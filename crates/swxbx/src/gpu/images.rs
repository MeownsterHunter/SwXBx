//! Render images and transfers to canonical arena bytes.

use super::image_layout::format_aspects;
use super::recording_device::RecordingDevice as Device;
use super::{
    commands::Commands, find_memory_type, image_layout::ImageDescriptor,
    texture_transfer::Transfer, Context,
};
use crate::tiling::{ImageKind, ImageShape, Layout, TileShape};
use ash::{vk, vk::Handle};
use std::{collections::HashMap, sync::Arc};

#[derive(Clone, Copy)]
pub(super) struct ImageInfo {
    /// Owned Vulkan image handle.
    pub image: vk::Image,
    /// Base-level image extent in texels.
    pub extent: vk::Extent3D,
    /// Public Vulkan format of the image.
    pub format: vk::Format,
    /// Tracked Vulkan image layout at recording time.
    pub layout: vk::ImageLayout,
    /// Typed dimensions, layers, and mip levels.
    pub shape: ImageShape,
    /// Storage size in bytes.
    pub size: u64,
}

pub(super) struct Image {
    pub wait_on_drop: bool,
    context: Arc<Context>,
    memory: vk::DeviceMemory,
    /// Image handle and tracked transfer metadata.
    pub info: ImageInfo,
    views: HashMap<[i32; 4], vk::ImageView>,
    packing: Option<Layout>,
}

impl Image {
    /// Create reusable execution resources; return None if Vulkan setup fails.
    pub fn new(
        context: &Arc<Context>,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Option<Self> {
        let mut image = Self::with_descriptor(
            context,
            &ImageDescriptor {
                shape: ImageShape {
                    width,
                    height,
                    depth: 1,
                    layers: 1,
                    levels: 1,
                    kind: ImageKind::D2,
                },
                format,
            },
        )?;
        image.info.size = u64::from(width)
            * u64::from(height)
            * if format == vk::Format::D32_SFLOAT_S8_UINT {
                5
            } else {
                4
            };
        image.packing = None;
        Some(image)
    }

    /// Allocate a typed image after checking native format support.
    pub fn with_descriptor(context: &Arc<Context>, descriptor: &ImageDescriptor) -> Option<Self> {
        let shape = descriptor.shape;
        let format = descriptor.format;
        let packing = if format == vk::Format::D32_SFLOAT_S8_UINT {
            if shape.depth != 1 || shape.layers != 1 || shape.levels != 1 {
                return None;
            }
            None
        } else {
            Some(descriptor.packing(TileShape {
                height_log2: 0,
                depth_log2: 0,
            })?)
        };
        let extent = vk::Extent3D {
            width: shape.width,
            height: shape.height,
            depth: shape.depth,
        };
        let ty = match shape.kind {
            ImageKind::D1 | ImageKind::D1Array => vk::ImageType::TYPE_1D,
            ImageKind::D3 => vk::ImageType::TYPE_3D,
            _ => vk::ImageType::TYPE_2D,
        };
        let flags = if shape.kind == ImageKind::Cube {
            vk::ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            vk::ImageCreateFlags::empty()
        };
        let props = unsafe {
            context
                .instance
                .get_physical_device_format_properties(context.physical_device, format)
        };
        let mut usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        if !props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::TRANSFER_SRC | vk::FormatFeatureFlags::TRANSFER_DST)
        {
            return None;
        }
        if !props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
        {
            return None;
        }
        usage |= vk::ImageUsageFlags::SAMPLED;
        if format_aspects(format) != vk::ImageAspectFlags::COLOR {
            if !props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            {
                return None;
            }
            usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        } else {
            if props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT)
            {
                usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
            }
            // Guest images may be bound as surfaces (storage images).
            if props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
            {
                usage |= vk::ImageUsageFlags::STORAGE;
            }
        }
        let limits = unsafe {
            context
                .instance
                .get_physical_device_image_format_properties(
                    context.physical_device,
                    format,
                    ty,
                    vk::ImageTiling::OPTIMAL,
                    usage,
                    flags,
                )
                .ok()?
        };
        if shape.width == 0
            || shape.height == 0
            || shape.depth == 0
            || shape.width > limits.max_extent.width
            || shape.height > limits.max_extent.height
            || shape.depth > limits.max_extent.depth
            || shape.levels > limits.max_mip_levels
            || shape.levels
                > 32 - shape
                    .width
                    .max(shape.height)
                    .max(shape.depth)
                    .leading_zeros()
            || shape.layers > limits.max_array_layers
        {
            return None;
        }
        let size = packing
            .as_ref()
            .map_or(u64::from(shape.width) * u64::from(shape.height) * 5, |p| {
                p.linear_size() as u64
            });
        let image = unsafe {
            context
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .flags(flags)
                        .image_type(ty)
                        .format(format)
                        .extent(extent)
                        .mip_levels(shape.levels)
                        .array_layers(shape.layers)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .ok()?
        };
        let mut result = Self {
            wait_on_drop: true,
            context: Arc::clone(context),
            memory: vk::DeviceMemory::null(),
            views: HashMap::new(),
            packing,
            info: ImageInfo {
                image,
                extent,
                format,
                layout: vk::ImageLayout::UNDEFINED,
                shape,
                size,
            },
        };
        let requirements = unsafe { context.device.get_image_memory_requirements(image) };
        let ty = find_memory_type(
            context,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        result.memory = unsafe {
            context
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(ty),
                    None,
                )
                .ok()?
        };
        unsafe {
            context
                .device
                .bind_image_memory(image, result.memory, 0)
                .ok()?;
        }
        Some(result)
    }

    /// Create or reuse the whole-resource sampled image view.
    pub fn view(&mut self) -> Option<vk::ImageView> {
        self.view_with_components(vk::ComponentMapping::default())
    }

    pub fn view_with_components(
        &mut self,
        components: vk::ComponentMapping,
    ) -> Option<vk::ImageView> {
        let key = [
            components.r.as_raw(),
            components.g.as_raw(),
            components.b.as_raw(),
            components.a.as_raw(),
        ];
        if let Some(view) = self.views.get(&key) {
            return Some(*view);
        }
        let view = unsafe {
            self.context
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(self.info.image)
                        .view_type(match self.info.shape.kind {
                            ImageKind::D1 => vk::ImageViewType::TYPE_1D,
                            ImageKind::D1Array => vk::ImageViewType::TYPE_1D_ARRAY,
                            ImageKind::D2 => vk::ImageViewType::TYPE_2D,
                            ImageKind::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
                            ImageKind::D3 => vk::ImageViewType::TYPE_3D,
                            ImageKind::Cube if self.info.shape.layers == 6 => {
                                vk::ImageViewType::CUBE
                            }
                            ImageKind::Cube => vk::ImageViewType::CUBE_ARRAY,
                        })
                        .format(self.info.format)
                        .components(components)
                        .subresource_range(image_range(self.info)),
                    None,
                )
                .ok()?
        };
        self.views.insert(key, view);
        Some(view)
    }

    /// Every subresource is in the shader-read layout; each barrier here
    /// spans the whole mip and layer range, so the image layout is theirs.
    fn shader_readable(&self) -> bool {
        self.info.layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    }

    /// Record a barrier to the requested layout and update tracked layout.
    pub fn transition(&mut self, command: vk::CommandBuffer, layout: vk::ImageLayout) {
        transition(&self.context.recorder(), command, self.info, layout);
        self.info.layout = layout;
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            // Includes presentation copies and callers using the context directly.
            if self.wait_on_drop {
                let _ = self.context.wait_queue();
            }
            for view in self.views.values() {
                self.context.device.destroy_image_view(*view, None);
            }
            self.context.device.destroy_image(self.info.image, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}

pub(super) struct Buffer {
    pub wait_on_drop: bool,
    context: Arc<Context>,
    memory: vk::DeviceMemory,
    /// Owned mapped staging buffer handle.
    pub buffer: vk::Buffer,
    size: u64,
    mapped: *mut u8,
}

// SAFETY: mapped bytes belong to this owner; the queue and CPU access are serialized.
unsafe impl Send for Buffer {}

impl Buffer {
    /// Create reusable execution resources; return None if Vulkan setup fails.
    pub fn new(context: &Arc<Context>, size: u64) -> Option<Self> {
        let buffer = unsafe {
            context
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(
                            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .ok()?
        };
        let mut result = Self {
            wait_on_drop: true,
            context: Arc::clone(context),
            memory: vk::DeviceMemory::null(),
            buffer,
            size,
            mapped: std::ptr::null_mut(),
        };
        let req = unsafe { context.device.get_buffer_memory_requirements(buffer) };
        let visible =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let ty = find_memory_type(
            context,
            req.memory_type_bits,
            visible | vk::MemoryPropertyFlags::HOST_CACHED,
        )
        .or_else(|| find_memory_type(context, req.memory_type_bits, visible))?;
        result.memory = unsafe {
            context
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(ty),
                    None,
                )
                .ok()?
        };
        unsafe {
            context
                .device
                .bind_buffer_memory(buffer, result.memory, 0)
                .ok()?;
        }
        result.mapped = unsafe {
            context
                .device
                .map_memory(result.memory, 0, size, vk::MemoryMapFlags::empty())
                .ok()?
                .cast()
        };
        Some(result)
    }

    /// Copy bytes into mapped staging storage; return None for an oversized range.
    pub fn write(&self, bytes: &[u8]) -> Option<()> {
        if bytes.len() as u64 > self.size {
            return None;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.mapped, bytes.len());
        }
        Some(())
    }
    /// Return size bytes copied from mapped staging storage.
    pub fn read(&self, size: usize) -> Option<Vec<u8>> {
        if size as u64 > self.size {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(self.mapped, size) }.to_vec())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            if self.wait_on_drop {
                let _ = self.context.wait_queue();
            }
            if !self.mapped.is_null() {
                self.context.device.unmap_memory(self.memory);
            }
            self.context.device.destroy_buffer(self.buffer, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}

struct DrawFramebuffer {
    context: Arc<Context>,
    handle: vk::Framebuffer,
    views: Vec<vk::ImageView>,
    pipeline: Arc<super::graphics::GraphicsPipeline>,
}
impl Drop for DrawFramebuffer {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_framebuffer(self.handle, None);
        }
    }
}

const RETAINED_SLOTS: usize = 16;

#[derive(Default)]
struct DrawResources {
    framebuffer: Option<Arc<DrawFramebuffer>>,
    states: Vec<Arc<super::graphics::Draw>>,
}

pub(super) struct Images {
    context: Arc<Context>,
    commands: Commands,
    images: HashMap<u64, Image>,
    recycled: Vec<Image>,
    staging: Option<Buffer>,
    transfer: Option<Transfer>,
    framebuffer: Option<Arc<DrawFramebuffer>>,
    draw_resources: Vec<DrawResources>,
    parallel_recording: bool,
}

impl Drop for Images {
    fn drop(&mut self) {
        let _ = self.commands.wait();
        self.framebuffer = None;
        self.draw_resources.clear();
    }
}

impl Images {
    /// Create reusable execution resources; return None if Vulkan setup fails.
    pub fn new(context: &Arc<Context>) -> Option<Self> {
        Some(Self {
            context: Arc::clone(context),
            commands: Commands::with_capacity(context, RETAINED_SLOTS)?,
            images: HashMap::new(),
            recycled: Vec::new(),
            staging: None,
            transfer: None,
            framebuffer: None,
            parallel_recording: false,
            draw_resources: (0..RETAINED_SLOTS)
                .map(|_| DrawResources::default())
                .collect(),
        })
    }

    pub fn set_parallel_recording(&mut self, enabled: bool) {
        self.parallel_recording = enabled;
    }

    /// Wait for retained command submissions to complete; return None on failure.
    pub(super) fn submission(&self) -> Option<super::commands::Submission> {
        self.commands.submission()
    }

    pub fn wait(&self) -> Option<()> {
        self.commands.wait()
    }

    /// Create or reuse a compatible color or depth image for this key.
    pub fn ensure(&mut self, key: u64, width: u64, height: u64, depth: bool) -> bool {
        self.ensure_format(
            key,
            width,
            height,
            if depth {
                vk::Format::D32_SFLOAT
            } else {
                vk::Format::R8G8B8A8_UNORM
            },
        )
    }

    /// Create or reuse an image with explicit format and dimensions.
    pub fn ensure_format(&mut self, key: u64, width: u64, height: u64, format: vk::Format) -> bool {
        let Ok(width) = u32::try_from(width.max(1)) else {
            return false;
        };
        let Ok(height) = u32::try_from(height.max(1)) else {
            return false;
        };
        if let Some(image) = self.images.get(&key) {
            return image.info.extent.width == width
                && image.info.extent.height == height
                && image.info.format == format;
        }
        let Some(image) = Image::new(&self.context, width, height, format) else {
            return false;
        };
        self.images.insert(key, image);
        true
    }

    /// Invalidate transfer recordings before pooling or destroying the removed image.
    pub fn remove(&mut self, key: u64) {
        self.framebuffer = None;
        if let Some(image) = self.images.remove(&key) {
            // Destroying a referenced object invalidates executable recordings.
            self.commands.invalidate_cached();
            if self.commands.wait().is_some() {
                for resources in &mut self.draw_resources {
                    resources.framebuffer = None;
                    resources.states.clear();
                }
                if self.recycled.len() < 8 {
                    self.recycled.push(image);
                }
            }
        }
    }

    /// Create or reuse a typed image; fail if its native format is unsupported.
    pub fn ensure_descriptor(&mut self, key: u64, descriptor: &ImageDescriptor) -> bool {
        if let Some(image) = self.images.get(&key) {
            return image.info.shape == descriptor.shape
                && image.info.format == descriptor.format
                && image.packing.is_some();
        }
        let image = if let Some(i) = self.recycled.iter().position(|i| {
            i.info.shape == descriptor.shape
                && i.info.format == descriptor.format
                && i.packing.is_some()
        }) {
            self.recycled.swap_remove(i)
        } else {
            let Some(image) = Image::with_descriptor(&self.context, descriptor) else {
                return false;
            };
            image
        };
        self.images.insert(key, image);
        true
    }

    fn staging(&mut self, size: u64) -> Option<()> {
        self.commands.wait()?;
        if self.staging.as_ref().is_none_or(|b| b.size < size) {
            self.staging = Some(Buffer::new(&self.context, size)?);
        }
        Some(())
    }

    /// Convert and transfer one image between arena and tiled storage.
    pub fn tiled_transfer(
        &mut self,
        key: u64,
        address: u64,
        packing: &Layout,
        load: bool,
    ) -> Option<()> {
        self.tiled_batch(&[(key, address, packing)], load)
    }

    /// Transfer up to 256 images using shared scratch and one submission.
    pub fn tiled_batch(&mut self, resources: &[(u64, u64, &Layout)], load: bool) -> Option<()> {
        if resources.is_empty() || resources.len() > 256 {
            return None;
        }
        for &(key, address, packing) in resources {
            let image = self.images.get(&key)?;
            if address == 0
                || !address.is_multiple_of(4)
                || image.info.shape != packing.shape()
                || image.packing.as_ref()?.format() != packing.format()
            {
                return None;
            }
            address.checked_add(packing.tiled_size() as u64)?;
        }
        if self.transfer.is_none() {
            self.transfer = Some(Transfer::new(&self.context)?);
        }
        // Reordering is valid only for unique images and disjoint arena ranges.
        let grouped = resources.iter().all(|r| r.2.linear_size() <= 256 * 1024)
            && resources
                .iter()
                .enumerate()
                .all(|(i, &(key, address, layout))| {
                    resources[..i].iter().all(|&(other, start, packing)| {
                        key != other
                            && (address >= start + packing.tiled_size() as u64
                                || start >= address + layout.tiled_size() as u64)
                    })
                });
        let mut offsets = Vec::with_capacity(resources.len());
        let mut span = 0usize;
        for &(_, _, packing) in resources {
            offsets.push(if grouped { span } else { 0 });
            let size = packing.linear_size().checked_add(255)? & !255;
            span = if grouped {
                span.checked_add(size)?
            } else {
                span.max(size)
            };
        }
        let required = span.checked_mul(self.commands.slots())?;
        if self.transfer.as_ref()?.capacity() < required {
            self.commands.wait()?;
            self.commands.invalidate_cached();
        }
        self.transfer.as_mut()?.ensure_capacity(required)?;
        // Keep the slot stride fixed until allocation growth waits for every fence.
        let ring_stride = self.transfer.as_ref()?.capacity() / self.commands.slots();
        let ring_base = ring_stride.checked_mul(self.commands.slot())?;
        for offset in &mut offsets {
            *offset = offset.checked_add(ring_base)?;
        }
        let transfer = self.transfer.as_ref()?;
        let cache = resources
            .iter()
            .zip(&offsets)
            .map(|(&(key, address, packing), &offset)| {
                let image = &self.images[&key];
                super::commands::TransferKey {
                    image: image.info.image.as_raw(),
                    scratch: transfer.buffer().as_raw(),
                    address,
                    scratch_address: transfer.address() + offset as u64,
                    layout: image.info.layout.as_raw(),
                    load,
                    conversion: true,
                    packing: packing.clone(),
                }
            })
            .collect();
        let (cmd, reused) = self.commands.begin_cached(cache)?;
        if reused {
            self.replay_layouts(resources.iter().map(|resource| resource.0), load);
            return self.commands.submit(false, None);
        }
        let transfer = self.transfer.as_ref()?;
        buffer_barrier(&self.context.recorder(), cmd);
        if grouped && load {
            for (&(_, address, packing), &offset) in resources.iter().zip(&offsets) {
                transfer.record_disjoint(cmd, address, packing, true, offset)?;
            }
            buffer_barrier(&self.context.recorder(), cmd);
        }
        for (&(key, address, packing), &offset) in resources.iter().zip(&offsets) {
            let image = self.images.get_mut(&key)?;
            if !grouped && load {
                transfer.record_at(cmd, address, packing, true, offset)?;
            }
            image.transition(
                cmd,
                if load {
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL
                } else {
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                },
            );
            let regions = buffer_regions(image.info, image.packing.as_ref(), offset as u64);
            unsafe {
                if load {
                    self.context.recorder().cmd_copy_buffer_to_image(
                        cmd,
                        transfer.buffer(),
                        image.info.image,
                        image.info.layout,
                        &regions,
                    );
                } else {
                    self.context.recorder().cmd_copy_image_to_buffer(
                        cmd,
                        image.info.image,
                        image.info.layout,
                        transfer.buffer(),
                        &regions,
                    );
                }
            }
            if !grouped {
                buffer_barrier(&self.context.recorder(), cmd);
                if !load {
                    transfer.record_at(cmd, address, packing, false, offset)?;
                }
            }
        }
        buffer_barrier(&self.context.recorder(), cmd);
        if grouped && !load {
            for (&(_, address, packing), &offset) in resources.iter().zip(&offsets) {
                transfer.record_disjoint(cmd, address, packing, false, offset)?;
            }
            buffer_barrier(&self.context.recorder(), cmd);
        }
        self.commands.submit(false, None)
    }

    pub fn linear_batch(
        &mut self,
        resources: &[(u64, vk::Buffer, u64, &Layout)],
        load: bool,
    ) -> Option<()> {
        if resources.is_empty() || resources.len() > 256 {
            return None;
        }
        let mut cache = Vec::with_capacity(resources.len());
        for &(key, buffer, offset, supplied) in resources {
            let image = self.images.get(&key)?;
            let packing = image.packing.as_ref()?;
            if image.info.shape != supplied.shape() || packing.format() != supplied.format() {
                return None;
            }
            let alignment = u64::from(packing.format().bytes).max(4);
            if buffer == vk::Buffer::null() || !offset.is_multiple_of(alignment) {
                return None;
            }
            offset.checked_add(packing.linear_size() as u64)?;
            cache.push(super::commands::TransferKey {
                image: image.info.image.as_raw(),
                scratch: buffer.as_raw(),
                address: offset,
                scratch_address: 0,
                layout: image.info.layout.as_raw(),
                load,
                conversion: false,
                packing: packing.clone(),
            });
        }
        let ordered = resources
            .iter()
            .enumerate()
            .any(|(i, &(key, buffer, offset, packing))| {
                resources[..i]
                    .iter()
                    .any(|&(other_key, other, start, layout)| {
                        key == other_key
                            || (!load
                                && buffer == other
                                && offset < start + layout.linear_size() as u64
                                && start < offset + packing.linear_size() as u64)
                    })
            });
        let (cmd, reused) = self.commands.begin_cached(cache)?;
        if reused {
            self.replay_layouts(resources.iter().map(|resource| resource.0), load);
            return self.commands.submit(false, None);
        }
        buffer_barrier(&self.context.recorder(), cmd);
        for &(key, buffer, offset, _) in resources {
            let image = self.images.get_mut(&key)?;
            image.transition(
                cmd,
                if load {
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL
                } else {
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                },
            );
            let regions = buffer_regions(image.info, image.packing.as_ref(), offset);
            unsafe {
                if load {
                    self.context.recorder().cmd_copy_buffer_to_image(
                        cmd,
                        buffer,
                        image.info.image,
                        image.info.layout,
                        &regions,
                    );
                } else {
                    self.context.recorder().cmd_copy_image_to_buffer(
                        cmd,
                        image.info.image,
                        image.info.layout,
                        buffer,
                        &regions,
                    );
                }
            }
            if ordered {
                buffer_barrier(&self.context.recorder(), cmd);
            }
        }
        buffer_barrier(&self.context.recorder(), cmd);
        self.commands.submit(false, None)
    }

    /// Transition the image for shader reads and writes and return its
    /// whole-resource view in the image's own format, or None when the image
    /// cannot be a storage image.
    pub fn storage(&mut self, key: u64) -> Option<vk::ImageView> {
        let (cmd, _) = self.commands.begin_direct()?;
        let image = self.images.get_mut(&key)?;
        let props = unsafe {
            image
                .context
                .instance
                .get_physical_device_format_properties(
                    image.context.physical_device,
                    image.info.format,
                )
        };
        if !props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
        {
            self.commands.submit(false, None)?;
            return None;
        }
        image.transition(cmd, vk::ImageLayout::GENERAL);
        let view = image.view()?;
        self.commands.submit(false, None)?;
        Some(view)
    }

    /// The format of a keyed image.
    pub fn format(&self, key: u64) -> Option<vk::Format> {
        Some(self.images.get(&key)?.info.format)
    }

    /// Transition the image for shader reads and return its whole-resource view.
    pub fn sampled(&mut self, key: u64) -> Option<vk::ImageView> {
        self.sampled_with_components(key, vk::ComponentMapping::default())
    }

    /// A replayed transfer recording leaves every image in the layout its
    /// recorded barriers end in, exactly as the first recording did.
    fn replay_layouts(&mut self, keys: impl Iterator<Item = u64>, load: bool) {
        let layout = transfer_layout(load);
        for key in keys {
            self.images
                .entry(key)
                .and_modify(|image| image.info.layout = layout);
        }
    }

    /// An image whose tracked layout is already shader-readable needs no
    /// barrier: every write to it moves it out of that layout first.
    pub fn sampled_with_components(
        &mut self,
        key: u64,
        components: vk::ComponentMapping,
    ) -> Option<vk::ImageView> {
        if self.images.get(&key).is_some_and(Image::shader_readable) {
            return self
                .images
                .get_mut(&key)
                .and_then(|image| image.view_with_components(components));
        }
        let (cmd, _) = self.commands.begin_direct()?;
        let image = self.images.get_mut(&key)?;
        image.transition(cmd, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let view = image.view_with_components(components)?;
        self.commands.submit(false, None)?;
        Some(view)
    }

    /// Clear selected channels of a color image and complete the transfer.
    pub fn clear_color(&mut self, key: u64, color: [f32; 4], mask: u32) -> Option<()> {
        if mask & 15 == 0 {
            return Some(());
        }
        if self.images.get(&key)?.info.format != vk::Format::R8G8B8A8_UNORM {
            return None;
        }
        if mask & 15 != 15 {
            // Vulkan transfer clears cannot mask channels. Preserve the others.
            let (_, _, mut bytes) = self.readback(key)?;
            let values = color.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
            for pixel in bytes.as_chunks_mut::<4>().0 {
                for c in 0..4 {
                    if mask & (1 << c) != 0 {
                        pixel[c] = values[c];
                    }
                }
            }
            return self.upload(key, &bytes);
        }
        let (cmd, _) = if self.parallel_recording {
            self.commands.begin()?
        } else {
            self.commands.begin_direct()?
        };
        let image = self.images.get_mut(&key)?;
        image.transition(cmd, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        unsafe {
            self.context.recorder().cmd_clear_color_image(
                cmd,
                image.info.image,
                image.info.layout,
                &vk::ClearColorValue { float32: color },
                &[image_range(image.info)],
            );
        }
        self.commands.submit(false, None)
    }

    /// Record one group and retain its resources until its command slot retires.
    pub fn draw(
        &mut self,
        keys: &[u64],
        depth: Option<u64>,
        memory: &super::GlobalMemory,
        draws: &[super::graphics::PendingDraw],
        query: Option<vk::QueryPool>,
    ) -> Option<()> {
        #[cfg(feature = "draw-metrics")]
        let _span = crate::draw_metrics::PipelineSpan::new(11);
        let pipeline = &draws.first()?.pipeline;
        let extent = self.images.get(keys.first()?)?.info.extent;
        let mut views = [vk::ImageView::null(); 9];
        for (at, key) in keys.iter().enumerate() {
            views[at] = self.images.get_mut(key)?.view()?;
        }
        if let Some(depth) = depth {
            views[keys.len()] = self.images.get_mut(&depth)?.view()?;
        }
        let views = &views[..keys.len() + usize::from(depth.is_some())];
        if self
            .framebuffer
            .as_ref()
            .is_none_or(|f| f.views != views || !Arc::ptr_eq(&f.pipeline, pipeline))
        {
            let handle = unsafe {
                self.context
                    .device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(pipeline.render_pass)
                            .attachments(views)
                            .width(extent.width)
                            .height(extent.height)
                            .layers(1),
                        None,
                    )
                    .ok()?
            };
            self.framebuffer = Some(Arc::new(DrawFramebuffer {
                context: self.context.clone(),
                handle,
                views: views.to_vec(),
                pipeline: pipeline.clone(),
            }));
        }
        let framebuffer = self.framebuffer.as_ref()?.clone();
        let device = &self.context.recorder();
        let result = (|| {
            let slot = self.commands.slot();
            let (cmd, _) = self.commands.begin_direct()?;
            let resources = &mut self.draw_resources[slot];
            resources.states.clear();
            resources.framebuffer = Some(framebuffer.clone());
            buffer_barrier(device, cmd);
            for key in keys {
                self.images
                    .get_mut(key)?
                    .transition(cmd, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            }
            if let Some(depth) = depth {
                self.images
                    .get_mut(&depth)?
                    .transition(cmd, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            }
            unsafe {
                if let Some(query) = query {
                    device.cmd_reset_query_pool(cmd, query, 0, 1);
                }
                device.cmd_begin_render_pass(
                    cmd,
                    &vk::RenderPassBeginInfo::default()
                        .render_pass(pipeline.render_pass)
                        .framebuffer(framebuffer.handle)
                        .render_area(vk::Rect2D::default().extent(vk::Extent2D {
                            width: extent.width,
                            height: extent.height,
                        })),
                    vk::SubpassContents::INLINE,
                );
                if let Some(query) = query {
                    device.cmd_begin_query(cmd, query, 0, vk::QueryControlFlags::empty());
                }
                let mut previous = None;
                let mut previous_pipeline = None;
                let mut previous_index = None;
                for draw in draws {
                    let pipeline = &draw.pipeline;
                    let pipeline_id = Arc::as_ptr(pipeline);
                    let id = Arc::as_ptr(&draw.state);
                    if previous != Some(id) || previous_pipeline != Some(pipeline_id) {
                        resources.states.push(draw.state.clone());
                        pipeline.record(cmd, memory, &draw.state);
                        previous = Some(id);
                        previous_pipeline = Some(pipeline_id);
                    }
                    let bind_index = if let super::graphics::DrawVertices::Elements {
                        buffer,
                        offset,
                        index_type,
                        ..
                    } = draw.vertices
                    {
                        let index = (buffer, offset, index_type);
                        let changed = previous_index != Some(index);
                        previous_index = Some(index);
                        changed
                    } else {
                        false
                    };
                    pipeline.record_vertices(
                        cmd,
                        draw.vertices,
                        draw.count,
                        bind_index,
                        draw.geometry,
                    );
                }
                if let Some(query) = query {
                    device.cmd_end_query(cmd, query, 0);
                }
                device.cmd_end_render_pass(cmd);
            }
            self.commands.submit(false, None)
        })();
        if result.is_none() {
            let _ = self.context.wait_queue();
        }
        result
    }

    /// Clear a depth image and complete the transfer.
    pub fn clear_depth(&mut self, key: u64, depth: f32) -> Option<()> {
        if self.images.get(&key)?.info.format != vk::Format::D32_SFLOAT {
            return None;
        }
        let (cmd, _) = if self.parallel_recording {
            self.commands.begin()?
        } else {
            self.commands.begin_direct()?
        };
        let image = self.images.get_mut(&key)?;
        image.transition(cmd, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        unsafe {
            self.context.recorder().cmd_clear_depth_stencil_image(
                cmd,
                image.info.image,
                image.info.layout,
                &vk::ClearDepthStencilValue { depth, stencil: 0 },
                &[image_range(image.info)],
            );
        }
        self.commands.submit(false, None)
    }

    pub fn info(&self, key: u64) -> Option<ImageInfo> {
        self.images.get(&key).map(|image| image.info)
    }

    /// Let a readback command own the source transition and update its tracked layout.
    pub fn readback_source(
        &mut self,
        key: u64,
        record: impl FnOnce(ImageInfo) -> Option<()>,
    ) -> Option<()> {
        let image = self.images.get_mut(&key)?;
        record(image.info)?;
        image.info.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        Some(())
    }

    /// Make an image transfer-readable and return its tracked metadata.
    pub fn source(&mut self, key: u64) -> Option<ImageInfo> {
        let (cmd, _) = self.commands.begin_direct()?;
        let image = self.images.get_mut(&key)?;
        image.transition(cmd, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        let info = image.info;
        self.commands.submit(false, None)?;
        Some(info)
    }

    /// Copy packed-linear bytes between a checked arena buffer range and an image.
    pub fn arena_transfer(
        &mut self,
        key: u64,
        buffer: vk::Buffer,
        offset: u64,
        load: bool,
    ) -> Option<()> {
        let info = self.images.get(&key)?.info;
        let alignment =
            super::image_layout::format_block(info.format).map_or(4, |f| u64::from(f.bytes).max(4));
        if !offset.is_multiple_of(alignment) {
            return None;
        }
        let (cmd, _) = self.commands.begin_direct()?;
        let image = self.images.get_mut(&key)?;
        buffer_barrier(&self.context.recorder(), cmd);
        let layout = if load {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        } else {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        };
        image.transition(cmd, layout);
        let regions = buffer_regions(image.info, image.packing.as_ref(), offset);
        unsafe {
            if load {
                self.context.recorder().cmd_copy_buffer_to_image(
                    cmd,
                    buffer,
                    image.info.image,
                    layout,
                    &regions,
                );
            } else {
                self.context.recorder().cmd_copy_image_to_buffer(
                    cmd,
                    image.info.image,
                    layout,
                    buffer,
                    &regions,
                );
                buffer_barrier(&self.context.recorder(), cmd);
            }
        }
        self.commands.submit(false, None)
    }

    /// Upload packed-linear bytes into the existing image.
    pub fn upload(&mut self, key: u64, bytes: &[u8]) -> Option<()> {
        let info = self.images.get(&key)?.info;
        if bytes.len() as u64 != info.size {
            return None;
        }
        self.staging(info.size)?;
        let staging = self.staging.as_ref()?;
        staging.write(bytes)?;
        let buffer = staging.buffer;
        self.arena_transfer(key, buffer, 0, true)?;
        self.commands.wait()
    }

    /// Copy all mip levels and array layers between compatible images.
    pub fn copy(&mut self, destination: u64, source: u64) -> Option<()> {
        if destination == source {
            return Some(());
        }
        let src = self.images.get(&source)?.info;
        let dst = self.images.get(&destination)?.info;
        if src.shape != dst.shape || src.format != dst.format {
            return None;
        }
        let regions: Vec<_> = (0..src.shape.levels)
            .map(|level| {
                let extent = vk::Extent3D {
                    width: (src.extent.width >> level).max(1),
                    height: (src.extent.height >> level).max(1),
                    depth: (src.extent.depth >> level).max(1),
                };
                vk::ImageCopy::default()
                    .src_subresource(
                        layers(src.format)
                            .mip_level(level)
                            .layer_count(src.shape.layers),
                    )
                    .dst_subresource(
                        layers(dst.format)
                            .mip_level(level)
                            .layer_count(dst.shape.layers),
                    )
                    .extent(extent)
            })
            .collect();
        self.record_copy(destination, source, &regions)
    }

    fn record_copy(
        &mut self,
        destination: u64,
        source: u64,
        regions: &[vk::ImageCopy],
    ) -> Option<()> {
        let (cmd, _) = self.commands.begin()?;
        self.images
            .get_mut(&source)?
            .transition(cmd, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        self.images
            .get_mut(&destination)?
            .transition(cmd, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let src = self.images.get(&source)?.info;
        let dst = self.images.get(&destination)?.info;
        unsafe {
            self.context
                .recorder()
                .cmd_copy_image(cmd, src.image, src.layout, dst.image, dst.layout, regions);
        }
        self.commands.submit(false, None)
    }

    /// Copy checked compatible image regions and complete the transfer.
    pub fn copy_region(
        &mut self,
        destination: u64,
        source: u64,
        from: super::image_layout::ImageRegion,
        to: super::image_layout::ImageRegion,
    ) -> Option<()> {
        if source == destination || from.extent != to.extent {
            return None;
        }
        let src = self.images.get(&source)?.info;
        let dst = self.images.get(&destination)?.info;
        if src.format != dst.format || !valid_region(src, from) || !valid_region(dst, to) {
            return None;
        }
        let offsets = |v: [u32; 3]| vk::Offset3D {
            x: v[0] as i32,
            y: v[1] as i32,
            z: v[2] as i32,
        };
        self.record_copy(
            destination,
            source,
            &[vk::ImageCopy::default()
                .src_subresource(
                    layers(src.format)
                        .mip_level(from.level)
                        .base_array_layer(from.layer),
                )
                .src_offset(offsets(from.offset))
                .dst_subresource(
                    layers(dst.format)
                        .mip_level(to.level)
                        .base_array_layer(to.layer),
                )
                .dst_offset(offsets(to.offset))
                .extent(vk::Extent3D {
                    width: from.extent[0],
                    height: from.extent[1],
                    depth: from.extent[2],
                })],
        )
    }

    /// Blit checked compatible image regions with the chosen filter.
    pub fn blit_region(
        &mut self,
        destination: u64,
        source: u64,
        from: super::image_layout::BlitRegion,
        to: super::image_layout::BlitRegion,
        filter: vk::Filter,
    ) -> Option<()> {
        if source == destination || !matches!(filter, vk::Filter::NEAREST | vk::Filter::LINEAR) {
            return None;
        }
        let src = self.images.get(&source)?.info;
        let dst = self.images.get(&destination)?.info;
        if super::image_layout::numeric_class(src.format)
            != super::image_layout::numeric_class(dst.format)
            || (super::image_layout::numeric_class(src.format)
                == super::image_layout::NumericClass::Depth
                && (src.format != dst.format || filter != vk::Filter::NEAREST))
        {
            return None;
        }
        if !can_blit(&self.context, src.format, dst.format)
            || !valid_blit(src, from)
            || !valid_blit(dst, to)
        {
            return None;
        }
        let block = super::image_layout::format_block(src.format)?;
        let block_dst = super::image_layout::format_block(dst.format)?;
        if block.width != 1 || block.height != 1 || block_dst.width != 1 || block_dst.height != 1 {
            return None;
        }
        if filter == vk::Filter::LINEAR
            && !unsafe {
                self.context
                    .instance
                    .get_physical_device_format_properties(self.context.physical_device, src.format)
            }
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            return None;
        }
        let (cmd, _) = self.commands.begin()?;
        self.images
            .get_mut(&source)?
            .transition(cmd, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        self.images
            .get_mut(&destination)?
            .transition(cmd, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let offsets = |v: [[i32; 3]; 2]| {
            v.map(|v| vk::Offset3D {
                x: v[0],
                y: v[1],
                z: v[2],
            })
        };
        unsafe {
            self.context.recorder().cmd_blit_image(
                cmd,
                src.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_subresource(
                        layers(src.format)
                            .mip_level(from.level)
                            .base_array_layer(from.layer),
                    )
                    .src_offsets(offsets(from.offsets))
                    .dst_subresource(
                        layers(dst.format)
                            .mip_level(to.level)
                            .base_array_layer(to.layer),
                    )
                    .dst_offsets(offsets(to.offsets))],
                filter,
            );
        }
        self.commands.submit(false, None)
    }

    /// Read the keyed image into tightly packed RGBA bytes and dimensions.
    pub fn readback(&mut self, key: u64) -> Option<(u32, u32, Vec<u8>)> {
        let info = self.images.get(&key)?.info;
        self.staging(info.size)?;
        let buffer = self.staging.as_ref()?.buffer;
        self.arena_transfer(key, buffer, 0, false)?;
        self.commands.wait()?;
        Some((
            info.extent.width,
            info.extent.height,
            self.staging.as_ref()?.read(info.size as usize)?,
        ))
    }

    /// Copy every mip and layer into packed-linear host bytes.
    pub fn read_image(&mut self, image: &mut Image) -> Option<Vec<u8>> {
        self.staging(image.info.size)?;
        let staging = self.staging.as_ref()?;
        let (cmd, _) = self.commands.begin()?;
        image.transition(cmd, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        unsafe {
            self.context.recorder().cmd_copy_image_to_buffer(
                cmd,
                image.info.image,
                image.info.layout,
                staging.buffer,
                &buffer_regions(image.info, image.packing.as_ref(), 0),
            );
        }
        buffer_barrier(&self.context.recorder(), cmd);
        self.commands.submit(false, None)?;
        self.commands.wait()?;
        staging.read(image.info.size as usize)
    }

    /// Scale and convert the source into an offscreen color target.
    pub fn blit_offscreen(&mut self, source: ImageInfo, target: &mut Image) -> Option<()> {
        if !can_blit(&self.context, source.format, target.info.format) {
            return None;
        }
        let (cmd, _) = self.commands.begin()?;
        record_present(
            &self.context.recorder(),
            cmd,
            source,
            target.info,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        target.info.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        self.commands.submit(false, None)
    }
}

fn transfer_layout(load: bool) -> vk::ImageLayout {
    [
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    ][usize::from(load)]
}

fn image_range(info: ImageInfo) -> vk::ImageSubresourceRange {
    range(info.format)
        .level_count(info.shape.levels)
        .layer_count(info.shape.layers)
}

pub(super) fn range(format: vk::Format) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(format_aspects(format))
        .level_count(1)
        .layer_count(1)
}

fn layers(format: vk::Format) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::default()
        .aspect_mask(
            if format_aspects(format).contains(vk::ImageAspectFlags::DEPTH) {
                vk::ImageAspectFlags::DEPTH
            } else {
                format_aspects(format)
            },
        )
        .layer_count(1)
}

fn buffer_regions(
    info: ImageInfo,
    packing: Option<&Layout>,
    offset: u64,
) -> Vec<vk::BufferImageCopy> {
    if let Some(packing) = packing {
        let mut regions = Vec::new();
        for layer in 0..info.shape.layers {
            for (level, p) in packing.levels().iter().enumerate() {
                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(
                            offset
                                + (layer as usize * packing.linear_layer_stride() + p.linear_offset)
                                    as u64,
                        )
                        .image_subresource(
                            layers(info.format)
                                .mip_level(level as u32)
                                .base_array_layer(layer),
                        )
                        .image_extent(vk::Extent3D {
                            width: p.extent[0],
                            height: p.extent[1],
                            depth: p.extent[2],
                        }),
                );
            }
        }
        return regions;
    }
    let depth = vk::BufferImageCopy::default()
        .buffer_offset(offset)
        .image_subresource(layers(info.format))
        .image_extent(info.extent);
    if info.format == vk::Format::D32_SFLOAT_S8_UINT {
        vec![
            depth,
            vk::BufferImageCopy::default()
                .buffer_offset(
                    offset + u64::from(info.extent.width) * u64::from(info.extent.height) * 4,
                )
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::STENCIL)
                        .layer_count(1),
                )
                .image_extent(info.extent),
        ]
    } else {
        vec![depth]
    }
}

pub(super) fn transition(
    device: &Device<'_>,
    cmd: vk::CommandBuffer,
    image: ImageInfo,
    layout: vk::ImageLayout,
) {
    let (src_stage, src_access) = if image.layout == vk::ImageLayout::UNDEFINED {
        (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::empty(),
        )
    } else {
        (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        )
    };
    let (dst_stage, dst_access) = if layout == vk::ImageLayout::PRESENT_SRC_KHR {
        (
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::AccessFlags::empty(),
        )
    } else if layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
        (
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
    } else if layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
        (
            vk::PipelineStageFlags::VERTEX_SHADER
                | vk::PipelineStageFlags::FRAGMENT_SHADER
                | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ,
        )
    } else if layout == vk::ImageLayout::GENERAL {
        (
            vk::PipelineStageFlags::VERTEX_SHADER
                | vk::PipelineStageFlags::FRAGMENT_SHADER
                | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        )
    } else if layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
        (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
    } else {
        (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
        )
    };
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(dst_access)
                .old_layout(image.layout)
                .new_layout(layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image.image)
                .subresource_range(image_range(image))],
        );
    }
}

pub(super) fn buffer_barrier(device: &Device<'_>, cmd: vk::CommandBuffer) {
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::MEMORY_READ
                        | vk::AccessFlags::MEMORY_WRITE
                        | vk::AccessFlags::HOST_READ,
                )],
            &[],
            &[],
        );
    }
}

pub(super) fn can_blit(context: &Context, source: vk::Format, destination: vk::Format) -> bool {
    unsafe {
        let src = context
            .instance
            .get_physical_device_format_properties(context.physical_device, source);
        let dst = context
            .instance
            .get_physical_device_format_properties(context.physical_device, destination);
        src.optimal_tiling_features
            .contains(vk::FormatFeatureFlags::BLIT_SRC)
            && dst
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::BLIT_DST)
    }
}

/// Shared by swapchain and offscreen presentation, including conversion and scaling.
pub(super) fn record_present(
    device: &Device<'_>,
    cmd: vk::CommandBuffer,
    source: ImageInfo,
    destination: ImageInfo,
    final_layout: vk::ImageLayout,
) {
    transition(device, cmd, source, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
    transition(
        device,
        cmd,
        destination,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );
    let offsets = |extent: vk::Extent3D| {
        [
            vk::Offset3D::default(),
            vk::Offset3D {
                x: extent.width as i32,
                y: extent.height as i32,
                z: 1,
            },
        ]
    };
    unsafe {
        device.cmd_blit_image(
            cmd,
            source.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            destination.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::ImageBlit::default()
                .src_subresource(layers(source.format))
                .src_offsets(offsets(source.extent))
                .dst_subresource(layers(destination.format))
                .dst_offsets(offsets(destination.extent))],
            vk::Filter::NEAREST,
        );
    }
    transition(
        device,
        cmd,
        ImageInfo {
            layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ..destination
        },
        final_layout,
    );
}

fn mip_extent(info: ImageInfo, level: u32) -> Option<[u32; 3]> {
    if level >= info.shape.levels {
        return None;
    }
    Some([
        (info.extent.width >> level).max(1),
        (info.extent.height >> level).max(1),
        (info.extent.depth >> level).max(1),
    ])
}
fn valid_region(info: ImageInfo, r: super::image_layout::ImageRegion) -> bool {
    let Some(extent) = mip_extent(info, r.level) else {
        return false;
    };
    if r.layer >= info.shape.layers {
        return false;
    }
    let Some(block) = super::image_layout::format_block(info.format) else {
        return false;
    };
    let align = [u32::from(block.width), u32::from(block.height), 1];
    (0..3).all(|i| {
        r.extent[i] > 0
            && r.offset[i].is_multiple_of(align[i])
            && r.offset[i].checked_add(r.extent[i]).is_some_and(|end| {
                end <= extent[i] && (r.extent[i].is_multiple_of(align[i]) || end == extent[i])
            })
    })
}
fn valid_blit(info: ImageInfo, r: super::image_layout::BlitRegion) -> bool {
    let Some(extent) = mip_extent(info, r.level) else {
        return false;
    };
    r.layer < info.shape.layers
        && (0..3).all(|i| {
            r.offsets[0][i] != r.offsets[1][i]
                && r.offsets
                    .iter()
                    .all(|v| v[i] >= 0 && v[i] as u32 <= extent[i])
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Vulkan device with unrestricted floating depth"]
    fn depth_stencil_upload_keeps_planes_and_offset_exact() {
        let context = Arc::new(Context::new().expect("Vulkan device required"));
        let mut images = Images::new(&context).unwrap();
        assert!(images.ensure_format(1, 64, 64, vk::Format::D32_SFLOAT_S8_UINT));
        let mut memory = super::super::GlobalMemory::new(&context, 1024 * 1024 * 1024).unwrap();
        memory.allocate_pool(1, 0, 65536).unwrap();
        let mut source = vec![0xff; 65536];
        let (buffer, _) = memory.image_region(1, 24576, 20480).unwrap();
        for values in [[0.75_f32; 4], [-0.5, 0.0, 1.0, 1.5]] {
            let depth: Vec<_> = values.into_iter().flat_map(f32::to_le_bytes).collect();
            source[24576..40960].copy_from_slice(&depth.repeat(1024));
            memory.write_pool(1, 0, &source).unwrap();
            images.arena_transfer(1, buffer, 24576, true).unwrap();
            images.wait().unwrap();
            let (_, _, bytes) = images.readback(1).unwrap();
            assert_eq!(bytes, source[24576..45056]);
        }
    }
}
