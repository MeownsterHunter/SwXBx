//! The optional Vulkan execution context.
//!
//! This module contains only public Vulkan pieces and swxbx's own bookkeeping.
//! It deliberately does not describe or reproduce the target API.

use ash::{vk, Device, Entry, Instance};
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

mod command_workers;
mod commands;
pub mod formats;
pub mod graphics;
mod graphics_libraries;
mod graphics_modules;
pub mod image_enums;
pub mod image_layout;
pub(crate) mod images;
mod memory;
pub mod operations;
mod pipeline_disk;
mod pipeline_workers;
mod present;
mod readbacks;
mod recording_device;
#[cfg(test)]
mod shader_features_tests;
mod subgroups;
pub mod texture_transfer;
pub mod textures;
pub mod uniforms;
mod validation;
use images::{Image, Images};
pub(crate) use present::Unoffered;
use present::Window;
pub mod pipelines;
pub use memory::{GlobalMemory, PoolFault, TransferStatistics};
pub use subgroups::SubgroupFeatures;

/// Optional device capabilities available to translated modules.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShaderFeatures {
    /// Enabled subgroup controls and stage limits. Evidence: the public contract.
    pub subgroups: SubgroupFeatures,
    /// Whether storage-buffer 8-bit access is enabled.
    pub storage_buffer8_bit_access: bool,
    /// Whether storage-buffer 16-bit access is enabled.
    pub storage_buffer16_bit_access: bool,
    /// Whether 8-bit shader integer arithmetic is enabled.
    pub shader_int8: bool,
    /// Whether 16-bit shader integer arithmetic is enabled.
    pub shader_int16: bool,
    /// Enabled helper demotion. Spec: vulkan_core.h 1.4.363, lines 7446 and 7750.
    /// Core 1.3 or the EXT feature structure. Evidence: the public contract.
    pub demote_to_helper_invocation: bool,
    /// Enabled maximal reconvergence. Spec: vulkan_core.h 1.4.363, line 13110.
    /// KHR feature structure. Evidence: the public contract.
    pub maximal_reconvergence: bool,
    /// Whether storage images may be read without a declared format.
    pub storage_image_read_without_format: bool,
    /// Whether storage images may be written without a declared format.
    pub storage_image_write_without_format: bool,
    /// Whether vertex-pipeline stages may store to storage resources.
    pub vertex_pipeline_stores: bool,
    /// Whether the fragment stage may store to storage resources.
    pub fragment_stores: bool,
    /// Whether pipelines may include a geometry stage.
    pub geometry_shader: bool,
    /// Whether shaders may read draw parameters (BaseInstance, BaseVertex,
    /// DrawIndex): Vulkan 1.1 shaderDrawParameters, SPIR-V capability
    /// DrawParameters (4427).
    pub draw_parameters: bool,
}

impl ShaderFeatures {
    /// Check a translated module's declared and encoded device requirements.
    pub fn validate_translated_shader(
        self,
        shader: &crate::startup_cache::TranslatedShader,
    ) -> Result<(), &'static str> {
        self.validate_requirements(
            shader.requires_demote_to_helper_invocation,
            shader.requires_maximal_reconvergence,
        )?;
        self.validate_shader(&shader.words)
    }

    fn validate_requirements(self, demote: bool, reconvergence: bool) -> Result<(), &'static str> {
        if demote && !self.demote_to_helper_invocation {
            return Err("shaderDemoteToHelperInvocation is unavailable");
        }
        if reconvergence && !self.maximal_reconvergence {
            return Err("shaderMaximalReconvergence is unavailable");
        }
        Ok(())
    }

    /// Check the optional capabilities used by flat global memory modules.
    pub fn validate_global_shader(self, words: &[u32]) -> Result<(), &'static str> {
        self.validate_shader(words)
    }

    /// Check instruction-encoded device requirements. Evidence: the public contract.
    /// This is a feature check, not a replacement for SPIR-V validation.
    pub fn validate_shader(self, words: &[u32]) -> Result<(), &'static str> {
        if words.len() < 5 || words[0] != 0x0723_0203 {
            return Err("invalid SPIR-V header");
        }
        let mut offset = 5;
        while offset < words.len() {
            let count = (words[offset] >> 16) as usize;
            let opcode = words[offset] & 0xffff;
            if count == 0 || count > words.len() - offset {
                return Err("invalid SPIR-V instruction length");
            }
            if opcode == 17 {
                if count != 2 {
                    return Err("invalid OpCapability length");
                }
                match words[offset + 1] {
                    5379 => self.validate_requirements(true, false)?,
                    39 if !self.shader_int8 => return Err("shaderInt8 is unavailable"),
                    22 if !self.shader_int16 => return Err("shaderInt16 is unavailable"),
                    4448 if !self.storage_buffer8_bit_access => {
                        return Err("storageBuffer8BitAccess is unavailable")
                    }
                    4433 if !self.storage_buffer16_bit_access => {
                        return Err("storageBuffer16BitAccess is unavailable")
                    }
                    55 if !self.storage_image_read_without_format => {
                        return Err("shaderStorageImageReadWithoutFormat is unavailable")
                    }
                    56 if !self.storage_image_write_without_format => {
                        return Err("shaderStorageImageWriteWithoutFormat is unavailable")
                    }
                    2 if !self.geometry_shader => {
                        return Err("geometryShader is unavailable on this device")
                    }
                    4427 if !self.draw_parameters => {
                        return Err(
                            "shaderDrawParameters (SPIR-V capability DrawParameters) is \
                                    unavailable on this device",
                        )
                    }
                    _ => {}
                }
            }
            match opcode {
                5380 | 5381 => {
                    let expected = if opcode == 5380 { 1 } else { 3 };
                    if count != expected {
                        return Err("invalid helper invocation instruction length");
                    }
                    self.validate_requirements(true, false)?;
                }
                16 | 331 => {
                    if count < 3 {
                        return Err("invalid execution mode instruction length");
                    }
                    if words[offset + 2] == 6023 {
                        if count != 3 {
                            return Err("invalid MaximallyReconvergesKHR length");
                        }
                        self.validate_requirements(false, true)?;
                    }
                }
                _ => {}
            }
            offset += count;
        }
        Ok(())
    }
}

/// A device and its one graphics queue. Failure is represented as `None` so a
/// host without Vulkan keeps using the CPU executor.
pub struct Context {
    /// Loaded Vulkan entry points.
    pub entry: Entry,
    /// Owned Vulkan instance; the context destroys it on drop.
    pub instance: Instance,
    /// Physical device selected for execution.
    pub physical_device: vk::PhysicalDevice,
    /// Owned logical device; the context destroys it on drop.
    pub device: Device,
    /// Graphics queue selected for submission.
    pub queue: vk::Queue,
    /// Queue family index of the graphics queue.
    pub queue_family: u32,
    /// Optional shader capabilities enabled on the device.
    pub shader_features: ShaderFeatures,
    /// Indices of families for which the device created queues.
    pub queue_families: Vec<u32>,
    swapchain: bool,
    /// Whether graphics library compilation is enabled. Evidence: the public contract.
    pub graphics_pipeline_libraries: bool,
    /// Whether the device advertises linking suitable for draw-time use.
    pub graphics_pipeline_fast_linking: bool,
    /// Whether samplers may carry a border colour of their own.
    pub(crate) custom_border_colors: CustomBorderColors,
    pub(crate) push_descriptors: Option<ash::khr::push_descriptor::Device>,
    pub(crate) max_push_descriptors: u32,
    /// The physical device's limits, read once at creation.
    pub(crate) limits: vk::PhysicalDeviceLimits,
    command_workers: command_workers::Runtime,
    completion: Mutex<QueueCompletion>,
    validation: Option<validation::Validation>,
}

/// The device's custom border colours, which VK_EXT_custom_border_color
/// provides. SwXBx uses them only when the device samples them without a
/// format, so one sampler serves every view it meets.
pub(crate) enum CustomBorderColors {
    /// The device has no format-free custom border colours.
    Absent,
    /// Up to `samplers` samplers with a custom colour may exist at once;
    /// `live` counts those that do.
    Present {
        samplers: u32,
        live: std::sync::atomic::AtomicU32,
    },
}

impl CustomBorderColors {
    /// Count one more custom-colour sampler, or None when the device has
    /// none or its limit is reached.
    pub(crate) fn reserve(&self) -> Option<()> {
        use std::sync::atomic::Ordering;
        match self {
            Self::Absent => None,
            Self::Present { samplers, live } => live
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                    (n < *samplers).then_some(n + 1)
                })
                .ok()
                .map(drop),
        }
    }

    /// Count one custom-colour sampler destroyed.
    pub(crate) fn release(&self) {
        if let Self::Present { live, .. } = self {
            live.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Why the device cannot sample a custom border colour at all, or None
    /// when it can.
    pub(crate) fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Absent => {
                Some("this device has no custom border colours it samples without a format")
            }
            Self::Present { .. } => None,
        }
    }

    /// The most samplers with a custom colour that may exist at once.
    pub(crate) fn limit(&self) -> u32 {
        match self {
            Self::Absent => 0,
            Self::Present { samplers, .. } => *samplers,
        }
    }
}

/// A created device: its physical device, logical device, queue family,
/// shader features, graphics pipeline libraries (enabled, fast linking) and
/// custom border colours.
type CreatedDevice = (
    vk::PhysicalDevice,
    Device,
    u32,
    ShaderFeatures,
    (bool, bool),
    CustomBorderColors,
);

struct QueueCompletion {
    semaphore: vk::Semaphore,
    value: u64,
    position: u64,
}

struct ResolvedTexture {
    description: crate::api::TextureDescription,
    image: Option<Arc<image_layout::ResolvedImage>>,
}

struct CoherentImage {
    revision: u64,
    writes: u64,
    binding: (u64, u64, usize),
    packing: Option<crate::tiling::Layout>,
}

/// Why a texture copy did not happen.
#[derive(Debug)]
pub enum CopyFailure {
    /// The copy asks for something the images cannot hold; the reason says
    /// what.
    Refused(String),
    /// The device failed while copying.
    Device,
}

/// Vulkan images, presentation, pipelines, and lazily allocated arena storage.
pub struct Backend {
    pub(crate) command_contract: Option<operations::CommandContract>,
    pub(crate) execution: operations::Execution,
    pub(crate) first_draw: Option<graphics::FirstDrawContract>,
    /// The image each depth format token of the installed depth and raster
    /// contract draws with, resolved when the contract was installed.
    pub(crate) depth_images: Vec<(u64, formats::DepthImage)>,
    pub(crate) texture_contract: textures::TextureContract,
    pub(crate) image_contract: image_layout::ImageContract,
    pub(crate) copy_decoder: Option<image_layout::CopyDecoder>,
    pub(crate) blend_contract: Option<graphics::BlendContract>,
    samplers: crate::keymap::KeyMap<(u64, u32), textures::Sampler>,
    pub(crate) uniforms: uniforms::UniformBufferContract,
    pub(crate) graphics: graphics::GraphicsPipelines,
    windows: HashMap<u64, Window>,
    offscreen: Option<Image>,
    readbacks: readbacks::Readbacks,
    images: Images,
    pending_draws: Vec<graphics::PendingDraw>,
    draw_targets: Vec<u64>,
    draw_depth: Option<u64>,
    bindings: crate::keymap::KeyMap<u64, (u64, u64, usize)>,
    tiled: crate::keymap::KeyMap<u64, crate::tiling::Layout>,
    coherent_images: crate::keymap::KeyMap<u64, CoherentImage>,
    /// Depth targets whose image holds their only copy
    /// (`DepthImage::D24S8AsD32S8`), with the memory they cover: pool,
    /// offset and bytes.
    image_truth: crate::keymap::KeyMap<u64, ((u64, u64, u64), formats::HeldPlanes)>,
    /// Memory such a depth target covered when it was released: swxbx never
    /// stored the image's contents there, so the memory does not hold them
    /// until the host writes it. Another such target over exactly that
    /// memory takes it over.
    orphaned_truth: Vec<(u64, u64, u64)>,
    /// Each texture's interpretation under the image contract, kept with the
    /// description it was made from. A texture is resolved once, not per use.
    resolved: std::cell::RefCell<crate::keymap::KeyMap<u64, ResolvedTexture>>,
    /// Flat mapped arena, present after the first successful pool allocation.
    pub global_memory: Option<GlobalMemory>,
    context: Arc<Context>,
    scale: f32,
}

impl Drop for Backend {
    fn drop(&mut self) {
        // Retire every command user before field destruction starts.
        let _ = self.context.wait_queue();
    }
}

impl Backend {
    /// Create a backend with output scale at least one; return None on setup failure.
    pub fn new(scale: f32) -> Option<Self> {
        Self::from_context(Arc::new(Context::new()?), scale)
    }

    fn from_context(context: Arc<Context>, scale: f32) -> Option<Self> {
        let images = Images::new(&context)?;
        Some(Self {
            command_contract: None,
            execution: operations::Execution::new(&context)?,
            first_draw: None,
            depth_images: Vec::new(),
            texture_contract: textures::TextureContract::default(),
            image_contract: image_layout::ImageContract::default(),
            copy_decoder: None,
            blend_contract: None,
            samplers: Default::default(),
            uniforms: uniforms::UniformBufferContract::default(),
            graphics: graphics::GraphicsPipelines::new(&context)?,
            windows: HashMap::new(),
            offscreen: None,
            readbacks: readbacks::Readbacks::new(&context),
            images,
            resolved: Default::default(),
            pending_draws: Vec::new(),
            draw_targets: Vec::new(),
            draw_depth: None,
            bindings: Default::default(),
            tiled: Default::default(),
            coherent_images: Default::default(),
            image_truth: Default::default(),
            orphaned_truth: Vec::new(),
            context,
            scale: scale.max(1.0),
            global_memory: None,
        })
    }

    /// The caller has accepted Host's pointer and callback lifetime contract.
    pub(crate) fn with_host(host: &crate::Host) -> Option<Self> {
        if host.vulkan.is_null() {
            return Self::new(host.render_scale);
        }
        let surface_host = unsafe { &*host.vulkan };
        if surface_host.extension_count == 0
            || surface_host.extension_count > 64
            || surface_host.extensions.is_null()
        {
            return None;
        }
        let pointers = unsafe {
            std::slice::from_raw_parts(
                surface_host.extensions,
                surface_host.extension_count as usize,
            )
        };
        let mut extensions = Vec::new();
        for &pointer in pointers {
            if pointer.is_null() {
                return None;
            }
            extensions.push(unsafe { std::ffi::CStr::from_ptr(pointer) }.to_owned());
        }
        if !extensions
            .iter()
            .any(|name| name.as_c_str() == ash::khr::surface::NAME)
        {
            return None;
        }
        let context = Arc::new(Context::with_extensions(&extensions, true)?);
        Self::from_context(context, host.render_scale)
    }

    /// Allocate a pool in the lazy arena and return its guest GPU base, or None.
    pub fn allocate_pool(&mut self, key: u64, storage: u64, size: u64) -> Option<u64> {
        if self.global_memory.is_none() {
            self.global_memory = Some(GlobalMemory::new(
                &self.context,
                crate::global_memory::ARENA_SIZE,
            )?);
        }
        let memory = self.global_memory.as_mut()?;
        memory.track_submissions();
        memory.allocate_pool(key, storage, size)
    }

    /// Release a pool and its dependent textures; return whether it existed.
    pub fn release_pool(&mut self, key: u64) -> bool {
        let textures: Vec<_> = self
            .bindings
            .iter()
            .filter_map(|(&texture, &(pool, _, _))| (pool == key).then_some(texture))
            .collect();
        for texture in textures {
            self.release_texture(texture);
        }
        self.global_memory
            .as_mut()
            .is_some_and(|memory| memory.release_pool(key))
    }

    pub(crate) fn resolved_image(
        &self,
        d: &crate::api::TextureDescription,
    ) -> Option<image_layout::ResolvedImage> {
        self.image_contract.resolve(d)
    }
    /// The contract's interpretation of texture `key`, resolved when its
    /// description is first seen and reused while that description holds.
    pub(crate) fn texture_image(
        &self,
        key: u64,
        d: &crate::api::TextureDescription,
    ) -> Option<Arc<image_layout::ResolvedImage>> {
        let mut resolved = self.resolved.borrow_mut();
        match resolved.get(&key) {
            Some(known) if known.description == *d => known.image.clone(),
            _ => {
                let image = self.image_contract.resolve(d).map(Arc::new);
                resolved.insert(
                    key,
                    ResolvedTexture {
                        description: d.clone(),
                        image: image.clone(),
                    },
                );
                image
            }
        }
    }
    /// Storage bytes of texture `key` under the image contract.
    pub(crate) fn texture_bytes(
        &self,
        key: u64,
        d: &crate::api::TextureDescription,
    ) -> Option<usize> {
        let r = self.texture_image(key, d)?;
        Some(match r.storage {
            image_layout::Storage::Linear => r.packing.linear_size(),
            image_layout::Storage::Tiled(_) => r.packing.tiled_size(),
        })
    }
    pub(crate) fn has_image_contract(&self) -> bool {
        !self.image_contract.rules.is_empty()
    }
    pub(crate) fn texture_storage_size(&self, d: &crate::api::TextureDescription) -> Option<usize> {
        let r = self.image_contract.resolve(d)?;
        Some(match r.storage {
            image_layout::Storage::Linear => r.packing.linear_size(),
            image_layout::Storage::Tiled(_) => r.packing.tiled_size(),
        })
    }

    /// Install validated image rules before any arena image bindings exist.
    pub fn set_image_contract(&mut self, contract: image_layout::ImageContract) -> bool {
        if contract.validate().is_err() || !self.bindings.is_empty() {
            return false;
        }
        self.image_contract = contract;
        self.resolved.get_mut().clear();
        true
    }

    /// Return the shared execution context owned by this backend.
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Check a translated shader against this device and create its Vulkan
    /// module once. This proves driver acceptance without inventing the
    /// pipeline state that a shader-cache record does not contain.
    pub fn validate_shader_module(
        &self,
        shader: &crate::startup_cache::TranslatedShader,
    ) -> Result<(), String> {
        self.context
            .shader_features
            .validate_translated_shader(shader)
            .map_err(str::to_owned)?;
        let module = self.context.create_global_shader_module(&shader.words)?;
        // SAFETY: this backend owns the device and module, and no command or
        // pipeline receives the module before its destruction here.
        unsafe {
            self.context.device.destroy_shader_module(module, None);
        }
        Ok(())
    }

    /// Create or reuse a color or depth image for the key; return false on failure.
    pub fn ensure(&mut self, key: u64, width: u64, height: u64, depth: bool) -> bool {
        self.images.ensure(key, width, height, depth)
    }

    /// The base level uses swxbx's existing four-byte, tightly packed storage
    /// choice. Guest format integers remain uninterpreted.
    pub(crate) fn ensure_texture(
        &mut self,
        key: u64,
        description: &crate::api::TextureDescription,
        depth: bool,
    ) -> bool {
        if !self.image_contract.rules.is_empty() {
            let Some(resolved) = self.texture_image(key, description) else {
                return false;
            };
            if depth && resolved.descriptor.format != vk::Format::D32_SFLOAT {
                return false;
            }
            let bytes = match resolved.storage {
                image_layout::Storage::Linear => resolved.packing.linear_size(),
                image_layout::Storage::Tiled(_) => resolved.packing.tiled_size(),
            };
            if description.pool == 0
                || self
                    .global_memory
                    .as_ref()
                    .and_then(|m| m.image_region(description.pool, description.pool_offset, bytes))
                    .is_none()
            {
                return false;
            }
            if !self.images.ensure_descriptor(key, &resolved.descriptor) {
                return false;
            }
            self.bindings
                .insert(key, (description.pool, description.pool_offset, bytes));
            if matches!(resolved.storage, image_layout::Storage::Tiled(_)) {
                if self.tiled.get(&key) != Some(&resolved.packing) {
                    self.tiled.insert(key, resolved.packing.clone());
                }
            } else {
                self.tiled.remove(&key);
            }
            return true;
        }
        let width = description.width.max(1);
        let height = description.height.max(1);
        if description.depth > 1
            || (description.stride != 0 && description.stride != width.saturating_mul(4))
        {
            return false;
        }
        let Some(bytes) = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| usize::try_from(n).ok())
        else {
            return false;
        };
        if description.pool != 0 {
            let Some(memory) = self.global_memory.as_ref() else {
                return false;
            };
            if memory
                .image_region(description.pool, description.pool_offset, bytes)
                .is_none()
            {
                return false;
            }
        }
        let created = self.images.format(key).is_none();
        if !self.ensure(key, width, height, depth) {
            return false;
        }
        // An image with no memory behind it holds only what swxbx wrote,
        // so a new one starts as zero, never undefined. Images over memory
        // load it before their first use.
        if created && description.pool == 0 {
            let cleared = if depth {
                self.images.clear_depth(key, Some(0.0), None)
            } else {
                self.images.clear_color(key, [0.0; 4], 0xf)
            };
            if cleared.is_none() {
                return false;
            }
        }
        if description.pool != 0 {
            self.bindings
                .insert(key, (description.pool, description.pool_offset, bytes));
        } else {
            self.bindings.remove(&key);
        }
        true
    }

    /// Create the typed image used by an image-to-image copy. Colour images
    /// use the image contract; depth/stencil images use the depth contract,
    /// whose planar memory layout is not an ordinary texel-block layout.
    pub(crate) fn ensure_copy_texture(
        &mut self,
        key: u64,
        description: &crate::api::TextureDescription,
    ) -> Result<(), String> {
        let is_2d = self
            .first_draw
            .as_ref()
            .is_some_and(|contract| contract.target_2d == description.target);
        if let Some(image) = is_2d
            .then(|| self.depth_image(description.format))
            .flatten()
        {
            self.depth_storage(key, description, image.guest())?;
            return self
                .ensure_depth_stencil(key, description, image)
                .then_some(())
                .ok_or_else(|| "the mapped depth image could not be materialized".into());
        }
        if self.resolved_image(description).is_some() {
            return self
                .ensure_texture(key, description, false)
                .then_some(())
                .ok_or_else(|| "the image contract could not materialize the texture".into());
        }
        Err(format!(
            "format token {} is neither an image-contract format nor a mapped 2D depth format",
            description.format
        ))
    }

    /// Check that a depth target's memory is what swxbx draws depth over:
    /// tightly packed planes (stride 0) in linear storage, in the format the
    /// image contract, when there is one, stores its token as. A tiled depth
    /// target is refused; swxbx has no tiled depth path. Source: the public contract.
    pub(crate) fn depth_storage(
        &self,
        key: u64,
        description: &crate::api::TextureDescription,
        format: vk::Format,
    ) -> Result<(), String> {
        if description.stride != 0 {
            return Err(format!(
                "its stride is {}; swxbx draws depth over tightly packed planes (stride 0)",
                description.stride
            ));
        }
        let Some(stored) = self.texture_image(key, description) else {
            return Ok(());
        };
        if stored.descriptor.format != format {
            return Err(format!(
                "the image contract stores format token {} as {}, but the depth and raster \
                 contract draws with it as {}",
                description.format,
                formats::Name(stored.descriptor.format),
                formats::Name(format)
            ));
        }
        if matches!(stored.storage, image_layout::Storage::Tiled(_)) {
            return Err(format!(
                "the image contract stores format token {} tiled; swxbx draws depth only over \
                 linear memory",
                description.format
            ));
        }
        Ok(())
    }

    /// Create or reuse the image a depth-stencil clear writes, and return its
    /// format. With a depth and raster contract that is the format its depth
    /// table maps the texture's token to, the image a draw uses; without one,
    /// swxbx clears depth as D32_SFLOAT. Source: the public contract.
    pub(crate) fn ensure_depth_clear(
        &mut self,
        key: u64,
        description: &crate::api::TextureDescription,
    ) -> Result<vk::Format, (crate::Status, String)> {
        let unsupported = |why: String| (crate::Status::Unimplemented, why);
        let placed = |format: vk::Format| {
            (
                crate::Status::BadArgument,
                format!(
                    "no device image in {} over its memory ({}x{}, stride {}, at offset \
                     {} of memory pool {})",
                    formats::Name(format),
                    description.width,
                    description.height,
                    description.stride,
                    description.pool_offset,
                    description.pool
                ),
            )
        };
        if self
            .first_draw
            .as_ref()
            .and_then(|c| c.depth_raster.as_ref())
            .is_none()
        {
            let format = vk::Format::D32_SFLOAT;
            return if self.ensure_texture(key, description, true) {
                Ok(format)
            } else {
                Err(placed(format))
            };
        }
        let image = self.depth_image(description.format).ok_or_else(|| {
            unsupported(format!(
                "its format token {} is not mapped to exactly one depth format by the depth and \
                 raster contract",
                description.format
            ))
        })?;
        self.depth_storage(key, description, image.guest())
            .map_err(unsupported)?;
        if self.ensure_depth_stencil(key, description, image) {
            Ok(image.format())
        } else {
            Err(placed(image.format()))
        }
    }

    /// The image a depth format token draws with under the installed depth
    /// and raster contract.
    pub(crate) fn depth_image(&self, token: u64) -> Option<formats::DepthImage> {
        let mut found = self.depth_images.iter().filter(|(t, _)| *t == token);
        match (found.next(), found.next()) {
            (Some(&(_, image)), None) => Some(image),
            _ => None,
        }
    }

    /// Why memory `pool`, `offset`, `bytes` cannot be read or written for
    /// texture `key`: it overlaps a depth target whose image holds its only
    /// copy. None when it does not.
    fn image_truth_fault(&self, key: u64, pool: u64, offset: u64, bytes: u64) -> Option<String> {
        let overlaps = |&(p, o, n): &(u64, u64, u64)| {
            p == pool
                && offset.checked_add(bytes).is_none_or(|end| o < end)
                && o.checked_add(n).is_none_or(|end| offset < end)
        };
        let range = format!(
            "memory pool {pool} bytes {offset:#x}..{:#x}",
            offset.saturating_add(bytes)
        );
        let live = self
            .image_truth
            .iter()
            .find(|&(&owner, (region, _))| owner != key && overlaps(region))
            .map(|(owner, _)| {
                format!(
                    "{range} overlap depth target texture {owner}, a D24_UNORM_S8_UINT target \
                     swxbx draws in a D32_SFLOAT_S8_UINT image; its contents live only in that \
                     image, not in memory"
                )
            });
        live.or_else(|| {
            self.orphaned_truth.iter().find(|r| overlaps(r)).map(|_| {
                format!(
                    "{range} overlap memory a released D24_UNORM_S8_UINT depth target covered; \
                     swxbx drew it in a D32_SFLOAT_S8_UINT image and never stored its contents \
                     there"
                )
            })
        })
    }

    /// The host wrote guest memory `address`, `size`: what it wrote is what
    /// that memory holds, so released depth images no longer stand for it.
    pub(crate) fn host_wrote(&mut self, address: u64, size: u64) {
        if self.orphaned_truth.is_empty() {
            return;
        }
        let Some(memory) = self.global_memory.as_ref() else {
            return;
        };
        for (pool, start, bytes) in memory.pool_spans(address, size) {
            let end = start.saturating_add(bytes);
            self.orphaned_truth = std::mem::take(&mut self.orphaned_truth)
                .into_iter()
                .flat_map(|(p, o, n)| {
                    let region_end = o.saturating_add(n);
                    if p != pool || end <= o || region_end <= start {
                        return vec![(p, o, n)];
                    }
                    [(o, start.min(region_end)), (end.max(o), region_end)]
                        .into_iter()
                        .filter(|(from, to)| from < to)
                        .map(|(from, to)| (p, from, to - from))
                        .collect()
                })
                .collect();
        }
    }

    /// Refuse by name a use of texture `key` other than drawing to or
    /// clearing it when its image holds its only copy.
    fn refuse_image_truth(&self, key: u64, used: &str) -> Option<()> {
        if self.image_truth.contains_key(&key) {
            crate::unsupported::refuse(
                crate::Status::Unimplemented,
                format!(
                    "depth target texture {key} is a D24_UNORM_S8_UINT target swxbx draws in a \
                     D32_SFLOAT_S8_UINT image, which holds its only copy; it cannot be {used}"
                ),
            );
            return None;
        }
        Some(())
    }

    /// Create or reuse the depth image of a draw or depth clear, over the
    /// caller format's depth plane and, for a format with stencil, the
    /// stencil plane after it. A widened image does not load or store that
    /// memory; it marks it as held by the image. Source: the public contract., 0066.
    pub(crate) fn ensure_depth_stencil(
        &mut self,
        key: u64,
        description: &crate::api::TextureDescription,
        image: formats::DepthImage,
    ) -> bool {
        let Some(planes) = formats::depth_planes(image.guest()) else {
            return false;
        };
        let Some(bytes) = description
            .width
            .checked_mul(description.height)
            .map(|texels| planes.size(texels))
            .and_then(|n| usize::try_from(n).ok())
        else {
            return false;
        };
        if description.width == 0
            || description.height == 0
            || description.pool == 0
            || description.stride != 0
        {
            return false;
        }
        if self
            .global_memory
            .as_ref()
            .and_then(|m| m.image_region(description.pool, description.pool_offset, bytes))
            .is_none()
        {
            return false;
        }
        let region = (description.pool, description.pool_offset, bytes as u64);
        if image == formats::DepthImage::D24S8AsD32S8 {
            self.orphaned_truth.retain(|r| *r != region);
        }
        if let Some(why) = self.image_truth_fault(key, region.0, region.1, region.2) {
            crate::unsupported::refuse(
                crate::Status::Unimplemented,
                format!("depth target texture {key}: {why}"),
            );
            return false;
        }
        if !self
            .images
            .ensure_format(key, description.width, description.height, image.format())
        {
            return false;
        }
        match image {
            formats::DepthImage::Exact(_) => {
                self.image_truth.remove(&key);
                self.bindings
                    .insert(key, (description.pool, description.pool_offset, bytes));
            }
            formats::DepthImage::D24S8AsD32S8 => {
                self.bindings.remove(&key);
                self.coherent_images.remove(&key);
                let held = match self.image_truth.get(&key) {
                    Some(&(held_region, held)) if held_region == region => held,
                    _ => formats::HeldPlanes::Unwritten,
                };
                self.image_truth.insert(key, (region, held));
            }
        }
        true
    }

    fn sync_texture(&mut self, key: u64, load: bool) -> Option<()> {
        #[cfg(feature = "draw-metrics")]
        let _span = crate::draw_metrics::PipelineSpan::new(12);
        if let Some(&(pool, offset, bytes)) = self.bindings.get(&key) {
            if let Some(why) = self.image_truth_fault(key, pool, offset, bytes as u64) {
                crate::unsupported::refuse(
                    crate::Status::Unimplemented,
                    format!("texture {key}: {why}"),
                );
                return None;
            }
            let binding = (pool, offset, bytes);
            let packing = self.tiled.get(&key);
            let (buffer, offset) = self
                .global_memory
                .as_ref()?
                .image_region(pool, offset, bytes)?;
            if load
                && self.coherent_images.get(&key).is_some_and(|stored| {
                    self.global_memory.as_ref().is_some_and(|memory| {
                        memory.unwritten_since(offset..offset + bytes as u64, stored.writes)
                    }) && stored.binding == binding
                        && stored.packing.as_ref() == packing
                })
            {
                return Some(());
            }
            if let Some(packing) = packing {
                let address = unsafe {
                    self.context.device.get_buffer_device_address(
                        &vk::BufferDeviceAddressInfo::default().buffer(buffer),
                    )
                }
                .checked_add(offset)?;
                self.images.tiled_transfer(key, address, packing, load)?;
            } else {
                self.images.arena_transfer(key, buffer, offset, load)?;
            }
            {
                let memory = self.global_memory.as_ref()?;
                memory.submitted(offset, bytes as u64, !load, self.images.submission()?);
                if let Some(revision) = memory.revision() {
                    let writes = memory.write_count();
                    let stored = self
                        .coherent_images
                        .entry(key)
                        .or_insert_with(|| CoherentImage {
                            revision,
                            writes,
                            binding,
                            packing: packing.cloned(),
                        });
                    stored.revision = revision;
                    stored.writes = writes;
                    if stored.binding != binding || stored.packing.as_ref() != packing {
                        stored.binding = binding;
                        stored.packing = packing.cloned();
                    }
                }
            }
        }
        Some(())
    }

    pub(crate) fn sampled_texture(
        &mut self,
        key: u64,
        description: &crate::api::TextureDescription,
    ) -> Option<vk::ImageView> {
        self.refuse_image_truth(key, "sampled")?;
        if !self.ensure_texture(key, description, false) {
            return None;
        }
        self.sync_texture(key, true)?;
        if self.image_contract.swizzles.is_empty() {
            self.images.sampled(key)
        } else {
            let components = self.texture_image(key, description)?.components;
            self.images.sampled_with_components(key, components)
        }
    }

    /// Load a guest image from the arena and return a view for shader reads
    /// and writes. A raw view format must be the image's own format; the
    /// image holds no other view format.
    pub(crate) fn storage_texture(
        &mut self,
        key: u64,
        description: &crate::api::TextureDescription,
        format: crate::surfaces::SurfaceFormat,
    ) -> Option<vk::ImageView> {
        if !self.ensure_texture(key, description, false) {
            return None;
        }
        if let crate::surfaces::SurfaceFormat::Raw(raw) = format {
            if self.images.format(key)? != raw.vk() {
                return None;
            }
        }
        self.sync_texture(key, true)?;
        self.images.storage(key)
    }

    /// Store guest images a shader wrote back to the arena.
    pub(crate) fn store_textures(&mut self, keys: &[u64]) -> Option<()> {
        for &key in keys {
            self.sync_texture(key, false)?;
        }
        Some(())
    }

    pub(crate) fn sampler(
        &mut self,
        pool: u64,
        id: u32,
        key: textures::SamplerKey,
    ) -> Option<vk::Sampler> {
        if self.samplers.get(&(pool, id)).is_none_or(|s| s.key != key) {
            if self.samplers.contains_key(&(pool, id)) {
                self.context.wait_queue()?;
            }
            self.graphics.invalidate_descriptors();
            self.samplers
                .insert((pool, id), textures::Sampler::new(&self.context, key)?);
        }
        Some(self.samplers.get(&(pool, id))?.handle)
    }

    /// Create a sampler through the recorded-state mapping and device checks.
    pub fn mapped_sampler(
        &mut self,
        pool: u64,
        id: u32,
        description: &textures::SamplerDescription,
        contract: &textures::TextureContract,
    ) -> Option<vk::Sampler> {
        contract.validate().ok()?;
        let key = textures::SamplerKey::new(description, contract)?;
        self.sampler(pool, id, key)
    }

    /// Allocate or reuse a typed host image. Compressed payloads remain opaque.
    pub fn ensure_image(&mut self, key: u64, descriptor: &image_layout::ImageDescriptor) -> bool {
        self.images.ensure_descriptor(key, descriptor)
    }

    /// Return a whole-resource view after making transfers visible to shaders.
    pub fn sampled_image(&mut self, key: u64) -> Option<vk::ImageView> {
        self.sampled_image_with_components(key, vk::ComponentMapping::default())
    }

    /// Return a sampled view with explicit host component selection.
    pub fn sampled_image_with_components(
        &mut self,
        key: u64,
        components: vk::ComponentMapping,
    ) -> Option<vk::ImageView> {
        if [components.r, components.g, components.b, components.a]
            .iter()
            .any(|c| !(0..=6).contains(&c.as_raw()))
        {
            return None;
        }
        let view = self.images.sampled_with_components(key, components)?;
        self.images.wait()?;
        Some(view)
    }

    pub(crate) fn has_swizzle_contract(&self) -> bool {
        !self.image_contract.swizzles.is_empty()
    }

    /// Upload a whole typed image in packed-linear order; return false on failure.
    pub fn upload_image(&mut self, key: u64, bytes: &[u8]) -> bool {
        if self.refuse_image_truth(key, "uploaded to").is_none() {
            return false;
        }
        self.coherent_images.remove(&key);
        self.images.upload(key, bytes).is_some()
    }

    /// Transfer tiled pool storage directly through its device-visible arena buffer.
    pub fn load_tiled(
        &mut self,
        key: u64,
        pool: u64,
        offset: u64,
        packing: &crate::tiling::Layout,
    ) -> bool {
        self.transfer_tiled(key, pool, offset, packing, true)
            .is_some()
    }
    /// Write an image into tiled arena storage; return false for invalid ranges or transfers.
    pub fn store_tiled(
        &mut self,
        key: u64,
        pool: u64,
        offset: u64,
        packing: &crate::tiling::Layout,
    ) -> bool {
        self.transfer_tiled(key, pool, offset, packing, false)
            .is_some()
    }
    /// Batch bounded transfers in one reusable command recording.
    pub fn load_tiled_batch(
        &mut self,
        resources: &[(u64, u64, u64, &crate::tiling::Layout)],
    ) -> bool {
        self.transfer_tiled_batch(resources, true).is_some()
    }
    /// Store up to 256 image transfers with shared scratch and submission.
    pub fn store_tiled_batch(
        &mut self,
        resources: &[(u64, u64, u64, &crate::tiling::Layout)],
    ) -> bool {
        self.transfer_tiled_batch(resources, false).is_some()
    }
    fn transfer_tiled_batch(
        &mut self,
        resources: &[(u64, u64, u64, &crate::tiling::Layout)],
        load: bool,
    ) -> Option<()> {
        if resources.is_empty() || resources.len() > 256 {
            return None;
        }
        let memory = self.global_memory.as_ref()?;
        let mut ranges = Vec::with_capacity(resources.len());
        for &(key, pool, offset, packing) in resources {
            let (buffer, at) = memory.image_region(pool, offset, packing.tiled_size())?;
            let address = unsafe {
                self.context.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(buffer),
                )
            }
            .checked_add(at)?;
            ranges.push((key, address, packing));
        }
        self.images.tiled_batch(&ranges, load)?;
        if load {
            for &(key, ..) in resources {
                self.coherent_images.remove(&key);
            }
        }
        for &(_, pool, offset, packing) in resources {
            let size = packing.tiled_size();
            let (_, at) = memory.image_region(pool, offset, size)?;
            memory.submitted(at, size as u64, !load, self.images.submission()?);
        }
        Some(())
    }

    fn transfer_tiled(
        &mut self,
        key: u64,
        pool: u64,
        offset: u64,
        packing: &crate::tiling::Layout,
        load: bool,
    ) -> Option<()> {
        let (buffer, at) =
            self.global_memory
                .as_ref()?
                .image_region(pool, offset, packing.tiled_size())?;
        if !at.is_multiple_of(4) {
            return None;
        }
        let address = unsafe {
            self.context
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        }
        .checked_add(at)?;
        self.images.tiled_transfer(key, address, packing, load)?;
        if load {
            self.coherent_images.remove(&key);
        }
        self.global_memory.as_ref()?.submitted(
            at,
            packing.tiled_size() as u64,
            !load,
            self.images.submission()?,
        );
        Some(())
    }

    /// Copy packed linear arena images in one submission, without conversion.
    pub fn load_linear_batch(
        &mut self,
        resources: &[(u64, u64, u64, &crate::tiling::Layout)],
    ) -> bool {
        self.transfer_linear_batch(resources, true).is_some()
    }

    /// Copy images to packed linear arena storage in one submission.
    pub fn store_linear_batch(
        &mut self,
        resources: &[(u64, u64, u64, &crate::tiling::Layout)],
    ) -> bool {
        self.transfer_linear_batch(resources, false).is_some()
    }

    fn transfer_linear_batch(
        &mut self,
        resources: &[(u64, u64, u64, &crate::tiling::Layout)],
        load: bool,
    ) -> Option<()> {
        if resources.is_empty() || resources.len() > 256 {
            return None;
        }
        let memory = self.global_memory.as_ref()?;
        let mut ranges = Vec::with_capacity(resources.len());
        for &(key, pool, offset, packing) in resources {
            let (buffer, at) = memory.image_region(pool, offset, packing.linear_size())?;
            ranges.push((key, buffer, at, packing));
        }
        self.images.linear_batch(&ranges, load)?;
        if load {
            for &(key, ..) in resources {
                self.coherent_images.remove(&key);
            }
        }
        for &(_, pool, offset, packing) in resources {
            let size = packing.linear_size();
            let (_, at) = memory.image_region(pool, offset, size)?;
            memory.submitted(at, size as u64, !load, self.images.submission()?);
        }
        Some(())
    }

    /// Complete all transfers before CPU access to shared arena storage.
    pub fn wait_transfers(&self) -> bool {
        self.images.wait().is_some()
    }

    /// Remove the image and arena binding for a texture key.
    pub fn release_texture(&mut self, key: u64) {
        self.coherent_images.remove(&key);
        self.graphics.invalidate_descriptors();
        self.images.remove(key);
        self.bindings.remove(&key);
        self.tiled.remove(&key);
        if let Some((region, _)) = self.image_truth.remove(&key) {
            self.orphaned_truth.push(region);
        }
    }

    /// Clear selected RGBA channels; mask bits zero through three select channels.
    /// `color` holds the clear's four 32-bit words, read in the image
    /// format's numeric class. A clear swxbx cannot do for the image's
    /// format is refused by name with `Status::Unimplemented`; other
    /// failures are refused by name with `Status::InternalError`.
    pub fn clear_color(
        &mut self,
        key: u64,
        color: [f32; 4],
        mask: u32,
    ) -> Result<(), crate::Status> {
        let internal = |what: &str| {
            crate::unsupported::refuse(
                crate::Status::InternalError,
                format!("clear texture {key}: the device could not {what}"),
            )
        };
        self.sync_texture(key, true)
            .ok_or_else(|| internal("bring the image up to date"))?;
        let format = self
            .images
            .format(key)
            .ok_or_else(|| internal("find the image"))?;
        formats::color_clear(format, color, mask)
            .map_err(|why| crate::unsupported::refuse(crate::Status::Unimplemented, why))?;
        (|| {
            self.images.clear_color(key, color, mask)?;
            self.sync_texture(key, false)
        })()
        .ok_or_else(|| internal("record the clear"))
    }

    /// Clear the depth and, when `stencil` is given, the stencil of a depth
    /// image to the given values.
    pub fn clear_depth(&mut self, key: u64, depth: Option<f32>, stencil: Option<u32>) -> bool {
        (|| {
            self.sync_texture(key, true)?;
            self.images.clear_depth(key, depth, stencil)?;
            if let Some((_, held)) = self.image_truth.get_mut(&key) {
                *held = held.cleared(depth, stencil);
            }
            self.sync_texture(key, false)
        })()
        .is_some()
    }

    /// Why a draw that tests depth and stencil as given cannot read depth
    /// target `key`: it is drawn in a widened image no clear has written the
    /// plane of. None when it can.
    pub(crate) fn unheld_depth(
        &self,
        key: u64,
        depth_test: bool,
        stencil_test: bool,
    ) -> Option<String> {
        let &(_, held) = self.image_truth.get(&key)?;
        held.missing(depth_test, stencil_test).map(|plane| {
            format!(
                "depth target texture {key} is a D24_UNORM_S8_UINT target swxbx draws in a \
                 D32_SFLOAT_S8_UINT image, which cannot be loaded from memory, and the draw tests \
                 its {plane} before a clear wrote it"
            )
        })
    }

    pub(crate) fn set_parallel_recording(&mut self, enabled: bool) {
        self.images.set_parallel_recording(enabled);
    }

    pub(crate) fn reserve_draws(&mut self, count: usize) {
        self.pending_draws
            .reserve(count.saturating_sub(self.pending_draws.len()));
    }

    pub(crate) fn queue_draw(
        &mut self,
        textures: &[u64],
        depth: Option<u64>,
        draw: graphics::PendingDraw,
    ) -> Option<()> {
        if !self.pending_draws.is_empty()
            && (textures != self.draw_targets || depth != self.draw_depth)
        {
            self.finish_draws()?;
        }
        if self.pending_draws.is_empty() {
            self.draw_targets.clear();
            self.draw_targets.extend_from_slice(textures);
            self.draw_depth = depth;
        }
        self.pending_draws.push(draw);
        Some(())
    }

    pub(crate) fn finish_draws(&mut self) -> Option<()> {
        if self.pending_draws.is_empty() {
            return Some(());
        }
        // Take reusable vectors to release the borrow across resource synchronization.
        let mut draws = std::mem::take(&mut self.pending_draws);
        let targets = std::mem::take(&mut self.draw_targets);
        let depth = self.draw_depth;
        let result = (|| {
            for &texture in &targets {
                self.sync_texture(texture, true)?;
            }
            if let Some(depth) = depth {
                self.sync_texture(depth, true)?;
            }
            let query = self.execution.occlusion_query();
            self.images
                .draw(&targets, depth, self.global_memory.as_ref()?, &draws, query)?;
            let memory = self.global_memory.as_ref()?;
            memory.submitted_all(false, self.images.submission()?);
            for draw in &draws {
                for info in draw.pipeline.write_buffers(&draw.state) {
                    memory.submitted(info.offset, info.range, true, self.images.submission()?);
                }
            }
            if draws.iter().any(|d| d.pipeline.global_writes()) {
                memory.submitted_all(true, self.images.submission()?);
            }
            self.execution
                .collect_occlusion(self.images.submission()?)?;
            for &texture in &targets {
                self.sync_texture(texture, false)?;
            }
            for draw in &draws {
                for &texture in &draw.state.storage_images {
                    self.sync_texture(texture, false)?;
                }
            }
            if let Some(depth) = depth {
                self.sync_texture(depth, false)?;
            }
            // Start the group on the GPU while the next one is recorded.
            self.context.command_workers.flush_direct();
            Some(())
        })();
        draws.clear();
        self.pending_draws = draws;
        self.draw_targets = targets;
        result
    }

    /// Return unscaled RGBA image bytes and dimensions, or None on failure.
    pub fn readback(&mut self, key: u64) -> Option<(u32, u32, Vec<u8>)> {
        self.refuse_image_truth(key, "read back")?;
        self.sync_texture(key, true)?;
        self.images.readback(key)
    }

    /// Upload tightly packed four-byte color texels into the keyed image.
    pub fn upload(&mut self, key: u64, data: &[u8], width: u64, height: u64) -> bool {
        (|| {
            self.refuse_image_truth(key, "uploaded to")?;
            if !self.ensure(key, width, height, false) {
                return None;
            }
            self.images.upload(key, data)?;
            self.sync_texture(key, false)
        })()
        .is_some()
    }

    pub(crate) fn copy_from_arena(
        &mut self,
        key: u64,
        pool: u64,
        offset: u64,
        size: usize,
    ) -> bool {
        (|| {
            self.refuse_image_truth(key, "copied to")?;
            if let Some(why) = self.image_truth_fault(key, pool, offset, size as u64) {
                crate::unsupported::refuse(
                    crate::Status::Unimplemented,
                    format!("copy to texture {key}: {why}"),
                );
                return None;
            }
            let (buffer, offset) = self
                .global_memory
                .as_ref()?
                .image_region(pool, offset, size)?;
            self.images.arena_transfer(key, buffer, offset, true)?;
            self.global_memory.as_ref()?.submitted(
                offset,
                size as u64,
                false,
                self.images.submission()?,
            );
            self.sync_texture(key, false)
        })()
        .is_some()
    }

    /// Copy equal-format image regions. A region the images cannot hold is
    /// refused with its reason before any device work.
    pub fn copy_region(
        &mut self,
        destination: u64,
        source: u64,
        from: image_layout::ImageRegion,
        to: image_layout::ImageRegion,
    ) -> Result<(), CopyFailure> {
        if let Some(fault) = self.images.copy_fault(destination, source, from, to) {
            return Err(CopyFailure::Refused(fault));
        }
        let source_held = self.image_truth.get(&source).map(|(_, held)| *held);
        let destination_held = self.image_truth.get(&destination).map(|(_, held)| *held);
        match (source_held, destination_held) {
            (Some(held), Some(_))
                if self.images.complete_region(source, from)
                    && self.images.complete_region(destination, to) =>
            {
                self.images
                    .copy_region(destination, source, from, to)
                    .ok_or(CopyFailure::Device)?;
                if let Some((_, destination_held)) = self.image_truth.get_mut(&destination) {
                    *destination_held = held;
                }
                return Ok(());
            }
            (Some(_), _) | (_, Some(_)) => {
                let key = if source_held.is_some() {
                    source
                } else {
                    destination
                };
                return Err(CopyFailure::Refused(format!(
                    "depth target texture {key} is a D24_UNORM_S8_UINT target swxbx draws in a \
                     D32_SFLOAT_S8_UINT image, which holds its only copy; only a complete copy to \
                     another such image is representable"
                )));
            }
            (None, None) => {}
        }
        (|| {
            self.sync_texture(source, true)?;
            self.sync_texture(destination, true)?;
            self.images.copy_region(destination, source, from, to)?;
            self.sync_texture(destination, false)
        })()
        .ok_or(CopyFailure::Device)
    }
    /// Blit compatible image regions with the selected filter; return false on failure.
    pub fn blit_region(
        &mut self,
        destination: u64,
        source: u64,
        from: image_layout::BlitRegion,
        to: image_layout::BlitRegion,
        filter: vk::Filter,
    ) -> bool {
        (|| {
            self.refuse_image_truth(source, "blitted")?;
            self.refuse_image_truth(destination, "blitted to")?;
            self.sync_texture(source, true)?;
            self.sync_texture(destination, true)?;
            self.images
                .blit_region(destination, source, from, to, filter)?;
            self.sync_texture(destination, false)
        })()
        .is_some()
    }

    /// Copy a whole image between existing compatible images; return false on failure.
    pub fn copy(&mut self, destination: u64, source: u64) -> bool {
        (|| {
            self.refuse_image_truth(source, "copied")?;
            self.refuse_image_truth(destination, "copied to")?;
            self.sync_texture(source, true)?;
            self.images.copy(destination, source)?;
            self.sync_texture(destination, false)
        })()
        .is_some()
    }

    pub(crate) fn open_window(
        &mut self,
        host: &crate::Host,
        key: u64,
        native_window: u64,
    ) -> Option<()> {
        use ash::vk::Handle;
        if host.vulkan.is_null() {
            return Some(());
        }
        if !self.context.swapchain || self.windows.contains_key(&key) {
            return None;
        }
        let surface_host = unsafe { &*host.vulkan };
        let raw = unsafe {
            (surface_host.create_surface)(
                host.user,
                self.context.instance.handle().as_raw(),
                key,
                native_window,
            )
        };
        if raw == 0 {
            return None;
        }
        let window = Window::new(&self.context, vk::SurfaceKHR::from_raw(raw))?;
        self.windows.insert(key, window);
        Some(())
    }

    /// Ask the surface of a window just opened whether it offers the mode.
    pub(crate) fn window_takes(
        &self,
        key: u64,
        mode: crate::PresentationMode,
    ) -> Result<(), present::Unoffered> {
        self.windows
            .get(&key)
            .ok_or(present::Unoffered::CannotBeAsked)?
            .offers(mode)
    }

    /// Whether any native window is open.
    pub(crate) fn has_windows(&self) -> bool {
        !self.windows.is_empty()
    }

    /// An open window whose surface does not offer the mode or cannot be asked.
    pub(crate) fn window_lacking(
        &self,
        mode: crate::PresentationMode,
    ) -> Option<(u64, present::Unoffered)> {
        self.windows
            .iter()
            .find_map(|(&key, window)| window.offers(mode).err().map(|why| (key, why)))
    }

    pub(crate) fn close_window(&mut self, key: u64) {
        self.windows.remove(&key);
    }

    pub(crate) fn present_window(
        &mut self,
        host: &crate::Host,
        window: u64,
        texture: u64,
        interval: crate::PresentInterval,
        scaling: crate::Scaling,
        config: crate::PresentationConfig,
    ) -> Result<bool, crate::Status> {
        let failed = crate::Status::InternalError;
        let mut extent = vk::Extent2D::default();
        let surface_host = unsafe { &*host.vulkan };
        unsafe {
            (surface_host.drawable_size)(host.user, window, &mut extent.width, &mut extent.height);
        }
        self.sync_texture(texture, true).ok_or(failed)?;
        let key = window;
        // The blit reads the texture's stored bytes. Source: the public contract.
        let source = self.images.shown_source(texture).map_err(|(status, why)| {
            crate::unsupported::refuse(
                status,
                format!("window {key:#x} cannot show texture {texture}: {why}"),
            )
        })?;
        let window = self.windows.get_mut(&window).ok_or(failed)?;
        // The blit converts from the texture's own format. Source: the public contract.
        window.blits_from(source.format).map_err(|why| {
            crate::unsupported::refuse(
                crate::Status::Unimplemented,
                format!("window {key:#x} cannot show texture {texture}: {why}"),
            )
        })?;
        // FIFO holds each present for one vertical blank: N presents of the
        // image hold it for the N blanks its interval asks. Mailbox takes
        // only intervals 0 and 1, both one present. Source: the public contract.
        let presents = match config.mode {
            crate::PresentationMode::Fifo => interval.fifo_presents(),
            crate::PresentationMode::Mailbox => 1,
        };
        for _ in 1..presents {
            if !window
                .present(source, extent, scaling, config)
                .ok_or(failed)?
            {
                return Ok(false);
            }
        }
        window
            .present(source, extent, scaling, config)
            .ok_or(failed)
    }

    /// Offscreen presentation uses the same blit and barriers as a swapchain.
    /// Returned bytes use the destination format, including BGRA order and sRGB encoding.
    pub fn present_offscreen(
        &mut self,
        texture: u64,
        format: vk::Format,
        width: u32,
        height: u32,
    ) -> Option<(u32, u32, Vec<u8>)> {
        if !matches!(
            format,
            vk::Format::R8G8B8A8_UNORM
                | vk::Format::B8G8R8A8_UNORM
                | vk::Format::R8G8B8A8_SRGB
                | vk::Format::B8G8R8A8_SRGB
        ) {
            return None;
        }
        self.sync_texture(texture, true)?;
        let source = self
            .images
            .shown_source(texture)
            .map_err(|(status, why)| {
                crate::unsupported::refuse(
                    status,
                    format!("offscreen present cannot show texture {texture}: {why}"),
                )
            })
            .ok()?;
        if !images::can_blit(&self.context, source.format, format) {
            crate::unsupported::refuse(
                crate::Status::Unimplemented,
                format!(
                    "offscreen present cannot show texture {texture}: the device cannot blit {} \
                     into {}",
                    formats::Name(source.format),
                    formats::Name(format)
                ),
            );
            return None;
        }
        if !self.offscreen.as_ref().is_some_and(|image| {
            image.info.format == format
                && image.info.extent.width == width
                && image.info.extent.height == height
        }) {
            self.offscreen = Some(Image::new(&self.context, width, height, format)?);
        }
        let target = self.offscreen.as_mut()?;
        self.images.blit_offscreen(source, target)?;
        Some((width, height, self.images.read_image(target)?))
    }

    pub(crate) fn enqueue_callback(
        &mut self,
        slot: usize,
        texture: u64,
        width: u64,
        height: u64,
    ) -> Option<()> {
        let width = (width.max(1) as f32 * self.scale).round() as u32;
        let height = (height.max(1) as f32 * self.scale).round() as u32;
        let current = self.coherent_images.get(&texture).is_some_and(|stored| {
            self.global_memory.as_ref().and_then(GlobalMemory::revision) == Some(stored.revision)
                && self.bindings.get(&texture) == Some(&stored.binding)
                && self.tiled.get(&texture) == stored.packing.as_ref()
        });
        let source = self.images.info(texture)?;
        if !self.tiled.contains_key(&texture)
            && source.format == vk::Format::R8G8B8A8_UNORM
            && source.shape.kind == crate::tiling::ImageKind::D2
            && source.extent
                == (vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
        {
            if let Some(&(pool, offset, bytes)) = self.bindings.get(&texture) {
                let size = (width as usize)
                    .checked_mul(height as usize)?
                    .checked_mul(4)?;
                if size <= bytes {
                    let (buffer, offset) = self
                        .global_memory
                        .as_ref()?
                        .image_region(pool, offset, size)?;
                    self.readbacks
                        .submit_buffer(slot, buffer, offset, width, height)?;
                    self.global_memory.as_ref()?.submitted(
                        offset,
                        size as u64,
                        false,
                        self.readbacks.submission(slot)?,
                    );
                    if !current {
                        self.sync_texture(texture, true)?;
                    }
                    return Some(());
                }
            }
        }
        if !current {
            self.sync_texture(texture, true)?;
        }
        // The readback blits the texture's stored bytes. Source: the public contract.
        let source = self
            .images
            .shown_source(texture)
            .map_err(|(status, why)| {
                crate::unsupported::refuse(
                    status,
                    format!("presentation readback cannot show texture {texture}: {why}"),
                )
            })
            .ok()?;
        self.readbacks.submit(slot, source, width, height)
    }

    pub(crate) fn read_callback(
        &self,
        slot: usize,
        wait: bool,
    ) -> Option<Option<(u32, u32, Vec<u8>)>> {
        self.readbacks.read(slot, wait)
    }

    pub(crate) fn reset_presentations(&mut self) {
        self.readbacks = readbacks::Readbacks::new(&self.context);
    }

    pub(crate) fn finish(&self) -> bool {
        self.images.wait().is_some()
            && self.readbacks.wait().is_some()
            && self.windows.values().all(Window::wait)
    }
}

impl Context {
    /// Retire preceding work on the execution queue at a timeline checkpoint.
    /// Queue access must be serialized with submissions by the caller.
    pub(crate) fn wait_queue(&self) -> Option<()> {
        let mut completion = self.completion.lock().unwrap_or_else(|p| p.into_inner());
        if self.command_workers.unchanged_since(completion.position)? {
            return Some(());
        }
        if let Some(latest) = self.command_workers.latest_timeline() {
            let semaphores = [latest.timeline.semaphore()];
            let values = [latest.value];
            let info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            loop {
                self.command_workers.healthy()?;
                if !latest.timeline.wait_posted(latest.value) {
                    continue;
                }
                match unsafe { self.device.wait_semaphores(&info, 100_000_000) } {
                    Ok(()) => {
                        self.command_workers.retire(latest.position);
                        completion.position = latest.position;
                        return Some(());
                    }
                    Err(vk::Result::TIMEOUT) => {}
                    Err(_) => return None,
                }
            }
        }
        let value = completion.value.checked_add(1)?;
        let semaphores = [completion.semaphore];
        let values = [value];
        unsafe {
            let position = self
                .command_workers
                .checkpoint(completion.semaphore, value)
                .ok()?;
            completion.value = value;
            self.device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&semaphores)
                        .values(&values),
                    u64::MAX,
                )
                .ok()?;
            completion.position = position;
            Some(())
        }
    }

    /// Create a module after checking the optional global memory features.
    /// The caller still validates SPIR-V and owns the returned Vulkan module.
    pub fn create_global_shader_module(&self, words: &[u32]) -> Result<vk::ShaderModule, String> {
        self.shader_features
            .validate_shader(words)
            .map_err(str::to_owned)?;
        // SAFETY: the feature checks ran above; Vulkan consumes the supplied words.
        unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
        }
        .map_err(|error| format!("create global shader module: {error:?}"))
    }
    /// Create a context, preferring a discrete GPU and then any usable GPU.
    pub fn new() -> Option<Self> {
        Self::with_extensions(&[], false)
    }

    fn with_extensions(extensions: &[CString], swapchain: bool) -> Option<Self> {
        let validation_mode = validation::Mode::from_env();
        // SAFETY: loading the system Vulkan loader is the boundary of this
        // optional backend; ash validates the returned function table.
        let entry = unsafe { Entry::load() };
        let entry = if validation_mode.is_some() {
            entry.expect("validation loader required")
        } else {
            entry.ok()?
        };
        let api_version = unsafe { entry.try_enumerate_instance_version() }
            .ok()?
            .unwrap_or(vk::API_VERSION_1_0)
            .min(vk::API_VERSION_1_3);
        if api_version < vk::API_VERSION_1_2 {
            return None;
        }
        let app = CString::new("swxbx").ok()?;
        let info = vk::ApplicationInfo::default()
            .application_name(&app)
            .engine_name(&app)
            .api_version(api_version);
        let mut extensions = extensions.to_vec();
        let mut validation = validation_mode.map(|_| validation::Validation::new());
        if validation.is_some() {
            for name in [
                ash::ext::debug_utils::NAME,
                ash::ext::layer_settings::NAME,
                ash::ext::validation_features::NAME,
            ] {
                if !extensions
                    .iter()
                    .any(|existing| existing.as_c_str() == name)
                {
                    extensions.push(name.to_owned());
                }
            }
        }
        if swapchain {
            let supported = unsafe { entry.enumerate_instance_extension_properties(None).ok()? };
            for name in [
                ash::khr::get_surface_capabilities2::NAME,
                ash::ext::surface_maintenance1::NAME,
            ] {
                if !supported.iter().any(|extension| unsafe { std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()) } == name) { return None; }
                if !extensions
                    .iter()
                    .any(|existing| existing.as_c_str() == name)
                {
                    extensions.push(name.to_owned());
                }
            }
        }
        let pointers: Vec<_> = extensions.iter().map(|name| name.as_ptr()).collect();
        let mut create = vk::InstanceCreateInfo::default()
            .application_info(&info)
            .enabled_extension_names(&pointers);
        let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
        let settings = validation_mode.map_or_else(Vec::new, |mode| {
            let layers = unsafe { entry.enumerate_instance_layer_properties() }
                .expect("Vulkan validation layer enumeration required");
            let layer = layers
                .iter()
                .find(|layer| unsafe {
                    std::ffi::CStr::from_ptr(layer.layer_name.as_ptr())
                        == c"VK_LAYER_KHRONOS_validation"
                })
                .expect("Vulkan validation layer required");
            mode.settings(layer.spec_version)
        });
        let mut layer_settings = vk::LayerSettingsCreateInfoEXT::default().settings(&settings);
        let mut features = validation_mode.map(validation::Mode::features);
        let mut debug_info = validation.as_ref().map(validation::Validation::create_info);
        if let Some(debug_info) = debug_info.as_mut() {
            create = create
                .enabled_layer_names(&layers)
                .push_next(&mut layer_settings)
                .push_next(features.as_mut().unwrap())
                .push_next(debug_info);
        }
        // SAFETY: `create` points only to local, immutable Vulkan structs.
        let instance = unsafe { entry.create_instance(&create, None) };
        let instance = if validation.is_some() {
            instance.expect("Vulkan validation instance required")
        } else {
            instance.ok()?
        };
        if let Some(validation) = validation.as_mut() {
            validation.attach(&entry, &instance);
        }
        // SAFETY: the instance is live and owns this enumeration.
        let result = Self::create_device(
            &instance,
            swapchain,
            api_version,
            matches!(validation_mode, Some(validation::Mode::Gpu)),
        );
        let Some((
            pick,
            device,
            family,
            shader_features,
            (libraries, fast_linking),
            custom_border_colors,
        )) = result
        else {
            if let Some(validation) = validation.as_mut() {
                validation.detach();
            }
            // SAFETY: no device was created and the instance is owned here.
            unsafe { instance.destroy_instance(None) };
            assert!(validation.is_none(), "Vulkan validation device required");
            return None;
        };
        // SAFETY: queue zero was requested by create_device.
        let queue = unsafe { device.get_device_queue(family, 0) };
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(pick) }
            .iter()
            .enumerate()
            .filter_map(|(i, family)| (family.queue_count > 0).then_some(i as u32))
            .collect();
        let push_descriptors = Self::supports_push(&instance, pick)
            .then(|| ash::khr::push_descriptor::Device::new(&instance, &device));
        let mut push = vk::PhysicalDevicePushDescriptorPropertiesKHR::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut push);
        if push_descriptors.is_some() {
            unsafe {
                instance.get_physical_device_properties2(pick, &mut properties);
            }
        }
        let mut ty =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let semaphore = unsafe {
            device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut ty), None)
        };
        let Ok(semaphore) = semaphore else {
            unsafe {
                device.destroy_device(None);
                if let Some(validation) = validation.as_mut() {
                    validation.detach();
                }
                instance.destroy_instance(None);
            }
            return None;
        };
        Some(Self {
            validation,
            graphics_pipeline_libraries: libraries,
            graphics_pipeline_fast_linking: fast_linking,
            custom_border_colors,
            push_descriptors,
            max_push_descriptors: push.max_push_descriptors,
            limits: unsafe { instance.get_physical_device_properties(pick) }.limits,
            completion: Mutex::new(QueueCompletion {
                semaphore,
                value: 0,
                position: 0,
            }),
            entry,
            instance,
            physical_device: pick,
            command_workers: command_workers::Runtime::new(&device, queue, family),
            device,
            queue,
            queue_family: family,
            shader_features,
            queue_families,
            swapchain,
        })
    }

    fn supports_push(instance: &Instance, device: vk::PhysicalDevice) -> bool {
        std::env::var_os("SWXBX_DISABLE_PUSH_DESCRIPTORS").is_none() && unsafe {
            instance.enumerate_device_extension_properties(device)
        }
        .is_ok_and(|extensions| {
            extensions.iter().any(|e| unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) }
                    == ash::khr::push_descriptor::NAME)
        })
    }

    fn recorder(&self) -> recording_device::RecordingDevice<'_> {
        recording_device::RecordingDevice::new(&self.device)
    }

    fn create_device(
        instance: &Instance,
        swapchain: bool,
        api_version: u32,
        gpu_validation: bool,
    ) -> Option<CreatedDevice> {
        // SAFETY: instance is live for all enumeration and feature queries below.
        let mut devices = unsafe { instance.enumerate_physical_devices().ok()? };
        devices.sort_by_key(|&device| unsafe {
            instance.get_physical_device_properties(device).device_type
                != vk::PhysicalDeviceType::DISCRETE_GPU
        });
        for pick in devices {
            if swapchain {
                let supported =
                    unsafe { instance.enumerate_device_extension_properties(pick) }.ok()?;
                if ![
                    ash::khr::swapchain::NAME,
                    ash::ext::swapchain_maintenance1::NAME,
                ]
                .iter()
                .all(|name| {
                    supported.iter().any(|extension| {
                        (unsafe { std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()) })
                            == *name
                    })
                }) {
                    continue;
                }
            }
            if unsafe { instance.get_physical_device_properties(pick).api_version }
                < vk::API_VERSION_1_2
            {
                continue;
            }
            let families = unsafe { instance.get_physical_device_queue_family_properties(pick) };
            let Some(family) = families.iter().position(|f| {
                f.queue_count > 0
                    && f.queue_flags
                        .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            }) else {
                continue;
            };
            let family = family as u32;
            let mut maintenance = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default();
            let supported = unsafe { instance.enumerate_device_extension_properties(pick) }.ok()?;
            let has = |name| {
                supported.iter().any(|e| {
                    (unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) }) == name
                })
            };
            let core13 = api_version >= vk::API_VERSION_1_3
                && unsafe { instance.get_physical_device_properties(pick).api_version }
                    >= vk::API_VERSION_1_3;
            let subgroup_extension = supported.iter().find(|e| unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) } == ash::ext::subgroup_size_control::NAME);
            let subgroup_controls = core13 || subgroup_extension.is_some();
            let legacy_subgroups =
                !core13 && subgroup_extension.is_some_and(|e| e.spec_version == 1);
            let mut subgroup_features = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
            let mut subgroup_properties =
                vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            if subgroup_controls {
                let mut properties =
                    vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_properties);
                unsafe { instance.get_physical_device_properties2(pick, &mut properties) };
                if legacy_subgroups {
                    subgroup_features.subgroup_size_control = vk::TRUE;
                    subgroup_features.compute_full_subgroups = vk::TRUE;
                } else {
                    let mut query =
                        vk::PhysicalDeviceFeatures2::default().push_next(&mut subgroup_features);
                    unsafe { instance.get_physical_device_features2(pick, &mut query) };
                }
            }
            let available = has(ash::ext::graphics_pipeline_library::NAME)
                && has(ash::khr::pipeline_library::NAME)
                && std::env::var_os("SWXBX_DISABLE_PIPELINE_LIBRARIES").is_none();
            let mut library_features =
                vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT::default();
            if available {
                let mut query =
                    vk::PhysicalDeviceFeatures2::default().push_next(&mut library_features);
                unsafe { instance.get_physical_device_features2(pick, &mut query) };
            }
            let libraries = available && library_features.graphics_pipeline_library != vk::FALSE;
            let mut library_properties =
                vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT::default();
            if libraries {
                let mut query =
                    vk::PhysicalDeviceProperties2::default().push_next(&mut library_properties);
                unsafe { instance.get_physical_device_properties2(pick, &mut query) };
            }
            let fast_linking = libraries
                && library_properties.graphics_pipeline_library_fast_linking != vk::FALSE
                && std::env::var_os("SWXBX_DISABLE_FAST_LINKING").is_none();
            let border_extension = has(ash::ext::custom_border_color::NAME);
            let mut border_features = vk::PhysicalDeviceCustomBorderColorFeaturesEXT::default();
            let mut border_properties = vk::PhysicalDeviceCustomBorderColorPropertiesEXT::default();
            if border_extension {
                let mut query =
                    vk::PhysicalDeviceFeatures2::default().push_next(&mut border_features);
                unsafe { instance.get_physical_device_features2(pick, &mut query) };
                let mut query =
                    vk::PhysicalDeviceProperties2::default().push_next(&mut border_properties);
                unsafe { instance.get_physical_device_properties2(pick, &mut query) };
            }
            let custom_borders = border_extension
                && border_features.custom_border_colors != vk::FALSE
                && border_features.custom_border_color_without_format != vk::FALSE
                && border_properties.max_custom_border_color_samplers > 0;
            let demote_extension =
                !core13 && has(ash::ext::shader_demote_to_helper_invocation::NAME);
            let reconvergence_extension = has(ash::khr::shader_maximal_reconvergence::NAME);
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut demote = vk::PhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT::default();
            let mut reconvergence =
                vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default();
            let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut features = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut v11)
                .push_next(&mut v12);
            if core13 {
                features = features.push_next(&mut v13);
            } else if demote_extension {
                features = features.push_next(&mut demote);
            }
            if reconvergence_extension {
                features = features.push_next(&mut reconvergence);
            }
            if swapchain {
                features = features.push_next(&mut maintenance);
            }
            unsafe { instance.get_physical_device_features2(pick, &mut features) };
            let core = features.features;
            if swapchain && maintenance.swapchain_maintenance1 == vk::FALSE {
                continue;
            }
            if core.shader_int64 == vk::FALSE
                || v12.buffer_device_address == vk::FALSE
                || v12.timeline_semaphore == vk::FALSE
            {
                continue;
            }
            if gpu_validation
                && (core.vertex_pipeline_stores_and_atomics == vk::FALSE
                    || core.fragment_stores_and_atomics == vk::FALSE
                    || v12.vulkan_memory_model == vk::FALSE
                    || v12.vulkan_memory_model_device_scope == vk::FALSE
                    || v12.scalar_block_layout == vk::FALSE)
            {
                continue;
            }
            let shader_features = ShaderFeatures {
                subgroups: subgroups::SubgroupFeatures::new(
                    &subgroup_features,
                    &subgroup_properties,
                ),
                storage_buffer8_bit_access: v12.storage_buffer8_bit_access != vk::FALSE,
                storage_buffer16_bit_access: v11.storage_buffer16_bit_access != vk::FALSE,
                draw_parameters: v11.shader_draw_parameters != vk::FALSE,
                shader_int8: v12.shader_int8 != vk::FALSE,
                shader_int16: core.shader_int16 != vk::FALSE,
                demote_to_helper_invocation: if core13 {
                    v13.shader_demote_to_helper_invocation != vk::FALSE
                } else {
                    demote_extension && demote.shader_demote_to_helper_invocation != vk::FALSE
                },
                maximal_reconvergence: reconvergence_extension
                    && reconvergence.shader_maximal_reconvergence != vk::FALSE,
                storage_image_read_without_format: core.shader_storage_image_read_without_format
                    != vk::FALSE,
                storage_image_write_without_format: core.shader_storage_image_write_without_format
                    != vk::FALSE,
                vertex_pipeline_stores: core.vertex_pipeline_stores_and_atomics != vk::FALSE,
                fragment_stores: core.fragment_stores_and_atomics != vk::FALSE,
                geometry_shader: core.geometry_shader != vk::FALSE,
            };
            let enabled = vk::PhysicalDeviceFeatures::default()
                .vertex_pipeline_stores_and_atomics(shader_features.vertex_pipeline_stores)
                .fragment_stores_and_atomics(shader_features.fragment_stores)
                .shader_storage_image_read_without_format(
                    shader_features.storage_image_read_without_format,
                )
                .shader_storage_image_write_without_format(
                    shader_features.storage_image_write_without_format,
                )
                .independent_blend(core.independent_blend != 0)
                .sampler_anisotropy(core.sampler_anisotropy != 0)
                .shader_int64(true)
                .draw_indirect_first_instance(core.draw_indirect_first_instance == vk::TRUE)
                .image_cube_array(core.image_cube_array != 0)
                .texture_compression_bc(core.texture_compression_bc != 0)
                .texture_compression_astc_ldr(core.texture_compression_astc_ldr != 0)
                .shader_int16(shader_features.shader_int16)
                .fill_mode_non_solid(core.fill_mode_non_solid != 0)
                .depth_bias_clamp(core.depth_bias_clamp != 0)
                .shader_clip_distance(core.shader_clip_distance != 0)
                .geometry_shader(shader_features.geometry_shader);
            let mut enabled11 = vk::PhysicalDeviceVulkan11Features::default()
                .storage_buffer16_bit_access(shader_features.storage_buffer16_bit_access)
                .shader_draw_parameters(shader_features.draw_parameters);
            let mut enabled12 = vk::PhysicalDeviceVulkan12Features::default()
                .vulkan_memory_model(gpu_validation)
                .vulkan_memory_model_device_scope(gpu_validation)
                .scalar_block_layout(gpu_validation)
                .buffer_device_address(true)
                .timeline_semaphore(true)
                .sampler_mirror_clamp_to_edge(v12.sampler_mirror_clamp_to_edge != vk::FALSE)
                .storage_buffer8_bit_access(shader_features.storage_buffer8_bit_access)
                .shader_int8(shader_features.shader_int8);
            let mut enabled13 = vk::PhysicalDeviceVulkan13Features::default()
                .shader_demote_to_helper_invocation(shader_features.demote_to_helper_invocation)
                .subgroup_size_control(subgroup_features.subgroup_size_control != vk::FALSE)
                .compute_full_subgroups(subgroup_features.compute_full_subgroups != vk::FALSE);
            let mut enabled_demote =
                vk::PhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT::default()
                    .shader_demote_to_helper_invocation(
                        shader_features.demote_to_helper_invocation,
                    );
            let mut enabled_reconvergence =
                vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default()
                    .shader_maximal_reconvergence(shader_features.maximal_reconvergence);
            let priority = [1.0_f32];
            let queue_info: Vec<_> = families
                .iter()
                .enumerate()
                .filter(|(_, properties)| properties.queue_count > 0)
                .map(|(i, _)| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(i as u32)
                        .queue_priorities(&priority)
                })
                .collect();
            let mut extensions = Vec::new();
            if shader_features.demote_to_helper_invocation && !core13 {
                extensions.push(ash::ext::shader_demote_to_helper_invocation::NAME.as_ptr());
            }
            if shader_features.maximal_reconvergence {
                extensions.push(ash::khr::shader_maximal_reconvergence::NAME.as_ptr());
            }
            if subgroup_controls && !core13 {
                extensions.push(ash::ext::subgroup_size_control::NAME.as_ptr());
            }
            if swapchain {
                extensions.push(ash::khr::swapchain::NAME.as_ptr());
                extensions.push(ash::ext::swapchain_maintenance1::NAME.as_ptr());
            }
            if Self::supports_push(instance, pick) {
                extensions.push(ash::khr::push_descriptor::NAME.as_ptr());
            }
            if has(ash::ext::depth_range_unrestricted::NAME) {
                extensions.push(ash::ext::depth_range_unrestricted::NAME.as_ptr());
            }
            if custom_borders {
                extensions.push(ash::ext::custom_border_color::NAME.as_ptr());
            }
            if libraries {
                extensions.push(ash::ext::graphics_pipeline_library::NAME.as_ptr());
                extensions.push(ash::khr::pipeline_library::NAME.as_ptr());
            }
            let mut enabled_library =
                vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT::default()
                    .graphics_pipeline_library(libraries);
            let mut device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&extensions)
                .enabled_features(&enabled)
                .push_next(&mut enabled11)
                .push_next(&mut enabled12);
            let mut enabled_maintenance =
                vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
                    .swapchain_maintenance1(true);
            if swapchain {
                device_info = device_info.push_next(&mut enabled_maintenance);
            }
            if core13 {
                device_info = device_info.push_next(&mut enabled13);
            } else if shader_features.demote_to_helper_invocation {
                device_info = device_info.push_next(&mut enabled_demote);
            }
            if shader_features.maximal_reconvergence {
                device_info = device_info.push_next(&mut enabled_reconvergence);
            }
            if subgroup_controls && !core13 && !legacy_subgroups {
                device_info = device_info.push_next(&mut subgroup_features);
            }
            if libraries {
                device_info = device_info.push_next(&mut enabled_library);
            }
            let mut enabled_borders = vk::PhysicalDeviceCustomBorderColorFeaturesEXT::default()
                .custom_border_colors(true)
                .custom_border_color_without_format(true);
            if custom_borders {
                device_info = device_info.push_next(&mut enabled_borders);
            }
            // SAFETY: the selected family supports graphics and the create info is valid.
            if let Ok(device) = unsafe { instance.create_device(pick, &device_info, None) } {
                return Some((
                    pick,
                    device,
                    family,
                    shader_features,
                    (libraries, fast_linking),
                    if custom_borders {
                        CustomBorderColors::Present {
                            samplers: border_properties.max_custom_border_color_samplers,
                            live: 0.into(),
                        }
                    } else {
                        CustomBorderColors::Absent
                    },
                ));
            }
        }
        None
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let _ = self.wait_queue();
        self.command_workers.shutdown();
        // SAFETY: all work submitted by this small context is complete before
        // it is dropped by the owning instance.
        unsafe {
            let completion = self.completion.get_mut().unwrap_or_else(|p| p.into_inner());
            self.device.destroy_semaphore(completion.semaphore, None);
            self.device.destroy_device(None);
            if let Some(validation) = self.validation.as_mut() {
                validation.detach();
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn find_memory_type(context: &Context, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
    let memory = unsafe {
        context
            .instance
            .get_physical_device_memory_properties(context.physical_device)
    };
    (0..memory.memory_type_count).find(|&i| {
        bits & (1 << i) != 0
            && memory.memory_types[i as usize]
                .property_flags
                .contains(flags)
    })
}

#[cfg(test)]
mod tests {
    use super::{Backend, ShaderFeatures};

    #[test]
    #[ignore = "requires a graphics device"]
    fn sampler_replacement_retires_queued_users() {
        use super::{textures, Context};
        use ash::vk;
        use std::{sync::mpsc, sync::Arc, sync::Mutex, time::Duration};

        let mut backend = Backend::new(1.0).expect("graphics device required");
        let contract = textures::TextureContract {
            filters: vec![(0, vk::Filter::NEAREST), (1, vk::Filter::LINEAR)],
            wraps: vec![(0, vk::SamplerAddressMode::REPEAT)],
            ..Default::default()
        };
        let key = |min_filter| {
            textures::SamplerKey::new(
                &textures::SamplerDescription {
                    min_filter,
                    max_anisotropy: 1.0,
                    ..Default::default()
                },
                &contract,
            )
            .unwrap()
        };
        let original = backend.sampler(1, 0, key(0)).unwrap();
        let replacement = key(1);
        let context: Arc<Context> = backend.context.clone();
        let (entered, entry) = mpsc::sync_channel(1);
        let (release, released) = mpsc::sync_channel(1);
        let released = Mutex::new(released);
        let operation: super::recording_device::Operation = Arc::new(move |_, _| {
            entered.send(()).unwrap();
            released.lock().unwrap().recv().unwrap();
        });
        let completion = context.command_workers.enqueue(vec![operation], None, None);
        entry.recv_timeout(Duration::from_secs(5)).unwrap();
        let (changed, change) = mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            let sampler = backend.sampler(1, 0, replacement).unwrap();
            changed.send(sampler != original).unwrap();
        });
        let early = change.recv_timeout(Duration::from_millis(100));
        release.send(()).unwrap();
        completion.wait().unwrap();
        worker.join().unwrap();
        assert!(matches!(early, Err(mpsc::RecvTimeoutError::Timeout)));
        assert!(change.recv_timeout(Duration::from_secs(5)).unwrap());
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn callback_source_reuse_observes_cpu_writes_and_aliased_image_stores() {
        let mut backend = Backend::new(1.0).expect("Vulkan device required");
        backend.allocate_pool(10, 0, 64).unwrap();
        for key in [1, 2] {
            assert!(backend.ensure(key, 4, 4, false));
            backend.bindings.insert(key, (10, 0, 64));
        }
        assert!(backend.clear_color(1, [1.0, 0.0, 0.0, 1.0], 0xf).is_ok());
        backend.enqueue_callback(0, 1, 4, 4).unwrap();
        assert_eq!(
            &backend.read_callback(0, true).unwrap().unwrap().2[..4],
            &[255, 0, 0, 255]
        );
        assert!(backend.clear_color(2, [0.0, 0.0, 1.0, 1.0], 0xf).is_ok());
        backend.enqueue_callback(0, 1, 4, 4).unwrap();
        assert_eq!(
            &backend.read_callback(0, true).unwrap().unwrap().2[..4],
            &[0, 0, 255, 255]
        );
        backend
            .global_memory
            .as_mut()
            .unwrap()
            .write_pool(10, 0, &[0, 255, 0, 255].repeat(16))
            .unwrap();
        backend.enqueue_callback(0, 1, 4, 4).unwrap();
        assert_eq!(
            &backend.read_callback(0, true).unwrap().unwrap().2[..4],
            &[0, 255, 0, 255]
        );
        assert!(backend.clear_color(1, [1.0, 0.0, 0.0, 1.0], 0xf).is_ok());
        assert!(backend.upload_image(1, &[0, 0, 255, 255].repeat(16)));
        backend.enqueue_callback(0, 1, 4, 4).unwrap();
        assert_eq!(
            &backend.read_callback(0, true).unwrap().unwrap().2[..4],
            &[255, 0, 0, 255]
        );
    }

    #[test]
    fn rejects_unavailable_narrow_capabilities_before_module_creation() {
        let supported = ShaderFeatures {
            subgroups: Default::default(),
            storage_buffer8_bit_access: true,
            storage_buffer16_bit_access: true,
            shader_int8: true,
            shader_int16: true,
            ..Default::default()
        };
        for capability in [39, 22, 4448, 4433] {
            let words = [0x0723_0203, 0x10300, 0, 1, 0, (2 << 16) | 17, capability];
            assert!(ShaderFeatures::default()
                .validate_global_shader(&words)
                .is_err());
            assert!(supported.validate_global_shader(&words).is_ok());
        }
        assert!(supported.validate_global_shader(&[]).is_err());
        assert!(supported
            .validate_global_shader(&[0x0723_0203, 0x10300, 0, 1, 0, 0])
            .is_err());
        assert!(supported
            .validate_global_shader(&[0x0723_0203, 0x10300, 0, 1, 0, (3 << 16) | 17, 39])
            .is_err());
    }

    #[test]
    fn clear_and_readback_at_scale_one() {
        let Some(mut backend) = Backend::new(1.0) else {
            return;
        };
        assert!(backend.ensure(1, 4, 4, false));
        assert!(backend.clear_color(1, [1.0, 0.5, 0.0, 1.0], 0xf).is_ok());
        let (width, height, pixels) = backend.readback(1).expect("Vulkan readback");
        assert_eq!((width, height), (4, 4));
        assert_eq!(&pixels[..4], &[255, 128, 0, 255]);
    }

    #[test]
    fn clear_and_readback_at_scale_two() {
        let Some(mut backend) = Backend::new(2.0) else {
            return;
        };
        assert!(backend.ensure(1, 2, 1, false));
        assert!(backend.clear_color(1, [0.0, 0.0, 1.0, 1.0], 0xf).is_ok());
        backend.enqueue_callback(0, 1, 2, 1).unwrap();
        let (width, height, pixels) = backend
            .read_callback(0, true)
            .flatten()
            .expect("Vulkan presentation");
        assert_eq!((width, height), (4, 2));
        assert_eq!(&pixels[..4], &[0, 0, 255, 255]);
    }
}
