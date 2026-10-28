//! Window surface, swapchain and presentation. Source: the public contract. and 0034.

use super::{
    commands::Commands,
    images::{can_blit, record_present, ImageInfo, Placement},
    Context,
};
use ash::vk;
use std::{collections::VecDeque, sync::Arc};

/// The surface present mode a requested presentation mode needs.
fn present_mode(mode: crate::PresentationMode) -> vk::PresentModeKHR {
    match mode {
        crate::PresentationMode::Fifo => vk::PresentModeKHR::FIFO,
        crate::PresentationMode::Mailbox => vk::PresentModeKHR::MAILBOX,
    }
}

/// Why a surface cannot take a presentation mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unoffered {
    /// The surface answered and the mode is not among its modes.
    Lacks,
    /// The surface could not be asked which modes it offers.
    CannotBeAsked,
}

struct Swapchain {
    context: Arc<Context>,
    loader: ash::khr::swapchain::Device,
    handle: vk::SwapchainKHR,
    extent: vk::Extent2D,
    mode: vk::PresentModeKHR,
    images: Vec<ImageInfo>,
    ready: Vec<vk::Semaphore>,
    retired: Vec<vk::Fence>,
    presented: Vec<bool>,
    in_flight: VecDeque<usize>,
    opaque: Option<OpaquePass>,
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            let fences: Vec<_> = self
                .retired
                .iter()
                .zip(&self.presented)
                .filter_map(|(&f, &p)| p.then_some(f))
                .collect();
            if !fences.is_empty() {
                let _ = self.context.device.wait_for_fences(&fences, true, u64::MAX);
            }
            for &fence in &self.retired {
                self.context.device.destroy_fence(fence, None);
            }
            for &semaphore in &self.ready {
                self.context.device.destroy_semaphore(semaphore, None);
            }
            if let Some(opaque) = self.opaque.as_mut() {
                opaque.destroy();
            }
            self.loader.destroy_swapchain(self.handle, None);
        }
    }
}

struct OpaquePass {
    context: Arc<Context>,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    premultiply: vk::Pipeline,
    opaque: vk::Pipeline,
    views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
}

fn opaque_blend_attachments() -> [vk::PipelineColorBlendAttachmentState; 2] {
    let rgb = vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B;
    [
        vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ZERO)
            .dst_color_blend_factor(vk::BlendFactor::DST_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ZERO)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(rgb),
        vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::A),
    ]
}

impl OpaquePass {
    fn new(context: &Arc<Context>, format: vk::Format, images: &[ImageInfo]) -> Option<Self> {
        let attachment = vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let reference = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let references = [reference];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&references);
        let attachments = [attachment];
        let subpasses = [subpass];
        let render_pass = unsafe {
            context
                .device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&attachments)
                        .subpasses(&subpasses),
                    None,
                )
                .ok()?
        };
        let layout = unsafe {
            context
                .device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
                .ok()?
        };
        let words = |bytes: &[u8]| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect::<Vec<_>>()
        };
        let vertex = context
            .create_global_shader_module(&words(include_bytes!("present_alpha.vert.spv")))
            .ok()?;
        let fragment = context
            .create_global_shader_module(&words(include_bytes!("present_alpha.frag.spv")))
            .ok()?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment)
                .name(c"main"),
        ];
        let input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        // Preserve the destination alpha while premultiplying RGB, then replace
        // alpha in a separate draw. Keeping those writes separate makes the
        // destination-alpha dependency explicit on every Vulkan implementation.
        let create = |attachment: vk::PipelineColorBlendAttachmentState| {
            let attachments = [attachment];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
            let info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&input)
                .input_assembly_state(&assembly)
                .viewport_state(&viewport)
                .rasterization_state(&raster)
                .multisample_state(&multisample)
                .color_blend_state(&blend)
                .dynamic_state(&dynamic)
                .layout(layout)
                .render_pass(render_pass);
            unsafe {
                context
                    .device
                    .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
                    .ok()
                    .map(|pipelines| pipelines[0])
            }
        };
        let [premultiply_state, opaque_state] = opaque_blend_attachments();
        let premultiply = create(premultiply_state)?;
        let opaque = match create(opaque_state) {
            Some(pipeline) => pipeline,
            None => {
                unsafe { context.device.destroy_pipeline(premultiply, None) };
                return None;
            }
        };
        unsafe {
            context.device.destroy_shader_module(vertex, None);
            context.device.destroy_shader_module(fragment, None);
        }
        let mut views = Vec::with_capacity(images.len());
        let mut framebuffers = Vec::with_capacity(images.len());
        for image in images {
            let view = unsafe {
                context
                    .device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image.image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .level_count(1)
                                    .layer_count(1),
                            ),
                        None,
                    )
                    .ok()?
            };
            let framebuffer_attachments = [view];
            let framebuffer = unsafe {
                context
                    .device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(&framebuffer_attachments)
                            .width(image.extent.width)
                            .height(image.extent.height)
                            .layers(1),
                        None,
                    )
                    .ok()?
            };
            views.push(view);
            framebuffers.push(framebuffer);
        }
        Some(Self {
            context: Arc::clone(context),
            render_pass,
            layout,
            premultiply,
            opaque,
            views,
            framebuffers,
        })
    }

    fn record(&self, command: vk::CommandBuffer, index: usize, extent: vk::Extent2D) {
        let area = vk::Rect2D::default().extent(extent);
        let recorder = self.context.recorder();
        unsafe {
            recorder.cmd_begin_render_pass(
                command,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[index])
                    .render_area(area),
                vk::SubpassContents::INLINE,
            );
            recorder.cmd_set_viewport(
                command,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            recorder.cmd_set_scissor(command, &[area]);
            recorder.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, self.premultiply);
            recorder.cmd_draw(command, 3, 1, 0, 0);
            recorder.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, self.opaque);
            recorder.cmd_draw(command, 3, 1, 0, 0);
            recorder.cmd_end_render_pass(command);
        }
    }

    unsafe fn destroy(&mut self) {
        for framebuffer in self.framebuffers.drain(..) {
            self.context.device.destroy_framebuffer(framebuffer, None);
        }
        for view in self.views.drain(..) {
            self.context.device.destroy_image_view(view, None);
        }
        self.context.device.destroy_pipeline(self.premultiply, None);
        self.context.device.destroy_pipeline(self.opaque, None);
        self.context
            .device
            .destroy_pipeline_layout(self.layout, None);
        self.context
            .device
            .destroy_render_pass(self.render_pass, None);
    }
}

pub(super) struct Window {
    context: Arc<Context>,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    present_family: u32,
    present_queue: vk::Queue,
    commands: Option<Commands>,
    chain: Option<Swapchain>,
    rebuild: bool,
    failed: bool,
}

impl Window {
    /// Takes ownership of the surface, including on failure.
    pub fn new(context: &Arc<Context>, surface: vk::SurfaceKHR) -> Option<Self> {
        let mut window = Self {
            context: Arc::clone(context),
            surface_loader: ash::khr::surface::Instance::new(&context.entry, &context.instance),
            surface,
            present_family: 0,
            present_queue: vk::Queue::null(),
            commands: None,
            chain: None,
            rebuild: true,
            failed: false,
        };
        let family = context
            .queue_families
            .iter()
            .copied()
            .find(|&family| unsafe {
                window
                    .surface_loader
                    .get_physical_device_surface_support(context.physical_device, family, surface)
                    .unwrap_or(false)
            })?;
        window.present_family = family;
        window.present_queue = unsafe { context.device.get_device_queue(family, 0) };
        window.commands = Some(Commands::new(context)?);
        Some(window)
    }

    /// Ask this surface whether it offers the requested mode.
    pub fn offers(&self, mode: crate::PresentationMode) -> Result<(), Unoffered> {
        let modes = unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(
                    self.context.physical_device,
                    self.surface,
                )
                .map_err(|_| Unoffered::CannotBeAsked)?
        };
        if modes.contains(&present_mode(mode)) {
            Ok(())
        } else {
            Err(Unoffered::Lacks)
        }
    }

    pub fn wait(&self) -> bool {
        self.commands.as_ref().is_none_or(|c| c.wait().is_some())
    }

    /// Check that a texture of `format` can be blitted into this window's
    /// swapchain images, or say why not. Before the first present the
    /// swapchain format is the one the surface would be given. Source: the public contract.
    pub fn blits_from(&self, format: vk::Format) -> Result<(), String> {
        let target = match self.chain.as_ref().and_then(|c| c.images.first()) {
            Some(image) => image.format,
            None => {
                let formats = unsafe {
                    self.surface_loader.get_physical_device_surface_formats(
                        self.context.physical_device,
                        self.surface,
                    )
                }
                .map_err(|e| format!("its surface formats cannot be asked: {e:?}"))?;
                choose_format(&self.context, &formats)
                    .ok_or("its surface offers no format swxbx can blit into")?
                    .format
            }
        };
        if can_blit(&self.context, format, target) {
            Ok(())
        } else {
            Err(format!(
                "the device cannot blit {} into the window's {} images",
                super::formats::Name(format),
                super::formats::Name(target)
            ))
        }
    }

    fn configure(&mut self, desired: vk::Extent2D, mode: crate::PresentationMode) -> Option<bool> {
        let context = &self.context;
        let caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(context.physical_device, self.surface)
                .ok()?
        };
        if desired.width == 0
            || desired.height == 0
            || caps.max_image_extent.width == 0
            || caps.max_image_extent.height == 0
        {
            return Some(false);
        }
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: desired
                    .width
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: desired
                    .height
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        if extent.width == 0 || extent.height == 0 {
            return Some(false);
        }
        // The mode was asked of this surface when the policy was set or the
        // window opened, whichever came last.
        let mode = present_mode(mode);
        if !self.rebuild
            && self
                .chain
                .as_ref()
                .is_some_and(|c| c.extent == extent && c.mode == mode)
        {
            return Some(true);
        }
        let required_usage =
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        if !caps.supported_usage_flags.contains(required_usage) {
            return None;
        }
        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(context.physical_device, self.surface)
                .ok()?
        };
        let format = choose_format(context, &formats)?;
        let alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|alpha| caps.supported_composite_alpha.contains(*alpha))?;
        let mut count = caps.min_image_count.saturating_add(1);
        if caps.max_image_count != 0 {
            count = count.min(caps.max_image_count);
        }
        let families = [context.queue_family, self.present_family];
        let mut info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(required_usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(alpha)
            .present_mode(mode)
            .clipped(true);
        if families[0] != families[1] {
            info = info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families);
        }
        self.commands.as_ref()?.wait()?;
        self.chain.take();
        let loader = ash::khr::swapchain::Device::new(&context.instance, &context.device);
        if std::env::var_os("SWXBX_PRESENT_TRACE").is_some() {
            eprintln!(
                "swxbx present: swapchain extent={}x{} desired={}x{} caps_current={}x{} format={:?}/{:?} alpha={:?} supported_alpha={:?} mode={:?} images={}",
                extent.width, extent.height, desired.width, desired.height,
                caps.current_extent.width, caps.current_extent.height,
                format.format.as_raw(), format.color_space.as_raw(), alpha.as_raw(), caps.supported_composite_alpha.as_raw(), mode.as_raw(), count
            );
        }
        let handle = unsafe { loader.create_swapchain(&info, None).ok()? };
        let mut chain = Swapchain {
            context: Arc::clone(context),
            loader,
            handle,
            extent,
            mode,
            images: Vec::new(),
            ready: Vec::new(),
            retired: Vec::new(),
            presented: Vec::new(),
            in_flight: VecDeque::new(),
            opaque: None,
        };
        chain.images = unsafe { chain.loader.get_swapchain_images(handle).ok()? }
            .into_iter()
            .map(|image| ImageInfo {
                image,
                extent: vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                format: format.format,
                layout: vk::ImageLayout::UNDEFINED,
                shape: crate::tiling::ImageShape {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                    layers: 1,
                    levels: 1,
                    kind: crate::tiling::ImageKind::D2,
                },
                size: u64::from(extent.width) * u64::from(extent.height) * 4,
            })
            .collect();
        chain.opaque = Some(OpaquePass::new(context, format.format, &chain.images)?);
        for _ in &chain.images {
            chain.retired.push(unsafe {
                context
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .ok()?
            });
            chain.presented.push(false);
            chain.ready.push(unsafe {
                context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .ok()?
            });
        }
        self.chain = Some(chain);
        self.rebuild = false;
        Some(true)
    }

    /// Present a transfer-readable source through the native swapchain, recreating it as needed.
    pub fn present(
        &mut self,
        source: ImageInfo,
        desired: vk::Extent2D,
        scaling: crate::Scaling,
        config: crate::PresentationConfig,
    ) -> Option<bool> {
        if self.failed {
            return None;
        }
        if self.commands.as_ref()?.slots() != config.frames_in_flight as usize {
            self.commands = Some(Commands::with_capacity(
                &self.context,
                config.frames_in_flight as usize,
            )?);
        }
        for _ in 0..2 {
            if !self.configure(desired, config.mode)? {
                return Some(false);
            }
            let chain = self.chain.as_mut()?;
            while chain.in_flight.len() >= config.frames_in_flight as usize {
                let index = chain.in_flight.pop_front()?;
                unsafe {
                    self.context
                        .device
                        .wait_for_fences(&[chain.retired[index]], true, u64::MAX)
                        .ok()?;
                }
            }
            let commands = self.commands.as_mut()?;
            let (cmd, acquire) = commands.begin()?;
            let acquired = unsafe {
                chain
                    .loader
                    .acquire_next_image(chain.handle, u64::MAX, acquire, vk::Fence::null())
            };
            let (index, suboptimal) = match acquired {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.rebuild = true;
                    continue;
                }
                Err(_) => return None,
            };
            // After successful acquisition, any failure makes this window unusable:
            // its acquire semaphore may remain signalled.
            let result = (|| {
                let target = chain.images[index as usize];
                record_present(
                    &self.context.recorder(),
                    cmd,
                    source,
                    target,
                    Placement::of(scaling, target.extent),
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                );
                chain
                    .opaque
                    .as_ref()?
                    .record(cmd, index as usize, chain.extent);
                let ready = chain.ready[index as usize];
                commands.submit(true, Some(ready))?;
                chain.images[index as usize].layout = vk::ImageLayout::PRESENT_SRC_KHR;
                let waits = [ready];
                chain.in_flight.retain(|&prior| prior != index as usize);
                let fences = [chain.retired[index as usize]];
                if chain.presented[index as usize] {
                    unsafe {
                        self.context
                            .device
                            .wait_for_fences(&fences, true, u64::MAX)
                            .ok()?;
                        self.context.device.reset_fences(&fences).ok()?;
                    }
                    chain.presented[index as usize] = false;
                }
                match self.context.command_workers.present(
                    self.present_queue,
                    &chain.loader,
                    chain.handle,
                    index,
                    &waits,
                    fences[0],
                ) {
                    Ok(outdated) => {
                        chain.presented[index as usize] = true;
                        chain.in_flight.push_back(index as usize);
                        self.rebuild = suboptimal || outdated;
                    }
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        chain.presented[index as usize] = true;
                        chain.in_flight.push_back(index as usize);
                        self.rebuild = true;
                    }
                    Err(vk::Result::ERROR_SURFACE_LOST_KHR) => {
                        chain.presented[index as usize] = true;
                        return None;
                    }
                    Err(_) => return None,
                }
                Some(true)
            })();
            if result.is_none() {
                self.failed = true;
            }
            return result;
        }
        None
    }
}

fn choose_format(
    context: &Context,
    formats: &[vk::SurfaceFormatKHR],
) -> Option<vk::SurfaceFormatKHR> {
    for format in [
        vk::Format::B8G8R8A8_UNORM,
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_SRGB,
        vk::Format::R8G8B8A8_SRGB,
    ] {
        let required =
            vk::FormatFeatureFlags::BLIT_DST | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND;
        if !super::formats::optimal_features(context, format).contains(required) {
            continue;
        }
        if let Some(surface_format) = formats.iter().find(|f| {
            (f.format == format || f.format == vk::Format::UNDEFINED)
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        }) {
            return Some(vk::SurfaceFormatKHR {
                format,
                color_space: surface_format.color_space,
            });
        }
    }
    None
}

impl Drop for Window {
    fn drop(&mut self) {
        self.commands.take();
        self.chain.take();
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_present_premultiplies_before_replacing_alpha() {
        let [premultiply, opaque] = opaque_blend_attachments();
        let rgb =
            vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B;
        assert!(premultiply.color_write_mask == rgb);
        assert!(premultiply.dst_color_blend_factor == vk::BlendFactor::DST_ALPHA);
        assert!(premultiply.src_color_blend_factor == vk::BlendFactor::ZERO);
        assert!(opaque.color_write_mask == vk::ColorComponentFlags::A);
        assert_eq!(opaque.blend_enable, vk::FALSE);
    }
}
