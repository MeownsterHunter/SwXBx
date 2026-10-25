//! Window surface, swapchain and presentation.

use super::{
    commands::Commands,
    images::{can_blit, record_present, ImageInfo},
    Context,
};
use ash::vk;
use std::{collections::VecDeque, sync::Arc};

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
            self.loader.destroy_swapchain(self.handle, None);
        }
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

    pub fn wait(&self) -> bool {
        self.commands.as_ref().is_none_or(|c| c.wait().is_some())
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
        let modes = unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(context.physical_device, self.surface)
                .ok()?
        };
        let mode = if mode == crate::PresentationMode::Mailbox
            && modes.contains(&vk::PresentModeKHR::MAILBOX)
        {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };
        if !self.rebuild
            && self
                .chain
                .as_ref()
                .is_some_and(|c| c.extent == extent && c.mode == mode)
        {
            return Some(true);
        }
        if !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
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
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
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
                    vk::ImageLayout::PRESENT_SRC_KHR,
                );
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
        if !can_blit(context, vk::Format::R8G8B8A8_UNORM, format) {
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
