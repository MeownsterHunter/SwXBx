//! Owned native recording operations.
#![allow(clippy::too_many_arguments)]

use ash::vk::{self, Handle};
use std::{
    cell::RefCell,
    collections::HashMap,
    ops::Deref,
    ptr,
    sync::atomic::{AtomicU64, Ordering},
};

pub(super) type Operation =
    std::sync::Arc<dyn Fn(&ash::Device, vk::CommandBuffer) + Send + Sync + 'static>;

thread_local! {
    static CAPTURES: RefCell<HashMap<u64, std::sync::Weak<std::sync::Mutex<Vec<Operation>>>>> = RefCell::new(HashMap::new());
}

static NEXT_CAPTURE: AtomicU64 = AtomicU64::new(1);
const CAPTURE_TAG: u64 = 0xc000_0000_0000_0000;

pub(super) struct Capture {
    command: vk::CommandBuffer,
    operations: std::sync::Arc<std::sync::Mutex<Vec<Operation>>>,
    owner: std::thread::ThreadId,
}
impl Capture {
    pub fn new() -> Self {
        let id = NEXT_CAPTURE.fetch_add(1, Ordering::Relaxed);
        assert!(id < CAPTURE_TAG, "capture identity exhausted");
        let command = vk::CommandBuffer::from_raw(CAPTURE_TAG | id);
        let operations = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        CAPTURES.with(|captures| {
            let mut captures = captures.borrow_mut();
            captures.retain(|_, value| value.strong_count() != 0);
            assert!(captures
                .insert(command.as_raw(), std::sync::Arc::downgrade(&operations))
                .is_none());
        });
        Self {
            command,
            operations,
            owner: std::thread::current().id(),
        }
    }
    pub fn command(&self) -> vk::CommandBuffer {
        self.command
    }
    pub fn finish(self) -> Vec<Operation> {
        assert_eq!(
            self.owner,
            std::thread::current().id(),
            "recording thread changed"
        );
        std::mem::take(&mut *self.operations.lock().unwrap_or_else(|p| p.into_inner()))
    }
}

pub(super) struct RecordingDevice<'a> {
    device: &'a ash::Device,
}
impl<'a> RecordingDevice<'a> {
    pub fn new(device: &'a ash::Device) -> Self {
        Self { device }
    }

    fn record_or_run(&self, command: vk::CommandBuffer, operation: Operation) {
        let capture = CAPTURES.with(|captures| {
            captures
                .borrow()
                .get(&command.as_raw())
                .and_then(std::sync::Weak::upgrade)
        });
        if let Some(capture) = capture {
            capture
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(operation);
        } else {
            assert!(
                command.as_raw() & CAPTURE_TAG != CAPTURE_TAG,
                "inactive captured command"
            );
            operation(self.device, command);
        }
    }

    pub unsafe fn cmd_bind_pipeline(
        &self,
        command: vk::CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    ) {
        if command.as_raw() & CAPTURE_TAG != CAPTURE_TAG {
            unsafe {
                self.device.cmd_bind_pipeline(command, bind_point, pipeline);
            }
            return;
        }
        self.record_or_run(
            command,
            std::sync::Arc::new(move |device, command| unsafe {
                device.cmd_bind_pipeline(command, bind_point, pipeline);
            }),
        );
    }

    pub unsafe fn cmd_bind_descriptor_sets(
        &self,
        command: vk::CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        if command.as_raw() & CAPTURE_TAG != CAPTURE_TAG {
            unsafe {
                self.device.cmd_bind_descriptor_sets(
                    command,
                    bind_point,
                    layout,
                    first_set,
                    descriptor_sets,
                    dynamic_offsets,
                );
            }
            return;
        }
        let descriptor_sets = descriptor_sets.to_vec();
        let dynamic_offsets = dynamic_offsets.to_vec();
        self.record_or_run(
            command,
            std::sync::Arc::new(move |device, command| unsafe {
                device.cmd_bind_descriptor_sets(
                    command,
                    bind_point,
                    layout,
                    first_set,
                    &descriptor_sets,
                    &dynamic_offsets,
                );
            }),
        );
    }

    pub unsafe fn cmd_dispatch(&self, command: vk::CommandBuffer, x: u32, y: u32, z: u32) {
        if command.as_raw() & CAPTURE_TAG != CAPTURE_TAG {
            unsafe {
                self.device.cmd_dispatch(command, x, y, z);
            }
            return;
        }
        self.record_or_run(
            command,
            std::sync::Arc::new(move |device, command| unsafe {
                device.cmd_dispatch(command, x, y, z);
            }),
        );
    }

    pub unsafe fn cmd_push_constants(
        &self,
        command: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        values: &[u8],
    ) {
        if command.as_raw() & CAPTURE_TAG != CAPTURE_TAG {
            unsafe {
                self.device
                    .cmd_push_constants(command, layout, stage_flags, offset, values);
            }
            return;
        }
        let values = values.to_vec();
        self.record_or_run(
            command,
            std::sync::Arc::new(move |device, command| unsafe {
                device.cmd_push_constants(command, layout, stage_flags, offset, &values);
            }),
        );
    }

    pub unsafe fn cmd_copy_buffer_to_image(
        &self,
        command: vk::CommandBuffer,
        src_buffer: vk::Buffer,
        dst_image: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        if command.as_raw() & CAPTURE_TAG != CAPTURE_TAG {
            unsafe {
                self.device
                    .cmd_copy_buffer_to_image(command, src_buffer, dst_image, dst_layout, regions);
            }
            return;
        }
        let regions = regions.to_vec();
        self.record_or_run(
            command,
            std::sync::Arc::new(move |device, command| unsafe {
                device
                    .cmd_copy_buffer_to_image(command, src_buffer, dst_image, dst_layout, &regions);
            }),
        );
    }

    pub unsafe fn cmd_copy_image_to_buffer(
        &self,
        command: vk::CommandBuffer,
        src_image: vk::Image,
        src_layout: vk::ImageLayout,
        dst_buffer: vk::Buffer,
        regions: &[vk::BufferImageCopy],
    ) {
        if command.as_raw() & CAPTURE_TAG != CAPTURE_TAG {
            unsafe {
                self.device
                    .cmd_copy_image_to_buffer(command, src_image, src_layout, dst_buffer, regions);
            }
            return;
        }
        let regions = regions.to_vec();
        self.record_or_run(
            command,
            std::sync::Arc::new(move |device, command| unsafe {
                device
                    .cmd_copy_image_to_buffer(command, src_image, src_layout, dst_buffer, &regions);
            }),
        );
    }

    pub unsafe fn cmd_copy_image(
        &self,
        command: vk::CommandBuffer,
        src_image: vk::Image,
        src_layout: vk::ImageLayout,
        dst_image: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::ImageCopy],
    ) {
        if command.as_raw() & CAPTURE_TAG != CAPTURE_TAG {
            unsafe {
                self.device.cmd_copy_image(
                    command, src_image, src_layout, dst_image, dst_layout, regions,
                );
            }
            return;
        }
        let regions = regions.to_vec();
        self.record_or_run(
            command,
            std::sync::Arc::new(move |device, command| unsafe {
                device.cmd_copy_image(
                    command, src_image, src_layout, dst_image, dst_layout, &regions,
                );
            }),
        );
    }

    pub unsafe fn cmd_blit_image(
        &self,
        command: vk::CommandBuffer,
        src_image: vk::Image,
        src_layout: vk::ImageLayout,
        dst_image: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::ImageBlit],
        filter: vk::Filter,
    ) {
        if command.as_raw() & CAPTURE_TAG != CAPTURE_TAG {
            unsafe {
                self.device.cmd_blit_image(
                    command, src_image, src_layout, dst_image, dst_layout, regions, filter,
                );
            }
            return;
        }
        let regions = regions.to_vec();
        self.record_or_run(
            command,
            std::sync::Arc::new(move |device, command| unsafe {
                device.cmd_blit_image(
                    command, src_image, src_layout, dst_image, dst_layout, &regions, filter,
                );
            }),
        );
    }

    pub unsafe fn cmd_clear_color_image(
        &self,
        command: vk::CommandBuffer,
        image: vk::Image,
        layout: vk::ImageLayout,
        color: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        if command.as_raw() & CAPTURE_TAG != CAPTURE_TAG {
            unsafe {
                self.device
                    .cmd_clear_color_image(command, image, layout, color, ranges);
            }
            return;
        }
        let color = *color;
        let ranges = ranges.to_vec();
        self.record_or_run(
            command,
            std::sync::Arc::new(move |device, command| unsafe {
                device.cmd_clear_color_image(command, image, layout, &color, &ranges);
            }),
        );
    }

    pub unsafe fn cmd_clear_depth_stencil_image(
        &self,
        command: vk::CommandBuffer,
        image: vk::Image,
        layout: vk::ImageLayout,
        value: &vk::ClearDepthStencilValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        if command.as_raw() & CAPTURE_TAG != CAPTURE_TAG {
            unsafe {
                self.device
                    .cmd_clear_depth_stencil_image(command, image, layout, value, ranges);
            }
            return;
        }
        let value = *value;
        let ranges = ranges.to_vec();
        self.record_or_run(
            command,
            std::sync::Arc::new(move |device, command| unsafe {
                device.cmd_clear_depth_stencil_image(command, image, layout, &value, &ranges);
            }),
        );
    }

    pub unsafe fn cmd_begin_render_pass(
        &self,
        command: vk::CommandBuffer,
        begin: &vk::RenderPassBeginInfo<'_>,
        contents: vk::SubpassContents,
    ) {
        if command.as_raw() & CAPTURE_TAG != CAPTURE_TAG {
            unsafe {
                self.device.cmd_begin_render_pass(command, begin, contents);
            }
            return;
        }
        assert!(
            begin.p_next.is_null(),
            "render pass p_next cannot be captured"
        );
        let render_pass = begin.render_pass;
        let framebuffer = begin.framebuffer;
        let render_area = begin.render_area;
        let clear_values = if begin.clear_value_count == 0 {
            Vec::new()
        } else {
            assert!(
                !begin.p_clear_values.is_null(),
                "render pass clear values are null"
            );
            unsafe {
                std::slice::from_raw_parts(begin.p_clear_values, begin.clear_value_count as usize)
                    .to_vec()
            }
        };
        self.record_or_run(
            command,
            std::sync::Arc::new(move |device, command| unsafe {
                let begin = vk::RenderPassBeginInfo::default()
                    .render_pass(render_pass)
                    .framebuffer(framebuffer)
                    .render_area(render_area)
                    .clear_values(&clear_values);
                device.cmd_begin_render_pass(command, &begin, contents);
            }),
        );
    }

    pub unsafe fn cmd_end_render_pass(&self, command: vk::CommandBuffer) {
        if command.as_raw() & CAPTURE_TAG != CAPTURE_TAG {
            unsafe {
                self.device.cmd_end_render_pass(command);
            }
            return;
        }
        self.record_or_run(
            command,
            std::sync::Arc::new(move |device, command| unsafe {
                device.cmd_end_render_pass(command);
            }),
        );
    }

    pub unsafe fn cmd_pipeline_barrier(
        &self,
        command: vk::CommandBuffer,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier<'_>],
        buffer_barriers: &[vk::BufferMemoryBarrier<'_>],
        image_barriers: &[vk::ImageMemoryBarrier<'_>],
    ) {
        if command.as_raw() & CAPTURE_TAG != CAPTURE_TAG {
            unsafe {
                self.device.cmd_pipeline_barrier(
                    command,
                    src_stage_mask,
                    dst_stage_mask,
                    dependency_flags,
                    memory_barriers,
                    buffer_barriers,
                    image_barriers,
                );
            }
            return;
        }
        let memory_barriers: Vec<_> = memory_barriers
            .iter()
            .map(OwnedMemoryBarrier::new)
            .collect();
        let buffer_barriers: Vec<_> = buffer_barriers
            .iter()
            .map(OwnedBufferMemoryBarrier::new)
            .collect();
        let image_barriers: Vec<_> = image_barriers
            .iter()
            .map(OwnedImageMemoryBarrier::new)
            .collect();
        self.record_or_run(
            command,
            std::sync::Arc::new(move |device, command| unsafe {
                let memory_barriers: Vec<_> = memory_barriers
                    .iter()
                    .map(OwnedMemoryBarrier::get)
                    .collect();
                let buffer_barriers: Vec<_> = buffer_barriers
                    .iter()
                    .map(OwnedBufferMemoryBarrier::get)
                    .collect();
                let image_barriers: Vec<_> = image_barriers
                    .iter()
                    .map(OwnedImageMemoryBarrier::get)
                    .collect();
                device.cmd_pipeline_barrier(
                    command,
                    src_stage_mask,
                    dst_stage_mask,
                    dependency_flags,
                    &memory_barriers,
                    &buffer_barriers,
                    &image_barriers,
                );
            }),
        );
    }
}

impl Deref for RecordingDevice<'_> {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        self.device
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if self.owner == std::thread::current().id() {
            CAPTURES.with(|captures| {
                captures.borrow_mut().remove(&self.command.as_raw());
            });
        }
    }
}

struct OwnedMemoryBarrier {
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
}

impl OwnedMemoryBarrier {
    fn new(value: &vk::MemoryBarrier<'_>) -> Self {
        assert!(
            value.p_next.is_null(),
            "memory barrier p_next cannot be captured"
        );
        Self {
            src_access_mask: value.src_access_mask,
            dst_access_mask: value.dst_access_mask,
        }
    }

    fn get(&self) -> vk::MemoryBarrier<'static> {
        vk::MemoryBarrier {
            p_next: ptr::null(),
            src_access_mask: self.src_access_mask,
            dst_access_mask: self.dst_access_mask,
            ..Default::default()
        }
    }
}

struct OwnedBufferMemoryBarrier {
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
    src_queue_family_index: u32,
    dst_queue_family_index: u32,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

impl OwnedBufferMemoryBarrier {
    fn new(value: &vk::BufferMemoryBarrier<'_>) -> Self {
        assert!(
            value.p_next.is_null(),
            "buffer barrier p_next cannot be captured"
        );
        Self {
            src_access_mask: value.src_access_mask,
            dst_access_mask: value.dst_access_mask,
            src_queue_family_index: value.src_queue_family_index,
            dst_queue_family_index: value.dst_queue_family_index,
            buffer: value.buffer,
            offset: value.offset,
            size: value.size,
        }
    }

    fn get(&self) -> vk::BufferMemoryBarrier<'static> {
        vk::BufferMemoryBarrier {
            p_next: ptr::null(),
            src_access_mask: self.src_access_mask,
            dst_access_mask: self.dst_access_mask,
            src_queue_family_index: self.src_queue_family_index,
            dst_queue_family_index: self.dst_queue_family_index,
            buffer: self.buffer,
            offset: self.offset,
            size: self.size,
            ..Default::default()
        }
    }
}

struct OwnedImageMemoryBarrier {
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_queue_family_index: u32,
    dst_queue_family_index: u32,
    image: vk::Image,
    subresource_range: vk::ImageSubresourceRange,
}

impl OwnedImageMemoryBarrier {
    fn new(value: &vk::ImageMemoryBarrier<'_>) -> Self {
        assert!(
            value.p_next.is_null(),
            "image barrier p_next cannot be captured"
        );
        Self {
            src_access_mask: value.src_access_mask,
            dst_access_mask: value.dst_access_mask,
            old_layout: value.old_layout,
            new_layout: value.new_layout,
            src_queue_family_index: value.src_queue_family_index,
            dst_queue_family_index: value.dst_queue_family_index,
            image: value.image,
            subresource_range: value.subresource_range,
        }
    }

    fn get(&self) -> vk::ImageMemoryBarrier<'static> {
        vk::ImageMemoryBarrier {
            p_next: ptr::null(),
            src_access_mask: self.src_access_mask,
            dst_access_mask: self.dst_access_mask,
            old_layout: self.old_layout,
            new_layout: self.new_layout,
            src_queue_family_index: self.src_queue_family_index,
            dst_queue_family_index: self.dst_queue_family_index,
            image: self.image,
            subresource_range: self.subresource_range,
            ..Default::default()
        }
    }
}
