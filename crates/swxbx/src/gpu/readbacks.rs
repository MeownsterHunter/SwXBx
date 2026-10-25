//! Per-frame offscreen readback resources.

use super::{
    commands::Commands,
    images::{buffer_barrier, can_blit, record_present, transition, Buffer, Image, ImageInfo},
    Context,
};
use ash::vk;
use std::sync::Arc;

struct Slot {
    commands: Commands,
    target: Image,
    buffer: Buffer,
}

impl Drop for Slot {
    fn drop(&mut self) {
        let _ = self.commands.wait();
        // These resources are private to this slot and retired by its timeline.
        self.target.wait_on_drop = false;
        self.buffer.wait_on_drop = false;
    }
}

pub(super) struct Readbacks {
    context: Arc<Context>,
    slots: Vec<Option<Slot>>,
}

impl Readbacks {
    pub fn new(context: &Arc<Context>) -> Self {
        Self {
            context: Arc::clone(context),
            slots: Vec::new(),
        }
    }

    pub fn submit(
        &mut self,
        index: usize,
        source: ImageInfo,
        width: u32,
        height: u32,
    ) -> Option<()> {
        if !can_blit(&self.context, source.format, vk::Format::R8G8B8A8_UNORM) {
            return None;
        }
        self.ensure_slot(index, width, height)?;
        let slot = self.slots[index].as_mut()?;
        let (cmd, _) = slot.commands.begin_direct()?;
        let copy = if source.format == slot.target.info.format
            && source.extent == slot.target.info.extent
            && source.shape.kind == crate::tiling::ImageKind::D2
        {
            transition(
                &self.context.recorder(),
                cmd,
                source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            source
        } else {
            record_present(
                &self.context.recorder(),
                cmd,
                source,
                slot.target.info,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            slot.target.info.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            slot.target.info
        };
        unsafe {
            self.context.recorder().cmd_copy_image_to_buffer(
                cmd,
                copy.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                slot.buffer.buffer,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(copy.extent)],
            );
        }
        buffer_barrier(&self.context.recorder(), cmd);
        slot.commands.submit(false, None)?;
        self.context.command_workers.flush_direct();
        Some(())
    }

    fn ensure_slot(&mut self, index: usize, width: u32, height: u32) -> Option<()> {
        self.slots
            .resize_with(self.slots.len().max(index + 1), || None);
        if !self.slots[index].as_ref().is_some_and(|s| {
            s.target.info.extent.width == width && s.target.info.extent.height == height
        }) {
            let target = Image::new(&self.context, width, height, vk::Format::R8G8B8A8_UNORM)?;
            let buffer = Buffer::new(&self.context, target.info.size)?;
            self.slots[index] = Some(Slot {
                commands: Commands::with_capacity(&self.context, 1)?,
                target,
                buffer,
            });
        }
        Some(())
    }

    pub fn submit_buffer(
        &mut self,
        index: usize,
        source: vk::Buffer,
        offset: u64,
        width: u32,
        height: u32,
    ) -> Option<()> {
        self.ensure_slot(index, width, height)?;
        let slot = self.slots[index].as_mut()?;
        let (cmd, _) = slot.commands.begin_direct()?;
        buffer_barrier(&self.context.recorder(), cmd);
        unsafe {
            self.context.recorder().cmd_copy_buffer(
                cmd,
                source,
                slot.buffer.buffer,
                &[vk::BufferCopy::default()
                    .src_offset(offset)
                    .size(slot.target.info.size)],
            );
        }
        buffer_barrier(&self.context.recorder(), cmd);
        slot.commands.submit(false, None)?;
        self.context.command_workers.flush_direct();
        Some(())
    }

    pub(super) fn submission(&self, index: usize) -> Option<super::commands::Submission> {
        self.slots.get(index)?.as_ref()?.commands.submission()
    }

    pub fn read(&self, index: usize, wait: bool) -> Option<Option<(u32, u32, Vec<u8>)>> {
        let slot = self.slots.get(index)?.as_ref()?;
        let value = slot.commands.completion();
        if wait {
            slot.commands.wait_value(value)?;
        } else if !slot.commands.ready(value)? {
            return Some(None);
        }
        let info = slot.target.info;
        Some(Some((
            info.extent.width,
            info.extent.height,
            slot.buffer.read(info.size as usize)?,
        )))
    }

    pub fn wait(&self) -> Option<()> {
        for slot in self.slots.iter().flatten() {
            slot.commands.wait()?;
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::Backend;

    struct Gate {
        context: Arc<Context>,
        semaphore: vk::Semaphore,
    }
    impl Gate {
        fn open(&self) {
            unsafe {
                self.context
                    .device
                    .signal_semaphore(
                        &vk::SemaphoreSignalInfo::default()
                            .semaphore(self.semaphore)
                            .value(1),
                    )
                    .unwrap();
            }
        }
    }
    impl Drop for Gate {
        fn drop(&mut self) {
            if unsafe {
                self.context
                    .device
                    .get_semaphore_counter_value(self.semaphore)
            }
            .unwrap_or(1)
                == 0
            {
                self.open();
            }
            let _ = self.context.wait_queue();
            unsafe {
                self.context.device.destroy_semaphore(self.semaphore, None);
            }
        }
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn pending_gpu_readback_can_be_polled_without_waiting() {
        let mut backend = Backend::new(1.0).expect("GPU is required");
        assert!(backend.ensure(1, 2, 2, false));
        assert!(backend.clear_color(1, [1.0, 0.0, 0.0, 1.0], 15));
        assert!(backend.finish());
        let context = Arc::clone(backend.context());
        let mut ty =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let semaphore = unsafe {
            context
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut ty), None)
                .unwrap()
        };
        let gate = Gate {
            context: Arc::clone(&context),
            semaphore,
        };
        let waits = [semaphore];
        let values = [1];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stages)
            .push_next(&mut timeline);
        unsafe {
            context
                .device
                .queue_submit(context.queue, &[submit], vk::Fence::null())
                .unwrap();
        }
        backend.enqueue_callback(0, 1, 2, 2).unwrap();
        assert!(backend.read_callback(0, false).unwrap().is_none());
        gate.open();
        let (_, _, pixels) = backend.read_callback(0, true).unwrap().unwrap();
        assert_eq!(pixels, [255, 0, 0, 255].repeat(4));
    }
}
