//! Bounded command reuse with timeline completion values.

use super::Context;
use ash::vk;
use std::sync::Arc;

const FRAMES: usize = 2;

#[derive(Clone, PartialEq, Eq)]
pub(super) struct TransferKey {
    /// Image handle included in the cached transfer identity.
    pub image: u64,
    /// Scratch buffer handle included in the cached transfer identity.
    pub scratch: u64,
    /// Arena buffer device address for this transfer.
    pub address: u64,
    /// Scratch buffer device address for this transfer.
    pub scratch_address: u64,
    /// Tracked Vulkan image layout at recording time.
    pub layout: i32,
    /// True copies arena storage to the image; false copies it back.
    pub load: bool,
    pub conversion: bool,
    /// Checked byte packing for the image storage.
    pub packing: crate::tiling::Layout,
}

#[derive(Clone)]
pub(super) struct Submission {
    // Fields drop in order. Release the semaphore before its final device owner.
    timeline: Arc<super::command_workers::Timeline>,
    value: u64,
    receipt: (super::command_workers::Completion, u64),
    context: Arc<Context>,
}

impl Submission {
    pub fn ready(&self) -> Option<bool> {
        self.receipt.0.ready()?;
        if !self.timeline.is_posted(self.value) {
            return Some(false);
        }
        let complete = unsafe {
            self.context
                .device
                .get_semaphore_counter_value(self.timeline.semaphore())
                .ok()?
                >= self.value
        };
        if complete {
            self.context.command_workers.retire(self.receipt.1);
        }
        Some(complete)
    }

    pub fn wait(&self) -> Option<()> {
        self.context.command_workers.flush_direct();
        let semaphores = [self.timeline.semaphore()];
        let values = [self.value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        while !self.ready()? {
            self.context.command_workers.healthy()?;
            if !self.timeline.wait_posted(self.value) {
                continue;
            }
            match unsafe { self.context.device.wait_semaphores(&info, 100_000_000) } {
                Ok(()) | Err(vk::Result::TIMEOUT) => {}
                Err(_) => return None,
            }
        }
        Some(())
    }
}

struct Frame {
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    completion: u64,
    acquire: vk::Semaphore,
    cached: Option<Vec<TransferKey>>,
    receipt: Option<(super::command_workers::Completion, u64)>,
}

pub(super) struct Commands {
    context: Arc<Context>,
    frames: Vec<Frame>,
    next: usize,
    timeline: Option<Arc<super::command_workers::Timeline>>,
    submitted: u64,
    capture: Option<super::recording_device::Capture>,
    reusing: bool,
    pending: Option<Vec<TransferKey>>,
}

impl Commands {
    /// Create reusable execution resources; return None if Vulkan setup fails.
    pub fn new(context: &Arc<Context>) -> Option<Self> {
        Self::with_capacity(context, FRAMES)
    }

    pub fn with_capacity(context: &Arc<Context>, capacity: usize) -> Option<Self> {
        let mut commands = Self {
            context: Arc::clone(context),
            frames: Vec::new(),
            next: 0,
            timeline: None,
            submitted: 0,
            capture: None,
            reusing: false,
            pending: None,
        };
        commands.timeline = Some(super::command_workers::Timeline::new(&context.device)?);
        for _ in 0..capacity {
            let device = &context.device;
            // Own each partially created frame before the next fallible call.
            let pool = unsafe {
                device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(context.queue_family),
                        None,
                    )
                    .ok()?
            };
            commands.frames.push(Frame {
                pool,
                command: vk::CommandBuffer::null(),
                completion: 0,
                acquire: vk::Semaphore::null(),
                cached: None,
                receipt: None,
            });
            let frame = commands.frames.last_mut()?;
            frame.command = unsafe {
                device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .ok()?[0]
            };
            frame.acquire = unsafe {
                device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .ok()?
            };
        }
        Some(commands)
    }

    pub fn slot(&self) -> usize {
        self.next
    }

    pub fn slots(&self) -> usize {
        self.frames.len()
    }

    /// Wait and discard cached recordings before their resources change.
    pub fn invalidate_cached(&mut self) {
        for frame in &mut self.frames {
            frame.cached = None;
        }
        self.pending = None;
        self.reusing = false;
    }

    /// Begin a fresh command recording and return its reusable completion semaphore.
    pub fn begin(&mut self) -> Option<(vk::CommandBuffer, vk::Semaphore)> {
        self.capture.take();
        self.reusing = false;
        self.pending = None;
        let frame = &self.frames[self.next];
        self.wait_value(frame.completion)?;
        self.frames[self.next].cached = None;
        let capture = super::recording_device::Capture::new();
        let command = capture.command();
        self.capture = Some(capture);
        Some((command, self.frames[self.next].acquire))
    }

    /// Select a transfer recording by complete identity; report whether it already exists.
    pub fn begin_cached(&mut self, key: Vec<TransferKey>) -> Option<(vk::CommandBuffer, bool)> {
        let frame = &self.frames[self.next];
        if frame.cached.as_ref() == Some(&key) {
            self.wait_value(frame.completion)?;
            self.capture.take();
            self.reusing = true;
            self.pending = None;
            return Some((frame.command, true));
        }
        let (command, _) = self.begin_with_flags(vk::CommandBufferUsageFlags::empty())?;
        self.pending = Some(key);
        Some((command, false))
    }

    pub fn begin_direct(&mut self) -> Option<(vk::CommandBuffer, vk::Semaphore)> {
        self.begin_with_flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
    }
    fn begin_with_flags(
        &mut self,
        flags: vk::CommandBufferUsageFlags,
    ) -> Option<(vk::CommandBuffer, vk::Semaphore)> {
        self.capture.take();
        self.reusing = false;
        self.pending = None;
        self.frames[self.next].cached = None;
        let frame = &self.frames[self.next];
        let device = &self.context.device;
        self.wait_value(frame.completion)?;
        unsafe {
            device
                .reset_command_pool(frame.pool, vk::CommandPoolResetFlags::empty())
                .ok()?;
            device
                .begin_command_buffer(
                    frame.command,
                    &vk::CommandBufferBeginInfo::default().flags(flags),
                )
                .ok()?;
        }
        Some((frame.command, frame.acquire))
    }

    /// Submit the selected command; optionally wait and signal the supplied semaphore.
    pub fn submit(&mut self, wait: bool, signal: Option<vk::Semaphore>) -> Option<()> {
        let value = self.submitted.checked_add(1)?;
        let frame = &mut self.frames[self.next];
        let operations = self.capture.take().map(|capture| capture.finish());
        let receipt = if let Some(operations) = operations {
            self.context.command_workers.enqueue_timeline(
                operations,
                wait.then_some(frame.acquire),
                signal,
                Some((self.timeline.as_ref()?.clone(), value)),
            )
        } else {
            if !self.reusing {
                unsafe {
                    self.context.device.end_command_buffer(frame.command).ok()?;
                }
            }
            if let Some(key) = self.pending.take() {
                frame.cached = Some(key);
            }
            self.context.command_workers.direct(
                frame.command,
                wait.then_some(frame.acquire),
                signal,
                (self.timeline.as_ref()?.clone(), value),
            )
        };
        frame.receipt = Some(receipt);
        self.submitted = value;
        frame.completion = value;
        self.next = (self.next + 1) % self.frames.len();
        Some(())
    }

    /// Wait for retained command submissions to complete; return None on failure.
    pub fn wait(&self) -> Option<()> {
        self.wait_value(self.submitted)
    }

    pub fn submission(&self) -> Option<Submission> {
        let frame = self
            .frames
            .iter()
            .find(|f| f.completion == self.submitted)?;
        Some(Submission {
            context: self.context.clone(),
            timeline: self.timeline.as_ref()?.clone(),
            value: self.submitted,
            receipt: frame.receipt.as_ref()?.clone(),
        })
    }

    pub fn completion(&self) -> u64 {
        self.submitted
    }

    pub fn ready(&self, value: u64) -> Option<bool> {
        self.context.command_workers.healthy()?;
        if !self.timeline.as_ref()?.is_posted(value) {
            self.context.command_workers.flush_direct();
            return Some(false);
        }
        let complete = unsafe {
            self.context
                .device
                .get_semaphore_counter_value(self.timeline.as_ref()?.semaphore())
                .ok()?
                >= value
        };
        if !complete {
            self.context.command_workers.flush_direct();
            if let Some(receipt) = self
                .frames
                .iter()
                .find(|f| f.completion == value)
                .and_then(|f| f.receipt.as_ref())
            {
                receipt.0.ready()?;
            }
        }
        if complete {
            if let Some((_, position)) = self
                .frames
                .iter()
                .find(|f| f.completion == value)
                .and_then(|f| f.receipt.as_ref())
            {
                self.context.command_workers.retire(*position);
            }
        }
        Some(complete)
    }

    pub fn wait_value(&self, value: u64) -> Option<()> {
        if value == 0 || self.ready(value)? {
            return Some(());
        }
        while !self.timeline.as_ref()?.wait_posted(value) {
            self.context.command_workers.healthy()?;
        }
        let semaphores = [self.timeline.as_ref()?.semaphore()];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        loop {
            if let Some(receipt) = self
                .frames
                .iter()
                .find(|f| f.completion == value)
                .and_then(|f| f.receipt.as_ref())
            {
                receipt.0.ready()?;
            }
            match unsafe { self.context.device.wait_semaphores(&info, 100_000_000) } {
                Ok(()) => {
                    if let Some((_, position)) = self
                        .frames
                        .iter()
                        .find(|f| f.completion == value)
                        .and_then(|f| f.receipt.as_ref())
                    {
                        self.context.command_workers.retire(*position);
                    }
                    return Some(());
                }
                Err(vk::Result::TIMEOUT) => {}
                Err(_) => return None,
            }
        }
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        let _ = self.wait();
        if let Some(timeline) = self.timeline.take() {
            self.context.command_workers.forget_timeline(&timeline);
            drop(timeline);
        }
        unsafe {
            let device = &self.context.device;

            for frame in &self.frames {
                device.destroy_semaphore(frame.acquire, None);
                device.destroy_command_pool(frame.pool, None);
            }
        }
    }
}
