//! Worker-local pools and ordered queue ownership.
use super::recording_device::Operation;
use ash::vk;
#[cfg(test)]
use std::time::Duration;
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    thread::{self, JoinHandle},
};
#[derive(Clone, Default)]
pub(super) struct Completion(Arc<(Mutex<Option<bool>>, Condvar)>);
impl Completion {
    fn finish(&self, success: bool) {
        let (state, changed) = &*self.0;
        *state.lock().unwrap_or_else(|p| p.into_inner()) = Some(success);
        changed.notify_all();
    }
    pub fn ready(&self) -> Option<bool> {
        match *self.0 .0.lock().unwrap_or_else(|p| p.into_inner()) {
            Some(false) => None,
            Some(true) => Some(true),
            None => Some(false),
        }
    }
    pub fn wait(&self) -> Option<()> {
        let (state, changed) = &*self.0;
        let mut result = state.lock().unwrap_or_else(|p| p.into_inner());
        while result.is_none() {
            result = changed.wait(result).unwrap_or_else(|p| p.into_inner());
        }
        result.unwrap().then_some(())
    }
}
pub(super) struct Timeline {
    device: ash::Device,
    semaphore: vk::Semaphore,
    queued: Mutex<u64>,
    posted: Condvar,
}
impl Timeline {
    pub fn new(device: &ash::Device) -> Option<Arc<Self>> {
        let mut ty =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let semaphore = unsafe {
            device
                .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut ty), None)
                .ok()?
        };
        Some(Arc::new(Self {
            device: device.clone(),
            semaphore,
            queued: Mutex::new(0),
            posted: Condvar::new(),
        }))
    }
    fn post(&self, value: u64) {
        let mut queued = self.queued.lock().unwrap_or_else(|p| p.into_inner());
        *queued = (*queued).max(value);
        self.posted.notify_all();
    }
    pub fn is_posted(&self, value: u64) -> bool {
        *self.queued.lock().unwrap_or_else(|p| p.into_inner()) >= value
    }
    pub fn wait_posted(&self, value: u64) -> bool {
        let queued = self.queued.lock().unwrap_or_else(|p| p.into_inner());
        if *queued >= value {
            return true;
        }
        let (queued, _) = self
            .posted
            .wait_timeout(queued, std::time::Duration::from_millis(100))
            .unwrap_or_else(|p| p.into_inner());
        *queued >= value
    }
    pub fn semaphore(&self) -> vk::Semaphore {
        self.semaphore
    }
}
impl Drop for Timeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}
#[derive(Clone)]
pub(super) struct TimelineWait {
    pub timeline: Arc<Timeline>,
    pub value: u64,
    pub position: u64,
}

struct Job {
    ticket: u64,
    operations: Vec<Operation>,
    wait: Option<vk::Semaphore>,
    signal: Option<vk::Semaphore>,
    timeline: Option<(Arc<Timeline>, u64)>,
    completion: Completion,
}
enum Payload {
    Recorded {
        command: vk::CommandBuffer,
        fence: vk::Fence,
        wait: Option<vk::Semaphore>,
        signal: Option<vk::Semaphore>,
        timeline: Option<(Arc<Timeline>, u64)>,
    },
    Direct(Box<DirectBatch>),
    Checkpoint(vk::Semaphore, u64, mpsc::SyncSender<Result<(), vk::Result>>),
    Failed,
    Present {
        queue: vk::Queue,
        loader: ash::khr::swapchain::Device,
        swapchain: vk::SwapchainKHR,
        index: u32,
        waits: Vec<vk::Semaphore>,
        fence: vk::Fence,
        reply: mpsc::SyncSender<Result<bool, vk::Result>>,
    },
}
const DIRECT_CAPACITY: usize = 16;
struct Direct {
    command: vk::CommandBuffer,
    wait: Option<vk::Semaphore>,
    signal: Option<vk::Semaphore>,
    timeline: (Arc<Timeline>, u64),
}
struct DirectBatch {
    entries: [Option<Direct>; DIRECT_CAPACITY],
    len: usize,
}
impl Default for DirectBatch {
    fn default() -> Self {
        Self {
            entries: std::array::from_fn(|_| None),
            len: 0,
        }
    }
}
struct Ready {
    ticket: u64,
    payload: Payload,
    completion: Completion,
}
pub(super) struct Runtime {
    next: AtomicU64,
    record_next: AtomicU64,
    sequence: AtomicU64,
    retired: AtomicU64,
    direct: Mutex<DirectBatch>,
    direct_failure: Completion,
    latest: Mutex<Option<TimelineWait>>,
    jobs: Vec<mpsc::Sender<Job>>,
    ready: Option<mpsc::Sender<Ready>>,
    workers: Vec<JoinHandle<()>>,
    submit: Option<JoinHandle<()>>,
}
struct Pool {
    device: ash::Device,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
}
impl Pool {
    fn new(device: ash::Device, family: u32) -> Option<Self> {
        let pool = unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                    None,
                )
                .ok()?
        };
        let mut resources = Self {
            device,
            pool,
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
        };
        resources.command = unsafe {
            resources
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .ok()?[0]
        };
        resources.fence = unsafe {
            resources
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .ok()?
        };
        Some(resources)
    }
    fn record(&self, operations: &[Operation]) -> Option<()> {
        unsafe {
            self.device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .ok()?;
            self.device
                .begin_command_buffer(
                    self.command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .ok()?;
            self.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)],
                &[],
                &[],
            );
            for operation in operations {
                operation(&self.device, self.command);
            }
            self.device.end_command_buffer(self.command).ok()?;
        }
        Some(())
    }
}
impl Drop for Pool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}
impl Runtime {
    pub fn new(device: &ash::Device, queue: vk::Queue, family: u32) -> Self {
        let (ready, receiver) = mpsc::channel();
        let submit_device = device.clone();
        let direct_failure = Completion::default();
        let submit_failure = direct_failure.clone();
        let submit = thread::Builder::new()
            .name("queue-submit".into())
            .spawn(move || submit_loop(submit_device, queue, receiver, submit_failure))
            .expect("submit thread");
        let count = thread::available_parallelism()
            .map_or(2, usize::from)
            .clamp(2, 8);
        let mut jobs = Vec::new();
        let mut workers = Vec::new();
        for index in 0..count {
            let (sender, receiver) = mpsc::channel::<Job>();
            jobs.push(sender);
            let ready = ready.clone();
            let device = device.clone();
            workers.push(
                thread::Builder::new()
                    .name(format!("command-worker-{index}"))
                    .spawn(move || {
                        let pool = Pool::new(device, family);
                        let mut healthy = pool.is_some();
                        for job in receiver {
                            let result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    if healthy {
                                        pool.as_ref()?.record(&job.operations)?;
                                        Some(())
                                    } else {
                                        None
                                    }
                                }))
                                .ok()
                                .flatten();
                            let payload = if result.is_some() {
                                let pool = pool.as_ref().unwrap();
                                Payload::Recorded {
                                    command: pool.command,
                                    fence: pool.fence,
                                    wait: job.wait,
                                    signal: job.signal,
                                    timeline: job.timeline,
                                }
                            } else {
                                Payload::Failed
                            };
                            let completion = job.completion.clone();
                            if ready
                                .send(Ready {
                                    ticket: job.ticket,
                                    payload,
                                    completion,
                                })
                                .is_err()
                            {
                                job.completion.finish(false);
                                break;
                            }
                            // One active buffer per private pool. Reset only after fence retirement.
                            healthy &= job.completion.wait().is_some();
                        }
                    })
                    .expect("command worker"),
            );
        }
        Self {
            next: AtomicU64::new(0),
            record_next: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
            retired: AtomicU64::new(0),
            direct: Mutex::new(DirectBatch::default()),
            direct_failure,
            latest: Mutex::new(None),
            jobs,
            ready: Some(ready),
            workers,
            submit: Some(submit),
        }
    }
    #[cfg(test)]
    pub fn enqueue(
        &self,
        operations: Vec<Operation>,
        wait: Option<vk::Semaphore>,
        signal: Option<vk::Semaphore>,
    ) -> Completion {
        self.enqueue_timeline(operations, wait, signal, None).0
    }
    pub fn enqueue_timeline(
        &self,
        operations: Vec<Operation>,
        wait: Option<vk::Semaphore>,
        signal: Option<vk::Semaphore>,
        timeline: Option<(Arc<Timeline>, u64)>,
    ) -> (Completion, u64) {
        let mut direct = self.direct.lock().unwrap_or_else(|p| p.into_inner());
        self.flush_batch(&mut direct);
        let ticket = self.next.fetch_add(1, Ordering::Relaxed);
        let position = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        *self.latest.lock().unwrap_or_else(|p| p.into_inner()) =
            timeline.as_ref().map(|(timeline, value)| TimelineWait {
                timeline: timeline.clone(),
                value: *value,
                position,
            });
        let completion = Completion::default();
        let job = Job {
            ticket,
            operations,
            wait,
            signal,
            timeline,
            completion: completion.clone(),
        };
        let worker = self.record_next.fetch_add(1, Ordering::Relaxed) as usize % self.jobs.len();
        if self.jobs[worker].send(job).is_err() {
            let _ = self.ready.as_ref().unwrap().send(Ready {
                ticket,
                payload: Payload::Failed,
                completion: completion.clone(),
            });
        }
        (completion, position)
    }
    pub fn direct(
        &self,
        command: vk::CommandBuffer,
        wait: Option<vk::Semaphore>,
        signal: Option<vk::Semaphore>,
        timeline: (Arc<Timeline>, u64),
    ) -> (Completion, u64) {
        let completion = self.direct_failure.clone();
        let mut direct = self.direct.lock().unwrap_or_else(|p| p.into_inner());
        let position = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        *self.latest.lock().unwrap_or_else(|p| p.into_inner()) = Some(TimelineWait {
            timeline: timeline.0.clone(),
            value: timeline.1,
            position,
        });
        let index = direct.len;
        direct.entries[index] = Some(Direct {
            command,
            wait,
            signal,
            timeline,
        });
        direct.len += 1;
        if direct.len == DIRECT_CAPACITY {
            self.flush_batch(&mut direct);
        }
        (completion, position)
    }
    pub fn flush_direct(&self) {
        self.flush_batch(&mut self.direct.lock().unwrap_or_else(|p| p.into_inner()));
    }
    fn flush_batch(&self, direct: &mut DirectBatch) {
        if direct.len == 0 {
            return;
        }
        let completion = self.direct_failure.clone();
        let batch = Box::new(std::mem::take(direct));
        let ticket = self.next.fetch_add(1, Ordering::Relaxed);
        if self
            .ready
            .as_ref()
            .unwrap()
            .send(Ready {
                ticket,
                payload: Payload::Direct(batch),
                completion,
            })
            .is_err()
        {
            self.direct_failure.finish(false);
        }
    }
    pub fn latest_timeline(&self) -> Option<TimelineWait> {
        let mut direct = self.direct.lock().unwrap_or_else(|p| p.into_inner());
        let latest = self
            .latest
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()?;
        if latest.position != self.sequence.load(Ordering::Relaxed) {
            return None;
        }
        self.flush_batch(&mut direct);
        Some(latest)
    }
    pub fn forget_timeline(&self, timeline: &Arc<Timeline>) {
        let _direct = self.direct.lock().unwrap_or_else(|p| p.into_inner());
        let mut latest = self.latest.lock().unwrap_or_else(|p| p.into_inner());
        if latest
            .as_ref()
            .is_some_and(|wait| Arc::ptr_eq(&wait.timeline, timeline))
        {
            latest.take();
        }
    }
    pub fn healthy(&self) -> Option<()> {
        self.direct_failure.ready().map(|_| ())
    }
    pub fn retire(&self, position: u64) {
        self.retired.fetch_max(position, Ordering::Release);
    }
    pub fn unchanged_since(&self, position: u64) -> Option<bool> {
        self.direct_failure.ready()?;
        let direct = self.direct.lock().unwrap_or_else(|p| p.into_inner());
        let sequence = self.sequence.load(Ordering::Relaxed);
        Some(
            direct.len == 0
                && (sequence == position || self.retired.load(Ordering::Acquire) >= sequence),
        )
    }
    pub fn checkpoint(&self, semaphore: vk::Semaphore, value: u64) -> Result<u64, vk::Result> {
        let (reply, result) = mpsc::sync_channel(1);
        let position = self.action(Payload::Checkpoint(semaphore, value, reply))?;
        result
            .recv()
            .unwrap_or(Err(vk::Result::ERROR_DEVICE_LOST))?;
        Ok(position)
    }
    fn action(&self, payload: Payload) -> Result<u64, vk::Result> {
        let mut direct = self.direct.lock().unwrap_or_else(|p| p.into_inner());
        self.flush_batch(&mut direct);
        let ticket = self.next.fetch_add(1, Ordering::Relaxed);
        let position = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        self.latest.lock().unwrap_or_else(|p| p.into_inner()).take();
        self.ready
            .as_ref()
            .unwrap()
            .send(Ready {
                ticket,
                payload,
                completion: Completion::default(),
            })
            .map(|()| position)
            .map_err(|_| vk::Result::ERROR_DEVICE_LOST)
    }
    pub fn present(
        &self,
        queue: vk::Queue,
        loader: &ash::khr::swapchain::Device,
        swapchain: vk::SwapchainKHR,
        index: u32,
        waits: &[vk::Semaphore],
        fence: vk::Fence,
    ) -> Result<bool, vk::Result> {
        let (reply, result) = mpsc::sync_channel(1);
        self.action(Payload::Present {
            queue,
            loader: loader.clone(),
            swapchain,
            index,
            waits: waits.to_vec(),
            fence,
            reply,
        })?;
        result.recv().unwrap_or(Err(vk::Result::ERROR_DEVICE_LOST))
    }
    pub fn shutdown(&mut self) {
        if self.ready.is_none() {
            return;
        }
        self.flush_direct();
        self.jobs.clear();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
        self.ready.take();
        if let Some(submit) = self.submit.take() {
            let _ = submit.join();
        }
        self.latest
            .get_mut()
            .unwrap_or_else(|p| p.into_inner())
            .take();
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn submit_direct(
    device: &ash::Device,
    queue: vk::Queue,
    batch: &DirectBatch,
) -> Result<(), vk::Result> {
    let mut buffers = [vk::CommandBuffer::null(); DIRECT_CAPACITY];
    if batch.entries[..batch.len]
        .iter()
        .flatten()
        .all(|entry| entry.wait.is_none() && entry.signal.is_none())
    {
        let mut signals = [vk::Semaphore::null(); DIRECT_CAPACITY];
        let mut values = [0; DIRECT_CAPACITY];
        let mut count = 0;
        for (i, entry) in batch.entries[..batch.len].iter().flatten().enumerate() {
            buffers[i] = entry.command;
            let at = signals[..count]
                .iter()
                .position(|&s| s == entry.timeline.0.semaphore());
            let at = at.unwrap_or_else(|| {
                let at = count;
                signals[at] = entry.timeline.0.semaphore();
                count += 1;
                at
            });
            values[at] = values[at].max(entry.timeline.1);
        }
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values[..count]);
        let info = vk::SubmitInfo::default()
            .command_buffers(&buffers[..batch.len])
            .signal_semaphores(&signals[..count])
            .push_next(&mut timeline);
        return unsafe { device.queue_submit(queue, &[info], vk::Fence::null()) };
    }
    let mut waits = [[vk::Semaphore::null(); 1]; DIRECT_CAPACITY];
    let mut signals = [[vk::Semaphore::null(); 2]; DIRECT_CAPACITY];
    let mut values = [[0; 2]; DIRECT_CAPACITY];
    for (i, entry) in batch.entries[..batch.len].iter().enumerate() {
        let entry = entry.as_ref().unwrap();
        buffers[i] = entry.command;
        waits[i][0] = entry.wait.unwrap_or(vk::Semaphore::null());
        signals[i] = [
            entry.timeline.0.semaphore(),
            entry.signal.unwrap_or(vk::Semaphore::null()),
        ];
        values[i] = [entry.timeline.1, 0];
    }
    let wait_values = [0];
    let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
    let mut timelines: [_; DIRECT_CAPACITY] = std::array::from_fn(|i| {
        let count = batch.entries[i]
            .as_ref()
            .map_or(0, |entry| 1 + usize::from(entry.signal.is_some()));
        vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values[i][..count])
    });
    let mut submits = [vk::SubmitInfo::default(); DIRECT_CAPACITY];
    for (i, (submit, timeline)) in submits[..batch.len]
        .iter_mut()
        .zip(&mut timelines)
        .enumerate()
    {
        let entry = batch.entries[i].as_ref().unwrap();
        let count = 1 + usize::from(entry.signal.is_some());
        let mut info = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&buffers[i]))
            .signal_semaphores(&signals[i][..count]);
        if entry.wait.is_some() {
            *timeline = timeline.wait_semaphore_values(&wait_values);
            info = info.wait_semaphores(&waits[i]).wait_dst_stage_mask(&stages);
        }
        *submit = info.push_next(timeline);
    }
    unsafe { device.queue_submit(queue, &submits[..batch.len], vk::Fence::null()) }
}

fn insert_ready(ready: &mut VecDeque<Option<Ready>>, next: u64, item: Ready) {
    let index = (item.ticket - next) as usize;
    if ready.len() <= index {
        ready.resize_with(index + 1, || None);
    }
    assert!(
        ready[index].replace(item).is_none(),
        "duplicate submission ticket"
    );
}

fn fail_queue(device: &ash::Device, failed: &mut bool) {
    if !*failed {
        unsafe {
            let _ = device.device_wait_idle();
        }
        *failed = true;
    }
}

fn submit_loop(
    device: ash::Device,
    queue: vk::Queue,
    receiver: mpsc::Receiver<Ready>,
    direct_failure: Completion,
) {
    let mut next = 0;
    let mut ready = VecDeque::new();
    let mut pending: VecDeque<(vk::Fence, Completion)> = VecDeque::new();
    let mut failed = false;
    let mut disconnected = false;
    loop {
        let input = if pending.is_empty() && !disconnected {
            receiver
                .recv()
                .map_err(|_| mpsc::RecvTimeoutError::Disconnected)
        } else {
            match receiver.try_recv() {
                Ok(item) => Ok(item),
                Err(mpsc::TryRecvError::Disconnected) => Err(mpsc::RecvTimeoutError::Disconnected),
                Err(mpsc::TryRecvError::Empty) => {
                    if let Some((fence, _)) = pending.front() {
                        match unsafe { device.wait_for_fences(&[*fence], true, 100_000) } {
                            Ok(()) | Err(vk::Result::TIMEOUT) => {}
                            Err(_) => fail_queue(&device, &mut failed),
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout)
                }
            }
        };
        match input {
            Ok(item) => {
                insert_ready(&mut ready, next, item);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => disconnected = true,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        while let Ok(item) = receiver.try_recv() {
            insert_ready(&mut ready, next, item);
        }
        while ready.front().is_some_and(Option::is_some) {
            let item = ready.pop_front().flatten().unwrap();
            next += 1;
            match item.payload {
                Payload::Recorded {
                    command,
                    fence,
                    wait,
                    signal,
                    timeline,
                } if !failed => {
                    let commands = [command];
                    let waits = [wait.unwrap_or(vk::Semaphore::null())];
                    let signals = [
                        signal.unwrap_or(vk::Semaphore::null()),
                        timeline
                            .as_ref()
                            .map_or(vk::Semaphore::null(), |t| t.0.semaphore()),
                    ];
                    let values = [0, timeline.as_ref().map_or(0, |t| t.1)];
                    let start = usize::from(signal.is_none());
                    let end = 1 + usize::from(timeline.is_some());
                    let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
                    let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                        .signal_semaphore_values(&values[start..end]);
                    let mut info = vk::SubmitInfo::default()
                        .command_buffers(&commands)
                        .signal_semaphores(&signals[start..end]);
                    if wait.is_some() {
                        timeline_info = timeline_info.wait_semaphore_values(&[0]);
                        info = info.wait_semaphores(&waits).wait_dst_stage_mask(&stages);
                    }
                    let info = info.push_next(&mut timeline_info);
                    let result = unsafe {
                        device
                            .reset_fences(&[fence])
                            .and_then(|()| device.queue_submit(queue, &[info], fence))
                    };
                    if result.is_ok() {
                        if let Some((timeline, value)) = &timeline {
                            timeline.post(*value);
                        }
                        pending.push_back((fence, item.completion));
                    } else {
                        fail_queue(&device, &mut failed);
                        item.completion.finish(false);
                    }
                }
                Payload::Direct(batch) => {
                    let result = if failed {
                        Err(vk::Result::ERROR_DEVICE_LOST)
                    } else {
                        submit_direct(&device, queue, &batch)
                    };
                    if result.is_err() {
                        fail_queue(&device, &mut failed);
                    }
                    if result.is_ok() {
                        for (i, entry) in batch.entries[..batch.len].iter().flatten().enumerate() {
                            if !batch.entries[i + 1..batch.len]
                                .iter()
                                .flatten()
                                .any(|later| Arc::ptr_eq(&later.timeline.0, &entry.timeline.0))
                            {
                                entry.timeline.0.post(entry.timeline.1);
                            }
                        }
                    }
                    if result.is_err() {
                        direct_failure.finish(false);
                    }
                }
                Payload::Checkpoint(semaphore, value, reply) => {
                    let semaphores = [semaphore];
                    let values = [value];
                    let mut timeline =
                        vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
                    let info = vk::SubmitInfo::default()
                        .signal_semaphores(&semaphores)
                        .push_next(&mut timeline);
                    let result = if failed {
                        Err(vk::Result::ERROR_DEVICE_LOST)
                    } else {
                        unsafe { device.queue_submit(queue, &[info], vk::Fence::null()) }
                    };
                    if result.is_err() {
                        fail_queue(&device, &mut failed);
                    }
                    let _ = reply.send(result);
                    item.completion.finish(!failed);
                }
                Payload::Recorded { .. } | Payload::Failed => {
                    fail_queue(&device, &mut failed);
                    item.completion.finish(false);
                }
                Payload::Present {
                    queue,
                    loader,
                    swapchain,
                    index,
                    waits,
                    fence,
                    reply,
                } => {
                    let chains = [swapchain];
                    let indices = [index];
                    let fences = [fence];
                    let mut retirement =
                        vk::SwapchainPresentFenceInfoEXT::default().fences(&fences);
                    let info = vk::PresentInfoKHR::default()
                        .wait_semaphores(&waits)
                        .swapchains(&chains)
                        .image_indices(&indices)
                        .push_next(&mut retirement);
                    let result = if failed {
                        Err(vk::Result::ERROR_DEVICE_LOST)
                    } else {
                        unsafe { loader.queue_present(queue, &info) }
                    };
                    if result == Err(vk::Result::ERROR_DEVICE_LOST) {
                        fail_queue(&device, &mut failed);
                    }
                    let _ = reply.send(result);
                    item.completion.finish(!failed);
                }
            }
        }
        if failed {
            direct_failure.finish(false);
            for (_, completion) in pending.drain(..) {
                completion.finish(false);
            }
        } else {
            while let Some((fence, _)) = pending.front() {
                match unsafe { device.get_fence_status(*fence) } {
                    Ok(true) => {
                        pending.pop_front().unwrap().1.finish(true);
                    }
                    Ok(false) => break,
                    Err(_) => {
                        fail_queue(&device, &mut failed);
                        break;
                    }
                }
            }
        }
        if disconnected && pending.is_empty() {
            for item in ready.into_iter().flatten() {
                item.completion.finish(false);
            }
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::Context;

    #[test]
    #[ignore = "requires a graphics device"]
    fn contexts_retire_before_cross_thread_teardown() {
        thread::scope(|scope| {
            for worker in 0..4 {
                scope.spawn(move || {
                    for round in 0..4 {
                        let context = Arc::new(Context::new().expect("graphics device required"));
                        let weak = Arc::downgrade(&context);
                        let mut backend = crate::gpu::Backend::from_context(context.clone(), 1.0)
                            .expect("backend resources");
                        backend.images.set_parallel_recording(round % 2 == 0);
                        assert!(backend.ensure(1, 4, 4, false));
                        let mut commands =
                            super::super::commands::Commands::with_capacity(&context, 2).unwrap();
                        for index in 0..24 {
                            if (worker + index) % 2 == 0 {
                                commands.begin_direct().unwrap();
                            } else {
                                commands.begin().unwrap();
                            }
                            commands.submit(false, None).unwrap();
                            let color = if index % 2 == 0 {
                                [1.0, 0.0, 0.0, 1.0]
                            } else {
                                [0.0, 0.0, 1.0, 1.0]
                            };
                            assert!(backend.clear_color(1, color, 15));
                            backend.enqueue_callback(index % 2, 1, 4, 4).unwrap();
                        }
                        let retained = commands.submission().unwrap();
                        drop(context);
                        thread::spawn(move || {
                            drop(backend);
                            drop(commands);
                            assert!(weak.upgrade().is_some());
                            retained.wait().unwrap();
                            drop(retained);
                            assert!(weak.upgrade().is_none());
                        })
                        .join()
                        .unwrap();
                    }
                });
            }
        });
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn retained_batch_completes_repeated_and_distinct_timeline_thresholds() {
        let context = Arc::new(Context::new().expect("Vulkan device required"));
        let mut first = super::super::commands::Commands::new(&context).unwrap();
        let mut second = super::super::commands::Commands::new(&context).unwrap();
        for _ in 0..2 {
            first.begin_direct().unwrap();
            first.submit(false, None).unwrap();
            second.begin_direct().unwrap();
            second.submit(false, None).unwrap();
        }
        first.wait().unwrap();
        second.wait().unwrap();
        for value in 1..=2 {
            assert_eq!(first.ready(value), Some(true));
            assert_eq!(second.ready(value), Some(true));
        }
        context.wait_queue().unwrap();
        assert_eq!(context.completion.lock().unwrap().value, 0);
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn incomplete_poll_flushes_retained_work_and_checkpoints_reuse_only_idle_prefixes() {
        let context = Arc::new(Context::new().expect("Vulkan device required"));
        context.wait_queue().unwrap();
        assert_eq!(context.completion.lock().unwrap().value, 0);
        let mut commands = super::super::commands::Commands::new(&context).unwrap();
        commands.begin_direct().unwrap();
        commands.submit(false, None).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !commands.ready(commands.completion()).unwrap() {
            assert!(std::time::Instant::now() < deadline);
            thread::yield_now();
        }
        context.wait_queue().unwrap();
        let position = context.completion.lock().unwrap().value;
        assert_eq!(position, 0);
        context.wait_queue().unwrap();
        assert_eq!(context.completion.lock().unwrap().value, position);
        commands.begin_direct().unwrap();
        commands.submit(false, None).unwrap();
        context.wait_queue().unwrap();
        assert_eq!(context.completion.lock().unwrap().value, position);
        assert_eq!(commands.ready(commands.completion()), Some(true));
        let untracked = context.command_workers.enqueue(Vec::new(), None, None);
        context.wait_queue().unwrap();
        untracked.wait().unwrap();
        assert_eq!(context.completion.lock().unwrap().value, position + 1);
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn direct_draw_commands_share_order_with_recording_workers() {
        let context = Arc::new(Context::new().expect("Vulkan device required"));
        let (entered, entry) = mpsc::sync_channel(1);
        let (release, released) = mpsc::sync_channel(1);
        let released = Mutex::new(released);
        let first: Operation = Arc::new(move |_, _| {
            entered.send(()).unwrap();
            released.lock().unwrap().recv().unwrap();
        });
        let first = context.command_workers.enqueue(vec![first], None, None);
        entry.recv_timeout(Duration::from_secs(5)).unwrap();
        let mut direct = super::super::commands::Commands::new(&context).unwrap();
        direct.begin_direct().unwrap();
        direct.submit(false, None).unwrap();
        let (recorded, recording) = mpsc::sync_channel(1);
        let last: Operation = Arc::new(move |_, _| {
            recorded.send(()).unwrap();
        });
        let last = context.command_workers.enqueue(vec![last], None, None);
        recording.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(direct.ready(direct.completion()), Some(false));
        assert_eq!(last.ready(), Some(false));
        release.send(()).unwrap();
        first.wait().unwrap();
        direct.wait().unwrap();
        last.wait().unwrap();
        assert_eq!(direct.ready(direct.completion()), Some(true));
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn workers_record_in_parallel_but_retire_in_ticket_order() {
        let context = Context::new().expect("Vulkan device required");
        let (entered, entry) = mpsc::sync_channel(1);
        let (release, released) = mpsc::sync_channel(1);
        let released = Mutex::new(released);
        let first: Operation = Arc::new(move |_, _| {
            entered.send(()).unwrap();
            released.lock().unwrap().recv().unwrap();
        });
        let first = context.command_workers.enqueue(vec![first], None, None);
        entry.recv_timeout(Duration::from_secs(5)).unwrap();
        let (recorded, recording) = mpsc::sync_channel(1);
        let second: Operation = Arc::new(move |_, _| {
            recorded.send(()).unwrap();
        });
        let second = context.command_workers.enqueue(vec![second], None, None);
        recording.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(second.0 .0.lock().unwrap().is_none());
        release.send(()).unwrap();
        first.wait().unwrap();
        second.wait().unwrap();
        // Reuse every worker's same pool after retirement.
        for _ in 0..32 {
            context
                .command_workers
                .enqueue(Vec::new(), None, None)
                .wait()
                .unwrap();
        }
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn failed_recording_unblocks_the_ordered_tail() {
        let context = Arc::new(Context::new().expect("Vulkan device required"));
        let fail: Operation = Arc::new(|_, _| panic!("recording fixture"));
        let first = context.command_workers.enqueue(vec![fail], None, None);
        let second = context.command_workers.enqueue(Vec::new(), None, None);
        assert!(first.wait().is_none());
        assert!(second.wait().is_none());
        let mut direct = super::super::commands::Commands::new(&context).unwrap();
        direct.begin_direct().unwrap();
        direct.submit(false, None).unwrap();
        assert!(direct.wait().is_none());
        assert!(direct.ready(direct.completion()).is_none());
        let mut readback = super::super::commands::Commands::with_capacity(&context, 1).unwrap();
        readback.begin_direct().unwrap();
        readback.submit(false, None).unwrap();
        assert!(readback.wait().is_none());
        assert!(context.wait_queue().is_none());
    }
}
