//! Host-selected presentation policy and callback timing.

use crate::{Host, Instance, Status};
use std::{collections::VecDeque, time::Instant};

/// Ordering policy for pending frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum PresentationMode {
    /// Deliver every frame in submission order.
    #[default]
    Fifo = 0,
    /// Replace pending frames with the newest submitted frame.
    Mailbox = 1,
}

/// Host policy, independent of guest presentation interval values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentationConfig {
    /// Maximum queued frames, from one through sixteen. The default is two.
    pub frames_in_flight: u32,
    /// FIFO or mailbox ordering. The default is FIFO.
    pub mode: PresentationMode,
}

impl Default for PresentationConfig {
    fn default() -> Self {
        Self {
            frames_in_flight: 2,
            mode: PresentationMode::Fifo,
        }
    }
}

/// C-compatible snapshot. Latency ends immediately before callback entry.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStatistics {
    /// Accepted presentation requests.
    pub submitted: u64,
    /// Frames handed to the callback or native presentation engine.
    pub delivered: u64,
    /// Pending frames replaced by mailbox policy.
    pub dropped: u64,
    /// Native requests skipped while the drawable has zero extent.
    pub skipped: u64,
    /// Requests that failed during preparation or delivery.
    pub failed: u64,
    /// Frames currently awaiting host delivery.
    pub pending: u32,
    /// Largest pending frame count.
    pub peak_pending: u32,
    /// Most recent delivered frame latency in nanoseconds.
    pub last_latency_ns: u64,
    /// Arithmetic mean of delivered latency samples in nanoseconds.
    pub mean_latency_ns: f64,
    /// Population variance of delivered latency samples in square nanoseconds.
    pub variance_latency_ns2: f64,
}

pub(crate) struct Frame {
    pub window: u64,
    pub started: Instant,
    pub slot: usize,
    pub pixels: Option<(u32, u32, Vec<u8>)>,
}

#[derive(Default)]
pub(crate) struct Presentation {
    pub config: PresentationConfig,
    pub pending: VecDeque<Frame>,
    stats: FrameStatistics,
    squared_deviations: f64,
    next: usize,
}

impl Presentation {
    pub fn statistics(&self) -> FrameStatistics {
        FrameStatistics {
            pending: self.pending.len() as u32,
            ..self.stats
        }
    }

    pub fn begin(&mut self) -> usize {
        self.stats.submitted += 1;
        let slot = (0..self.config.frames_in_flight as usize)
            .map(|offset| (self.next + offset) % self.config.frames_in_flight as usize)
            .find(|slot| self.pending.iter().all(|frame| frame.slot != *slot))
            .expect("queue capacity was reserved");
        self.next = (self.next + 1) % self.config.frames_in_flight as usize;
        slot
    }

    #[cfg(feature = "vulkan")]
    pub fn skipped(&mut self) {
        self.stats.skipped += 1;
    }

    pub fn failed(&mut self) {
        self.stats.failed += 1;
    }

    pub fn drop_pending(&mut self, window: u64) {
        let before = self.pending.len();
        self.pending.retain(|frame| frame.window != window);
        self.stats.dropped += (before - self.pending.len()) as u64;
    }

    pub fn push(&mut self, frame: Frame) {
        self.pending.push_back(frame);
        self.stats.peak_pending = self.stats.peak_pending.max(self.pending.len() as u32);
    }

    pub fn delivered(&mut self, started: Instant) {
        self.stats.delivered += 1;
        self.stats.last_latency_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        let sample = self.stats.last_latency_ns as f64;
        let delta = sample - self.stats.mean_latency_ns;
        self.stats.mean_latency_ns += delta / self.stats.delivered as f64;
        self.squared_deviations += delta * (sample - self.stats.mean_latency_ns);
        self.stats.variance_latency_ns2 = self.squared_deviations / self.stats.delivered as f64;
    }

    pub fn deliver_one(
        &mut self,
        instance: &Instance,
        host: &Host,
        wait: bool,
    ) -> Result<bool, Status> {
        let Some(frame) = self.pending.front() else {
            return Ok(false);
        };
        let pixels = if let Some(pixels) = &frame.pixels {
            Some(pixels.clone())
        } else {
            #[cfg(feature = "vulkan")]
            {
                let gpu = instance.gpu.lock().unwrap_or_else(|p| p.into_inner());
                let read = gpu.as_ref().and_then(|b| b.read_callback(frame.slot, wait));
                match read {
                    Some(pixels) => pixels,
                    None => {
                        self.pending.pop_front();
                        return Err(Status::InternalError);
                    }
                }
            }
            #[cfg(not(feature = "vulkan"))]
            {
                let _ = (instance, wait);
                return Err(Status::InternalError);
            }
        };
        let Some((width, height, pixels)) = pixels else {
            return Ok(false);
        };
        let frame = self.pending.pop_front().unwrap();
        if let Some(vblank) = host.wait_vblank {
            unsafe {
                vblank(host.user);
            }
        }
        self.delivered(frame.started);
        if let Some(present) = host.present {
            unsafe {
                present(
                    host.user,
                    frame.window,
                    width,
                    height,
                    pixels.as_ptr(),
                    u64::from(width) * 4,
                );
            }
        }
        Ok(true)
    }
}

impl Instance {
    /// Drain pending callbacks and replace the host policy. Invalid capacities fail.
    /// Callbacks must not reenter presentation methods or guest submission.
    pub fn configure_presentation(&self, config: PresentationConfig) -> Status {
        if !(1..=16).contains(&config.frames_in_flight) {
            return Status::BadArgument;
        }
        let mut state = self.presentation.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(host) = self.host() {
            while !state.pending.is_empty() {
                if let Err(status) = state.deliver_one(self, host, true) {
                    state.failed();
                    return status;
                }
            }
        }
        #[cfg(feature = "vulkan")]
        if let Some(gpu) = self.gpu.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
            gpu.reset_presentations();
        }
        state.config = config;
        state.next = 0;
        Status::Ok
    }

    /// Deliver one ready frame without waiting for GPU completion, or drain all
    /// pending frames when wait is true. The host supplies any vblank wait.
    pub fn poll_presentations(&self, wait: bool) -> Status {
        let mut state = self.presentation.lock().unwrap_or_else(|p| p.into_inner());
        let Some(host) = self.host() else {
            return Status::Ok;
        };
        loop {
            match state.deliver_one(self, host, wait) {
                Ok(true) if wait => continue,
                Ok(_) => return Status::Ok,
                Err(status) => {
                    state.failed();
                    return status;
                }
            }
        }
    }

    /// Return cumulative frame counts and delivered latency moments.
    pub fn frame_statistics(&self) -> FrameStatistics {
        self.presentation
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .statistics()
    }
}
