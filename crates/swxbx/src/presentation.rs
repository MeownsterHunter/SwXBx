//! Host-selected presentation policy and callback timing. Source: the public contract.

use crate::{Host, Instance, Status};
use std::{collections::VecDeque, num::NonZeroU32, time::Instant};

/// How long the guest asks each presented image to stay on screen, as stated
/// through the window's present interval. Source: the public contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentInterval {
    /// Interval 0: present without waiting for a vertical blank.
    Immediate,
    /// Interval N: hold each image for at least N vertical blanks.
    Blanks(NonZeroU32),
}

impl PresentInterval {
    /// The interval every window starts with: one vertical blank per image.
    pub const ONE: Self = Self::Blanks(NonZeroU32::MIN);

    /// Read the guest's signed interval; a negative count is no interval.
    pub fn from_guest(value: i32) -> Result<Self, String> {
        match u32::try_from(value) {
            Ok(blanks) => Ok(NonZeroU32::new(blanks).map_or(Self::Immediate, Self::Blanks)),
            Err(_) => Err(format!(
                "present interval {value}: an interval counts vertical blanks and cannot be negative"
            )),
        }
    }

    /// The value the guest stated.
    pub fn guest_value(self) -> u32 {
        match self {
            Self::Immediate => 0,
            Self::Blanks(blanks) => blanks.get(),
        }
    }

    /// Native FIFO presents of one image: each FIFO present holds the image
    /// for one vertical blank, so N presents hold it for N.
    #[cfg(feature = "vulkan")]
    pub(crate) fn fifo_presents(self) -> u32 {
        match self {
            Self::Immediate => 1,
            Self::Blanks(blanks) => blanks.get(),
        }
    }

    /// Host vertical-blank waits before an RGBA callback delivery.
    pub(crate) fn vblank_waits(self) -> u32 {
        match self {
            Self::Immediate => 0,
            Self::Blanks(blanks) => blanks.get(),
        }
    }

    /// Whether this host, under this presentation mode, can show images the
    /// way the interval asks; the reason names what is missing when it cannot.
    /// Native presentation: FIFO waits for every vertical blank, so it holds
    /// an image for N blanks by presenting it N times, and cannot skip the
    /// wait; mailbox never waits, so it takes interval 0 and cannot hold an
    /// image for more than one blank. Interval 1 is each mode's own pacing.
    /// RGBA callback hosts wait through `wait_vblank`. No host presents
    /// nothing, so every interval holds.
    pub(crate) fn shown_by(
        self,
        host: Option<&Host>,
        mode: PresentationMode,
    ) -> Result<(), String> {
        let Some(host) = host else {
            return Ok(());
        };
        let native = !host.vulkan.is_null();
        match self {
            Self::Immediate if native && mode == PresentationMode::Fifo => Err(
                "present interval 0 asks not to wait for a vertical blank, and the host's \
                 presentation mode Fifo waits for every one"
                    .into(),
            ),
            Self::Blanks(blanks) if blanks.get() > 1 && native && mode == PresentationMode::Mailbox => {
                Err(format!(
                    "present interval {blanks} asks to hold each image for {blanks} vertical blanks, \
                     and the host's presentation mode Mailbox replaces images instead of holding them"
                ))
            }
            Self::Blanks(blanks) if blanks.get() > 1 && !native && host.wait_vblank.is_none() => {
                Err(format!(
                    "present interval {blanks} asks to hold each image for {blanks} vertical blanks, \
                     and the host supplies no vertical-blank wait"
                ))
            }
            _ => Ok(()),
        }
    }
}

/// How a presented image fills the window's drawable. The integrator states it
/// at every native present through `HostVulkan::scaling`; there is no default.
/// Source: the public contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scaling {
    /// Fill the whole drawable, whatever its aspect.
    Stretch,
    /// Keep this aspect, centred, with black bars on the remaining sides.
    Fit(Aspect),
}

/// A positive, finite width-to-height ratio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aspect {
    width: f32,
    height: f32,
}

impl Aspect {
    /// The ratio `width : height`; both sides must be positive and finite.
    pub fn new(width: f32, height: f32) -> Result<Self, String> {
        if width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0 {
            Ok(Self { width, height })
        } else {
            Err(format!(
                "fit aspect {width}:{height} is not a ratio of two positive finite numbers"
            ))
        }
    }

    /// Width side of the ratio.
    pub fn width(self) -> f32 {
        self.width
    }

    /// Height side of the ratio.
    pub fn height(self) -> f32 {
        self.height
    }
}

/// C form of [`Scaling`], written by the host's `scaling` callback.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HostScaling {
    /// `SCALING_STRETCH` or `SCALING_FIT`.
    pub kind: u32,
    /// Fit only: the width side of the aspect.
    pub aspect_width: f32,
    /// Fit only: the height side of the aspect.
    pub aspect_height: f32,
}

/// [`HostScaling::kind`]: fill the drawable.
pub const SCALING_STRETCH: u32 = 0;
/// [`HostScaling::kind`]: keep the aspect, with black bars.
pub const SCALING_FIT: u32 = 1;

impl HostScaling {
    /// What SwXBx hands the callback: a kind no host states, so a callback
    /// that writes nothing is refused rather than read as a default.
    pub const UNSTATED: Self = Self {
        kind: u32::MAX,
        aspect_width: 0.0,
        aspect_height: 0.0,
    };

    /// Read the host's statement.
    pub fn parse(self) -> Result<Scaling, String> {
        match self.kind {
            SCALING_STRETCH => Ok(Scaling::Stretch),
            SCALING_FIT => Aspect::new(self.aspect_width, self.aspect_height).map(Scaling::Fit),
            u32::MAX => Err("the host stated no scaling for the window".into()),
            kind => Err(format!(
                "scaling kind {kind} is neither stretch (0) nor fit (1)"
            )),
        }
    }
}

/// Ordering policy for pending frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum PresentationMode {
    /// Deliver every frame in submission order.
    #[default]
    Fifo = 0,
    /// Replace pending frames with the newest submitted frame. A surface that
    /// does not offer it refuses it by name.
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
    /// Largest observed pending frame count.
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
    /// The window's interval when the frame was presented.
    pub interval: PresentInterval,
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
            for _ in 0..frame.interval.vblank_waits() {
                unsafe {
                    vblank(host.user);
                }
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
    /// The mode is host policy; whether a surface offers it is a fact of each
    /// window. With a Vulkan-surface host, mailbox needs an open window, and
    /// every open window's surface must offer it; otherwise the mode is refused
    /// by name and the policy is kept. A window opened later in mailbox is
    /// asked when it opens. A callback host presents itself and is not asked.
    /// FIFO is offered by every surface. Callbacks must not reenter
    /// presentation methods or guest submission. Source: the public contract.
    pub fn configure_presentation(&self, config: PresentationConfig) -> Status {
        if !(1..=16).contains(&config.frames_in_flight) {
            return Status::BadArgument;
        }
        // Held across the surface check so a window cannot open between the
        // check and the new policy.
        let mut state = self.presentation.lock().unwrap_or_else(|p| p.into_inner());
        #[cfg(feature = "vulkan")]
        if let Err(status) = self.surfaces_take(config.mode) {
            return status;
        }
        // A window's present interval was accepted under the old mode; the new
        // mode must show it too.
        if let Some((window, why)) =
            self.objects
                .window_intervals()
                .into_iter()
                .find_map(|(window, interval)| {
                    interval
                        .shown_by(self.host(), config.mode)
                        .err()
                        .map(|why| (window, why))
                })
        {
            return crate::unsupported::refuse(
                Status::BadArgument,
                format!(
                    "presentation mode {:?}: window {window:#x}: {why}",
                    config.mode
                ),
            );
        }
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

    /// Whether the open windows' surfaces can take `mode` under this host.
    #[cfg(feature = "vulkan")]
    fn surfaces_take(&self, mode: PresentationMode) -> Result<(), Status> {
        // Every surface offers FIFO; a callback host presents itself.
        if mode == PresentationMode::Fifo {
            return Ok(());
        }
        if self.host().is_none_or(|host| host.vulkan.is_null()) {
            return Ok(());
        }
        let gpu = self.gpu.lock().unwrap_or_else(|p| p.into_inner());
        let Some(backend) = gpu.as_ref().filter(|backend| backend.has_windows()) else {
            return Err(crate::unsupported::refuse(
                Status::BadArgument,
                format!("presentation mode {mode:?}: no open window whose surface can be asked"),
            ));
        };
        match backend.window_lacking(mode) {
            Some((window, why)) => Err(refuse_unoffered(mode, window, why)),
            None => Ok(()),
        }
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

/// Refuse a mode a window's surface cannot take, naming why.
#[cfg(feature = "vulkan")]
pub(crate) fn refuse_unoffered(
    mode: PresentationMode,
    window: u64,
    why: crate::gpu::Unoffered,
) -> Status {
    match why {
        crate::gpu::Unoffered::Lacks => crate::unsupported::refuse(
            Status::BadArgument,
            format!("presentation mode {mode:?}: the surface of window {window:#x} does not offer it"),
        ),
        crate::gpu::Unoffered::CannotBeAsked => crate::unsupported::refuse(
            Status::InternalError,
            format!(
                "presentation mode {mode:?}: the surface of window {window:#x} cannot be asked which modes it offers"
            ),
        ),
    }
}
