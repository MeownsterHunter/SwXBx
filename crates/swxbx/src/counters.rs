//! Counter reports: which counter a guest token selects, and the report the
//! graphics processor writes for it. Source: the public contract.
//!
//! A report is sixteen bytes at the given GPU address: the counter value as a
//! little-endian 64-bit word, then the GPU timer as a little-endian 64-bit
//! word. The layout and the timer frequency follow the Maxwell semaphore
//! counter report in the reference host (MIT), `the reference host.Graphics.Gpu/Engine/Threed/
//! SemaphoreUpdater.cs` (`CounterData`, `ReportCounter`) and
//! `the reference host.Graphics.Gpu/GpuContext.cs` (`ConvertNanosecondsToTicks`).
//! Copyright (c) the reference host Team and Contributors, MIT licensed.

use std::collections::BTreeMap;
#[cfg(feature = "vulkan")]
use std::time::Instant;

/// Bytes in one counter report.
#[cfg(feature = "vulkan")]
pub(crate) const REPORT_SIZE: u64 = 16;

/// A counter swxbx can report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Counter {
    /// No counter value. The first word is zero and the report carries the
    /// GPU timer, so it marks when the queue reached it.
    Timestamp,
    /// Samples that passed the depth and stencil tests since the last
    /// reset of this counter. Zero before the first reset.
    SamplesPassed,
}

/// Which guest counter token selects which counter. The tokens are not
/// recovered interface facts; each is a host hypothesis or an observation
/// recorded in the public contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterContract {
    tokens: BTreeMap<u64, Counter>,
}

impl CounterContract {
    /// A contract from token and counter pairs. A token named twice keeps
    /// its last counter.
    pub fn new(tokens: impl IntoIterator<Item = (u64, Counter)>) -> Self {
        Self {
            tokens: tokens.into_iter().collect(),
        }
    }

    /// The counter a token selects, if the contract names it.
    pub fn counter(&self, token: u64) -> Option<Counter> {
        self.tokens.get(&token).copied()
    }
}

impl Default for CounterContract {
    /// The contract every instance starts with. The only token observed in
    /// a program's reports is 0, and that program advanced once its reports
    /// carried a rising timer (the public contract), so 0 selects the timestamp.
    fn default() -> Self {
        Self::new([(0, Counter::Timestamp)])
    }
}

/// The GPU timer: 614.4 MHz, counted from when the clock was made.
#[cfg(feature = "vulkan")]
pub(crate) struct GpuClock {
    epoch: Instant,
}

#[cfg(feature = "vulkan")]
impl GpuClock {
    pub(crate) fn new() -> Self {
        Self {
            epoch: Instant::now(),
        }
    }

    /// Timer ticks now.
    pub(crate) fn ticks(&self) -> u64 {
        let nanoseconds = u64::try_from(self.epoch.elapsed().as_nanos()).unwrap_or(u64::MAX);
        ticks(nanoseconds)
    }
}

/// Timer ticks in a span of nanoseconds, at 384 ticks per 625 ns. Dividing
/// first keeps the product in range; the remainder is converted apart.
/// Ported from the reference host `GpuContext.ConvertNanosecondsToTicks` (MIT).
#[cfg(feature = "vulkan")]
fn ticks(nanoseconds: u64) -> u64 {
    let divided = nanoseconds / 625;
    let remainder = nanoseconds - divided * 625;
    divided * 384 + remainder * 384 / 625
}

/// The bytes of one report.
#[cfg(feature = "vulkan")]
pub(crate) fn report(value: u64, ticks: u64) -> [u8; REPORT_SIZE as usize] {
    let mut bytes = [0; REPORT_SIZE as usize];
    bytes[..8].copy_from_slice(&value.to_le_bytes());
    bytes[8..].copy_from_slice(&ticks.to_le_bytes());
    bytes
}
