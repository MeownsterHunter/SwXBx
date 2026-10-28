//! Optional counters for the original draw benchmark. Source: the public contract.
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};
static NANOS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static CALLS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Pipeline lookup, descriptor preparation, and uniform resolution totals.
pub fn take() -> [(u64, u64); 3] {
    std::array::from_fn(|i| {
        (
            NANOS[i].swap(0, Ordering::Relaxed),
            CALLS[i].swap(0, Ordering::Relaxed),
        )
    })
}
pub(crate) struct Span(usize, Instant);
impl Span {
    pub(crate) fn new(index: usize) -> Self {
        Self(index, Instant::now())
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        NANOS[self.0].fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS[self.0].fetch_add(1, Ordering::Relaxed);
    }
}

/// Actual descriptor updates, pool creations, state binds, retained draw hits, and push writes.
pub fn take_counts() -> [u64; 5] {
    std::array::from_fn(|i| COUNTS[i].swap(0, Ordering::Relaxed))
}
static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
pub(crate) fn count(index: usize) {
    COUNTS[index].fetch_add(1, Ordering::Relaxed);
}

/// Compilation stages: request, layout, modules, four subsets, link,
/// descriptor lock, driver cache lock, completion polling, draw completion, attachment transfers, and host upload
/// and download, download completion, and mapped reads. Source: the public contract.
pub fn take_pipeline() -> [(u64, u64, u64); 17] {
    std::array::from_fn(|i| {
        (
            PIPELINE_NANOS[i].swap(0, Ordering::Relaxed),
            PIPELINE_CALLS[i].swap(0, Ordering::Relaxed),
            PIPELINE_MAX[i].swap(0, Ordering::Relaxed),
        )
    })
}
static PIPELINE_NANOS: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];
static PIPELINE_CALLS: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];
static PIPELINE_MAX: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];
pub(crate) struct PipelineSpan(usize, Instant);
impl PipelineSpan {
    pub(crate) fn new(index: usize) -> Self {
        Self(index, Instant::now())
    }
}
impl Drop for PipelineSpan {
    fn drop(&mut self) {
        let nanos = self.1.elapsed().as_nanos() as u64;
        PIPELINE_NANOS[self.0].fetch_add(nanos, Ordering::Relaxed);
        PIPELINE_CALLS[self.0].fetch_add(1, Ordering::Relaxed);
        PIPELINE_MAX[self.0].fetch_max(nanos, Ordering::Relaxed);
    }
}
