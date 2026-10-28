//! Vulkan backing for the flat address space. Source: the public contract.-flat-global-memory, 0038 and 0039.

use super::commands::Submission;
use super::Context;
use crate::dirty_ranges::Ranges;
use crate::global_memory::{
    AddressMap, Allocator, GUEST_BASE, PUSH_GLOBAL_DELTA_OFFSET, PUSH_GLOBAL_DELTA_SIZE,
};
use ash::{vk, Device};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    ops::Range,
    sync::Arc,
};

const PAGE_SIZE: u64 = 4096;

#[derive(Default)]
struct PoolSync {
    tracked: bool,
    direct: bool,
    size: u64,
}

struct Access {
    range: Range<u64>,
    write: bool,
    submission: Submission,
}

// Physical pointers have no descriptor bound. Keep their writeback conservative.
pub(super) fn physical_writes(words: &[u32]) -> bool {
    writes_to_storage(words, &[5349])
}

pub(super) fn storage_writes(words: &[u32]) -> bool {
    writes_to_storage(words, &[2, 12])
}

fn writes_to_storage(words: &[u32], classes: &[u32]) -> bool {
    use std::collections::BTreeSet;
    let mut instructions = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let length = (words[at] >> 16) as usize;
        if length == 0 || at + length > words.len() {
            return true;
        }
        instructions.push(&words[at..at + length]);
        at += length;
    }
    let types: BTreeSet<_> = instructions
        .iter()
        .filter(|w| w[0] & 0xffff == 32 && w.len() >= 4 && classes.contains(&w[2]))
        .map(|w| w[1])
        .collect();
    let pointers: BTreeSet<_> = instructions
        .iter()
        .filter(|w| w.len() >= 3 && types.contains(&w[1]))
        .map(|w| w[2])
        .collect();
    instructions.iter().any(|w| {
        let op = w[0] & 0xffff;
        match op {
            62..=64 | 228 | 319 => w.get(1).is_some_and(|id| pointers.contains(id)),
            229..=242 | 318 => w.get(3).is_some_and(|id| pointers.contains(id)),
            _ => false,
        }
    })
}

pub(super) fn descriptor_writable(words: &[u32], set: u32, binding: u32) -> bool {
    if !storage_writes(words) {
        return false;
    }
    let mut ops = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let size = (words[at] >> 16) as usize;
        if size == 0 || at + size > words.len() {
            return true;
        }
        ops.push(&words[at..at + size]);
        at += size;
    }
    let matches = |id, decoration, value| {
        ops.iter().any(|w| {
            w.len() >= 4 && w[0] & 0xffff == 71 && w[1] == id && w[2] == decoration && w[3] == value
        })
    };
    let Some(variable) = ops.iter().find(|w| {
        w.len() >= 4 && w[0] & 0xffff == 59 && matches(w[2], 34, set) && matches(w[2], 33, binding)
    }) else {
        return true;
    };
    if ops
        .iter()
        .any(|w| w.len() >= 3 && w[0] & 0xffff == 71 && w[1] == variable[2] && w[2] == 24)
    {
        return false;
    }
    let Some(pointer) = ops
        .iter()
        .find(|w| w.len() >= 4 && w[0] & 0xffff == 32 && w[1] == variable[1])
    else {
        return true;
    };
    let Some(structure) = ops
        .iter()
        .find(|w| w.len() >= 2 && w[0] & 0xffff == 30 && w[1] == pointer[3])
    else {
        return true;
    };
    !(0..structure.len() - 2).all(|member| {
        ops.iter().any(|w| {
            w.len() >= 4
                && w[0] & 0xffff == 72
                && w[1] == structure[1]
                && w[2] == member as u32
                && w[3] == 24
        })
    })
}

/// Payload bytes moved between guest storage and the mapped arena.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransferStatistics {
    /// Bytes copied into the arena, excluding direct host writes.
    pub uploaded: u64,
    /// Bytes copied out of the arena, excluding direct host reads.
    pub downloaded: u64,
    /// Bytes inspected through callbacks for hosts without write notifications.
    pub inspected: u64,
}

/// Why host access to pool storage was not granted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolFault {
    /// The pool is not registered, or the range does not lie inside it.
    UnknownRange,
    /// Waiting for device access to the range failed.
    WaitFailed,
}

/// One buffer and one stable device address for every live guest pool.
/// Methods require exclusive access, including across queue submission and readback.
/// Raw external submissions must finish before host access or release. Backend
/// submissions register their ranges and completion tokens here.
pub struct GlobalMemory {
    context: Arc<Context>,
    device: Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: usize,
    addresses: AddressMap,
    allocator: Allocator,
    revision: Cell<Option<u64>>,
    writes: WriteStamps,
    host_dirty: Ranges,
    device_dirty: RefCell<Ranges>,
    pending: RefCell<Vec<Access>>,
    pools: BTreeMap<u64, PoolSync>,
    submission_tracking: bool,
    statistics: TransferStatistics,
}

/// The write count at which each arena page last changed. A write with no
/// known range raises the floor, which every page then reports. Each group of
/// `GROUP_PAGES` pages also holds its newest page stamp, so a reader whose
/// range saw no write since its count learns that from the groups alone.
struct WriteStamps {
    count: Cell<u64>,
    floor: Cell<u64>,
    pages: RefCell<Vec<u64>>,
    groups: RefCell<Vec<u64>>,
}

/// Pages summarized by one group stamp.
const GROUP_PAGES: usize = 64;

impl WriteStamps {
    fn new(size: u64) -> Self {
        let pages = size.div_ceil(PAGE_SIZE) as usize;
        Self {
            count: Cell::new(0),
            floor: Cell::new(0),
            pages: RefCell::new(vec![0; pages]),
            groups: RefCell::new(vec![0; pages.div_ceil(GROUP_PAGES)]),
        }
    }

    fn next(&self) -> u64 {
        self.count.set(self.count.get() + 1);
        self.count.get()
    }

    fn everywhere(&self) {
        self.floor.set(self.next());
    }

    fn range(&self, range: Range<u64>) {
        let count = self.next();
        let mut pages = self.pages.borrow_mut();
        let span = Self::span(pages.len(), range);
        pages[span.clone()].fill(count);
        if !span.is_empty() {
            self.groups.borrow_mut()[span.start / GROUP_PAGES..=(span.end - 1) / GROUP_PAGES]
                .fill(count);
        }
    }

    fn unwritten_since(&self, range: Range<u64>, count: u64) -> bool {
        if self.floor.get() > count {
            return false;
        }
        let pages = self.pages.borrow();
        let groups = self.groups.borrow();
        let span = Self::span(pages.len(), range);
        let mut start = span.start;
        while start < span.end {
            let group = start / GROUP_PAGES;
            let end = ((group + 1) * GROUP_PAGES).min(span.end);
            if groups[group] > count && pages[start..end].iter().any(|&page| page > count) {
                return false;
            }
            start = end;
        }
        true
    }

    /// The pages a byte range touches, clipped to the arena.
    fn span(pages: usize, range: Range<u64>) -> Range<usize> {
        let end = (range.end.div_ceil(PAGE_SIZE) as usize).min(pages);
        let start = ((range.start / PAGE_SIZE) as usize).min(end);
        start..end
    }
}

impl GlobalMemory {
    /// Allocate and map a device-address buffer; return None if no compatible allocation succeeds.
    pub fn new(context: &Arc<Context>, size: u64) -> Option<Self> {
        if size == 0
            || !size.is_multiple_of(16)
            || GUEST_BASE.checked_add(size)? > u64::from(u32::MAX) + 1
        {
            return None;
        }
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: the device has bufferDeviceAddress enabled and size is nonzero.
        let buffer = unsafe { context.device.create_buffer(&info, None).ok()? };
        let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
        let visible =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let properties = unsafe {
            context
                .instance
                .get_physical_device_memory_properties(context.physical_device)
        };
        let mut candidates: Vec<_> = (0..properties.memory_type_count)
            .filter(|&ty| {
                requirements.memory_type_bits & (1 << ty) != 0
                    && properties.memory_types[ty as usize]
                        .property_flags
                        .contains(visible)
            })
            .collect();
        // Submission reads the mapped arena as well as writing it. Prefer cached
        // coherent memory, then device-local memory among equally cached choices.
        candidates.sort_by_key(|&ty| {
            let flags = properties.memory_types[ty as usize].property_flags;
            (
                !flags.contains(vk::MemoryPropertyFlags::HOST_CACHED),
                !flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
            )
        });
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let memory = candidates.into_iter().find_map(|ty| {
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(ty)
                .push_next(&mut flags);
            unsafe { context.device.allocate_memory(&alloc, None).ok() }
        });
        let Some(memory) = memory else {
            unsafe {
                context.device.destroy_buffer(buffer, None);
            }
            return None;
        };
        let addresses = (|| {
            unsafe { context.device.bind_buffer_memory(buffer, memory, 0).ok()? };
            let host = unsafe {
                context.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(buffer),
                )
            };
            AddressMap::new(GUEST_BASE, host, size)
        })();
        let Some(addresses) = addresses else {
            // SAFETY: no work was submitted using the buffer.
            unsafe {
                context.device.destroy_buffer(buffer, None);
                context.device.free_memory(memory, None);
            }
            return None;
        };
        let mapped = unsafe {
            context
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        };
        let Ok(mapped) = mapped else {
            unsafe {
                context.device.destroy_buffer(buffer, None);
                context.device.free_memory(memory, None);
            }
            return None;
        };
        Some(Self {
            context: Arc::clone(context),
            device: context.device.clone(),
            buffer,
            memory,
            mapped: mapped as usize,
            addresses,
            allocator: Allocator::new(size),
            revision: Cell::new(Some(1)),
            writes: WriteStamps::new(size),
            host_dirty: Ranges::default(),
            device_dirty: RefCell::default(),
            pending: RefCell::default(),
            pools: BTreeMap::new(),
            submission_tracking: false,
            statistics: TransferStatistics::default(),
        })
    }

    pub(crate) fn revision(&self) -> Option<u64> {
        self.revision.get()
    }

    pub(crate) fn mark_written(&self) {
        self.revision
            .set(self.revision.get().and_then(|value| value.checked_add(1)));
        self.writes.everywhere();
    }

    /// Record a write whose arena range is known. Readers of other ranges
    /// keep their contents current.
    pub(crate) fn mark_range_written(&self, range: Range<u64>) {
        self.revision
            .set(self.revision.get().and_then(|value| value.checked_add(1)));
        self.writes.range(range);
    }

    /// The write count a reader records when it takes a coherent copy.
    pub(crate) fn write_count(&self) -> u64 {
        self.writes.count.get()
    }

    /// Whether no write reached an arena range after a reader's write count.
    pub(crate) fn unwritten_since(&self, range: Range<u64>, count: u64) -> bool {
        self.revision.get().is_some() && self.writes.unwritten_since(range, count)
    }

    /// Cumulative transfer payload and legacy inspection counts.
    pub fn transfer_statistics(&self) -> TransferStatistics {
        self.statistics
    }

    pub(crate) fn notify_storage_write(&mut self, address: u64, size: u64) {
        for (key, offset, bytes) in self.pool_spans(address, size) {
            let _ = self.notify_host_write(key, offset, bytes);
        }
    }

    /// The pools a guest address range falls in, as pool key, offset in the
    /// pool and bytes.
    pub(crate) fn pool_spans(&self, address: u64, size: u64) -> Vec<(u64, u64, u64)> {
        let Some(end) = address.checked_add(size) else {
            return Vec::new();
        };
        self.allocator
            .pools
            .iter()
            .filter_map(|(&key, pool)| {
                let start = address.max(pool.storage);
                let limit = end.min(pool.storage + pool.size);
                (start < limit).then(|| (key, start - pool.storage, limit - start))
            })
            .collect()
    }

    pub(crate) fn track_submissions(&mut self) {
        self.submission_tracking = true;
    }

    /// Enable page tracking for a pool and its aliases. The initial upload is retained.
    /// The host must report every subsequent storage write before submission.
    pub fn track_host_writes(&mut self, key: u64) -> Option<()> {
        let pool = *self.allocator.pools.get(&key)?;
        if self.pools.get(&pool.block)?.tracked {
            return Some(());
        }
        self.pools.get_mut(&pool.block)?.tracked = true;
        for alias in self
            .allocator
            .pools
            .values()
            .filter(|p| p.block == pool.block)
        {
            self.host_dirty
                .insert(alias.offset..alias.offset + alias.size);
        }
        Some(())
    }

    /// Mark a changed host range. Dirty pages are clipped to the pool and coalesced.
    pub fn notify_host_write(&mut self, key: u64, offset: u64, size: u64) -> Option<()> {
        let pool = *self.allocator.pools.get(&key)?;
        if offset.checked_add(size)? > pool.size {
            return None;
        }
        if size == 0 {
            return Some(());
        }
        let start = pool.offset + offset;
        let end = start + size;
        if self.pools.get(&pool.block)?.direct {
            self.wait_range(start..end, true)?;
            self.mark_range_written(start..end);
        } else {
            self.host_dirty.insert(
                (start & !(PAGE_SIZE - 1)).max(pool.offset)
                    ..((end + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)).min(pool.offset + pool.size),
            );
        }
        Some(())
    }

    /// Return coherent backing for host-managed storage without boundary copies.
    /// Initialize the contents before use. Aliases share this choice.
    ///
    /// # Safety
    /// The pointer is valid until the last alias is released. Host callbacks must
    /// use this backing. Before each host access call wait_pool; before a host
    /// write also call notify_host_write. Do not access it during submission.
    pub unsafe fn map_pool(&mut self, key: u64) -> Result<*mut u8, PoolFault> {
        let pool = *self
            .allocator
            .pools
            .get(&key)
            .ok_or(PoolFault::UnknownRange)?;
        self.wait_range(pool.offset..pool.offset + pool.size, true)
            .ok_or(PoolFault::WaitFailed)?;
        self.pools
            .get_mut(&pool.block)
            .ok_or(PoolFault::UnknownRange)?
            .direct = true;
        self.host_dirty.remove(pool.offset..pool.offset + pool.size);
        Ok(unsafe { (self.mapped as *mut u8).add(pool.offset as usize) })
    }

    /// Wait for submissions that overlap a host access. Writes also wait for readers.
    pub fn wait_pool(
        &self,
        key: u64,
        offset: u64,
        size: usize,
        write: bool,
    ) -> Result<(), PoolFault> {
        let at = self
            .pool_offset(key, offset, size)
            .ok_or(PoolFault::UnknownRange)?;
        self.wait_range(at..at + size as u64, write)
            .ok_or(PoolFault::WaitFailed)
    }

    fn wait_range(&self, range: Range<u64>, write: bool) -> Option<()> {
        if range.is_empty() {
            return Some(());
        }
        let mut pending = self.pending.borrow_mut();
        for access in pending.iter() {
            if (write || access.write)
                && range.start < access.range.end
                && access.range.start < range.end
            {
                access.submission.wait()?;
            }
        }
        let mut kept = Vec::with_capacity(pending.len());
        for access in pending.drain(..) {
            if !access.submission.ready()? {
                kept.push(access);
            }
        }
        *pending = kept;
        Some(())
    }

    pub(crate) fn device_written(&self, offset: u64, size: u64) {
        self.mark_range_written(offset..offset + size);
        self.device_dirty.borrow_mut().insert(offset..offset + size);
    }

    pub(super) fn submitted(&self, offset: u64, size: u64, write: bool, submission: Submission) {
        if write {
            self.device_written(offset, size);
        }
        let mut pending = self.pending.borrow_mut();
        pending.retain(|access| access.submission.ready() != Some(true));
        pending.push(Access {
            range: offset..offset + size,
            write,
            submission,
        });
    }

    pub(super) fn submitted_all(&self, write: bool, submission: Submission) {
        for pool in self.allocator.pools.values() {
            self.submitted(pool.offset, pool.size, write, submission.clone());
        }
    }

    // Sort aliases together and visit each arena byte through only one callback.
    fn live_ranges(&self) -> Vec<(Range<u64>, u64, u64)> {
        let mut pools: Vec<_> = self.allocator.pools.values().copied().collect();
        pools.sort_by_key(|p| (p.offset, std::cmp::Reverse(p.size)));
        let mut end = 0;
        pools
            .into_iter()
            .filter_map(|p| {
                let start = p.offset.max(end);
                end = end.max(p.offset + p.size);
                (start < end).then(|| (start..end, p.storage + start - p.offset, p.block))
            })
            .collect()
    }

    /// Return the checked guest-to-device address mapping for the arena.
    pub fn addresses(&self) -> AddressMap {
        self.addresses
    }

    /// Borrow the Vulkan context retained by the arena.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// A bounded uniform descriptor for a live pool. Evidence: the public contract.
    /// The caller retains this arena and pool until descriptor users finish.
    pub fn uniform_buffer_info(
        &self,
        key: u64,
        offset: u64,
        size: u64,
    ) -> Option<vk::DescriptorBufferInfo> {
        let at = self.pool_offset(key, offset, usize::try_from(size).ok()?)?;
        // SAFETY: querying immutable properties of this context's physical device.
        let limits = unsafe {
            self.context
                .instance
                .get_physical_device_properties(self.context.physical_device)
        }
        .limits;
        if size == 0
            || size > u64::from(limits.max_uniform_buffer_range)
            || !at.is_multiple_of(limits.min_uniform_buffer_offset_alignment)
        {
            return None;
        }
        Some(
            vk::DescriptorBufferInfo::default()
                .buffer(self.buffer)
                .offset(at)
                .range(size),
        )
    }

    /// Storage fallback for constant banks with a device-aligned arena range.
    pub(crate) fn storage_buffer_info(
        &self,
        key: u64,
        offset: u64,
        size: u64,
    ) -> Option<vk::DescriptorBufferInfo> {
        let at = self.pool_offset(key, offset, usize::try_from(size).ok()?)?;
        let limits = unsafe {
            self.context
                .instance
                .get_physical_device_properties(self.context.physical_device)
        }
        .limits;
        if size == 0
            || size > u64::from(limits.max_storage_buffer_range)
            || !at.is_multiple_of(limits.min_storage_buffer_offset_alignment)
        {
            return None;
        }
        Some(
            vk::DescriptorBufferInfo::default()
                .buffer(self.buffer)
                .offset(at)
                .range(size),
        )
    }

    /// Canonical tightly packed base-level texels in a live arena pool.
    pub(crate) fn image_region(
        &self,
        key: u64,
        offset: u64,
        size: usize,
    ) -> Option<(vk::Buffer, u64)> {
        let (buffer, at) = self.buffer_region(key, offset, size)?;
        // Four-byte colour and depth texels require four-byte copy offsets.
        at.is_multiple_of(4).then_some((buffer, at))
    }

    /// A bounded nonempty slice of the canonical arena. Callers check use-specific alignment.
    pub(crate) fn buffer_region(
        &self,
        key: u64,
        offset: u64,
        size: usize,
    ) -> Option<(vk::Buffer, u64)> {
        let at = self.pool_offset(key, offset, size)?;
        (size > 0).then_some((self.buffer, at))
    }

    /// Whether the arena contains an allocation for this pool key.
    pub fn contains_pool(&self, key: u64) -> bool {
        self.allocator.pools.contains_key(&key)
    }

    /// Allocate a 16-byte-aligned pool range and return its guest GPU address.
    pub fn allocate_pool(&mut self, key: u64, storage: u64, size: u64) -> Option<u64> {
        let aliased = self.allocator.pools.values().any(|pool| {
            storage >= pool.storage
                && storage
                    .checked_add(size)
                    .is_some_and(|end| end <= pool.storage + pool.size)
        });
        let pool = self.allocator.allocate(key, storage, size)?;
        self.mark_written();
        self.pools.entry(pool.block).or_insert(PoolSync {
            size: (size + 15) & !15,
            ..PoolSync::default()
        });
        self.host_dirty.insert(pool.offset..pool.offset + pool.size);
        if !aliased {
            let reserved = (size + 15) & !15;
            // SAFETY: this new block is within the buffer. New or recycled bytes
            // and the complete-word padding start zeroed before use.
            unsafe {
                if self
                    .wait_range(pool.offset..pool.offset + reserved, true)
                    .is_none()
                {
                    self.allocator.release(key);
                    return None;
                }
                std::ptr::write_bytes(
                    (self.mapped as *mut u8).add(pool.offset as usize),
                    0,
                    reserved as usize,
                );
            }
        }
        self.addresses.guest(pool.offset)
    }

    /// Release a pool range; return false if its key is unknown.
    pub fn release_pool(&mut self, key: u64) -> bool {
        let Some(pool) = self.allocator.pools.get(&key).copied() else {
            return false;
        };
        if self
            .wait_range(pool.block..pool.block + self.pools[&pool.block].size, true)
            .is_none()
        {
            return false;
        }
        self.mark_written();
        self.allocator.release(key);
        if !self.allocator.pools.values().any(|p| p.block == pool.block) {
            let sync = self.pools.remove(&pool.block).expect("live block");
            self.host_dirty.remove(pool.block..pool.block + sync.size);
            self.device_dirty
                .borrow_mut()
                .remove(pool.block..pool.block + sync.size);
        }
        true
    }

    fn pool_offset(&self, key: u64, offset: u64, size: usize) -> Option<u64> {
        let pool = self.allocator.pools.get(&key)?;
        (offset.checked_add(size as u64)? <= pool.size).then(|| pool.offset + offset)
    }

    /// Copy bytes to a checked pool range and flush noncoherent memory as needed.
    pub fn write_pool(&mut self, key: u64, offset: u64, bytes: &[u8]) -> Option<()> {
        self.write_at(self.pool_offset(key, offset, bytes.len())?, bytes)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Option<()> {
        if bytes.is_empty() {
            return Some(());
        }
        // SAFETY: the range is inside the mapped buffer, coherent memory needs no
        // flush, and exclusive access prevents simultaneous host or queue writes.
        unsafe {
            self.wait_range(offset..offset + bytes.len() as u64, true)?;
            self.device_written(offset, bytes.len() as u64);
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (self.mapped as *mut u8).add(offset as usize),
                bytes.len(),
            );
        }
        Some(())
    }

    /// Read coherent bytes after completion. Submitted shader writes must include
    /// a shader-write to host-read memory barrier before this method is called.
    pub fn read_pool(&mut self, key: u64, offset: u64, bytes: &mut [u8]) -> Option<()> {
        let offset = self.pool_offset(key, offset, bytes.len())?;
        if bytes.is_empty() {
            return Some(());
        }
        // SAFETY: waiting completes submitted writes; the range is inside coherent
        // host-visible memory and the output slice is valid for its length.
        unsafe {
            self.wait_range(offset..offset + bytes.len() as u64, false)?;
            std::ptr::copy_nonoverlapping(
                (self.mapped as *const u8).add(offset as usize),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        Some(())
    }

    /// Upload dirty pages. Legacy hosts are inspected for changes through bounded reads.
    /// Failed callbacks retain the unfinished ranges for retry.
    pub fn upload(&mut self, mut read: impl FnMut(u64, &mut [u8]) -> bool) -> bool {
        let mut scratch = Vec::new();
        for (range, storage, block) in self.live_ranges() {
            let mode = &self.pools[&block];
            if mode.direct {
                continue;
            }
            if !mode.tracked {
                scratch.resize(64 * 1024, 0);
                let mut at = range.start;
                while at < range.end {
                    let end = (at + scratch.len() as u64).min(range.end);
                    let bytes = &mut scratch[..(end - at) as usize];
                    if !read(storage + at - range.start, bytes) {
                        return false;
                    }
                    self.statistics.inspected += bytes.len() as u64;
                    if self.wait_range(at..end, true).is_none() {
                        return false;
                    }
                    for (page, source) in bytes.chunks(PAGE_SIZE as usize).enumerate() {
                        let start = at + (page as u64 * PAGE_SIZE);
                        let target = unsafe {
                            std::slice::from_raw_parts_mut(
                                (self.mapped as *mut u8).add(start as usize),
                                source.len(),
                            )
                        };
                        if target != source {
                            target.copy_from_slice(source);
                            self.statistics.uploaded += source.len() as u64;
                            self.mark_range_written(start..start + source.len() as u64);
                        }
                    }
                    self.host_dirty.remove(at..end);
                    at = end;
                }
            } else {
                for dirty in self.host_dirty.intersections(range.clone()) {
                    let mut at = dirty.start;
                    while at < dirty.end {
                        let end = (at + 64 * 1024).min(dirty.end);
                        if self.wait_range(at..end, true).is_none() {
                            return false;
                        }
                        // A staging read preserves the arena if the callback fails.
                        scratch.resize((end - at) as usize, 0);
                        if !read(storage + at - range.start, &mut scratch) {
                            return false;
                        }
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                scratch.as_ptr(),
                                (self.mapped as *mut u8).add(at as usize),
                                scratch.len(),
                            );
                        }
                        self.statistics.uploaded += end - at;
                        self.host_dirty.remove(at..end);
                        self.mark_range_written(at..end);
                        at = end;
                    }
                }
            }
        }
        true
    }

    /// Download declared device writes after only their overlapping completions.
    /// Direct mappings need neither: the host reads them in place and waits
    /// with `wait_pool` first, as `map_pool` requires, and the pending
    /// accesses that wait checks stay recorded. External raw submissions must
    /// be completed by their caller before using an untracked arena.
    pub fn download(&mut self, mut write: impl FnMut(u64, &[u8]) -> bool) -> bool {
        for (range, storage, block) in self.live_ranges() {
            let ranges = if self.submission_tracking {
                self.device_dirty.borrow().intersections(range.clone())
            } else {
                vec![range.clone()]
            };
            for dirty in ranges {
                if self.pools[&block].direct {
                    self.device_dirty.borrow_mut().remove(dirty);
                    continue;
                }
                #[cfg(feature = "draw-metrics")]
                let wait_span = crate::draw_metrics::PipelineSpan::new(15);
                if self.wait_range(dirty.clone(), false).is_none() {
                    return false;
                }
                #[cfg(feature = "draw-metrics")]
                drop(wait_span);
                let mut at = dirty.start;
                while at < dirty.end {
                    let end = (at + 64 * 1024).min(dirty.end);
                    #[cfg(feature = "draw-metrics")]
                    let _span = crate::draw_metrics::PipelineSpan::new(16);
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            (self.mapped as *const u8).add(at as usize),
                            (end - at) as usize,
                        )
                    };
                    if !write(storage + at - range.start, bytes) {
                        return false;
                    }
                    self.statistics.downloaded += end - at;
                    self.device_dirty.borrow_mut().remove(at..end);
                    at = end;
                }
            }
        }
        true
    }

    /// Describe the eight-byte global delta range for the selected shader stages.
    pub fn push_constant_range(stages: vk::ShaderStageFlags) -> vk::PushConstantRange {
        vk::PushConstantRange::default()
            .stage_flags(stages)
            .offset(PUSH_GLOBAL_DELTA_OFFSET)
            .size(PUSH_GLOBAL_DELTA_SIZE)
    }

    /// # Safety
    /// `command` must be recording on this device and `layout` must include
    /// push_constant_range(stages). Push again before every global draw or dispatch.
    pub unsafe fn push_delta(
        &self,
        command: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
    ) {
        super::recording_device::RecordingDevice::new(&self.device).cmd_push_constants(
            command,
            layout,
            stages,
            PUSH_GLOBAL_DELTA_OFFSET,
            &self.addresses.delta().to_ne_bytes(),
        );
    }
}

impl Drop for GlobalMemory {
    fn drop(&mut self) {
        // SAFETY: the owning device outlives this object and all submitted users
        // finish before either the buffer or its memory is freed.
        unsafe {
            let _ = self.wait_range(0..u64::MAX, true);
            self.device.unmap_memory(self.memory);
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
#[path = "memory_tests.rs"]
mod tests;
