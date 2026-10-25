//! Flat guest address allocation.

#[cfg(any(test, feature = "vulkan"))]
use std::collections::BTreeMap;

/// Base of the flat guest GPU address range.
pub const GUEST_BASE: u64 = 0x1_0000;
/// Default flat arena size in bytes, equal to 1 GiB.
pub const ARENA_SIZE: u64 = 1024 * 1024 * 1024;
/// Byte offset of the global address delta in push constants.
pub const PUSH_GLOBAL_DELTA_OFFSET: u32 = 0;
/// Size in bytes of the 64-bit global address delta.
pub const PUSH_GLOBAL_DELTA_SIZE: u32 = 8;

/// Checked mapping between one guest range and a host device-address range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddressMap {
    /// Base guest GPU address.
    pub guest_base: u64,
    /// Base host buffer device address.
    pub host_base: u64,
    /// Length of the storage range in bytes.
    pub size: u64,
}

impl AddressMap {
    /// Construct a mapping; reject zero bases, non-16-byte-aligned bases, empty or overflowing ranges.
    pub fn new(guest_base: u64, host_base: u64, size: u64) -> Option<Self> {
        if size == 0
            || guest_base == 0
            || host_base == 0
            || !guest_base.is_multiple_of(16)
            || !host_base.is_multiple_of(16)
        {
            return None;
        }
        guest_base.checked_add(size)?;
        host_base.checked_add(size)?;
        Some(Self {
            guest_base,
            host_base,
            size,
        })
    }

    /// Wrapping host-minus-guest delta used by translated global-memory access.
    pub fn delta(self) -> u64 {
        self.host_base.wrapping_sub(self.guest_base)
    }

    /// Return the guest address at a byte offset, or None outside the range.
    pub fn guest(self, offset: u64) -> Option<u64> {
        (offset < self.size).then(|| self.guest_base + offset)
    }

    /// Resolve a guest address to a host device address, or None outside the range.
    pub fn host(self, guest: u64) -> Option<u64> {
        let offset = guest.checked_sub(self.guest_base)?;
        (offset < self.size).then(|| guest.wrapping_add(self.delta()))
    }
}

/// Allocated arena range and its backing program storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolAllocation {
    /// Byte offset from the arena base.
    pub offset: u64,
    /// Length of the storage range in bytes.
    pub size: u64,
    /// Base address of host-managed program storage.
    pub storage: u64,
    pub(crate) block: u64,
}

/// First-fit ranges with shared backing for contained CPU storage aliases.
/// Partial overlaps are rejected because extending a live block would move pointers.
#[cfg(any(test, feature = "vulkan"))]
pub(crate) struct Allocator {
    free: BTreeMap<u64, u64>,
    blocks: BTreeMap<u64, u64>,
    /// Current pool allocations keyed by program object address.
    pub pools: BTreeMap<u64, PoolAllocation>,
}

#[cfg(any(test, feature = "vulkan"))]
impl Allocator {
    /// Start with one free arena range of size bytes.
    pub fn new(size: u64) -> Self {
        Self {
            free: BTreeMap::from([(0, size)]),
            blocks: BTreeMap::new(),
            pools: BTreeMap::new(),
        }
    }

    /// Allocate or alias a checked 16-byte-aligned range; reject duplicates and partial overlaps.
    pub fn allocate(&mut self, key: u64, storage: u64, size: u64) -> Option<PoolAllocation> {
        if size == 0 || self.pools.contains_key(&key) {
            return None;
        }
        let end = storage.checked_add(size)?;
        let alias = self
            .pools
            .values()
            .find(|pool| storage >= pool.storage && end <= pool.storage + pool.size)
            .copied();
        let pool = if let Some(pool) = alias {
            let offset = pool.offset + storage - pool.storage;
            if !offset.is_multiple_of(16) {
                return None;
            }
            PoolAllocation {
                offset,
                size,
                storage,
                block: pool.block,
            }
        } else {
            if self
                .pools
                .values()
                .any(|pool| storage < pool.storage + pool.size && pool.storage < end)
            {
                return None;
            }
            let reserved = size.checked_add(15)? & !15;
            let (&offset, &available) =
                self.free.iter().find(|(_, length)| **length >= reserved)?;
            self.free.remove(&offset);
            if available > reserved {
                self.free.insert(offset + reserved, available - reserved);
            }
            self.blocks.insert(offset, reserved);
            PoolAllocation {
                offset,
                size,
                storage,
                block: offset,
            }
        };
        self.pools.insert(key, pool);
        Some(pool)
    }

    /// Release the keyed pool and free its backing block after its last alias.
    pub fn release(&mut self, key: u64) -> bool {
        let Some(pool) = self.pools.remove(&key) else {
            return false;
        };
        if self.pools.values().any(|other| other.block == pool.block) {
            return true;
        }
        let mut start = pool.block;
        let mut size = self.blocks.remove(&start).expect("live allocation block");
        if let Some((&previous, &length)) = self.free.range(..start).next_back() {
            if previous + length == start {
                self.free.remove(&previous);
                start = previous;
                size += length;
            }
        }
        if let Some(length) = self.free.remove(&(start + size)) {
            size += length;
        }
        self.free.insert(start, size);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_delta_preserves_interior_addresses_and_alignment() {
        for host in [0x1000, 0x1_0000, 0x1234_0000_0000] {
            let map = AddressMap::new(GUEST_BASE, host, ARENA_SIZE).unwrap();
            for offset in [0, 1, 15, 16, 0xffff, ARENA_SIZE - 1] {
                let guest = map.guest(offset).unwrap();
                assert_eq!(map.host(guest), Some(host + offset));
                assert_eq!(guest.wrapping_add(map.delta()), host + offset);
                assert_eq!(guest % 16, (host + offset) % 16);
                assert!(guest <= u64::from(u32::MAX));
            }
            assert_eq!(map.host(GUEST_BASE - 1), None);
            assert_eq!(map.guest(ARENA_SIZE), None);
            assert_eq!(map.host(GUEST_BASE + ARENA_SIZE), None);
        }
    }

    #[test]
    fn rejects_invalid_or_wrapping_ranges() {
        for (guest, host, size) in [
            (0, 16, 1),
            (16, 0, 1),
            (17, 16, 1),
            (16, 17, 1),
            (16, 16, 0),
            (u64::MAX - 15, 16, 16),
            (16, u64::MAX - 15, 16),
        ] {
            assert_eq!(AddressMap::new(guest, host, size), None);
        }
    }

    #[test]
    fn allocations_reuse_coalesced_space_without_moving_live_pools() {
        let mut alloc = Allocator::new(64);
        assert_eq!(alloc.allocate(1, 0x1000, 17).unwrap().offset, 0);
        assert_eq!(alloc.allocate(2, 0x2000, 16).unwrap().offset, 32);
        assert_eq!(alloc.allocate(3, 0x3000, 16).unwrap().offset, 48);
        assert!(alloc.allocate(4, 0x4000, 1).is_none());
        assert!(alloc.allocate(1, 0x1000, 17).is_none());
        assert!(alloc.release(1));
        assert!(alloc.release(2));
        assert_eq!(alloc.allocate(4, 0x4000, 48).unwrap().offset, 0);
        assert_eq!(alloc.pools[&3].offset, 48);
        assert!(alloc.release(3));
        assert!(alloc.release(4));
        assert_eq!(alloc.allocate(5, 0x5000, 64).unwrap().offset, 0);
    }

    #[test]
    fn contained_aliases_share_bytes_and_outlive_the_original_pool() {
        let mut alloc = Allocator::new(64);
        alloc.allocate(1, 0x1000, 64).unwrap();
        assert_eq!(alloc.allocate(2, 0x1010, 16).unwrap().offset, 16);
        assert!(alloc.allocate(3, 0x1020, 64).is_none());
        assert!(alloc.allocate(3, 0x1001, 1).is_none());
        assert!(alloc.release(1));
        assert!(alloc.allocate(3, 0x2000, 1).is_none());
        assert!(alloc.release(2));
        assert_eq!(alloc.allocate(3, 0x2000, 64).unwrap().offset, 0);
        assert!(alloc.allocate(4, u64::MAX, 2).is_none());
    }
}
