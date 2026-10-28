//! Maps keyed by SwXBx's own handles, addresses and small records.
//!
//! These keys are object handles and packed integers, not attacker-chosen
//! text, so the per-draw lookups use a multiply-mix hasher instead of the
//! keyed SipHash default. The final avalanche spreads aligned addresses
//! across the low bits a table indexes with.
use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
};

/// A hash map whose keys are SwXBx handles or small integer records.
pub(crate) type KeyMap<K, V> = HashMap<K, V, BuildHasherDefault<KeyHasher>>;

/// Multiply-mix hasher over 64-bit words with the public-domain murmur3 `fmix64` finalizer.
#[derive(Default, Clone, Copy)]
pub(crate) struct KeyHasher(u64);

const MIX: u64 = 0x9e37_79b9_7f4a_7c15;

impl KeyHasher {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(23) ^ word).wrapping_mul(MIX);
    }
}

impl Hasher for KeyHasher {
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for &word in words {
            self.add(u64::from_le_bytes(word));
        }
        let mut last = [0; 8];
        last[..rest.len()].copy_from_slice(rest);
        self.add(u64::from_le_bytes(last) ^ ((rest.len() as u64) << 56));
    }
    fn write_u8(&mut self, v: u8) {
        self.add(u64::from(v));
    }
    fn write_u16(&mut self, v: u16) {
        self.add(u64::from(v));
    }
    fn write_u32(&mut self, v: u32) {
        self.add(u64::from(v));
    }
    fn write_u64(&mut self, v: u64) {
        self.add(v);
    }
    fn write_usize(&mut self, v: usize) {
        self.add(v as u64);
    }
    fn write_i32(&mut self, v: i32) {
        self.add(v as u32 as u64);
    }
    fn write_i64(&mut self, v: i64) {
        self.add(v as u64);
    }
    fn finish(&self) -> u64 {
        let mut h = self.0;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^ (h >> 33)
    }
}
