//! Coalesced half-open byte ranges.
use std::{collections::BTreeMap, ops::Range};

#[derive(Default, Debug)]
pub(crate) struct Ranges(BTreeMap<u64, u64>);
impl Ranges {
    pub(crate) fn insert(&mut self, mut range: Range<u64>) {
        if range.is_empty() {
            return;
        }
        if let Some((&start, &end)) = self.0.range(..=range.start).next_back() {
            if end >= range.start {
                self.0.remove(&start);
                range.start = start;
                range.end = range.end.max(end);
            }
        }
        while let Some((&start, &end)) = self.0.range(range.start..=range.end).next() {
            self.0.remove(&start);
            range.end = range.end.max(end);
        }
        self.0.insert(range.start, range.end);
    }
    pub(crate) fn remove(&mut self, range: Range<u64>) {
        if range.is_empty() {
            return;
        }
        let overlaps: Vec<_> = self
            .0
            .range(..range.end)
            .filter(|(_, end)| **end > range.start)
            .map(|(&s, &e)| s..e)
            .collect();
        for old in overlaps {
            self.0.remove(&old.start);
            if old.start < range.start {
                self.0.insert(old.start, range.start);
            }
            if old.end > range.end {
                self.0.insert(range.end, old.end);
            }
        }
    }
    pub(crate) fn intersections(&self, range: Range<u64>) -> Vec<Range<u64>> {
        self.0
            .range(..range.end)
            .filter_map(|(&s, &e)| {
                let at = s.max(range.start)..e.min(range.end);
                (!at.is_empty()).then_some(at)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesces_clips_and_splits_without_losing_bytes() {
        let mut ranges = Ranges::default();
        for range in [9..12, 1..4, 4..7, 2..6, 7..9, 20..25, 3..3] {
            ranges.insert(range);
        }
        assert_eq!(ranges.intersections(0..30), [1..12, 20..25]);
        assert_eq!(ranges.intersections(3..22), [3..12, 20..22]);
        ranges.remove(5..10);
        assert_eq!(ranges.intersections(0..30), [1..5, 10..12, 20..25]);
        ranges.remove(11..24);
        assert_eq!(ranges.intersections(0..30), [1..5, 10..11, 24..25]);
        ranges.remove(0..u64::MAX);
        assert!(ranges.intersections(0..u64::MAX).is_empty());
    }

    #[test]
    fn range_operations_match_a_byte_set() {
        let mut ranges = Ranges::default();
        let mut bytes = [false; 128];
        let mut state = 7_u64;
        for step in 0..2000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let start = (state >> 32) as usize % bytes.len();
            let end = (start + (state as usize % 32)).min(bytes.len());
            if step % 3 == 0 {
                ranges.remove(start as u64..end as u64);
            } else {
                ranges.insert(start as u64..end as u64);
            }
            bytes[start..end].fill(step % 3 != 0);
            let mut actual = [false; 128];
            for range in ranges.intersections(0..128) {
                actual[range.start as usize..range.end as usize].fill(true);
            }
            assert_eq!(actual, bytes);
        }
    }
}
