//! Per-buffer recording ownership and immutable publication.
use super::objects::RecordedCommand;
use arc_swap::ArcSwap;
use std::{
    cell::RefCell,
    collections::HashMap,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering},
        Arc,
    },
};
static NEXT_REGISTRY: AtomicU64 = AtomicU64::new(1);
type Last = Option<(u64, u64, Arc<Recorder>)>;
thread_local! {
    static LAST: RefCell<Last> = const { RefCell::new(None) };
}
#[derive(Default, Debug)]
struct State {
    open: bool,
    sequence: u64,
    commands: Vec<RecordedCommand>,
}
#[derive(Debug)]
struct Published {
    commands: AtomicPtr<Vec<RecordedCommand>>,
    #[cfg(test)]
    len: usize,
}
impl Published {
    fn new(commands: Vec<RecordedCommand>) -> Self {
        #[cfg(test)]
        let len = commands.len();
        Self {
            commands: AtomicPtr::new(Box::into_raw(Box::new(commands))),
            #[cfg(test)]
            len,
        }
    }
    fn take(&self) -> Option<Vec<RecordedCommand>> {
        let commands = self.commands.swap(ptr::null_mut(), Ordering::AcqRel);
        if commands.is_null() {
            None
        } else {
            // The exchange transfers sole ownership independently of active recording.
            Some(*unsafe { Box::from_raw(commands) })
        }
    }
}
impl Drop for Published {
    fn drop(&mut self) {
        let commands = self.commands.swap(ptr::null_mut(), Ordering::AcqRel);
        if !commands.is_null() {
            unsafe {
                drop(Box::from_raw(commands));
            }
        }
    }
}
#[derive(Debug)]
struct Recorder {
    state: AtomicPtr<State>,
    live: AtomicBool,
    completed: ArcSwap<HashMap<u64, Arc<Published>>>,
    recycled: AtomicPtr<Vec<RecordedCommand>>,
}
impl Recorder {
    fn new() -> Self {
        Self {
            state: AtomicPtr::new(Box::into_raw(Box::default())),
            live: AtomicBool::new(true),
            completed: ArcSwap::from_pointee(HashMap::new()),
            recycled: AtomicPtr::new(ptr::null_mut()),
        }
    }
    fn access<R>(&self, f: impl FnOnce(&mut State) -> R) -> Option<R> {
        if !self.live.load(Ordering::Acquire) {
            return None;
        }
        let state = self.state.swap(ptr::null_mut(), Ordering::AcqRel);
        if state.is_null() {
            return None;
        }
        // The exchange grants sole ownership. A concurrent call fails immediately.
        let mut lease = Lease {
            recorder: self,
            state: Some(unsafe { Box::from_raw(state) }),
        };
        if !self.live.load(Ordering::Acquire) {
            return None;
        }
        Some(f(lease.state.as_mut().unwrap()))
    }
    fn clear_recycled(&self) {
        let recycled = self.recycled.swap(ptr::null_mut(), Ordering::AcqRel);
        if !recycled.is_null() {
            unsafe {
                drop(Box::from_raw(recycled));
            }
        }
    }
    fn close(&self) {
        self.live.store(false, Ordering::Release);
        self.clear_recycled();
        self.completed.store(Arc::new(HashMap::new()));
        let state = self.state.swap(ptr::null_mut(), Ordering::AcqRel);
        if !state.is_null() {
            unsafe {
                drop(Box::from_raw(state));
            }
        }
    }
}
struct Lease<'a> {
    recorder: &'a Recorder,
    state: Option<Box<State>>,
}
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        let state = Box::into_raw(self.state.take().unwrap());
        self.recorder.state.store(state, Ordering::Release);
        // Finalization can race an active lease without reclaiming its storage.
        if !self.recorder.live.load(Ordering::Acquire) {
            self.recorder.completed.store(Arc::new(HashMap::new()));
            let state = self.recorder.state.swap(ptr::null_mut(), Ordering::AcqRel);
            if !state.is_null() {
                unsafe {
                    drop(Box::from_raw(state));
                }
            }
        }
    }
}
impl Drop for Recorder {
    fn drop(&mut self) {
        self.close();
    }
}
pub(super) struct Recordings {
    id: u64,
    entries: ArcSwap<HashMap<u64, Arc<Recorder>>>,
}
impl Default for Recordings {
    fn default() -> Self {
        Self {
            id: NEXT_REGISTRY.fetch_add(1, Ordering::Relaxed),
            entries: ArcSwap::from_pointee(HashMap::new()),
        }
    }
}
impl Recordings {
    fn with_entry<R>(&self, address: u64, f: impl FnOnce(&Arc<Recorder>) -> R) -> Option<R> {
        LAST.with(|last| {
            let cached = last.borrow();
            if let Some((id, key, entry)) = cached.as_ref() {
                if *id == self.id && *key == address && entry.live.load(Ordering::Acquire) {
                    return Some(f(entry));
                }
            }
            drop(cached);
            let entry = self.entries.load().get(&address)?.clone();
            *last.borrow_mut() = Some((self.id, address, entry.clone()));
            entry.live.load(Ordering::Acquire).then(|| f(&entry))
        })
    }
    fn entry(&self, address: u64) -> Option<Arc<Recorder>> {
        self.with_entry(address, Arc::clone)
    }
    pub fn initialize(&self, address: u64) {
        self.finalize(address);
        let entry = Arc::new(Recorder::new());
        self.entries.rcu(|current| {
            let mut updated = (**current).clone();
            updated.insert(address, entry.clone());
            updated
        });
    }
    pub fn finalize(&self, address: u64) {
        if let Some(entry) = self.entry(address) {
            entry.close();
        }
        self.entries.rcu(|current| {
            let mut updated = (**current).clone();
            updated.remove(&address);
            updated
        });
    }
    pub fn begin(&self, address: u64) -> bool {
        self.entry(address)
            .and_then(|e| {
                e.access(|state| {
                    state.open = true;
                    state.commands.clear();
                    let recycled = e.recycled.swap(ptr::null_mut(), Ordering::AcqRel);
                    if !recycled.is_null() {
                        let recycled = *unsafe { Box::from_raw(recycled) };
                        if recycled.capacity() > state.commands.capacity() {
                            state.commands = recycled;
                        }
                    }
                })
            })
            .is_some()
    }
    pub fn push(&self, address: u64, command: RecordedCommand) -> bool {
        self.with_entry(address, |entry| {
            entry
                .access(|state| {
                    if state.open {
                        state.commands.push(command);
                    }
                })
                .is_some()
        })
        .unwrap_or(true)
    }
    pub fn end(&self, address: u64) -> Option<u64> {
        let entry = self.entry(address)?;
        entry.access(|state| {
            state.open = false;
            state.sequence += 1;
            let current = state.sequence & 0xffff;
            let handle = address | current << 48;
            let published = Arc::new(Published::new(std::mem::take(&mut state.commands)));
            entry.completed.rcu(|completed| {
                let mut updated = (**completed).clone();
                updated.retain(|h, _| current.wrapping_sub(h >> 48) & 0xffff < 64);
                updated.insert(handle, published.clone());
                updated
            });
            handle
        })
    }
    pub fn recycle(&self, handle: u64, mut commands: Vec<RecordedCommand>) {
        commands.clear();
        let Some(entry) = self.entry(handle & ((1 << 48) - 1)) else {
            return;
        };
        let previous = entry
            .recycled
            .swap(Box::into_raw(Box::new(commands)), Ordering::AcqRel);
        if !previous.is_null() {
            unsafe {
                drop(Box::from_raw(previous));
            }
        }
        if !entry.live.load(Ordering::Acquire) {
            entry.clear_recycled();
        }
    }
    pub fn take(&self, handle: u64) -> Option<Vec<RecordedCommand>> {
        let entry = self.entry(handle & ((1 << 48) - 1))?;
        let published = entry.completed.load().get(&handle)?.clone();
        published.take()
    }
    #[cfg(test)]
    pub fn len(&self, handle: u64) -> Option<usize> {
        let entry = self.entry(handle & ((1 << 48) - 1))?;
        let published = entry.completed.load().get(&handle)?.clone();
        (!published.commands.load(Ordering::Acquire).is_null()).then_some(published.len)
    }
}
impl Drop for Recordings {
    fn drop(&mut self) {
        // Cached generations retain only their closed metadata after registry shutdown.
        for entry in self.entries.load().values() {
            entry.close();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{mpsc, Arc};

    #[test]
    fn completed_capacity_returns_without_claiming_active_recording() {
        let recordings = Recordings::default();
        recordings.initialize(0x1000);
        assert!(recordings.begin(0x1000));
        for _ in 0..1024 {
            assert!(recordings.push(0x1000, RecordedCommand::SetViewport([0; 5])));
        }
        let handle = recordings.end(0x1000).unwrap();
        let commands = recordings.take(handle).unwrap();
        let storage = commands.as_ptr();
        assert!(recordings.begin(0x1000));
        assert!(recordings.push(0x1000, RecordedCommand::SetViewport([1; 5])));
        recordings.recycle(handle, commands);
        let next = recordings.end(0x1000).unwrap();
        assert_eq!(
            recordings.take(next).unwrap(),
            vec![RecordedCommand::SetViewport([1; 5])]
        );
        assert!(recordings.begin(0x1000));
        let entry = recordings.entry(0x1000).unwrap();
        assert!(entry
            .access(|state| state.commands.as_ptr() == storage)
            .unwrap());
    }

    #[test]
    fn independent_buffers_publish_on_many_threads_and_migrate() {
        let recordings = Arc::new(Recordings::default());
        for key in 1..=8 {
            recordings.initialize(key);
        }
        std::thread::scope(|scope| {
            for key in 1..=8 {
                let recordings = &recordings;
                scope.spawn(move || {
                    assert!(recordings.begin(key));
                    for value in 0..4096 {
                        assert!(
                            recordings.push(key, RecordedCommand::SetDepthRange([key, value, 0]))
                        );
                    }
                });
            }
        });
        // Finish from a different thread, then claim the published list on the caller.
        let handles = std::thread::spawn({
            let recordings = recordings.clone();
            move || {
                (1..=8)
                    .map(|key| recordings.end(key).unwrap())
                    .collect::<Vec<_>>()
            }
        })
        .join()
        .unwrap();
        for (key, handle) in (1..=8).zip(handles) {
            let commands = recordings.take(handle).unwrap();
            assert_eq!(commands.len(), 4096);
            for (value, command) in commands.into_iter().enumerate() {
                assert_eq!(
                    command,
                    RecordedCommand::SetDepthRange([key, value as u64, 0])
                );
            }
            assert!(recordings.take(handle).is_none());
        }
    }

    #[test]
    fn contended_buffer_rejects_immediately_and_recovers_after_panic() {
        let recordings = Recordings::default();
        recordings.initialize(1);
        let (entered, entry) = mpsc::sync_channel(1);
        let (release, released) = mpsc::sync_channel(1);
        std::thread::scope(|scope| {
            let recordings = &recordings;
            scope.spawn(move || {
                recordings.entry(1).unwrap().access(|_| {
                    entered.send(()).unwrap();
                    released.recv().unwrap();
                });
            });
            entry.recv().unwrap();
            assert!(!recordings.begin(1));
            release.send(()).unwrap();
        });
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            recordings.entry(1).unwrap().access(|_| panic!("fixture"));
        }));
        assert!(result.is_err());
        assert!(recordings.begin(1));
        assert!(recordings.push(1, RecordedCommand::SetDepthRange([1, 2, 3])));
        recordings.finalize(1);
        recordings.initialize(1);
        assert!(recordings.begin(1));
        assert!(recordings
            .take(recordings.end(1).unwrap())
            .unwrap()
            .is_empty());
    }
    #[test]
    fn completed_list_can_be_claimed_while_the_next_recording_is_owned() {
        let recordings = Recordings::default();
        recordings.initialize(1);
        assert!(recordings.begin(1));
        assert!(recordings.push(1, RecordedCommand::SetDepthRange([1, 2, 3])));
        let old = recordings.end(1).unwrap();
        assert!(recordings.begin(1));
        let (entered, entry) = mpsc::sync_channel(1);
        let (release, released) = mpsc::sync_channel(1);
        std::thread::scope(|scope| {
            let recordings = &recordings;
            scope.spawn(move || {
                recordings.entry(1).unwrap().access(|state| {
                    entered.send(()).unwrap();
                    released.recv().unwrap();
                    state
                        .commands
                        .push(RecordedCommand::SetDepthRange([1, 20, 30]));
                });
            });
            entry.recv().unwrap();
            let completed = recordings.take(old);
            release.send(()).unwrap();
            assert_eq!(
                completed,
                Some(vec![RecordedCommand::SetDepthRange([1, 2, 3])])
            );
        });
        let next = recordings.end(1).unwrap();
        assert_eq!(
            recordings.take(next),
            Some(vec![RecordedCommand::SetDepthRange([1, 20, 30])])
        );
        assert!(recordings.take(old).is_none());
    }
}
