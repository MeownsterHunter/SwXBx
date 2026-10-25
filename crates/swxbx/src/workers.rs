//! Bounded owned pipeline requests.
/// Cumulative request reuse and insertion counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Requests that reused an existing entry.
    pub hits: u64,
    /// Requests that created a new entry.
    pub misses: u64,
}
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU8, Ordering},
        mpsc, Arc, Condvar, Mutex, OnceLock,
    },
    thread::{self, JoinHandle},
    time::Instant,
};

/// Completion state of an owned compilation request.
pub enum PipelineStatus<P> {
    /// The owned job is waiting for a worker.
    Queued,
    /// A worker is compiling the owned job.
    Compiling,
    /// Compilation completed and the shared result is available.
    Ready(Arc<P>),
    /// Compilation failed; the diagnostic stays available until explicit retry.
    Failed(String),
}

struct RequestState<P> {
    completed: Condvar,
    phase: AtomicU8,
    result: OnceLock<Result<Arc<P>, String>>,
    waiter: Mutex<()>,
}

/// Owned completion handle. Replacing a program replaces its handle, so an old
/// completion cannot modify the current program or follow a guest pointer.
pub struct PipelineRequest<P>(Arc<RequestState<P>>);

impl<P> Clone for PipelineRequest<P> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<P> PipelineRequest<P> {
    /// Create an already completed request retaining the supplied shared result.
    pub fn ready(value: Arc<P>) -> Self {
        Self::completed(Ok(value))
    }

    pub(crate) fn completed(result: Result<Arc<P>, String>) -> Self {
        Self(Arc::new(RequestState {
            completed: Condvar::new(),
            phase: AtomicU8::new(2),
            result: OnceLock::from(result),
            waiter: Mutex::new(()),
        }))
    }

    /// Wait until the owned job completes or fails.
    pub fn wait(&self) -> PipelineStatus<P> {
        if self.0.result.get().is_some() {
            return self.poll();
        }
        let guard = self.0.waiter.lock().unwrap_or_else(|p| p.into_inner());
        let _guard = self
            .0
            .completed
            .wait_while(guard, |_| self.0.result.get().is_none())
            .unwrap_or_else(|p| p.into_inner());
        self.poll()
    }

    /// Wait for completion until one shared draw deadline, then return a snapshot.
    pub fn wait_until(&self, deadline: Instant) -> PipelineStatus<P> {
        if self.0.result.get().is_some() {
            return self.poll();
        }
        let guard = self.0.waiter.lock().unwrap_or_else(|p| p.into_inner());
        let (_guard, _) = self
            .0
            .completed
            .wait_timeout_while(
                guard,
                deadline.saturating_duration_since(Instant::now()),
                |_| self.0.result.get().is_none(),
            )
            .unwrap_or_else(|p| p.into_inner());
        self.poll()
    }

    /// Return the current phase without waiting for compilation.
    pub fn poll(&self) -> PipelineStatus<P> {
        #[cfg(feature = "draw-metrics")]
        let _span = crate::draw_metrics::PipelineSpan::new(10);
        if let Some(result) = self.0.result.get() {
            return match result {
                Ok(pipeline) => PipelineStatus::Ready(pipeline.clone()),
                Err(error) => PipelineStatus::Failed(error.clone()),
            };
        }
        match self.0.phase.load(Ordering::Acquire) {
            0 => PipelineStatus::Queued,
            _ => PipelineStatus::Compiling,
        }
    }
}

/// Failure to enqueue a new owned compilation request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestError {
    /// The bounded job channel has no available slot.
    QueueFull,
    /// The worker channel has disconnected.
    Stopped,
}

struct CompileJob<K, P> {
    program: K,
    state: Arc<RequestState<P>>,
}

/// Bounded compiler pool. Construction and shutdown belong outside frame recording.
/// Nonblocking requests and polling do not wait for compilation.
pub(crate) struct AsyncPipelines<K, P> {
    requests: HashMap<K, PipelineRequest<P>>,
    sender: Option<mpsc::SyncSender<CompileJob<K, P>>>,
    workers: Vec<JoinHandle<()>>,
    pub(crate) stats: CacheStats,
}

impl<K: Eq + std::hash::Hash + Clone + Send + 'static, P: Send + Sync + 'static>
    AsyncPipelines<K, P>
{
    /// Converts a populated cache into a worker service, preserving ready pipelines.
    /// Queue capacity bounds waiting work; worker_count bounds active work.
    pub fn new(
        pipelines: HashMap<K, Arc<P>>,
        stats: CacheStats,
        worker_count: usize,
        queue_capacity: usize,
        compile: impl Fn(&K) -> Result<Arc<P>, String> + Send + Sync + 'static,
    ) -> Result<Self, String> {
        if worker_count == 0 || queue_capacity == 0 {
            return Err("compiler pool requires workers and queue capacity".into());
        }
        let (sender, receiver) = mpsc::sync_channel::<CompileJob<K, P>>(queue_capacity);
        let receiver = Arc::new(Mutex::new(receiver));
        let compiler = Arc::new(compile);
        let mut result = Self {
            requests: pipelines
                .into_iter()
                .map(|(program, pipeline)| {
                    (
                        program,
                        PipelineRequest(Arc::new(RequestState {
                            completed: Condvar::new(),
                            phase: AtomicU8::new(2),
                            result: OnceLock::from(Ok(pipeline)),
                            waiter: Mutex::new(()),
                        })),
                    )
                })
                .collect(),
            sender: Some(sender),
            workers: Vec::new(),
            stats,
        };
        for index in 0..worker_count {
            let receiver = receiver.clone();
            let compiler = compiler.clone();
            let worker = thread::Builder::new()
                .name(format!("pipeline-{index}"))
                .spawn(move || loop {
                    let job = {
                        let receiver = receiver.lock().unwrap();
                        receiver.recv()
                    };
                    let Ok(job) = job else { break };
                    job.state.phase.store(1, Ordering::Release);
                    let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        compiler(&job.program)
                    }))
                    .unwrap_or_else(|_| Err("pipeline compiler panicked".into()));
                    let _guard = job.state.waiter.lock().unwrap();
                    assert!(job.state.result.set(compiled).is_ok());
                    job.state.phase.store(2, Ordering::Release);
                    job.state.completed.notify_all();
                })
                .map_err(|error| format!("start pipeline worker: {error}"))?;
            result.workers.push(worker);
        }
        Ok(result)
    }

    /// Duplicate requests share one result. QueueFull leaves no cache entry.
    /// Failed requests require explicit retry.
    pub fn request(&mut self, program: K) -> Result<PipelineRequest<P>, RequestError> {
        self.enqueue(program, false)
    }

    /// Wait for queue admission. Compilation remains owned by the workers.
    #[cfg(feature = "vulkan")]
    pub fn request_blocking(&mut self, program: K) -> Result<PipelineRequest<P>, RequestError> {
        self.enqueue(program, true)
    }

    fn enqueue(&mut self, program: K, blocking: bool) -> Result<PipelineRequest<P>, RequestError> {
        if let Some(request) = self.requests.get(&program) {
            self.stats.hits += 1;
            return Ok(request.clone());
        }
        let state = Arc::new(RequestState {
            completed: Condvar::new(),
            phase: AtomicU8::new(0),
            result: OnceLock::new(),
            waiter: Mutex::new(()),
        });
        let sender = self.sender.as_ref().ok_or(RequestError::Stopped)?;
        let job = CompileJob {
            program: program.clone(),
            state: state.clone(),
        };
        if blocking {
            sender.send(job).map_err(|_| RequestError::Stopped)?;
        } else {
            sender.try_send(job).map_err(|error| match error {
                mpsc::TrySendError::Full(_) => RequestError::QueueFull,
                mpsc::TrySendError::Disconnected(_) => RequestError::Stopped,
            })?;
        }
        let request = PipelineRequest(state);
        self.requests.insert(program, request.clone());
        self.stats.misses += 1;
        Ok(request)
    }

    /// Forget only a failed request. Existing handles retain their result.
    #[cfg(feature = "vulkan")]
    pub fn retry_failed(&mut self, program: &K) -> bool {
        if self
            .requests
            .get(program)
            .is_some_and(|request| matches!(request.poll(), PipelineStatus::Failed(_)))
        {
            self.requests.remove(program);
            true
        } else {
            false
        }
    }

    pub(crate) fn pending(&self) -> usize {
        self.requests
            .values()
            .filter(|request| {
                matches!(
                    request.poll(),
                    PipelineStatus::Queued | PipelineStatus::Compiling
                )
            })
            .count()
    }
}

impl<K, P> Drop for AsyncPipelines<K, P> {
    fn drop(&mut self) {
        self.sender.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl<P> std::fmt::Debug for PipelineRequest<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineRequest")
            .field("phase", &self.0.phase.load(Ordering::Acquire))
            .finish()
    }
}
impl<P> PartialEq for PipelineRequest<P> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl<P> Eq for PipelineRequest<P> {}

#[cfg(test)]
mod deadline_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn one_deadline_bounds_waiting_and_completion_wakes_waiters() {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let mut pool = AsyncPipelines::new(
            HashMap::new(),
            CacheStats::default(),
            1,
            1,
            move |key: &u32| {
                started_tx.send(()).unwrap();
                release_rx.lock().unwrap().recv().unwrap();
                Ok(Arc::new(*key))
            },
        )
        .unwrap();
        let first = pool.request(7).unwrap();
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let second = pool.request(8).unwrap();
        assert_eq!(pool.request(9).unwrap_err(), RequestError::QueueFull);
        let start = Instant::now();
        let deadline = start + Duration::from_millis(5);
        assert!(matches!(
            first.wait_until(deadline),
            PipelineStatus::Compiling
        ));
        assert!(matches!(
            second.wait_until(deadline),
            PipelineStatus::Queued
        ));
        assert!(
            start.elapsed() < Duration::from_millis(250),
            "wait exceeded shared deadline"
        );
        release_tx.send(()).unwrap();
        assert!(
            matches!(first.wait_until(Instant::now() + Duration::from_secs(5)), PipelineStatus::Ready(value) if *value == 7)
        );
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        release_tx.send(()).unwrap();
        assert!(
            matches!(second.wait_until(Instant::now() + Duration::from_secs(5)), PipelineStatus::Ready(value) if *value == 8)
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn blocking_admission_preserves_work_when_the_queue_is_full() {
        use std::time::Duration;
        let (started, receive) = mpsc::channel();
        let (release, resume) = mpsc::channel();
        let resume = Mutex::new(resume);
        let mut pool = AsyncPipelines::new(
            HashMap::new(),
            CacheStats::default(),
            1,
            1,
            move |key: &u32| {
                started.send(*key).unwrap();
                resume.lock().unwrap().recv().unwrap();
                Ok(Arc::new(*key))
            },
        )
        .unwrap();
        let first = pool.request(1).unwrap();
        assert_eq!(receive.recv_timeout(Duration::from_secs(5)).unwrap(), 1);
        let second = pool.request(2).unwrap();
        assert_eq!(pool.request(3).unwrap_err(), RequestError::QueueFull);
        let third = std::thread::scope(|scope| {
            let admission = scope.spawn(|| pool.request_blocking(3).unwrap());
            release.send(()).unwrap();
            assert_eq!(receive.recv_timeout(Duration::from_secs(5)).unwrap(), 2);
            let request = admission.join().unwrap();
            release.send(()).unwrap();
            assert_eq!(receive.recv_timeout(Duration::from_secs(5)).unwrap(), 3);
            release.send(()).unwrap();
            request
        });
        for (expected, request) in [first, second, third].iter().enumerate() {
            assert!(
                matches!(request.wait(), PipelineStatus::Ready(value) if *value == expected as u32 + 1)
            );
        }
        assert_eq!(pool.pending(), 0);
        assert_eq!(pool.stats.misses, 3);
    }

    #[test]
    fn completed_requests_ignore_the_waiter_lock() {
        let ready = PipelineRequest::ready(Arc::new(7));
        let _guard = ready.0.waiter.lock().unwrap();
        assert!(matches!(ready.poll(), PipelineStatus::Ready(value) if *value == 7));
        assert!(matches!(ready.wait(), PipelineStatus::Ready(value) if *value == 7));
        assert!(
            matches!(ready.wait_until(Instant::now()), PipelineStatus::Ready(value) if *value == 7)
        );
        let failed = PipelineRequest::<u32>::completed(Err("retained failure".into()));
        let _guard = failed.0.waiter.lock().unwrap();
        assert!(
            matches!(failed.poll(), PipelineStatus::Failed(error) if error == "retained failure")
        );
    }

    #[test]
    fn waiting_preserves_failed_results() {
        let mut pool =
            AsyncPipelines::<u32, u32>::new(HashMap::new(), CacheStats::default(), 1, 1, |_| {
                Err("synthetic failure".into())
            })
            .unwrap();
        let request = pool.request(1).unwrap();
        assert!(
            matches!(request.wait_until(Instant::now() + Duration::from_secs(5)), PipelineStatus::Failed(error) if error == "synthetic failure")
        );
        assert!(matches!(request.poll(), PipelineStatus::Failed(_)));
        assert!(matches!(request.wait(), PipelineStatus::Failed(_)));
    }
}
