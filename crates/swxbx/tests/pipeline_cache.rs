//! Host cache and scheduling checks.
#![cfg(feature = "vulkan")]

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    time::{Duration, Instant},
};
use swxbx::{
    gpu::{
        pipelines::{
            AsyncComputePipelines, ComputePipeline, ComputePipelines, PipelineRequest,
            PipelineStatus, RequestError, TranslationIdentity,
        },
        Context,
    },
    ShaderTranslator,
};

const PROGRAM: &[u8] = b"original host fixture header and code";

fn directory(name: &str) -> PathBuf {
    let path = env::temp_dir().join(format!("pipeline-{name}-{}", std::process::id()));
    if path.exists() {
        fs::remove_dir_all(&path).unwrap();
    }
    fs::create_dir_all(&path).unwrap();
    path
}

fn shader(directory: &Path) -> Vec<u32> {
    let path = directory.join("cache.spv");
    let status = Command::new("glslangValidator")
        .args(["-V", "--target-env", "vulkan1.2", "-o"])
        .arg(&path)
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/shaders/pipeline_cache.comp"))
        .output()
        .expect("glslangValidator must be installed");
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stdout)
    );
    let validated = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2"])
        .arg(&path)
        .status()
        .expect("spirv-val must be installed");
    assert!(validated.success());
    words(&path)
}

fn words(path: &Path) -> Vec<u32> {
    let bytes = fs::read(path).unwrap();
    let (words, remainder) = bytes.as_chunks::<4>();
    assert!(remainder.is_empty());
    words.iter().map(|&word| u32::from_le_bytes(word)).collect()
}

struct Translator {
    words: Vec<u32>,
    calls: AtomicUsize,
}

impl ShaderTranslator for Translator {
    fn translate(&self, _: u32, _: &[u8]) -> Result<Vec<u32>, String> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(self.words.clone())
    }
}

fn identity() -> TranslationIdentity {
    TranslationIdentity {
        version: "host-test-1".into(),
        configuration: "default".into(),
    }
}

fn child(root: &Path, mode: &str) -> Command {
    let mut command = Command::new(env::current_exe().unwrap());
    command
        .args(["--exact", "cache_process", "--ignored", "--nocapture"])
        .env("SWXBX_CACHE_TEST_ROOT", root)
        .env("SWXBX_CACHE_TEST_MODE", mode);
    command
}

fn run_child(root: &Path, mode: &str) {
    let output = child(root, mode).output().unwrap();
    assert!(
        output.status.success(),
        "child {mode}: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn files(root: &Path, extension: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for namespace in fs::read_dir(root.join("store")).unwrap() {
        for file in fs::read_dir(namespace.unwrap().path()).unwrap() {
            let path = file.unwrap().path();
            if path.extension().is_some_and(|value| value == extension) {
                paths.push(path);
            }
        }
    }
    paths
}

#[test]
#[ignore = "subprocess helper, invoked by the restart proof"]
fn cache_process() {
    let Some(root) = env::var_os("SWXBX_CACHE_TEST_ROOT").map(PathBuf::from) else {
        return;
    };
    let mode = env::var("SWXBX_CACHE_TEST_MODE").unwrap();
    let context = Arc::new(Context::new().expect("Vulkan compute context"));
    let translator = Arc::new(Translator {
        words: words(&root.join("cache.spv")),
        calls: AtomicUsize::new(0),
    });
    let mut cache = ComputePipelines::persistent(
        &context,
        translator.clone(),
        &root.join("store"),
        &identity(),
    )
    .unwrap();
    let driver_loaded = cache.persistence_stats().driver_cache_loaded;
    let first = cache.get_or_compile(PROGRAM).unwrap();
    let second = cache.get_or_compile(PROGRAM).unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    let expected_calls = usize::from(
        mode == "cold" || mode == "translation-corrupt" || mode == "translation-invalid",
    );
    assert_eq!(translator.calls.load(Ordering::Relaxed), expected_calls);
    assert_eq!(
        cache.persistence_stats().translation_hits,
        u64::from(expected_calls == 0)
    );
    assert_eq!(driver_loaded, mode != "cold" && mode != "driver-corrupt");
    let diagnostics = cache.take_diagnostics();
    if mode.ends_with("corrupt") || mode.ends_with("invalid") {
        assert!(
            !diagnostics.is_empty(),
            "corruption must produce a diagnostic"
        );
    } else {
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
    }
}

#[test]
#[ignore = "requires Vulkan, glslangValidator and spirv-val; never skips"]
fn disk_cache_survives_restarts_recovers_corruption_and_concurrent_processes() {
    let root = directory("restart");
    shader(&root);
    run_child(&root, "cold");
    run_child(&root, "hit");
    let translation = files(&root, "spirv").pop().unwrap();
    fs::write(&translation, b"truncated").unwrap();
    run_child(&root, "translation-corrupt");
    run_child(&root, "hit");
    // Keep a valid checksum but break the SPIR-V header. Reflection rejects it
    // before Vulkan sees the module, and fresh translation repairs the entry.
    let mut bytes = fs::read(&translation).unwrap();
    let words_offset = 84 + 8 + PROGRAM.len();
    bytes[words_offset] ^= 1;
    let checksum = *blake3::hash(&bytes[84..]).as_bytes();
    bytes[52..84].copy_from_slice(&checksum);
    fs::write(&translation, bytes).unwrap();
    run_child(&root, "translation-invalid");
    run_child(&root, "hit");
    let driver = files(&root, "bin").pop().unwrap();
    let mut bytes = fs::read(&driver).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(&driver, bytes).unwrap();
    run_child(&root, "driver-corrupt");
    run_child(&root, "hit");

    // Both children load and publish the same destination concurrently.
    let mut first = child(&root, "hit").spawn().unwrap();
    let mut second = child(&root, "hit").spawn().unwrap();
    assert!(first.wait().unwrap().success());
    assert!(second.wait().unwrap().success());
    run_child(&root, "hit");
    fs::remove_dir_all(root).unwrap();
}

struct GatedTranslator {
    words: Vec<u32>,
    calls: AtomicUsize,
    entered: mpsc::Sender<()>,
    gate: Arc<(Mutex<bool>, Condvar)>,
}

impl ShaderTranslator for GatedTranslator {
    fn translate(&self, _: u32, program: &[u8]) -> Result<Vec<u32>, String> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.entered.send(()).unwrap();
        let (lock, wake) = &*self.gate;
        let _guard = wake
            .wait_while(lock.lock().unwrap(), |released| !*released)
            .unwrap();
        drop(_guard);
        match program {
            b"failure" => Err("test translation failure".into()),
            b"panic" => panic!("test translator panic"),
            _ => Ok(self.words.clone()),
        }
    }
}

fn ready(request: &PipelineRequest) -> Arc<ComputePipeline> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match request.poll() {
            PipelineStatus::Ready(pipeline) => return pipeline,
            PipelineStatus::Failed(error) => panic!("{error}"),
            _ => assert!(Instant::now() < deadline, "compilation timed out"),
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn failed(request: &PipelineRequest) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match request.poll() {
            PipelineStatus::Failed(_) => return,
            PipelineStatus::Ready(_) => panic!("expected failed compilation"),
            _ => assert!(Instant::now() < deadline, "compilation timed out"),
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
#[ignore = "requires Vulkan, glslangValidator and spirv-val; never skips"]
fn async_workers_deduplicate_skip_pending_reuse_and_retry() {
    let root = directory("async");
    let context = Arc::new(Context::new().expect("Vulkan compute context"));
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let (entered, receiver) = mpsc::channel();
    let translator = Arc::new(GatedTranslator {
        words: shader(&root),
        calls: AtomicUsize::new(0),
        entered,
        gate: gate.clone(),
    });
    let cache = ComputePipelines::persistent(
        &context,
        translator.clone(),
        &root.join("store"),
        &identity(),
    )
    .unwrap();
    let mut pool = AsyncComputePipelines::new(cache, 2, 1).unwrap();
    let first = pool.request(b"first").unwrap();
    receiver.recv_timeout(Duration::from_secs(10)).unwrap();
    let duplicate = pool.request(b"first").unwrap();
    let replaced = pool.request(b"replacement").unwrap();
    receiver.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(matches!(first.poll(), PipelineStatus::Compiling));
    assert!(matches!(duplicate.poll(), PipelineStatus::Compiling));
    assert!(matches!(replaced.poll(), PipelineStatus::Compiling));
    let queued = pool.request(b"queued").unwrap();
    assert!(matches!(queued.poll(), PipelineStatus::Queued));
    assert!(matches!(
        pool.request(b"overflow"),
        Err(RequestError::QueueFull)
    ));
    assert_eq!(translator.calls.load(Ordering::Relaxed), 2);

    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    let pipeline = ready(&first);
    assert!(Arc::ptr_eq(&pipeline, &ready(&duplicate)));
    assert!(Arc::ptr_eq(
        &pipeline,
        &ready(&pool.request(b"first").unwrap())
    ));
    assert!(!Arc::ptr_eq(&pipeline, &ready(&replaced)));
    ready(&queued);
    let overflow = pool.request(b"overflow").unwrap();
    ready(&overflow);
    assert_eq!(translator.calls.load(Ordering::Relaxed), 4);
    let failure = pool.request(b"failure").unwrap();
    failed(&failure);
    assert!(pool.retry_failed(b"failure"));
    let retry = pool.request(b"failure").unwrap();
    failed(&retry);
    let panicked = pool.request(b"panic").unwrap();
    failed(&panicked);
    ready(&pool.request(b"after-panic").unwrap());
    assert!(!pool.retry_failed(b"first"));
    assert!(pool.take_diagnostics().is_empty());
    drop(pool);
    // Disk translation is used by an asynchronous request in a new service.
    let second_translator = Arc::new(Translator {
        words: translator.words.clone(),
        calls: AtomicUsize::new(0),
    });
    let cache = ComputePipelines::persistent(
        &context,
        second_translator.clone(),
        &root.join("store"),
        &identity(),
    )
    .unwrap();
    let mut pool = AsyncComputePipelines::new(cache, 1, 1).unwrap();
    ready(&pool.request(b"first").unwrap());
    assert_eq!(second_translator.calls.load(Ordering::Relaxed), 0);
    assert_eq!(pool.persistence_stats().translation_hits, 1);
    drop(pool);
    assert!(pipeline.bindings().is_empty());
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "requires Vulkan, glslangValidator and spirv-val; never skips"]
fn unavailable_cache_directory_preserves_compilation() {
    let root = directory("io-error");
    let context = Arc::new(Context::new().expect("Vulkan compute context"));
    let translator = Arc::new(Translator {
        words: shader(&root),
        calls: AtomicUsize::new(0),
    });
    let path = root.join("not-a-directory");
    fs::write(&path, b"occupied").unwrap();
    let cache =
        ComputePipelines::persistent(&context, translator.clone(), &path, &identity()).unwrap();
    let mut pool = AsyncComputePipelines::new(cache, 1, 1).unwrap();
    ready(&pool.request(PROGRAM).unwrap());
    assert_eq!(translator.calls.load(Ordering::Relaxed), 1);
    assert!(!pool.take_diagnostics().is_empty());
    drop(pool);
    fs::remove_dir_all(root).unwrap();
}
