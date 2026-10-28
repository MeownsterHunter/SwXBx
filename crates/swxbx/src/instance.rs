//! An instance: what a host creates once and forwards a program's graphics
//! calls to.

use crate::api::{self, Handler, Objects};
use crate::functions::{self, FunctionId};
#[cfg(feature = "vulkan")]
use crate::gpu::Backend;
use crate::observe::{CallSnapshot, FunctionShape};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::{c_char, c_void};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Host shader-stage identifier; registration uses the unknown stage until established.
pub type ShaderStage = u32;
/// Stage token used when observed shader records do not establish a stage.
pub const SHADER_STAGE_UNKNOWN: ShaderStage = 0;

#[derive(Clone, Copy)]
pub(crate) struct PendingShader {
    /// Registration index whose deferred shader bytes must be retried.
    pub record_index: usize,
    /// Resolved program-memory shader address.
    pub address: u64,
    /// GPU address of the same program, which places its constant data.
    pub gpu_address: u64,
    /// Readable program bytes from the address to the end of its pool.
    pub limit: usize,
    /// Number of failed deferred byte reads.
    pub failures: u8,
}

/// Thread-safe host hook for translating bounded shader bytes into SPIR-V.
pub trait ShaderTranslator: Send + Sync {
    /// Translate supplied shader bytes into SPIR-V words or a diagnostic.
    fn translate(&self, stage: ShaderStage, code: &[u8]) -> Result<Vec<u32>, String>;

    /// Return the exact runtime identity from the translator's public cache API.
    /// None means required translation state is absent. Source: the public contract.
    fn translation_cache_key(
        &self,
        _stage: ShaderStage,
        _code: &[u8],
        _context: &crate::startup_cache::TranslationContext,
    ) -> Result<Option<String>, String> {
        Err("translator cache identity is unavailable".into())
    }

    /// Encode an opaque disk record for the supplied translation context.
    /// None disables persistence. Errors are diagnostics and preserve translation.
    /// Records must retain every execution requirement. Evidence: the public contract.
    fn encode_cache_record(
        &self,
        _shader: &crate::startup_cache::TranslatedShader,
        _context: &crate::startup_cache::TranslationContext,
    ) -> Result<Option<Vec<u8>>, String> {
        Ok(None)
    }

    /// Validate an opaque record against the key and translator identity.
    /// None means disabled, stale, or incompatible. Malformed records return
    /// a diagnostic. Results restore all requirements and the supplied key.
    /// Evidence: the public contract.
    fn decode_cache_record(
        &self,
        _key: &str,
        _bytes: &[u8],
    ) -> Result<Option<crate::startup_cache::TranslatedShader>, String> {
        Ok(None)
    }

    /// What translating `code` needs beyond the code and the instance's
    /// context. Code that needs the draw's input primitive (a geometry
    /// stage) is translated at each draw, with
    /// [`TranslationContext::geometry_input`](crate::startup_cache::TranslationContext::geometry_input)
    /// stating that draw's primitive; other code translates when it is
    /// registered.
    fn translation_needs(
        &self,
        _stage: ShaderStage,
        _code: &[u8],
    ) -> crate::startup_cache::TranslationNeeds {
        crate::startup_cache::TranslationNeeds::Code
    }

    /// Adapters with state or subgroup requirements must override this method.
    fn translate_for_context(
        &self,
        stage: ShaderStage,
        code: &[u8],
        context: &crate::startup_cache::TranslationContext,
    ) -> Result<crate::startup_cache::TranslatedShader, String> {
        if *context != crate::startup_cache::TranslationContext::default() {
            return Err("translator context is unsupported".into());
        }
        Ok(crate::startup_cache::TranslatedShader {
            key: String::new(),
            words: self.translate(stage, code)?,
            requires_subgroup_size_32: false,
            requires_demote_to_helper_invocation: false,
            requires_maximal_reconvergence: false,
            surfaces: Vec::new(),
            textures: Vec::new(),
        })
    }
}

/// What the host provides. The program's memory is the host's to manage, so
/// the library reads and writes it through these callbacks.
///
/// Both callbacks return 0 on success and any other value when the range is
/// not accessible. They are `unsafe` to call: the pointer must be valid for
/// `size` bytes.
///
/// The host's side of the contract, which `Instance::with_host` takes on
/// trust: `user` and both callbacks stay usable for the life of the
/// instance, and may be used from several threads at once, because a program
/// can call the graphics API from any of its threads. A non-null `vulkan`
/// pointer follows the lifetime and callback contract in docs/host-interface.md.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Host {
    /// Passed back as the first argument of every callback.
    pub user: *mut c_void,
    /// Copy a program-memory range into out; return zero on success, nonzero on failure.
    pub read_memory: Option<
        unsafe extern "C" fn(user: *mut c_void, address: u64, out: *mut u8, size: u64) -> i32,
    >,
    /// Copy data into a program-memory range; return zero on success, nonzero on failure.
    pub write_memory: Option<
        unsafe extern "C" fn(user: *mut c_void, address: u64, data: *const u8, size: u64) -> i32,
    >,
    /// Receive borrowed RGBA frame bytes; copy them before returning to retain the frame.
    pub present: Option<
        unsafe extern "C" fn(
            user: *mut c_void,
            window_object: u64,
            width: u32,
            height: u32,
            rgba: *const u8,
            stride_bytes: u64,
        ),
    >,
    /// The output scale. Values below 1 are treated as 1.0.
    pub render_scale: f32,
    /// Waits for one vertical blank. RGBA callback presentation calls it once
    /// per blank of the window's present interval before each delivery.
    pub wait_vblank: Option<unsafe extern "C" fn(user: *mut c_void)>,
    /// Optional Vulkan presentation contract (version 6). See docs/host-interface.md.
    pub vulkan: *const HostVulkan,
}

/// Host-owned platform integration. Vulkan handles cross this C boundary as u64.
/// Source: the public contract.-presentation.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HostVulkan {
    /// Number of NUL-terminated names in extensions; native setup accepts one through 64.
    pub extension_count: u32,
    /// Borrowed array of Vulkan instance extension names, including the surface extension.
    pub extensions: *const *const c_char,
    /// Return a surface for the supplied instance and native handle; zero means failure.
    /// The library owns and destroys the returned surface.
    pub create_surface: unsafe extern "C" fn(
        user: *mut c_void,
        instance: u64,
        window_object: u64,
        native_window: u64,
    ) -> u64,
    /// Write the current drawable width and height; zero suspends presentation.
    pub drawable_size: unsafe extern "C" fn(
        user: *mut c_void,
        window_object: u64,
        width: *mut u32,
        height: *mut u32,
    ),
    /// Write how the presented image fills the drawable, at every present.
    /// There is no default: the library hands in `HostScaling::UNSTATED`, and
    /// a present whose scaling is unstated or invalid is refused by name.
    /// Source: the public contract.
    pub scaling: unsafe extern "C" fn(
        user: *mut c_void,
        window_object: u64,
        scaling: *mut crate::presentation::HostScaling,
    ),
}

impl Host {
    /// Ask the native host how window `window`'s presented image fills its
    /// drawable. Only for a host whose `vulkan` is non-null.
    #[cfg(feature = "vulkan")]
    pub(crate) fn scaling(&self, window: u64) -> Result<crate::presentation::Scaling, String> {
        let mut stated = crate::presentation::HostScaling::UNSTATED;
        // SAFETY: `vulkan` is non-null and, with its callbacks and `user`,
        // valid for the instance's life (the `with_host` contract).
        unsafe { ((*self.vulkan).scaling)(self.user, window, &mut stated) };
        stated.parse()
    }
}

// SAFETY: a `Host` only reaches an instance through `Instance::with_host`,
// whose caller guarantees that `user` and the callbacks may be shared between
// threads. Native integration pointers follow the same documented contract.
unsafe impl Send for Host {}
unsafe impl Sync for Host {}

/// The registers that carry a call's arguments and results under the
/// program's standard calling convention: eight integer registers, eight
/// floating-point registers (their low 64 bits) and the stack pointer, for
/// arguments that did not fit in registers.
///
/// The host fills this in before a call. On return `x[0]`, `x[1]` and `d[0]`
/// hold the results.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Registers {
    /// Eight integer argument registers; the first two also carry results.
    pub x: [u64; 8],
    /// Low 64 bits of eight floating-point argument registers; the first also carries a result.
    pub d: [u64; 8],
    /// Program stack pointer for arguments beyond the register set.
    pub sp: u64,
}

/// Outcome of a call into the library.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The function ran.
    Ok = 0,
    /// The function has no behaviour yet. The call was counted and the
    /// result registers were set to zero.
    Unimplemented = 1,
    /// The function id is outside the table.
    BadFunction = 2,
    /// A pointer argument was null, or a file could not be written.
    BadArgument = 3,
    /// The library panicked while servicing the call.
    InternalError = 4,
}

thread_local! {
    /// The call this thread last reported and what its address arguments
    /// pointed to, until the matching return is reported.
    static PENDING: Cell<Option<(u32, CallSnapshot, Registers)>> = const { Cell::new(None) };
}

/// Host connection, object records, observation counters, and optional execution services.
#[doc(hidden)]
pub struct InstanceState {
    host: Option<Host>,
    pub(crate) presentation: Mutex<crate::presentation::Presentation>,
    /// Whether a batch stops or skips at a command swxbx cannot run.
    pub(crate) unsupported_commands: crate::unsupported::UnsupportedCommands,
    /// The first batch since the last present that stopped before its end:
    /// the command and the reason. The next present refuses while it holds.
    stopped_batch: Mutex<Option<String>>,
    /// Which guest counter token selects which counter. Source: the public contract.
    counters: Mutex<crate::counters::CounterContract>,
    /// swxbx's record of the program's objects.
    pub objects: Objects,
    handlers: Vec<Option<Handler>>,
    /// A rising count that stands in for the graphics processor's clock in
    /// counter reports.
    requested: Vec<AtomicBool>,
    calls: Vec<AtomicU64>,
    shapes: Vec<Mutex<FunctionShape>>,
    /// Names the program asked for that are not in the table, with how often.
    unknown_requests: Mutex<BTreeMap<String, u64>>,
    shader_translator: Mutex<Option<Arc<dyn ShaderTranslator>>>,
    /// The translation hook each program registered under, by program
    /// handle. Its later translations (a geometry stage per primitive, a
    /// stage per texel image) use it, whatever hook is installed by then.
    program_translators: Mutex<HashMap<u64, Arc<dyn ShaderTranslator>>>,
    pub(crate) startup_cache: Mutex<Option<crate::startup_cache::StartupCache>>,
    shader_translation_enabled: AtomicBool,
    shader_dump_directory: Mutex<Option<PathBuf>>,
    shader_dump_sequence: AtomicU64,
    shader_dump_hashes: Mutex<BTreeSet<u64>>,
    successful_shader_translations: AtomicU64,
    distinct_shader_outputs: Mutex<BTreeSet<u64>>,
    shader_translation_errors: Mutex<BTreeMap<String, u64>>,
    /// Stage translations specialized by the textures draws bind.
    #[cfg(feature = "vulkan")]
    pub(crate) specializations: Mutex<crate::api::specialization::Specializations>,
    pending_shaders: Mutex<HashMap<u64, Vec<PendingShader>>>,
    shader_late_reads: Mutex<BTreeMap<&'static str, u64>>,
    shader_zero_header: AtomicU64,
    shader_header_without_code: AtomicU64,
    #[cfg(feature = "vulkan")]
    pub(crate) gpu: Mutex<Option<Backend>>,
    #[cfg(feature = "vulkan")]
    host_writes: Mutex<crate::dirty_ranges::Ranges>,
    #[cfg(feature = "vulkan")]
    recording_strides: arc_swap::ArcSwap<[Option<u64>; 2]>,
    /// The backend's copy decoder and image rules, for the recording call.
    #[cfg(feature = "vulkan")]
    pub(crate) copy_rules: arc_swap::ArcSwap<crate::api::CopyRules>,
}

impl InstanceState {
    /// Remember where a batch stopped, unless an earlier batch since the
    /// last present already did.
    pub(crate) fn batch_stopped(&self, at: String) {
        self.stopped_batch
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get_or_insert(at);
    }

    /// Where the first batch since the last present stopped, if one did;
    /// the next frame starts with none.
    pub(crate) fn take_stopped_batch(&self) -> Option<String> {
        self.stopped_batch
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
    }

    /// An instance with no host. It can record requests and calls, which is
    /// all the library does so far.
    pub fn new() -> Self {
        Self::build(None)
    }

    /// An instance that uses `host` to reach the program's memory.
    ///
    /// # Safety
    /// `host.user` and the callbacks must stay usable for the life of the
    /// instance and must tolerate being used from several threads at once.
    /// A non-null `host.vulkan` points to a valid HostVulkan for that lifetime.
    /// Its extension array and strings are valid during this call. Native windows
    /// stay live until finalization, and its callbacks must not reenter swxbx.
    pub unsafe fn with_host(host: Host) -> Self {
        Self::build(Some(host))
    }

    fn build(host: Option<Host>) -> Self {
        let count = functions::count();
        Self {
            host,
            presentation: Mutex::default(),
            unsupported_commands: crate::unsupported::UnsupportedCommands::from_env(),
            stopped_batch: Mutex::default(),
            counters: Mutex::default(),
            objects: Objects::new(),
            handlers: functions::all()
                .map(|(_, name)| api::handler(name))
                .collect(),
            requested: (0..count).map(|_| AtomicBool::new(false)).collect(),
            calls: (0..count).map(|_| AtomicU64::new(0)).collect(),
            shapes: (0..count).map(|_| Mutex::default()).collect(),
            unknown_requests: Mutex::new(BTreeMap::new()),
            shader_translator: Mutex::new(None),
            program_translators: Mutex::new(HashMap::new()),
            startup_cache: Mutex::new(None),
            shader_translation_enabled: AtomicBool::new(false),
            shader_dump_directory: Mutex::new(None),
            shader_dump_sequence: AtomicU64::new(0),
            shader_dump_hashes: Mutex::new(BTreeSet::new()),
            successful_shader_translations: AtomicU64::new(0),
            distinct_shader_outputs: Mutex::new(BTreeSet::new()),
            shader_translation_errors: Mutex::new(BTreeMap::new()),
            #[cfg(feature = "vulkan")]
            specializations: Mutex::default(),
            pending_shaders: Mutex::new(HashMap::new()),
            shader_late_reads: Mutex::new(BTreeMap::new()),
            shader_zero_header: AtomicU64::new(0),
            shader_header_without_code: AtomicU64::new(0),
            #[cfg(feature = "vulkan")]
            gpu: Mutex::new(host.and_then(|h| Backend::with_host(&h))),
            #[cfg(feature = "vulkan")]
            host_writes: Mutex::default(),
            #[cfg(feature = "vulkan")]
            recording_strides: arc_swap::ArcSwap::from_pointee([None, None]),
            #[cfg(feature = "vulkan")]
            copy_rules: arc_swap::ArcSwap::default(),
        }
    }

    /// Load translated modules before registration and start bounded workers.
    /// Cache read errors are diagnostic misses. Worker configuration errors fail.
    pub fn with_startup_cache(
        config: crate::startup_cache::StartupCacheConfig,
        translator: Arc<dyn ShaderTranslator>,
    ) -> Result<Self, String> {
        let instance = Self::new();
        instance.configure_startup_cache(config, translator)?;
        Ok(instance)
    }

    /// Construct a hosted instance and load its startup caches.
    ///
    /// # Safety
    /// The same callback and lifetime contract as `with_host` applies.
    pub unsafe fn with_host_startup_cache(
        host: Host,
        config: crate::startup_cache::StartupCacheConfig,
        translator: Arc<dyn ShaderTranslator>,
    ) -> Result<Self, String> {
        let instance = Self::build(Some(host));
        instance.configure_startup_cache(config, translator)?;
        Ok(instance)
    }

    /// Configure outside calls or submission. Exclusive access prevents replacing
    /// the translator underneath current owned requests. Existing objects must
    /// be registered again if translation inputs change.
    pub fn configure_startup_cache(
        &self,
        config: crate::startup_cache::StartupCacheConfig,
        translator: Arc<dyn ShaderTranslator>,
    ) -> Result<(), String> {
        #[cfg(feature = "vulkan")]
        let directory = config.directory.clone();
        #[cfg(feature = "vulkan")]
        let workers = config.workers;
        #[cfg(feature = "vulkan")]
        let capacity = config.queue_capacity;
        #[cfg(feature = "vulkan")]
        let identity = crate::gpu::pipelines::TranslationIdentity {
            version: "startup graphics interface 1".into(),
            configuration: format!("{:?}", config.context),
        };
        let cache = crate::startup_cache::StartupCache::open(config, translator.clone())?;
        #[cfg(feature = "vulkan")]
        let mut cache = cache;
        #[cfg(feature = "vulkan")]
        if let Some(backend) = self.gpu.lock().unwrap().as_mut() {
            let mut graphics = crate::gpu::graphics::GraphicsPipelines::persistent(
                backend.context(),
                &directory,
                &identity,
                workers,
                capacity,
            )?;
            cache.schedule(&mut graphics);
            backend.graphics = graphics;
        }
        *self.startup_cache.lock().unwrap() = Some(cache);
        *self.shader_translator.lock().unwrap() = Some(translator);
        self.shader_translation_enabled
            .store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Select inputs for subsequent registrations. Existing requests keep their
    /// original context; register a program again to translate a different variant.
    pub fn set_shader_translation_context(
        &self,
        context: crate::startup_cache::TranslationContext,
    ) -> bool {
        let mut cache = self.startup_cache.lock().unwrap();
        let Some(cache) = cache.as_mut() else {
            return false;
        };
        cache.set_context(context);
        true
    }

    /// The context registrations currently translate in; the default when no
    /// startup cache is configured.
    #[cfg(feature = "vulkan")]
    pub(crate) fn translation_context(&self) -> crate::startup_cache::TranslationContext {
        self.startup_cache
            .lock()
            .unwrap()
            .as_ref()
            .map(|cache| cache.context())
            .unwrap_or_default()
    }

    /// Return startup cache counters, or None if no startup cache is configured.
    pub fn startup_cache_stats(&self) -> Option<crate::startup_cache::StartupCacheStats> {
        self.startup_cache
            .lock()
            .unwrap()
            .as_ref()
            .map(|c| c.stats())
    }

    /// Drain accumulated startup cache diagnostics; return empty when unconfigured.
    pub fn take_startup_cache_diagnostics(&self) -> Vec<String> {
        self.startup_cache
            .lock()
            .unwrap()
            .as_ref()
            .map(|c| c.diagnostics())
            .unwrap_or_default()
    }

    pub(crate) fn request_cached_translation(
        &self,
        code: &[u8],
    ) -> Option<crate::api::ShaderTranslation> {
        self.cached_translation(code, None, true)
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn retry_cached_translation(
        &self,
        code: &[u8],
        context: &crate::startup_cache::TranslationContext,
        blocking: bool,
    ) -> Option<crate::api::ShaderTranslation> {
        if !blocking {
            return self.cached_translation(code, Some(context), false);
        }
        self.startup_cache.lock().unwrap().as_mut().map(|cache| {
            match cache.request_blocking(SHADER_STAGE_UNKNOWN, code, context) {
                Ok(Some(request)) => crate::api::ShaderTranslation::Cached(request),
                Ok(None) => unreachable!("blocking admission must return a request"),
                Err(error) => crate::api::ShaderTranslation::Error(error),
            }
        })
    }

    fn cached_translation(
        &self,
        code: &[u8],
        context: Option<&crate::startup_cache::TranslationContext>,
        count_lookup: bool,
    ) -> Option<crate::api::ShaderTranslation> {
        self.startup_cache.lock().unwrap().as_mut().map(|cache| {
            let captured = context.cloned().unwrap_or_else(|| cache.context());
            let result = if context.is_some() {
                cache.request_in_context(SHADER_STAGE_UNKNOWN, code, &captured, count_lookup)
            } else {
                cache.request(SHADER_STAGE_UNKNOWN, code, count_lookup)
            };
            match result {
                Ok(Some(request)) => crate::api::ShaderTranslation::Cached(request),
                Ok(None) => crate::api::ShaderTranslation::Deferred(code.to_vec(), captured),
                Err(error) => crate::api::ShaderTranslation::Error(error),
            }
        })
    }

    /// The host this instance was created with.
    pub fn host(&self) -> Option<&Host> {
        self.host.as_ref()
    }

    /// Select which guest counter token selects which counter. Every
    /// instance starts with `CounterContract::default()`. Recordings made
    /// after this call use the new contract. Source: the public contract.
    pub fn set_counter_contract(&self, contract: crate::counters::CounterContract) {
        *self.counters.lock().unwrap_or_else(|p| p.into_inner()) = contract;
    }

    /// The counter a guest token selects under the current contract.
    pub(crate) fn counter(&self, token: u64) -> Option<crate::counters::Counter> {
        self.counters
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .counter(token)
    }

    /// Select explicit command hypotheses. See the public contract.
    #[cfg(feature = "vulkan")]
    pub fn set_command_contract(
        &self,
        contract: Option<crate::gpu::operations::CommandContract>,
    ) -> bool {
        if contract.as_ref().is_some_and(|c| !c.valid()) {
            return false;
        }
        let mut gpu = self.gpu.lock().unwrap_or_else(|p| p.into_inner());
        let Some(backend) = gpu.as_mut() else {
            return false;
        };
        if backend.finish_draws().is_none() {
            return false;
        }
        backend.command_contract = contract;
        true
    }

    /// Enable the bounded first draw experiment with explicit host enum choices.
    /// Returns false when this instance has no Vulkan backend, or when a
    /// color or depth format table entry is one swxbx cannot draw with; the
    /// entry is named through `swxbx_last_error`. Source: the public contract.
    #[cfg(feature = "vulkan")]
    pub fn set_first_draw_contract(
        &self,
        contract: Option<crate::gpu::graphics::FirstDrawContract>,
    ) -> bool {
        if let Some(Err(why)) = contract.as_ref().map(|c| c.validate()) {
            crate::unsupported::refuse(
                crate::Status::BadArgument,
                format!("first-draw contract: {why}"),
            );
            return false;
        }
        let mut gpu = self.gpu.lock().unwrap_or_else(|p| p.into_inner());
        let Some(backend) = gpu.as_mut() else {
            return false;
        };
        let mut depth_images = Vec::new();
        for &(token, format) in contract
            .iter()
            .filter_map(|c| c.depth_raster.as_ref())
            .flat_map(|d| &d.depth_formats)
        {
            match crate::gpu::formats::DepthImage::resolve(backend.context(), format) {
                Ok(image) => depth_images.push((token, image)),
                Err(why) => {
                    crate::unsupported::refuse(
                        crate::Status::BadArgument,
                        format!("first-draw contract: depth target format token {token}: {why}"),
                    );
                    return false;
                }
            }
        }
        backend.depth_images = depth_images;
        let strides = contract.as_ref().map_or([None, None], |contract| {
            [
                contract.attribute_state_stride,
                contract.stream_state_stride,
            ]
        });
        backend.first_draw = contract;
        self.recording_strides.store(Arc::new(strides));
        true
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn recording_stride(&self, kind: &str) -> Option<u64> {
        let index = usize::from(kind != "VertexAttribState");
        self.recording_strides.load()[index]
    }

    /// Configure explicit stage/index-to-bank choices for translated graphics.
    /// Invalid mappings leave the current contract unchanged. Source: the public contract.
    #[cfg(feature = "vulkan")]
    pub fn set_uniform_buffer_contract(
        &self,
        contract: crate::gpu::uniforms::UniformBufferContract,
    ) -> Result<(), String> {
        contract.validate()?;
        let mut gpu = self.gpu.lock().unwrap_or_else(|p| p.into_inner());
        let backend = gpu.as_mut().ok_or("Vulkan backend is unavailable")?;
        backend.uniforms = contract;
        Ok(())
    }

    /// Configure explicit texture slot and sampler enum choices. Source: the public contract.
    #[cfg(feature = "vulkan")]
    pub fn set_texture_contract(
        &self,
        contract: crate::gpu::textures::TextureContract,
    ) -> Result<(), String> {
        contract.validate()?;
        let mut gpu = self.gpu.lock().unwrap_or_else(|p| p.into_inner());
        gpu.as_mut()
            .ok_or("Vulkan backend is unavailable")?
            .texture_contract = contract;
        Ok(())
    }

    /// Decode opaque recorded image-copy arguments using a host-supplied contract.
    #[cfg(feature = "vulkan")]
    pub fn set_image_copy_decoder(
        &self,
        decoder: Option<crate::gpu::image_layout::CopyDecoder>,
    ) -> bool {
        let mut gpu = self.gpu.lock().unwrap_or_else(|p| p.into_inner());
        let Some(backend) = gpu.as_mut() else {
            return false;
        };
        backend.copy_decoder = decoder.clone();
        self.copy_rules.rcu(|rules| crate::api::CopyRules {
            decoder: decoder.clone(),
            images: rules.images.clone(),
        });
        true
    }

    /// Interpret image flags, targets and formats only through explicit host rules.
    #[cfg(feature = "vulkan")]
    pub fn set_image_contract(
        &self,
        contract: crate::gpu::image_layout::ImageContract,
    ) -> Result<(), String> {
        contract.validate()?;
        let mut gpu = self.gpu.lock().unwrap_or_else(|p| p.into_inner());
        let backend = gpu.as_mut().ok_or("Vulkan backend is unavailable")?;
        if !backend.set_image_contract(contract.clone()) {
            return Err("image rules cannot change while arena images are live".into());
        }
        self.copy_rules.rcu(|rules| crate::api::CopyRules {
            decoder: rules.decoder.clone(),
            images: contract.clone(),
        });
        Ok(())
    }

    /// Opt into blend and channel argument hypotheses. Source: the public contract.
    #[cfg(feature = "vulkan")]
    pub fn set_blend_contract(
        &self,
        contract: Option<crate::gpu::graphics::BlendContract>,
    ) -> Result<(), String> {
        if let Some(c) = &contract {
            c.validate()?;
        }
        let mut gpu = self.gpu.lock().unwrap_or_else(|p| p.into_inner());
        gpu.as_mut()
            .ok_or("Vulkan backend is unavailable")?
            .blend_contract = contract;
        Ok(())
    }

    /// Return graphics request hit/miss counters, or None without a Vulkan backend.
    #[cfg(feature = "vulkan")]
    pub fn graphics_cache_stats(&self) -> Option<crate::gpu::pipelines::CacheStats> {
        self.gpu
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|backend| backend.graphics.stats())
    }

    /// Configure private persistent graphics cache files and a bounded worker pool.
    /// Call outside queue submission. Reconfiguration drains the previous workers.
    #[cfg(feature = "vulkan")]
    pub fn set_graphics_pipeline_cache(
        &self,
        directory: &std::path::Path,
        identity: &crate::gpu::pipelines::TranslationIdentity,
        worker_count: usize,
        queue_capacity: usize,
    ) -> Result<(), String> {
        let mut gpu = self.gpu.lock().unwrap_or_else(|p| p.into_inner());
        let backend = gpu.as_mut().ok_or("Vulkan backend is unavailable")?;
        let policy = backend.graphics.policy();
        backend.graphics = crate::gpu::graphics::GraphicsPipelines::persistent(
            backend.context(),
            directory,
            identity,
            worker_count,
            queue_capacity,
        )?;
        backend.graphics.set_policy(policy)?;
        Ok(())
    }

    /// Block at submission by default, or explicitly allow logged skipped draws.
    /// Call outside submission. Timed skip budgets above 16 milliseconds are rejected.
    #[cfg(feature = "vulkan")]
    pub fn set_pending_draw_policy(
        &self,
        policy: crate::gpu::graphics::PendingDrawPolicy,
    ) -> Result<(), String> {
        self.gpu
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_mut()
            .ok_or("Vulkan backend is unavailable")?
            .graphics
            .set_policy(policy)
    }

    /// Return successful graphics part, link, and whole-pipeline creation counts.
    #[cfg(feature = "vulkan")]
    pub fn graphics_compilation_stats(
        &self,
    ) -> Option<crate::gpu::graphics::GraphicsCompilationStats> {
        self.gpu
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|backend| backend.graphics.compilation_stats())
    }

    /// Return whether graphics library compilation and draw-time linking are enabled.
    #[cfg(feature = "vulkan")]
    pub fn graphics_library_support(&self) -> Option<(bool, bool)> {
        self.gpu
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|backend| {
                (
                    backend.context().graphics_pipeline_libraries,
                    backend.context().graphics_pipeline_fast_linking,
                )
            })
    }

    /// Queued and compiling graphics requests. Polling performs no disk I/O.
    #[cfg(feature = "vulkan")]
    pub fn graphics_pending_count(&self) -> Option<usize> {
        self.gpu
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_mut()
            .map(|backend| backend.graphics.pending())
    }

    /// Return graphics disk-cache counters, or None without a Vulkan backend.
    #[cfg(feature = "vulkan")]
    pub fn graphics_persistence_stats(&self) -> Option<crate::gpu::pipelines::PersistenceStats> {
        self.gpu
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|backend| backend.graphics.persistence_stats())
    }

    /// Collect cache diagnostics away from queue submission.
    #[cfg(feature = "vulkan")]
    pub fn take_graphics_cache_diagnostics(&self) -> Vec<String> {
        self.gpu
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|backend| backend.graphics.take_diagnostics())
            .unwrap_or_default()
    }

    /// Explicitly forget a failed compilation, retaining all successful requests.
    #[cfg(feature = "vulkan")]
    pub fn retry_graphics_pipeline(
        &self,
        stages: &[Vec<u32>],
        input: crate::gpu::graphics::VertexInput,
        topology: crate::gpu::graphics::PrimitiveTopology,
        state: crate::gpu::graphics::DrawPipelineState,
    ) -> Result<bool, String> {
        let mut gpu = self.gpu.lock().unwrap_or_else(|p| p.into_inner());
        let backend = gpu.as_mut().ok_or("Vulkan backend is unavailable")?;
        backend.graphics.retry_failed(
            stages,
            input,
            topology,
            state,
            backend.uniforms.storage_buffers,
        )
    }

    /// Replace the ordinary translation hook; configure startup caching exclusively instead.
    pub fn set_shader_translator(&self, translator: Option<Arc<dyn ShaderTranslator>>) {
        *self
            .shader_translator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = translator;
    }

    /// Return a shared copy of the current translation hook, if one is installed.
    pub fn shader_translator(&self) -> Option<Arc<dyn ShaderTranslator>> {
        self.shader_translator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Remember the hook `program`'s shaders register under.
    pub(crate) fn set_program_translator(
        &self,
        program: u64,
        translator: Arc<dyn ShaderTranslator>,
    ) {
        self.program_translators
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(program, translator);
    }

    /// The hook `program`'s shaders registered under.
    pub(crate) fn program_translator(&self, program: u64) -> Option<Arc<dyn ShaderTranslator>> {
        self.program_translators
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&program)
            .cloned()
    }

    /// Enable or disable translation during future shader registration.
    pub fn set_shader_translation_enabled(&self, enabled: bool) {
        self.shader_translation_enabled
            .store(enabled, Ordering::Relaxed);
    }

    /// Set the directory for optional translated shader dumps. `None` disables
    /// dumping. The directory receives output derived from the observed
    /// program's shaders, never the original shader bytes.
    pub fn set_shader_dump_directory(&self, directory: Option<&Path>) {
        *self
            .shader_dump_directory
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = directory.map(Path::to_path_buf);
    }

    pub(crate) fn shader_dump_directory(&self) -> Option<PathBuf> {
        self.shader_dump_directory
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub(crate) fn shader_translation_enabled(&self) -> bool {
        self.shader_translation_enabled.load(Ordering::Relaxed)
    }

    pub(crate) fn record_shader_translation_error(&self, error: String) {
        *self
            .shader_translation_errors
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(error)
            .or_default() += 1;
    }

    pub(crate) fn record_successful_shader_translation(&self, words: &[u32]) {
        self.successful_shader_translations
            .fetch_add(1, Ordering::Relaxed);
        self.distinct_shader_outputs
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(hash_bytes(
                &words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
            ));
    }

    pub(crate) fn clear_pending_shaders(&self, program: u64) {
        self.pending_shaders.lock().unwrap().remove(&program);
    }

    pub(crate) fn add_pending_shader(&self, program: u64, pending: PendingShader) {
        self.pending_shaders
            .lock()
            .unwrap()
            .entry(program)
            .or_default()
            .push(pending);
    }

    pub(crate) fn pending_shaders(&self, program: u64) -> Vec<PendingShader> {
        self.pending_shaders
            .lock()
            .unwrap()
            .get(&program)
            .cloned()
            .unwrap_or_default()
    }
    pub(crate) fn pending_programs(&self) -> Vec<u64> {
        self.pending_shaders
            .lock()
            .unwrap()
            .keys()
            .copied()
            .collect()
    }

    pub(crate) fn replace_pending_shader(&self, program: u64, record_index: usize) {
        if let Some(entries) = self.pending_shaders.lock().unwrap().get_mut(&program) {
            entries.retain(|entry| entry.record_index != record_index);
        }
    }

    pub(crate) fn fail_pending_shader(&self, program: u64, record_index: usize) {
        if let Some(entry) = self
            .pending_shaders
            .lock()
            .unwrap()
            .get_mut(&program)
            .and_then(|entries| {
                entries
                    .iter_mut()
                    .find(|entry| entry.record_index == record_index)
            })
        {
            entry.failures = entry.failures.saturating_add(1);
        }
    }

    pub(crate) fn record_late_read(&self, point: &'static str) {
        *self
            .shader_late_reads
            .lock()
            .unwrap()
            .entry(point)
            .or_default() += 1;
    }
    pub(crate) fn record_shader_zero_header(&self, zero: bool) {
        if zero {
            self.shader_zero_header.fetch_add(1, Ordering::Relaxed)
        } else {
            self.shader_header_without_code
                .fetch_add(1, Ordering::Relaxed)
        };
    }

    pub(crate) fn next_shader_dump(&self, bytes: &[u8]) -> Option<(u64, u64)> {
        let hash = hash_bytes(bytes);
        let mut hashes = self
            .shader_dump_hashes
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if !hashes.insert(hash) {
            return None;
        }
        let sequence = self.shader_dump_sequence.fetch_add(1, Ordering::Relaxed);
        Some((sequence, hash))
    }

    /// Record that the program asked for `name` and return its id.
    pub fn request(&self, name: &str) -> Option<FunctionId> {
        match functions::lookup(name) {
            Some(id) => {
                self.requested[id.0 as usize].store(true, Ordering::Relaxed);
                Some(id)
            }
            None => {
                let mut unknown = self
                    .unknown_requests
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                *unknown.entry(name.to_string()).or_default() += 1;
                None
            }
        }
    }

    /// Read program memory through the host. False when there is no host,
    /// no read callback, or the range is not accessible.
    pub fn read_memory(&self, address: u64, out: &mut [u8]) -> bool {
        let Some(read) = self.host.as_ref().and_then(|host| host.read_memory) else {
            return false;
        };
        let user = self
            .host
            .as_ref()
            .map_or(std::ptr::null_mut(), |host| host.user);
        // SAFETY: `out` is valid for its length; the host's callback reports
        // an inaccessible range instead of faulting.
        unsafe { read(user, address, out.as_mut_ptr(), out.len() as u64) == 0 }
    }

    /// Enable host write notifications for a pool and all its aliases.
    /// Every subsequent host storage write must be reported before submission.
    #[cfg(feature = "vulkan")]
    pub fn track_pool_writes(&self, pool: u64) -> bool {
        self.gpu
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_mut()
            .and_then(|b| b.global_memory.as_mut())
            .and_then(|m| m.track_host_writes(pool))
            .is_some()
    }

    /// Report a completed host storage write by byte address and length.
    /// Do not race notifications or storage writes against submission.
    #[cfg(feature = "vulkan")]
    pub fn notify_memory_write(&self, address: u64, size: u64) -> bool {
        let Some(end) = address.checked_add(size) else {
            return false;
        };
        self.host_writes
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(address..end);
        true
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn take_host_writes(&self) -> crate::dirty_ranges::Ranges {
        std::mem::take(&mut *self.host_writes.lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// Adopt coherent arena storage for a registered pool, preserving its
    /// contents, once queued batches have run. Batches still queued are not
    /// seen: `Instance::map_pool` runs them first and is the only way to reach
    /// this. A refusal is named through `swxbx_last_error`.
    ///
    /// # Safety
    /// Follow `GlobalMemory::map_pool` for pointer lifetime, callback routing and
    /// access synchronization.
    #[cfg(feature = "vulkan")]
    pub(crate) unsafe fn map_pool_after_queued(&self, pool: u64) -> Result<*mut u8, Status> {
        let mut gpu = self.gpu.lock().unwrap_or_else(|p| p.into_inner());
        let Some(memory) = gpu.as_mut().and_then(|b| b.global_memory.as_mut()) else {
            return Err(no_device_memory("map", pool));
        };
        for range in self.take_host_writes().intersections(0..u64::MAX) {
            memory.notify_storage_write(range.start, range.end - range.start);
        }
        if !memory.upload(|address, out| self.read_memory(address, out)) {
            return Err(crate::unsupported::refuse(
                Status::InternalError,
                format!(
                    "map pool {pool:#x}: reading guest memory into device storage before mapping failed"
                ),
            ));
        }
        unsafe { memory.map_pool(pool) }.map_err(|fault| pool_fault("map", pool, fault))
    }

    /// Wait for overlapping device access to mapped storage from batches that
    /// have already run. Batches still queued are not seen: `Instance::wait_pool`
    /// runs them first and is the only way to reach this. A refusal is named
    /// through `swxbx_last_error`.
    #[cfg(feature = "vulkan")]
    pub(crate) fn wait_pool_after_queued(
        &self,
        pool: u64,
        offset: u64,
        size: usize,
        write: bool,
    ) -> Status {
        let gpu = self.gpu.lock().unwrap_or_else(|p| p.into_inner());
        let Some(memory) = gpu.as_ref().and_then(|b| b.global_memory.as_ref()) else {
            return no_device_memory("wait on", pool);
        };
        match memory.wait_pool(pool, offset, size, write) {
            Ok(()) => Status::Ok,
            Err(crate::gpu::PoolFault::UnknownRange) => crate::unsupported::refuse(
                Status::BadArgument,
                format!(
                    "wait on pool {pool:#x}: {size} bytes at offset {offset:#x} do not lie inside a registered pool"
                ),
            ),
            Err(fault) => pool_fault("wait on", pool, fault),
        }
    }

    /// Cumulative payload copies and legacy inspection bytes for this instance.
    #[cfg(feature = "vulkan")]
    pub fn transfer_statistics(&self) -> Option<crate::gpu::TransferStatistics> {
        self.gpu
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .and_then(|b| b.global_memory.as_ref())
            .map(|m| m.transfer_statistics())
    }

    /// Write program memory through the host.
    pub fn write_memory(&self, address: u64, data: &[u8]) -> bool {
        if !self.write_memory_untracked(address, data) {
            return false;
        }
        #[cfg(feature = "vulkan")]
        self.notify_memory_write(address, data.len() as u64);
        true
    }

    pub(crate) fn write_memory_untracked(&self, address: u64, data: &[u8]) -> bool {
        let Some(write) = self.host.as_ref().and_then(|host| host.write_memory) else {
            return false;
        };
        let user = self
            .host
            .as_ref()
            .map_or(std::ptr::null_mut(), |host| host.user);
        // SAFETY: `data` is valid for its length.
        unsafe { write(user, address, data.as_ptr(), data.len() as u64) == 0 }
    }

    /// A host that lets the original implementation run reports what it
    /// returned here, so results are sampled along with arguments.
    pub fn returned(&self, function: FunctionId, registers: &Registers) -> Status {
        if function.0 as usize >= self.shapes.len() {
            return Status::BadFunction;
        }
        // The snapshot only belongs to this return if the thread's last
        // reported call was the same function.
        let snapshot = PENDING
            .take()
            .filter(|(pending, _, _)| *pending == function.0)
            .map(|(_, snapshot, registers)| (snapshot, registers));
        if let Some((_, call_registers)) = snapshot.as_ref() {
            self.observe_pool_return(function, call_registers, registers);
        }
        self.shape(function).record_return(
            self.host.as_ref(),
            registers,
            snapshot.as_ref().map(|(snapshot, _)| snapshot),
        );
        Status::Ok
    }

    fn observe_pool_return(
        &self,
        function: FunctionId,
        call_registers: &Registers,
        return_registers: &Registers,
    ) {
        let Some(name) = functions::all()
            .find(|(id, _)| *id == function)
            .map(|(_, name)| name)
        else {
            return;
        };
        if name == "nvnMemoryPoolGetBufferAddress" {
            let pool = call_registers.x[0];
            let gpu_address = return_registers.x[0];
            self.objects.update(pool, |object| {
                if let crate::api::Object::MemoryPool {
                    observed_gpu_address: base,
                    ..
                } = object
                {
                    *base = Some(gpu_address);
                }
            });
        }
    }

    fn shape(&self, function: FunctionId) -> std::sync::MutexGuard<'_, FunctionShape> {
        self.shapes[function.0 as usize]
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The sampled shapes of every function that was called, as text.
    pub fn shapes_report(&self) -> String {
        let mut out = String::from(
            "# swxbx shapes: what the argument and result registers held, per function\n",
        );
        for (id, name) in functions::all() {
            let calls = self.call_count(id);
            if calls > 0 {
                self.shape(id).write(&mut out, name, calls);
            }
        }
        out
    }

    /// Return this instance's call count for the function id; unknown ids report zero.
    pub fn call_count(&self, function: FunctionId) -> u64 {
        self.calls
            .get(function.0 as usize)
            .map_or(0, |counter| counter.load(Ordering::Relaxed))
    }

    /// A snapshot of what was requested and called.
    pub fn census(&self) -> Census {
        let functions = functions::all()
            .map(|(id, name)| CensusEntry {
                name,
                requested: self.requested[id.0 as usize].load(Ordering::Relaxed),
                calls: self.calls[id.0 as usize].load(Ordering::Relaxed),
            })
            .collect();
        let unknown_requests = self
            .unknown_requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let pending_shaders = self
            .pending_shaders
            .lock()
            .unwrap()
            .values()
            .map(|entries| entries.iter().filter(|entry| entry.failures < 64).count() as u64)
            .sum();
        let mut shader_translation_errors = self.shader_translation_errors.lock().unwrap().clone();
        if pending_shaders > 0 {
            *shader_translation_errors
                .entry("no code after header".into())
                .or_default() += pending_shaders;
        }
        Census {
            functions,
            unknown_requests,
            successful_shader_translations: self
                .successful_shader_translations
                .load(Ordering::Relaxed),
            distinct_shader_outputs: self
                .distinct_shader_outputs
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .len(),
            shader_translation_errors,
            pending_shaders,
            shader_late_reads: self.shader_late_reads.lock().unwrap().clone(),
            shader_zero_header: self.shader_zero_header.load(Ordering::Relaxed),
            shader_header_without_code: self.shader_header_without_code.load(Ordering::Relaxed),
        }
    }
}

/// Refuse a pool access on an instance that has no device memory.
#[cfg(feature = "vulkan")]
fn no_device_memory(access: &str, pool: u64) -> Status {
    crate::unsupported::refuse(
        Status::BadArgument,
        format!("{access} pool {pool:#x}: no device memory exists, so no pool is registered"),
    )
}

/// Refuse a pool access by what the device memory found.
#[cfg(feature = "vulkan")]
fn pool_fault(access: &str, pool: u64, fault: crate::gpu::PoolFault) -> Status {
    match fault {
        crate::gpu::PoolFault::UnknownRange => crate::unsupported::refuse(
            Status::BadArgument,
            format!("{access} pool {pool:#x}: the pool is not registered"),
        ),
        crate::gpu::PoolFault::WaitFailed => crate::unsupported::refuse(
            Status::InternalError,
            format!("{access} pool {pool:#x}: waiting for device access to it failed"),
        ),
    }
}

/// Host connection whose shutdown joins all queued execution.
pub struct Instance {
    state: Arc<InstanceState>,
    pub(crate) queue: crate::execution::QueueExecutor,
}

impl std::ops::Deref for Instance {
    type Target = InstanceState;
    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl Instance {
    /// Create an instance without host callbacks.
    pub fn new() -> Self {
        Self::from_state(InstanceState::new())
    }

    /// Record a typed texel-buffer binding supplied by a host adapter.
    #[cfg(feature = "vulkan")]
    pub fn record_texel_buffer(
        &self,
        command_buffer: u64,
        stage: u64,
        index: u64,
        address: u64,
        size: u64,
        format: ash::vk::Format,
    ) -> Status {
        let command =
            crate::api::RecordedCommand::State(crate::api::StateCommand::BindTexelBuffer {
                stage,
                index,
                address,
                size,
                format: format.as_raw() as u32,
            });
        self.objects
            .record_host_command(command_buffer, command)
            .map_or_else(|status| status, |_| Status::Ok)
    }

    /// Create a hosted instance.
    ///
    /// # Safety
    /// Callbacks and their context remain valid until shutdown returns and support concurrent calls.
    pub unsafe fn with_host(host: Host) -> Self {
        Self::from_state(unsafe { InstanceState::with_host(host) })
    }

    /// Wait for overlapping device access before using mapped storage. Batches
    /// submitted on this instance before the call count: they run first, so
    /// their access is known. A refusal is named through `swxbx_last_error`.
    /// Source: the public contract.
    #[cfg(feature = "vulkan")]
    pub fn wait_pool(&self, pool: u64, offset: u64, size: usize, write: bool) -> Status {
        match self.run_queued("wait on", pool) {
            Ok(()) => self.state.wait_pool_after_queued(pool, offset, size, write),
            Err(status) => status,
        }
    }

    /// Adopt coherent arena storage for a registered pool, preserving its
    /// contents. Batches submitted on this instance before the call run first,
    /// so none of them runs after the pool's backing changed. A refusal is
    /// named through `swxbx_last_error`. Source: the public contract.
    ///
    /// # Safety
    /// Follow `GlobalMemory::map_pool` for pointer lifetime, callback routing and
    /// access synchronization.
    #[cfg(feature = "vulkan")]
    pub unsafe fn map_pool(&self, pool: u64) -> Result<*mut u8, Status> {
        self.run_queued("map", pool)?;
        unsafe { self.state.map_pool_after_queued(pool) }
    }

    /// Run every batch submitted before a host access to `pool`.
    #[cfg(feature = "vulkan")]
    fn run_queued(&self, access: &str, pool: u64) -> Result<(), Status> {
        match self.queue.drain(crate::execution::Reach::Queued) {
            crate::execution::Drained::Ran => Ok(()),
            crate::execution::Drained::Lost => Err(crate::unsupported::refuse(
                Status::InternalError,
                format!(
                    "{access} pool {pool:#x}: the device was lost, so earlier batches never ran"
                ),
            )),
            crate::execution::Drained::OnExecutionThread => Err(crate::unsupported::refuse(
                Status::BadArgument,
                format!(
                    "{access} pool {pool:#x}: called on SwXBx's execution thread, which cannot wait for the batches it runs"
                ),
            )),
        }
    }

    fn from_state(state: InstanceState) -> Self {
        let state = Arc::new(state);
        let queue = crate::execution::QueueExecutor::new(state.clone());
        Self { state, queue }
    }

    /// Load startup caches and configure translation before use.
    pub fn with_startup_cache(
        config: crate::startup_cache::StartupCacheConfig,
        translator: Arc<dyn ShaderTranslator>,
    ) -> Result<Self, String> {
        Ok(Self::from_state(InstanceState::with_startup_cache(
            config, translator,
        )?))
    }

    /// Create a hosted instance with startup caches.
    ///
    /// # Safety
    /// The callback lifetime and concurrency contract of with_host applies.
    pub unsafe fn with_host_startup_cache(
        host: Host,
        config: crate::startup_cache::StartupCacheConfig,
        translator: Arc<dyn ShaderTranslator>,
    ) -> Result<Self, String> {
        Ok(Self::from_state(unsafe {
            InstanceState::with_host_startup_cache(host, config, translator)?
        }))
    }

    /// Drain previous execution before replacing cache configuration.
    pub fn configure_startup_cache(
        &mut self,
        config: crate::startup_cache::StartupCacheConfig,
        translator: Arc<dyn ShaderTranslator>,
    ) -> Result<(), String> {
        if self.queue.drain(crate::execution::Reach::Completed) != crate::execution::Drained::Ran {
            return Err("queued execution failed".into());
        }
        self.state.configure_startup_cache(config, translator)
    }

    /// Run a call. A function with a handler runs it; any other known
    /// function is counted and reported as unimplemented with zeroed
    /// results.
    pub fn call(&self, function: FunctionId, registers: &mut Registers) -> Status {
        // A recording call refuses misuse on this thread; its reason in
        // `swxbx_last_error` belongs to this call. Source: the public contract.
        crate::clear_last_error();
        let name = functions::name(function);
        if name.is_some_and(|name| {
            !name.contains("CommandBuffer")
                && !name.ends_with("QueueSubmitCommands")
                && !name.ends_with("QueueFlush")
        }) {
            // Waiting orders this call after earlier batches. A batch that
            // stopped early was reported when it ran; only a lost device
            // fails this call. Source: the public contract. Presenting copies the
            // texture on the device queue behind those batches, so it waits
            // for them to run, not for the device to finish them.
            let reach = if name.is_some_and(|name| name.ends_with("QueuePresentTexture")) {
                crate::execution::Reach::Queued
            } else {
                crate::execution::Reach::Completed
            };
            if self.queue.drain(reach) != crate::execution::Drained::Ran {
                return Status::InternalError;
            }
        }
        let Some(counter) = self.calls.get(function.0 as usize) else {
            return Status::BadFunction;
        };
        // The first calls of each function are sampled for their shape; after
        // that only the counter is touched.
        let snapshot = if counter.fetch_add(1, Ordering::Relaxed) < crate::observe::SAMPLE_LIMIT {
            self.shape(function)
                .record_call(self.host.as_ref(), &self.objects, registers)
        } else {
            None
        };
        PENDING.set(snapshot.map(|snapshot| (function.0, snapshot, *registers)));
        if let Some(Some(handler)) = self.handlers.get(function.0 as usize) {
            return handler(self, function, registers);
        }
        registers.x[0] = 0;
        registers.x[1] = 0;
        registers.d[0] = 0;
        Status::Unimplemented
    }
}
impl Drop for Instance {
    fn drop(&mut self) {
        self.queue.shutdown();
        let _ = self.poll_presentations(true);
        self.unsupported_commands.finish();
    }
}

impl Default for Instance {
    fn default() -> Self {
        Self::new()
    }
}

struct CensusEntry {
    name: &'static str,
    requested: bool,
    calls: u64,
}

/// Which functions a program requested and how often it called each. It
/// holds names and counts only, never a program's data.
pub struct Census {
    functions: Vec<CensusEntry>,
    unknown_requests: BTreeMap<String, u64>,
    successful_shader_translations: u64,
    distinct_shader_outputs: usize,
    shader_translation_errors: BTreeMap<String, u64>,
    pending_shaders: u64,
    shader_late_reads: BTreeMap<&'static str, u64>,
    shader_zero_header: u64,
    shader_header_without_code: u64,
}

impl Census {
    /// Number of distinct functions called at least once.
    pub fn functions_called(&self) -> usize {
        self.functions
            .iter()
            .filter(|entry| entry.calls > 0)
            .count()
    }

    /// Number of distinct known functions the program requested.
    pub fn functions_requested(&self) -> usize {
        self.functions
            .iter()
            .filter(|entry| entry.requested)
            .count()
    }
}

/// The text form: a summary line, then one line per function that was
/// requested or called, then the unknown names.
impl fmt::Display for Census {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "# swxbx census: {} of {} functions requested, {} called, {} unknown names requested",
            self.functions_requested(),
            self.functions.len(),
            self.functions_called(),
            self.unknown_requests.len()
        )?;
        writeln!(f, "# calls requested name")?;
        for entry in &self.functions {
            if entry.requested || entry.calls > 0 {
                let requested = if entry.requested { "yes" } else { "no" };
                writeln!(f, "{} {} {}", entry.calls, requested, entry.name)?;
            }
        }
        for (name, count) in &self.unknown_requests {
            writeln!(f, "unknown {count} {name}")?;
        }
        for (error, count) in &self.shader_translation_errors {
            writeln!(f, "shader-translation-error {count} {error}")?;
        }
        for (point, count) in &self.shader_late_reads {
            writeln!(f, "shader-late-read {point} {count}")?;
        }
        writeln!(f, "shader-still-pending {}", self.pending_shaders)?;
        writeln!(f, "shader-zero-header {}", self.shader_zero_header)?;
        writeln!(
            f,
            "shader-header-without-code {}",
            self.shader_header_without_code
        )?;
        writeln!(
            f,
            "shader-translations-successful {} distinct-outputs {}",
            self.successful_shader_translations, self.distinct_shader_outputs
        )?;
        Ok(())
    }
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct FakeTranslator;

    impl ShaderTranslator for FakeTranslator {
        /// Translate supplied shader bytes into SPIR-V words or a diagnostic.
        fn translate(&self, stage: ShaderStage, code: &[u8]) -> Result<Vec<u32>, String> {
            Ok(vec![stage, code.len() as u32])
        }
    }

    #[test]
    fn shader_translator_can_be_registered_and_called() {
        let instance = Instance::new();
        assert!(instance.shader_translator().is_none());
        instance.set_shader_translator(Some(Arc::new(FakeTranslator)));
        let translator = instance.shader_translator().expect("translator");
        assert_eq!(translator.translate(7, &[1, 2, 3]), Ok(vec![7, 3]));
        instance.set_shader_translator(None);
        assert!(instance.shader_translator().is_none());
    }

    #[test]
    fn a_call_is_counted_and_reported_unimplemented() {
        let instance = Instance::new();
        let id = FunctionId(3);
        let mut registers = Registers {
            x: [1, 2, 3, 4, 5, 6, 7, 8],
            d: [9; 8],
            sp: 0x1000,
        };
        assert_eq!(instance.call(id, &mut registers), Status::Unimplemented);
        assert_eq!(instance.call(id, &mut registers), Status::Unimplemented);
        assert_eq!(instance.call_count(id), 2);
        assert_eq!((registers.x[0], registers.x[1], registers.d[0]), (0, 0, 0));
        // Arguments other than the result registers are left alone.
        assert_eq!(registers.x[2], 3);
        assert_eq!(registers.sp, 0x1000);
    }

    #[test]
    fn an_id_outside_the_table_is_refused() {
        let instance = Instance::new();
        let id = FunctionId(functions::count() as u32);
        assert_eq!(
            instance.call(id, &mut Registers::default()),
            Status::BadFunction
        );
        assert_eq!(instance.call_count(id), 0);
    }

    #[test]
    fn the_census_lists_requests_calls_and_unknown_names() {
        let instance = Instance::new();
        let (first, first_name) = functions::all().next().unwrap();
        let (second, second_name) = functions::all().nth(1).unwrap();
        assert_eq!(instance.request(first_name), Some(first));
        assert_eq!(instance.request("somethingElse"), None);
        assert_eq!(instance.request("somethingElse"), None);
        instance.call(second, &mut Registers::default());

        let census = instance.census();
        assert_eq!(census.functions_requested(), 1);
        assert_eq!(census.functions_called(), 1);

        let text = census.to_string();
        assert!(text.contains(&format!("0 yes {first_name}\n")), "{text}");
        assert!(text.contains(&format!("1 no {second_name}\n")), "{text}");
        assert!(text.contains("unknown 2 somethingElse\n"), "{text}");
        assert!(
            text.contains("shader-translations-successful 0 distinct-outputs 0\n"),
            "{text}"
        );
        assert_eq!(text.lines().count(), 9, "{text}");
    }

    #[test]
    fn counting_is_safe_from_several_threads() {
        let instance = Instance::new();
        let id = FunctionId(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..1000 {
                        instance.call(id, &mut Registers::default());
                    }
                });
            }
        });
        assert_eq!(instance.call_count(id), 4000);
    }

    #[test]
    fn observed_pool_address_is_used_for_shader_resolution() {
        let instance = Instance::new();
        let defaults = FunctionId(
            functions::lookup("nvnMemoryPoolBuilderSetDefaults")
                .unwrap()
                .0,
        );
        let storage = FunctionId(
            functions::lookup("nvnMemoryPoolBuilderSetStorage")
                .unwrap()
                .0,
        );
        let initialize = FunctionId(functions::lookup("nvnMemoryPoolInitialize").unwrap().0);
        let address = FunctionId(
            functions::lookup("nvnMemoryPoolGetBufferAddress")
                .unwrap()
                .0,
        );

        let mut registers = Registers {
            x: [20, 0, 0, 0, 0, 0, 0, 0],
            ..Registers::default()
        };
        assert_eq!(instance.call(defaults, &mut registers), Status::Ok);
        registers.x[0] = 20;
        registers.x[1] = 0x1000;
        registers.x[2] = 0x100;
        assert_eq!(instance.call(storage, &mut registers), Status::Ok);
        registers.x = [21, 20, 0, 0, 0, 0, 0, 0];
        assert_eq!(instance.call(initialize, &mut registers), Status::Ok);

        let call = Registers {
            x: [21, 0, 0, 0, 0, 0, 0, 0],
            ..Registers::default()
        };
        let result = Registers {
            x: [0x9000, 0, 0, 0, 0, 0, 0, 0],
            ..Registers::default()
        };
        let mut call_registers = call;
        assert_eq!(instance.call(address, &mut call_registers), Status::Ok);
        instance.returned(address, &result);

        assert_eq!(
            instance.objects.resolve_gpu_address(0x9080),
            Ok(crate::api::GpuAddress {
                pool: 21,
                offset: 0x80,
                remaining: 0x80,
                program_address: 0x1080,
            })
        );
    }
}
