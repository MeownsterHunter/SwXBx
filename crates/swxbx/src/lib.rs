//! A graphics-call library with CPU execution and an optional Vulkan backend.
//!
//! A host creates an [`Instance`], owns program memory, and forwards calls with
//! [`Registers`]. The library records objects and command lists, counts calls,
//! and executes supported clears, copies, and presentation. Unsupported calls
//! return [`Status::Unimplemented`].
//!
//! The `vulkan` feature adds arena-backed resources and bounded graphics draws.
//! Draw execution needs explicit host contracts for unresolved formats and state.
//! [`ShaderTranslator`] supplies translated SPIR-V; no translator is included.
//! [`startup_cache`] loads translated output and schedules owned worker requests.
//!
//! The repository guide lives in `docs/guide`. Guest behavior is limited to the
//! signature, shape, census, and provenance records. Public host execution uses
//! Vulkan contracts and original synthetic tests.
#![deny(missing_docs)]

pub mod api;
pub mod counters;
#[cfg(any(test, feature = "vulkan"))]
mod dirty_ranges;
#[cfg(feature = "draw-metrics")]
pub mod draw_metrics;
mod execution;
pub mod frame_transport;
pub mod functions;
pub mod global_memory;
#[cfg(feature = "vulkan")]
pub mod gpu;
mod instance;
mod keymap;
pub mod observe;
pub mod presentation;
pub use presentation::{
    Aspect, FrameStatistics, HostScaling, PresentInterval, PresentationConfig, PresentationMode,
    Scaling, SCALING_FIT, SCALING_STRETCH,
};
pub mod startup_cache;
pub mod surfaces;
pub mod tiling;
mod unsupported;
mod workers;

pub use instance::{
    Census, Host, HostVulkan, Instance, Registers, ShaderStage, ShaderTranslator, Status,
    SHADER_STAGE_UNKNOWN,
};

use functions::FunctionId;
use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;

/// Version of the host interface. It changes when a host built against an
/// older `include/swxbx.h` could no longer use the library.
pub const HOST_INTERFACE_VERSION: u32 = 6;

/// Returned by lookups for a name the library does not know.
pub const FUNCTION_NONE: u32 = u32::MAX;

const MAX_TRANSLATED_SHADER_WORDS: u64 = 16 * 1024 * 1024;

/// Host-owned SPIR-V returned by [`HostShaderTranslator::translate`].
///
/// The words remain owned by the host. SwXBx copies them before invoking the
/// matching [`HostShaderTranslator::release`] callback.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct HostTranslatedShader {
    /// Pointer to `word_count` SPIR-V words.
    pub words: *const u32,
    /// Number of words at `words`.
    pub word_count: u64,
}

/// Owned C callback contract for translating guest shader bytes.
///
/// A successful call to [`swxbx_instance_set_shader_translator`] transfers
/// ownership of `user` to SwXBx. Callbacks may run concurrently. `release`
/// runs once for every successful nonempty output, and `destroy_user` runs
/// once after the translator and all background requests stop using it.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct HostShaderTranslator {
    /// Opaque host context passed to every callback.
    pub user: *mut c_void,
    /// Translate bounded shader bytes. Zero means success.
    pub translate: Option<
        unsafe extern "C" fn(
            user: *mut c_void,
            stage: ShaderStage,
            code: *const u8,
            code_size: u64,
            output: *mut HostTranslatedShader,
        ) -> i32,
    >,
    /// Release one nonempty output previously returned by `translate`.
    pub release:
        Option<unsafe extern "C" fn(user: *mut c_void, words: *const u32, word_count: u64)>,
    /// Release the owned host context after its final use.
    pub destroy_user: Option<unsafe extern "C" fn(user: *mut c_void)>,
}

struct CShaderTranslator(HostShaderTranslator);

struct HostOutput<'a> {
    translator: &'a HostShaderTranslator,
    output: HostTranslatedShader,
}

impl Drop for HostOutput<'_> {
    fn drop(&mut self) {
        if let Some(release) = self.translator.release {
            unsafe {
                release(
                    self.translator.user,
                    self.output.words,
                    self.output.word_count,
                )
            };
        }
    }
}

// SAFETY: successful installation requires the host to uphold the documented
// concurrent-callback contract for its context and callbacks.
unsafe impl Send for CShaderTranslator {}
unsafe impl Sync for CShaderTranslator {}

impl Drop for CShaderTranslator {
    fn drop(&mut self) {
        if let Some(destroy) = self.0.destroy_user {
            unsafe { destroy(self.0.user) };
        }
    }
}

impl ShaderTranslator for CShaderTranslator {
    fn translate(&self, stage: ShaderStage, code: &[u8]) -> Result<Vec<u32>, String> {
        let translate = self
            .0
            .translate
            .ok_or_else(|| "host shader translator has no translate callback".to_string())?;
        if self.0.release.is_none() {
            return Err("host shader translator has no release callback".into());
        }
        let mut output = HostTranslatedShader::default();
        let status = unsafe {
            translate(
                self.0.user,
                stage,
                code.as_ptr(),
                code.len() as u64,
                &mut output,
            )
        };
        if status != 0 {
            return Err(format!(
                "host shader translation failed with status {status}"
            ));
        }
        if output.words.is_null() || output.word_count == 0 {
            return Err("host shader translator returned no words".into());
        }
        let output = HostOutput {
            translator: &self.0,
            output,
        };
        if output.output.word_count > MAX_TRANSLATED_SHADER_WORDS {
            return Err("host shader translation is too large".into());
        }
        let word_count = usize::try_from(output.output.word_count)
            .map_err(|_| "host shader translation is too large".to_string())?;
        let words = unsafe { std::slice::from_raw_parts(output.output.words, word_count) }.to_vec();
        Ok(words)
    }
}

static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Name why this thread's call failed, for `swxbx_last_error`.
pub(crate) fn set_last_error(message: &str) {
    LAST_ERROR.with(|error| {
        *error.borrow_mut() =
            Some(CString::new(message.replace('\0', " ")).expect("NUL bytes were replaced"))
    });
}

/// This thread's last reason, if a call on it failed with one.
pub(crate) fn last_error() -> Option<String> {
    LAST_ERROR.with(|error| {
        error
            .borrow()
            .as_ref()
            .map(|message| message.to_string_lossy().into_owned())
    })
}

/// Forget this thread's last reason, so the next one belongs to the next call.
pub(crate) fn clear_last_error() {
    LAST_ERROR.with(|error| *error.borrow_mut() = None);
}

fn ffi_guard<T>(default: T, f: impl FnOnce() -> T) -> T {
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown panic");
            LAST_ERROR.with(|error| {
                *error.borrow_mut() = Some(
                    CString::new(message)
                        .unwrap_or_else(|_| CString::new("panic message contained NUL").unwrap()),
                )
            });
            default
        }
    }
}

/// Why the most recent failed call on this thread failed: a refused
/// command's reason or a panic message. Null when there was none.
#[no_mangle]
pub extern "C" fn swxbx_last_error() -> *const c_char {
    // Not wrapped in `ffi_guard`, which would clear the message first.
    LAST_ERROR.with(|error| {
        error
            .borrow()
            .as_ref()
            .map_or(std::ptr::null(), |message| message.as_ptr())
    })
}

/// The library version as a NUL-terminated string that lives as long as the
/// library is loaded.
#[no_mangle]
pub extern "C" fn swxbx_version() -> *const c_char {
    ffi_guard(std::ptr::null(), || VERSION.as_ptr().cast())
}

/// The host interface version this library was built with.
#[no_mangle]
pub extern "C" fn swxbx_host_interface_version() -> u32 {
    ffi_guard(0, || HOST_INTERFACE_VERSION)
}

/// Number of functions in the table. Valid function ids are `0..count`.
#[no_mangle]
pub extern "C" fn swxbx_function_count() -> u32 {
    ffi_guard(0, || functions::count() as u32)
}

/// Name of a function as a NUL-terminated string owned by the library, or
/// null for an id outside the table.
#[no_mangle]
pub extern "C" fn swxbx_function_name(function: u32) -> *const c_char {
    ffi_guard(std::ptr::null(), || {
        functions::c_name(FunctionId(function)).map_or(std::ptr::null(), CStr::as_ptr)
    })
}

/// Id of the function with this name, or `FUNCTION_NONE`.
///
/// # Safety
/// `name` must be null or point to a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn swxbx_function_lookup(name: *const c_char) -> u32 {
    ffi_guard(FUNCTION_NONE, || {
        if name.is_null() {
            return FUNCTION_NONE;
        }
        // SAFETY: the caller passes a NUL-terminated string.
        let name = unsafe { CStr::from_ptr(name) };
        name.to_str()
            .ok()
            .and_then(functions::lookup)
            .map_or(FUNCTION_NONE, |id| id.0)
    })
}

/// Enable host write notifications for a registered pool and its aliases.
/// Every subsequent host storage write must be reported before submission.
///
/// # Safety
/// `instance` must be a live instance pointer, or null.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_track_pool_writes(
    instance: *const Instance,
    pool: u64,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        #[cfg(feature = "vulkan")]
        {
            if instance.track_pool_writes(pool) {
                Status::Ok
            } else {
                Status::BadArgument
            }
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (instance, pool);
            Status::Unimplemented
        }
    })
}

/// Report a host storage write by byte address and length before the next submission.
///
/// # Safety
/// `instance` must be a live instance pointer, or null.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_notify_memory_write(
    instance: *const Instance,
    address: u64,
    size: u64,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        #[cfg(feature = "vulkan")]
        {
            if instance.notify_memory_write(address, size) {
                Status::Ok
            } else {
                Status::BadArgument
            }
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (instance, address, size);
            Status::Unimplemented
        }
    })
}

/// Adopt coherent arena storage for a registered pool. Batches submitted on this
/// instance before the call run first, so none of them runs after the pool's
/// backing changed. Returns null on failure and names the reason through
/// `swxbx_last_error`.
///
/// # Safety
/// `instance` must be live or null. The host must satisfy the mapping and access
/// contract of `GlobalMemory::map_pool`, including callback routing and waits.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_map_pool(instance: *const Instance, pool: u64) -> *mut u8 {
    ffi_guard(std::ptr::null_mut(), || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return std::ptr::null_mut();
        };
        #[cfg(feature = "vulkan")]
        {
            unsafe { instance.map_pool(pool) }.unwrap_or(std::ptr::null_mut())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (instance, pool);
            std::ptr::null_mut()
        }
    })
}

/// Wait for device access to a mapped host range. Nonzero `write` waits for readers too.
/// Batches submitted on this instance before this call run first and their
/// device access is waited for, so a host write after such a submit never races
/// a batch that has not read yet; batches submitted concurrently from another
/// thread are not covered. Returns `BadArgument` for a range outside a registered
/// pool or a call on SwXBx's execution thread, and `InternalError` when the
/// device was lost or the wait failed, naming the reason through
/// `swxbx_last_error`.
///
/// # Safety
/// `instance` must be a live instance pointer, or null. Finish the host read
/// or write before submitting work that uses the range.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_wait_pool(
    instance: *const Instance,
    pool: u64,
    offset: u64,
    size: u64,
    write: u32,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        let Ok(size) = usize::try_from(size) else {
            return Status::BadArgument;
        };
        #[cfg(feature = "vulkan")]
        {
            instance.wait_pool(pool, offset, size, write != 0)
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (instance, pool, offset, size, write);
            Status::Unimplemented
        }
    })
}

/// Create an instance. `host` may be null for a host that provides nothing
/// yet; the structure is copied. Destroy the result with
/// `swxbx_instance_destroy`.
///
/// # Safety
/// `host` must be null or point to a valid `Host`. Its `user` pointer and
/// callbacks must stay usable for the life of the instance and must tolerate
/// being used from several threads at once.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_create(host: *const Host) -> *mut Instance {
    ffi_guard(std::ptr::null_mut(), || {
        // SAFETY: the caller passes null or a valid `Host`.
        let instance = match unsafe { host.as_ref() } {
            Some(host) => unsafe { Instance::with_host(*host) },
            None => Instance::new(),
        };
        Box::into_raw(Box::new(instance))
    })
}

/// Destroy an instance. Null is accepted.
///
/// # Safety
/// `instance` must be null or come from `swxbx_instance_create` and not
/// have been destroyed already. No other call on the instance may be running
/// or start afterwards.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_destroy(instance: *mut Instance) {
    ffi_guard((), || {
        if !instance.is_null() {
            drop(unsafe { Box::from_raw(instance) });
        }
    })
}

/// Set presentation capacity and mode, zero for FIFO or one for mailbox. With a
/// Vulkan-surface host, mailbox needs an open window and every open window's
/// surface must offer it. Otherwise it returns `BadArgument` (no open window, or
/// a surface without the mode) or `InternalError` (a surface that cannot be
/// asked), names the reason through `swxbx_last_error`, and keeps the previous
/// policy. A window opened later while mailbox is set is asked when it opens and
/// refused the same way. FIFO is always accepted.
///
/// # Safety
/// `instance` must be live. Presentation callbacks must not reenter the library.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_set_presentation(
    instance: *const Instance,
    frames_in_flight: u32,
    mode: u32,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        let mode = match mode {
            0 => PresentationMode::Fifo,
            1 => PresentationMode::Mailbox,
            _ => return Status::BadArgument,
        };
        instance.configure_presentation(PresentationConfig {
            frames_in_flight,
            mode,
        })
    })
}

/// Deliver one ready callback, or drain all pending callbacks when wait is one.
///
/// # Safety
/// `instance` must be live. Presentation callbacks must not reenter the library.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_poll_presentations(
    instance: *const Instance,
    wait: u32,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        if wait > 1 {
            return Status::BadArgument;
        }
        instance.poll_presentations(wait != 0)
    })
}

/// Copy cumulative frame counts and latency moments into host storage.
///
/// # Safety
/// `instance` must be live and `out` must be writable for one FrameStatistics.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_frame_statistics(
    instance: *const Instance,
    out: *mut FrameStatistics,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        let Some(out) = (unsafe { out.as_mut() }) else {
            return Status::BadArgument;
        };
        *out = instance.frame_statistics();
        Status::Ok
    })
}

/// Set or clear the directory for local debugging dumps of translated
/// shaders. The files contain translated output derived from the observed
/// program's shaders and never the original shader bytes.
///
/// # Safety
/// `instance` must be live and `directory` must be null or NUL-terminated
/// UTF-8.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_set_shader_dump_directory(
    instance: *const Instance,
    directory: *const c_char,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        let path = if directory.is_null() {
            None
        } else {
            let value = unsafe { CStr::from_ptr(directory) };
            let Ok(value) = value.to_str() else {
                return Status::BadArgument;
            };
            Some(Path::new(value))
        };
        instance.set_shader_dump_directory(path);
        Status::Ok
    })
}

/// Install or clear a host shader translator.
///
/// A non-null translator must provide `translate`, `release`, and
/// `destroy_user`. On success SwXBx owns its `user` context until the
/// translator is replaced or the instance is destroyed. Passing null clears
/// the translator after outstanding requests release their references.
///
/// # Safety
/// `instance` must be live. `translator` must be null or point to a valid
/// [`HostShaderTranslator`] whose callbacks satisfy its ownership and
/// concurrency contract.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_set_shader_translator(
    instance: *const Instance,
    translator: *const HostShaderTranslator,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        let Some(translator) = (unsafe { translator.as_ref() }) else {
            instance.set_shader_translator(None);
            instance.set_shader_translation_enabled(false);
            return Status::Ok;
        };
        if translator.translate.is_none()
            || translator.release.is_none()
            || translator.destroy_user.is_none()
        {
            return Status::BadArgument;
        }
        instance.set_shader_translator(Some(Arc::new(CShaderTranslator(*translator))));
        instance.set_shader_translation_enabled(true);
        Status::Ok
    })
}

/// The program asked its bootstrap function for `name`. Records the request
/// and returns the function id, or `FUNCTION_NONE` for a name the library
/// does not know (which is recorded too).
///
/// # Safety
/// `instance` must be live; `name` must be null or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_request(
    instance: *const Instance,
    name: *const c_char,
) -> u32 {
    ffi_guard(FUNCTION_NONE, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return FUNCTION_NONE;
        };
        if name.is_null() {
            return FUNCTION_NONE;
        }
        let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        instance.request(&name).map_or(FUNCTION_NONE, |id| id.0)
    })
}

/// The program called `function`. `registers` holds its argument registers
/// on entry and receives its result registers.
///
/// # Safety
/// `instance` must be live. `registers` must point to a valid `Registers`
/// that nothing else reads or writes until the call returns. Calls on one
/// instance may run on several threads at once, each with its own registers.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_call(
    instance: *const Instance,
    function: u32,
    registers: *mut Registers,
) -> Status {
    ffi_guard(Status::InternalError, || {
        match unsafe { (instance.as_ref(), registers.as_mut()) } {
            (Some(instance), Some(registers)) => instance.call(FunctionId(function), registers),
            _ => Status::BadArgument,
        }
    })
}

/// The original implementation of `function` returned. `registers` holds
/// its result registers. Only hosts that let the original implementation run
/// call this; it lets results be sampled along with arguments.
///
/// # Safety
/// `instance` must be live and `registers` must point to a valid `Registers`.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_returned(
    instance: *const Instance,
    function: u32,
    registers: *const Registers,
) -> Status {
    ffi_guard(Status::InternalError, || {
        match unsafe { (instance.as_ref(), registers.as_ref()) } {
            (Some(instance), Some(registers)) => instance.returned(FunctionId(function), registers),
            _ => Status::BadArgument,
        }
    })
}

/// Write the sampled argument and result shapes to a text file.
///
/// # Safety
/// `instance` must be live; `path` must be NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_write_shapes(
    instance: *const Instance,
    path: *const c_char,
) -> Status {
    ffi_guard(Status::InternalError, || {
        // SAFETY: the caller passes a live instance.
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        if path.is_null() {
            return Status::BadArgument;
        }
        // SAFETY: the caller passes a NUL-terminated string.
        let path = unsafe { CStr::from_ptr(path) }.to_string_lossy();
        match std::fs::write(&*path, instance.shapes_report()) {
            Ok(()) => Status::Ok,
            Err(_) => Status::BadArgument,
        }
    })
}

/// How many times `function` was called on this instance.
///
/// # Safety
/// `instance` must be null or live.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_call_count(
    instance: *const Instance,
    function: u32,
) -> u64 {
    ffi_guard(0, || {
        unsafe { instance.as_ref() }.map_or(0, |instance| instance.call_count(FunctionId(function)))
    })
}

/// Write the census (what was requested and called) to a text file.
/// Returns `Status::Ok`, or `Status::BadArgument` when the file could not be
/// written.
///
/// # Safety
/// `instance` must be live; `path` must be NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn swxbx_instance_write_census(
    instance: *const Instance,
    path: *const c_char,
) -> Status {
    ffi_guard(Status::InternalError, || {
        // SAFETY: the caller passes a live instance.
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        if path.is_null() {
            return Status::BadArgument;
        }
        // SAFETY: the caller passes a NUL-terminated string.
        let path = unsafe { CStr::from_ptr(path) }.to_string_lossy();
        match std::fs::write(&*path, instance.census().to_string()) {
            Ok(()) => Status::Ok,
            Err(_) => Status::BadArgument,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TranslatorCounters {
        releases: Arc<AtomicUsize>,
        destroys: Arc<AtomicUsize>,
    }

    unsafe extern "C" fn translate_shader(
        _: *mut c_void,
        stage: ShaderStage,
        code: *const u8,
        code_size: u64,
        output: *mut HostTranslatedShader,
    ) -> i32 {
        assert_eq!(stage, 7);
        let code = unsafe { std::slice::from_raw_parts(code, code_size as usize) };
        assert_eq!(code, b"guest shader");
        let words = vec![0x0723_0203, 7, code.len() as u32].into_boxed_slice();
        let output_words = words.as_ptr();
        let output_count = words.len() as u64;
        std::mem::forget(words);
        unsafe {
            *output = HostTranslatedShader {
                words: output_words,
                word_count: output_count,
            };
        }
        0
    }

    unsafe extern "C" fn release_shader(user: *mut c_void, words: *const u32, word_count: u64) {
        let counters = unsafe { &*user.cast::<TranslatorCounters>() };
        counters.releases.fetch_add(1, Ordering::SeqCst);
        let words = std::ptr::slice_from_raw_parts_mut(words.cast_mut(), word_count as usize);
        drop(unsafe { Box::from_raw(words) });
    }

    unsafe extern "C" fn destroy_translator(user: *mut c_void) {
        let counters = unsafe { Box::from_raw(user.cast::<TranslatorCounters>()) };
        counters.destroys.fetch_add(1, Ordering::SeqCst);
    }

    unsafe extern "C" fn oversized_shader(
        _: *mut c_void,
        _: ShaderStage,
        _: *const u8,
        _: u64,
        output: *mut HostTranslatedShader,
    ) -> i32 {
        unsafe {
            *output = HostTranslatedShader {
                words: std::ptr::NonNull::<u32>::dangling().as_ptr(),
                word_count: MAX_TRANSLATED_SHADER_WORDS + 1,
            };
        }
        0
    }

    unsafe extern "C" fn release_oversized(user: *mut c_void, _: *const u32, _: u64) {
        let counters = unsafe { &*user.cast::<TranslatorCounters>() };
        counters.releases.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn version_is_a_c_string_matching_the_package() {
        // SAFETY: `swxbx_version` returns a static NUL-terminated string.
        let version = unsafe { CStr::from_ptr(swxbx_version()) };
        assert_eq!(version.to_str(), Ok(env!("CARGO_PKG_VERSION")));
        assert_eq!(swxbx_host_interface_version(), HOST_INTERFACE_VERSION);
    }

    #[test]
    fn a_panic_is_reported_and_cleared_by_the_next_call() {
        let status = ffi_guard(Status::Ok, || -> Status { panic!("boom") });
        assert_eq!(status, Status::Ok);
        // SAFETY: a non-null result is a NUL-terminated string owned by the
        // library until the next call on this thread.
        let message = unsafe { CStr::from_ptr(swxbx_last_error()) };
        assert_eq!(message.to_str(), Ok("boom"));
        swxbx_version();
        assert!(swxbx_last_error().is_null());
    }

    #[test]
    fn c_interface_round_trips_names_and_counts_calls() {
        let count = swxbx_function_count();
        assert!(count > 0);
        // SAFETY: ids below the count have names; the strings are static.
        let first = unsafe { CStr::from_ptr(swxbx_function_name(0)) }.to_owned();
        assert_eq!(unsafe { swxbx_function_lookup(first.as_ptr()) }, 0);
        assert!(swxbx_function_name(count).is_null());

        let unknown = CString::new("notAFunction").unwrap();
        assert_eq!(
            unsafe { swxbx_function_lookup(unknown.as_ptr()) },
            FUNCTION_NONE
        );

        // SAFETY: a null host is allowed, and the instance is destroyed below.
        let instance = unsafe { swxbx_instance_create(std::ptr::null()) };
        assert_eq!(
            unsafe { swxbx_instance_request(instance, first.as_ptr()) },
            0
        );
        assert_eq!(
            unsafe { swxbx_instance_request(instance, unknown.as_ptr()) },
            FUNCTION_NONE
        );

        let mut registers = Registers::default();
        registers.x[0] = 7;
        let status = unsafe { swxbx_instance_call(instance, 0, &mut registers) };
        assert_eq!(status, Status::Unimplemented);
        assert_eq!(registers.x[0], 0);
        assert_eq!(unsafe { swxbx_instance_call_count(instance, 0) }, 1);
        assert_eq!(
            unsafe { swxbx_instance_call(instance, count, &mut registers) },
            Status::BadFunction
        );

        unsafe { swxbx_instance_destroy(instance) };
    }

    #[test]
    fn c_translator_copies_releases_and_owns_its_context() {
        let releases = Arc::new(AtomicUsize::new(0));
        let destroys = Arc::new(AtomicUsize::new(0));
        let user = Box::into_raw(Box::new(TranslatorCounters {
            releases: releases.clone(),
            destroys: destroys.clone(),
        }));
        let callbacks = HostShaderTranslator {
            user: user.cast(),
            translate: Some(translate_shader),
            release: Some(release_shader),
            destroy_user: Some(destroy_translator),
        };

        let instance = unsafe { swxbx_instance_create(std::ptr::null()) };
        assert_eq!(
            unsafe { swxbx_instance_set_shader_translator(instance, &callbacks) },
            Status::Ok
        );
        let installed = unsafe { &*instance }.shader_translator().unwrap();
        assert_eq!(
            installed.translate(7, b"guest shader").unwrap(),
            [0x0723_0203, 7, 12]
        );
        assert_eq!(releases.load(Ordering::SeqCst), 1);

        assert_eq!(
            unsafe { swxbx_instance_set_shader_translator(instance, std::ptr::null()) },
            Status::Ok
        );
        assert_eq!(destroys.load(Ordering::SeqCst), 0);
        drop(installed);
        assert_eq!(destroys.load(Ordering::SeqCst), 1);
        unsafe { swxbx_instance_destroy(instance) };
    }

    #[test]
    fn c_translator_rejects_an_incomplete_contract_without_taking_ownership() {
        let instance = unsafe { swxbx_instance_create(std::ptr::null()) };
        let callbacks = HostShaderTranslator::default();
        assert_eq!(
            unsafe { swxbx_instance_set_shader_translator(instance, &callbacks) },
            Status::BadArgument
        );
        assert!(unsafe { &*instance }.shader_translator().is_none());
        unsafe { swxbx_instance_destroy(instance) };
    }

    #[test]
    fn c_translator_releases_an_oversized_output_without_reading_it() {
        let releases = Arc::new(AtomicUsize::new(0));
        let destroys = Arc::new(AtomicUsize::new(0));
        let callbacks = HostShaderTranslator {
            user: Box::into_raw(Box::new(TranslatorCounters {
                releases: releases.clone(),
                destroys: destroys.clone(),
            }))
            .cast(),
            translate: Some(oversized_shader),
            release: Some(release_oversized),
            destroy_user: Some(destroy_translator),
        };
        let instance = unsafe { swxbx_instance_create(std::ptr::null()) };
        assert_eq!(
            unsafe { swxbx_instance_set_shader_translator(instance, &callbacks) },
            Status::Ok
        );
        let installed = unsafe { &*instance }.shader_translator().unwrap();
        assert_eq!(
            installed.translate(0, b"x").unwrap_err(),
            "host shader translation is too large"
        );
        assert_eq!(releases.load(Ordering::SeqCst), 1);
        drop(installed);
        unsafe { swxbx_instance_destroy(instance) };
        assert_eq!(destroys.load(Ordering::SeqCst), 1);
    }
}
