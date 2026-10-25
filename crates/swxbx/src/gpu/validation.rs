//! Strict Vulkan validation for synthetic tests.

use ash::{vk, Entry, Instance};
use std::{
    ffi::{c_void, CStr},
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Copy)]
pub(super) enum Mode {
    Sync,
    Gpu,
}

impl Mode {
    pub fn from_env() -> Option<Self> {
        match std::env::var("VULKAN_VALIDATION") {
            Err(std::env::VarError::NotPresent) => None,
            Ok(value) if value == "sync" => Some(Self::Sync),
            Ok(value) if value == "gpu" => Some(Self::Gpu),
            _ => panic!("VULKAN_VALIDATION must be sync or gpu"),
        }
    }

    pub fn features(self) -> vk::ValidationFeaturesEXT<'static> {
        match self {
            Self::Sync => vk::ValidationFeaturesEXT::default().enabled_validation_features(&[
                vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
            ]),
            Self::Gpu => vk::ValidationFeaturesEXT::default()
                .enabled_validation_features(&[
                    vk::ValidationFeatureEnableEXT::GPU_ASSISTED,
                    vk::ValidationFeatureEnableEXT::GPU_ASSISTED_RESERVE_BINDING_SLOT,
                ])
                .disabled_validation_features(&[vk::ValidationFeatureDisableEXT::CORE_CHECKS]),
        }
    }

    pub fn settings(self, layer_version: u32) -> Vec<vk::LayerSettingEXT<'static>> {
        let mut settings = vec![(c"enable_message_limit", false)];
        match self {
            Self::Sync => settings.push((
                if layer_version >= vk::make_api_version(0, 1, 4, 363) {
                    c"syncval_full_validation"
                } else {
                    c"syncval_submit_time_validation"
                },
                true,
            )),
            Self::Gpu if layer_version >= vk::make_api_version(0, 1, 4, 357) => {
                settings.push((c"gpuav_mesh_shading", false));
                settings.push((c"gpuav_validate_trace_ray", false));
            }
            Self::Gpu => {}
        }
        settings
            .into_iter()
            .map(|(name, enabled)| vk::LayerSettingEXT {
                p_layer_name: c"VK_LAYER_KHRONOS_validation".as_ptr(),
                p_setting_name: name.as_ptr(),
                ty: vk::LayerSettingTypeEXT::BOOL32,
                value_count: 1,
                p_values: if enabled { &vk::TRUE } else { &vk::FALSE } as *const u32
                    as *const c_void,
                ..Default::default()
            })
            .collect()
    }
}

pub(super) struct Validation {
    messages: Box<AtomicUsize>,
    messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl Validation {
    pub fn new() -> Self {
        Self {
            messages: Box::new(AtomicUsize::new(0)),
            messenger: None,
        }
    }

    pub fn create_info(&self) -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
        vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(report))
            .user_data(&*self.messages as *const AtomicUsize as *mut c_void)
    }

    pub fn attach(&mut self, entry: &Entry, instance: &Instance) {
        let loader = ash::ext::debug_utils::Instance::new(entry, instance);
        let messenger = unsafe { loader.create_debug_utils_messenger(&self.create_info(), None) }
            .expect("validation messenger required");
        self.messenger = Some((loader, messenger));
    }

    pub fn detach(&mut self) {
        if let Some((loader, messenger)) = self.messenger.take() {
            unsafe { loader.destroy_debug_utils_messenger(messenger, None) };
        }
    }
}

impl Drop for Validation {
    fn drop(&mut self) {
        let count = self.messages.load(Ordering::Relaxed);
        if !std::thread::panicking() {
            assert_eq!(count, 0, "Vulkan validation reported {count} messages");
        }
    }
}

unsafe extern "system" fn report(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    // The boxed counter lives through instance destruction, including callbacks
    // during instance creation and device teardown. A callback must never unwind.
    let messages = &*(user as *const AtomicUsize);
    messages.fetch_add(1, Ordering::Relaxed);
    let message = CStr::from_ptr((*data).p_message).to_string_lossy();
    use std::io::Write;
    let severity = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        "error"
    } else {
        "warning"
    };
    let _ = writeln!(
        std::io::stderr().lock(),
        "Vulkan validation {severity}: {message}"
    );
    vk::FALSE
}
