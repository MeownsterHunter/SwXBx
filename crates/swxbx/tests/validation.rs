//! Strict validation failure checks against a live messenger.
#![cfg(feature = "vulkan")]

use ash::vk;
use std::process::Command;

#[test]
#[ignore = "requires a Vulkan device and validation layer"]
fn validation_message_probe() {
    let Ok(severity) = std::env::var("VALIDATION_PROBE") else {
        return;
    };
    let context = swxbx::gpu::Context::new().expect("Vulkan validation context required");
    let loader = ash::ext::debug_utils::Instance::new(&context.entry, &context.instance);
    let severity = match severity.as_str() {
        "warning" => vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
        "error" => vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        _ => panic!("invalid probe severity"),
    };
    unsafe {
        loader.submit_debug_utils_message(
            severity,
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            &vk::DebugUtilsMessengerCallbackDataEXT::default()
                .message_id_name(c"synthetic-validation-probe")
                .message(c"synthetic validation message"),
        );
    }
    drop(context);
}

#[test]
#[ignore = "requires a Vulkan device and validation layer"]
fn warnings_and_errors_fail_the_process() {
    for severity in ["warning", "error"] {
        let result = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "validation_message_probe",
                "--ignored",
                "--nocapture",
            ])
            .env("VULKAN_VALIDATION", "sync")
            .env("VALIDATION_PROBE", severity)
            .output()
            .expect("validation probe must run");
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            !result.status.success(),
            "validation message must fail the test"
        );
        assert!(stderr.contains("synthetic validation message"), "{stderr}");
        assert!(
            stderr.contains("Vulkan validation reported 1 messages"),
            "{stderr}"
        );
    }
}
