//! Headless swapchain pacing and retirement checks.
#![cfg(feature = "vulkan")]

use ash::{vk, vk::Handle};
use std::{
    ffi::c_void,
    sync::{
        atomic::{AtomicU32, Ordering},
        Mutex,
    },
};
use swxbx::{Host, HostVulkan, Instance, PresentationConfig, PresentationMode, Registers, Status};

struct State {
    memory: Mutex<Vec<u8>>,
    extent: AtomicU32,
}

unsafe extern "C" fn read(user: *mut c_void, address: u64, out: *mut u8, size: u64) -> i32 {
    let state = unsafe { &*user.cast::<State>() };
    let memory = state.memory.lock().unwrap();
    let bytes = &memory[address as usize..(address + size) as usize];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
    }
    0
}
unsafe extern "C" fn write(user: *mut c_void, address: u64, data: *const u8, size: u64) -> i32 {
    let state = unsafe { &*user.cast::<State>() };
    let mut memory = state.memory.lock().unwrap();
    let bytes = &mut memory[address as usize..(address + size) as usize];
    unsafe {
        std::ptr::copy_nonoverlapping(data, bytes.as_mut_ptr(), bytes.len());
    }
    0
}
unsafe extern "C" fn surface(_: *mut c_void, instance: u64, _: u64, _: u64) -> u64 {
    let Ok(entry) = (unsafe { ash::Entry::load() }) else {
        return 0;
    };
    let instance =
        unsafe { ash::Instance::load(entry.static_fn(), vk::Instance::from_raw(instance)) };
    let loader = ash::ext::headless_surface::Instance::new(&entry, &instance);
    unsafe { loader.create_headless_surface(&vk::HeadlessSurfaceCreateInfoEXT::default(), None) }
        .map_or(0, |s| s.as_raw())
}
unsafe extern "C" fn size(user: *mut c_void, _: u64, width: *mut u32, height: *mut u32) {
    let state = unsafe { &*user.cast::<State>() };
    let extent = state.extent.load(Ordering::Relaxed);
    unsafe {
        *width = extent;
        *height = extent;
    }
}
fn call(instance: &Instance, suffix: &str, args: &[u64]) -> u64 {
    let id = swxbx::functions::all()
        .find(|(_, n)| n.get(3..) == Some(suffix))
        .unwrap()
        .0;
    let mut r = Registers::default();
    r.x[..args.len()].copy_from_slice(args);
    assert_eq!(instance.call(id, &mut r), Status::Ok, "{suffix}");
    r.x[0]
}

#[test]
#[ignore = "requires a GPU with headless surfaces and presentation maintenance"]
fn native_modes_resize_capacity_and_teardown() {
    let state = State {
        memory: Mutex::new(vec![0; 0x2000]),
        extent: AtomicU32::new(32),
    };
    let extensions = [
        ash::khr::surface::NAME.as_ptr(),
        ash::ext::headless_surface::NAME.as_ptr(),
    ];
    let native = HostVulkan {
        extension_count: 2,
        extensions: extensions.as_ptr(),
        create_surface: surface,
        drawable_size: size,
    };
    let instance = unsafe {
        Instance::with_host(Host {
            user: std::ptr::from_ref(&state).cast_mut().cast(),
            read_memory: Some(read),
            write_memory: Some(write),
            vulkan: &native,
            ..Host::default()
        })
    };
    call(&instance, "MemoryPoolBuilderSetDefaults", &[1]);
    call(&instance, "MemoryPoolBuilderSetStorage", &[1, 0x1000, 256]);
    call(&instance, "MemoryPoolInitialize", &[2, 1]);
    assert_eq!(
        call(&instance, "MemoryPoolGetBufferAddress", &[2]),
        swxbx::global_memory::GUEST_BASE
    );
    call(&instance, "TextureBuilderSetDefaults", &[3]);
    call(&instance, "TextureBuilderSetSize2D", &[3, 2, 2]);
    call(&instance, "TextureBuilderSetStorage", &[3, 2, 0]);
    call(&instance, "TextureInitialize", &[4, 3]);
    state.memory.lock().unwrap()[0x100..0x108].copy_from_slice(&4_u64.to_le_bytes());
    call(&instance, "WindowBuilderSetDefaults", &[8]);
    call(&instance, "WindowBuilderSetTextures", &[8, 1, 0x100]);
    assert_eq!(call(&instance, "WindowInitialize", &[9, 8]), 1);
    for mode in [PresentationMode::Fifo, PresentationMode::Mailbox] {
        for capacity in [1, 2, 4] {
            assert_eq!(
                instance.configure_presentation(PresentationConfig {
                    frames_in_flight: capacity,
                    mode
                }),
                Status::Ok
            );
            for frame in 0..12 {
                state
                    .extent
                    .store(if frame < 6 { 32 } else { 64 }, Ordering::Relaxed);
                call(&instance, "QueuePresentTexture", &[0, 9, 0]);
            }
        }
    }
    state.extent.store(0, Ordering::Relaxed);
    call(&instance, "QueuePresentTexture", &[0, 9, 0]);
    assert_eq!(instance.frame_statistics().skipped, 1);
    call(&instance, "QueueFinish", &[0]);
    assert_eq!(instance.frame_statistics().delivered, 72);
    call(&instance, "WindowFinalize", &[9]);
}
