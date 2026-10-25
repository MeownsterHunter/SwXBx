//! Synthetic presentation and arena tests.
#![cfg(feature = "vulkan")]

use ash::vk;
use std::{ffi::c_void, sync::Mutex};
use swxbx::{functions, gpu::Backend, Host, Instance, Registers, Status};

struct State {
    memory: Mutex<Vec<u8>>,
    frames: Mutex<Vec<(u32, u32, Vec<u8>)>>,
}

unsafe extern "C" fn read(user: *mut c_void, address: u64, out: *mut u8, size: u64) -> i32 {
    let state = unsafe { &*user.cast::<State>() };
    let memory = state.memory.lock().unwrap();
    let Some(end) = address.checked_add(size) else {
        return 1;
    };
    let Some(bytes) = memory.get(address as usize..end as usize) else {
        return 1;
    };
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
    }
    0
}

unsafe extern "C" fn write(user: *mut c_void, address: u64, data: *const u8, size: u64) -> i32 {
    let state = unsafe { &*user.cast::<State>() };
    let mut memory = state.memory.lock().unwrap();
    let Some(end) = address.checked_add(size) else {
        return 1;
    };
    let Some(bytes) = memory.get_mut(address as usize..end as usize) else {
        return 1;
    };
    unsafe {
        std::ptr::copy_nonoverlapping(data, bytes.as_mut_ptr(), bytes.len());
    }
    0
}

unsafe extern "C" fn present(
    user: *mut c_void,
    _window: u64,
    width: u32,
    height: u32,
    rgba: *const u8,
    stride: u64,
) {
    let state = unsafe { &*user.cast::<State>() };
    assert_eq!(stride, u64::from(width) * 4);
    let bytes = unsafe { std::slice::from_raw_parts(rgba, (stride * u64::from(height)) as usize) };
    state
        .frames
        .lock()
        .unwrap()
        .push((width, height, bytes.to_vec()));
}

fn call(instance: &Instance, name: &str, args: &[u64]) -> u64 {
    let mut registers = Registers::default();
    registers.x[..args.len()].copy_from_slice(args);
    assert_eq!(
        instance.call(functions::lookup(name).unwrap(), &mut registers),
        Status::Ok,
        "{name}"
    );
    if name.get(3..) == Some("QueuePresentTexture") {
        assert_eq!(instance.poll_presentations(true), Status::Ok);
    }
    registers.x[0]
}

fn put(state: &State, address: usize, bytes: &[u8]) {
    state.memory.lock().unwrap()[address..address + bytes.len()].copy_from_slice(bytes);
}

fn record(instance: &Instance, state: &State, texture: u64, color: [f32; 4], mask: u64) {
    put(state, 0x100, &texture.to_le_bytes());
    put(
        state,
        0x200,
        &color
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    call(instance, "nvnCommandBufferBeginRecording", &[7]);
    call(
        instance,
        "nvnCommandBufferSetRenderTargets",
        &[7, 1, 0x100, 0, 0],
    );
    // State records share the recording with clears and remain unexecuted.
    call(instance, "nvnCommandBufferBindBlendState", &[7, 0x700]);
    call(instance, "nvnCommandBufferBarrier", &[7, 0x12]);
    call(instance, "nvnCommandBufferClearColor", &[7, 0, 0x200, mask]);
    let handle = call(instance, "nvnCommandBufferEndRecording", &[7]);
    put(state, 0x300, &handle.to_le_bytes());
    call(instance, "nvnQueueSubmitCommands", &[0, 1, 0x300]);
}

fn copy_texture(instance: &Instance, state: &State, source: u64, destination: u64) {
    call(instance, "nvnCommandBufferBeginRecording", &[7]);
    call(
        instance,
        "nvnCommandBufferCopyTextureToTexture",
        &[7, source, destination],
    );
    let handle = call(instance, "nvnCommandBufferEndRecording", &[7]);
    put(state, 0x300, &handle.to_le_bytes());
    call(instance, "nvnQueueSubmitCommands", &[0, 1, 0x300]);
}

#[test]
#[ignore = "requires a Vulkan device; CI runs with a software driver and no display"]
fn api_clear_copy_and_present_share_arena_bytes() {
    let state = Box::new(State {
        memory: Mutex::new(vec![0; 0x2000]),
        frames: Mutex::new(Vec::new()),
    });
    let host = Host {
        user: std::ptr::from_ref(&*state).cast_mut().cast(),
        read_memory: Some(read),
        write_memory: Some(write),
        present: Some(present),
        render_scale: 2.0,
        ..Host::default()
    };
    // State and callbacks remain live until the instance is destroyed.
    let instance = unsafe { Instance::with_host(host) };
    call(&instance, "nvnMemoryPoolBuilderSetDefaults", &[1]);
    call(
        &instance,
        "nvnMemoryPoolBuilderSetStorage",
        &[1, 0x1000, 256],
    );
    assert_eq!(call(&instance, "nvnMemoryPoolInitialize", &[2, 1]), 1);
    let base = call(&instance, "nvnMemoryPoolGetBufferAddress", &[2]);
    assert_eq!(
        base,
        swxbx::global_memory::GUEST_BASE,
        "Vulkan must be active"
    );
    for (texture, offset) in [(4, 16), (5, 64), (6, 16)] {
        call(&instance, "nvnTextureBuilderSetDefaults", &[3]);
        call(&instance, "nvnTextureBuilderSetSize2D", &[3, 3, 2]);
        call(&instance, "nvnTextureBuilderSetStorage", &[3, 2, offset]);
        assert_eq!(call(&instance, "nvnTextureInitialize", &[texture, 3]), 1);
        assert_eq!(
            call(&instance, "nvnTextureGetTextureAddress", &[texture]),
            base + offset
        );
    }
    put(&state, 0x100, &5_u64.to_le_bytes());
    call(&instance, "nvnWindowBuilderSetDefaults", &[8]);
    call(&instance, "nvnWindowBuilderSetTextures", &[8, 1, 0x100]);
    call(&instance, "nvnWindowInitialize", &[9, 8]);
    call(&instance, "nvnCommandBufferInitialize", &[7, 0]);

    // Texture 4 is not in the window. Its clear still writes canonical arena bytes.
    record(&instance, &state, 4, [1.0, 0.5, 0.25, 1.0], 15);
    // A second object aliases the same bytes. Its channel mask preserves the rest.
    record(&instance, &state, 6, [0.0, 0.0, 0.0, 0.0], 1);
    copy_texture(&instance, &state, 4, 5);
    call(&instance, "nvnQueuePresentTexture", &[0, 9, 0]);
    {
        let memory = state.memory.lock().unwrap();
        for offset in [16, 64] {
            assert!(memory[0x1000 + offset..0x1000 + offset + 24]
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .all(|p| p == [0, 128, 64, 255]));
        }
        assert!(memory[0x1000..0x1010].iter().all(|&b| b == 0));
        assert!(memory[0x1010 + 24..0x1040].iter().all(|&b| b == 0));
    }
    {
        let frames = state.frames.lock().unwrap();
        assert_eq!((frames[0].0, frames[0].1), (6, 4));
        assert!(frames[0]
            .2
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .all(|p| p == [0, 128, 64, 255]));
    }
    // Repeated clears and presents exercise command/fence reuse and retained layouts.
    for frame in 0..48 {
        let color = if frame % 2 == 0 {
            [0.0, 0.0, 1.0, 1.0]
        } else {
            [1.0, 0.0, 0.0, 1.0]
        };
        record(&instance, &state, 5, color, 15);
        call(&instance, "nvnQueuePresentTexture", &[0, 9, 0]);
        let frames = state.frames.lock().unwrap();
        let expected = if frame % 2 == 0 {
            [0, 0, 255, 255]
        } else {
            [255, 0, 0, 255]
        };
        assert!(frames
            .last()
            .unwrap()
            .2
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .all(|p| p == expected));
    }
    // A buffer copy must observe an earlier image clear in the same recording.
    put(&state, 0x100, &4_u64.to_le_bytes());
    let color: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
    put(
        &state,
        0x200,
        &color
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    call(&instance, "nvnCommandBufferBeginRecording", &[7]);
    call(
        &instance,
        "nvnCommandBufferSetRenderTargets",
        &[7, 1, 0x100, 0, 0],
    );
    call(&instance, "nvnCommandBufferClearColor", &[7, 0, 0x200, 15]);
    call(
        &instance,
        "nvnCommandBufferCopyBufferToTexture",
        &[7, base + 16, 5],
    );
    let handle = call(&instance, "nvnCommandBufferEndRecording", &[7]);
    put(&state, 0x300, &handle.to_le_bytes());
    call(&instance, "nvnQueueSubmitCommands", &[0, 1, 0x300]);
    call(&instance, "nvnQueuePresentTexture", &[0, 9, 0]);
    assert!(
        state
            .frames
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .2
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| *p == [0, 255, 0, 255]),
        "buffer copy must see the preceding clear's arena bytes"
    );
    put(&state, 0x100, &5_u64.to_le_bytes());

    let uploaded = [23, 45, 67, 89].repeat(6);
    put(&state, 0x10c0, &uploaded);
    call(&instance, "nvnCommandBufferBeginRecording", &[7]);
    call(
        &instance,
        "nvnCommandBufferCopyBufferToTexture",
        &[7, base + 192, 5],
    );
    let handle = call(&instance, "nvnCommandBufferEndRecording", &[7]);
    put(&state, 0x300, &handle.to_le_bytes());
    call(&instance, "nvnQueueSubmitCommands", &[0, 1, 0x300]);
    call(&instance, "nvnQueuePresentTexture", &[0, 9, 0]);
    assert!(state
        .frames
        .lock()
        .unwrap()
        .last()
        .unwrap()
        .2
        .as_chunks::<4>()
        .0
        .iter()
        .all(|p| *p == [23, 45, 67, 89]));

    // Depth has its own render-target slot, not the last colour image.
    call(&instance, "nvnTextureBuilderSetDefaults", &[3]);
    call(&instance, "nvnTextureBuilderSetSize2D", &[3, 3, 2]);
    call(&instance, "nvnTextureBuilderSetStorage", &[3, 2, 128]);
    call(&instance, "nvnTextureInitialize", &[10, 3]);
    call(&instance, "nvnCommandBufferBeginRecording", &[7]);
    call(
        &instance,
        "nvnCommandBufferSetRenderTargets",
        &[7, 1, 0x100, 0, 10],
    );
    let mut registers = Registers {
        x: [7, 0, 1, 0, 0, 0, 0, 0],
        ..Registers::default()
    };
    registers.d[0] = u64::from(0.25_f32.to_bits());
    assert_eq!(
        instance.call(
            functions::lookup("nvnCommandBufferClearDepthStencil").unwrap(),
            &mut registers
        ),
        Status::Ok
    );
    let handle = call(&instance, "nvnCommandBufferEndRecording", &[7]);
    put(&state, 0x300, &handle.to_le_bytes());
    call(&instance, "nvnQueueSubmitCommands", &[0, 1, 0x300]);
    let finish = functions::all()
        .find(|(_, name)| name.ends_with("QueueFinish"))
        .unwrap()
        .1;
    call(&instance, finish, &[0]);
    assert!(state.memory.lock().unwrap()[0x1080..0x1098]
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .all(|p| p == 0.25_f32.to_le_bytes()));
    call(&instance, "nvnQueueFinish", &[0]);
    call(&instance, "nvnWindowFinalize", &[9]);
    for texture in [4, 5, 6, 10] {
        call(&instance, "nvnTextureFinalize", &[texture]);
    }
    call(&instance, "nvnMemoryPoolFinalize", &[2]);
    drop(instance);
}

#[test]
#[ignore = "requires a Vulkan device; CI runs with a software driver and no display"]
fn offscreen_present_converts_bgra_and_srgb_and_scales() {
    let mut backend = Backend::new(1.0).expect("Vulkan device is required");
    assert!(backend.ensure(1, 3, 2, false));
    for frame in 0..48 {
        assert!(backend.clear_color(1, [1.0, 0.5, 0.25, 1.0], 15));
        let format = if frame % 2 == 0 {
            vk::Format::B8G8R8A8_UNORM
        } else {
            vk::Format::B8G8R8A8_SRGB
        };
        let width = if frame % 3 == 0 { 7 } else { 2 };
        let (w, h, bytes) = backend
            .present_offscreen(1, format, width, 5)
            .expect("offscreen present");
        assert_eq!((w, h), (width, 5));
        let expected: [u8; 4] = if frame % 2 == 0 {
            [64, 128, 255, 255]
        } else {
            [137, 188, 255, 255]
        };
        assert_eq!(bytes.len(), (width * 5 * 4) as usize);
        for pixel in bytes.as_chunks::<4>().0 {
            for (actual, wanted) in pixel.iter().zip(expected) {
                assert!(actual.abs_diff(wanted) <= 1, "{pixel:?} vs {expected:?}");
            }
        }
    }
    // Invalid destinations are rejected before recording a blit.
    assert!(backend
        .present_offscreen(1, vk::Format::D32_SFLOAT, 3, 2)
        .is_none());
    assert!(backend
        .present_offscreen(1, vk::Format::R8G8B8A8_UNORM, 0, 2)
        .is_none());
}
