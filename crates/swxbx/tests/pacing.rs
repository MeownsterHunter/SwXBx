//! Original callback ordering and GPU retirement checks. Source: the public contract.

use std::{ffi::c_void, sync::Mutex};
use swxbx::{Host, Instance, PresentationConfig, PresentationMode, Registers, Status};

struct State {
    memory: Mutex<Vec<u8>>,
    frames: Mutex<Vec<(u64, Vec<u8>)>>,
    vblanks: Mutex<usize>,
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

unsafe extern "C" fn present(
    user: *mut c_void,
    window: u64,
    width: u32,
    height: u32,
    rgba: *const u8,
    stride: u64,
) {
    let state = unsafe { &*user.cast::<State>() };
    assert_eq!((width, height, stride), (2, 2, 8));
    let bytes = unsafe { std::slice::from_raw_parts(rgba, (stride * u64::from(height)) as usize) };
    assert!(bytes.as_chunks::<4>().0.iter().all(|p| p == &bytes[..4]));
    state.frames.lock().unwrap().push((window, bytes.to_vec()));
}

unsafe extern "C" fn vblank(user: *mut c_void) {
    let state = unsafe { &*user.cast::<State>() };
    *state.vblanks.lock().unwrap() += 1;
}

fn call(instance: &Instance, suffix: &str, args: &[u64]) -> u64 {
    let id = swxbx::functions::all()
        .find(|(_, name)| name.get(3..) == Some(suffix))
        .unwrap()
        .0;
    let mut r = Registers::default();
    r.x[..args.len()].copy_from_slice(args);
    assert_eq!(instance.call(id, &mut r), Status::Ok, "{suffix}");
    r.x[0]
}

fn setup(state: &State) -> Instance {
    let instance = unsafe {
        Instance::with_host(Host {
            user: std::ptr::from_ref(state).cast_mut().cast(),
            read_memory: Some(read),
            write_memory: Some(write),
            present: Some(present),
            wait_vblank: Some(vblank),
            ..Host::default()
        })
    };
    call(&instance, "MemoryPoolBuilderSetDefaults", &[1]);
    call(&instance, "MemoryPoolBuilderSetStorage", &[1, 0x1000, 256]);
    assert_eq!(call(&instance, "MemoryPoolInitialize", &[2, 1]), 1);
    #[cfg(feature = "vulkan")]
    assert_eq!(
        call(&instance, "MemoryPoolGetBufferAddress", &[2]),
        swxbx::global_memory::GUEST_BASE,
        "GPU must be active"
    );
    call(&instance, "TextureBuilderSetDefaults", &[3]);
    call(&instance, "TextureBuilderSetSize2D", &[3, 2, 2]);
    call(&instance, "TextureBuilderSetStorage", &[3, 2, 0]);
    assert_eq!(call(&instance, "TextureInitialize", &[4, 3]), 1);
    state.memory.lock().unwrap()[0x100..0x108].copy_from_slice(&4_u64.to_le_bytes());
    for window in [9, 10] {
        call(&instance, "WindowBuilderSetDefaults", &[8]);
        call(&instance, "WindowBuilderSetTextures", &[8, 1, 0x100]);
        call(&instance, "WindowInitialize", &[window, 8]);
    }
    call(&instance, "CommandBufferInitialize", &[7, 0]);
    instance
}

fn frame(instance: &Instance, state: &State, value: u8, window: u64) {
    let color = [f32::from(value) / 255.0, 0.0, 0.0, 1.0];
    state.memory.lock().unwrap()[0x200..0x210].copy_from_slice(
        &color
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    call(instance, "CommandBufferBeginRecording", &[7]);
    call(
        instance,
        "CommandBufferSetRenderTargets",
        &[7, 1, 0x100, 0, 0],
    );
    call(instance, "CommandBufferClearColor", &[7, 0, 0x200, 15]);
    let handle = call(instance, "CommandBufferEndRecording", &[7]);
    state.memory.lock().unwrap()[0x300..0x308].copy_from_slice(&handle.to_le_bytes());
    call(instance, "QueueSubmitCommands", &[0, 1, 0x300]);
    call(instance, "QueuePresentTexture", &[0, window, 0]);
}

fn state() -> State {
    State {
        memory: Mutex::new(vec![0; 0x2000]),
        frames: Mutex::default(),
        vblanks: Mutex::default(),
    }
}

#[test]
#[cfg_attr(feature = "vulkan", ignore = "requires a GPU")]
fn fifo_preserves_snapshots_and_bounds_backpressure() {
    let state = state();
    let instance = setup(&state);
    assert_eq!(PresentationConfig::default().frames_in_flight, 2);
    for capacity in [1, 2, 4, 16] {
        assert_eq!(
            instance.configure_presentation(PresentationConfig {
                frames_in_flight: capacity,
                mode: PresentationMode::Fifo
            }),
            Status::Ok
        );
        let before = state.frames.lock().unwrap().len();
        for value in 0..40 {
            frame(&instance, &state, value, 9);
        }
        assert_eq!(instance.frame_statistics().pending, capacity);
        assert_eq!(instance.poll_presentations(true), Status::Ok);
        let frames = state.frames.lock().unwrap();
        assert_eq!(frames.len() - before, 40);
        for (value, (_, bytes)) in frames[before..].iter().enumerate() {
            assert_eq!(&bytes[..4], &[value as u8, 0, 0, 255]);
        }
    }
    let stats = instance.frame_statistics();
    assert_eq!(
        (
            stats.submitted,
            stats.delivered,
            stats.dropped,
            stats.failed,
            stats.pending,
            stats.peak_pending
        ),
        (160, 160, 0, 0, 0, 16)
    );
    assert_eq!(*state.vblanks.lock().unwrap(), 160);
    assert!(stats.mean_latency_ns > 0.0);
    assert!(stats.variance_latency_ns2.is_finite() && stats.variance_latency_ns2 >= 0.0);
}

#[test]
#[cfg_attr(feature = "vulkan", ignore = "requires a GPU")]
fn mailbox_replaces_only_the_same_window_and_policy_changes_drain() {
    let state = state();
    let instance = setup(&state);
    assert_eq!(
        instance.configure_presentation(PresentationConfig {
            frames_in_flight: 2,
            mode: PresentationMode::Mailbox
        }),
        Status::Ok
    );
    frame(&instance, &state, 10, 9);
    frame(&instance, &state, 20, 10);
    frame(&instance, &state, 30, 9);
    assert!(state.frames.lock().unwrap().is_empty());
    assert_eq!(instance.frame_statistics().dropped, 1);
    assert_eq!(
        instance.configure_presentation(PresentationConfig::default()),
        Status::Ok
    );
    let frames = state.frames.lock().unwrap();
    assert_eq!(
        frames
            .iter()
            .map(|(window, bytes)| (*window, bytes[0]))
            .collect::<Vec<_>>(),
        [(10, 20), (9, 30)]
    );
    drop(frames);
    frame(&instance, &state, 40, 9);
    call(&instance, "QueueFinish", &[0]);
    frame(&instance, &state, 50, 9);
    drop(instance);
    assert_eq!(state.frames.lock().unwrap().len(), 4);
    assert_eq!(*state.vblanks.lock().unwrap(), 4);
}

#[test]
fn c_boundary_rejects_invalid_policy_and_null_statistics() {
    let instance = Instance::new();
    unsafe {
        assert_eq!(
            swxbx::swxbx_instance_set_presentation(&instance, 0, 0),
            Status::BadArgument
        );
        assert_eq!(
            swxbx::swxbx_instance_set_presentation(&instance, 17, 0),
            Status::BadArgument
        );
        assert_eq!(
            swxbx::swxbx_instance_set_presentation(&instance, 2, 99),
            Status::BadArgument
        );
        assert_eq!(
            swxbx::swxbx_instance_poll_presentations(&instance, 2),
            Status::BadArgument
        );
        assert_eq!(
            swxbx::swxbx_instance_frame_statistics(&instance, std::ptr::null_mut()),
            Status::BadArgument
        );
        let mut stats = swxbx::FrameStatistics::default();
        assert_eq!(
            swxbx::swxbx_instance_frame_statistics(&instance, &mut stats),
            Status::Ok
        );
        assert_eq!(stats.submitted, 0);
    }
}
