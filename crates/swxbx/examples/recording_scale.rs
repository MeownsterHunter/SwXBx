//! Synthetic recording and ordered clear workload.
use std::{
    ffi::c_void,
    sync::{
        atomic::{AtomicU64, Ordering},
        Barrier,
    },
    time::Instant,
};
use swxbx::{functions, Host, Instance, Registers, Status};
const COMMANDS: usize = 8192;
const FRAMES: usize = 240;
struct Memory {
    handles: [AtomicU64; 8],
}
unsafe extern "C" fn read(user: *mut c_void, address: u64, out: *mut u8, size: u64) -> i32 {
    let memory = unsafe { &*user.cast::<Memory>() };
    let bytes = match address {
        0x100..=0x138 if address.is_multiple_of(8) => memory.handles
            [((address - 0x100) / 8) as usize]
            .load(Ordering::Acquire)
            .to_le_bytes()
            .to_vec(),
        0x200 => [0.25_f32, 0.5, 0.75, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
        0x300 => 0x500_u64.to_le_bytes().to_vec(),
        _ => return 1,
    };
    if size as usize > bytes.len() {
        return 1;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, size as usize);
    }
    0
}
fn function(suffix: &str) -> functions::FunctionId {
    functions::all()
        .find(|(_, name)| name.ends_with(suffix))
        .unwrap()
        .0
}
fn call(instance: &Instance, id: functions::FunctionId, args: &[u64]) -> u64 {
    let mut registers = Registers::default();
    registers.x[..args.len()].copy_from_slice(args);
    assert_eq!(instance.call(id, &mut registers), Status::Ok);
    registers.x[0]
}
#[repr(C)]
struct Timespec {
    seconds: i64,
    nanos: i64,
}
unsafe extern "C" {
    fn clock_gettime(clock: i32, value: *mut Timespec) -> i32;
}
fn cpu_seconds() -> f64 {
    let mut time = Timespec {
        seconds: 0,
        nanos: 0,
    };
    // Process CPU time includes every recording and execution thread.
    assert_eq!(unsafe { clock_gettime(2, &mut time) }, 0);
    time.seconds as f64 + time.nanos as f64 * 1e-9
}
fn main() {
    println!("threads,cpu_ms_per_frame,wall_ms_per_frame,frames_per_second");
    for threads in [1, 2, 4, 8] {
        let memory = Memory {
            handles: std::array::from_fn(|_| AtomicU64::new(0)),
        };
        // The fixture remains live through instance shutdown. Reads are thread safe.
        let instance = unsafe {
            Instance::with_host(Host {
                user: (&memory as *const Memory).cast_mut().cast(),
                read_memory: Some(read),
                ..Host::default()
            })
        };
        instance
            .graphics_cache_stats()
            .expect("Vulkan execution required");
        let init = function("CommandBufferInitialize");
        let begin = function("CommandBufferBeginRecording");
        let viewport = function("CommandBufferSetViewport");
        let target = function("CommandBufferSetRenderTargets");
        let clear = function("CommandBufferClearColor");
        let end = function("CommandBufferEndRecording");
        let submit = function("QueueSubmitCommands");
        let finish = function("QueueFinish");
        call(&instance, function("TextureBuilderSetDefaults"), &[0x400]);
        call(
            &instance,
            function("TextureBuilderSetSize2D"),
            &[0x400, 16, 16],
        );
        call(&instance, function("TextureInitialize"), &[0x500, 0x400]);
        for thread in 0..threads {
            call(&instance, init, &[0x1000 + thread as u64 * 0x100, 0]);
        }
        let barrier = Barrier::new(threads + 1);
        std::thread::scope(|scope| {
            for thread in 0..threads {
                let instance = &instance;
                let memory = &memory;
                let barrier = &barrier;
                scope.spawn(move || {
                    let key = 0x1000 + thread as u64 * 0x100;
                    for _ in 0..FRAMES + 20 {
                        barrier.wait();
                        call(instance, begin, &[key]);
                        for _ in 0..COMMANDS / threads {
                            call(instance, viewport, &[key, 0, 0, 16, 16]);
                        }
                        call(instance, target, &[key, 1, 0x300, 0, 0, 0]);
                        call(instance, clear, &[key, 0, 0x200, 15]);
                        memory.handles[thread]
                            .store(call(instance, end, &[key]), Ordering::Release);
                        barrier.wait();
                    }
                });
            }
            let mut start = Instant::now();
            let mut cpu = cpu_seconds();
            for frame in 0..FRAMES + 20 {
                if frame == 20 {
                    start = Instant::now();
                    cpu = cpu_seconds();
                }
                barrier.wait();
                barrier.wait();
                call(&instance, submit, &[0x600, threads as u64, 0x100]);
                call(&instance, finish, &[0x600]);
            }
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "{threads},{:.6},{:.6},{:.3}",
                (cpu_seconds() - cpu) * 1000.0 / FRAMES as f64,
                elapsed * 1000.0 / FRAMES as f64,
                FRAMES as f64 / elapsed
            );
        });
    }
}
