//! Measure an original scene of small textured draws. Source: the public contract.
mod support;

use ash::vk;
use std::{
    fs,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use support::{call, compile, hosted, put, register_marker, Memory, Shaders};
use swxbx::{
    global_memory::GUEST_BASE,
    gpu::{
        graphics::{FirstDrawContract, PrimitiveTopology, VertexFormat},
        textures::{TextureContract, TextureMapping},
        uniforms::GraphicsStage,
    },
};

fn main() {
    let mut args = std::env::args().skip(1);
    let frames = args.next().map(|s| s.parse().unwrap()).unwrap_or(5);
    let draws = args.next().map(|s| s.parse().unwrap()).unwrap_or(10_000);
    let indexed = args.next().is_some_and(|s| s == "indexed");
    run(frames, draws, indexed);
}

fn run(frames: usize, draws_per_frame: usize, indexed: bool) {
    scene(frames, draws_per_frame, indexed, false);
}

fn scene(frames: usize, draws_per_frame: usize, indexed: bool, fail_tail: bool) {
    assert!(frames > 0 && draws_per_frame > 0 && draws_per_frame.is_multiple_of(100));
    let scratch = std::env::temp_dir().join(format!("draw-bench-{}", std::process::id()));
    fs::create_dir(&scratch).expect("create temporary shader directory");
    let vertex = compile(
        &scratch,
        "vert",
        "#version 450\nlayout(location=0) in vec4 p; void main(){ gl_Position=p; }",
    );
    let fragment = compile(
        &scratch,
        "frag",
        "#version 450\nlayout(set=1,binding=0) uniform texture2D t;
        layout(set=1,binding=1) uniform sampler s;
        layout(set=0,binding=0) uniform Bank { vec4 values[4096]; } bank;
        layout(location=0) out vec4 c;
        void main(){ c=texture(sampler2D(t,s), gl_FragCoord.xy/64.0)*bank.values[0]; }",
    );
    fs::remove_dir_all(&scratch).unwrap();

    // Host memory outlives the instance. Pool storage starts at byte 0x1000.
    let memory = Memory(Mutex::new(vec![0; 0x9000]), Mutex::default());
    let instance = hosted(&memory);
    assert!(
        instance.set_first_draw_contract(Some(FirstDrawContract {
            topologies: vec![(1, PrimitiveTopology::TriangleList)],
            attribute_formats: vec![(2, VertexFormat::Float4)],
            attribute_state_stride: None,
            stream_state_stride: None,
            index_u16: 6,
            index_u32: 7,
            cull_none: 3,
            color_formats: vec![(4, ash::vk::Format::R8G8B8A8_UNORM)],
            target_2d: 5,
            identity_swizzle: [0; 4],
            depth_raster: None,
        })),
        "a compatible Vulkan device is required"
    );
    instance
        .set_texture_contract(TextureContract {
            bindings: vec![TextureMapping {
                stage: 5,
                index: 3,
                target: GraphicsStage::Fragment,
                set: 1,
                image: 0,
                sampler: 1,
            }],
            filters: vec![(100, vk::Filter::NEAREST)],
            wraps: vec![(200, vk::SamplerAddressMode::CLAMP_TO_EDGE)],
            compare_disabled: 300,
            ..Default::default()
        })
        .unwrap();
    instance
        .set_uniform_buffer_contract(swxbx::gpu::uniforms::UniformBufferContract {
            bindings: vec![swxbx::gpu::uniforms::UniformBankMapping {
                stage: 5,
                index: 0,
                target: GraphicsStage::Fragment,
                bank: 0,
            }],
            storage_buffers: false,
        })
        .unwrap();
    instance.set_shader_translator(Some(Arc::new(Shaders { vertex, fragment })));
    instance.set_shader_translation_enabled(true);

    call(&instance, "MemoryPoolBuilderSetDefaults", &[1]);
    call(
        &instance,
        "MemoryPoolBuilderSetStorage",
        &[1, 0x1000, 0x8000],
    );
    assert_eq!(call(&instance, "MemoryPoolInitialize", &[2, 1]), 1);
    let base = call(&instance, "MemoryPoolGetBufferAddress", &[2]);
    assert_eq!(base, GUEST_BASE, "flat arena allocation must succeed");

    // Color target at pool offset 0x1000 and a 2 by 2 sampled texture at 0x6000.
    for (object, size, offset) in [
        (4, 64, 0x1000),
        (31, 2, 0x6000),
        (36, 2, 0x6100),
        (37, 2, 0x6200),
    ] {
        call(&instance, "TextureBuilderSetDefaults", &[3]);
        call(&instance, "TextureBuilderSetSize2D", &[3, size, size]);
        call(&instance, "TextureBuilderSetFormat", &[3, 4]);
        call(&instance, "TextureBuilderSetTarget", &[3, 5]);
        call(&instance, "TextureBuilderSetStorage", &[3, 2, offset]);
        assert_eq!(call(&instance, "TextureInitialize", &[object, 3]), 1);
    }
    put(
        &memory,
        0x7000,
        &[
            255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255,
        ],
    );
    for address in [0x7100, 0x7200] {
        put(
            &memory,
            address,
            &[
                128, 128, 128, 255, 128, 128, 128, 255, 128, 128, 128, 255, 128, 128, 128, 255,
            ],
        );
    }
    put(
        &memory,
        0x7800,
        &[1.0_f32; 4]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    put(
        &memory,
        0x7900,
        &[0.5_f32, 0.5, 0.5, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    put(
        &memory,
        0x1200,
        &[0_u16, 1, 2]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    put(&memory, 0x100, &4_u64.to_le_bytes());
    call(&instance, "WindowBuilderSetDefaults", &[8]);
    call(&instance, "WindowBuilderSetTextures", &[8, 1, 0x100]);
    call(&instance, "WindowInitialize", &[9, 8]);
    call(&instance, "CommandBufferInitialize", &[7, 0]);

    register_marker(&memory, 0x1300, 2);
    register_marker(&memory, 0x1500, 1);
    put(&memory, 0x400, &(base + 0x300).to_le_bytes());
    put(&memory, 0x440, &(base + 0x500).to_le_bytes());
    call(&instance, "ProgramInitialize", &[12, 0]);
    assert_eq!(call(&instance, "ProgramSetShaders", &[12, 2, 0x400]), 1);

    let vertices = [
        [-0.75_f32, -0.75, 0.0, 1.0],
        [0.75, -0.75, 0.0, 1.0],
        [0.0, 0.75, 0.0, 1.0],
    ];
    put(
        &memory,
        0x1080,
        &vertices
            .into_iter()
            .flatten()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    for (kind, object) in [
        ("VertexStreamState", 20),
        ("VertexAttribState", 21),
        ("ColorState", 22),
        ("DepthStencilState", 23),
        ("PolygonState", 24),
    ] {
        call(&instance, &format!("{kind}SetDefaults"), &[object]);
    }
    call(&instance, "VertexStreamStateSetStride", &[20, 16]);
    call(&instance, "VertexStreamStateSetDivisor", &[20, 0]);
    call(&instance, "VertexAttribStateSetFormat", &[21, 2, 0]);
    call(&instance, "VertexAttribStateSetStreamIndex", &[21, 0]);
    call(&instance, "ColorStateSetBlendEnable", &[22, 0, 0]);
    for setting in ["DepthTestEnable", "DepthWriteEnable", "StencilTestEnable"] {
        call(
            &instance,
            &format!("DepthStencilStateSet{setting}"),
            &[23, 0],
        );
    }
    call(&instance, "PolygonStateSetCullFace", &[24, 3]);
    call(&instance, "TexturePoolInitialize", &[32, 2, 0, 512]);
    call(&instance, "TexturePoolRegisterTexture", &[32, 256, 31, 0]);
    call(&instance, "TexturePoolRegisterTexture", &[32, 258, 36, 0]);
    call(&instance, "TexturePoolRegisterTexture", &[32, 259, 37, 0]);
    call(&instance, "SamplerPoolInitialize", &[33, 2, 0, 512]);
    call(&instance, "SamplerBuilderSetDefaults", &[34]);
    call(&instance, "SamplerBuilderSetMinMagFilter", &[34, 100, 100]);
    call(&instance, "SamplerBuilderSetWrapMode", &[34, 200, 200, 200]);
    call(&instance, "SamplerBuilderSetCompare", &[34, 300, 0]);
    let id = swxbx::functions::all()
        .find(|(_, name)| name.get(3..) == Some("SamplerBuilderSetMaxAnisotropy"))
        .unwrap()
        .0;
    let mut registers = swxbx::Registers::default();
    registers.x[0] = 34;
    registers.d[0] = u64::from(1.0_f32.to_bits());
    assert_eq!(instance.call(id, &mut registers), swxbx::Status::Ok);
    call(&instance, "SamplerInitialize", &[35, 34]);
    call(&instance, "SamplerPoolRegisterSampler", &[33, 257, 35]);
    put(
        &memory,
        0x200,
        &[0.0_f32, 0.0, 1.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );

    // The first submission can skip a draw while its pipeline compiles.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        call(&instance, "CommandBufferBeginRecording", &[7]);
        call(
            &instance,
            "CommandBufferSetRenderTargets",
            &[7, 1, 0x100, 0, 0],
        );
        call(&instance, "CommandBufferClearColor", &[7, 0, 0x200, 15]);
        call(&instance, "CommandBufferBindProgram", &[7, 12, 0xffff]);
        for (kind, object) in [
            ("ColorState", 22),
            ("DepthStencilState", 23),
            ("PolygonState", 24),
        ] {
            call(&instance, &format!("CommandBufferBind{kind}"), &[7, object]);
        }
        call(&instance, "CommandBufferSetViewport", &[7, 0, 0, 64, 64]);
        call(&instance, "CommandBufferSetScissor", &[7, 0, 0, 64, 64]);
        call(
            &instance,
            "CommandBufferBindVertexBuffer",
            &[7, 0, base + 0x80, 48],
        );
        call(&instance, "CommandBufferBindVertexStreamState", &[7, 1, 20]);
        call(&instance, "CommandBufferBindVertexAttribState", &[7, 1, 21]);
        call(&instance, "CommandBufferSetTexturePool", &[7, 32]);
        call(&instance, "CommandBufferSetSamplerPool", &[7, 33]);
        call(
            &instance,
            "CommandBufferBindSeparateTexture",
            &[7, 5, 3, 256],
        );
        call(
            &instance,
            "CommandBufferBindSeparateSampler",
            &[7, 5, 3, 257],
        );
        call(
            &instance,
            "CommandBufferBindUniformBuffer",
            &[7, 5, 0, base + 0x6800, 16],
        );
        call(&instance, "CommandBufferDrawArrays", &[7, 1, 0, 3]);
        let handle = call(&instance, "CommandBufferEndRecording", &[7]);
        put(&memory, 0x300, &handle.to_le_bytes());
        call(&instance, "QueueSubmitCommands", &[0, 1, 0x300]);
        call(&instance, "QueueFinish", &[0]);
        let pixels = memory.0.lock().unwrap()[0x2000..0x6000].to_vec();
        if pixels
            .as_chunks::<4>()
            .0
            .iter()
            .any(|p| *p != [0, 0, 255, 255])
        {
            break;
        }
        assert!(Instant::now() < deadline, "draw did not complete");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(instance.take_graphics_cache_diagnostics().is_empty());
    let locked = memory.0.lock().unwrap();
    let pixels = &locked[0x2000..0x6000];
    assert_eq!(&pixels[..4], &[0, 0, 255, 255]);
    assert!(pixels.as_chunks::<4>().0.contains(&[255, 255, 255, 255]));
    assert!(pixels.as_chunks::<4>().0.contains(&[0, 0, 0, 255]));
    // Check coverage and exact nearest-filter colors away from raster edges.
    let triangle = [[8.0_f32, 8.0], [56.0, 8.0], [32.0, 56.0]];
    for y in 0..64 {
        for x in 0..64 {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let edges = [0, 1, 2].map(|n| {
                let a = triangle[n];
                let b = triangle[(n + 1) % 3];
                ((b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]))
                    / (b[0] - a[0]).hypot(b[1] - a[1])
            });
            if edges.iter().any(|e| e.abs() < 1.0) {
                continue;
            }
            let expected = if edges.iter().any(|e| *e < 0.0) {
                [0, 0, 255, 255]
            } else if (x / 32 + y / 32) % 2 == 0 {
                [255, 255, 255, 255]
            } else {
                [0, 0, 0, 255]
            };
            let at = (y * 64 + x) * 4;
            assert_eq!(&pixels[at..at + 4], &expected, "pixel {x},{y}");
        }
    }
    drop(locked);
    let mut record_cpu = 0;
    let mut submit_cpu = 0;
    let mut record_allocs = 0;
    let mut submit_allocs = 0;
    let mut commands = 0;
    let mut elapsed = Duration::ZERO;
    let mut metrics = [(0_u64, 0_u64); 3];
    let mut counts = [0_u64; 5];
    let calls = std::cell::Cell::new(0_u64);
    let mut ids = std::collections::HashMap::new();
    for (id, name) in swxbx::functions::all() {
        ids.insert(&name[3..], id);
    }
    let invoke = |suffix: &str, args: &[u64]| {
        calls.set(calls.get() + 1);
        let mut r = swxbx::Registers::default();
        r.x[..args.len()].copy_from_slice(args);
        assert_eq!(instance.call(ids[suffix], &mut r), swxbx::Status::Ok);
        r.x[0]
    };
    // One full frame warms capacities, descriptor identities and all sampled images.
    for frame in 0..=frames {
        let factor = if frame.is_multiple_of(2) {
            1.0_f32
        } else {
            0.5_f32
        };
        put(
            &memory,
            0x7800,
            &[factor; 4]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        let call_start = calls.get();
        let wall = Instant::now();
        let allocation_start = ALLOCATIONS.load(std::sync::atomic::Ordering::Relaxed);
        let cpu = cpu_time();
        invoke("CommandBufferBeginRecording", &[7]);
        invoke("CommandBufferSetRenderTargets", &[7, 1, 0x100, 0, 0]);
        invoke("CommandBufferBindProgram", &[7, 12, 0xffff]);
        for (suffix, object) in [
            ("ColorState", 22),
            ("DepthStencilState", 23),
            ("PolygonState", 24),
        ] {
            invoke(&format!("CommandBufferBind{suffix}"), &[7, object]);
        }
        invoke("CommandBufferSetViewport", &[7, 0, 0, 64, 64]);
        invoke("CommandBufferSetScissor", &[7, 0, 0, 64, 64]);
        invoke("CommandBufferBindVertexBuffer", &[7, 0, base + 0x80, 48]);
        invoke("CommandBufferBindVertexStreamState", &[7, 1, 20]);
        invoke("CommandBufferBindVertexAttribState", &[7, 1, 21]);
        invoke("CommandBufferSetTexturePool", &[7, 32]);
        invoke("CommandBufferSetSamplerPool", &[7, 33]);
        invoke("CommandBufferBindSeparateSampler", &[7, 5, 3, 257]);
        for draw in 0..draws_per_frame {
            let width = draws_per_frame / 100;
            if draw.is_multiple_of(width) {
                let group = draw / width;
                invoke(
                    "CommandBufferBindSeparateTexture",
                    &[7, 5, 3, [256, 258, 259][group % 3]],
                );
                invoke(
                    "CommandBufferBindUniformBuffer",
                    &[7, 5, 0, base + 0x6800 + (group % 2) as u64 * 256, 16],
                );
                invoke(
                    "CommandBufferSetScissor",
                    &[7, 0, 0, if group.is_multiple_of(2) { 64 } else { 48 }, 64],
                );
            }
            if indexed {
                invoke(
                    "CommandBufferDrawElementsBaseVertex",
                    &[7, 1, 6, 3, base + 0x200, 0],
                );
            } else {
                invoke("CommandBufferDrawArrays", &[7, 1, 0, 3]);
            }
        }
        if fail_tail {
            invoke("CommandBufferDrawArrays", &[7, 1, u64::from(u32::MAX), 0]);
            invoke("CommandBufferDrawArrays", &[7, 1, 0, 4]);
        }
        let handle = invoke("CommandBufferEndRecording", &[7]);
        let recorded_commands = calls.get() - call_start - 2;
        let recorded_cpu = cpu_time() - cpu;
        let recorded_allocs =
            ALLOCATIONS.load(std::sync::atomic::Ordering::Relaxed) - allocation_start;
        put(&memory, 0x300, &handle.to_le_bytes());
        swxbx::draw_metrics::take();
        swxbx::draw_metrics::take_counts();
        let allocation_start = ALLOCATIONS.load(std::sync::atomic::Ordering::Relaxed);
        let cpu = process_cpu_time();
        let mut submit = swxbx::Registers::default();
        submit.x[..3].copy_from_slice(&[0, 1, 0x300]);
        assert_eq!(
            instance.call(ids["QueueSubmitCommands"], &mut submit),
            swxbx::Status::Ok
        );
        assert_eq!(
            instance.call(ids["QueueFinish"], &mut swxbx::Registers::default()),
            if fail_tail {
                swxbx::Status::BadArgument
            } else {
                swxbx::Status::Ok
            }
        );
        let submitted_cpu = process_cpu_time() - cpu;
        let submitted_allocs =
            ALLOCATIONS.load(std::sync::atomic::Ordering::Relaxed) - allocation_start;
        let spans = swxbx::draw_metrics::take();
        let operations = swxbx::draw_metrics::take_counts();
        if frame != 0 {
            record_cpu += recorded_cpu;
            submit_cpu += submitted_cpu;
            record_allocs += recorded_allocs;
            submit_allocs += submitted_allocs;
            commands += recorded_commands;
            elapsed += wall.elapsed();
            for i in 0..5 {
                counts[i] += operations[i];
            }
            for i in 0..3 {
                metrics[i].0 += spans[i].0;
                metrics[i].1 += spans[i].1;
            }
        }
        let pixels = memory.0.lock().unwrap();
        let at = 0x2000 + (20 * 64 + 20) * 4;
        assert_eq!(
            &pixels[at..at + 4],
            &[128, 128, 128, 255],
            "last group must use its own texture and uniform range"
        );
        let at = 0x2000 + (10 * 64 + 52) * 4;
        let expected = if frame.is_multiple_of(2) {
            [128, 128, 128, 255]
        } else {
            [64, 64, 64, 128]
        };
        assert_eq!(
            &pixels[at..at + 4],
            &expected,
            "scissor must preserve the preceding group's updated uniform bytes"
        );
    }
    let draws = frames as f64 * draws_per_frame as f64;
    println!("frames={frames} draws_per_frame={draws_per_frame} record_ns_per_command={:.2} submit_cpu_ns_per_draw={:.2} fps={:.3} record_allocations_per_frame={} submit_allocations_per_frame={}",
        record_cpu as f64/commands as f64,submit_cpu as f64/draws,frames as f64/elapsed.as_secs_f64(),record_allocs/frames as u64,submit_allocs/frames as u64);
    for (label, (nanos, calls)) in ["pipeline", "descriptor", "uniform"]
        .into_iter()
        .zip(metrics)
    {
        println!(
            "{label}_ns_per_draw={:.2} {label}_calls_per_frame={} {label}_ns_per_call={:.2}",
            nanos as f64 / draws,
            calls / frames as u64,
            nanos as f64 / calls.max(1) as f64
        );
    }
    println!("descriptor_updates_per_frame={} descriptor_pools_per_frame={} state_binds_per_frame={} retained_draws_per_frame={} push_writes_per_frame={}", counts[0]/frames as u64, counts[1]/frames as u64, counts[2]/frames as u64, counts[3]/frames as u64, counts[4]/frames as u64);
    assert_eq!(
        &counts[..4],
        &[
            0,
            0,
            frames as u64 * 100,
            frames as u64 * (draws_per_frame as u64 - 100 + u64::from(fail_tail))
        ],
        "warm groups must reuse descriptors and bind once"
    );
    assert!(
        submit_allocs / (frames as u64) < 5_000,
        "allocation count must depend on groups"
    );
    assert!(instance.take_graphics_cache_diagnostics().is_empty());
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires a Vulkan GPU and shader tools"]
    fn grouped_draws_reuse_resources() {
        super::run(1, 10_000, false);
    }
    #[test]
    #[ignore = "requires a Vulkan GPU and shader tools"]
    fn grouped_indexed_draws_reuse_resources() {
        super::run(1, 10_000, true);
    }
    #[test]
    #[ignore = "requires a Vulkan GPU and shader tools"]
    fn retained_bounds_and_partial_completion() {
        super::scene(1, 10_000, false, true);
    }
}

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
#[repr(C)]
struct Timespec {
    seconds: i64,
    nanos: i64,
}
unsafe extern "C" {
    fn clock_gettime(clock: i32, time: *mut Timespec) -> i32;
}
fn process_cpu_time() -> u64 {
    let mut time = Timespec {
        seconds: 0,
        nanos: 0,
    };
    assert_eq!(unsafe { clock_gettime(2, &mut time) }, 0);
    time.seconds as u64 * 1_000_000_000 + time.nanos as u64
}

fn cpu_time() -> u64 {
    let mut time = Timespec {
        seconds: 0,
        nanos: 0,
    };
    assert_eq!(unsafe { clock_gettime(3, &mut time) }, 0);
    time.seconds as u64 * 1_000_000_000 + time.nanos as u64
}
