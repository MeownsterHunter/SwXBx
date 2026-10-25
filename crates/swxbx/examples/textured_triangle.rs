//! Draw a checkerboard triangle and write a PPM image.
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
        uniforms::UniformStage,
    },
};

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .expect("supply an output PPM filename");
    let scratch = std::env::temp_dir().join(format!("swxbx-triangle-{}", std::process::id()));
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
        layout(location=0) out vec4 c;
        void main(){ c=texture(sampler2D(t,s), gl_FragCoord.xy/64.0); }",
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
            rgba8: 4,
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
                target: UniformStage::Fragment,
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
    for (object, size, offset) in [(4, 64, 0x1000), (31, 2, 0x6000)] {
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
    let draw_frame = || {
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
        call(&instance, "CommandBufferDrawArrays", &[7, 1, 0, 3]);
        let handle = call(&instance, "CommandBufferEndRecording", &[7]);
        put(&memory, 0x300, &handle.to_le_bytes());
        call(&instance, "QueueSubmitCommands", &[0, 1, 0x300]);
        call(&instance, "TextureGetLevels", &[4]);
    };
    loop {
        draw_frame();
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
    let frames = std::env::args()
        .nth(2)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(0);
    let mode = match std::env::args().nth(3).as_deref() {
        None | Some("fifo") => swxbx::PresentationMode::Fifo,
        Some("mailbox") => swxbx::PresentationMode::Mailbox,
        _ => panic!("mode must be fifo or mailbox"),
    };
    let capacity = std::env::args()
        .nth(4)
        .map(|s| s.parse().unwrap())
        .unwrap_or(2);
    assert_eq!(
        instance.configure_presentation(swxbx::PresentationConfig {
            frames_in_flight: capacity,
            mode
        }),
        swxbx::Status::Ok
    );
    let tick = match std::env::args().nth(5).as_deref() {
        Some("serial") => 1,
        Some(value) => value.parse::<usize>().expect("host tick period"),
        None => 0,
    };
    for batch in [32, frames] {
        if frames == 0 {
            break;
        }
        memory.1.lock().unwrap().samples.clear();
        for frame in 0..batch {
            draw_frame();
            if mode == swxbx::PresentationMode::Mailbox {
                memory.1.lock().unwrap().pending.clear();
            }
            memory.1.lock().unwrap().pending.push_back(Instant::now());
            call(&instance, "QueuePresentTexture", &[0, 9, 0]);
            if tick != 0 && (frame + 1) % tick == 0 {
                assert_eq!(instance.poll_presentations(true), swxbx::Status::Ok);
            }
        }
        assert_eq!(instance.poll_presentations(true), swxbx::Status::Ok);
    }
    if frames != 0 {
        let measurement = memory.1.lock().unwrap();
        let delivered = if mode == swxbx::PresentationMode::Mailbox {
            if tick == 0 {
                1
            } else {
                frames.div_ceil(tick)
            }
        } else {
            frames
        };
        assert_eq!(measurement.samples.len(), delivered);
        let n = measurement.samples.len() as f64;
        let mean = measurement.samples.iter().sum::<f64>() / n;
        let variance = measurement
            .samples
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / n;
        println!(
            "callback samples={} mean_us={mean:.3} variance_us2={variance:.3} stddev_us={:.3}",
            measurement.samples.len(),
            variance.sqrt()
        );
        let stats = instance.frame_statistics();
        assert_eq!(stats.pending, 0);
        assert_eq!(stats.failed, 0);
        assert!(stats.peak_pending <= capacity);
        assert_eq!(stats.submitted, stats.delivered + stats.dropped);
        println!(
            "submitted={} delivered={} dropped={} peak_pending={}",
            stats.submitted, stats.delivered, stats.dropped, stats.peak_pending
        );
    }
    let memory = memory.0.lock().unwrap();
    let pixels = &memory[0x2000..0x6000];
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
    let mut ppm = b"P6\n64 64\n255\n".to_vec();
    for pixel in pixels.as_chunks::<4>().0 {
        ppm.extend_from_slice(&pixel[..3]);
    }
    fs::write(output, ppm).unwrap();
    println!("Wrote a blue background and a black/white textured triangle.");
}
