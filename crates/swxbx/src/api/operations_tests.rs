//! Synthetic command programs and readback checks.
use super::*;
use crate::{functions, Instance, Registers, Status};

fn call(instance: &Instance, suffix: &str, args: &[u64]) -> u64 {
    let mut r = Registers::default();
    r.x[..args.len()].copy_from_slice(args);
    let id = functions::all()
        .find(|(_, name)| name.get(3..) == Some(suffix))
        .unwrap()
        .0;
    assert_eq!(instance.call(id, &mut r), Status::Ok, "{suffix}");
    r.x[0]
}

#[test]
fn operations_are_owned_deferred_full_width_records() {
    let instance = Instance::new();
    for command in [7, 8] {
        call(&instance, "CommandBufferInitialize", &[command, 0]);
        call(&instance, "CommandBufferBeginRecording", &[command]);
        let args = [command, 1 << 42, 2 << 42, 3 << 42, 4, 5, 6, 7];
        for &kind in NAMES {
            call(&instance, &format!("CommandBuffer{kind}"), &args);
        }
        let handle = call(&instance, "CommandBufferEndRecording", &[command]);
        let recorded = instance.objects.recording(handle).unwrap();
        assert_eq!(recorded.len(), NAMES.len());
        for (record, &kind) in recorded.iter().zip(NAMES) {
            assert_eq!(
                record,
                &RecordedCommand::Operation {
                    kind,
                    arguments: args[1..].try_into().unwrap(),
                }
            );
        }
        call(&instance, "CommandBufferBeginRecording", &[command]);
        assert!(instance.objects.recording(handle).is_none());
    }
}

#[cfg(feature = "vulkan")]
mod gpu {
    use super::*;
    use crate::{
        api::objects::{Object, ShaderTranslation},
        counters::{Counter, CounterContract},
        gpu::{
            graphics::{FirstDrawContract, PrimitiveTopology, VertexFormat},
            operations::{CommandContract, ComputeBufferMapping},
        },
        Host,
    };
    use std::{
        ffi::c_void,
        fs,
        process::Command,
        sync::{Arc, Mutex},
    };

    unsafe extern "C" fn read(user: *mut c_void, address: u64, out: *mut u8, size: u64) -> i32 {
        let memory = unsafe { &*user.cast::<Mutex<Vec<u8>>>() }.lock().unwrap();
        let Some(bytes) = memory.get(address as usize..address.saturating_add(size) as usize)
        else {
            return 1;
        };
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
        }
        0
    }
    unsafe extern "C" fn write(user: *mut c_void, address: u64, data: *const u8, size: u64) -> i32 {
        let mut memory = unsafe { &*user.cast::<Mutex<Vec<u8>>>() }.lock().unwrap();
        let Some(bytes) = memory.get_mut(address as usize..address.saturating_add(size) as usize)
        else {
            return 1;
        };
        unsafe {
            std::ptr::copy_nonoverlapping(data, bytes.as_mut_ptr(), bytes.len());
        }
        0
    }

    fn shader(stage: &str, source: &str) -> Vec<u32> {
        let dir =
            std::env::temp_dir().join(format!("command-shaders-{}-{stage}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let input = dir.join(format!("source.{stage}"));
        let output = dir.join("module.spv");
        fs::write(&input, source).unwrap();
        let compile = Command::new("glslangValidator")
            .args(["-V", "--target-env", "vulkan1.2", "-o"])
            .arg(&output)
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            compile.status.success(),
            "{}",
            String::from_utf8_lossy(&compile.stdout)
        );
        let validate = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            validate.status.success(),
            "{}",
            String::from_utf8_lossy(&validate.stderr)
        );
        let bytes = fs::read(output).unwrap();
        fs::remove_dir_all(dir).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    }

    struct Fixture {
        instance: Instance,
        memory: Box<Mutex<Vec<u8>>>,
        base: u64,
    }
    impl Fixture {
        fn new() -> Self {
            let memory = Box::new(Mutex::new(vec![0; 0x30000]));
            let host = Host {
                user: std::ptr::from_ref(&*memory).cast_mut().cast(),
                read_memory: Some(read),
                write_memory: Some(write),
                ..Host::default()
            };
            let instance = unsafe { Instance::with_host(host) };
            assert!(
                instance.set_command_contract(Some(CommandContract {
                    condition_nonzero: 90,
                    condition_zero: 91,
                    compute_stage: 5,
                    compute_buffers: vec![ComputeBufferMapping {
                        index: 0,
                        set: 0,
                        binding: 0
                    }],
                    clear: Some(Arc::new(|a| (a[1..] == [0, 0, 0, 0, 0, 0]).then_some((
                        a[0],
                        [0.0, 0.0, 1.0, 1.0],
                        15
                    )))),
                })),
                "Vulkan backend is required"
            );
            instance.set_counter_contract(CounterContract::new([
                (80, Counter::SamplesPassed),
                (81, Counter::Timestamp),
            ]));
            call(&instance, "MemoryPoolBuilderSetDefaults", &[1]);
            call(
                &instance,
                "MemoryPoolBuilderSetStorage",
                &[1, 0x1000, 0x20000],
            );
            call(&instance, "MemoryPoolInitialize", &[2, 1]);
            let base = call(&instance, "MemoryPoolGetBufferAddress", &[2]);
            call(&instance, "CommandBufferInitialize", &[7, 0]);
            Self {
                instance,
                memory,
                base,
            }
        }
        fn put(&self, offset: usize, bytes: &[u8]) {
            self.memory.lock().unwrap()[0x1000 + offset..0x1000 + offset + bytes.len()]
                .copy_from_slice(bytes);
        }
        fn words(&self, offset: usize, values: &[u32]) {
            self.put(
                offset,
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
        }
        fn word(&self, offset: usize) -> u32 {
            u32::from_le_bytes(
                self.memory.lock().unwrap()[0x1000 + offset..0x1004 + offset]
                    .try_into()
                    .unwrap(),
            )
        }
        fn wide(&self, offset: usize) -> u64 {
            u64::from_le_bytes(
                self.memory.lock().unwrap()[0x1000 + offset..0x1008 + offset]
                    .try_into()
                    .unwrap(),
            )
        }
        fn begin(&self) {
            call(&self.instance, "CommandBufferBeginRecording", &[7]);
        }
        fn command(&self, suffix: &str, args: &[u64]) {
            let mut a = vec![7];
            a.extend_from_slice(args);
            call(&self.instance, &format!("CommandBuffer{suffix}"), &a);
        }
        fn submit(&self, expected: Status) {
            let handle = call(&self.instance, "CommandBufferEndRecording", &[7]);
            self.memory.lock().unwrap()[0x100..0x108].copy_from_slice(&handle.to_le_bytes());
            let mut r = Registers::default();
            r.x[..3].copy_from_slice(&[0, 1, 0x100]);
            let id = functions::all()
                .find(|(_, n)| n.get(3..) == Some("QueueSubmitCommands"))
                .unwrap()
                .0;
            let recording = self.instance.objects.get(7);
            assert_eq!(self.instance.call(id, &mut r), Status::Ok);
            assert_eq!(
                self.instance.queue.drain().status(),
                expected,
                "{recording:?}"
            );
        }
        fn compute(&self) {
            let words = shader(
                "comp",
                r#"#version 450
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
layout(set=0, binding=0, std430) buffer Output { uint values[]; };
void main() {
    uint i = gl_GlobalInvocationID.x + gl_NumWorkGroups.x *
        (gl_GlobalInvocationID.y + gl_NumWorkGroups.y * gl_GlobalInvocationID.z);
    values[i] = 100 + i + gl_NumWorkGroups.x * 1000 +
        gl_NumWorkGroups.y * 10000 + gl_NumWorkGroups.z * 100000;
    if (i == 0) {
        values[64] = 3; values[65] = 2; values[66] = 1;
        values[80] = 3; values[81] = 2; values[82] = 0; values[83] = 0;
        values[96] = 1;
    }
}"#,
            );
            self.instance.objects.put(
                30,
                Object::Program {
                    shader_translations: vec![ShaderTranslation::Spirv(words)],
                    device: 0,
                    shader_records: Vec::new(),
                },
            );
        }
    }

    impl Fixture {
        fn graphics_cold(&self) {
            assert!(self
                .instance
                .set_first_draw_contract(Some(FirstDrawContract {
                    topologies: vec![(4, PrimitiveTopology::TriangleList)],
                    attribute_formats: vec![(14, VertexFormat::Float4), (15, VertexFormat::Float2)],
                    attribute_state_stride: Some(16),
                    stream_state_stride: Some(16),
                    cull_none: 16,
                    rgba8: 17,
                    target_2d: 18,
                    identity_swizzle: [0; 4],
                    index_u16: 19,
                    index_u32: 20,
                    depth_raster: None,
                })));
            for (key, offset) in [(4, 0x1000), (5, 0x2000)] {
                call(&self.instance, "TextureBuilderSetDefaults", &[3]);
                call(&self.instance, "TextureBuilderSetSize2D", &[3, 32, 32]);
                call(&self.instance, "TextureBuilderSetFormat", &[3, 17]);
                call(&self.instance, "TextureBuilderSetTarget", &[3, 18]);
                call(&self.instance, "TextureBuilderSetStorage", &[3, 2, offset]);
                call(&self.instance, "TextureInitialize", &[key, 3]);
            }
            let vertex = shader(
                "vert",
                r#"#version 450
layout(location=0) in vec4 position;
layout(location=1) in vec2 shift;
void main() {
    gl_Position = position;
    gl_Position.xy += shift + vec2(float(gl_InstanceIndex) * 0.8, 0.0);
}"#,
            );
            let fragment = shader(
                "frag",
                r#"#version 450
layout(location=0) out vec4 color;
void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }"#,
            );
            self.instance.objects.put(
                31,
                Object::Program {
                    device: 0,
                    shader_records: Vec::new(),
                    shader_translations: vec![
                        ShaderTranslation::Spirv(vertex),
                        ShaderTranslation::Spirv(fragment),
                    ],
                },
            );
            for key in [20, 36] {
                call(&self.instance, "VertexStreamStateSetDefaults", &[key]);
            }
            call(&self.instance, "VertexStreamStateSetStride", &[20, 16]);
            call(&self.instance, "VertexStreamStateSetDivisor", &[20, 0]);
            call(&self.instance, "VertexStreamStateSetStride", &[36, 8]);
            call(&self.instance, "VertexStreamStateSetDivisor", &[36, 1]);
            for key in [21, 37] {
                call(&self.instance, "VertexAttribStateSetDefaults", &[key]);
            }
            call(&self.instance, "VertexAttribStateSetFormat", &[21, 14, 0]);
            call(&self.instance, "VertexAttribStateSetStreamIndex", &[21, 0]);
            call(&self.instance, "VertexAttribStateSetFormat", &[37, 15, 0]);
            call(&self.instance, "VertexAttribStateSetStreamIndex", &[37, 1]);
            call(&self.instance, "ColorStateSetDefaults", &[22]);
            call(&self.instance, "ColorStateSetBlendEnable", &[22, 0, 0]);
            call(&self.instance, "DepthStencilStateSetDefaults", &[23]);
            for setting in ["DepthTestEnable", "DepthWriteEnable", "StencilTestEnable"] {
                call(
                    &self.instance,
                    &format!("DepthStencilStateSet{setting}"),
                    &[23, 0],
                );
            }
            call(&self.instance, "PolygonStateSetDefaults", &[24]);
            call(&self.instance, "PolygonStateSetCullFace", &[24, 16]);
            let vertices: [f32; 12] = [
                -0.9, -0.5, 0.0, 1.0, -0.3, -0.5, 0.0, 1.0, -0.6, 0.4, 0.0, 1.0,
            ];
            self.put(
                0x4000,
                &vertices
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            );
            self.put(0x5000, &[0; 32]);
        }

        fn graphics(&self) {
            self.graphics_cold();
            // Compile the asynchronous graphics pipeline before measuring reports.
            self.draw_state();
            self.command("DrawArrays", &[4, 0, 3]);
            self.submit(Status::Ok);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while self.instance.graphics_pending_count().unwrap() != 0 {
                assert!(std::time::Instant::now() < deadline);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert!(self.instance.take_graphics_cache_diagnostics().is_empty());
        }

        fn draw_state(&self) {
            self.draw_state_target(4);
        }

        fn draw_state_target(&self, target: u64) {
            self.begin();
            self.memory.lock().unwrap()[0x180..0x188].copy_from_slice(&target.to_le_bytes());
            self.command("SetRenderTargets", &[1, 0x180, 0, 0, 0]);
            self.put(
                0x7000,
                &[0.0_f32, 0.0, 1.0, 1.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            );
            self.command("ClearColor", &[0, 0x8000, 15]);
            self.command("BindProgram", &[31, 0]);
            self.command("BindVertexBuffer", &[0, self.base + 0x4000, 48]);
            self.command("BindVertexBuffer", &[1, self.base + 0x5000, 32]);
            self.command("BindVertexStreamState", &[2, 20]);
            self.command("BindVertexAttribState", &[2, 21]);
            for (kind, key) in [
                ("ColorState", 22),
                ("DepthStencilState", 23),
                ("PolygonState", 24),
            ] {
                self.command(&format!("Bind{kind}"), &[key]);
            }
            self.command("SetViewport", &[0, 0, 32, 32]);
            self.command("SetScissor", &[0, 0, 32, 32]);
        }

        fn pixels(&self, key: u64) -> Vec<u8> {
            self.instance
                .gpu
                .lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .readback(key)
                .unwrap()
                .2
        }
        fn triangles(&self, key: u64, left: bool, right: bool) {
            let pixels = self.pixels(key);
            for (x, red) in [(6, left), (19, right)] {
                let at = (12 * 32 + x) * 4;
                assert_eq!(
                    &pixels[at..at + 4],
                    if red {
                        &[255, 0, 0, 255]
                    } else {
                        &[0, 0, 255, 255]
                    }
                );
            }
            let offset = if key == 4 { 0x1000 } else { 0x2000 };
            assert_eq!(
                pixels,
                self.memory.lock().unwrap()[0x1000 + offset..0x2000 + offset]
            );
        }
    }

    struct GatedTranslator {
        words: Vec<u32>,
        gate: Arc<(Mutex<bool>, std::sync::Condvar)>,
        started: std::sync::mpsc::Sender<()>,
    }
    impl crate::ShaderTranslator for GatedTranslator {
        fn translate(&self, _: crate::ShaderStage, _: &[u8]) -> Result<Vec<u32>, String> {
            let _ = self.started.send(());
            let mut open = self.gate.0.lock().unwrap();
            while !*open {
                open = self.gate.1.wait(open).unwrap();
            }
            Ok(self.words.clone())
        }
        fn translation_cache_key(
            &self,
            _: crate::ShaderStage,
            code: &[u8],
            _: &crate::startup_cache::TranslationContext,
        ) -> Result<Option<String>, String> {
            Ok(Some(format!("{:064x}", code[0] + 1)))
        }
    }

    #[test]
    #[ignore = "requires a graphics device and shader tools; never skips"]
    fn cold_draw_writes_once_then_later_submission_samples_exact_pixels() {
        use crate::{
            gpu::{
                textures::{TextureContract, TextureMapping},
                uniforms::UniformStage,
            },
            startup_cache::StartupCacheConfig,
        };
        use std::{sync::mpsc, time::Duration};

        for mode in 0..3 {
            let mut f = Fixture::new();
            f.graphics_cold();
            let directory = std::env::temp_dir()
                .join(format!("single-write-cache-{}-{mode}", std::process::id()));
            fs::create_dir_all(&directory).unwrap();
            f.instance
                .set_graphics_pipeline_cache(
                    &directory.join("pipelines"),
                    &crate::gpu::pipelines::TranslationIdentity {
                        version: "single write 1".into(),
                        configuration: String::new(),
                    },
                    1,
                    1,
                )
                .unwrap();
            assert_eq!(f.instance.graphics_cache_stats().unwrap().misses, 0);
            let vertices: [f32; 12] = [
                -1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0,
            ];
            f.put(
                0x4000,
                &vertices
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            );
            let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
            if mode != 0 {
                let Some(Object::Program {
                    mut shader_translations,
                    ..
                }) = f.instance.objects.get(31)
                else {
                    panic!("missing producer program");
                };
                let ShaderTranslation::Spirv(words) = shader_translations.pop().unwrap() else {
                    panic!("missing producer fragment");
                };
                let (started, receive) = mpsc::channel();
                let mut config = StartupCacheConfig::new(directory.join("translations"));
                config.workers = 1;
                config.queue_capacity = 1;
                f.instance
                    .configure_startup_cache(
                        config,
                        Arc::new(GatedTranslator {
                            words,
                            gate: gate.clone(),
                            started,
                        }),
                    )
                    .unwrap();
                let pending = f.instance.request_cached_translation(&[0]).unwrap();
                receive.recv_timeout(Duration::from_secs(5)).unwrap();
                let translation = if mode == 1 {
                    pending
                } else {
                    let _queued = f.instance.request_cached_translation(&[1]).unwrap();
                    let deferred = f.instance.request_cached_translation(&[2]).unwrap();
                    assert!(matches!(deferred, ShaderTranslation::Deferred(..)));
                    deferred
                };
                shader_translations.push(translation);
                f.instance.objects.put(
                    31,
                    Object::Program {
                        shader_translations,
                        shader_records: Vec::new(),
                        device: 0,
                    },
                );
            }
            f.draw_state();
            f.command("DrawArrays", &[4, 0, 3]);
            let completed_early = std::thread::scope(|scope| {
                let (done, receive) = mpsc::channel();
                let fixture = &f;
                let submit = scope.spawn(move || {
                    fixture.submit(Status::Ok);
                    done.send(()).unwrap();
                });
                let early = mode != 0 && receive.recv_timeout(Duration::from_millis(50)).is_ok();
                *gate.0.lock().unwrap() = true;
                gate.1.notify_all();
                submit.join().unwrap();
                early
            });
            assert!(
                !completed_early,
                "submission completed before translation, mode {mode}"
            );
            for (at, pixel) in f.pixels(4).as_chunks::<4>().0.iter().enumerate() {
                assert_eq!(*pixel, [255, 0, 0, 255], "producer mode {mode}, pixel {at}");
            }

            f.instance
                .set_texture_contract(TextureContract {
                    bindings: vec![TextureMapping {
                        stage: 4,
                        index: 0,
                        target: UniformStage::Fragment,
                        set: 0,
                        image: 0,
                        sampler: 1,
                    }],
                    filters: vec![(100, ash::vk::Filter::NEAREST)],
                    wraps: vec![(101, ash::vk::SamplerAddressMode::CLAMP_TO_EDGE)],
                    compare_disabled: 102,
                    ..TextureContract::default()
                })
                .unwrap();
            call(&f.instance, "TexturePoolInitialize", &[40, 2, 0, 512]);
            call(&f.instance, "TexturePoolRegisterTexture", &[40, 256, 4, 0]);
            call(&f.instance, "SamplerPoolInitialize", &[41, 2, 0, 512]);
            call(&f.instance, "SamplerBuilderSetDefaults", &[42]);
            call(&f.instance, "SamplerBuilderSetCompare", &[42, 102, 0]);
            call(
                &f.instance,
                "SamplerBuilderSetMinMagFilter",
                &[42, 100, 100],
            );
            call(
                &f.instance,
                "SamplerBuilderSetWrapMode",
                &[42, 101, 101, 101],
            );
            let mut r = Registers::default();
            r.x[0] = 42;
            r.d[0] = u64::from(1.0_f32.to_bits());
            let id = functions::all()
                .find(|(_, name)| name.get(3..) == Some("SamplerBuilderSetMaxAnisotropy"))
                .unwrap()
                .0;
            assert_eq!(f.instance.call(id, &mut r), Status::Ok);
            call(&f.instance, "SamplerInitialize", &[43, 42]);
            call(&f.instance, "SamplerPoolRegisterSampler", &[41, 257, 43]);
            let fragment = shader(
                "frag",
                r#"#version 450
layout(set=0, binding=0) uniform texture2D source_image;
layout(set=0, binding=1) uniform sampler source_sampler;
layout(location=0) out vec4 color;
void main() {
    color = texelFetch(sampler2D(source_image, source_sampler), ivec2(gl_FragCoord.xy), 0);
}"#,
            );
            let Some(Object::Program {
                mut shader_translations,
                ..
            }) = f.instance.objects.get(31)
            else {
                panic!("missing producer program");
            };
            shader_translations.truncate(1);
            shader_translations.push(ShaderTranslation::Spirv(fragment));
            f.instance.objects.put(
                32,
                Object::Program {
                    shader_translations,
                    shader_records: Vec::new(),
                    device: 0,
                },
            );
            f.draw_state_target(5);
            f.command("BindProgram", &[32, 0]);
            f.command("SetTexturePool", &[40]);
            f.command("SetSamplerPool", &[41]);
            f.command("BindSeparateTexture", &[4, 0, 256]);
            f.command("BindSeparateSampler", &[4, 0, 257]);
            f.command("DrawArrays", &[4, 0, 3]);
            f.submit(Status::Ok);
            for (at, pixel) in f.pixels(5).as_chunks::<4>().0.iter().enumerate() {
                assert_eq!(*pixel, [255, 0, 0, 255], "sample mode {mode}, pixel {at}");
            }
            assert!(f.instance.take_graphics_cache_diagnostics().is_empty());
            drop(f);
            fs::remove_dir_all(directory).unwrap();
        }
    }

    #[test]
    #[ignore = "requires a graphics device and shader tools; never skips"]
    fn explicit_skip_logs_each_unfinished_draw_and_block_restores_execution() {
        use crate::{gpu::graphics::PendingDrawPolicy, startup_cache::StartupCacheConfig};
        for policy in [
            PendingDrawPolicy::Skip,
            PendingDrawPolicy::Wait(std::time::Duration::ZERO),
        ] {
            let mut f = Fixture::new();
            f.graphics_cold();
            let directory =
                std::env::temp_dir().join(format!("skip-choice-cache-{}", std::process::id()));
            let Some(Object::Program {
                mut shader_translations,
                ..
            }) = f.instance.objects.get(31)
            else {
                panic!("missing program");
            };
            let ShaderTranslation::Spirv(words) = shader_translations.pop().unwrap() else {
                panic!("missing fragment");
            };
            let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
            let (started, receive) = std::sync::mpsc::channel();
            f.instance
                .configure_startup_cache(
                    StartupCacheConfig::new(directory.clone()),
                    Arc::new(GatedTranslator {
                        words,
                        gate: gate.clone(),
                        started,
                    }),
                )
                .unwrap();
            shader_translations.push(f.instance.request_cached_translation(&[0]).unwrap());
            receive
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            f.instance.objects.put(
                31,
                Object::Program {
                    shader_translations,
                    shader_records: Vec::new(),
                    device: 0,
                },
            );
            f.instance.set_pending_draw_policy(policy).unwrap();
            f.draw_state();
            for _ in 0..3 {
                f.command("DrawArrays", &[4, 0, 3]);
            }
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f.submit(Status::Ok)));
            *gate.0.lock().unwrap() = true;
            gate.1.notify_all();
            result.unwrap();
            let diagnostics = f.instance.take_graphics_cache_diagnostics();
            assert_eq!(
                diagnostics,
                vec!["skipped draw: shader translation pending"; 3]
            );
            f.instance
                .set_pending_draw_policy(PendingDrawPolicy::Block)
                .unwrap();
            f.draw_state();
            f.command("DrawArrays", &[4, 0, 3]);
            f.submit(Status::Ok);
            f.triangles(4, true, false);
            assert!(f.instance.take_graphics_cache_diagnostics().is_empty());
            drop(f);
            fs::remove_dir_all(directory).unwrap();
        }
    }

    #[test]
    #[ignore = "requires Vulkan, glslangValidator and spirv-val"]
    fn indirect_instancing_reports_conditionals_and_pass_transfers_read_back() {
        let f = Fixture::new();
        f.graphics();
        f.compute();
        f.draw_state();
        f.command("ResetCounter", &[80]);
        f.command("DrawArraysInstanced", &[4, 0, 3, 0, 2]);
        f.command("ReportCounter", &[80, f.base + 0xa008]);
        f.submit(Status::Ok);
        f.triangles(4, true, true);
        assert!(f.wide(0xa008) > 0);
        assert!(f.wide(0xa010) > 0);

        f.draw_state();
        f.command("DrawArraysInstanced", &[4, 0, 3, 1, 1]);
        f.submit(Status::Ok);
        f.triangles(4, false, true);

        f.draw_state();
        f.command("BindProgram", &[30, 0]);
        f.command("BindUniformBuffer", &[5, 0, f.base + 0x8000, 1024]);
        f.command("DispatchCompute", &[1, 1, 1]);
        f.command("BindProgram", &[31, 0]);
        // Compute buffer bindings do not become unsupported graphics banks.
        f.command("ClearBuffer", &[f.base + 0x8180, 4, 0]);
        f.command("SetRenderEnableConditional", &[f.base + 0x8180, 90]);
        f.command("ResetCounter", &[80]);
        f.command("DrawArraysIndirect", &[4, f.base + 0x8140]);
        f.command("ReportCounter", &[80, f.base + 0xa020]);
        f.command(
            "CopyBufferToBuffer",
            &[f.base + 0x8140, f.base + 0x8300, 16, 0],
        );
        f.command("SetRenderEnableConditional", &[f.base + 0x8180, 91]);
        f.command("DrawArraysIndirect", &[4, f.base + 0x8300]);
        f.command("ReportCounter", &[80, f.base + 0xa038]);
        f.command("CopyTextureToTexture", &[4, 5]);
        f.command("ClearTexture", &[4]);
        f.submit(Status::Ok);
        assert_eq!(f.wide(0xa020), 0);
        assert!(f.wide(0xa038) > 0);
        f.triangles(4, false, false);
        f.triangles(5, true, true);

        for (token, width) in [(19, 2), (20, 4)] {
            let indices = [0xffff_u32, 1, 2, 3];
            f.put(
                0x6000,
                &indices
                    .into_iter()
                    .flat_map(|i| i.to_le_bytes().into_iter().take(width))
                    .collect::<Vec<_>>(),
            );
            f.words(0x8400, &[3, 2, 1, (-1_i32) as u32, 0]);
            f.draw_state();
            f.command(
                "DrawElementsIndirect",
                &[4, token, f.base + 0x6000, f.base + 0x8400],
            );
            f.submit(Status::Ok);
            f.triangles(4, true, true);
            f.draw_state();
            f.command(
                "DrawElementsInstanced",
                &[
                    4,
                    token,
                    3,
                    f.base + 0x6000 + width as u64,
                    u64::from((-1_i32) as u32),
                    0,
                    2,
                ],
            );
            f.submit(Status::Ok);
            f.triangles(4, true, true);
        }

        f.words(0x8400, &[3, 1, 1, 0, 0]);
        f.draw_state();
        f.command(
            "DrawElementsIndirect",
            &[4, 19, f.base - 2, f.base + 0x8400],
        );
        f.submit(Status::BadArgument);
        f.draw_state();
        f.command(
            "CopyBufferToBuffer",
            &[f.base + 0x8140, f.base + 0x1000, 16, 0],
        );
        f.command("DrawArraysIndirect", &[4, f.base + 0x1000]);
        f.submit(Status::BadArgument);

        f.draw_state();
        f.command("SetRenderEnable", &[0]);
        f.command("DrawArrays", &[4, 0, 3]);
        f.command("DrawArraysInstanced", &[4, 0, 3, 0, 2]);
        f.submit(Status::Ok);
        f.triangles(4, false, false);

        f.draw_state();
        f.command("BindVertexBuffer", &[1, f.base + 0x5000, 8]);
        f.command("DrawArraysInstanced", &[4, 0, 3, 0, 2]);
        f.submit(Status::BadArgument);
        f.draw_state();
        f.command("DrawArraysInstanced", &[4, 0, 3, 0, 0]);
        f.submit(Status::Ok);
        f.triangles(4, false, false);

        // A transfer invalidates the index bounds of an otherwise reusable draw.
        call(&f.instance, "VertexStreamStateSetDivisor", &[36, 0]);
        f.draw_state();
        f.command("DrawArrays", &[4, 0, 3]);
        f.submit(Status::Ok);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while f.instance.graphics_pending_count().unwrap() != 0 {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        f.put(0x6000, &[0, 0, 1, 0, 2, 0]);
        f.draw_state();
        f.command("DrawElementsBaseVertex", &[4, 19, 3, f.base + 0x6000, 0]);
        f.command("ClearBuffer", &[f.base + 0x6000, 4, u64::from(u32::MAX)]);
        f.command("DrawElementsBaseVertex", &[4, 19, 3, f.base + 0x6000, 0]);
        f.submit(Status::BadArgument);
    }

    #[test]
    #[ignore = "requires Vulkan, glslangValidator and spirv-val"]
    fn compute_indirect_fill_copy_and_reports_read_back() {
        let f = Fixture::new();
        f.compute();
        f.put(0x8000, &[0xaa; 1024]);
        f.put(0xa000, &[0xcc; 64]);
        f.begin();
        f.command("ClearBuffer", &[f.base + 0x8000, 1024, 0]);
        f.command("BindProgram", &[30, 0]);
        f.command("BindUniformBuffer", &[5, 0, f.base + 0x8000, 1024]);
        f.command("DispatchCompute", &[2, 3, 4]);
        f.command("ReportCounter", &[81, f.base + 0xa008]);
        f.command(
            "CopyBufferToBuffer",
            &[f.base + 0x8000, f.base + 0x9000, 96, 0],
        );
        f.command("ClearBuffer", &[f.base + 0x8000, 96, u64::from(u32::MAX)]);
        f.command("DispatchComputeIndirect", &[f.base + 0x8100]);
        f.command("ReportCounter", &[81, f.base + 0xa020]);
        assert_eq!(f.word(0x8000), 0xaaaa_aaaa, "recording must not execute");
        assert_eq!(f.wide(0xa008), 0xcccc_cccc_cccc_cccc);
        f.submit(Status::Ok);
        for i in 0..24 {
            assert_eq!(
                f.word(0x8000 + i * 4),
                if i < 6 { 123100 + i as u32 } else { u32::MAX }
            );
            assert_eq!(f.word(0x9000 + i * 4), 432100 + i as u32);
        }
        assert_eq!(f.word(0x8000 + 24 * 4), 0);
        assert_eq!(f.wide(0xa008), 0);
        assert!(f.wide(0xa010) > 0);
        assert!(f.wide(0xa028) >= f.wide(0xa010));
        assert_eq!(f.wide(0xa000), 0xcccc_cccc_cccc_cccc);
        assert_eq!(f.wide(0xa018), 0xcccc_cccc_cccc_cccc);
        assert_eq!(f.wide(0xa030), 0xcccc_cccc_cccc_cccc);

        for (kind, args, status) in [
            ("DispatchCompute", vec![u64::MAX, 1, 1], Status::BadArgument),
            (
                "DispatchComputeIndirect",
                vec![f.base + 0x20000 - 8],
                Status::BadArgument,
            ),
            (
                "DispatchComputeIndirect",
                vec![f.base + 0x8101],
                Status::BadArgument,
            ),
            (
                "ClearBuffer",
                vec![f.base + 0x8001, 4, 1],
                Status::BadArgument,
            ),
            (
                "ClearBuffer",
                vec![f.base + 0x8000, 3, 1],
                Status::BadArgument,
            ),
            (
                "CopyBufferToBuffer",
                vec![f.base + 0x8000, f.base + 0x8004, 16, 0],
                Status::BadArgument,
            ),
            (
                "ReportCounter",
                vec![999, f.base + 0xa008],
                Status::Unimplemented,
            ),
            (
                "ReportCounter",
                vec![81, f.base + 0x20000 - 8],
                Status::BadArgument,
            ),
            (
                "SetRenderEnableConditional",
                vec![f.base + 0x8180, 999],
                Status::Unimplemented,
            ),
        ] {
            f.begin();
            f.command(kind, &args);
            f.submit(status);
        }
    }
}
