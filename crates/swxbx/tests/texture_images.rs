//! Original pattern images, sampled on the GPU and compared byte for byte. Source: the public contract., 0043.
#![cfg(feature = "vulkan")]
use ash::vk;
use std::{
    ffi::CString,
    io::Write,
    process::{Command, Stdio},
};
use swxbx::{
    gpu::{
        image_layout::{format_block, BlitRegion, ImageDescriptor, ImageRegion},
        Backend, GlobalMemory,
    },
    tiling::{ImageKind, ImageShape, Layout, TileShape},
};

fn packing(shape: ImageShape, format: vk::Format, height: u8, depth: u8) -> Layout {
    Layout::new(
        shape,
        format_block(format).unwrap(),
        TileShape {
            height_log2: height,
            depth_log2: depth,
        },
    )
    .unwrap()
}
fn pattern(layout: &Layout) -> Vec<u8> {
    let mut bytes = vec![0; layout.linear_size()];
    for layer in 0..layout.shape().layers {
        for (level, p) in layout.levels().iter().enumerate() {
            for z in 0..p.blocks[2] {
                for y in 0..p.blocks[1] {
                    for x in 0..p.row_bytes {
                        let at = layout
                            .linear_byte_offset(level as u32, layer, x as u32, y, z)
                            .unwrap();
                        bytes[at] = ((x * 13
                            + y as usize * 29
                            + z as usize * 71
                            + level * 37
                            + layer as usize * 19)
                            % 251) as u8;
                    }
                }
            }
        }
    }
    bytes
}
fn assert_active(layout: &Layout, actual: &[u8], expected: &[u8]) {
    assert_eq!(actual.len(), layout.linear_size());
    for layer in 0..layout.shape().layers {
        for p in layout.levels() {
            let at = layer as usize * layout.linear_layer_stride() + p.linear_offset;
            let size = p.row_bytes * p.blocks[1] as usize * p.blocks[2] as usize;
            assert_eq!(&actual[at..at + size], &expected[at..at + size]);
        }
    }
}
fn sample_shader(kind: ImageKind, layers: u32) -> Vec<u32> {
    sample_shader_variant(kind, layers, 0, false)
}
fn sample_shader_variant(kind: ImageKind, layers: u32, class: u32, wide: bool) -> Vec<u32> {
    compile_sample_shader(kind, layers, class, wide, &[])
}
fn compile_sample_shader(
    kind: ImageKind,
    layers: u32,
    class: u32,
    wide: bool,
    extra: &[String],
) -> Vec<u32> {
    let shape = match kind {
        ImageKind::D1 => 5,
        ImageKind::D1Array => 6,
        ImageKind::D2 => 0,
        ImageKind::D2Array => 1,
        ImageKind::D3 => 2,
        ImageKind::Cube if layers == 6 => 3,
        ImageKind::Cube => 4,
    };
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_2 as u32,
    );
    options.add_macro_definition("SHAPE", Some(&shape.to_string()));
    options.add_macro_definition("CLASS", Some(&class.to_string()));
    options.add_macro_definition("WIDE", Some(&u32::from(wide).to_string()));
    for definition in extra {
        let (name, value) = definition
            .strip_prefix("-D")
            .unwrap()
            .split_once('=')
            .unwrap();
        options.add_macro_definition(name, Some(value));
    }
    let module = compiler
        .compile_into_spirv(
            include_str!("shaders/sample_image.comp"),
            shaderc::ShaderKind::Compute,
            "sample_image.comp",
            "main",
            Some(&options),
        )
        .unwrap();
    let mut validator = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    validator
        .stdin
        .take()
        .unwrap()
        .write_all(module.as_binary_u8())
        .unwrap();
    let result = validator.wait_with_output().unwrap();
    assert!(
        result.status.success(),
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    module.as_binary().to_vec()
}

#[test]
#[ignore = "requires spirv-val"]
fn shader_compilation_without_scratch_storage() {
    const PROBE: &str = "SAMPLE_SHADER_COMPILER_PROBE";
    if std::env::var_os(PROBE).is_some() {
        let first = sample_shader(ImageKind::D2, 1);
        let second = sample_shader(ImageKind::D2, 1);
        assert_eq!(first.first(), Some(&0x0723_0203));
        assert_eq!(first, second);
        return;
    }
    let executable = std::env::current_exe().unwrap();
    let result = Command::new(&executable)
        .args([
            "--exact",
            "shader_compilation_without_scratch_storage",
            "--include-ignored",
            "--nocapture",
        ])
        .env(PROBE, "1")
        .env("TMPDIR", &executable)
        .env("TMP", &executable)
        .env("TEMP", &executable)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

fn sampled(backend: &mut Backend, key: u64, layout: &Layout) -> Vec<u8> {
    let words = sample_shader(layout.shape().kind, layout.shape().layers);
    sampled_words(
        backend,
        key,
        layout,
        &words,
        vk::ComponentMapping::default(),
    )
}
fn sampled_words(
    backend: &mut Backend,
    key: u64,
    layout: &Layout,
    words: &[u32],
    components: vk::ComponentMapping,
) -> Vec<u8> {
    sampled_state(backend, key, layout, words, components, None)
}
fn sampled_state(
    backend: &mut Backend,
    key: u64,
    layout: &Layout,
    words: &[u32],
    components: vk::ComponentMapping,
    state: Option<(
        &swxbx::gpu::textures::SamplerDescription,
        &swxbx::gpu::textures::TextureContract,
    )>,
) -> Vec<u8> {
    let view = backend
        .sampled_image_with_components(key, components)
        .expect("sampled view");
    let contract = swxbx::gpu::textures::TextureContract {
        filters: vec![(0, vk::Filter::NEAREST)],
        wraps: vec![(0, vk::SamplerAddressMode::CLAMP_TO_EDGE)],
        lod: Some(vk::SamplerMipmapMode::NEAREST),
        ..Default::default()
    };
    let default_description = swxbx::gpu::textures::SamplerDescription {
        max_anisotropy: 1.0,
        lod_clamp: [0.0, 32.0],
        ..Default::default()
    };
    let (description, selected_contract) = state.unwrap_or((&default_description, &contract));
    let sampler = backend
        .mapped_sampler(0, 0, description, selected_contract)
        .expect("mapped sampler");
    let context = backend.context();
    let device = &context.device;
    let mut output =
        GlobalMemory::new(context, layout.linear_size().next_multiple_of(16) as u64).unwrap();
    let guest = output
        .allocate_pool(1, 0x1000, layout.linear_size() as u64)
        .unwrap();
    let address = output.addresses().host(guest).unwrap();
    // All handles belong to this context and survive until the queue completes.
    unsafe {
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .unwrap();
        let sets = [set_layout];
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(32)];
        let pipeline_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&sets)
                    .push_constant_ranges(&ranges),
                None,
            )
            .unwrap();
        let sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        }];
        let descriptors = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
            .unwrap();
        let set = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptors)
                    .set_layouts(&sets),
            )
            .unwrap()[0];
        let images = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .sampler(sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&images)],
            &[],
        );
        let module = context.create_global_shader_module(words).unwrap();
        let entry = CString::new("main").unwrap();
        let pipeline = device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(&entry),
                    )
                    .layout(pipeline_layout)],
                None,
            )
            .unwrap()[0];
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(context.queue_family),
                None,
            )
            .unwrap();
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap()[0];
        device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[set],
            &[],
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)],
            &[],
            &[],
        );
        for layer in 0..layout.shape().layers {
            for (level, p) in layout.levels().iter().enumerate() {
                let mut push = Vec::from(address.to_ne_bytes());
                for value in [
                    p.extent[0],
                    p.extent[1],
                    p.extent[2],
                    layer,
                    level as u32,
                    (layer as usize * layout.linear_layer_stride() + p.linear_offset) as u32,
                ] {
                    push.extend(value.to_ne_bytes());
                }
                device.cmd_push_constants(
                    cmd,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push,
                );
                device.cmd_dispatch(
                    cmd,
                    (p.extent[0] * p.extent[1] * p.extent[2]).div_ceil(64),
                    1,
                    1,
                );
            }
        }
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)],
            &[],
            &[],
        );
        device.end_command_buffer(cmd).unwrap();
        device
            .queue_submit(
                context.queue,
                &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                vk::Fence::null(),
            )
            .unwrap();
        device.queue_wait_idle(context.queue).unwrap();
        device.destroy_command_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_shader_module(module, None);
        device.destroy_descriptor_pool(descriptors, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
    }
    let mut bytes = vec![0; layout.linear_size()];
    output.read_pool(1, 0, &mut bytes).unwrap();
    bytes
}

#[test]
#[ignore = "requires a real Vulkan device and shader tools; never silently skips"]
fn tiled_mips_sample_exactly_and_preserve_padding() {
    let mut backend = Backend::new(1.0).expect("required GPU");
    backend.global_memory = Some(GlobalMemory::new(backend.context(), 8 * 1024 * 1024).unwrap());
    let mut cases = Vec::new();
    for (kind, layers) in [(ImageKind::D1, 1), (ImageKind::D1Array, 3)] {
        cases.push((
            ImageShape {
                width: 37,
                height: 1,
                depth: 1,
                layers,
                levels: 6,
                kind,
            },
            0,
            0,
        ));
    }
    for h in [0, 1, 3, 5] {
        cases.push((
            ImageShape {
                width: 67,
                height: 35,
                depth: 1,
                layers: 1,
                levels: 7,
                kind: ImageKind::D2,
            },
            h,
            0,
        ));
    }
    cases.extend([
        (
            ImageShape {
                width: 19,
                height: 11,
                depth: 1,
                layers: 3,
                levels: 5,
                kind: ImageKind::D2Array,
            },
            2,
            0,
        ),
        (
            ImageShape {
                width: 17,
                height: 13,
                depth: 9,
                layers: 1,
                levels: 5,
                kind: ImageKind::D3,
            },
            2,
            2,
        ),
        (
            ImageShape {
                width: 17,
                height: 17,
                depth: 1,
                layers: 6,
                levels: 5,
                kind: ImageKind::Cube,
            },
            3,
            0,
        ),
        (
            ImageShape {
                width: 9,
                height: 9,
                depth: 1,
                layers: 12,
                levels: 4,
                kind: ImageKind::Cube,
            },
            1,
            0,
        ),
    ]);
    for (i, (shape, h, d)) in cases.into_iter().enumerate() {
        let layout = packing(shape, vk::Format::R8G8B8A8_UINT, h, d);
        let linear = pattern(&layout);
        let mut tiled = vec![0x9b; layout.tiled_size()];
        layout.encode(&linear, &mut tiled).unwrap();
        let key = i as u64 + 1;
        backend
            .allocate_pool(
                key,
                0x1000 + key * 0x100000,
                layout.tiled_size() as u64 + 32,
            )
            .unwrap();
        backend
            .global_memory
            .as_mut()
            .unwrap()
            .write_pool(key, 0, &vec![0x9b; layout.tiled_size() + 32])
            .unwrap();
        backend
            .global_memory
            .as_mut()
            .unwrap()
            .write_pool(key, 16, &tiled)
            .unwrap();
        assert!(backend.ensure_image(
            key,
            &ImageDescriptor {
                shape,
                format: vk::Format::R8G8B8A8_UINT
            }
        ));
        assert!(backend.load_tiled(key, key, 16, &layout));
        assert!(backend.wait_transfers());
        assert_active(&layout, &backend.readback(key).unwrap().2, &linear);
        assert_active(&layout, &sampled(&mut backend, key, &layout), &linear);
        assert!(backend.store_tiled(key, key, 16, &layout));
        assert!(backend.wait_transfers());
        let mut restored = vec![0; layout.tiled_size() + 32];
        backend
            .global_memory
            .as_mut()
            .unwrap()
            .read_pool(key, 0, &mut restored)
            .unwrap();
        assert_eq!(&restored[..16], &[0x9b; 16]);
        assert_eq!(&restored[16..16 + layout.tiled_size()], &tiled);
        assert_eq!(&restored[16 + layout.tiled_size()..], &[0x9b; 16]);
        backend.release_pool(key);
    }
}

#[test]
#[ignore = "requires a real Vulkan device; compressed support is queried"]
fn compressed_blocks_pass_through_all_mips() {
    let mut backend = Backend::new(1.0).expect("required GPU");
    backend.global_memory = Some(GlobalMemory::new(backend.context(), 8 * 1024 * 1024).unwrap());
    let mut formats = vec![
        vk::Format::BC1_RGB_UNORM_BLOCK,
        vk::Format::BC1_RGBA_SRGB_BLOCK,
        vk::Format::BC2_UNORM_BLOCK,
        vk::Format::BC3_UNORM_BLOCK,
        vk::Format::BC4_UNORM_BLOCK,
        vk::Format::BC4_SNORM_BLOCK,
        vk::Format::BC5_UNORM_BLOCK,
        vk::Format::BC5_SNORM_BLOCK,
        vk::Format::BC6H_UFLOAT_BLOCK,
        vk::Format::BC6H_SFLOAT_BLOCK,
        vk::Format::BC7_UNORM_BLOCK,
        vk::Format::BC7_SRGB_BLOCK,
    ];
    formats.extend(
        (vk::Format::ASTC_4X4_UNORM_BLOCK.as_raw()..=vk::Format::ASTC_12X12_SRGB_BLOCK.as_raw())
            .map(vk::Format::from_raw),
    );
    for (i, format) in formats.into_iter().enumerate() {
        let shape = ImageShape {
            width: 37,
            height: 29,
            depth: 1,
            layers: 2,
            levels: 6,
            kind: ImageKind::D2Array,
        };
        let layout = packing(shape, format, 3, 0);
        let properties = unsafe {
            backend
                .context()
                .instance
                .get_physical_device_format_properties(backend.context().physical_device, format)
        };
        let supported = properties.optimal_tiling_features.contains(
            vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::TRANSFER_SRC
                | vk::FormatFeatureFlags::TRANSFER_DST,
        );
        if !supported {
            assert!(
                format.as_raw() >= vk::Format::ASTC_4X4_UNORM_BLOCK.as_raw(),
                "BC format must be supported on this test device"
            );
            eprintln!("ASTC format {} unavailable", format.as_raw());
            continue;
        }
        let linear = pattern(&layout);
        let mut tiled = vec![0x69; layout.tiled_size()];
        layout.encode(&linear, &mut tiled).unwrap();
        let key = i as u64 + 100;
        backend
            .allocate_pool(key, key * 0x100000, layout.tiled_size() as u64)
            .unwrap();
        backend
            .global_memory
            .as_mut()
            .unwrap()
            .write_pool(key, 0, &tiled)
            .unwrap();
        assert!(backend.ensure_image(key, &ImageDescriptor { shape, format }));
        assert!(backend.load_tiled(key, key, 0, &layout));
        assert!(backend.wait_transfers());
        assert_active(&layout, &backend.readback(key).unwrap().2, &linear);
        assert!(backend.store_tiled(key, key, 0, &layout));
        assert!(backend.wait_transfers());
        let mut restored = vec![0; layout.tiled_size()];
        backend
            .global_memory
            .as_mut()
            .unwrap()
            .read_pool(key, 0, &mut restored)
            .unwrap();
        assert_eq!(restored, tiled);
        backend.release_pool(key);
    }
}

#[test]
#[ignore = "requires a real Vulkan device"]
fn image_regions_and_blits_have_exact_pixels() {
    let mut backend = Backend::new(1.0).expect("required GPU");
    let shape = ImageShape {
        width: 17,
        height: 13,
        depth: 1,
        layers: 3,
        levels: 5,
        kind: ImageKind::D2Array,
    };
    let layout = packing(shape, vk::Format::R8G8B8A8_UINT, 1, 0);
    let original = pattern(&layout);
    let zero = vec![0; layout.linear_size()];
    for key in [1, 2] {
        assert!(backend.ensure_image(
            key,
            &ImageDescriptor {
                shape,
                format: vk::Format::R8G8B8A8_UINT
            }
        ));
        assert!(backend.upload_image(key, if key == 1 { &original } else { &zero }));
    }
    assert!(backend.copy(2, 1));
    assert_active(&layout, &backend.readback(2).unwrap().2, &original);
    assert!(backend.upload_image(2, &vec![0; layout.linear_size()]));
    let from = ImageRegion {
        level: 1,
        layer: 2,
        offset: [1, 1, 0],
        extent: [3, 2, 1],
    };
    let to = ImageRegion {
        layer: 1,
        offset: [2, 2, 0],
        ..from
    };
    assert!(backend.copy_region(2, 1, from, to).is_ok());
    let result = backend.readback(2).unwrap().2;
    for y in 0..2 {
        for x in 0..3 {
            let src = layout
                .linear_byte_offset(1, 2, (x + 1) * 4, y + 1, 0)
                .unwrap();
            let dst = layout
                .linear_byte_offset(1, 1, (x + 2) * 4, y + 2, 0)
                .unwrap();
            assert_eq!(&result[dst..dst + 4], &original[src..src + 4]);
        }
    }
    let small = ImageShape {
        width: 3,
        height: 2,
        depth: 1,
        layers: 1,
        levels: 1,
        kind: ImageKind::D2,
    };
    let large = ImageShape {
        width: 6,
        height: 4,
        ..small
    };
    let small_layout = packing(small, vk::Format::R8G8B8A8_UNORM, 0, 0);
    let large_layout = packing(large, vk::Format::R8G8B8A8_UNORM, 0, 0);
    let bytes = pattern(&small_layout);
    assert!(backend.ensure_image(
        3,
        &ImageDescriptor {
            shape: small,
            format: vk::Format::R8G8B8A8_UNORM
        }
    ));
    assert!(backend.ensure_image(
        4,
        &ImageDescriptor {
            shape: large,
            format: vk::Format::R8G8B8A8_UNORM
        }
    ));
    assert!(backend.upload_image(3, &bytes));
    assert!(backend.blit_region(
        4,
        3,
        BlitRegion {
            level: 0,
            layer: 0,
            offsets: [[0, 0, 0], [3, 2, 1]]
        },
        BlitRegion {
            level: 0,
            layer: 0,
            offsets: [[0, 0, 0], [6, 4, 1]]
        },
        vk::Filter::NEAREST
    ));
    let pixels = backend.readback(4).unwrap().2;
    for y in 0..4 {
        for x in 0..6 {
            let src = small_layout
                .linear_byte_offset(0, 0, (x / 2) * 4, y / 2, 0)
                .unwrap();
            let dst = large_layout.linear_byte_offset(0, 0, x * 4, y, 0).unwrap();
            assert_eq!(&pixels[dst..dst + 4], &bytes[src..src + 4]);
        }
    }
    assert!(backend
        .copy_region(
            2,
            1,
            ImageRegion {
                offset: [u32::MAX, 0, 0],
                ..from
            },
            to
        )
        .is_err());
    assert!(backend.copy_region(1, 1, from, from).is_err());
}

#[test]
#[ignore = "requires a real Vulkan device; exercises the recorded command executor"]
fn recorded_tiled_copies_and_blits_update_arena_bytes() {
    use std::{
        ffi::c_void,
        sync::{Arc, Mutex},
    };
    use swxbx::{
        functions,
        gpu::image_layout::{CopyOperation, ImageContract, ImageRule, Storage},
        Host, Instance, Registers, Status,
    };
    unsafe extern "C" fn read(user: *mut c_void, address: u64, out: *mut u8, size: u64) -> i32 {
        let memory = unsafe { &*user.cast::<Mutex<Vec<u8>>>() }.lock().unwrap();
        let Some(bytes) = memory.get(address as usize..address as usize + size as usize) else {
            return 1;
        };
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
        }
        0
    }
    unsafe extern "C" fn write(
        user: *mut c_void,
        address: u64,
        bytes: *const u8,
        size: u64,
    ) -> i32 {
        let mut memory = unsafe { &*user.cast::<Mutex<Vec<u8>>>() }.lock().unwrap();
        let Some(out) = memory.get_mut(address as usize..address as usize + size as usize) else {
            return 1;
        };
        unsafe {
            std::ptr::copy_nonoverlapping(bytes, out.as_mut_ptr(), out.len());
        }
        0
    }
    fn call(i: &Instance, suffix: &str, args: &[u64]) -> u64 {
        let id = functions::all()
            .find(|(_, name)| name.ends_with(suffix))
            .unwrap()
            .0;
        let mut r = Registers::default();
        r.x[..args.len()].copy_from_slice(args);
        assert_eq!(i.call(id, &mut r), Status::Ok, "{suffix}");
        r.x[0]
    }
    let memory = Mutex::new(vec![0; 256 * 1024]);
    let host = Host {
        user: std::ptr::from_ref(&memory).cast_mut().cast(),
        read_memory: Some(read),
        write_memory: Some(write),
        render_scale: 1.0,
        ..Host::default()
    };
    // Callbacks and their storage stay live until this instance is destroyed.
    let instance = unsafe { Instance::with_host(host) };
    let tile = TileShape {
        height_log2: 2,
        depth_log2: 0,
    };
    instance
        .set_image_contract(ImageContract {
            swizzles: Vec::new(),
            depth_stencil_modes: Vec::new(),
            rules: vec![ImageRule {
                evidence: [swxbx::gpu::image_enums::Evidence::Hypothesis; 3],
                rationale: ["original synthetic fixture"; 3],
                flags: 0x400,
                target: 0x500,
                format: 0x600,
                host_format: vk::Format::R8G8B8A8_UINT,
                kind: ImageKind::D2Array,
                coordinates: swxbx::gpu::image_layout::Coordinates::Normalized,
                storage: Storage::Tiled(tile),
            }],
        })
        .expect("required GPU");
    let shape = ImageShape {
        width: 17,
        height: 11,
        depth: 1,
        layers: 2,
        levels: 5,
        kind: ImageKind::D2Array,
    };
    let layout = packing(shape, vk::Format::R8G8B8A8_UINT, 2, 0);
    let original = pattern(&layout);
    let span = layout.tiled_size();
    let pool_size = span * 3;
    call(&instance, "MemoryPoolBuilderSetDefaults", &[1]);
    call(
        &instance,
        "MemoryPoolBuilderSetStorage",
        &[1, 0x1000, pool_size as u64],
    );
    assert_eq!(call(&instance, "MemoryPoolInitialize", &[2, 1]), 1);
    for (key, offset) in [(4, 0), (5, span)] {
        call(&instance, "TextureBuilderSetDefaults", &[3]);
        call(&instance, "TextureBuilderSetFlags", &[3, 0x400]);
        call(&instance, "TextureBuilderSetTarget", &[3, 0x500]);
        call(&instance, "TextureBuilderSetFormat", &[3, 0x600]);
        call(&instance, "TextureBuilderSetSize3D", &[3, 17, 11, 2]);
        call(&instance, "TextureBuilderSetLevels", &[3, 5]);
        call(
            &instance,
            "TextureBuilderSetStorage",
            &[3, 2, offset as u64],
        );
        assert_eq!(
            call(&instance, "TextureBuilderGetStorageSize", &[3]),
            span as u64
        );
        assert_eq!(call(&instance, "TextureInitialize", &[key, 3]), 1);
    }
    call(&instance, "TextureViewSetDefaults", &[30]);
    call(&instance, "TextureViewSetLevels", &[30, 2, 1]);
    call(&instance, "TextureViewSetLayers", &[30, 1, 1]);
    assert_eq!(
        call(&instance, "TextureGetViewOffset", &[4, 30]),
        (layout.array_stride() + layout.levels()[2].tiled_offset) as u64
    );
    let mut source = vec![0xab; span];
    layout.encode(&original, &mut source).unwrap();
    {
        let mut bytes = memory.lock().unwrap();
        bytes[0x1000..0x1000 + pool_size].fill(0xab);
        bytes[0x1000..0x1000 + span].copy_from_slice(&source);
    }
    assert!(instance.set_image_copy_decoder(Some(Arc::new(|args| {
        assert_eq!(&args[3..7], &[0x111, 0x222, 0x333, 0x444]);
        let from = ImageRegion {
            level: 1,
            layer: 1,
            offset: [1, 1, 0],
            extent: [3, 2, 1],
        };
        let to = ImageRegion {
            level: 1,
            layer: 0,
            offset: [2, 2, 0],
            extent: [3, 2, 1],
        };
        Some(if args[7] == 0 {
            CopyOperation::Copy {
                source: args[1],
                destination: args[2],
                from,
                to,
            }
        } else {
            CopyOperation::Blit {
                source: args[1],
                destination: args[2],
                from: BlitRegion {
                    level: 1,
                    layer: 1,
                    offsets: [[0, 0, 0], [2, 2, 1]],
                },
                to: BlitRegion {
                    level: 0,
                    layer: 0,
                    offsets: [[0, 0, 0], [4, 4, 1]],
                },
                filter: vk::Filter::NEAREST,
            }
        })
    }))));
    call(&instance, "CommandBufferInitialize", &[7, 0]);
    for mode in [0, 1] {
        call(&instance, "CommandBufferBeginRecording", &[7]);
        call(
            &instance,
            "CommandBufferCopyTextureToTexture",
            &[7, 4, 5, 0x111, 0x222, 0x333, 0x444, mode],
        );
        let handle = call(&instance, "CommandBufferEndRecording", &[7]);
        memory.lock().unwrap()[0x100..0x108].copy_from_slice(&handle.to_le_bytes());
        call(&instance, "QueueSubmitCommands", &[0, 1, 0x100]);
        call(&instance, "QueueFinish", &[0]);
        let bytes = memory.lock().unwrap();
        let mut decoded = vec![0; layout.linear_size()];
        layout
            .decode(&bytes[0x1000 + span..0x1000 + span * 2], &mut decoded)
            .unwrap();
        if mode == 0 {
            for y in 0..2 {
                for x in 0..3 {
                    let a = layout
                        .linear_byte_offset(1, 1, (x + 1) * 4, y + 1, 0)
                        .unwrap();
                    let b = layout
                        .linear_byte_offset(1, 0, (x + 2) * 4, y + 2, 0)
                        .unwrap();
                    assert_eq!(&decoded[b..b + 4], &original[a..a + 4]);
                }
            }
        } else {
            for y in 0..4 {
                for x in 0..4 {
                    let a = layout
                        .linear_byte_offset(1, 1, (x / 2) * 4, y / 2, 0)
                        .unwrap();
                    let b = layout.linear_byte_offset(0, 0, x * 4, y, 0).unwrap();
                    assert_eq!(&decoded[b..b + 4], &original[a..a + 4]);
                }
            }
        }
        assert_eq!(&bytes[0x1000..0x1000 + span], &source);
        assert!(bytes[0x1000 + span * 2..0x1000 + span * 3]
            .iter()
            .all(|b| *b == 0xab));
    }
}

#[test]
#[ignore = "required GPU"]
fn batched_transfers_replay_with_changed_bytes_and_preserve_guards() {
    let mut backend = Backend::new(1.0).expect("required GPU");
    backend.global_memory = Some(GlobalMemory::new(backend.context(), 4 * 1024 * 1024).unwrap());
    backend
        .global_memory
        .as_mut()
        .unwrap()
        .allocate_pool(1, 0x1000, 4 * 1024 * 1024)
        .unwrap();
    let layouts: Vec<_> = (0..12)
        .map(|i| {
            let height = [0, 2, 5][i % 3];
            packing(
                ImageShape {
                    width: 67,
                    height: 35,
                    depth: 1,
                    layers: 1,
                    levels: 7,
                    kind: ImageKind::D2,
                },
                vk::Format::R8G8B8A8_UINT,
                height,
                0,
            )
        })
        .collect();
    let resources: Vec<_> = layouts
        .iter()
        .enumerate()
        .map(|(i, layout)| {
            let key = 100 + i as u64;
            assert!(backend.ensure_image(
                key,
                &ImageDescriptor {
                    shape: layout.shape(),
                    format: vk::Format::R8G8B8A8_UINT,
                },
            ));
            (key, 1, 256 + i as u64 * 0x20000, layout)
        })
        .collect();
    let mut expected = Vec::new();
    for generation in 0..5 {
        expected.clear();
        for (i, &(_, pool, offset, layout)) in resources.iter().enumerate() {
            let mut linear = pattern(layout);
            for byte in &mut linear {
                *byte = byte.wrapping_add((generation * 17 + i * 31) as u8);
            }
            let mut tiled = vec![0xa5; layout.tiled_size() + 32];
            layout.encode(&linear, &mut tiled[16..]).unwrap();
            assert!(backend
                .global_memory
                .as_mut()
                .unwrap()
                .write_pool(pool, offset - 16, &tiled)
                .is_some());
            expected.push((linear, tiled));
        }
        if generation == 0 {
            // Growing the ring waits for both earlier submissions before replacement.
            for count in [1, 6, 12] {
                assert!(backend.load_tiled_batch(&resources[..count]));
            }
        } else {
            assert!(backend.load_tiled_batch(&resources));
        }
        assert!(backend.wait_transfers());
    }
    // Reuse both ring slots before waiting, then vary the batch span and wrap.
    for count in [12, 1, 6, 2, 12] {
        assert!(backend.load_tiled_batch(&resources[..count]));
    }
    assert!(backend.wait_transfers());
    for (&(key, _, _, layout), (linear, _)) in resources.iter().zip(&expected) {
        assert_active(layout, &backend.readback(key).unwrap().2, linear);
    }
    for _ in 0..5 {
        for &(_, pool, offset, layout) in &resources {
            assert!(backend
                .global_memory
                .as_mut()
                .unwrap()
                .write_pool(pool, offset - 16, &vec![0xa5; layout.tiled_size() + 32])
                .is_some());
        }
        assert!(backend.store_tiled_batch(&resources));
        assert!(backend.wait_transfers());
    }
    for _ in 0..5 {
        assert!(backend.store_tiled_batch(&resources));
    }
    assert!(backend.wait_transfers());
    for (&(_, pool, offset, layout), (_, tiled)) in resources.iter().zip(&expected) {
        let mut actual = vec![0; layout.tiled_size() + 32];
        backend
            .global_memory
            .as_mut()
            .unwrap()
            .read_pool(pool, offset - 16, &mut actual)
            .unwrap();
        assert_eq!(&actual, tiled);
    }
    let resident: Vec<_> = resources
        .iter()
        .enumerate()
        .map(|(i, &(key, pool, _, layout))| {
            let offset = 2 * 1024 * 1024 + i as u64 * 0x10000;
            let mut guarded = vec![0xa5; layout.linear_size() + 32];
            guarded[16..16 + layout.linear_size()].copy_from_slice(&expected[i].0);
            backend
                .global_memory
                .as_mut()
                .unwrap()
                .write_pool(pool, offset - 16, &guarded)
                .unwrap();
            (key, pool, offset, layout)
        })
        .collect();
    for _ in 0..5 {
        assert!(backend.load_linear_batch(&resident));
    }
    assert!(backend.wait_transfers());
    for (&(key, _, _, layout), (linear, _)) in resident.iter().zip(&expected) {
        assert_active(layout, &backend.readback(key).unwrap().2, linear);
    }
    for _ in 0..5 {
        assert!(backend.store_linear_batch(&resident));
    }
    assert!(backend.wait_transfers());
    for (i, &(_, pool, offset, layout)) in resident.iter().enumerate() {
        let mut actual = vec![0; layout.linear_size() + 32];
        backend
            .global_memory
            .as_mut()
            .unwrap()
            .read_pool(pool, offset - 16, &mut actual)
            .unwrap();
        assert!(actual[..16]
            .iter()
            .chain(&actual[16 + layout.linear_size()..])
            .all(|&b| b == 0xa5));
        assert_active(
            layout,
            &actual[16..16 + layout.linear_size()],
            &expected[i].0,
        );
    }
    // Ordered writes to an overlapping resident range preserve the last image.
    let (first_key, pool, offset, layout) = resident[0];
    let second_key = resident[1].0;
    let overlapping = [
        (first_key, pool, offset, layout),
        (second_key, pool, offset, layout),
    ];
    assert!(backend.store_linear_batch(&overlapping));
    assert!(backend.wait_transfers());
    let mut last = vec![0; layout.linear_size()];
    backend
        .global_memory
        .as_mut()
        .unwrap()
        .read_pool(pool, offset, &mut last)
        .unwrap();
    assert_active(layout, &last, &expected[1].0);

    // Two sources for one destination image also retain submission order.
    let duplicate = [
        (first_key, resident[2].1, resident[2].2, layout),
        resident[0],
    ];
    assert!(backend.load_linear_batch(&duplicate));
    assert!(backend.wait_transfers());
    assert_active(
        layout,
        &backend.readback(first_key).unwrap().2,
        &expected[1].0,
    );
    assert!(!backend.load_linear_batch(&[]));
    assert!(!backend.load_linear_batch(&[(100, 1, 4 * 1024 * 1024, &layouts[0])]));
    for &(key, _, _, _) in &resources {
        backend.release_texture(key);
    }
    for &(key, _, _, layout) in &resources {
        assert!(backend.ensure_image(
            key,
            &ImageDescriptor {
                shape: layout.shape(),
                format: vk::Format::R8G8B8A8_UINT,
            }
        ));
    }
    assert!(backend.load_tiled_batch(&resources));
    assert!(backend.wait_transfers());
    for (&(key, _, _, layout), (linear, _)) in resources.iter().zip(&expected) {
        assert_active(layout, &backend.readback(key).unwrap().2, linear);
    }
    assert!(!backend.load_tiled_batch(&[]));
    assert!(!backend.load_tiled_batch(&[(100, 1, 4 * 1024 * 1024, &layouts[0])]));
}

fn catalogue_red_texel(format: vk::Format) -> Option<(Vec<u8>, [u32; 4])> {
    use swxbx::gpu::image_layout::{numeric_class, NumericClass};
    let raw = format.as_raw();
    let float = matches!(
        numeric_class(format),
        NumericClass::Float | NumericClass::Depth
    );
    let one = if float { 1.0_f32.to_bits() } else { 1 };
    let mut expected = [one, 0, 0, one];
    let packed: Option<(u32, usize)> = match raw {
        2 => Some((0xf000, 2)),
        3 => Some((0x00f0, 2)),
        4 => Some((0xf800, 2)),
        5 => Some((0x001f, 2)),
        6 => Some((0xf800, 2)),
        7 => Some((0x003e, 2)),
        8 => Some((0x7c00, 2)),
        58 => Some((0x3ff << 20, 4)),
        64 => Some((0x3ff, 4)),
        62 => Some((1 << 20, 4)),
        68 => Some((1, 4)),
        122 => Some((15 << 6, 4)),
        123 => Some(((16 << 27) | 256, 4)),
        _ => None,
    };
    if let Some((value, bytes)) = packed {
        if !matches!(raw, 4 | 5 | 122 | 123) {
            expected[3] = 0;
        }
        return Some((value.to_le_bytes()[..bytes].to_vec(), expected));
    }
    let (components, bits, bgra) = match raw {
        9..=15 | 127 => (1, 8, false),
        16..=22 => (2, 8, false),
        37..=43 => (4, 8, false),
        44..=50 => (4, 8, true),
        70..=76 | 124 => (1, 16, false),
        77..=83 => (2, 16, false),
        91..=97 => (4, 16, false),
        98..=100 | 126 => (1, 32, false),
        101..=103 => (2, 32, false),
        107..=109 => (4, 32, false),
        _ => return None,
    };
    let sf = matches!(raw, 76 | 83 | 97 | 100 | 103 | 109 | 126);
    let sn = matches!(raw, 10 | 17 | 38 | 71 | 78 | 92);
    let value = if sf {
        if bits == 16 {
            0x3c00
        } else {
            1.0_f32.to_bits()
        }
    } else if numeric_class(format) == NumericClass::Signed {
        expected[0] = u32::MAX;
        u32::MAX
    } else if !float {
        1
    } else if sn {
        (1_u32 << (bits - 1)) - 1
    } else {
        (1_u32 << bits) - 1
    };
    let mut texel = vec![0; components * bits / 8];
    let at = if bgra { 2 * bits / 8 } else { 0 };
    texel[at..at + bits / 8].copy_from_slice(&value.to_le_bytes()[..bits / 8]);
    if components == 4 {
        expected[3] = 0;
    }
    Some((texel, expected))
}

#[test]
#[ignore = "requires a Vulkan device and shader tools; unavailable host formats are reported"]
fn host_format_catalogue_samples_and_reads_back_exact_values() {
    use swxbx::gpu::{
        image_enums::transfer_formats,
        image_layout::{numeric_class, NumericClass},
    };
    let mut backend = Backend::new(1.0).expect("required GPU");
    let shape = ImageShape {
        width: 4,
        height: 4,
        depth: 1,
        layers: 1,
        levels: 1,
        kind: ImageKind::D2,
    };
    let output_layout = packing(shape, vk::Format::R32G32B32A32_UINT, 0, 0);
    let shaders = [
        sample_shader_variant(ImageKind::D2, 1, 0, true),
        sample_shader_variant(ImageKind::D2, 1, 1, true),
        sample_shader_variant(ImageKind::D2, 1, 2, true),
    ];
    let mut tested = 0;
    let mut unavailable = 0;
    for format in transfer_formats() {
        let properties = unsafe {
            backend
                .context()
                .instance
                .get_physical_device_format_properties(backend.context().physical_device, format)
        };
        if !properties.optimal_tiling_features.contains(
            vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::TRANSFER_SRC
                | vk::FormatFeatureFlags::TRANSFER_DST,
        ) {
            eprintln!("host format {} unavailable", format.as_raw());
            unavailable += 1;
            continue;
        }
        let key = format.as_raw() as u64 + 10000;
        let layout = packing(shape, format, 0, 0);
        let mut payload = vec![0; layout.linear_size()];
        if format == vk::Format::BC7_UNORM_BLOCK || format == vk::Format::BC7_SRGB_BLOCK {
            payload[0] = 0x40;
        }
        if format.as_raw() >= vk::Format::ASTC_4X4_UNORM_BLOCK.as_raw() {
            for block in payload.as_chunks_mut::<16>().0 {
                block[..8].copy_from_slice(&[0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
            }
        }
        assert!(
            backend.ensure_image(key, &ImageDescriptor { shape, format }),
            "create format {}",
            format.as_raw()
        );
        assert!(
            backend.upload_image(key, &payload),
            "upload format {}",
            format.as_raw()
        );
        let (_, _, readback) = backend.readback(key).unwrap();
        assert_active(&layout, &readback, &payload);
        let class = match numeric_class(format) {
            NumericClass::Unsigned => 0,
            NumericClass::Signed => 2,
            NumericClass::Float | NumericClass::Depth => 1,
        };
        let one = if class == 1 { 1.0_f32.to_bits() } else { 1 };
        let alpha_zero = matches!(format.as_raw(), 2 | 3 | 6 | 7 | 8 | 37..=43 | 44..=50 | 58..=69 | 91..=97 | 107..=109 | 135..=138 | 145..=146 | 157..=184);
        let expected = [0, 0, 0, if alpha_zero { 0 } else { one }];
        for component in [
            vk::ComponentSwizzle::IDENTITY,
            vk::ComponentSwizzle::ZERO,
            vk::ComponentSwizzle::ONE,
            vk::ComponentSwizzle::R,
            vk::ComponentSwizzle::G,
            vk::ComponentSwizzle::B,
            vk::ComponentSwizzle::A,
        ] {
            let components = vk::ComponentMapping {
                r: component,
                g: component,
                b: component,
                a: component,
            };
            let bytes = sampled_words(
                &mut backend,
                key,
                &output_layout,
                &shaders[class],
                components,
            );
            let selected = match component.as_raw() {
                0 => expected,
                1 => [0; 4],
                2 => [one; 4],
                c => [expected[(c - 3) as usize]; 4],
            };
            let pixel: Vec<_> = selected.iter().flat_map(|v| v.to_ne_bytes()).collect();
            for actual in bytes.as_chunks::<16>().0 {
                assert_eq!(
                    actual.as_slice(),
                    pixel,
                    "sample format {} component {}",
                    format.as_raw(),
                    component.as_raw()
                );
            }
        }
        if let Some((texel, expected)) = catalogue_red_texel(format) {
            let mut red = vec![0; layout.linear_size()];
            for pixel in red[..layout.levels()[0].row_bytes * shape.height as usize]
                .chunks_exact_mut(texel.len())
            {
                pixel.copy_from_slice(&texel);
            }
            assert!(backend.upload_image(key, &red));
            assert_active(&layout, &backend.readback(key).unwrap().2, &red);
            let actual = sampled_words(
                &mut backend,
                key,
                &output_layout,
                &shaders[class],
                vk::ComponentMapping::default(),
            );
            let pixel: Vec<_> = expected.iter().flat_map(|v| v.to_ne_bytes()).collect();
            for value in actual.as_chunks::<16>().0 {
                assert_eq!(
                    value.as_slice(),
                    pixel,
                    "nonzero sample format {}",
                    format.as_raw()
                );
            }
        }
        backend.release_texture(key);
        tested += 1;
    }
    eprintln!("host format catalogue: {tested} exact sampling/readback cases, {unavailable} unavailable formats");
    assert!(tested > 60, "catalogue must exercise the native device");
}

#[test]
#[ignore = "requires a Vulkan device and shader tools"]
fn component_selection_distinguishes_every_source_channel() {
    let mut backend = Backend::new(1.0).expect("required GPU");
    let shape = ImageShape {
        width: 1,
        height: 1,
        depth: 1,
        layers: 1,
        levels: 1,
        kind: ImageKind::D2,
    };
    let format = vk::Format::R8G8B8A8_UINT;
    assert!(backend.ensure_image(1, &ImageDescriptor { shape, format }));
    let mut payload = vec![0; packing(shape, format, 0, 0).linear_size()];
    payload[..4].copy_from_slice(&[9, 31, 77, 113]);
    assert!(backend.upload_image(1, &payload));
    let output = packing(shape, vk::Format::R32G32B32A32_UINT, 0, 0);
    let words = sample_shader_variant(ImageKind::D2, 1, 0, true);
    for (raw, expected) in [(1, 0_u32), (2, 1), (3, 9), (4, 31), (5, 77), (6, 113)] {
        let c = vk::ComponentSwizzle::from_raw(raw);
        let bytes = sampled_words(
            &mut backend,
            1,
            &output,
            &words,
            vk::ComponentMapping {
                r: c,
                g: c,
                b: c,
                a: c,
            },
        );
        assert_eq!(&bytes[..16], expected.to_ne_bytes().repeat(4));
    }
}

#[test]
#[ignore = "requires a Vulkan device and shader tools"]
fn sampler_filters_wraps_borders_and_comparison_sample_exactly() {
    use swxbx::gpu::{
        image_enums::{EnumRule, Evidence},
        textures::{MinFilter, MipFilter, SamplerDescription, SamplerEnums, TextureContract},
    };
    fn rule<T>(guest: u64, host: T) -> EnumRule<T> {
        EnumRule {
            guest,
            host,
            evidence: Evidence::Hypothesis,
            rationale: "original sampler fixture",
        }
    }
    let mut backend = Backend::new(1.0).expect("required GPU");
    let shape = ImageShape {
        width: 2,
        height: 1,
        depth: 1,
        layers: 1,
        levels: 1,
        kind: ImageKind::D2,
    };
    let format = vk::Format::R32_SFLOAT;
    assert!(backend.ensure_image(1, &ImageDescriptor { shape, format }));
    let mut pixels = vec![0; packing(shape, format, 0, 0).linear_size()];
    pixels[4..8].copy_from_slice(&1.0_f32.to_le_bytes());
    assert!(backend.upload_image(1, &pixels));
    let output_shape = ImageShape { width: 1, ..shape };
    let output = packing(output_shape, vk::Format::R32G32B32A32_UINT, 0, 0);
    let contract = TextureContract {
        enums: Some(SamplerEnums {
            min_filters: vec![
                rule(
                    11,
                    MinFilter {
                        texel: vk::Filter::NEAREST,
                        mip: MipFilter::Nearest,
                    },
                ),
                rule(
                    12,
                    MinFilter {
                        texel: vk::Filter::LINEAR,
                        mip: MipFilter::Linear,
                    },
                ),
            ],
            mag_filters: vec![rule(11, vk::Filter::NEAREST), rule(12, vk::Filter::LINEAR)],
            wraps: vec![
                rule(20, vk::SamplerAddressMode::REPEAT),
                rule(21, vk::SamplerAddressMode::MIRRORED_REPEAT),
                rule(22, vk::SamplerAddressMode::CLAMP_TO_EDGE),
                rule(23, vk::SamplerAddressMode::CLAMP_TO_BORDER),
            ],
            compare_modes: vec![rule(30, false), rule(31, true)],
            compare_functions: (0..8)
                .map(|i| rule(i as u64 + 32, vk::CompareOp::from_raw(i)))
                .collect(),
            anisotropy: true,
            integer_border: false,
        }),
        lod: Some(vk::SamplerMipmapMode::NEAREST),
        ..Default::default()
    };
    let mut state = SamplerDescription {
        min_filter: 11,
        mag_filter: 11,
        wrap: [22; 3],
        max_anisotropy: 1.0,
        compare_mode: 30,
        ..Default::default()
    };
    for (filter, wrap, border, coordinate, expected) in [
        (11, 22, [0.0; 4], 0.5, 1.0_f32),
        (12, 22, [0.0; 4], 0.5, 0.5),
        (11, 20, [0.0; 4], 1.25, 0.0),
        (11, 21, [0.0; 4], 1.25, 1.0),
        (11, 22, [0.0; 4], -0.25, 0.0),
        (11, 23, [1.0; 4], -0.25, 1.0),
        (11, 23, [0.0, 0.0, 0.0, 1.0], -0.25, 0.0),
    ] {
        state.min_filter = filter;
        state.mag_filter = filter;
        state.wrap = [wrap; 3];
        state.border_color = border;
        let words = compile_sample_shader(
            ImageKind::D2,
            1,
            1,
            true,
            &[
                "-DPROBE=1".into(),
                format!("-DPROBE_U={coordinate:.8}"),
                "-DPROBE_V=0.5".into(),
                "-DPROBE_LOD=0.0".into(),
            ],
        );
        let bytes = sampled_state(
            &mut backend,
            1,
            &output,
            &words,
            vk::ComponentMapping::default(),
            Some((&state, &contract)),
        );
        assert_eq!(
            u32::from_ne_bytes(bytes[..4].try_into().unwrap()),
            expected.to_bits()
        );
    }
    let depth_format = vk::Format::D32_SFLOAT;
    assert!(backend.ensure_image(
        2,
        &ImageDescriptor {
            shape: output_shape,
            format: depth_format
        }
    ));
    let mut depth = vec![0; packing(output_shape, depth_format, 0, 0).linear_size()];
    depth[..4].copy_from_slice(&0.5_f32.to_le_bytes());
    assert!(backend.upload_image(2, &depth));
    state.min_filter = 11;
    state.mag_filter = 11;
    state.wrap = [22; 3];
    state.border_color = [0.0; 4];
    state.compare_mode = 31;
    state.compare_func = 33;
    for (reference, expected) in [(0.25, 1.0_f32), (0.75, 0.0)] {
        let words = compile_sample_shader(
            ImageKind::D2,
            1,
            1,
            true,
            &[
                "-DPROBE=1".into(),
                "-DPROBE_COMPARE=1".into(),
                "-DPROBE_U=0.5".into(),
                "-DPROBE_V=0.5".into(),
                "-DPROBE_LOD=0.0".into(),
                format!("-DPROBE_REF={reference:.8}"),
            ],
        );
        let bytes = sampled_state(
            &mut backend,
            2,
            &output,
            &words,
            vk::ComponentMapping::default(),
            Some((&state, &contract)),
        );
        assert_eq!(
            u32::from_ne_bytes(bytes[..4].try_into().unwrap()),
            expected.to_bits()
        );
    }
    for (raw, expected) in [
        (0, 0.0_f32),
        (1, 1.0),
        (2, 0.0),
        (3, 1.0),
        (4, 0.0),
        (5, 1.0),
        (6, 0.0),
        (7, 1.0),
    ] {
        state.compare_func = raw + 32;
        let words = compile_sample_shader(
            ImageKind::D2,
            1,
            1,
            true,
            &[
                "-DPROBE=1".into(),
                "-DPROBE_COMPARE=1".into(),
                "-DPROBE_U=0.5".into(),
                "-DPROBE_V=0.5".into(),
                "-DPROBE_LOD=0.0".into(),
                "-DPROBE_REF=0.25".into(),
            ],
        );
        let bytes = sampled_state(
            &mut backend,
            2,
            &output,
            &words,
            vk::ComponentMapping::default(),
            Some((&state, &contract)),
        );
        assert_eq!(
            u32::from_ne_bytes(bytes[..4].try_into().unwrap()),
            expected.to_bits()
        );
    }
    state.compare_mode = 30;
    let mip_shape = ImageShape { levels: 2, ..shape };
    let mip_layout = packing(mip_shape, format, 0, 0);
    let mut mip_bytes = vec![0; mip_layout.linear_size()];
    let at = mip_layout.levels()[1].linear_offset;
    mip_bytes[at..at + 4].copy_from_slice(&1.0_f32.to_le_bytes());
    assert!(backend.ensure_image(
        3,
        &ImageDescriptor {
            shape: mip_shape,
            format
        }
    ));
    assert!(backend.upload_image(3, &mip_bytes));
    state.lod_clamp = [0.0, 1.0];
    state.min_filter = 12;
    state.mag_filter = 12;
    let words = compile_sample_shader(
        ImageKind::D2,
        1,
        1,
        true,
        &[
            "-DPROBE=1".into(),
            "-DPROBE_U=0.5".into(),
            "-DPROBE_V=0.5".into(),
            "-DPROBE_LOD=0.5".into(),
        ],
    );
    let bytes = sampled_state(
        &mut backend,
        3,
        &output,
        &words,
        vk::ComponentMapping::default(),
        Some((&state, &contract)),
    );
    assert_eq!(
        u32::from_ne_bytes(bytes[..4].try_into().unwrap()),
        0.5_f32.to_bits()
    );
    let limits = unsafe {
        backend
            .context()
            .instance
            .get_physical_device_properties(backend.context().physical_device)
    }
    .limits;
    let features = unsafe {
        backend
            .context()
            .instance
            .get_physical_device_features(backend.context().physical_device)
    };
    for value in [1.0, 4.0, 8.0] {
        state.max_anisotropy = value;
        let expected = value <= limits.max_sampler_anisotropy
            && (value == 1.0 || features.sampler_anisotropy != vk::FALSE);
        assert_eq!(
            backend.mapped_sampler(0, 1, &state, &contract).is_some(),
            expected
        );
    }
    state.max_anisotropy = f32::INFINITY;
    assert!(backend.mapped_sampler(0, 1, &state, &contract).is_none());
    state.max_anisotropy = 1.0;
    state.min_filter = 99;
    assert!(backend.mapped_sampler(0, 1, &state, &contract).is_none());
}
