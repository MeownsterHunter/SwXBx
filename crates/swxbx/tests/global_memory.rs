//! Vulkan global-memory checks.

#![cfg(feature = "vulkan")]

use ash::vk;
use std::{ffi::CString, fs, path::PathBuf, process::Command, sync::Arc};
use swxbx::gpu::{Context, GlobalMemory};

fn shader(name: &str) -> Vec<u32> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = std::env::temp_dir().join(format!("{name}-{}.spv", std::process::id()));
    fs::create_dir_all(output.parent().unwrap()).unwrap();
    let compiled = Command::new("glslangValidator")
        .args(["-V", "--target-env", "vulkan1.2", "-o"])
        .arg(&output)
        .arg(root.join("tests/shaders").join(format!("{name}.comp")))
        .output()
        .expect("glslangValidator must be installed");
    assert!(
        compiled.status.success(),
        "shader compilation: {}{}",
        String::from_utf8_lossy(&compiled.stdout),
        String::from_utf8_lossy(&compiled.stderr)
    );
    let validated = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2"])
        .arg(&output)
        .output()
        .expect("spirv-val must be installed");
    assert!(
        validated.status.success(),
        "SPIR-V validation: {}",
        String::from_utf8_lossy(&validated.stderr)
    );
    let bytes = fs::read(&output).unwrap();
    fs::remove_file(output).unwrap();
    let (words, remainder) = bytes.as_chunks::<4>();
    assert!(remainder.is_empty());
    words
        .iter()
        .map(|&chunk| u32::from_le_bytes(chunk))
        .collect()
}

fn dispatch(context: &Context, memory: &GlobalMemory, words: &[u32], root: u64) {
    let module = context
        .create_global_shader_module(words)
        .expect("required shader features");
    let stage_flags = vk::ShaderStageFlags::COMPUTE;
    // The experiment adds its root pointer after the published delta range.
    let ranges = [GlobalMemory::push_constant_range(stage_flags).size(16)];
    let device = &context.device;
    // SAFETY: all objects belong to this context and are destroyed after queue idle.
    unsafe {
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges),
                None,
            )
            .unwrap();
        let entry = CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(stage_flags)
            .module(module)
            .name(&entry);
        let pipelines = device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            )
            .unwrap();
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(context.queue_family),
                None,
            )
            .unwrap();
        let command = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap()[0];
        device
            .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipelines[0]);
        memory.push_delta(command, layout, stage_flags);
        device.cmd_push_constants(command, layout, stage_flags, 8, &root.to_ne_bytes());
        device.cmd_dispatch(command, 1, 1, 1);
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
        device.end_command_buffer(command).unwrap();
        device
            .queue_submit(
                context.queue,
                &[vk::SubmitInfo::default().command_buffers(&[command])],
                vk::Fence::null(),
            )
            .unwrap();
        device.queue_wait_idle(context.queue).unwrap();
        device.destroy_command_pool(pool, None);
        device.destroy_pipeline(pipelines[0], None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_shader_module(module, None);
    }
}

#[test]
#[ignore = "requires Vulkan 1.2, glslangValidator and spirv-val; never silently skips"]
fn physical_accesses_follow_guest_pointers_across_pools() {
    let context =
        Arc::new(Context::new().expect("Vulkan with bufferDeviceAddress and shaderInt64"));
    let mut memory = GlobalMemory::new(&context, 64 * 1024).expect("flat Vulkan arena");
    let first = memory.allocate_pool(1, 0x1000, 32).unwrap();
    let second = memory.allocate_pool(2, 0x2000, 32).unwrap();
    let map = memory.addresses();
    assert_eq!(map.host(first).unwrap().wrapping_sub(first), map.delta());
    assert_eq!(map.host(second).unwrap().wrapping_sub(second), map.delta());
    assert_eq!(second - first, 32);
    assert!(second <= u64::from(u32::MAX));
    let mut first_bytes = [0xaa; 32];
    first_bytes[..8].copy_from_slice(&second.to_le_bytes());
    first_bytes[12..16].copy_from_slice(&99_u32.to_le_bytes());
    let mut second_bytes = [0xbb; 32];
    second_bytes[8..12].copy_from_slice(&41_u32.to_le_bytes());
    assert!(memory.upload(|address, out| {
        let bytes = match address {
            0x1000 => &first_bytes,
            0x2000 => &second_bytes,
            _ => return false,
        };
        out.copy_from_slice(bytes);
        true
    }));
    dispatch(&context, &memory, &shader("global_memory"), first);
    first_bytes[8..12].copy_from_slice(&42_u32.to_le_bytes());
    second_bytes[8..12].copy_from_slice(&99_u32.to_le_bytes());
    assert!(memory.download(|address, bytes| {
        let expected = match address {
            0x1000 => &first_bytes,
            0x2000 => &second_bytes,
            _ => return false,
        };
        assert_eq!(bytes, expected);
        true
    }));
    let alias = memory.allocate_pool(3, 0x1010, 16).unwrap();
    assert_eq!(alias, first + 16);
    assert!(memory.release_pool(1));
    let mut bytes = [0; 16];
    memory.read_pool(3, 0, &mut bytes).unwrap();
    assert_eq!(bytes, [0xaa; 16]);
    assert!(memory.write_pool(3, 15, &[1, 2]).is_none());
    assert!(memory.allocate_pool(4, 0x3000, 64 * 1024).is_none());
    assert!(memory.release_pool(3));
    assert_eq!(memory.allocate_pool(4, 0x4000, 17), Some(first));
    assert_eq!(memory.addresses(), map);
    let mut reset = [1; 17];
    memory.read_pool(4, 0, &mut reset).unwrap();
    assert_eq!(reset, [0; 17]);
    let context_lifetime = Arc::downgrade(&context);
    drop(context);
    memory.read_pool(4, 0, &mut reset).unwrap();
    assert!(context_lifetime.upgrade().is_some());
    drop(memory);
    assert!(context_lifetime.upgrade().is_none());
}

#[test]
#[ignore = "requires Vulkan narrow storage and integer features, glslangValidator and spirv-val"]
fn adjacent_narrow_stores_preserve_neighbouring_bytes() {
    let context = Arc::new(Context::new().expect("Vulkan with flat global memory features"));
    let mut memory = GlobalMemory::new(&context, 64 * 1024).unwrap();
    let address = memory.allocate_pool(1, 0x1000, 16).unwrap();
    memory.write_pool(1, 0, &[0xaa; 16]).unwrap();
    dispatch(&context, &memory, &shader("global_memory_narrow"), address);
    let mut bytes = [0; 16];
    memory.read_pool(1, 0, &mut bytes).unwrap();
    assert_eq!(&bytes[..4], &[0x31, 0x32, 0x67, 0x45]);
    assert_eq!(&bytes[4..], &[0xaa; 12]);
}
