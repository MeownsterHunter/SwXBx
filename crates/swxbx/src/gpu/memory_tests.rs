use super::*;
use crate::gpu::{commands::Commands, images::buffer_barrier};
use std::time::Instant;

#[cfg(target_os = "linux")]
fn cpu_ns() -> u64 {
    #[repr(C)]
    struct Time {
        seconds: i64,
        nanos: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: i32, time: *mut Time) -> i32;
    }
    let mut time = Time {
        seconds: 0,
        nanos: 0,
    };
    assert_eq!(unsafe { clock_gettime(3, &mut time) }, 0);
    time.seconds as u64 * 1_000_000_000 + time.nanos as u64
}

#[test]
#[cfg(target_os = "linux")]
#[ignore = "requires a GPU; measures large-pool transfer cost"]
fn memory_transfer_cost() {
    const SIZE: usize = 512 * 1024 * 1024;
    const AT: usize = SIZE / 2;
    let context = Arc::new(Context::new().expect("GPU is required"));
    let mut memory = GlobalMemory::new(&context, SIZE as u64).unwrap();
    memory.allocate_pool(1, 0x1000, SIZE as u64).unwrap();
    memory.track_submissions();
    memory.track_host_writes(1).unwrap();
    let mut guest = vec![0_u8; SIZE];
    let mut commands = Commands::new(&context).unwrap();
    assert!(memory.upload(|address, out| {
        let start = (address - 0x1000) as usize;
        out.copy_from_slice(&guest[start..start + out.len()]);
        true
    }));
    println!("mode,sample,uploaded_bytes,downloaded_bytes,cpu_ns,elapsed_ns");
    for sample in 0..12_u32 {
        guest[AT..AT + 4].copy_from_slice(&sample.to_le_bytes());
        let mut uploaded = 0;
        let mut downloaded = 0;
        let cpu = cpu_ns();
        let elapsed = Instant::now();
        memory.notify_host_write(1, AT as u64, 4).unwrap();
        assert!(memory.upload(|address, out| {
            let start = (address - 0x1000) as usize;
            out.copy_from_slice(&guest[start..start + out.len()]);
            uploaded += out.len();
            true
        }));
        let (command, _) = commands.begin_direct().unwrap();
        unsafe {
            buffer_barrier(&context.recorder(), command);
            context
                .device
                .cmd_fill_buffer(command, memory.buffer, AT as u64, 4, sample + 1);
            buffer_barrier(&context.recorder(), command);
        }
        commands.submit(false, None).unwrap();
        memory.submitted(AT as u64, 4, true, commands.submission().unwrap());
        assert!(memory.download(|address, bytes| {
            let start = (address - 0x1000) as usize;
            guest[start..start + bytes.len()].copy_from_slice(bytes);
            downloaded += bytes.len();
            true
        }));
        assert_eq!((uploaded, downloaded), (PAGE_SIZE as usize, 4));
        let elapsed = elapsed.elapsed().as_nanos();
        let cpu = cpu_ns() - cpu;
        assert_eq!(
            u32::from_le_bytes(guest[AT..AT + 4].try_into().unwrap()),
            sample + 1
        );
        assert_eq!(guest[AT - 1], 0);
        assert_eq!(guest[AT + 4], 0);
        println!("tracked,{sample},{uploaded},{downloaded},{cpu},{elapsed}");
    }
    let mut expected = vec![0; SIZE];
    expected[AT..AT + 4].copy_from_slice(&12_u32.to_le_bytes());
    assert!(guest == expected, "only the declared bytes may change");
    drop(guest);
    drop(expected);
    let pointer = unsafe { memory.map_pool(1) }.unwrap();
    let before = memory.transfer_statistics();
    for sample in 0..12_u32 {
        let cpu = cpu_ns();
        let elapsed = Instant::now();
        memory.notify_host_write(1, AT as u64, 4).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(sample.to_le_bytes().as_ptr(), pointer.add(AT), 4);
        }
        assert!(memory.upload(|_, _| panic!("direct mapping must not upload")));
        let (command, _) = commands.begin_direct().unwrap();
        unsafe {
            buffer_barrier(&context.recorder(), command);
            context
                .device
                .cmd_fill_buffer(command, memory.buffer, AT as u64, 4, sample + 1);
            buffer_barrier(&context.recorder(), command);
        }
        commands.submit(false, None).unwrap();
        memory.submitted(AT as u64, 4, true, commands.submission().unwrap());
        assert!(memory.download(|_, _| panic!("direct mapping must not download")));
        let elapsed = elapsed.elapsed().as_nanos();
        let cpu = cpu_ns() - cpu;
        assert_eq!(memory.transfer_statistics(), before);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(pointer.add(AT), 4) },
            &((sample + 1).to_le_bytes())
        );
        println!("mapped,{sample},0,0,{cpu},{elapsed}");
    }
}

fn arena(size: u64) -> GlobalMemory {
    let context = Arc::new(Context::new().expect("GPU is required"));
    let mut memory = GlobalMemory::new(&context, size).unwrap();
    memory.track_submissions();
    memory
}

#[test]
#[ignore = "requires a GPU"]
fn dirty_pages_coalesce_across_aliases_and_preserve_neighbors() {
    let mut memory = arena(64 * 1024 * 1024);
    const SIZE: usize = 32 * 1024 * 1024;
    memory.allocate_pool(1, 0x1000, SIZE as u64).unwrap();
    memory.allocate_pool(2, 0x2000, 8192).unwrap();
    memory.track_host_writes(1).unwrap();
    let mut host = vec![0x31; SIZE];
    let read = |address: u64, out: &mut [u8]| {
        let at = (address - 0x1000) as usize;
        out.copy_from_slice(&host[at..at + out.len()]);
        true
    };
    assert!(memory.upload(read));
    let before = memory.transfer_statistics();
    host[4095..4097].copy_from_slice(&[0x73, 0x74]);
    host[6000] = 0x75;
    memory.notify_host_write(1, 4095, 2).unwrap();
    memory.notify_host_write(2, 1904, 1).unwrap();
    assert!(memory.upload(|address, out| {
        let at = (address - 0x1000) as usize;
        out.copy_from_slice(&host[at..at + out.len()]);
        true
    }));
    assert_eq!(
        memory.transfer_statistics().uploaded - before.uploaded,
        8192
    );
    let mut actual = vec![0; 8193];
    memory.read_pool(1, 0, &mut actual).unwrap();
    assert_eq!(actual, host[..8193]);
    assert!(memory.upload(|_, _| panic!("clean pool must not be read")));
    assert!(memory.release_pool(1));
    memory.notify_host_write(2, 1, 1).unwrap();
    let mut copied = 0;
    assert!(memory.upload(|address, out| {
        let at = (address - 0x1000) as usize;
        out.copy_from_slice(&host[at..at + out.len()]);
        copied += out.len();
        true
    }));
    assert_eq!(copied, 4096);
    assert!(memory.release_pool(2));
    memory.allocate_pool(3, 0x8000, SIZE as u64).unwrap();
    assert!(memory.download(|_, _| panic!("released writes must not survive reuse")));
}

#[test]
#[ignore = "requires a GPU"]
fn failed_callbacks_retain_only_unfinished_ranges() {
    let mut memory = arena(128 * 1024);
    memory.allocate_pool(1, 0x1000, 128 * 1024).unwrap();
    memory.track_host_writes(1).unwrap();
    assert!(!memory.upload(|address, out| {
        out.fill(0x61);
        address == 0x1000
    }));
    let mut bytes = [0; 4];
    memory.read_pool(1, 64 * 1024, &mut bytes).unwrap();
    assert_eq!(bytes, [0; 4]);
    let mut count = 0;
    assert!(memory.upload(|address, out| {
        assert_eq!(address, 0x11000);
        out.fill(0x62);
        count += out.len();
        true
    }));
    assert_eq!(count, 64 * 1024);
    memory.write_pool(1, 8, &[1, 2, 3, 4]).unwrap();
    memory.write_pool(1, 32, &[5, 6, 7, 8]).unwrap();
    let mut outputs = Vec::new();
    assert!(!memory.download(|address, out| {
        if address == 0x1020 {
            return false;
        }
        outputs.push((address, out.to_vec()));
        true
    }));
    assert_eq!(outputs, [(0x1008, vec![1, 2, 3, 4])]);
    outputs.clear();
    assert!(memory.download(|address, out| {
        outputs.push((address, out.to_vec()));
        true
    }));
    assert_eq!(outputs, [(0x1020, vec![5, 6, 7, 8])]);
    assert!(memory.notify_host_write(1, u64::MAX, 2).is_none());
    assert!(memory.notify_host_write(1, 128 * 1024, 1).is_none());
    assert!(memory.notify_host_write(1, 128 * 1024, 0).is_some());
}

#[test]
#[ignore = "requires a GPU"]
fn coherent_mapping_skips_callbacks_and_waits_for_device_writes() {
    let mut memory = arena(64 * 1024);
    memory.allocate_pool(1, 0x1000, 64 * 1024).unwrap();
    let pointer = unsafe { memory.map_pool(1) }.unwrap();
    memory.notify_host_write(1, 0, 64 * 1024).unwrap();
    unsafe {
        std::ptr::write_bytes(pointer, 0x39, 64 * 1024);
    }
    assert!(memory.upload(|_, _| panic!("direct storage must not be copied")));
    let mut commands = Commands::new(&memory.context).unwrap();
    let (command, _) = commands.begin_direct().unwrap();
    unsafe {
        buffer_barrier(&memory.context.recorder(), command);
        memory
            .context
            .device
            .cmd_fill_buffer(command, memory.buffer, 4096, 4, 0x12345678);
        buffer_barrier(&memory.context.recorder(), command);
    }
    commands.submit(false, None).unwrap();
    memory.submitted(4096, 4, true, commands.submission().unwrap());
    assert!(memory.download(|_, _| panic!("direct storage must not be copied")));
    memory.wait_pool(1, 4096, 4, false).unwrap();
    assert_eq!(
        unsafe { std::slice::from_raw_parts(pointer.add(4096), 4) },
        &0x12345678_u32.to_ne_bytes()
    );
    assert_eq!(unsafe { *pointer.add(4095) }, 0x39);
    assert_eq!(unsafe { *pointer.add(4100) }, 0x39);
    assert_eq!(memory.transfer_statistics(), TransferStatistics::default());
}

struct Gate {
    context: Arc<Context>,
    semaphore: vk::Semaphore,
}
impl Gate {
    fn open(&self) {
        unsafe {
            self.context
                .device
                .signal_semaphore(
                    &vk::SemaphoreSignalInfo::default()
                        .semaphore(self.semaphore)
                        .value(1),
                )
                .unwrap();
        }
    }
}
impl Drop for Gate {
    fn drop(&mut self) {
        if unsafe {
            self.context
                .device
                .get_semaphore_counter_value(self.semaphore)
        }
        .unwrap_or(1)
            == 0
        {
            self.open();
        }
        self.context.wait_queue().unwrap();
        unsafe {
            self.context.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

#[test]
#[ignore = "requires a GPU"]
fn reads_wait_only_for_overlapping_submission_writes() {
    use std::{sync::mpsc, time::Duration};
    let mut memory = arena(64 * 1024);
    memory.allocate_pool(1, 0x1000, 64 * 1024).unwrap();
    memory.write_pool(1, 8192, &[0x41; 4]).unwrap();
    assert!(memory.download(|_, _| true));
    let context = memory.context.clone();
    let mut ty = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let gate = Gate {
        semaphore: unsafe {
            context
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut ty), None)
                .unwrap()
        },
        context: context.clone(),
    };
    let waits = [gate.semaphore];
    let values = [1];
    let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
    unsafe {
        context
            .device
            .queue_submit(
                context.queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&waits)
                    .wait_dst_stage_mask(&stages)
                    .push_next(&mut timeline)],
                vk::Fence::null(),
            )
            .unwrap();
    }
    let mut commands = Commands::new(&context).unwrap();
    let (command, _) = commands.begin_direct().unwrap();
    unsafe {
        context
            .device
            .cmd_fill_buffer(command, memory.buffer, 4096, 4, 0x12345678);
        buffer_barrier(&context.recorder(), command);
    }
    commands.submit(false, None).unwrap();
    memory.submitted(4096, 4, true, commands.submission().unwrap());
    let (send, receive) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let mut bytes = [0; 4];
        memory.read_pool(1, 8192, &mut bytes).unwrap();
        send.send(bytes).unwrap();
        memory.read_pool(1, 4096, &mut bytes).unwrap();
        send.send(bytes).unwrap();
        memory
    });
    let unrelated = receive.recv_timeout(Duration::from_secs(3));
    let blocked = receive.recv_timeout(Duration::from_millis(50));
    gate.open();
    let written = receive.recv_timeout(Duration::from_secs(3));
    let mut memory = worker.join().unwrap();
    assert_eq!(unrelated.unwrap(), [0x41; 4]);
    assert!(blocked.is_err());
    assert_eq!(written.unwrap(), 0x12345678_u32.to_ne_bytes());
    let mut guest = [0x77; 8192];
    assert!(memory.download(|address, out| {
        let at = (address - 0x1000) as usize;
        guest[at..at + out.len()].copy_from_slice(out);
        true
    }));
    assert_eq!(guest[4095], 0x77);
    assert_eq!(guest[4100], 0x77);
}

fn compute_shader(source: &str) -> Vec<u32> {
    use std::{fs, process::Command};
    let folder = std::env::temp_dir().join(format!("range-shader-{}", std::process::id()));
    fs::create_dir_all(&folder).unwrap();
    let input = folder.join("source.comp");
    let output = folder.join("module.spv");
    fs::write(&input, source).unwrap();
    let compile = Command::new("glslangValidator")
        .args(["-V", "--target-env", "vulkan1.2", "-o"])
        .arg(&output)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "{}{}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
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
    let bytes = fs::read(&output).unwrap();
    fs::remove_dir_all(folder).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

#[test]
#[ignore = "requires a GPU and shader tools"]
fn storage_bindings_and_query_reports_download_bounded_ranges() {
    let mut memory = arena(32 * 1024 * 1024);
    memory.allocate_pool(1, 0x1000, 32 * 1024 * 1024).unwrap();
    memory.write_pool(1, 4096, &41_u32.to_le_bytes()).unwrap();
    assert!(memory.download(|_, _| true));
    let before = memory.transfer_statistics();
    let info = memory.storage_buffer_info(1, 4096, 4).unwrap();
    let mut execution = crate::gpu::operations::Execution::new(&memory.context).unwrap();
    execution
        .dispatch(&memory, &compute_shader("#version 450\nlayout(local_size_x=1) in; layout(set=0,binding=0,std430) buffer Data { uint value; } data; void main() { data.value += 1; }\n"), false, &[(0, 0, info)], [1, 1, 1], None)
        .unwrap();
    let mut outputs = Vec::new();
    assert!(memory.download(|address, bytes| {
        outputs.push((address, bytes.to_vec()));
        true
    }));
    assert_eq!(outputs, [(0x2000, 42_u32.to_le_bytes().to_vec())]);
    assert_eq!(
        memory.transfer_statistics().downloaded - before.downloaded,
        4
    );
    let region = memory.buffer_region(1, 8192, 16).unwrap();
    execution
        .report(&memory, region, crate::counters::Counter::Timestamp)
        .unwrap();
    outputs.clear();
    assert!(memory.download(|address, bytes| {
        outputs.push((address, bytes.to_vec()));
        true
    }));
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].0, 0x3000);
    assert_eq!(outputs[0].1.len(), 16);
    assert_eq!(&outputs[0].1[..8], &[0; 8]);
    assert_ne!(&outputs[0].1[8..], &[0; 8]);
    assert_eq!(
        memory.transfer_statistics().downloaded - before.downloaded,
        20
    );
}

#[test]
#[ignore = "requires a GPU"]
fn image_writeback_does_not_download_the_surrounding_pool() {
    use crate::{api::TextureDescription, gpu::Backend};
    let mut backend = Backend::new(1.0).expect("GPU is required");
    backend.global_memory = Some(GlobalMemory::new(backend.context(), 32 * 1024 * 1024).unwrap());
    backend.allocate_pool(1, 0x1000, 32 * 1024 * 1024).unwrap();
    let description = TextureDescription {
        pool: 1,
        pool_offset: 4096,
        width: 2,
        height: 2,
        ..TextureDescription::default()
    };
    assert!(backend.ensure_texture(2, &description, false));
    assert!(backend.clear_color(2, [1.0, 0.0, 0.0, 1.0], 15));
    let memory = backend.global_memory.as_mut().unwrap();
    let mut outputs = Vec::new();
    assert!(memory.download(|address, bytes| {
        outputs.push((address, bytes.to_vec()));
        true
    }));
    assert_eq!(outputs, [(0x2000, [255, 0, 0, 255].repeat(4))]);
    assert_eq!(memory.transfer_statistics().downloaded, 16);
}

#[test]
#[ignore = "requires a GPU and shader tools"]
fn readonly_storage_bindings_produce_no_writeback() {
    let mut memory = arena(32 * 1024 * 1024);
    memory.allocate_pool(1, 0x1000, 32 * 1024 * 1024).unwrap();
    memory.write_pool(1, 4096, &41_u32.to_le_bytes()).unwrap();
    assert!(memory.download(|_, _| true));
    let source = memory.storage_buffer_info(1, 4096, 4096).unwrap();
    let destination = memory.storage_buffer_info(1, 8192, 4).unwrap();
    let words = compute_shader("#version 450\nlayout(local_size_x=1) in; layout(set=0,binding=0,std430) readonly buffer Input { uint value; } input_data; layout(set=0,binding=1,std430) buffer Output { uint value; } output_data; void main() { output_data.value = input_data.value + 1; }\n");
    let mut execution = crate::gpu::operations::Execution::new(&memory.context).unwrap();
    execution
        .dispatch(
            &memory,
            &words,
            false,
            &[(0, 0, source), (0, 1, destination)],
            [1, 1, 1],
            None,
        )
        .unwrap();
    let mut outputs = Vec::new();
    assert!(memory.download(|address, bytes| {
        outputs.push((address, bytes.to_vec()));
        true
    }));
    assert_eq!(outputs, [(0x3000, 42_u32.to_le_bytes().to_vec())]);
}
