//! Original host memory and synthetic shader markers for the triangle example.
use std::{ffi::c_void, fs, path::Path, process::Command, sync::Mutex, time::Instant};
use swxbx::{functions, Host, Instance, Registers, ShaderStage, ShaderTranslator, Status};

pub(super) struct Memory(pub Mutex<Vec<u8>>, pub Mutex<Measurement>);

#[derive(Default)]
pub(super) struct Measurement {
    pub pending: std::collections::VecDeque<Instant>,
    pub samples: Vec<f64>,
}

unsafe extern "C" fn present(
    user: *mut c_void,
    _: u64,
    width: u32,
    height: u32,
    rgba: *const u8,
    stride: u64,
) {
    let handed = Instant::now();
    let memory = unsafe { &*user.cast::<Memory>() };
    let mut measurement = memory.1.lock().unwrap();
    let start = measurement.pending.pop_front().expect("present timestamp");
    measurement
        .samples
        .push(handed.duration_since(start).as_secs_f64() * 1e6);
    assert_eq!((width, height, stride), (64, 64, 256));
    let pixels = unsafe { std::slice::from_raw_parts(rgba, (stride * u64::from(height)) as usize) };
    assert_eq!(&pixels[..4], &[0, 0, 255, 255]);
    assert!(pixels.as_chunks::<4>().0.contains(&[255, 255, 255, 255]));
    assert!(pixels.as_chunks::<4>().0.contains(&[0, 0, 0, 255]));
}

unsafe extern "C" fn read(user: *mut c_void, address: u64, out: *mut u8, size: u64) -> i32 {
    // SAFETY: the example keeps Memory alive until the instance is dropped.
    let memory = unsafe { &*user.cast::<Memory>() }.0.lock().unwrap();
    let Some(end) = address.checked_add(size) else {
        return 1;
    };
    let Some(bytes) = memory.get(address as usize..end as usize) else {
        return 1;
    };
    // SAFETY: the library supplies an output buffer valid for size bytes.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len()) };
    0
}

unsafe extern "C" fn write(user: *mut c_void, address: u64, data: *const u8, size: u64) -> i32 {
    // SAFETY: the example keeps Memory alive until the instance is dropped.
    let mut memory = unsafe { &*user.cast::<Memory>() }.0.lock().unwrap();
    let Some(end) = address.checked_add(size) else {
        return 1;
    };
    let Some(bytes) = memory.get_mut(address as usize..end as usize) else {
        return 1;
    };
    // SAFETY: the library supplies an input buffer valid for size bytes.
    unsafe { std::ptr::copy_nonoverlapping(data, bytes.as_mut_ptr(), bytes.len()) };
    0
}

pub(super) fn hosted(memory: &Memory) -> Instance {
    let host = Host {
        user: std::ptr::from_ref(memory).cast_mut().cast(),
        read_memory: Some(read),
        write_memory: Some(write),
        present: Some(present),
        ..Host::default()
    };
    // SAFETY: the caller keeps memory alive and drops the instance first.
    unsafe { Instance::with_host(host) }
}

pub(super) fn call(instance: &Instance, suffix: &str, args: &[u64]) -> u64 {
    let id = functions::all()
        .find(|(_, name)| name.get(3..) == Some(suffix))
        .expect("known function suffix")
        .0;
    let mut registers = Registers::default();
    registers.x[..args.len()].copy_from_slice(args);
    assert_eq!(instance.call(id, &mut registers), Status::Ok, "{suffix}");
    registers.x[0]
}

pub(super) fn put(memory: &Memory, address: usize, bytes: &[u8]) {
    memory.0.lock().unwrap()[address..address + bytes.len()].copy_from_slice(bytes);
}

pub(super) struct Shaders {
    pub vertex: Vec<u32>,
    pub fragment: Vec<u32>,
}

impl ShaderTranslator for Shaders {
    fn translate(&self, _: ShaderStage, bytes: &[u8]) -> Result<Vec<u32>, String> {
        // These markers identify original GLSL, not hardware instruction bytes.
        match bytes.first() {
            Some(1) => Ok(self.vertex.clone()),
            Some(2) => Ok(self.fragment.clone()),
            _ => Err("unknown synthetic marker".into()),
        }
    }
}

pub(super) fn register_marker(memory: &Memory, address: usize, marker: u8) {
    // Observed envelope only, with an original marker header and nonzero payload.
    // The example hook never interprets this as a compiled hardware program.
    let mut bytes = [0; 0xa0];
    bytes[..4].copy_from_slice(&0x1234_5678_u32.to_le_bytes());
    bytes[0x30] = marker;
    bytes[0x80..0x88].fill(0x5a);
    // The program ends at an unconditional branch to itself (the public contract).
    bytes[0x98..].copy_from_slice(&0xe240_0fff_ff87_000f_u64.to_le_bytes());
    put(memory, address, &bytes);
}

pub(super) fn compile(directory: &Path, stage: &str, source: &str) -> Vec<u32> {
    let input = directory.join(format!("triangle.{stage}"));
    let output = directory.join(format!("triangle.{stage}.spv"));
    fs::write(&input, source).unwrap();
    for (tool, args) in [
        (
            "glslangValidator",
            vec!["-V", "--target-env", "vulkan1.2", "-o"],
        ),
        ("spirv-val", vec!["--target-env", "vulkan1.2"]),
    ] {
        let mut command = Command::new(tool);
        command.args(args).arg(&output);
        if tool == "glslangValidator" {
            command.arg(&input);
        }
        let result = command.output().expect("shader tools must be installed");
        assert!(
            result.status.success(),
            "{tool}: {} {}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }
    fs::read(output)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}
