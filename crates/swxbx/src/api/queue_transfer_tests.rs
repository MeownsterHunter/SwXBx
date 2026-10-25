use super::*;
use crate::api::objects::StateCommand;
use crate::{
    gpu::{operations::CommandContract, Backend, GlobalMemory},
    Host, Instance,
};
use std::{
    ffi::c_void,
    sync::{Arc, Mutex},
};

unsafe extern "C" fn read(user: *mut c_void, address: u64, out: *mut u8, size: u64) -> i32 {
    let bytes = unsafe { &*user.cast::<Mutex<Vec<u8>>>() }.lock().unwrap();
    let Some(source) = bytes.get(address as usize..address.saturating_add(size) as usize) else {
        return 1;
    };
    unsafe {
        std::ptr::copy_nonoverlapping(source.as_ptr(), out, source.len());
    }
    0
}
unsafe extern "C" fn write(user: *mut c_void, address: u64, data: *const u8, size: u64) -> i32 {
    let mut bytes = unsafe { &*user.cast::<Mutex<Vec<u8>>>() }.lock().unwrap();
    let Some(target) = bytes.get_mut(address as usize..address.saturating_add(size) as usize)
    else {
        return 1;
    };
    unsafe {
        std::ptr::copy_nonoverlapping(data, target.as_mut_ptr(), target.len());
    }
    0
}

#[test]
#[ignore = "requires a GPU"]
fn submission_uses_host_notifications_and_exact_device_writeback() {
    const SIZE: usize = 32 * 1024 * 1024;
    let host = Arc::new(Mutex::new(vec![0x31; SIZE]));
    let instance = unsafe {
        Instance::with_host(Host {
            user: Arc::as_ptr(&host).cast_mut().cast(),
            read_memory: Some(read),
            write_memory: Some(write),
            ..Host::default()
        })
    };
    let mut backend = Backend::new(1.0).expect("GPU is required");
    backend.global_memory = Some(GlobalMemory::new(backend.context(), SIZE as u64).unwrap());
    let address = backend.allocate_pool(1, 0, SIZE as u64).unwrap();
    instance.objects.put(
        1,
        Object::MemoryPool {
            device: 0,
            flags: 0,
            storage: 0,
            size: SIZE as u64,
            gpu_address: Some(address),
            observed_gpu_address: None,
        },
    );
    *instance.gpu.lock().unwrap() = Some(backend);
    assert!(instance.set_command_contract(Some(CommandContract {
        condition_nonzero: 1,
        condition_zero: 2,
        compute_stage: 5,
        compute_buffers: vec![],
        clear: None
    })));
    assert_eq!(
        unsafe { crate::swxbx_instance_track_pool_writes(&instance, 1) },
        Status::Ok
    );
    assert_eq!(execute(&instance, vec![]), Status::Ok);
    let before = instance.transfer_statistics().unwrap();
    host.lock().unwrap()[4095..4097].copy_from_slice(&[0x71, 0x72]);
    assert_eq!(
        unsafe { crate::swxbx_instance_notify_memory_write(&instance, 4095, 2) },
        Status::Ok
    );
    let commands = vec![RecordedCommand::State(StateCommand::ClearBuffer {
        address: address + 4100,
        size: 4,
        value: 0x12345678,
    })];
    assert_eq!(execute(&instance, vec![(0, commands)]), Status::Ok);
    let after = instance.transfer_statistics().unwrap();
    assert_eq!(after.uploaded - before.uploaded, 8192);
    assert_eq!(after.downloaded - before.downloaded, 4);
    assert_eq!(after.inspected - before.inspected, 0);
    let bytes = host.lock().unwrap();
    assert_eq!(&bytes[4095..4097], &[0x71, 0x72]);
    assert_eq!(&bytes[4100..4104], &0x12345678_u32.to_le_bytes());
    assert_eq!(bytes[4099], 0x31);
    assert_eq!(bytes[4104], 0x31);
    drop(bytes);
    assert_eq!(execute(&instance, vec![]), Status::Ok);
    assert_eq!(instance.transfer_statistics().unwrap(), after);
    assert!(instance.write_memory(8192, &[0x53; 4]));
    assert_eq!(execute(&instance, vec![]), Status::Ok);
    assert_eq!(
        instance.transfer_statistics().unwrap().uploaded - after.uploaded,
        4096
    );
    let mut bytes = [0; 4];
    instance
        .gpu
        .lock()
        .unwrap()
        .as_mut()
        .unwrap()
        .global_memory
        .as_mut()
        .unwrap()
        .read_pool(1, 8192, &mut bytes)
        .unwrap();
    assert_eq!(bytes, [0x53; 4]);
}
