//! Memory pools. Signatures: docs/signatures/0003-objects.md.
//!
//! The program gives a pool its storage, a block of its own memory. swxbx
//! keeps CPU storage separate from GPU addresses when Vulkan is active.
//! The flat backing and its implementation choices are documented in
//! the documented contract

use super::{objects::Object, succeed, Handler};
use crate::instance::{Instance, Registers, Status};

fn builder_update(
    instance: &Instance,
    registers: &Registers,
    change: impl FnOnce(&mut Object),
) -> Status {
    instance.objects.update(registers.x[0], change);
    Status::Ok
}

fn pool_field(
    instance: &Instance,
    address: u64,
    pick: impl FnOnce(u64, u64, u64) -> u64,
) -> Option<u64> {
    match instance.objects.get(address) {
        Some(Object::MemoryPool {
            flags,
            storage,
            size,
            ..
        }) => Some(pick(flags, storage, size)),
        _ => None,
    }
}

/// Return the handler for a supported name in this command family.
pub fn handler(name: &str) -> Option<Handler> {
    Some(match name {
        "nvnMemoryPoolBuilderSetDefaults" => |instance, _, registers| {
            instance.objects.put(
                registers.x[0],
                Object::MemoryPoolBuilder {
                    device: 0,
                    flags: 0,
                    storage: 0,
                    size: 0,
                },
            );
            Status::Ok
        },
        "nvnMemoryPoolBuilderSetDevice" => |instance, _, registers| {
            let value = registers.x[1];
            builder_update(instance, registers, |object| {
                if let Object::MemoryPoolBuilder { device, .. } = object {
                    *device = value;
                }
            })
        },
        "nvnMemoryPoolBuilderSetFlags" => |instance, _, registers| {
            let value = registers.x[1];
            builder_update(instance, registers, |object| {
                if let Object::MemoryPoolBuilder { flags, .. } = object {
                    *flags = value;
                }
            })
        },
        "nvnMemoryPoolBuilderSetStorage" => |instance, _, registers| {
            let (memory, bytes) = (registers.x[1], registers.x[2]);
            builder_update(instance, registers, |object| {
                if let Object::MemoryPoolBuilder { storage, size, .. } = object {
                    *storage = memory;
                    *size = bytes;
                }
            })
        },
        "nvnMemoryPoolInitialize" => |instance, _, registers| {
            let Some(Object::MemoryPoolBuilder {
                device,
                flags,
                storage,
                size,
            }) = instance.objects.get(registers.x[1])
            else {
                return Status::BadArgument;
            };
            if instance.objects.get(registers.x[0]).is_some() {
                registers.x[0] = 0;
                return Status::BadArgument;
            }
            #[cfg(not(feature = "vulkan"))]
            let gpu_address = None;
            #[cfg(feature = "vulkan")]
            let gpu_address = {
                let mut gpu = instance.gpu.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(backend) = gpu.as_mut() {
                    let gpu_address = backend.allocate_pool(registers.x[0], storage, size);
                    if gpu_address.is_none() {
                        registers.x[0] = 0;
                        return Status::BadArgument;
                    }
                    gpu_address
                } else {
                    None
                }
            };
            instance.objects.put(
                registers.x[0],
                Object::MemoryPool {
                    device,
                    flags,
                    storage,
                    size,
                    gpu_address,
                    observed_gpu_address: None,
                },
            );
            succeed(registers)
        },
        "nvnMemoryPoolFinalize" => |instance, _, registers| {
            #[cfg(feature = "vulkan")]
            if let Some(backend) = instance
                .gpu
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_mut()
            {
                if backend
                    .global_memory
                    .as_ref()
                    .is_some_and(|memory| memory.contains_pool(registers.x[0]))
                    && !backend.release_pool(registers.x[0])
                {
                    return Status::InternalError;
                }
            }
            instance.objects.remove(registers.x[0]);
            Status::Ok
        },
        // Mapping a pool gives the program a pointer to its contents; the
        // contents are the program's own storage.
        "nvnMemoryPoolMap" => {
            |instance, _, registers| match pool_field(instance, registers.x[0], |_, storage, _| {
                storage
            }) {
                Some(value) => {
                    registers.x[0] = value;
                    Status::Ok
                }
                None => Status::BadArgument,
            }
        }
        "nvnMemoryPoolGetBufferAddress" => {
            |instance, _, registers| match instance.objects.get(registers.x[0]) {
                Some(Object::MemoryPool {
                    storage,
                    gpu_address,
                    observed_gpu_address,
                    ..
                }) => {
                    registers.x[0] = gpu_address.or(observed_gpu_address).unwrap_or(storage);
                    Status::Ok
                }
                _ => Status::BadArgument,
            }
        }
        "nvnMemoryPoolGetSize" => {
            |instance, _, registers| match pool_field(instance, registers.x[0], |_, _, size| size) {
                Some(value) => {
                    registers.x[0] = value;
                    Status::Ok
                }
                None => Status::BadArgument,
            }
        }
        "nvnMemoryPoolGetFlags" => {
            |instance, _, registers| match pool_field(instance, registers.x[0], |flags, _, _| flags)
            {
                Some(value) => {
                    registers.x[0] = value;
                    Status::Ok
                }
                None => Status::BadArgument,
            }
        }
        _ => return None,
    })
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;
    use crate::api::objects::TextureDescription;
    use crate::gpu::{Backend, GlobalMemory};
    use std::sync::{Arc, Mutex};

    fn call(instance: &Instance, name: &str, args: &[u64]) -> Registers {
        let mut registers = Registers::default();
        registers.x[..args.len()].copy_from_slice(args);
        let function = instance.request(name).unwrap();
        assert_eq!(
            instance.call(function, &mut registers),
            Status::Ok,
            "{name}"
        );
        registers
    }

    #[test]
    #[ignore = "requires Vulkan, exercises the public API with actual buffer addresses"]
    fn pool_and_texture_gpu_addresses_share_backing_but_map_returns_cpu_storage() {
        let mut backend = Backend::new(1.0).expect("Vulkan context");
        backend.global_memory = Some(GlobalMemory::new(backend.context(), 64 * 1024).unwrap());
        let instance = Instance::new();
        *instance.gpu.lock().unwrap() = Some(backend);
        call(&instance, "nvnMemoryPoolBuilderSetDefaults", &[1]);
        call(
            &instance,
            "nvnMemoryPoolBuilderSetStorage",
            &[1, 0x1000, 0x100],
        );
        assert_eq!(call(&instance, "nvnMemoryPoolInitialize", &[2, 1]).x[0], 1);
        let gpu_base = call(&instance, "nvnMemoryPoolGetBufferAddress", &[2]).x[0];
        let observed = Registers {
            x: [0x800000, 0, 0, 0, 0, 0, 0, 0],
            ..Registers::default()
        };
        instance.returned(
            instance.request("nvnMemoryPoolGetBufferAddress").unwrap(),
            &observed,
        );
        assert_eq!(
            instance
                .objects
                .resolve_gpu_address(0x800020)
                .unwrap()
                .program_address,
            0x1020
        );
        assert_eq!(
            call(&instance, "nvnMemoryPoolGetBufferAddress", &[2]).x[0],
            gpu_base
        );
        assert_eq!(gpu_base, crate::global_memory::GUEST_BASE);
        assert_eq!(call(&instance, "nvnMemoryPoolMap", &[2]).x[0], 0x1000);
        let resolution = instance.objects.resolve_gpu_address(gpu_base + 32).unwrap();
        assert_eq!(resolution.program_address, 0x1020);
        assert_eq!(resolution.remaining, 0xe0);
        instance.objects.put(
            3,
            Object::Texture {
                description: TextureDescription {
                    pool: 2,
                    pool_offset: 32,
                    ..TextureDescription::default()
                },
                image: Arc::new(Mutex::new(None)),
            },
        );
        assert_eq!(
            call(&instance, "nvnTextureGetTextureAddress", &[3]).x[0],
            gpu_base + 32
        );
        call(
            &instance,
            "nvnMemoryPoolBuilderSetStorage",
            &[1, 0x1010, 16],
        );
        call(&instance, "nvnMemoryPoolInitialize", &[4, 1]);
        assert_eq!(
            call(&instance, "nvnMemoryPoolGetBufferAddress", &[4]).x[0],
            gpu_base + 16
        );
        call(&instance, "nvnMemoryPoolFinalize", &[2]);
        assert_eq!(
            instance
                .objects
                .resolve_gpu_address(gpu_base + 16)
                .unwrap()
                .pool,
            4
        );
        call(&instance, "nvnMemoryPoolFinalize", &[4]);
        assert!(instance.objects.resolve_gpu_address(gpu_base + 16).is_err());
        call(
            &instance,
            "nvnMemoryPoolBuilderSetStorage",
            &[1, 0x3000, 0x10000],
        );
        call(&instance, "nvnMemoryPoolInitialize", &[5, 1]);
        assert_eq!(
            call(&instance, "nvnMemoryPoolGetBufferAddress", &[5]).x[0],
            gpu_base
        );
        call(
            &instance,
            "nvnMemoryPoolBuilderSetStorage",
            &[1, 0x20000, 16],
        );
        let mut registers = Registers {
            x: [6, 1, 0, 0, 0, 0, 0, 0],
            ..Registers::default()
        };
        assert_eq!(
            instance.call(
                instance.request("nvnMemoryPoolInitialize").unwrap(),
                &mut registers
            ),
            Status::BadArgument
        );
        assert_eq!(registers.x[0], 0);
        assert!(instance.objects.get(6).is_none());
    }
}
