//! Device, device builder and the handles the device hands out.

use super::{objects::Object, succeed, write_u32, Handler};
use crate::instance::Status;
#[cfg(test)]
use crate::{
    functions::FunctionId,
    instance::{Instance, Registers},
};

static INTEGERS: &str = include_str!("../../../../data/device-integers.txt");

/// The configured answer for a device integer selector.
pub fn device_integer(selector: u64) -> Option<u32> {
    INTEGERS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .find_map(|line| {
            let mut fields = line.split_whitespace();
            let key = parse_hex(fields.next()?)?;
            let value = parse_hex(fields.next()?)?;
            (key == selector).then_some(value as u32)
        })
}

fn parse_hex(text: &str) -> Option<u64> {
    u64::from_str_radix(text.trim_start_matches("0x"), 16).ok()
}

/// Return the handler for a supported name in this command family.
pub fn handler(name: &str) -> Option<Handler> {
    Some(match name {
        "nvnDeviceBuilderSetDefaults" => |instance, _, registers| {
            instance
                .objects
                .put(registers.x[0], Object::DeviceBuilder { flags: 0 });
            Status::Ok
        },
        "nvnDeviceBuilderSetFlags" => |instance, _, registers| {
            let value = registers.x[1];
            instance.objects.update(registers.x[0], |object| {
                if let Object::DeviceBuilder { flags } = object {
                    *flags = value;
                }
            });
            Status::Ok
        },
        "nvnDeviceInitialize" => |instance, _, registers| {
            let flags = match instance.objects.get(registers.x[1]) {
                Some(Object::DeviceBuilder { flags }) => flags,
                _ => 0,
            };
            instance.objects.put(
                registers.x[0],
                Object::Device {
                    flags,
                    depth_mode: 0,
                    window_origin_mode: 0,
                },
            );
            succeed(registers)
        },
        "nvnDeviceGetInteger" => |instance, _, registers| {
            // Answer only selectors named by the device contract.
            // Anything else is left to the host.
            let Some(value) = device_integer(registers.x[1]) else {
                return Status::Unimplemented;
            };
            if write_u32(instance, registers.x[2], value) {
                registers.x[0] = 0;
                Status::Ok
            } else {
                Status::BadArgument
            }
        },
        "nvnDeviceSetDepthMode" => |instance, _, registers| {
            let value = registers.x[1];
            instance.objects.update(registers.x[0], |object| {
                if let Object::Device { depth_mode, .. } = object {
                    *depth_mode = value;
                }
            });
            Status::Ok
        },
        "nvnDeviceSetWindowOriginMode" => |instance, _, registers| {
            let value = registers.x[1];
            instance.objects.update(registers.x[0], |object| {
                if let Object::Device {
                    window_origin_mode, ..
                } = object
                {
                    *window_origin_mode = value;
                }
            });
            Status::Ok
        },
        // Registration succeeds because Vulkan does not require separate
        // fast-clear value registration.
        "nvnDeviceRegisterFastClearColor" | "nvnDeviceRegisterFastClearDepth" => {
            |_instance, _, registers| succeed(registers)
        }
        // A handle is whatever the program will hand back later. The pool
        // id it registered the object under is the simplest value that
        // identifies it. The host contract leaves the value open, so this
        // implementation uses the registered pool identifier.
        "nvnDeviceGetSeparateTextureHandle"
        | "nvnDeviceGetSeparateSamplerHandle"
        | "nvnDeviceGetImageHandle" => |_instance, _, registers| {
            registers.x[0] = registers.x[1];
            Status::Ok
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_answer_table_is_read() {
        assert_eq!(device_integer(0x00), Some(0x37));
        assert_eq!(device_integer(0x24), Some(0x100000));
        assert_eq!(device_integer(0x02), None);
    }

    #[test]
    fn a_device_is_built_from_its_builder() {
        let instance = Instance::new();
        let mut registers = Registers::default();
        registers.x[0] = 0x1000;
        handler("nvnDeviceBuilderSetDefaults").unwrap()(&instance, FunctionId(0), &mut registers);
        registers.x[1] = 0x100;
        handler("nvnDeviceBuilderSetFlags").unwrap()(&instance, FunctionId(0), &mut registers);
        registers.x[0] = 0x2000;
        registers.x[1] = 0x1000;
        assert_eq!(
            handler("nvnDeviceInitialize").unwrap()(&instance, FunctionId(0), &mut registers),
            Status::Ok
        );
        assert_eq!(registers.x[0], 1);
        assert_eq!(
            instance.objects.get(0x2000),
            Some(Object::Device {
                flags: 0x100,
                depth_mode: 0,
                window_origin_mode: 0
            })
        );
    }
}
