//! State objects: blend, channel mask, colour, depth-stencil, polygon,
//! multisample, vertex attribute and vertex stream state.
//! Signatures: docs/signatures/0003-objects.md.
//!
//! The settings are recorded as the program gave them, under the setting's
//! name (the function name without the kind prefix and "Set"). What each
//! value means is not interpreted yet; that comes with drawing.

use super::{objects::Object, Handler};
use crate::functions::{self, FunctionId};
use crate::instance::{Instance, Registers, Status};

const KINDS: &[&str] = &[
    "BlendState",
    "ChannelMaskState",
    "ColorState",
    "DepthStencilState",
    "PolygonState",
    "MultisampleState",
    "VertexAttribState",
    "VertexStreamState",
];

/// Splits a state function's name into its kind and its setting, or `None`
/// for any other function.
fn kind_and_setting(name: &str) -> Option<(&'static str, &str)> {
    functions::lookup(name)?;
    let rest = name.get(3..)?;
    let kind = KINDS.iter().find(|kind| rest.starts_with(**kind))?;
    let setting = rest[kind.len()..].strip_prefix("Set")?;
    Some((kind, setting))
}

/// Return the handler for a supported name in this command family.
pub fn handler(name: &str) -> Option<Handler> {
    let (_, setting) = kind_and_setting(name)?;
    Some(if setting == "Defaults" {
        set_defaults
    } else {
        set_value
    })
}

fn set_defaults(instance: &Instance, function: FunctionId, registers: &mut Registers) -> Status {
    let kind = functions::name(function)
        .and_then(kind_and_setting)
        .map_or("State", |(kind, _)| kind);
    instance.objects.put(
        registers.x[0],
        Object::State {
            kind,
            settings: Vec::new(),
        },
    );
    Status::Ok
}

fn set_value(instance: &Instance, function: FunctionId, registers: &mut Registers) -> Status {
    let Some((kind, setting)) = functions::name(function).and_then(kind_and_setting) else {
        return Status::BadFunction;
    };
    let arguments = [
        registers.x[1],
        registers.x[2],
        registers.x[3],
        registers.x[4],
        registers.x[5],
        registers.d[0],
    ];
    let found = instance.objects.update(registers.x[0], |object| {
        if let Object::State { settings, .. } = object {
            settings.push((setting, arguments));
        }
    });
    if !found {
        // A state object the program filled without SetDefaults first, or
        // one swxbx never saw. Start one.
        instance.objects.put(
            registers.x[0],
            Object::State {
                kind,
                settings: vec![(setting, arguments)],
            },
        );
    }
    Status::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_names_split_into_kind_and_setting() {
        assert_eq!(
            kind_and_setting("nvnDepthStencilStateSetDepthFunc"),
            Some(("DepthStencilState", "DepthFunc"))
        );
        assert_eq!(
            kind_and_setting("nvnBlendStateSetDefaults"),
            Some(("BlendState", "Defaults"))
        );
        assert_eq!(kind_and_setting("nvnDeviceInitialize"), None);
        assert_eq!(kind_and_setting("nvnCommandBufferBindBlendState"), None);
    }

    #[test]
    fn settings_are_recorded_under_the_object() {
        let instance = Instance::new();
        let defaults = functions::lookup("nvnPolygonStateSetDefaults").unwrap();
        let cull = functions::lookup("nvnPolygonStateSetCullFace").unwrap();
        let mut registers = Registers::default();
        registers.x[0] = 0x5000;
        set_defaults(&instance, defaults, &mut registers);
        registers.x[1] = 2;
        set_value(&instance, cull, &mut registers);
        assert_eq!(
            instance.objects.get(0x5000),
            Some(Object::State {
                kind: "PolygonState",
                settings: vec![("CullFace", [2, 0, 0, 0, 0, 0])],
            })
        );
    }
}
