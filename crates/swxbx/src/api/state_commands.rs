//! Command-state recording under explicit host contracts.
//! Polygon offset retains a raw three-register hypothesis for opt-in execution.
//! No command here executes, writes
//! program memory, signals synchronization, or allocates rendering resources.

use super::{
    commands::record,
    objects::{Object, RecordedCommand, StateCommand},
    Handler,
};
use crate::{
    functions::{self, FunctionId},
    instance::{Instance, Registers, Status},
};

pub(super) const NAMES: &[&str] = &[
    "nvnCommandBufferBindBlendState",
    "nvnCommandBufferBindChannelMaskState",
    "nvnCommandBufferBindColorState",
    "nvnCommandBufferBindDepthStencilState",
    "nvnCommandBufferBindMultisampleState",
    "nvnCommandBufferBindPolygonState",
    "nvnCommandBufferBarrier",
    "nvnCommandBufferSetPolygonOffsetClamp",
    "nvnCommandBufferSetStencilMask",
    "nvnCommandBufferSetStencilRef",
    "nvnCommandBufferSetStencilValueMask",
    "nvnCommandBufferSetTiledCacheAction",
    "nvnCommandBufferBindUniformBuffer",
    "nvnCommandBufferBindVertexBuffer",
    "nvnCommandBufferBindSeparateTexture",
    "nvnCommandBufferBindImage",
    "nvnCommandBufferClearBuffer",
    "nvnCommandBufferDispatchCompute",
    "nvnCommandBufferBindVertexAttribState",
    "nvnCommandBufferBindVertexStreamState",
    "nvnCommandBufferBindSeparateSampler",
    "nvnCommandBufferFenceSync",
    "nvnCommandBufferSaveZCullData",
    "nvnCommandBufferRestoreZCullData",
];

pub(super) const TEXTURE_NAMES: &[&str] = &[
    "nvnCommandBufferBindTexture",
    "nvnCommandBufferSetTexturePool",
    "nvnCommandBufferSetSamplerPool",
];

/// Return the handler for a supported name in this command family.
pub fn handler(name: &str) -> Option<Handler> {
    (NAMES.contains(&name) || TEXTURE_NAMES.contains(&name)).then_some(record_state)
}

fn record_state(instance: &Instance, function: FunctionId, r: &mut Registers) -> Status {
    let Some(name) = functions::name(function).and_then(|n| n.strip_prefix("nvnCommandBuffer"))
    else {
        return Status::BadFunction;
    };
    let command = match name {
        "BindBlendState"
        | "BindChannelMaskState"
        | "BindColorState"
        | "BindDepthStencilState"
        | "BindMultisampleState"
        | "BindPolygonState" => {
            let kind = name.strip_prefix("Bind").expect("binding");
            let settings = match instance.objects.get(r.x[1]) {
                Some(Object::State {
                    kind: actual,
                    settings,
                }) if actual == kind => Some(settings),
                _ => None,
            };
            StateCommand::BindState {
                kind,
                address: r.x[1],
                settings,
            }
        }
        "SetStencilMask" | "SetStencilRef" | "SetStencilValueMask" => StateCommand::Stencil {
            setting: name.strip_prefix("Set").expect("setting"),
            faces: r.x[1],
            value: r.x[2],
        },
        "SetPolygonOffsetClamp" => StateCommand::PolygonOffset([r.d[0], r.d[1], r.d[2]]),
        "BindVertexAttribState" | "BindVertexStreamState" => {
            let kind = name.strip_prefix("Bind").expect("binding");
            let first_settings = if r.x[1] == 0 {
                None
            } else {
                match instance.objects.get(r.x[2]) {
                    Some(Object::State {
                        kind: actual,
                        settings,
                    }) if actual == kind => Some(settings),
                    _ => None,
                }
            };
            // Counted object spacing is a host choice, not a guest layout. 0028.
            #[cfg(feature = "vulkan")]
            let stride = instance.recording_stride(kind);
            #[cfg(not(feature = "vulkan"))]
            let stride: Option<u64> = None;
            let count = r.x[1];
            let address = r.x[2];
            let experiment_settings =
                (count <= 16 && stride.is_some() && (count <= 1 || stride != Some(0)))
                    .then(|| {
                        (0..count)
                            .map(|index| {
                                let at = index.checked_mul(stride?)?.checked_add(address)?;
                                match instance.objects.get(at) {
                                    Some(Object::State {
                                        kind: actual,
                                        settings,
                                    }) if actual == kind => Some(settings),
                                    _ => None,
                                }
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .flatten();
            StateCommand::BindStates {
                kind,
                count: r.x[1],
                address: r.x[2],
                first_settings,
                experiment_settings,
            }
        }
        "BindSeparateSampler" => StateCommand::BindSamplerReference {
            stage: r.x[1],
            index: r.x[2],
            reference: r.x[3],
        },
        "FenceSync" => StateCommand::FenceSync {
            sync: r.x[1],
            condition: r.x[2],
            flags: r.x[3],
        },
        "SaveZCullData" => StateCommand::SaveZCullData {
            address: r.x[1],
            size: r.x[2],
        },
        "RestoreZCullData" => StateCommand::RestoreZCullData {
            address: r.x[1],
            size: r.x[2],
        },
        "Barrier" => StateCommand::Barrier(r.x[1]),
        "SetTiledCacheAction" => StateCommand::TiledCacheAction(r.x[1]),
        "BindUniformBuffer" => StateCommand::BindUniformBuffer {
            stage: r.x[1],
            index: r.x[2],
            address: r.x[3],
            size: r.x[4],
        },
        "BindVertexBuffer" => StateCommand::BindVertexBuffer {
            index: r.x[1],
            address: r.x[2],
            size: r.x[3],
        },
        "SetTexturePool" | "SetSamplerPool" => StateCommand::SetDescriptorPool {
            sampler: name == "SetSamplerPool",
            pool: r.x[1],
        },
        "BindTexture" | "BindSeparateTexture" | "BindImage" => StateCommand::BindHandle {
            kind: name.strip_prefix("Bind").expect("binding"),
            stage: r.x[1],
            index: r.x[2],
            handle: r.x[3],
        },
        "ClearBuffer" => StateCommand::ClearBuffer {
            address: r.x[1],
            size: r.x[2],
            value: r.x[3],
        },
        "DispatchCompute" => StateCommand::DispatchCompute([r.x[1], r.x[2], r.x[3]]),
        _ => return Status::BadFunction,
    };
    record(instance, function.0, r, RecordedCommand::State(command))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(instance: &Instance, suffix: &str, r: &mut Registers) -> Status {
        let id = functions::lookup(&format!(
            "{}{suffix}",
            &functions::all().next().unwrap().1[..3]
        ))
        .unwrap();
        instance.call(id, r)
    }

    fn start(instance: &Instance, address: u64) {
        let mut r = Registers::default();
        r.x[0] = address;
        r.x[1] = 0x100;
        assert_eq!(
            call(instance, "CommandBufferInitialize", &mut r),
            Status::Ok
        );
        r.x[0] = address;
        assert_eq!(
            call(instance, "CommandBufferBeginRecording", &mut r),
            Status::Ok
        );
    }

    fn end(instance: &Instance, address: u64) -> Vec<RecordedCommand> {
        let mut r = Registers::default();
        r.x[0] = address;
        assert_eq!(
            call(instance, "CommandBufferEndRecording", &mut r),
            Status::Ok
        );
        instance.objects.recording(r.x[0]).unwrap()
    }

    #[test]
    fn polygon_offset_keeps_full_float_words_in_order() {
        let instance = Instance::new();
        start(&instance, 0x200);
        let mut r = Registers::default();
        r.x[0] = 0x200;
        r.d[..3].copy_from_slice(&[
            0x1234_5678_0000_0001,
            0x2345_6789_0000_0002,
            0x3456_789a_0000_0003,
        ]);
        assert_eq!(
            call(&instance, "CommandBufferSetPolygonOffsetClamp", &mut r),
            Status::Ok
        );
        assert_eq!(
            end(&instance, 0x200),
            vec![RecordedCommand::State(StateCommand::PolygonOffset([
                0x1234_5678_0000_0001,
                0x2345_6789_0000_0002,
                0x3456_789a_0000_0003
            ]))]
        );
    }
    #[test]
    fn scalar_commands_keep_order_and_full_values_without_executing() {
        use StateCommand::*;
        let instance = Instance::new();
        start(&instance, 0x200);
        // Exercise the full register width.
        let a = 0x1234_5678_0000_0001;
        let b = 0x2345_6789_0000_0002;
        let c = 0x3456_789a_0000_0003;
        let d = 0x4567_89ab_0000_0004;
        let cases = [
            ("CommandBufferBarrier", Barrier(a)),
            ("CommandBufferSetTiledCacheAction", TiledCacheAction(a)),
            (
                "CommandBufferSetStencilMask",
                Stencil {
                    setting: "StencilMask",
                    faces: a,
                    value: b,
                },
            ),
            (
                "CommandBufferSetStencilRef",
                Stencil {
                    setting: "StencilRef",
                    faces: a,
                    value: b,
                },
            ),
            (
                "CommandBufferSetStencilValueMask",
                Stencil {
                    setting: "StencilValueMask",
                    faces: a,
                    value: b,
                },
            ),
            (
                "CommandBufferBindUniformBuffer",
                BindUniformBuffer {
                    stage: a,
                    index: b,
                    address: c,
                    size: d,
                },
            ),
            (
                "CommandBufferBindVertexBuffer",
                BindVertexBuffer {
                    index: a,
                    address: b,
                    size: c,
                },
            ),
            (
                "CommandBufferBindSeparateTexture",
                BindHandle {
                    kind: "SeparateTexture",
                    stage: a,
                    index: b,
                    handle: c,
                },
            ),
            (
                "CommandBufferBindImage",
                BindHandle {
                    kind: "Image",
                    stage: a,
                    index: b,
                    handle: c,
                },
            ),
            (
                "CommandBufferClearBuffer",
                ClearBuffer {
                    address: a,
                    size: b,
                    value: c,
                },
            ),
            ("CommandBufferDispatchCompute", DispatchCompute([a, b, c])),
            (
                "CommandBufferBindVertexAttribState",
                BindStates {
                    kind: "VertexAttribState",
                    count: a,
                    address: b,
                    first_settings: None,
                    experiment_settings: None,
                },
            ),
            (
                "CommandBufferBindVertexStreamState",
                BindStates {
                    kind: "VertexStreamState",
                    count: a,
                    address: b,
                    first_settings: None,
                    experiment_settings: None,
                },
            ),
            (
                "CommandBufferBindSeparateSampler",
                BindSamplerReference {
                    stage: a,
                    index: b,
                    reference: c,
                },
            ),
            (
                "CommandBufferFenceSync",
                FenceSync {
                    sync: a,
                    condition: b,
                    flags: c,
                },
            ),
            (
                "CommandBufferSaveZCullData",
                SaveZCullData {
                    address: a,
                    size: b,
                },
            ),
            (
                "CommandBufferRestoreZCullData",
                RestoreZCullData {
                    address: a,
                    size: b,
                },
            ),
        ];
        for (suffix, _) in &cases {
            let mut r = Registers {
                x: [0x200, a, b, c, d, 0xdead, 0xbeef, 0xbad],
                d: [0xf00d; 8],
                ..Registers::default()
            };
            assert_eq!(call(&instance, suffix, &mut r), Status::Ok, "{suffix}");
            // Same result convention as the existing command recorder.
            assert_eq!((r.x[0], r.x[1], r.d[0]), (0, 0, 0));
            assert_eq!((r.x[2], r.x[7], r.d[7]), (b, 0xbad, 0xf00d));
        }
        assert_eq!(
            end(&instance, 0x200),
            cases
                .into_iter()
                .map(|(_, command)| RecordedCommand::State(command))
                .collect::<Vec<_>>()
        );
        // No host is needed to record even an unreadable address or handle.
        assert_eq!(instance.objects.get(a), None);
    }

    #[test]
    fn bindings_snapshot_known_settings_and_leave_unknown_layouts_alone() {
        let instance = Instance::new();
        start(&instance, 0x200);
        let kinds = [
            "BlendState",
            "ChannelMaskState",
            "ColorState",
            "DepthStencilState",
            "MultisampleState",
            "PolygonState",
        ];
        let mut expected = Vec::new();
        for kind in kinds {
            instance.objects.put(
                0x300,
                Object::State {
                    kind,
                    settings: vec![("Observed", [7; 6])],
                },
            );
            let mut r = Registers::default();
            r.x[0] = 0x200;
            r.x[1] = 0x300;
            assert_eq!(
                call(&instance, &format!("CommandBufferBind{kind}"), &mut r),
                Status::Ok
            );
            expected.push(RecordedCommand::State(StateCommand::BindState {
                kind,
                address: 0x300,
                settings: Some(vec![("Observed", [7; 6])]),
            }));
            // Mutation after binding must not change the earlier record.
            instance.objects.put(
                0x300,
                Object::State {
                    kind,
                    settings: vec![("Observed", [9; 6])],
                },
            );
            r.x[0] = 0x200;
            r.x[1] = 0x300;
            assert_eq!(
                call(&instance, &format!("CommandBufferBind{kind}"), &mut r),
                Status::Ok
            );
            expected.push(RecordedCommand::State(StateCommand::BindState {
                kind,
                address: 0x300,
                settings: Some(vec![("Observed", [9; 6])]),
            }));
        }
        for address in [0x300, 0x999] {
            let mut r = Registers::default();
            r.x[0] = 0x200;
            r.x[1] = address;
            // The last object above is a polygon state, not a blend state.
            assert_eq!(
                call(&instance, "CommandBufferBindBlendState", &mut r),
                Status::Ok
            );
            expected.push(RecordedCommand::State(StateCommand::BindState {
                kind: "BlendState",
                address,
                settings: None,
            }));
        }
        assert_eq!(end(&instance, 0x200), expected);
        assert_eq!(instance.objects.get(0x999), None);
    }

    #[test]
    fn counted_bindings_snapshot_only_the_known_base_object() {
        let instance = Instance::new();
        start(&instance, 0x200);
        let mut expected = Vec::new();
        for kind in ["VertexAttribState", "VertexStreamState"] {
            let mut r = Registers::default();
            r.x[0] = 0x300;
            assert_eq!(
                call(&instance, &format!("{kind}SetDefaults"), &mut r),
                Status::Ok
            );
            let setting = if kind == "VertexAttribState" {
                "StreamIndex"
            } else {
                "Stride"
            };
            for value in [1, 2] {
                r = Registers::default();
                r.x[0] = 0x300;
                r.x[1] = value;
                assert_eq!(
                    call(&instance, &format!("{kind}Set{setting}"), &mut r),
                    Status::Ok
                );
                r.x[..3].copy_from_slice(&[0x200, 5, 0x300]);
                assert_eq!(
                    call(&instance, &format!("CommandBufferBind{kind}"), &mut r),
                    Status::Ok
                );
                expected.push(RecordedCommand::State(StateCommand::BindStates {
                    kind,
                    count: 5,
                    address: 0x300,
                    first_settings: Some(
                        (1..=value).map(|v| (setting, [v, 0, 0, 0, 0, 0])).collect(),
                    ),
                    experiment_settings: None,
                }));
            }
            // Zero count, unknown base and another state family stay opaque.
            instance.objects.put(
                0x400,
                Object::State {
                    kind: "BlendState",
                    settings: vec![],
                },
            );
            for (count, address) in [(0, 0x300), (0, 0), (2, 0x999), (1, 0x400)] {
                r.x[..3].copy_from_slice(&[0x200, count, address]);
                assert_eq!(
                    call(&instance, &format!("CommandBufferBind{kind}"), &mut r),
                    Status::Ok
                );
                expected.push(RecordedCommand::State(StateCommand::BindStates {
                    kind,
                    count,
                    address,
                    first_settings: None,
                    experiment_settings: None,
                }));
            }
        }
        assert_eq!(end(&instance, 0x200), expected);
        assert_eq!(instance.objects.get(0x999), None);
    }

    #[test]
    fn new_handlers_never_follow_references_and_respect_recording_lifetimes() {
        use crate::instance::Host;
        use std::{
            ffi::c_void,
            sync::atomic::{AtomicU64, Ordering},
        };

        unsafe extern "C" fn read(user: *mut c_void, _: u64, _: *mut u8, _: u64) -> i32 {
            // SAFETY: user points to the live counter below.
            unsafe { &*user.cast::<AtomicU64>() }.fetch_add(1, Ordering::Relaxed);
            1
        }
        let mut reads = Box::new(AtomicU64::new(0));
        // SAFETY: the counter outlives the instance and is safe to share.
        let instance = unsafe {
            Instance::with_host(Host {
                user: (&mut *reads as *mut AtomicU64).cast(),
                read_memory: Some(read),
                ..Host::default()
            })
        };
        for name in &NAMES[17..] {
            start(&instance, 0x200);
            start(&instance, 0x300);
            reads.store(0, Ordering::Relaxed);
            let id = functions::lookup(name).unwrap();
            let mut r = Registers::default();
            // Direct handler calls exclude diagnostic register sampling.
            for address in [0x200, 0x300] {
                r.x = [address, u64::MAX, u64::MAX, u64::MAX, 4, 5, 6, 7];
                assert_eq!(record_state(&instance, id, &mut r), Status::Ok);
            }
            assert_eq!(reads.load(Ordering::Relaxed), 0, "{name}");
            let first = end(&instance, 0x200);
            assert_eq!(first.len(), 1, "{name}");
            // Calls after EndRecording do not enter the next recording.
            r.x[..4].copy_from_slice(&[0x200, 1, 2, 3]);
            assert_eq!(record_state(&instance, id, &mut r), Status::Ok);
            r.x[0] = 0x200;
            assert_eq!(
                call(&instance, "CommandBufferBeginRecording", &mut r),
                Status::Ok
            );
            assert!(end(&instance, 0x200).is_empty());
            assert_eq!(end(&instance, 0x300), first, "{name}");
        }
    }

    #[test]
    fn recording_lifetimes_keep_buffers_independent() {
        let instance = Instance::new();
        start(&instance, 0x200);
        start(&instance, 0x300);
        let mut r = Registers::default();
        r.x[0] = 0x200;
        r.x[1] = 0x12;
        assert_eq!(call(&instance, "CommandBufferBarrier", &mut r), Status::Ok);
        r.x[0] = 0x300;
        r.x[1] = 0x40;
        assert_eq!(call(&instance, "CommandBufferBarrier", &mut r), Status::Ok);
        assert_eq!(
            end(&instance, 0x200),
            vec![RecordedCommand::State(StateCommand::Barrier(0x12))]
        );
        // Recording outside Begin/End follows the existing recorder: ignored.
        r.x[0] = 0x200;
        assert_eq!(call(&instance, "CommandBufferBarrier", &mut r), Status::Ok);
        r.x[0] = 0x200;
        assert_eq!(
            call(&instance, "CommandBufferBeginRecording", &mut r),
            Status::Ok
        );
        assert!(end(&instance, 0x200).is_empty());
        assert_eq!(
            end(&instance, 0x300),
            vec![RecordedCommand::State(StateCommand::Barrier(0x40))]
        );
    }

    #[test]
    fn setter_changes_do_not_rewrite_bound_state() {
        let instance = Instance::new();
        start(&instance, 0x200);
        let mut r = Registers::default();
        r.x[0] = 0x300;
        assert_eq!(
            call(&instance, "PolygonStateSetDefaults", &mut r),
            Status::Ok
        );
        for value in [1, 2] {
            r = Registers::default();
            r.x[0] = 0x300;
            r.x[1] = value;
            assert_eq!(
                call(&instance, "PolygonStateSetCullFace", &mut r),
                Status::Ok
            );
            r.x[0] = 0x200;
            r.x[1] = 0x300;
            assert_eq!(
                call(&instance, "CommandBufferBindPolygonState", &mut r),
                Status::Ok
            );
        }
        assert_eq!(
            end(&instance, 0x200),
            vec![
                RecordedCommand::State(StateCommand::BindState {
                    kind: "PolygonState",
                    address: 0x300,
                    settings: Some(vec![("CullFace", [1, 0, 0, 0, 0, 0])]),
                }),
                RecordedCommand::State(StateCommand::BindState {
                    kind: "PolygonState",
                    address: 0x300,
                    settings: Some(vec![
                        ("CullFace", [1, 0, 0, 0, 0, 0]),
                        ("CullFace", [2, 0, 0, 0, 0, 0]),
                    ]),
                }),
            ]
        );
    }

    #[test]
    fn submission_consumes_state_without_writing_or_rendering() {
        use super::super::objects::{TextureDescription, TextureImage};
        use crate::instance::Host;
        use std::{
            ffi::c_void,
            sync::{
                atomic::{AtomicU64, Ordering},
                Arc, Mutex,
            },
        };

        struct Memory {
            handle: AtomicU64,
            writes: AtomicU64,
            buffer: [u8; 16],
        }
        unsafe extern "C" fn read(user: *mut c_void, address: u64, out: *mut u8, size: u64) -> i32 {
            // SAFETY: the boxed Memory outlives the instance; out is provided
            // by the memory callback contract and is valid for size bytes.
            let memory = unsafe { &*user.cast::<Memory>() };
            let handle = memory.handle.load(Ordering::Relaxed).to_le_bytes();
            let bytes: &[u8] = match address {
                0x1000 if size <= 8 => &handle,
                0x8000 if size <= 16 => &memory.buffer,
                _ => return 1,
            };
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, size as usize);
            }
            0
        }
        unsafe extern "C" fn write(user: *mut c_void, _: u64, _: *const u8, _: u64) -> i32 {
            // SAFETY: same live boxed Memory as the read callback.
            unsafe { &*user.cast::<Memory>() }
                .writes
                .fetch_add(1, Ordering::Relaxed);
            1
        }
        let mut memory = Box::new(Memory {
            handle: AtomicU64::new(0),
            writes: AtomicU64::new(0),
            buffer: [0xa5; 16],
        });
        // SAFETY: memory and callbacks remain valid for the instance lifetime.
        let instance = unsafe {
            Instance::with_host(Host {
                user: (&mut *memory as *mut Memory).cast(),
                read_memory: Some(read),
                write_memory: Some(write),
                present: None,
                render_scale: 1.0,
                wait_vblank: None,
                vulkan: std::ptr::null(),
            })
        };
        let pixels = Arc::new(Mutex::new(Some(TextureImage {
            width: 1,
            height: 1,
            depth: 1,
            pixels: vec![11, 22, 33, 44],
        })));
        instance.objects.put(
            0x500,
            Object::Texture {
                description: TextureDescription::default(),
                image: pixels.clone(),
            },
        );
        let pool = Object::MemoryPool {
            device: 0x100,
            flags: 0,
            storage: 0x8000,
            size: 16,
            gpu_address: None,
            observed_gpu_address: None,
        };
        instance.objects.put(0x600, pool.clone());
        let sync = Object::Sync { device: 0x100 };
        instance.objects.put(0x8000, sync.clone());
        start(&instance, 0x200);
        for name in NAMES {
            let mut r = Registers::default();
            r.x[..5].copy_from_slice(&[0x200, 0x8000, 16, 0, 8]);
            assert_eq!(
                instance.call(functions::lookup(name).unwrap(), &mut r),
                Status::Ok
            );
        }
        let mut r = Registers::default();
        r.x[0] = 0x200;
        assert_eq!(
            call(&instance, "CommandBufferEndRecording", &mut r),
            Status::Ok
        );
        let handle = r.x[0];
        memory.handle.store(handle, Ordering::Relaxed);
        // Assert that the submission actually consumed a nonempty recording.
        assert_eq!(instance.objects.recording_len(handle), Some(NAMES.len()));
        r.x[..3].copy_from_slice(&[0x700, 1, 0x1000]);
        #[cfg(feature = "vulkan")]
        let expected = if instance.gpu.lock().unwrap().is_some() {
            Status::Unimplemented
        } else {
            Status::Ok
        };
        #[cfg(not(feature = "vulkan"))]
        let expected = Status::Ok;
        assert_eq!(call(&instance, "QueueSubmitCommands", &mut r), Status::Ok);
        assert_eq!(instance.queue.drain().status(), expected);
        assert!(instance.objects.recording(handle).is_none());
        assert_eq!(memory.writes.load(Ordering::Relaxed), 0);
        assert_eq!(memory.buffer, [0xa5; 16]);
        assert_eq!(instance.objects.get(0x600), Some(pool));
        assert_eq!(instance.objects.get(0x8000), Some(sync));
        assert_eq!(
            pixels.lock().unwrap().as_ref().unwrap().pixels,
            [11, 22, 33, 44]
        );
    }

    #[test]
    fn open_commands_keep_raw_arguments_and_resolver_never_invents_an_address() {
        let instance = Instance::new();
        start(&instance, 0x200);
        let arguments = Registers {
            x: [0x200, 0x300, 0, 0x400, 0x500, 0xf, 6, 7],
            d: [0, 0, 0x7f7fffff, 3, 4, 5, 6, 7],
            sp: 0x900,
        };
        let mut expected = Vec::new();
        {
            let name = "nvnCommandBufferClearTexture";
            let id = functions::lookup(name).unwrap();
            let mut r = arguments;
            assert_eq!(instance.call(id, &mut r), Status::Ok);
            expected.push(RecordedCommand::Operation {
                kind: "ClearTexture",
                arguments: arguments.x[1..].try_into().unwrap(),
            });
            // Opaque inputs remain untouched during recording.
            assert_eq!(&r.d[1..], &arguments.d[1..]);
        }
        assert_eq!(end(&instance, 0x200), expected);
        let resolver = functions::lookup("nvnDeviceGetProcAddress").unwrap();
        for device in [0, 0x100] {
            let mut r = arguments;
            r.x[..2].copy_from_slice(&[device, 0x400]);
            assert_eq!(instance.call(resolver, &mut r), Status::Unimplemented);
            assert_eq!(r.x[0], 0);
        }
    }
}
