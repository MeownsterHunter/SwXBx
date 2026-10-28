//! Command buffers: set-up and recording bookkeeping.
//! Signatures: docs/signatures/0002-command-buffer.md.
//!
//! Recorded commands are handled in commands.rs and state_commands.rs.
//! This module manages the command buffer and its recording handles.

use super::{accept, objects::Object, queue::signal_event, succeed, Handler};
use crate::instance::Status;

/// Return the handler for a supported name in this command family.
pub fn handler(name: &str) -> Option<Handler> {
    Some(match name {
        "nvnCommandBufferInitialize" => |instance, _, registers| {
            instance.objects.put(
                registers.x[0],
                Object::CommandBuffer {
                    device: registers.x[1],
                    command_memory: Vec::new(),
                    control_memory: Vec::new(),
                },
            );
            succeed(registers)
        },
        "nvnCommandBufferFinalize" => |instance, _, registers| {
            instance.objects.remove(registers.x[0]);
            Status::Ok
        },
        "nvnCommandBufferAddCommandMemory" => |instance, _, registers| {
            let entry = (registers.x[1], registers.x[2], registers.x[3]);
            instance.objects.update(registers.x[0], |object| {
                if let Object::CommandBuffer { command_memory, .. } = object {
                    command_memory.push(entry);
                }
            });
            Status::Ok
        },
        "nvnCommandBufferAddControlMemory" => |instance, _, registers| {
            let entry = (registers.x[1], registers.x[2]);
            instance.objects.update(registers.x[0], |object| {
                if let Object::CommandBuffer { control_memory, .. } = object {
                    control_memory.push(entry);
                }
            });
            Status::Ok
        },
        // Observed to answer the size the program had added.
        "nvnCommandBufferGetCommandMemorySize" => |instance, _, registers| {
            registers.x[0] = match instance.objects.get(registers.x[0]) {
                Some(Object::CommandBuffer { command_memory, .. }) => {
                    command_memory.iter().map(|(_, _, size)| size).sum()
                }
                _ => 0,
            };
            Status::Ok
        },
        // The callbacks would ask the program for more memory; swxbx
        // records no commands and never needs any. Accepted and ignored.
        "nvnCommandBufferSetMemoryCallback"
        | "nvnCommandBufferSetMemoryCallbackData"
        | "nvnCommandBufferSetShaderScratchMemory" => accept,
        // Commands that the graphics processor would execute later and that
        // the program waits on. Nothing executes commands yet, so their
        // effect is produced at recording time (note 0011).
        //
        // SignalEvent: the value written is taken from the fourth register,
        // which was 1 in every observed call; signatures 0002 leaves the
        // argument order open.
        "nvnCommandBufferSignalEvent" => |instance, _, registers| {
            let value = registers.x[4] as u32;
            signal_event(instance, registers.x[1], if value == 0 { 1 } else { value });
            Status::Ok
        },
        "nvnCommandBufferBeginRecording" => |instance, _, registers| {
            if !instance.objects.begin_recording(registers.x[0]) {
                return Status::BadArgument;
            }
            Status::Ok
        },
        // Returns a handle the program later submits. The command buffer's
        // own address, with a count in the top bits so successive recordings
        // differ, is swxbx's choice; signatures 0002 leaves the real value
        // open.
        "nvnCommandBufferEndRecording" => |instance, _, registers| {
            let Some(handle) = instance.objects.end_recording(registers.x[0]) else {
                return Status::BadArgument;
            };
            registers.x[0] = handle;
            Status::Ok
        },
        _ => return None,
    })
}
