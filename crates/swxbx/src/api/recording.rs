//! Command buffers: set-up and recording bookkeeping.
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
        // Return the total command-memory size added by the program.
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
        // These synchronization commands take effect during recording because
        // this path does not submit them to the graphics processor.
        //
        // SignalEvent takes the value from the fourth register. The host
        // contract leaves the remaining argument order open.
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
        // Return a handle for later submission. Combine the command-buffer
        // address with a sequence count so successive recordings differ.
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
