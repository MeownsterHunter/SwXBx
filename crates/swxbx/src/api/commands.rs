use super::{objects::RecordedCommand, read_f32x4, read_u64, Handler};
use crate::instance::{Instance, Registers, Status};

pub(super) fn record(
    instance: &Instance,
    function: u32,
    registers: &mut Registers,
    command: RecordedCommand,
) -> Status {
    if !instance.objects.record_command(registers.x[0], command) {
        return Status::BadArgument;
    }
    let _ = function;
    // Recorded commands return nothing swxbx knows of. The result
    // registers are cleared, as for an unhandled call, so a caller that does
    // read a result gets zero rather than its own first argument back.
    registers.x[0] = 0;
    registers.x[1] = 0;
    registers.d[0] = 0;
    Status::Ok
}

/// Return the handler for a supported name in this command family.
pub fn handler(name: &str) -> Option<Handler> {
    if !name.starts_with("nvnCommandBuffer") {
        return None;
    }
    Some(match name {
        "nvnCommandBufferBindProgram" => |instance, _, r| {
            crate::api::resources::retry_pending(instance, r.x[1], "bind");
            record(instance, 0, r, RecordedCommand::BindProgram(r.x[1]))
        },
        "nvnCommandBufferSetRenderTargets" => |instance, _, r| {
            let count = r.x[1].min(16);
            let mut colors = Vec::with_capacity(count as usize);
            for i in 0..count {
                let Some(value) = read_u64(instance, r.x[2] + i * 8) else {
                    return Status::BadArgument;
                };
                colors.push(value);
            }
            record(
                instance,
                0,
                r,
                RecordedCommand::SetRenderTargets {
                    colors,
                    depth: r.x[4],
                    views: [r.x[3], r.x[5]],
                },
            )
        },
        "nvnCommandBufferClearColor" => |instance, _, r| {
            let Some(color) = read_f32x4(instance, r.x[2]) else {
                return Status::BadArgument;
            };
            record(
                instance,
                0,
                r,
                RecordedCommand::ClearColor {
                    index: r.x[1] as u32,
                    color,
                    mask: r.x[3] as u32,
                },
            )
        },
        "nvnCommandBufferClearDepthStencil" => |instance, _, r| {
            record(
                instance,
                0,
                r,
                RecordedCommand::ClearDepthStencil {
                    depth: f32::from_bits(r.d[0] as u32),
                    depth_write: r.x[2] as u32,
                    stencil: r.x[3] as u32,
                    stencil_mask: r.x[4] as u32,
                },
            )
        },
        // The host contract defines x1 as the source GPU address and x2 as
        // the destination texture. The remaining pointers stay opaque.
        "nvnCommandBufferCopyBufferToTexture" => |instance, _, r| {
            record(
                instance,
                0,
                r,
                RecordedCommand::CopyBufferToTexture {
                    buffer: r.x[1],
                    texture: r.x[2],
                },
            )
        },
        "nvnCommandBufferCopyTextureToTexture" => |instance, _, r| {
            record(
                instance,
                0,
                r,
                RecordedCommand::CopyTextureToTexture {
                    source: r.x[1],
                    destination: r.x[2],
                    arguments: r.x,
                },
            )
        },
        "nvnCommandBufferSetViewport" => |i, _, r| {
            record(
                i,
                0,
                r,
                RecordedCommand::SetViewport(r.x[..5].try_into().unwrap()),
            )
        },
        "nvnCommandBufferSetScissor" => |i, _, r| {
            record(
                i,
                0,
                r,
                RecordedCommand::SetScissor(r.x[..5].try_into().unwrap()),
            )
        },
        "nvnCommandBufferDrawArrays" => |i, _, r| {
            record(
                i,
                0,
                r,
                RecordedCommand::DrawArrays {
                    primitive: r.x[1] as u32,
                    first: r.x[2] as u32,
                    count: r.x[3] as u32,
                },
            )
        },
        "nvnCommandBufferDrawArraysInstanced" => |i, _, r| {
            record(
                i,
                0,
                r,
                RecordedCommand::DrawArraysInstanced {
                    primitive: r.x[1] as u32,
                    first: r.x[2] as u32,
                    count: r.x[3] as u32,
                    base_instance: r.x[4] as u32,
                    instances: r.x[5] as u32,
                },
            )
        },
        "nvnCommandBufferDrawElementsBaseVertex" => |i, _, r| {
            record(
                i,
                0,
                r,
                RecordedCommand::DrawElementsBaseVertex {
                    primitive: r.x[1] as u32,
                    index_type: r.x[2] as u32,
                    count: r.x[3] as u32,
                    indices: r.x[4],
                    // Signed low word is the host experiment's interpretation.
                    base_vertex: r.x[5] as i32,
                },
            )
        },
        "nvnCommandBufferSetDepthRange" => |i, _, r| {
            record(
                i,
                0,
                r,
                RecordedCommand::SetDepthRange([r.x[1], r.d[0], r.d[1]]),
            )
        },
        // Retain the command for a later executor, but report it as unimplemented
        // so the host can apply its fallback policy.
        _ => |instance, function, r| {
            let _ = record(
                instance,
                function.0,
                r,
                RecordedCommand::Raw {
                    function: function.0,
                    registers: r.x,
                },
            );
            Status::Unimplemented
        },
    })
}
