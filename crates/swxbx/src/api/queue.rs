//! Queue, sync, event and window objects.
//! Signatures: docs/signatures/0001-setup.md, 0003-objects.md, 0004-pointers.md.

use super::{
    accept,
    objects::{Object, RecordedCommand, TextureDescription, TextureImage},
    read_u64, succeed, write_u32, Handler,
};
use crate::instance::{Registers, Status};

fn builder_update(
    instance: &crate::instance::InstanceState,
    registers: &Registers,
    change: impl FnOnce(&mut Object),
) -> Status {
    instance.objects.update(registers.x[0], change);
    Status::Ok
}

const COPY_CHUNK: usize = 64 * 1024;

fn read_pool_bytes(
    instance: &crate::instance::InstanceState,
    address: u64,
    size: usize,
) -> Option<Vec<u8>> {
    let resolution = instance.objects.resolve_gpu_address(address).ok()?;
    if size as u64 > resolution.remaining {
        return None;
    }
    let mut bytes = vec![0; size];
    for (offset, chunk) in bytes.chunks_mut(COPY_CHUNK).enumerate() {
        let address = resolution
            .program_address
            .checked_add((offset * COPY_CHUNK) as u64)?;
        if !instance.read_memory(address, chunk) {
            return None;
        }
    }
    Some(bytes)
}

/// Bytes in a texture's base level at four bytes per texel, or `None` when
/// the recorded size is larger than swxbx's image limit.
fn copy_size(description: &TextureDescription) -> Option<usize> {
    let texels = description
        .width
        .max(1)
        .checked_mul(description.height.max(1))?
        .checked_mul(description.depth.max(1))?;
    (texels <= CPU_IMAGE_TEXELS).then(|| (texels * 4) as usize)
}

fn copy_cpu(
    instance: &crate::instance::InstanceState,
    source: u64,
    destination: u64,
    data: Vec<u8>,
) {
    let Some(Object::Texture { image, description }) = instance.objects.get(destination) else {
        return;
    };
    let level = TextureDescription {
        depth: 1,
        ..description.clone()
    };
    let Some(size) = copy_size(&level) else {
        return;
    };
    let mut image = image.lock().unwrap_or_else(|p| p.into_inner());
    let target = image.get_or_insert_with(|| TextureImage {
        width: description.width.max(1) as u32,
        height: description.height.max(1) as u32,
        depth: 1,
        pixels: vec![0; size],
    });
    let _ = source;
    let length = target.pixels.len().min(data.len());
    target.pixels[..length].copy_from_slice(&data[..length]);
}

/// Marks an event signalled, in swxbx's record and in the event's storage
/// word in program memory.
/// Largest CPU image, in texels: 4096 by 4096.
const CPU_IMAGE_TEXELS: u64 = 4096 * 4096;

pub(crate) fn signal_event(instance: &crate::instance::InstanceState, event: u64, new_value: u32) {
    let mut storage = 0;
    instance.objects.update(event, |object| {
        if let Object::Event { value, storage: at } = object {
            *value = new_value;
            storage = *at;
        }
    });
    if storage != 0 {
        write_u32(instance, storage, new_value);
    }
}

pub(crate) fn execute(
    instance: &crate::instance::InstanceState,
    buffers: Vec<(u64, Vec<RecordedCommand>)>,
) -> Status {
    for program in instance.pending_programs() {
        crate::api::resources::retry_pending(instance, program, "submit");
    }
    instance.unsupported_commands.submitted();
    // Only textures a window presents get a CPU image. A program
    // clears thousands of other render targets, and filling CPU
    // copies of all of them stalled the program it was tried on.
    let presented = instance.objects.window_textures();
    #[cfg(feature = "vulkan")]
    let mut gpu = instance.gpu.lock().unwrap_or_else(|p| p.into_inner());
    #[cfg(feature = "vulkan")]
    let host_writes = instance.take_host_writes();
    #[cfg(feature = "vulkan")]
    if let Some(backend) = gpu.as_mut() {
        backend.set_parallel_recording(buffers.len() > 1);
    }
    #[cfg(feature = "vulkan")]
    if let Some(memory) = gpu
        .as_mut()
        .and_then(|backend| backend.global_memory.as_mut())
    {
        #[cfg(feature = "draw-metrics")]
        let _span = crate::draw_metrics::PipelineSpan::new(13);
        memory.track_submissions();
        for range in host_writes.intersections(0..u64::MAX) {
            memory.notify_storage_write(range.start, range.end - range.start);
        }
        if !memory.upload(|address, bytes| instance.read_memory(address, bytes)) {
            return Status::BadArgument;
        }
    }
    #[cfg(feature = "vulkan")]
    if let Some(backend) = gpu.as_mut() {
        for range in host_writes.intersections(0..u64::MAX) {
            backend.host_wrote(range.start, range.end - range.start);
        }
    }
    let submitted = (|| {
        for (handle, commands) in buffers {
            #[cfg(feature = "vulkan")]
            if let Some(backend) = gpu.as_mut() {
                backend.reserve_draws(commands.len());
            }
            let mut targets = Vec::new();
            let mut depth_target = 0;
            let mut target_views = [0, 0];
            #[cfg(feature = "vulkan")]
            let mut draw_state = super::drawing::State::default();
            #[cfg(feature = "vulkan")]
            let mut operation_state = super::operations::State::default();
            let mut commands = commands;
            'commands: for command in commands.drain(..) {
                // A stopped batch names the reason its own command gave.
                crate::clear_last_error();
                let status = 'command: {
                    #[cfg(feature = "vulkan")]
                    if !matches!(
                        &command,
                        RecordedCommand::DrawArrays { .. }
                            | RecordedCommand::DrawElementsBaseVertex { .. }
                            | RecordedCommand::SetViewport(_)
                            | RecordedCommand::SetScissor(_)
                            | RecordedCommand::SetDepthRange(_)
                            | RecordedCommand::BindProgram(_)
                            | RecordedCommand::BindState { .. }
                            | RecordedCommand::State(_)
                    ) {
                        if let Some(backend) = gpu.as_mut() {
                            if backend.finish_draws().is_none() {
                                break 'command crate::unsupported::refuse(Status::InternalError, "the device could not finish the draws recorded before this command");
                            }
                        }
                    }
                    #[cfg(feature = "vulkan")]
                    if let Some(backend) = gpu.as_mut() {
                        if let Some(result) = operation_state.execute(
                            &command,
                            instance,
                            backend,
                            &mut draw_state,
                            super::operations::Targets {
                                colors: &targets,
                                depth: depth_target,
                                views: target_views,
                            },
                        ) {
                            if let Err(status) = result {
                                break 'command status;
                            }
                            continue 'commands;
                        }
                        if !operation_state.enabled
                            && matches!(
                                command,
                                RecordedCommand::DrawArrays { .. }
                                    | RecordedCommand::DrawElementsBaseVertex { .. }
                            )
                        {
                            continue 'commands;
                        }
                    }
                    match command {
                        RecordedCommand::SetRenderTargets {
                            colors,
                            depth,
                            views,
                        } => {
                            targets = colors;
                            depth_target = depth;
                            target_views = views;
                            continue 'commands;
                        }
                        RecordedCommand::ClearColor { index, color, mask } => {
                            let Some(texture) = targets.get(index as usize).copied() else {
                                continue 'commands;
                            };
                            let Some(Object::Texture { description, image }) =
                                instance.objects.get(texture)
                            else {
                                break 'command crate::unsupported::refuse(Status::BadArgument, format_args!("ClearColor: colour target {index} is object {texture}, which is not a texture"));
                            };
                            #[cfg(feature = "vulkan")]
                            if let Some(backend) = gpu.as_mut() {
                                if !backend.ensure_texture(texture, &description, false) {
                                    break 'command crate::unsupported::refuse(
                                        Status::BadArgument,
                                        format_args!(
                                            "ClearColor: texture {texture} has no device image"
                                        ),
                                    );
                                }
                                if let Err(status) = backend.clear_color(texture, color, mask) {
                                    break 'command status;
                                }
                                continue 'commands;
                            }
                            if !presented.contains(&texture) {
                                continue 'commands;
                            }
                            // Only the first layer of the base level is kept on the
                            // CPU, and only for sizes a screen can have: a game
                            // clears large arrays and volumes every frame, and
                            // copying those on the CPU stalled it. swxbx's own
                            // limit, until clears run on the GPU.
                            let (width, height) =
                                (description.width.max(1), description.height.max(1));
                            if width * height > CPU_IMAGE_TEXELS {
                                continue 'commands;
                            }
                            let mut image = image.lock().unwrap_or_else(|p| p.into_inner());
                            let image = image.get_or_insert_with(|| TextureImage {
                                width: width as u32,
                                height: height as u32,
                                depth: 1,
                                pixels: vec![0; (width * height * 4) as usize],
                            });
                            // Bits 0..3 mean red, green, blue and alpha. This is swxbx's own choice.
                            let values = color.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
                            for pixel in image.pixels.as_chunks_mut::<4>().0 {
                                for c in 0..4 {
                                    if mask & (1 << c) != 0 {
                                        pixel[c] = values[c];
                                    }
                                }
                            }
                        }
                        RecordedCommand::ClearDepthStencil {
                            depth,
                            depth_write,
                            stencil,
                            stencil_mask,
                        } => {
                            #[cfg(not(feature = "vulkan"))]
                            let _ = (depth, depth_write, stencil, stencil_mask, depth_target);
                            let _ = stencil;
                            #[cfg(feature = "vulkan")]
                            if let Some(backend) = gpu.as_mut() {
                                // The clear writes the depth target in the
                                // format the depth table gives its token, the
                                // image a draw uses. Source: the public contract.
                                let reject = |status, reason: std::fmt::Arguments<'_>| {
                                    crate::unsupported::refuse(
                                        status,
                                        format_args!(
                                            "clear depth-stencil of depth target \
                                             {depth_target:#x}: {reason}"
                                        ),
                                    )
                                };
                                let stencil_bits = stencil_mask & 0xff;
                                if (depth_write == 0 && stencil_bits == 0) || depth_target == 0 {
                                    continue 'commands;
                                }
                                if stencil_bits != 0 && stencil_bits != 0xff {
                                    break 'command reject(
                                        Status::Unimplemented,
                                        format_args!(
                                            "stencil mask {stencil_mask:#x} keeps some stencil \
                                             bits; swxbx clears all eight or none"
                                        ),
                                    );
                                }
                                let Some(Object::Texture { description, .. }) =
                                    instance.objects.get(depth_target)
                                else {
                                    break 'command reject(
                                        Status::BadArgument,
                                        format_args!("it is not a texture"),
                                    );
                                };
                                let format =
                                    match backend.ensure_depth_clear(depth_target, &description) {
                                        Ok(format) => format,
                                        Err((status, why)) => {
                                            break 'command reject(status, format_args!("{why}"))
                                        }
                                    };
                                if stencil_bits != 0
                                    && crate::gpu::formats::depth_planes(format)
                                        .is_none_or(|p| p.stencil == 0)
                                {
                                    break 'command reject(
                                        Status::Unimplemented,
                                        format_args!(
                                            "it clears stencil, and {} has none",
                                            crate::gpu::formats::Name(format)
                                        ),
                                    );
                                }
                                if !backend.clear_depth(
                                    depth_target,
                                    (depth_write != 0).then_some(depth),
                                    (stencil_bits != 0).then_some(stencil & 0xff),
                                ) {
                                    break 'command reject(
                                        Status::InternalError,
                                        format_args!(
                                            "the device clear of {} failed",
                                            crate::gpu::formats::Name(format)
                                        ),
                                    );
                                }
                            }
                        }
                        RecordedCommand::CopyBufferToTexture { buffer, texture } => {
                            // Each rejection names its check, so a run shows
                            // which one a program's copy fails. Source: the public contract.
                            let reject = |reason: std::fmt::Arguments<'_>| {
                                crate::unsupported::refuse(
                                    Status::BadArgument,
                                    format_args!("copy buffer to texture: {reason}"),
                                )
                            };
                            let Some(Object::Texture { description, .. }) =
                                instance.objects.get(texture)
                            else {
                                break 'command reject(format_args!(
                                    "{texture:#x} is not a recorded texture"
                                ));
                            };
                            let shape = format_args!(
                                "target {} format {} {}x{}x{} levels {} stride {} pool {}",
                                description.target,
                                description.format,
                                description.width,
                                description.height,
                                description.depth,
                                description.levels,
                                description.stride,
                                if description.pool == 0 { "none" } else { "set" },
                            )
                            .to_string();
                            #[cfg(feature = "vulkan")]
                            if let Some(backend) = gpu.as_mut() {
                                let size = if backend.has_image_contract() {
                                    let Some(r) = backend.resolved_image(&description) else {
                                        break 'command reject(format_args!(
                                            "no image contract rule ({shape})"
                                        ));
                                    };
                                    if let Err(why) = crate::gpu::formats::supports(
                                        backend.context(),
                                        r.descriptor.format,
                                        crate::gpu::formats::FormatUse::Sampled,
                                    ) {
                                        break 'command reject(format_args!("{why} ({shape})"));
                                    }
                                    r.packing.linear_size()
                                } else {
                                    let Some(size) = copy_size(&description) else {
                                        break 'command reject(format_args!(
                                            "base level above the image limit ({shape})"
                                        ));
                                    };
                                    size
                                };
                                if !backend.ensure_texture(texture, &description, false) {
                                    break 'command reject(format_args!(
                                        "no device image for the texture ({shape})"
                                    ));
                                }
                                let Ok(source) = instance.objects.resolve_gpu_address(buffer)
                                else {
                                    break 'command reject(format_args!(
                                        "source {buffer:#x} is in no pool ({shape})"
                                    ));
                                };
                                if size as u64 > source.remaining {
                                    break 'command reject(format_args!(
                                        "copy needs {size} bytes, source pool has {} left ({shape})",
                                        source.remaining
                                    ));
                                }
                                if !backend.copy_from_arena(
                                    texture,
                                    source.pool,
                                    source.offset,
                                    size,
                                ) {
                                    break 'command crate::unsupported::refuse(Status::InternalError, format_args!("the device could not upload texture {texture} from its memory pool"));
                                }
                                continue 'commands;
                            }
                            let Some(size) = copy_size(&description) else {
                                break 'command reject(format_args!(
                                    "base level above the image limit ({shape})"
                                ));
                            };
                            let Some(data) = read_pool_bytes(instance, buffer, size) else {
                                break 'command reject(format_args!(
                                    "source {buffer:#x} cannot supply {size} bytes ({shape})"
                                ));
                            };
                            copy_cpu(instance, buffer, texture, data);
                        }
                        RecordedCommand::CopyTextureToTexture {
                            source,
                            destination,
                            arguments,
                        } => {
                            #[cfg(not(feature = "vulkan"))]
                            let _ = arguments;
                            #[cfg(feature = "vulkan")]
                            if let Some(backend) = gpu.as_mut() {
                                if let Some(decoder) = &backend.copy_decoder {
                                    let Some(operation) = decoder(arguments) else {
                                        break 'command crate::unsupported::refuse(
                                            Status::BadArgument,
                                            "copy texture to texture: the installed copy \
                                             decoder does not describe these arguments",
                                        );
                                    };
                                    use crate::gpu::image_layout::CopyOperation;
                                    let (src, dst) = match operation {
                                        CopyOperation::Copy {
                                            source,
                                            destination,
                                            ..
                                        }
                                        | CopyOperation::Blit {
                                            source,
                                            destination,
                                            ..
                                        } => (source, destination),
                                    };
                                    for texture in [src, dst] {
                                        let Some(Object::Texture { description, .. }) =
                                            instance.objects.get(texture)
                                        else {
                                            break 'command crate::unsupported::refuse(
                                                Status::BadArgument,
                                                format_args!(
                                                    "copy texture to texture: {texture:#x} is not \
                                                     a texture"
                                                ),
                                            );
                                        };
                                        if let Err(reason) =
                                            backend.ensure_copy_texture(texture, &description)
                                        {
                                            break 'command crate::unsupported::refuse(
                                                Status::BadArgument,
                                                format_args!(
                                                    "copy texture to texture: texture \
                                                     {texture:#x} has no image: {reason}"
                                                ),
                                            );
                                        }
                                    }
                                    match operation {
                                        CopyOperation::Copy { from, to, .. } => {
                                            match backend.copy_region(dst, src, from, to) {
                                                Ok(()) => {}
                                                Err(crate::gpu::CopyFailure::Refused(reason)) => {
                                                    break 'command crate::unsupported::refuse(
                                                        Status::BadArgument,
                                                        format_args!(
                                                            "copy texture {src:#x} to \
                                                             {dst:#x}: {reason}"
                                                        ),
                                                    );
                                                }
                                                Err(crate::gpu::CopyFailure::Device) => {
                                                    break 'command crate::unsupported::refuse(Status::InternalError, format_args!("copy texture {src:#x} to {dst:#x}: the device could not record the copy"));
                                                }
                                            }
                                        }
                                        CopyOperation::Blit {
                                            from, to, filter, ..
                                        } => {
                                            if !backend.blit_region(dst, src, from, to, filter) {
                                                break 'command crate::unsupported::refuse(
                                                    Status::Unimplemented,
                                                    format_args!(
                                                        "blit texture {src:#x} {from:?} to \
                                                         {dst:#x} {to:?} with filter {}: \
                                                         swxbx cannot blit these images",
                                                        filter.as_raw()
                                                    ),
                                                );
                                            }
                                        }
                                    }
                                    continue 'commands;
                                }
                                for texture in [source, destination] {
                                    let Some(Object::Texture { description, .. }) =
                                        instance.objects.get(texture)
                                    else {
                                        break 'command crate::unsupported::refuse(
                                            Status::BadArgument,
                                            format_args!("copy: object {texture} is not a texture"),
                                        );
                                    };
                                    if let Err(reason) =
                                        backend.ensure_copy_texture(texture, &description)
                                    {
                                        break 'command crate::unsupported::refuse(
                                            Status::BadArgument,
                                            format_args!(
                                                "copy: texture {texture} has no device image: \
                                                 {reason}"
                                            ),
                                        );
                                    }
                                }
                                if !backend.copy(destination, source) {
                                    break 'command crate::unsupported::refuse(Status::InternalError, format_args!("the device could not copy texture {source} to {destination}"));
                                }
                                continue 'commands;
                            }
                            let Some(Object::Texture { image, .. }) = instance.objects.get(source)
                            else {
                                break 'command crate::unsupported::refuse(
                                    Status::BadArgument,
                                    format_args!("copy: source object {source} is not a texture"),
                                );
                            };
                            let Some(data) = image
                                .lock()
                                .unwrap_or_else(|p| p.into_inner())
                                .as_ref()
                                .map(|i| i.pixels.clone())
                            else {
                                break 'command crate::unsupported::refuse(
                                    Status::BadArgument,
                                    format_args!(
                                        "copy: texture {source} has no pixels kept on the CPU"
                                    ),
                                );
                            };
                            copy_cpu(instance, source, destination, data);
                        }
                        RecordedCommand::DrawArrays {
                            primitive,
                            first,
                            count,
                        } => {
                            #[cfg(feature = "vulkan")]
                            if let Some(backend) = gpu.as_mut() {
                                if let Err(status) = draw_state.execute(
                                    instance,
                                    backend,
                                    super::drawing::Request {
                                        targets: &targets,
                                        depth: depth_target,
                                        views: target_views,
                                        primitive,
                                        geometry: crate::gpu::operations::Geometry::default(),
                                        vertices: super::drawing::Vertices::Arrays { first },
                                        count,
                                    },
                                ) {
                                    break 'command status;
                                }
                                continue 'commands;
                            }
                            let _ = (primitive, first, count, target_views);
                            break 'command crate::unsupported::refuse(Status::Unimplemented, "DrawArrays needs the Vulkan device, which this instance does not have");
                        }
                        RecordedCommand::DrawElementsBaseVertex {
                            primitive,
                            index_type,
                            count,
                            indices,
                            base_vertex,
                        } => {
                            #[cfg(feature = "vulkan")]
                            if let Some(backend) = gpu.as_mut() {
                                if let Err(status) = draw_state.execute(
                                    instance,
                                    backend,
                                    super::drawing::Request {
                                        targets: &targets,
                                        depth: depth_target,
                                        views: target_views,
                                        primitive,
                                        geometry: crate::gpu::operations::Geometry::default(),
                                        count,
                                        vertices: super::drawing::Vertices::Elements {
                                            index_type,
                                            indices,
                                            base_vertex,
                                        },
                                    },
                                ) {
                                    break 'command status;
                                }
                                continue 'commands;
                            }
                            let _ = (
                                primitive,
                                index_type,
                                count,
                                indices,
                                base_vertex,
                                target_views,
                            );
                            break 'command crate::unsupported::refuse(Status::Unimplemented, "DrawElementsBaseVertex needs the Vulkan device, which this instance does not have");
                        }
                        RecordedCommand::Operation { .. }
                        | RecordedCommand::ReportCounter { .. }
                        | RecordedCommand::ResetCounter(_)
                        | RecordedCommand::DrawArraysInstanced { .. } => {
                            break 'command crate::unsupported::refuse(
                                Status::Unimplemented,
                                format_args!(
                                    "{} needs the Vulkan device, which this instance does not have",
                                    crate::unsupported::command_name(&command)
                                ),
                            )
                        }
                        command @ (RecordedCommand::SetViewport(_)
                        | RecordedCommand::SetScissor(_)
                        | RecordedCommand::SetDepthRange(_)
                        | RecordedCommand::BindProgram(_)
                        | RecordedCommand::BindState { .. }
                        | RecordedCommand::Raw { .. }
                        | RecordedCommand::State(_)) => {
                            #[cfg(feature = "vulkan")]
                            draw_state.record(command);
                            #[cfg(not(feature = "vulkan"))]
                            let _ = command;
                            continue 'commands;
                        }
                    }
                    Status::Ok
                };
                // Stop ends the batch here, as the public contract requires. Skip
                // records the command and continues. Source: the public contract.
                if let Err(stopped) = instance.unsupported_commands.settle(&command, status) {
                    instance.batch_stopped(stopped.at);
                    return stopped.status;
                }
            }
            instance.objects.recycle_recording(handle, commands);
        }
        Status::Ok
    })();
    #[cfg(feature = "vulkan")]
    if let Some(backend) = gpu.as_mut() {
        if backend.finish_draws().is_none() {
            return Status::InternalError;
        }
    }
    #[cfg(feature = "vulkan")]
    if let Some(memory) = gpu
        .as_mut()
        .and_then(|backend| backend.global_memory.as_mut())
    {
        #[cfg(feature = "draw-metrics")]
        let _span = crate::draw_metrics::PipelineSpan::new(14);
        if !memory.download(|address, bytes| instance.write_memory_untracked(address, bytes)) {
            return Status::BadArgument;
        }
    }
    if submitted != Status::Ok {
        return submitted;
    }
    Status::Ok
}

/// Return the handler for a supported name in this command family.
pub fn handler(name: &str) -> Option<Handler> {
    Some(match name {
        "nvnQueueBuilderSetDefaults" => |instance, _, registers| {
            instance.objects.put(
                registers.x[0],
                Object::QueueBuilder {
                    device: 0,
                    control_memory_size: 0,
                    command_flush_threshold: 0,
                },
            );
            Status::Ok
        },
        "nvnQueueBuilderSetDevice" => |instance, _, registers| {
            let value = registers.x[1];
            builder_update(instance, registers, |object| {
                if let Object::QueueBuilder { device, .. } = object {
                    *device = value;
                }
            })
        },
        "nvnQueueBuilderSetControlMemorySize" => |instance, _, registers| {
            let value = registers.x[1];
            builder_update(instance, registers, |object| {
                if let Object::QueueBuilder {
                    control_memory_size,
                    ..
                } = object
                {
                    *control_memory_size = value;
                }
            })
        },
        "nvnQueueBuilderSetCommandFlushThreshold" => |instance, _, registers| {
            let value = registers.x[1];
            builder_update(instance, registers, |object| {
                if let Object::QueueBuilder {
                    command_flush_threshold,
                    ..
                } = object
                {
                    *command_flush_threshold = value;
                }
            })
        },
        "nvnQueueInitialize" => |instance, _, registers| {
            let device = match instance.objects.get(registers.x[1]) {
                Some(Object::QueueBuilder { device, .. }) => device,
                _ => 0,
            };
            instance
                .objects
                .put(registers.x[0], Object::Queue { device });
            succeed(registers)
        },
        "nvnQueueSubmitCommands" => |instance, _, r| {
            let mut buffers = Vec::new();
            for i in 0..r.x[1].min(1024) {
                let Some(handle) = read_u64(instance, r.x[2] + i * 8) else {
                    return Status::BadArgument;
                };
                if let Some(commands) = instance.objects.recording(handle) {
                    buffers.push((handle, commands));
                }
            }
            let status = instance.queue.submit(buffers);
            if status == Status::Ok {
                r.x[0] = 0;
            }
            status
        },
        "nvnQueueFlush" => |instance, _, registers| {
            for program in instance.pending_programs() {
                crate::api::resources::retry_pending(instance, program, "flush");
            }
            accept(
                instance,
                crate::functions::lookup("nvnQueueFlush").unwrap(),
                registers,
            )
        },
        "nvnQueueFinish" => |instance, _, registers| {
            let status = instance.poll_presentations(true);
            if status != Status::Ok {
                return status;
            }
            #[cfg(feature = "vulkan")]
            if instance
                .gpu
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_ref()
                .is_some_and(|backend| !backend.finish())
            {
                return Status::InternalError;
            }
            for program in instance.pending_programs() {
                crate::api::resources::retry_pending(instance, program, "finish");
            }
            accept(
                instance,
                crate::functions::lookup("nvnQueueFinish").unwrap(),
                registers,
            )
        },
        "nvnQueuePresentTexture" => |instance, _, r| {
            let started = std::time::Instant::now();
            let window = r.x[1];
            let Some(Object::Window {
                textures,
                present_interval,
                ..
            }) = instance.objects.get(window)
            else {
                return Status::BadArgument;
            };
            let Some(texture) = textures.get(r.x[2] as usize).copied() else {
                return Status::BadArgument;
            };
            let Some(Object::Texture { description, image }) = instance.objects.get(texture) else {
                return Status::BadArgument;
            };
            #[cfg(not(feature = "vulkan"))]
            let _ = &description;
            // The commands before this present have run (the call waited
            // for them). If one of their batches stopped, the frame lacks
            // the rest of that batch: nothing is presented, and the window
            // keeps the last frame it was given.
            if let Some(stopped) = instance.take_stopped_batch() {
                return crate::unsupported::refuse(
                    Status::Unimplemented,
                    format!(
                        "window {window:#x}: not presented, because a batch submitted since the last present stopped at {stopped}"
                    ),
                );
            }
            let Some(host) = instance.host() else {
                r.x[0] = 0;
                return Status::Ok;
            };
            let mut pacing = instance
                .presentation
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if pacing.config.mode == crate::PresentationMode::Mailbox {
                pacing.drop_pending(window);
            }
            if pacing.pending.len() == pacing.config.frames_in_flight as usize {
                if let Err(status) = pacing.deliver_one(instance, host, true) {
                    pacing.failed();
                    return status;
                }
            }
            let slot = pacing.begin();
            let prepare = (|| {
                #[cfg(feature = "vulkan")]
                if let Some(backend) = instance
                    .gpu
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .as_mut()
                {
                    if !backend.ensure_texture(texture, &description, false) {
                        return Err(Status::BadArgument);
                    }
                    if !host.vulkan.is_null() {
                        let scaling = host.scaling(window).map_err(|why| {
                            crate::unsupported::refuse(
                                Status::BadArgument,
                                format!("window {window:#x}: {why}"),
                            )
                        })?;
                        let handed = backend.present_window(
                            host,
                            window,
                            texture,
                            present_interval,
                            scaling,
                            pacing.config,
                        )?;
                        if handed {
                            pacing.delivered(started);
                        } else {
                            pacing.skipped();
                        }
                        return Ok(());
                    }
                    if host.present.is_some() {
                        backend
                            .enqueue_callback(slot, texture, description.width, description.height)
                            .ok_or(Status::InternalError)?;
                        pacing.push(crate::presentation::Frame {
                            window,
                            interval: present_interval,
                            started,
                            slot,
                            pixels: None,
                        });
                    }
                    return Ok(());
                }
                if !host.vulkan.is_null() {
                    return Err(Status::InternalError);
                }
                if host.present.is_some() {
                    let image = image
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .as_ref()
                        .cloned()
                        .ok_or(Status::BadArgument)?;
                    pacing.push(crate::presentation::Frame {
                        window,
                        interval: present_interval,
                        started,
                        slot,
                        pixels: Some((image.width, image.height, image.pixels)),
                    });
                }
                Ok(())
            })();
            if let Err(status) = prepare {
                pacing.failed();
                return status;
            }
            r.x[0] = 0;
            Status::Ok
        },
        // Observed to return 1. Waiting on a sync that nothing signals would
        // block the program forever, so the wait is reported as satisfied.
        "nvnQueueWaitSync" => |_instance, _, registers| succeed(registers),
        "nvnSyncInitialize" => |instance, _, registers| {
            instance.objects.put(
                registers.x[0],
                Object::Sync {
                    device: registers.x[1],
                },
            );
            succeed(registers)
        },
        "nvnSyncFinalize" => |instance, _, registers| {
            instance.objects.remove(registers.x[0]);
            Status::Ok
        },
        // Observed to return 0, taken to mean the wait completed.
        "nvnSyncWait" => accept,
        "nvnEventBuilderSetStorage" => |instance, _, registers| {
            instance.objects.put(
                registers.x[0],
                Object::EventBuilder {
                    pool: registers.x[1],
                    offset: registers.x[2],
                },
            );
            Status::Ok
        },
        // The event lives in pool memory the program chose
        // (EventBuilderSetStorage), and the program was seen reading that
        // memory directly rather than calling EventGetValue (note 0011).
        "nvnEventInitialize" => |instance, _, registers| {
            let storage = match instance.objects.get(registers.x[1]) {
                Some(Object::EventBuilder { pool, offset }) => match instance.objects.get(pool) {
                    Some(Object::MemoryPool { storage, .. }) => storage + offset,
                    _ => 0,
                },
                _ => 0,
            };
            instance
                .objects
                .put(registers.x[0], Object::Event { value: 0, storage });
            succeed(registers)
        },
        "nvnEventGetValue" => |instance, _, registers| {
            registers.x[0] = match instance.objects.get(registers.x[0]) {
                Some(Object::Event { value, .. }) => u64::from(value),
                _ => 0,
            };
            Status::Ok
        },
        // Signatures 0003 leaves the arguments open. The event is marked
        // signalled, which is the least the name promises.
        "nvnEventSignal" => |instance, _, registers| {
            signal_event(instance, registers.x[0], 1);
            Status::Ok
        },
        "nvnWindowBuilderSetDefaults" => |instance, _, registers| {
            instance.objects.put(
                registers.x[0],
                Object::WindowBuilder {
                    device: 0,
                    native_window: 0,
                    textures: Vec::new(),
                    present_interval: crate::presentation::PresentInterval::ONE,
                },
            );
            Status::Ok
        },
        "nvnWindowBuilderSetDevice" => |instance, _, registers| {
            let value = registers.x[1];
            builder_update(instance, registers, |object| {
                if let Object::WindowBuilder { device, .. } = object {
                    *device = value;
                }
            })
        },
        "nvnWindowBuilderSetNativeWindow" => |instance, _, registers| {
            let value = registers.x[1];
            builder_update(instance, registers, |object| {
                if let Object::WindowBuilder { native_window, .. } = object {
                    *native_window = value;
                }
            })
        },
        "nvnWindowBuilderSetTextures" => |instance, _, registers| {
            // An array of `count` texture object addresses (signatures 0004).
            let count = registers.x[1].min(16);
            let mut list = Vec::with_capacity(count as usize);
            for index in 0..count {
                match read_u64(instance, registers.x[2] + index * 8) {
                    Some(address) => list.push(address),
                    None => return Status::BadArgument,
                }
            }
            builder_update(instance, registers, |object| {
                if let Object::WindowBuilder { textures, .. } = object {
                    *textures = list;
                }
            })
        },
        // Read once where it arrives; whether this host and mode can show it
        // is asked when the window initializes. Source: the public contract.
        "nvnWindowBuilderSetPresentInterval" => |instance, _, registers| {
            let builder = registers.x[0];
            let interval = match crate::presentation::PresentInterval::from_guest(
                registers.x[1] as u32 as i32,
            ) {
                Ok(interval) => interval,
                Err(why) => {
                    return crate::unsupported::refuse(
                        Status::BadArgument,
                        format!("window builder {builder:#x}: {why}"),
                    )
                }
            };
            builder_update(instance, registers, |object| {
                if let Object::WindowBuilder {
                    present_interval, ..
                } = object
                {
                    *present_interval = interval;
                }
            })
        },
        "nvnWindowBuilderGetPresentInterval" => |instance, _, registers| {
            registers.x[0] = match instance.objects.get(registers.x[0]) {
                Some(Object::WindowBuilder {
                    present_interval, ..
                }) => u64::from(present_interval.guest_value()),
                _ => 1,
            };
            Status::Ok
        },
        "nvnWindowInitialize" => |instance, _, registers| {
            let Some(Object::WindowBuilder {
                device,
                textures,
                native_window,
                present_interval,
            }) = instance.objects.get(registers.x[1])
            else {
                return Status::BadArgument;
            };
            // Held so the policy cannot change between the checks below and
            // the opened window. Source: the public contract. and 0064.
            let pacing = instance
                .presentation
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if let Err(why) = present_interval.shown_by(instance.host(), pacing.config.mode) {
                let window = registers.x[0];
                registers.x[0] = 0;
                return crate::unsupported::refuse(
                    Status::BadArgument,
                    format!("window {window:#x}: {why}"),
                );
            }
            #[cfg(feature = "vulkan")]
            if let Some(host) = instance.host() {
                if !host.vulkan.is_null() {
                    let mut gpu = instance.gpu.lock().unwrap_or_else(|p| p.into_inner());
                    let Some(backend) = gpu.as_mut() else {
                        registers.x[0] = 0;
                        return Status::InternalError;
                    };
                    let window = registers.x[0];
                    if backend.open_window(host, window, native_window).is_none() {
                        registers.x[0] = 0;
                        return Status::InternalError;
                    }
                    let mode = pacing.config.mode;
                    if mode == crate::PresentationMode::Mailbox {
                        if let Err(why) = backend.window_takes(window, mode) {
                            backend.close_window(window);
                            registers.x[0] = 0;
                            return crate::presentation::refuse_unoffered(mode, window, why);
                        }
                    }
                }
            }
            #[cfg(not(feature = "vulkan"))]
            if instance.host().is_some_and(|host| !host.vulkan.is_null()) {
                registers.x[0] = 0;
                return Status::Unimplemented;
            }
            let _ = native_window;
            instance.objects.put(
                registers.x[0],
                Object::Window {
                    device,
                    textures,
                    present_interval,
                    next_texture: 0,
                },
            );
            succeed(registers)
        },
        "nvnWindowFinalize" => |instance, _, registers| {
            let status = instance.poll_presentations(true);
            if status != Status::Ok {
                return status;
            }
            #[cfg(feature = "vulkan")]
            if let Some(backend) = instance
                .gpu
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_mut()
            {
                backend.close_window(registers.x[0]);
            }
            instance.objects.remove(registers.x[0]);
            Status::Ok
        },
        // The interval is honoured at every present or refused here, by name,
        // when this host and presentation mode cannot show it. Source: the public contract.
        "nvnWindowSetPresentInterval" => |instance, _, registers| {
            let window = registers.x[0];
            let interval = match crate::presentation::PresentInterval::from_guest(
                registers.x[1] as u32 as i32,
            ) {
                Ok(interval) => interval,
                Err(why) => {
                    return crate::unsupported::refuse(
                        Status::BadArgument,
                        format!("window {window:#x}: {why}"),
                    )
                }
            };
            // Held across the update so the mode cannot change between the
            // check and the stored interval.
            let pacing = instance
                .presentation
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if let Err(why) = interval.shown_by(instance.host(), pacing.config.mode) {
                return crate::unsupported::refuse(
                    Status::BadArgument,
                    format!("window {window:#x}: {why}"),
                );
            }
            builder_update(instance, registers, |object| {
                if let Object::Window {
                    present_interval, ..
                } = object
                {
                    *present_interval = interval;
                }
            })
        },
        "nvnWindowGetPresentInterval" => |instance, _, registers| {
            registers.x[0] = match instance.objects.get(registers.x[0]) {
                Some(Object::Window {
                    present_interval, ..
                }) => u64::from(present_interval.guest_value()),
                _ => 1,
            };
            Status::Ok
        },
        // Hands out the window's textures in turn and writes the index
        // through the third argument (signatures 0004); returns 0 as
        // observed.
        "nvnWindowAcquireTexture" => |instance, _, registers| {
            let mut index = 0;
            let found = instance.objects.update(registers.x[0], |object| {
                if let Object::Window {
                    textures,
                    next_texture,
                    ..
                } = object
                {
                    index = *next_texture;
                    let count = textures.len().max(1) as u32;
                    *next_texture = (*next_texture + 1) % count;
                }
            });
            if !found || !write_u32(instance, registers.x[2], index) {
                return Status::BadArgument;
            }
            registers.x[0] = 0;
            Status::Ok
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Host;
    use crate::Instance;
    use std::ffi::c_void;

    unsafe extern "C" fn read(_user: *mut c_void, address: u64, out: *mut u8, size: u64) -> i32 {
        let Some(offset) = address.checked_sub(0x1000) else {
            return 1;
        };
        if offset.checked_add(size).is_none_or(|end| end > 4) {
            return 1;
        }
        let bytes = [10_u8, 20, 30, 40];
        // SAFETY: the callback contract supplies a valid output slice and the
        // checked range lies in this project's fixture.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr().add(offset as usize), out, size as usize)
        };
        0
    }

    #[test]
    fn buffer_copy_resolves_gpu_addresses_before_reading_cpu_storage() {
        // SAFETY: the stateless callback accepts only its synthetic memory range.
        let instance = unsafe {
            Instance::with_host(Host {
                user: std::ptr::null_mut(),
                read_memory: Some(read),
                write_memory: None,
                present: None,
                render_scale: 1.0,
                wait_vblank: None,
                vulkan: std::ptr::null(),
            })
        };
        instance.objects.put(
            1,
            Object::MemoryPool {
                device: 0,
                flags: 0,
                storage: 0x1000,
                size: 4,
                gpu_address: Some(0x10000),
                observed_gpu_address: None,
            },
        );
        assert_eq!(read_pool_bytes(&instance, 0x10001, 2), Some(vec![20, 30]));
        assert_eq!(read_pool_bytes(&instance, 0x10003, 2), None);
        assert_eq!(read_pool_bytes(&instance, 0x1001, 2), None);
    }
}

#[cfg(all(test, feature = "vulkan"))]
#[path = "queue_transfer_tests.rs"]
mod transfer_tests;
