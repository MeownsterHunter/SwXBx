//! Deferred command hypotheses. Evidence and limits:
use super::{commands::record, objects::RecordedCommand, Handler};

const NAMES: &[&str] = &[
    "DispatchComputeIndirect",
    "DrawArraysIndirect",
    "DrawElementsIndirect",
    "DrawElementsInstanced",
    "SetRenderEnable",
    "SetRenderEnableConditional",
    "CopyBufferToBuffer",
    "ClearTexture",
];

pub(super) fn handler(name: &str) -> Option<Handler> {
    let suffix = name.get(3..)?.strip_prefix("CommandBuffer")?;
    match suffix {
        "ReportCounter" => return Some(report_counter),
        "ResetCounter" => return Some(reset_counter),
        _ => {}
    }
    NAMES.iter().find(|&&n| n == suffix)?;
    Some(|instance, function, r| {
        let Some(kind) = crate::functions::name(function)
            .and_then(|n| n.get(3..))
            .and_then(|n| n.strip_prefix("CommandBuffer"))
            .and_then(|n| NAMES.iter().copied().find(|&known| known == n))
        else {
            return crate::Status::BadArgument;
        };
        record(
            instance,
            function.0,
            r,
            RecordedCommand::Operation {
                kind,
                arguments: r.x[1..].try_into().expect("seven arguments"),
            },
        )
    })
}

/// Record a counter report: x1 is the client counter token and x2 is the report's
/// GPU address. The token is parsed here, once, into the
/// counter it selects. A token the contract does not name stays an opaque
/// operation, which execution reports as unimplemented.
fn report_counter(
    instance: &crate::Instance,
    function: crate::functions::FunctionId,
    r: &mut crate::Registers,
) -> crate::Status {
    let command = match instance.counter(r.x[1]) {
        Some(counter) => RecordedCommand::ReportCounter {
            counter,
            address: r.x[2],
        },
        None => opaque("ReportCounter", r),
    };
    record(instance, function.0, r, command)
}

/// Record a counter reset: x1 is the guest counter token.
fn reset_counter(
    instance: &crate::Instance,
    function: crate::functions::FunctionId,
    r: &mut crate::Registers,
) -> crate::Status {
    let command = match instance.counter(r.x[1]) {
        Some(counter) => RecordedCommand::ResetCounter(counter),
        None => opaque("ResetCounter", r),
    };
    record(instance, function.0, r, command)
}

fn opaque(kind: &'static str, r: &crate::Registers) -> RecordedCommand {
    RecordedCommand::Operation {
        kind,
        arguments: r.x[1..].try_into().expect("seven arguments"),
    }
}

#[cfg(feature = "vulkan")]
pub(super) struct Targets<'a> {
    pub colors: &'a [u64],
    pub depth: u64,
    pub views: [u64; 2],
}

#[cfg(feature = "vulkan")]
mod execution {
    use super::*;
    use crate::gpu::{graphics::execution_model, operations::Geometry, Backend};
    use crate::{
        api::objects::{Object, ShaderTranslation, StateCommand},
        counters::{Counter, REPORT_SIZE},
        instance::InstanceState as Instance,
        Status,
    };
    use std::collections::HashMap;

    pub(crate) struct State {
        program: Option<u64>,
        buffers: HashMap<(u64, u64), (u64, u64)>,
        pub enabled: bool,
    }
    impl Default for State {
        fn default() -> Self {
            Self {
                program: None,
                buffers: HashMap::new(),
                enabled: true,
            }
        }
    }

    fn region(
        instance: &Instance,
        backend: &Backend,
        address: u64,
        size: u64,
    ) -> Result<(ash::vk::Buffer, u64), Status> {
        let r = instance
            .objects
            .resolve_gpu_address(address)
            .map_err(|_| Status::BadArgument)?;
        if size > r.remaining {
            return Err(Status::BadArgument);
        }
        backend
            .global_memory
            .as_ref()
            .ok_or(Status::BadArgument)?
            .buffer_region(
                r.pool,
                r.offset,
                usize::try_from(size).map_err(|_| Status::BadArgument)?,
            )
            .ok_or(Status::BadArgument)
    }

    fn words(
        instance: &Instance,
        backend: &mut Backend,
        address: u64,
        count: usize,
    ) -> Result<Vec<u32>, Status> {
        let r = instance
            .objects
            .resolve_gpu_address(address)
            .map_err(|_| Status::BadArgument)?;
        let mut bytes = vec![0; count * 4];
        if bytes.len() as u64 > r.remaining || !address.is_multiple_of(4) {
            return Err(Status::BadArgument);
        }
        backend
            .global_memory
            .as_mut()
            .ok_or(Status::BadArgument)?
            .read_pool(r.pool, r.offset, &mut bytes)
            .ok_or(Status::InternalError)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect())
    }

    /// Write a counter report once every earlier command in queue order has
    /// been submitted.
    fn report_counter(
        instance: &Instance,
        backend: &mut Backend,
        counter: Counter,
        address: u64,
    ) -> Result<(), Status> {
        backend.finish_draws().ok_or(Status::InternalError)?;
        if !address.is_multiple_of(8) {
            return Err(Status::BadArgument);
        }
        let destination = region(instance, backend, address, REPORT_SIZE)?;
        let memory = backend.global_memory.as_ref().ok_or(Status::BadArgument)?;
        memory.mark_written();
        backend
            .execution
            .report(memory, destination, counter)
            .ok_or(Status::InternalError)
    }

    /// Start a new accumulation interval.
    fn reset_counter(backend: &mut Backend, counter: Counter) -> Result<(), Status> {
        backend.finish_draws().ok_or(Status::InternalError)?;
        match counter {
            Counter::SamplesPassed => backend
                .execution
                .reset_occlusion()
                .ok_or(Status::InternalError),
            Counter::Timestamp => Err(Status::Unimplemented),
        }
    }

    impl State {
        pub fn execute(
            &mut self,
            command: &RecordedCommand,
            instance: &Instance,
            backend: &mut Backend,
            draw: &mut super::super::drawing::State,
            target: super::Targets<'_>,
        ) -> Option<Result<(), Status>> {
            match command {
                RecordedCommand::BindProgram(program) => {
                    self.program = Some(*program);
                    return None;
                }
                RecordedCommand::State(StateCommand::BindUniformBuffer {
                    stage,
                    index,
                    address,
                    size,
                }) => {
                    self.buffers.insert((*stage, *index), (*address, *size));
                    if backend
                        .command_contract
                        .as_ref()
                        .is_some_and(|c| c.compute_stage == *stage)
                    {
                        return Some(Ok(()));
                    }
                    return None;
                }
                _ => {}
            }
            let result = match command {
                RecordedCommand::Operation { kind, arguments: a } => {
                    self.operation(kind, *a, instance, backend, draw, target)
                }
                RecordedCommand::State(StateCommand::ClearBuffer {
                    address,
                    size,
                    value,
                }) => self.operation(
                    "ClearBuffer",
                    [*address, *size, *value, 0, 0, 0, 0],
                    instance,
                    backend,
                    draw,
                    target,
                ),
                RecordedCommand::State(StateCommand::DispatchCompute(groups)) => self.operation(
                    "DispatchCompute",
                    [groups[0], groups[1], groups[2], 0, 0, 0, 0],
                    instance,
                    backend,
                    draw,
                    target,
                ),
                RecordedCommand::ReportCounter { counter, address } => {
                    draw.invalidate();
                    report_counter(instance, backend, *counter, *address)
                }
                RecordedCommand::ResetCounter(counter) => {
                    draw.invalidate();
                    reset_counter(backend, *counter)
                }
                RecordedCommand::DrawArraysInstanced {
                    primitive,
                    first,
                    count,
                    base_instance,
                    instances,
                } => {
                    if !self.enabled {
                        return Some(Ok(()));
                    }
                    self.operation(
                        "DrawArraysInstanced",
                        [
                            u64::from(*primitive),
                            u64::from(*first),
                            u64::from(*count),
                            u64::from(*base_instance),
                            u64::from(*instances),
                            0,
                            0,
                        ],
                        instance,
                        backend,
                        draw,
                        target,
                    )
                }
                _ => return None,
            };
            Some(result)
        }

        fn operation(
            &mut self,
            kind: &str,
            a: [u64; 7],
            instance: &Instance,
            backend: &mut Backend,
            draw: &mut super::super::drawing::State,
            target: super::Targets<'_>,
        ) -> Result<(), Status> {
            let super::Targets {
                colors: targets,
                depth,
                views,
            } = target;
            draw.invalidate();
            backend.finish_draws().ok_or(Status::InternalError)?;
            let c = backend
                .command_contract
                .clone()
                .ok_or(Status::Unimplemented)?;
            let integer = |n| u32::try_from(n).map_err(|_| Status::BadArgument);
            match kind {
                "ClearBuffer" => {
                    let r = region(instance, backend, a[0], a[1])?;
                    if a[1] == 0 || !a[1].is_multiple_of(4) || !r.1.is_multiple_of(4) {
                        return Err(Status::BadArgument);
                    }
                    backend
                        .global_memory
                        .as_ref()
                        .ok_or(Status::BadArgument)?
                        .mark_written();
                    backend
                        .execution
                        .fill(
                            backend.global_memory.as_ref().ok_or(Status::BadArgument)?,
                            r,
                            a[1],
                            integer(a[2])?,
                        )
                        .ok_or(Status::InternalError)
                }
                "CopyBufferToBuffer" => {
                    if a[3] != 0 {
                        return Err(Status::Unimplemented);
                    }
                    let source = region(instance, backend, a[0], a[2])?;
                    let destination = region(instance, backend, a[1], a[2])?;
                    if source.0 == destination.0
                        && source.1 < destination.1 + a[2]
                        && destination.1 < source.1 + a[2]
                    {
                        return Err(Status::BadArgument);
                    }
                    backend
                        .global_memory
                        .as_ref()
                        .ok_or(Status::BadArgument)?
                        .mark_written();
                    backend
                        .execution
                        .copy(
                            backend.global_memory.as_ref().ok_or(Status::BadArgument)?,
                            source,
                            destination,
                            a[2],
                        )
                        .ok_or(Status::InternalError)
                }
                "DispatchCompute" | "DispatchComputeIndirect" => {
                    let indirect = if kind == "DispatchComputeIndirect" {
                        Some(region(instance, backend, a[0], 12)?)
                    } else {
                        None
                    };
                    let groups = if indirect.is_some() {
                        let g = words(instance, backend, a[0], 3)?;
                        [g[0], g[1], g[2]]
                    } else {
                        [integer(a[0])?, integer(a[1])?, integer(a[2])?]
                    };
                    let limits = unsafe {
                        backend
                            .context()
                            .instance
                            .get_physical_device_properties(backend.context().physical_device)
                    }
                    .limits;
                    if groups
                        .iter()
                        .zip(limits.max_compute_work_group_count)
                        .any(|(&n, max)| n > max)
                    {
                        return Err(Status::BadArgument);
                    }
                    let Some(Object::Program {
                        shader_translations,
                        ..
                    }) = instance
                        .objects
                        .get(self.program.ok_or(Status::Unimplemented)?)
                    else {
                        return Err(Status::BadArgument);
                    };
                    let mut stages = Vec::new();
                    for t in shader_translations {
                        let t = match t {
                            ShaderTranslation::Deferred(code, context) => instance
                                .retry_cached_translation(&code, &context, true)
                                .ok_or(Status::Unimplemented)?,
                            t => t,
                        };
                        match t {
                            ShaderTranslation::Spirv(w) => stages.push((w, false, Vec::new())),
                            ShaderTranslation::Cached(r) => match r.wait() {
                                crate::startup_cache::TranslationStatus::Ready(s) => {
                                    backend
                                        .context()
                                        .shader_features
                                        .validate_translated_shader(&s)
                                        .map_err(|error| {
                                            eprintln!("compute pipeline: {error}");
                                            Status::Unimplemented
                                        })?;
                                    stages.push((
                                        s.words.clone(),
                                        s.requires_subgroup_size_32,
                                        s.surfaces.clone(),
                                    ))
                                }
                                _ => return Err(Status::Unimplemented),
                            },
                            _ => return Err(Status::Unimplemented),
                        }
                    }
                    let compute: Vec<_> = stages
                        .iter()
                        .filter(|s| execution_model(&s.0).ok() == Some(5))
                        .collect();
                    if compute.len() != 1 {
                        return Err(Status::Unimplemented);
                    }
                    let (shader, requires_subgroup_size_32, surfaces) = compute[0];
                    let reflected = crate::gpu::pipelines::compute_bindings(shader)
                        .map_err(|_| Status::Unimplemented)?;
                    let mut resources = Vec::new();
                    let mut images = Vec::new();
                    let mut stored = Vec::new();
                    for b in reflected {
                        if b.count != 1 {
                            return Err(Status::Unimplemented);
                        }
                        if b.descriptor_type == ash::vk::DescriptorType::STORAGE_IMAGE {
                            let skip = |reason: &str| {
                                eprintln!("compute dispatch: {reason}");
                                Status::Unimplemented
                            };
                            let surface = surfaces
                                .iter()
                                .find(|s| (s.set, s.binding) == (b.set, b.binding))
                                .ok_or_else(|| skip("storage image without a typed surface"))?;
                            let contract = backend.texture_contract.clone();
                            let texture = draw.bound_surface(
                                instance,
                                &contract,
                                crate::gpu::textures::SurfaceStage::Compute,
                                surface,
                                &skip,
                            )?;
                            let Some(Object::Texture { description: d, .. }) =
                                instance.objects.get(texture)
                            else {
                                return Err(Status::BadArgument);
                            };
                            let kind = backend
                                .resolved_image(&d)
                                .ok_or_else(|| skip("surface image has no image contract"))?
                                .descriptor
                                .shape
                                .kind;
                            use crate::surfaces::SurfaceDimension as D;
                            use crate::tiling::ImageKind as K;
                            if !matches!(
                                (surface.dimension, kind),
                                (D::D1, K::D1)
                                    | (D::D1Array, K::D1Array)
                                    | (D::D2, K::D2)
                                    | (D::D2Array, K::D2Array)
                                    | (D::D3, K::D3)
                            ) {
                                return Err(skip("surface dimension differs from its image"));
                            }
                            let view = backend
                                .storage_texture(texture, &d, surface.format)
                                .ok_or_else(|| skip("surface image has no storage view"))?;
                            stored.push(texture);
                            images.push((b.set, b.binding, view));
                            continue;
                        }
                        let m = c
                            .compute_buffers
                            .iter()
                            .find(|m| (m.set, m.binding) == (b.set, b.binding))
                            .ok_or(Status::Unimplemented)?;
                        let &(address, size) = self
                            .buffers
                            .get(&(c.compute_stage, m.index))
                            .ok_or(Status::Unimplemented)?;
                        let r = instance
                            .objects
                            .resolve_gpu_address(address)
                            .map_err(|_| Status::BadArgument)?;
                        if size > r.remaining {
                            return Err(Status::BadArgument);
                        }
                        let memory = backend.global_memory.as_ref().ok_or(Status::BadArgument)?;
                        let info = match b.descriptor_type {
                            ash::vk::DescriptorType::UNIFORM_BUFFER => {
                                memory.uniform_buffer_info(r.pool, r.offset, size)
                            }
                            ash::vk::DescriptorType::STORAGE_BUFFER => {
                                memory.storage_buffer_info(r.pool, r.offset, size)
                            }
                            _ => return Err(Status::Unimplemented),
                        }
                        .ok_or(Status::BadArgument)?;
                        resources.push((b.set, b.binding, info));
                    }
                    backend
                        .execution
                        .dispatch_with_images(
                            backend.global_memory.as_ref().ok_or(Status::BadArgument)?,
                            shader,
                            *requires_subgroup_size_32,
                            &resources,
                            &images,
                            groups,
                            indirect,
                        )
                        .ok_or(Status::Unimplemented)?;
                    backend.store_textures(&stored).ok_or(Status::InternalError)
                }
                "SetRenderEnable" => {
                    self.enabled = match a[0] {
                        0 => false,
                        1 => true,
                        _ => return Err(Status::Unimplemented),
                    };
                    Ok(())
                }
                "SetRenderEnableConditional" => {
                    let value = words(instance, backend, a[0], 1)?[0];
                    self.enabled = if a[1] == c.condition_nonzero {
                        value != 0
                    } else if a[1] == c.condition_zero {
                        value == 0
                    } else {
                        return Err(Status::Unimplemented);
                    };
                    Ok(())
                }
                "ClearTexture" => {
                    let (texture, color, mask) = c
                        .clear
                        .as_ref()
                        .and_then(|decode| decode(a))
                        .ok_or(Status::Unimplemented)?;
                    if !color.iter().all(|v| v.is_finite()) || mask & !15 != 0 {
                        return Err(Status::BadArgument);
                    }
                    let Some(Object::Texture { description, .. }) = instance.objects.get(texture)
                    else {
                        return Err(Status::BadArgument);
                    };
                    if !backend.ensure_texture(texture, &description, false) {
                        return Err(Status::BadArgument);
                    }
                    if backend.clear_color(texture, color, mask) {
                        Ok(())
                    } else {
                        Err(Status::InternalError)
                    }
                }
                "DrawArraysIndirect"
                | "DrawElementsIndirect"
                | "DrawArraysInstanced"
                | "DrawElementsInstanced" => {
                    if !self.enabled {
                        return Ok(());
                    }
                    let mut geometry = Geometry::default();
                    let vertices;
                    let count;
                    if kind == "DrawArraysIndirect" {
                        let w = words(instance, backend, a[1], 4)?;
                        geometry = Geometry {
                            instances: w[1],
                            first_instance: w[3],
                            indirect: Some(region(instance, backend, a[1], 16)?),
                            index_skip: 0,
                        };
                        vertices = super::super::drawing::Vertices::Arrays { first: w[2] };
                        count = w[0];
                    } else if kind == "DrawElementsIndirect" {
                        let w = words(instance, backend, a[3], 5)?;
                        let contract = backend.first_draw.as_ref().ok_or(Status::Unimplemented)?;
                        let width = if a[1] == u64::from(contract.index_u16) {
                            2_u64
                        } else if a[1] == u64::from(contract.index_u32) {
                            4
                        } else {
                            return Err(Status::Unimplemented);
                        };
                        let indices = a[2]
                            .checked_add(u64::from(w[2]) * width)
                            .ok_or(Status::BadArgument)?;
                        if w[0] != 0 {
                            region(
                                instance,
                                backend,
                                a[2],
                                (u64::from(w[2]) + u64::from(w[0])) * width,
                            )?;
                        }
                        geometry = Geometry {
                            instances: w[1],
                            first_instance: w[4],
                            indirect: Some(region(instance, backend, a[3], 20)?),
                            index_skip: u64::from(w[2]) * width,
                        };
                        vertices = super::super::drawing::Vertices::Elements {
                            index_type: integer(a[1])?,
                            indices,
                            base_vertex: w[3] as i32,
                        };
                        count = w[0];
                    } else if kind == "DrawArraysInstanced" {
                        vertices = super::super::drawing::Vertices::Arrays {
                            first: integer(a[1])?,
                        };
                        count = integer(a[2])?;
                        geometry.instances = integer(a[4])?;
                        geometry.first_instance = integer(a[3])?;
                    } else {
                        vertices = super::super::drawing::Vertices::Elements {
                            index_type: integer(a[1])?,
                            indices: a[3],
                            base_vertex: a[4] as u32 as i32,
                        };
                        count = integer(a[2])?;
                        geometry.instances = integer(a[6])?;
                        geometry.first_instance = integer(a[5])?;
                    }
                    if geometry.instances == 0 || count == 0 {
                        return Ok(());
                    }
                    if geometry.indirect.is_some() {
                        let (address, size) = if kind == "DrawArraysIndirect" {
                            (a[1], 16)
                        } else {
                            (a[3], 20)
                        };
                        let r = instance
                            .objects
                            .resolve_gpu_address(address)
                            .map_err(|_| Status::BadArgument)?;
                        for key in targets.iter().copied().chain((depth != 0).then_some(depth)) {
                            let Some(Object::Texture { description: d, .. }) =
                                instance.objects.get(key)
                            else {
                                return Err(Status::BadArgument);
                            };
                            let bytes =
                                backend
                                    .texture_storage_size(&d)
                                    .map(|n| n as u64)
                                    .or_else(|| {
                                        d.width
                                            .checked_mul(d.height)?
                                            .checked_mul(if key == depth { 5 } else { 4 })
                                    })
                                    .ok_or(Status::BadArgument)?;
                            if r.pool == d.pool
                                && r.offset
                                    < d.pool_offset
                                        .checked_add(bytes)
                                        .ok_or(Status::BadArgument)?
                                && d.pool_offset < r.offset + size
                            {
                                return Err(Status::BadArgument);
                            }
                        }
                    }
                    if geometry.indirect.is_some() && geometry.first_instance != 0 {
                        let supported = unsafe {
                            backend
                                .context()
                                .instance
                                .get_physical_device_features(backend.context().physical_device)
                        };
                        if supported.draw_indirect_first_instance == ash::vk::FALSE {
                            return Err(Status::Unimplemented);
                        }
                    }
                    draw.execute(
                        instance,
                        backend,
                        super::super::drawing::Request {
                            targets,
                            depth,
                            views,
                            primitive: integer(a[0])?,
                            vertices,
                            count,
                            geometry,
                        },
                    )
                }
                _ => Err(Status::Unimplemented),
            }
        }
    }
}
#[cfg(feature = "vulkan")]
pub(crate) use execution::State;

#[cfg(test)]
#[path = "operations_tests.rs"]
mod tests;
