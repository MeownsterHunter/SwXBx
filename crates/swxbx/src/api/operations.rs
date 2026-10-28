//! Deferred command hypotheses. Evidence and limits: the public contract.
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

/// Record a counter report: x1 is the guest counter token, x2 the report's
/// GPU address (signatures 0002). The token is parsed here, once, into the
/// counter it selects. A token the contract does not name stays an opaque
/// operation, which execution reports as unimplemented. Source: the public contract.
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

/// Record a counter reset: x1 is the guest counter token. Source: the public contract.
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

pub(super) fn bindless_texture_location(
    handle: &crate::surfaces::BindlessHandle,
) -> Option<(u8, u16)> {
    use crate::surfaces::{BindlessHandle, HandleOperation, HandleWord};
    match *handle {
        BindlessHandle::Word(HandleWord::ConstantBuffer { bank, offset }) => Some((bank, offset)),
        BindlessHandle::Combined {
            operation: HandleOperation::Or,
            a: HandleWord::ConstantBuffer { bank, offset },
            ..
        } => Some((bank, offset)),
        _ => None,
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

    /// A status whose reason is recorded as `swxbx_last_error`, so the
    /// batch it stops names why. Source: the public contract.
    pub(crate) struct Named(Status);

    /// Refuse with `status`, naming `reason`.
    fn no(status: Status, reason: impl std::fmt::Display) -> Named {
        Named(crate::unsupported::refuse(status, reason))
    }

    /// A status from a callee that names its own refusals.
    fn named(status: Status) -> Named {
        Named(status)
    }

    pub(crate) struct State {
        program: Option<u64>,
        buffers: HashMap<(u64, u64), (u64, u64)>,
        texel_buffers: HashMap<(u64, u64), (u64, u64, ash::vk::Format)>,
        pub enabled: bool,
    }
    impl Default for State {
        fn default() -> Self {
            Self {
                program: None,
                buffers: HashMap::new(),
                texel_buffers: HashMap::new(),
                enabled: true,
            }
        }
    }

    fn region(
        instance: &Instance,
        backend: &Backend,
        address: u64,
        size: u64,
    ) -> Result<(ash::vk::Buffer, u64), Named> {
        let r = resolve(instance, address)?;
        if size > r.remaining {
            return Err(no(
                Status::BadArgument,
                format!("{size} bytes at GPU address {address:#x} run past the end of its pool"),
            ));
        }
        let length = usize::try_from(size).map_err(|_| {
            no(
                Status::BadArgument,
                format!("{size} bytes at GPU address {address:#x} are too many to address"),
            )
        })?;
        memory(&backend.global_memory)?
            .buffer_region(r.pool, r.offset, length)
            .ok_or_else(|| {
                no(
                    Status::BadArgument,
                    format!(
                        "{size} bytes at GPU address {address:#x} are not in device-visible \
                         memory"
                    ),
                )
            })
    }

    /// The pool and offset a GPU address names.
    fn resolve(
        instance: &Instance,
        address: u64,
    ) -> Result<crate::api::objects::GpuAddress, Named> {
        instance.objects.resolve_gpu_address(address).map_err(|_| {
            no(
                Status::BadArgument,
                format!("GPU address {address:#x} is in no memory pool"),
            )
        })
    }

    /// The device's view of guest memory.
    fn memory(
        memory: &Option<crate::gpu::GlobalMemory>,
    ) -> Result<&crate::gpu::GlobalMemory, Named> {
        memory.as_ref().ok_or_else(|| {
            no(
                Status::BadArgument,
                "no memory pool is mapped on the device",
            )
        })
    }

    /// Finish the draws recorded before this command.
    fn finish(backend: &mut Backend) -> Result<(), Named> {
        backend.finish_draws().ok_or_else(|| {
            no(
                Status::InternalError,
                "the device could not finish the draws recorded before this command",
            )
        })
    }

    fn words(
        instance: &Instance,
        backend: &mut Backend,
        address: u64,
        count: usize,
    ) -> Result<Vec<u32>, Named> {
        let r = resolve(instance, address)?;
        let mut bytes = vec![0; count * 4];
        if bytes.len() as u64 > r.remaining || !address.is_multiple_of(4) {
            return Err(no(
                Status::BadArgument,
                format!(
                    "{count} words at GPU address {address:#x} are not 4-byte aligned or run \
                     past the end of its pool"
                ),
            ));
        }
        backend
            .global_memory
            .as_mut()
            .ok_or_else(|| {
                no(
                    Status::BadArgument,
                    "no memory pool is mapped on the device",
                )
            })?
            .read_pool(r.pool, r.offset, &mut bytes)
            .ok_or_else(|| {
                no(
                    Status::InternalError,
                    format!("the device could not read {count} words at GPU address {address:#x}"),
                )
            })?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect())
    }

    /// Write a counter report once every earlier command in queue order has
    /// been submitted. Source: the public contract.
    fn report_counter(
        instance: &Instance,
        backend: &mut Backend,
        counter: Counter,
        address: u64,
    ) -> Result<(), Named> {
        finish(backend)?;
        if !address.is_multiple_of(8) {
            return Err(no(
                Status::BadArgument,
                format!("counter report address {address:#x} is not 8-byte aligned"),
            ));
        }
        let destination = region(instance, backend, address, REPORT_SIZE)?;
        let memory = memory(&backend.global_memory)?;
        memory.mark_written();
        backend
            .execution
            .report(memory, destination, counter)
            .ok_or_else(|| {
                no(
                    Status::InternalError,
                    format!("the device could not write the {counter:?} report"),
                )
            })
    }

    /// Start a new accumulation interval. Source: the public contract.
    fn reset_counter(backend: &mut Backend, counter: Counter) -> Result<(), Named> {
        finish(backend)?;
        match counter {
            Counter::SamplesPassed => backend.execution.reset_occlusion().ok_or_else(|| {
                no(
                    Status::InternalError,
                    "the device could not reset the samples-passed counter",
                )
            }),
            Counter::Timestamp => Err(no(
                Status::Unimplemented,
                "swxbx does not reset the timestamp counter",
            )),
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
                RecordedCommand::State(StateCommand::BindTexelBuffer {
                    stage,
                    index,
                    address,
                    size,
                    format,
                }) => {
                    self.texel_buffers.insert(
                        (*stage, *index),
                        (*address, *size, ash::vk::Format::from_raw(*format as i32)),
                    );
                    return Some(Ok(()));
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
            Some(result.map_err(|Named(status)| status))
        }

        fn operation(
            &mut self,
            kind: &str,
            a: [u64; 7],
            instance: &Instance,
            backend: &mut Backend,
            draw: &mut super::super::drawing::State,
            target: super::Targets<'_>,
        ) -> Result<(), Named> {
            let super::Targets {
                colors: targets,
                depth,
                views,
            } = target;
            draw.invalidate();
            finish(backend)?;
            if let "DrawArraysInstanced" | "DrawElementsInstanced" = kind {
                return self.instanced(kind, a, instance, backend, draw, target);
            }
            let c = backend.command_contract.clone().ok_or_else(|| {
                no(
                    Status::Unimplemented,
                    format!("{kind} needs a command contract, and none is installed"),
                )
            })?;
            let integer = |n: u64| {
                u32::try_from(n).map_err(|_| {
                    no(
                        Status::BadArgument,
                        format!("{kind}: argument {n:#x} does not fit in 32 bits"),
                    )
                })
            };
            match kind {
                "ClearBuffer" => {
                    let r = region(instance, backend, a[0], a[1])?;
                    if a[1] == 0 || !a[1].is_multiple_of(4) || !r.1.is_multiple_of(4) {
                        return Err(no(
                            Status::BadArgument,
                            format!(
                                "ClearBuffer of {} bytes at {:#x}: the size and offset must be \
                                 non-zero multiples of 4",
                                a[1], a[0]
                            ),
                        ));
                    }
                    let value = integer(a[2])?;
                    memory(&backend.global_memory)?.mark_written();
                    backend
                        .execution
                        .fill(memory(&backend.global_memory)?, r, a[1], value)
                        .ok_or_else(|| {
                            no(
                                Status::InternalError,
                                "the device could not record the buffer clear",
                            )
                        })
                }
                "CopyBufferToBuffer" => {
                    if a[3] != 0 {
                        return Err(no(
                            Status::Unimplemented,
                            format!(
                                "CopyBufferToBuffer with flags {:#x}; swxbx copies with none",
                                a[3]
                            ),
                        ));
                    }
                    let source = region(instance, backend, a[0], a[2])?;
                    let destination = region(instance, backend, a[1], a[2])?;
                    if source.0 == destination.0
                        && source.1 < destination.1 + a[2]
                        && destination.1 < source.1 + a[2]
                    {
                        return Err(no(
                            Status::BadArgument,
                            format!(
                                "CopyBufferToBuffer of {} bytes from {:#x} to {:#x}: the ranges \
                                 overlap",
                                a[2], a[0], a[1]
                            ),
                        ));
                    }
                    memory(&backend.global_memory)?.mark_written();
                    backend
                        .execution
                        .copy(memory(&backend.global_memory)?, source, destination, a[2])
                        .ok_or_else(|| {
                            no(
                                Status::InternalError,
                                "the device could not record the buffer copy",
                            )
                        })
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
                        return Err(no(
                            Status::BadArgument,
                            format!(
                                "{kind} of {groups:?} work groups exceeds this device's {:?}",
                                limits.max_compute_work_group_count
                            ),
                        ));
                    }
                    let program = self.program.ok_or_else(|| {
                        no(
                            Status::Unimplemented,
                            format!("{kind} with no program bound"),
                        )
                    })?;
                    let Some(Object::Program {
                        shader_translations,
                        ..
                    }) = instance.objects.get(program)
                    else {
                        return Err(no(
                            Status::BadArgument,
                            format!("{kind}: the bound object {program} is not a program"),
                        ));
                    };
                    let mut stages = Vec::new();
                    for t in shader_translations {
                        let t = match t {
                            ShaderTranslation::Deferred(code, context) => instance
                                .retry_cached_translation(&code, &context, true)
                                .ok_or_else(|| {
                                    no(
                                        Status::Unimplemented,
                                        format!(
                                            "{kind}: the bound program's stage has no translation"
                                        ),
                                    )
                                })?,
                            t => t,
                        };
                        match t {
                            ShaderTranslation::Spirv(w) => {
                                stages.push((w, false, Vec::new(), Vec::new()))
                            }
                            ShaderTranslation::ByInputPrimitive(_) => {
                                return Err(no(
                                    Status::Unimplemented,
                                    "dispatch: the bound program holds a geometry stage, \
                                     which draws run and dispatches do not",
                                ))
                            }
                            ShaderTranslation::Cached(r) => match r.wait() {
                                crate::startup_cache::TranslationStatus::Ready(s) => {
                                    backend
                                        .context()
                                        .shader_features
                                        .validate_translated_shader(&s)
                                        .map_err(|error| {
                                            no(
                                                Status::Unimplemented,
                                                format!("compute pipeline: {error}"),
                                            )
                                        })?;
                                    stages.push((
                                        s.words.clone(),
                                        s.requires_subgroup_size_32,
                                        s.surfaces.clone(),
                                        s.textures.clone(),
                                    ))
                                }
                                _ => {
                                    return Err(no(
                                        Status::Unimplemented,
                                        format!(
                                            "{kind}: the bound program's cached translation \
                                             failed"
                                        ),
                                    ))
                                }
                            },
                            ShaderTranslation::Error(error) => {
                                return Err(no(
                                    Status::Unimplemented,
                                    format!("{kind}: compute translation failed: {error}"),
                                ))
                            }
                            _ => {
                                return Err(no(
                                    Status::Unimplemented,
                                    format!(
                                        "{kind}: the bound program holds a stage translated \
                                         for a draw"
                                    ),
                                ))
                            }
                        }
                    }
                    let compute: Vec<_> = stages
                        .iter()
                        .filter(|s| execution_model(&s.0).ok() == Some(5))
                        .collect();
                    if compute.len() != 1 {
                        return Err(no(
                            Status::Unimplemented,
                            format!(
                                "{kind}: the bound program holds {} compute stages, not one",
                                compute.len()
                            ),
                        ));
                    }
                    let (shader, requires_subgroup_size_32, surfaces, bindless_textures) =
                        compute[0];
                    let reflected =
                        crate::gpu::pipelines::compute_bindings(shader).map_err(|error| {
                            no(
                                Status::Unimplemented,
                                format!("{kind}: the compute stage's bindings: {error:?}"),
                            )
                        })?;
                    let mut resources = Vec::new();
                    let mut images = Vec::new();
                    let mut sampled = Vec::new();
                    let mut texels = Vec::new();
                    let mut stored = Vec::new();
                    for b in reflected {
                        if b.count != 1 {
                            return Err(no(
                                Status::Unimplemented,
                                format!(
                                    "{kind}: binding {} of set {} is an array of {}",
                                    b.binding, b.set, b.count
                                ),
                            ));
                        }
                        if b.descriptor_type == ash::vk::DescriptorType::STORAGE_IMAGE {
                            let skip = |reason: &str| {
                                no(Status::Unimplemented, format!("compute dispatch: {reason}"))
                            };
                            let surface = surfaces
                                .iter()
                                .find(|s| (s.set, s.binding) == (b.set, b.binding))
                                .ok_or_else(|| skip("storage image without a typed surface"))?;
                            let contract = backend.texture_contract.clone();
                            let texture = draw
                                .bound_surface(
                                    instance,
                                    &contract,
                                    crate::gpu::textures::SurfaceStage::Compute,
                                    surface,
                                    c.compute_texel_location_base
                                        .map(|base| (c.compute_stage, base)),
                                    &|status, reason| {
                                        crate::unsupported::refuse(
                                            status,
                                            format!("compute dispatch: {reason}"),
                                        )
                                    },
                                )
                                .map_err(named)?;
                            let Some(Object::Texture { description: d, .. }) =
                                instance.objects.get(texture)
                            else {
                                return Err(no(
                                    Status::BadArgument,
                                    format!("compute dispatch: object {texture} is not a texture"),
                                ));
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
                        if matches!(
                            b.descriptor_type,
                            ash::vk::DescriptorType::SAMPLED_IMAGE
                                | ash::vk::DescriptorType::SAMPLER
                        ) {
                            let image = b.descriptor_type == ash::vk::DescriptorType::SAMPLED_IMAGE;
                            let contract = backend.texture_contract.clone();
                            let source = if let Some(texture) = bindless_textures.iter().find(|t| {
                                t.set == b.set
                                    && if image {
                                        t.image == b.binding
                                    } else {
                                        t.sampler == b.binding
                                    }
                            }) {
                                let (texture_word, sampler_word) =
                                    super::super::drawing::handle_locations(&texture.source)
                                        .ok_or_else(|| {
                                            no(
                                                Status::Unimplemented,
                                                "compute dispatch: unsupported bindless texture handle",
                                            )
                                        })?;
                                let (bank, offset) =
                                    if image { texture_word } else { sampler_word };
                                let base = c.compute_texel_location_base.ok_or_else(|| {
                                    no(
                                        Status::Unimplemented,
                                        "compute dispatch: bindless texture locations are unmapped",
                                    )
                                })?;
                                (
                                    c.compute_stage,
                                    base + (u64::from(bank) << 16 | u64::from(offset)),
                                )
                            } else if b.set == 1 && b.binding % 2 == u32::from(!image) {
                                (c.compute_stage, u64::from(b.binding / 2))
                            } else {
                                return Err(no(
                                    Status::Unimplemented,
                                    format!(
                                        "compute dispatch: no host mapping for set {} binding {}",
                                        b.set, b.binding
                                    ),
                                ));
                            };
                            let refuse = |status, reason| {
                                crate::unsupported::refuse(
                                    status,
                                    format!("compute dispatch: {reason}"),
                                )
                            };
                            if !draw.has_bound_sampled_object(source, image) {
                                continue;
                            }
                            let (pool, id, object) = draw
                                .bound_sampled_object(instance, &contract, source, image, &refuse)
                                .map_err(named)?;
                            let info = if image {
                                let description = instance
                                    .objects
                                    .texture_description(object)
                                    .ok_or_else(|| {
                                        no(
                                            Status::BadArgument,
                                            format!("compute texture {object} is not a texture"),
                                        )
                                    })?;
                                let _resolved =
                                    backend.texture_image(object, &description).ok_or_else(
                                        || {
                                            no(
                                        Status::Unimplemented,
                                        format!("compute texture {object} has no image contract"),
                                    )
                                        },
                                    )?;
                                let view = backend
                                    .sampled_texture(object, &description)
                                    .ok_or_else(|| {
                                        no(
                                            Status::Unimplemented,
                                            format!("compute texture {object} has no sampled view"),
                                        )
                                    })?;
                                ash::vk::DescriptorImageInfo::default()
                                    .image_view(view)
                                    .image_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            } else {
                                let description = instance
                                    .objects
                                    .with(object, |candidate| match candidate {
                                        Object::Sampler(description) => Some(description.clone()),
                                        _ => None,
                                    })
                                    .flatten()
                                    .ok_or_else(|| {
                                        no(
                                            Status::BadArgument,
                                            format!("compute sampler {object} is not a sampler"),
                                        )
                                    })?;
                                let key =
                                    crate::gpu::textures::SamplerKey::new(&description, &contract)
                                        .ok_or_else(|| {
                                            no(
                                                Status::Unimplemented,
                                                format!("compute sampler {object} is unsupported"),
                                            )
                                        })?;
                                let sampler = backend.sampler(pool, id, key).ok_or_else(|| {
                                    no(
                                        Status::InternalError,
                                        format!("compute sampler {object} could not be created"),
                                    )
                                })?;
                                ash::vk::DescriptorImageInfo::default().sampler(sampler)
                            };
                            sampled.push((b.set, b.binding, info));
                            continue;
                        }
                        if matches!(
                            b.descriptor_type,
                            ash::vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                                | ash::vk::DescriptorType::STORAGE_TEXEL_BUFFER
                        ) {
                            let direct = c
                                .compute_texel_buffers
                                .iter()
                                .find(|m| (m.set, m.binding) == (b.set, b.binding));
                            let index = if let Some(mapping) = direct {
                                mapping.index
                            } else {
                                let source = bindless_textures
                                    .iter()
                                    .find(|texture| {
                                        texture.set == b.set && texture.image == b.binding
                                    })
                                    .map(|texture| texture.source)
                                    .or_else(|| {
                                        surfaces
                                            .iter()
                                            .find(|surface| {
                                                (surface.set, surface.binding)
                                                    == (b.set, b.binding)
                                                    && surface.dimension
                                                        == crate::surfaces::SurfaceDimension::Buffer
                                            })
                                            .and_then(|surface| match surface.source {
                                                crate::surfaces::SurfaceSource::Bindless(source) => {
                                                    Some(source)
                                                }
                                                crate::surfaces::SurfaceSource::Bound { .. } => None,
                                            })
                                    })
                                    .ok_or_else(|| {
                                        no(
                                            Status::Unimplemented,
                                            format!(
                                                "compute dispatch: texel buffer at set {} binding {} \
                                                 has no fixed or bindless mapping",
                                                b.set, b.binding
                                            ),
                                        )
                                    })?;
                                let (bank, offset) = bindless_texture_location(&source)
                                    .ok_or_else(|| {
                                        no(
                                            Status::Unimplemented,
                                            "compute dispatch: unsupported bindless texel handle",
                                        )
                                    })?;
                                c.compute_texel_location_base
                                    .and_then(|base| {
                                        base.checked_add(
                                            (u64::from(bank) << 16) | u64::from(offset),
                                        )
                                    })
                                    .ok_or_else(|| {
                                        no(
                                            Status::Unimplemented,
                                            "compute dispatch: bindless texel locations are unmapped",
                                        )
                                    })?
                            };
                            let &(address, size, format) = self
                                .texel_buffers
                                .get(&(c.compute_stage, index))
                                .ok_or_else(|| {
                                    no(
                                        Status::Unimplemented,
                                        format!(
                                            "compute dispatch: texel buffer at set {} binding {} \
                                             maps to host index {index}, which is not bound",
                                            b.set, b.binding
                                        ),
                                    )
                                })?;
                            let r = resolve(instance, address)?;
                            let length = usize::try_from(size).map_err(|_| {
                                no(
                                    Status::BadArgument,
                                    "compute dispatch: texel-buffer range is too large",
                                )
                            })?;
                            let (buffer, offset) = memory(&backend.global_memory)?
                                .buffer_region(r.pool, r.offset, length)
                                .ok_or_else(|| {
                                    no(
                                        Status::BadArgument,
                                        "compute dispatch: texel buffer is not device-visible",
                                    )
                                })?;
                            texels.push((b.set, b.binding, buffer, offset, size, format));
                            continue;
                        }
                        let m =
                            c.compute_buffers
                                .iter()
                                .find(|m| (m.set, m.binding) == (b.set, b.binding))
                                .ok_or_else(|| {
                                    no(
                                        Status::Unimplemented,
                                        format!(
                                        "{kind}: the command contract maps no descriptor type {} \
                                         resource to \
                                         set {} binding {}",
                                        b.descriptor_type.as_raw(), b.set, b.binding
                                    ),
                                    )
                                })?;
                        let &(address, size) = self
                            .buffers
                            .get(&(c.compute_stage, m.index))
                            .ok_or_else(|| {
                                no(
                                    Status::Unimplemented,
                                    format!(
                                        "{kind}: the compute stage reads uniform buffer {}, \
                                         which is not bound; bound indices are {:?}",
                                        m.index,
                                        self.buffers
                                            .keys()
                                            .filter(|(stage, _)| *stage == c.compute_stage)
                                            .map(|(_, index)| *index)
                                            .collect::<Vec<_>>()
                                    ),
                                )
                            })?;
                        let r = resolve(instance, address)?;
                        if size > r.remaining {
                            return Err(no(
                                Status::BadArgument,
                                format!(
                                    "{kind}: buffer {} of {size} bytes at {address:#x} runs past \
                                     the end of its pool",
                                    m.index
                                ),
                            ));
                        }
                        let memory = memory(&backend.global_memory)?;
                        let info = match b.descriptor_type {
                            ash::vk::DescriptorType::UNIFORM_BUFFER => {
                                memory.uniform_buffer_info(r.pool, r.offset, size)
                            }
                            ash::vk::DescriptorType::STORAGE_BUFFER => {
                                memory.storage_buffer_info(r.pool, r.offset, size)
                            }
                            other => {
                                return Err(no(
                                    Status::Unimplemented,
                                    format!(
                                        "{kind}: set {} binding {} is descriptor type {}",
                                        b.set,
                                        b.binding,
                                        other.as_raw()
                                    ),
                                ))
                            }
                        }
                        .ok_or_else(|| {
                            no(
                                Status::BadArgument,
                                format!(
                                    "{kind}: buffer {} at {address:#x} is not in device-visible \
                                     memory",
                                    m.index
                                ),
                            )
                        })?;
                        resources.push((b.set, b.binding, info));
                    }
                    backend
                        .execution
                        .dispatch_with_images(
                            memory(&backend.global_memory)?,
                            shader,
                            *requires_subgroup_size_32,
                            &resources,
                            &images,
                            &sampled,
                            &texels,
                            groups,
                            indirect,
                        )
                        .ok_or_else(|| {
                            no(
                                Status::Unimplemented,
                                format!("the device could not record the {kind}"),
                            )
                        })?;
                    backend.store_textures(&stored).ok_or_else(|| {
                        no(
                            Status::InternalError,
                            format!("the device could not keep the images the {kind} wrote"),
                        )
                    })
                }
                "SetRenderEnable" => {
                    self.enabled = match a[0] {
                        0 => false,
                        1 => true,
                        other => {
                            return Err(no(
                                Status::Unimplemented,
                                format!("SetRenderEnable with value {other}"),
                            ))
                        }
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
                        return Err(no(
                            Status::Unimplemented,
                            format!(
                                "SetRenderEnableConditional with condition token {}, which the \
                                 command contract does not map",
                                a[1]
                            ),
                        ));
                    };
                    Ok(())
                }
                "ClearTexture" => {
                    let (texture, color, mask) = c
                        .clear
                        .as_ref()
                        .and_then(|decode| decode(a))
                        .ok_or_else(|| {
                            no(
                                Status::Unimplemented,
                                "ClearTexture arguments the command contract does not decode",
                            )
                        })?;
                    if !color.iter().all(|v| v.is_finite()) || mask & !15 != 0 {
                        return Err(no(
                            Status::BadArgument,
                            format!("ClearTexture of texture {texture} with colour {color:?} and mask {mask:#x}"),
                        ));
                    }
                    let Some(Object::Texture { description, .. }) = instance.objects.get(texture)
                    else {
                        return Err(no(
                            Status::BadArgument,
                            format!("ClearTexture: object {texture} is not a texture"),
                        ));
                    };
                    if !backend.ensure_texture(texture, &description, false) {
                        return Err(no(
                            Status::BadArgument,
                            format!("ClearTexture: texture {texture} has no device image"),
                        ));
                    }
                    backend.clear_color(texture, color, mask).map_err(named)
                }
                "DrawArraysIndirect" | "DrawElementsIndirect" => {
                    if !self.enabled {
                        return Ok(());
                    }
                    let (geometry, vertices, count) = if kind == "DrawArraysIndirect" {
                        let w = words(instance, backend, a[1], 4)?;
                        (
                            Geometry {
                                instances: w[1],
                                first_instance: w[3],
                                indirect: Some(region(instance, backend, a[1], 16)?),
                                index_skip: 0,
                            },
                            super::super::drawing::Vertices::Arrays { first: w[2] },
                            w[0],
                        )
                    } else {
                        let w = words(instance, backend, a[3], 5)?;
                        let contract = backend.first_draw.as_ref().ok_or_else(|| {
                            no(
                                Status::Unimplemented,
                                "no first-draw contract is installed, so swxbx cannot \
                                 interpret draws",
                            )
                        })?;
                        let width = if a[1] == u64::from(contract.index_u16) {
                            2_u64
                        } else if a[1] == u64::from(contract.index_u32) {
                            4
                        } else {
                            return Err(no(
                                Status::Unimplemented,
                                format!(
                                    "{kind}: index type token {} is not a 16- or 32-bit index \
                                     in the first-draw contract",
                                    a[1]
                                ),
                            ));
                        };
                        let indices =
                            a[2].checked_add(u64::from(w[2]) * width).ok_or_else(|| {
                                no(
                                    Status::BadArgument,
                                    format!("{kind}: the first index overflows the address space"),
                                )
                            })?;
                        if w[0] != 0 {
                            region(
                                instance,
                                backend,
                                a[2],
                                (u64::from(w[2]) + u64::from(w[0])) * width,
                            )?;
                        }
                        (
                            Geometry {
                                instances: w[1],
                                first_instance: w[4],
                                indirect: Some(region(instance, backend, a[3], 20)?),
                                index_skip: u64::from(w[2]) * width,
                            },
                            super::super::drawing::Vertices::Elements {
                                index_type: integer(a[1])?,
                                indices,
                                base_vertex: w[3] as i32,
                            },
                            w[0],
                        )
                    };
                    if geometry.instances == 0 || count == 0 {
                        return Ok(());
                    }
                    if geometry.indirect.is_some() {
                        let (address, size) = if kind == "DrawArraysIndirect" {
                            (a[1], 16)
                        } else {
                            (a[3], 20)
                        };
                        let r = resolve(instance, address)?;
                        for key in targets.iter().copied().chain((depth != 0).then_some(depth)) {
                            let Some(Object::Texture { description: d, .. }) =
                                instance.objects.get(key)
                            else {
                                return Err(no(
                                    Status::BadArgument,
                                    format!("{kind}: target {key} is not a texture"),
                                ));
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
                                    .ok_or_else(|| {
                                        no(
                                            Status::BadArgument,
                                            format!("{kind}: target {key} is too large to address"),
                                        )
                                    })?;
                            if r.pool == d.pool
                                && r.offset < d.pool_offset.saturating_add(bytes)
                                && d.pool_offset < r.offset + size
                            {
                                return Err(no(
                                    Status::BadArgument,
                                    format!(
                                        "{kind}: the indirect arguments at {address:#x} lie in \
                                         target {key}"
                                    ),
                                ));
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
                            return Err(no(
                                Status::Unimplemented,
                                format!(
                                    "{kind} starts at instance {}, and this device draws \
                                     indirect from instance 0 only",
                                    geometry.first_instance
                                ),
                            ));
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
                    .map_err(named)
                }
                other => Err(no(
                    Status::Unimplemented,
                    format!("swxbx does not execute {other}"),
                )),
            }
        }

        /// Run an instanced draw with NVN's own arguments, which need no
        /// command contract: `DrawArraysInstanced(mode, first, count,
        /// baseInstance, instanceCount)` and `DrawElementsInstanced(mode,
        /// type, count, indices, baseVertex, baseInstance, instanceCount)`.
        fn instanced(
            &mut self,
            kind: &str,
            a: [u64; 7],
            instance: &Instance,
            backend: &mut Backend,
            draw: &mut super::super::drawing::State,
            target: super::Targets<'_>,
        ) -> Result<(), Named> {
            if !self.enabled {
                return Ok(());
            }
            let integer = |n: u64| {
                u32::try_from(n).map_err(|_| {
                    no(
                        Status::BadArgument,
                        format!("{kind}: argument {n:#x} does not fit in 32 bits"),
                    )
                })
            };
            let (vertices, count, first_instance, instances) = match kind {
                "DrawArraysInstanced" => (
                    super::super::drawing::Vertices::Arrays {
                        first: integer(a[1])?,
                    },
                    integer(a[2])?,
                    integer(a[3])?,
                    integer(a[4])?,
                ),
                _ => (
                    super::super::drawing::Vertices::Elements {
                        index_type: integer(a[1])?,
                        indices: a[3],
                        base_vertex: a[4] as u32 as i32,
                    },
                    integer(a[2])?,
                    integer(a[5])?,
                    integer(a[6])?,
                ),
            };
            if instances == 0 || count == 0 {
                return Ok(());
            }
            let super::Targets {
                colors: targets,
                depth,
                views,
            } = target;
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
                    geometry: Geometry {
                        instances,
                        first_instance,
                        ..Geometry::default()
                    },
                },
            )
            .map_err(named)
        }
    }
}
#[cfg(feature = "vulkan")]
pub(crate) use execution::State;

#[cfg(test)]
#[path = "operations_tests.rs"]
mod tests;
