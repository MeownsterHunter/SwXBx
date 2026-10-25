//! Bounded execution of recorded draws under explicit host contracts.
use super::objects::{Object, RecordedCommand, ShaderTranslation, StateCommand, StateSettings};
use crate::{
    gpu::{
        graphics::{
            execution_model, mapped, DepthRasterContract, Draw, DrawPipelineState, DrawVertices,
            VertexAttribute, VertexBinding, VertexInput,
        },
        uniforms::{self, UniformStage, BANK_SIZE},
        Backend,
    },
    instance::InstanceState as Instance,
    Status,
};
use ash::vk;
use std::{collections::HashMap, sync::Arc};

type Settings = StateSettings;

pub(super) struct Request<'a> {
    /// Color texture object keys in attachment order.
    pub targets: &'a [u64],
    /// Depth/stencil texture key, or zero without an attachment.
    pub depth: u64,
    /// Opaque color and depth view arguments; the bounded draw requires zero.
    pub views: [u64; 2],
    /// Raw primitive token interpreted only through a host contract.
    pub primitive: u32,
    /// Array or indexed geometry selection.
    pub vertices: Vertices,
    /// Number of vertices or indices in the draw.
    pub count: u32,
    pub geometry: crate::gpu::operations::Geometry,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Vertices {
    Arrays {
        first: u32,
    },
    Elements {
        index_type: u32,
        indices: u64,
        base_vertex: i32,
    },
}

#[derive(Clone, Copy)]
enum IndexFormat {
    U16,
    U32,
}

impl IndexFormat {
    fn width(self) -> usize {
        match self {
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }
    fn vulkan(self) -> vk::IndexType {
        match self {
            Self::U16 => vk::IndexType::UINT16,
            Self::U32 => vk::IndexType::UINT32,
        }
    }
}

#[derive(Clone, Copy)]
enum TextureReference {
    Separate(u64),
    Combined(u64),
}

#[derive(Default)]
pub(super) struct State {
    program: Option<u64>,
    buffers: HashMap<u64, (u64, u64)>,
    vertex_states: HashMap<&'static str, Option<Vec<Settings>>>,
    uniforms: HashMap<(u64, u64), (u64, u64)>,
    texture_pool: Option<u64>,
    sampler_pool: Option<u64>,
    textures: HashMap<(u64, u64), TextureReference>,
    samplers: HashMap<(u64, u64), TextureReference>,
    images: HashMap<(u64, u64), u64>,
    blends: HashMap<u64, Option<Settings>>,
    states: HashMap<&'static str, Option<Settings>>,
    viewport: Option<[u64; 5]>,
    scissor: Option<[u64; 5]>,
    unsupported: bool,
    stencil_commands: Vec<(&'static str, u64, u64)>,
    bias: Option<[u64; 3]>,
    prepared: Option<Prepared>,
}

struct Prepared {
    targets: Vec<u64>,
    depth: u64,
    primitive: u32,
    draw: Arc<Draw>,
    pipeline: Arc<crate::gpu::graphics::GraphicsPipeline>,
    fetches: Vec<(u32, u32, u32, u64)>,
    source: Vertices,
    count: u32,
    vertices: DrawVertices,
}
impl State {
    /// Arena writes can change index values used by the previous bounds check.
    pub fn invalidate(&mut self) {
        self.prepared = None;
    }

    /// Update retained draw state from a recorded command.
    pub fn record(&mut self, command: RecordedCommand) {
        self.prepared = None;
        match command {
            RecordedCommand::State(StateCommand::SetDescriptorPool { sampler, pool }) => {
                if sampler {
                    self.sampler_pool = Some(pool);
                } else {
                    self.texture_pool = Some(pool);
                }
            }
            RecordedCommand::State(StateCommand::BindHandle {
                kind,
                stage,
                index,
                handle,
            }) if matches!(kind, "Texture" | "SeparateTexture") => {
                if kind == "Texture" {
                    self.textures
                        .insert((stage, index), TextureReference::Combined(handle));
                    self.samplers
                        .insert((stage, index), TextureReference::Combined(handle));
                } else {
                    self.textures
                        .insert((stage, index), TextureReference::Separate(handle));
                }
            }
            RecordedCommand::State(StateCommand::BindHandle {
                kind: "Image",
                stage,
                index,
                handle,
            }) => {
                self.images.insert((stage, index), handle);
            }
            RecordedCommand::State(StateCommand::BindSamplerReference {
                stage,
                index,
                reference,
            }) => {
                self.samplers
                    .insert((stage, index), TextureReference::Separate(reference));
            }
            RecordedCommand::State(StateCommand::BindState {
                kind: "BlendState",
                settings,
                ..
            }) => {
                if let Some(target) = settings.as_ref().and_then(|s| value(s, "BlendTarget").ok()) {
                    self.blends.insert(target[0], settings);
                } else {
                    self.unsupported = true;
                }
            }
            RecordedCommand::State(StateCommand::BindUniformBuffer {
                stage,
                index,
                address,
                size,
            }) => {
                self.uniforms.insert((stage, index), (address, size));
            }
            RecordedCommand::BindProgram(program) => self.program = Some(program),
            RecordedCommand::State(StateCommand::BindVertexBuffer {
                index: stream,
                address,
                size,
            }) => {
                self.buffers.insert(stream, (address, size));
            }
            RecordedCommand::State(StateCommand::BindStates {
                kind,
                count,
                first_settings,
                experiment_settings,
                ..
            }) => {
                let settings = experiment_settings.or_else(|| match count {
                    0 => Some(Vec::new()),
                    1 => first_settings.map(|settings| vec![settings]),
                    _ => None,
                });
                self.vertex_states.insert(kind, settings);
            }
            RecordedCommand::BindState { kind, settings }
            | RecordedCommand::State(StateCommand::BindState { kind, settings, .. })
                if matches!(
                    kind,
                    "ColorState" | "ChannelMaskState" | "DepthStencilState" | "PolygonState"
                ) =>
            {
                self.states.insert(kind, settings);
            }
            RecordedCommand::State(StateCommand::Stencil {
                setting,
                faces,
                value,
            }) => {
                self.stencil_commands.push((setting, faces, value));
            }
            RecordedCommand::State(StateCommand::PolygonOffset(bias)) => self.bias = Some(bias),
            RecordedCommand::SetViewport(viewport) => self.viewport = Some(viewport),
            RecordedCommand::SetScissor(scissor) => self.scissor = Some(scissor),
            RecordedCommand::SetDepthRange(range) => {
                // The experiment fixes the Vulkan depth interval to [0, 1].
                if range != [0, 0, u64::from(1.0_f32.to_bits())] {
                    self.unsupported = true;
                }
            }
            _ => self.unsupported = true,
        }
    }

    fn settings(&self, kind: &str, allowed: &[&str]) -> Result<&Settings, Status> {
        let settings = self
            .states
            .get(kind)
            .and_then(Option::as_ref)
            .ok_or(Status::Unimplemented)?;
        if settings.iter().any(|(name, _)| !allowed.contains(name)) {
            return Err(Status::Unimplemented);
        }
        Ok(settings)
    }

    fn pipeline_state(
        &self,
        extension: Option<DepthRasterContract>,
        cull_none: u64,
        attachment: bool,
    ) -> Result<DrawPipelineState, Status> {
        let settings = self.settings(
            "DepthStencilState",
            &[
                "DepthTestEnable",
                "DepthWriteEnable",
                "StencilTestEnable",
                "DepthFunc",
                "StencilFunc",
                "StencilOp",
            ],
        )?;
        let mut state = DrawPipelineState {
            attachment,
            depth_test: boolean(value(settings, "DepthTestEnable")?[0])?,
            depth_write: boolean(value(settings, "DepthWriteEnable")?[0])?,
            stencil_test: boolean(value(settings, "StencilTestEnable")?[0])?,
            ..Default::default()
        };
        let polygon = self.settings("PolygonState", &["CullFace", "PolygonMode"])?;
        let cull = value(polygon, "CullFace")?[0];
        let Some(extension) = extension else {
            if state.attachment
                || state.depth_test
                || state.depth_write
                || state.stencil_test
                || settings
                    .iter()
                    .any(|(name, _)| matches!(*name, "DepthFunc" | "StencilFunc" | "StencilOp"))
                || polygon.iter().any(|(name, _)| *name == "PolygonMode")
                || cull != cull_none
                || !self.stencil_commands.is_empty()
                || self.bias.is_some()
            {
                return Err(Status::Unimplemented);
            }
            return Ok(state);
        };
        if (state.depth_test || state.depth_write || state.stencil_test) && !attachment {
            return Err(Status::Unimplemented);
        }
        state.clockwise = extension.clockwise;
        state.cull = if cull == cull_none {
            0
        } else {
            [1, 2, 3][token(cull, &extension.cull)? as usize]
        };
        if let Ok(mode) = value(polygon, "PolygonMode") {
            state.polygon = token(mode[0], &extension.polygon)?;
        }
        if state.depth_test {
            state.depth_compare = token(value(settings, "DepthFunc")?[0], &extension.compare)?;
        }
        // Replay face-specific setters so a later front-only setter preserves the back face.
        let mut functions = [false; 2];
        let mut ops = [false; 2];
        for &(name, args) in settings {
            if !matches!(name, "StencilFunc" | "StencilOp") {
                continue;
            }
            let faces = face_mask(args[0], &extension)?;
            for i in 0..2 {
                if faces & (1 << i) == 0 {
                    continue;
                }
                if name == "StencilFunc" {
                    state.stencil[i].compare = token(args[1], &extension.compare)?;
                    state.stencil[i].reference = byte(args[2])?;
                    state.stencil[i].mask = byte(args[3])?;
                    functions[i] = true;
                } else {
                    // Hypothesis: fail, depth fail, pass.
                    state.stencil[i].fail = token(args[1], &extension.stencil_ops)?;
                    state.stencil[i].depth_fail = token(args[2], &extension.stencil_ops)?;
                    state.stencil[i].pass = token(args[3], &extension.stencil_ops)?;
                    ops[i] = true;
                }
            }
        }
        let mut masks = [false; 2];
        for &(setting, selector, v) in &self.stencil_commands {
            let faces = face_mask(selector, &extension)?;
            let v = byte(v)?;
            for (i, stencil) in state.stencil.iter_mut().enumerate() {
                if faces & (1 << i) == 0 {
                    continue;
                }
                match setting {
                    "StencilMask" => {
                        stencil.write_mask = v;
                        masks[i] = true;
                    }
                    "StencilRef" => stencil.reference = v,
                    "StencilValueMask" => stencil.mask = v,
                    _ => return Err(Status::Unimplemented),
                }
            }
        }
        if state.stencil_test && !(0..2).all(|i| functions[i] && ops[i] && masks[i]) {
            return Err(Status::Unimplemented);
        }
        if let Some(bias) = self.bias {
            if !extension.polygon_offset {
                return Err(Status::Unimplemented);
            }
            for (i, bits) in bias.into_iter().enumerate() {
                let bits = u32::try_from(bits).map_err(|_| Status::BadArgument)?;
                let v = f32::from_bits(bits);
                if !v.is_finite() {
                    return Err(Status::BadArgument);
                }
                state.bias[i] = if v == 0.0 { 0 } else { bits };
            }
        }
        Ok(state)
    }

    /// Execute one bounded draw using explicit host contracts; report unsupported state.
    pub fn execute(
        &mut self,
        instance: &Instance,
        backend: &mut Backend,
        request: Request<'_>,
    ) -> Result<(), Status> {
        if let Some(prepared) = &self.prepared {
            if request.geometry == crate::gpu::operations::Geometry::default()
                && request.views == [0, 0]
                && prepared.targets == request.targets
                && prepared.depth == request.depth
                && prepared.primitive == request.primitive
            {
                let vertices = match request.vertices {
                    Vertices::Arrays { first } => {
                        for &(stride, offset, bytes, size) in &prepared.fetches {
                            if vertex_end(first, request.count, stride, offset, bytes)
                                .ok_or(Status::BadArgument)?
                                > size
                            {
                                return Err(Status::BadArgument);
                            }
                        }
                        Some(DrawVertices::Arrays { first })
                    }
                    source if source == prepared.source && request.count == prepared.count => {
                        Some(prepared.vertices)
                    }
                    _ => None,
                };
                if let Some(vertices) = vertices {
                    #[cfg(feature = "draw-metrics")]
                    crate::draw_metrics::count(3);
                    return backend
                        .queue_draw(
                            request.targets,
                            (request.depth != 0).then_some(request.depth),
                            crate::gpu::graphics::PendingDraw {
                                pipeline: prepared.pipeline.clone(),
                                state: prepared.draw.clone(),
                                vertices,
                                count: request.count,
                                geometry: request.geometry,
                            },
                        )
                        .ok_or(Status::InternalError);
                }
            }
        }
        let Request {
            targets,
            depth,
            views,
            primitive,
            vertices,
            count,
            geometry,
        } = request;
        let contract = backend.first_draw.as_ref().ok_or(Status::Unimplemented)?;
        let topology = mapped(&contract.topologies, primitive).ok_or(Status::Unimplemented)?;
        if self.unsupported || (targets.is_empty() || targets.len() > 8) || views != [0, 0] {
            return Err(Status::Unimplemented);
        }
        let Some(Object::Texture { description, .. }) = instance.objects.get(targets[0]) else {
            return Err(Status::BadArgument);
        };
        let mut descriptions = Vec::new();
        for &target in targets {
            let Some(Object::Texture { description: d, .. }) = instance.objects.get(target) else {
                return Err(Status::BadArgument);
            };
            if d.width == 0
                || d.height == 0
                || d.width != description.width
                || d.height != description.height
            {
                return Err(Status::BadArgument);
            }
            if d.pool == 0
                || d.format != contract.rgba8
                || d.depth > 1
                || d.levels > 1
                || d.target != contract.target_2d
                || d.swizzle != contract.identity_swizzle
                || d.flags != 0
                || d.depth_stencil_mode != 0
            {
                return Err(Status::Unimplemented);
            }
            let bytes = d
                .width
                .checked_mul(d.height)
                .and_then(|n| n.checked_mul(4))
                .ok_or(Status::BadArgument)?;
            if descriptions
                .iter()
                .any(|other: &super::objects::TextureDescription| {
                    other.pool == d.pool
                        && ranges_overlap(other.pool_offset, bytes, d.pool_offset, bytes)
                })
            {
                return Err(Status::Unimplemented);
            }
            descriptions.push(d);
        }
        let streams = self.vertex_settings("VertexStreamState", &["Stride", "Divisor"])?;
        let attributes = self.vertex_settings("VertexAttribState", &["Format", "StreamIndex"])?;
        let mut pipeline_state =
            self.pipeline_state(contract.depth_raster, contract.cull_none, depth != 0)?;
        pipeline_state.color_count = targets.len() as u32;
        pipeline_state.colors = self.color_state(backend.blend_contract.as_ref(), targets.len())?;
        let depth_description = if depth != 0 {
            let extension = contract.depth_raster.ok_or(Status::Unimplemented)?;
            let Some(Object::Texture { description: d, .. }) = instance.objects.get(depth) else {
                return Err(Status::BadArgument);
            };
            if targets.contains(&depth)
                || d.width != description.width
                || d.height != description.height
            {
                return Err(Status::BadArgument);
            }
            if d.format != extension.d32s8
                || d.target != contract.target_2d
                || d.swizzle != contract.identity_swizzle
                || d.depth > 1
                || d.levels > 1
                || d.flags != 0
                || d.depth_stencil_mode != 0
            {
                return Err(Status::Unimplemented);
            }
            if descriptions.iter().any(|color| {
                color.pool == d.pool
                    && ranges_overlap(
                        color.pool_offset,
                        color.width.saturating_mul(color.height).saturating_mul(4),
                        d.pool_offset,
                        d.width.saturating_mul(d.height).saturating_mul(5),
                    )
            }) {
                return Err(Status::Unimplemented);
            }
            Some(d)
        } else {
            None
        };
        let mut attachment_ranges = [(0_u64, 0_u64, 0_u64); 9];
        for (at, d) in descriptions
            .iter()
            .chain(depth_description.iter())
            .enumerate()
        {
            let bytes = backend.texture_storage_size(d).map_or_else(
                || {
                    d.width
                        .saturating_mul(d.height)
                        .saturating_mul(if at < descriptions.len() { 4 } else { 5 })
                },
                |bytes| bytes as u64,
            );
            attachment_ranges[at] = (d.pool, d.pool_offset, bytes);
        }
        let attachment_ranges =
            &attachment_ranges[..descriptions.len() + usize::from(depth_description.is_some())];
        let context = backend.context().clone();
        let memory = backend.global_memory.as_mut().ok_or(Status::BadArgument)?;
        let mut input = VertexInput {
            bindings: Vec::new(),
            attributes: Vec::new(),
        };
        let mut buffers = Vec::new();
        let mut fetches = Vec::new();
        let mut independent = true;
        for attribute in attributes {
            let format = value(attribute, "Format")?;
            let format_kind =
                mapped(&contract.attribute_formats, format[0]).ok_or(Status::Unimplemented)?;
            let offset = u32::try_from(format[1]).map_err(|_| Status::BadArgument)?;
            let binding = u32::try_from(value(attribute, "StreamIndex")?[0])
                .map_err(|_| Status::BadArgument)?;
            let stream = streams.get(binding as usize).ok_or(Status::Unimplemented)?;
            let divisor = value(stream, "Divisor")?[0];
            if divisor > 1 || binding >= 32 {
                return Err(Status::Unimplemented);
            }
            if divisor == 1 {
                pipeline_state.instance_bindings |= 1 << binding;
            }
            let stride =
                u32::try_from(value(stream, "Stride")?[0]).map_err(|_| Status::BadArgument)?;
            let (bytes, alignment) = format_kind.layout();
            let (address, size) = *self
                .buffers
                .get(&u64::from(binding))
                .ok_or(Status::Unimplemented)?;
            let resolution = instance
                .objects
                .resolve_gpu_address(address)
                .map_err(|_| Status::BadArgument)?;
            // Every fetched element, including the arena binding address, must be aligned.
            let (buffer, at) = memory
                .image_region(
                    resolution.pool,
                    resolution.offset,
                    usize::try_from(size).map_err(|_| Status::BadArgument)?,
                )
                .ok_or(Status::BadArgument)?;
            if size == 0
                || size > resolution.remaining
                || !stride.is_multiple_of(alignment)
                || !(at + u64::from(offset)).is_multiple_of(u64::from(alignment))
            {
                return Err(Status::BadArgument);
            }
            if divisor == 1 {
                if vertex_end(
                    geometry.first_instance,
                    geometry.instances,
                    stride,
                    offset,
                    bytes,
                )
                .ok_or(Status::BadArgument)?
                    > size
                {
                    return Err(Status::BadArgument);
                }
            } else if let Vertices::Arrays { first } = vertices {
                if vertex_end(first, count, stride, offset, bytes).ok_or(Status::BadArgument)?
                    > size
                {
                    return Err(Status::BadArgument);
                }
            }
            independent &= !attachment_ranges.iter().any(|&(pool, offset, bytes)| {
                pool == resolution.pool && ranges_overlap(offset, bytes, resolution.offset, size)
            });
            if divisor == 0 {
                fetches.push((stride, offset, bytes, size));
            } else {
                independent = false;
            }
            input.attributes.push(VertexAttribute {
                binding,
                format: format_kind,
                offset,
            });
            if !input.bindings.iter().any(|b| b.binding == binding) {
                input.bindings.push(VertexBinding { binding, stride });
                buffers.push((binding, buffer, at));
            }
        }
        input.bindings.sort_unstable_by_key(|b| b.binding);
        buffers.sort_unstable_by_key(|b| b.0);
        let source = vertices;
        let vertices = match vertices {
            Vertices::Arrays { first } => DrawVertices::Arrays { first },
            Vertices::Elements {
                index_type,
                indices,
                base_vertex,
            } => {
                if contract.index_u16 == contract.index_u32 {
                    return Err(Status::Unimplemented);
                }
                let format = if index_type == contract.index_u16 {
                    IndexFormat::U16
                } else if index_type == contract.index_u32 {
                    IndexFormat::U32
                } else {
                    return Err(Status::Unimplemented);
                };
                // An empty indexed draw fetches neither indices nor vertices.
                if count == 0 {
                    return Ok(());
                }
                let index_resolution = instance
                    .objects
                    .resolve_gpu_address(indices)
                    .map_err(|_| Status::BadArgument)?;
                let length = u64::from(count) * format.width() as u64;
                independent &= !attachment_ranges.iter().any(|&(pool, offset, bytes)| {
                    pool == index_resolution.pool
                        && ranges_overlap(offset, bytes, index_resolution.offset, length)
                });
                if length > index_resolution.remaining {
                    return Err(Status::BadArgument);
                }
                let (index_buffer, index_offset) = memory
                    .buffer_region(
                        index_resolution.pool,
                        index_resolution.offset,
                        usize::try_from(length).map_err(|_| Status::BadArgument)?,
                    )
                    .ok_or(Status::BadArgument)?;
                if !index_offset.is_multiple_of(format.width() as u64) {
                    return Err(Status::BadArgument);
                }
                // Read canonical bytes, including earlier completed GPU writes in this submission.
                // Bounded chunks avoid allocating a count-sized host copy.
                let mut bytes = [0; 64 * 1024];
                let mut read = 0;
                while read < length {
                    let chunk = (length - read).min(bytes.len() as u64) as usize;
                    memory
                        .read_pool(
                            index_resolution.pool,
                            index_resolution.offset + read,
                            &mut bytes[..chunk],
                        )
                        .ok_or(Status::InternalError)?;
                    for &(stride, offset, attribute_bytes, size) in &fetches {
                        let end = indexed_vertex_end(
                            &bytes[..chunk],
                            format,
                            base_vertex,
                            stride,
                            offset,
                            attribute_bytes,
                        )
                        .ok_or(Status::BadArgument)?;
                        if end > size {
                            return Err(Status::BadArgument);
                        }
                    }
                    read += chunk as u64;
                }
                DrawVertices::Elements {
                    buffer: index_buffer,
                    offset: index_offset,
                    index_type: format.vulkan(),
                    base_vertex,
                }
            }
        };
        let viewport = Viewport::new(self.viewport, description.width, description.height)?;
        let scissor = rectangle(self.scissor, description.width, description.height)?;
        let translations = match instance
            .objects
            .get(self.program.ok_or(Status::Unimplemented)?)
        {
            Some(Object::Program {
                shader_translations,
                ..
            }) => shader_translations,
            _ => return Err(Status::BadArgument),
        };
        let blocking = backend.graphics.policy() == crate::gpu::graphics::PendingDrawPolicy::Block;
        let mut translation_keys = Vec::new();
        let mut stages = Vec::new();
        let mut surfaces = Vec::new();
        for translation in translations {
            let translation = match translation {
                ShaderTranslation::Deferred(code, context) => instance
                    .retry_cached_translation(&code, &context, blocking)
                    .ok_or(Status::Unimplemented)?,
                other => other,
            };
            match translation {
                ShaderTranslation::Spirv(words) => stages.push(Arc::new(
                    crate::gpu::graphics::Stage::checked(context.shader_features, words).map_err(
                        |error| {
                            backend.graphics.log_skipped_draw(error);
                            Status::Unimplemented
                        },
                    )?,
                )),
                ShaderTranslation::Cached(request) => match if blocking {
                    request.wait()
                } else {
                    request.poll()
                } {
                    crate::startup_cache::TranslationStatus::Ready(shader) => {
                        let stage =
                            backend
                                .graphics
                                .translated_stage(&shader)
                                .map_err(|error| {
                                    backend.graphics.log_skipped_draw(error);
                                    Status::Unimplemented
                                })?;
                        if shader.requires_subgroup_size_32 {
                            let index = match execution_model(&shader.words)
                                .map_err(|_| Status::Unimplemented)?
                            {
                                0 => 0,
                                4 => 1,
                                _ => return Err(Status::Unimplemented),
                            };
                            pipeline_state.subgroup_size_32[index] = true;
                        }
                        if !shader.surfaces.is_empty() {
                            let stage = match execution_model(&shader.words)
                                .map_err(|_| Status::Unimplemented)?
                            {
                                0 => UniformStage::Vertex,
                                4 => UniformStage::Fragment,
                                _ => return Err(Status::Unimplemented),
                            };
                            surfaces.extend(shader.surfaces.iter().map(|s| (stage, *s)));
                        }
                        translation_keys.push(shader.key.clone());
                        stages.push(stage);
                    }
                    crate::startup_cache::TranslationStatus::Queued
                    | crate::startup_cache::TranslationStatus::Compiling => {
                        backend
                            .graphics
                            .log_skipped_draw("shader translation pending");
                        return Ok(());
                    }
                    _ => return Err(Status::Unimplemented),
                },
                ShaderTranslation::Deferred(..) => {
                    backend.graphics.log_skipped_draw("translation queue full");
                    return Ok(());
                }
                ShaderTranslation::Error(error) => {
                    backend
                        .graphics
                        .log_skipped_draw(&format!("shader translation failed: {error}"));
                    return Err(Status::Unimplemented);
                }
            }
        }
        let limits = unsafe {
            context
                .instance
                .get_physical_device_properties(context.physical_device)
        }
        .limits;
        if description.width > u64::from(limits.max_framebuffer_width)
            || description.height > u64::from(limits.max_framebuffer_height)
            || viewport
                .size()
                .iter()
                .zip(limits.max_viewport_dimensions)
                .any(|(&size, limit)| size > limit)
            || limits.viewport_bounds_range[0] > 0.0
            || viewport
                .size()
                .iter()
                .any(|&size| size as f32 > limits.viewport_bounds_range[1])
        {
            return Err(Status::BadArgument);
        }
        if self.uniforms.keys().any(|&(stage, index)| {
            !backend
                .uniforms
                .bindings
                .iter()
                .any(|m| m.stage == stage && m.index == index)
        }) {
            backend
                .graphics
                .log_skipped_draw("recorded uniform has no contract mapping");
            return Err(Status::Unimplemented);
        }
        let storage = backend
            .graphics
            .uses_storage(backend.uniforms.storage_buffers);
        if let Some(cache) = instance.startup_cache.lock().unwrap().as_mut() {
            cache.schedule(&mut backend.graphics);
        }
        let recipe_input = input.clone();
        let pipeline = backend
            .graphics
            .request_stages(&stages, input, topology, pipeline_state, storage)
            .map_err(|error| {
                backend.graphics.log_skipped_draw(&error);
                Status::Unimplemented
            })?;
        let pending_banks;
        let banks = if let Some(pipeline) = &pipeline {
            &pipeline.banks
        } else {
            pending_banks = [UniformStage::Vertex, UniformStage::Fragment]
                .into_iter()
                .map(|stage| {
                    let words = stages
                        .iter()
                        .map(|stage| stage.words())
                        .find(|words| execution_model(words).ok() == Some(stage.model()))
                        .ok_or(Status::Unimplemented)?;
                    uniforms::banks(words, stage).map_err(|_| Status::Unimplemented)
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            &pending_banks
        };
        #[cfg(feature = "draw-metrics")]
        let uniform_span = crate::draw_metrics::Span::new(2);
        let mut uniform_buffers = Vec::new();
        for bank in banks {
            let stage = bank.stage;
            let mapping = backend
                .uniforms
                .bindings
                .iter()
                .find(|m| m.target == stage && m.bank == bank.bank)
                .ok_or_else(|| {
                    backend.graphics.log_skipped_draw(&format!(
                        "shader uniform bank has no contract mapping: stage={stage:?} bank={}",
                        bank.bank
                    ));
                    Status::Unimplemented
                })?;
            let &(address, size) = self
                .uniforms
                .get(&(mapping.stage, mapping.index))
                .ok_or_else(|| {
                    backend
                        .graphics
                        .log_skipped_draw("shader uniform bank is not bound");
                    Status::Unimplemented
                })?;
            let resolution = instance
                .objects
                .resolve_gpu_address(address)
                .map_err(|_| Status::BadArgument)?;
            if size == 0
                || size > BANK_SIZE
                || !size.is_multiple_of(16)
                || size > resolution.remaining
            {
                return Err(Status::BadArgument);
            }
            independent &= !attachment_ranges.iter().any(|&(pool, offset, bytes)| {
                pool == resolution.pool && ranges_overlap(offset, bytes, resolution.offset, size)
            });
            let info = if storage {
                memory.storage_buffer_info(resolution.pool, resolution.offset, size)
            } else {
                memory.uniform_buffer_info(resolution.pool, resolution.offset, size)
            }
            .ok_or(Status::BadArgument)?;
            uniform_buffers.push(info);
        }
        #[cfg(feature = "draw-metrics")]
        drop(uniform_span);
        let Some(pipeline) = pipeline else {
            backend
                .graphics
                .log_skipped_draw("pipeline compilation pending or queue full");
            return Ok(());
        };
        if translation_keys.len() == stages.len() {
            if let Some(cache) = instance.startup_cache.lock().unwrap().as_mut() {
                cache.remember(crate::startup_cache::PipelineRecipe {
                    keys: translation_keys,
                    input: recipe_input,
                    topology,
                    state: pipeline_state,
                    storage,
                });
            }
        }
        for (&target, d) in targets.iter().zip(&descriptions) {
            if !backend.ensure_texture(target, d, false) {
                return Err(Status::BadArgument);
            }
        }
        if let Some(description) = &depth_description {
            if !backend.ensure_depth_stencil(depth, description) {
                return Err(Status::BadArgument);
            }
        }
        let texture_contract = backend.texture_contract.clone();
        let mut texture_infos = Vec::new();
        let mut storage_images = Vec::new();
        if pipeline
            .textures
            .iter()
            .any(|b| b.ty == vk::DescriptorType::STORAGE_IMAGE)
        {
            // Surfaces load from and store to the arena around their own
            // draw; earlier draws land first.
            backend.finish_draws().ok_or(Status::InternalError)?;
        }
        for b in &pipeline.textures {
            if b.ty == vk::DescriptorType::STORAGE_IMAGE {
                let skip = |reason: &str| {
                    backend.graphics.log_skipped_draw(reason);
                    Status::Unimplemented
                };
                let surface = surfaces
                    .iter()
                    .find(|(stage, s)| {
                        *stage == b.stage && s.set == b.set && s.binding == b.binding
                    })
                    .ok_or_else(|| skip("storage image without a typed surface binding"))?
                    .1;
                let texture = self.bound_surface(
                    instance,
                    &texture_contract,
                    b.stage.into(),
                    &surface,
                    &skip,
                )?;
                let Some(Object::Texture { description: d, .. }) = instance.objects.get(texture)
                else {
                    return Err(Status::BadArgument);
                };
                if depth == texture || targets.contains(&texture) {
                    backend
                        .graphics
                        .log_skipped_draw("surface is also a render target");
                    return Err(Status::Unimplemented);
                }
                let c = backend.first_draw.as_ref().ok_or(Status::Unimplemented)?;
                if backend.has_image_contract() {
                    let resolved = backend.resolved_image(&d).ok_or(Status::Unimplemented)?;
                    if d.pool == 0 || b.image_kind != Some(resolved.descriptor.shape.kind) {
                        return Err(Status::Unimplemented);
                    }
                } else if b.image_kind != Some(crate::tiling::ImageKind::D2)
                    || d.pool == 0
                    || d.format != c.rgba8
                    || d.target != c.target_2d
                    || d.depth > 1
                    || d.levels > 1
                    || d.flags != 0
                {
                    return Err(Status::Unimplemented);
                }
                let view = backend
                    .storage_texture(texture, &d, surface.format)
                    .ok_or_else(|| {
                        backend
                            .graphics
                            .log_skipped_draw("surface image has no storage view");
                        Status::Unimplemented
                    })?;
                storage_images.push(texture);
                texture_infos.push(
                    vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::GENERAL),
                );
                continue;
            }
            let mapping = texture_contract
                .bindings
                .iter()
                .find(|m| {
                    m.target == b.stage
                        && m.set == b.set
                        && if b.ty == vk::DescriptorType::SAMPLED_IMAGE {
                            m.image == b.binding
                        } else {
                            m.sampler == b.binding
                        }
                })
                .ok_or_else(|| {
                    backend.graphics.log_skipped_draw(&format!(
                        "shader texture has no contract mapping: stage={:?} set={} binding={} type={}",
                        b.stage, b.set, b.binding, b.ty.as_raw()
                    ));
                    Status::Unimplemented
                })?;
            let source = (mapping.stage, mapping.index);
            let image = b.ty == vk::DescriptorType::SAMPLED_IMAGE;
            let reference = if image {
                self.textures.get(&source)
            } else {
                self.samplers.get(&source)
            }
            .ok_or_else(|| {
                backend
                    .graphics
                    .log_skipped_draw("shader texture or sampler is not bound");
                Status::Unimplemented
            })?;
            let id = match reference {
                TextureReference::Separate(handle) => {
                    u32::try_from(*handle).map_err(|_| Status::Unimplemented)?
                }
                TextureReference::Combined(handle) => {
                    let (_, texture, sampler) = texture_contract
                        .combined
                        .iter()
                        .find(|(h, _, _)| h == handle)
                        .ok_or(Status::Unimplemented)?;
                    if image {
                        *texture
                    } else {
                        *sampler
                    }
                }
            };
            if image {
                let pool = self.texture_pool.ok_or_else(|| {
                    backend
                        .graphics
                        .log_skipped_draw("texture pool is not bound");
                    Status::Unimplemented
                })?;
                let Some(Object::TexturePool { registered, .. }) = instance.objects.get(pool)
                else {
                    return Err(Status::BadArgument);
                };
                let &(texture, view) = registered.get(&id).ok_or_else(|| {
                    backend
                        .graphics
                        .log_skipped_draw("texture handle is not registered");
                    Status::Unimplemented
                })?;
                if view != 0 {
                    return Err(Status::Unimplemented);
                }
                let Some(Object::Texture { description: d, .. }) = instance.objects.get(texture)
                else {
                    return Err(Status::BadArgument);
                };
                if d.width == 0 || d.height == 0 {
                    return Err(Status::BadArgument);
                }
                let sample_bytes = backend
                    .texture_storage_size(&d)
                    .map_or(d.width.saturating_mul(d.height).saturating_mul(4), |n| {
                        n as u64
                    });
                if depth == texture
                    || depth_description.as_ref().is_some_and(|other| {
                        d.pool == other.pool
                            && ranges_overlap(
                                d.pool_offset,
                                sample_bytes,
                                other.pool_offset,
                                other.width.saturating_mul(other.height).saturating_mul(5),
                            )
                    })
                {
                    backend
                        .graphics
                        .log_skipped_draw("sampled texture description is unsupported");
                    return Err(Status::Unimplemented);
                }
                if targets.contains(&texture)
                    || descriptions.iter().any(|other| {
                        d.pool == other.pool
                            && ranges_overlap(
                                d.pool_offset,
                                sample_bytes,
                                other.pool_offset,
                                backend.texture_storage_size(other).map_or(
                                    other.width.saturating_mul(other.height).saturating_mul(4),
                                    |n| n as u64,
                                ),
                            )
                    })
                {
                    return Err(Status::Unimplemented);
                }
                let c = backend.first_draw.as_ref().ok_or(Status::Unimplemented)?;
                if backend.has_image_contract() {
                    let resolved = backend.resolved_image(&d).ok_or(Status::Unimplemented)?;
                    let shape = resolved.descriptor.shape;
                    if d.pool == 0
                        || (!backend.has_swizzle_contract() && d.swizzle != c.identity_swizzle)
                        || b.image_kind != Some(shape.kind)
                        || (shape.kind == crate::tiling::ImageKind::Cube
                            && b.arrayed != (shape.layers > 6))
                        || !matches!(
                            crate::gpu::image_layout::numeric_class(resolved.descriptor.format),
                            crate::gpu::image_layout::NumericClass::Float
                                | crate::gpu::image_layout::NumericClass::Depth
                        )
                    {
                        return Err(Status::Unimplemented);
                    }
                } else if b.image_kind != Some(crate::tiling::ImageKind::D2)
                    || b.arrayed
                    || d.pool == 0
                    || d.format != c.rgba8
                    || d.target != c.target_2d
                    || d.depth > 1
                    || d.levels > 1
                    || d.swizzle != c.identity_swizzle
                    || d.flags != 0
                    || d.depth_stencil_mode != 0
                {
                    return Err(Status::Unimplemented);
                }
                let view = backend
                    .sampled_texture(texture, &d)
                    .ok_or(Status::BadArgument)?;
                texture_infos.push(
                    vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                );
            } else {
                let pool = self.sampler_pool.ok_or_else(|| {
                    backend
                        .graphics
                        .log_skipped_draw("sampler pool is not bound");
                    Status::Unimplemented
                })?;
                let Some(Object::SamplerPool { registered, .. }) = instance.objects.get(pool)
                else {
                    return Err(Status::BadArgument);
                };
                let sampler = registered.get(&id).ok_or_else(|| {
                    backend
                        .graphics
                        .log_skipped_draw("sampler handle is not registered");
                    Status::Unimplemented
                })?;
                let Some(Object::Sampler(d)) = instance.objects.get(*sampler) else {
                    return Err(Status::BadArgument);
                };
                let key = crate::gpu::textures::SamplerKey::new(&d, &texture_contract).ok_or_else(
                    || {
                        backend
                            .graphics
                            .log_skipped_draw("sampler description is unsupported");
                        Status::Unimplemented
                    },
                )?;
                let sampler = backend
                    .sampler(pool, id, key)
                    .ok_or(Status::InternalError)?;
                texture_infos.push(vk::DescriptorImageInfo::default().sampler(sampler));
            }
        }
        let descriptors = backend
            .graphics
            .descriptors(&pipeline, &uniform_buffers, &texture_infos)
            .map_err(|_| Status::InternalError)?;
        let surfaces_bound = !storage_images.is_empty();
        let draw = Arc::new(Draw {
            descriptors,
            storage_images,
            buffers,
            viewport: viewport.vulkan(),
            scissor: vk::Rect2D::default().extent(vk::Extent2D {
                width: scissor[0],
                height: scissor[1],
            }),
        });
        if independent
            && !surfaces_bound
            && pipeline.batch_safe
            && geometry == crate::gpu::operations::Geometry::default()
        {
            self.prepared = Some(Prepared {
                targets: targets.to_vec(),
                depth,
                primitive,
                draw: draw.clone(),
                pipeline: pipeline.clone(),
                fetches,
                source,
                count,
                vertices,
            });
        }
        backend
            .queue_draw(
                targets,
                depth_description.map(|_| depth),
                crate::gpu::graphics::PendingDraw {
                    pipeline,
                    state: draw,
                    vertices,
                    count,
                    geometry,
                },
            )
            .ok_or(Status::InternalError)?;
        if surfaces_bound {
            // A later draw may sample or load what this one writes.
            backend.finish_draws().ok_or(Status::InternalError)?;
        }
        Ok(())
    }

    /// The guest texture a bound surface names: the host contract maps the
    /// module's bound slot to the recorded image binding, and the texture
    /// pool resolves that handle. `skip` says why a surface cannot bind.
    pub(super) fn bound_surface(
        &self,
        instance: &Instance,
        contract: &crate::gpu::textures::TextureContract,
        target: crate::gpu::textures::SurfaceStage,
        surface: &crate::surfaces::SurfaceBinding,
        skip: &dyn Fn(&str) -> Status,
    ) -> Result<u64, Status> {
        let crate::surfaces::SurfaceSource::Bound { slot } = surface.source else {
            return Err(skip("bindless surface"));
        };
        let mapping = contract
            .surfaces
            .iter()
            .find(|m| m.target == target && m.slot == slot)
            .ok_or_else(|| skip("surface slot without a host mapping"))?;
        let handle = *self
            .images
            .get(&(mapping.stage, mapping.index))
            .ok_or_else(|| skip("surface slot without a bound image"))?;
        let id = u32::try_from(handle).map_err(|_| Status::Unimplemented)?;
        let pool = self.texture_pool.ok_or(Status::Unimplemented)?;
        let Some(Object::TexturePool { registered, .. }) = instance.objects.get(pool) else {
            return Err(Status::BadArgument);
        };
        let &(texture, view) = registered.get(&id).ok_or(Status::Unimplemented)?;
        if view != 0 {
            return Err(Status::Unimplemented);
        }
        Ok(texture)
    }

    fn color_state(
        &self,
        contract: Option<&crate::gpu::graphics::BlendContract>,
        count: usize,
    ) -> Result<[crate::gpu::graphics::ColorAttachmentState; 8], Status> {
        let mut colors = [crate::gpu::graphics::ColorAttachmentState::default(); 8];
        let settings = self.settings("ColorState", &["BlendEnable"])?;
        let mut known = [false; 8];
        for &(_, args) in settings {
            let at = usize::try_from(args[0]).map_err(|_| Status::BadArgument)?;
            let c = colors.get_mut(at).ok_or(Status::Unimplemented)?;
            c.enable = boolean(args[1])?;
            known[at] = true;
        }
        if !known[..count].iter().all(|k| *k) {
            return Err(Status::Unimplemented);
        }
        for (at, color) in colors.iter_mut().enumerate() {
            if !color.enable {
                continue;
            }
            let contract = contract.ok_or(Status::Unimplemented)?;
            let settings = self
                .blends
                .get(&(at as u64))
                .and_then(Option::as_ref)
                .ok_or(Status::Unimplemented)?;
            if settings
                .iter()
                .any(|(name, _)| !matches!(*name, "BlendTarget" | "BlendFunc" | "BlendEquation"))
            {
                return Err(Status::Unimplemented);
            }
            let f = value(settings, "BlendFunc")?;
            let e = value(settings, "BlendEquation")?;
            for i in 0..4 {
                color.factors[i] = mapped(&contract.factors, f[contract.function_order[i]])
                    .ok_or(Status::Unimplemented)?
                    .as_raw();
            }
            for i in 0..2 {
                color.operations[i] = mapped(&contract.operations, e[contract.equation_order[i]])
                    .ok_or(Status::Unimplemented)?
                    .as_raw();
            }
        }
        if self.states.contains_key("ChannelMaskState") {
            let c = contract.ok_or(Status::Unimplemented)?;
            for &(_, args) in self.settings("ChannelMaskState", &["ChannelMask"])? {
                let at = usize::try_from(args[0]).map_err(|_| Status::BadArgument)?;
                let color = colors.get_mut(at).ok_or(Status::Unimplemented)?;
                color.write_mask = 0;
                for i in 0..4 {
                    if boolean(args[1 + c.channel_order[i]])? {
                        color.write_mask |= 1 << i;
                    }
                }
            }
        }
        Ok(colors)
    }

    fn vertex_settings(&self, kind: &str, allowed: &[&str]) -> Result<&Vec<Settings>, Status> {
        let settings = self
            .vertex_states
            .get(kind)
            .and_then(Option::as_ref)
            .ok_or(Status::Unimplemented)?;
        if settings
            .iter()
            .flatten()
            .any(|(name, _)| !allowed.contains(name))
        {
            return Err(Status::Unimplemented);
        }
        Ok(settings)
    }
}

fn ranges_overlap(a: u64, an: u64, b: u64, bn: u64) -> bool {
    a.checked_add(an).is_none_or(|end| b < end) && b.checked_add(bn).is_none_or(|end| a < end)
}

fn boolean(v: u64) -> Result<bool, Status> {
    match v {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(Status::Unimplemented),
    }
}
fn token(v: u64, tokens: &[u64]) -> Result<i32, Status> {
    let mut matches = tokens.iter().enumerate().filter(|(_, t)| **t == v);
    let (index, _) = matches.next().ok_or(Status::Unimplemented)?;
    if matches.next().is_some() {
        return Err(Status::BadArgument);
    }
    Ok(index as i32)
}
fn face_mask(v: u64, extension: &DepthRasterContract) -> Result<u32, Status> {
    Ok([1, 2, 3][token(v, &extension.faces)? as usize])
}
fn byte(v: u64) -> Result<u32, Status> {
    u8::try_from(v)
        .map(u32::from)
        .map_err(|_| Status::BadArgument)
}

fn value(settings: &Settings, name: &str) -> Result<[u64; 6], Status> {
    settings
        .iter()
        .rev()
        .find(|(key, _)| *key == name)
        .map(|(_, value)| *value)
        .ok_or(Status::Unimplemented)
}

/// Vertical direction of a recorded viewport. Guest Y negation becomes a
/// negative host viewport height whose origin row is the region's far edge.
/// A recorded height is a 32-bit signed integer; a negative one selects the
/// mirrored form over the same rows.
enum Rows {
    Upright(u32),
    Mirrored(u32),
}

/// A recorded viewport covering columns and rows from zero.
struct Viewport {
    width: u32,
    rows: Rows,
}

impl Viewport {
    fn new(recorded: Option<[u64; 5]>, width: u64, height: u64) -> Result<Self, Status> {
        let [_, x, y, w, h] = recorded.ok_or(Status::Unimplemented)?;
        let rows = match h as u32 as i32 {
            h @ 1.. if y == 0 => Rows::Upright(h.unsigned_abs()),
            h @ ..0 if y == u64::from(h.unsigned_abs()) => Rows::Mirrored(h.unsigned_abs()),
            0 => return Err(Status::BadArgument),
            _ => return Err(Status::Unimplemented),
        };
        if x != 0 {
            return Err(Status::Unimplemented);
        }
        let (Rows::Upright(span) | Rows::Mirrored(span)) = rows;
        if w == 0 || w > width || u64::from(span) > height {
            return Err(Status::BadArgument);
        }
        Ok(Self {
            width: u32::try_from(w).map_err(|_| Status::BadArgument)?,
            rows,
        })
    }

    fn size(&self) -> [u32; 2] {
        let (Rows::Upright(span) | Rows::Mirrored(span)) = self.rows;
        [self.width, span]
    }

    fn vulkan(&self) -> vk::Viewport {
        let (y, height) = match self.rows {
            Rows::Upright(span) => (0.0, span as f32),
            Rows::Mirrored(span) => (span as f32, -(span as f32)),
        };
        vk::Viewport {
            x: 0.0,
            y,
            width: self.width as f32,
            height,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}

fn rectangle(recorded: Option<[u64; 5]>, width: u64, height: u64) -> Result<[u32; 2], Status> {
    let [_, x, y, w, h] = recorded.ok_or(Status::Unimplemented)?;
    if x != 0 || y != 0 {
        return Err(Status::Unimplemented);
    }
    if w == 0 || h == 0 || w > width || h > height {
        return Err(Status::BadArgument);
    }
    Ok([
        u32::try_from(w).map_err(|_| Status::BadArgument)?,
        u32::try_from(h).map_err(|_| Status::BadArgument)?,
    ])
}

fn vertex_end(first: u32, count: u32, stride: u32, offset: u32, bytes: u32) -> Option<u64> {
    if count == 0 {
        return Some(0);
    }
    let last = first.checked_add(count - 1)?;
    u64::from(last)
        .checked_mul(u64::from(stride))?
        .checked_add(u64::from(offset))?
        .checked_add(u64::from(bytes))
}

fn indexed_vertex_end(
    bytes: &[u8],
    format: IndexFormat,
    base: i32,
    stride: u32,
    offset: u32,
    attribute_bytes: u32,
) -> Option<u64> {
    let mut end = 0;
    for bytes in bytes.chunks_exact(format.width()) {
        let index = match format {
            IndexFormat::U16 => u32::from(u16::from_le_bytes(bytes.try_into().ok()?)),
            IndexFormat::U32 => u32::from_le_bytes(bytes.try_into().ok()?),
        };
        let vertex = u32::try_from(i64::from(index) + i64::from(base)).ok()?;
        end = end.max(vertex_end(vertex, 1, stride, offset, attribute_bytes)?);
    }
    Some(end)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn contract() -> DepthRasterContract {
        DepthRasterContract {
            d32s8: 90,
            compare: [100, 101, 102, 103, 104, 105, 106, 107],
            stencil_ops: [200, 201, 202, 203, 204, 205, 206, 207],
            faces: [301, 302, 303],
            cull: [401, 402, 403],
            polygon: [500, 501, 502],
            clockwise: false,
            polygon_offset: true,
        }
    }
    fn state() -> State {
        let mut state = State::default();
        state.states.insert(
            "DepthStencilState",
            Some(vec![
                ("DepthTestEnable", [0; 6]),
                ("DepthWriteEnable", [0; 6]),
                ("StencilTestEnable", [1, 0, 0, 0, 0, 0]),
                ("StencilFunc", [303, 107, 4, 15, 0, 0]),
                ("StencilOp", [303, 200, 201, 202, 0, 0]),
                ("StencilFunc", [301, 102, 7, 31, 0, 0]),
            ]),
        );
        state.states.insert(
            "PolygonState",
            Some(vec![
                ("CullFace", [0; 6]),
                ("PolygonMode", [500, 0, 0, 0, 0, 0]),
            ]),
        );
        state.stencil_commands = vec![
            ("StencilMask", 303, 255),
            ("StencilMask", 302, 3),
            ("StencilRef", 302, 9),
            ("StencilValueMask", 301, 127),
        ];
        state
    }
    #[test]
    fn face_setters_and_command_masks_preserve_the_other_face() {
        let s = state().pipeline_state(Some(contract()), 0, true).unwrap();
        assert_eq!(
            (
                s.stencil[0].compare,
                s.stencil[0].reference,
                s.stencil[0].mask,
                s.stencil[0].write_mask
            ),
            (2, 7, 127, 255)
        );
        assert_eq!(
            (
                s.stencil[1].compare,
                s.stencil[1].reference,
                s.stencil[1].mask,
                s.stencil[1].write_mask
            ),
            (7, 9, 15, 3)
        );
        assert_eq!(
            (
                s.stencil[0].fail,
                s.stencil[0].depth_fail,
                s.stencil[0].pass
            ),
            (0, 1, 2)
        );
    }
    #[test]
    fn rejects_unknown_tokens_missing_masks_and_nonfinite_bias() {
        let mut s = state();
        assert_eq!(s.pipeline_state(None, 0, true), Err(Status::Unimplemented));
        assert_eq!(
            s.pipeline_state(Some(contract()), 0, false),
            Err(Status::Unimplemented)
        );
        s.stencil_commands.clear();
        assert_eq!(
            s.pipeline_state(Some(contract()), 0, true),
            Err(Status::Unimplemented)
        );
        s.stencil_commands.push(("StencilMask", 303, 256));
        assert_eq!(
            s.pipeline_state(Some(contract()), 0, true),
            Err(Status::BadArgument)
        );
        s.stencil_commands = vec![("StencilMask", 303, 255)];
        s.bias = Some([u64::from(f32::NAN.to_bits()), 0, 0]);
        assert_eq!(
            s.pipeline_state(Some(contract()), 0, true),
            Err(Status::BadArgument)
        );
        s.bias = Some([0, u64::from(2.0_f32.to_bits()), 0]);
        let parsed = s.pipeline_state(Some(contract()), 0, true).unwrap();
        assert_eq!(parsed.bias, [0, 2.0_f32.to_bits(), 0]);
        let mut c = contract();
        c.polygon_offset = false;
        assert_eq!(
            s.pipeline_state(Some(c), 0, true),
            Err(Status::Unimplemented)
        );
        assert_eq!(token(999, &c.compare), Err(Status::Unimplemented));
        assert_eq!(token(1, &[1, 1]), Err(Status::BadArgument));
    }
    #[test]
    fn vertex_bounds_include_first_offset_and_last_attribute() {
        assert_eq!(vertex_end(2, 3, 32, 8, 16), Some(152));
        assert_eq!(vertex_end(2, 3, 32, 8, 4), Some(140));
        assert_eq!(vertex_end(u32::MAX, 2, 16, 0, 16), None);
        assert_eq!(vertex_end(u32::MAX, 0, 16, 0, 16), Some(0));
        assert_eq!(vertex_end(10, 3, 0, 2, 4), Some(6));
    }

    #[test]
    fn indexed_bounds_decode_width_and_signed_base_without_wrapping() {
        assert_eq!(
            indexed_vertex_end(&[3, 0, 1, 0, 2, 0], IndexFormat::U16, 1, 32, 8, 16),
            Some(152)
        );
        let bytes: Vec<_> = [4_u32, 2, 3]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert_eq!(
            indexed_vertex_end(&bytes, IndexFormat::U32, -1, 32, 8, 16),
            Some(120)
        );
        assert_eq!(
            indexed_vertex_end(&bytes, IndexFormat::U32, -3, 32, 8, 16),
            None
        );
        assert_eq!(
            indexed_vertex_end(&u32::MAX.to_le_bytes(), IndexFormat::U32, 1, 32, 8, 16),
            None
        );
        assert_eq!(
            indexed_vertex_end(&65536_u32.to_le_bytes(), IndexFormat::U32, 0, 16, 0, 16),
            Some(1048592)
        );
        assert_eq!(
            indexed_vertex_end(&u16::MAX.to_le_bytes(), IndexFormat::U16, -65535, 16, 0, 16),
            Some(16)
        );
    }
}
