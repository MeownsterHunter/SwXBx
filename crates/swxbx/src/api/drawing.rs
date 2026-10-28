//! Bounded execution of recorded draws. Evidence: signatures 0009, the public contract, 0028, 0029.
use super::misuse::{listed, value, StreamFault, ATTRIBUTE_SETTINGS, STREAM_SETTINGS};
use super::objects::{Object, RecordedCommand, ShaderTranslation, StateCommand, StateSettings};
use crate::keymap::KeyMap;
use crate::{
    gpu::{
        graphics::{
            mapped, DepthRasterContract, Draw, DrawPipelineState, DrawVertices, VertexAttribute,
            VertexBinding, VertexInput,
        },
        uniforms::{self, GraphicsStage, BANK_SIZE},
        Backend,
    },
    instance::InstanceState as Instance,
    Status,
};
use ash::vk;
use std::sync::Arc;

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
    /// The index value in one little-endian element of this width.
    fn read(self, bytes: &[u8]) -> u32 {
        match self {
            Self::U16 => u32::from(u16::from_le_bytes([bytes[0], bytes[1]])),
            Self::U32 => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
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
pub(super) enum TextureReference {
    Separate(u64),
    Combined(u64),
}

#[derive(Default)]
pub(super) struct State {
    program: Option<u64>,
    buffers: KeyMap<u64, (u64, u64)>,
    vertex_states: KeyMap<&'static str, Option<Vec<Settings>>>,
    uniforms: KeyMap<(u64, u64), (u64, u64)>,
    texture_pool: Option<u64>,
    sampler_pool: Option<u64>,
    textures: KeyMap<(u64, u64), TextureReference>,
    samplers: KeyMap<(u64, u64), TextureReference>,
    images: KeyMap<(u64, u64), u64>,
    blends: KeyMap<u64, Option<Settings>>,
    states: KeyMap<&'static str, Option<Settings>>,
    viewport: Option<[u64; 5]>,
    scissor: Option<[u64; 5]>,
    unsupported: bool,
    stencil_commands: Vec<(&'static str, u64, u64)>,
    bias: Option<[u64; 3]>,
    prepared: Option<Prepared>,
    /// Reused host copy of an indexed draw's indices for the bounds check.
    index_bytes: Vec<u8>,
}

struct Prepared {
    targets: Vec<u64>,
    depth: u64,
    primitive: u32,
    draw: Arc<Draw>,
    pipeline: Arc<crate::gpu::graphics::GraphicsPipeline>,
    fetches: Vec<Fetch>,
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
                // A zero range unbinds the stream.
                if size == 0 {
                    self.buffers.remove(&stream);
                } else {
                    self.buffers.insert(stream, (address, size));
                }
            }
            RecordedCommand::State(StateCommand::BindStates {
                kind,
                count,
                first_settings,
                experiment_settings,
                ..
            }) => {
                self.vertex_states
                    .insert(kind, listed(count, first_settings, experiment_settings));
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

    /// The bound state of `kind`; the error names it when it is not bound or
    /// sets something swxbx does not draw with. Source: the public contract.
    fn settings(&self, kind: &str, allowed: &[&str]) -> Result<&Settings, Refusal> {
        let settings = self
            .states
            .get(kind)
            .and_then(Option::as_ref)
            .ok_or_else(|| Refusal::unimplemented(format!("no {kind} is bound")))?;
        if let Some((name, _)) = settings.iter().find(|(name, _)| !allowed.contains(name)) {
            return Err(Refusal::unimplemented(format!(
                "the bound {kind} sets {name}, which swxbx does not draw with"
            )));
        }
        Ok(settings)
    }

    fn pipeline_state(
        &self,
        extension: Option<&DepthRasterContract>,
        cull_none: u64,
        attachment: bool,
    ) -> Result<DrawPipelineState, Refusal> {
        const KIND: &str = "DepthStencilState";
        let settings = self.settings(
            KIND,
            &[
                "DepthTestEnable",
                "DepthWriteEnable",
                "StencilTestEnable",
                "DepthFunc",
                "StencilFunc",
                "StencilOp",
            ],
        )?;
        let enabled =
            |name: &str| boolean(setting(settings, name, KIND)?[0], &format!("{KIND} {name}"));
        let mut state = DrawPipelineState {
            attachment,
            depth_test: enabled("DepthTestEnable")?,
            depth_write: enabled("DepthWriteEnable")?,
            stencil_test: enabled("StencilTestEnable")?,
            ..Default::default()
        };
        let polygon = self.settings("PolygonState", &["CullFace", "PolygonMode"])?;
        let cull = setting(polygon, "CullFace", "PolygonState")?[0];
        let Some(extension) = extension else {
            let asked = if state.attachment {
                Some("a depth target".to_string())
            } else if state.depth_test || state.depth_write || state.stencil_test {
                Some("depth or stencil testing or depth writes".to_string())
            } else if let Some((name, _)) = settings
                .iter()
                .find(|(name, _)| matches!(*name, "DepthFunc" | "StencilFunc" | "StencilOp"))
            {
                Some(format!("{KIND} {name}"))
            } else if polygon.iter().any(|(name, _)| *name == "PolygonMode") {
                Some("a polygon mode".to_string())
            } else if cull != cull_none {
                Some(format!("cull face token {cull}"))
            } else if let Some((name, _, _)) = self.stencil_commands.first() {
                Some(format!("the stencil setting {name}"))
            } else if self.bias.is_some() {
                Some("a polygon offset".to_string())
            } else {
                None
            };
            return match asked {
                Some(asked) => Err(Refusal::unimplemented(format!(
                    "the draw uses {asked}, but no depth and raster contract is installed"
                ))),
                None => Ok(state),
            };
        };
        if (state.depth_test || state.depth_write || state.stencil_test) && !attachment {
            return Err(Refusal::unimplemented(
                "depth or stencil testing or depth writes are enabled, but the draw has no \
                 depth target",
            ));
        }
        state.clockwise = extension.clockwise;
        state.cull = if cull == cull_none {
            0
        } else {
            [1, 2, 3][token(cull, &extension.cull, "cull face")? as usize]
        };
        if let Ok(mode) = value(polygon, "PolygonMode") {
            state.polygon = token(mode[0], &extension.polygon, "polygon mode")?;
        }
        if state.depth_test {
            state.depth_compare = token(
                setting(settings, "DepthFunc", KIND)?[0],
                &extension.compare,
                "depth function",
            )?;
        }
        // Replay face-specific setters so a later front-only setter preserves the back face.
        let mut functions = [false; 2];
        let mut ops = [false; 2];
        for &(name, args) in settings {
            if !matches!(name, "StencilFunc" | "StencilOp") {
                continue;
            }
            let faces = face_mask(args[0], extension)?;
            for i in 0..2 {
                if faces & (1 << i) == 0 {
                    continue;
                }
                if name == "StencilFunc" {
                    state.stencil[i].compare =
                        token(args[1], &extension.compare, "stencil function")?;
                    state.stencil[i].reference = byte(args[2], "stencil reference")?;
                    state.stencil[i].mask = byte(args[3], "stencil value mask")?;
                    functions[i] = true;
                } else {
                    // Hypothesis: fail, depth fail, pass.
                    state.stencil[i].fail =
                        token(args[1], &extension.stencil_ops, "stencil fail operation")?;
                    state.stencil[i].depth_fail = token(
                        args[2],
                        &extension.stencil_ops,
                        "stencil depth-fail operation",
                    )?;
                    state.stencil[i].pass =
                        token(args[3], &extension.stencil_ops, "stencil pass operation")?;
                    ops[i] = true;
                }
            }
        }
        let mut masks = [false; 2];
        for &(setting, selector, v) in &self.stencil_commands {
            let faces = face_mask(selector, extension)?;
            let v = byte(v, setting)?;
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
                    _ => {
                        return Err(Refusal::unimplemented(format!(
                            "the stencil setting {setting} is not one swxbx draws with"
                        )))
                    }
                }
            }
        }
        if state.stencil_test {
            if let Some(i) = (0..2).find(|&i| !(functions[i] && ops[i] && masks[i])) {
                let face = ["front", "back"][i];
                let unset = [
                    (!functions[i]).then_some("function"),
                    (!ops[i]).then_some("operations"),
                    (!masks[i]).then_some("write mask"),
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(", ");
                return Err(Refusal::unimplemented(format!(
                    "stencil testing is enabled, but the {face} face's stencil {unset} \
                     was never set"
                )));
            }
        }
        if let Some(bias) = self.bias {
            if !extension.polygon_offset {
                return Err(Refusal::unimplemented(
                    "a polygon offset is set, but the depth and raster contract does not \
                     allow one",
                ));
            }
            for (i, bits) in bias.into_iter().enumerate() {
                let Ok(bits) = u32::try_from(bits) else {
                    return Err(Refusal::bad(format!(
                        "polygon offset value {i} is {bits:#x}, wider than a 32-bit float"
                    )));
                };
                let v = f32::from_bits(bits);
                if !v.is_finite() {
                    return Err(Refusal::bad(format!(
                        "polygon offset value {i} is {v}, not finite"
                    )));
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
                        for fetch in &prepared.fetches {
                            fetch
                                .consecutive("vertices", first, request.count)
                                .map_err(|reason| {
                                    backend.graphics.refuse_draw(Status::BadArgument, reason)
                                })?;
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
                        .ok_or_else(|| {
                            backend.graphics.refuse_draw(
                                Status::InternalError,
                                "the device could not record this draw",
                            )
                        });
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
        // Each refusal below names the fact that stopped the draw.
        // Source: the public contract.
        let stop = |status: Status, reason: String| backend.graphics.refuse_draw(status, reason);
        let Some(contract) = backend.first_draw.as_ref() else {
            return Err(stop(
                Status::Unimplemented,
                "no first-draw contract is installed, so swxbx cannot interpret draws".into(),
            ));
        };
        let topology = mapped(&contract.topologies, primitive).ok_or_else(|| {
            stop(
                Status::Unimplemented,
                format!(
                    "primitive token {primitive} is not mapped to exactly one topology by the \
                     first-draw contract"
                ),
            )
        })?;
        if self.unsupported {
            return Err(stop(
                Status::Unimplemented,
                "this command buffer set state swxbx cannot draw with: a blend state without \
                 a target, a depth range other than 0 to 1, or a state command draws do not read"
                    .into(),
            ));
        }
        if targets.is_empty() || targets.len() > 8 {
            return Err(stop(
                Status::Unimplemented,
                format!(
                    "the draw has {} color targets; swxbx draws to 1 to 8",
                    targets.len()
                ),
            ));
        }
        if views != [0, 0] {
            return Err(stop(
                Status::Unimplemented,
                format!(
                    "the render targets were set with texture views ({:#x} color, {:#x} depth); \
                     swxbx draws only to whole textures",
                    views[0], views[1]
                ),
            ));
        }
        let mut descriptions: Vec<super::objects::TextureDescription> = Vec::new();
        let mut color_formats = [vk::Format::R8G8B8A8_UNORM; 8];
        let mut color_bytes = [0_u64; 8];
        for (at, &target) in targets.iter().enumerate() {
            let Some(d) = instance.objects.texture_description(target) else {
                return Err(stop(
                    Status::BadArgument,
                    format!("color target {at} is texture {target}, which is not a texture"),
                ));
            };
            if d.width == 0 || d.height == 0 {
                return Err(stop(
                    Status::BadArgument,
                    format!(
                        "color target {at}, texture {target}, is {}x{} texels",
                        d.width, d.height
                    ),
                ));
            }
            if let Some(first) = descriptions.first() {
                if d.width != first.width || d.height != first.height {
                    return Err(stop(
                        Status::BadArgument,
                        format!(
                            "color target {at}, texture {target}, is {}x{} texels but color \
                             target 0 is {}x{}",
                            d.width, d.height, first.width, first.height
                        ),
                    ));
                }
            }
            // A 3D image bound whole is drawn as an array of its depth
            // slices; the draw's layer output selects the slice
            // (the reference host Graphics.Vulkan TextureView.cs, FramebufferParams.cs).
            let volume = backend
                .texture_image(target, &d)
                .is_some_and(|r| r.descriptor.shape.kind == crate::tiling::ImageKind::D3);
            let unsupported = if d.pool == 0 {
                Some("it has no memory pool".to_string())
            } else if d.levels > 1 {
                Some(format!(
                    "it has {} mip levels; swxbx draws to one level",
                    d.levels
                ))
            } else if !volume && d.target != contract.target_2d {
                Some(format!(
                    "its target token {} is not the contract's 2D target {}, and the image \
                     contract does not resolve it to a 3D image",
                    d.target, contract.target_2d
                ))
            } else if !volume && d.depth > 1 {
                Some(format!(
                    "it has depth {} but is a 2D image; swxbx draws to one 2D level",
                    d.depth
                ))
            } else if volume && d.depth > u64::from(backend.context().limits.max_framebuffer_layers)
            {
                Some(format!(
                    "it is a 3D image of {} slices, and this device draws to at most {} layers",
                    d.depth,
                    backend.context().limits.max_framebuffer_layers
                ))
            } else if d.swizzle != contract.identity_swizzle {
                Some(format!(
                    "its swizzle {:?} is not the identity swizzle {:?}",
                    d.swizzle, contract.identity_swizzle
                ))
            } else if d.flags != 0 {
                Some(format!("it has builder flags {:#x}", d.flags))
            } else if d.depth_stencil_mode != 0 {
                Some(format!(
                    "it has depth-stencil mode {}",
                    d.depth_stencil_mode
                ))
            } else {
                None
            };
            let format = unsupported
                .map_or_else(|| target_format(backend, contract, target, &d), Err)
                .map_err(|why| {
                    stop(
                        Status::Unimplemented,
                        format!("color target {at}, texture {target}, cannot be drawn to: {why}"),
                    )
                })?;
            let texel =
                u64::from(crate::gpu::formats::color_target(format).map_err(|why| {
                    stop(Status::Unimplemented, format!("color target {at}: {why}"))
                })?);
            let Some(bytes) = d
                .width
                .checked_mul(d.height)
                .and_then(|n| n.checked_mul(d.depth.max(1)))
                .and_then(|n| n.checked_mul(texel))
            else {
                return Err(stop(
                    Status::BadArgument,
                    format!(
                        "color target {at}, texture {target}, is {}x{} texels, too large to address",
                        d.width, d.height
                    ),
                ));
            };
            if let Some(other) = descriptions
                .iter()
                .zip(color_bytes)
                .position(|(other, size)| {
                    other.pool == d.pool
                        && ranges_overlap(other.pool_offset, size, d.pool_offset, bytes)
                })
            {
                return Err(stop(
                    Status::Unimplemented,
                    format!(
                        "color targets {other} and {at} (textures {} and {target}) share memory \
                         in pool {}",
                        targets[other], d.pool
                    ),
                ));
            }
            color_formats[at] = format;
            color_bytes[at] = bytes;
            descriptions.push(d);
        }
        let description = descriptions[0].clone();
        let streams = self
            .vertex_settings("VertexStreamState", "vertex stream", STREAM_SETTINGS)
            .map_err(|reason| stop(Status::Unimplemented, reason))?;
        let attributes = self
            .vertex_settings("VertexAttribState", "vertex attribute", ATTRIBUTE_SETTINGS)
            .map_err(|reason| stop(Status::Unimplemented, reason))?;
        let refused = |Refusal(status, reason)| stop(status, reason);
        let mut pipeline_state = self
            .pipeline_state(
                contract.depth_raster.as_ref(),
                contract.cull_none,
                depth != 0,
            )
            .map_err(refused)?;
        pipeline_state.color_count = targets.len() as u32;
        for (state, format) in pipeline_state.color_formats.iter_mut().zip(color_formats) {
            *state = format.as_raw();
        }
        pipeline_state.colors = self
            .color_state(backend.blend_contract.as_ref(), targets.len())
            .map_err(refused)?;
        let depth_description = if depth != 0 {
            if contract.depth_raster.is_none() {
                return Err(stop(
                    Status::Unimplemented,
                    format!(
                        "the draw has depth target texture {depth}, but the first-draw contract \
                         has no depth and raster interpretation"
                    ),
                ));
            }
            let Some(d) = instance.objects.texture_description(depth) else {
                return Err(stop(
                    Status::BadArgument,
                    format!("the depth target is texture {depth}, which is not a texture"),
                ));
            };
            if targets.contains(&depth) {
                return Err(stop(
                    Status::BadArgument,
                    format!("texture {depth} is both a color target and the depth target"),
                ));
            }
            if d.width != description.width || d.height != description.height {
                return Err(stop(
                    Status::BadArgument,
                    format!(
                        "depth target texture {depth} is {}x{} texels but the color targets are \
                         {}x{}",
                        d.width, d.height, description.width, description.height
                    ),
                ));
            }
            let unsupported = if d.target != contract.target_2d {
                Some(format!(
                    "its target token {} is not the contract's 2D target {}",
                    d.target, contract.target_2d
                ))
            } else if d.swizzle != contract.identity_swizzle {
                Some(format!(
                    "its swizzle {:?} is not the identity swizzle {:?}",
                    d.swizzle, contract.identity_swizzle
                ))
            } else if d.depth > 1 || d.levels > 1 {
                Some(format!(
                    "it has depth {} and {} mip levels; swxbx draws to one 2D level",
                    d.depth, d.levels
                ))
            } else if d.flags != 0 {
                Some(format!("it has builder flags {:#x}", d.flags))
            } else if d.depth_stencil_mode != 0 {
                Some(format!(
                    "it has depth-stencil mode {}",
                    d.depth_stencil_mode
                ))
            } else {
                None
            };
            let (image, planes) = unsupported
                .map_or_else(
                    || depth_format(backend, depth, &d, pipeline_state.stencil_test),
                    Err,
                )
                .map_err(|why| {
                    stop(
                        Status::Unimplemented,
                        format!("depth target texture {depth} cannot be drawn to: {why}"),
                    )
                })?;
            if image == crate::gpu::formats::DepthImage::D24S8AsD32S8
                && pipeline_state.bias != [0; 3]
            {
                return Err(stop(
                    Status::Unimplemented,
                    format!(
                        "depth target texture {depth} is D24_UNORM_S8_UINT, which this device \
                         draws in a D32_SFLOAT_S8_UINT image, and the draw sets a polygon \
                         offset, which moves depth by a different amount in that format"
                    ),
                ));
            }
            pipeline_state.depth_format = image.format().as_raw();
            let depth_bytes = planes.size(d.width.saturating_mul(d.height));
            if let Some(color) = descriptions
                .iter()
                .zip(color_bytes)
                .position(|(color, size)| {
                    color.pool == d.pool
                        && ranges_overlap(color.pool_offset, size, d.pool_offset, depth_bytes)
                })
            {
                return Err(stop(
                    Status::Unimplemented,
                    format!(
                        "color target {color} (texture {}) and depth target texture {depth} \
                         share memory in pool {}",
                        targets[color], d.pool
                    ),
                ));
            }
            Some(DepthTarget {
                description: d,
                image,
                bytes: depth_bytes,
            })
        } else {
            None
        };
        let mut attachment_ranges = [(0_u64, 0_u64, 0_u64); 9];
        for (at, (&key, d)) in targets.iter().zip(&descriptions).enumerate() {
            let bytes = backend
                .texture_bytes(key, d)
                .map_or(color_bytes[at], |bytes| bytes as u64);
            attachment_ranges[at] = (d.pool, d.pool_offset, bytes);
        }
        if let Some(t) = &depth_description {
            attachment_ranges[descriptions.len()] =
                (t.description.pool, t.description.pool_offset, t.bytes);
        }
        let attachment_ranges =
            &attachment_ranges[..descriptions.len() + usize::from(depth_description.is_some())];
        let context = backend.context().clone();
        let Some(memory) = backend.global_memory.as_mut() else {
            return Err(stop(
                Status::BadArgument,
                "the draw reads vertex memory, but no memory pool is registered".into(),
            ));
        };
        let mut input = VertexInput {
            bindings: Vec::new(),
            attributes: Vec::new(),
        };
        let mut buffers = Vec::new();
        let mut fetches = Vec::new();
        let mut independent = true;
        for (attribute_index, attribute) in attributes.iter().enumerate() {
            let format = value(attribute, "Format")?;
            let format_kind = mapped(&contract.attribute_formats, format[0]).ok_or_else(|| {
                stop(
                    Status::Unimplemented,
                    format!(
                        "vertex attribute {attribute_index} has format token {}, which the \
                         first-draw contract does not map to exactly one vertex format",
                        format[0]
                    ),
                )
            })?;
            let features = unsafe {
                context.instance.get_physical_device_format_properties(
                    context.physical_device,
                    format_kind.vk(),
                )
            }
            .buffer_features;
            if !features.contains(vk::FormatFeatureFlags::VERTEX_BUFFER) {
                return Err(stop(
                    Status::Unimplemented,
                    format!(
                        "vertex attribute {attribute_index} has format {format_kind:?} (VkFormat \
                         {}), which this device does not read from a vertex buffer",
                        format_kind.vk().as_raw()
                    ),
                ));
            }
            let offset = u32::try_from(format[1]).map_err(|_| {
                stop(
                    Status::BadArgument,
                    format!(
                        "vertex attribute {attribute_index} has relative offset {}, which does \
                         not fit 32 bits",
                        format[1]
                    ),
                )
            })?;
            let stream_index = value(attribute, "StreamIndex")?[0];
            let binding = u32::try_from(stream_index).map_err(|_| {
                stop(
                    Status::BadArgument,
                    format!(
                        "vertex attribute {attribute_index} reads vertex stream {stream_index}, \
                         which does not fit 32 bits"
                    ),
                )
            })?;
            let stream = streams.get(binding as usize).ok_or_else(|| {
                stop(
                    Status::BadArgument,
                    StreamFault::Undescribed {
                        stream: u64::from(binding),
                        streams: streams.len(),
                    }
                    .to_string(),
                )
            })?;
            let (address, size) = *self.buffers.get(&u64::from(binding)).ok_or_else(|| {
                stop(
                    Status::BadArgument,
                    StreamFault::Unbound {
                        stream: u64::from(binding),
                    }
                    .to_string(),
                )
            })?;
            let divisor = value(stream, "Divisor")?[0];
            if divisor > 1 {
                return Err(stop(
                    Status::Unimplemented,
                    format!(
                        "vertex stream {binding} has instance divisor {divisor}; swxbx draws \
                         divisors 0 and 1"
                    ),
                ));
            }
            if binding >= 32 {
                return Err(stop(
                    Status::Unimplemented,
                    format!("vertex attribute {attribute_index} reads vertex stream {binding}; swxbx draws streams 0 to 31"),
                ));
            }
            if divisor == 1 {
                pipeline_state.instance_bindings |= 1 << binding;
            }
            let stride_value = value(stream, "Stride")?[0];
            let stride = u32::try_from(stride_value).map_err(|_| {
                stop(
                    Status::BadArgument,
                    format!(
                        "vertex stream {binding} has stride {stride_value}, which does not fit \
                         32 bits"
                    ),
                )
            })?;
            let (bytes, alignment) = format_kind.layout();
            let resolution = instance
                .objects
                .resolve_gpu_address(address)
                .map_err(|error| {
                    stop(
                        Status::BadArgument,
                        format!(
                            "the vertex buffer bound to stream {binding} starts at address \
                             {address:#x}, {}",
                            match error {
                                super::objects::GpuAddressError::UnknownAddress =>
                                    "which no registered memory pool contains".to_string(),
                                super::objects::GpuAddressError::OutsidePool { pool } =>
                                    format!("which lies outside memory pool {pool}"),
                            }
                        ),
                    )
                })?;
            if size > resolution.remaining {
                return Err(stop(
                    Status::BadArgument,
                    format!(
                        "the vertex buffer bound to stream {binding} is {size} bytes at address \
                         {address:#x}, but memory pool {} has {} bytes left from there",
                        resolution.pool, resolution.remaining
                    ),
                ));
            }
            // Every fetched element, including the arena binding address, must be aligned.
            let Some((buffer, at)) = usize::try_from(size)
                .ok()
                .and_then(|length| memory.image_region(resolution.pool, resolution.offset, length))
            else {
                let held = usize::try_from(size).ok().and_then(|length| {
                    memory.buffer_region(resolution.pool, resolution.offset, length)
                });
                return Err(stop(
                    Status::BadArgument,
                    match held {
                        Some(_) => format!(
                            "the vertex buffer bound to stream {binding} starts at offset {} of \
                             memory pool {}, not on a four-byte boundary",
                            resolution.offset, resolution.pool
                        ),
                        None => format!(
                            "the vertex buffer bound to stream {binding}, {size} bytes at offset \
                             {} of memory pool {}, is not memory swxbx holds for that pool",
                            resolution.offset, resolution.pool
                        ),
                    },
                ));
            };
            if !stride.is_multiple_of(alignment)
                || !(at + u64::from(offset)).is_multiple_of(u64::from(alignment))
            {
                return Err(stop(
                    Status::BadArgument,
                    format!(
                        "vertex attribute {attribute_index} reads {bytes}-byte elements needing \
                         {alignment}-byte alignment from stream {binding}, but the stream's \
                         stride is {stride} and the attribute starts {offset} bytes into a \
                         buffer at offset {} of memory pool {}",
                        resolution.offset, resolution.pool
                    ),
                ));
            }
            let fetch = Fetch {
                attribute: attribute_index,
                stream: binding,
                stride,
                offset,
                bytes,
                size,
            };
            let fetched = if divisor == 1 {
                fetch.consecutive("instances", geometry.first_instance, geometry.instances)
            } else if let Vertices::Arrays { first } = vertices {
                fetch.consecutive("vertices", first, count)
            } else {
                Ok(())
            };
            fetched.map_err(|reason| stop(Status::BadArgument, reason))?;
            independent &= !attachment_ranges.iter().any(|&(pool, offset, bytes)| {
                pool == resolution.pool && ranges_overlap(offset, bytes, resolution.offset, size)
            });
            if divisor == 0 {
                fetches.push(fetch);
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
                    return Err(stop(
                        Status::Unimplemented,
                        format!(
                            "the first-draw contract gives 16-bit and 32-bit indices the same \
                             token {}, so indexed draws cannot be read",
                            contract.index_u16
                        ),
                    ));
                }
                let format = if index_type == contract.index_u16 {
                    IndexFormat::U16
                } else if index_type == contract.index_u32 {
                    IndexFormat::U32
                } else {
                    return Err(stop(
                        Status::Unimplemented,
                        format!(
                            "index type token {index_type} is neither the contract's 16-bit \
                             token {} nor its 32-bit token {}",
                            contract.index_u16, contract.index_u32
                        ),
                    ));
                };
                // An empty indexed draw fetches neither indices nor vertices.
                if count == 0 {
                    return Ok(());
                }
                let width = format.width() as u64;
                let index_resolution =
                    instance
                        .objects
                        .resolve_gpu_address(indices)
                        .map_err(|error| {
                            stop(
                                Status::BadArgument,
                                format!(
                                    "the index buffer starts at address {indices:#x}, {}",
                                    match error {
                                        super::objects::GpuAddressError::UnknownAddress =>
                                            "which no registered memory pool contains".to_string(),
                                        super::objects::GpuAddressError::OutsidePool { pool } =>
                                            format!("which lies outside memory pool {pool}"),
                                    }
                                ),
                            )
                        })?;
                let length = u64::from(count) * width;
                independent &= !attachment_ranges.iter().any(|&(pool, offset, bytes)| {
                    pool == index_resolution.pool
                        && ranges_overlap(offset, bytes, index_resolution.offset, length)
                });
                if length > index_resolution.remaining {
                    return Err(stop(
                        Status::BadArgument,
                        format!(
                            "the draw reads {count} indices of {width} bytes at address \
                             {indices:#x}, but memory pool {} has {} bytes left from there",
                            index_resolution.pool, index_resolution.remaining
                        ),
                    ));
                }
                let Some((index_buffer, index_offset)) =
                    usize::try_from(length).ok().and_then(|length| {
                        memory.buffer_region(index_resolution.pool, index_resolution.offset, length)
                    })
                else {
                    return Err(stop(
                        Status::BadArgument,
                        format!(
                            "the draw's {count} indices, {length} bytes at offset {} of memory \
                             pool {}, are not device memory swxbx holds",
                            index_resolution.offset, index_resolution.pool
                        ),
                    ));
                };
                if !index_offset.is_multiple_of(width) {
                    return Err(stop(
                        Status::BadArgument,
                        format!(
                            "the index buffer at address {indices:#x} is not aligned to its \
                             {width}-byte indices"
                        ),
                    ));
                }
                // Read canonical bytes, including earlier completed GPU writes in this submission.
                // Bounded chunks avoid allocating a count-sized host copy.
                let bytes = &mut self.index_bytes;
                bytes.resize(length.min(64 * 1024) as usize, 0);
                let mut read = 0;
                while read < length {
                    let chunk = (length - read).min(bytes.len() as u64) as usize;
                    memory
                        .read_pool(
                            index_resolution.pool,
                            index_resolution.offset + read,
                            &mut bytes[..chunk],
                        )
                        .ok_or_else(|| {
                            stop(
                                Status::InternalError,
                                format!(
                                    "swxbx could not read the draw's indices at offset {} of \
                                     memory pool {}",
                                    index_resolution.offset + read,
                                    index_resolution.pool
                                ),
                            )
                        })?;
                    for fetch in &fetches {
                        if let Some(reason) =
                            fetch.indexed(&bytes[..chunk], format, base_vertex, read / width)
                        {
                            return Err(stop(Status::BadArgument, reason));
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
        let viewport = Viewport::new(self.viewport, description.width, description.height)
            .map_err(|status| {
                stop(
                    status,
                    match self.viewport {
                        None => "no viewport is set".to_string(),
                        Some([_, x, y, w, h]) => format!(
                            "the viewport at ({x}, {y}) is {w} wide and {} high; swxbx draws \
                             a nonempty viewport at the left edge, upright from the top or \
                             mirrored from the bottom, within the {}x{} targets",
                            h as u32 as i32, description.width, description.height
                        ),
                    },
                )
            })?;
        let scissor =
            rectangle(self.scissor, description.width, description.height).map_err(|status| {
                stop(
                    status,
                    match self.scissor {
                        None => "no scissor is set".to_string(),
                        Some([_, x, y, w, h]) => format!(
                            "the scissor at ({x}, {y}) is {w}x{h}; swxbx draws a nonempty \
                             scissor within the {}x{} targets",
                            description.width, description.height
                        ),
                    },
                )
            })?;
        let Some(program) = self.program else {
            return Err(stop(Status::Unimplemented, "no program is bound".into()));
        };
        let translations = instance
            .objects
            .with(program, |object| match object {
                Object::Program {
                    shader_translations,
                    ..
                } => Some(shader_translations.clone()),
                _ => None,
            })
            .flatten()
            .ok_or_else(|| {
                stop(
                    Status::BadArgument,
                    format!("the bound program {program} is not a program"),
                )
            })?;
        let texels = super::specialization::TexelImages::bound(
            instance,
            &backend.texture_contract,
            &backend.image_contract,
            self.texture_pool,
            |stage, index| self.textures.get(&(stage, index)),
        );
        let blocking = backend.graphics.policy() == crate::gpu::graphics::PendingDrawPolicy::Block;
        let mut translation_keys = Vec::new();
        let mut stages = Vec::new();
        let mut surfaces = Vec::new();
        let mut bindless = Vec::new();
        for (index, translation) in translations.into_iter().enumerate() {
            let translation = match translation {
                ShaderTranslation::Deferred(code, context) => instance
                    .retry_cached_translation(&code, &context, blocking)
                    .ok_or_else(|| {
                        backend.graphics.refuse_draw(
                            Status::Unimplemented,
                            format!(
                                "shader {index} of program {program} waits for translation, \
                                 and no startup cache is present to translate it"
                            ),
                        )
                    })?,
                other => other,
            };
            let (shader, geometry_input) = match translation {
                ShaderTranslation::Spirv(words) => {
                    stages.push(Arc::new(
                        crate::gpu::graphics::Stage::checked(context.shader_features, words)
                            .map_err(|error| {
                                backend.graphics.refuse_draw(Status::Unimplemented, error)
                            })?,
                    ));
                    continue;
                }
                // A geometry stage translates for the primitive this draw
                // assembles; every other stage translated when it registered.
                ShaderTranslation::ByInputPrimitive(code) => {
                    let primitive = topology.geometry_input();
                    let shader = instance
                        .specializations
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .by_primitive(instance, program, &code, primitive)?;
                    (shader, Some(primitive))
                }
                ShaderTranslation::Cached(request) => match if blocking {
                    request.wait()
                } else {
                    request.poll()
                } {
                    crate::startup_cache::TranslationStatus::Ready(shader) => (shader, None),
                    crate::startup_cache::TranslationStatus::Queued
                    | crate::startup_cache::TranslationStatus::Compiling => {
                        backend
                            .graphics
                            .log_skipped_draw("shader translation pending");
                        return Ok(());
                    }
                    crate::startup_cache::TranslationStatus::Failed(error) => {
                        return Err(backend.graphics.refuse_draw(
                            Status::Unimplemented,
                            format!(
                                "shader {index} of program {program} failed to translate: {error}"
                            ),
                        ))
                    }
                },
                ShaderTranslation::Deferred(..) => {
                    backend.graphics.log_skipped_draw("translation queue full");
                    return Ok(());
                }
                ShaderTranslation::Error(error) => {
                    return Err(backend.graphics.refuse_draw(
                        Status::Unimplemented,
                        format!("shader {index} of program {program} failed to translate: {error}"),
                    ));
                }
            };
            let shader = if texels.is_empty() {
                shader
            } else {
                instance
                    .specializations
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .stage(instance, program, index, &shader, geometry_input, &texels)?
            };
            let stage = backend
                .graphics
                .translated_stage(&shader)
                .map_err(|error| backend.graphics.refuse_draw(Status::Unimplemented, error))?;
            let graphics_stage = GraphicsStage::of(&shader.words).map_err(|error| {
                backend.graphics.refuse_draw(
                    Status::Unimplemented,
                    format!("shader {index} of program {program}: {error}"),
                )
            })?;
            if shader.requires_subgroup_size_32 {
                pipeline_state.subgroup_size_32[graphics_stage.slot()] = true;
            }
            surfaces.extend(shader.surfaces.iter().map(|s| (graphics_stage, *s)));
            bindless.extend(shader.textures.iter().map(|t| (graphics_stage, *t)));
            // The startup cache holds translations made at registration;
            // a pipeline with a stage translated for its draw is not
            // remembered for the next start.
            if geometry_input.is_none() {
                translation_keys.push(shader.key.clone());
            }
            stages.push(stage);
        }
        let limits = context.limits;
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
            return Err(backend.graphics.refuse_draw(
                Status::BadArgument,
                format!(
                    "the {}x{} targets with a {:?} viewport exceed this device: framebuffers \
                     up to {}x{}, viewports up to {:?} within {:?}",
                    description.width,
                    description.height,
                    viewport.size(),
                    limits.max_framebuffer_width,
                    limits.max_framebuffer_height,
                    limits.max_viewport_dimensions,
                    limits.viewport_bounds_range
                ),
            ));
        }
        if let Some((stage, index)) = self.uniforms.keys().find(|&&(stage, index)| {
            !backend
                .uniforms
                .bindings
                .iter()
                .any(|m| m.stage == stage && m.index == index)
        }) {
            return Err(backend.graphics.refuse_draw(
                Status::Unimplemented,
                format!(
                    "recorded uniform has no contract mapping: uniform buffer {index} of \
                     stage {stage:?} is bound, and the host contract maps no shader bank to it"
                ),
            ));
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
            .map_err(|error| backend.graphics.refuse_draw(Status::Unimplemented, error))?;
        let pending_banks;
        let banks = if let Some(pipeline) = &pipeline {
            &pipeline.banks
        } else {
            // In pipeline stage order, as a pipeline lists its banks.
            let mut ordered: Vec<_> = stages.iter().collect();
            ordered.sort_by_key(|stage| GraphicsStage::of(stage.words()).ok());
            pending_banks = ordered
                .into_iter()
                .map(|stage| {
                    let words = stage.words();
                    let stage = GraphicsStage::of(words).map_err(|error| {
                        backend
                            .graphics
                            .refuse_draw(Status::Unimplemented, format!("a bound shader: {error}"))
                    })?;
                    uniforms::banks(words, stage).map_err(|error| {
                        backend.graphics.refuse_draw(
                            Status::Unimplemented,
                            format!("the {stage:?} shader's uniform banks: {error}"),
                        )
                    })
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
                    backend.graphics.refuse_draw(
                        Status::Unimplemented,
                        format!(
                            "shader uniform bank has no contract mapping: stage={stage:?} bank={}",
                            bank.bank
                        ),
                    )
                })?;
            let &(address, size) = self
                .uniforms
                .get(&(mapping.stage, mapping.index))
                .ok_or_else(|| {
                    backend.graphics.refuse_draw(
                        Status::Unimplemented,
                        format!(
                            "shader uniform bank is not bound: the {stage:?} shader reads bank \
                             {}, which is uniform buffer {} of stage {:?}",
                            bank.bank, mapping.index, mapping.stage
                        ),
                    )
                })?;
            let resolution = instance.objects.resolve_gpu_address(address).map_err(|_| {
                backend.graphics.refuse_draw(
                    Status::BadArgument,
                    format!(
                        "uniform buffer {} of stage {:?} is at GPU address {address:#x}, \
                             which no memory pool maps",
                        mapping.index, mapping.stage
                    ),
                )
            })?;
            if size == 0
                || size > BANK_SIZE
                || !size.is_multiple_of(16)
                || size > resolution.remaining
            {
                return Err(backend.graphics.refuse_draw(
                    Status::BadArgument,
                    format!(
                        "uniform buffer {} of stage {:?} is {size} bytes with {} bytes of its \
                         pool left; swxbx binds a nonzero multiple of 16 bytes up to {} \
                         that fits its pool",
                        mapping.index, mapping.stage, resolution.remaining, BANK_SIZE
                    ),
                ));
            }
            independent &= !attachment_ranges.iter().any(|&(pool, offset, bytes)| {
                pool == resolution.pool && ranges_overlap(offset, bytes, resolution.offset, size)
            });
            let info = if storage {
                memory.storage_buffer_info(resolution.pool, resolution.offset, size)
            } else {
                memory.uniform_buffer_info(resolution.pool, resolution.offset, size)
            }
            .ok_or_else(|| {
                backend.graphics.refuse_draw(
                    Status::BadArgument,
                    format!(
                        "uniform buffer {} of stage {:?} ({size} bytes at offset {} of memory \
                         pool {}) is not in the device arena at the device's buffer offset \
                         alignment and range",
                        mapping.index, mapping.stage, resolution.offset, resolution.pool
                    ),
                )
            })?;
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
                return Err(backend.graphics.refuse_draw(
                    Status::BadArgument,
                    format!(
                        "color target {target} ({}x{}, format {}, stride {}, at offset {} of \
                         memory pool {}) has no device image over its memory",
                        d.width, d.height, d.format, d.stride, d.pool_offset, d.pool
                    ),
                ));
            }
        }
        if let Some(t) = &depth_description {
            if !backend.ensure_depth_stencil(depth, &t.description, t.image) {
                let d = &t.description;
                return Err(backend.graphics.refuse_draw(
                    Status::BadArgument,
                    format!(
                        "depth target {depth} ({}x{}, {}, stride {}, at offset {} of memory \
                         pool {}) has no device image over its memory",
                        d.width,
                        d.height,
                        crate::gpu::formats::Name(t.image.guest()),
                        d.stride,
                        d.pool_offset,
                        d.pool
                    ),
                ));
            }
            if let Some(why) = backend.unheld_depth(
                depth,
                pipeline_state.depth_test,
                pipeline_state.stencil_test,
            ) {
                return Err(backend.graphics.refuse_draw(Status::Unimplemented, why));
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
            backend.finish_draws().ok_or_else(|| {
                backend.graphics.refuse_draw(
                    Status::InternalError,
                    "the draws before this surface draw did not finish on the device",
                )
            })?;
        }
        // Texture formats and linear samplers by the module pair they fill, so
        // a linear sampler is checked against the format it reads. Source: the public contract.
        let mut sampled_formats = Vec::new();
        let mut linear_samplers = Vec::new();
        for b in &pipeline.textures {
            if b.ty == vk::DescriptorType::STORAGE_IMAGE {
                let slot = format!(
                    "{:?} surface binding {} of set {}",
                    b.stage, b.binding, b.set
                );
                let refuse =
                    |status: Status, reason: String| backend.graphics.refuse_draw(status, reason);
                let surface = surfaces
                    .iter()
                    .find(|(stage, s)| {
                        *stage == b.stage && s.set == b.set && s.binding == b.binding
                    })
                    .ok_or_else(|| {
                        refuse(
                            Status::Unimplemented,
                            format!("storage image without a typed surface binding: {slot}"),
                        )
                    })?
                    .1;
                let texture = self.bound_surface(
                    instance,
                    &texture_contract,
                    b.stage.into(),
                    &surface,
                    None,
                    &refuse,
                )?;
                let Some(d) = instance.objects.texture_description(texture) else {
                    return Err(refuse(
                        Status::BadArgument,
                        format!("{slot} names object {texture}, which is not a texture"),
                    ));
                };
                if depth == texture || targets.contains(&texture) {
                    return Err(refuse(
                        Status::Unimplemented,
                        format!(
                            "surface is also a render target: {slot} names texture {texture}, \
                             which this draw also renders to"
                        ),
                    ));
                }
                let Some(c) = backend.first_draw.as_ref() else {
                    return Err(refuse(
                        Status::Unimplemented,
                        "the host gave no first-draw contract, so swxbx cannot bind surfaces"
                            .into(),
                    ));
                };
                if backend.has_image_contract() {
                    let resolved = backend.texture_image(texture, &d).ok_or_else(|| {
                        refuse(
                            Status::Unimplemented,
                            format!(
                                "{slot}: the image contract has no layout for texture {texture} \
                                 (format {}, target {})",
                                d.format, d.target
                            ),
                        )
                    })?;
                    if d.pool == 0 {
                        return Err(refuse(
                            Status::Unimplemented,
                            format!("{slot}: texture {texture} has no memory pool"),
                        ));
                    }
                    if b.image_kind != Some(resolved.descriptor.shape.kind) {
                        return Err(refuse(
                            Status::Unimplemented,
                            format!(
                                "{slot}: the shader declares a {:?} image and texture {texture} \
                                 is {:?}",
                                b.image_kind, resolved.descriptor.shape.kind
                            ),
                        ));
                    }
                } else if b.image_kind != Some(crate::tiling::ImageKind::D2)
                    || d.pool == 0
                    || c.color_format(d.format) != Some(vk::Format::R8G8B8A8_UNORM)
                    || d.target != c.target_2d
                    || d.depth > 1
                    || d.levels > 1
                    || d.flags != 0
                {
                    return Err(refuse(
                        Status::Unimplemented,
                        format!(
                            "{slot}: without an image contract swxbx binds one-level 2D \
                             RGBA8 surfaces over memory; the shader declares {:?} and texture \
                             {texture} has format {}, target {}, depth {}, {} levels, flags \
                             {:#x}, pool {}",
                            b.image_kind, d.format, d.target, d.depth, d.levels, d.flags, d.pool
                        ),
                    ));
                }
                let view = backend
                    .storage_texture(texture, &d, surface.format)
                    .ok_or_else(|| {
                        backend.graphics.refuse_draw(
                            Status::Unimplemented,
                            format!(
                                "surface image has no storage view: {slot} binds texture \
                                 {texture} as {:?}",
                                surface.format
                            ),
                        )
                    })?;
                storage_images.push(texture);
                texture_infos.push(
                    vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::GENERAL),
                );
                continue;
            }
            let image = b.ty == vk::DescriptorType::SAMPLED_IMAGE;
            let declared = bindless.iter().find(|(stage, t)| {
                *stage == b.stage
                    && t.set == b.set
                    && if image {
                        t.image == b.binding
                    } else {
                        t.sampler == b.binding
                    }
            });
            // `pair` names the module's texture and sampler pair.
            let (source, pair) = match declared {
                Some((_, t)) => {
                    let (texture, sampler) = handle_locations(&t.source).ok_or_else(|| {
                        backend.graphics.refuse_draw(
                            Status::Unimplemented,
                            format!(
                                "bindless texture handle {:?} at binding {} of set {} of the \
                                 {:?} shader is neither one constant-buffer word nor two \
                                 constant-buffer words or-ed; swxbx binds only those",
                                t.source, b.binding, b.set, b.stage
                            ),
                        )
                    })?;
                    let mapping = texture_contract
                        .locations
                        .iter()
                        .find(|m| m.target == b.stage)
                        .ok_or_else(|| {
                            backend.graphics.refuse_draw(
                                Status::Unimplemented,
                                format!(
                                    "the {:?} shader samples a bindless texture, and the host \
                                     contract names no location mapping for that stage",
                                    b.stage
                                ),
                            )
                        })?;
                    let (bank, offset) = if image { texture } else { sampler };
                    (
                        (mapping.stage, mapping.index(bank, offset)),
                        (b.stage, t.set, t.image),
                    )
                }
                None => {
                    let mapping = texture_contract
                        .bindings
                        .iter()
                        .find(|m| {
                            m.target == b.stage
                                && m.set == b.set
                                && if image {
                                    m.image == b.binding
                                } else {
                                    m.sampler == b.binding
                                }
                        })
                        .ok_or_else(|| {
                            backend.graphics.refuse_draw(
                                Status::Unimplemented,
                                format!(
                                    "shader texture has no contract mapping: stage={:?} set={} \
                                     binding={} type={}",
                                    b.stage,
                                    b.set,
                                    b.binding,
                                    b.ty.as_raw()
                                ),
                            )
                        })?;
                    (
                        (mapping.stage, mapping.index),
                        (mapping.target, mapping.set, mapping.image),
                    )
                }
            };
            let reference = if image {
                self.textures.get(&source)
            } else {
                self.samplers.get(&source)
            }
            .ok_or_else(|| {
                backend.graphics.refuse_draw(
                    Status::Unimplemented,
                    format!(
                        "shader texture or sampler is not bound: the {:?} shader reads {} {} \
                         of stage {:?}, and none is bound",
                        b.stage,
                        if image { "texture" } else { "sampler" },
                        source.1,
                        source.0
                    ),
                )
            })?;
            let id = match reference {
                TextureReference::Separate(handle) => u32::try_from(*handle).map_err(|_| {
                    backend.graphics.refuse_draw(
                        Status::Unimplemented,
                        format!("handle {handle:#x} does not fit a pool index"),
                    )
                })?,
                TextureReference::Combined(handle) => {
                    let (_, texture, sampler) = texture_contract
                        .combined
                        .iter()
                        .find(|(h, _, _)| h == handle)
                        .ok_or_else(|| {
                            backend.graphics.refuse_draw(
                                Status::Unimplemented,
                                format!(
                                    "combined handle {handle:#x} is not in the host \
                                     contract's combined handles"
                                ),
                            )
                        })?;
                    if image {
                        *texture
                    } else {
                        *sampler
                    }
                }
            };
            if image {
                let pool = self.texture_pool.ok_or_else(|| {
                    backend.graphics.refuse_draw(
                        Status::Unimplemented,
                        format!("texture pool is not bound: texture handle {id} needs one"),
                    )
                })?;
                let Some(registration) = instance
                    .objects
                    .with(pool, |object| match object {
                        Object::TexturePool { registered, .. } => {
                            Some(registered.get(&id).copied())
                        }
                        _ => None,
                    })
                    .flatten()
                else {
                    return Err(backend.graphics.refuse_draw(
                        Status::BadArgument,
                        format!("the bound texture pool {pool} is not a texture pool"),
                    ));
                };
                let (texture, view) = registration.ok_or_else(|| {
                    backend.graphics.refuse_draw(
                        Status::Unimplemented,
                        format!(
                            "texture handle is not registered: handle {id} in texture pool \
                             {pool}"
                        ),
                    )
                })?;
                if view != 0 {
                    return Err(backend.graphics.refuse_draw(
                        Status::Unimplemented,
                        format!(
                            "texture handle {id} registers texture {texture} with view {view}; \
                             swxbx samples whole textures (view 0)"
                        ),
                    ));
                }
                let Some(d) = instance.objects.texture_description(texture) else {
                    return Err(backend.graphics.refuse_draw(
                        Status::BadArgument,
                        format!(
                            "texture handle {id} registers object {texture}, which is not a \
                             texture"
                        ),
                    ));
                };
                if d.width == 0 || d.height == 0 {
                    return Err(backend.graphics.refuse_draw(
                        Status::BadArgument,
                        format!("sampled texture {texture} is {}x{}", d.width, d.height),
                    ));
                }
                // A bindless handle's module was translated without texel
                // coordinates: those are specialized for bound textures only.
                if declared.is_some()
                    && backend.image_contract.coordinates(&d)
                        == Some(crate::gpu::image_layout::Coordinates::Texels)
                {
                    return Err(backend.graphics.refuse_draw(
                        Status::Unimplemented,
                        format!(
                            "sampled texture {texture} reads texel coordinates through a \
                             bindless handle; swxbx translates texel coordinates for bound \
                             textures only"
                        ),
                    ));
                }
                let sample_bytes = backend
                    .texture_bytes(texture, &d)
                    .map_or(d.width.saturating_mul(d.height).saturating_mul(4), |n| {
                        n as u64
                    });
                if depth == texture
                    || depth_description.as_ref().is_some_and(|other| {
                        d.pool == other.description.pool
                            && ranges_overlap(
                                d.pool_offset,
                                sample_bytes,
                                other.description.pool_offset,
                                other.bytes,
                            )
                    })
                {
                    return Err(backend.graphics.refuse_draw(
                        Status::Unimplemented,
                        format!(
                            "sampled texture description is unsupported: texture {texture} \
                             is or overlaps the depth target {depth}"
                        ),
                    ));
                }
                if targets.contains(&texture)
                    || targets.iter().zip(&descriptions).zip(color_bytes).any(
                        |((&key, other), size)| {
                            d.pool == other.pool
                                && ranges_overlap(
                                    d.pool_offset,
                                    sample_bytes,
                                    other.pool_offset,
                                    backend.texture_bytes(key, other).map_or(size, |n| n as u64),
                                )
                        },
                    )
                {
                    return Err(backend.graphics.refuse_draw(
                        Status::Unimplemented,
                        format!(
                            "sampled texture {texture} is or overlaps a color target of this \
                             draw"
                        ),
                    ));
                }
                let refuse = |reason: String| {
                    backend.graphics.refuse_draw(
                        Status::Unimplemented,
                        format!("sampled texture {texture}: {reason}"),
                    )
                };
                let Some(c) = backend.first_draw.as_ref() else {
                    return Err(refuse(
                        "the host gave no first-draw contract, so swxbx cannot sample \
                         textures"
                            .into(),
                    ));
                };
                if backend.has_image_contract() {
                    let resolved = backend.texture_image(texture, &d).ok_or_else(|| {
                        refuse(format!(
                            "the image contract has no layout for format {}, target {}",
                            d.format, d.target
                        ))
                    })?;
                    let shape = resolved.descriptor.shape;
                    let class = crate::gpu::image_layout::numeric_class(resolved.descriptor.format);
                    if d.pool == 0 {
                        return Err(refuse("it has no memory pool".into()));
                    }
                    if !backend.has_swizzle_contract() && d.swizzle != c.identity_swizzle {
                        return Err(refuse(format!(
                            "swizzle {:?} needs a swizzle contract, and the host gave none",
                            d.swizzle
                        )));
                    }
                    if b.image_kind != Some(shape.kind) {
                        return Err(refuse(format!(
                            "the shader declares a {:?} image and the texture is {:?}",
                            b.image_kind, shape.kind
                        )));
                    }
                    if shape.kind == crate::tiling::ImageKind::Cube
                        && b.arrayed != (shape.layers > 6)
                    {
                        return Err(refuse(format!(
                            "the shader declares a cube {} and the texture has {} layers",
                            if b.arrayed { "array" } else { "image" },
                            shape.layers
                        )));
                    }
                    if !matches!(
                        class,
                        crate::gpu::image_layout::NumericClass::Float
                            | crate::gpu::image_layout::NumericClass::Depth
                    ) {
                        return Err(refuse(format!(
                            "its device format {} does not read as float or depth; swxbx \
                             samples float and depth formats",
                            resolved.descriptor.format.as_raw()
                        )));
                    }
                    crate::gpu::formats::supports(
                        backend.context(),
                        resolved.descriptor.format,
                        crate::gpu::formats::FormatUse::Sampled,
                    )
                    .map_err(refuse)?;
                    sampled_formats.push((pair, texture, resolved.descriptor.format));
                } else if b.image_kind != Some(crate::tiling::ImageKind::D2)
                    || b.arrayed
                    || d.pool == 0
                    || c.color_format(d.format) != Some(vk::Format::R8G8B8A8_UNORM)
                    || d.target != c.target_2d
                    || d.depth > 1
                    || d.levels > 1
                    || d.swizzle != c.identity_swizzle
                    || d.flags != 0
                    || d.depth_stencil_mode != 0
                {
                    return Err(refuse(format!(
                        "without an image contract swxbx samples one-level 2D RGBA8 textures \
                         over memory with the identity swizzle; the shader declares {:?}{} and \
                         the texture has format {}, target {}, depth {}, {} levels, swizzle \
                         {:?}, flags {:#x}, depth-stencil mode {}, pool {}",
                        b.image_kind,
                        if b.arrayed { " arrayed" } else { "" },
                        d.format,
                        d.target,
                        d.depth,
                        d.levels,
                        d.swizzle,
                        d.flags,
                        d.depth_stencil_mode,
                        d.pool
                    )));
                }
                let view = backend.sampled_texture(texture, &d).ok_or_else(|| {
                    backend.graphics.refuse_draw(
                        Status::BadArgument,
                        format!(
                            "sampled texture {texture} ({}x{}, format {}, at offset {} of \
                             memory pool {}) has no device image over its memory",
                            d.width, d.height, d.format, d.pool_offset, d.pool
                        ),
                    )
                })?;
                texture_infos.push(
                    vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                );
            } else {
                let pool = self.sampler_pool.ok_or_else(|| {
                    backend.graphics.refuse_draw(
                        Status::Unimplemented,
                        format!("sampler pool is not bound: sampler handle {id} needs one"),
                    )
                })?;
                let Some(registration) = instance
                    .objects
                    .with(pool, |object| match object {
                        Object::SamplerPool { registered, .. } => {
                            Some(registered.get(&id).copied())
                        }
                        _ => None,
                    })
                    .flatten()
                else {
                    return Err(backend.graphics.refuse_draw(
                        Status::BadArgument,
                        format!("the bound sampler pool {pool} is not a sampler pool"),
                    ));
                };
                let sampler = registration.ok_or_else(|| {
                    backend.graphics.refuse_draw(
                        Status::Unimplemented,
                        format!(
                            "sampler handle is not registered: handle {id} in sampler pool \
                             {pool}"
                        ),
                    )
                })?;
                let Some(d) = instance
                    .objects
                    .with(sampler, |object| match object {
                        Object::Sampler(d) => Some(d.clone()),
                        _ => None,
                    })
                    .flatten()
                else {
                    return Err(backend.graphics.refuse_draw(
                        Status::BadArgument,
                        format!(
                            "sampler handle {id} registers object {sampler}, which is not a \
                             sampler"
                        ),
                    ));
                };
                let key = crate::gpu::textures::SamplerKey::new(&d, &texture_contract).ok_or_else(
                    || {
                        backend.graphics.refuse_draw(
                            Status::Unimplemented,
                            format!(
                                "sampler description is unsupported: sampler {sampler} sets \
                                 a filter, address mode, compare, LOD, anisotropy or border \
                                 value swxbx or the host contract cannot map"
                            ),
                        )
                    },
                )?;
                if key.filters_linearly() {
                    linear_samplers.push((pair, sampler));
                }
                if key.mirrors_once()
                    && !crate::gpu::textures::mirror_clamp_to_edge(backend.context())
                {
                    return Err(backend.graphics.refuse_draw(
                        Status::Unimplemented,
                        format!(
                            "sampler {sampler} mirrors the texture once and clamps to its \
                             edge, which this device cannot sample"
                        ),
                    ));
                }
                if let Some(color) = key.custom_border() {
                    if let Some(why) = backend.context().custom_border_colors.refusal() {
                        return Err(backend.graphics.refuse_draw(
                            Status::Unimplemented,
                            format!(
                                "sampler {sampler} clamps to border colour {color:?}, which no \
                                 fixed Vulkan border colour states, and {why}"
                            ),
                        ));
                    }
                }
                let created = backend.sampler(pool, id, key);
                let Some(sampler) = created else {
                    let limit = backend.context().custom_border_colors.limit();
                    return Err(backend.graphics.refuse_draw(
                        Status::InternalError,
                        match key.custom_border() {
                            Some(color) => format!(
                                "the device could not create sampler {sampler} with custom \
                                 border colour {color:?}; it holds at most {limit} samplers with \
                                 a custom border colour at once"
                            ),
                            None => format!("the device could not create sampler {sampler}"),
                        },
                    ));
                };
                texture_infos.push(vk::DescriptorImageInfo::default().sampler(sampler));
            }
        }
        for &(slot, sampler) in &linear_samplers {
            for &(_, texture, format) in sampled_formats.iter().filter(|(s, _, _)| *s == slot) {
                crate::gpu::formats::filters_linearly(backend.context(), format).map_err(
                    |why| {
                        backend.graphics.refuse_draw(
                            Status::Unimplemented,
                            format!(
                                "sampler {sampler} filters sampled texture {texture} linearly, \
                                 and {why}"
                            ),
                        )
                    },
                )?;
            }
        }
        let descriptors = backend
            .graphics
            .descriptors(&pipeline, &uniform_buffers, &texture_infos)
            .map_err(|error| {
                backend.graphics.refuse_draw(
                    Status::InternalError,
                    format!("the draw's descriptor set could not be written: {error}"),
                )
            })?;
        let surfaces_bound = !storage_images.is_empty();
        let draw = Arc::new(Draw {
            descriptors,
            storage_images,
            buffers,
            viewport: viewport.vulkan(),
            scissor,
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
            .ok_or_else(|| {
                backend.graphics.refuse_draw(
                    Status::InternalError,
                    "the device could not record this draw",
                )
            })?;
        if surfaces_bound {
            // A later draw may sample or load what this one writes.
            backend.finish_draws().ok_or_else(|| {
                backend.graphics.refuse_draw(
                    Status::InternalError,
                    "the surface draw did not finish on the device",
                )
            })?;
        }
        Ok(())
    }

    /// The guest texture a bound surface names: the host contract maps the
    /// module's bound slot to the recorded image binding, and the texture
    /// pool resolves that handle. `refuse` says why a surface cannot bind.
    pub(super) fn bound_surface(
        &self,
        instance: &Instance,
        contract: &crate::gpu::textures::TextureContract,
        target: crate::gpu::textures::SurfaceStage,
        surface: &crate::surfaces::SurfaceBinding,
        bindless_location: Option<(u64, u64)>,
        refuse: &dyn Fn(Status, String) -> Status,
    ) -> Result<u64, Status> {
        let unimplemented = |reason: String| refuse(Status::Unimplemented, reason);
        let (stage, index) = match surface.source {
            crate::surfaces::SurfaceSource::Bound { slot } => {
                let mapping = contract
                    .surfaces
                    .iter()
                    .find(|m| m.target == target && m.slot == slot)
                    .ok_or_else(|| {
                        unimplemented(format!(
                            "surface slot without a host mapping: {target:?} slot {slot}"
                        ))
                    })?;
                (mapping.stage, mapping.index)
            }
            crate::surfaces::SurfaceSource::Bindless(handle) => {
                let (stage, base) = bindless_location.ok_or_else(|| {
                    unimplemented(format!(
                        "bindless surface at binding {} of set {} has no location contract",
                        surface.binding, surface.set
                    ))
                })?;
                let (bank, offset) = crate::api::operations::bindless_texture_location(&handle)
                    .ok_or_else(|| {
                        unimplemented(format!(
                            "bindless surface at binding {} of set {} has an unsupported handle",
                            surface.binding, surface.set
                        ))
                    })?;
                (stage, base + (u64::from(bank) << 16 | u64::from(offset)))
            }
        };
        let handle = *self.images.get(&(stage, index)).ok_or_else(|| {
            unimplemented(format!(
                "surface without a bound image: {target:?} reads image {index} of stage \
                     {stage}, and none is bound"
            ))
        })?;
        let id = u32::try_from(handle).map_err(|_| {
            unimplemented(format!(
                "surface image handle {handle:#x} does not fit a texture pool index"
            ))
        })?;
        let pool = self.texture_pool.ok_or_else(|| {
            unimplemented(format!(
                "surface image handle {id} needs a texture pool, and none is bound"
            ))
        })?;
        let Some(registration) = instance
            .objects
            .with(pool, |object| match object {
                Object::TexturePool { registered, .. } => Some(registered.get(&id).copied()),
                _ => None,
            })
            .flatten()
        else {
            return Err(refuse(
                Status::BadArgument,
                format!("the bound texture pool {pool} is not a texture pool"),
            ));
        };
        let (texture, view) = registration.ok_or_else(|| {
            unimplemented(format!(
                "surface image handle {id} is not registered in texture pool {pool}"
            ))
        })?;
        if view != 0 {
            return Err(unimplemented(format!(
                "surface image handle {id} registers texture {texture} with view {view}; \
                 swxbx binds whole textures (view 0)"
            )));
        }
        Ok(texture)
    }

    /// Resolve one recorded texture or sampler reference through its pool.
    pub(super) fn bound_sampled_object(
        &self,
        instance: &Instance,
        contract: &crate::gpu::textures::TextureContract,
        source: (u64, u64),
        image: bool,
        refuse: &dyn Fn(Status, String) -> Status,
    ) -> Result<(u64, u32, u64), Status> {
        let reference = if image {
            self.textures.get(&source)
        } else {
            self.samplers.get(&source)
        }
        .ok_or_else(|| {
            let bound = if image {
                self.textures.keys()
            } else {
                self.samplers.keys()
            }
            .filter(|(stage, _)| *stage == source.0)
            .map(|(_, index)| format!("{index:#x}"))
            .collect::<Vec<_>>()
            .join(",");
            refuse(
                Status::Unimplemented,
                format!(
                    "compute {} {:#x} of stage {} is not bound (bound: [{}])",
                    if image { "texture" } else { "sampler" },
                    source.1,
                    source.0,
                    bound,
                ),
            )
        })?;
        let id = match reference {
            TextureReference::Separate(handle) => u32::try_from(*handle).map_err(|_| {
                refuse(
                    Status::Unimplemented,
                    format!("compute handle {handle:#x} does not fit a pool index"),
                )
            })?,
            TextureReference::Combined(handle) => {
                let (_, texture, sampler) = contract
                    .combined
                    .iter()
                    .find(|(candidate, _, _)| candidate == handle)
                    .ok_or_else(|| {
                        refuse(
                            Status::Unimplemented,
                            format!("compute combined handle {handle:#x} is not mapped"),
                        )
                    })?;
                if image {
                    *texture
                } else {
                    *sampler
                }
            }
        };
        let pool = if image {
            self.texture_pool
        } else {
            self.sampler_pool
        }
        .ok_or_else(|| {
            refuse(
                Status::Unimplemented,
                format!(
                    "compute {} pool is not bound",
                    if image { "texture" } else { "sampler" }
                ),
            )
        })?;
        let object = instance
            .objects
            .with(pool, |object| match object {
                Object::TexturePool { registered, .. } if image => registered.get(&id).copied(),
                Object::SamplerPool { registered, .. } if !image => {
                    registered.get(&id).copied().map(|object| (object, 0))
                }
                _ => None,
            })
            .flatten()
            .ok_or_else(|| {
                refuse(
                    Status::Unimplemented,
                    format!("compute handle {id} is not registered in pool {pool}"),
                )
            })?;
        if image && object.1 != 0 {
            return Err(refuse(
                Status::Unimplemented,
                format!(
                    "compute texture handle {id} uses unsupported view {}",
                    object.1
                ),
            ));
        }
        Ok((pool, id, object.0))
    }

    pub(super) fn has_bound_sampled_object(&self, source: (u64, u64), image: bool) -> bool {
        if image {
            self.textures.contains_key(&source)
        } else {
            self.samplers.contains_key(&source)
        }
    }

    /// The color attachments' blend and write-mask state; the error names the
    /// setting execution cannot draw with. Source: the public contract.
    fn color_state(
        &self,
        contract: Option<&crate::gpu::graphics::BlendContract>,
        count: usize,
    ) -> Result<[crate::gpu::graphics::ColorAttachmentState; 8], Refusal> {
        let mut colors = [crate::gpu::graphics::ColorAttachmentState::default(); 8];
        let settings = self.settings("ColorState", &["BlendEnable"])?;
        let mut known = [false; 8];
        for &(_, args) in settings {
            let Some(at) = usize::try_from(args[0])
                .ok()
                .filter(|&at| at < colors.len())
            else {
                return Err(Refusal::unimplemented(format!(
                    "the bound ColorState sets BlendEnable for color target {}; swxbx draws \
                     to targets 0 to 7",
                    args[0]
                )));
            };
            colors[at].enable = boolean(
                args[1],
                &format!("ColorState BlendEnable for color target {at}"),
            )?;
            known[at] = true;
        }
        if let Some(at) = known[..count].iter().position(|k| !*k) {
            return Err(Refusal::unimplemented(format!(
                "the bound ColorState does not set BlendEnable for color target {at}"
            )));
        }
        for (at, color) in colors.iter_mut().enumerate() {
            if !color.enable {
                continue;
            }
            let Some(contract) = contract else {
                return Err(Refusal::unimplemented(format!(
                    "blending is enabled for color target {at}, but no blend contract is \
                     installed"
                )));
            };
            let settings = self
                .blends
                .get(&(at as u64))
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    Refusal::unimplemented(format!(
                        "blending is enabled for color target {at}, but no BlendState for \
                         that target is bound"
                    ))
                })?;
            if let Some((name, _)) = settings
                .iter()
                .find(|(name, _)| !matches!(*name, "BlendTarget" | "BlendFunc" | "BlendEquation"))
            {
                return Err(Refusal::unimplemented(format!(
                    "the BlendState for color target {at} sets {name}, which swxbx does not \
                     draw with"
                )));
            }
            let f = setting(settings, "BlendFunc", "BlendState")?;
            let e = setting(settings, "BlendEquation", "BlendState")?;
            for i in 0..4 {
                let token = f[contract.function_order[i]];
                color.factors[i] = mapped(&contract.factors, token)
                    .ok_or_else(|| {
                        Refusal::unimplemented(format!(
                            "blend factor token {token} for color target {at} is not mapped \
                             to exactly one factor by the blend contract"
                        ))
                    })?
                    .as_raw();
            }
            for i in 0..2 {
                let token = e[contract.equation_order[i]];
                color.operations[i] = mapped(&contract.operations, token)
                    .ok_or_else(|| {
                        Refusal::unimplemented(format!(
                            "blend equation token {token} for color target {at} is not mapped \
                             to exactly one operation by the blend contract"
                        ))
                    })?
                    .as_raw();
            }
        }
        if self.states.contains_key("ChannelMaskState") {
            let Some(c) = contract else {
                return Err(Refusal::unimplemented(
                    "a ChannelMaskState is bound, but no blend contract is installed",
                ));
            };
            for &(_, args) in self.settings("ChannelMaskState", &["ChannelMask"])? {
                let Some(at) = usize::try_from(args[0])
                    .ok()
                    .filter(|&at| at < colors.len())
                else {
                    return Err(Refusal::unimplemented(format!(
                        "the bound ChannelMaskState sets a mask for color target {}; swxbx \
                         draws to targets 0 to 7",
                        args[0]
                    )));
                };
                let color = &mut colors[at];
                color.write_mask = 0;
                for i in 0..4 {
                    if boolean(
                        args[1 + c.channel_order[i]],
                        &format!("ChannelMaskState channel {i} for color target {at}"),
                    )? {
                        color.write_mask |= 1 << i;
                    }
                }
            }
        }
        Ok(colors)
    }

    /// The bound vertex stream or attribute states (`noun`); the error names
    /// why execution cannot read them. Source: the public contract.
    fn vertex_settings(
        &self,
        kind: &str,
        noun: &str,
        allowed: &[&str],
    ) -> Result<&Vec<Settings>, String> {
        let settings = match self.vertex_states.get(kind) {
            None => return Err(format!("no {noun} states are bound")),
            Some(None) => {
                return Err(format!(
                    "the bound {noun} states cannot be read: there are more than 16, the \
                     first-draw contract gives no spacing between them, or one of them is not \
                     a {noun} state"
                ))
            }
            Some(Some(settings)) => settings,
        };
        if let Some((at, (name, _))) = settings.iter().enumerate().find_map(|(at, state)| {
            state
                .iter()
                .find(|(name, _)| !allowed.contains(name))
                .map(|setting| (at, setting))
        }) {
            return Err(format!(
                "{noun} state {at} sets {name}, which swxbx does not draw with"
            ));
        }
        Ok(settings)
    }
}

fn ranges_overlap(a: u64, an: u64, b: u64, bn: u64) -> bool {
    a.checked_add(an).is_none_or(|end| b < end) && b.checked_add(bn).is_none_or(|end| a < end)
}

/// Why a draw cannot run: the status its batch stops on and the words that
/// name the fact that was wrong. Source: the public contract.
#[derive(Debug, PartialEq)]
pub(super) struct Refusal(pub Status, pub String);

impl Refusal {
    fn unimplemented(reason: impl Into<String>) -> Self {
        Self(Status::Unimplemented, reason.into())
    }
    fn bad(reason: impl Into<String>) -> Self {
        Self(Status::BadArgument, reason.into())
    }
}

/// The last value `settings` (a bound `kind`) gives `name`.
fn setting(settings: &Settings, name: &str, kind: &str) -> Result<[u64; 6], Refusal> {
    value(settings, name)
        .map_err(|_| Refusal::unimplemented(format!("the bound {kind} sets no {name}")))
}

fn boolean(v: u64, what: &str) -> Result<bool, Refusal> {
    match v {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(Refusal::unimplemented(format!("{what} is {v}, not 0 or 1"))),
    }
}
fn token(v: u64, tokens: &[u64], what: &str) -> Result<i32, Refusal> {
    let mut matches = tokens.iter().enumerate().filter(|(_, t)| **t == v);
    let (index, _) = matches.next().ok_or_else(|| {
        Refusal::unimplemented(format!(
            "{what} token {v} is not one the depth and raster contract maps"
        ))
    })?;
    if matches.next().is_some() {
        return Err(Refusal::bad(format!(
            "{what} token {v} appears more than once in the depth and raster contract"
        )));
    }
    Ok(index as i32)
}
/// A draw's depth target: its description, the image its token draws
/// with, and the bytes of memory it covers. Source: the public contract.
struct DepthTarget {
    description: super::objects::TextureDescription,
    image: crate::gpu::formats::DepthImage,
    bytes: u64,
}

/// The format a color target is drawn in: the first-draw contract's color
/// table maps its token, the image contract stores it in the same format, and
/// the device can draw into and blend it. Source: the public contract.
fn target_format(
    backend: &Backend,
    contract: &crate::gpu::graphics::FirstDrawContract,
    key: u64,
    d: &super::objects::TextureDescription,
) -> Result<vk::Format, String> {
    let format = contract.color_format(d.format).ok_or_else(|| {
        format!(
            "its format token {} is not mapped to exactly one color format by the first-draw \
             contract",
            d.format
        )
    })?;
    if backend.has_image_contract() {
        let stored = backend
            .texture_image(key, d)
            .ok_or_else(|| {
                format!(
                    "the image contract has no layout for format token {}, target {}",
                    d.format, d.target
                )
            })?
            .descriptor
            .format;
        if stored != format {
            return Err(format!(
                "the image contract stores format token {} as {}, but the first-draw \
                 contract draws it as {}",
                d.format,
                crate::gpu::formats::Name(stored),
                crate::gpu::formats::Name(format)
            ));
        }
    } else if format != vk::Format::R8G8B8A8_UNORM {
        return Err(format!(
            "format token {} draws as {}; without an image contract swxbx draws only \
             to R8G8B8A8_UNORM (VkFormat 37)",
            d.format,
            crate::gpu::formats::Name(format)
        ));
    }
    crate::gpu::formats::supports(
        backend.context(),
        format,
        crate::gpu::formats::FormatUse::ColorTarget,
    )?;
    Ok(format)
}

/// The image a depth target is drawn with, resolved when the depth and
/// raster contract was installed, and the planes of the caller's format,
/// which lay out the memory it covers. A stencil test needs a format with
/// stencil. Source: the public contract.
fn depth_format(
    backend: &Backend,
    key: u64,
    d: &super::objects::TextureDescription,
    stencil_test: bool,
) -> Result<
    (
        crate::gpu::formats::DepthImage,
        crate::gpu::formats::DepthPlanes,
    ),
    String,
> {
    let image = backend.depth_image(d.format).ok_or_else(|| {
        format!(
            "its format token {} is not mapped to exactly one depth format by the depth and \
             raster contract",
            d.format
        )
    })?;
    let planes = crate::gpu::formats::depth_target(image.guest())?;
    backend.depth_storage(key, d, image.guest())?;
    if stencil_test && planes.stencil == 0 {
        return Err(format!(
            "the draw tests stencil, and {} has no stencil",
            crate::gpu::formats::Name(image.guest())
        ));
    }
    Ok((image, planes))
}

fn face_mask(v: u64, extension: &DepthRasterContract) -> Result<u32, Refusal> {
    Ok([1, 2, 3][token(v, &extension.faces, "stencil face")? as usize])
}
fn byte(v: u64, what: &str) -> Result<u32, Refusal> {
    u8::try_from(v)
        .map(u32::from)
        .map_err(|_| Refusal::bad(format!("{what} {v} does not fit 8 bits")))
}

/// Vertical direction of a recorded viewport. the reference host's GPU layer folds guest
/// Y negation into a negative host viewport height whose origin row is the
/// region's far edge (`StateUpdater.UpdateViewportTransform`), and its Vulkan
/// backend hands that region to Vulkan unchanged (`PipelineBase.SetViewports`).
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

fn rectangle(recorded: Option<[u64; 5]>, width: u64, height: u64) -> Result<vk::Rect2D, Status> {
    let [_, x, y, w, h] = recorded.ok_or(Status::Unimplemented)?;
    if w == 0
        || h == 0
        || x.checked_add(w).is_none_or(|right| right > width)
        || y.checked_add(h).is_none_or(|bottom| bottom > height)
    {
        return Err(Status::BadArgument);
    }
    Ok(vk::Rect2D {
        offset: vk::Offset2D {
            x: i32::try_from(x).map_err(|_| Status::BadArgument)?,
            y: i32::try_from(y).map_err(|_| Status::BadArgument)?,
        },
        extent: vk::Extent2D {
            width: u32::try_from(w).map_err(|_| Status::BadArgument)?,
            height: u32::try_from(h).map_err(|_| Status::BadArgument)?,
        },
    })
}

/// One vertex attribute's fetch from its bound stream, as the bounds checks
/// and their refusals see it.
#[derive(Clone, Copy)]
struct Fetch {
    attribute: usize,
    stream: u32,
    stride: u32,
    offset: u32,
    bytes: u32,
    size: u64,
}

impl Fetch {
    /// Check `count` consecutive vertices or instances (`what`) from `first`;
    /// the error names the last one and the byte it reads past the stream's
    /// bound range. Source: the public contract.
    fn consecutive(&self, what: &str, first: u32, count: u32) -> Result<(), String> {
        let end = vertex_end(first, count, self.stride, self.offset, self.bytes);
        if end.is_some_and(|end| end <= self.size) {
            return Ok(());
        }
        let last = u64::from(first) + u64::from(count) - 1;
        Err(format!(
            "the draw reads {what} {first} to {last}; vertex attribute {} reads {} bytes at \
             offset {} of each from vertex stream {} with stride {}, {}",
            self.attribute,
            self.bytes,
            self.offset,
            self.stream,
            self.stride,
            self.past(end)
        ))
    }

    /// Check the vertices `indices` select; the error names the first index
    /// whose vertex reads past the stream's bound range. `before` counts the
    /// draw's indices before this chunk. Source: the public contract.
    fn indexed(
        &self,
        indices: &[u8],
        format: IndexFormat,
        base: i32,
        before: u64,
    ) -> Option<String> {
        let (at, index, end) =
            indices
                .chunks_exact(format.width())
                .enumerate()
                .find_map(|(at, one)| {
                    let end =
                        indexed_vertex_end(one, format, base, self.stride, self.offset, self.bytes);
                    end.is_none_or(|end| end > self.size)
                        .then(|| (at, format.read(one), end))
                })?;
        let vertex = i64::from(index) + i64::from(base);
        let selected = format!(
            "index {} of the draw has value {index}, which with base vertex {base} selects vertex \
             {vertex}",
            before + at as u64
        );
        Some(match u32::try_from(vertex) {
            Err(_) => format!("{selected}, which no vertex stream can hold"),
            Ok(_) => format!(
                "{selected}; vertex attribute {} reads {} bytes at offset {} of it from vertex \
                 stream {} with stride {}, {}",
                self.attribute,
                self.bytes,
                self.offset,
                self.stream,
                self.stride,
                self.past(end)
            ),
        })
    }

    fn past(&self, end: Option<u64>) -> String {
        match end {
            Some(end) => format!(
                "ending at byte {end}, past the {} bytes bound to the stream",
                self.size
            ),
            None => "past any addressable byte".to_string(),
        }
    }
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

/// The (bank, byte offset) locations of a bindless handle's texture and
/// sampler words: one constant-buffer word holds both; of two words or-ed,
/// the first holds the texture and the second the sampler
/// (`crate::gpu::textures::LocationMapping`). Other handles have none.
pub(super) fn handle_locations(
    handle: &crate::surfaces::BindlessHandle,
) -> Option<((u8, u16), (u8, u16))> {
    use crate::surfaces::{BindlessHandle, HandleOperation, HandleWord};
    match *handle {
        BindlessHandle::Word(HandleWord::ConstantBuffer { bank, offset }) => {
            Some(((bank, offset), (bank, offset)))
        }
        BindlessHandle::Combined {
            operation: HandleOperation::Or,
            a:
                HandleWord::ConstantBuffer {
                    bank: ta,
                    offset: to,
                },
            b:
                HandleWord::ConstantBuffer {
                    bank: sa,
                    offset: so,
                },
        } => Some(((ta, to), (sa, so))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangle_accepts_a_nonzero_origin_within_the_target() {
        let rectangle = rectangle(Some([7, 142, 85, 670, 40]), 1920, 1080).unwrap();
        assert_eq!(rectangle.offset.x, 142);
        assert_eq!(rectangle.offset.y, 85);
        assert_eq!(rectangle.extent.width, 670);
        assert_eq!(rectangle.extent.height, 40);
    }

    #[test]
    fn rectangle_accepts_an_edge_aligned_scissor() {
        let rectangle = rectangle(Some([7, 1250, 1040, 670, 40]), 1920, 1080).unwrap();
        assert_eq!(rectangle.offset.x, 1250);
        assert_eq!(rectangle.offset.y, 1040);
        assert_eq!(rectangle.extent.width, 670);
        assert_eq!(rectangle.extent.height, 40);
    }

    #[test]
    fn rectangle_rejects_empty_out_of_bounds_and_overflowing_scissors() {
        assert!(matches!(
            rectangle(Some([7, 142, 85, 0, 40]), 1920, 1080),
            Err(Status::BadArgument)
        ));
        assert!(matches!(
            rectangle(Some([7, 1251, 1040, 670, 40]), 1920, 1080),
            Err(Status::BadArgument)
        ));
        assert!(matches!(
            rectangle(Some([7, u64::MAX, 0, 2, 1]), 1920, 1080),
            Err(Status::BadArgument)
        ));
    }

    fn contract() -> DepthRasterContract {
        DepthRasterContract {
            depth_formats: vec![(90, ash::vk::Format::D32_SFLOAT_S8_UINT)],
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
        let s = state().pipeline_state(Some(&contract()), 0, true).unwrap();
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
        assert_eq!(
            s.pipeline_state(None, 0, true).map_err(|r| r.0),
            Err(Status::Unimplemented)
        );
        assert_eq!(
            s.pipeline_state(Some(&contract()), 0, false)
                .map_err(|r| r.0),
            Err(Status::Unimplemented)
        );
        s.stencil_commands.clear();
        assert_eq!(
            s.pipeline_state(Some(&contract()), 0, true)
                .map_err(|r| r.0),
            Err(Status::Unimplemented)
        );
        s.stencil_commands.push(("StencilMask", 303, 256));
        assert_eq!(
            s.pipeline_state(Some(&contract()), 0, true)
                .map_err(|r| r.0),
            Err(Status::BadArgument)
        );
        s.stencil_commands = vec![("StencilMask", 303, 255)];
        s.bias = Some([u64::from(f32::NAN.to_bits()), 0, 0]);
        assert_eq!(
            s.pipeline_state(Some(&contract()), 0, true)
                .map_err(|r| r.0),
            Err(Status::BadArgument)
        );
        s.bias = Some([0, u64::from(2.0_f32.to_bits()), 0]);
        let parsed = s.pipeline_state(Some(&contract()), 0, true).unwrap();
        assert_eq!(parsed.bias, [0, 2.0_f32.to_bits(), 0]);
        let mut c = contract();
        c.polygon_offset = false;
        assert_eq!(
            s.pipeline_state(Some(&c), 0, true).map_err(|r| r.0),
            Err(Status::Unimplemented)
        );
        assert_eq!(
            token(999, &c.compare, "depth function").map_err(|r| r.0),
            Err(Status::Unimplemented)
        );
        assert_eq!(
            token(1, &[1, 1], "depth function").map_err(|r| r.0),
            Err(Status::BadArgument)
        );
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
