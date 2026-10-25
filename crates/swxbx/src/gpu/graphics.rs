//! Bounded drawing with public Vulkan state and host choices.
use super::{
    graphics_libraries::{Compiler, Part},
    graphics_modules::{ShaderModule, ShaderModules},
    pipelines::{CacheStats, DriverCache, PersistenceStats, TranslationIdentity},
    uniforms::{self, Bank, UniformStage, BANK_SIZE},
    Context, GlobalMemory,
};
use ash::vk;
use std::{
    collections::HashMap,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

/// Opt-in host interpretation, not established guest enums.
/// Viewport/scissor use Vulkan pixel coordinates. Without the depth/raster extension, raster state uses fill,
/// one sample and all color channels. Duplicate tokens are rejected at use.
#[derive(Clone, Debug)]
pub struct FirstDrawContract {
    /// Explicit mapping from caller tokens to supported primitive topologies.
    pub topologies: Vec<(u32, PrimitiveTopology)>,
    /// Explicit mapping from caller tokens to vertex formats.
    pub attribute_formats: Vec<(u64, VertexFormat)>,
    /// Host-selected object spacing for counted bindings. None supports count one only.
    pub attribute_state_stride: Option<u64>,
    /// Host-selected object spacing for counted stream state; None permits count one.
    pub stream_state_stride: Option<u64>,
    /// Distinct host-selected tokens for tightly packed unsigned index elements.
    pub index_u16: u32,
    /// Host token for tightly packed unsigned 32-bit indices.
    pub index_u32: u32,
    /// Host token that disables culling.
    pub cull_none: u64,
    /// Host token for the supported four-byte color format.
    pub rgba8: u64,
    /// Host token for a two-dimensional render target.
    pub target_2d: u64,
    /// Raw swizzle values accepted as identity for the first draw path.
    pub identity_swizzle: [u64; 4],
    /// Optional explicit interpretation of depth, stencil, and raster state.
    pub depth_raster: Option<DepthRasterContract>,
}

/// Host-selected tokens, not guest enum claims. See
#[derive(Clone, Copy, Debug)]
pub struct DepthRasterContract {
    /// Host token for the supported depth/stencil format.
    pub d32s8: u64,
    /// Never, less, equal, less-or-equal, greater, not-equal, greater-or-equal, always.
    pub compare: [u64; 8],
    /// Keep, zero, replace, increment-clamp, decrement-clamp, invert, increment-wrap, decrement-wrap.
    pub stencil_ops: [u64; 8],
    /// Front, back, both.
    pub faces: [u64; 3],
    /// Front, back, both. No culling uses FirstDrawContract::cull_none.
    pub cull: [u64; 3],
    /// Fill, line, point.
    pub polygon: [u64; 3],
    /// No recorded front-face setter is established. This is a host choice.
    pub clockwise: bool,
    /// Opt into the d0 slope, d1 constant, d2 clamp hypothesis.
    pub polygon_offset: bool,
}

#[derive(
    Clone, Copy, Debug, Default, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
/// One face's interpreted Vulkan stencil state.
pub struct StencilState {
    /// Raw Vulkan stencil operation when the stencil test fails.
    pub fail: i32,
    /// Raw Vulkan stencil operation when the depth test fails.
    pub depth_fail: i32,
    /// Raw Vulkan stencil operation when both tests pass.
    pub pass: i32,
    /// Raw Vulkan stencil comparison operation.
    pub compare: i32,
    /// Mask applied before stencil comparison.
    pub mask: u32,
    /// Mask of components or bits that may be written.
    pub write_mask: u32,
    /// Stencil reference value.
    pub reference: u32,
}
impl StencilState {
    fn vk(self) -> vk::StencilOpState {
        vk::StencilOpState::default()
            .fail_op(vk::StencilOp::from_raw(self.fail))
            .depth_fail_op(vk::StencilOp::from_raw(self.depth_fail))
            .pass_op(vk::StencilOp::from_raw(self.pass))
            .compare_op(vk::CompareOp::from_raw(self.compare))
            .compare_mask(self.mask)
            .write_mask(self.write_mask)
            .reference(self.reference)
    }
}

/// Explicit hypotheses for blend setter argument order and channel order.
/// Factor and operation integers are public Vulkan values, not guest enums.
#[derive(Clone)]
pub struct BlendContract {
    /// Explicit mapping from caller tokens to Vulkan blend factors.
    pub factors: Vec<(u64, vk::BlendFactor)>,
    /// Explicit mapping from caller tokens to Vulkan blend operations.
    pub operations: Vec<(u64, vk::BlendOp)>,
    /// Indices in the four arguments: source color, destination color, source alpha, destination alpha.
    pub function_order: [usize; 4],
    /// Indices in the two equation arguments: color, alpha.
    pub equation_order: [usize; 2],
    /// Indices in the four channel arguments: red, green, blue, alpha.
    pub channel_order: [usize; 4],
}
impl BlendContract {
    pub(crate) fn validate(&self) -> Result<(), String> {
        for order in [
            &self.function_order[..],
            &self.equation_order[..],
            &self.channel_order[..],
        ] {
            let mut sorted = order.to_vec();
            sorted.sort_unstable();
            if sorted != (0..order.len()).collect::<Vec<_>>() {
                return Err("invalid blend argument permutation".into());
            }
        }
        if self
            .factors
            .iter()
            .any(|(_, v)| !(0..=10).contains(&v.as_raw()))
            || self
                .operations
                .iter()
                .any(|(_, v)| !(0..=4).contains(&v.as_raw()))
        {
            return Err("unsupported blend factor or operation".into());
        }
        Ok(())
    }
}

/// Interpreted blending and write mask for one color attachment.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ColorAttachmentState {
    /// Whether this operation is enabled.
    pub enable: bool,
    /// Source color, destination color, source alpha, and destination alpha factors.
    pub factors: [i32; 4],
    /// Color and alpha blend operations as raw Vulkan values.
    pub operations: [i32; 2],
    /// Mask of components or bits that may be written.
    pub write_mask: u32,
}
impl Default for ColorAttachmentState {
    fn default() -> Self {
        Self {
            enable: false,
            factors: [1, 0, 1, 0],
            operations: [0; 2],
            write_mask: 15,
        }
    }
}
impl ColorAttachmentState {
    fn vk(self) -> vk::PipelineColorBlendAttachmentState {
        vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(self.enable)
            .src_color_blend_factor(vk::BlendFactor::from_raw(self.factors[0]))
            .dst_color_blend_factor(vk::BlendFactor::from_raw(self.factors[1]))
            .src_alpha_blend_factor(vk::BlendFactor::from_raw(self.factors[2]))
            .dst_alpha_blend_factor(vk::BlendFactor::from_raw(self.factors[3]))
            .color_blend_op(vk::BlendOp::from_raw(self.operations[0]))
            .alpha_blend_op(vk::BlendOp::from_raw(self.operations[1]))
            .color_write_mask(vk::ColorComponentFlags::from_raw(self.write_mask))
    }
}

/// Interpreted pipeline state, also used for explicit cache retry.
/// Enum integers are public Vulkan values. Retry validates them before use.
#[derive(
    Clone, Copy, Debug, Default, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub struct DrawPipelineState {
    /// Required 32-lane subgroups for vertex and fragment stages, in that order.
    #[serde(default)]
    pub subgroup_size_32: [bool; 2],
    /// Binding bits selecting one element per instance.
    #[serde(default)]
    pub instance_bindings: u32,
    /// Zero selects the single-target retry contract. Otherwise 1 through 8.
    pub color_count: u32,
    /// Per-attachment blend and channel-mask state.
    pub colors: [ColorAttachmentState; 8],
    /// Whether the pipeline includes a depth/stencil attachment.
    pub attachment: bool,
    /// Whether depth comparison is enabled.
    pub depth_test: bool,
    /// Whether passing fragments write depth.
    pub depth_write: bool,
    /// Raw Vulkan depth comparison operation.
    pub depth_compare: i32,
    /// Whether stencil testing is enabled.
    pub stencil_test: bool,
    /// Front and back face stencil states.
    pub stencil: [StencilState; 2],
    /// Raw Vulkan cull-mode flags.
    pub cull: u32,
    /// Raw Vulkan polygon mode.
    pub polygon: i32,
    /// Whether clockwise winding defines the front face.
    pub clockwise: bool,
    /// Slope, constant, clamp as finite f32 bits.
    pub bias: [u32; 3],
}

/// Primitive assembly supported by the bounded draw path.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PrimitiveTopology {
    /// Each group of three vertices forms an independent triangle.
    TriangleList,
    /// Each vertex after the first two completes another triangle.
    TriangleStrip,
    /// The first vertex joins each successive pair to form a triangle.
    TriangleFan,
}

impl PrimitiveTopology {
    fn vk(self) -> vk::PrimitiveTopology {
        match self {
            Self::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            Self::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
            Self::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
        }
    }
}

/// Float-converting vertex formats from public Vulkan documentation,
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VertexFormat {
    /// One 32-bit floating-point component.
    Float,
    /// Two 32-bit floating-point components.
    Float2,
    /// Three 32-bit floating-point components.
    Float3,
    /// Four 32-bit floating-point components.
    Float4,
    /// Two 16-bit floating-point components.
    Half2,
    /// Four 16-bit floating-point components.
    Half4,
    /// Four unsigned 8-bit components normalized to zero through one.
    Unorm8x4,
    /// Four signed 8-bit components normalized to minus one through one.
    Snorm8x4,
    /// Two unsigned 16-bit components normalized to zero through one.
    Unorm16x2,
    /// Four unsigned 16-bit components normalized to zero through one.
    Unorm16x4,
    /// Two signed 16-bit components normalized to minus one through one.
    Snorm16x2,
    /// Four signed 16-bit components normalized to minus one through one.
    Snorm16x4,
}

impl VertexFormat {
    pub(crate) fn vk(self) -> vk::Format {
        match self {
            Self::Float => vk::Format::R32_SFLOAT,
            Self::Float2 => vk::Format::R32G32_SFLOAT,
            Self::Float3 => vk::Format::R32G32B32_SFLOAT,
            Self::Float4 => vk::Format::R32G32B32A32_SFLOAT,
            Self::Half2 => vk::Format::R16G16_SFLOAT,
            Self::Half4 => vk::Format::R16G16B16A16_SFLOAT,
            Self::Unorm8x4 => vk::Format::R8G8B8A8_UNORM,
            Self::Snorm8x4 => vk::Format::R8G8B8A8_SNORM,
            Self::Unorm16x2 => vk::Format::R16G16_UNORM,
            Self::Unorm16x4 => vk::Format::R16G16B16A16_UNORM,
            Self::Snorm16x2 => vk::Format::R16G16_SNORM,
            Self::Snorm16x4 => vk::Format::R16G16B16A16_SNORM,
        }
    }

    pub(crate) fn layout(self) -> (u32, u32) {
        match self {
            Self::Float => (4, 4),
            Self::Float2 => (8, 4),
            Self::Float3 => (12, 4),
            Self::Float4 => (16, 4),
            Self::Half2 | Self::Unorm16x2 | Self::Snorm16x2 => (4, 2),
            Self::Half4 | Self::Unorm16x4 | Self::Snorm16x4 => (8, 2),
            Self::Unorm8x4 | Self::Snorm8x4 => (4, 1),
        }
    }

    fn components(self) -> u32 {
        match self {
            Self::Float => 1,
            Self::Float2 | Self::Half2 | Self::Unorm16x2 | Self::Snorm16x2 => 2,
            Self::Float3 => 3,
            Self::Float4
            | Self::Half4
            | Self::Unorm8x4
            | Self::Snorm8x4
            | Self::Unorm16x4
            | Self::Snorm16x4 => 4,
        }
    }
}

pub(crate) fn mapped<T: Copy, K: PartialEq>(table: &[(K, T)], token: K) -> Option<T> {
    let mut matches = table.iter().filter(|(key, _)| *key == token);
    let result = matches.next()?.1;
    matches.next().is_none().then_some(result)
}

/// Complete vertex binding and attribute layout used in a graphics cache key.
#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VertexInput {
    /// Vertex streams in binding order.
    pub bindings: Vec<VertexBinding>,
    /// Vertex attributes in location order.
    pub attributes: Vec<VertexAttribute>,
}

/// One per-vertex stream binding.
#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VertexBinding {
    /// Zero-based vertex stream binding number.
    pub binding: u32,
    /// Distance between successive elements in bytes.
    pub stride: u32,
}

/// One attribute; its position in the attribute list determines its location.
#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VertexAttribute {
    /// Zero-based vertex stream binding number.
    pub binding: u32,
    /// Component storage format decoded by Vulkan.
    pub format: VertexFormat,
    /// Byte offset of this attribute within each vertex.
    pub offset: u32,
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub(crate) struct Key {
    pub(super) vertex: Vec<u32>,
    pub(super) fragment: Vec<u32>,
    pub(super) input: VertexInput,
    pub(super) topology: PrimitiveTopology,
    pub(super) storage: bool,
    pub(super) state: DrawPipelineState,
}

/// Host choice when a draw needs unfinished compilation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PendingDrawPolicy {
    /// Wait for translation, queue admission, compilation and linking at submission.
    #[default]
    Block,
    /// Explicitly discard unfinished draws and log each discarded draw.
    Skip,
    /// Explicitly allow discarding a draw after this wait budget, with a log entry.
    /// The host setter accepts budgets up to 16 milliseconds.
    Wait(Duration),
}

/// Successful driver creation counts, excluding cache request hits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphicsCompilationStats {
    /// Vertex input, pre-rasterization, fragment shader, and fragment output counts.
    pub library_parts: [u64; 4],
    /// Executable links, including startup links.
    pub links: u64,
    /// Whole pipelines created on devices without library support.
    pub whole: u64,
}

/// A shader module this device accepts, with its pipeline identity hashed
/// once. Draws carry stages, so a module is scanned and hashed when it is
/// first seen rather than on every draw.
pub(crate) struct Stage {
    words: Vec<u32>,
    hash: u64,
}

impl Stage {
    /// Accept a module whose instructions the device supports.
    pub(crate) fn checked(
        features: super::ShaderFeatures,
        words: Vec<u32>,
    ) -> Result<Self, &'static str> {
        features
            .validate_shader(&words)
            .map(|()| Self::accepted(words))
    }

    /// Accept a translated module whose declared and encoded requirements the
    /// device supports.
    fn translated(
        features: super::ShaderFeatures,
        shader: &crate::startup_cache::TranslatedShader,
    ) -> Result<Self, &'static str> {
        features
            .validate_translated_shader(shader)
            .map(|()| Self::accepted(shader.words.clone()))
    }

    fn accepted(words: Vec<u32>) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        words.hash(&mut hasher);
        Self {
            hash: hasher.finish(),
            words,
        }
    }

    pub(crate) fn words(&self) -> &[u32] {
        &self.words
    }
}

/// A translated module held for as long as its accepted stage is reused, so
/// its address identifies it.
type TranslatedStage = (
    Arc<crate::startup_cache::TranslatedShader>,
    Result<Arc<Stage>, &'static str>,
);

pub(crate) struct GraphicsPipelines {
    driver: Arc<DriverCache>,
    translated: HashMap<usize, TranslatedStage>,
    requires_storage: bool,
    pool: Compiler,
    policy: PendingDrawPolicy,
    ready_ids: HashMap<u64, Vec<usize>>,
    ready: Vec<(Key, Arc<GraphicsPipeline>)>,
    descriptors: HashMap<vk::Pipeline, DescriptorCache>,
}
impl GraphicsPipelines {
    /// Create reusable execution resources; return None if Vulkan setup fails.
    pub fn new(context: &Arc<Context>) -> Option<Self> {
        Self::build(context, None, 2, 64).ok()
    }
    /// Create bounded graphics workers using private host-owned persistence.
    pub fn persistent(
        context: &Arc<Context>,
        directory: &Path,
        identity: &TranslationIdentity,
        workers: usize,
        capacity: usize,
    ) -> Result<Self, String> {
        Self::build(context, Some((directory, identity)), workers, capacity)
    }
    fn build(
        context: &Arc<Context>,
        persistence: Option<(&Path, &TranslationIdentity)>,
        workers: usize,
        capacity: usize,
    ) -> Result<Self, String> {
        let driver = DriverCache::new(
            context,
            b"swxbx graphics main interface 3 disk 1",
            persistence,
        )?;
        let pool = Compiler::new(
            &driver,
            context.graphics_pipeline_libraries,
            workers,
            capacity,
        )?;
        let limits = unsafe {
            context
                .instance
                .get_physical_device_properties(context.physical_device)
        }
        .limits;
        Ok(Self {
            driver,
            pool,
            policy: PendingDrawPolicy::default(),
            ready_ids: HashMap::new(),
            translated: HashMap::new(),
            ready: Vec::new(),
            descriptors: HashMap::new(),
            requires_storage: u64::from(limits.max_uniform_buffer_range) < BANK_SIZE,
        })
    }
    /// Wait by default. Only an explicit host policy may return an unfinished request.
    #[cfg(test)]
    pub fn request(
        &mut self,
        stages: &[Vec<u32>],
        input: VertexInput,
        topology: PrimitiveTopology,
        state: DrawPipelineState,
        storage: bool,
    ) -> Result<Option<Arc<GraphicsPipeline>>, String> {
        let features = self.driver.context.shader_features;
        let stages = stages
            .iter()
            .map(|words| Stage::checked(features, words.clone()).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        self.request_stages(&stages, input, topology, state, storage)
    }

    /// The accepted stage of a translated module, checked once per module.
    pub(crate) fn translated_stage(
        &mut self,
        shader: &Arc<crate::startup_cache::TranslatedShader>,
    ) -> Result<Arc<Stage>, &'static str> {
        let features = self.driver.context.shader_features;
        self.translated
            .entry(Arc::as_ptr(shader) as usize)
            .or_insert_with(|| {
                (
                    Arc::clone(shader),
                    Stage::translated(features, shader).map(Arc::new),
                )
            })
            .1
            .clone()
    }

    /// Return the pipeline for accepted stages and draw state.
    pub(crate) fn request_stages(
        &mut self,
        stages: &[Arc<Stage>],
        input: VertexInput,
        topology: PrimitiveTopology,
        state: DrawPipelineState,
        storage: bool,
    ) -> Result<Option<Arc<GraphicsPipeline>>, String> {
        #[cfg(feature = "draw-metrics")]
        let _span = crate::draw_metrics::Span::new(0);
        use std::hash::{Hash, Hasher};
        let storage = self.uses_storage(storage);
        let mut shader_hashes = [0; 2];
        if stages.len() == 2 {
            for (hash, stage) in shader_hashes.iter_mut().zip(stages) {
                *hash = stage.hash;
            }
            shader_hashes.sort_unstable();
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (&shader_hashes, &input, topology, state, storage).hash(&mut hasher);
        let fingerprint = hasher.finish();
        if let Some(ids) = self.ready_ids.get(&fingerprint) {
            for &id in ids {
                let (key, pipeline) = &self.ready[id];
                if key.input == input
                    && key.topology == topology
                    && key.state == state
                    && key.storage == storage
                    && stages.len() == 2
                    && ((stages[0].words == key.vertex && stages[1].words == key.fragment)
                        || (stages[1].words == key.vertex && stages[0].words == key.fragment))
                {
                    self.pool.hit();
                    return Ok(Some(pipeline.clone()));
                }
            }
        }
        let words: Vec<_> = stages.iter().map(|stage| stage.words.clone()).collect();
        let key = Key::new(&words, input, topology, state, storage)?;
        let deadline = match self.policy {
            PendingDrawPolicy::Block | PendingDrawPolicy::Skip => None,
            PendingDrawPolicy::Wait(budget) => Some(Instant::now() + budget),
        };
        let pipeline = self.pool.request(
            key.clone(),
            deadline,
            self.policy == PendingDrawPolicy::Block,
        )?;
        if let Some(pipeline) = &pipeline {
            let id = self.ready.len();
            self.ready.push((key, pipeline.clone()));
            self.ready_ids.entry(fingerprint).or_default().push(id);
        }
        Ok(pipeline)
    }
    pub(crate) fn set_policy(&mut self, policy: PendingDrawPolicy) -> Result<(), String> {
        if matches!(policy, PendingDrawPolicy::Wait(budget) if budget > Duration::from_millis(16)) {
            return Err("draw wait budget exceeds 16 milliseconds".into());
        }
        self.policy = policy;
        Ok(())
    }
    pub(crate) fn log_skipped_draw(&self, reason: &str) {
        let message = format!("skipped draw: {reason}");
        eprintln!("{message}");
        self.driver.diagnostic(message);
    }

    pub(crate) fn policy(&self) -> PendingDrawPolicy {
        self.policy
    }

    pub(crate) fn queue_startup(
        &mut self,
        stages: &[&crate::startup_cache::TranslatedShader],
        recipe: &crate::startup_cache::PipelineRecipe,
    ) -> Result<bool, String> {
        let mut state = recipe.state;
        state.subgroup_size_32 = [false; 2];
        for stage in stages {
            self.driver
                .context
                .shader_features
                .validate_translated_shader(stage)?;
            if stage.requires_subgroup_size_32 {
                let index = match execution_model(&stage.words)? {
                    0 => 0,
                    4 => 1,
                    _ => return Err("unsupported graphics stage".into()),
                };
                state.subgroup_size_32[index] = true;
            }
        }
        let words: Vec<_> = stages.iter().map(|s| s.words.clone()).collect();
        let key = Key::new(
            &words,
            recipe.input.clone(),
            recipe.topology,
            state,
            self.uses_storage(recipe.storage),
        )?;
        self.pool.queue(key)
    }

    pub(super) fn invalidate_descriptors(&mut self) {
        self.descriptors.clear();
    }

    pub(crate) fn descriptors(
        &mut self,
        pipeline: &GraphicsPipeline,
        buffers: &[vk::DescriptorBufferInfo],
        images: &[vk::DescriptorImageInfo],
    ) -> Result<Arc<DrawDescriptors>, String> {
        pipeline.descriptors(
            buffers,
            images,
            self.descriptors.entry(pipeline.pipeline).or_default(),
        )
    }

    /// Forget a failed graphics request so a later request can compile again.
    pub fn retry_failed(
        &mut self,
        stages: &[Vec<u32>],
        input: VertexInput,
        topology: PrimitiveTopology,
        state: DrawPipelineState,
        storage: bool,
    ) -> Result<bool, String> {
        Ok(self.pool.retry_failed(&Key::new(
            stages,
            input,
            topology,
            state,
            self.uses_storage(storage),
        )?))
    }
    /// Whether the host requested storage banks or device uniform limits require them.
    pub fn uses_storage(&self, requested: bool) -> bool {
        requested || self.requires_storage
    }
    pub(crate) fn compilation_stats(&self) -> GraphicsCompilationStats {
        self.pool.compilation_stats()
    }
    /// Return cumulative request reuse and insertion counters.
    pub fn stats(&self) -> CacheStats {
        self.pool.stats()
    }
    /// Number of queued or compiling graphics requests.
    pub fn pending(&mut self) -> usize {
        self.pool.pending()
    }
    /// Return persistent cache loading and reuse counters.
    pub fn persistence_stats(&self) -> PersistenceStats {
        self.driver.persistence_stats()
    }
    /// Drain accumulated compilation and persistence diagnostics.
    pub fn take_diagnostics(&self) -> Vec<String> {
        self.driver.take_diagnostics()
    }
}
impl Key {
    pub(super) fn validate_interfaces(&self) -> Result<(), String> {
        let inputs = float_interface(&self.vertex, 1)?;
        let vertex_outputs = float_interface(&self.vertex, 3)?;
        let fragment_inputs = float_interface(&self.fragment, 1)?;
        if inputs.is_empty()
            || inputs.iter().any(|&(location, components)| {
                self.input
                    .attributes
                    .get(location as usize)
                    .is_none_or(|attribute| attribute.format.components() > components)
            })
            || fragment_inputs
                .iter()
                .any(|input| !vertex_outputs.contains(input))
            || float_interface(&self.fragment, 3)?
                != (0..self.state.color_count.max(1))
                    .map(|i| (i, 4))
                    .collect::<Vec<_>>()
        {
            return Err(format!(
                "incompatible graphics interfaces: shader_inputs={inputs:?} bound_attributes={:?} vertex_outputs={vertex_outputs:?} fragment_inputs={fragment_inputs:?} fragment_outputs={:?} color_count={}",
                self.input.attributes,
                float_interface(&self.fragment, 3)?,
                self.state.color_count,
            ));
        }
        Ok(())
    }

    pub(crate) fn new(
        stages: &[Vec<u32>],
        input: VertexInput,
        topology: PrimitiveTopology,
        state: DrawPipelineState,
        storage: bool,
    ) -> Result<Self, String> {
        if state.color_count > 8
            || state.colors.iter().any(|c| {
                c.write_mask > 15
                    || c.factors.iter().any(|v| !(0..=10).contains(v))
                    || c.operations.iter().any(|v| !(0..=4).contains(v))
            })
            || !(0..=7).contains(&state.depth_compare)
            || state.stencil.iter().any(|s| {
                !(0..=7).contains(&s.compare)
                    || [s.fail, s.depth_fail, s.pass]
                        .iter()
                        .any(|op| !(0..=7).contains(op))
            })
            || state.cull > 3
            || !(0..=2).contains(&state.polygon)
            || state
                .bias
                .iter()
                .any(|bits| !f32::from_bits(*bits).is_finite())
        {
            return Err("invalid interpreted depth or raster state".into());
        }
        if stages.len() != 2 {
            return Err("first draw requires exactly two translated stages".into());
        }
        let mut vertex = None;
        let mut fragment = None;
        for words in stages {
            let model = execution_model(words)?;
            let slot = match model {
                0 => &mut vertex,
                4 => &mut fragment,
                _ => return Err("unsupported graphics stage".into()),
            };
            if slot.replace(words).is_some() {
                return Err("duplicate graphics stage".into());
            }
        }
        let vertex = vertex.ok_or("missing vertex stage")?;
        let fragment = fragment.ok_or("missing fragment stage")?;
        Ok(Self {
            vertex: vertex.clone(),
            fragment: fragment.clone(),
            input,
            topology,
            storage,
            state,
        })
    }
}

pub(crate) struct Draw {
    /// Vertex binding number, buffer handle, and byte offset tuples.
    pub buffers: Vec<(u32, vk::Buffer, u64)>,
    /// Dynamic Vulkan viewport in pixel coordinates.
    pub viewport: vk::Viewport,
    /// Dynamic Vulkan scissor in pixel coordinates.
    pub scissor: vk::Rect2D,
    /// Immutable descriptors shared by draws with identical resource ranges.
    pub descriptors: Arc<DrawDescriptors>,
    /// Guest images bound as storage images, stored back after the draw.
    pub storage_images: Vec<u64>,
}

#[derive(Clone, Copy)]
pub(crate) enum DrawVertices {
    Arrays {
        first: u32,
    },
    Elements {
        buffer: vk::Buffer,
        offset: u64,
        index_type: vk::IndexType,
        base_vertex: i32,
    },
}

pub(super) fn instructions(words: &[u32]) -> Result<Vec<(u32, &[u32])>, String> {
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err("invalid SPIR-V header".into());
    }
    let mut result = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 || count > words.len() - at {
            return Err("invalid SPIR-V instruction length".into());
        }
        result.push((words[at] & 0xffff, &words[at + 1..at + count]));
        at += count;
    }
    Ok(result)
}

pub(crate) fn execution_model(words: &[u32]) -> Result<u32, String> {
    let entries: Vec<_> = instructions(words)?
        .into_iter()
        .filter(|(op, _)| *op == 15)
        .collect();
    match entries.as_slice() {
        [(15, operands)] if operands.len() >= 4 => Ok(operands[0]),
        _ => Err("expected one graphics entry point".into()),
    }
}

// Inspect scalar/vector float interfaces from public types and decorations.
fn float_interface(words: &[u32], storage: u32) -> Result<Vec<(u32, u32)>, String> {
    let instructions = instructions(words)?;
    let mut types = HashMap::new();
    let mut locations = HashMap::new();
    let mut builtins = Vec::new();
    for &(op, args) in &instructions {
        match (op, args) {
            (19..=33, [id, ..]) => {
                types.insert(*id, (op, args));
            }
            (71, [id, 30, location]) => {
                locations.insert(*id, *location);
            }
            (71, [id, 11, _]) => builtins.push(*id),
            (72, [id, _, 11, _]) => builtins.push(*id),
            (71, [_, 31 | 32, _]) => {
                return Err("component/index interface decorations are unsupported".into())
            }
            _ => {}
        }
    }
    let mut result = Vec::new();
    for &(op, args) in &instructions {
        if op != 59 || args.len() < 3 || args[2] != storage {
            continue;
        }
        let Some(&(32, [_, actual, ty])) = types.get(&args[0]) else {
            return Err("invalid interface pointer".into());
        };
        if *actual != storage {
            return Err("interface storage mismatch".into());
        }
        if builtins.contains(&args[1]) || builtins.contains(ty) {
            if execution_model(words)? == 4 && storage == 3 {
                return Err("fragment builtin output is unsupported".into());
            }
            continue;
        }
        let (scalar, components) = match types.get(ty) {
            Some((23, [_, scalar, components @ 2..=4])) => (scalar, *components),
            Some((22, [_, 32])) => (ty, 1),
            _ => return Err("draw requires scalar/vector float interfaces".into()),
        };
        if !matches!(types.get(scalar), Some((22, [_, 32]))) {
            return Err("first draw requires 32-bit floats".into());
        }
        let location = *locations
            .get(&args[1])
            .ok_or("missing interface location")?;
        if result.iter().any(|&(at, _)| at == location) {
            return Err("duplicate interface location".into());
        }
        result.push((location, components));
    }
    result.sort_unstable();
    Ok(result)
}

pub(crate) struct GraphicsPipeline {
    context: Arc<Context>,
    /// Compatible render pass owned by the graphics pipeline.
    pub render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layouts: Vec<vk::DescriptorSetLayout>,
    /// Reflected uniform banks required by the translated stages.
    pub banks: Vec<Bank>,
    /// Reflected sampled image and sampler bindings required by the stages.
    pub textures: Vec<super::textures::Binding>,
    /// Whether banks use storage buffers.
    pub storage: bool,
    pub batch_safe: bool,
    global_writes: bool,
    writes_banks: bool,
    modules: Vec<Arc<ShaderModule>>,
    push_set: Option<u32>,
    resource_owner: Option<Arc<GraphicsPipeline>>,
}

impl GraphicsPipeline {
    pub(super) fn create(
        context: &Arc<Context>,
        cache: vk::PipelineCache,
        key: &Key,
        part: Option<Part>,
        modules: &ShaderModules,
    ) -> Result<Self, String> {
        let fixed = matches!(part, Some(Part::Input | Part::Output));
        let mut banks = Vec::new();
        let mut textures = Vec::new();
        if !fixed {
            context.shader_features.validate_shader(&key.vertex)?;
            context.shader_features.validate_shader(&key.fragment)?;
            key.validate_interfaces()?;
            banks = uniforms::banks(&key.vertex, UniformStage::Vertex)?;
            banks.extend(uniforms::banks(&key.fragment, UniformStage::Fragment)?);
            textures = super::textures::bindings(&key.vertex, UniformStage::Vertex)?;
            textures.extend(super::textures::bindings(
                &key.fragment,
                UniformStage::Fragment,
            )?);
        }
        let color_count = key.state.color_count.max(1);
        let limits = unsafe {
            context
                .instance
                .get_physical_device_properties(context.physical_device)
        }
        .limits;
        if color_count > limits.max_color_attachments {
            return Err("too many color attachments".into());
        }
        if key.input.attributes.len() > limits.max_vertex_input_attributes as usize {
            return Err("too many vertex attributes".into());
        }
        for binding in &key.input.bindings {
            if binding.binding >= 32
                || binding.binding >= limits.max_vertex_input_bindings
                || binding.stride > limits.max_vertex_input_binding_stride
            {
                return Err("vertex binding exceeds device limits".into());
            }
        }
        for attribute in &key.input.attributes {
            if attribute.offset > limits.max_vertex_input_attribute_offset {
                return Err("vertex attribute exceeds device limits".into());
            }
            let supported = unsafe {
                context.instance.get_physical_device_format_properties(
                    context.physical_device,
                    attribute.format.vk(),
                )
            };
            if !supported
                .buffer_features
                .contains(vk::FormatFeatureFlags::VERTEX_BUFFER)
            {
                return Err("vertex format is unavailable".into());
            }
        }
        let (per_stage, total, range) = if key.storage {
            (
                limits.max_per_stage_descriptor_storage_buffers,
                limits.max_descriptor_set_storage_buffers,
                limits.max_storage_buffer_range,
            )
        } else {
            (
                limits.max_per_stage_descriptor_uniform_buffers,
                limits.max_descriptor_set_uniform_buffers,
                limits.max_uniform_buffer_range,
            )
        };
        if !banks.is_empty()
            && (u64::from(range) < BANK_SIZE
                || limits.max_bound_descriptor_sets < 2
                || banks.len() > total as usize
                || [UniformStage::Vertex, UniformStage::Fragment]
                    .iter()
                    .any(|stage| {
                        let count = banks.iter().filter(|b| b.stage == *stage).count();
                        count > per_stage as usize
                            || count > limits.max_per_stage_resources as usize
                    }))
        {
            return Err("constant banks exceed device descriptor limits".into());
        }
        let of = |ty: vk::DescriptorType| textures.iter().filter(move |b| b.ty == ty);
        let features = context.shader_features;
        if of(vk::DescriptorType::STORAGE_IMAGE).any(|b| match b.stage {
            UniformStage::Vertex => !features.vertex_pipeline_stores,
            UniformStage::Fragment => !features.fragment_stores,
        }) {
            return Err("storage images in draw stages are unavailable".into());
        }
        if of(vk::DescriptorType::SAMPLED_IMAGE).count()
            > limits.max_descriptor_set_sampled_images as usize
            || of(vk::DescriptorType::SAMPLER).count() > limits.max_descriptor_set_samplers as usize
            || of(vk::DescriptorType::STORAGE_IMAGE).count()
                > limits.max_descriptor_set_storage_images as usize
            || [UniformStage::Vertex, UniformStage::Fragment]
                .iter()
                .any(|stage| {
                    let count = |ty| of(ty).filter(|b| b.stage == *stage).count();
                    let images = count(vk::DescriptorType::SAMPLED_IMAGE);
                    let samplers = count(vk::DescriptorType::SAMPLER);
                    let storage = count(vk::DescriptorType::STORAGE_IMAGE);
                    let banks = banks.iter().filter(|b| b.stage == *stage).count();
                    images > limits.max_per_stage_descriptor_sampled_images as usize
                        || samplers > limits.max_per_stage_descriptor_samplers as usize
                        || storage > limits.max_per_stage_descriptor_storage_images as usize
                        || banks + images + samplers + storage
                            > limits.max_per_stage_resources as usize
                })
        {
            return Err("texture descriptors exceed device limits".into());
        }
        if key.state.colors[..color_count as usize]
            .iter()
            .any(|c| c.enable)
            && !unsafe {
                context.instance.get_physical_device_format_properties(
                    context.physical_device,
                    vk::Format::R8G8B8A8_UNORM,
                )
            }
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND)
        {
            return Err("color attachment blending is unavailable".into());
        }
        let features = unsafe {
            context
                .instance
                .get_physical_device_features(context.physical_device)
        };
        if color_count > 1
            && features.independent_blend == 0
            && key.state.colors[..color_count as usize]
                .iter()
                .any(|c| *c != key.state.colors[0])
        {
            return Err("independent blend state is unavailable".into());
        }
        if key.state.polygon != 0 && features.fill_mode_non_solid == 0 {
            return Err("non-solid polygon modes are unavailable".into());
        }
        if f32::from_bits(key.state.bias[2]) != 0.0 && features.depth_bias_clamp == 0 {
            return Err("depth bias clamp is unavailable".into());
        }
        #[cfg(feature = "draw-metrics")]
        let layout_span = crate::draw_metrics::PipelineSpan::new(1);
        let mut result = Self {
            context: context.clone(),
            render_pass: vk::RenderPass::null(),
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            set_layouts: Vec::new(),
            modules: Vec::new(),
            push_set: None,
            resource_owner: None,
            writes_banks: key.storage
                && (super::memory::storage_writes(&key.vertex)
                    || super::memory::storage_writes(&key.fragment)),
            global_writes: super::memory::physical_writes(&key.vertex)
                || super::memory::physical_writes(&key.fragment),
            batch_safe: fixed
                || [&key.vertex, &key.fragment].iter().all(|words| {
                    instructions(words).is_ok_and(|ops| {
                        !ops.iter()
                            .any(|&(op, args)| op == 32 && args.get(1) == Some(&5349))
                    })
                }),
            banks,
            textures,
            storage: key.storage,
        };
        // SAFETY: validated public Vulkan states, with owned partial handles.
        unsafe {
            let mut attachments = vec![
                vk::AttachmentDescription::default()
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
                color_count as usize
            ];
            if key.state.attachment {
                attachments.push(
                    vk::AttachmentDescription::default()
                        .format(vk::Format::D32_SFLOAT_S8_UINT)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::LOAD)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .stencil_load_op(vk::AttachmentLoadOp::LOAD)
                        .stencil_store_op(vk::AttachmentStoreOp::STORE)
                        .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
                );
            }
            let colors: Vec<_> = (0..color_count)
                .map(|i| {
                    vk::AttachmentReference::default()
                        .attachment(i)
                        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                })
                .collect();
            let depth_ref = vk::AttachmentReference::default()
                .attachment(color_count)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            let mut subpass = vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&colors);
            if key.state.attachment {
                subpass = subpass.depth_stencil_attachment(&depth_ref);
            }
            let subpasses = [subpass];
            result.render_pass = context
                .device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&attachments)
                        .subpasses(&subpasses),
                    None,
                )
                .map_err(|e| format!("create draw render pass: {e:?}"))?;
            if !result.banks.is_empty() || !result.textures.is_empty() {
                if limits.max_bound_descriptor_sets < 4 {
                    return Err("four draw descriptor sets are unavailable".into());
                }
                if context.push_descriptors.is_some() {
                    result.push_set = (0..4)
                        .filter_map(|set| {
                            let count =
                                result.banks.iter().filter(|b| b.stage.set() == set).count()
                                    + result
                                        .textures
                                        .iter()
                                        .filter(|b| b.stage_set() == set)
                                        .count();
                            (count != 0 && count <= context.max_push_descriptors.min(64) as usize)
                                .then_some((set, count))
                        })
                        .max_by_key(|&(_, count)| count)
                        .map(|(set, _)| set);
                }
                for set in 0..4 {
                    let stage = if set % 2 == 0 {
                        UniformStage::Vertex
                    } else {
                        UniformStage::Fragment
                    };
                    let bindings: Vec<_> =
                        result
                            .banks
                            .iter()
                            .filter(|b| b.stage == stage && set < 2)
                            .map(|b| {
                                vk::DescriptorSetLayoutBinding::default()
                                    .binding(b.bank)
                                    .descriptor_type(result.descriptor_type())
                                    .descriptor_count(1)
                                    .stage_flags(stage.flags())
                            })
                            .chain(result.textures.iter().filter(|b| b.stage_set() == set).map(
                                |b| {
                                    vk::DescriptorSetLayoutBinding::default()
                                        .binding(b.lowered())
                                        .descriptor_type(b.ty)
                                        .descriptor_count(1)
                                        .stage_flags(stage.flags())
                                },
                            ))
                            .collect();
                    let info = vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&bindings)
                        .flags(if result.push_set == Some(set) {
                            vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR
                        } else {
                            vk::DescriptorSetLayoutCreateFlags::empty()
                        });
                    let mut support = vk::DescriptorSetLayoutSupport::default();
                    context
                        .device
                        .get_descriptor_set_layout_support(&info, &mut support);
                    if support.supported == vk::FALSE {
                        return Err("constant bank layout is unavailable".into());
                    }
                    result.set_layouts.push(
                        context
                            .device
                            .create_descriptor_set_layout(&info, None)
                            .map_err(|e| format!("create constant bank layout: {e:?}"))?,
                    );
                }
            }
            let ranges = [GlobalMemory::push_constant_range(
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            )];
            result.layout = context
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&result.set_layouts)
                        .push_constant_ranges(&ranges),
                    None,
                )
                .map_err(|e| format!("create draw layout: {e:?}"))?;
        }
        #[cfg(feature = "draw-metrics")]
        drop(layout_span);
        let mut subgroup_sizes = [vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(32); 2];
        let mut stages = Vec::new();
        for (index, ((words, stage, subset), subgroup_size)) in [
            (&key.vertex, UniformStage::Vertex, Part::Raster),
            (&key.fragment, UniformStage::Fragment, Part::Fragment),
        ]
        .into_iter()
        .zip(subgroup_sizes.iter_mut())
        .enumerate()
        {
            if part.is_some_and(|p| p != subset) {
                continue;
            }
            let module = modules.get(context, words, stage, key.storage)?;
            let mut info = vk::PipelineShaderStageCreateInfo::default()
                .stage(stage.flags())
                .module(module.handle)
                .name(c"main");
            if key.state.subgroup_size_32[index] {
                context.shader_features.subgroups.validate(stage.flags())?;
                info = info.push_next(subgroup_size);
            }
            stages.push(info);
            result.modules.push(module);
        }
        let bindings: Vec<_> = key
            .input
            .bindings
            .iter()
            .map(|b| {
                vk::VertexInputBindingDescription::default()
                    .binding(b.binding)
                    .stride(b.stride)
                    .input_rate(if key.state.instance_bindings & (1 << b.binding) != 0 {
                        vk::VertexInputRate::INSTANCE
                    } else {
                        vk::VertexInputRate::VERTEX
                    })
            })
            .collect();
        let attributes: Vec<_> = key
            .input
            .attributes
            .iter()
            .enumerate()
            .map(|(i, a)| {
                vk::VertexInputAttributeDescription::default()
                    .binding(a.binding)
                    .location(i as u32)
                    .format(a.format.vk())
                    .offset(a.offset)
            })
            .collect();
        let input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let assembly =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(key.topology.vk());
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::from_raw(key.state.polygon))
            .cull_mode(vk::CullModeFlags::from_raw(key.state.cull))
            .front_face(if key.state.clockwise {
                vk::FrontFace::CLOCKWISE
            } else {
                vk::FrontFace::COUNTER_CLOCKWISE
            })
            .depth_bias_enable(key.state.bias != [0; 3])
            .depth_bias_slope_factor(f32::from_bits(key.state.bias[0]))
            .depth_bias_constant_factor(f32::from_bits(key.state.bias[1]))
            .depth_bias_clamp(f32::from_bits(key.state.bias[2]))
            .line_width(1.0);
        let samples = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(key.state.depth_test)
            .depth_write_enable(key.state.depth_write)
            .depth_compare_op(vk::CompareOp::from_raw(key.state.depth_compare))
            .stencil_test_enable(key.state.stencil_test)
            .front(key.state.stencil[0].vk())
            .back(key.state.stencil[1].vk());
        let colors: Vec<_> = key.state.colors[..color_count as usize]
            .iter()
            .map(|c| c.vk())
            .collect();
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&colors);
        let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamics = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&samples)
            .depth_stencil_state(&depth)
            .color_blend_state(&blend)
            .dynamic_state(&dynamics)
            .layout(result.layout)
            .render_pass(result.render_pass);
        let mut subset = vk::GraphicsPipelineLibraryCreateInfoEXT::default();
        if let Some(part) = part {
            subset.flags = part.flags();
            info = info
                .flags(vk::PipelineCreateFlags::LIBRARY_KHR)
                .push_next(&mut subset);
        }
        #[cfg(feature = "draw-metrics")]
        let driver_span =
            crate::draw_metrics::PipelineSpan::new(part.map_or(3, |p| 3 + p as usize));
        let created = unsafe {
            context
                .device
                .create_graphics_pipelines(cache, &[info], None)
        };
        #[cfg(feature = "draw-metrics")]
        drop(driver_span);
        match created {
            Ok(pipelines) => result.pipeline = pipelines[0],
            Err((pipelines, error)) => {
                unsafe {
                    for pipeline in pipelines {
                        context.device.destroy_pipeline(pipeline, None);
                    }
                }
                return Err(format!("create graphics pipeline: {error:?}"));
            }
        }
        Ok(result)
    }

    pub(super) fn link(key: &Key, parts: &[Arc<Self>]) -> Result<Self, String> {
        #[cfg(feature = "draw-metrics")]
        let _span = crate::draw_metrics::PipelineSpan::new(7);
        let mut banks = uniforms::banks(&key.vertex, UniformStage::Vertex)?;
        banks.extend(uniforms::banks(&key.fragment, UniformStage::Fragment)?);
        let mut textures = super::textures::bindings(&key.vertex, UniformStage::Vertex)?;
        textures.extend(super::textures::bindings(
            &key.fragment,
            UniformStage::Fragment,
        )?);
        let owner = &parts[1];
        let handles: Vec<_> = parts.iter().map(|p| p.pipeline).collect();
        let mut libraries = vk::PipelineLibraryCreateInfoKHR::default().libraries(&handles);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .layout(owner.layout)
            .render_pass(owner.render_pass)
            .push_next(&mut libraries);
        // Worker links do not acquire the compilation cache or perform disk I/O.
        let created = unsafe {
            owner
                .context
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        };
        let pipeline = match created {
            Ok(values) => values[0],
            Err((values, error)) => {
                unsafe {
                    for value in values {
                        owner.context.device.destroy_pipeline(value, None);
                    }
                }
                return Err(format!("link graphics pipeline: {error:?}"));
            }
        };
        Ok(Self {
            context: owner.context.clone(),
            render_pass: owner.render_pass,
            pipeline,
            layout: owner.layout,
            set_layouts: owner.set_layouts.clone(),
            banks,
            textures,
            storage: key.storage,
            writes_banks: key.storage
                && (super::memory::storage_writes(&key.vertex)
                    || super::memory::storage_writes(&key.fragment)),
            global_writes: super::memory::physical_writes(&key.vertex)
                || super::memory::physical_writes(&key.fragment),
            batch_safe: [&key.vertex, &key.fragment].iter().all(|words| {
                instructions(words).is_ok_and(|ops| {
                    !ops.iter()
                        .any(|&(op, args)| op == 32 && args.get(1) == Some(&5349))
                })
            }),
            modules: Vec::new(),
            push_set: owner.push_set,
            resource_owner: Some(owner.clone()),
        })
    }

    fn descriptor_type(&self) -> vk::DescriptorType {
        if self.storage {
            vk::DescriptorType::STORAGE_BUFFER
        } else {
            vk::DescriptorType::UNIFORM_BUFFER
        }
    }

    /// Allocate and populate descriptors for reflected banks and sampled resources.
    fn descriptors(
        &self,
        buffers: &[vk::DescriptorBufferInfo],
        images: &[vk::DescriptorImageInfo],
        cache: &mut DescriptorCache,
    ) -> Result<Arc<DrawDescriptors>, String> {
        #[cfg(feature = "draw-metrics")]
        let _span = crate::draw_metrics::Span::new(1);
        if buffers.len() != self.banks.len() || images.len() != self.textures.len() {
            return Err("missing constant bank ranges".into());
        }
        use ash::vk::Handle;
        let key = DescriptorKey {
            buffers: buffers
                .iter()
                .map(|b| (b.buffer.as_raw(), b.offset, b.range))
                .collect(),
            images: images
                .iter()
                .map(|i| {
                    (
                        i.sampler.as_raw(),
                        i.image_view.as_raw(),
                        i.image_layout.as_raw(),
                    )
                })
                .collect(),
        };
        if let Some(&id) = cache.ids.get(&key) {
            return Ok(cache.entries[id].clone());
        }
        if cache.entries.len() == 256 {
            cache.ids.clear();
            cache.entries.clear();
        }
        let mut result = DrawDescriptors {
            context: self.context.clone(),
            pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            buffers: buffers.to_vec(),
            images: images.to_vec(),
        };
        if self.banks.is_empty() && self.textures.is_empty() {
            return Ok(Arc::new(result));
        }
        unsafe {
            let mut sizes = Vec::new();
            for (ty, count) in [
                (self.descriptor_type(), buffers.len()),
                (
                    vk::DescriptorType::SAMPLED_IMAGE,
                    self.textures
                        .iter()
                        .filter(|b| b.ty == vk::DescriptorType::SAMPLED_IMAGE)
                        .count(),
                ),
                (
                    vk::DescriptorType::SAMPLER,
                    self.textures
                        .iter()
                        .filter(|b| b.ty == vk::DescriptorType::SAMPLER)
                        .count(),
                ),
                (
                    vk::DescriptorType::STORAGE_IMAGE,
                    self.textures
                        .iter()
                        .filter(|b| b.ty == vk::DescriptorType::STORAGE_IMAGE)
                        .count(),
                ),
            ] {
                if count != 0 {
                    sizes.push(
                        vk::DescriptorPoolSize::default()
                            .ty(ty)
                            .descriptor_count(count as u32),
                    );
                }
            }
            #[cfg(feature = "draw-metrics")]
            crate::draw_metrics::count(1);
            result.pool = self
                .context
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(4)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(|e| format!("create draw descriptor pool: {e:?}"))?;
            result.sets = vec![vk::DescriptorSet::null(); self.set_layouts.len()];
            for (set, &layout) in self.set_layouts.iter().enumerate() {
                if self.push_set == Some(set as u32) {
                    continue;
                }
                result.sets[set] = self
                    .context
                    .device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(result.pool)
                            .set_layouts(std::slice::from_ref(&layout)),
                    )
                    .map_err(|e| format!("allocate draw descriptors: {e:?}"))?[0];
            }
            let mut writes: Vec<_> = self
                .banks
                .iter()
                .zip(buffers)
                .map(|(bank, buffer)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(result.sets[bank.stage.set() as usize])
                        .dst_binding(bank.bank)
                        .descriptor_type(self.descriptor_type())
                        .buffer_info(std::slice::from_ref(buffer))
                })
                .collect();
            writes.extend(self.textures.iter().zip(images).map(|(b, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(result.sets[b.stage_set() as usize])
                    .dst_binding(b.lowered())
                    .descriptor_type(b.ty)
                    .image_info(std::slice::from_ref(info))
            }));
            writes.retain(|w| w.dst_set != vk::DescriptorSet::null());
            #[cfg(feature = "draw-metrics")]
            if !writes.is_empty() {
                crate::draw_metrics::count(0);
            }
            self.context.device.update_descriptor_sets(&writes, &[]);
        }
        let result = Arc::new(result);
        let id = cache.entries.len();
        cache.entries.push(result.clone());
        cache.ids.insert(key, id);
        Ok(result)
    }

    pub(crate) fn write_buffers<'a>(&self, draw: &'a Draw) -> &'a [vk::DescriptorBufferInfo] {
        if self.writes_banks {
            &draw.descriptors.buffers
        } else {
            &[]
        }
    }

    pub(crate) fn global_writes(&self) -> bool {
        self.global_writes
    }

    /// Caller owns the active compatible render pass, live arena slice and completion.
    pub unsafe fn record(&self, command: vk::CommandBuffer, memory: &GlobalMemory, draw: &Draw) {
        #[cfg(feature = "draw-metrics")]
        crate::draw_metrics::count(2);
        let device = &self.context.device;
        device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        for &(binding, buffer, offset) in &draw.buffers {
            device.cmd_bind_vertex_buffers(command, binding, &[buffer], &[offset]);
        }
        for (set, &handle) in draw.descriptors.sets.iter().enumerate() {
            if handle != vk::DescriptorSet::null() {
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    set as u32,
                    &[handle],
                    &[],
                );
            }
        }
        if let Some(set) = self.push_set {
            #[cfg(feature = "draw-metrics")]
            crate::draw_metrics::count(4);
            let mut writes = [vk::WriteDescriptorSet::default(); 64];
            let mut count = 0;
            for (bank, buffer) in self.banks.iter().zip(&draw.descriptors.buffers) {
                if bank.stage.set() == set {
                    writes[count] = vk::WriteDescriptorSet::default()
                        .dst_binding(bank.bank)
                        .descriptor_type(self.descriptor_type())
                        .buffer_info(std::slice::from_ref(buffer));
                    count += 1;
                }
            }
            for (binding, image) in self.textures.iter().zip(&draw.descriptors.images) {
                if binding.stage_set() == set {
                    writes[count] = vk::WriteDescriptorSet::default()
                        .dst_binding(binding.lowered())
                        .descriptor_type(binding.ty)
                        .image_info(std::slice::from_ref(image));
                    count += 1;
                }
            }
            self.context
                .push_descriptors
                .as_ref()
                .unwrap()
                .cmd_push_descriptor_set(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    set,
                    &writes[..count],
                );
        }
        device.cmd_set_viewport(command, 0, &[draw.viewport]);
        device.cmd_set_scissor(command, 0, &[draw.scissor]);
        memory.push_delta(
            command,
            self.layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        );
    }

    /// Record geometry after the retained draw state has been bound.
    pub unsafe fn record_vertices(
        &self,
        command: vk::CommandBuffer,
        vertices: DrawVertices,
        count: u32,
        bind_index: bool,
        geometry: super::operations::Geometry,
    ) {
        let device = &self.context.device;
        match vertices {
            DrawVertices::Arrays { first } => {
                if let Some((buffer, offset)) = geometry.indirect {
                    device.cmd_draw_indirect(command, buffer, offset, 1, 16);
                } else {
                    device.cmd_draw(
                        command,
                        count,
                        geometry.instances,
                        first,
                        geometry.first_instance,
                    );
                }
            }
            DrawVertices::Elements {
                buffer,
                offset,
                index_type,
                base_vertex,
            } => {
                if bind_index {
                    device.cmd_bind_index_buffer(
                        command,
                        buffer,
                        offset - geometry.index_skip,
                        index_type,
                    );
                }
                if let Some((buffer, offset)) = geometry.indirect {
                    device.cmd_draw_indexed_indirect(command, buffer, offset, 1, 20);
                } else {
                    device.cmd_draw_indexed(
                        command,
                        count,
                        geometry.instances,
                        0,
                        base_vertex,
                        geometry.first_instance,
                    );
                }
            }
        }
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_pipeline(self.pipeline, None);
            if self.resource_owner.is_some() {
                return;
            }
            self.context
                .device
                .destroy_pipeline_layout(self.layout, None);
            self.context
                .device
                .destroy_render_pass(self.render_pass, None);
            for layout in &self.set_layouts {
                self.context
                    .device
                    .destroy_descriptor_set_layout(*layout, None);
            }
        }
    }
}

#[derive(Hash, PartialEq, Eq)]
struct DescriptorKey {
    buffers: Vec<(u64, u64, u64)>,
    images: Vec<(u64, u64, i32)>,
}
#[derive(Default)]
struct DescriptorCache {
    ids: HashMap<DescriptorKey, usize>,
    entries: Vec<Arc<DrawDescriptors>>,
}
pub(crate) struct PendingDraw {
    pub geometry: super::operations::Geometry,
    pub pipeline: Arc<GraphicsPipeline>,
    pub state: Arc<Draw>,
    pub vertices: DrawVertices,
    pub count: u32,
}

/// Per-draw sets and pool retained until synchronous submission completes.
pub(crate) struct DrawDescriptors {
    context: Arc<Context>,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    buffers: Vec<vk::DescriptorBufferInfo>,
    images: Vec<vk::DescriptorImageInfo>,
}
impl Drop for DrawDescriptors {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::pipelines::stage_bindings;
    fn instruction(words: &mut Vec<u32>, opcode: u32, operands: &[u32]) {
        words.push(((operands.len() as u32 + 1) << 16) | opcode);
        words.extend_from_slice(operands);
    }
    fn interface(model: u32, storage: u32, location: u32) -> Vec<u32> {
        let mut words = vec![0x0723_0203, 0x0001_0300, 0, 100, 0];
        instruction(&mut words, 15, &[model, 90, 0x6e69_616d, 0]);
        instruction(&mut words, 22, &[1, 32]);
        instruction(&mut words, 23, &[2, 1, 4]);
        instruction(&mut words, 32, &[3, storage, 2]);
        instruction(&mut words, 59, &[3, 4, storage]);
        instruction(&mut words, 71, &[4, 30, location]);
        words
    }
    #[test]
    fn retry_key_rejects_invalid_public_state_before_driver_use() {
        let stages = [interface(0, 1, 0), interface(4, 3, 0)];
        let input = VertexInput {
            bindings: vec![VertexBinding {
                binding: 0,
                stride: 16,
            }],
            attributes: vec![VertexAttribute {
                binding: 0,
                format: VertexFormat::Float4,
                offset: 0,
            }],
        };
        let invalid = DrawPipelineState {
            polygon: 3,
            ..Default::default()
        };
        assert!(Key::new(
            &stages,
            input.clone(),
            PrimitiveTopology::TriangleList,
            invalid,
            false
        )
        .is_err());
        let invalid = DrawPipelineState {
            bias: [f32::INFINITY.to_bits(), 0, 0],
            ..Default::default()
        };
        assert!(Key::new(
            &stages,
            input.clone(),
            PrimitiveTopology::TriangleList,
            invalid,
            false
        )
        .is_err());
        assert!(Key::new(
            &stages,
            input.clone(),
            PrimitiveTopology::TriangleList,
            DrawPipelineState::default(),
            false
        )
        .unwrap()
        .validate_interfaces()
        .is_ok());
    }
    #[test]
    fn derives_stage_and_single_float4_location_from_translation() {
        let vertex = interface(0, 1, 3);
        assert_eq!(execution_model(&vertex).unwrap(), 0);
        assert_eq!(float_interface(&vertex, 1).unwrap(), [(3, 4)]);
        assert!(float_interface(&vertex, 3).unwrap().is_empty());
        assert!(stage_bindings(&vertex, 0).unwrap().is_empty());
        assert!(stage_bindings(&vertex, 4).is_err());
        let fragment = interface(4, 3, 0);
        assert_eq!(execution_model(&fragment).unwrap(), 4);
        assert_eq!(float_interface(&fragment, 3).unwrap(), [(0, 4)]);
    }
    #[test]
    fn accepts_extra_float_attributes_and_rejects_bad_interfaces() {
        let mut words = interface(0, 1, 0);
        instruction(&mut words, 59, &[3, 5, 1]);
        instruction(&mut words, 71, &[5, 30, 1]);
        assert_eq!(float_interface(&words, 1).unwrap(), [(0, 4), (1, 4)]);
        instruction(&mut words, 71, &[5, 30, 0]);
        assert!(float_interface(&words, 1)
            .unwrap_err()
            .contains("duplicate"));
        let mut words = interface(0, 1, 0);
        words[12] = 16; // Float width.
        assert!(float_interface(&words, 1).is_err());
        let mut words = interface(4, 3, 0);
        instruction(&mut words, 71, &[4, 11, 22]); // FragDepth.
        assert!(float_interface(&words, 3).unwrap_err().contains("builtin"));
        let mut words = interface(0, 1, 0);
        words.push(0);
        assert!(execution_model(&words).unwrap_err().contains("length"));
    }

    #[test]
    fn accepts_sparse_vertex_inputs_and_unused_vertex_outputs() {
        let mut vertex = interface(0, 1, 3);
        instruction(&mut vertex, 32, &[5, 3, 2]);
        instruction(&mut vertex, 59, &[5, 6, 3]);
        instruction(&mut vertex, 71, &[6, 30, 1]);
        let mut fragment = interface(4, 1, 1);
        instruction(&mut fragment, 32, &[5, 3, 2]);
        instruction(&mut fragment, 59, &[5, 6, 3]);
        instruction(&mut fragment, 71, &[6, 30, 0]);
        let input = VertexInput {
            bindings: vec![VertexBinding {
                binding: 0,
                stride: 16,
            }],
            attributes: vec![
                VertexAttribute {
                    binding: 0,
                    format: VertexFormat::Float4,
                    offset: 0,
                };
                4
            ],
        };
        let stages = [vertex, fragment];
        assert!(Key::new(
            &stages,
            input.clone(),
            PrimitiveTopology::TriangleList,
            DrawPipelineState::default(),
            false
        )
        .is_ok());
        let mut narrower_bound_attribute = input.clone();
        narrower_bound_attribute.attributes[3].format = VertexFormat::Float3;
        assert!(Key::new(
            &stages,
            narrower_bound_attribute,
            PrimitiveTopology::TriangleList,
            DrawPipelineState::default(),
            false
        )
        .unwrap()
        .validate_interfaces()
        .is_ok());

        let mut narrow_shader_input = interface(0, 1, 3);
        let vector_type = narrow_shader_input
            .windows(4)
            .position(|words| words[0] >> 16 == 4 && words[0] & 0xffff == 23)
            .unwrap();
        narrow_shader_input[vector_type + 3] = 2;
        assert!(Key::new(
            &[narrow_shader_input, interface(4, 1, 1)],
            input,
            PrimitiveTopology::TriangleList,
            DrawPipelineState::default(),
            false
        )
        .unwrap()
        .validate_interfaces()
        .is_err());
    }
}
