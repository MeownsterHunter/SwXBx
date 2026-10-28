//! Bounded sampled images and explicit host interpretations. Source: the public contract.; 0032.
use super::image_enums::{validate_rules, EnumRule};
use super::{
    graphics::{instructions, mapped},
    pipelines::stage_bindings,
    uniforms::GraphicsStage,
    Context,
};
pub use crate::api::SamplerDescription;
use crate::tiling::ImageKind;
use ash::vk;
use std::{collections::HashSet, sync::Arc};

/// One recorded slot mapped to the translator's original descriptor pair.
#[derive(Clone, Copy, Debug)]
pub struct TextureMapping {
    /// Raw stage token supplied by the caller.
    pub stage: u64,
    /// Raw binding index supplied by the caller.
    pub index: u64,
    /// Translated shader stage receiving the descriptor pair.
    pub target: GraphicsStage,
    /// Descriptor set number in the translated module.
    pub set: u32,
    /// Sampled-image descriptor binding number in the translated module.
    pub image: u32,
    /// Separate sampler descriptor binding number in the translated module.
    pub sampler: u32,
}

/// A bindless texture is named by the locations of its handle words: the
/// texture by the word that holds its texture id, the sampler by the word
/// that holds its sampler id. A handle of one constant-buffer word names
/// that word for both; a handle of two words or-ed names the first for the
/// texture and the second for the sampler (the reference host (MIT)
/// `Shader/Translation/Optimizations/BindlessElimination.cs` and
/// `Shader/TextureHandle.cs` `ReadPackedId`; Copyright (c) the reference host Team and
/// Contributors). The caller binds the texture and sampler of a location
/// with the separate texture and sampler binding calls of stage token
/// `stage`, at index `base + (bank << 16 | byte offset)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocationMapping {
    /// Raw stage token supplied by the caller.
    pub stage: u64,
    /// Translated shader stage whose bindless textures the indices name.
    pub target: GraphicsStage,
    /// Index of bank 0, byte offset 0.
    pub base: u64,
}

/// The indices a location mapping spans: every bank and byte offset.
pub(crate) const LOCATION_SPAN: u64 = 1 << 24;

impl LocationMapping {
    /// The binding-call index of the word at `bank`, byte `offset`.
    pub(crate) fn index(&self, bank: u8, offset: u16) -> u64 {
        self.base + (u64::from(bank) << 16 | u64::from(offset))
    }
}

/// The translated stage a surface slot belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SurfaceStage {
    /// Vertex shader stage.
    Vertex,
    /// Geometry shader stage.
    Geometry,
    /// Fragment shader stage.
    Fragment,
    /// Compute shader stage.
    Compute,
}

impl From<GraphicsStage> for SurfaceStage {
    fn from(stage: GraphicsStage) -> Self {
        match stage {
            GraphicsStage::Vertex => Self::Vertex,
            GraphicsStage::Geometry => Self::Geometry,
            GraphicsStage::Fragment => Self::Fragment,
        }
    }
}

/// One recorded image binding slot mapped to a translated module's bound
/// surface slot.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceMapping {
    /// Raw stage token the image binding call names.
    pub stage: u64,
    /// Raw binding index the image binding call names.
    pub index: u64,
    /// Translated stage whose module reads and writes the surface.
    pub target: SurfaceStage,
    /// Bound surface slot the module's instructions name.
    pub slot: u16,
}

/// Minification combines a texel filter and a mip filter.
#[derive(Clone, Copy)]
pub struct MinFilter {
    /// Filter applied to texels within a mip level.
    pub texel: vk::Filter,
    /// Which mip levels minification reads.
    pub mip: MipFilter,
}

/// The mip levels a minifying sample reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MipFilter {
    /// Only the texture's base level, whatever the sampler's LOD range.
    BaseLevel,
    /// The nearest level to the computed LOD.
    Nearest,
    /// A blend of the two levels around the computed LOD.
    Linear,
}

impl MipFilter {
    /// The Vulkan mipmap mode and LOD clamp that read these levels. Vulkan
    /// has no mode without mipmapping; its specification reads only the
    /// base level with NEAREST and the clamp [0, 0.25] (VkSamplerCreateInfo,
    /// "Mapping of OpenGL to Vulkan filter modes"; the reference host
    /// the reference host.Graphics.Vulkan/SamplerHolder.cs:23-27 does the same).
    fn host(self, clamp: [f32; 2]) -> (vk::SamplerMipmapMode, [f32; 2]) {
        match self {
            Self::BaseLevel => (vk::SamplerMipmapMode::NEAREST, [0.0, 0.25]),
            Self::Nearest => (vk::SamplerMipmapMode::NEAREST, clamp),
            Self::Linear => (vk::SamplerMipmapMode::LINEAR, clamp),
        }
    }
}

/// Every numeric correspondence is supplied and classified by the host.
#[derive(Clone, Default)]
pub struct SamplerEnums {
    /// Caller minification tokens mapped to texel and mip filters.
    pub min_filters: Vec<EnumRule<MinFilter>>,
    /// Caller magnification tokens mapped to texel filters.
    pub mag_filters: Vec<EnumRule<vk::Filter>>,
    /// Caller wrap tokens mapped to supported address modes.
    pub wraps: Vec<EnumRule<vk::SamplerAddressMode>>,
    /// Caller comparison tokens mapped to enabled or disabled sampling.
    pub compare_modes: Vec<EnumRule<bool>>,
    /// Caller comparison function tokens mapped to Vulkan comparisons.
    pub compare_functions: Vec<EnumRule<vk::CompareOp>>,
    /// Opt into the observed anisotropy float and device feature checks.
    pub anisotropy: bool,
    /// Select integer rather than float fixed border colors.
    pub integer_border: bool,
}

impl SamplerEnums {
    fn validate(&self) -> Result<(), String> {
        validate_rules(&self.min_filters)?;
        validate_rules(&self.mag_filters)?;
        validate_rules(&self.wraps)?;
        validate_rules(&self.compare_modes)?;
        validate_rules(&self.compare_functions)?;
        if self
            .min_filters
            .iter()
            .any(|r| !matches!(r.host.texel, vk::Filter::NEAREST | vk::Filter::LINEAR))
            || self
                .mag_filters
                .iter()
                .any(|r| !matches!(r.host, vk::Filter::NEAREST | vk::Filter::LINEAR))
            || self.wraps.iter().any(|r| {
                !matches!(
                    r.host,
                    vk::SamplerAddressMode::REPEAT
                        | vk::SamplerAddressMode::MIRRORED_REPEAT
                        | vk::SamplerAddressMode::CLAMP_TO_EDGE
                        | vk::SamplerAddressMode::CLAMP_TO_BORDER
                        | vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE
                )
            })
            || self
                .compare_functions
                .iter()
                .any(|r| !(0..=7).contains(&r.host.as_raw()))
        {
            return Err("unsupported sampler enum rule".into());
        }
        Ok(())
    }
}

fn enum_value<T: Copy>(rules: &[EnumRule<T>], guest: u64) -> Option<T> {
    Some(rules.iter().find(|r| r.guest == guest)?.host)
}

/// Host choices for unresolved enums and bindings. No implicit guest mapping.
#[derive(Clone, Default)]
pub struct TextureContract {
    /// Explicit mappings used to connect recorded slots to host resources.
    pub bindings: Vec<TextureMapping>,
    /// Explicit mappings from recorded image slots to bound surface slots.
    pub surfaces: Vec<SurfaceMapping>,
    /// Evidence-tagged sampler rules; when absent, use the legacy host choices.
    pub enums: Option<SamplerEnums>,
    /// Explicit mapping from caller filter tokens to Vulkan filters.
    pub filters: Vec<(u64, vk::Filter)>,
    /// Explicit mapping from caller wrap tokens to Vulkan address modes.
    pub wraps: Vec<(u64, vk::SamplerAddressMode)>,
    /// Raw token accepted as disabled comparison sampling.
    pub compare_disabled: u64,
    /// Opt into the recorded LOD float order and an explicit mip filter.
    pub lod: Option<vk::SamplerMipmapMode>,
    /// Explicit combined-handle interpretation: handle, texture id, sampler id.
    /// Separate handles returned by this library use registration ids.
    pub combined: Vec<(u64, u32, u32)>,
    /// Where the caller binds each stage's bindless textures, by location.
    pub locations: Vec<LocationMapping>,
}
impl TextureContract {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.lod.is_some_and(|m| {
            !matches!(
                m,
                vk::SamplerMipmapMode::NEAREST | vk::SamplerMipmapMode::LINEAR
            )
        }) {
            return Err("unsupported mip filter".into());
        }
        if let Some(enums) = &self.enums {
            enums.validate()?;
        }
        let mut sources = HashSet::new();
        let mut targets = HashSet::new();
        for m in &self.bindings {
            if m.set > 1
                || m.image >= 256
                || m.sampler >= 256
                || m.image == m.sampler
                || !sources.insert((m.stage, m.index))
                || !targets.insert((m.target, m.set, m.image))
                || !targets.insert((m.target, m.set, m.sampler))
            {
                return Err("duplicate or invalid texture mapping".into());
            }
        }
        let mut sources = HashSet::new();
        let mut targets = HashSet::new();
        if self
            .surfaces
            .iter()
            .any(|m| !sources.insert((m.stage, m.index)) || !targets.insert((m.target, m.slot)))
        {
            return Err("duplicate surface mapping".into());
        }
        let mut handles = HashSet::new();
        if self.combined.iter().any(|(h, _, _)| !handles.insert(*h)) {
            return Err("duplicate combined texture handle".into());
        }
        let mut stages = HashSet::new();
        for l in &self.locations {
            let span = l.base
                ..l.base
                    .checked_add(LOCATION_SPAN)
                    .ok_or("location indices overflow")?;
            if !stages.insert(l.target)
                || self
                    .bindings
                    .iter()
                    .any(|m| m.stage == l.stage && span.contains(&m.index))
                || self.locations.iter().any(|o| {
                    o.target != l.target
                        && o.stage == l.stage
                        && o.base < span.end
                        && span.start < o.base.saturating_add(LOCATION_SPAN)
                })
            {
                return Err("duplicate location mapping, or one over bound texture indices".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Binding {
    /// Translated shader stage.
    pub stage: GraphicsStage,
    /// Descriptor set in the translated module.
    pub set: u32,
    /// Descriptor binding number in the translated module.
    pub binding: u32,
    /// Reflected sampled-image or sampler descriptor type.
    pub ty: vk::DescriptorType,
    /// Reflected image dimension, present for sampled-image bindings.
    pub image_kind: Option<ImageKind>,
    /// Whether the reflected image type is an array.
    pub arrayed: bool,
}
impl Binding {
    /// Combine source set and binding into the flattened descriptor binding.
    pub fn lowered(self) -> u32 {
        self.set * 256 + self.binding
    }
    /// Return the descriptor set used for this stage in the graphics pipeline.
    pub fn stage_set(self) -> u32 {
        self.stage.texture_set()
    }
}

/// Reflect separate float images, samplers and storage images with their
/// image view shapes.
///
/// Sampled images and samplers live in set 1 below binding 256 (bound
/// textures) and in set 0 at any binding (bindless textures; the shader translator
/// places them from 4096). Storage images
/// (guest surfaces) live in set 0 (bindless views) and set 2 (bound views
/// and their aliases); their bindings only need to stay distinct once the
/// sets are flattened into the stage's one descriptor set.
pub(crate) fn bindings(words: &[u32], stage: GraphicsStage) -> Result<Vec<Binding>, String> {
    let reflected = stage_bindings(words, stage.model())?;
    let instructions = instructions(words)?;
    for &(op, args) in &instructions {
        let shape =
            |args: &[u32]| matches!((args[2], args[4]), (0 | 1, 0 | 1) | (2, 0) | (3, 0 | 1));
        let sampled = |args: &[u32]| {
            args[3] <= 1
                && args[6] == 1
                && args[7] == 0
                && shape(args)
                && instructions.contains(&(22, &[args[1], 32]))
        };
        let storage = |args: &[u32]| {
            args[3] == 0
                && args[6] == 2
                && args[2] != 3
                && shape(args)
                && (instructions.contains(&(22, &[args[1], 32]))
                    || instructions.contains(&(21, &[args[1], 32, 0]))
                    || instructions.contains(&(21, &[args[1], 32, 1])))
        };
        if op == 25 && (args.len() != 8 || args[5] != 0 || !(sampled(args) || storage(args))) {
            return Err("unsupported sampled image type".into());
        }
    }
    let mut lowered = HashSet::new();
    reflected
        .into_iter()
        .filter(|b| b.descriptor_type != vk::DescriptorType::UNIFORM_BUFFER)
        .map(|b| {
            let placed = match b.descriptor_type {
                vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::SAMPLER => match b.set {
                    0 => true,
                    1 => b.binding < 256,
                    _ => false,
                },
                vk::DescriptorType::STORAGE_IMAGE => matches!(b.set, 0 | 2),
                _ => false,
            };
            let flat = b
                .set
                .checked_mul(256)
                .and_then(|s| s.checked_add(b.binding));
            if !placed || b.count != 1 || !flat.is_some_and(|flat| lowered.insert(flat)) {
                return Err(format!(
                    "unsupported draw texture descriptor: {} at set {} binding {}",
                    match b.descriptor_type {
                        vk::DescriptorType::SAMPLED_IMAGE => "sampled image",
                        vk::DescriptorType::SAMPLER => "sampler",
                        vk::DescriptorType::STORAGE_IMAGE => "storage image",
                        _ => "descriptor",
                    },
                    b.set,
                    b.binding
                ));
            }
            let mut image_kind = None;
            let mut arrayed = false;
            if matches!(
                b.descriptor_type,
                vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::STORAGE_IMAGE
            ) {
                let variable = instructions
                    .iter()
                    .find_map(|(op, a)| {
                        if *op != 59 || a.len() < 3 {
                            return None;
                        }
                        let id = a[1];
                        let set = instructions
                            .iter()
                            .any(|(op, d)| *op == 71 && *d == [id, 34, b.set]);
                        let binding = instructions
                            .iter()
                            .any(|(op, d)| *op == 71 && *d == [id, 33, b.binding]);
                        (set && binding).then_some(a[0])
                    })
                    .ok_or("missing image variable")?;
                let pointer = instructions
                    .iter()
                    .find(|(op, a)| *op == 32 && a.first() == Some(&variable))
                    .ok_or("missing image pointer")?
                    .1;
                let mut ty = pointer[2];
                for _ in 0..instructions.len() {
                    let (op, a) = instructions
                        .iter()
                        .find(|(op, a)| matches!(*op, 25 | 28) && a.first() == Some(&ty))
                        .ok_or("missing image type")?;
                    if *op == 28 {
                        ty = a[1];
                        continue;
                    }
                    if *op != 25 {
                        return Err("invalid image descriptor type".into());
                    }
                    arrayed = a[4] != 0;
                    image_kind = Some(match (a[2], arrayed) {
                        (0, false) => ImageKind::D1,
                        (0, true) => ImageKind::D1Array,
                        (1, false) => ImageKind::D2,
                        (1, true) => ImageKind::D2Array,
                        (2, false) => ImageKind::D3,
                        (3, _) => ImageKind::Cube,
                        _ => return Err("unsupported image shape".into()),
                    });
                    break;
                }
            }
            Ok(Binding {
                stage,
                set: b.set,
                binding: b.binding,
                ty: b.descriptor_type,
                image_kind,
                arrayed,
            })
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SamplerKey {
    min: i32,
    mag: i32,
    wrap: [i32; 3],
    lod: [u32; 3],
    mipmap: i32,
    compare: Option<i32>,
    anisotropy: u32,
    border: Border,
}

/// What a sampler returns for coordinates outside the texture. Provenance:
/// the reference host `SamplerHolder` (MIT), which uses a fixed Vulkan border colour on
/// an exact match and a custom one otherwise.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Border {
    /// No address mode samples the border, so its colour is never read.
    Unsampled,
    /// One of Vulkan's fixed border colours, matched bit for bit.
    Fixed(vk::BorderColor),
    /// A float colour no fixed border colour states, as each component's
    /// bits; it needs the device's custom border colours.
    Custom([u32; 4]),
}

impl std::fmt::Debug for Border {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsampled => f.write_str("Unsampled"),
            Self::Fixed(color) => write!(f, "Fixed({})", color.as_raw()),
            Self::Custom(bits) => write!(f, "Custom({:?})", bits.map(f32::from_bits)),
        }
    }
}

impl Border {
    fn new(color: [f32; 4], integer: bool, sampled: bool) -> Option<Self> {
        const ZERO: u32 = 0;
        const ONE: u32 = 0x3f80_0000;
        if !sampled {
            return Some(Self::Unsampled);
        }
        let bits = color.map(f32::to_bits);
        let fixed = match (bits, integer) {
            ([ZERO, ZERO, ZERO, ZERO], false) => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            ([ZERO, ZERO, ZERO, ONE], false) => vk::BorderColor::FLOAT_OPAQUE_BLACK,
            ([ONE, ONE, ONE, ONE], false) => vk::BorderColor::FLOAT_OPAQUE_WHITE,
            ([ZERO, ZERO, ZERO, ZERO], true) => vk::BorderColor::INT_TRANSPARENT_BLACK,
            ([ZERO, ZERO, ZERO, ONE], true) => vk::BorderColor::INT_OPAQUE_BLACK,
            ([ONE, ONE, ONE, ONE], true) => vk::BorderColor::INT_OPAQUE_WHITE,
            (bits, false) => return Some(Self::Custom(bits)),
            (_, true) => return None,
        };
        Some(Self::Fixed(fixed))
    }
}

impl SamplerKey {
    /// Create a checked sampler interpretation or Vulkan sampler; return None for unsupported state.
    pub fn new(d: &SamplerDescription, c: &TextureContract) -> Option<Self> {
        if (c.lod.is_none() && (d.lod_bias != 0.0 || d.lod_clamp != [0.0; 2]))
            || !d.lod_bias.is_finite()
            || !d.lod_clamp.iter().all(|v| v.is_finite())
            || d.lod_clamp[0] < 0.0
            || d.lod_clamp[1] < d.lod_clamp[0]
            || !d.max_anisotropy.is_finite()
            || d.max_anisotropy < 1.0
        {
            return None;
        }
        let (min, mag, (mipmap, lod_clamp), compare, anisotropy, integer_border) =
            if let Some(enums) = &c.enums {
                let min = enum_value(&enums.min_filters, d.min_filter)?;
                let compare = if enum_value(&enums.compare_modes, d.compare_mode)? {
                    Some(enum_value(&enums.compare_functions, d.compare_func)?.as_raw())
                } else {
                    None
                };
                if !enums.anisotropy && d.max_anisotropy != 1.0 {
                    return None;
                }
                (
                    min.texel,
                    enum_value(&enums.mag_filters, d.mag_filter)?,
                    min.mip.host(d.lod_clamp),
                    compare,
                    d.max_anisotropy,
                    enums.integer_border,
                )
            } else {
                if d.compare_mode != c.compare_disabled
                    || d.max_anisotropy != 1.0
                    || d.border_color != [0.0; 4]
                {
                    return None;
                }
                (
                    mapped(&c.filters, d.min_filter)?,
                    mapped(&c.filters, d.mag_filter)?,
                    (c.lod.unwrap_or(vk::SamplerMipmapMode::NEAREST), d.lod_clamp),
                    None,
                    1.0,
                    false,
                )
            };
        if ![min, mag]
            .iter()
            .all(|v| matches!(*v, vk::Filter::NEAREST | vk::Filter::LINEAR))
        {
            return None;
        }
        let mut wrap = [0; 3];
        for (i, token) in d.wrap.iter().enumerate() {
            let value = if let Some(enums) = &c.enums {
                enum_value(&enums.wraps, *token)?
            } else {
                mapped(&c.wraps, *token)?
            };
            if !matches!(
                value,
                vk::SamplerAddressMode::REPEAT
                    | vk::SamplerAddressMode::MIRRORED_REPEAT
                    | vk::SamplerAddressMode::CLAMP_TO_EDGE
                    | vk::SamplerAddressMode::CLAMP_TO_BORDER
                    | vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE
            ) {
                return None;
            }
            wrap[i] = value.as_raw();
        }
        let border = Border::new(
            d.border_color,
            integer_border,
            wrap.contains(&vk::SamplerAddressMode::CLAMP_TO_BORDER.as_raw()),
        )?;
        Some(Self {
            min: min.as_raw(),
            mag: mag.as_raw(),
            wrap,
            lod: [
                lod_clamp[0].to_bits(),
                lod_clamp[1].to_bits(),
                d.lod_bias.to_bits(),
            ],
            mipmap: mipmap.as_raw(),
            compare,
            anisotropy: anisotropy.to_bits(),
            border,
        })
    }
}

pub(crate) struct Sampler {
    context: Arc<Context>,
    /// Complete interpreted sampler configuration.
    pub key: SamplerKey,
    /// Owned Vulkan sampler handle.
    pub handle: vk::Sampler,
}
impl SamplerKey {
    /// Whether the sampler filters or blends mip levels linearly, which
    /// needs the texture's format to filter linearly. Source: the public contract.
    pub(crate) fn filters_linearly(&self) -> bool {
        self.min == vk::Filter::LINEAR.as_raw()
            || self.mag == vk::Filter::LINEAR.as_raw()
            || self.mipmap == vk::SamplerMipmapMode::LINEAR.as_raw()
    }

    /// Whether an axis mirrors the texture once and then clamps to its edge,
    /// which needs the device's samplerMirrorClampToEdge feature.
    pub(crate) fn mirrors_once(&self) -> bool {
        self.wrap
            .contains(&vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE.as_raw())
    }

    /// The custom border colour the sampler samples, which needs the
    /// device's custom border colours.
    pub(crate) fn custom_border(&self) -> Option<[f32; 4]> {
        match self.border {
            Border::Custom(bits) => Some(bits.map(f32::from_bits)),
            Border::Unsampled | Border::Fixed(_) => None,
        }
    }
}

/// Whether the device samples with MIRROR_CLAMP_TO_EDGE. SwXBx enables the
/// Vulkan 1.2 feature whenever the device has it.
pub(crate) fn mirror_clamp_to_edge(context: &Context) -> bool {
    let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut v12);
    unsafe {
        context
            .instance
            .get_physical_device_features2(context.physical_device, &mut features)
    };
    v12.sampler_mirror_clamp_to_edge != vk::FALSE
}

impl Sampler {
    /// Create a checked sampler interpretation or Vulkan sampler; return None for unsupported state.
    pub fn new(context: &Arc<Context>, key: SamplerKey) -> Option<Self> {
        let limits = unsafe {
            context
                .instance
                .get_physical_device_properties(context.physical_device)
        }
        .limits;
        if f32::from_bits(key.lod[2]).abs() > limits.max_sampler_lod_bias {
            return None;
        }
        if key.mirrors_once() && !mirror_clamp_to_edge(context) {
            return None;
        }
        let anisotropy = f32::from_bits(key.anisotropy);
        let features = unsafe {
            context
                .instance
                .get_physical_device_features(context.physical_device)
        };
        if anisotropy > limits.max_sampler_anisotropy
            || (anisotropy > 1.0 && features.sampler_anisotropy == vk::FALSE)
        {
            return None;
        }
        let mut custom = vk::SamplerCustomBorderColorCreateInfoEXT::default();
        let border = match key.border {
            Border::Unsampled => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            Border::Fixed(color) => color,
            Border::Custom(bits) => {
                context.custom_border_colors.reserve()?;
                // The device samples custom colours without a format, so the
                // colour holds for every view the sampler meets.
                custom = custom.custom_border_color(vk::ClearColorValue {
                    float32: bits.map(f32::from_bits),
                });
                vk::BorderColor::FLOAT_CUSTOM_EXT
            }
        };
        let mut info = vk::SamplerCreateInfo::default();
        if let Border::Custom(_) = key.border {
            info = info.push_next(&mut custom);
        }
        let created = unsafe {
            context.device.create_sampler(
                &info
                    .min_filter(vk::Filter::from_raw(key.min))
                    .mag_filter(vk::Filter::from_raw(key.mag))
                    .mipmap_mode(vk::SamplerMipmapMode::from_raw(key.mipmap))
                    .address_mode_u(vk::SamplerAddressMode::from_raw(key.wrap[0]))
                    .address_mode_v(vk::SamplerAddressMode::from_raw(key.wrap[1]))
                    .address_mode_w(vk::SamplerAddressMode::from_raw(key.wrap[2]))
                    .border_color(border)
                    .compare_enable(key.compare.is_some())
                    .compare_op(
                        key.compare
                            .map_or(vk::CompareOp::ALWAYS, vk::CompareOp::from_raw),
                    )
                    .anisotropy_enable(anisotropy > 1.0)
                    .max_anisotropy(anisotropy)
                    .min_lod(f32::from_bits(key.lod[0]))
                    .max_lod(f32::from_bits(key.lod[1]))
                    .mip_lod_bias(f32::from_bits(key.lod[2])),
                None,
            )
        };
        let Ok(handle) = created else {
            if let Border::Custom(_) = key.border {
                context.custom_border_colors.release();
            }
            return None;
        };
        Some(Self {
            context: context.clone(),
            key,
            handle,
        })
    }
}
impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_sampler(self.handle, None);
        }
        if let Border::Custom(_) = self.key.border {
            self.context.custom_border_colors.release();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_bounds_require_explicit_host_order_and_finite_values() {
        let mut contract = TextureContract {
            filters: vec![(7, vk::Filter::LINEAR)],
            wraps: vec![(9, vk::SamplerAddressMode::REPEAT)],
            compare_disabled: 11,
            ..Default::default()
        };
        let mut description = SamplerDescription {
            min_filter: 7,
            mag_filter: 7,
            wrap: [9; 3],
            max_anisotropy: 1.0,
            compare_mode: 11,
            lod_clamp: [0.0, 5.0],
            ..Default::default()
        };
        assert!(SamplerKey::new(&description, &contract).is_none());
        contract.lod = Some(vk::SamplerMipmapMode::LINEAR);
        let first = SamplerKey::new(&description, &contract).unwrap();
        description.lod_bias = 0.5;
        assert_ne!(
            first.lod,
            SamplerKey::new(&description, &contract).unwrap().lod
        );
        description.lod_clamp = [5.0, 1.0];
        assert!(SamplerKey::new(&description, &contract).is_none());
        description.lod_clamp = [0.0, f32::INFINITY];
        assert!(SamplerKey::new(&description, &contract).is_none());
        description.lod_clamp = [0.0, 1.0];
        description.lod_bias = f32::NAN;
        assert!(SamplerKey::new(&description, &contract).is_none());
    }
}
