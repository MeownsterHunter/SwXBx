//! Which formats swxbx draws into, samples and presents, and the device
//! features each of those uses needs. Source: the public contract.
//!
//! The classification reads the format facts in `image_layout`. The feature
//! sets are ported from the reference host (MIT), `the reference host.Graphics.Vulkan/
//! FormatCapabilities.cs` (`OptimalFormatSupports`, `ConvertToVkFormat`) and
//! `VulkanRenderer.cs` (`GetCapabilities`, compressed formats and ASTC).
//! Copyright (c) the reference host Team and Contributors, MIT licensed.
//!
//! Where the reference host substitutes another format for one the device lacks, swxbx
//! refuses the format by name instead.
use super::{
    image_layout::{format_aspects, format_block, numeric_class, NumericClass},
    Context,
};
use ash::vk;

/// A Vulkan format named for an integrator: its `VkFormat` value.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Name(pub vk::Format);

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VkFormat {}", self.0.as_raw())
    }
}

impl std::fmt::Debug for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

/// A token-to-format table, printed as token and `VkFormat` value pairs.
pub(crate) fn table_debug(entries: &[(u64, vk::Format)]) -> Vec<(u64, Name)> {
    entries
        .iter()
        .map(|&(token, format)| (token, Name(format)))
        .collect()
}

/// Bytes per texel of a depth format as swxbx keeps it in memory: the depth
/// plane, then a one-byte stencil plane when the format has stencil, starting
/// at the next four-byte boundary, where Vulkan copies it from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthPlanes {
    /// Bytes of one depth value, as a Vulkan copy of the depth aspect writes it.
    pub depth: u8,
    /// Bytes of one stencil value: one, or zero for a depth-only format.
    pub stencil: u8,
}

impl DepthPlanes {
    /// Where the stencil plane of `texels` texels starts: after the depth
    /// plane, padded to four bytes.
    pub fn stencil_offset(self, texels: u64) -> u64 {
        texels
            .saturating_mul(u64::from(self.depth))
            .next_multiple_of(4)
    }

    /// Bytes both planes of `texels` texels take.
    pub fn size(self, texels: u64) -> u64 {
        match self.stencil {
            0 => texels.saturating_mul(u64::from(self.depth)),
            stencil => self
                .stencil_offset(texels)
                .saturating_add(texels.saturating_mul(u64::from(stencil))),
        }
    }
}

/// The depth formats swxbx draws with. A copy of the depth aspect of a
/// 24-bit format writes four bytes per texel.
pub fn depth_planes(format: vk::Format) -> Option<DepthPlanes> {
    use vk::Format as F;
    let (depth, stencil) = match format {
        F::D16_UNORM => (2, 0),
        F::X8_D24_UNORM_PACK32 | F::D32_SFLOAT => (4, 0),
        F::D16_UNORM_S8_UINT => (2, 1),
        F::D24_UNORM_S8_UINT | F::D32_SFLOAT_S8_UINT => (4, 1),
        _ => return None,
    };
    Some(DepthPlanes { depth, stencil })
}

/// Every core Vulkan format whose texel-block geometry swxbx knows, so it
/// can lay out, upload and copy images of it. An integrator that names
/// formats by their `VkFormat` value can map each to itself. Whether the
/// device supports a use of one is checked where the format is used.
pub fn known() -> impl Iterator<Item = vk::Format> {
    (1..=vk::Format::ASTC_12X12_SRGB_BLOCK.as_raw())
        .map(vk::Format::from_raw)
        .filter(|&format| format_block(format).is_some())
}
/// How many colour components a non-compressed Vulkan color format has, by
/// its name: the R, G, B and A components it stores (padding and shared
/// exponents are not components). None for depth, stencil and compressed
/// formats. A vertex input that reads such a format gets the missing
/// components of (0, 0, 0, 1). Generated from `vulkan_core.h` names.
pub fn channels(format: vk::Format) -> Option<u8> {
    use vk::Format as F;
    Some(match format {
        F::R8_UNORM
        | F::R8_SNORM
        | F::R8_USCALED
        | F::R8_SSCALED
        | F::R8_UINT
        | F::R8_SINT
        | F::R8_SRGB
        | F::R16_UNORM
        | F::R16_SNORM
        | F::R16_USCALED
        | F::R16_SSCALED
        | F::R16_UINT
        | F::R16_SINT
        | F::R16_SFLOAT
        | F::R32_UINT
        | F::R32_SINT
        | F::R32_SFLOAT
        | F::R64_UINT
        | F::R64_SINT
        | F::R64_SFLOAT => 1,
        F::R4G4_UNORM_PACK8
        | F::R8G8_UNORM
        | F::R8G8_SNORM
        | F::R8G8_USCALED
        | F::R8G8_SSCALED
        | F::R8G8_UINT
        | F::R8G8_SINT
        | F::R8G8_SRGB
        | F::R16G16_UNORM
        | F::R16G16_SNORM
        | F::R16G16_USCALED
        | F::R16G16_SSCALED
        | F::R16G16_UINT
        | F::R16G16_SINT
        | F::R16G16_SFLOAT
        | F::R32G32_UINT
        | F::R32G32_SINT
        | F::R32G32_SFLOAT
        | F::R64G64_UINT
        | F::R64G64_SINT
        | F::R64G64_SFLOAT => 2,
        F::R5G6B5_UNORM_PACK16
        | F::B5G6R5_UNORM_PACK16
        | F::R8G8B8_UNORM
        | F::R8G8B8_SNORM
        | F::R8G8B8_USCALED
        | F::R8G8B8_SSCALED
        | F::R8G8B8_UINT
        | F::R8G8B8_SINT
        | F::R8G8B8_SRGB
        | F::B8G8R8_UNORM
        | F::B8G8R8_SNORM
        | F::B8G8R8_USCALED
        | F::B8G8R8_SSCALED
        | F::B8G8R8_UINT
        | F::B8G8R8_SINT
        | F::B8G8R8_SRGB
        | F::R16G16B16_UNORM
        | F::R16G16B16_SNORM
        | F::R16G16B16_USCALED
        | F::R16G16B16_SSCALED
        | F::R16G16B16_UINT
        | F::R16G16B16_SINT
        | F::R16G16B16_SFLOAT
        | F::R32G32B32_UINT
        | F::R32G32B32_SINT
        | F::R32G32B32_SFLOAT
        | F::R64G64B64_UINT
        | F::R64G64B64_SINT
        | F::R64G64B64_SFLOAT
        | F::B10G11R11_UFLOAT_PACK32
        | F::E5B9G9R9_UFLOAT_PACK32 => 3,
        F::R4G4B4A4_UNORM_PACK16
        | F::B4G4R4A4_UNORM_PACK16
        | F::R5G5B5A1_UNORM_PACK16
        | F::B5G5R5A1_UNORM_PACK16
        | F::A1R5G5B5_UNORM_PACK16
        | F::R8G8B8A8_UNORM
        | F::R8G8B8A8_SNORM
        | F::R8G8B8A8_USCALED
        | F::R8G8B8A8_SSCALED
        | F::R8G8B8A8_UINT
        | F::R8G8B8A8_SINT
        | F::R8G8B8A8_SRGB
        | F::B8G8R8A8_UNORM
        | F::B8G8R8A8_SNORM
        | F::B8G8R8A8_USCALED
        | F::B8G8R8A8_SSCALED
        | F::B8G8R8A8_UINT
        | F::B8G8R8A8_SINT
        | F::B8G8R8A8_SRGB
        | F::A8B8G8R8_UNORM_PACK32
        | F::A8B8G8R8_SNORM_PACK32
        | F::A8B8G8R8_USCALED_PACK32
        | F::A8B8G8R8_SSCALED_PACK32
        | F::A8B8G8R8_UINT_PACK32
        | F::A8B8G8R8_SINT_PACK32
        | F::A8B8G8R8_SRGB_PACK32
        | F::A2R10G10B10_UNORM_PACK32
        | F::A2R10G10B10_SNORM_PACK32
        | F::A2R10G10B10_USCALED_PACK32
        | F::A2R10G10B10_SSCALED_PACK32
        | F::A2R10G10B10_UINT_PACK32
        | F::A2R10G10B10_SINT_PACK32
        | F::A2B10G10R10_UNORM_PACK32
        | F::A2B10G10R10_SNORM_PACK32
        | F::A2B10G10R10_USCALED_PACK32
        | F::A2B10G10R10_SSCALED_PACK32
        | F::A2B10G10R10_UINT_PACK32
        | F::A2B10G10R10_SINT_PACK32
        | F::R16G16B16A16_UNORM
        | F::R16G16B16A16_SNORM
        | F::R16G16B16A16_USCALED
        | F::R16G16B16A16_SSCALED
        | F::R16G16B16A16_UINT
        | F::R16G16B16A16_SINT
        | F::R16G16B16A16_SFLOAT
        | F::R32G32B32A32_UINT
        | F::R32G32B32A32_SINT
        | F::R32G32B32A32_SFLOAT
        | F::R64G64B64A64_UINT
        | F::R64G64B64A64_SINT
        | F::R64G64B64A64_SFLOAT => 4,
        _ => return None,
    })
}

/// Whether swxbx can draw color into this format: a color format of one
/// texel per block that reads as float, which is what translated fragment
/// shaders write. Returns the bytes per texel.
pub fn color_target(format: vk::Format) -> Result<u8, String> {
    let block = format_block(format).filter(|b| b.width == 1 && b.height == 1);
    match block {
        Some(block)
            if format_aspects(format) == vk::ImageAspectFlags::COLOR
                && numeric_class(format) == NumericClass::Float =>
        {
            Ok(block.bytes)
        }
        _ => Err(format!(
            "{} is not a color format swxbx draws into (one texel per block, read as \
             float)",
            Name(format)
        )),
    }
}

/// The value a colour clear of an image of `format` writes, from the
/// clear's four 32-bit words: floats for a format that reads as float
/// (normalized, scaled, float and sRGB formats), the same bits as unsigned
/// or signed integers for an integer format, as the clear registers hold
/// them. A clear that keeps some channels (`mask` short of 0xf) reads the
/// image back and rewrites it as bytes, which swxbx does only for
/// R8G8B8A8_UNORM.
pub(crate) fn color_clear(
    format: vk::Format,
    words: [f32; 4],
    mask: u32,
) -> Result<vk::ClearColorValue, String> {
    if compressed(format) || format_aspects(format) != vk::ImageAspectFlags::COLOR {
        return Err(format!(
            "colour clear of a {} image: swxbx clears colour formats of one texel per block",
            Name(format)
        ));
    }
    if mask & 15 != 15 && format != vk::Format::R8G8B8A8_UNORM {
        return Err(format!(
            "colour clear of channels {mask:#x} of a {} image: swxbx keeps uncleared channels \
             only in R8G8B8A8_UNORM images",
            Name(format)
        ));
    }
    let bits = words.map(f32::to_bits);
    Ok(match numeric_class(format) {
        NumericClass::Float => vk::ClearColorValue { float32: words },
        NumericClass::Unsigned => vk::ClearColorValue { uint32: bits },
        NumericClass::Signed => vk::ClearColorValue {
            int32: bits.map(|b| b as i32),
        },
        NumericClass::Depth => {
            return Err(format!(
                "colour clear of a {} image, which holds depth",
                Name(format)
            ))
        }
    })
}

/// Whether swxbx can draw depth into this format.
pub fn depth_target(format: vk::Format) -> Result<DepthPlanes, String> {
    depth_planes(format)
        .ok_or_else(|| format!("{} is not a depth format swxbx draws with", Name(format)))
}

/// The image swxbx draws a caller's depth format with, resolved once when
/// the depth and raster contract is installed. Provenance: the reference host
/// `FormatCapabilities` (MIT), which draws D24_UNORM_S8_UINT as
/// D32_SFLOAT_S8_UINT on a device without it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DepthImage {
    /// The caller's format, which the device draws depth with.
    Exact(vk::Format),
    /// D24_UNORM_S8_UINT on a device that cannot draw with it, drawn in a
    /// D32_SFLOAT_S8_UINT image, which holds depth as 32-bit floats rather
    /// than 24-bit normalized values. The image is not the memory's layout,
    /// so it holds the target's only copy: swxbx neither loads nor stores it
    /// through memory and refuses by name anything else that reads or writes
    /// the memory the target covers, and a polygon offset, which moves depth
    /// by a format-dependent amount.
    D24S8AsD32S8,
}

impl DepthImage {
    /// The format of the image swxbx draws with.
    pub fn format(self) -> vk::Format {
        match self {
            Self::Exact(format) => format,
            Self::D24S8AsD32S8 => vk::Format::D32_SFLOAT_S8_UINT,
        }
    }

    /// The caller's format, which lays out the memory the target covers.
    pub fn guest(self) -> vk::Format {
        match self {
            Self::Exact(format) => format,
            Self::D24S8AsD32S8 => vk::Format::D24_UNORM_S8_UINT,
        }
    }

    /// The image the device draws `format` with, or why it cannot.
    pub(crate) fn resolve(context: &Context, format: vk::Format) -> Result<Self, String> {
        depth_target(format)?;
        match supports(context, format, FormatUse::DepthTarget) {
            Ok(()) => Ok(Self::Exact(format)),
            Err(_)
                if format == vk::Format::D24_UNORM_S8_UINT
                    && supports(
                        context,
                        vk::Format::D32_SFLOAT_S8_UINT,
                        FormatUse::DepthTarget,
                    )
                    .is_ok() =>
            {
                Ok(Self::D24S8AsD32S8)
            }
            Err(why) => Err(why),
        }
    }
}

/// Which planes of a depth target drawn in a widened image swxbx has
/// written since the image was made: they hold nothing until a clear writes
/// them, because the image cannot be loaded from memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeldPlanes {
    Unwritten,
    Depth,
    Stencil,
    Both,
}

impl HeldPlanes {
    /// The planes held after a clear of `depth` and `stencil`.
    pub(crate) fn cleared(self, depth: Option<f32>, stencil: Option<u32>) -> Self {
        let depth = depth.is_some() || matches!(self, Self::Depth | Self::Both);
        let stencil = stencil.is_some() || matches!(self, Self::Stencil | Self::Both);
        match (depth, stencil) {
            (false, false) => Self::Unwritten,
            (true, false) => Self::Depth,
            (false, true) => Self::Stencil,
            (true, true) => Self::Both,
        }
    }

    /// The plane a draw that tests depth (`depth_test`) and stencil
    /// (`stencil_test`) reads but this image does not hold.
    pub(crate) fn missing(self, depth_test: bool, stencil_test: bool) -> Option<&'static str> {
        match self {
            Self::Unwritten | Self::Stencil if depth_test => Some("depth"),
            Self::Unwritten | Self::Depth if stencil_test => Some("stencil"),
            _ => None,
        }
    }
}

/// What swxbx does with an image of a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormatUse {
    /// Drawn into as a color target, without blending.
    ColorTarget,
    /// Drawn into and blended as a color target.
    BlendedColorTarget,
    /// Drawn with as the depth or depth-stencil target.
    DepthTarget,
    /// Read by shaders.
    Sampled,
}

impl FormatUse {
    /// Optimal-tiling features the use needs. Every swxbx image is sampled
    /// and moves to and from memory by transfer (the reference host ConvertToVkFormat).
    fn features(self, format: vk::Format) -> vk::FormatFeatureFlags {
        use vk::FormatFeatureFlags as F;
        let base = F::SAMPLED_IMAGE | F::TRANSFER_SRC | F::TRANSFER_DST;
        match self {
            Self::ColorTarget => base | F::COLOR_ATTACHMENT,
            Self::BlendedColorTarget => base | F::COLOR_ATTACHMENT | F::COLOR_ATTACHMENT_BLEND,
            Self::DepthTarget => base | F::DEPTH_STENCIL_ATTACHMENT,
            Self::Sampled if compressed(format) => {
                base | F::SAMPLED_IMAGE_FILTER_LINEAR | F::BLIT_SRC
            }
            Self::Sampled => base,
        }
    }
}

fn compressed(format: vk::Format) -> bool {
    format_block(format).is_some_and(|b| b.width != 1 || b.height != 1)
}

fn astc(format: vk::Format) -> bool {
    (vk::Format::ASTC_4X4_UNORM_BLOCK.as_raw()..=vk::Format::ASTC_12X12_SRGB_BLOCK.as_raw())
        .contains(&format.as_raw())
}

/// Optimal-tiling features the device has for a format.
pub(crate) fn optimal_features(context: &Context, format: vk::Format) -> vk::FormatFeatureFlags {
    // SAFETY: the physical device belongs to the context's live instance.
    unsafe {
        context
            .instance
            .get_physical_device_format_properties(context.physical_device, format)
    }
    .optimal_tiling_features
}

/// Check that the device supports `format` for `usage`, or name what it lacks.
pub(crate) fn supports(
    context: &Context,
    format: vk::Format,
    usage: FormatUse,
) -> Result<(), String> {
    if usage == FormatUse::Sampled && astc(format) {
        // SAFETY: as above.
        let core = unsafe {
            context
                .instance
                .get_physical_device_features(context.physical_device)
        };
        if core.texture_compression_astc_ldr == vk::FALSE {
            return Err(format!(
                "the device has no ASTC texture support, so it cannot sample {}; swxbx \
                 does not decode ASTC itself",
                Name(format)
            ));
        }
    }
    let wanted = usage.features(format);
    let missing = wanted & !optimal_features(context, format);
    if missing.is_empty() {
        return Ok(());
    }
    let what = match usage {
        FormatUse::ColorTarget => "draw into",
        FormatUse::BlendedColorTarget => "draw and blend into",
        FormatUse::DepthTarget => "draw depth with",
        FormatUse::Sampled => "sample",
    };
    Err(format!(
        "the device cannot {what} {}: it lacks {}",
        Name(format),
        feature_names(missing)
    ))
}

/// Name the format features swxbx asks for, as Vulkan spells them.
fn feature_names(flags: vk::FormatFeatureFlags) -> String {
    use vk::FormatFeatureFlags as F;
    [
        (F::SAMPLED_IMAGE, "SAMPLED_IMAGE"),
        (
            F::SAMPLED_IMAGE_FILTER_LINEAR,
            "SAMPLED_IMAGE_FILTER_LINEAR",
        ),
        (F::TRANSFER_SRC, "TRANSFER_SRC"),
        (F::TRANSFER_DST, "TRANSFER_DST"),
        (F::BLIT_SRC, "BLIT_SRC"),
        (F::COLOR_ATTACHMENT, "COLOR_ATTACHMENT"),
        (F::COLOR_ATTACHMENT_BLEND, "COLOR_ATTACHMENT_BLEND"),
        (F::DEPTH_STENCIL_ATTACHMENT, "DEPTH_STENCIL_ATTACHMENT"),
    ]
    .into_iter()
    .filter(|&(flag, _)| flags.contains(flag))
    .map(|(_, name)| name)
    .collect::<Vec<_>>()
    .join(", ")
}

/// Check that the device filters `format` linearly, which a sampler with a
/// linear filter or linear mip blending needs of the texture it reads.
pub(crate) fn filters_linearly(context: &Context, format: vk::Format) -> Result<(), String> {
    if optimal_features(context, format)
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        Ok(())
    } else {
        Err(format!(
            "the device cannot filter {} linearly: it lacks SAMPLED_IMAGE_FILTER_LINEAR",
            Name(format)
        ))
    }
}

/// Whether a present or readback can show a texture of this format: a color
/// format of one texel per block that reads as float, as an image-to-image
/// blit takes it.
pub fn shown(format: vk::Format) -> Result<(), String> {
    color_target(format).map(drop).map_err(|_| {
        format!(
            "{} is not a color format of one texel per block that reads as float, which is \
             what swxbx presents and reads back",
            Name(format)
        )
    })
}

/// The UNORM format with the same bytes as an sRGB color format. A present or
/// readback blits from it, so the stored bytes reach the screen unchanged
/// instead of being decoded from sRGB.
pub fn byte_alias(format: vk::Format) -> Option<vk::Format> {
    use vk::Format as F;
    Some(match format {
        F::R8_SRGB => F::R8_UNORM,
        F::R8G8_SRGB => F::R8G8_UNORM,
        F::R8G8B8_SRGB => F::R8G8B8_UNORM,
        F::B8G8R8_SRGB => F::B8G8R8_UNORM,
        F::R8G8B8A8_SRGB => F::R8G8B8A8_UNORM,
        F::B8G8R8A8_SRGB => F::B8G8R8A8_UNORM,
        F::A8B8G8R8_SRGB_PACK32 => F::A8B8G8R8_UNORM_PACK32,
        _ => return None,
    })
}
