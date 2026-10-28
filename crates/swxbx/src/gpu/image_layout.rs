//! Explicit host interpretations of opaque texture fields. Source: the public contract.; 0032.
use super::image_enums::{validate_rules, EnumRule, Evidence};
use crate::{
    api::TextureDescription,
    tiling::{BlockFormat, ImageKind, ImageShape, Layout, TileShape},
};
use ash::vk;
use std::collections::HashSet;

/// How a texture's sample coordinates address it. NVN's RECTANGLE target
/// reads texels, in [0, extent]; every other target reads [0, 1]. The guest
/// driver clears the descriptor's normalized-coordinates bit for rectangle
/// textures (the reference host Gpu/Image/TextureDescriptor.cs:203-208), so a shader
/// that samples one is translated for texel coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Coordinates {
    /// [0, 1] across the texture.
    #[default]
    Normalized,
    /// [0, extent] in texels of level zero. Two-dimensional images only.
    Texels,
}

/// Storage selected by a host after interpreting recorded builder fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    /// Packed-linear storage with the layout's mip and layer alignment.
    Linear,
    /// Block-linear storage with explicit tile exponents.
    Tiled(TileShape),
}

/// One complete flags/target/format tuple. No guest enum is implicit.
#[derive(Clone, Copy)]
pub struct ImageRule {
    /// Flags, target and format interpretations, in that order.
    pub evidence: [Evidence; 3],
    /// Supporting evidence or hypothesis for flags, target and format, in order.
    pub rationale: [&'static str; 3],
    /// Raw flags matched exactly by this rule.
    pub flags: u64,
    /// Raw target token matched exactly by this rule.
    pub target: u64,
    /// Raw format token matched exactly by this rule.
    pub format: u64,
    /// Public Vulkan format selected by the host.
    pub host_format: vk::Format,
    /// Image dimension and layer organization.
    pub kind: ImageKind,
    /// How shaders address the texture: the rule for an NVN RECTANGLE
    /// target token says Texels.
    pub coordinates: Coordinates,
    /// Base address of host-managed program storage.
    pub storage: Storage,
}

/// Complete host rules for interpreting otherwise opaque image builder values.
#[derive(Clone, Default)]
pub struct ImageContract {
    /// Unique flags, target, and format rules; no implicit mapping is added.
    pub rules: Vec<ImageRule>,
    /// Empty retains the legacy host identity choice. Nonempty requires every token.
    pub swizzles: Vec<EnumRule<vk::ComponentSwizzle>>,
    /// Explicit caller selectors mapped to one depth or stencil aspect.
    pub depth_stencil_modes: Vec<EnumRule<vk::ImageAspectFlags>>,
}

/// Typed Vulkan image shape and format independent of guest enum values.
#[derive(Clone)]
pub struct ImageDescriptor {
    /// Base texel dimensions, layers, and mip levels.
    pub shape: ImageShape,
    /// Public Vulkan image format.
    pub format: vk::Format,
}
impl ImageDescriptor {
    /// Calculate checked byte packing, or None for unsupported geometry or shape.
    pub fn packing(&self, tile: TileShape) -> Option<Layout> {
        Layout::new(self.shape, format_block(self.format)?, tile).ok()
    }
}

pub(crate) struct ResolvedImage {
    /// Resolved typed host image shape and format.
    pub descriptor: ImageDescriptor,
    /// Checked byte packing for the image storage.
    pub packing: Layout,
    /// Host-selected linear or tiled storage representation.
    pub storage: Storage,
    pub components: vk::ComponentMapping,
}
impl ImageContract {
    /// Reject duplicate rules, unsupported format geometry, or excessive tile exponents.
    pub fn validate(&self) -> Result<(), String> {
        validate_rules(&self.swizzles)?;
        validate_rules(&self.depth_stencil_modes)?;
        if self
            .depth_stencil_modes
            .iter()
            .any(|r| !matches!(r.host.as_raw(), 2 | 4))
        {
            return Err("depth/stencil selection must name one aspect".into());
        }
        if self
            .swizzles
            .iter()
            .any(|r| !(0..=6).contains(&r.host.as_raw()))
        {
            return Err("unsupported component swizzle".into());
        }
        let mut keys = HashSet::new();
        for r in &self.rules {
            if !keys.insert((r.flags, r.target, r.format))
                || format_block(r.host_format).is_none()
                || r.rationale.iter().any(|s| s.trim().is_empty())
            {
                return Err("duplicate image rule or unsupported format geometry".into());
            }
            if r.coordinates == Coordinates::Texels && r.kind != ImageKind::D2 {
                return Err(format!(
                    "image rule for target {} reads texel coordinates but is not two-dimensional",
                    r.target
                ));
            }
            if let Storage::Tiled(t) = r.storage {
                if t.height_log2 > 5 || t.depth_log2 > 5 {
                    return Err("image tile exponent exceeds five".into());
                }
            }
        }
        Ok(())
    }
    /// How shaders address a texture with this description, or None when
    /// no rule covers it.
    pub(crate) fn coordinates(&self, d: &TextureDescription) -> Option<Coordinates> {
        self.rules
            .iter()
            .find(|r| (r.flags, r.target, r.format) == (d.flags, d.target, d.format))
            .map(|r| r.coordinates)
    }

    pub(crate) fn resolve(&self, d: &TextureDescription) -> Option<ResolvedImage> {
        let r = self
            .rules
            .iter()
            .find(|r| (r.flags, r.target, r.format) == (d.flags, d.target, d.format))?;
        let aspects = format_aspects(r.host_format);
        if aspects != vk::ImageAspectFlags::COLOR {
            if self.depth_stencil_modes.is_empty() {
                if d.depth_stencil_mode != 0 {
                    return None;
                }
            } else if self
                .depth_stencil_modes
                .iter()
                .find(|r| r.guest == d.depth_stencil_mode)?
                .host
                != aspects
            {
                return None;
            }
        }
        let width = u32::try_from(d.width).ok()?;
        let height = u32::try_from(d.height).ok()?;
        let count = u32::try_from(d.depth.max(1)).ok()?;
        let (depth, layers) = if r.kind == ImageKind::D3 {
            (count, 1)
        } else if matches!(r.kind, ImageKind::D1 | ImageKind::D2) {
            (1, 1)
        } else {
            (1, count)
        };
        if matches!(r.kind, ImageKind::D1 | ImageKind::D2) && count != 1 {
            return None;
        }
        let descriptor = ImageDescriptor {
            shape: ImageShape {
                width,
                height,
                depth,
                layers,
                levels: u32::try_from(d.levels.max(1)).ok()?,
                kind: r.kind,
            },
            format: r.host_format,
        };
        let tile = match r.storage {
            Storage::Linear => TileShape {
                height_log2: 0,
                depth_log2: 0,
            },
            Storage::Tiled(t) => t,
        };
        let packing = descriptor.packing(tile)?;
        // A nonzero stride is accepted only for the packed single-level linear case.
        if d.stride != 0
            && (r.storage != Storage::Linear
                || descriptor.shape.levels != 1
                || d.stride != packing.levels()[0].row_bytes as u64)
        {
            return None;
        }
        let mut components = [vk::ComponentSwizzle::IDENTITY; 4];
        if !self.swizzles.is_empty() {
            for (component, token) in components.iter_mut().zip(d.swizzle) {
                *component = self.swizzles.iter().find(|r| r.guest == token)?.host;
            }
        }
        Some(ResolvedImage {
            components: vk::ComponentMapping {
                r: components[0],
                g: components[1],
                b: components[2],
                a: components[3],
            },
            descriptor,
            packing,
            storage: r.storage,
        })
    }
}

/// Public host format block geometry. Payloads stay opaque during transfers.
pub fn format_block(format: vk::Format) -> Option<BlockFormat> {
    use vk::Format as F;
    let bytes = match format {
        F::R8_UNORM | F::R8_SNORM | F::R8_UINT | F::R8_SINT | F::R8_SRGB | F::S8_UINT => 1,
        F::R8G8_UNORM
        | F::R8G8_SNORM
        | F::R8G8_UINT
        | F::R8G8_SINT
        | F::R8G8_SRGB
        | F::R16_UNORM
        | F::R16_SNORM
        | F::R16_UINT
        | F::R16_SINT
        | F::R16_SFLOAT
        | F::D16_UNORM
        | F::R4G4B4A4_UNORM_PACK16
        | F::B4G4R4A4_UNORM_PACK16
        | F::R5G6B5_UNORM_PACK16
        | F::B5G6R5_UNORM_PACK16
        | F::R5G5B5A1_UNORM_PACK16
        | F::B5G5R5A1_UNORM_PACK16
        | F::A1R5G5B5_UNORM_PACK16 => 2,
        F::R8G8B8A8_UNORM
        | F::R8G8B8A8_SNORM
        | F::R8G8B8A8_UINT
        | F::R8G8B8A8_SINT
        | F::R8G8B8A8_SRGB
        | F::B8G8R8A8_UNORM
        | F::B8G8R8A8_SRGB
        | F::R16G16_UNORM
        | F::R16G16_SNORM
        | F::R16G16_UINT
        | F::R16G16_SINT
        | F::R16G16_SFLOAT
        | F::R32_UINT
        | F::R32_SINT
        | F::R32_SFLOAT
        | F::D32_SFLOAT
        | F::A2B10G10R10_UNORM_PACK32
        | F::A2R10G10B10_UNORM_PACK32
        | F::A2R10G10B10_UINT_PACK32
        | F::A2B10G10R10_UINT_PACK32
        | F::B10G11R11_UFLOAT_PACK32
        | F::E5B9G9R9_UFLOAT_PACK32
        | F::X8_D24_UNORM_PACK32 => 4,
        F::D16_UNORM_S8_UINT | F::D24_UNORM_S8_UINT => 4,
        F::D32_SFLOAT_S8_UINT => 8,
        F::R16G16B16A16_UNORM
        | F::R16G16B16A16_SNORM
        | F::R16G16B16A16_UINT
        | F::R16G16B16A16_SINT
        | F::R16G16B16A16_SFLOAT
        | F::R32G32_UINT
        | F::R32G32_SINT
        | F::R32G32_SFLOAT => 8,
        F::R32G32B32A32_UINT | F::R32G32B32A32_SINT | F::R32G32B32A32_SFLOAT => 16,
        _ => {
            let raw = format.as_raw();
            if (F::BC1_RGB_UNORM_BLOCK.as_raw()..=F::BC7_SRGB_BLOCK.as_raw()).contains(&raw) {
                let bytes = if matches!(
                    format,
                    F::BC1_RGB_UNORM_BLOCK
                        | F::BC1_RGB_SRGB_BLOCK
                        | F::BC1_RGBA_UNORM_BLOCK
                        | F::BC1_RGBA_SRGB_BLOCK
                        | F::BC4_UNORM_BLOCK
                        | F::BC4_SNORM_BLOCK
                ) {
                    8
                } else {
                    16
                };
                return Some(BlockFormat {
                    width: 4,
                    height: 4,
                    bytes,
                });
            }
            let first = F::ASTC_4X4_UNORM_BLOCK.as_raw();
            let last = F::ASTC_12X12_SRGB_BLOCK.as_raw();
            if (first..=last).contains(&raw) {
                const EXTENTS: [(u8, u8); 14] = [
                    (4, 4),
                    (5, 4),
                    (5, 5),
                    (6, 5),
                    (6, 6),
                    (8, 5),
                    (8, 6),
                    (8, 8),
                    (10, 5),
                    (10, 6),
                    (10, 8),
                    (10, 10),
                    (12, 10),
                    (12, 12),
                ];
                let (width, height) = EXTENTS[((raw - first) / 2) as usize];
                return Some(BlockFormat {
                    width,
                    height,
                    bytes: 16,
                });
            }
            return None;
        }
    };
    Some(BlockFormat {
        width: 1,
        height: 1,
        bytes,
    })
}

pub(crate) fn format_aspects(format: vk::Format) -> vk::ImageAspectFlags {
    use vk::Format as F;
    match format {
        F::D16_UNORM | F::X8_D24_UNORM_PACK32 | F::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
        F::S8_UINT => vk::ImageAspectFlags::STENCIL,
        F::D16_UNORM_S8_UINT | F::D24_UNORM_S8_UINT | F::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::COLOR,
    }
}

/// A checked texel-space box in one image mip and array layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageRegion {
    /// Zero-based mip level.
    pub level: u32,
    /// Zero-based array layer.
    pub layer: u32,
    /// Origin of the region in texels.
    pub offset: [u32; 3],
    /// Extent of the region in texels.
    pub extent: [u32; 3],
}

/// One mip/layer and two signed texel corners defining a blit region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlitRegion {
    /// Zero-based mip level.
    pub level: u32,
    /// Zero-based array layer.
    pub layer: u32,
    /// First and opposite corners in signed texel coordinates.
    pub offsets: [[i32; 3]; 2],
}

/// Host interpretation of a recorded opaque image-copy argument set.
#[derive(Clone, Copy)]
pub enum CopyOperation {
    /// Equal-format transfer between explicit image regions.
    Copy {
        /// Source image object key.
        source: u64,
        /// Destination image object key.
        destination: u64,
        /// Source subresource and region.
        from: ImageRegion,
        /// Destination subresource and region.
        to: ImageRegion,
    },
    /// Filtered transfer between explicit image regions.
    Blit {
        /// Source image object key.
        source: u64,
        /// Destination image object key.
        destination: u64,
        /// Source subresource and region.
        from: BlitRegion,
        /// Destination subresource and region.
        to: BlitRegion,
        /// Vulkan filter used for the blit.
        filter: vk::Filter,
    },
}

/// Thread-safe host decoder for eight opaque recorded copy argument registers.
pub type CopyDecoder = std::sync::Arc<dyn Fn([u64; 8]) -> Option<CopyOperation> + Send + Sync>;

/// Numeric type used when sampling a public Vulkan format.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NumericClass {
    /// Floating-point samples, including normalized and scaled formats.
    Float,
    /// Unsigned integer samples.
    Unsigned,
    /// Signed integer samples.
    Signed,
    /// Depth samples represented as floating-point values.
    Depth,
}
/// Classify integer and depth formats; other formats use float samples.
/// Every 8-, 16- and 32-bit-component `_UINT` and `_SINT` colour format is listed.
pub fn numeric_class(format: vk::Format) -> NumericClass {
    use vk::Format as F;
    match format {
        F::S8_UINT
        | F::A2R10G10B10_UINT_PACK32
        | F::A2B10G10R10_UINT_PACK32
        | F::R8_UINT
        | F::R8G8_UINT
        | F::R8G8B8A8_UINT
        | F::R16_UINT
        | F::R16G16_UINT
        | F::R16G16B16A16_UINT
        | F::R32_UINT
        | F::R32G32_UINT
        | F::R32G32B32A32_UINT
        | F::R8G8B8_UINT
        | F::B8G8R8_UINT
        | F::B8G8R8A8_UINT
        | F::A8B8G8R8_UINT_PACK32
        | F::R16G16B16_UINT
        | F::R32G32B32_UINT => NumericClass::Unsigned,
        F::R8_SINT
        | F::R8G8_SINT
        | F::R8G8B8A8_SINT
        | F::R16_SINT
        | F::R16G16_SINT
        | F::R16G16B16A16_SINT
        | F::R32_SINT
        | F::R32G32_SINT
        | F::R32G32B32A32_SINT
        | F::R8G8B8_SINT
        | F::B8G8R8_SINT
        | F::B8G8R8A8_SINT
        | F::A8B8G8R8_SINT_PACK32
        | F::A2R10G10B10_SINT_PACK32
        | F::A2B10G10R10_SINT_PACK32
        | F::R16G16B16_SINT
        | F::R32G32B32_SINT => NumericClass::Signed,
        F::D16_UNORM
        | F::X8_D24_UNORM_PACK32
        | F::D32_SFLOAT
        | F::D16_UNORM_S8_UINT
        | F::D24_UNORM_S8_UINT
        | F::D32_SFLOAT_S8_UINT => NumericClass::Depth,
        _ => NumericClass::Float,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_depth_stencil_formats_have_copy_geometry() {
        assert_eq!(
            format_block(vk::Format::D16_UNORM_S8_UINT).unwrap().bytes,
            4
        );
        assert_eq!(
            format_block(vk::Format::D24_UNORM_S8_UINT).unwrap().bytes,
            4
        );
        assert_eq!(
            format_block(vk::Format::D32_SFLOAT_S8_UINT).unwrap().bytes,
            8
        );
    }

    #[test]
    fn opaque_builder_fields_need_an_exact_rule() {
        let rule = ImageRule {
            evidence: [crate::gpu::image_enums::Evidence::Hypothesis; 3],
            rationale: ["original synthetic fixture"; 3],
            flags: 17,
            target: 19,
            format: 23,
            host_format: vk::Format::BC1_RGBA_UNORM_BLOCK,
            kind: ImageKind::D2Array,
            coordinates: Coordinates::Normalized,
            storage: Storage::Tiled(TileShape {
                height_log2: 3,
                depth_log2: 0,
            }),
        };
        let mut contract = ImageContract {
            rules: vec![rule],
            ..Default::default()
        };
        contract.validate().unwrap();
        let mut description = TextureDescription {
            flags: 17,
            target: 19,
            format: 23,
            width: 37,
            height: 29,
            depth: 3,
            levels: 6,
            ..Default::default()
        };
        let image = contract.resolve(&description).unwrap();
        assert_eq!(image.descriptor.shape.layers, 3);
        assert_eq!(image.packing.levels()[0].blocks, [10, 8, 1]);
        assert_eq!(image.packing.levels()[0].tile.height_log2, 0);
        description.flags = 18;
        assert!(contract.resolve(&description).is_none());
        description.flags = 17;
        description.stride = 80;
        assert!(contract.resolve(&description).is_none());
        contract.rules.push(rule);
        assert!(contract.validate().is_err());
    }
}
