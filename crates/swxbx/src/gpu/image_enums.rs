//! Evidence-tagged host interpretations.
use super::image_layout::{format_block, ImageContract, ImageRule, Storage};
use crate::tiling::ImageKind;
use ash::vk;
use std::collections::HashSet;

/// Confidence concerns the guest interpretation, rather than host support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence {
    /// Supported by recorded observations or permitted public evidence.
    Established,
    /// Host interpretation that still requires confirming guest evidence.
    Hypothesis,
}

/// A host must name the recorded evidence or the hypothesis behind each rule.
#[derive(Clone, Copy, Debug)]
pub struct EnumRule<T> {
    /// Raw caller token matched exactly by this rule.
    pub guest: u64,
    /// Typed host value selected for the caller token.
    pub host: T,
    /// Confidence in the guest interpretation.
    pub evidence: Evidence,
    /// Nonempty description of the supporting evidence or hypothesis.
    pub rationale: &'static str,
}

pub(crate) fn validate_rules<T>(rules: &[EnumRule<T>]) -> Result<(), String> {
    let mut keys = HashSet::new();
    if rules
        .iter()
        .any(|r| !keys.insert(r.guest) || r.rationale.trim().is_empty())
    {
        return Err("duplicate enum token or missing rationale".into());
    }
    Ok(())
}

/// Factorized rules still require exact flags, target and format matches.
/// The recorded evidence does not supply a default guest enum table.
#[derive(Clone, Default)]
pub struct ImageEnums {
    /// Explicit caller format tokens mapped to public Vulkan formats.
    pub formats: Vec<EnumRule<vk::Format>>,
    /// Explicit caller target tokens mapped to image dimensions and layers.
    pub targets: Vec<EnumRule<ImageKind>>,
    /// Exact caller flags mapped to linear or tiled storage.
    pub storage: Vec<EnumRule<Storage>>,
    /// Caller component selectors; an empty table retains identity selection.
    pub swizzles: Vec<EnumRule<vk::ComponentSwizzle>>,
    /// Caller selectors mapped to one depth or stencil aspect.
    pub depth_stencil_modes: Vec<EnumRule<vk::ImageAspectFlags>>,
}

impl ImageEnums {
    /// Build and validate every exact storage, target and format combination.
    pub fn contract(&self) -> Result<ImageContract, String> {
        validate_rules(&self.formats)?;
        validate_rules(&self.targets)?;
        validate_rules(&self.storage)?;
        validate_rules(&self.swizzles)?;
        validate_rules(&self.depth_stencil_modes)?;
        let mut rules = Vec::new();
        for storage in &self.storage {
            for target in &self.targets {
                for format in &self.formats {
                    rules.push(ImageRule {
                        evidence: [storage.evidence, target.evidence, format.evidence],
                        rationale: [storage.rationale, target.rationale, format.rationale],
                        flags: storage.guest,
                        target: target.guest,
                        format: format.guest,
                        host_format: format.host,
                        kind: target.host,
                        storage: storage.host,
                    });
                }
            }
        }
        let contract = ImageContract {
            rules,
            swizzles: self.swizzles.clone(),
            depth_stencil_modes: self.depth_stencil_modes.clone(),
        };
        contract.validate()?;
        Ok(contract)
    }
}

/// Host formats whose single-aspect payloads can use the transfer path.
/// Host format semantics are established by the public format specification.
/// This list establishes no correspondence with guest enum numbers.
pub fn transfer_formats() -> Vec<vk::Format> {
    (1..=184)
        .map(vk::Format::from_raw)
        .filter(|f| format_block(*f).is_some())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::TextureDescription;

    fn hypothesis<T>(guest: u64, host: T) -> EnumRule<T> {
        EnumRule {
            guest,
            host,
            evidence: Evidence::Hypothesis,
            rationale: "original synthetic fixture",
        }
    }

    #[test]
    fn depth_and_stencil_modes_require_the_selected_aspect() {
        let mut enums = ImageEnums {
            formats: vec![hypothesis(23, vk::Format::D16_UNORM)],
            targets: vec![hypothesis(19, ImageKind::D2)],
            storage: vec![hypothesis(17, Storage::Linear)],
            depth_stencil_modes: vec![hypothesis(7, vk::ImageAspectFlags::DEPTH)],
            ..Default::default()
        };
        let mut description = TextureDescription {
            flags: 17,
            target: 19,
            format: 23,
            width: 1,
            height: 1,
            depth_stencil_mode: 7,
            ..Default::default()
        };
        assert!(enums.contract().unwrap().resolve(&description).is_some());
        description.depth_stencil_mode = 8;
        assert!(enums.contract().unwrap().resolve(&description).is_none());
        description.depth_stencil_mode = 7;
        enums.formats[0].host = vk::Format::S8_UINT;
        assert!(enums.contract().unwrap().resolve(&description).is_none());
        enums.depth_stencil_modes[0].host = vk::ImageAspectFlags::STENCIL;
        assert!(enums.contract().unwrap().resolve(&description).is_some());
        enums.depth_stencil_modes[0].host =
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
        assert!(enums.contract().is_err());
    }

    #[test]
    fn factorized_rules_preserve_exact_matching_and_unknown_rejection() {
        let mut enums = ImageEnums {
            formats: vec![hypothesis(23, vk::Format::R8G8B8A8_UINT)],
            targets: vec![hypothesis(19, ImageKind::D2)],
            storage: vec![hypothesis(17, Storage::Linear)],
            swizzles: vec![hypothesis(2, vk::ComponentSwizzle::R)],
            ..Default::default()
        };
        let contract = enums.contract().unwrap();
        let mut description = TextureDescription {
            flags: 17,
            target: 19,
            format: 23,
            width: 1,
            height: 1,
            swizzle: [2; 4],
            ..Default::default()
        };
        let image = contract.resolve(&description).unwrap();
        assert!(image.components.r == vk::ComponentSwizzle::R);
        description.swizzle[3] = 9;
        assert!(contract.resolve(&description).is_none());
        description.swizzle = [2; 4];
        description.flags = 18;
        assert!(contract.resolve(&description).is_none());
        enums.formats.push(enums.formats[0]);
        assert!(enums.contract().is_err());
    }
}
