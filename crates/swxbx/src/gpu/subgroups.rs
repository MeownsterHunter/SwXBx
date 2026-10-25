//! Fixed subgroup requirements from public stage limits.
use ash::vk;

/// Enabled device controls and limits for translated subgroup operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct SubgroupFeatures {
    /// Whether pipeline stages may request a fixed subgroup size.
    pub size_control: bool,
    /// Whether compute stages may require every lane to be active.
    pub compute_full_subgroups: bool,
    /// Smallest supported subgroup size.
    pub min_size: u32,
    /// Largest supported subgroup size.
    pub max_size: u32,
    /// Stage bits that permit a required size.
    pub required_stages: u32,
    /// Maximum number of subgroups in a compute workgroup.
    pub max_compute_workgroup_subgroups: u32,
}

impl SubgroupFeatures {
    pub(super) fn new(
        features: &vk::PhysicalDeviceSubgroupSizeControlFeatures<'_>,
        properties: &vk::PhysicalDeviceSubgroupSizeControlProperties<'_>,
    ) -> Self {
        Self {
            size_control: features.subgroup_size_control != vk::FALSE,
            compute_full_subgroups: features.compute_full_subgroups != vk::FALSE,
            min_size: properties.min_subgroup_size,
            max_size: properties.max_subgroup_size,
            required_stages: properties.required_subgroup_size_stages.as_raw(),
            max_compute_workgroup_subgroups: properties.max_compute_workgroup_subgroups,
        }
    }

    /// Check whether this stage supports the required 32-lane subgroup size.
    pub fn validate(self, stage: vk::ShaderStageFlags) -> Result<(), String> {
        let name = match stage {
            vk::ShaderStageFlags::VERTEX => "vertex",
            vk::ShaderStageFlags::FRAGMENT => "fragment",
            vk::ShaderStageFlags::COMPUTE => "compute",
            _ => return Err("unsupported subgroup stage".into()),
        };
        if !self.size_control
            || self.min_size > 32
            || self.max_size < 32
            || self.required_stages & stage.as_raw() == 0
        {
            return Err(format!(
                "{name} stage requires subgroup size 32, unavailable on this device"
            ));
        }
        if stage == vk::ShaderStageFlags::COMPUTE && !self.compute_full_subgroups {
            return Err(
                "compute stage requires full 32-lane subgroups, unavailable on this device".into(),
            );
        }
        Ok(())
    }

    pub(super) fn validate_compute(self, words: &[u32]) -> Result<(), String> {
        self.validate(vk::ShaderStageFlags::COMPUTE)?;
        let instructions = super::graphics::instructions(words)?;
        let local = instructions
            .iter()
            .find_map(|&(op, args)| match (op, args) {
                (16, [_, 17, x, y, z]) => Some([*x, *y, *z]),
                _ => None,
            })
            .ok_or("required subgroup size 32 needs a literal compute local size")?;
        let total = local
            .into_iter()
            .try_fold(1_u64, |n, size| n.checked_mul(u64::from(size)))
            .ok_or("compute local size overflow")?;
        if local.contains(&0) || !local[0].is_multiple_of(32) {
            return Err("full 32-lane subgroups require compute local size X to be a positive multiple of 32".into());
        }
        if total > 32 * u64::from(self.max_compute_workgroup_subgroups) {
            return Err("compute local size exceeds the device limit for 32-lane subgroups".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
