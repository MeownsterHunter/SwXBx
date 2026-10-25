//! Stage-local constant banks. Sources and experiment limits:
use super::{graphics::instructions, pipelines::stage_bindings};
use ash::vk;
use std::collections::{HashMap, HashSet};

pub(crate) const BANK_SIZE: u64 = 4096 * 16;

/// A translated graphics stage, independent of recorded stage tokens.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum UniformStage {
    /// Vertex shader stage.
    Vertex,
    /// Fragment shader stage.
    Fragment,
}
impl UniformStage {
    pub(crate) fn model(self) -> u32 {
        match self {
            Self::Vertex => 0,
            Self::Fragment => 4,
        }
    }
    pub(crate) fn set(self) -> u32 {
        match self {
            Self::Vertex => 0,
            Self::Fragment => 1,
        }
    }
    pub(crate) fn flags(self) -> vk::ShaderStageFlags {
        match self {
            Self::Vertex => vk::ShaderStageFlags::VERTEX,
            Self::Fragment => vk::ShaderStageFlags::FRAGMENT,
        }
    }
}

/// Host-selected interpretation of one recorded stage/index pair.
/// Client stage meanings and binding-to-bank mappings remain unresolved.
#[derive(Clone, Copy, Debug)]
pub struct UniformBankMapping {
    /// Raw stage token supplied by the caller.
    pub stage: u64,
    /// Raw binding index supplied by the caller.
    pub index: u64,
    /// Translated shader stage receiving the bank.
    pub target: UniformStage,
    /// Uniform bank number in the translated module.
    pub bank: u32,
}

/// Opt-in bank mapping for the bounded draw experiment. No implicit mapping exists.
#[derive(Clone, Debug, Default)]
pub struct UniformBufferContract {
    /// Explicit mappings used to connect recorded slots to host resources.
    pub bindings: Vec<UniformBankMapping>,
    /// Use read-only storage buffers even when 64 KiB uniform blocks fit.
    /// Otherwise the backend selects storage only when the device limit requires it.
    pub storage_buffers: bool,
}
impl UniformBufferContract {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let mut sources = HashSet::new();
        let mut targets = HashSet::new();
        for m in &self.bindings {
            if m.bank >= 32
                || !sources.insert((m.stage, m.index))
                || !targets.insert((m.target, m.bank))
            {
                return Err("duplicate or invalid uniform bank mapping".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Bank {
    /// Translated shader stage.
    pub stage: UniformStage,
    /// Uniform bank number in the translated module.
    pub bank: u32,
}

/// Accept only the translator's set-zero Block { float4[4096] } bank interface.
/// Generic descriptor reflection still checks entry points, arrays and push constants.
pub(crate) fn banks(words: &[u32], stage: UniformStage) -> Result<Vec<Bank>, String> {
    let bindings: Vec<_> = stage_bindings(words, stage.model())?
        .into_iter()
        .filter(|b| b.descriptor_type == vk::DescriptorType::UNIFORM_BUFFER)
        .collect();
    super::textures::bindings(words, stage)?;
    if bindings.iter().any(|b| {
        b.set != 0
            || b.binding >= 32
            || b.count != 1
            || b.descriptor_type != vk::DescriptorType::UNIFORM_BUFFER
    }) {
        return Err("draw supports only constant bank descriptors".into());
    }
    let instructions = instructions(words)?;
    let mut types = HashMap::new();
    let mut constants = HashMap::new();
    let mut strides = HashMap::new();
    let mut offsets = HashMap::new();
    for &(op, args) in &instructions {
        match (op, args) {
            (19..=33, [id, ..]) => {
                types.insert(*id, (op, args));
            }
            (43, [_, id, value]) => {
                constants.insert(*id, *value);
            }
            (71, [id, 6, stride]) => {
                strides.insert(*id, *stride);
            }
            (72, [id, member, 35, offset]) => {
                offsets.insert((*id, *member), *offset);
            }
            _ => {}
        }
    }
    for &(op, args) in &instructions {
        let (59, [pointer, _, 2, ..]) = (op, args) else {
            continue;
        };
        let Some(&(32, [_, 2, structure])) = types.get(pointer) else {
            return Err("invalid constant bank pointer".into());
        };
        let Some(&(30, [_, array])) = types.get(structure) else {
            return Err("expected one constant bank array".into());
        };
        let Some(&(28, [_, vector, length])) = types.get(array) else {
            return Err("expected fixed constant bank array".into());
        };
        let Some(&(23, [_, scalar, 4])) = types.get(vector) else {
            return Err("expected constant bank float4".into());
        };
        if !matches!(types.get(scalar), Some((22, [_, 32])))
            || constants.get(length) != Some(&4096)
            || strides.get(array) != Some(&16)
            || offsets.get(&(*structure, 0)) != Some(&0)
        {
            return Err("unsupported constant bank layout".into());
        }
    }
    Ok(bindings
        .into_iter()
        .map(|b| Bank {
            stage,
            bank: b.binding,
        })
        .collect())
}

/// Keep bank numbers, separate stage sets, and optionally change Uniform pointers
/// and variables to StorageBuffer. Block layout and all loads remain unchanged.
pub(crate) fn lower(words: &[u32], stage: UniformStage, storage: bool) -> Result<Vec<u32>, String> {
    banks(words, stage)?;
    if storage && words[1] < 0x0001_0300 {
        return Err("storage constant banks require SPIR-V 1.3".into());
    }
    let instructions = instructions(words)?;
    let variables: Vec<_> = instructions
        .iter()
        .filter_map(|(op, args)| match (*op, *args) {
            (59, [_, variable, 2, ..]) => Some(*variable),
            _ => None,
        })
        .collect();
    let textures: Vec<_> = instructions
        .iter()
        .filter_map(|(op, args)| match (*op, *args) {
            (59, [_, variable, 0, ..]) => Some(*variable),
            _ => None,
        })
        .collect();
    let sets: HashMap<_, _> = instructions
        .iter()
        .filter_map(|(op, args)| match (*op, *args) {
            (71, [variable, 34, set]) => Some((*variable, *set)),
            _ => None,
        })
        .collect();
    let mut result = words[..5].to_vec();
    let mut decorated = false;
    for (op, args) in instructions {
        if storage && !decorated && (19..=39).contains(&op) {
            for variable in &variables {
                result.extend([3 << 16 | 71, *variable, 24]); // NonWritable.
            }
            decorated = true;
        }
        let mut args = args.to_vec();
        match op {
            71 if args.get(1) == Some(&34) => {
                args[2] = if textures.contains(&args[0]) {
                    stage.set() + 2
                } else {
                    stage.set()
                }
            }
            71 if args.get(1) == Some(&33) && textures.contains(&args[0]) => {
                args[2] += sets[&args[0]] * 256
            }
            32 if storage && args.get(1) == Some(&2) => args[1] = 12,
            59 if storage && args.get(2) == Some(&2) => args[2] = 12,
            _ => {}
        }
        result.push(((args.len() as u32 + 1) << 16) | op);
        result.extend(args);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn instruction(words: &mut Vec<u32>, op: u32, args: &[u32]) {
        words.push(((args.len() as u32 + 1) << 16) | op);
        words.extend(args);
    }
    fn bank() -> Vec<u32> {
        let mut words = vec![0x0723_0203, 0x0001_0300, 0, 100, 0];
        for (op, args) in [
            (17, vec![1]),
            (14, vec![0, 1]),
            (15, vec![4, 90, 0x6e69_616d, 0, 91]),
            (16, vec![90, 7]),
            (71, vec![91, 30, 0]),
            (71, vec![8, 34, 0]),
            (71, vec![8, 33, 2]),
            (71, vec![5, 2]),
            (71, vec![4, 6, 16]),
            (72, vec![5, 0, 35, 0]),
            (22, vec![1, 32]),
            (23, vec![2, 1, 4]),
            (21, vec![3, 32, 0]),
            (43, vec![3, 9, 4096]),
            (28, vec![4, 2, 9]),
            (30, vec![5, 4]),
            (32, vec![6, 2, 5]),
            (32, vec![7, 2, 2]),
            (59, vec![6, 8, 2]),
            (19, vec![10]),
            (33, vec![12, 10]),
            (32, vec![11, 3, 2]),
            (43, vec![3, 16, 0]),
            (59, vec![11, 91, 3]),
            (54, vec![10, 90, 0, 12]),
            (248, vec![92]),
            (65, vec![7, 15, 8, 16, 16]),
            (61, vec![2, 17, 15]),
            (62, vec![91, 17]),
            (253, vec![]),
            (56, vec![]),
        ] {
            instruction(&mut words, op, &args);
        }
        words
    }
    #[test]
    fn reflects_only_complete_constant_bank_shapes() {
        let words = bank();
        assert_eq!(
            banks(&words, UniformStage::Fragment).unwrap(),
            vec![Bank {
                stage: UniformStage::Fragment,
                bank: 2
            }]
        );
        for (op, operand, value) in [(43, 2, 1024), (71, 2, 8), (72, 3, 4), (22, 1, 16)] {
            let mut changed = words.clone();
            let mut at = 5;
            while at < changed.len() {
                let count = (changed[at] >> 16) as usize;
                let args = &changed[at + 1..at + count];
                if changed[at] & 0xffff == op && (op != 71 || args[1] == 6) {
                    changed[at + 1 + operand] = value;
                    break;
                }
                at += count;
            }
            assert!(banks(&changed, UniformStage::Fragment).is_err());
        }
    }
    #[test]
    fn keeps_binding_and_layout_while_separating_sets_and_lowering_storage() {
        let words = bank();
        let uniform = lower(&words, UniformStage::Fragment, false).unwrap();
        let storage = lower(&words, UniformStage::Fragment, true).unwrap();
        for (words, kind) in [
            (&uniform, vk::DescriptorType::UNIFORM_BUFFER),
            (&storage, vk::DescriptorType::STORAGE_BUFFER),
        ] {
            let binding = stage_bindings(words, 4).unwrap()[0];
            assert_eq!((binding.set, binding.binding, binding.count), (1, 2, 1));
            assert_eq!(binding.descriptor_type.as_raw(), kind.as_raw());
            let instructions = instructions(words).unwrap();
            assert!(instructions.contains(&(71, &[4, 6, 16])));
            assert!(instructions.contains(&(72, &[5, 0, 35, 0])));
        }
        assert!(instructions(&storage).unwrap().contains(&(71, &[8, 24])));
    }
    #[test]
    #[ignore = "requires spirv-val"]
    fn validates_uniform_and_storage_bank_rewrites() {
        use std::{fs, process::Command};
        let directory =
            std::env::temp_dir().join(format!("uniform-validation-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        for storage in [false, true] {
            let words = lower(&bank(), UniformStage::Fragment, storage).unwrap();
            let path = directory.join(format!("bank-{storage}.spv"));
            fs::write(
                &path,
                words
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let result = Command::new("spirv-val")
                .args(["--target-env", "vulkan1.2"])
                .arg(&path)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "bank validation: {}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
        fs::remove_dir_all(directory).unwrap();
    }
}
