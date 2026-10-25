//! Retained worker-owned shader modules.
use super::{
    uniforms::{self, UniformStage},
    Context,
};
use ash::vk;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[derive(Hash, PartialEq, Eq)]
struct Key {
    words: Vec<u32>,
    stage: UniformStage,
    storage: bool,
}

pub(super) struct ShaderModule {
    context: Arc<Context>,
    pub(super) handle: vk::ShaderModule,
}
impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_shader_module(self.handle, None);
        }
    }
}

#[derive(Default)]
pub(super) struct ShaderModules(Mutex<HashMap<Key, Arc<ShaderModule>>>);
impl ShaderModules {
    pub(super) fn get(
        &self,
        context: &Arc<Context>,
        words: &[u32],
        stage: UniformStage,
        storage: bool,
    ) -> Result<Arc<ShaderModule>, String> {
        let key = Key {
            words: words.to_vec(),
            stage,
            storage,
        };
        // Only compiler workers access this cache.
        let mut modules = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(module) = modules.get(&key) {
            return Ok(module.clone());
        }
        #[cfg(feature = "draw-metrics")]
        let _span = crate::draw_metrics::PipelineSpan::new(2);
        let words = uniforms::lower(words, stage, storage)?;
        let module = Arc::new(ShaderModule {
            context: context.clone(),
            handle: context.create_global_shader_module(&words)?,
        });
        modules.insert(key, module.clone());
        Ok(module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a graphics device; never skips"]
    fn modules_survive_users_and_separate_storage_choices() {
        let context = Arc::new(Context::new().expect("graphics context"));
        // Empty vertex entry point using public instruction encodings.
        let mut words = vec![0x0723_0203, 0x0001_0300, 0, 5, 0];
        for (op, args) in [
            (17, vec![1]),
            (14, vec![0, 1]),
            (15, vec![0, 3, 0x6e69_616d, 0]),
            (19, vec![1]),
            (33, vec![2, 1]),
            (54, vec![1, 3, 0, 2]),
            (248, vec![4]),
            (253, vec![]),
            (56, vec![]),
        ] {
            words.push(((args.len() as u32 + 1) << 16) | op);
            words.extend(args);
        }
        let cache = ShaderModules::default();
        let first = cache
            .get(&context, &words, UniformStage::Vertex, false)
            .unwrap();
        let retained = Arc::downgrade(&first);
        drop(first);
        let second = cache
            .get(&context, &words, UniformStage::Vertex, false)
            .unwrap();
        assert!(Arc::ptr_eq(&retained.upgrade().unwrap(), &second));
        let storage = cache
            .get(&context, &words, UniformStage::Vertex, true)
            .unwrap();
        assert!(!Arc::ptr_eq(&second, &storage));
        assert!(cache
            .get(&context, &[], UniformStage::Vertex, false)
            .is_err());
        drop(cache);
        assert!(retained.upgrade().is_some());
        drop(second);
        assert!(retained.upgrade().is_none());
    }
}
