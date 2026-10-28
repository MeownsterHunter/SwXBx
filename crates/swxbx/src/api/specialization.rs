//! Translation specialized by the textures a draw binds.
//!
//! NVN's RECTANGLE target samples with texel coordinates, in [0, extent],
//! where every other target samples in [0, 1]. The guest driver records that
//! in the texture descriptor (the reference host Gpu/Image/TextureDescriptor.cs:203-208),
//! and the shader code is the same either way, so the coordinate space is
//! translation state: the reference host keys a program on it
//! (Gpu/Shader/GpuAccessor.cs:186-190). A program registers before any
//! texture is bound, so its stages translate first with every sampled image
//! normalized. A draw whose bound textures include a rectangle one gets the
//! stages that sample it translated again, with those sampled images named
//! in [`TranslationContext::texel_images`]. The integrator states nothing:
//! the texture's target says it, through the image contract's
//! [`Coordinates`] for that target.
//!
//! A geometry stage's code does not name the primitive it reads either: the
//! draw's topology does (the reference host Gpu/Shader/GpuAccessor.cs
//! QueryPrimitiveTopology). Its code is held untranslated, and each draw
//! gets it translated for the primitive that draw assembles.
use super::{drawing::TextureReference as Bound, objects::Object};
use crate::{
    gpu::{
        image_layout::{Coordinates, ImageContract},
        textures::TextureContract,
        uniforms::GraphicsStage,
    },
    instance::InstanceState as Instance,
    startup_cache::{InputPrimitive, SampledImage, TranslatedShader, TranslationContext},
    Status,
};
use ash::vk;
use std::{
    collections::{hash_map::Entry, HashMap},
    sync::Arc,
};

/// Sampled images that read texel coordinates, by the translated stage
/// that declares them.
pub(super) struct TexelImages(Vec<(GraphicsStage, SampledImage)>);

impl TexelImages {
    /// The contract's sampled images whose bound texture's target reads
    /// texel coordinates. A slot that does not resolve to a texture names
    /// nothing here; the draw's texture binding refuses it if a stage
    /// samples it.
    pub fn bound<'a>(
        instance: &Instance,
        textures: &TextureContract,
        images: &ImageContract,
        pool: Option<u64>,
        bound: impl Fn(u64, u64) -> Option<&'a Bound>,
    ) -> Self {
        let Some(pool) = pool else {
            return Self(Vec::new());
        };
        Self(
            textures
                .bindings
                .iter()
                .filter_map(|mapping| {
                    let id = match *bound(mapping.stage, mapping.index)? {
                        Bound::Separate(handle) => u32::try_from(handle).ok()?,
                        Bound::Combined(handle) => {
                            textures.combined.iter().find(|(h, _, _)| *h == handle)?.1
                        }
                    };
                    let texture = instance
                        .objects
                        .with(pool, |object| match object {
                            Object::TexturePool { registered, .. } => {
                                registered.get(&id).map(|&(texture, _)| texture)
                            }
                            _ => None,
                        })
                        .flatten()?;
                    let coordinates = instance
                        .objects
                        .with(texture, |object| match object {
                            Object::Texture { description, .. } => images.coordinates(description),
                            _ => None,
                        })
                        .flatten()?;
                    (coordinates == Coordinates::Texels).then_some((
                        mapping.target,
                        SampledImage {
                            set: mapping.set,
                            binding: mapping.image,
                        },
                    ))
                })
                .collect(),
        )
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Specialized translations, and the sampled images each base translation
/// declares. Both hold the base translation, so its address names it for
/// as long as the entry lives.
#[derive(Default)]
pub(crate) struct Specializations {
    declared: HashMap<usize, (Arc<TranslatedShader>, GraphicsStage, Vec<SampledImage>)>,
    translated:
        HashMap<(usize, TranslationContext), (Arc<TranslatedShader>, Arc<TranslatedShader>)>,
    /// Geometry stages by their held code's address and the primitive they
    /// read. Each entry holds the code, so its address names it.
    primitives: HashMap<(usize, InputPrimitive), PrimitiveTranslation>,
}

/// Held geometry code and its translation for one primitive.
type PrimitiveTranslation = (Arc<Vec<u8>>, Arc<TranslatedShader>);

impl Specializations {
    /// The translation of a stage's held code for the primitive the draw
    /// assembles.
    pub(super) fn by_primitive(
        &mut self,
        instance: &Instance,
        program: u64,
        code: &Arc<Vec<u8>>,
        primitive: InputPrimitive,
    ) -> Result<Arc<TranslatedShader>, Status> {
        let key = (Arc::as_ptr(code) as usize, primitive);
        if let Some((_, translated)) = self.primitives.get(&key) {
            return Ok(Arc::clone(translated));
        }
        let refuse = |error: &dyn std::fmt::Display| {
            crate::unsupported::refuse(
                Status::Unimplemented,
                format_args!(
                    "draw: translating a geometry stage for the {primitive:?} the draw \
                     assembles failed: {error}"
                ),
            )
        };
        let translator = instance
            .program_translator(program)
            .ok_or_else(|| refuse(&"the program registered under no shader translator"))?;
        let context = TranslationContext {
            geometry_input: Some(primitive),
            ..instance.translation_context()
        };
        let translated = Arc::new(
            translator
                .translate_for_context(crate::SHADER_STAGE_UNKNOWN, code, &context)
                .map_err(|error| refuse(&error))?,
        );
        self.primitives
            .insert(key, (Arc::clone(code), Arc::clone(&translated)));
        Ok(translated)
    }

    /// The translation of `base`'s stage for the draw's texel images: `base`
    /// itself when the stage samples none of them. `geometry_input` is the
    /// primitive `base` was translated for, which a geometry stage's
    /// retranslation keeps.
    pub(super) fn stage(
        &mut self,
        instance: &Instance,
        program: u64,
        index: usize,
        base: &Arc<TranslatedShader>,
        geometry_input: Option<InputPrimitive>,
        texels: &TexelImages,
    ) -> Result<Arc<TranslatedShader>, Status> {
        let address = Arc::as_ptr(base) as usize;
        let (_, stage, declared) = match self.declared.entry(address) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let refuse = |error: String| {
                    crate::unsupported::refuse(
                        Status::Unimplemented,
                        format_args!("draw: reading a translated stage's sampled images: {error}"),
                    )
                };
                let stage = GraphicsStage::of(&base.words).map_err(refuse)?;
                let images = crate::gpu::textures::bindings(&base.words, stage)
                    .map_err(refuse)?
                    .into_iter()
                    .filter(|binding| binding.ty == vk::DescriptorType::SAMPLED_IMAGE)
                    .map(|binding| SampledImage {
                        set: binding.set,
                        binding: binding.binding,
                    })
                    .collect();
                entry.insert((Arc::clone(base), stage, images))
            }
        };
        let mut texel_images: Vec<SampledImage> = texels
            .0
            .iter()
            .filter(|(texel_stage, image)| texel_stage == stage && declared.contains(image))
            .map(|&(_, image)| image)
            .collect();
        if texel_images.is_empty() {
            return Ok(Arc::clone(base));
        }
        texel_images.sort_unstable();
        texel_images.dedup();
        let stage = *stage;
        let context = TranslationContext {
            texel_images,
            geometry_input,
            ..instance.translation_context()
        };
        let key = (address, context);
        if let Some((_, translated)) = self.translated.get(&key) {
            return Ok(Arc::clone(translated));
        }
        let refuse = |error: &dyn std::fmt::Display| {
            crate::unsupported::refuse(
                Status::Unimplemented,
                format_args!(
                    "draw: the {stage:?} stage samples texel-coordinate (rectangle) textures \
                     at {:?}, and translating it for them failed: {error}",
                    key.1.texel_images
                ),
            )
        };
        let code_address = instance
            .objects
            .with(program, |object| match object {
                Object::Program { shader_records, .. } => shader_records
                    .get(index)
                    .map(|record| record.gpu_addresses[0]),
                _ => None,
            })
            .flatten()
            .ok_or_else(|| refuse(&"the program holds no record for it"))?;
        let code = super::resources::shader_bytes(instance, code_address)
            .map_err(|error| refuse(&error.message()))?;
        let translator = instance
            .program_translator(program)
            .ok_or_else(|| refuse(&"the program registered under no shader translator"))?;
        let translated = Arc::new(
            translator
                .translate_for_context(crate::SHADER_STAGE_UNKNOWN, &code, &key.1)
                .map_err(|error| refuse(&error))?,
        );
        self.translated
            .insert(key, (Arc::clone(base), Arc::clone(&translated)));
        Ok(translated)
    }
}
