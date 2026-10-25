//! Textures, texture views, samplers, the pools that register them, and
//! programs.

use super::{
    objects::{Object, SamplerDescription, ShaderRecord, ShaderTranslation, TextureDescription},
    read_f32x4, succeed, Handler,
};
use crate::instance::{PendingShader, Registers, Status, SHADER_STAGE_UNKNOWN};
use std::fs;

use super::program_memory::{GuestProgram, ProgramMemory, ProgramRead, SHADER_CODE_OFFSET};

/// The translator's input for the program at `gpu_address`, read once.
fn shader_bytes(
    instance: &crate::instance::InstanceState,
    gpu_address: u64,
) -> Result<Vec<u8>, ProgramRead> {
    let resolution = instance
        .objects
        .resolve_gpu_address(gpu_address)
        .map_err(|_| ProgramRead::Unavailable)?;
    program_record(
        instance,
        gpu_address,
        resolution.program_address,
        usize::try_from(resolution.remaining).map_err(|_| ProgramRead::Unavailable)?,
    )
}

fn program_record(
    instance: &crate::instance::InstanceState,
    gpu_address: u64,
    program_address: u64,
    available: usize,
) -> Result<Vec<u8>, ProgramRead> {
    let read = |offset: usize, out: &mut [u8]| {
        program_address
            .checked_add(offset as u64)
            .is_some_and(|at| instance.read_memory(at, out))
    };
    GuestProgram::read(&ProgramMemory {
        gpu_address,
        available,
        read: &read,
    })
    .map(|program| program.record())
}

pub(crate) fn retry_pending(
    instance: &crate::instance::InstanceState,
    program: u64,
    point: &'static str,
) {
    let entries = instance.pending_shaders(program);
    for entry in entries {
        if entry.failures >= 64 {
            continue;
        }
        let Ok(program_bytes) =
            program_record(instance, entry.gpu_address, entry.address, entry.limit)
        else {
            instance.fail_pending_shader(program, entry.record_index);
            continue;
        };
        let Some(translator) = instance.shader_translator() else {
            continue;
        };
        if let Some(translation) = instance.request_cached_translation(&program_bytes) {
            instance.replace_pending_shader(program, entry.record_index);
            instance.record_late_read(point);
            instance.objects.update(program, |object| {
                if let Object::Program {
                    shader_translations,
                    ..
                } = object
                {
                    if let Some(slot) = shader_translations.get_mut(entry.record_index) {
                        *slot = translation;
                    }
                }
            });
            continue;
        }
        match translator.translate_for_context(
            SHADER_STAGE_UNKNOWN,
            &program_bytes,
            &Default::default(),
        ) {
            Ok(shader) => {
                let words = &shader.words;
                instance.record_successful_shader_translation(words);
                instance.record_late_read(point);
                instance.replace_pending_shader(program, entry.record_index);
                instance.objects.update(program, |object| {
                    if let Object::Program {
                        shader_translations,
                        ..
                    } = object
                    {
                        if let Some(slot) = shader_translations.get_mut(entry.record_index) {
                            *slot = ShaderTranslation::Cached(
                                crate::startup_cache::TranslationRequest::ready(
                                    std::sync::Arc::new(shader),
                                ),
                            );
                        }
                    }
                });
            }
            Err(error) => {
                instance.record_shader_translation_error(error);
                instance.fail_pending_shader(program, entry.record_index);
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::api::program_memory::{COMPUTE_MAGIC, GRAPHICS_MAGIC};
    use crate::api::{Object, ShaderTranslation};
    use crate::Instance;
    use crate::{Host, Registers, ShaderStage, ShaderTranslator, SHADER_STAGE_UNKNOWN};
    use std::ffi::c_void;
    use std::fs;
    use std::sync::{Arc, Mutex};

    struct Memory {
        base: u64,
        bytes: Vec<u8>,
    }

    unsafe extern "C" fn read_memory(
        user: *mut c_void,
        address: u64,
        out: *mut u8,
        size: u64,
    ) -> i32 {
        let memory = unsafe { &*(user.cast::<Memory>()) };
        let Some(start) = address.checked_sub(memory.base).map(|value| value as usize) else {
            return 1;
        };
        let end = start.saturating_add(size as usize);
        if end > memory.bytes.len() {
            return 1;
        }
        unsafe { std::slice::from_raw_parts_mut(out, size as usize) }
            .copy_from_slice(&memory.bytes[start..end]);
        0
    }

    type SeenTranslations = Arc<Mutex<Vec<(ShaderStage, Vec<u8>)>>>;

    struct FakeTranslator(SeenTranslations);

    impl ShaderTranslator for FakeTranslator {
        fn translate(&self, stage: ShaderStage, code: &[u8]) -> Result<Vec<u32>, String> {
            self.0.lock().unwrap().push((stage, code.to_vec()));
            Ok(vec![0x0723_0203, code.len() as u32])
        }
        fn translate_for_context(
            &self,
            stage: ShaderStage,
            code: &[u8],
            _: &crate::startup_cache::TranslationContext,
        ) -> Result<crate::startup_cache::TranslatedShader, String> {
            Ok(crate::startup_cache::TranslatedShader {
                key: String::new(),
                words: self.translate(stage, code)?,
                requires_subgroup_size_32: false,
                requires_demote_to_helper_invocation: true,
                requires_maximal_reconvergence: true,
                surfaces: Vec::new(),
            })
        }
    }

    /// The compiler's end marker, an unconditional branch to itself, for
    /// slots two and three, and for slot one, whose target is the bundle's
    /// scheduling word.
    const END: u64 = 0xe240_0fff_ff87_000f;
    const END_FIRST_SLOT: u64 = 0xe240_0fff_ff07_000f;

    fn program(memory: &[u8], gpu_address: u64) -> Result<Vec<u8>, ProgramRead> {
        let read = |offset: usize, out: &mut [u8]| {
            let Some(bytes) = memory.get(offset..offset + out.len()) else {
                return false;
            };
            out.copy_from_slice(bytes);
            true
        };
        GuestProgram::read(&ProgramMemory {
            gpu_address,
            available: memory.len(),
            read: &read,
        })
        .map(|program| program.record())
    }

    #[test]
    fn program_layout_follows_the_magic_word() {
        let mut graphics = vec![0u8; 0x200];
        graphics[..4].copy_from_slice(&GRAPHICS_MAGIC.to_le_bytes());
        graphics[0x30] = 0xaa;
        graphics[0x80..0x88].copy_from_slice(&[1; 8]);
        graphics[0x98..0xa0].copy_from_slice(&END.to_le_bytes());
        graphics[0x100..0x104].copy_from_slice(&0x40u32.to_le_bytes());
        let bytes = program(&graphics, 0x1000).unwrap();
        assert_eq!(
            (bytes.len(), bytes[0], bytes[0x50]),
            (0x50 + 0x20 + 0x100 + 40, 0xaa, 1)
        );
        assert_eq!(bytes[0x70], 0x40);
        assert_eq!(&bytes[bytes.len() - 16..], b"SBXCB1RECORD0001");

        let mut compute = vec![0u8; 0x200];
        compute[..4].copy_from_slice(&COMPUTE_MAGIC.to_le_bytes());
        compute[0x100..0x108].copy_from_slice(&[2; 8]);
        compute[0x108..0x110].copy_from_slice(&END_FIRST_SLOT.to_le_bytes());
        let bytes = program(&compute, 0x1000).unwrap();
        assert!(bytes[..0x50].iter().all(|&b| b == 0));
        assert_eq!(&bytes[0x50..0x58], &[2; 8]);
        assert_eq!(bytes.len(), 0x50 + 0x20 + 40);

        assert_eq!(
            program(&[0u8; 0x200], 0x1000),
            Err(ProgramRead::UnknownLayout)
        );
        graphics[0x98..0xa0].fill(0);
        assert_eq!(program(&graphics, 0x1000), Err(ProgramRead::NoEnd));
    }

    #[test]
    fn late_read_retains_declared_requirements_without_disk_caching() {
        let mut memory = Box::new(Memory {
            base: 0x1000,
            bytes: vec![0; 0x200],
        });
        memory.bytes[..4].copy_from_slice(&GRAPHICS_MAGIC.to_le_bytes());
        memory.bytes[0x80..0x88].copy_from_slice(&[1, 0, 0, 0, 2, 0, 0, 0]);
        memory.bytes[0x98..0xa0].copy_from_slice(&END.to_le_bytes());
        let instance = unsafe {
            Instance::with_host(Host {
                user: (&mut *memory as *mut Memory).cast(),
                read_memory: Some(read_memory),
                ..Default::default()
            })
        };
        instance.set_shader_translator(Some(Arc::new(FakeTranslator(Arc::new(Mutex::new(
            Vec::new(),
        ))))));
        instance.objects.put(
            9,
            Object::Program {
                device: 0,
                shader_records: Vec::new(),
                shader_translations: vec![ShaderTranslation::Error("pending".into())],
            },
        );
        instance.add_pending_shader(
            9,
            PendingShader {
                record_index: 0,
                address: 0x1000,
                gpu_address: 0x1000,
                limit: 0x200,
                failures: 0,
            },
        );
        retry_pending(&instance, 9, "test");
        let Some(Object::Program {
            mut shader_translations,
            ..
        }) = instance.objects.get(9)
        else {
            panic!("program must exist")
        };
        let ShaderTranslation::Cached(request) = shader_translations.pop().unwrap() else {
            panic!("owned translation must exist")
        };
        let crate::startup_cache::TranslationStatus::Ready(shader) = request.wait() else {
            panic!("translation must be complete")
        };
        assert!(shader.requires_demote_to_helper_invocation);
        assert!(shader.requires_maximal_reconvergence);
        // Header, one bundle, constants to the end of memory, record footer.
        assert_eq!(shader.words, vec![0x0723_0203, 0x50 + 0x20 + 0x100 + 40]);
    }

    #[test]
    fn enabled_translation_uses_resolved_bytes_and_retains_spirv() {
        // Header, one bundle, constants from +0x100 to the pool end, footer.
        const RECORD: usize = 0x50 + 0x20 + 0x3f00 + 40;
        let base = 0x1000;
        let mut memory = Box::new(Memory {
            base,
            bytes: vec![0; 0x4000],
        });
        let record = 0x2000usize - base as usize;
        let code = 0x1080usize - base as usize;
        memory.bytes[record..record + 8].copy_from_slice(&0x1000u64.to_le_bytes());
        // Graphics program layout: magic, header at +0x30, code at +0x80.
        memory.bytes[0..4].copy_from_slice(&GRAPHICS_MAGIC.to_le_bytes());
        memory.bytes[code..code + 4].copy_from_slice(&1u32.to_le_bytes());
        memory.bytes[code + 4..code + 8].copy_from_slice(&2u32.to_le_bytes());
        memory.bytes[code + 0x18..code + 0x20].copy_from_slice(&END.to_le_bytes());
        let host = Host {
            user: (&mut *memory as *mut Memory).cast(),
            read_memory: Some(read_memory),
            write_memory: None,
            present: None,
            render_scale: 1.0,
            wait_vblank: None,
            vulkan: std::ptr::null(),
        };
        let instance = unsafe { Instance::with_host(host) };
        instance.objects.put(
            7,
            Object::MemoryPool {
                device: 0,
                flags: 0,
                storage: base,
                size: 0x4000,
                gpu_address: None,
                observed_gpu_address: None,
            },
        );
        let seen = Arc::new(Mutex::new(Vec::new()));
        instance.set_shader_translator(Some(Arc::new(FakeTranslator(seen.clone()))));
        instance.set_shader_translation_enabled(true);
        let dump_directory =
            std::env::temp_dir().join(format!("swxbx-spirv-dump-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dump_directory);
        instance.set_shader_dump_directory(Some(&dump_directory));
        let mut init = Registers {
            x: [9, 0, 0, 0, 0, 0, 0, 0],
            ..Registers::default()
        };
        assert_eq!(
            instance.call(
                crate::functions::lookup("nvnProgramInitialize").unwrap(),
                &mut init
            ),
            Status::Ok
        );
        let mut setup = Registers {
            x: [9, 1, 0x2000, 0, 0, 0, 0, 0],
            ..Registers::default()
        };
        assert_eq!(
            instance.call(
                crate::functions::lookup("nvnProgramSetShaders").unwrap(),
                &mut setup
            ),
            Status::Ok
        );
        assert_eq!(seen.lock().unwrap()[0].0, SHADER_STAGE_UNKNOWN);
        assert_eq!(seen.lock().unwrap()[0].1.len(), RECORD);
        assert_eq!(
            u32::from_le_bytes(seen.lock().unwrap()[0].1[0x50..0x54].try_into().unwrap()),
            1
        );
        let Some(Object::Program {
            shader_records,
            mut shader_translations,
            ..
        }) = instance.objects.get(9)
        else {
            panic!("program must exist")
        };
        assert_eq!(shader_records[0].record_address, 0x2000);
        assert_eq!(shader_records[0].gpu_addresses, [0x1000, 0]);
        let ShaderTranslation::Cached(request) = shader_translations.pop().unwrap() else {
            panic!("owned translation must exist")
        };
        let crate::startup_cache::TranslationStatus::Ready(shader) = request.wait() else {
            panic!("translation must be complete")
        };
        assert_eq!(shader.words, vec![0x0723_0203, RECORD as u32]);
        assert!(shader.requires_demote_to_helper_invocation);
        assert!(shader.requires_maximal_reconvergence);
        assert_eq!(
            fs::read(dump_directory.join("0-213e63a4374f9163.spv")).unwrap(),
            [0x03, 0x02, 0x23, 0x07, 0x98, 0x3f, 0, 0]
        );
        fs::remove_dir_all(dump_directory).unwrap();
    }
}

fn translate_shaders(
    instance: &crate::instance::InstanceState,
    program: u64,
    records: &[ShaderRecord],
) -> Vec<super::objects::ShaderTranslation> {
    if !instance.shader_translation_enabled() {
        return Vec::new();
    }
    let Some(translator) = instance.shader_translator() else {
        return Vec::new();
    };
    records
        .iter()
        .enumerate()
        .map(|(record_index, record)| {
            dump_record(instance, record);
            let code = match shader_bytes(instance, record.gpu_addresses[0]) {
                Ok(code) => code,
                Err(ProgramRead::NoEnd) => {
                    if let Ok(resolution) = instance
                        .objects
                        .resolve_gpu_address(record.gpu_addresses[0])
                    {
                        let limit = usize::try_from(resolution.remaining).unwrap_or(0);
                        let mut prefix = vec![0; SHADER_CODE_OFFSET.min(limit)];
                        let zero = !prefix.is_empty()
                            && instance.read_memory(resolution.program_address, &mut prefix)
                            && prefix.iter().all(|&byte| byte == 0);
                        instance.record_shader_zero_header(zero);
                        instance.add_pending_shader(
                            program,
                            PendingShader {
                                record_index,
                                address: resolution.program_address,
                                gpu_address: record.gpu_addresses[0],
                                limit,
                                failures: 0,
                            },
                        );
                    }
                    return super::objects::ShaderTranslation::Error(
                        ProgramRead::NoEnd.message().into(),
                    );
                }
                Err(error) => {
                    instance.record_shader_translation_error(error.message().into());
                    return super::objects::ShaderTranslation::Error(error.message().into());
                }
            };
            if let Some(translation) = instance.request_cached_translation(&code) {
                return translation;
            }
            match translator.translate_for_context(SHADER_STAGE_UNKNOWN, &code, &Default::default())
            {
                Ok(shader) => {
                    let words = &shader.words;
                    instance.record_successful_shader_translation(words);
                    dump_translation(instance, words);
                    super::objects::ShaderTranslation::Cached(
                        crate::startup_cache::TranslationRequest::ready(std::sync::Arc::new(
                            shader,
                        )),
                    )
                }
                Err(error) => {
                    dump_translation_error(instance, &error);
                    instance.record_shader_translation_error(error.clone());
                    super::objects::ShaderTranslation::Error(error)
                }
            }
        })
        .collect()
}

/// Opt-in observation for working out shader record layouts: when
/// SWXBX_RECORD_DUMP_DIR is set, write each SetShaders record's raw words
/// and, for every word that resolves to program memory, the bytes from
/// 0x80 before it to 0x180 after it. Written once per record address. The
/// output contains program data and belongs in a private directory.
fn dump_record(instance: &crate::instance::InstanceState, record: &ShaderRecord) {
    let Some(directory) = std::env::var_os("SWXBX_RECORD_DUMP_DIR") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    // Programs reuse record memory, so key by the record's contents.
    let key = record
        .raw_words
        .iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, w| {
            (h ^ w).wrapping_mul(0x0100_0000_01b3)
        });
    let path = directory.join(format!("{key:016x}.txt"));
    if path.exists() {
        return;
    }
    let mut text = String::new();
    for (index, word) in record.raw_words.iter().enumerate() {
        text.push_str(&format!("word +0x{:02x}: {word:#018x}\n", index * 8));
    }
    for (index, word) in record.raw_words.iter().enumerate() {
        let Ok(resolution) = instance.objects.resolve_gpu_address(*word) else {
            continue;
        };
        let start = resolution.program_address.saturating_sub(0x80);
        let mut bytes = vec![0u8; 0x200];
        if !instance.read_memory(start, &mut bytes) {
            continue;
        }
        text.push_str(&format!(
            "window for word +0x{:02x} (program address {:#x}, from -0x80):\n",
            index * 8,
            resolution.program_address
        ));
        for (row, chunk) in bytes.chunks(16).enumerate() {
            let hex = chunk
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
            text.push_str(&format!("  {:+5}: {hex}\n", row as i64 * 16 - 0x80));
        }
    }
    let _ = fs::create_dir_all(&directory).and_then(|()| fs::write(path, text));
}

fn dump_translation(instance: &crate::instance::InstanceState, words: &[u32]) {
    let Some(directory) = instance.shader_dump_directory() else {
        return;
    };
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let Some((sequence, hash)) = instance.next_shader_dump(&bytes) else {
        return;
    };
    let path = directory.join(format!("{sequence}-{hash:016x}.spv"));
    let _ = fs::create_dir_all(&directory).and_then(|()| fs::write(path, bytes));
}

fn dump_translation_error(instance: &crate::instance::InstanceState, error: &str) {
    let Some(directory) = instance.shader_dump_directory() else {
        return;
    };
    let message = error.lines().next().unwrap_or("");
    let bytes = format!("{message}\n").into_bytes();
    let Some((sequence, hash)) = instance.next_shader_dump(&bytes) else {
        return;
    };
    let path = directory.join(format!("{sequence}-{hash:016x}.err"));
    let _ = fs::create_dir_all(&directory).and_then(|()| fs::write(path, bytes));
}

fn texture_builder_update(
    instance: &crate::instance::InstanceState,
    registers: &Registers,
    change: impl FnOnce(&mut TextureDescription),
) -> Status {
    instance.objects.update(registers.x[0], |object| {
        if let Object::TextureBuilder(description) = object {
            change(description);
        }
    });
    Status::Ok
}

fn sampler_builder_update(
    instance: &crate::instance::InstanceState,
    registers: &Registers,
    change: impl FnOnce(&mut SamplerDescription),
) -> Status {
    instance.objects.update(registers.x[0], |object| {
        if let Object::SamplerBuilder(description) = object {
            change(description);
        }
    });
    Status::Ok
}

fn texture_field(
    instance: &crate::instance::InstanceState,
    address: u64,
    pick: impl FnOnce(&TextureDescription) -> u64,
) -> Option<u64> {
    match instance.objects.get(address) {
        Some(Object::Texture { description, .. }) => Some(pick(&description)),
        _ => None,
    }
}

/// How much pool storage a texture needs, by swxbx's own reckoning.
///
/// The answer depends on the format and its tiling. Four bytes per texel per
/// level, with each
/// level at least one texel, is enough for every format a Vulkan
/// implementation would store uncompressed at 32 bits, and never smaller
/// than what the program will later read or write through the pool.
fn storage_size(description: &TextureDescription) -> u64 {
    let (mut width, mut height) = (description.width.max(1), description.height.max(1));
    let depth = description.depth.max(1);
    let mut total = 0;
    for _ in 0..description.levels.max(1) {
        total += width * height * depth * 4;
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }
    total.next_multiple_of(STORAGE_ALIGNMENT)
}

/// Conservative storage alignment used by the host contract.
const STORAGE_ALIGNMENT: u64 = 0x10000;

/// Return the handler for a supported name in this command family.
pub fn handler(name: &str) -> Option<Handler> {
    Some(match name {
        "nvnTextureBuilderSetDefaults" => |instance, _, registers| {
            instance.objects.put(
                registers.x[0],
                Object::TextureBuilder(TextureDescription::default()),
            );
            Status::Ok
        },
        "nvnTextureBuilderSetDevice" => |instance, _, registers| {
            let value = registers.x[1];
            texture_builder_update(instance, registers, |t| t.device = value)
        },
        "nvnTextureBuilderSetFlags" => |instance, _, registers| {
            let value = registers.x[1];
            texture_builder_update(instance, registers, |t| t.flags = value)
        },
        "nvnTextureBuilderSetTarget" => |instance, _, registers| {
            let value = registers.x[1];
            texture_builder_update(instance, registers, |t| t.target = value)
        },
        "nvnTextureBuilderSetFormat" => |instance, _, registers| {
            let value = registers.x[1];
            texture_builder_update(instance, registers, |t| t.format = value)
        },
        "nvnTextureBuilderSetLevels" => |instance, _, registers| {
            let value = registers.x[1];
            texture_builder_update(instance, registers, |t| t.levels = value)
        },
        "nvnTextureBuilderSetSize1D" => |instance, _, registers| {
            let value = registers.x[1];
            texture_builder_update(instance, registers, |t| {
                t.width = value;
                t.height = 1;
                t.depth = 1;
            })
        },
        "nvnTextureBuilderSetSize2D" => |instance, _, registers| {
            let (w, h) = (registers.x[1], registers.x[2]);
            texture_builder_update(instance, registers, |t| {
                t.width = w;
                t.height = h;
                t.depth = 1;
            })
        },
        "nvnTextureBuilderSetSize3D" => |instance, _, registers| {
            let (w, h, d) = (registers.x[1], registers.x[2], registers.x[3]);
            texture_builder_update(instance, registers, |t| {
                t.width = w;
                t.height = h;
                t.depth = d;
            })
        },
        "nvnTextureBuilderSetStride" => |instance, _, registers| {
            let value = registers.x[1];
            texture_builder_update(instance, registers, |t| t.stride = value)
        },
        "nvnTextureBuilderSetSwizzle" => |instance, _, registers| {
            let value = [
                registers.x[1],
                registers.x[2],
                registers.x[3],
                registers.x[4],
            ];
            texture_builder_update(instance, registers, |t| t.swizzle = value)
        },
        "nvnTextureBuilderSetDepthStencilMode" => |instance, _, registers| {
            let value = registers.x[1];
            texture_builder_update(instance, registers, |t| t.depth_stencil_mode = value)
        },
        "nvnTextureBuilderSetStorage" => |instance, _, registers| {
            let (pool, offset) = (registers.x[1], registers.x[2]);
            texture_builder_update(instance, registers, |t| {
                t.pool = pool;
                t.pool_offset = offset;
            })
        },
        "nvnTextureBuilderGetStorageSize" => {
            |instance, _, registers| match instance.objects.get(registers.x[0]) {
                Some(Object::TextureBuilder(description)) => {
                    #[cfg(feature = "vulkan")]
                    if let Some(backend) = instance
                        .gpu
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .as_ref()
                    {
                        if backend.has_image_contract() {
                            let Some(bytes) = backend.texture_storage_size(&description) else {
                                return Status::BadArgument;
                            };
                            registers.x[0] = bytes as u64;
                            return Status::Ok;
                        }
                    }
                    registers.x[0] = storage_size(&description);
                    Status::Ok
                }
                _ => Status::BadArgument,
            }
        }
        "nvnTextureBuilderGetStorageAlignment" => |_instance, _, registers| {
            registers.x[0] = STORAGE_ALIGNMENT;
            Status::Ok
        },
        "nvnTextureInitialize" => |instance, _, registers| {
            let Some(Object::TextureBuilder(description)) = instance.objects.get(registers.x[1])
            else {
                return Status::BadArgument;
            };
            #[cfg(feature = "vulkan")]
            if let Some(backend) = instance
                .gpu
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_mut()
            {
                backend.release_texture(registers.x[0]);
            }
            instance.objects.put(
                registers.x[0],
                Object::Texture {
                    description,
                    // The image is shared so object-table clones stay cheap.
                    image: std::sync::Arc::new(std::sync::Mutex::new(None)),
                },
            );
            succeed(registers)
        },
        "nvnTextureFinalize" => |instance, _, registers| {
            #[cfg(feature = "vulkan")]
            if let Some(backend) = instance
                .gpu
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_mut()
            {
                backend.release_texture(registers.x[0]);
            }
            instance.objects.remove(registers.x[0]);
            Status::Ok
        },
        "nvnTextureGetFlags" => {
            |instance, _, registers| texture_answer(instance, registers, |t| t.flags)
        }
        "nvnTextureGetTarget" => {
            |instance, _, registers| texture_answer(instance, registers, |t| t.target)
        }
        "nvnTextureGetLevels" => {
            |instance, _, registers| texture_answer(instance, registers, |t| t.levels)
        }
        // Multisample textures are not part of this bounded contract.
        "nvnTextureGetSamples" => {
            |instance, _, registers| texture_answer(instance, registers, |_| 0)
        }
        // Pool aliases use the same flat address plus the texture offset.
        "nvnTextureGetTextureAddress" => |instance, _, registers| {
            let Some(Object::Texture { description, .. }) = instance.objects.get(registers.x[0])
            else {
                return Status::BadArgument;
            };
            let (base, size) = match instance.objects.get(description.pool) {
                Some(Object::MemoryPool {
                    storage,
                    gpu_address,
                    observed_gpu_address,
                    size,
                    ..
                }) => (
                    gpu_address.or(observed_gpu_address).unwrap_or(storage),
                    size,
                ),
                _ => return Status::BadArgument,
            };
            if description.pool_offset >= size {
                return Status::BadArgument;
            }
            let Some(address) = base.checked_add(description.pool_offset) else {
                return Status::BadArgument;
            };
            registers.x[0] = address;
            Status::Ok
        },
        // A view's offset into the texture's storage. Levels are laid out
        // in order by swxbx's own storage_size rule.
        "nvnTextureGetViewOffset" => |instance, _, registers| {
            let Some(Object::Texture { description, .. }) = instance.objects.get(registers.x[0])
            else {
                return Status::BadArgument;
            };
            let base_level = match instance.objects.get(registers.x[1]) {
                Some(Object::TextureView { base_level, .. }) => base_level,
                _ => 0,
            };
            #[cfg(feature = "vulkan")]
            if let Some(backend) = instance
                .gpu
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_ref()
            {
                if backend.has_image_contract() {
                    let Some(r) = backend.resolved_image(&description) else {
                        return Status::BadArgument;
                    };
                    let Some(level) = r.packing.levels().get(base_level as usize) else {
                        return Status::BadArgument;
                    };
                    let base_layer = match instance.objects.get(registers.x[1]) {
                        Some(Object::TextureView { base_layer, .. }) => base_layer,
                        _ => 0,
                    };
                    if base_layer >= r.descriptor.shape.layers {
                        return Status::BadArgument;
                    }
                    let (offset, stride) = match r.storage {
                        crate::gpu::image_layout::Storage::Linear => {
                            (level.linear_offset, r.packing.linear_layer_stride())
                        }
                        crate::gpu::image_layout::Storage::Tiled(_) => {
                            (level.tiled_offset, r.packing.array_stride())
                        }
                    };
                    registers.x[0] = offset as u64 + u64::from(base_layer) * stride as u64;
                    return Status::Ok;
                }
            }
            let mut offset = 0;
            let (mut width, mut height) = (description.width.max(1), description.height.max(1));
            for _ in 0..base_level {
                offset += width * height * description.depth.max(1) * 4;
                width = (width / 2).max(1);
                height = (height / 2).max(1);
            }
            registers.x[0] = offset;
            Status::Ok
        },
        // The program asks how much memory a depth texture's culling data
        // needs and later saves and restores it. SwXBx keeps none, so the
        // minimum allocation preserves the program's layout.
        "nvnTextureGetZCullStorageSize" => |_instance, _, registers| {
            registers.x[0] = 0x100;
            Status::Ok
        },
        "nvnTextureViewSetDefaults" => |instance, _, registers| {
            instance.objects.put(
                registers.x[0],
                Object::TextureView {
                    base_level: 0,
                    levels: 0,
                    base_layer: 0,
                    layers: 0,
                },
            );
            Status::Ok
        },
        "nvnTextureViewSetLevels" => |instance, _, registers| {
            let (base, count) = (registers.x[1] as u32, registers.x[2] as u32);
            instance.objects.update(registers.x[0], |object| {
                if let Object::TextureView {
                    base_level, levels, ..
                } = object
                {
                    *base_level = base;
                    *levels = count;
                }
            });
            Status::Ok
        },
        "nvnTextureViewSetLayers" => |instance, _, registers| {
            let (base, count) = (registers.x[1] as u32, registers.x[2] as u32);
            instance.objects.update(registers.x[0], |object| {
                if let Object::TextureView {
                    base_layer, layers, ..
                } = object
                {
                    *base_layer = base;
                    *layers = count;
                }
            });
            Status::Ok
        },
        "nvnTexturePoolInitialize" => |instance, _, registers| {
            instance.objects.put(
                registers.x[0],
                Object::TexturePool {
                    memory: registers.x[1],
                    offset: registers.x[2],
                    count: registers.x[3],
                    registered: Default::default(),
                },
            );
            succeed(registers)
        },
        "nvnSamplerPoolInitialize" => |instance, _, registers| {
            instance.objects.put(
                registers.x[0],
                Object::SamplerPool {
                    memory: registers.x[1],
                    offset: registers.x[2],
                    count: registers.x[3],
                    registered: Default::default(),
                },
            );
            succeed(registers)
        },
        "nvnTexturePoolRegisterTexture" | "nvnTexturePoolRegisterImage" => {
            |instance, _, registers| {
                let (id, texture) = (registers.x[1] as u32, registers.x[2]);
                instance.objects.update(registers.x[0], |object| {
                    if let Object::TexturePool { registered, .. } = object {
                        registered.insert(id, (texture, registers.x[3]));
                    }
                });
                Status::Ok
            }
        }
        "nvnSamplerPoolRegisterSampler" => |instance, _, registers| {
            let (id, sampler) = (registers.x[1] as u32, registers.x[2]);
            instance.objects.update(registers.x[0], |object| {
                if let Object::SamplerPool { registered, .. } = object {
                    registered.insert(id, sampler);
                }
            });
            Status::Ok
        },
        "nvnSamplerBuilderSetDefaults" => |instance, _, registers| {
            instance.objects.put(
                registers.x[0],
                Object::SamplerBuilder(SamplerDescription::default()),
            );
            Status::Ok
        },
        "nvnSamplerBuilderSetDevice" => |instance, _, registers| {
            let value = registers.x[1];
            sampler_builder_update(instance, registers, |s| s.device = value)
        },
        "nvnSamplerBuilderSetMinMagFilter" => |instance, _, registers| {
            let (min, mag) = (registers.x[1], registers.x[2]);
            sampler_builder_update(instance, registers, |s| {
                s.min_filter = min;
                s.mag_filter = mag;
            })
        },
        "nvnSamplerBuilderSetWrapMode" => |instance, _, registers| {
            let wrap = [registers.x[1], registers.x[2], registers.x[3]];
            sampler_builder_update(instance, registers, |s| s.wrap = wrap)
        },
        "nvnSamplerBuilderSetMaxAnisotropy" => |instance, _, registers| {
            let value = f32::from_bits(registers.d[0] as u32);
            sampler_builder_update(instance, registers, |s| s.max_anisotropy = value)
        },
        "nvnSamplerBuilderSetCompare" => |instance, _, registers| {
            let (mode, func) = (registers.x[1], registers.x[2]);
            sampler_builder_update(instance, registers, |s| {
                s.compare_mode = mode;
                s.compare_func = func;
            })
        },
        "nvnSamplerBuilderSetBorderColor" => |instance, _, registers| {
            let Some(color) = read_f32x4(instance, registers.x[1]) else {
                return Status::BadArgument;
            };
            sampler_builder_update(instance, registers, |s| s.border_color = color)
        },
        "nvnSamplerBuilderSetLodBias" => |instance, _, registers| {
            let value = f32::from_bits(registers.d[0] as u32);
            sampler_builder_update(instance, registers, |s| s.lod_bias = value)
        },
        "nvnSamplerBuilderSetLodClamp" => |instance, _, registers| {
            let value = [
                f32::from_bits(registers.d[0] as u32),
                f32::from_bits(registers.d[1] as u32),
            ];
            sampler_builder_update(instance, registers, |s| s.lod_clamp = value)
        },
        "nvnSamplerInitialize" => |instance, _, registers| {
            let Some(Object::SamplerBuilder(description)) = instance.objects.get(registers.x[1])
            else {
                return Status::BadArgument;
            };
            instance
                .objects
                .put(registers.x[0], Object::Sampler(description));
            succeed(registers)
        },
        "nvnProgramInitialize" => |instance, _, registers| {
            instance.objects.put(
                registers.x[0],
                Object::Program {
                    device: registers.x[1],
                    shader_records: Vec::new(),
                    shader_translations: Vec::new(),
                },
            );
            succeed(registers)
        },
        // The record has two GPU-shaped values, but the host contract does not
        // identify the code value or establish a code size. Retain both values
        // and the opaque words for deferred translation.
        "nvnProgramSetShaders" => |instance, _, registers| {
            instance.clear_pending_shaders(registers.x[0]);
            let count = registers.x[1].min(8);
            let records: Vec<ShaderRecord> = (0..count)
                .filter_map(|index| {
                    let record_address = registers.x[2] + index * 0x40;
                    let mut bytes = [0u8; 0x40];
                    if !instance.read_memory(record_address, &mut bytes) {
                        return None;
                    }
                    let mut raw_words = [0u64; 8];
                    for (index, word) in raw_words.iter_mut().enumerate() {
                        let start = index * 8;
                        *word = u64::from_le_bytes(
                            bytes[start..start + 8].try_into().expect("8 bytes"),
                        );
                    }
                    Some(ShaderRecord {
                        record_address,
                        gpu_addresses: [raw_words[0], raw_words[6]],
                        raw_words,
                    })
                })
                .collect();
            instance.objects.update(registers.x[0], |object| {
                if let Object::Program { shader_records, .. } = object {
                    *shader_records = records;
                }
            });
            let translations = instance
                .objects
                .get(registers.x[0])
                .and_then(|object| match object {
                    Object::Program { shader_records, .. } => {
                        Some(translate_shaders(instance, registers.x[0], &shader_records))
                    }
                    _ => None,
                })
                .unwrap_or_default();
            instance.objects.update(registers.x[0], |object| {
                if let Object::Program {
                    shader_translations,
                    ..
                } = object
                {
                    *shader_translations = translations;
                }
            });
            succeed(registers)
        },
        _ => return None,
    })
}

fn texture_answer(
    instance: &crate::instance::InstanceState,
    registers: &mut Registers,
    pick: impl FnOnce(&TextureDescription) -> u64,
) -> Status {
    match texture_field(instance, registers.x[0], pick) {
        Some(value) => {
            registers.x[0] = value;
            Status::Ok
        }
        None => Status::BadArgument,
    }
}
