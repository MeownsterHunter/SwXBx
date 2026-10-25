//! Private, disposable pipeline cache files.
use ash::vk;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const VERSION: u32 = 1;
const MAX_FILE: u64 = 64 * 1024 * 1024;
const HEADER: usize = 84;
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// The host must change this identity whenever translator output can change.
/// Include generation, options and translator version in these strings.
#[derive(Clone, Debug)]
pub struct TranslationIdentity {
    /// Translator implementation version used to invalidate persisted output.
    pub version: String,
    /// Identity of every translator option that can affect generated output.
    pub configuration: String,
}

pub(super) struct DiskCache {
    directory: PathBuf,
    namespace: [u8; 32],
    vendor: u32,
    device: u32,
    pipeline_uuid: [u8; 16],
}

impl DiskCache {
    #[cfg(test)]
    fn new(
        directory: &Path,
        identity: &TranslationIdentity,
        properties: &vk::PhysicalDeviceProperties,
        ids: &vk::PhysicalDeviceIDProperties<'_>,
    ) -> Self {
        Self::for_interface(
            directory,
            identity,
            properties,
            ids,
            b"swxbx compute main interface 1 disk 1",
        )
    }

    pub(super) fn for_interface(
        directory: &Path,
        identity: &TranslationIdentity,
        properties: &vk::PhysicalDeviceProperties,
        ids: &vk::PhysicalDeviceIDProperties<'_>,
        interface: &[u8],
    ) -> Self {
        let mut hash = blake3::Hasher::new();
        hash.update(interface);
        for field in [
            identity.version.as_bytes(),
            identity.configuration.as_bytes(),
        ] {
            hash.update(&(field.len() as u64).to_le_bytes());
            hash.update(field);
        }
        hash.update(&ids.device_uuid);
        hash.update(&ids.driver_uuid);
        hash.update(&properties.pipeline_cache_uuid);
        hash.update(&properties.vendor_id.to_le_bytes());
        hash.update(&properties.device_id.to_le_bytes());
        let namespace = *hash.finalize().as_bytes();
        Self {
            directory: directory.join(blake3::Hash::from(namespace).to_hex().as_str()),
            namespace,
            vendor: properties.vendor_id,
            device: properties.device_id,
            pipeline_uuid: properties.pipeline_cache_uuid,
        }
    }

    fn read(&self, name: &str, magic: &[u8; 8]) -> io::Result<Option<Vec<u8>>> {
        let file = match File::open(self.directory.join(name)) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let mut bytes = Vec::new();
        file.take(MAX_FILE + 1).read_to_end(&mut bytes)?;
        if bytes.len() < HEADER
            || bytes.len() as u64 > MAX_FILE
            || &bytes[..8] != magic
            || bytes[8..12] != VERSION.to_le_bytes()
            || bytes[12..44] != self.namespace
            || bytes[44..52] != ((bytes.len() - HEADER) as u64).to_le_bytes()
            || bytes[52..HEADER] != *blake3::hash(&bytes[HEADER..]).as_bytes()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid cache envelope",
            ));
        }
        Ok(Some(bytes.split_off(HEADER)))
    }

    fn write(&self, name: &str, magic: &[u8; 8], payload: &[u8]) -> io::Result<()> {
        if payload.len() as u64 > MAX_FILE - HEADER as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cache data too large",
            ));
        }
        fs::create_dir_all(&self.directory)?;
        // create_new also handles PID reuse and leftovers from an interrupted writer.
        let (temporary, mut file) = loop {
            let path = self.directory.join(format!(
                ".{name}.{}.{}.tmp",
                std::process::id(),
                TEMP_ID.fetch_add(1, Ordering::Relaxed)
            ));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => break (path, file),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        };
        let result = (|| {
            file.write_all(magic)?;
            file.write_all(&VERSION.to_le_bytes())?;
            file.write_all(&self.namespace)?;
            file.write_all(&(payload.len() as u64).to_le_bytes())?;
            file.write_all(blake3::hash(payload).as_bytes())?;
            file.write_all(payload)?;
            file.sync_all()?;
            fs::rename(&temporary, self.directory.join(name))?;
            File::open(&self.directory)?.sync_all()
        })();
        if result.is_err() {
            let _ = fs::remove_file(temporary);
        }
        result
    }

    fn translation_name(program: &[u8]) -> String {
        format!("{}.spirv", blake3::hash(program).to_hex())
    }

    pub(super) fn load_translation(&self, program: &[u8]) -> io::Result<Option<Vec<u32>>> {
        let Some(bytes) = self.read(&Self::translation_name(program), b"NVTRANS\0")? else {
            return Ok(None);
        };
        let invalid = || io::Error::new(io::ErrorKind::InvalidData, "invalid translation record");
        let length = bytes.get(..8).ok_or_else(invalid)?;
        // Compare every input byte so a filename digest collision cannot alias code.
        if length != (program.len() as u64).to_le_bytes()
            || bytes.get(8..8 + program.len()) != Some(program)
        {
            return Err(invalid());
        }
        let (words, remainder) = bytes[8 + program.len()..].as_chunks::<4>();
        if !remainder.is_empty() || words.is_empty() {
            return Err(invalid());
        }
        Ok(Some(
            words.iter().map(|&word| u32::from_le_bytes(word)).collect(),
        ))
    }

    pub(super) fn save_translation(&self, program: &[u8], words: &[u32]) -> io::Result<()> {
        let mut bytes = (program.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(program);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        self.write(&Self::translation_name(program), b"NVTRANS\0", &bytes)
    }

    pub(super) fn load_shader(
        &self,
        program: &[u8],
    ) -> io::Result<Option<crate::startup_cache::TranslatedShader>> {
        let Some(record) = self.load_translation(program)? else {
            return Ok(None);
        };
        let Some((&required, words)) = record.split_first() else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing execution requirement metadata",
            ));
        };
        if required > 7 || words.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid execution requirement metadata",
            ));
        }
        Ok(Some(crate::startup_cache::TranslatedShader {
            key: String::new(),
            words: words.to_vec(),
            requires_subgroup_size_32: required & 1 != 0,
            requires_demote_to_helper_invocation: required & 4 != 0,
            requires_maximal_reconvergence: required & 2 != 0,
            surfaces: Vec::new(),
        }))
    }

    pub(super) fn save_shader(
        &self,
        program: &[u8],
        shader: &crate::startup_cache::TranslatedShader,
    ) -> io::Result<()> {
        // The record has no form for surface bindings; such a module is
        // remembered in memory only.
        if !shader.surfaces.is_empty() {
            return Ok(());
        }
        let mut record = vec![
            u32::from(shader.requires_subgroup_size_32)
                | (u32::from(shader.requires_demote_to_helper_invocation) << 2)
                | (u32::from(shader.requires_maximal_reconvergence) << 1),
        ];
        record.extend_from_slice(&shader.words);
        self.save_translation(program, &record)
    }

    fn valid_driver_header(&self, bytes: &[u8]) -> bool {
        bytes.len() >= 32
            && bytes[..4] == 32_u32.to_le_bytes()
            && bytes[4..8] == 1_u32.to_le_bytes()
            && bytes[8..12] == self.vendor.to_le_bytes()
            && bytes[12..16] == self.device.to_le_bytes()
            && bytes[16..32] == self.pipeline_uuid
    }

    pub(super) fn load_driver(&self) -> io::Result<Option<Vec<u8>>> {
        let bytes = self.read("driver.bin", b"NVDRIVER")?;
        if bytes
            .as_ref()
            .is_some_and(|bytes| !self.valid_driver_header(bytes))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "incompatible driver cache",
            ));
        }
        Ok(bytes)
    }

    pub(super) fn save_driver(&self, bytes: &[u8]) -> io::Result<()> {
        if !self.valid_driver_header(bytes) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected driver cache header",
            ));
        }
        self.write("driver.bin", b"NVDRIVER", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(root: &Path, version: &str, configuration: &str, device: u8, driver: u8) -> DiskCache {
        let properties = vk::PhysicalDeviceProperties {
            vendor_id: 123,
            device_id: 456,
            pipeline_cache_uuid: [7; 16],
            ..Default::default()
        };
        let ids = vk::PhysicalDeviceIDProperties::default()
            .device_uuid([device; 16])
            .driver_uuid([driver; 16]);
        DiskCache::new(
            root,
            &TranslationIdentity {
                version: version.into(),
                configuration: configuration.into(),
            },
            &properties,
            &ids,
        )
    }

    fn root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "disk-{name}-{}-{}",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn driver_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        for value in [32_u32, 1, 123, 456] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[7; 16]);
        bytes.extend_from_slice(b"synthetic driver payload");
        bytes
    }

    #[test]
    fn shader_records_preserve_and_validate_requirements() {
        let root = root("subgroup-metadata");
        let disk = cache(&root, "v1", "", 1, 2);
        for flags in 0..8 {
            let shader = crate::startup_cache::TranslatedShader {
                key: String::new(),
                words: vec![1, 2, 3],
                requires_subgroup_size_32: flags & 1 != 0,
                requires_demote_to_helper_invocation: flags & 4 != 0,
                requires_maximal_reconvergence: flags & 2 != 0,
                surfaces: Vec::new(),
            };
            disk.save_shader(b"program", &shader).unwrap();
            let reopened = cache(&root, "v1", "", 1, 2);
            assert_eq!(reopened.load_shader(b"program").unwrap(), Some(shader));
        }
        disk.save_translation(b"program", &[8, 1, 2, 3]).unwrap();
        assert!(disk.load_shader(b"program").is_err());
        disk.save_translation(b"program", &[1]).unwrap();
        assert!(disk.load_shader(b"program").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn versions_options_and_device_driver_uuids_separate_namespaces() {
        let root = root("namespace");
        let first = cache(&root, "v1", "options-a", 1, 2);
        first.save_translation(b"program", &[1, 2, 3]).unwrap();
        assert_eq!(
            first.load_translation(b"program").unwrap(),
            Some(vec![1, 2, 3])
        );
        for other in [
            cache(&root, "v2", "options-a", 1, 2),
            cache(&root, "v1", "options-b", 1, 2),
            cache(&root, "v1", "options-a", 3, 2),
            cache(&root, "v1", "options-a", 1, 4),
        ] {
            assert_ne!(first.directory, other.directory);
            assert!(other.load_translation(b"program").unwrap().is_none());
        }
        let properties = vk::PhysicalDeviceProperties {
            vendor_id: 123,
            device_id: 456,
            pipeline_cache_uuid: [7; 16],
            ..Default::default()
        };
        let ids = vk::PhysicalDeviceIDProperties::default()
            .device_uuid([1; 16])
            .driver_uuid([2; 16]);
        let graphics = DiskCache::for_interface(
            &root,
            &TranslationIdentity {
                version: "v1".into(),
                configuration: "options-a".into(),
            },
            &properties,
            &ids,
            b"swxbx graphics main interface 1 disk 1",
        );
        first.save_driver(&driver_bytes()).unwrap();
        assert_ne!(first.directory, graphics.directory);
        assert!(graphics.load_translation(b"program").unwrap().is_none());
        assert!(graphics.load_driver().unwrap().is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_partial_corrupt_oversized_and_wrong_version_records() {
        let root = root("corruption");
        let cache = cache(&root, "v1", "", 1, 2);
        cache.save_translation(b"program", &[1, 2, 3]).unwrap();
        let path = cache
            .directory
            .join(DiskCache::translation_name(b"program"));
        let valid = fs::read(&path).unwrap();
        for end in [0, 7, 12, 44, 52, HEADER, valid.len() - 1] {
            fs::write(&path, &valid[..end]).unwrap();
            assert!(cache.load_translation(b"program").is_err());
        }
        for at in [0, 8, 12, 44, 52, HEADER, valid.len() - 1] {
            let mut corrupt = valid.clone();
            corrupt[at] ^= 1;
            fs::write(&path, corrupt).unwrap();
            assert!(cache.load_translation(b"program").is_err());
        }
        let file = File::create(&path).unwrap();
        file.set_len(MAX_FILE + 1).unwrap();
        assert!(cache.load_translation(b"program").is_err());
        cache.save_translation(b"program", &[4, 5]).unwrap();
        assert_eq!(
            cache.load_translation(b"program").unwrap(),
            Some(vec![4, 5])
        );
        // A complete envelope with a different input also misses.
        cache
            .write(
                &DiskCache::translation_name(b"program"),
                b"NVTRANS\0",
                b"wrong contents",
            )
            .unwrap();
        assert!(cache.load_translation(b"program").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn decodes_and_checks_each_driver_header_field() {
        let root = root("driver");
        let cache = cache(&root, "v1", "", 1, 2);
        let valid = driver_bytes();
        cache.save_driver(&valid).unwrap();
        assert_eq!(cache.load_driver().unwrap(), Some(valid.clone()));
        // Recompute the envelope checksum to exercise header checks independently.
        for at in [0, 4, 8, 12, 16] {
            let mut corrupt = valid.clone();
            corrupt[at] ^= 1;
            cache.write("driver.bin", b"NVDRIVER", &corrupt).unwrap();
            assert!(cache.load_driver().is_err());
        }
        cache
            .write("driver.bin", b"NVDRIVER", &valid[..31])
            .unwrap();
        assert!(cache.load_driver().is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn concurrent_writers_publish_only_complete_records_and_ignore_temporary_files() {
        let root = root("concurrent");
        let cache = std::sync::Arc::new(cache(&root, "v1", "", 1, 2));
        cache.save_translation(b"program", &[0; 256]).unwrap();
        fs::write(cache.directory.join(".interrupted.tmp"), b"partial").unwrap();
        let workers: Vec<_> = (1..=4)
            .map(|value| {
                let cache = cache.clone();
                std::thread::spawn(move || {
                    for _ in 0..20 {
                        cache.save_translation(b"program", &[value; 256]).unwrap();
                        let words = cache.load_translation(b"program").unwrap().unwrap();
                        assert_eq!(words.len(), 256);
                        assert!(words.iter().all(|&word| word == words[0]));
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        assert!(cache.directory.join(".interrupted.tmp").is_file());
        fs::remove_dir_all(root).unwrap();
    }
}
