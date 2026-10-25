//! Lock-free frame exchange for application adapters and presentation hosts.

use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
};

/// Magic at the start of every shared-frame transport.
pub const MAGIC: [u8; 8] = *b"NVNSHDW2";
/// Fixed metadata bytes before the two pixel slots.
pub const HEADER_BYTES: u64 = 64;

/// A double-buffered RGBA8 frame publisher.
///
/// An odd generation means a write is in progress. An even generation names
/// the complete slot `(generation / 2) % 2`. Readers verify the generation
/// before and after copying pixels and retry if it changed.
pub struct SharedFrameWriter {
    file: File,
    width: u32,
    height: u32,
    byte_count: u64,
    generation: u64,
}

impl SharedFrameWriter {
    /// Open or replace the transport at `path` for tightly packed RGBA8 frames.
    pub fn open(path: &Path, width: u32, height: u32) -> io::Result<Self> {
        let byte_count = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "frame size overflow"))?;
        if byte_count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "frame dimensions must be nonzero",
            ));
        }
        let file_bytes = HEADER_BYTES
            .checked_add(byte_count.checked_mul(2).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "transport size overflow")
            })?)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "transport size overflow")
            })?;
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        let generation = if file.metadata()?.len() >= 16 {
            let mut prefix = [0_u8; 16];
            file.seek(SeekFrom::Start(0))?;
            file.read_exact(&mut prefix)?;
            if prefix[..8] == MAGIC {
                let observed = u64::from_le_bytes(prefix[8..16].try_into().unwrap());
                observed - (observed & 1)
            } else {
                0
            }
        } else {
            0
        };
        file.set_len(file_bytes)?;
        let mut header = [0_u8; HEADER_BYTES as usize];
        header[..8].copy_from_slice(&MAGIC);
        header[8..16].copy_from_slice(&generation.to_le_bytes());
        header[16..20].copy_from_slice(&width.to_le_bytes());
        header[20..24].copy_from_slice(&height.to_le_bytes());
        header[24..32].copy_from_slice(&byte_count.to_le_bytes());
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;
        file.flush()?;
        Ok(Self {
            file,
            width,
            height,
            byte_count,
            generation,
        })
    }

    /// Publish one complete frame and return its stable even generation.
    pub fn publish(&mut self, pixels: &[u8]) -> io::Result<u64> {
        if pixels.len() as u64 != self.byte_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "pixel payload does not match transport dimensions",
            ));
        }
        let busy = self
            .generation
            .checked_add(1)
            .ok_or_else(|| io::Error::other("frame generation overflow"))?;
        let complete = busy
            .checked_add(1)
            .ok_or_else(|| io::Error::other("frame generation overflow"))?;
        let slot = (complete / 2) % 2;

        self.file.seek(SeekFrom::Start(8))?;
        self.file.write_all(&busy.to_le_bytes())?;
        self.file
            .seek(SeekFrom::Start(HEADER_BYTES + slot * self.byte_count))?;
        self.file.write_all(pixels)?;
        self.file.seek(SeekFrom::Start(16))?;
        self.file.write_all(&self.width.to_le_bytes())?;
        self.file.write_all(&self.height.to_le_bytes())?;
        self.file.write_all(&self.byte_count.to_le_bytes())?;
        self.file.seek(SeekFrom::Start(8))?;
        self.file.write_all(&complete.to_le_bytes())?;
        self.file.flush()?;
        self.generation = complete;
        Ok(complete)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn read_complete(path: &Path) -> (u64, u32, u32, Vec<u8>) {
        let mut file = File::open(path).unwrap();
        let mut header = [0_u8; HEADER_BYTES as usize];
        file.read_exact(&mut header).unwrap();
        assert_eq!(header[..8], MAGIC);
        let generation = u64::from_le_bytes(header[8..16].try_into().unwrap());
        assert_eq!(generation % 2, 0);
        let width = u32::from_le_bytes(header[16..20].try_into().unwrap());
        let height = u32::from_le_bytes(header[20..24].try_into().unwrap());
        let byte_count = u64::from_le_bytes(header[24..32].try_into().unwrap());
        let slot = (generation / 2) % 2;
        file.seek(SeekFrom::Start(HEADER_BYTES + slot * byte_count))
            .unwrap();
        let mut pixels = vec![0; byte_count as usize];
        file.read_exact(&mut pixels).unwrap();
        (generation, width, height, pixels)
    }

    #[test]
    fn publishes_alternating_complete_slots_without_reallocation() {
        let name = format!(
            "swxbx-frame-transport-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(name);
        let mut writer = SharedFrameWriter::open(&path, 2, 1).unwrap();
        assert_eq!(writer.publish(&[1; 8]).unwrap(), 2);
        assert_eq!(read_complete(&path), (2, 2, 1, vec![1; 8]));
        let length = std::fs::metadata(&path).unwrap().len();
        assert_eq!(writer.publish(&[2; 8]).unwrap(), 4);
        assert_eq!(read_complete(&path), (4, 2, 1, vec![2; 8]));
        assert_eq!(std::fs::metadata(&path).unwrap().len(), length);
        drop(writer);
        let mut reopened = SharedFrameWriter::open(&path, 2, 1).unwrap();
        assert_eq!(reopened.publish(&[3; 8]).unwrap(), 6);
        assert_eq!(read_complete(&path), (6, 2, 1, vec![3; 8]));
        std::fs::remove_file(path).unwrap();
    }
}
