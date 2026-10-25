//! One guest program read from program memory: its shader header, its code up
//! to the compiler's end marker, and the constant data that follows the code.
//! Parsed once here; the translator receives the result as one record.

/// Program data starts with a magic word that selects the layout:
/// 0x12345678 is a graphics program
/// with the 80-byte shader header at +0x30 and code at +0x80; 0x12345679 is
/// a compute program with no header and code at +0x100. The translator gets
/// the header followed by the code; a compute program gets an all-zero
/// header, which parses as the compute shader type.
pub(super) const GRAPHICS_MAGIC: u32 = 0x1234_5678;
pub(super) const COMPUTE_MAGIC: u32 = 0x1234_5679;
const SHADER_PREFIX: usize = 0x30;
pub(super) const SHADER_CODE_OFFSET: usize = 0x80;
const COMPUTE_CODE_OFFSET: usize = 0x100;
const SHADER_HEADER_SIZE: usize = SHADER_CODE_OFFSET - SHADER_PREFIX;

/// Largest program window read: code and constant data each fit one
/// 64 KiB constant-bank-sized window.
const MAX_SHADER_BYTES: usize = 64 * 1024;

/// A scheduling word followed by three instruction slots.
const BUNDLE: usize = 32;
const WORD: usize = 8;

/// Constant buffers start on 256-byte GPU addresses before Turing
/// (Mesa src/nouveau/vulkan/nvk_physical_device.h:62-65).
const CONSTANT_BUFFER_ALIGNMENT: u64 = 256;

/// Extended translator program record:
/// header, code, cb1 bytes, then three little-endian u64 lengths and this tag.
const RECORD_TAG: &[u8; 16] = b"SBXCB1RECORD0001";

/// Why program memory did not yield a program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProgramRead {
    /// The program's memory could not be read.
    Unavailable,
    /// The first word is neither layout magic.
    UnknownLayout,
    /// No end marker in the readable window yet; the code may still be uploading.
    NoEnd,
}

impl ProgramRead {
    pub(super) fn message(self) -> &'static str {
        match self {
            Self::Unavailable => "shader bytes unavailable",
            Self::UnknownLayout => "unknown shader layout magic",
            // The census counts programs awaiting code under this name.
            Self::NoEnd => "no code after header",
        }
    }
}

/// Program memory at one GPU address, readable through `read` from `available`
/// bytes onward (offsets relative to the program's start).
pub(super) struct ProgramMemory<'r> {
    /// GPU address of the program's first byte.
    pub gpu_address: u64,
    /// Readable bytes from the program's start to the end of its pool.
    pub available: usize,
    /// Copy program bytes at a program-relative offset; false when unreadable.
    pub read: &'r dyn Fn(usize, &mut [u8]) -> bool,
}

/// A program as the guest compiled it.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct GuestProgram {
    header: [u8; SHADER_HEADER_SIZE],
    /// Code through the end of the bundle holding the end marker.
    code: Vec<u8>,
    /// Bytes from the first constant-buffer-aligned address after the code.
    constant_buffer1: Vec<u8>,
}

/// The program's end: an unconditional branch whose target is itself. Targets
/// are relative to the next slot; a target on a scheduling word lands on the
/// first slot after it.
fn ends_program(word: u64, slot_address: usize) -> bool {
    let bra = word >> 52 == 0xe24;
    let always = (word >> 16) & 0xf == 7 && word & 0x1f == 0xf;
    let offset = (((word >> 20) as u32) << 8) as i32 >> 8;
    let target = (slot_address as i64 + 8 + i64::from(offset)) as usize;
    let landed = target + WORD * usize::from(target.is_multiple_of(BUNDLE));
    bra && always && landed == slot_address
}

/// Offset of the end of the bundle holding the end marker in `code`, which
/// starts on a bundle.
fn code_end(code: &[u8]) -> Option<usize> {
    code.as_chunks::<BUNDLE>()
        .0
        .iter()
        .enumerate()
        .find_map(|(bundle, bytes)| {
            bytes[WORD..]
                .as_chunks::<WORD>()
                .0
                .iter()
                .enumerate()
                .any(|(slot, word)| {
                    ends_program(
                        u64::from_le_bytes(*word),
                        bundle * BUNDLE + WORD + slot * WORD,
                    )
                })
                .then_some((bundle + 1) * BUNDLE)
        })
}

impl GuestProgram {
    /// Read one program: layout, code to its end marker, then constant data.
    pub(super) fn read(memory: &ProgramMemory) -> Result<Self, ProgramRead> {
        let limit = memory.available.min(MAX_SHADER_BYTES);
        let mut prefix = [0u8; SHADER_CODE_OFFSET];
        if limit < SHADER_CODE_OFFSET || !(memory.read)(0, &mut prefix) {
            return Err(ProgramRead::Unavailable);
        }
        let (header, code_offset) = match u32::from_le_bytes(prefix[..4].try_into().unwrap()) {
            COMPUTE_MAGIC => ([0u8; SHADER_HEADER_SIZE], COMPUTE_CODE_OFFSET),
            GRAPHICS_MAGIC => (
                prefix[SHADER_PREFIX..].try_into().unwrap(),
                SHADER_CODE_OFFSET,
            ),
            _ => return Err(ProgramRead::UnknownLayout),
        };
        let window = limit.saturating_sub(code_offset) / BUNDLE * BUNDLE;
        let mut code = vec![0u8; window];
        let mut end = Err(ProgramRead::NoEnd);
        for start in (0..window).step_by(4096) {
            let chunk = start..window.min(start + 4096);
            if !(memory.read)(code_offset + start, &mut code[chunk.clone()]) {
                return Err(ProgramRead::Unavailable);
            }
            if let Some(chunk_end) = code_end(&code[chunk]) {
                end = Ok(start + chunk_end);
                break;
            }
        }
        let end = end?;
        code.truncate(end);
        let code_end = memory.gpu_address + (code_offset + end) as u64;
        let constants =
            (code_end.next_multiple_of(CONSTANT_BUFFER_ALIGNMENT) - memory.gpu_address) as usize;
        let constant_bytes = memory
            .available
            .saturating_sub(constants)
            .min(MAX_SHADER_BYTES)
            / 4
            * 4;
        let mut constant_buffer1 = vec![0u8; constant_bytes];
        if !constant_buffer1.is_empty() && !(memory.read)(constants, &mut constant_buffer1) {
            return Err(ProgramRead::Unavailable);
        }
        Ok(Self {
            header,
            code,
            constant_buffer1,
        })
    }

    /// The translator's input: header, code, then the cb1 snapshot and the
    /// record footer, so translation and its cache identity both see cb1.
    pub(super) fn record(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            self.header.len() + self.code.len() + self.constant_buffer1.len() + 40,
        );
        bytes.extend_from_slice(&self.header);
        bytes.extend_from_slice(&self.code);
        bytes.extend_from_slice(&self.constant_buffer1);
        for length in [
            self.header.len(),
            self.code.len(),
            self.constant_buffer1.len(),
        ] {
            bytes.extend_from_slice(&(length as u64).to_le_bytes());
        }
        bytes.extend_from_slice(RECORD_TAG);
        bytes
    }
}
