//! Storage-image (surface) bindings a translated module declares.
//!
//! A translator that emits storage images reports each view it declared:
//! where the module expects it (descriptor set and binding), which guest
//! image handle it reads and writes, the view's dimension, and the view
//! format its access needs. SwXBx builds the descriptor layout from the
//! module and binds each view from the guest image handle at draw or
//! dispatch time. The values are the translator's own; the adapter parses
//! them once at its edge into these types.
//!
//! Surface targets and raw access sizes are normalized into typed image
//! dimensions and formats at this boundary.

/// One storage-image view a translated module declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceBinding {
    /// Descriptor set number in the translated module.
    pub set: u32,
    /// Descriptor binding number in the translated module.
    pub binding: u32,
    /// The guest image handle the view reads and writes.
    pub source: SurfaceSource,
    /// The view's image dimension.
    pub dimension: SurfaceDimension,
    /// The view format the module's accesses need.
    pub format: SurfaceFormat,
}

/// Where a surface's guest image handle comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SurfaceSource {
    /// A bound slot of the stage's surface table; the host binds a guest
    /// image to it with the image binding call.
    Bound {
        /// Bound surface table index the instruction names.
        slot: u16,
    },
    /// A handle the module computes from constants and constant buffers.
    Bindless(BindlessHandle),
}

/// A bindless handle: one word, or two words combined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindlessHandle {
    /// The handle is this word.
    Word(HandleWord),
    /// The handle is two words combined by a logic operation.
    Combined {
        /// How the words combine.
        operation: HandleOperation,
        /// First word.
        a: HandleWord,
        /// Second word.
        b: HandleWord,
    },
}

/// A word a host can read at draw time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HandleWord {
    /// A literal word.
    Constant(u32),
    /// A word of a bound constant buffer.
    ConstantBuffer {
        /// Constant buffer bank.
        bank: u8,
        /// Byte offset in the bank.
        offset: u16,
    },
}

/// The logic operation that combines two handle words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HandleOperation {
    /// `a & b`.
    And,
    /// `a | b`.
    Or,
    /// `a ^ b`.
    Xor,
    /// `!b`.
    PassInvertedB,
}

/// Image dimension of a storage view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SurfaceDimension {
    /// One-dimensional image.
    D1,
    /// Texel buffer.
    Buffer,
    /// One-dimensional image array.
    D1Array,
    /// Two-dimensional image.
    D2,
    /// Two-dimensional image array.
    D2Array,
    /// Three-dimensional image.
    D3,
}

/// The view format a storage view's accesses need.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SurfaceFormat {
    /// Formatted access: the view keeps the image's own format and the
    /// module reads and writes it without a declared format.
    Formatted,
    /// Raw access at a fixed texel size and signedness.
    Raw(RawFormat),
}

/// Raw storage view formats (SPIR-V `ImageFormat` R8ui..Rgba32ui, and R32i
/// for signed surface atomics).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RawFormat {
    /// One unsigned byte.
    R8Uint,
    /// One signed byte.
    R8Sint,
    /// One unsigned 16-bit word.
    R16Uint,
    /// One signed 16-bit word.
    R16Sint,
    /// One unsigned 32-bit word.
    R32Uint,
    /// One signed 32-bit word (signed surface atomics).
    R32Sint,
    /// Two 32-bit words.
    Rg32Uint,
    /// Four 32-bit words.
    Rgba32Uint,
}

#[cfg(feature = "vulkan")]
impl RawFormat {
    /// The Vulkan format of the raw view.
    pub fn vk(self) -> ash::vk::Format {
        use ash::vk::Format;
        match self {
            Self::R8Uint => Format::R8_UINT,
            Self::R8Sint => Format::R8_SINT,
            Self::R16Uint => Format::R16_UINT,
            Self::R16Sint => Format::R16_SINT,
            Self::R32Uint => Format::R32_UINT,
            Self::R32Sint => Format::R32_SINT,
            Self::Rg32Uint => Format::R32G32_UINT,
            Self::Rgba32Uint => Format::R32G32B32A32_UINT,
        }
    }
}
