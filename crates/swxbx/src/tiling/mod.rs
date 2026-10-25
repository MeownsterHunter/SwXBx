//! CPU conversion between packed-linear image bytes and a block-linear layout.
//!

use std::{error::Error, fmt, thread};

const GOB_WIDTH: usize = 64;
const GOB_HEIGHT: usize = 8;
const GOB_SIZE: usize = GOB_WIDTH * GOB_HEIGHT;
const SECTOR_SIZE: usize = 16;
const MAX_LEVELS: u32 = 16;
const PARALLEL_MIN_BYTES: usize = 256 * 1024;
const STREAM_MIN_BYTES: usize = 1024 * 1024;
const SECTOR_OFFSETS: [[usize; 4]; 8] = [
    [0, 32, 256, 288],
    [16, 48, 272, 304],
    [64, 96, 320, 352],
    [80, 112, 336, 368],
    [128, 160, 384, 416],
    [144, 176, 400, 432],
    [192, 224, 448, 480],
    [208, 240, 464, 496],
];

/// Image dimensions and array organization supported by checked packing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageKind {
    /// One one-dimensional image with height, depth and layer count one.
    D1,
    /// Array of one-dimensional images with height and depth one.
    D1Array,
    /// One two-dimensional image with depth and layer count one.
    D2,
    /// Array of two-dimensional images with depth one.
    D2Array,
    /// One three-dimensional image with layer count one.
    D3,
    /// Square cube faces in layers; the layer count is divisible by six.
    Cube,
}

/// Base texel dimensions, layers, and complete or partial mip chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageShape {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Depth in texels.
    pub depth: u32,
    /// Number of array layers; cube faces count as layers.
    pub layers: u32,
    /// Borrow mip layouts in increasing level order.
    pub levels: u32,
    /// Image dimension and layer organization.
    pub kind: ImageKind,
}

/// Opaque format-block geometry used for packing without decoding payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockFormat {
    /// Width of one format block in texels.
    pub width: u8,
    /// Height of one format block in texels.
    pub height: u8,
    /// Bytes per opaque format block, supported for sizes 1, 2, 4, 8, and 16.
    pub bytes: u8,
}

/// Block-linear tile exponents in units of GOBs, each limited to zero through five.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileShape {
    /// Base-two exponent of the tile height in eight-row GOBs.
    pub height_log2: u8,
    /// Base-two exponent of the tile depth in GOBs.
    pub depth_log2: u8,
}

/// Checked storage geometry for one mip level within an array layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelLayout {
    /// Width, height, and depth of this mip in texels.
    pub extent: [u32; 3],
    /// Mip dimensions rounded up to whole format blocks.
    pub blocks: [u32; 3],
    /// Active format-block bytes in one row.
    pub row_bytes: usize,
    /// Effective tile after mip-dependent clamping.
    pub tile: TileShape,
    /// Byte offset of this mip within one tiled array layer.
    pub tiled_offset: usize,
    /// Total tiled storage size in bytes, including padding.
    pub tiled_size: usize,
    /// Bytes between tiled block rows.
    pub tiled_row_stride: usize,
    /// Bytes between tiled depth slabs.
    pub tiled_depth_stride: usize,
    /// Byte offset of this mip within one packed-linear layer.
    pub linear_offset: usize,
    /// Total packed-linear storage size in bytes, including padding.
    pub linear_size: usize,
}

/// Checked mip offsets, layer strides, and sizes for linear and tiled storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    shape: ImageShape,
    format: BlockFormat,
    levels: Box<[LevelLayout]>,
    array_stride: usize,
    linear_layer_stride: usize,
    tiled_size: usize,
    linear_size: usize,
}

/// Invalid image geometry, arithmetic overflow, or insufficient conversion storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// A dimension, layer count, or mip count is zero.
    ZeroDimension,
    /// Dimensions or layer counts do not match the selected image kind.
    InvalidShape,
    /// Format block geometry or element size is unsupported.
    InvalidFormat,
    /// A tile exponent exceeds five.
    InvalidTile,
    /// Offset or size arithmetic cannot be represented.
    Overflow,
    /// The source buffer is smaller than the required representation.
    SourceSize {
        /// Minimum required byte length.
        expected: usize,
        /// Actual supplied byte length.
        actual: usize,
    },
    /// The destination buffer is smaller than the required representation.
    DestinationSize {
        /// Minimum required byte length.
        expected: usize,
        /// Actual supplied byte length.
        actual: usize,
    },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => {
                f.write_str("image dimensions, layers, and levels must be nonzero")
            }
            Self::InvalidShape => f.write_str("image dimensions do not match the image kind"),
            Self::InvalidFormat => f.write_str("unsupported block format"),
            Self::InvalidTile => f.write_str("tile exponents are out of range"),
            Self::Overflow => f.write_str("image layout size overflow"),
            Self::SourceSize { expected, actual } => {
                write!(f, "source is {actual} bytes, expected at least {expected}")
            }
            Self::DestinationSize { expected, actual } => {
                write!(
                    f,
                    "destination is {actual} bytes, expected at least {expected}"
                )
            }
        }
    }
}

impl Error for LayoutError {}

impl Layout {
    /// Compute checked packing; reject invalid shape, block geometry, tile exponents, or overflow.
    pub fn new(
        shape: ImageShape,
        format: BlockFormat,
        tile: TileShape,
    ) -> Result<Self, LayoutError> {
        validate(shape, format, tile)?;
        let mut levels = Vec::with_capacity(shape.levels as usize);
        let mut tiled_end = 0usize;
        let mut linear_end = 0usize;

        for level in 0..shape.levels {
            let extent = [
                mip_extent(shape.width, level),
                mip_extent(shape.height, level),
                if shape.kind == ImageKind::D3 {
                    mip_extent(shape.depth, level)
                } else {
                    1
                },
            ];
            let blocks = [
                div_ceil_u32(extent[0], u32::from(format.width)),
                div_ceil_u32(extent[1], u32::from(format.height)),
                extent[2],
            ];
            let row_bytes = usize::try_from(blocks[0])
                .ok()
                .and_then(|v| v.checked_mul(usize::from(format.bytes)))
                .ok_or(LayoutError::Overflow)?;
            let level_tile = TileShape {
                height_log2: clamp_log2(tile.height_log2, blocks[1], GOB_HEIGHT as u32),
                depth_log2: clamp_log2(tile.depth_log2, blocks[2], 1),
            };
            let tile_height = GOB_HEIGHT << level_tile.height_log2;
            let tile_depth = 1usize << level_tile.depth_log2;
            let tile_bytes = GOB_SIZE
                .checked_shl(u32::from(level_tile.height_log2 + level_tile.depth_log2))
                .ok_or(LayoutError::Overflow)?;
            let tile_columns = div_ceil(row_bytes, GOB_WIDTH);
            let tile_rows = div_ceil(blocks[1] as usize, tile_height);
            let tile_depths = div_ceil(blocks[2] as usize, tile_depth);
            let tiled_row_stride = tile_columns
                .checked_mul(tile_bytes)
                .ok_or(LayoutError::Overflow)?;
            let tiled_depth_stride = tile_rows
                .checked_mul(tiled_row_stride)
                .ok_or(LayoutError::Overflow)?;
            let tiled_size = tile_depths
                .checked_mul(tiled_depth_stride)
                .ok_or(LayoutError::Overflow)?;
            let tiled_offset = align_up(tiled_end, tile_bytes)?;
            tiled_end = tiled_offset
                .checked_add(tiled_size)
                .ok_or(LayoutError::Overflow)?;

            let active_linear_size = row_bytes
                .checked_mul(blocks[1] as usize)
                .and_then(|v| v.checked_mul(blocks[2] as usize))
                .ok_or(LayoutError::Overflow)?;
            let linear_offset = align_up(linear_end, SECTOR_SIZE)?;
            let linear_size = align_up(active_linear_size, SECTOR_SIZE)?;
            linear_end = linear_offset
                .checked_add(linear_size)
                .ok_or(LayoutError::Overflow)?;
            levels.push(LevelLayout {
                extent,
                blocks,
                row_bytes,
                tile: level_tile,
                tiled_offset,
                tiled_size,
                tiled_row_stride,
                tiled_depth_stride,
                linear_offset,
                linear_size,
            });
        }

        let layers = storage_layers(shape);
        let base_tile = levels[0].tile;
        let base_tile_bytes = GOB_SIZE
            .checked_shl(u32::from(base_tile.height_log2 + base_tile.depth_log2))
            .ok_or(LayoutError::Overflow)?;
        let array_stride = align_up(tiled_end, base_tile_bytes)?;
        let linear_layer_stride = align_up(linear_end, SECTOR_SIZE)?;
        let tiled_size = array_stride
            .checked_mul(layers)
            .ok_or(LayoutError::Overflow)?;
        let linear_size = linear_layer_stride
            .checked_mul(layers)
            .ok_or(LayoutError::Overflow)?;

        Ok(Self {
            shape,
            format,
            levels: levels.into_boxed_slice(),
            array_stride,
            linear_layer_stride,
            tiled_size,
            linear_size,
        })
    }

    /// Return the base image shape used to construct this layout.
    pub fn shape(&self) -> ImageShape {
        self.shape
    }

    /// Return the opaque format-block geometry.
    pub fn format(&self) -> BlockFormat {
        self.format
    }

    /// Borrow mip layouts in increasing level order.
    pub fn levels(&self) -> &[LevelLayout] {
        &self.levels
    }

    /// Byte distance between consecutive tiled array layers.
    pub fn array_stride(&self) -> usize {
        self.array_stride
    }

    /// Byte distance between consecutive packed-linear array layers.
    pub fn linear_layer_stride(&self) -> usize {
        self.linear_layer_stride
    }

    /// Total tiled storage size in bytes, including padding.
    pub fn tiled_size(&self) -> usize {
        self.tiled_size
    }

    /// Total packed-linear storage size in bytes, including padding.
    pub fn linear_size(&self) -> usize {
        self.linear_size
    }

    /// Resolve layer, mip, block coordinates, and byte-in-block to a checked tiled offset.
    pub fn tiled_byte_offset(
        &self,
        level: u32,
        layer: u32,
        x_byte: u32,
        y: u32,
        z: u32,
    ) -> Option<usize> {
        let level = self.levels.get(level as usize)?;
        if layer as usize >= storage_layers(self.shape)
            || x_byte as usize >= level.row_bytes
            || y >= level.blocks[1]
            || z >= level.blocks[2]
        {
            return None;
        }
        let h = level.tile.height_log2 as usize;
        let d = level.tile.depth_log2 as usize;
        let x = x_byte as usize;
        let y = y as usize;
        let z = z as usize;
        let tile_columns = div_ceil(level.row_bytes, GOB_WIDTH);
        let tile_x = x / GOB_WIDTH;
        let gob_y = y / GOB_HEIGHT;
        let tile_y = gob_y >> h;
        let tile_z = z >> d;
        let gob_y_in_tile = gob_y & ((1usize << h) - 1);
        let gob_z_in_tile = z & ((1usize << d) - 1);
        let tile_bytes = GOB_SIZE << (h + d);
        let tile_index = tile_z
            .checked_mul(level.tiled_depth_stride / tile_bytes)?
            .checked_add(tile_y.checked_mul(tile_columns)?)?
            .checked_add(tile_x)?;
        let gob_index = (gob_z_in_tile << h) | gob_y_in_tile;
        (layer as usize)
            .checked_mul(self.array_stride)?
            .checked_add(level.tiled_offset)?
            .checked_add(tile_index.checked_mul(tile_bytes)?)?
            .checked_add(gob_index.checked_mul(GOB_SIZE)?)?
            .checked_add(sector_offset(x % GOB_WIDTH, y % GOB_HEIGHT))
    }

    /// Resolve layer, mip, block coordinates, and byte-in-block to a checked linear offset.
    pub fn linear_byte_offset(
        &self,
        level: u32,
        layer: u32,
        x_byte: u32,
        y: u32,
        z: u32,
    ) -> Option<usize> {
        let level = self.levels.get(level as usize)?;
        if layer as usize >= storage_layers(self.shape)
            || x_byte as usize >= level.row_bytes
            || y >= level.blocks[1]
            || z >= level.blocks[2]
        {
            return None;
        }
        (layer as usize)
            .checked_mul(self.linear_layer_stride)?
            .checked_add(level.linear_offset)?
            .checked_add(
                (z as usize)
                    .checked_mul(level.blocks[1] as usize)?
                    .checked_mul(level.row_bytes)?,
            )?
            .checked_add((y as usize).checked_mul(level.row_bytes)?)?
            .checked_add(x_byte as usize)
    }

    /// Copy active bytes from tiled to linear storage; preserve destination padding.
    pub fn decode(&self, tiled: &[u8], linear: &mut [u8]) -> Result<(), LayoutError> {
        self.check_buffers(tiled, linear, true)?;
        self.convert_serial(tiled, linear, true)
    }

    /// Copy active bytes from linear to tiled storage; preserve destination padding.
    pub fn encode(&self, linear: &[u8], tiled: &mut [u8]) -> Result<(), LayoutError> {
        self.check_buffers(linear, tiled, false)?;
        self.convert_serial(linear, tiled, false)
    }

    /// Decode disjoint ranges with at most the requested worker count; small images run serially.
    pub fn decode_parallel(
        &self,
        tiled: &[u8],
        linear: &mut [u8],
        threads: usize,
    ) -> Result<(), LayoutError> {
        self.check_buffers(tiled, linear, true)?;
        self.convert_parallel(tiled, linear, true, threads)
    }

    /// Encode disjoint ranges with at most the requested worker count; small images run serially.
    pub fn encode_parallel(
        &self,
        linear: &[u8],
        tiled: &mut [u8],
        threads: usize,
    ) -> Result<(), LayoutError> {
        self.check_buffers(linear, tiled, false)?;
        self.convert_parallel(linear, tiled, false, threads)
    }

    fn check_buffers(
        &self,
        source: &[u8],
        destination: &[u8],
        decode: bool,
    ) -> Result<(), LayoutError> {
        let (source_size, destination_size) = if decode {
            (self.tiled_size, self.linear_size)
        } else {
            (self.linear_size, self.tiled_size)
        };
        if source.len() < source_size {
            return Err(LayoutError::SourceSize {
                expected: source_size,
                actual: source.len(),
            });
        }
        if destination.len() < destination_size {
            return Err(LayoutError::DestinationSize {
                expected: destination_size,
                actual: destination.len(),
            });
        }
        Ok(())
    }

    fn convert_serial(
        &self,
        source: &[u8],
        destination: &mut [u8],
        decode: bool,
    ) -> Result<(), LayoutError> {
        for layer in 0..storage_layers(self.shape) as u32 {
            for level in 0..self.levels.len() as u32 {
                self.convert_level(source, destination, decode, layer, level, 0)?;
            }
        }
        Ok(())
    }

    fn convert_parallel(
        &self,
        source: &[u8],
        destination: &mut [u8],
        decode: bool,
        threads: usize,
    ) -> Result<(), LayoutError> {
        let work = storage_layers(self.shape) * self.levels.len();
        let workers = threads.max(1).min(work.max(1));
        if self.linear_size < PARALLEL_MIN_BYTES {
            return self.convert_serial(source, destination, decode);
        }
        if self.shape.kind == ImageKind::D2 && threads > 1 {
            for level in 0..self.levels.len() as u32 {
                if self.levels[level as usize].linear_size < PARALLEL_MIN_BYTES {
                    self.convert_level(source, destination, decode, 0, level, 0)?;
                } else {
                    self.convert_level_parallel(source, destination, decode, threads, level)?;
                }
            }
            return Ok(());
        }
        if self.shape.kind == ImageKind::D3 && self.levels.len() == 1 && threads > 1 {
            return self.convert_level_parallel(source, destination, decode, threads, 0);
        }
        if workers == 1 {
            return self.convert_serial(source, destination, decode);
        }
        let per_worker = div_ceil(work, workers);
        let destination_size = destination_len(self, decode);
        let mut rest = &mut destination[..destination_size];
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for first in (0..work).step_by(per_worker) {
                let end = (first + per_worker).min(work);
                let start_offset = self.work_range(first, decode).0;
                let end_offset = self.work_range(end - 1, decode).1;
                let skip = start_offset - (destination_len(self, decode) - rest.len());
                let (_, after_skip) = rest.split_at_mut(skip);
                let (chunk, next) = after_skip.split_at_mut(end_offset - start_offset);
                rest = next;
                handles.push(scope.spawn(move || {
                    for index in first..end {
                        let layer = (index / self.levels.len()) as u32;
                        let level = (index % self.levels.len()) as u32;
                        self.convert_level(source, chunk, decode, layer, level, start_offset)?;
                    }
                    Ok(())
                }));
            }
            for handle in handles {
                handle.join().map_err(|_| LayoutError::InvalidShape)??;
            }
            Ok(())
        })
    }

    fn convert_level_parallel(
        &self,
        source: &[u8],
        destination: &mut [u8],
        decode: bool,
        threads: usize,
        level_index: u32,
    ) -> Result<(), LayoutError> {
        let level = self.levels[level_index as usize];
        let tile_height = GOB_HEIGHT << level.tile.height_log2;
        let tile_rows = div_ceil(level.blocks[1] as usize, tile_height);
        let tile_depth = 1usize << level.tile.depth_log2;
        let tile_depths = div_ceil(level.blocks[2] as usize, tile_depth);
        let split_depth = self.shape.kind == ImageKind::D3;
        let units = if split_depth { tile_depths } else { tile_rows };
        let workers = threads
            .min(units)
            .min(level.linear_size / (2 * 1024 * 1024))
            .max(1);
        if workers == 1 {
            return self.convert_level(source, destination, decode, 0, level_index, 0);
        }
        let units_per_worker = div_ceil(units, workers);
        let tile_row_bytes = level.tiled_row_stride;
        thread::scope(|scope| {
            let mut handles = Vec::new();
            let mut consumed = 0usize;
            let mut rest = destination;
            for first in (0..units).step_by(units_per_worker) {
                let end = (first + units_per_worker).min(units);
                let start = if decode {
                    if split_depth {
                        level.linear_offset
                            + first * tile_depth * level.blocks[1] as usize * level.row_bytes
                    } else {
                        level.linear_offset + first * tile_height * level.row_bytes
                    }
                } else {
                    level.tiled_offset
                        + if split_depth {
                            first * level.tiled_depth_stride
                        } else {
                            first * tile_row_bytes
                        }
                };
                let finish = if decode {
                    if split_depth {
                        level.linear_offset
                            + (end * tile_depth).min(level.blocks[2] as usize)
                                * level.blocks[1] as usize
                                * level.row_bytes
                    } else {
                        level.linear_offset
                            + (end * tile_height).min(level.blocks[1] as usize) * level.row_bytes
                    }
                } else {
                    level.tiled_offset
                        + if split_depth {
                            end * level.tiled_depth_stride
                        } else {
                            end * tile_row_bytes
                        }
                };
                let (_, after_gap) = rest.split_at_mut(start - consumed);
                let (chunk, next) = after_gap.split_at_mut(finish - start);
                rest = next;
                consumed = finish;
                handles.push(scope.spawn(move || {
                    let (first_z, end_z, first_y, end_y) = if split_depth {
                        (first, end, 0, tile_rows)
                    } else {
                        (0, tile_depths, first, end)
                    };
                    self.convert_level_tiles(
                        source,
                        chunk,
                        decode,
                        0,
                        level_index,
                        start,
                        first_z,
                        end_z,
                        first_y,
                        end_y,
                    )
                }));
            }
            for handle in handles {
                handle.join().map_err(|_| LayoutError::InvalidShape)??;
            }
            Ok(())
        })
    }

    fn work_range(&self, index: usize, decode: bool) -> (usize, usize) {
        let layer = index / self.levels.len();
        let level = &self.levels[index % self.levels.len()];
        if decode {
            let start = layer * self.linear_layer_stride + level.linear_offset;
            (start, start + level.linear_size)
        } else {
            let start = layer * self.array_stride + level.tiled_offset;
            (start, start + level.tiled_size)
        }
    }

    fn convert_level(
        &self,
        source: &[u8],
        destination: &mut [u8],
        decode: bool,
        layer: u32,
        level_index: u32,
        destination_base: usize,
    ) -> Result<(), LayoutError> {
        let level = self.levels[level_index as usize];
        let tile_rows = div_ceil(
            level.blocks[1] as usize,
            GOB_HEIGHT << level.tile.height_log2,
        );
        self.convert_level_tiles(
            source,
            destination,
            decode,
            layer,
            level_index,
            destination_base,
            0,
            div_ceil(level.blocks[2] as usize, 1usize << level.tile.depth_log2),
            0,
            tile_rows,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn convert_level_tiles(
        &self,
        source: &[u8],
        destination: &mut [u8],
        decode: bool,
        layer: u32,
        level_index: u32,
        destination_base: usize,
        first_tile_depth: usize,
        end_tile_depth: usize,
        first_tile_row: usize,
        end_tile_row: usize,
    ) -> Result<(), LayoutError> {
        macro_rules! dispatch {
            ($bytes:expr) => {
                if decode {
                    self.convert_level_direction::<true, $bytes>(
                        source,
                        destination,
                        layer,
                        level_index,
                        destination_base,
                        first_tile_depth,
                        end_tile_depth,
                        first_tile_row,
                        end_tile_row,
                    )
                } else {
                    self.convert_level_direction::<false, $bytes>(
                        source,
                        destination,
                        layer,
                        level_index,
                        destination_base,
                        first_tile_depth,
                        end_tile_depth,
                        first_tile_row,
                        end_tile_row,
                    )
                }
            };
        }
        match self.format.bytes {
            1 => dispatch!(1),
            2 => dispatch!(2),
            4 => dispatch!(4),
            8 => dispatch!(8),
            16 => dispatch!(16),
            _ => unreachable!(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn convert_level_direction<const DECODE: bool, const BYTES: usize>(
        &self,
        source: &[u8],
        destination: &mut [u8],
        layer: u32,
        level_index: u32,
        destination_base: usize,
        first_tile_depth: usize,
        end_tile_depth: usize,
        first_tile_row: usize,
        end_tile_row: usize,
    ) -> Result<(), LayoutError> {
        let level = self.levels[level_index as usize];
        #[cfg(target_arch = "x86_64")]
        let paired = std::is_x86_feature_detected!("avx2");
        #[cfg(target_arch = "x86_64")]
        let wide = std::is_x86_feature_detected!("avx512f");
        debug_assert_eq!(level.row_bytes, level.blocks[0] as usize * BYTES);
        let h = level.tile.height_log2 as usize;
        let d = level.tile.depth_log2 as usize;
        let tile_bytes = GOB_SIZE << (h + d);
        let tile_columns = div_ceil(level.row_bytes, GOB_WIDTH);
        let tile_rows = div_ceil(level.blocks[1] as usize, GOB_HEIGHT << h);
        let tile_depths = div_ceil(level.blocks[2] as usize, 1usize << d);
        let tiled_level_base = layer as usize * self.array_stride + level.tiled_offset;
        let linear_level_base = layer as usize * self.linear_layer_stride + level.linear_offset;
        #[cfg(target_arch = "x86_64")]
        let stream = level.linear_size >= STREAM_MIN_BYTES
            && level.row_bytes.is_multiple_of(64)
            && (destination.as_ptr() as usize).is_multiple_of(16);
        #[cfg(not(target_arch = "x86_64"))]
        let stream = false;
        #[cfg(target_arch = "x86_64")]
        let prefix = if DECODE && stream {
            (64usize.wrapping_sub(
                (destination.as_ptr() as usize)
                    .wrapping_add(linear_level_base)
                    .wrapping_sub(destination_base),
            ) & 63)
                / 16
        } else {
            0
        };
        debug_assert!(end_tile_depth <= tile_depths);
        for tile_z in first_tile_depth..end_tile_depth {
            for tile_y in first_tile_row..end_tile_row {
                let rows_per_group = if DECODE || level.row_bytes <= 256 {
                    8
                } else if level.row_bytes <= 512 {
                    4
                } else if level.row_bytes <= 1024 {
                    2
                } else {
                    1
                }
                .min(1usize << h);
                let bounds = [1usize << d, (1usize << h) / rows_per_group, tile_columns];
                for first in 0..bounds[0] {
                    for second in 0..bounds[1] {
                        for third in 0..bounds[2] {
                            for fourth in 0..rows_per_group {
                                let (tile_x, gob_z, gob_in_y) =
                                    (third, first, second * rows_per_group + fourth);
                                let z = tile_z * (1usize << d) + gob_z;
                                let y = tile_y * (GOB_HEIGHT << h) + gob_in_y * GOB_HEIGHT;
                                if z >= level.blocks[2] as usize || y >= level.blocks[1] as usize {
                                    continue;
                                }
                                let tile_index =
                                    (tile_z * tile_rows + tile_y) * tile_columns + tile_x;
                                let tile_base = tiled_level_base + tile_index * tile_bytes;
                                let x = tile_x * GOB_WIDTH;
                                let width = (level.row_bytes - x).min(GOB_WIDTH);
                                let height = (level.blocks[1] as usize - y).min(GOB_HEIGHT);
                                let tiled_base = tile_base + ((gob_z << h) | gob_in_y) * GOB_SIZE;
                                let linear_base = linear_level_base
                                    + (z * level.blocks[1] as usize + y) * level.row_bytes
                                    + x;
                                #[cfg(target_arch = "x86_64")]
                                if DECODE && stream && tile_x + 2 < tile_columns {
                                    unsafe {
                                        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                                        for line in (0..GOB_SIZE).step_by(64) {
                                            _mm_prefetch(
                                                source
                                                    .as_ptr()
                                                    .add(tiled_base + 2 * tile_bytes + line)
                                                    .cast(),
                                                _MM_HINT_T0,
                                            );
                                        }
                                    }
                                }
                                #[cfg(target_arch = "x86_64")]
                                if stream
                                    && !DECODE
                                    && tile_x + 4 < tile_columns
                                    && height == GOB_HEIGHT
                                {
                                    unsafe {
                                        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                                        for row in 0..GOB_HEIGHT {
                                            _mm_prefetch(
                                                source
                                                    .as_ptr()
                                                    .add(linear_base + row * level.row_bytes + 256)
                                                    .cast(),
                                                _MM_HINT_T0,
                                            );
                                        }
                                    }
                                }
                                #[cfg(target_arch = "x86_64")]
                                if DECODE
                                    && wide
                                    && prefix == 0
                                    && width == GOB_WIDTH
                                    && height == GOB_HEIGHT
                                {
                                    // SAFETY: this complete GOB and all eight output rows
                                    // are in the checked level and worker range.
                                    unsafe {
                                        wide_decode::<BYTES>(
                                            source.as_ptr().add(tiled_base),
                                            destination
                                                .as_mut_ptr()
                                                .add(linear_base - destination_base),
                                            level.blocks[0] as usize,
                                            stream,
                                        );
                                    }
                                    continue;
                                }
                                #[cfg(target_arch = "x86_64")]
                                if DECODE && paired && width == GOB_WIDTH && height == GOB_HEIGHT {
                                    unsafe {
                                        macro_rules! copy {
                                            ($prefix:expr, $stream:expr) => {
                                                paired_decode::<BYTES, $prefix, $stream>(
                                                    source.as_ptr().add(tiled_base),
                                                    destination
                                                        .as_mut_ptr()
                                                        .add(linear_base - destination_base),
                                                    level.blocks[0] as usize,
                                                    tile_bytes,
                                                    tile_x,
                                                    tile_columns,
                                                )
                                            };
                                        }
                                        if !stream {
                                            copy!(0, false);
                                        } else {
                                            match prefix {
                                                0 => copy!(0, true),
                                                1 => copy!(1, true),
                                                2 => copy!(2, true),
                                                3 => copy!(3, true),
                                                _ => unreachable!(),
                                            }
                                        }
                                    }
                                    continue;
                                }
                                #[cfg(target_arch = "x86_64")]
                                if !DECODE
                                    && paired
                                    && stream
                                    && width == GOB_WIDTH
                                    && height == GOB_HEIGHT
                                {
                                    // SAFETY: full rows, aligned output and complete
                                    // adjacent GOBs are established here.
                                    unsafe {
                                        let src = source.as_ptr().add(linear_base);
                                        let dst = destination
                                            .as_mut_ptr()
                                            .add(tiled_base - destination_base);
                                        let previous = gob_in_y > 0;
                                        let next = gob_in_y + 1 < (1usize << h)
                                            && y + 16 <= level.blocks[1] as usize;
                                        if (dst as usize).is_multiple_of(32) {
                                            paired_encode::<BYTES, 0>(
                                                src,
                                                dst,
                                                level.blocks[0] as usize,
                                                previous,
                                                next,
                                            );
                                        } else {
                                            paired_encode::<BYTES, 1>(
                                                src,
                                                dst,
                                                level.blocks[0] as usize,
                                                previous,
                                                next,
                                            );
                                        }
                                    }
                                    continue;
                                }
                                #[cfg(target_arch = "x86_64")]
                                if !DECODE && wide && width == GOB_WIDTH && height == GOB_HEIGHT {
                                    unsafe {
                                        wide_encode::<BYTES>(
                                            source.as_ptr().add(linear_base),
                                            destination
                                                .as_mut_ptr()
                                                .add(tiled_base - destination_base),
                                            level.blocks[0] as usize,
                                            stream,
                                            gob_in_y > 0,
                                            gob_in_y + 1 < (1usize << h)
                                                && y + 16 <= level.blocks[1] as usize,
                                        );
                                    }
                                    continue;
                                }
                                #[cfg(target_arch = "x86_64")]
                                if prefix != 0 {
                                    // SAFETY: streaming selects complete 64-byte rows and
                                    // 16-byte alignment. Only existing neighbor columns
                                    // are read; prefix and tail writes stay in active rows.
                                    unsafe {
                                        let src = source.as_ptr().add(tiled_base);
                                        let dst = destination
                                            .as_mut_ptr()
                                            .add(linear_base - destination_base);
                                        match prefix {
                                            1 => decode_shifted_gob::<1>(
                                                src,
                                                dst,
                                                level.row_bytes,
                                                tile_bytes,
                                                tile_x,
                                                tile_columns,
                                                height,
                                            ),
                                            2 => decode_shifted_gob::<2>(
                                                src,
                                                dst,
                                                level.row_bytes,
                                                tile_bytes,
                                                tile_x,
                                                tile_columns,
                                                height,
                                            ),
                                            3 => decode_shifted_gob::<3>(
                                                src,
                                                dst,
                                                level.row_bytes,
                                                tile_bytes,
                                                tile_x,
                                                tile_columns,
                                                height,
                                            ),
                                            _ => unreachable!(),
                                        }
                                    }
                                    continue;
                                }
                                if width == GOB_WIDTH && height == GOB_HEIGHT {
                                    Self::convert_full_gob::<DECODE, BYTES>(
                                        source,
                                        destination,
                                        tiled_base,
                                        linear_base,
                                        level.row_bytes,
                                        destination_base,
                                        stream,
                                    )?;
                                    continue;
                                }
                                // Layout construction, worker ranges and the outer edge
                                // bounds prove every sector here. No address query per byte.
                                for (row, offsets) in SECTOR_OFFSETS.iter().enumerate().take(height)
                                {
                                    for sector in (0..width).step_by(SECTOR_SIZE) {
                                        let count = (width - sector).min(SECTOR_SIZE);
                                        let tiled = tiled_base + offsets[sector / SECTOR_SIZE];
                                        let linear = linear_base + row * level.row_bytes + sector;
                                        let (src, dst) = if DECODE {
                                            (tiled, linear)
                                        } else {
                                            (linear, tiled)
                                        };
                                        unsafe {
                                            std::ptr::copy_nonoverlapping(
                                                source.as_ptr().add(src),
                                                destination
                                                    .as_mut_ptr()
                                                    .add(dst - destination_base),
                                                count,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        if stream {
            // SAFETY: the streaming stores above must become globally visible
            // before this worker returns and its destination may be consumed.
            unsafe { std::arch::x86_64::_mm_sfence() };
        }
        Ok(())
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn convert_full_gob<const DECODE: bool, const BYTES: usize>(
        source: &[u8],
        destination: &mut [u8],
        tiled_base: usize,
        linear_base: usize,
        row_bytes: usize,
        destination_base: usize,
        stream: bool,
    ) -> Result<(), LayoutError> {
        // Specialize the sector element count outside the GOB loop.
        // Full GOBs and worker destination ranges are proved by the caller.
        macro_rules! sector {
            ($row:expr, $index:expr) => {{
                let element = $index * (16 / BYTES);
                let linear = linear_base + $row * row_bytes + element * BYTES;
                let tiled = tiled_base + SECTOR_OFFSETS[$row][$index];
                let (src, dst) = if DECODE {
                    (tiled, linear)
                } else {
                    (linear, tiled)
                };
                unsafe {
                    let src = source.as_ptr().add(src);
                    let dst = destination.as_mut_ptr().add(dst - destination_base);
                    #[cfg(target_arch = "x86_64")]
                    {
                        use std::arch::x86_64::{_mm_loadu_si128, _mm_storeu_si128};
                        if stream {
                            stream_copy_16(src, dst);
                        } else {
                            _mm_storeu_si128(dst.cast(), _mm_loadu_si128(src.cast()));
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        let _ = stream;
                        std::ptr::copy_nonoverlapping(src, dst, 16);
                    }
                }
            }};
        }
        if DECODE {
            macro_rules! row {
                ($row:expr) => {
                    sector!($row, 0);
                    sector!($row, 1);
                    sector!($row, 2);
                    sector!($row, 3);
                };
            }
            row!(0);
            row!(1);
            row!(2);
            row!(3);
            row!(4);
            row!(5);
            row!(6);
            row!(7);
        } else {
            // Consecutive tiled sectors complete output cache lines together.
            macro_rules! pair {
                ($a:expr, $b:expr, $sector:expr) => {
                    sector!($a, $sector);
                    sector!($b, $sector);
                    sector!($a, $sector + 1);
                    sector!($b, $sector + 1);
                };
            }
            pair!(0, 1, 0);
            pair!(2, 3, 0);
            pair!(4, 5, 0);
            pair!(6, 7, 0);
            pair!(0, 1, 2);
            pair!(2, 3, 2);
            pair!(4, 5, 2);
            pair!(6, 7, 2);
        }
        Ok(())
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn paired_decode<const BYTES: usize, const PREFIX: usize, const STREAM: bool>(
    source: *const u8,
    destination: *mut u8,
    row_elements: usize,
    tile_bytes: usize,
    column: usize,
    columns: usize,
) {
    let row_bytes = row_elements * BYTES;
    use std::arch::x86_64::*;
    unsafe {
        macro_rules! row {
            ($row:expr) => {{
                let offsets = SECTOR_OFFSETS[$row];
                let dst = destination.add($row * row_bytes);
                if PREFIX != 0 && column == 0 {
                    for sector in 0..PREFIX {
                        _mm_storeu_si128(
                            dst.add(sector * 16).cast(),
                            _mm_loadu_si128(source.add(offsets[sector]).cast()),
                        );
                    }
                }
                if PREFIX != 0 && column + 1 == columns {
                    for sector in PREFIX..4 {
                        _mm_storeu_si128(
                            dst.add(sector * 16).cast(),
                            _mm_loadu_si128(source.add(offsets[sector]).cast()),
                        );
                    }
                } else {
                    macro_rules! pair {
                        ($pair:expr) => {{
                            let sector = PREFIX + $pair * 2;
                            let first = if sector < 4 {
                                source.add(offsets[sector])
                            } else {
                                source.add(tile_bytes + offsets[sector - 4])
                            };
                            let second = if sector + 1 < 4 {
                                source.add(offsets[sector + 1])
                            } else {
                                source.add(tile_bytes + offsets[sector - 3])
                            };
                            let value = _mm256_set_m128i(
                                _mm_loadu_si128(second.cast()),
                                _mm_loadu_si128(first.cast()),
                            );
                            let output = dst.add(sector * 16);
                            if STREAM {
                                _mm256_stream_si256(output.cast(), value);
                            } else {
                                _mm256_storeu_si256(output.cast(), value);
                            }
                        }};
                    }
                    pair!(0);
                    pair!(1);
                }
            }};
        }
        row!(0);
        row!(1);
        row!(2);
        row!(3);
        row!(4);
        row!(5);
        row!(6);
        row!(7);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn paired_encode<const BYTES: usize, const PREFIX: usize>(
    source: *const u8,
    destination: *mut u8,
    row_elements: usize,
    previous: bool,
    next: bool,
) {
    use std::arch::x86_64::*;
    let row_bytes = row_elements * BYTES;
    unsafe {
        if PREFIX != 0 {
            let first = _mm_loadu_si128(source.cast());
            if previous {
                let tail = _mm_loadu_si128(source.sub(row_bytes).add(48).cast());
                _mm256_stream_si256(destination.sub(16).cast(), _mm256_set_m128i(first, tail));
            } else {
                _mm_storeu_si128(destination.cast(), first);
            }
        }
        // Physical sectors select two rows and two columns per cache line.
        // These indices become constants for each unrolled pair.
        macro_rules! sector {
            ($index:expr) => {{
                let index = $index;
                let row = (index / 4 % 4) * 2 + index % 2;
                let column = index / 16 * 2 + index % 4 / 2;
                source.add(row * row_bytes + column * (16 / BYTES) * BYTES)
            }};
        }
        macro_rules! pair {
            ($pair:expr) => {{
                let index = PREFIX + $pair * 2;
                let first = _mm_loadu_si128(sector!(PREFIX + $pair * 2).cast());
                if PREFIX != 0 && $pair == 15 {
                    if !next {
                        _mm_storeu_si128(destination.add(index * 16).cast(), first);
                    }
                } else {
                    let second = _mm_loadu_si128(sector!(PREFIX + $pair * 2 + 1).cast());
                    _mm256_stream_si256(
                        destination.add(index * 16).cast(),
                        _mm256_set_m128i(second, first),
                    );
                }
            }};
        }
        pair!(0);
        pair!(1);
        pair!(2);
        pair!(3);
        pair!(4);
        pair!(5);
        pair!(6);
        pair!(7);
        pair!(8);
        pair!(9);
        pair!(10);
        pair!(11);
        pair!(12);
        pair!(13);
        pair!(14);
        pair!(15);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn wide_decode<const BYTES: usize>(
    source: *const u8,
    destination: *mut u8,
    row_elements: usize,
    stream: bool,
) {
    let row_bytes = row_elements * BYTES;
    use std::arch::x86_64::*;
    unsafe {
        let even = _mm512_setr_epi64(0, 1, 4, 5, 8, 9, 12, 13);
        let odd = _mm512_setr_epi64(2, 3, 6, 7, 10, 11, 14, 15);
        for pair in 0..4 {
            let a = _mm512_loadu_si512(source.add(pair * 64).cast());
            let b = _mm512_loadu_si512(source.add(pair * 64 + 256).cast());
            let first = _mm512_permutex2var_epi64(a, even, b);
            let second = _mm512_permutex2var_epi64(a, odd, b);
            let dst = destination.add(pair * 2 * row_bytes);
            if stream {
                _mm512_stream_si512(dst.cast(), first);
                _mm512_stream_si512(dst.add(row_bytes).cast(), second);
            } else {
                _mm512_storeu_si512(dst.cast(), first);
                _mm512_storeu_si512(dst.add(row_bytes).cast(), second);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn load_sectors(source: *const u8) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::*;
    unsafe { _mm512_loadu_si512(source.cast()) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn wide_encode<const BYTES: usize>(
    source: *const u8,
    destination: *mut u8,
    row_elements: usize,
    stream: bool,
    previous: bool,
    next: bool,
) {
    let row_bytes = row_elements * BYTES;
    use std::arch::x86_64::*;
    // Pair 16-byte sectors from two rows into physical cache lines.
    unsafe {
        let encode0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
        let encode1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
        let mut lines = [_mm512_setzero_si512(); 8];
        for pair in 0..4 {
            let a = load_sectors(source.add(pair * 2 * row_bytes).cast());
            let b = load_sectors(source.add((pair * 2 + 1) * row_bytes).cast());
            lines[pair] = _mm512_permutex2var_epi64(a, encode0, b);
            lines[pair + 4] = _mm512_permutex2var_epi64(a, encode1, b);
        }
        let shift = (64 - destination as usize % 64) % 64;
        if stream && shift != 0 {
            // Cached prefix and suffix enclose seven complete streaming lines.
            let lanes = shift / 8;
            if previous {
                // The previous GOB owns the same eight-row strip at y - 8.
                // Its final sectors and this prefix make a complete cache line.
                let a = load_sectors(source.sub(2 * row_bytes));
                let b = load_sectors(source.sub(row_bytes));
                let tail = _mm512_permutex2var_epi64(a, encode1, b);
                let value = match lanes {
                    2 => _mm512_alignr_epi64::<2>(lines[0], tail),
                    4 => _mm512_alignr_epi64::<4>(lines[0], tail),
                    6 => _mm512_alignr_epi64::<6>(lines[0], tail),
                    _ => unreachable!(),
                };
                _mm512_stream_si512(destination.add(shift).sub(64).cast(), value);
            } else {
                _mm512_mask_storeu_epi64(destination.cast(), ((1u16 << lanes) - 1) as u8, lines[0]);
            }
            for line in 0..7 {
                let value = match lanes {
                    2 => _mm512_alignr_epi64::<2>(lines[line + 1], lines[line]),
                    4 => _mm512_alignr_epi64::<4>(lines[line + 1], lines[line]),
                    6 => _mm512_alignr_epi64::<6>(lines[line + 1], lines[line]),
                    _ => unreachable!(),
                };
                _mm512_stream_si512(destination.add(shift + line * 64).cast(), value);
            }
            let value = match lanes {
                2 => _mm512_alignr_epi64::<2>(lines[7], lines[7]),
                4 => _mm512_alignr_epi64::<4>(lines[7], lines[7]),
                6 => _mm512_alignr_epi64::<6>(lines[7], lines[7]),
                _ => unreachable!(),
            };
            if !next {
                _mm512_mask_storeu_epi64(
                    destination.add(448 + shift).cast(),
                    ((1u16 << (8 - lanes)) - 1) as u8,
                    value,
                );
            }
        } else {
            for (line, value) in lines.into_iter().enumerate() {
                let dst = destination.add(line * 64).cast();
                if stream {
                    _mm512_stream_si512(dst, value);
                } else {
                    _mm512_storeu_si512(dst, value);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn decode_shifted_gob<const PREFIX: usize>(
    source: *const u8,
    destination: *mut u8,
    row_bytes: usize,
    tile_bytes: usize,
    column: usize,
    columns: usize,
    height: usize,
) {
    for (row, offsets) in SECTOR_OFFSETS.iter().enumerate().take(height) {
        // SAFETY: caller supplies complete byte rows and valid GOB columns.
        unsafe {
            let dst = destination.add(row * row_bytes);
            if column == 0 {
                for (sector, &offset) in offsets.iter().enumerate().take(PREFIX) {
                    std::ptr::copy_nonoverlapping(source.add(offset), dst.add(sector * 16), 16);
                }
            }
            if column + 1 < columns {
                for sector in 0..4 {
                    let input_sector = PREFIX + sector;
                    let (gob, index) = if input_sector < 4 {
                        (0, input_sector)
                    } else {
                        (tile_bytes, input_sector - 4)
                    };
                    stream_copy_16(
                        source.add(gob + offsets[index]),
                        dst.add((PREFIX + sector) * 16),
                    );
                }
            } else {
                for (sector, &offset) in offsets.iter().enumerate().skip(PREFIX) {
                    std::ptr::copy_nonoverlapping(source.add(offset), dst.add(sector * 16), 16);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn stream_copy_16(source: *const u8, destination: *mut u8) {
    use std::arch::x86_64::{_mm_loadu_si128, _mm_stream_si128};
    let value = unsafe { _mm_loadu_si128(source.cast()) };
    unsafe { _mm_stream_si128(destination.cast(), value) };
}

fn validate(shape: ImageShape, format: BlockFormat, tile: TileShape) -> Result<(), LayoutError> {
    if shape.width == 0
        || shape.height == 0
        || shape.depth == 0
        || shape.layers == 0
        || shape.levels == 0
    {
        return Err(LayoutError::ZeroDimension);
    }
    let valid_shape = match shape.kind {
        ImageKind::D1 => shape.height == 1 && shape.depth == 1 && shape.layers == 1,
        ImageKind::D1Array => shape.height == 1 && shape.depth == 1,
        ImageKind::D2 => shape.depth == 1 && shape.layers == 1,
        ImageKind::D2Array => shape.depth == 1,
        ImageKind::D3 => shape.layers == 1,
        ImageKind::Cube => {
            shape.depth == 1 && shape.width == shape.height && shape.layers.is_multiple_of(6)
        }
    };
    if !valid_shape {
        return Err(LayoutError::InvalidShape);
    }
    if format.width == 0 || format.height == 0 || !matches!(format.bytes, 1 | 2 | 4 | 8 | 16) {
        return Err(LayoutError::InvalidFormat);
    }
    if tile.height_log2 > 5 || tile.depth_log2 > 5 || shape.levels > MAX_LEVELS {
        return Err(LayoutError::InvalidTile);
    }
    Ok(())
}

fn storage_layers(shape: ImageShape) -> usize {
    if shape.kind == ImageKind::D3 {
        1
    } else {
        shape.layers as usize
    }
}

fn mip_extent(value: u32, level: u32) -> u32 {
    value.checked_shr(level).unwrap_or(0).max(1)
}

fn clamp_log2(mut log2: u8, extent: u32, unit: u32) -> u8 {
    while log2 > 0 && (unit << (log2 - 1)) >= extent {
        log2 -= 1;
    }
    log2
}

fn div_ceil(value: usize, divisor: usize) -> usize {
    value / divisor + usize::from(!value.is_multiple_of(divisor))
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value / divisor + u32::from(!value.is_multiple_of(divisor))
}

fn align_up(value: usize, alignment: usize) -> Result<usize, LayoutError> {
    let mask = alignment.checked_sub(1).ok_or(LayoutError::Overflow)?;
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or(LayoutError::Overflow)
}

fn destination_len(layout: &Layout, decode: bool) -> usize {
    if decode {
        layout.linear_size
    } else {
        layout.tiled_size
    }
}

#[inline(always)]
fn sector_offset(x: usize, y: usize) -> usize {
    x / 32 * 256 + y / 2 * 64 + (x % 32) / 16 * 32 + y % 2 * 16 + x % 16
}

#[cfg(test)]
mod tests;
