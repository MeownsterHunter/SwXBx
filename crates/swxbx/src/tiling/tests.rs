use super::*;

fn oracle_offset(
    layout: &Layout,
    level_index: usize,
    layer: usize,
    x: usize,
    y: usize,
    z: usize,
) -> usize {
    let level = layout.levels()[level_index];
    let h = level.tile.height_log2 as usize;
    let d = level.tile.depth_log2 as usize;
    let tile_width = 64;
    let tile_height = 8 << h;
    let tile_depth = 1 << d;
    let columns = div_ceil(level.row_bytes, tile_width);
    let rows = div_ceil(level.blocks[1] as usize, tile_height);
    let tile_x = x / tile_width;
    let tile_y = y / tile_height;
    let tile_z = z / tile_depth;
    let gob_y = (y % tile_height) / 8;
    let gob_z = z % tile_depth;
    let tile_index = (tile_z * rows + tile_y) * columns + tile_x;
    let gob_index = gob_z * (1 << h) + gob_y;
    layer * layout.array_stride()
        + level.tiled_offset
        + tile_index * (512 << (h + d))
        + gob_index * 512
        + (x % 64) / 32 * 256
        + (y % 8) / 2 * 64
        + (x % 32) / 16 * 32
        + (y % 2) * 16
        + x % 16
}

fn shape(kind: ImageKind) -> ImageShape {
    ImageShape {
        width: 137,
        height: match kind {
            ImageKind::D1 | ImageKind::D1Array => 1,
            ImageKind::Cube => 137,
            _ => 73,
        },
        depth: if kind == ImageKind::D3 { 9 } else { 1 },
        layers: match kind {
            ImageKind::D1 | ImageKind::D2 => 1,
            ImageKind::D1Array => 3,
            ImageKind::D2Array => 3,
            ImageKind::D3 => 1,
            ImageKind::Cube => 6,
        },
        levels: 5,
        kind,
    }
}

#[test]
fn address_matches_independent_tile_oracle() {
    let layout = Layout::new(
        shape(ImageKind::D3),
        BlockFormat {
            width: 1,
            height: 1,
            bytes: 4,
        },
        TileShape {
            height_log2: 3,
            depth_log2: 2,
        },
    )
    .unwrap();
    for (level_index, level) in layout.levels().iter().enumerate() {
        for z in 0..level.blocks[2] as usize {
            for y in 0..level.blocks[1] as usize {
                for x in 0..level.row_bytes {
                    assert_eq!(
                        layout.tiled_byte_offset(
                            level_index as u32,
                            0,
                            x as u32,
                            y as u32,
                            z as u32
                        ),
                        Some(oracle_offset(&layout, level_index, 0, x, y, z))
                    );
                }
            }
        }
    }
}

#[test]
fn round_trip_all_kinds_and_block_sizes_preserves_padding() {
    for kind in [
        ImageKind::D1,
        ImageKind::D1Array,
        ImageKind::D2,
        ImageKind::D2Array,
        ImageKind::D3,
        ImageKind::Cube,
    ] {
        for bytes in [1, 2, 4, 8, 16] {
            let layout = Layout::new(
                shape(kind),
                BlockFormat {
                    width: 4,
                    height: 4,
                    bytes,
                },
                TileShape {
                    height_log2: 4,
                    depth_log2: 3,
                },
            )
            .unwrap();
            let mut linear = vec![0xa5; layout.linear_size()];
            for layer in 0..storage_layers(layout.shape()) as u32 {
                for (level_index, level) in layout.levels().iter().enumerate() {
                    for z in 0..level.blocks[2] {
                        for y in 0..level.blocks[1] {
                            for x in 0..level.row_bytes {
                                let at = layout
                                    .linear_byte_offset(level_index as u32, layer, x as u32, y, z)
                                    .unwrap();
                                linear[at] = at.wrapping_mul(37) as u8;
                            }
                        }
                    }
                }
            }
            let original = linear.clone();
            let mut tiled = vec![0x3c; layout.tiled_size()];
            layout.encode(&linear, &mut tiled).unwrap();
            let mut decoded = vec![0xa5; layout.linear_size()];
            layout.decode(&tiled, &mut decoded).unwrap();
            assert_eq!(decoded, original);
            let serial = tiled.clone();
            let mut parallel = vec![0x3c; layout.tiled_size()];
            layout.encode_parallel(&linear, &mut parallel, 4).unwrap();
            assert_eq!(parallel, serial);
            let mut parallel_decoded = vec![0xa5; layout.linear_size()];
            layout
                .decode_parallel(&parallel, &mut parallel_decoded, 4)
                .unwrap();
            assert_eq!(parallel_decoded, original);
        }
    }
}

#[test]
fn geometry_and_padding_regression() {
    let layout = Layout::new(
        ImageShape {
            width: 65,
            height: 9,
            depth: 1,
            layers: 2,
            levels: 3,
            kind: ImageKind::D2Array,
        },
        BlockFormat {
            width: 1,
            height: 1,
            bytes: 1,
        },
        TileShape {
            height_log2: 2,
            depth_log2: 0,
        },
    )
    .unwrap();
    assert_eq!(layout.levels()[0].row_bytes, 65);
    assert_eq!(layout.levels()[0].tile.height_log2, 1);
    assert_eq!(layout.levels()[1].extent, [32, 4, 1]);
    assert_eq!(layout.levels()[2].extent, [16, 2, 1]);
    assert!(layout
        .levels()
        .windows(2)
        .all(|pair| pair[1].linear_offset % 16 == 0
            && pair[1].linear_offset >= pair[0].linear_offset + pair[0].linear_size));
    assert_eq!(
        layout.array_stride()
            % (512 << (layout.levels()[0].tile.height_log2 + layout.levels()[0].tile.depth_log2)),
        0
    );
}

#[test]
fn rejects_invalid_shapes_and_short_buffers() {
    let invalid = ImageShape {
        width: 1,
        height: 1,
        depth: 2,
        layers: 1,
        levels: 1,
        kind: ImageKind::D2,
    };
    assert_eq!(
        Layout::new(
            invalid,
            BlockFormat {
                width: 1,
                height: 1,
                bytes: 4
            },
            TileShape {
                height_log2: 0,
                depth_log2: 0
            }
        ),
        Err(LayoutError::InvalidShape)
    );
    let layout = Layout::new(
        shape(ImageKind::D2),
        BlockFormat {
            width: 1,
            height: 1,
            bytes: 4,
        },
        TileShape {
            height_log2: 0,
            depth_log2: 0,
        },
    )
    .unwrap();
    assert!(matches!(
        layout.decode(&[], &mut vec![0; layout.linear_size()]),
        Err(LayoutError::SourceSize { .. })
    ));
}

#[test]
fn tiny_array_stride_uses_clamped_base_tile() {
    let layout = Layout::new(
        ImageShape {
            width: 1,
            height: 1,
            depth: 1,
            layers: 2,
            levels: 1,
            kind: ImageKind::D2Array,
        },
        BlockFormat {
            width: 1,
            height: 1,
            bytes: 1,
        },
        TileShape {
            height_log2: 5,
            depth_log2: 0,
        },
    )
    .unwrap();
    assert_eq!(layout.levels()[0].tile.height_log2, 0);
    assert_eq!(layout.array_stride(), 512);
}

#[test]
fn parallel_block_rows_match_serial_for_single_level_2d() {
    let layout = Layout::new(
        ImageShape {
            width: 1024,
            height: 1024,
            depth: 1,
            layers: 1,
            levels: 1,
            kind: ImageKind::D2,
        },
        BlockFormat {
            width: 1,
            height: 1,
            bytes: 4,
        },
        TileShape {
            height_log2: 4,
            depth_log2: 0,
        },
    )
    .unwrap();
    let linear: Vec<_> = (0..layout.linear_size()).map(|i| i as u8).collect();
    let mut serial = vec![0x55; layout.tiled_size()];
    let mut parallel = serial.clone();
    layout.encode(&linear, &mut serial).unwrap();
    layout.encode_parallel(&linear, &mut parallel, 4).unwrap();
    assert_eq!(parallel, serial);
    let mut decoded = vec![0; layout.linear_size()];
    layout.decode_parallel(&parallel, &mut decoded, 4).unwrap();
    assert_eq!(decoded, linear);
}

#[test]
fn aligned_large_streaming_round_trip_is_exact() {
    fn aligned(len: usize, value: u8, shift: usize) -> (Vec<u8>, usize) {
        let bytes = vec![value; len + 127];
        let offset = (64 - bytes.as_ptr() as usize % 64) % 64;
        assert_eq!((unsafe { bytes.as_ptr().add(offset) } as usize) % 64, 0);
        (bytes, offset + shift)
    }
    for width in [1024, 4096] {
        let layout = Layout::new(
            ImageShape {
                width,
                height: width,
                depth: 1,
                layers: 1,
                levels: 1,
                kind: ImageKind::D2,
            },
            BlockFormat {
                width: 1,
                height: 1,
                bytes: 4,
            },
            TileShape {
                height_log2: 4,
                depth_log2: 0,
            },
        )
        .unwrap();
        for shift in [0, 16, 32, 48] {
            let (mut linear, linear_at) = aligned(layout.linear_size(), 0x5a, shift);
            let linear = &mut linear[linear_at..linear_at + layout.linear_size()];
            for (index, byte) in linear.iter_mut().enumerate() {
                *byte = index.wrapping_mul(37) as u8;
            }
            let expected = linear.to_vec();
            let (mut tiled, tiled_at) = aligned(layout.tiled_size(), 0xc3, shift);
            {
                let tiled_image = &mut tiled[tiled_at..tiled_at + layout.tiled_size()];
                layout.encode(linear, tiled_image).unwrap();
                linear.fill(0x5a);
                layout.decode(tiled_image, linear).unwrap();
                assert_eq!(linear, expected);
                layout.encode_parallel(linear, tiled_image, 4).unwrap();
                linear.fill(0x5a);
                layout.decode_parallel(tiled_image, linear, 4).unwrap();
                assert_eq!(linear, expected);
            }
            assert!(tiled[..tiled_at].iter().all(|byte| *byte == 0xc3));
            assert!(tiled[tiled_at + layout.tiled_size()..]
                .iter()
                .all(|byte| *byte == 0xc3));
        }
    }
}

#[test]
fn seventeen_rows_keep_a_four_gob_tile_and_pack_next_mip_exactly() {
    let layout = Layout::new(
        ImageShape {
            width: 64,
            height: 17,
            depth: 1,
            layers: 1,
            levels: 2,
            kind: ImageKind::D2,
        },
        BlockFormat {
            width: 1,
            height: 1,
            bytes: 1,
        },
        TileShape {
            height_log2: 3,
            depth_log2: 0,
        },
    )
    .unwrap();
    assert_eq!(layout.levels()[0].tile.height_log2, 2);
    assert_eq!(layout.levels()[0].tiled_offset, 0);
    assert_eq!(layout.levels()[0].tiled_size, 2048);
    assert_eq!(layout.levels()[1].extent, [32, 8, 1]);
    assert_eq!(layout.levels()[1].tile.height_log2, 0);
    assert_eq!(layout.levels()[1].tiled_offset, 2048);
    assert_eq!(layout.levels()[1].tiled_size, 512);
    assert_eq!(layout.levels()[1].linear_offset, 1088);
    assert_eq!(layout.tiled_size(), 4096);
}

#[test]
fn cube_faces_must_be_square() {
    let result = Layout::new(
        ImageShape {
            width: 8,
            height: 4,
            depth: 1,
            layers: 6,
            levels: 1,
            kind: ImageKind::Cube,
        },
        BlockFormat {
            width: 1,
            height: 1,
            bytes: 4,
        },
        TileShape {
            height_log2: 0,
            depth_log2: 0,
        },
    );
    assert_eq!(result, Err(LayoutError::InvalidShape));
}

#[test]
fn specialized_streaming_matches_oracle_with_partial_rows_and_guards() {
    for bytes in [1, 2, 4, 8, 16] {
        for kind in [ImageKind::D2, ImageKind::D3] {
            let layout = Layout::new(
                ImageShape {
                    width: if kind == ImageKind::D2 {
                        4096 / u32::from(bytes)
                    } else {
                        1024 / u32::from(bytes)
                    },
                    height: if kind == ImageKind::D2 { 1025 } else { 65 },
                    depth: if kind == ImageKind::D2 { 1 } else { 65 },
                    layers: 1,
                    levels: if kind == ImageKind::D2 { 10 } else { 1 },
                    kind,
                },
                BlockFormat {
                    width: 1,
                    height: 1,
                    bytes,
                },
                TileShape {
                    height_log2: 4,
                    depth_log2: 3,
                },
            )
            .unwrap();
            for shift in [0, 1, 16, 32, 48] {
                let mut source = vec![0x5a; layout.linear_size() + 128];
                let source_at = (64 - source.as_ptr() as usize % 64) % 64 + shift;
                let mut expected = vec![0xc3; layout.tiled_size()];
                for (mip, level) in layout.levels().iter().enumerate() {
                    for z in 0..level.blocks[2] as usize {
                        for y in 0..level.blocks[1] as usize {
                            for x in 0..level.row_bytes {
                                let at = level.linear_offset
                                    + (z * level.blocks[1] as usize + y) * level.row_bytes
                                    + x;
                                let value =
                                    (at as u8).wrapping_mul(37).wrapping_add((at >> 8) as u8);
                                source[source_at + at] = value;
                                expected[oracle_offset(&layout, mip, 0, x, y, z)] = value;
                            }
                        }
                    }
                }
                let mut destination = vec![0xc3; layout.tiled_size() + 128];
                let destination_at = (64 - destination.as_ptr() as usize % 64) % 64 + shift;
                let tiled = &mut destination[destination_at..destination_at + layout.tiled_size()];
                layout.encode(&source[source_at..], tiled).unwrap();
                assert_eq!(
                    tiled, &expected,
                    "encode {kind:?}, bytes {bytes}, shift {shift}"
                );
                let mut decoded = vec![0x5a; layout.linear_size() + 128];
                let decoded_at = (64 - decoded.as_ptr() as usize % 64) % 64 + shift;
                layout
                    .decode(
                        tiled,
                        &mut decoded[decoded_at..decoded_at + layout.linear_size()],
                    )
                    .unwrap();
                assert_eq!(
                    &decoded[decoded_at..decoded_at + layout.linear_size()],
                    &source[source_at..source_at + layout.linear_size()]
                );
                tiled.fill(0xc3);
                layout
                    .encode_parallel(&source[source_at..], tiled, 4)
                    .unwrap();
                assert_eq!(
                    tiled, &expected,
                    "parallel encode {kind:?}, bytes {bytes}, shift {shift}"
                );
                decoded[decoded_at..decoded_at + layout.linear_size()].fill(0x5a);
                layout
                    .decode_parallel(
                        tiled,
                        &mut decoded[decoded_at..decoded_at + layout.linear_size()],
                        4,
                    )
                    .unwrap();
                assert_eq!(
                    &decoded[decoded_at..decoded_at + layout.linear_size()],
                    &source[source_at..source_at + layout.linear_size()]
                );
                assert!(decoded[..decoded_at]
                    .iter()
                    .chain(&decoded[decoded_at + layout.linear_size()..])
                    .all(|&b| b == 0x5a));
                assert!(destination[..destination_at]
                    .iter()
                    .chain(&destination[destination_at + layout.tiled_size()..])
                    .all(|&b| b == 0xc3));
            }
        }
    }
}
