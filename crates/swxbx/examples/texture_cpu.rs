use std::{
    hint::black_box,
    ops::{Deref, DerefMut},
    time::{Duration, Instant},
};
use swxbx::tiling::{BlockFormat, ImageKind, ImageShape, Layout, TileShape};

struct AlignedBytes {
    backing: Vec<u8>,
    offset: usize,
    len: usize,
}

impl AlignedBytes {
    fn new(len: usize, value: u8, aligned: bool) -> Self {
        let backing = vec![value; len + 80];
        let base = backing.as_ptr() as usize;
        let aligned_offset = (64 - base % 64) % 64;
        let offset = if aligned {
            aligned_offset
        } else {
            aligned_offset + 16
        };
        let result = Self {
            backing,
            offset,
            len,
        };
        assert_eq!(
            (result.as_ptr() as usize) % 64,
            if aligned { 0 } else { 16 }
        );
        result
    }
}

impl Deref for AlignedBytes {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.backing[self.offset..self.offset + self.len]
    }
}

impl DerefMut for AlignedBytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.backing[self.offset..self.offset + self.len]
    }
}

struct Case {
    name: &'static str,
    shape: ImageShape,
}

fn fill_linear(layout: &Layout) -> Vec<u8> {
    let mut bytes = vec![0x5a; layout.linear_size()];
    for layer in 0..layout.shape().layers {
        for (level_index, level) in layout.levels().iter().enumerate() {
            for z in 0..level.blocks[2] {
                for y in 0..level.blocks[1] {
                    let at = layout
                        .linear_byte_offset(level_index as u32, layer, 0, y, z)
                        .unwrap();
                    for (x, byte) in bytes[at..at + level.row_bytes].iter_mut().enumerate() {
                        *byte = ((at + x) as u8).wrapping_mul(37).wrapping_add(11);
                    }
                }
            }
        }
    }
    bytes
}

fn useful_bytes(layout: &Layout) -> usize {
    layout.shape().layers as usize
        * layout
            .levels()
            .iter()
            .map(|level| level.row_bytes * level.blocks[1] as usize * level.blocks[2] as usize)
            .sum::<usize>()
}

fn loops_for(bytes: usize) -> usize {
    std::env::var("TEXTURE_BENCH_LOOPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| (512 * 1024 * 1024usize / bytes.max(1)).clamp(8, 20_000))
}

fn measure(mut run: impl FnMut(), loops: usize) -> Duration {
    for _ in 0..3 {
        run();
    }
    let samples = std::env::var("TEXTURE_BENCH_SAMPLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3usize)
        .max(1);
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        for _ in 0..loops {
            run();
        }
        times.push(start.elapsed());
    }
    times.sort_unstable();
    times[times.len() / 2]
}

fn rate(bytes: usize, loops: usize, duration: Duration) -> f64 {
    bytes as f64 * loops as f64 / duration.as_secs_f64() / 1e9
}

fn main() {
    let cases = [
        Case {
            name: "2d-64",
            shape: ImageShape {
                width: 64,
                height: 64,
                depth: 1,
                layers: 1,
                levels: 1,
                kind: ImageKind::D2,
            },
        },
        Case {
            name: "2d-1024",
            shape: ImageShape {
                width: 1024,
                height: 1024,
                depth: 1,
                layers: 1,
                levels: 1,
                kind: ImageKind::D2,
            },
        },
        Case {
            name: "2d-4096",
            shape: ImageShape {
                width: 4096,
                height: 4096,
                depth: 1,
                layers: 1,
                levels: 1,
                kind: ImageKind::D2,
            },
        },
        Case {
            name: "3d-256x64",
            shape: ImageShape {
                width: 256,
                height: 256,
                depth: 64,
                layers: 1,
                levels: 1,
                kind: ImageKind::D3,
            },
        },
        Case {
            name: "mips-1024",
            shape: ImageShape {
                width: 1024,
                height: 1024,
                depth: 1,
                layers: 1,
                levels: 11,
                kind: ImageKind::D2,
            },
        },
        Case {
            name: "mips-4096",
            shape: ImageShape {
                width: 4096,
                height: 4096,
                depth: 1,
                layers: 1,
                levels: 13,
                kind: ImageKind::D2,
            },
        },
    ];
    let workers = std::env::var("TEXTURE_BENCH_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    println!("case useful_bytes loops memcpy_ms memcpy_gbps encode_ms encode_gbps decode_ms decode_gbps parallel_encode_ms parallel_encode_gbps parallel_decode_ms parallel_decode_gbps workers");
    let filter = std::env::args().nth(1).unwrap_or_default();
    for bytes in [1, 2, 4, 8, 16] {
        for case in &cases {
            if !format!("{}-b{}", case.name, bytes).contains(&filter) {
                continue;
            }
            let layout = Layout::new(
                case.shape,
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
            let pattern = fill_linear(&layout);
            for (alignment, aligned) in [("normal16", false), ("aligned64", true)] {
                let mut linear = AlignedBytes::new(layout.linear_size(), 0x5a, aligned);
                linear.copy_from_slice(&pattern);
                let mut tiled = AlignedBytes::new(layout.tiled_size(), 0xc3, aligned);
                layout.encode(&linear, &mut tiled).unwrap();
                let mut decoded = AlignedBytes::new(layout.linear_size(), 0x5a, aligned);
                layout.decode(&tiled, &mut decoded).unwrap();
                assert_eq!(&*decoded, &*linear);
                black_box(
                    decoded
                        .iter()
                        .fold(0u8, |sum, byte| sum.wrapping_add(*byte)),
                );
                let useful = useful_bytes(&layout);
                let loops = loops_for(useful);
                let mut copied = vec![0; linear.len()];
                let memcpy = measure(|| copied.copy_from_slice(black_box(&linear)), loops);
                let serial_encode = measure(
                    || {
                        layout
                            .encode(black_box(&linear), black_box(&mut tiled))
                            .unwrap();
                    },
                    loops,
                );
                let serial_decode = measure(
                    || {
                        layout
                            .decode(black_box(&tiled), black_box(&mut decoded))
                            .unwrap();
                    },
                    loops,
                );
                let parallel_encode = measure(
                    || {
                        layout
                            .encode_parallel(black_box(&linear), black_box(&mut tiled), workers)
                            .unwrap();
                    },
                    loops,
                );
                let parallel_decode = measure(
                    || {
                        layout
                            .decode_parallel(black_box(&tiled), black_box(&mut decoded), workers)
                            .unwrap();
                    },
                    loops,
                );
                black_box(tiled.iter().fold(0u8, |sum, byte| sum.wrapping_add(*byte)));
                black_box(
                    decoded
                        .iter()
                        .fold(0u8, |sum, byte| sum.wrapping_add(*byte)),
                );
                println!(
                "{}-b{}-{} {} {} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} {}",
                case.name,
                bytes,
                alignment,
                useful,
                loops,
                memcpy.as_secs_f64() * 1000.0,
                rate(useful, loops, memcpy),
                serial_encode.as_secs_f64() * 1000.0,
                rate(useful, loops, serial_encode),
                serial_decode.as_secs_f64() * 1000.0,
                rate(useful, loops, serial_decode),
                parallel_encode.as_secs_f64() * 1000.0,
                rate(useful, loops, parallel_encode),
                parallel_decode.as_secs_f64() * 1000.0,
                rate(useful, loops, parallel_decode),
                workers
            );
            }
        }
    }
}
