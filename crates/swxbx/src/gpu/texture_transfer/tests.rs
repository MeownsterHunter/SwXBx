use super::{Scratch, Transfer};
use crate::{
    gpu::{commands::Commands, Context},
    tiling::{BlockFormat, ImageKind, ImageShape, Layout, TileShape},
};
use std::sync::Arc;

#[test]
#[ignore = "requires a compute-capable graphics device"]
fn compute_layout_bytes() {
    let context = Arc::new(Context::new().expect("compute device"));
    let mut commands = Commands::new(&context).expect("command resources");
    let mut transfer = Transfer::new(&context).expect("conversion pipeline");
    let mut layouts = Vec::new();
    for shape in [
        ImageShape {
            width: 65,
            height: 33,
            depth: 1,
            layers: 1,
            levels: 7,
            kind: ImageKind::D2,
        },
        ImageShape {
            width: 17,
            height: 13,
            depth: 1,
            layers: 3,
            levels: 5,
            kind: ImageKind::D2Array,
        },
        ImageShape {
            width: 19,
            height: 17,
            depth: 5,
            layers: 1,
            levels: 5,
            kind: ImageKind::D3,
        },
        ImageShape {
            width: 9,
            height: 9,
            depth: 1,
            layers: 12,
            levels: 4,
            kind: ImageKind::Cube,
        },
    ] {
        for format in [
            BlockFormat {
                width: 1,
                height: 1,
                bytes: 1,
            },
            BlockFormat {
                width: 1,
                height: 1,
                bytes: 2,
            },
            BlockFormat {
                width: 1,
                height: 1,
                bytes: 4,
            },
            BlockFormat {
                width: 1,
                height: 1,
                bytes: 8,
            },
            BlockFormat {
                width: 1,
                height: 1,
                bytes: 16,
            },
            BlockFormat {
                width: 4,
                height: 4,
                bytes: 8,
            },
            BlockFormat {
                width: 4,
                height: 4,
                bytes: 16,
            },
            BlockFormat {
                width: 5,
                height: 4,
                bytes: 16,
            },
            BlockFormat {
                width: 12,
                height: 12,
                bytes: 16,
            },
        ] {
            layouts.push(
                Layout::new(
                    shape,
                    format,
                    TileShape {
                        height_log2: 3,
                        depth_log2: 2,
                    },
                )
                .expect("layout"),
            );
        }
    }
    let largest = layouts.iter().map(Layout::linear_size).max().unwrap();
    transfer.ensure_capacity(largest).expect("scratch");
    let scratch_buffer = transfer.buffer();
    let scratch_address = transfer.address();
    for layout in layouts {
        transfer
            .ensure_capacity(layout.linear_size())
            .expect("scratch reuse");
        assert!(transfer.buffer() == scratch_buffer);
        assert_eq!(transfer.address(), scratch_address);
        let arena = Scratch::new(&context, layout.tiled_size() + 32).expect("arena bytes");
        let tiled: Vec<_> = (0..arena.capacity)
            .map(|i| ((i * 37 + i / 7 + 11) % 251) as u8)
            .collect();
        unsafe {
            std::ptr::copy_nonoverlapping(tiled.as_ptr(), arena.mapped as *mut u8, tiled.len());
        }
        let mut expected = vec![0; layout.linear_size()];
        layout
            .decode(&tiled[16..], &mut expected)
            .expect("CPU decode");
        let (command, _) = commands.begin().expect("begin load");
        transfer
            .record(command, arena.address + 16, &layout, true)
            .expect("record load");
        commands.submit(false, None).expect("submit load");
        commands.wait().expect("wait load");
        let actual = transfer
            .read(layout.linear_size())
            .expect("read linear bytes");
        assert_eq!(
            actual,
            expected,
            "load {:?} {:?}",
            layout.shape(),
            layout.format()
        );

        for (i, byte) in expected.iter_mut().enumerate() {
            *byte = ((i * 19 + i / 13 + 23) % 253) as u8;
        }
        transfer
            .write(&expected)
            .expect("write changed linear bytes");
        let mut expected_tiled = tiled.clone();
        layout
            .encode(&expected, &mut expected_tiled[16..])
            .expect("CPU encode");
        let (command, _) = commands.begin().expect("begin store");
        transfer
            .record(command, arena.address + 16, &layout, false)
            .expect("record store");
        commands.submit(false, None).expect("submit store");
        commands.wait().expect("wait store");
        let actual_tiled =
            unsafe { std::slice::from_raw_parts(arena.mapped as *const u8, arena.capacity) };
        assert_eq!(
            actual_tiled,
            expected_tiled,
            "store {:?} {:?}",
            layout.shape(),
            layout.format()
        );
    }
}
