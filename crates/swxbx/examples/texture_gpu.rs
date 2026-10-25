#[cfg(feature = "vulkan")]
mod run {
    use ash::vk;
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };
    use swxbx::{
        gpu::{
            image_layout::ImageDescriptor, texture_transfer::Transfer, Backend, Context,
            GlobalMemory,
        },
        tiling::{BlockFormat, ImageKind, ImageShape, Layout, TileShape},
    };

    struct Timer {
        context: Arc<Context>,
        pool: vk::CommandPool,
        command: vk::CommandBuffer,
        fence: vk::Fence,
        queries: vk::QueryPool,
        period: f64,
        mask: u64,
    }

    impl Timer {
        fn new(context: &Arc<Context>) -> Self {
            let families = unsafe {
                context
                    .instance
                    .get_physical_device_queue_family_properties(context.physical_device)
            };
            let bits = families[context.queue_family as usize].timestamp_valid_bits;
            assert!(bits != 0, "queue timestamps are required");
            let period = unsafe {
                context
                    .instance
                    .get_physical_device_properties(context.physical_device)
            }
            .limits
            .timestamp_period as f64;
            let pool = unsafe {
                context
                    .device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(context.queue_family),
                        None,
                    )
                    .expect("timing command pool")
            };
            let command = unsafe {
                context
                    .device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .expect("timing command")[0]
            };
            let fence = unsafe {
                context
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .expect("timing fence")
            };
            let queries = unsafe {
                context
                    .device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )
                    .expect("timing queries")
            };
            Self {
                context: Arc::clone(context),
                pool,
                command,
                fence,
                queries,
                period,
                mask: if bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                },
            }
        }

        fn measure(
            &mut self,
            transfer: &Transfer,
            address: u64,
            layout: &Layout,
            load: bool,
            loops: usize,
        ) -> Duration {
            self.measure_batch(&[(transfer, address, layout)], load, loops)
        }

        fn measure_batch(
            &mut self,
            resources: &[(&Transfer, u64, &Layout)],
            load: bool,
            loops: usize,
        ) -> Duration {
            let device = &self.context.device;
            unsafe {
                device
                    .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                    .expect("reset timing command");
                device
                    .reset_fences(&[self.fence])
                    .expect("reset timing fence");
                device
                    .begin_command_buffer(
                        self.command,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .expect("begin timing command");
                device.cmd_reset_query_pool(self.command, self.queries, 0, 2);
                device.cmd_write_timestamp(
                    self.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.queries,
                    0,
                );
            }
            for _ in 0..loops {
                for &(transfer, address, layout) in resources {
                    transfer
                        .record(self.command, address, layout, load)
                        .expect("record timed conversion");
                }
            }
            unsafe {
                device.cmd_write_timestamp(
                    self.command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.queries,
                    1,
                );
                device
                    .end_command_buffer(self.command)
                    .expect("end timing command");
                device
                    .queue_submit(
                        self.context.queue,
                        &[vk::SubmitInfo::default().command_buffers(&[self.command])],
                        self.fence,
                    )
                    .expect("submit timing command");
                device
                    .wait_for_fences(&[self.fence], true, u64::MAX)
                    .expect("wait timing command");
            }
            let mut ticks = [0u64; 2];
            unsafe {
                device
                    .get_query_pool_results(
                        self.queries,
                        0,
                        &mut ticks,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .expect("read timing queries");
            }
            let elapsed = ticks[1].wrapping_sub(ticks[0]) & self.mask;
            Duration::from_secs_f64(elapsed as f64 * self.period / 1e9)
        }
    }

    impl Drop for Timer {
        fn drop(&mut self) {
            unsafe {
                let device = &self.context.device;
                device
                    .device_wait_idle()
                    .expect("complete timing resources");
                device.destroy_query_pool(self.queries, None);
                device.destroy_fence(self.fence, None);
                device.destroy_command_pool(self.pool, None);
            }
        }
    }

    fn useful_bytes(layout: &Layout) -> usize {
        layout.shape().layers as usize
            * layout
                .levels()
                .iter()
                .map(|level| level.row_bytes * level.blocks[1] as usize * level.blocks[2] as usize)
                .sum::<usize>()
    }

    fn wall_time(mut run: impl FnMut(), loops: usize) -> Duration {
        for _ in 0..3 {
            run();
        }
        let start = Instant::now();
        for _ in 0..loops {
            run();
        }
        start.elapsed()
    }

    fn rate(bytes: usize, loops: usize, duration: Duration) -> f64 {
        bytes as f64 * loops as f64 / duration.as_secs_f64() / 1e9
    }

    fn batch64() {
        let shape = ImageShape {
            width: 64,
            height: 64,
            depth: 1,
            layers: 1,
            levels: 1,
            kind: ImageKind::D2,
        };
        let layout = Layout::new(
            shape,
            BlockFormat {
                width: 1,
                height: 1,
                bytes: 4,
            },
            TileShape {
                height_log2: 4,
                depth_log2: 3,
            },
        )
        .expect("batch layout");
        let stride = layout.tiled_size();
        let mut arena_bytes = vec![0xc3; stride * 128];
        let mut linear_images = Vec::new();
        for image in 0..64 {
            let linear: Vec<_> = (0..layout.linear_size())
                .map(|at| ((at * 37 + image * 17 + 11) % 251) as u8)
                .collect();
            layout
                .encode(
                    &linear,
                    &mut arena_bytes[image * stride..(image + 1) * stride],
                )
                .expect("original batch image");
            arena_bytes[(image + 64) * stride..(image + 64) * stride + linear.len()]
                .copy_from_slice(&linear);
            linear_images.push(linear);
        }
        let mut backend = Backend::new(1.0).expect("batch graphics device");
        let context = Arc::clone(backend.context());
        backend.global_memory =
            Some(GlobalMemory::new(&context, arena_bytes.len() as u64).expect("batch arena"));
        let guest = backend
            .allocate_pool(1, 0, arena_bytes.len() as u64)
            .expect("batch pool");
        let arena = backend.global_memory.as_mut().unwrap();
        arena
            .write_pool(1, 0, &arena_bytes)
            .expect("initial batch upload");
        let base_address = arena.addresses().host(guest).expect("batch arena address");
        let resources: Vec<_> = (0..64u64)
            .map(|image| (image + 2, 1, image * stride as u64, &layout))
            .collect();
        for &(key, _, _, _) in &resources {
            assert!(backend.ensure_image(
                key,
                &ImageDescriptor {
                    shape,
                    format: vk::Format::R8G8B8A8_UNORM,
                }
            ));
        }
        assert!(backend.load_tiled_batch(&resources));
        assert!(backend.wait_transfers());
        for (image, &(key, _, _, _)) in resources.iter().enumerate() {
            let (_, _, actual) = backend.readback(key).expect("batch image readback");
            assert_eq!(actual, linear_images[image], "batch image {image}");
        }
        assert!(backend.store_tiled_batch(&resources));
        assert!(backend.wait_transfers());
        let mut roundtrip = vec![0; arena_bytes.len()];
        backend
            .global_memory
            .as_mut()
            .unwrap()
            .read_pool(1, 0, &mut roundtrip)
            .expect("batch arena roundtrip");
        assert_eq!(roundtrip, arena_bytes, "batch arena preflight");

        let bytes = useful_bytes(&layout) * 64;
        let loops = (256 * 1024 * 1024 / bytes).clamp(8, 2048);
        let load_wall = wall_time(
            || {
                assert!(backend.load_tiled_batch(&resources));
                assert!(backend.wait_transfers());
            },
            loops,
        );
        let store_wall = wall_time(
            || {
                assert!(backend.store_tiled_batch(&resources));
                assert!(backend.wait_transfers());
            },
            loops,
        );

        let resident: Vec<_> = resources
            .iter()
            .map(|&(key, pool, offset, layout)| (key, pool, offset + (stride * 64) as u64, layout))
            .collect();
        assert!(backend.load_linear_batch(&resident));
        assert!(backend.wait_transfers());
        for (image, &(key, _, _, _)) in resident.iter().enumerate() {
            assert_eq!(
                backend.readback(key).unwrap().2,
                linear_images[image],
                "resident load preflight"
            );
        }
        assert!(backend.store_linear_batch(&resident));
        assert!(backend.wait_transfers());
        backend
            .global_memory
            .as_mut()
            .unwrap()
            .read_pool(1, 0, &mut roundtrip)
            .unwrap();
        assert_eq!(roundtrip, arena_bytes, "resident store preflight");
        let load_resident = wall_time(
            || {
                assert!(backend.load_linear_batch(&resident));
                assert!(backend.wait_transfers());
            },
            loops,
        );
        let store_resident = wall_time(
            || {
                assert!(backend.store_linear_batch(&resident));
                assert!(backend.wait_transfers());
            },
            loops,
        );
        println!(
            "resident-batch64 {:.3} {:.3} time-ratio {:.3} {:.3}",
            rate(bytes, loops, load_resident),
            rate(bytes, loops, store_resident),
            load_wall.as_secs_f64() / load_resident.as_secs_f64(),
            store_wall.as_secs_f64() / store_resident.as_secs_f64()
        );

        let transfers: Vec<_> = (0..64)
            .map(|_| {
                let mut transfer = Transfer::new(&context).expect("batch compute pipeline");
                transfer
                    .ensure_capacity(layout.linear_size())
                    .expect("batch compute scratch");
                transfer
            })
            .collect();
        let compute_resources: Vec<_> = transfers
            .iter()
            .enumerate()
            .map(|(image, transfer)| (transfer, base_address + (image * stride) as u64, &layout))
            .collect();
        let mut timer = Timer::new(&context);
        timer.measure_batch(&compute_resources, true, 3);
        for (image, transfer) in transfers.iter().enumerate() {
            assert_eq!(
                transfer
                    .read(layout.linear_size())
                    .expect("batch compute readback"),
                linear_images[image],
                "batch compute image {image}",
            );
        }
        let load_compute = timer.measure_batch(&compute_resources, true, loops);
        let store_compute = timer.measure_batch(&compute_resources, false, loops);
        backend
            .global_memory
            .as_mut()
            .unwrap()
            .read_pool(1, 0, &mut roundtrip)
            .expect("batch compute roundtrip");
        assert_eq!(roundtrip, arena_bytes, "batch compute arena preflight");
        println!(
            "2d-64-batch64 {} {} {:.3} {:.3} {:.3} {:.3}",
            bytes,
            loops,
            rate(bytes, loops, load_wall),
            rate(bytes, loops, store_wall),
            rate(bytes, loops, load_compute),
            rate(bytes, loops, store_compute),
        );
    }

    pub fn main() {
        let cases = [
            (
                "2d-64",
                ImageShape {
                    width: 64,
                    height: 64,
                    depth: 1,
                    layers: 1,
                    levels: 1,
                    kind: ImageKind::D2,
                },
            ),
            (
                "2d-1024",
                ImageShape {
                    width: 1024,
                    height: 1024,
                    depth: 1,
                    layers: 1,
                    levels: 1,
                    kind: ImageKind::D2,
                },
            ),
            (
                "2d-4096",
                ImageShape {
                    width: 4096,
                    height: 4096,
                    depth: 1,
                    layers: 1,
                    levels: 1,
                    kind: ImageKind::D2,
                },
            ),
            (
                "3d-256x64",
                ImageShape {
                    width: 256,
                    height: 256,
                    depth: 64,
                    layers: 1,
                    levels: 1,
                    kind: ImageKind::D3,
                },
            ),
            (
                "mips-1024",
                ImageShape {
                    width: 1024,
                    height: 1024,
                    depth: 1,
                    layers: 1,
                    levels: 11,
                    kind: ImageKind::D2,
                },
            ),
            (
                "mips-4096",
                ImageShape {
                    width: 4096,
                    height: 4096,
                    depth: 1,
                    layers: 1,
                    levels: 13,
                    kind: ImageKind::D2,
                },
            ),
        ];
        println!("case useful_bytes loops load_e2e_gbps store_e2e_gbps load_compute_only_gbps store_compute_only_gbps");
        for (name, shape) in cases {
            let layout = Layout::new(
                shape,
                BlockFormat {
                    width: 1,
                    height: 1,
                    bytes: 4,
                },
                TileShape {
                    height_log2: 4,
                    depth_log2: 3,
                },
            )
            .expect("benchmark layout");
            let mut linear = vec![0; layout.linear_size()];
            for (level_index, level) in layout.levels().iter().enumerate() {
                let active = level.row_bytes * level.blocks[1] as usize * level.blocks[2] as usize;
                for at in 0..active {
                    linear[level.linear_offset + at] =
                        ((at * 37 + level_index * 13 + 11) % 251) as u8;
                }
            }
            let mut tiled = vec![0xc3; layout.tiled_size()];
            layout
                .encode(&linear, &mut tiled)
                .expect("generate original tiled input");
            let mut backend = Backend::new(1.0).expect("graphics device");
            let context = Arc::clone(backend.context());
            let resident_offset = (layout.tiled_size() + 255) & !255;
            let arena_size = (resident_offset + linear.len()) as u64;
            backend.global_memory =
                Some(GlobalMemory::new(&context, arena_size).expect("device-visible arena"));
            let guest = backend.allocate_pool(1, 0, arena_size).expect("arena pool");
            let arena = backend.global_memory.as_mut().unwrap();
            arena.write_pool(1, 0, &tiled).expect("upload arena once");
            arena
                .write_pool(1, resident_offset as u64, &linear)
                .expect("resident linear population");
            let address = arena.addresses().host(guest).expect("arena device address");
            assert!(backend.ensure_image(
                2,
                &ImageDescriptor {
                    shape,
                    format: vk::Format::R8G8B8A8_UNORM
                }
            ));
            assert!(backend.load_tiled(2, 1, 0, &layout));
            assert!(backend.wait_transfers());
            let (_, _, readback) = backend.readback(2).expect("image readback");
            assert_eq!(readback, linear, "image preflight {name}");
            assert!(backend.store_tiled(2, 1, 0, &layout));
            assert!(backend.wait_transfers());
            let mut roundtrip = vec![0; tiled.len()];
            backend
                .global_memory
                .as_mut()
                .unwrap()
                .read_pool(1, 0, &mut roundtrip)
                .expect("arena preflight");
            assert_eq!(roundtrip, tiled, "tiled padding preflight {name}");

            let bytes = useful_bytes(&layout);
            let loops = (256 * 1024 * 1024 / bytes).clamp(8, 2048);
            let load_wall = wall_time(
                || {
                    assert!(backend.load_tiled(2, 1, 0, &layout));
                    assert!(backend.wait_transfers());
                },
                loops,
            );
            let store_wall = wall_time(
                || {
                    assert!(backend.store_tiled(2, 1, 0, &layout));
                    assert!(backend.wait_transfers());
                },
                loops,
            );

            let resident = [(2, 1, resident_offset as u64, &layout)];
            assert!(backend.load_linear_batch(&resident));
            assert!(backend.wait_transfers());
            assert_eq!(
                backend.readback(2).unwrap().2,
                linear,
                "resident load preflight"
            );
            assert!(backend.store_linear_batch(&resident));
            assert!(backend.wait_transfers());
            let mut resident_bytes = vec![0; linear.len()];
            backend
                .global_memory
                .as_mut()
                .unwrap()
                .read_pool(1, resident_offset as u64, &mut resident_bytes)
                .unwrap();
            assert_eq!(resident_bytes, linear, "resident store preflight");
            let load_resident = wall_time(
                || {
                    assert!(backend.load_linear_batch(&resident));
                    assert!(backend.wait_transfers());
                },
                loops,
            );
            let store_resident = wall_time(
                || {
                    assert!(backend.store_linear_batch(&resident));
                    assert!(backend.wait_transfers());
                },
                loops,
            );
            println!(
                "resident-{} {:.3} {:.3} time-ratio {:.3} {:.3}",
                name,
                rate(bytes, loops, load_resident),
                rate(bytes, loops, store_resident),
                load_wall.as_secs_f64() / load_resident.as_secs_f64(),
                store_wall.as_secs_f64() / store_resident.as_secs_f64()
            );

            let mut transfer = Transfer::new(&context).expect("resident compute pipeline");
            transfer
                .ensure_capacity(layout.linear_size())
                .expect("resident compute scratch");
            let mut timer = Timer::new(&context);
            timer.measure(&transfer, address, &layout, true, 3);
            assert_eq!(
                transfer
                    .read(layout.linear_size())
                    .expect("compute preflight"),
                linear
            );
            let load_compute = timer.measure(&transfer, address, &layout, true, loops);
            let store_compute = timer.measure(&transfer, address, &layout, false, loops);
            backend
                .global_memory
                .as_mut()
                .unwrap()
                .read_pool(1, 0, &mut roundtrip)
                .expect("compute roundtrip");
            assert_eq!(roundtrip, tiled, "compute padding preflight {name}");
            println!(
                "{} {} {} {:.3} {:.3} {:.3} {:.3}",
                name,
                bytes,
                loops,
                rate(bytes, loops, load_wall),
                rate(bytes, loops, store_wall),
                rate(bytes, loops, load_compute),
                rate(bytes, loops, store_compute)
            );
        }
        batch64();
    }
}

#[cfg(feature = "vulkan")]
fn main() {
    run::main();
}

#[cfg(not(feature = "vulkan"))]
fn main() {
    panic!("enable the graphics feature to run this benchmark");
}
