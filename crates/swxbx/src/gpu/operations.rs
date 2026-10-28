//! Explicit command hypotheses and synchronized arena operations. Source: the public contract.
use super::{commands::Commands, pipelines::ComputePipeline, Context, GlobalMemory};
use ash::vk;
use std::{collections::HashMap, sync::Arc};

/// Host interpretation of a recorded buffer binding for compute.
#[derive(Clone, Debug)]
pub struct ComputeBufferMapping {
    /// Recorded binding index at the selected compute stage.
    pub index: u64,
    /// Reflected descriptor set.
    pub set: u32,
    /// Reflected descriptor binding.
    pub binding: u32,
}

/// Host interpretation of a recorded texel-buffer binding for compute.
#[derive(Clone, Debug)]
pub struct ComputeTexelBufferMapping {
    /// Recorded texture handle at the compute stage.
    pub index: u64,
    /// Reflected descriptor set.
    pub set: u32,
    /// Reflected descriptor binding.
    pub binding: u32,
}

/// Decode an unresolved whole-texture clear into a key, color and channel mask.
pub type ClearDecoder = Arc<dyn Fn([u64; 7]) -> Option<(u64, [f32; 4], u32)> + Send + Sync>;

/// Opt-in hypotheses. Tokens and record layouts are not recovered interface facts.
#[derive(Clone)]
pub struct CommandContract {
    /// Predicate mode selecting a nonzero 32-bit arena word.
    pub condition_nonzero: u64,
    /// Predicate mode selecting a zero 32-bit arena word.
    pub condition_zero: u64,
    /// Recorded stage token selecting compute buffer bindings.
    pub compute_stage: u64,
    /// Explicit binding-to-descriptor mappings.
    pub compute_buffers: Vec<ComputeBufferMapping>,
    /// Explicit texture-handle-to-texel-buffer descriptor mappings.
    pub compute_texel_buffers: Vec<ComputeTexelBufferMapping>,
    /// Base index for compute bindless-handle words, encoded as
    /// `base + (constant_buffer_bank << 16 | byte_offset)`.
    pub compute_texel_location_base: Option<u64>,
    /// Unresolved clear record decoder, absent by default.
    pub clear: Option<ClearDecoder>,
}
impl CommandContract {
    pub(crate) fn valid(&self) -> bool {
        self.condition_nonzero != self.condition_zero
            && self.compute_buffers.iter().enumerate().all(|(i, a)| {
                self.compute_buffers[..i]
                    .iter()
                    .all(|b| a.index != b.index && (a.set, a.binding) != (b.set, b.binding))
            })
            && self.compute_texel_buffers.iter().enumerate().all(|(i, a)| {
                self.compute_texel_buffers[..i]
                    .iter()
                    .all(|b| a.index != b.index && (a.set, a.binding) != (b.set, b.binding))
            })
            && self.compute_texel_location_base.is_none_or(|base| {
                self.compute_texel_buffers
                    .iter()
                    .all(|mapping| !(base..base.saturating_add(1 << 24)).contains(&mapping.index))
            })
    }
}

/// Instance range and optional Vulkan indirect argument buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub instances: u32,
    pub first_instance: u32,
    pub indirect: Option<(vk::Buffer, u64)>,
    pub index_skip: u64,
}
impl Default for Geometry {
    fn default() -> Self {
        Self {
            instances: 1,
            first_instance: 0,
            indirect: None,
            index_skip: 0,
        }
    }
}

pub(crate) struct Execution {
    context: Arc<Context>,
    commands: Commands,
    clock: crate::counters::GpuClock,
    occlusion: vk::QueryPool,
    counting: bool,
    samples: u64,
    pipelines: HashMap<(Vec<u32>, bool), Arc<ComputePipeline>>,
}
impl Execution {
    pub fn new(context: &Arc<Context>) -> Option<Self> {
        Some(Self {
            context: context.clone(),
            commands: Commands::new(context)?,
            clock: crate::counters::GpuClock::new(),
            occlusion: vk::QueryPool::null(),
            counting: false,
            samples: 0,
            pipelines: HashMap::new(),
        })
    }

    fn query(&self, ty: vk::QueryType) -> Option<vk::QueryPool> {
        unsafe {
            self.context
                .device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(ty)
                        .query_count(1),
                    None,
                )
                .ok()
        }
    }

    pub fn reset_occlusion(&mut self) -> Option<()> {
        if self.occlusion == vk::QueryPool::null() {
            self.occlusion = self.query(vk::QueryType::OCCLUSION)?;
        }
        self.samples = 0;
        self.counting = true;
        Some(())
    }

    pub fn occlusion_query(&self) -> Option<vk::QueryPool> {
        self.counting.then_some(self.occlusion)
    }

    pub(super) fn collect_occlusion(
        &mut self,
        submission: super::commands::Submission,
    ) -> Option<()> {
        if self.counting {
            submission.wait()?;
            let mut result = [0_u64];
            unsafe {
                self.context
                    .device
                    .get_query_pool_results(
                        self.occlusion,
                        0,
                        &mut result,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .ok()?;
            }
            self.samples = self.samples.checked_add(result[0])?;
        }
        Some(())
    }

    /// Full memory dependency at the boundary of each synchronous operation.
    unsafe fn barrier(&self, command: vk::CommandBuffer) {
        self.context.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::MEMORY_READ
                        | vk::AccessFlags::MEMORY_WRITE
                        | vk::AccessFlags::HOST_READ,
                )],
            &[],
            &[],
        );
    }

    fn submit(&mut self, command: vk::CommandBuffer) -> Option<()> {
        unsafe {
            self.barrier(command);
        }
        self.commands.submit(false, None)?;
        Some(())
    }

    pub fn fill(
        &mut self,
        memory: &GlobalMemory,
        region: (vk::Buffer, u64),
        size: u64,
        value: u32,
    ) -> Option<()> {
        if !region.1.is_multiple_of(4) || !size.is_multiple_of(4) || size == 0 {
            return None;
        }
        let (command, _) = self.commands.begin_direct()?;
        unsafe {
            self.barrier(command);
            self.context
                .device
                .cmd_fill_buffer(command, region.0, region.1, size, value);
        }
        self.submit(command)?;
        memory.submitted(region.1, size, true, self.commands.submission()?);
        Some(())
    }

    pub fn copy(
        &mut self,
        memory: &GlobalMemory,
        source: (vk::Buffer, u64),
        destination: (vk::Buffer, u64),
        size: u64,
    ) -> Option<()> {
        if size == 0
            || (source.0 == destination.0
                && source.1 < destination.1.checked_add(size)?
                && destination.1 < source.1.checked_add(size)?)
        {
            return None;
        }
        let (command, _) = self.commands.begin_direct()?;
        unsafe {
            self.barrier(command);
            self.context.device.cmd_copy_buffer(
                command,
                source.0,
                destination.0,
                &[vk::BufferCopy::default()
                    .src_offset(source.1)
                    .dst_offset(destination.1)
                    .size(size)],
            );
        }
        self.submit(command)?;
        let submission = self.commands.submission()?;
        memory.submitted(source.1, size, false, submission.clone());
        memory.submitted(destination.1, size, true, submission);
        Some(())
    }

    /// Write one counter report to `region` after every earlier submission
    /// on the queue. The timer is read when the report is processed, as the
    /// Maxwell command processor does in the reference host. Source: the public contract.
    pub fn report(
        &mut self,
        memory: &GlobalMemory,
        region: (vk::Buffer, u64),
        counter: crate::counters::Counter,
    ) -> Option<()> {
        let value = match counter {
            crate::counters::Counter::Timestamp => 0,
            crate::counters::Counter::SamplesPassed => self.samples,
        };
        let bytes = crate::counters::report(value, self.clock.ticks());
        let (command, _) = self.commands.begin_direct()?;
        unsafe {
            self.barrier(command);
            self.context
                .device
                .cmd_update_buffer(command, region.0, region.1, &bytes);
        }
        self.submit(command)?;
        memory.submitted(
            region.1,
            crate::counters::REPORT_SIZE,
            true,
            self.commands.submission()?,
        );
        Some(())
    }

    /// Dispatch with buffer resources only.
    #[cfg(test)]
    pub fn dispatch(
        &mut self,
        memory: &GlobalMemory,
        words: &[u32],
        requires_subgroup_size_32: bool,
        resources: &[(u32, u32, vk::DescriptorBufferInfo)],
        groups: [u32; 3],
        indirect: Option<(vk::Buffer, u64)>,
    ) -> Option<()> {
        self.dispatch_with_images(
            memory,
            words,
            requires_subgroup_size_32,
            resources,
            &[],
            &[],
            &[],
            groups,
            indirect,
        )
    }

    /// Dispatch with buffer resources and storage-image views (in the
    /// GENERAL layout), each named by its set and binding.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_with_images(
        &mut self,
        memory: &GlobalMemory,
        words: &[u32],
        requires_subgroup_size_32: bool,
        resources: &[(u32, u32, vk::DescriptorBufferInfo)],
        images: &[(u32, u32, vk::ImageView)],
        sampled: &[(u32, u32, vk::DescriptorImageInfo)],
        texels: &[(u32, u32, vk::Buffer, u64, u64, vk::Format)],
        groups: [u32; 3],
        indirect: Option<(vk::Buffer, u64)>,
    ) -> Option<()> {
        let key = (words.to_vec(), requires_subgroup_size_32);
        let pipeline = if let Some(p) = self.pipelines.get(&key) {
            p.clone()
        } else {
            let p = Arc::new(
                ComputePipeline::create(
                    &self.context,
                    vk::PipelineCache::null(),
                    words,
                    requires_subgroup_size_32,
                )
                .map_err(|error| eprintln!("compute pipeline: {error}"))
                .ok()?,
            );
            self.pipelines.insert(key, p.clone());
            p
        };
        let bindings = pipeline.bindings();
        if resources.len() + images.len() + sampled.len() + texels.len() > bindings.len()
            || bindings.iter().any(|b| {
                b.count != 1
                    || match b.descriptor_type {
                        vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::STORAGE_BUFFER => {
                            !resources.iter().any(|r| (r.0, r.1) == (b.set, b.binding))
                        }
                        vk::DescriptorType::STORAGE_IMAGE => {
                            !images.iter().any(|r| (r.0, r.1) == (b.set, b.binding))
                        }
                        vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::SAMPLER => false,
                        vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                        | vk::DescriptorType::STORAGE_TEXEL_BUFFER => {
                            !texels.iter().any(|r| (r.0, r.1) == (b.set, b.binding))
                        }
                        _ => true,
                    }
            })
        {
            return None;
        }
        let sizes: Vec<_> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorPoolSize::default()
                    .ty(b.descriptor_type)
                    .descriptor_count(1)
            })
            .collect();
        let context = self.context.clone();
        let device = &context.device;
        let layouts = pipeline.set_layouts();
        let pool = if layouts.is_empty() {
            vk::DescriptorPool::null()
        } else {
            unsafe {
                device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(layouts.len() as u32)
                            .pool_sizes(&sizes),
                        None,
                    )
                    .ok()?
            }
        };
        let views: Vec<_> = texels
            .iter()
            .map(|r| unsafe {
                device
                    .create_buffer_view(
                        &vk::BufferViewCreateInfo::default()
                            .buffer(r.2)
                            .format(r.5)
                            .offset(r.3)
                            .range(r.4),
                        None,
                    )
                    .ok()
            })
            .collect::<Option<_>>()?;
        let result = (|| {
            let sets = if layouts.is_empty() {
                Vec::new()
            } else {
                unsafe {
                    device
                        .allocate_descriptor_sets(
                            &vk::DescriptorSetAllocateInfo::default()
                                .descriptor_pool(pool)
                                .set_layouts(layouts),
                        )
                        .ok()?
                }
            };
            for b in bindings
                .iter()
                .filter(|b| b.descriptor_type == vk::DescriptorType::STORAGE_IMAGE)
            {
                let r = images.iter().find(|r| (r.0, r.1) == (b.set, b.binding))?;
                let info = vk::DescriptorImageInfo::default()
                    .image_view(r.2)
                    .image_layout(vk::ImageLayout::GENERAL);
                unsafe {
                    device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .dst_set(sets[b.set as usize])
                            .dst_binding(b.binding)
                            .descriptor_type(b.descriptor_type)
                            .image_info(std::slice::from_ref(&info))],
                        &[],
                    );
                }
            }
            for b in bindings.iter().filter(|b| {
                matches!(
                    b.descriptor_type,
                    vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::SAMPLER
                )
            }) {
                let Some(r) = sampled.iter().find(|r| (r.0, r.1) == (b.set, b.binding)) else {
                    continue;
                };
                unsafe {
                    device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .dst_set(sets[b.set as usize])
                            .dst_binding(b.binding)
                            .descriptor_type(b.descriptor_type)
                            .image_info(std::slice::from_ref(&r.2))],
                        &[],
                    );
                }
            }
            for b in bindings.iter().filter(|b| {
                matches!(
                    b.descriptor_type,
                    vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                        | vk::DescriptorType::STORAGE_TEXEL_BUFFER
                )
            }) {
                let at = texels
                    .iter()
                    .position(|r| (r.0, r.1) == (b.set, b.binding))?;
                unsafe {
                    device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .dst_set(sets[b.set as usize])
                            .dst_binding(b.binding)
                            .descriptor_type(b.descriptor_type)
                            .texel_buffer_view(std::slice::from_ref(&views[at]))],
                        &[],
                    );
                }
            }
            for b in bindings.iter().filter(|b| {
                !matches!(
                    b.descriptor_type,
                    vk::DescriptorType::STORAGE_IMAGE
                        | vk::DescriptorType::SAMPLED_IMAGE
                        | vk::DescriptorType::SAMPLER
                        | vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                        | vk::DescriptorType::STORAGE_TEXEL_BUFFER
                )
            }) {
                let r = resources
                    .iter()
                    .find(|r| (r.0, r.1) == (b.set, b.binding))?;
                unsafe {
                    device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .dst_set(sets[b.set as usize])
                            .dst_binding(b.binding)
                            .descriptor_type(b.descriptor_type)
                            .buffer_info(std::slice::from_ref(&r.2))],
                        &[],
                    );
                }
            }
            let (command, _) = self.commands.begin_direct()?;
            unsafe {
                self.barrier(command);
                if let Some((buffer, offset)) = indirect {
                    pipeline
                        .record_dispatch_indirect(command, memory, &sets, groups, buffer, offset)
                        .ok()?;
                } else {
                    pipeline
                        .record_dispatch(command, memory, &sets, groups)
                        .ok()?;
                }
            }
            self.submit(command)?;
            let submission = self.commands.submission()?;
            for b in bindings.iter().filter(|b| {
                !matches!(
                    b.descriptor_type,
                    vk::DescriptorType::STORAGE_IMAGE
                        | vk::DescriptorType::SAMPLED_IMAGE
                        | vk::DescriptorType::SAMPLER
                        | vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                        | vk::DescriptorType::STORAGE_TEXEL_BUFFER
                )
            }) {
                let info = &resources
                    .iter()
                    .find(|r| (r.0, r.1) == (b.set, b.binding))?
                    .2;
                memory.submitted(
                    info.offset,
                    info.range,
                    b.descriptor_type == vk::DescriptorType::STORAGE_BUFFER
                        && super::memory::descriptor_writable(words, b.set, b.binding),
                    submission.clone(),
                );
            }
            if super::memory::physical_writes(words) {
                memory.submitted_all(true, submission.clone());
            }
            for (index, r) in texels.iter().enumerate() {
                let writable = bindings.iter().any(|b| {
                    (b.set, b.binding) == (r.0, r.1)
                        && b.descriptor_type == vk::DescriptorType::STORAGE_TEXEL_BUFFER
                });
                memory.submitted(r.3, r.4, writable, submission.clone());
                let _ = views[index];
            }
            if let Some((_, at)) = indirect {
                memory.submitted(at, 12, false, submission);
            }
            // Descriptor resources are owned locally until this dispatch finishes.
            self.commands.wait()
        })();
        unsafe {
            if result.is_none() {
                let _ = self.commands.wait();
            }
            device.destroy_descriptor_pool(pool, None);
            for view in views {
                device.destroy_buffer_view(view, None);
            }
        }
        result
    }
}
impl Drop for Execution {
    fn drop(&mut self) {
        unsafe {
            let _ = self.commands.wait();
            self.context.device.destroy_query_pool(self.occlusion, None);
        }
    }
}
