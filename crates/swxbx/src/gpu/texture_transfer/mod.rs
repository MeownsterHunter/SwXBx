//! Reusable word-owned conversion between tiled arena bytes and linear scratch.
use super::{find_memory_type, Context};
use crate::tiling::Layout;
use ash::vk;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy)]
struct Parameters {
    tiled_address: u64,
    linear_address: u64,
    row_bytes: u64,
    tiled_size: u64,
    linear_size: u64,
    tile_columns: u64,
    rows: u32,
    depth: u32,
    height_log2: u32,
    depth_log2: u32,
    load: u32,
    padding: u32,
    word_base: u64,
}

struct Scratch {
    context: Arc<Context>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: usize,
    capacity: usize,
    address: u64,
}

impl Scratch {
    fn new(context: &Arc<Context>, capacity: usize) -> Option<Self> {
        let capacity = capacity.checked_add(15)? & !15;
        let size = u64::try_from(capacity).ok()?;
        let buffer = unsafe {
            context
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(
                            vk::BufferUsageFlags::STORAGE_BUFFER
                                | vk::BufferUsageFlags::TRANSFER_SRC
                                | vk::BufferUsageFlags::TRANSFER_DST
                                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .ok()?
        };
        let mut result = Self {
            context: Arc::clone(context),
            buffer,
            memory: vk::DeviceMemory::null(),
            mapped: 0,
            capacity,
            address: 0,
        };
        let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory_type = find_memory_type(
            context,
            requirements.memory_type_bits,
            host | vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|| find_memory_type(context, requirements.memory_type_bits, host))?;
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        result.memory = unsafe {
            context
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(memory_type)
                        .push_next(&mut flags),
                    None,
                )
                .ok()?
        };
        unsafe {
            context
                .device
                .bind_buffer_memory(buffer, result.memory, 0)
                .ok()?;
            result.mapped = context
                .device
                .map_memory(result.memory, 0, size, vk::MemoryMapFlags::empty())
                .ok()? as usize;
            result.address = context
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
        }
        (result.address != 0).then_some(result)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        unsafe {
            let _ = self.context.wait_queue();
            if self.mapped != 0 {
                self.context.device.unmap_memory(self.memory);
            }
            self.context.device.destroy_buffer(self.buffer, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}

/// Reusable compute conversion resources and packed-linear scratch storage.
pub struct Transfer {
    context: Arc<Context>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    scratch: Option<Scratch>,
    max_groups: u32,
}

impl Transfer {
    /// Create the conversion pipeline and command slots; return None on setup failure.
    pub fn new(context: &Arc<Context>) -> Option<Self> {
        let limits = unsafe {
            context
                .instance
                .get_physical_device_properties(context.physical_device)
        }
        .limits;
        if limits.max_compute_work_group_invocations < 64
            || limits.max_compute_work_group_size[0] < 64
            || limits.max_push_constants_size < std::mem::size_of::<Parameters>() as u32
        {
            return None;
        }
        let mut result = Self {
            context: Arc::clone(context),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            scratch: None,
            max_groups: limits.max_compute_work_group_count[0],
        };
        result.pipeline_layout = unsafe {
            context
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&[
                        vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .offset(0)
                            .size(std::mem::size_of::<Parameters>() as u32),
                    ]),
                    None,
                )
                .ok()?
        };
        let words: Vec<u32> = include_bytes!("convert.spv")
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        let shader = context.create_global_shader_module(&words).ok()?;
        let created = unsafe {
            context.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .layout(result.pipeline_layout)
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(shader)
                            .name(c"main"),
                    )],
                None,
            )
        };
        unsafe {
            context.device.destroy_shader_module(shader, None);
        }
        match created {
            Ok(pipelines) => result.pipeline = pipelines[0],
            Err((pipelines, _)) => {
                for pipeline in pipelines {
                    unsafe {
                        context.device.destroy_pipeline(pipeline, None);
                    }
                }
                return None;
            }
        }
        Some(result)
    }

    /// Grow scratch if necessary, invalidating recordings that use replaced storage.
    pub fn ensure_capacity(&mut self, size: usize) -> Option<()> {
        if size == 0 {
            return None;
        }
        if self
            .scratch
            .as_ref()
            .is_some_and(|scratch| scratch.capacity >= size)
        {
            return Some(());
        }
        let scratch = Scratch::new(&self.context, size)?;
        self.scratch = Some(scratch);
        Some(())
    }

    /// Current scratch capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.scratch.as_ref().map_or(0, |scratch| scratch.capacity)
    }

    /// Borrow the scratch buffer handle; it remains owned by this transfer object.
    pub fn buffer(&self) -> vk::Buffer {
        self.scratch
            .as_ref()
            .map_or(vk::Buffer::null(), |scratch| scratch.buffer)
    }

    /// Current scratch buffer device address; growth invalidates earlier addresses.
    pub fn address(&self) -> u64 {
        self.scratch.as_ref().map_or(0, |scratch| scratch.address)
    }

    /// The caller waits for previous scratch users before writing mapped bytes.
    pub fn write(&mut self, bytes: &[u8]) -> Option<()> {
        let scratch = self.scratch.as_ref()?;
        if bytes.len() > scratch.capacity {
            return None;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), scratch.mapped as *mut u8, bytes.len());
        }
        Some(())
    }

    /// The caller waits for GPU completion and a host-read dependency first.
    pub fn read(&self, size: usize) -> Option<Vec<u8>> {
        let scratch = self.scratch.as_ref()?;
        if size > scratch.capacity {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(scratch.mapped as *const u8, size) }.to_vec())
    }

    /// Records conversion only; the caller owns arena bounds and queue ordering.
    pub fn record(
        &self,
        command: vk::CommandBuffer,
        arena_address: u64,
        layout: &Layout,
        load: bool,
    ) -> Option<()> {
        self.record_at(command, arena_address, layout, load, 0)
    }

    /// Records into a checked staging range. The caller owns queue ordering.
    pub fn record_at(
        &self,
        command: vk::CommandBuffer,
        arena_address: u64,
        layout: &Layout,
        load: bool,
        offset: usize,
    ) -> Option<()> {
        self.record_range(command, arena_address, layout, load, offset, true)
    }

    /// Disjoint dispatches share dependencies supplied by the batch recorder.
    pub(super) fn record_disjoint(
        &self,
        command: vk::CommandBuffer,
        arena_address: u64,
        layout: &Layout,
        load: bool,
        offset: usize,
    ) -> Option<()> {
        self.record_range(command, arena_address, layout, load, offset, false)
    }

    fn record_range(
        &self,
        command: vk::CommandBuffer,
        arena_address: u64,
        layout: &Layout,
        load: bool,
        offset: usize,
        synchronize: bool,
    ) -> Option<()> {
        let scratch = self.scratch.as_ref()?;
        if arena_address == 0
            || !arena_address.is_multiple_of(4)
            || !offset.is_multiple_of(16)
            || offset.checked_add(layout.linear_size())? > scratch.capacity
            || self.max_groups == 0
        {
            return None;
        }
        let linear_address = scratch.address.checked_add(u64::try_from(offset).ok()?)?;
        let arena_end = arena_address.checked_add(u64::try_from(layout.tiled_size()).ok()?)?;
        let scratch_end = linear_address.checked_add(u64::try_from(layout.linear_size()).ok()?)?;
        if arena_address < scratch_end && linear_address < arena_end {
            return None;
        }
        let device = &self.context.recorder();
        unsafe {
            if synchronize {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(
                            vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE,
                        )
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )],
                    &[],
                    &[],
                );
            }
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
        }
        for layer in 0..layout.shape().layers {
            for level in layout.levels() {
                let tiled_offset = usize::try_from(layer)
                    .ok()?
                    .checked_mul(layout.array_stride())?
                    .checked_add(level.tiled_offset)?;
                let linear_offset = usize::try_from(layer)
                    .ok()?
                    .checked_mul(layout.linear_layer_stride())?
                    .checked_add(level.linear_offset)?;
                if !tiled_offset.is_multiple_of(4)
                    || !linear_offset.is_multiple_of(4)
                    || !level.tiled_size.is_multiple_of(4)
                    || !level.linear_size.is_multiple_of(4)
                    || level.row_bytes == 0
                    || level.blocks[1] == 0
                    || level.blocks[2] == 0
                    || u32::from(level.tile.height_log2) + u32::from(level.tile.depth_log2) > 31
                {
                    return None;
                }
                let mut parameters = Parameters {
                    tiled_address: arena_address.checked_add(u64::try_from(tiled_offset).ok()?)?,
                    linear_address: linear_address
                        .checked_add(u64::try_from(linear_offset).ok()?)?,
                    row_bytes: u64::try_from(level.row_bytes).ok()?,
                    tiled_size: u64::try_from(level.tiled_size).ok()?,
                    linear_size: u64::try_from(level.linear_size).ok()?,
                    tile_columns: u64::try_from(level.row_bytes.div_ceil(64)).ok()?,
                    rows: level.blocks[1],
                    depth: level.blocks[2],
                    height_log2: u32::from(level.tile.height_log2),
                    depth_log2: u32::from(level.tile.depth_log2),
                    load: u32::from(load),
                    padding: 0,
                    word_base: 0,
                };
                let words = if load {
                    parameters.linear_size
                } else {
                    parameters.tiled_size
                } / 4;
                while parameters.word_base < words {
                    let groups = (words - parameters.word_base)
                        .div_ceil(64)
                        .min(u64::from(self.max_groups)) as u32;
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            (&parameters as *const Parameters).cast::<u8>(),
                            std::mem::size_of::<Parameters>(),
                        )
                    };
                    unsafe {
                        device.cmd_push_constants(
                            command,
                            self.pipeline_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            bytes,
                        );
                        device.cmd_dispatch(command, groups, 1, 1);
                    }
                    parameters.word_base += u64::from(groups) * 64;
                }
            }
        }
        if synchronize {
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::MEMORY_READ
                                | vk::AccessFlags::MEMORY_WRITE
                                | vk::AccessFlags::HOST_READ,
                        )],
                    &[],
                    &[],
                );
            }
        }
        Some(())
    }
}

impl Drop for Transfer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.context.wait_queue();
            self.context.device.destroy_pipeline(self.pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

#[cfg(test)]
mod tests;
