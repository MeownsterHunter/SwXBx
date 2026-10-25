//! swxbx's record of the program's objects.
//!
//! The program allocates every object itself and passes its address. swxbx
//! keeps what it knows about each object here, keyed by that address, and
//! never writes into the program's object memory. That way the size and
//! layout of the objects need not be known. It assumes the program only
//! touches objects through the API, which has not been contradicted by
//! observation but has not been confirmed either (see docs/design.md).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

/// Owned command and argument snapshots produced by recording.
#[derive(Debug, Clone, PartialEq)]
pub enum RecordedCommand {
    /// Deferred command hypotheses. See
    Operation {
        /// Unprefixed command family.
        kind: &'static str,
        /// Full-width arguments, excluding the command object.
        arguments: [u64; 7],
    },
    /// Write a counter report when the queue reaches this command.
    ///
    ReportCounter {
        /// The counter the guest token selected.
        counter: crate::counters::Counter,
        /// GPU address of the sixteen-byte report.
        address: u64,
    },
    /// Start a new accumulation interval for a counter.
    ResetCounter(crate::counters::Counter),
    /// Select retained color and depth texture keys with opaque view arguments.
    SetRenderTargets {
        /// Color texture object keys in attachment order.
        colors: Vec<u64>,
        /// Depth value or depth texture key, according to the enclosing command.
        depth: u64,
        /// Opaque color and depth view arguments.
        views: [u64; 2],
    },
    /// Select the program object key.
    BindProgram(u64),
    /// Retain one known or opaque state object binding.
    BindState {
        /// State or binding category identified by the function family.
        kind: &'static str,
        /// Snapshot of retained setters for a known state object.
        settings: Option<Vec<(&'static str, [u64; 6])>>,
    },
    /// Clear selected channels of one current color target.
    ClearColor {
        /// Color attachment or binding index supplied by the caller.
        index: u32,
        /// Four floating-point clear components.
        color: [f32; 4],
        /// Raw color channel write mask.
        mask: u32,
    },
    /// Retain depth and stencil clear values and write masks.
    ClearDepthStencil {
        /// Depth value or depth texture key, according to the enclosing command.
        depth: f32,
        /// Raw depth write-enable argument.
        depth_write: u32,
        /// Stencil clear value.
        stencil: u32,
        /// Stencil bits selected for clearing.
        stencil_mask: u32,
    },
    /// Retain a buffer GPU address and destination texture key.
    CopyBufferToTexture {
        /// Source buffer GPU address.
        buffer: u64,
        /// Destination texture object key.
        texture: u64,
    },
    /// Retain texture keys and all eight raw copy registers.
    CopyTextureToTexture {
        /// Source texture object key.
        source: u64,
        /// Destination texture object key.
        destination: u64,
        /// All eight integer argument registers retained for host decoding.
        arguments: [u64; 8],
    },
    /// Retain command key, origin, width, and height as raw integers.
    SetViewport([u64; 5]),
    /// Retain command key, origin, width, and height as raw integers.
    SetScissor([u64; 5]),
    /// Retain the command key and two floating-point register bit patterns.
    SetDepthRange([u64; 3]),
    /// Retain non-indexed primitive, first vertex, and vertex count.
    DrawArrays {
        /// Raw primitive topology token.
        primitive: u32,
        /// First vertex in the selected stream.
        first: u32,
        /// Vertex, index, or state count supplied by the caller.
        count: u32,
    },
    /// Retain non-indexed instanced geometry arguments without general execution support.
    DrawArraysInstanced {
        /// Raw primitive topology token.
        primitive: u32,
        /// First vertex in the selected stream.
        first: u32,
        /// Vertex, index, or state count supplied by the caller.
        count: u32,
        /// First instance identifier retained from the call.
        base_instance: u32,
        /// Instance count retained from the call.
        instances: u32,
    },
    /// Retain indexed geometry and signed base-vertex adjustment.
    DrawElementsBaseVertex {
        /// Raw primitive topology token.
        primitive: u32,
        /// Raw index element format token.
        index_type: u32,
        /// Vertex, index, or state count supplied by the caller.
        count: u32,
        /// GPU address of the first index element.
        indices: u64,
        /// Signed adjustment added to each index.
        base_vertex: i32,
    },
    /// State retained without execution.
    State(StateCommand),
    /// Retain an unsupported command without assigning argument meanings.
    Raw {
        /// Function table id of the raw command.
        function: u32,
        /// Eight raw integer argument registers.
        registers: [u64; 8],
    },
}

/// Setter names paired with six retained integer argument registers.
pub type StateSettings = Vec<(&'static str, [u64; 6])>;

/// Recording fields from the host contract.
/// this representation does not assign enumeration meanings or integer widths.
#[derive(Debug, Clone, PartialEq)]
pub enum StateCommand {
    /// Retain one known or opaque state object binding.
    BindState {
        /// State or binding category identified by the function family.
        kind: &'static str,
        /// Raw address supplied by the caller; no pointer layout is inferred.
        address: u64,
        /// Snapshot of settings known through earlier setter calls. An unknown
        /// object stays unknown; its program-memory layout is never guessed.
        settings: Option<Vec<(&'static str, [u64; 6])>>,
    },
    /// Retain counted state bindings without guessing object spacing.
    BindStates {
        /// State or binding category identified by the function family.
        kind: &'static str,
        /// Vertex, index, or state count supplied by the caller.
        count: u64,
        /// Raw address supplied by the caller; no pointer layout is inferred.
        address: u64,
        /// Only the object at the supplied base address is known. No element
        /// stride or pointer-array layout is inferred for subsequent elements.
        first_settings: Option<StateSettings>,
        /// Bounded snapshots using explicit host spacing, only for opt-in drawing.
        /// This does not establish a program-memory layout.
        experiment_settings: Option<Vec<StateSettings>>,
    },
    /// Select the descriptor pool object key.
    SetDescriptorPool {
        /// True selects the sampler pool; false selects the texture pool.
        sampler: bool,
        /// Descriptor pool object key.
        pool: u64,
    },
    /// Retain an opaque sampler reference for a recorded stage and slot.
    BindSamplerReference {
        /// Raw recorded shader stage token.
        stage: u64,
        /// Color attachment or binding index supplied by the caller.
        index: u64,
        /// May be a handle or a pointer. Neither interpretation is decoded.
        reference: u64,
    },
    /// Retain synchronization object, condition, and flag values.
    FenceSync {
        /// Synchronization object key.
        sync: u64,
        /// Raw synchronization condition token.
        condition: u64,
        /// Raw synchronization flag bits.
        flags: u64,
    },
    /// Retain an opaque culling-data save range.
    SaveZCullData {
        /// Raw address supplied by the caller; no pointer layout is inferred.
        address: u64,
        /// Recorded range length in bytes.
        size: u64,
    },
    /// Retain an opaque culling-data restore range.
    RestoreZCullData {
        /// Raw address supplied by the caller; no pointer layout is inferred.
        address: u64,
        /// Recorded range length in bytes.
        size: u64,
    },
    /// Retain a face-specific stencil setter argument.
    Stencil {
        /// Stencil setter suffix identified by the recording handler.
        setting: &'static str,
        /// Raw face-selection token.
        faces: u64,
        /// Raw retained value for the enclosing command.
        value: u64,
    },
    /// Opt-in float-order hypothesis. See
    PolygonOffset([u64; 3]),
    /// Retain the raw barrier mask.
    Barrier(u64),
    /// Retain the raw tiled-cache action mask.
    TiledCacheAction(u64),
    /// Retain a uniform-buffer GPU address and byte size.
    BindUniformBuffer {
        /// Raw recorded shader stage token.
        stage: u64,
        /// Color attachment or binding index supplied by the caller.
        index: u64,
        /// Raw address supplied by the caller; no pointer layout is inferred.
        address: u64,
        /// Recorded range length in bytes.
        size: u64,
    },
    /// Retain a vertex stream GPU address and byte size.
    BindVertexBuffer {
        /// Color attachment or binding index supplied by the caller.
        index: u64,
        /// Raw address supplied by the caller; no pointer layout is inferred.
        address: u64,
        /// Recorded range length in bytes.
        size: u64,
    },
    /// Retain a texture or sampler handle binding.
    BindHandle {
        /// State or binding category identified by the function family.
        kind: &'static str,
        /// Raw recorded shader stage token.
        stage: u64,
        /// Color attachment or binding index supplied by the caller.
        index: u64,
        /// Raw texture or sampler handle.
        handle: u64,
    },
    /// Retain the GPU address, byte size, and clear value.
    ClearBuffer {
        /// Raw address supplied by the caller; no pointer layout is inferred.
        address: u64,
        /// Recorded range length in bytes.
        size: u64,
        /// Raw retained value for the enclosing command.
        value: u64,
    },
    /// Retain three raw workgroup counts without general command execution.
    DispatchCompute([u64; 3]),
}

/// Retained CPU image with tightly packed four-byte base-level texels.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextureImage {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Depth in texels.
    pub depth: u32,
    /// Opaque base-level four-byte texels in tightly packed order.
    pub pixels: Vec<u8>,
}

/// Shader registration values retained without decoding private layouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderRecord {
    /// Program address of the shader record.
    pub record_address: u64,
    /// The two GPU addresses retained from the record.
    pub gpu_addresses: [u64; 2],
    /// Eight words retained from the record for bounded observation.
    pub raw_words: [u64; 8],
}

/// Owned translation result, request, or failure for one registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderTranslation {
    /// An owned result, so obsolete completion cannot replace a newer program.
    Cached(crate::startup_cache::TranslationRequest<crate::startup_cache::TranslatedShader>),
    /// Owned bytes awaiting bounded queue space. Draws retry without guest reads.
    Deferred(Vec<u8>, crate::startup_cache::TranslationContext),
    /// Completed translated SPIR-V module.
    Spirv(Vec<u32>),
    /// Translation failed with this diagnostic.
    Error(String),
}

/// What swxbx knows about one object. Field meanings follow the signature
/// tables; unknown enumerations are kept as the integers the program passed.
#[derive(Debug, Clone, Default)]
pub enum Object {
    /// No object state has been established at this key.
    #[default]
    Unknown,
    /// Device builder settings retained from setter calls.
    DeviceBuilder {
        /// Raw flags retained from the caller; no enum meaning is implied.
        flags: u64,
    },
    /// Initialized device state and raw mode settings.
    Device {
        /// Raw flags retained from the caller; no enum meaning is implied.
        flags: u64,
        /// Raw depth-mode token.
        depth_mode: u64,
        /// Raw window-origin token.
        window_origin_mode: u64,
    },
    /// Queue builder settings retained from setter calls.
    QueueBuilder {
        /// Device object key supplied by the host.
        device: u64,
        /// Requested control-memory size in bytes.
        control_memory_size: u64,
        /// Raw command flush threshold.
        command_flush_threshold: u64,
    },
    /// Initialized queue and its device key.
    Queue {
        /// Device object key supplied by the host.
        device: u64,
    },
    /// Window builder state with native handle and texture keys.
    WindowBuilder {
        /// Device object key supplied by the host.
        device: u64,
        /// Opaque native window handle supplied by the host.
        native_window: u64,
        /// Texture object keys attached to the window.
        textures: Vec<u64>,
    },
    /// Presentation state for an initialized window.
    Window {
        /// Device object key supplied by the host.
        device: u64,
        /// Texture object keys attached to the window.
        textures: Vec<u64>,
        /// Zero requests unpaced presentation; positive values request FIFO.
        present_interval: u32,
        /// Index of the texture the next acquire hands out.
        next_texture: u32,
    },
    /// Synchronization object and its device key.
    Sync {
        /// Device object key supplied by the host.
        device: u64,
    },
    /// Event storage selected by the builder.
    EventBuilder {
        /// Memory-pool object key.
        pool: u64,
        /// Byte offset within the containing storage.
        offset: u64,
    },
    /// Event value and its program-memory storage.
    Event {
        /// The retained event value.
        value: u32,
        /// Program address of the event's storage word, which the program
        /// may read directly instead of calling EventGetValue.
        storage: u64,
    },
    /// Pool builder settings, including host storage range.
    MemoryPoolBuilder {
        /// Device object key supplied by the host.
        device: u64,
        /// Raw flags retained from the caller; no enum meaning is implied.
        flags: u64,
        /// Base address of host-managed program storage.
        storage: u64,
        /// Length of the storage range in bytes.
        size: u64,
    },
    /// Registered pool and its assigned or callback-provided GPU address.
    MemoryPool {
        /// Device object key supplied by the host.
        device: u64,
        /// Raw flags retained from the caller; no enum meaning is implied.
        flags: u64,
        /// Base address of host-managed program storage.
        storage: u64,
        /// Length of the storage range in bytes.
        size: u64,
        /// Assigned guest GPU base, if allocation succeeded.
        gpu_address: Option<u64>,
        /// Host callback return used only to resolve diagnostic reads.
        observed_gpu_address: Option<u64>,
    },
    /// Texture description retained from builder calls.
    TextureBuilder(TextureDescription),
    /// Texture description and shared optional CPU image.
    Texture {
        /// Snapshot of builder settings used to initialize the resource.
        description: TextureDescription,
        /// Shared CPU image storage, populated by supported execution paths.
        image: Arc<Mutex<Option<TextureImage>>>,
    },
    /// Recorded mip and layer selection.
    TextureView {
        /// First mip level selected by the view.
        base_level: u32,
        /// Number of mip levels, including the base level.
        levels: u32,
        /// First array layer selected by the view.
        base_layer: u32,
        /// Number of array layers; cube faces count as layers.
        layers: u32,
    },
    /// Registered texture descriptors keyed by registration id.
    TexturePool {
        /// Memory-pool object key backing descriptor storage.
        memory: u64,
        /// Byte offset within the containing storage.
        offset: u64,
        /// Number of elements in this range.
        count: u64,
        /// Objects registered through known calls; descriptor bytes are not guessed.
        registered: HashMap<u32, (u64, u64)>,
    },
    /// Registered sampler descriptors keyed by registration id.
    SamplerPool {
        /// Memory-pool object key backing descriptor storage.
        memory: u64,
        /// Byte offset within the containing storage.
        offset: u64,
        /// Number of elements in this range.
        count: u64,
        /// Objects registered through known calls; descriptor bytes are not guessed.
        registered: HashMap<u32, u64>,
    },
    /// Sampler settings retained from builder calls.
    SamplerBuilder(SamplerDescription),
    /// Initialized sampler settings.
    Sampler(SamplerDescription),
    /// Shader records and owned translation results for a program.
    Program {
        /// Device object key supplied by the host.
        device: u64,
        /// Shader records retained during registration.
        shader_records: Vec<ShaderRecord>,
        /// Translation requests or results in registration order.
        shader_translations: Vec<ShaderTranslation>,
    },
    /// A state object. Its fields are not interpreted yet; the program's
    /// calls are kept as (function, arguments) so they are not lost.
    State {
        /// State object category identified by setter calls.
        kind: &'static str,
        /// Setter names and retained raw arguments for known state objects.
        settings: Vec<(&'static str, [u64; 6])>,
    },
    /// Current recording and saved command lists keyed by returned handles.
    CommandBuffer {
        /// Device object key supplied by the host.
        device: u64,
        /// Registered command-memory tuples of pool, offset, and byte size.
        command_memory: Vec<(u64, u64, u64)>,
        /// Registered control-memory address and byte size pairs.
        control_memory: Vec<(u64, u64)>,
    },
}

impl PartialEq for Object {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Object::Texture { description: a, .. }, Object::Texture { description: b, .. }) => {
                a == b
            }
            (a, b) => format!("{a:?}") == format!("{b:?}"),
        }
    }
}

/// What the program told a texture builder, carried over to the texture.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextureDescription {
    /// Device object key.
    pub device: u64,
    /// Raw builder flags.
    pub flags: u64,
    /// Raw builder target token.
    pub target: u64,
    /// Raw builder format token.
    pub format: u64,
    /// Raw requested mip count.
    pub levels: u64,
    /// Width in texels.
    pub width: u64,
    /// Height in texels.
    pub height: u64,
    /// Depth in texels.
    pub depth: u64,
    /// Requested row stride in bytes; zero selects tightly packed storage.
    pub stride: u64,
    /// Four raw component selection tokens.
    pub swizzle: [u64; 4],
    /// Raw depth/stencil interpretation token.
    pub depth_stencil_mode: u64,
    /// Memory-pool object key backing storage.
    pub pool: u64,
    /// Byte offset within the memory pool.
    pub pool_offset: u64,
}

/// Sampler builder values carried into an initialized sampler.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplerDescription {
    /// Device object key.
    pub device: u64,
    /// Raw minification filter token.
    pub min_filter: u64,
    /// Raw magnification filter token.
    pub mag_filter: u64,
    /// Raw wrap tokens for the three texture axes.
    pub wrap: [u64; 3],
    /// Requested anisotropy as retained from the floating-point register.
    pub max_anisotropy: f32,
    /// Raw comparison-enable token.
    pub compare_mode: u64,
    /// Raw comparison-function token.
    pub compare_func: u64,
    /// Four retained floating-point border components.
    pub border_color: [f32; 4],
    /// Retained floating-point LOD bias.
    pub lod_bias: f32,
    /// Retained minimum and maximum LOD.
    pub lod_clamp: [f32; 2],
}

/// Thread-safe object table keyed by program object address.
#[derive(Default, Clone)]
pub struct Objects {
    map: Arc<Mutex<HashMap<u64, Object>>>,
    recordings: Arc<super::recordings::Recordings>,
    gpu_pools: Arc<Mutex<Vec<CachedGpuPool>>>,
}

#[derive(Clone, Copy)]
struct CachedGpuPool {
    pool: u64,
    storage: u64,
    size: u64,
    assigned: u64,
    observed: Option<u64>,
}

impl CachedGpuPool {
    fn base(self, alias: bool) -> Option<u64> {
        if alias {
            self.observed
        } else {
            Some(self.assigned)
        }
    }
}

/// Resolved GPU address and its corresponding host-managed storage range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuAddress {
    /// Memory-pool object key.
    pub pool: u64,
    /// Byte offset within the containing storage.
    pub offset: u64,
    /// Bytes remaining from this offset to the end of the pool.
    pub remaining: u64,
    /// Resolved address in host-managed program memory.
    pub program_address: u64,
}

/// Failure to map a GPU address into registered pool storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuAddressError {
    /// No registered pool contains the address.
    UnknownAddress,
    /// The address cannot resolve to a valid byte within the pool.
    OutsidePool {
        /// Pool object key whose range rejected the address.
        pool: u64,
    },
}

impl Objects {
    /// Construct an empty object table.
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<u64, Object>> {
        self.map
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn refresh_gpu_pools(&self, objects: &HashMap<u64, Object>) {
        let mut pools = self.gpu_pools.lock().unwrap_or_else(|p| p.into_inner());
        pools.clear();
        pools.extend(objects.iter().filter_map(|(&pool, object)| {
            let Object::MemoryPool {
                storage,
                size,
                gpu_address,
                observed_gpu_address,
                ..
            } = object
            else {
                return None;
            };
            let assigned = gpu_address.or(*observed_gpu_address).unwrap_or(*storage);
            Some(CachedGpuPool {
                pool,
                storage: *storage,
                size: *size,
                assigned,
                observed: observed_gpu_address.filter(|observed| *observed != assigned),
            })
        }));
    }

    /// Replace whatever was at `address` with `object`. A builder's
    /// SetDefaults call and every Initialize call start an object afresh.
    pub fn put(&self, address: u64, object: Object) {
        let mut map = self.lock();
        let changes_gpu_pool = matches!(object, Object::MemoryPool { .. })
            || matches!(map.get(&address), Some(Object::MemoryPool { .. }));
        if matches!(object, Object::CommandBuffer { .. }) {
            self.recordings.initialize(address);
        } else {
            self.recordings.finalize(address);
        }
        map.insert(address, object);
        if changes_gpu_pool {
            self.refresh_gpu_pools(&map);
        }
    }

    /// Return a cloned object record, or None if the key is absent.
    pub fn get(&self, address: u64) -> Option<Object> {
        self.lock().get(&address).cloned()
    }

    /// Remove and return an object record, or None if the key is absent.
    pub fn remove(&self, address: u64) -> Option<Object> {
        let mut map = self.lock();
        self.recordings.finalize(address);
        let removed = map.remove(&address);
        if matches!(&removed, Some(Object::MemoryPool { .. })) {
            self.refresh_gpu_pools(&map);
        }
        removed
    }

    /// Change the object at `address` in place. Returns false when there is
    /// no object there, which means the program used an object it never set
    /// up through the API, or one swxbx did not record.
    pub fn update(&self, address: u64, change: impl FnOnce(&mut Object)) -> bool {
        let mut map = self.lock();
        let (updated, changes_gpu_pool) = match map.get_mut(&address) {
            Some(object) => {
                let was_gpu_pool = matches!(object, Object::MemoryPool { .. });
                change(object);
                (
                    true,
                    was_gpu_pool || matches!(object, Object::MemoryPool { .. }),
                )
            }
            None => (false, false),
        };
        if changes_gpu_pool {
            self.refresh_gpu_pools(&map);
        }
        updated
    }

    /// Number of retained object records.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether the object table contains no records.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Resolve assigned or callback-provided GPU ranges to CPU storage. Callback addresses
    /// never replace an assigned Vulkan base.
    pub fn resolve_gpu_address(&self, address: u64) -> Result<GpuAddress, GpuAddressError> {
        let pools = self.gpu_pools.lock().unwrap_or_else(|p| p.into_inner());
        let mut matched_base = None;
        // Assigned ranges have precedence over the callback-provided alias.
        for alias in [false, true] {
            for cached in pools.iter().copied() {
                let Some(base) = cached.base(alias) else {
                    continue;
                };
                let Some(offset) = address.checked_sub(base) else {
                    continue;
                };
                if offset < cached.size {
                    let program_address = cached
                        .storage
                        .checked_add(offset)
                        .ok_or(GpuAddressError::UnknownAddress)?;
                    return Ok(GpuAddress {
                        pool: cached.pool,
                        offset,
                        remaining: cached.size - offset,
                        program_address,
                    });
                }
                matched_base = Some(cached.pool);
            }
        }
        Err(
            matched_base.map_or(GpuAddressError::UnknownAddress, |pool| {
                GpuAddressError::OutsidePool { pool }
            }),
        )
    }

    pub(crate) fn recycle_recording(&self, handle: u64, consumed: Vec<RecordedCommand>) {
        self.recordings.recycle(handle, consumed);
    }

    /// Every texture some window presents.
    pub fn window_textures(&self) -> std::collections::HashSet<u64> {
        self.lock()
            .values()
            .filter_map(|object| match object {
                Object::Window { textures, .. } => Some(textures.iter().copied()),
                _ => None,
            })
            .flatten()
            .collect()
    }

    /// Start a recording without touching the shared object table.
    pub fn begin_recording(&self, address: u64) -> bool {
        self.recordings.begin(address)
    }

    /// Append using an atomic per-buffer ownership transfer.
    pub fn record_command(&self, address: u64, command: RecordedCommand) -> bool {
        self.recordings.push(address, command)
    }

    /// Publish an owned command list and return its one-shot handle.
    pub fn end_recording(&self, address: u64) -> Option<u64> {
        self.recordings.end(address)
    }

    /// Claim a completed recording once, independently of other buffers.
    pub fn recording(&self, handle: u64) -> Option<Vec<RecordedCommand>> {
        self.recordings.take(handle)
    }

    #[cfg(test)]
    pub(crate) fn recording_len(&self, handle: u64) -> Option<usize> {
        self.recordings.len(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_registered_pool_ranges() {
        let objects = Objects::new();
        objects.put(
            1,
            Object::MemoryPool {
                device: 0,
                flags: 0,
                storage: 0x10_0000,
                size: 0x100,
                gpu_address: None,
                observed_gpu_address: None,
            },
        );
        assert_eq!(
            objects.resolve_gpu_address(0x10_0020),
            Ok(GpuAddress {
                pool: 1,
                offset: 0x20,
                remaining: 0xe0,
                program_address: 0x10_0020,
            })
        );
        assert_eq!(
            objects.resolve_gpu_address(0x10_0100),
            Err(GpuAddressError::OutsidePool { pool: 1 })
        );
        assert_eq!(
            objects.resolve_gpu_address(0x0f_0000),
            Err(GpuAddressError::UnknownAddress)
        );
    }
}
