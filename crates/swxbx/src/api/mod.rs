//! The behaviour behind the function table.
//!
//! Each implemented function is a handler that reads its arguments from the
//! registers, acts on the objects it refers to, and writes its results back.
//! A handler returns `Status::Ok` when it handled the call; a function with
//! no handler is reported as `Status::Unimplemented`, and the host decides
//! what to do about that.
//!
//! Handlers validate every interpreted argument before changing instance
//! state. Unresolved inputs remain explicit unsupported states.

mod commands;
mod device;
#[cfg(feature = "vulkan")]
mod drawing;
mod memory;
mod objects;
mod operations;
mod program_memory;
pub(crate) mod queue;
mod recording;
mod recordings;
mod resources;
mod state;
mod state_commands;

use crate::functions::FunctionId;
use crate::instance::{Instance, Registers, Status};
#[cfg(feature = "vulkan")]
pub use objects::SamplerDescription;
pub use objects::{GpuAddress, GpuAddressError, Object, Objects, ShaderRecord, ShaderTranslation};
pub(crate) use objects::{RecordedCommand, StateCommand};

/// A function's behaviour. Arguments arrive in `registers`; results go back
/// through it.
pub type Handler = fn(&Instance, FunctionId, &mut Registers) -> Status;

/// The handler for a function name, if the function has behaviour yet.
pub fn handler(name: &str) -> Option<Handler> {
    device::handler(name)
        .or_else(|| queue::handler(name))
        .or_else(|| memory::handler(name))
        .or_else(|| resources::handler(name))
        .or_else(|| state::handler(name))
        .or_else(|| operations::handler(name))
        .or_else(|| recording::handler(name))
        .or_else(|| state_commands::handler(name))
        .or_else(|| commands::handler(name))
}

/// Writes `value` through the pointer in `address`, through the host.
fn write_u32(instance: &crate::instance::InstanceState, address: u64, value: u32) -> bool {
    instance.write_memory(address, &value.to_le_bytes())
}

/// Reads a 64-bit word from program memory.
fn read_u64(instance: &crate::instance::InstanceState, address: u64) -> Option<u64> {
    let mut bytes = [0u8; 8];
    instance
        .read_memory(address, &mut bytes)
        .then(|| u64::from_le_bytes(bytes))
}

/// Reads four floats from program memory.
fn read_f32x4(instance: &crate::instance::InstanceState, address: u64) -> Option<[f32; 4]> {
    let mut bytes = [0u8; 16];
    instance.read_memory(address, &mut bytes).then(|| {
        let mut out = [0f32; 4];
        for (index, value) in out.iter_mut().enumerate() {
            *value =
                f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().expect("4 bytes"));
        }
        out
    })
}

/// A handler that records nothing and reports success. Used for functions
/// whose effect is on the host side only (flushing, waiting)
/// or that only tune the object (set-defaults calls).
fn accept(_instance: &Instance, _function: FunctionId, registers: &mut Registers) -> Status {
    registers.x[0] = 0;
    Status::Ok
}

/// A handler for the `...Initialize(object, builder)` family: records the
/// object and answers 1, the success value defined by the host contract.
fn succeed(registers: &mut Registers) -> Status {
    registers.x[0] = 1;
    Status::Ok
}

#[cfg(feature = "vulkan")]
pub(crate) use objects::TextureDescription;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions;

    #[test]
    fn every_handler_name_is_in_the_function_table() {
        // A handler for a name the table does not have would never run.
        for name in ALL_HANDLED
            .iter()
            .chain(state_commands::NAMES)
            .chain(state_commands::TEXTURE_NAMES)
        {
            assert!(functions::lookup(name).is_some(), "{name}");
            assert!(handler(name).is_some(), "{name}");
        }
    }

    /// Every name the handler tables claim. Kept here so the test above and
    /// the coverage figure in the README agree.
    pub(crate) const ALL_HANDLED: &[&str] = &[
        "nvnDeviceBuilderSetDefaults",
        "nvnDeviceBuilderSetFlags",
        "nvnDeviceInitialize",
        "nvnDeviceGetInteger",
        "nvnDeviceSetDepthMode",
        "nvnDeviceSetWindowOriginMode",
        "nvnDeviceRegisterFastClearColor",
        "nvnDeviceRegisterFastClearDepth",
        "nvnDeviceGetSeparateTextureHandle",
        "nvnDeviceGetSeparateSamplerHandle",
        "nvnDeviceGetImageHandle",
        "nvnQueueBuilderSetDefaults",
        "nvnQueueBuilderSetDevice",
        "nvnQueueBuilderSetControlMemorySize",
        "nvnQueueBuilderSetCommandFlushThreshold",
        "nvnQueueInitialize",
        "nvnQueueSubmitCommands",
        "nvnQueueFlush",
        "nvnQueueFinish",
        "nvnQueuePresentTexture",
        "nvnQueueWaitSync",
        "nvnSyncInitialize",
        "nvnSyncFinalize",
        "nvnSyncWait",
        "nvnEventBuilderSetStorage",
        "nvnEventInitialize",
        "nvnEventGetValue",
        "nvnEventSignal",
        "nvnWindowBuilderSetDefaults",
        "nvnWindowBuilderSetDevice",
        "nvnWindowBuilderSetNativeWindow",
        "nvnWindowBuilderSetTextures",
        "nvnWindowInitialize",
        "nvnWindowFinalize",
        "nvnWindowSetPresentInterval",
        "nvnWindowGetPresentInterval",
        "nvnWindowAcquireTexture",
        "nvnMemoryPoolBuilderSetDefaults",
        "nvnMemoryPoolBuilderSetDevice",
        "nvnMemoryPoolBuilderSetFlags",
        "nvnMemoryPoolBuilderSetStorage",
        "nvnMemoryPoolInitialize",
        "nvnMemoryPoolFinalize",
        "nvnMemoryPoolMap",
        "nvnMemoryPoolGetBufferAddress",
        "nvnMemoryPoolGetSize",
        "nvnMemoryPoolGetFlags",
        "nvnTextureBuilderSetDefaults",
        "nvnTextureBuilderSetDevice",
        "nvnTextureBuilderSetFlags",
        "nvnTextureBuilderSetTarget",
        "nvnTextureBuilderSetFormat",
        "nvnTextureBuilderSetLevels",
        "nvnTextureBuilderSetSize1D",
        "nvnTextureBuilderSetSize2D",
        "nvnTextureBuilderSetSize3D",
        "nvnTextureBuilderSetStride",
        "nvnTextureBuilderSetSwizzle",
        "nvnTextureBuilderSetDepthStencilMode",
        "nvnTextureBuilderSetStorage",
        "nvnTextureBuilderGetStorageSize",
        "nvnTextureBuilderGetStorageAlignment",
        "nvnTextureInitialize",
        "nvnTextureFinalize",
        "nvnTextureGetFlags",
        "nvnTextureGetTarget",
        "nvnTextureGetLevels",
        "nvnTextureGetSamples",
        "nvnTextureGetTextureAddress",
        "nvnTextureGetViewOffset",
        "nvnTextureGetZCullStorageSize",
        "nvnTextureViewSetDefaults",
        "nvnTextureViewSetLevels",
        "nvnTextureViewSetLayers",
        "nvnTexturePoolInitialize",
        "nvnSamplerPoolInitialize",
        "nvnTexturePoolRegisterTexture",
        "nvnTexturePoolRegisterImage",
        "nvnSamplerPoolRegisterSampler",
        "nvnSamplerBuilderSetDefaults",
        "nvnSamplerBuilderSetDevice",
        "nvnSamplerBuilderSetMinMagFilter",
        "nvnSamplerBuilderSetWrapMode",
        "nvnSamplerBuilderSetMaxAnisotropy",
        "nvnSamplerBuilderSetCompare",
        "nvnSamplerBuilderSetBorderColor",
        "nvnSamplerBuilderSetLodBias",
        "nvnSamplerBuilderSetLodClamp",
        "nvnSamplerInitialize",
        "nvnProgramInitialize",
        "nvnProgramSetShaders",
        "nvnBlendStateSetDefaults",
        "nvnBlendStateSetBlendTarget",
        "nvnBlendStateSetBlendFunc",
        "nvnBlendStateSetBlendEquation",
        "nvnChannelMaskStateSetDefaults",
        "nvnChannelMaskStateSetChannelMask",
        "nvnColorStateSetDefaults",
        "nvnColorStateSetBlendEnable",
        "nvnDepthStencilStateSetDefaults",
        "nvnDepthStencilStateSetDepthTestEnable",
        "nvnDepthStencilStateSetDepthWriteEnable",
        "nvnDepthStencilStateSetStencilTestEnable",
        "nvnDepthStencilStateSetDepthFunc",
        "nvnDepthStencilStateSetStencilFunc",
        "nvnDepthStencilStateSetStencilOp",
        "nvnPolygonStateSetDefaults",
        "nvnPolygonStateSetCullFace",
        "nvnPolygonStateSetPolygonMode",
        "nvnMultisampleStateSetDefaults",
        "nvnVertexAttribStateSetDefaults",
        "nvnVertexAttribStateSetFormat",
        "nvnVertexAttribStateSetStreamIndex",
        "nvnVertexStreamStateSetDefaults",
        "nvnVertexStreamStateSetStride",
        "nvnVertexStreamStateSetDivisor",
        "nvnCommandBufferInitialize",
        "nvnCommandBufferFinalize",
        "nvnCommandBufferAddCommandMemory",
        "nvnCommandBufferAddControlMemory",
        "nvnCommandBufferGetCommandMemorySize",
        "nvnCommandBufferSetMemoryCallback",
        "nvnCommandBufferSetMemoryCallbackData",
        "nvnCommandBufferBeginRecording",
        "nvnCommandBufferEndRecording",
        "nvnCommandBufferSetTexturePool",
        "nvnCommandBufferSetSamplerPool",
        "nvnCommandBufferSetShaderScratchMemory",
        "nvnCommandBufferSignalEvent",
        "nvnCommandBufferReportCounter",
    ];
}
