//! Shared bounded worker service. Evidence: the public contract and 0031.
pub(crate) use crate::workers::AsyncPipelines;
pub use crate::workers::RequestError;
/// Owned completion handle for an asynchronous compute or graphics compilation.
pub type PipelineRequest<P = super::pipelines::ComputePipeline> =
    crate::workers::PipelineRequest<P>;
/// Nonblocking snapshot of a pipeline request's current state.
pub type PipelineStatus<P = super::pipelines::ComputePipeline> = crate::workers::PipelineStatus<P>;
