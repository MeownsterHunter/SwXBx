//! Shared graphics subsets and bounded compilation.
use super::{
    graphics::{DrawPipelineState, GraphicsPipeline, Key, PrimitiveTopology, VertexInput},
    graphics_modules::ShaderModules,
    pipeline_workers::AsyncPipelines,
    pipelines::{CacheStats, DriverCache, PipelineRequest, PipelineStatus, RequestError},
};
use ash::vk;
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub(super) enum Part {
    Input,
    Raster,
    Fragment,
    Output,
}
impl Part {
    pub(super) fn flags(self) -> vk::GraphicsPipelineLibraryFlagsEXT {
        match self {
            Self::Input => vk::GraphicsPipelineLibraryFlagsEXT::VERTEX_INPUT_INTERFACE,
            Self::Raster => vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS,
            Self::Fragment => vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER,
            Self::Output => vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_OUTPUT_INTERFACE,
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct PartKey {
    part: Part,
    shader: Vec<u32>,
    input: Option<VertexInput>,
    topology: Option<PrimitiveTopology>,
    state: DrawPipelineState,
    layout: Vec<(u32, u32, i32)>,
    storage: bool,
}
impl PartKey {
    fn new(key: &Key, part: Part) -> Result<Self, String> {
        use super::uniforms::{self, UniformStage};
        let mut result = Self {
            part,
            shader: Vec::new(),
            input: None,
            topology: None,
            state: DrawPipelineState::default(),
            layout: Vec::new(),
            storage: false,
        };
        match part {
            Part::Input => {
                result.input = Some(key.input.clone());
                result.topology = Some(key.topology);
                result.state.instance_bindings = key.state.instance_bindings;
            }
            Part::Raster | Part::Fragment => {
                result.storage = key.storage;
                result.state.color_count = key.state.color_count.max(1);
                result.state.attachment = key.state.attachment;
                for (words, stage) in [
                    (&key.vertex, UniformStage::Vertex),
                    (&key.fragment, UniformStage::Fragment),
                ] {
                    result
                        .layout
                        .extend(uniforms::banks(words, stage)?.into_iter().map(|b| {
                            (
                                b.stage.set(),
                                b.bank,
                                if key.storage {
                                    vk::DescriptorType::STORAGE_BUFFER
                                } else {
                                    vk::DescriptorType::UNIFORM_BUFFER
                                }
                                .as_raw(),
                            )
                        }));
                    result.layout.extend(
                        super::textures::bindings(words, stage)?
                            .into_iter()
                            .map(|b| (b.stage_set(), b.lowered(), b.ty.as_raw())),
                    );
                }
                result.layout.sort_unstable();
                if part == Part::Raster {
                    result.state.subgroup_size_32[0] = key.state.subgroup_size_32[0];
                    result.shader = key.vertex.clone();
                    result.state.cull = key.state.cull;
                    result.state.polygon = key.state.polygon;
                    result.state.clockwise = key.state.clockwise;
                    result.state.bias = key.state.bias;
                } else {
                    result.state.subgroup_size_32[1] = key.state.subgroup_size_32[1];
                    result.shader = key.fragment.clone();
                    result.state.depth_test = key.state.depth_test;
                    result.state.depth_write = key.state.depth_write;
                    result.state.depth_compare = key.state.depth_compare;
                    result.state.stencil_test = key.state.stencil_test;
                    result.state.stencil = key.state.stencil;
                }
            }
            Part::Output => {
                result.state.color_count = key.state.color_count.max(1);
                result.state.attachment = key.state.attachment;
                result.state.colors[..result.state.color_count as usize]
                    .copy_from_slice(&key.state.colors[..result.state.color_count as usize]);
            }
        }
        Ok(result)
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
enum JobKey {
    Whole(Key),
    Part(PartKey),
    Link(Key),
}
#[derive(Clone)]
struct Job {
    id: JobKey,
    source: Key,
    parts: Vec<Arc<GraphicsPipeline>>,
}
impl PartialEq for Job {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Job {}
impl Hash for Job {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

struct Assembly {
    jobs: Vec<Job>,
    requests: Vec<Option<PipelineRequest<GraphicsPipeline>>>,
    link: Option<PipelineRequest<GraphicsPipeline>>,
}

pub(super) struct Compiler {
    pool: AsyncPipelines<Job, GraphicsPipeline>,
    assemblies: HashMap<Key, Assembly>,
    libraries: bool,
    stats: CacheStats,
    counts: Arc<[AtomicU64; 6]>,
}
impl Compiler {
    pub(super) fn new(
        driver: &Arc<DriverCache>,
        libraries: bool,
        workers: usize,
        capacity: usize,
    ) -> Result<Self, String> {
        let driver = driver.clone();
        let counts = Arc::new(std::array::from_fn(|_| AtomicU64::new(0)));
        let worker_counts: Arc<[AtomicU64; 6]> = counts.clone();
        let modules = ShaderModules::default();
        // Job equality and hashing use only the immutable id, never retained resources.
        #[allow(clippy::mutable_key_type)]
        let mut warmed = HashMap::new();
        if libraries {
            for job in common_jobs() {
                let JobKey::Part(part) = &job.id else {
                    unreachable!()
                };
                let pipeline = driver.create(|context, cache| {
                    GraphicsPipeline::create(context, cache, &job.source, Some(part.part), &modules)
                })?;
                counts[part.part as usize].fetch_add(1, Ordering::Relaxed);
                warmed.insert(job, Arc::new(pipeline));
            }
        }
        let pool = AsyncPipelines::new(
            warmed,
            CacheStats::default(),
            workers,
            capacity,
            move |job: &Job| {
                let pipeline = match &job.id {
                    JobKey::Whole(_) => driver.create(|context, cache| {
                        GraphicsPipeline::create(context, cache, &job.source, None, &modules)
                    })?,
                    JobKey::Part(part) => driver.create(|context, cache| {
                        GraphicsPipeline::create(
                            context,
                            cache,
                            &job.source,
                            Some(part.part),
                            &modules,
                        )
                    })?,
                    JobKey::Link(_) => GraphicsPipeline::link(&job.source, &job.parts)?,
                };
                let index = match &job.id {
                    JobKey::Whole(_) => 5,
                    JobKey::Part(part) => part.part as usize,
                    JobKey::Link(_) => 4,
                };
                worker_counts[index].fetch_add(1, Ordering::Relaxed);
                // Persistence stays on workers, including startup and slow links.
                driver.persist_driver();
                Ok(Arc::new(pipeline))
            },
        )?;
        Ok(Self {
            pool,
            assemblies: HashMap::new(),
            libraries,
            stats: CacheStats::default(),
            counts,
        })
    }

    fn ensure(&mut self, key: &Key) -> Result<(), String> {
        if self.assemblies.contains_key(key) {
            self.stats.hits += 1;
            return Ok(());
        }
        let jobs = if self.libraries {
            key.validate_interfaces()?;
            [Part::Input, Part::Raster, Part::Fragment, Part::Output]
                .into_iter()
                .map(|part| {
                    Ok(Job {
                        id: JobKey::Part(PartKey::new(key, part)?),
                        source: key.clone(),
                        parts: Vec::new(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?
        } else {
            vec![Job {
                id: JobKey::Whole(key.clone()),
                source: key.clone(),
                parts: Vec::new(),
            }]
        };
        let requests = vec![None; jobs.len()];
        self.assemblies.insert(
            key.clone(),
            Assembly {
                jobs,
                requests,
                link: None,
            },
        );
        self.stats.misses += 1;
        Ok(())
    }

    fn advance(
        pool: &mut AsyncPipelines<Job, GraphicsPipeline>,
        key: &Key,
        assembly: &mut Assembly,
        deadline: Option<Instant>,
        blocking: bool,
    ) -> Result<Option<Arc<GraphicsPipeline>>, String> {
        let mut parts = Vec::new();
        // Attempt every missing part before waiting, so independent jobs can overlap.
        for (job, request) in assembly.jobs.iter().zip(&mut assembly.requests) {
            if request.is_none() {
                let result = if blocking {
                    pool.request_blocking(job.clone())
                } else {
                    pool.request(job.clone())
                };
                match result {
                    Ok(value) => *request = Some(value),
                    Err(RequestError::QueueFull) => {}
                    Err(RequestError::Stopped) => return Err("graphics compiler stopped".into()),
                }
            }
        }
        for request in &assembly.requests {
            let Some(request) = request else {
                return Ok(None);
            };
            let status = if blocking {
                request.wait()
            } else {
                deadline.map_or_else(|| request.poll(), |end| request.wait_until(end))
            };
            match status {
                PipelineStatus::Ready(part) => parts.push(part),
                PipelineStatus::Failed(error) => return Err(error),
                PipelineStatus::Queued | PipelineStatus::Compiling => return Ok(None),
            }
        }
        if parts.len() == 1 {
            return Ok(parts.pop());
        }
        if assembly.link.is_none() {
            let job = Job {
                id: JobKey::Link(key.clone()),
                source: key.clone(),
                parts,
            };
            let result = if blocking {
                pool.request_blocking(job)
            } else {
                pool.request(job)
            };
            match result {
                Ok(request) => assembly.link = Some(request),
                Err(RequestError::QueueFull) => return Ok(None),
                Err(RequestError::Stopped) => return Err("graphics compiler stopped".into()),
            }
        }
        let request = assembly.link.as_ref().unwrap();
        let status = if blocking {
            request.wait()
        } else {
            deadline.map_or_else(|| request.poll(), |end| request.wait_until(end))
        };
        match status {
            PipelineStatus::Ready(result) => Ok(Some(result)),
            PipelineStatus::Failed(error) => Err(error),
            PipelineStatus::Queued | PipelineStatus::Compiling => Ok(None),
        }
    }

    pub(super) fn request(
        &mut self,
        key: Key,
        deadline: Option<Instant>,
        blocking: bool,
    ) -> Result<Option<Arc<GraphicsPipeline>>, String> {
        #[cfg(feature = "draw-metrics")]
        let _span = crate::draw_metrics::PipelineSpan::new(0);
        self.ensure(&key)?;
        let assembly = self.assemblies.get_mut(&key).unwrap();
        Self::advance(&mut self.pool, &key, assembly, deadline, blocking)
    }

    pub(super) fn compilation_stats(&self) -> super::graphics::GraphicsCompilationStats {
        let counts = self.counts.each_ref().map(|v| v.load(Ordering::Relaxed));
        super::graphics::GraphicsCompilationStats {
            library_parts: [counts[0], counts[1], counts[2], counts[3]],
            links: counts[4],
            whole: counts[5],
        }
    }

    pub(super) fn queue(&mut self, key: Key) -> Result<bool, String> {
        self.ensure(&key)?;
        let assembly = self.assemblies.get_mut(&key).unwrap();
        Self::advance(&mut self.pool, &key, assembly, None, false)?;
        Ok(assembly.requests.iter().all(Option::is_some))
    }
    pub(super) fn pending(&mut self) -> usize {
        // Startup polling also enqueues the final link after all parts finish.
        let mut count = 0;
        for (key, assembly) in &mut self.assemblies {
            if let Ok(None) = Self::advance(&mut self.pool, key, assembly, None, false) {
                count += 1;
            }
        }
        count
    }
    pub(super) fn hit(&mut self) {
        self.stats.hits += 1;
    }
    pub(super) fn stats(&self) -> CacheStats {
        self.stats
    }
    pub(super) fn retry_failed(&mut self, key: &Key) -> bool {
        let Some(assembly) = self.assemblies.get(key) else {
            return false;
        };
        let mut failed = assembly
            .link
            .as_ref()
            .is_some_and(|r| matches!(r.poll(), PipelineStatus::Failed(_)));
        for job in &assembly.jobs {
            failed |= self.pool.retry_failed(job);
        }
        failed |= self.pool.retry_failed(&Job {
            id: JobKey::Link(key.clone()),
            source: key.clone(),
            parts: Vec::new(),
        });
        if failed {
            self.assemblies.remove(key);
        }
        failed
    }
}

fn common_jobs() -> Vec<Job> {
    use super::graphics::{VertexAttribute, VertexBinding, VertexFormat};
    let mut key = Key {
        vertex: Vec::new(),
        fragment: Vec::new(),
        input: VertexInput {
            bindings: vec![VertexBinding {
                binding: 0,
                stride: 16,
            }],
            attributes: vec![VertexAttribute {
                binding: 0,
                format: VertexFormat::Float4,
                offset: 0,
            }],
        },
        topology: PrimitiveTopology::TriangleList,
        state: DrawPipelineState::default(),
        storage: false,
    };
    let mut jobs = Vec::new();
    for (stride, offset) in [(16, 0), (32, 8)] {
        key.input.bindings[0].stride = stride;
        key.input.attributes[0].offset = offset;
        for topology in [
            PrimitiveTopology::TriangleList,
            PrimitiveTopology::TriangleStrip,
            PrimitiveTopology::TriangleFan,
        ] {
            key.topology = topology;
            jobs.push(Job {
                id: JobKey::Part(PartKey::new(&key, Part::Input).unwrap()),
                source: key.clone(),
                parts: Vec::new(),
            });
        }
    }
    for color_count in 1..=4 {
        key.state.color_count = color_count;
        for attachment in [false, true] {
            key.state.attachment = attachment;
            jobs.push(Job {
                id: JobKey::Part(PartKey::new(&key, Part::Output).unwrap()),
                source: key.clone(),
                parts: Vec::new(),
            });
        }
    }
    jobs
}

#[cfg(test)]
mod tests {
    use super::super::graphics::{VertexAttribute, VertexBinding, VertexFormat};
    use super::*;

    fn key() -> Key {
        // Minimal public type declarations sufficient for descriptor reflection.
        fn interface(model: u32, storage: u32) -> Vec<u32> {
            let mut words = vec![0x0723_0203, 0x0001_0300, 0, 100, 0];
            for (opcode, operands) in [
                (15, vec![model, 90, 0x6e69_616d, 0]),
                (22, vec![1, 32]),
                (23, vec![2, 1, 4]),
                (32, vec![3, storage, 2]),
                (59, vec![3, 4, storage]),
                (71, vec![4, 30, 0]),
            ] {
                words.push(((operands.len() as u32 + 1) << 16) | opcode);
                words.extend(operands);
            }
            words
        }
        Key::new(
            &[interface(0, 1), interface(4, 3)],
            VertexInput {
                bindings: vec![VertexBinding {
                    binding: 0,
                    stride: 16,
                }],
                attributes: vec![VertexAttribute {
                    binding: 0,
                    format: VertexFormat::Float4,
                    offset: 0,
                }],
            },
            PrimitiveTopology::TriangleList,
            DrawPipelineState::default(),
            false,
        )
        .unwrap()
    }
    fn changed_parts(a: &Key, b: &Key) -> Vec<Part> {
        [Part::Input, Part::Raster, Part::Fragment, Part::Output]
            .into_iter()
            .filter(|&part| PartKey::new(a, part).unwrap() != PartKey::new(b, part).unwrap())
            .collect()
    }
    #[test]
    fn keys_separate_shader_input_raster_depth_and_output_changes() {
        let a = key();
        let mut b = a.clone();
        b.fragment.extend([2 << 16, 42]);
        assert!(changed_parts(&a, &b) == [Part::Fragment]);
        b = a.clone();
        b.vertex.extend([2 << 16, 42]);
        assert!(changed_parts(&a, &b) == [Part::Raster]);
        b = a.clone();
        b.input.bindings[0].stride = 32;
        assert!(changed_parts(&a, &b) == [Part::Input]);
        b = a.clone();
        b.state.instance_bindings = 1;
        assert!(changed_parts(&a, &b) == [Part::Input]);
        b = a.clone();
        b.topology = PrimitiveTopology::TriangleStrip;
        assert!(changed_parts(&a, &b) == [Part::Input]);
        b = a.clone();
        b.state.cull = 1;
        assert!(changed_parts(&a, &b) == [Part::Raster]);
        b = a.clone();
        b.state.depth_test = true;
        assert!(changed_parts(&a, &b) == [Part::Fragment]);
        b = a.clone();
        b.state.colors[0].write_mask = 7;
        assert!(changed_parts(&a, &b) == [Part::Output]);
        b = a.clone();
        b.state.color_count = 1;
        assert!(changed_parts(&a, &b).is_empty());
        b = a.clone();
        b.state.attachment = true;
        assert!(changed_parts(&a, &b) == [Part::Raster, Part::Fragment, Part::Output]);
    }
}
