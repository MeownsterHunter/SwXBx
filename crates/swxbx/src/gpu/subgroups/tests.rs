use super::*;
use crate::{
    gpu::{
        graphics::{
            Draw, DrawPipelineState, DrawVertices, PendingDraw, PrimitiveTopology, VertexAttribute,
            VertexBinding, VertexFormat, VertexInput,
        },
        operations::Geometry,
        Backend, Context,
    },
    startup_cache::{PipelineRecipe, TranslatedShader},
};
use std::{
    fs,
    process::Command,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

fn supported() -> SubgroupFeatures {
    SubgroupFeatures {
        size_control: true,
        compute_full_subgroups: true,
        min_size: 32,
        max_size: 64,
        required_stages: (vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT).as_raw(),
        max_compute_workgroup_subgroups: 2,
    }
}

#[test]
fn stage_checks_reject_missing_controls_ranges_and_full_subgroups() {
    let features = supported();
    for stage in [
        vk::ShaderStageFlags::COMPUTE,
        vk::ShaderStageFlags::FRAGMENT,
    ] {
        features.validate(stage).unwrap();
        for unsupported in [
            SubgroupFeatures {
                size_control: false,
                ..features
            },
            SubgroupFeatures {
                min_size: 64,
                ..features
            },
            SubgroupFeatures {
                max_size: 16,
                ..features
            },
            SubgroupFeatures {
                required_stages: 0,
                ..features
            },
        ] {
            assert!(unsupported
                .validate(stage)
                .unwrap_err()
                .contains("requires subgroup size 32"));
        }
    }
    assert!(features
        .validate(vk::ShaderStageFlags::VERTEX)
        .unwrap_err()
        .starts_with("vertex"));
    let partial = SubgroupFeatures {
        compute_full_subgroups: false,
        ..features
    };
    partial.validate(vk::ShaderStageFlags::FRAGMENT).unwrap();
    assert!(partial
        .validate(vk::ShaderStageFlags::COMPUTE)
        .unwrap_err()
        .contains("full 32-lane"));
}

#[test]
fn full_subgroups_validate_local_size_and_workgroup_limits() {
    let words = |x, y| {
        vec![
            0x0723_0203,
            0x0001_0300,
            0,
            10,
            0,
            (6 << 16) | 16,
            1,
            17,
            x,
            y,
            1,
        ]
    };
    supported().validate_compute(&words(32, 2)).unwrap();
    assert!(supported()
        .validate_compute(&words(16, 2))
        .unwrap_err()
        .contains("multiple of 32"));
    assert!(supported().validate_compute(&words(32, 0)).is_err());
    assert!(supported()
        .validate_compute(&words(32, 3))
        .unwrap_err()
        .contains("device limit"));
    assert!(supported()
        .validate_compute(&words(32, 2)[..5])
        .unwrap_err()
        .contains("literal"));
}

struct Directory(std::path::PathBuf);
impl Directory {
    fn new() -> Self {
        static ID: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "subgroup-proof-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn compile(&self, stage: &str, source: &str) -> Vec<u32> {
        let input = self.0.join(format!("shader.{stage}"));
        let output = self.0.join(format!("shader.{stage}.spv"));
        fs::write(&input, source).unwrap();
        let result = Command::new("glslangValidator")
            .args(["-V", "--target-env", "vulkan1.2", "-o"])
            .arg(&output)
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(&output)
            .status()
            .unwrap()
            .success());
        fs::read(output)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

const OPERATIONS: &str = r#"
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
uvec4 results() {
    uint lane = gl_SubgroupInvocationID;
    uint sum = 0;
    for (uint source = 0; source < 32; source++)
        sum += subgroupShuffle(lane + 1, source);
    bool votes = subgroupAll(lane < 32) && subgroupAny(lane == 31);
    uvec4 ballot = subgroupBallot(true);
    uint bit = 1u << (lane & 31u);
    bool masks = gl_SubgroupEqMask == uvec4(bit, 0, 0, 0)
        && gl_SubgroupLtMask == uvec4(bit - 1u, 0, 0, 0)
        && gl_SubgroupLeMask == uvec4(bit | (bit - 1u), 0, 0, 0)
        && gl_SubgroupGeMask == uvec4(~(bit - 1u), 0, 0, 0)
        && gl_SubgroupGtMask == uvec4(~(bit | (bit - 1u)), 0, 0, 0);
    bool correct = votes && masks && ballot == uvec4(0xffffffffu, 0, 0, 0);
    return uvec4(lane + 1u, correct ? 255u : 0u, sum, subgroupBallotBitCount(ballot));
}
"#;

#[test]
#[ignore = "requires a GPU with a default size of 64 and fixed compute size 32; never skips"]
fn compute_votes_ballots_shuffles_and_masks_read_back_exactly() {
    let directory = Directory::new();
    let words = directory.compile("comp", &format!("#version 450\n{OPERATIONS}\nlayout(local_size_x=64) in; layout(set=0,binding=0,std430) buffer Output {{uvec4 values[];}}; void main(){{values[gl_LocalInvocationIndex]=results();}}"));
    let mut backend = Backend::new(1.0).expect("GPU backend");
    backend.allocate_pool(1, 0, 4096).unwrap();
    let mut outputs = Vec::new();
    for required in [false, true, false, true] {
        let memory = backend.global_memory.as_mut().unwrap();
        memory.write_pool(1, 0, &[0; 1024]).unwrap();
        let buffer = memory.storage_buffer_info(1, 0, 1024).unwrap();
        backend
            .execution
            .dispatch(memory, &words, required, &[(0, 0, buffer)], [1, 1, 1], None)
            .unwrap();
        backend.context.wait_queue().unwrap();
        let mut bytes = vec![0; 1024];
        backend
            .global_memory
            .as_mut()
            .unwrap()
            .read_pool(1, 0, &mut bytes)
            .unwrap();
        let values: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let expected: Vec<_> = (0..64).flat_map(|i| [i % 32 + 1, 255, 528, 32]).collect();
        if required {
            assert_eq!(values, expected);
        } else {
            assert_ne!(
                values, expected,
                "negative control must expose the wide subgroup"
            );
        }
        outputs.push(values);
    }
    assert_eq!(outputs[0], outputs[2]);
    assert_eq!(outputs[1], outputs[3]);
}

fn input() -> VertexInput {
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
    }
}

fn exact_pixels(bytes: &[u8]) -> bool {
    let mut lanes = [0; 32];
    for pixel in bytes.as_chunks::<4>().0 {
        if !(1..=32).contains(&pixel[0]) || pixel[1..] != [255, 16, 32] {
            return false;
        }
        lanes[usize::from(pixel[0] - 1)] += 1;
    }
    lanes == [128; 32]
}

#[test]
#[ignore = "requires a GPU with a default size of 64 and fixed fragment size 32; never skips"]
fn fragment_votes_ballots_shuffles_and_masks_read_back_exactly() {
    let directory = Directory::new();
    let vertex = directory.compile(
        "vert",
        "#version 450\nlayout(location=0) in vec4 p; void main(){gl_Position=p;}",
    );
    let fragment = directory.compile("frag", &format!("#version 450\n{OPERATIONS}\nlayout(location=0) out vec4 color; void main(){{uvec4 v=results(); color=vec4(v.x,v.y,v.z&255u,v.w)/255.0;}}"));
    let context = Arc::new(Context::new().expect("GPU context"));
    let mut backend = Backend::from_context(context, 1.0).unwrap();
    backend.allocate_pool(1, 0, 4096).unwrap();
    let vertices: Vec<_> = [-1_f32, -1., 0., 1., 3., -1., 0., 1., -1., 3., 0., 1.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    backend
        .global_memory
        .as_mut()
        .unwrap()
        .write_pool(1, 0, &vertices)
        .unwrap();
    assert!(backend.ensure(2, 64, 64, false));
    let stages = [vertex, fragment];
    let mut outputs = Vec::new();
    for required in [false, true, false, true] {
        let state = DrawPipelineState {
            subgroup_size_32: [false, required],
            ..Default::default()
        };
        if required {
            let translated: Vec<_> = stages
                .iter()
                .enumerate()
                .map(|(i, words)| TranslatedShader {
                    key: String::new(),
                    words: words.clone(),
                    requires_subgroup_size_32: i == 1,
                    requires_demote_to_helper_invocation: false,
                    requires_maximal_reconvergence: false,
                    surfaces: Vec::new(),
                })
                .collect();
            let recipe = PipelineRecipe {
                keys: vec![],
                input: input(),
                topology: PrimitiveTopology::TriangleList,
                state: Default::default(),
                storage: false,
            };
            backend
                .graphics
                .queue_startup(&translated.iter().collect::<Vec<_>>(), &recipe)
                .unwrap();
        }
        let pipeline = backend
            .graphics
            .request(
                &stages,
                input(),
                PrimitiveTopology::TriangleList,
                state,
                false,
            )
            .unwrap()
            .unwrap();
        let descriptors = backend.graphics.descriptors(&pipeline, &[], &[]).unwrap();
        let (buffer, offset) = backend
            .global_memory
            .as_ref()
            .unwrap()
            .buffer_region(1, 0, vertices.len())
            .unwrap();
        let draw = Arc::new(Draw {
            storage_images: Vec::new(),
            buffers: vec![(0, buffer, offset)],
            viewport: vk::Viewport {
                x: 0.,
                y: 0.,
                width: 64.,
                height: 64.,
                min_depth: 0.,
                max_depth: 1.,
            },
            scissor: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: 64,
                    height: 64,
                },
            },
            descriptors,
        });
        backend
            .queue_draw(
                &[2],
                None,
                PendingDraw {
                    geometry: Geometry::default(),
                    pipeline,
                    state: draw,
                    vertices: DrawVertices::Arrays { first: 0 },
                    count: 3,
                },
            )
            .unwrap();
        backend.finish_draws().unwrap();
        let (_, _, pixels) = backend.readback(2).unwrap();
        if required {
            assert!(
                exact_pixels(&pixels),
                "every pixel must match the operations and every lane must occur 128 times"
            );
        } else {
            assert!(
                !exact_pixels(&pixels),
                "negative control must expose the wide subgroup"
            );
        }
        outputs.push(pixels);
    }
    assert_eq!(outputs[0], outputs[2]);
    assert_eq!(outputs[1], outputs[3]);
}
