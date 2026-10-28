//! Original synthetic feature checks. Evidence: the public contract.
use super::{
    graphics::{
        Draw, DrawPipelineState, DrawVertices, PendingDraw, PrimitiveTopology, VertexAttribute,
        VertexBinding, VertexFormat, VertexInput,
    },
    operations::Geometry,
    Backend, Context, ShaderFeatures,
};
use crate::startup_cache::TranslatedShader;
use ash::vk;
use std::sync::Arc;

fn module(instructions: &[u32]) -> Vec<u32> {
    let mut words = vec![0x0723_0203, 0x0001_0600, 0, 8, 0];
    words.extend_from_slice(instructions);
    words
}

fn supported() -> ShaderFeatures {
    ShaderFeatures {
        demote_to_helper_invocation: true,
        maximal_reconvergence: true,
        ..Default::default()
    }
}

#[test]
fn detects_demote_capability_without_extension_and_both_opcodes() {
    for instructions in [
        vec![(2 << 16) | 17, 5379],
        vec![(1 << 16) | 5380],
        vec![(3 << 16) | 5381, 1, 2],
        vec![(1 << 16), (2 << 16) | 17, 5379],
    ] {
        let words = module(&instructions);
        assert_eq!(
            ShaderFeatures::default().validate_shader(&words),
            Err("shaderDemoteToHelperInvocation is unavailable")
        );
        supported().validate_shader(&words).unwrap();
    }
}

#[test]
fn detects_maximal_reconvergence_execution_mode_without_extension() {
    for opcode in [16, 331] {
        let words = module(&[(3 << 16) | opcode, 1, 6023]);
        assert_eq!(
            ShaderFeatures::default().validate_shader(&words),
            Err("shaderMaximalReconvergence is unavailable")
        );
        supported().validate_shader(&words).unwrap();
    }
}

#[test]
fn ordinary_instructions_and_numeric_operands_do_not_require_features() {
    for instructions in [
        vec![],
        vec![(1 << 16)],
        vec![(2 << 16) | 17, 1],
        vec![(6 << 16) | 16, 1, 17, 5379, 5380, 6023],
        vec![(4 << 16) | 43, 1, 2, 5381],
    ] {
        ShaderFeatures::default()
            .validate_shader(&module(&instructions))
            .unwrap();
    }
}

#[test]
fn malformed_headers_and_instruction_lengths_are_rejected() {
    for words in [vec![], vec![0; 5], vec![0x0723_0203; 4]] {
        assert_eq!(
            supported().validate_shader(&words),
            Err("invalid SPIR-V header")
        );
    }
    for instructions in [
        vec![0],
        vec![(3 << 16) | 17, 5379],
        vec![(1 << 16) | 17],
        vec![(3 << 16) | 17, 5379, 0],
        vec![(2 << 16) | 5380, 0],
        vec![(2 << 16) | 5381, 1],
        vec![(4 << 16) | 5381, 1, 2, 3],
        vec![(2 << 16) | 16, 1],
        vec![(2 << 16) | 331, 1],
        vec![(4 << 16) | 16, 1, 6023, 0],
        vec![(3 << 16) | 16, 1],
        vec![(1 << 16), 0],
    ] {
        assert!(supported().validate_shader(&module(&instructions)).is_err());
    }
}

#[test]
fn declared_requirements_are_checked_even_without_instruction_markers() {
    let mut shader = TranslatedShader {
        key: String::new(),
        words: module(&[]),
        requires_subgroup_size_32: false,
        requires_demote_to_helper_invocation: true,
        requires_maximal_reconvergence: false,
        surfaces: Vec::new(),
        textures: Vec::new(),
    };
    assert_eq!(
        ShaderFeatures::default().validate_translated_shader(&shader),
        Err("shaderDemoteToHelperInvocation is unavailable")
    );
    supported().validate_translated_shader(&shader).unwrap();
    shader.requires_demote_to_helper_invocation = false;
    shader.requires_maximal_reconvergence = true;
    assert_eq!(
        ShaderFeatures::default().validate_translated_shader(&shader),
        Err("shaderMaximalReconvergence is unavailable")
    );
    supported().validate_translated_shader(&shader).unwrap();
    shader.requires_maximal_reconvergence = false;
    shader.words = module(&[(1 << 16) | 5380]);
    assert!(ShaderFeatures::default()
        .validate_translated_shader(&shader)
        .is_err());
    shader.words = module(&[(3 << 16) | 16, 1, 6023]);
    assert!(ShaderFeatures::default()
        .validate_translated_shader(&shader)
        .is_err());
}

fn compile(source: &str, kind: shaderc::ShaderKind) -> Vec<u32> {
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_2 as u32,
    );
    compiler
        .compile_into_spirv(source, kind, "feature-check", "main", Some(&options))
        .unwrap()
        .as_binary()
        .to_vec()
}

fn context() -> Arc<Context> {
    let context = Arc::new(Context::new().expect("offscreen device required"));
    eprintln!(
        "device features: shaderDemoteToHelperInvocation={}, shaderMaximalReconvergence={}",
        context.shader_features.demote_to_helper_invocation,
        context.shader_features.maximal_reconvergence,
    );
    context
}

#[test]
#[ignore = "requires an offscreen device and shader compiler"]
fn demoted_fragments_preserve_quad_derivatives_and_suppress_writes() {
    let context = context();
    if !context.shader_features.demote_to_helper_invocation {
        eprintln!("SKIP demotion readback: shaderDemoteToHelperInvocation is unavailable");
        return;
    }
    let vertex = compile(
        "#version 450\nlayout(location=0) in vec4 p; void main(){gl_Position=p;}",
        shaderc::ShaderKind::Vertex,
    );
    let fragment = compile(
        r#"#version 450
#extension GL_EXT_demote_to_helper_invocation : require
layout(location=0) out vec4 color;
void main() {
    if ((int(gl_FragCoord.x) & 1) == 0) demote;
    float dx = dFdxFine(gl_FragCoord.x);
    float dy = dFdyFine(gl_FragCoord.y);
    color = vec4(dx / 4.0, dy / 4.0, helperInvocationEXT() ? 0.0 : 1.0, 1.0);
}
"#,
        shaderc::ShaderKind::Fragment,
    );
    assert!(ShaderFeatures::default()
        .validate_shader(&fragment)
        .unwrap_err()
        .contains("shaderDemote"));
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
    assert!(backend.clear_color(2, [0.0; 4], 0xf).is_ok());
    let state = DrawPipelineState::default();
    let pipeline = backend
        .graphics
        .request(
            &[vertex, fragment],
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
    backend
        .queue_draw(
            &[2],
            None,
            PendingDraw {
                geometry: Geometry::default(),
                pipeline,
                state: Arc::new(Draw {
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
                }),
                vertices: DrawVertices::Arrays { first: 0 },
                count: 3,
            },
        )
        .unwrap();
    backend.finish_draws().unwrap();
    let (width, height, pixels) = backend.readback(2).unwrap();
    assert_eq!((width, height), (64, 64));
    for (i, pixel) in pixels.as_chunks::<4>().0.iter().enumerate() {
        let expected = if i % 64 % 2 == 0 {
            [0; 4]
        } else {
            [64, 64, 255, 255]
        };
        assert_eq!(pixel, &expected, "pixel {i}");
    }
}

#[test]
#[ignore = "requires an offscreen device and shader compiler"]
fn maximal_reconvergence_rejoins_divergent_compute_invocations() {
    let context = context();
    if !context.shader_features.maximal_reconvergence {
        eprintln!("SKIP reconvergence readback: shaderMaximalReconvergence is unavailable");
        return;
    }
    let words = compile(
        r#"#version 460
#extension GL_EXT_maximal_reconvergence : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
layout(local_size_x=128) in;
layout(set=0,binding=0,std430) buffer Output { uvec4 values[]; };
void main() [[maximally_reconverges]] {
    uint lane = gl_LocalInvocationIndex;
    uvec4 before = subgroupBallot(true);
    uint n = lane % 7u + 1u;
    uint sum = 0u;
    for (uint i = 1u; i <= n; i++) sum += i;
    if ((lane & 1u) == 0u) sum += 11u;
    else sum += 23u;
    uvec4 after = subgroupBallot(true);
    values[lane] = uvec4(sum, subgroupBallotBitCount(before),
        subgroupBallotBitCount(after), before == after ? 1u : 0u);
}
"#,
        shaderc::ShaderKind::Compute,
    );
    assert_eq!(
        ShaderFeatures::default().validate_shader(&words),
        Err("shaderMaximalReconvergence is unavailable")
    );
    let mut backend = Backend::from_context(context, 1.0).unwrap();
    backend.allocate_pool(1, 0, 4096).unwrap();
    let memory = backend.global_memory.as_mut().unwrap();
    memory.write_pool(1, 0, &[0; 2048]).unwrap();
    let buffer = memory.storage_buffer_info(1, 0, 2048).unwrap();
    backend
        .execution
        .dispatch(memory, &words, false, &[(0, 0, buffer)], [1, 1, 1], None)
        .unwrap();
    backend.context.wait_queue().unwrap();
    let mut bytes = [0; 2048];
    backend
        .global_memory
        .as_mut()
        .unwrap()
        .read_pool(1, 0, &mut bytes)
        .unwrap();
    for (lane, record) in bytes.as_chunks::<16>().0.iter().enumerate() {
        let values: Vec<_> = record
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let n = lane as u32 % 7 + 1;
        assert_eq!(
            values[0],
            n * (n + 1) / 2 + if lane % 2 == 0 { 11 } else { 23 }
        );
        assert!(values[1] > 0);
        assert_eq!(values[1], values[2]);
        assert_eq!(values[3], 1);
    }
}

#[test]
#[ignore = "requires an offscreen device"]
fn extension_feature_path_creates_modules_with_12_feature_negotiation() {
    let context = context();
    let gpu_validation = std::env::var("VULKAN_VALIDATION").is_ok_and(|m| m == "gpu");
    let (_, device, _, features, _, _) = Context::create_device(
        &context.instance,
        false,
        vk::API_VERSION_1_2,
        gpu_validation,
    )
    .expect("1.2 device");
    eprintln!(
        "extension features: shaderDemoteToHelperInvocation={}, shaderMaximalReconvergence={}",
        features.demote_to_helper_invocation, features.maximal_reconvergence,
    );
    for (enabled, source, kind) in [
        (
            features.demote_to_helper_invocation,
            "#version 450\n#extension GL_EXT_demote_to_helper_invocation : require\nvoid main(){demote;}",
            shaderc::ShaderKind::Fragment,
        ),
        (
            features.maximal_reconvergence,
            "#version 460\n#extension GL_EXT_maximal_reconvergence : require\nlayout(local_size_x=1) in;\nvoid main() [[maximally_reconverges]] {}",
            shaderc::ShaderKind::Compute,
        ),
    ] {
        if !enabled {
            eprintln!("SKIP extension module: requested optional feature is unavailable");
            continue;
        }
        let words = compile(source, kind);
        features.validate_shader(&words).unwrap();
        unsafe {
            let module = device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&words), None,
            ).unwrap();
            device.destroy_shader_module(module, None);
        }
    }
    unsafe { device.destroy_device(None) };
}

#[test]
#[ignore = "requires an offscreen device"]
fn module_and_pipeline_paths_reject_missing_features_before_driver_creation() {
    let mut context = Context::new().expect("offscreen device required");
    context.shader_features.demote_to_helper_invocation = false;
    context.shader_features.maximal_reconvergence = false;
    let context = Arc::new(context);
    let mut backend = Backend::from_context(context.clone(), 1.0).unwrap();
    let vertex = compile(
        "#version 450\nvoid main(){gl_Position=vec4(0.0);}",
        shaderc::ShaderKind::Vertex,
    );
    for (source, expected) in [
        ("#version 450\n#extension GL_EXT_demote_to_helper_invocation : require\nlayout(location=0) out vec4 color; void main(){demote; color=vec4(1.0);}", "shaderDemoteToHelperInvocation is unavailable"),
        ("#version 460\n#extension GL_EXT_maximal_reconvergence : require\nlayout(location=0) out vec4 color; void main() [[maximally_reconverges]] {color=vec4(1.0);}", "shaderMaximalReconvergence is unavailable"),
    ] {
        let fragment = compile(source, shaderc::ShaderKind::Fragment);
        let error = backend.graphics.request(
            &[vertex.clone(), fragment], VertexInput { bindings: vec![], attributes: vec![] },
            PrimitiveTopology::TriangleList, DrawPipelineState::default(), false,
        ).err().expect("graphics creation must reject the missing feature");
        assert_eq!(error, expected);
    }
    for (instructions, demote, maximal, expected) in [
        (
            vec![(2 << 16) | 17, 5379],
            true,
            false,
            "shaderDemoteToHelperInvocation is unavailable",
        ),
        (
            vec![(3 << 16) | 16, 1, 6023],
            false,
            true,
            "shaderMaximalReconvergence is unavailable",
        ),
    ] {
        let words = module(&instructions);
        assert_eq!(
            context.create_global_shader_module(&words).unwrap_err(),
            expected
        );
        let error = super::pipelines::ComputePipeline::create(
            &context,
            vk::PipelineCache::null(),
            &words,
            false,
        )
        .err()
        .expect("must reject before reflection or layout creation");
        assert_eq!(error, expected);
        let shader = TranslatedShader {
            key: String::new(),
            words: module(&[]),
            requires_subgroup_size_32: false,
            requires_demote_to_helper_invocation: demote,
            requires_maximal_reconvergence: maximal,
            surfaces: Vec::new(),
            textures: Vec::new(),
        };
        let recipe = crate::startup_cache::PipelineRecipe {
            keys: vec![],
            input: VertexInput {
                bindings: vec![],
                attributes: vec![],
            },
            topology: PrimitiveTopology::TriangleList,
            state: Default::default(),
            storage: false,
        };
        assert_eq!(
            backend
                .graphics
                .queue_startup(&[&shader], &recipe)
                .unwrap_err(),
            expected
        );
    }
}

#[test]
#[ignore = "requires an offscreen device"]
fn production_module_path_checks_demote_with_a_feature_mask() {
    let mut context = Context::new().expect("offscreen device required");
    assert!(
        context.shader_features.demote_to_helper_invocation,
        "device must support demotion for this experiment"
    );
    let words = compile(
        "#version 450\n#extension GL_EXT_demote_to_helper_invocation : require\nlayout(location=0) out vec4 color; void main(){if(gl_FragCoord.x < 4.0) demote; color=vec4(1.0);}",
        shaderc::ShaderKind::Fragment,
    );
    let shader = TranslatedShader {
        key: String::new(),
        words,
        requires_subgroup_size_32: false,
        requires_demote_to_helper_invocation: true,
        requires_maximal_reconvergence: false,
        surfaces: Vec::new(),
        textures: Vec::new(),
    };
    // Mask the advertised feature on a real enabled device. Rejection must happen
    // in the production path before reaching the driver.
    context.shader_features.demote_to_helper_invocation = false;
    let expected = "shaderDemoteToHelperInvocation is unavailable";
    assert_eq!(
        context
            .shader_features
            .validate_translated_shader(&shader)
            .unwrap_err(),
        expected
    );
    assert_eq!(
        context
            .create_global_shader_module(&shader.words)
            .unwrap_err(),
        expected
    );
    context.shader_features.demote_to_helper_invocation = true;
    context
        .shader_features
        .validate_translated_shader(&shader)
        .unwrap();
    let module = context.create_global_shader_module(&shader.words).unwrap();
    unsafe {
        context.device.destroy_shader_module(module, None);
    }
}
