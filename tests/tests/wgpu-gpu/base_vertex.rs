//! Faithful port of the CTS test
//! `webgpu:shader,execution,robust_access_vertex:vertex_buffer_access:*`
//! restricted to the parameterization that fails on the paravirtualized Metal
//! device in CI: `indexed=true, indirect=false, drawCallTestParameter=baseVertex`.
//!
//! The goal is to reproduce that failure in a wgpu GPU test so the necessary
//! ingredients can be narrowed down without having to run modified CTS in CI.
//!
//! The test mirrors the CTS `DrawCall` / `doTest` logic: multiple vertex buffers
//! (one instance-step-mode, the rest vertex-step-mode) with several attributes
//! each, an instanced indexed draw, and a vertex shader that flags failure
//! (drawing to the left pixel) if any attribute is outside the set of expected
//! values or if `@builtin(vertex_index)` is outside
//! `[baseVertex, baseVertex + numVertices)`. The render target is cleared green;
//! a red left pixel means a subcase failed.
//!
//! Deviation from the CTS: the CTS bakes `vertexIndexOffset` and the "0 is a
//! valid value" flag into the shader as literals. Doing so here would force a
//! distinct shader (and thus a distinct, slow Metal pipeline compile) per
//! `errorScale`, blowing the test's time budget. Instead those two values are
//! passed via a uniform, so the shader depends only on `(type, buffer_count)`.
//! This does not affect the vertex-fetch / `vertex_index` behavior under test.
//!
//! See metal_base_vertex.md for the full investigation.

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

pub fn all_tests(vec: &mut Vec<wgpu_test::GpuTestInitializer>) {
    vec.push(ROBUST_ACCESS_VERTEX_BASE_VERTEX);
    vec.push(VERTEX_ID_BASE_VERTEX_SWEEP);
}

const NUM_VERTICES: u32 = 4;
const ATTRIBUTES_PER_BUFFER: u32 = 2;
const ARBITRARY_VALUES: [i64; 4] = [990, 685, 446, 175];
const ERROR_SCALES: [u32; 6] = [0, 1, 4, 100, 10_000, 1_000_000];

#[derive(Clone, Copy)]
struct TypeInfo {
    name: &'static str,
    wgsl_type: &'static str,
    size_in_bytes: u32,
    /// Number of f32 components per attribute.
    components: u32,
    format: wgpu::VertexFormat,
    validation_func: &'static str,
}

const TYPE_INFOS: [TypeInfo; 4] = [
    TypeInfo {
        name: "float32",
        wgsl_type: "f32",
        size_in_bytes: 4,
        components: 1,
        format: wgpu::VertexFormat::Float32,
        validation_func: "return valid(v);",
    },
    TypeInfo {
        name: "float32x2",
        wgsl_type: "vec2<f32>",
        size_in_bytes: 8,
        components: 2,
        format: wgpu::VertexFormat::Float32x2,
        validation_func: "return valid(v.x) && valid(v.y);",
    },
    TypeInfo {
        name: "float32x3",
        wgsl_type: "vec3<f32>",
        size_in_bytes: 12,
        components: 3,
        format: wgpu::VertexFormat::Float32x3,
        validation_func: "return valid(v.x) && valid(v.y) && valid(v.z);",
    },
    TypeInfo {
        name: "float32x4",
        wgsl_type: "vec4<f32>",
        size_in_bytes: 16,
        components: 4,
        format: wgpu::VertexFormat::Float32x4,
        validation_func: "return (valid(v.x) && valid(v.y) && valid(v.z) && valid(v.w)) ||\n\
             (v.x == 0.0 && v.y == 0.0 && v.z == 0.0 && (v.w == 0.0 || v.w == 1.0));",
    },
];

#[derive(Clone, Copy)]
struct Subcase {
    type_info: TypeInfo,
    additional_buffers: u32,
    partial_last_number: bool,
    offset_vertex_buffer: bool,
    error_scale: u32,
}

/// A vertex shader / pipeline depends only on these; pipelines are cached to
/// avoid redundant (slow) Metal shader compiles.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    type_index: usize,
    buffer_count: u32,
}

fn generate_vertex_shader(buffer_count: u32, type_info: &TypeInfo) -> String {
    let total_attributes = buffer_count * ATTRIBUTES_PER_BUFFER;

    let mut layout = String::from("struct Attributes {\n");
    for i in 0..total_attributes {
        layout += &format!("  @location({i}) a_{i} : {},\n", type_info.wgsl_type);
    }
    layout += "};";

    let valid_body = ARBITRARY_VALUES
        .iter()
        .map(|v| format!("f == {v}.0"))
        .collect::<Vec<_>>()
        .join(" || ");

    let attributes_in_bounds = (0..total_attributes)
        .map(|i| format!("validationFunc(attributes.a_{i})"))
        .collect::<Vec<_>>()
        .join(" && ");

    format!(
        r#"
struct Params {{
  vertex_index_offset : u32,
  vertex_index_count : u32,
  allow_zero : u32,
  pad : u32,
}};
@group(0) @binding(0) var<uniform> params : Params;

{layout}

fn valid(f : f32) -> bool {{
  return {valid_body} || (params.allow_zero != 0u && f == 0.0);
}}

fn validationFunc(v : {wgsl_type}) -> bool {{
  {validation_func}
}}

@vertex fn main(
  @builtin(vertex_index) VertexIndex : u32,
  attributes : Attributes
  ) -> @builtin(position) vec4<f32> {{
  var attributesInBounds = {attributes_in_bounds};

  var indexInBoundsCountFromBaseVertex =
      (VertexIndex >= params.vertex_index_offset &&
      VertexIndex < params.vertex_index_offset + params.vertex_index_count);
  var indexInBounds = VertexIndex == 0u || indexInBoundsCountFromBaseVertex;

  var Position : vec4<f32>;
  if (attributesInBounds && indexInBounds) {{
    // Success case, move the vertex to the right of the viewport.
    Position = vec4<f32>(0.5, 0.0, 0.0, 1.0);
  }} else {{
    // Failure case, move the vertex to the left of the viewport.
    Position = vec4<f32>(-0.5, 0.0, 0.0, 1.0);
  }}
  return Position;
}}
"#,
        wgsl_type = type_info.wgsl_type,
        validation_func = type_info.validation_func,
    )
}

const FRAGMENT_SHADER: &str = r#"
@fragment fn main() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;

fn build_pipeline(
    ctx: &TestingContext,
    pipeline_layout: &wgpu::PipelineLayout,
    key: PipelineKey,
) -> wgpu::RenderPipeline {
    let type_info = &TYPE_INFOS[key.type_index];
    let buffer_count = key.buffer_count;

    // Build the vertex buffer layouts. Buffer 0 is instance step mode, the rest
    // are vertex step mode.
    let mut attribute_sets: Vec<Vec<wgpu::VertexAttribute>> = Vec::new();
    let mut location = 0;
    for _ in 0..buffer_count {
        let mut attributes = Vec::new();
        for j in 0..ATTRIBUTES_PER_BUFFER {
            attributes.push(wgpu::VertexAttribute {
                format: type_info.format,
                offset: (j * type_info.size_in_bytes) as u64,
                shader_location: location,
            });
            location += 1;
        }
        attribute_sets.push(attributes);
    }
    let buffers: Vec<Option<wgpu::VertexBufferLayout>> = (0..buffer_count as usize)
        .map(|i| {
            Some(wgpu::VertexBufferLayout {
                array_stride: (ATTRIBUTES_PER_BUFFER * type_info.size_in_bytes) as u64,
                step_mode: if i == 0 {
                    wgpu::VertexStepMode::Instance
                } else {
                    wgpu::VertexStepMode::Vertex
                },
                attributes: &attribute_sets[i],
            })
        })
        .collect();

    let vertex_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vertex"),
            source: wgpu::ShaderSource::Wgsl(
                generate_vertex_shader(buffer_count, type_info).into(),
            ),
        });
    let fragment_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fragment"),
            source: wgpu::ShaderSource::Wgsl(FRAGMENT_SHADER.into()),
        });

    ctx.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                buffers: &buffers,
                module: &vertex_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fragment_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
}

/// Records the draw for one subcase and returns the command buffer.
fn run_subcase(
    ctx: &TestingContext,
    subcase: &Subcase,
    pipeline: &wgpu::RenderPipeline,
    bind_group_layout: &wgpu::BindGroupLayout,
    color_attachment: &wgpu::TextureView,
) -> wgpu::CommandBuffer {
    let type_info = &subcase.type_info;
    let buffer_count = subcase.additional_buffers + 2;

    // A valid value is one in the buffer, or 0 for the OOB testing cases.
    let is_control =
        subcase.error_scale == 0 && !subcase.offset_vertex_buffer && !subcase.partial_last_number;
    let params = [
        subcase.error_scale,    // vertex_index_offset
        NUM_VERTICES,           // vertex_index_count
        u32::from(!is_control), // allow_zero
        0,
    ];
    let params_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: params_buffer.as_entire_binding(),
        }],
    });

    // Generate the vertex buffer contents: a flat array of f32 filled by
    // repeating the arbitrary values, shared by every buffer.
    let float_count = (NUM_VERTICES * ATTRIBUTES_PER_BUFFER * type_info.components) as usize;
    let vertex_array: Vec<f32> = (0..float_count)
        .map(|i| ARBITRARY_VALUES[i % ARBITRARY_VALUES.len()] as f32)
        .collect();

    // Buffer 0 (instance step mode) is kept in range: no partialLastNumber.
    let vertex_buffers: Vec<wgpu::Buffer> = (0..buffer_count)
        .map(|i| {
            let partial = subcase.partial_last_number && i != 0;
            let byte_len = vertex_array.len() * 4;
            let (size, write_floats) = if partial {
                (byte_len - 1, vertex_array.len() - 1)
            } else {
                (byte_len, vertex_array.len())
            };
            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vertex buffer"),
                size: size as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            ctx.queue.write_buffer(
                &buffer,
                0,
                bytemuck::cast_slice(&vertex_array[..write_floats]),
            );
            buffer
        })
        .collect();

    // Index buffer [0, 1, ..., vertexCountInIndexBuffer - 1].
    let index_data: Vec<u32> = (0..NUM_VERTICES).collect();
    let index_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("index buffer"),
        contents: bytemuck::cast_slice(&index_data),
        usage: wgpu::BufferUsages::INDEX,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_attachment,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, Some(&bind_group), &[]);

        // Bind the vertex buffers. Buffer 0 (instance step mode) is bound at
        // offset 0; the rest optionally at a 4-byte offset to induce OOB.
        for (i, buffer) in vertex_buffers.iter().enumerate() {
            let offset = if i != 0 && subcase.offset_vertex_buffer {
                4
            } else {
                0
            };
            rpass.set_vertex_buffer(i as u32, Some(buffer.slice(offset..)));
        }

        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        // drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance)
        // with baseVertex += errorScale.
        rpass.draw_indexed(0..NUM_VERTICES, subcase.error_scale as i32, 0..NUM_VERTICES);
    }

    encoder.finish()
}

async fn robust_access_vertex(ctx: TestingContext) {
    // 2x1 render target: failing vertices land in the left pixel (x=0), which
    // must stay the green clear color.
    let color_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("color"),
        size: wgpu::Extent3d {
            width: 2,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Readback buffer for the 2x1 texture (bytes_per_row padded to 256).
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(16),
                },
                count: None,
            }],
        });
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

    let mut failures = Vec::new();
    let mut pipeline_cache: std::collections::HashMap<PipelineKey, wgpu::RenderPipeline> =
        std::collections::HashMap::new();

    for (type_index, type_info) in TYPE_INFOS.into_iter().enumerate() {
        for additional_buffers in [0, 4] {
            for partial_last_number in [false, true] {
                for offset_vertex_buffer in [false, true] {
                    for error_scale in ERROR_SCALES {
                        let subcase = Subcase {
                            type_info,
                            additional_buffers,
                            partial_last_number,
                            offset_vertex_buffer,
                            error_scale,
                        };

                        let key = PipelineKey {
                            type_index,
                            buffer_count: additional_buffers + 2,
                        };
                        let pipeline = pipeline_cache
                            .entry(key)
                            .or_insert_with(|| build_pipeline(&ctx, &pipeline_layout, key))
                            .clone();

                        let draw =
                            run_subcase(&ctx, &subcase, &pipeline, &bind_group_layout, &color_view);

                        let mut copy_encoder = ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                        copy_encoder.copy_texture_to_buffer(
                            wgpu::TexelCopyTextureInfo {
                                texture: &color_texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::TexelCopyBufferInfo {
                                buffer: &readback,
                                layout: wgpu::TexelCopyBufferLayout {
                                    offset: 0,
                                    bytes_per_row: Some(256),
                                    rows_per_image: Some(1),
                                },
                            },
                            wgpu::Extent3d {
                                width: 2,
                                height: 1,
                                depth_or_array_layers: 1,
                            },
                        );

                        ctx.queue.submit([draw, copy_encoder.finish()]);

                        let slice = readback.slice(..);
                        slice.map_async(wgpu::MapMode::Read, |_| ());
                        ctx.async_poll(wgpu::PollType::wait_indefinitely())
                            .await
                            .unwrap();
                        let left_pixel = {
                            let data = slice.get_mapped_range().unwrap();
                            [data[0], data[1], data[2], data[3]]
                        };
                        readback.unmap();

                        // Left pixel must be the green clear color.
                        if left_pixel != [0, 255, 0, 255] {
                            let case = format!(
                                "type={} additionalBuffers={} partialLastNumber={} \
                                 offsetVertexBuffer={} errorScale={}",
                                subcase.type_info.name,
                                subcase.additional_buffers,
                                subcase.partial_last_number,
                                subcase.offset_vertex_buffer,
                                subcase.error_scale,
                            );
                            eprintln!("Failed: {case} -> left pixel {left_pixel:?}");
                            failures.push(case);
                        }
                    }
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} subcase(s) failed robust vertex access",
        failures.len()
    );
}

#[gpu_test]
static ROBUST_ACCESS_VERTEX_BASE_VERTEX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits())
    .run_async(robust_access_vertex);

// ---------------------------------------------------------------------------
// Diagnostic probe: read back the actual `@builtin(vertex_index)` the driver
// produces for a sweep of `base_vertex` values.
//
// The CTS port above narrowed the failure to a single ingredient: large
// `base_vertex`. This probe pins the mechanism by reading `vertex_index`
// directly. Each sample does one indexed draw of a single vertex with index
// value 0, so the expected `vertex_index` is exactly `base_vertex`. The slot to
// write is supplied via a uniform (independent of `base_vertex`), avoiding the
// circularity of keying the output on the value under test.
// ---------------------------------------------------------------------------

/// `base_vertex` values to probe, chosen to bracket the 10^4..10^6 threshold and
/// to reveal a power-of-two truncation boundary if there is one.
const BASE_VERTEX_SAMPLES: [i32; 13] = [
    0, 1, 100, 10_000, 65_535, 65_536, 100_000, 131_072, 262_144, 524_288, 1_000_000, 1_048_576,
    16_777_215,
];

const SWEEP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> output : array<u32>;

struct Params { slot : u32, pad0 : u32, pad1 : u32, pad2 : u32 };
@group(0) @binding(1) var<uniform> params : Params;

@vertex fn vs_main(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4<f32> {
  output[params.slot] = vertex_index;
  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment fn fs_main() -> @location(0) vec4<f32> {
  return vec4<f32>(0.0);
}
"#;

async fn vertex_id_base_vertex_sweep(ctx: TestingContext) {
    let sample_count = BASE_VERTEX_SAMPLES.len();
    let buffer_size = (sample_count * 4) as u64;

    let gpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("observed vertex_index"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let cpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let index_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("index buffer"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::INDEX,
    });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(4),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(16),
                    },
                    count: None,
                },
            ],
        });
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sweep"),
            source: wgpu::ShaderSource::Wgsl(SWEEP_SHADER.into()),
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                buffers: &[],
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

    let dummy = ctx
        .device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default());

    for (slot, &base_vertex) in BASE_VERTEX_SAMPLES.iter().enumerate() {
        let params = [slot as u32, 0, 0, 0];
        let params_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dummy,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&pipeline);
            rpass.set_bind_group(0, Some(&bind_group), &[]);
            rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..1, base_vertex, 0..1);
        }
        ctx.queue.submit([encoder.finish()]);
    }

    let mut copy_encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    copy_encoder.copy_buffer_to_buffer(&gpu_buffer, 0, &cpu_buffer, 0, buffer_size);
    ctx.queue.submit([copy_encoder.finish()]);

    let slice = cpu_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    let observed: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();

    let mut mismatches = 0;
    for (slot, &base_vertex) in BASE_VERTEX_SAMPLES.iter().enumerate() {
        // Index value is 0, so vertex_index should equal base_vertex.
        let expected = base_vertex as u32;
        let got = observed[slot];
        let status = if got == expected { "ok" } else { "MISMATCH" };
        eprintln!(
            "base_vertex={base_vertex} -> vertex_index={got} (expected {expected}) [{status}]"
        );
        if got != expected {
            mismatches += 1;
        }
    }

    assert_eq!(
        mismatches, 0,
        "{mismatches} base_vertex value(s) produced the wrong vertex_index"
    );
}

#[gpu_test]
static VERTEX_ID_BASE_VERTEX_SWEEP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
    )
    .run_async(vertex_id_base_vertex_sweep);
