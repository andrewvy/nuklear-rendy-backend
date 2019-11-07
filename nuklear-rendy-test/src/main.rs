//!
//! The mighty triangle example.
//! This examples shows colord triangle on white background.
//! Nothing fancy. Just prove that `rendy` works.
//!

#![cfg_attr(
    not(any(feature = "dx12", feature = "metal", feature = "vulkan")),
    allow(unused)
)]

use derivative::Derivative;

use rendy::{
    command::{Families, QueueId, RenderPassEncoder},
    factory::{Config, Factory, ImageState},
    graph::{render::*, Graph, GraphBuilder, GraphContext, NodeBuffer, NodeImage},
    hal,
    memory::Dynamic,
    mesh::PosColor,
    resource::{Buffer, BufferInfo, DescriptorSetLayout, Escape, Handle, ImageView},
    shader::{ShaderKind, SourceLanguage, SourceShaderInfo, SpirvShader},
    texture::Texture,
    wsi::winit::{EventsLoop, WindowBuilder},
};

use std::sync::{Arc, Mutex};

#[cfg(feature = "spirv-reflection")]
use rendy::shader::SpirvReflection;

#[cfg(not(feature = "spirv-reflection"))]
use rendy::mesh::AsVertex;

#[cfg(feature = "dx12")]
type Backend = rendy::dx12::Backend;

#[cfg(feature = "metal")]
type Backend = rendy::metal::Backend;

#[cfg(feature = "vulkan")]
type Backend = rendy::vulkan::Backend;

const MAX_VERTEX_MEMORY: usize = 512 * 1024;
const MAX_ELEMENT_MEMORY: usize = 128 * 1024;
const MAX_COMMANDS_MEMORY: usize = 64 * 1024;

lazy_static::lazy_static! {
    static ref VERTEX: SpirvShader = SourceShaderInfo::new(
        include_str!("../shaders/shader.vert"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/shader.vert").into(),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref FRAGMENT: SpirvShader = SourceShaderInfo::new(
        include_str!("../shaders/shader.frag"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/shader.frag").into(),
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).unwrap()
        .with_fragment(&*FRAGMENT).unwrap();
}

#[cfg(feature = "spirv-reflection")]
lazy_static::lazy_static! {
    static ref SHADER_REFLECTION: SpirvReflection = SHADERS.reflect().unwrap();
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Drawer<B: hal::Backend> {
    #[derivative(Debug="ignore")]
    cmd: nuklear::Buffer,
    queue: QueueId,
    textures: Vec<Texture<B>>,
    vbuf: Escape<Buffer<B>>,
}

unsafe impl<B: hal::Backend> Send for Drawer<B> {}

impl<B: hal::Backend> Drawer<B> {
    pub fn new(
        factory: &mut Factory<B>,
        queue: QueueId,
        command_buffer: nuklear::Buffer,
        texture_count: usize,
        vbo_size: usize,
        ebo_size: usize,
    ) -> Drawer<B> {
        let texture_count = 36;

        let mut vbuf = factory
            .create_buffer(
                BufferInfo {
                    size: vbo_size as u64,
                    usage: hal::buffer::Usage::VERTEX,
                },
                Dynamic,
            )
            .unwrap();

        Drawer {
            vbuf,
            queue,
            cmd: command_buffer,
            textures: Vec::with_capacity(texture_count + 1),
        }
    }

    pub fn add_texture(
        &mut self,
        factory: &mut Factory<B>,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> nuklear::Handle {
        let texture = rendy::texture::TextureBuilder::new()
            .set_data_width(width)
            .set_data_height(height)
            .set_raw_data(image, hal::format::Format::Rgba8Unorm)
            .build(
                ImageState {
                    queue: self.queue,
                    stage: hal::pso::PipelineStage::FRAGMENT_SHADER,
                    access: hal::image::Access::SHADER_READ,
                    layout: hal::image::Layout::ShaderReadOnlyOptimal,
                },
                factory,
            )
            .unwrap();

        self.textures.push(texture);

        nuklear::Handle::from_id(self.textures.len() as i32)
    }

    pub fn draw(
        &mut self,
        ctx: &mut nuklear::Context,
        cfg: &mut nuklear::ConvertConfig,
        encoder: &mut RenderPassEncoder<'_, B>,
        factory: &mut Factory<B>,
        width: u32,
        height: u32,
        scale: nuklear::Vec2,
    ) {
        for cmd in ctx.draw_command_iterator(&self.cmd) {
            if cmd.elem_count() < 1 {
                continue;
            }

            let id = cmd.texture().id().unwrap();

            let x = cmd.clip_rect().x * scale.x;
            let y = cmd.clip_rect().y * scale.y;
            let w = cmd.clip_rect().w * scale.x;
            let h = cmd.clip_rect().h * scale.y;

            let source_rect = hal::pso::Rect {
                x: (if x < 0f32 { 0f32 } else { x }) as i16,
                y: (if y < 0f32 { 0f32 } else { y }) as i16,
                w: (if x < 0f32 { w + x } else { w }) as i16,
                h: (if y < 0f32 { h + y } else { h }) as i16,
            };

            let texture = self.find_texture(id).unwrap();
        }
    }

    fn find_texture(&self, id: i32) -> Option<&ImageView<B>> {
        if id > 0 && id as usize <= self.textures.len() {
            Some(self.textures[(id - 1) as usize].view())
        } else {
            None
        }
    }
}

#[derive(Default, Debug)]
struct NuklearRenderPipelineDesc;

#[derive(Debug)]
struct NuklearRenderPipeline<B: hal::Backend> {
    vertex: Option<Escape<Buffer<B>>>,
    drawer: Drawer<B>,
}

impl<B, T> SimpleGraphicsPipelineDesc<B, T> for NuklearRenderPipelineDesc
where
    B: hal::Backend,
    T: ?Sized,
{
    type Pipeline = NuklearRenderPipeline<B>;

    fn depth_stencil(&self) -> Option<hal::pso::DepthStencilDesc> {
        None
    }

    fn load_shader_set(&self, factory: &mut Factory<B>, _aux: &T) -> rendy_shader::ShaderSet<B> {
        SHADERS.build(factory, Default::default()).unwrap()
    }

    fn vertices(
        &self,
    ) -> Vec<(
        Vec<hal::pso::Element<hal::format::Format>>,
        hal::pso::ElemStride,
        hal::pso::VertexInputRate,
    )> {
        #[cfg(feature = "spirv-reflection")]
        return vec![SHADER_REFLECTION
            .attributes_range(..)
            .unwrap()
            .gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex)];

        #[cfg(not(feature = "spirv-reflection"))]
        return vec![PosColor::vertex().gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex)];
    }

    fn build<'a>(
        self,
        _ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        queue: QueueId,
        _aux: &T,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<NuklearRenderPipeline<B>, failure::Error> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert!(set_layouts.is_empty());

        let mut allo = nuklear::Allocator::new_vec();

        let drawer = Drawer::new(factory, queue, nuklear::Buffer::with_size(&mut allo, MAX_COMMANDS_MEMORY), 36, MAX_VERTEX_MEMORY, MAX_ELEMENT_MEMORY);

        Ok(NuklearRenderPipeline { vertex: None, drawer })
    }
}

impl<B, T> SimpleGraphicsPipeline<B, T> for NuklearRenderPipeline<B>
where
    B: hal::Backend,
    T: ?Sized,
{
    type Desc = NuklearRenderPipelineDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        _index: usize,
        _aux: &T,
    ) -> PrepareResult {
        if self.vertex.is_none() {
            #[cfg(feature = "spirv-reflection")]
            let vbuf_size = SHADER_REFLECTION.attributes_range(..).unwrap().stride as u64 * 3;

            #[cfg(not(feature = "spirv-reflection"))]
            let vbuf_size = PosColor::vertex().stride as u64 * 3;

            let mut vbuf = factory
                .create_buffer(
                    BufferInfo {
                        size: vbuf_size,
                        usage: hal::buffer::Usage::VERTEX,
                    },
                    Dynamic,
                )
                .unwrap();

            unsafe {
                // Fresh buffer.
                factory
                    .upload_visible_buffer(
                        &mut vbuf,
                        0,
                        &[
                            PosColor {
                                position: [0.0, -0.5, 0.0].into(),
                                color: [1.0, 0.0, 0.0, 1.0].into(),
                            },
                            PosColor {
                                position: [0.5, 0.5, 0.0].into(),
                                color: [0.0, 1.0, 0.0, 1.0].into(),
                            },
                            PosColor {
                                position: [-0.5, 0.5, 0.0].into(),
                                color: [0.0, 0.0, 1.0, 1.0].into(),
                            },
                        ],
                    )
                    .unwrap();
            }

            self.vertex = Some(vbuf);
        }

        PrepareResult::DrawReuse
    }

    fn draw(
        &mut self,
        _layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        _index: usize,
        _aux: &T,
    ) {
        let vbuf = self.vertex.as_ref().unwrap();

        unsafe {
            encoder.bind_vertex_buffers(0, Some((vbuf.raw(), 0)));
            encoder.draw(0..3, 0..1);
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &T) {}
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn run(
    event_loop: &mut EventsLoop,
    factory: &mut Factory<Backend>,
    families: &mut Families<Backend>,
    mut graph: Graph<Backend, ()>,
) -> Result<(), failure::Error> {
    let started = std::time::Instant::now();

    let mut frames = 0u64..;
    let mut elapsed = started.elapsed();

    for _ in &mut frames {
        factory.maintain(families);
        event_loop.poll_events(|_| ());
        graph.run(factory, families, &());

        elapsed = started.elapsed();
        if elapsed >= std::time::Duration::new(5, 0) {
            break;
        }
    }

    let elapsed_ns = elapsed.as_secs() * 1_000_000_000 + elapsed.subsec_nanos() as u64;

    log::info!(
        "Elapsed: {:?}. Frames: {}. FPS: {}",
        elapsed,
        frames.start,
        frames.start * 1_000_000_000 / elapsed_ns
    );

    graph.dispose(factory, &());
    Ok(())
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn main() {
    env_logger::Builder::from_default_env().init();

    let config: Config = Default::default();

    let (mut factory, mut families): (Factory<Backend>, _) = rendy::factory::init(config).unwrap();

    let mut event_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title("Rendy example")
        .build(&event_loop)
        .unwrap();

    event_loop.poll_events(|_| ());

    let surface = factory.create_surface(&window);

    let mut graph_builder = GraphBuilder::<Backend, ()>::new();

    graph_builder.add_node(
        NuklearRenderPipeline::builder()
            .into_subpass()
            .with_color_surface()
            .into_pass()
            .with_surface(
                surface,
                Some(hal::command::ClearValue::Color([1.0, 1.0, 1.0, 1.0].into())),
            ),
    );

    let graph = graph_builder
        .build(&mut factory, &mut families, &())
        .unwrap();

    run(&mut event_loop, &mut factory, &mut families, graph).unwrap();
}

#[cfg(not(any(feature = "dx12", feature = "metal", feature = "vulkan")))]
fn main() {
    panic!("Specify feature: { dx12, metal, vulkan }");
}
