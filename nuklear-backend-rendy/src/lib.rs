use nuklear::{
    Buffer, Context, ConvertConfig, DrawVertexLayoutAttribute, DrawVertexLayoutElements,
    DrawVertexLayoutFormat, Handle, Size, Vec2,
};

use rendy::{
    hal::{self},
    resource::{DescriptorSet, Escape},
    shader::{ShaderKind, SourceLanguage, SourceShaderInfo, SpirvShader},
    texture::Texture,
};

lazy_static::lazy_static! {
    static ref VERTEX: SpirvShader = SourceShaderInfo::new(
        include_str!("../shaders/shader.vert"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/examples/sprite/shader.vert").into(),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref FRAGMENT: SpirvShader = SourceShaderInfo::new(
        include_str!("../shaders/shader.frag"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/examples/sprite/shader.frag").into(),
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

pub struct Drawer {
    cmd: Buffer,
}

impl Drawer {
    pub fn new(command_buffer: Buffer) -> Drawer {
        Drawer {
            cmd: command_buffer,
        }
    }

    pub fn add_texture(&mut self, image: &[u8], width: u32, height: u32) -> Handle {
        Handle::from_id(1)
    }

    pub fn draw(&mut self, ctx: &mut Context, cfg: &mut ConvertConfig) {}
}
