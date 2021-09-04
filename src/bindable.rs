use std::mem;
use wgpu::util::DeviceExt;
use std::marker::PhantomData;
use bytemuck::Pod;

use crate::{
    dimensions::Dimensions,
    directions::{RenderDir, RenderMotion},
};

// ---------------------------------------------------------------------

#[derive(Clone, Copy)]
pub enum BindAccess {
    ReadOnly,
    ReadSampled,
    WriteOnly,
}

pub trait Bindable {
    fn binding_resource(
        &self
    ) -> wgpu::BindingResource;

    fn binding_type(
        &self,
        access: BindAccess,
    ) -> wgpu::BindingType;
}

// ---------------------------------------------------------------------

pub struct Binder { }

impl Binder {
    // Take the arguments that get used by the a compute kernel,
    // create wgpu bindings for them, and bundle them all together.
    pub fn bind_up(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        entry_point: &str,
        args: &Vec<(BindAccess, &dyn Bindable)>,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
        let bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{} bind group layout", entry_point)),
                entries: &args.iter().enumerate().map(|(idx, (access, arg))| {
                    wgpu::BindGroupLayoutEntry {
                        binding: idx as _,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: arg.binding_type(*access),
                        count: None,
                    }}).collect::<Vec<_>>(),
            });
        let bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some(&format!("{} bind group", entry_point)),
                layout: &bind_group_layout,
                entries: &args.iter().enumerate().map(|(idx, (_, arg))| {
                    wgpu::BindGroupEntry {
                        binding: idx as _,
                        resource: arg.binding_resource(),
                    }}).collect::<Vec<_>>(),
            });

        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{} pipeline layout", entry_point)),
                bind_group_layouts: &[
                    &bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{} compute pipeline", entry_point)),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point,
            });

        (pipeline, bind_group)
    }

    // Used when you want to have different args for different directions
    // (e.g. switching source and destination buffers).
    pub fn bind_up_dir<'a, F>(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        entry_point: &str,
        args: &'a F,
    ) -> (wgpu::ComputePipeline, RenderMotion<wgpu::BindGroup>)
    where
        F: Fn(RenderDir) -> Vec<(BindAccess, &'a dyn Bindable)>
    {
        let bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{} bind group layout", entry_point)),
                entries: &args(RenderDir::Forward).iter().enumerate()
                    .map(|(idx, (access, arg))| {
                        wgpu::BindGroupLayoutEntry {
                            binding: idx as _,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: arg.binding_type(*access),
                            count: None,
                        }}).collect::<Vec<_>>(),
            });
        let bind_groups = RenderMotion::new(|dir| device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some(&format!("{} bind group", entry_point)),
                layout: &bind_group_layout,
                entries: &args(dir).iter().enumerate()
                    .map(|(idx, (_, arg))| {
                        wgpu::BindGroupEntry {
                            binding: idx as _,
                            resource: arg.binding_resource(),
                        }}).collect::<Vec<_>>(),
            }));

        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{} pipeline layout", entry_point)),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{} compute pipeline", entry_point)),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point,
            });

        (pipeline, bind_groups)
    }
}

// ---------------------------------------------------------------------

pub enum BufferType {
    Uniform,
    Storage
}

pub struct Buffer {
    buf: wgpu::Buffer,
    size: usize,
    ty: BufferType,
}

impl Buffer {
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        ty: BufferType,
        size: usize,
    ) -> Self {
        Buffer {
            buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                usage: match ty {
                    BufferType::Uniform => wgpu::BufferUsages::UNIFORM,
                    BufferType::Storage => wgpu::BufferUsages::STORAGE,
                } | wgpu::BufferUsages::COPY_SRC
                  | wgpu::BufferUsages::COPY_DST,
                size: size as _,
                mapped_at_creation: false,
            }),
            size,
            ty,
        }
    }

    pub fn new_init(
        device: &wgpu::Device,
        label: &str,
        ty: BufferType,
        contents: &[u8],
    ) -> Self {
        Buffer {
            buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: match ty {
                    BufferType::Uniform => wgpu::BufferUsages::UNIFORM,
                    BufferType::Storage => wgpu::BufferUsages::STORAGE,
                } | wgpu::BufferUsages::COPY_SRC
                  | wgpu::BufferUsages::COPY_DST,
            }),
            size: contents.len(),
            ty,
        }
    }

    pub fn buf(
        &self
    ) -> &wgpu::Buffer {
        &self.buf
    }
}

impl Bindable for Buffer {
    fn binding_resource(&self) -> wgpu::BindingResource {
        self.buf.as_entire_binding()
    }

    fn binding_type(&self, access: BindAccess) -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: match self.ty {
                BufferType::Storage => wgpu::BufferBindingType::Storage {
                    read_only: match access {
                        BindAccess::ReadOnly => true,
                        // XXX This suggests that the abstraction is wrong;
                        // you shouldn't be allowed to specify ReadSampled
                        // on a Buffer, because they don't support that.
                        BindAccess::ReadSampled => true,
                        BindAccess::WriteOnly => false,
                    },
                },
                BufferType::Uniform => wgpu::BufferBindingType::Uniform,
            },
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(self.size as _),
        }
    }
}

// ---------------------------------------------------------------------

pub struct Buffer2D<T> {
    buf: Buffer,
    dim: Dimensions,
    label: String,
    _phantom: PhantomData<T>, 
}

impl<T> Buffer2D<T> {
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        dim: Dimensions,
    ) -> Self {
        Buffer2D {
            buf: Buffer::new(
                device,
                label,
                BufferType::Storage,
                dim.area() * mem::size_of::<T>()
            ),
            dim,
            label: label.to_string(),
            _phantom: PhantomData,
        }
    }

    pub fn new_init(
        device: &wgpu::Device,
        label: &str,
        ty: BufferType,
        dim: Dimensions,
        contents: &[u8],
    ) -> Self {
        assert_eq!(contents.len(), dim.area() * mem::size_of::<T>());

        Buffer2D {
            buf: Buffer::new_init(device, label, ty, contents),
            dim,
            label: label.to_string(),
            _phantom: PhantomData,
        }
    }

    pub fn buf(&self) -> &wgpu::Buffer {
        &self.buf.buf()
    }

    pub fn dim(&self) -> Dimensions {
        self.dim
    }
}

impl<T> Buffer2D<T> where T: Pod {
    pub fn copyin_buf(
        &self,
        command_encoder: &mut wgpu::CommandEncoder,
        src: &Buffer2D<T>,
    ) {
        assert_eq!(src.dim(), self.dim());
        let size = self.dim().area() * mem::size_of::<T>();

        command_encoder.copy_buffer_to_buffer(
            &src.buf(), 0, self.buf(), 0, size as _);
    }

    pub fn copyin_vec(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &Vec<T>,
    ) {
        assert_eq!(data.len(), self.dim.area());

        let import_buf = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some(format!("{} data import buffer", self.label).as_str()),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC,
                contents: bytemuck::cast_slice(&data),
                });

        let mut command_encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some(format!("importing {} data", self.label).as_str())
            });
        command_encoder.copy_buffer_to_buffer(
            &import_buf, 0, self.buf(), 0,
            bytemuck::cast_slice::<_, u8>(&data).len() as u64
        );
        queue.submit(Some(command_encoder.finish()));
    }
}

impl<T> Bindable for Buffer2D<T> {
    fn binding_resource(&self) -> wgpu::BindingResource {
        self.buf.binding_resource()
    }

    fn binding_type(&self, access: BindAccess) -> wgpu::BindingType {
        self.buf.binding_type(access)
    }
}

// ---------------------------------------------------------------------

pub struct Texture {
    texture_view: wgpu::TextureView,
    format: wgpu::TextureFormat,
}

impl Texture {
    pub fn new(
        device: &wgpu::Device,
        dimensions: Dimensions,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: dimensions.width(),
                height: dimensions.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::STORAGE_BINDING,
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Texture {
            texture_view,
            format,
        }
    }
}

impl Bindable for Texture {
    fn binding_resource(
        &self
    ) -> wgpu::BindingResource {
        wgpu::BindingResource::TextureView(&self.texture_view)
    }

    fn binding_type(
        &self,
        access: BindAccess,
    ) -> wgpu::BindingType {
        match access {
            BindAccess::ReadOnly =>
                wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadOnly,
                    format: self.format,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },

            BindAccess::ReadSampled => 
                wgpu::BindingType::Texture {
                    sample_type: self.format.describe().sample_type,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },

            BindAccess::WriteOnly => 
                wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: self.format,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
        }
    }
}

// ---------------------------------------------------------------------

pub struct Sampler {
    sampler: wgpu::Sampler,
}

impl Sampler {
    pub fn new(
        device: &wgpu::Device,
        address_mode: wgpu::AddressMode,
        filter_mode: wgpu::FilterMode,
    ) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: address_mode,
            address_mode_v: address_mode,
            address_mode_w: address_mode,
            mag_filter: filter_mode,
            min_filter: filter_mode,
            mipmap_filter: filter_mode,
            ..Default::default()
        });

        Sampler {
            sampler,
        }
    }
}

impl Bindable for Sampler {
    fn binding_resource(
        &self
    ) -> wgpu::BindingResource {
        wgpu::BindingResource::Sampler(&self.sampler)
    }

    fn binding_type(
        &self,
        _: BindAccess,
    ) -> wgpu::BindingType {
        wgpu::BindingType::Sampler {
            comparison: false,
            filtering: true,
        }
    }
}
