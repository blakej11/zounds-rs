// Conway's Game of Life

use std::borrow::Cow;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};

use crate::{
    bindable::{Bindable, BindAccess, Binder, Buffer2D, BufferType},
    buffer_copy::BufferCopier,
    debug_buffer::DebugBuffer,
    dimensions::Dimensions,
    directions::{RenderDir, RenderMotion, RenderSources},
};

// ---------------------------------------------------------------------------
// Data that is shared between Rust and the compute pipeline in WGSL.

// Number of cells calculated in each gpu work group.
// This must match the value of the workgroup_size() annotation in life.wgsl
const WORKGROUP_SIZE: (u32, u32) = (8, 8);

// ---------------------------------------------------------------------------

pub struct Life {
    // Data for the compute shader.
    shader: wgpu::ShaderModule,
    pipeline: wgpu::ComputePipeline,
    bind_groups: RenderMotion<wgpu::BindGroup>,
    dimensions: Dimensions,
    cell_buffers: RenderSources<Buffer2D<f32>>,
    random_buf: Buffer2D<[u32; 4]>,
    cell_bc: BufferCopier<f32, f32>,
    rand_bc: BufferCopier<[u32; 4], [u32; 4]>,
    debug_buffer: DebugBuffer<f32>,
    frame_num: usize,
}

impl Life {
    pub fn new(
        device: &wgpu::Device,
        dimensions: Dimensions,
        params: &impl Bindable,
        texture: &impl Bindable,
        rng: &mut impl Rng,
    ) -> Self {
        // Load and compile the compute shader.
        let shader = device.create_shader_module(
            &wgpu::ShaderModuleDescriptor {
                label: Some("life algorithm"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    include_str!("life.wgsl"))),
            });

        // Allocate a pair of equal-sized GPU buffers to hold cell data.
        let cell_buffers: RenderSources<Buffer2D<f32>> =
            RenderSources::new(|dir| {
                let label = format!("Source for {:?}", dir);
                Buffer2D::new(device, &label, dimensions)
            });

        // Allocate a GPU buffer to hold random u32 data.
        // This is used to assign random sub-threshold values to dead cells,
        //   so they'd have interesting values if the threshold were changed.
        let random_data: Vec<[u32; 4]> = {
            let u = Uniform::new_inclusive(u32::MIN, u32::MAX);
            (0..dimensions.area()).map(|_| [
                u.sample(rng), u.sample(rng), u.sample(rng), u.sample(rng)
            ]).collect()
        };
        let random_buf: Buffer2D<[u32; 4]> = Buffer2D::new_init(
            device, "random data", BufferType::Storage, dimensions,
            bytemuck::cast_slice(&random_data),
        );

        // Allocate a buffer for looking at current algorithm state.
        let debug_buffer = DebugBuffer::new(&device, dimensions);

        // Get BufferCopiers for the resize method to use.
        let cell_bc: BufferCopier<f32, f32> = BufferCopier::new(device);
        let rand_bc: BufferCopier<[u32; 4], [u32; 4]> = BufferCopier::new(device);

        // Create bind groups for the arguments.
        let (pipeline, bind_groups) = Binder::bind_up_dir(
            device, &shader, "life",
            &|dir| {
                let v: Vec<(_, &dyn Bindable)> = vec![
                    (BindAccess::ReadOnly,  params),
                    (BindAccess::ReadOnly,  cell_buffers.src(dir)),
                    (BindAccess::WriteOnly, cell_buffers.dst(dir)),
                    (BindAccess::WriteOnly, &random_buf),
                    (BindAccess::WriteOnly, texture),
                ];
                v
            }
        );

        Life {
            shader,
            pipeline,
            bind_groups,
            dimensions,
            cell_buffers,
            random_buf,
            cell_bc,
            rand_bc,
            debug_buffer,
            frame_num: 0,
        }
    }

    // called on WindowEvent::Resized events
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dimensions: Dimensions,
        params: &impl Bindable,
        texture: &impl Bindable,
    ) {
        // Copy the old cell data over.
        let cell_buffers: RenderSources<Buffer2D<f32>> =
            RenderSources::new(|dir| {
                let label = format!("Source for {:?}", dir);
                Buffer2D::new(device, &label, dimensions)
            });
        RenderDir::iterate(|dir| {
            self.cell_bc.copy(device, queue,
                &self.cell_buffers.src(dir), &cell_buffers.src(dir))});

        // Copy the old random data over.
        let random_buf: Buffer2D<[u32; 4]> =
            Buffer2D::new(device, "random data", dimensions);
        self.rand_bc.copy(device, queue, &self.random_buf, &random_buf);

        // Bind up the new arguments.
        let (pipeline, bind_groups) = Binder::bind_up_dir(
            device, &self.shader, "life",
            &|dir| {
                let v: Vec<(_, &dyn Bindable)> = vec![
                    (BindAccess::ReadOnly,  params),
                    (BindAccess::ReadOnly,  cell_buffers.src(dir)),
                    (BindAccess::WriteOnly, cell_buffers.dst(dir)),
                    (BindAccess::WriteOnly, &random_buf),
                    (BindAccess::WriteOnly, texture),
                ];
                v
            }
        );

        self.pipeline = pipeline;
        self.bind_groups = bind_groups;
        self.dimensions = dimensions;
        self.cell_buffers = cell_buffers;
        self.random_buf = random_buf;
        self.debug_buffer = DebugBuffer::new(&device, dimensions);
    }

    // Take a single timestep in the Life algorithm.
    pub fn step(
        &mut self,
        command_encoder: &mut wgpu::CommandEncoder,
    ) {
        let debug = false;

        if debug {
            self.debug_buffer.enqueue_copyin(command_encoder, &self.src_buf());
        }

        let xdim = self.dimensions.width() + WORKGROUP_SIZE.0 - 1;
        let xgroups = xdim / WORKGROUP_SIZE.0;
        let ydim = self.dimensions.height() + WORKGROUP_SIZE.1 - 1;
        let ygroups = ydim / WORKGROUP_SIZE.1;

        let mut cpass = command_encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor {
                label: Some("Life grid step")
            });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_groups.get(self.dir()), &[]);
        cpass.dispatch(xgroups, ygroups, 1);
        self.frame_num += 1;
    }

    // Import some data into the Life grid.
    pub fn import(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cells: &Vec<f32>,
    ) {
        self.src_buf().copyin_vec(device, queue, cells);
    }

    fn src_buf(&self) -> &Buffer2D<f32> {
        self.cell_buffers.src(self.dir())
    }

    pub fn dir(&self) -> RenderDir {
        RenderDir::dir(self.frame_num)
    }

    #[allow(dead_code)]
    pub fn dump_debug(
        &self,
        device: &wgpu::Device,
    ) {
        log::debug!("Life data entering step {}:", self.frame_num());
        self.debug_buffer.display(device);
        log::debug!("");
    }

    #[allow(dead_code)]
    pub fn frame_num(&self) -> usize {
        self.frame_num
    }
}
