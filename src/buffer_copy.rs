use std::borrow::Cow;
use std::marker::PhantomData;
use bytemuck::{Pod, Zeroable};

use crate::{
    bindable::{Bindable, BindAccess, Binder, Buffer, Buffer2D, BufferType},
    dimensions::Dimensions,
};

// ---------------------------------------------------------------------------
// Data that is shared between Rust and the compute pipeline in WGSL.

// This must match the value of the workgroup_size() annotation in life.wgsl
const WORKGROUP_SIZE: (u32, u32) = (8, 8);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CopyParams {
    odx: u32,
    ody: u32,
    ndx: u32,
    ndy: u32,
    owidth: u32,
    nwidth: u32,
    width: u32,
    height: u32,
}

// ---------------------------------------------------------------------------

pub struct CopyShaderInfo {
    src_type: &'static str,         // source type in WGSL
    dst_type: &'static str,         // destination type in WGSL
    manip: Option<&'static str>,    // any manipulation needed besides "new=old"
}

// This trait can be implemented for any type pairs that it makes sense
// for the "copy" kernel to copy.
pub trait BufferCopyable {
    fn shader_info() -> CopyShaderInfo;
}

pub struct BC<T,U> (T,U);

impl BufferCopyable for BC<f32, f32> {
    fn shader_info() -> CopyShaderInfo {
        CopyShaderInfo {
            src_type: "f32",
            dst_type: "f32",
            manip: None,
        }
    }
}

impl BufferCopyable for BC<[u32; 4], [u32; 4]> {
    fn shader_info() -> CopyShaderInfo {
        CopyShaderInfo {
            src_type: "vec4<u32>",
            dst_type: "vec4<u32>",
            manip: None,
        }
    }
}

// ---------------------------------------------------------------------------

pub struct BufferCopier<T, U> {
    shader: wgpu::ShaderModule,
    _t: PhantomData<T>,
    _u: PhantomData<U>,
}

impl<T,U> BufferCopier<T,U> {
    pub fn new(
        device: &wgpu::Device,
    ) -> Self where
        BC<T, U>: BufferCopyable
    {
        // Perform the type-specific translations on the prototype shader.
        let shader_info = BC::<T, U>::shader_info();
        let shader_src =
            include_str!("buffer_copy.wgsl")
            .replace("{src_type}", shader_info.src_type)
            .replace("{dst_type}", shader_info.dst_type)
            .replace("{manip}", shader_info.manip.unwrap_or("new = old"));

        let shader = device.create_shader_module(
            &wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_src)),
            });

        BufferCopier {
            shader,
            _t: PhantomData,
            _u: PhantomData,
        }
    }

    fn copy_params(
        src_buf: &Buffer2D<T>,
        dst_buf: &Buffer2D<U>,
    ) -> CopyParams {
        let Dimensions { width: ow, height: oh } = src_buf.dim();
        let Dimensions { width: nw, height: nh } = dst_buf.dim();
        CopyParams {
            odx:    if nw < ow { (ow - nw) / 2 } else { 0 },
            ndx:    if nw > ow { (nw - ow) / 2 } else { 0 },
            ody:    if nh < oh { (oh - nh) / 2 } else { 0 },
            ndy:    if nh > oh { (nh - oh) / 2 } else { 0 },
            owidth: ow,
            nwidth: nw,
            width:  ow.min(nw),
            height: oh.min(nh),
        }
    }

    pub fn copy(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_buf: &Buffer2D<T>,
        dst_buf: &Buffer2D<U>,
    ) where
        BC<T, U>: BufferCopyable,
        U: Default + Pod,
    {
        // Derive the parameters needed for the kernel, and make a
        // Uniform buffer to holds them.
        let params = BufferCopier::copy_params(src_buf, dst_buf);
        let param_buf = Buffer::new_init(
            device,
            "copy data parameters",
            BufferType::Uniform,
            bytemuck::bytes_of(&params)
        );
        log::info!("buffer copy params: {:?}", &params);

        // Bind up the arguments.
        let args: Vec<(_, &dyn Bindable)> = vec![
            (BindAccess::ReadOnly,  &param_buf),
            (BindAccess::ReadOnly,  src_buf),
            (BindAccess::WriteOnly, dst_buf),
        ];
        let (pipeline, bind_group) =
            Binder::bind_up(device, &self.shader, "copy", &args);

        // XXX Workaround for https://github.com/gfx-rs/wgpu/pull/1561
        // which hasn't been fixed in Firefox Nightly as of 2021-08-02.
        #[cfg(target_arch = "wasm32")]
        dst_buf.copyin_vec(device, queue,
             &vec![Default::default(); dst_buf.dim().area()]);

        // Do the copy.
        let mut command_encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: None
            });
        {
            let mut cpass = command_encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: None
                });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch(
                (params.width + WORKGROUP_SIZE.0 - 1) / WORKGROUP_SIZE.0,
                (params.height + WORKGROUP_SIZE.1 - 1) / WORKGROUP_SIZE.1,
                1
            );
        }
        queue.submit(Some(command_encoder.finish()));
    }
}