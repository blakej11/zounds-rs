use std::fmt::Debug;
use bytemuck::Pod;

use crate::{
    bindable::Buffer2D,
    dimensions::Dimensions,
};

#[cfg(not(target_arch = "wasm32"))]
use pollster::block_on;

pub struct DebugBuffer<T> {
    buf: Buffer2D<T>,
}

impl<T> DebugBuffer<T> {
    pub fn new(
        device: &wgpu::Device,
        dim: Dimensions,
    ) -> Self {
        DebugBuffer {
            buf: Buffer2D::new(device, "debug buffer", dim),
        }
    }

    // Enqueue a copyin action into this debug buffer.
    //
    // Note that display()'ing this debug buffer after an enqueue_copyin()
    // will not show anything useful until the command encoder has been finished
    // and submitted to a command queue.
    pub fn enqueue_copyin(
        &self,
        command_encoder: &mut wgpu::CommandEncoder,
        buf: &Buffer2D<T>,
    ) where
        T: Pod
    {
        self.buf.copyin_buf(command_encoder, buf);
    }

    // Note the above caveat about using enqueue_copyin() with display().
    pub fn display(
        &self,
        device: &wgpu::Device,
    ) where
        T: Debug + Pod
    {
        // Start a request to map the debug buffer, and wait for it.
        let buf = self.buf.buf();
        let buffer_slice = buf.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        device.poll(wgpu::Maintain::Wait);

        // This doesn't work with wasm32.
        #[cfg(target_arch = "wasm32")]
        log::debug!("debug_buffer doesn't work under wasm32 :(");

        #[cfg(not(target_arch = "wasm32"))]
        match block_on(buffer_future) {
            Err(e) => {
                println!("failed to wait for buffer read: {}", e)
            }
            Ok(_) => {
                let data : Vec<u8> = buffer_slice.get_mapped_range().to_vec();
                let result : Vec<T> = bytemuck::cast_slice(&data).to_vec();
                for d in result {
                    println!("{:?}", d);
                }

                // Current API requires dropping the data before unmapping.
                drop(data);
                buf.unmap();
            }
        }
    }

    // Copy the given buffer into this debug buffer and display it immediately.
    // This avoids the caveat mentioned above.
    #[allow(dead_code)]
    pub fn copyin_and_display(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buf: &Buffer2D<T>,
    ) where
        T: Debug + Pod
    {
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: None
            });
        self.enqueue_copyin(&mut command_encoder, buf);
        queue.submit(Some(command_encoder.finish()));

        self.display(device);
    }
}