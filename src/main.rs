mod bindable;
mod buffer_copy;
mod debug_buffer;
mod dimensions;
mod directions;
mod life;
mod renderer;
mod window;

use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};

use winit::event::VirtualKeyCode;
use bytemuck::{Pod, Zeroable};

use crate::{
    bindable::{Buffer, BufferType, Texture},
    dimensions::Dimensions,
    life::Life,
    renderer::Renderer,
    window::WindowOps,
};

// ---------------------------------------------------------------------------
// Structures that are shared between Rust and the compute/fragment shaders.

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width : u32,
    height : u32,
    threshold : f32,
}

// ---------------------------------------------------------------------------

/// This holds all of the state used by the program.
struct LifeProg {
    dim: Dimensions, // XXX for debugging
    life: Life,
    renderer: Renderer,
}

impl LifeProg {
    fn params(
        device: &wgpu::Device,
        dim: Dimensions,
    ) -> Buffer {
        Buffer::new_init(
            device,
            "Life parameters",
            BufferType::Uniform,
            bytemuck::bytes_of(&Params {
                width: dim.width(),
                height: dim.height(),
                threshold: 0.7,
            }),
        )
    }
}

impl window::Example for LifeProg {

    /// Construct the initial instance of the LifeProg struct.
    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let dim = Dimensions::new(config.width, config.height);
        let ncells = dim.area();

        // Get a pseudo-random number generator.
        // We don't need crypto-strength PRNGs, so we use SmallRng.
        // Might consider seeding this with something from rand::thread_rng()
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        // Parameters for the game, shared between compute and fragment shaders.
        let params = LifeProg::params(device, dim);

        // Create a texture that's shared between compute and fragment shaders.
        let texture = Texture::new(&device, dim, wgpu::TextureFormat::R32Float);

        // Initialize the life algorithm.
        let mut life = Life::new(&device, dim, &params, &texture, &mut rng);

        // Initialize the vertex shader.
        let renderer = Renderer::new(&config, &device, &params, &texture);

        // Set the initial state for all cells in the life grid.
        let cell_data: Vec<f32> = {
            let u = Uniform::new_inclusive(0.0, 1.0);
            u.sample_iter(&mut rng).take(ncells).collect()
        };

        life.import(&device, &queue, &cell_data);

        // Step the algorithm a few times, so the initial image looks Life-like.
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: None
            });
        for _ in 0..100 {
            life.step(&mut command_encoder);
        }
        queue.submit(Some(command_encoder.finish()));

        LifeProg {
            dim,
            life,
            renderer,
        }
    }

    /// called on WindowEvent::Resized events
    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let dim = Dimensions::new(config.width, config.height);
        log::info!("main: resizing {:?} -> {:?}", self.dim, dim);

        // Parameters for the game, shared between compute and fragment shaders.
        let params = LifeProg::params(device, dim);

        // Create a texture that's shared between compute and fragment shaders.
        let texture = Texture::new(device, dim, wgpu::TextureFormat::R32Float);

        // Resize the life algorithm.
        self.life.resize(device, queue, dim, &params, &texture);

        // Rebind the renderer to the new params and texture args.
        self.renderer.resize(config, device, &params, &texture);

        self.dim = dim;
    }

    /// called to generate each new frame
    fn render(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &window::Spawner,
    ) {
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: None
            });

        // Run the life algorithm one step.
        self.life.step(&mut command_encoder);

        // Render the life cells into actual pixels, and display them.
        self.renderer.render(&mut command_encoder, &view);

        queue.submit(Some(command_encoder.finish()));

        // XXX: create and destroy a texture, to confirm that this is still going
        // even when console output stops (this shows up in RUST_LOG=debug output)
        let z = self.life.frame_num();
        if z % 100 == 0 {
            let dim = Dimensions::new(100, 100 + z as u32 / 100);
            let _ = Texture::new(&device, dim, wgpu::TextureFormat::R32Float);
        }
    }

    /// called when a key is pressed
    fn key_press(
        &mut self,
        keycode: VirtualKeyCode,
    ) -> Option<WindowOps> {
        match keycode {
            VirtualKeyCode::Escape => Some(WindowOps::Quit),
            VirtualKeyCode::F => Some(WindowOps::FullScreen),
            VirtualKeyCode::W => Some(WindowOps::UnFullScreen),
            _ => None,
        }
    }

    /// called when a mouse button is pressed or released
    fn mouse_press(
        &mut self,
        _state: winit::event::ElementState,
        _button: winit::event::MouseButton,
    ) {
        // println!("mouse_press: state {:?}, button {:?}", _state, _button);
        // empty
    }

    /// called when the mouse is moved
    fn mouse_move(
        &mut self,
        _x: f64,
        _y: f64,
    ) {
        // println!("mouse_move: x {:?}, y {:?}", _x, _y);
        // empty
    }

    /// called for any WindowEvent not handled by the framework
    fn update(
        &mut self,
        _event: winit::event::WindowEvent
    ) {
        // empty
    }
}

/// run example
fn main() {
    window::run::<LifeProg>();
}
