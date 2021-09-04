// Code to initialize the window and the GPU.
// based on https://github.com/gfx-rs/wgpu-rs/blob/master/examples/framework.rs
//   (licensed under https://choosealicense.com/licenses/mpl-2.0/)

use std::future::Future;
use std::rc::Rc;
use cfg_if::cfg_if;

use winit::{
    event::{self, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

#[cfg(not(target_arch = "wasm32"))]
use {
    std::time::{Duration, Instant},
    pollster::block_on,
};

#[cfg(target_arch = "wasm32")]
use {
    wasm_bindgen::{prelude::*, closure::Closure, JsCast, JsValue},
};

pub enum WindowOps {
    Quit,
    FullScreen,
    UnFullScreen,
}

pub trait Example: 'static + Sized {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::empty()
    }
    fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::default()
    }
    fn init(
        config: &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self;
    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    );
    fn key_press(
        &mut self,
        keycode: winit::event::VirtualKeyCode,
    ) -> Option<WindowOps>;
    fn mouse_press(
        &mut self,
        state: winit::event::ElementState,
        button: winit::event::MouseButton,
    );
    fn mouse_move(
        &mut self,
        x: f64,
        y: f64,
    );
    fn update(
        &mut self,
        event: WindowEvent,
    );
    fn render(
        &mut self,
        frame: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spawner: &Spawner,
    );
}

struct Setup {
    window: Rc<winit::window::Window>,
    event_loop: EventLoop<()>,
    instance: wgpu::Instance,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[cfg(target_arch = "wasm32")]
fn show_window_sizes(winit_window: &winit::window::Window)
{
    use winit::platform::web::WindowExtWebSys;

    let canvas = winit_window.canvas();

    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let body = document.body().unwrap();
    let root_element = document.document_element().unwrap();

    let  cwidth = canvas.client_width();
    let cheight = canvas.client_height();
    log::info!("window sizes:    canvas size {}x{}", cwidth, cheight);
    let  wwidth = window.inner_width().unwrap().as_f64().unwrap() as i32;
    let wheight = window.inner_height().unwrap().as_f64().unwrap() as i32;
    log::info!("window sizes:    window size {}x{}", wwidth, wheight);
    let  bwidth = body.client_width();
    let bheight = body.client_height();
    log::info!("window sizes:      body size {}x{}", bwidth, bheight);
    let  rwidth = root_element.client_width();
    let rheight = root_element.client_height();
    log::info!("window sizes: root elem size {}x{}", rwidth, rheight);
}


// Create the window and initialize the GPU.
// The program-specific initialization is done in start().
// Uses <E> for optional and required features, and for required limits.
async fn setup<E: Example>() -> Setup {
    // Initialize logging.
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            use winit::platform::web::WindowExtWebSys;

            console_log::init().expect("could not initialize logger");
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));

            let webwin = web_sys::window().unwrap();
            let document = webwin.document().unwrap();
            let body = document.body().unwrap();
            let root_element = document.document_element().unwrap();

            let event_loop = EventLoop::new();
            let window = winit::window::WindowBuilder::new()
                .build(&event_loop).unwrap();
            body.append_child(&web_sys::Element::from(window.canvas())).ok();

            // This seems to give the appropriate amount of room that the browser puts
            // on all sides of the canvas.
            //
            // I'm not at all clear on:
            // - why the client_height() property tells me this,
            //   but client_width() doesn't (!)
            // - why there doesn't seem to be some sort of DOM element that gives me
            //   the width and height I'm looking for directly
            let delta = root_element.client_height() - window.canvas().client_height();

            // I'd rather not have any border, but until I figure out how to
            // eliminate it, I can at least make it black.
            body.style().set_property("background-color", "black").unwrap();

            // Allow the winit window to be accessed from callbacks.
            let window = Rc::new(window);

            // A callback to determine how big the window is.
            let window_size = {
                let window = window.clone();
                move || {
                    show_window_sizes(&window);   // for debugging

                    let webwin = web_sys::window().unwrap();
                    let  width = webwin.inner_width().unwrap().as_f64().unwrap() as i32;
                    let height = webwin.inner_height().unwrap().as_f64().unwrap() as i32;
                    let  width = width - delta;
                    let height = height - delta;
                    log::info!("window::setup(): using new size {}x{}", width, height);
                    winit::dpi::LogicalSize::new(width, height)
                }};

            // Set the window's initial size.
            window.set_inner_size(window_size());

            // Resize the winit window whenever the browser window changes size.
            // (This generates "Resized" events which get passed to the Example.)
            {
                let window = window.clone();
                let closure = wasm_bindgen::closure::Closure::wrap(Box::new(
                    move |e: web_sys::Event| {
                        window.set_inner_size(window_size());
                    }) as Box<dyn FnMut(_)>);

                webwin
                    .add_event_listener_with_callback("resize",
                        closure.as_ref().unchecked_ref())
                    .unwrap();
                closure.forget();
            }

        } else {
            env_logger::init();

            let event_loop = EventLoop::new();
            let window = winit::window::WindowBuilder::new()
                .with_inner_size(winit::dpi::LogicalSize {
                    width: 1280,
                    height: 720,
                })
                .build(&event_loop).unwrap();
            let window = Rc::new(window);
        }
    }

    log::info!("Initializing the surface...");

    let backend = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);

    let instance = wgpu::Instance::new(backend);
    let (size, surface) = unsafe {
        let size = window.inner_size();
        let surface = instance.create_surface(&*window);
        (size, surface)
    };
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, backend)
        .await
        .expect("No suitable GPU adapters found on the system!");

    // this doesn't show anything useful under wasm
    #[cfg(not(target_arch = "wasm32"))]
    {
        let adapter_info = adapter.get_info();
        log::info!("Using {} ({:?})", adapter_info.name, adapter_info.backend);
    }

    let optional_features = E::optional_features();
    let required_features = E::required_features();
    let adapter_features = adapter.features();
    assert!(
        adapter_features.contains(required_features),
        "Adapter does not support required features for this example: {:?}",
        required_features - adapter_features
    );

    let needed_limits = E::required_limits();

    let trace_dir = std::env::var("WGPU_TRACE");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: (optional_features & adapter_features) | required_features,
                limits: needed_limits,
            },
            trace_dir.ok().as_ref().map(std::path::Path::new),
        )
        .await
        .expect("Unable to find a suitable GPU adapter!");

    Setup {
        window,
        event_loop,
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
    }
}

// Initialize the rest of the code, and run the event loop.
fn start<E: Example>(
    Setup {
        window,
        event_loop,
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
    }: Setup,
) {
    let spawner = Spawner::new();
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface.get_preferred_format(&adapter).unwrap(),
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };

    log::info!("Initializing the example...");
    let mut example = E::init(&config, &adapter, &device, &queue);
    log::info!("Example initialized.");

    // Force a no-op resize on native, to make sure it gets tested.
    #[cfg(not(target_arch = "wasm32"))]
    {
        log::info!("Performing no-op resize (non-wasm32)...");
        example.resize(&config, &device, &queue);
    }
    surface.configure(&device, &config);

    #[cfg(not(target_arch = "wasm32"))]
    let mut last_update_inst = Instant::now();
    #[cfg(not(target_arch = "wasm32"))]
    let mut last_frame_inst = Instant::now();
    #[cfg(not(target_arch = "wasm32"))]
    let (mut frame_count, mut accum_time) = (0, 0.0);

    log::info!("Entering render loop...");
    event_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapter); // force ownership by the closure
        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };

        match event {
            // fires after all RedrawRequested events have been processed,
            // and control flow is about to be taken away
            event::Event::RedrawEventsCleared => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    // Clamp to some max framerate to avoid busy-looping too much
                    // (we might be in wgpu::PresentMode::Mailbox, thus discarding superfluous frames)
                    //
                    // winit has window.current_monitor().video_modes() but that is a list of all full screen video modes.
                    // So without extra dependencies it's a bit tricky to get the max refresh rate we can run the window on.
                    // Therefore we just go with 60fps - sorry 120hz+ folks!
                    let target_frametime = Duration::from_secs_f64(1.0 / 60.0);
                    let time_since_last_frame = last_update_inst.elapsed();
                    if time_since_last_frame >= target_frametime {
                        window.request_redraw();
                        last_update_inst = Instant::now();
                    } else {
                        *control_flow = ControlFlow::WaitUntil(
                            Instant::now() + target_frametime - time_since_last_frame,
                        );
                    }

                    spawner.run_until_stalled();
                }

                #[cfg(target_arch = "wasm32")]
                window.request_redraw();
            }

            // fires when receiving a resize event from the OS
            event::Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                log::info!("Loop: Resized event: now {:?}", size);
                config.width = size.width.max(1);
                config.height = size.height.max(1);
                example.resize(&config, &device, &queue);
                surface.configure(&device, &config);
            },

            // fires when receiving some sort of window-related event from the OS
            event::Event::WindowEvent { event, .. } => match event {
                // fires when receiving a keypress
                WindowEvent::KeyboardInput { input, .. } => match input {
                    event::KeyboardInput {
                        virtual_keycode: Some(key),
                        state: event::ElementState::Pressed,
                        ..
                        // It would be nice to be able to pass along e.g. the
                        // state of the shift key here, but that's not kosher;
                        // cf. https://github.com/rust-windowing/winit/issues/1824
                    } => if let Some(op) = example.key_press(key) {
                        match op {
                            WindowOps::Quit =>
                                *control_flow = ControlFlow::Exit,
                            WindowOps::FullScreen =>
                                window.set_fullscreen(Some(
                                    winit::window::Fullscreen::Borderless(None)
                                )),
                            WindowOps::UnFullScreen =>
                                window.set_fullscreen(None),
                        }
                    },
                    // don't care about "virtual_keycode = None" or "state = Released"
                    _ => (),
                },

                // fires when receiving a mouse-down event
                WindowEvent::MouseInput { state, button, .. } => {
                    example.mouse_press(state, button);
                },

                // fires when receiving a mouse-move event
                WindowEvent::CursorMoved { 
                    position: winit::dpi::PhysicalPosition { x, y }, ..
                } => {
                    example.mouse_move(x, y);
                },

                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }

                _ => {
                    // all other events are passed along
                    example.update(event);
                }
            },

            // fires when a window should be redrawn
            event::Event::RedrawRequested(_) => {
                log::info!("Loop: Redraw-requested event");

                #[cfg(not(target_arch = "wasm32"))]
                {
                    accum_time += last_frame_inst.elapsed().as_secs_f32();
                    last_frame_inst = Instant::now();
                    frame_count += 1;
                    if frame_count == 100 {
                        println!(
                            "Avg frame time {}ms",
                            accum_time * 1000.0 / frame_count as f32
                        );
                        accum_time = 0.0;
                        frame_count = 0;
                    }
                }

                let frame = match surface.get_current_frame() {
                    Ok(frame) => frame,
                    Err(_) => {
                        surface.configure(&device, &config);
                        surface
                            .get_current_frame()
                            .expect("Failed to acquire next surface texture!")
                    }
                };
                let view = frame
                    .output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                example.render(&view, &device, &queue, &spawner);
            }
            _ => {}
        }
    });
}

// An object that can spawn a task, in both wasm and native forms.
#[cfg(not(target_arch = "wasm32"))]
pub struct Spawner<'a> {
    executor: async_executor::LocalExecutor<'a>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> Spawner<'a> {
    fn new() -> Self {
        Self {
            executor: async_executor::LocalExecutor::new(),
        }
    }

    #[allow(dead_code)]
    pub fn spawn_local(&self, future: impl Future<Output = ()> + 'a) {
        self.executor.spawn(future).detach();
    }

    fn run_until_stalled(&self) {
        while self.executor.try_tick() {}
    }
}

#[cfg(target_arch = "wasm32")]
pub struct Spawner {}

#[cfg(target_arch = "wasm32")]
impl Spawner {
    fn new() -> Self {
        Self {}
    }

    #[allow(dead_code)]
    pub fn spawn_local(&self, future: impl Future<Output = ()> + 'static) {
        wasm_bindgen_futures::spawn_local(future);
    }
}

// The main entry point, in both wasm and native forms.
// Invokes setup() and then start().
#[cfg(not(target_arch = "wasm32"))]
pub fn run<E: Example>() {
    let setup = block_on(setup::<E>());
    start::<E>(setup);
}

#[cfg(target_arch = "wasm32")]
pub fn run<E: Example>() {
    wasm_bindgen_futures::spawn_local(async move {
        let setup = setup::<E>().await;
        let start_closure = Closure::once_into_js(move || start::<E>(setup));

        // make sure to handle JS exceptions thrown inside start.
        // Otherwise wasm_bindgen_futures Queue would break and never handle any tasks again.
        // This is required, because winit uses JS exception for control flow to escape from `run`.
        if let Err(error) = call_catch(&start_closure) {
            let is_control_flow_exception = error.dyn_ref::<js_sys::Error>().map_or(false, |e| {
                e.message().includes("Using exceptions for control flow", 0)
            });

            if !is_control_flow_exception {
                web_sys::console::error_1(&error);
            }
        }

        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(catch, js_namespace = Function, js_name = "prototype.call.call")]
            fn call_catch(this: &JsValue) -> Result<(), JsValue>;
        }
    });
}
