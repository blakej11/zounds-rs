struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] tex_coord: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(
    [[location(0)]] position: vec4<f32>,
    [[location(1)]] tex_coord: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = position;
    out.tex_coord = tex_coord;
    return out;
}

fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h: f32 = hsv.x * 6.0f;
    let s: f32 = hsv.y;
    let v: f32 = hsv.z;

    let w: i32 = i32(h);
    let f: f32 = h - f32(w);
    let p: f32 = v * (1.0f - s);
    let q: f32 = v * (1.0f - (s * f));
    let t: f32 = v * (1.0f - (s * (1.0f - f)));

    var r: f32;
    var g: f32;
    var b: f32;

    // workaround for Naga bugs listed below
    r = v; g = t; b = p;

    switch (w) {
        // "default: fallthrough;" doesn't work right now
        // https://github.com/gfx-rs/naga/issues/1088
        // "case 6: { fallthrough; }" also doesn't work right now
        // https://github.com/gfx-rs/naga/issues/1099
        case 0: { r = v; g = t; b = p; }
        case 1: { r = q; g = v; b = p; }
        case 2: { r = p; g = v; b = t; }
        case 3: { r = p; g = q; b = v; }
        case 4: { r = t; g = p; b = v; }
        case 5: { r = v; g = p; b = q; }
    }

    return vec3<f32>(r, g, b);
}

[[block]]
struct LifeParams {
    width: u32;
    height: u32;
    threshold: f32;
};

[[group(0), binding(0)]] var<uniform> params: LifeParams;
[[group(0), binding(1)]] var texture: texture_2d<f32>;
[[group(0), binding(2)]] var sampler: sampler;

fn render(val: f32) -> vec3<f32> {
    let thresh: f32 = params.threshold;

    // return vec3<f32>(val, val, val); // XXX for debugging
    if (val <= thresh) {
        return vec3<f32>(0f, 0f, 0f);
    } else {
        let a: f32 = (val - thresh) / (1.0f - thresh);
        let b: f32 = (1.0f - a) * 0.7f;
        return hsv_to_rgb(vec3<f32>(b, 1.0f, 1.0f));
    }
}

[[stage(fragment)]]
fn fs_main(
    in: VertexOutput
) -> [[location(0)]] vec4<f32> {
    let value: f32 = textureSample(texture, sampler, in.tex_coord).x;
    let rgb: vec3<f32> = render(value);
    return vec4<f32>(rgb.x, rgb.y, rgb.z, 1.0);
}
