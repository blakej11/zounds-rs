[[block]]
struct LifeParams {
    width : u32;
    height : u32;
    threshold : f32;
};

[[block]]
struct Cells {
    cells : array<f32>;
};

[[block]]
struct RandState {
    state : array<vec4<u32>>;
};

[[group(0), binding(0)]] var<uniform> params: LifeParams;
[[group(0), binding(1)]] var<storage, read> cellSrc: Cells;
[[group(0), binding(2)]] var<storage, read_write> cellDst: Cells;
[[group(0), binding(3)]] var<storage, read_write> randState: RandState;
[[group(0), binding(4)]] var texture: texture_storage_2d<r32float, write>;

fn LCGStep(z: u32, A: u32, C: u32) -> u32 {
    return (A * z + C);
}

fn TausStep(z: u32, S1: u32, S2: u32, S3: u32, M: u32) -> u32 {
    return ((((z << S1) ^ z) >> S2) ^ ((z & M) << S3));
}

fn generate_random(pix: u32) -> f32 {
    let ostate : vec4<u32> = randState.state[pix];
    var nstate : vec4<u32>;

    nstate.w = LCGStep(ostate.w, 1664525u32, 1013904223u32);        // p = 2^32
    nstate.x = TausStep(ostate.x, 13u32, 19u32, 12u32, u32( -2));   // p = 2^31 - 1
    nstate.y = TausStep(ostate.y,  2u32, 25u32,  4u32, u32( -8));   // p = 2^30 - 1
    nstate.z = TausStep(ostate.z,  3u32, 11u32, 17u32, u32(-16));   // p = 2^28 - 1

    randState.state[pix] = nstate;

    return (2.3283064365387e-10 * f32(nstate.x ^ nstate.y ^ nstate.z ^ nstate.w));
}

[[stage(compute), workgroup_size(8, 8)]]
fn life([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let X : u32 = global_id.x;
    let Y : u32 = global_id.y;
    let W : u32 = params.width;
    let H : u32 = params.height;
    let thresh : f32 = params.threshold;

    if (X > W || Y > H) {
        return;
    }

    var count : i32 = 0;
    for (var y : i32 = i32(Y - 1u32); y <= i32(Y + 1u32); y = y + 1) {
        for (var x : i32 = i32(X - 1u32); x <= i32(X + 1u32); x = x + 1) {
            let yw : u32 = u32(y + i32(H)) % H;
            let xw : u32 = u32(x + i32(W)) % W;
            if (cellSrc.cells[yw * W + xw] > thresh) {
                count = count + 1;
            }
        }
    }

    let pix : u32 = Y * W + X;
    let ov : f32 = cellSrc.cells[pix];
    let was_alive : bool = ov > thresh;
    var nv : f32;

    // in the first clause, "3 or 4" includes the center cell
    if (was_alive && (count == 3 || count == 4)) {
        if (ov - 0.01 > thresh) {
            nv = ov - 0.01;
        } else {
            nv = ov;
        }
    } else {
        if (!was_alive && count == 3) {
            nv = 1.0;
        } else {
            nv = thresh; // generate_random(pix) * thresh;
        }
    }

    cellDst.cells[pix] = nv;

    let coord : vec2<i32> = vec2<i32>(i32(X), i32(Y));
    // all channels other than the first are ignored
    let value : vec4<f32> = vec4<f32>(nv, 0.0, 0.0, 1.0);

    textureStore(texture, coord, value);
}
