[[block]]
struct CopyParams {
    odx: u32;
    ody: u32;
    ndx: u32;
    ndy: u32;
    owidth: u32;
    nwidth: u32;
    width: u32;
    height: u32;
};

[[block]]
struct SrcArray {
    src_cells: array<{src_type}>;
};

[[block]]
struct DstArray {
    dst_cells: array<{dst_type}>;
};

[[group(0), binding(0)]] var<uniform> params: CopyParams;
[[group(0), binding(1)]] var<storage, read> src: SrcArray;
[[group(0), binding(2)]] var<storage, read_write> dst: DstArray;

// ----------------------------------------------------------------------
// Kernel for copying {src_type}'s to {dst_type}'s.

[[stage(compute), workgroup_size(8, 8)]]
fn copy([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let X: u32 = global_id.x;
    let Y: u32 = global_id.y;

    if (X < params.width && Y < params.height) {
        let op: u32 = (Y + params.ody) * params.owidth + (X + params.odx);
        let np: u32 = (Y + params.ndy) * params.nwidth + (X + params.ndx);

        let old: {src_type} = src.src_cells[op];
        var new: {dst_type};
        {manip};
        dst.dst_cells[np] = new;
    }
}
