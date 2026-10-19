// SPDX-License-Identifier: MIT OR Apache-2.0
// outline_blur.wgsl — Filters / Spatial / Outline (preblur/softness)
//
// One pass of a separable Gaussian blur — `mc_nodes::outline`'s own
// `dispatch_gaussian_2pass` runs this twice (horizontal, then vertical) for
// a full 2D blur, reused for both `preblur` (on the full-color input,
// before the select pass) and `softness` (on the grayscale-broadcast
// outline result, after resolve/rounding). Unlike `blur.wgsl`'s own dynamic
// step/tap scheme, this uses EDGE_OUTLINE_SPEC.md's own exact convention —
// integer support `ceil(3*sigma)`, weights `exp(-k²/(2σ²))` normalized to
// sum to one — since `preblur`/`softness` are tested against the research
// package's own exact Gaussian fixtures, not just "looks blurred."
// `mc_nodes::outline::dispatch` never calls this at `sigma_px == 0` (gated
// out entirely on the Rust side instead), so sigma is always > 0 here.
//
// `invert` only applies on the vertical pass (`direction == 1`) — the
// horizontal pass is never the pipeline's own last step.
//
// Inputs:  `input_tex`.
// Output:  `output`.
// Params:  resolution, direction, sigma_px, wrap, invert.

struct BlurParams {
    resolution: u32,
    direction: u32,
    sigma_px: f32,
    wrap: u32,
    invert: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: BlurParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    let sigma = params.sigma_px;
    let radius = i32(ceil(sigma * 3.0));
    let wrap = params.wrap != 0u;

    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var k = -radius; k <= radius; k = k + 1) {
        let weight = exp(-f32(k * k) / (2.0 * sigma * sigma));
        var coord = vec2<i32>(id.xy);
        if params.direction == 0u {
            coord.x = tile_coord(coord.x + k, size, wrap);
        } else {
            coord.y = tile_coord(coord.y + k, size, wrap);
        }
        sum = sum + textureLoad(input_tex, coord, 0) * weight;
        weight_sum = weight_sum + weight;
    }
    var result = sum / weight_sum;
    if (params.invert != 0u) {
        result = vec4<f32>(1.0 - result.rgb, result.a);
    }
    textureStore(output, vec2<i32>(id.xy), result);
}
