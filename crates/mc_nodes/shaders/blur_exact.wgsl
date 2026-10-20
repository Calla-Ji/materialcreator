// SPDX-License-Identifier: MIT OR Apache-2.0
// blur_exact.wgsl — Filters / Spatial / Blur (exact pass, sigma < 3px)
//
// A real, exact separable Gaussian tap sum — integer support `ceil(3*sigma)`
// (99.7% of a true Gaussian's own mass, unlike the old `blur.wgsl`'s own
// `±2σ` truncation), weights `exp(-k²/(2σ²))` normalized to sum to one.
// Same math as `outline_blur.wgsl`'s own preblur/softness pass (duplicated,
// not cross-called, per this crate's own convention of keeping each node's
// shaders self-contained), minus that one's `invert` flag, which Blur has
// no use for.
//
// Only used below `mc_nodes::blur::EXACT_MAX_SIGMA_PX` — a box blur's own
// integer-width quantization (see `blur_box.wgsl`) genuinely can't
// represent a sub-3px falloff well, but at that small a sigma the tap
// count here is tiny, so exact is both correct and cheap.
//
// Params:
//   direction — 0 = horizontal pass, 1 = vertical pass
//   sigma_px — the Gaussian's own standard deviation, in pixels.
//
// Inputs:  `input_tex`.
// Output:  `output`.

struct ExactBlurParams {
    resolution: u32,
    direction: u32,
    sigma_px: f32,
    wrap: u32,
}

@group(0) @binding(0) var<uniform> params: ExactBlurParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    let sigma = max(params.sigma_px, 0.001);
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
    textureStore(output, vec2<i32>(id.xy), sum / weight_sum);
}
