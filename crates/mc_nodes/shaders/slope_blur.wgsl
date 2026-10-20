// SPDX-License-Identifier: MIT OR Apache-2.0
// slope_blur.wgsl — Filters / Spatial / Flow Blur
//
// Inputs:  `input_tex`   — the texture to blur.
//          `control_tex` — drives both the blur's direction (its local
//          gradient) and strength (its local slope/steepness) — steep areas
//          of the control map blur `input` more, flat areas don't blur it
//          at all. Key for organic/eroded looks.
// Output:  `output` — wrap-addressed so it stays tileable.
// Params:  radius_px — maximum blur radius in pixels, reached at the control
//          map's steepest point.
//          strength — multiplies the control map's own per-texel gradient
//          before it drives the blur amount. A `height`/noise-style control
//          map's neighboring-texel difference is tiny (a smooth hill a few
//          hundred texels wide differs from its neighbor by a few
//          thousandths), so without this the node was all but inert — the
//          max reach was never actually approached by any realistic input.
//          No single built-in constant can fit every control map's own
//          amplitude and frequency, so — same reasoning as Height to
//          Normal's own `intensity` — this is a real exposed param, not a
//          one-off tuning number.
//          mode — matches Substance's own Slope Blur "Mode": Blur (0) is the
//          plain weighted directional average below; Min (1)/Max (2) are a
//          directional grayscale erosion/dilation instead — the darkest or
//          lightest sample in the same reach, no weighting — which is what
//          actually gives the crisp, branching "eroded crack" look artists
//          reach for on rock/terrain, since the original value survives
//          rather than being blended away. Chaos (3) samples one
//          pseudo-random point within the same reach instead of reducing
//          over the whole footprint, for a scattered/torn look.
//
// Blur mode samples every texel along the gradient direction out to
// `ceil(3*sigma)` (the same exact, un-skipped Gaussian tap sum as
// `blur_exact.wgsl`, just along an arbitrary per-pixel direction instead of
// a fixed axis) — replacing an earlier version that grew its *step* past 1
// texel once sigma exceeded 16px to stay inside a fixed sample budget,
// which periodically skipped texels and produced the same comb/aliasing
// artifact `blur.wgsl` once had past its own old 16px radius. `blur.wgsl`'s
// own fix (three successive box blurs) needs a shared axis-aligned sliding
// window across the whole image, which doesn't exist here — each pixel's
// own blur direction differs, so there's no window to share. `MAX_TAPS`
// below instead caps the *reach*, not the sample density: past it, the
// blur just slightly under-reaches the Gaussian's own tail (a soft quality
// cap), never skip-samples. Min/Max use the same un-skipped sampling out to
// `ceil(sigma)` (no `*3` — a hard morphological footprint has no Gaussian
// tail to capture, so the footprint is just `sigma` itself, keeping
// `radius_px`'s own "max reach" meaning consistent across every mode).
const MAX_TAPS: i32 = 64;

const MODE_BLUR: u32 = 0u;
const MODE_MIN: u32 = 1u;
const MODE_MAX: u32 = 2u;
const MODE_CHAOS: u32 = 3u;

struct SlopeBlurParams {
    resolution: u32,
    radius_px: f32,
    strength: f32,
    tiling_mode: u32,
    mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: SlopeBlurParams;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var control_tex: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;

fn control_at(x: i32, y: i32, size: i32, wrap_x: bool, wrap_y: bool) -> f32 {
    return textureLoad(
        control_tex,
        vec2<i32>(tile_coord(x, size, wrap_x), tile_coord(y, size, wrap_y)),
        0,
    ).r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let size = i32(params.resolution);
    let x = i32(id.x);
    let y = i32(id.y);
    let wrap_x = wraps_x(params.tiling_mode);
    let wrap_y = wraps_y(params.tiling_mode);

    let left = control_at(x - 1, y, size, wrap_x, wrap_y);
    let right = control_at(x + 1, y, size, wrap_x, wrap_y);
    let down = control_at(x, y - 1, size, wrap_x, wrap_y);
    let up = control_at(x, y + 1, size, wrap_x, wrap_y);
    let grad = vec2<f32>(right - left, up - down) * 0.5;
    let slope = length(grad);

    let sigma = params.radius_px * clamp(slope * params.strength, 0.0, 1.0);
    if sigma < 0.5 {
        textureStore(output, vec2<i32>(id.xy), textureLoad(input_tex, vec2<i32>(id.xy), 0));
        return;
    }

    let dir = grad / slope;

    if params.mode == MODE_CHAOS {
        // One pseudo-random point within the same reach, instead of
        // reducing over the whole footprint — `hash2` (common.wgsl) keyed
        // on the pixel's own integer coordinate is already how every other
        // seeded node in this crate gets deterministic, tileable per-texel
        // randomness (e.g. White Noise), reused as-is.
        let r = hash2(vec2<i32>(x, y), 0u) * 2.0 - 1.0;
        let dist = r * sigma;
        let offset = dir * dist;
        let coord = vec2<i32>(
            tile_coord(x + i32(round(offset.x)), size, wrap_x),
            tile_coord(y + i32(round(offset.y)), size, wrap_y),
        );
        textureStore(output, vec2<i32>(id.xy), textureLoad(input_tex, coord, 0));
        return;
    }

    if params.mode == MODE_MIN || params.mode == MODE_MAX {
        let taps = min(i32(ceil(sigma)), MAX_TAPS);
        var result = textureLoad(input_tex, vec2<i32>(id.xy), 0);
        for (var i = -taps; i <= taps; i = i + 1) {
            let dist = f32(i);
            let offset = dir * dist;
            let coord = vec2<i32>(
                tile_coord(x + i32(round(offset.x)), size, wrap_x),
                tile_coord(y + i32(round(offset.y)), size, wrap_y),
            );
            let sample = textureLoad(input_tex, coord, 0);
            if params.mode == MODE_MIN {
                result = min(result, sample);
            } else {
                result = max(result, sample);
            }
        }
        textureStore(output, vec2<i32>(id.xy), result);
        return;
    }

    // MODE_BLUR (also the fallback for any unrecognized value).
    let taps = min(i32(ceil(sigma * 3.0)), MAX_TAPS);

    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var i = -taps; i <= taps; i = i + 1) {
        let dist = f32(i);
        let weight = exp(-(dist * dist) / (2.0 * sigma * sigma));
        let offset = dir * dist;
        let coord = vec2<i32>(
            tile_coord(x + i32(round(offset.x)), size, wrap_x),
            tile_coord(y + i32(round(offset.y)), size, wrap_y),
        );
        sum = sum + textureLoad(input_tex, coord, 0) * weight;
        weight_sum = weight_sum + weight;
    }
    textureStore(output, vec2<i32>(id.xy), sum / weight_sum);
}
