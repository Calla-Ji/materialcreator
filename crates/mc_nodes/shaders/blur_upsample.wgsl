// SPDX-License-Identifier: MIT OR Apache-2.0
// blur_upsample.wgsl — Filters / Spatial / Blur (large-radius pyramid:
// upsample step)
//
// The last step of Blur's own large-radius pyramid: one direct bilinear
// jump from the small, already-blurred level straight back to full
// resolution — not reversing the downsample chain step by step, which
// would compound several generations of incremental softening instead of
// one true interpolation over the real span.
//
// Manual 4-tap bilinear blend via `textureLoad`, not a `Sampler`/
// `textureSampleLevel` binding — every other shader in this crate reads
// via `textureLoad` with its own explicit wrap/clamp math, so this keeps
// that same convention rather than introducing the first sampler use in
// `mc_nodes` (a real compute-shader sampler precedent exists in
// `mc_app::hdri`, but mc_app-side and a one-off, not this crate's own
// pattern).
//
// Inputs:  `source_tex` (the small blurred level).
// Output:  `dest_tex` (full resolution).
// Params:  dest_size, src_size, wrap_x, wrap_y.

struct UpsampleParams {
    dest_size: u32,
    src_size: u32,
    wrap_x: u32,
    wrap_y: u32,
}

@group(0) @binding(0) var<uniform> params: UpsampleParams;
@group(0) @binding(1) var source_tex: texture_2d<f32>;
@group(0) @binding(2) var dest_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.dest_size || id.y >= params.dest_size) {
        return;
    }
    let wrap_x = params.wrap_x != 0u;
    let wrap_y = params.wrap_y != 0u;
    let src_size = i32(params.src_size);
    let scale = f32(params.src_size) / f32(params.dest_size);

    // Pixel-center-aligned mapping from a full-resolution destination texel
    // to its continuous position in the small source texture.
    let src_pos = (vec2<f32>(id.xy) + 0.5) * scale - 0.5;
    let base = vec2<i32>(floor(src_pos));
    let frac = fract(src_pos);

    let x0 = tile_coord(base.x, src_size, wrap_x);
    let x1 = tile_coord(base.x + 1, src_size, wrap_x);
    let y0 = tile_coord(base.y, src_size, wrap_y);
    let y1 = tile_coord(base.y + 1, src_size, wrap_y);

    let c00 = textureLoad(source_tex, vec2<i32>(x0, y0), 0);
    let c10 = textureLoad(source_tex, vec2<i32>(x1, y0), 0);
    let c01 = textureLoad(source_tex, vec2<i32>(x0, y1), 0);
    let c11 = textureLoad(source_tex, vec2<i32>(x1, y1), 0);

    let top = mix(c00, c10, frac.x);
    let bottom = mix(c01, c11, frac.x);
    textureStore(dest_tex, vec2<i32>(id.xy), mix(top, bottom, frac.y));
}
