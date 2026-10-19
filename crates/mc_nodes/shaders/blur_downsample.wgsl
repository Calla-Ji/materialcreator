// SPDX-License-Identifier: MIT OR Apache-2.0
// blur_downsample.wgsl — Filters / Spatial / Blur (large-radius pyramid:
// downsample step)
//
// One 2x box-downsample step of Blur's own large-radius pyramid (see
// blur.rs's own doc comment for why it exists: `blur.wgsl`'s own sampling
// loop skip-samples once its radius exceeds 16px, so a large blur instead
// runs on a repeatedly-halved copy, always kept under that threshold).
// Reads `source_tex` at full size, writes `dest_tex` at half the
// resolution, averaging each 2x2 block — wrap-addressed so a tileable
// input stays tileable through the pyramid. Modeled on
// `mc_app::mipgen`'s own proven `mip_downsample.wgsl` box-filter arm,
// simplified: Blur's own working texture is always `rgba16float` (no
// second format to support) and Blur has never been normal-map-aware (no
// renormalization branch).
//
// Inputs:  `source_tex`.
// Output:  `dest_tex`.
// Params:  dest_size, wrap_x, wrap_y.

struct DownsampleParams {
    dest_size: u32,
    wrap_x: u32,
    wrap_y: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: DownsampleParams;
@group(0) @binding(1) var source_tex: texture_2d<f32>;
@group(0) @binding(2) var dest_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.dest_size || id.y >= params.dest_size) {
        return;
    }
    let src_size = i32(params.dest_size) * 2;
    let wrap_x = params.wrap_x != 0u;
    let wrap_y = params.wrap_y != 0u;
    let src_origin = vec2<i32>(id.xy) * 2;

    let x0 = tile_coord(src_origin.x, src_size, wrap_x);
    let x1 = tile_coord(src_origin.x + 1, src_size, wrap_x);
    let y0 = tile_coord(src_origin.y, src_size, wrap_y);
    let y1 = tile_coord(src_origin.y + 1, src_size, wrap_y);

    let c00 = textureLoad(source_tex, vec2<i32>(x0, y0), 0);
    let c10 = textureLoad(source_tex, vec2<i32>(x1, y0), 0);
    let c01 = textureLoad(source_tex, vec2<i32>(x0, y1), 0);
    let c11 = textureLoad(source_tex, vec2<i32>(x1, y1), 0);

    textureStore(dest_tex, vec2<i32>(id.xy), (c00 + c10 + c01 + c11) * 0.25);
}
