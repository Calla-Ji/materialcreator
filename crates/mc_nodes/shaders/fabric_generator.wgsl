// SPDX-License-Identifier: MIT OR Apache-2.0
// fabric_generator.wgsl — Generators / Patterns / Fabric Generator
//
// Inputs:  `fiber_tex` — a single fiber's own pattern (see
//          `thread_shape.wgsl`/the Thread node): Y is cross-section, X is
//          position along the fiber's length (constant along X for an
//          untwisted single-ply fiber; it only actually varies along X once
//          Thread has 2+ twisted plies) — sampled as the thread shape for
//          Plain Weave, Twill, and Rib Knit's straight thread segments, with
//          each grid cell's own local coordinate standing in for "how far
//          along this cell's stretch of the thread" (see `fiber_height`'s
//          own doc comment for why that's seamless cell-to-cell with no
//          extra bookkeeping). If nothing's wired into Thread in the graph,
//          `mc_nodes::fabric_generator::dispatch` renders one internally
//          from `thread_width`/`thread_roundness` below, so this is never
//          actually unbound. Stockinette's loops aren't straight thread
//          segments with a cross-section — it keeps its own built-in "V"
//          stitch shape regardless of what's wired into Fiber.
//          `ply_id_tex` — Thread's own Ply ID output (see
//          `thread_shape.wgsl`), sampled at the exact same coordinates as
//          whichever `fiber_tex` sample actually won at each pixel, so a
//          downstream Gradient Map can color each ply consistently with the
//          woven structure. If nothing's wired into the new "Thread Ply ID"
//          input, `dispatch` binds a flat "single ply" fallback.
// Output:  `output` — a height-like value tracing a woven or knit fabric's
//          thread structure, 0 in the valleys between threads up to 1 on a
//          thread's own ridge. Chain into Height to Normal the same way
//          Pattern Generator's output is meant to be used.
//          `ply_id_out` — the winning sample's own Ply ID, carried straight
//          through unmodified. Always 0 for Stockinette, which never
//          samples a fiber cross-section at all (its own built-in "V"
//          stitch shape has no ply data to report).
// Params:
//   weave_type         — 0 Plain Weave, 1 Twill, 2 Herringbone, 3 Rib Knit,
//                         4 Stockinette (Jersey Knit)
//   rows, columns       — thread/stitch density
//   thread_width        — Stockinette's own loop width, 0..1; also what
//                         `dispatch` renders the fallback Fiber at when
//                         nothing's connected to it
//   thread_roundness    — only affects the fallback Fiber `dispatch` renders
//                         when nothing's connected (has no effect on
//                         Stockinette, which doesn't use a thread
//                         cross-section at all)
//   thread_group_size   — Plain Weave only: threads per group acting as one
//                         woven unit (1 = plain weave, 2+ = basket weave)
//   twill_count         — Twill and Herringbone: the diagonal stripe's
//                         repeat period; Herringbone also reuses it as the
//                         width (in columns) of each band before the
//                         diagonal's direction reverses
//   rib_width           — Rib Knit only: columns per rib stripe
//   size_jitter,
//   luminance_jitter,
//   seed                — per-thread/stitch randomness, same convention as
//                         Pattern Generator
//
// Seamless tiling needs rows/columns to divide evenly by whatever period
// each weave type's own over/under or stagger pattern repeats on — same
// constraint, and same reason, as Pattern Generator's "rows must be even"
// note (see `mc_nodes::tile_generator`): Plain Weave/Basket Weave need rows
// AND columns even; Twill needs both a multiple of `twill_count`;
// Herringbone needs rows a multiple of `twill_count` (same as Twill) AND
// columns a multiple of `twill_count * 2` (an even number of reversed
// bands, so the band parity lines up again across the wrap); Rib Knit needs
// columns a multiple of `rib_width * 2`; Stockinette needs rows even (it
// reuses Pattern Generator's own running-bond stagger). The defaults below
// satisfy every one of these at once (24 is a multiple of both 4 and 8).

struct FabricGeneratorParams {
    resolution: u32,
    rows: u32,
    columns: u32,
    seed: u32,
    weave_type: u32,
    thread_width: f32,
    thread_roundness: f32,
    thread_group_size: u32,
    twill_count: u32,
    rib_width: u32,
    size_jitter: f32,
    luminance_jitter: f32,
}

@group(0) @binding(0) var<uniform> params: FabricGeneratorParams;
@group(0) @binding(1) var fiber_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var ply_id_tex: texture_2d<f32>;
@group(0) @binding(4) var ply_id_out: texture_storage_2d<rgba16float, write>;

const WEAVE_PLAIN: u32 = 0u;
const WEAVE_TWILL: u32 = 1u;
const WEAVE_HERRINGBONE: u32 = 2u;
const WEAVE_RIB_KNIT: u32 = 3u;
const WEAVE_STOCKINETTE: u32 = 4u;

fn wrap_i(v: i32, n: i32) -> i32 {
    return ((v % n) + n) % n;
}

// Reads `fiber_tex` at cross-section distance `t` (-0.5..0.5, `width_scale`
// stretching/shrinking which part of the fiber's own profile that maps to —
// the per-crossing jitter's stand-in for narrowing/widening an actual
// thread) and `along_frac` (0..1, this cell's fractional position along the
// thread's own length — see `fiber_tex`'s own doc comment on why each
// cell's independent `fract()` already lines up seamlessly cell-to-cell
// without any extra global coordinate).
fn fiber_height(t: f32, along_frac: f32, width_scale: f32) -> f32 {
    let size = i32(params.resolution);
    let sample_t = clamp(t / max(width_scale, 0.0001), -0.5, 0.5);
    let y = clamp(i32((sample_t + 0.5) * f32(size)), 0, size - 1);
    let x = clamp(i32(fract(along_frac) * f32(size)), 0, size - 1);
    return textureLoad(fiber_tex, vec2<i32>(x, y), 0).r;
}

// Same coordinate mapping as `fiber_height`, sampling `ply_id_tex` instead
// — call it with the exact same `(t, along_frac, width_scale)` as whichever
// `fiber_height` call it should carry the ply data for.
fn fiber_ply_id(t: f32, along_frac: f32, width_scale: f32) -> f32 {
    let size = i32(params.resolution);
    let sample_t = clamp(t / max(width_scale, 0.0001), -0.5, 0.5);
    let y = clamp(i32((sample_t + 0.5) * f32(size)), 0, size - 1);
    let x = clamp(i32(fract(along_frac) * f32(size)), 0, size - 1);
    return textureLoad(ply_id_tex, vec2<i32>(x, y), 0).r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);
    let columns = max(params.columns, 1u);
    let rows = max(params.rows, 1u);

    let col = i32(floor(uv.x * f32(columns)));
    let row = i32(floor(uv.y * f32(rows)));
    let xp = fract(uv.x * f32(columns)) - 0.5;
    let yp = fract(uv.y * f32(rows)) - 0.5;

    // Per-crossing jitter: a touch of per-thread width/brightness variation
    // so the weave doesn't read as a mechanically perfect grid.
    let cell = vec2<i32>(col, row);
    let h1 = hash2(cell, params.seed);
    let h2 = hash2(cell, params.seed + 101u);
    let width_scale = 1.0 - params.size_jitter * h1;
    let luminance = 1.0 - params.luminance_jitter * h2;

    var value = 0.0;
    var ply = 0.0;
    if params.weave_type == WEAVE_RIB_KNIT {
        let rib_width = max(i32(params.rib_width), 1);
        let raised = wrap_i(col, rib_width * 2) < rib_width;
        var h = fiber_height(xp, yp + 0.5, width_scale);
        // A gentle per-row undulation hints at a knit column's stacked
        // loops without simulating each individual one.
        h = h * (0.9 + 0.1 * cos(yp * 6.28318530718));
        value = select(h * 0.45, h, raised);
        ply = fiber_ply_id(xp, yp + 0.5, width_scale);
    } else if params.weave_type == WEAVE_STOCKINETTE {
        // Reuses Pattern Generator's own running-bond stagger (alternate
        // rows offset by half a column) so loops interlock the way real
        // stockinette stitches do, instead of lining up in a plain grid.
        var x = uv.x * f32(columns);
        if wrap_i(row, 2) != 0 {
            x = x + 0.5;
        }
        let loop_col = i32(floor(x));
        let lx = fract(x) - 0.5;
        let ly = yp;
        let loop_cell = vec2<i32>(wrap_i(loop_col, i32(columns)), row);
        let lh1 = hash2(loop_cell, params.seed);
        let lh2 = hash2(loop_cell, params.seed + 101u);
        let loop_width_scale = 1.0 - params.size_jitter * lh1;
        // A squashed rhombus — the classic simplified "V" stitch silhouette
        // stockinette knitting is known for. Not a thread cross-section, so
        // `fiber_tex` plays no part here.
        let d = abs(lx) * 2.2 + abs(ly) * 1.3;
        let edge = clamp(params.thread_width, 0.05, 1.0) * loop_width_scale;
        value = clamp(1.0 - d / edge, 0.0, 1.0) * (1.0 - params.luminance_jitter * lh2);
        // Stockinette never samples a fiber cross-section — no ply data.
        textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
        textureStore(ply_id_out, vec2<i32>(id.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    } else {
        // Plain Weave, Twill, and Herringbone all cross a vertical and a
        // horizontal thread at every cell; they only differ in which thread
        // is "on top" at a given crossing.
        let v_h = fiber_height(xp, yp + 0.5, width_scale);
        let h_h = fiber_height(yp, xp + 0.5, width_scale);
        let v_ply = fiber_ply_id(xp, yp + 0.5, width_scale);
        let h_ply = fiber_ply_id(yp, xp + 0.5, width_scale);

        var vertical_on_top: bool;
        if params.weave_type == WEAVE_TWILL || params.weave_type == WEAVE_HERRINGBONE {
            let n = max(i32(params.twill_count), 2);
            var diagonal = col - row;
            if params.weave_type == WEAVE_HERRINGBONE {
                // Reverse the diagonal's own direction every `n` columns —
                // the steady Twill stripe zigzags into the classic
                // herringbone chevron instead of running one direction
                // forever.
                if wrap_i(col / n, 2) != 0 {
                    diagonal = -diagonal;
                }
            }
            vertical_on_top = wrap_i(diagonal, n) < n / 2;
        } else {
            let group = max(i32(params.thread_group_size), 1);
            vertical_on_top = wrap_i(col / group + row / group, 2) == 0;
        }

        var top_h = h_h;
        var bottom_h = v_h;
        var top_ply = h_ply;
        var bottom_ply = v_ply;
        if vertical_on_top {
            top_h = v_h;
            bottom_h = h_h;
            top_ply = v_ply;
            bottom_ply = h_ply;
        }
        // The "under" thread is visibly recessed relative to the "over"
        // thread at each crossing — the ridge pattern that makes this read
        // as woven rather than a flat grid.
        value = max(top_h, bottom_h * 0.55);
        // Ply ID follows whichever sample actually won the max() above, not
        // just `vertical_on_top` — the "under" thread can still win if its
        // own recessed height is brighter than the "over" thread's.
        ply = select(bottom_ply, top_ply, top_h >= bottom_h * 0.55);
    }

    value = clamp(value * luminance, 0.0, 1.0);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
    textureStore(ply_id_out, vec2<i32>(id.xy), vec4<f32>(ply, ply, ply, 1.0));
}
