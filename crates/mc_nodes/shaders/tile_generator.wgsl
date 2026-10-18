// SPDX-License-Identifier: MIT OR Apache-2.0
// tile_generator.wgsl — Generators / Patterns / Pattern Generator
//
// Inputs:  `shape_tex` — an optional custom tile shape (e.g. the Shape
//          node's output): when connected, this is sampled (rotation- and
//          size-jittered the same as the built-in shape) instead of the
//          built-in rounded rect, so `gap`/`bevel` have no effect — the
//          shape's own silhouette and softness are whatever was authored
//          upstream. `has_shape` says which case this dispatch is; if
//          nothing's connected, `mc_nodes::tile_generator::dispatch` still
//          binds *something* here (never actually read).
// Output:  `output` — a brick/grid pattern: a height-like value, bright on
//          each tile's raised face (softened at the edges by `bevel`), 0 in
//          the gap between tiles.
// Params:
//   rows, columns      — grid density
//   offset             — Grid layout: stagger alternate rows by this
//                        fraction of a column width (0 = plain grid, 0.5 =
//                        classic brick). NOTE: for the stagger to tile
//                        seamlessly top-to-bottom, `rows` must be even — an
//                        odd row count flips row parity across the vertical
//                        wrap, same constraint every brick generator has.
//                        Ignored entirely outside `distribution` = Both:
//                        staggering alternate rows only makes sense across a
//                        full grid, and would otherwise knock every other
//                        tile sideways out of a single centered column/row.
//                        Herringbone layout: repurposed entirely — see
//                        `pattern_layout`/`herringbone_value`'s own doc
//                        comments, below/elsewhere in this file.
//   gap                — mortar gap width, as a fraction of a tile. Only
//                        affects the built-in shape (see `shape_tex` above).
//   bevel              — how much of the tile's edge fades from full
//                        brightness to the gap. Only affects the built-in
//                        shape.
//   size_jitter        — per-tile random shrink, 0..1
//   rotation_jitter     — per-tile random rotation, 0..1 (small angles —
//                        this is a per-tile *cosmetic* tilt, not a precise
//                        physically-fitted brick edge)
//   luminance_jitter   — per-tile random brightness variation, 0..1
//   seed               — changes the per-tile randomness
//   distribution       — which grid axes actually get a tile: 0 = Both (the
//                        standard full rows x columns grid), 1 = Horizontal
//                        Only ("turn this into vertical stripes": columns
//                        repeat as usual, rows don't exist at all — every
//                        pixel shows the full, uncropped tile height, so a
//                        deliberately tall connected Shape (e.g. a big
//                        Scale Y stretch) comes through at its own real
//                        height instead of being squeezed into a small
//                        centered window), 2 = Vertical Only (the exact
//                        mirror: rows repeat, every pixel shows the full
//                        uncropped tile width). Column indices are always
//                        wrapped for their per-tile hash regardless of this
//                        value, independent of it — that's what keeps the
//                        staggered brick seam matching edge to edge, not a
//                        user-facing choice. Rotation Jitter is skipped
//                        outside Both: it only has one consistent geometric
//                        meaning when both axes repeat at the same scale,
//                        and Horizontal/Vertical Only deliberately mix one
//                        repeating (scaled) axis with one unscaled, full-
//                        tile axis.
//   shape_size         — only used when `shape_tex` is connected: how large
//                        each stamped shape is, as a fraction of the WHOLE
//                        TILE (same units Shape's own Size param uses) —
//                        not a fraction of each cell, so it stays the same
//                        absolute size as Tile Count (rows/columns) changes
//                        instead of silently shrinking as the grid gets
//                        finer. Internally this is still a zoom into the
//                        connected Shape's own canvas, applied to the
//                        REPEATING axis only (both axes, in Both mode; no
//                        cross-cell blending — a value larger than the
//                        current cell pitch just clips hard at the cell
//                        edge rather than spilling into neighbors) — the
//                        uncropped axis in Horizontal/Vertical Only ignores
//                        it entirely, by design (see `distribution` above).
//                        Built-in shape uses `gap` for this instead.
//   pattern_layout     — 0 Grid (the built-in rect path above), 1
//                        Herringbone — only takes effect with no Shape
//                        connected (`has_shape` == 0). Real interlocking
//                        parquet planks, each a fully bounded rectangle (not
//                        a continuous zigzag ribbon) — see `herringbone_value`'s
//                        own doc comment for the full derivation. `columns`
//                        alone sets this layout's own density (`rows` is
//                        unused/hidden — see `pack_params` on the Rust
//                        side); `offset` picks between Herringbone's own two
//                        standard orientations instead of staggering rows
//                        (see `herringbone_value`'s own doc comment).
//   plank_length,
//   plank_width        — Herringbone only: how much of its own fixed-size
//                        slot each plank fills, along its length and width
//                        respectively (1.0 = the slot's own full extent) —
//                        independent of each other AND of the lattice's own
//                        spacing, which never moves as these change. See
//                        `herringbone_value`'s own doc comment.

struct TileGeneratorParams {
    resolution: u32,
    rows: u32,
    columns: u32,
    seed: u32,
    offset: f32,
    gap: f32,
    bevel: f32,
    size_jitter: f32,
    rotation_jitter: f32,
    luminance_jitter: f32,
    distribution: u32,
    has_shape: u32,
    shape_size: f32,
    pattern_layout: u32,
    plank_length: f32,
    plank_width: f32,
}

@group(0) @binding(0) var<uniform> params: TileGeneratorParams;
@group(0) @binding(1) var shape_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

const LAYOUT_HERRINGBONE: u32 = 1u;

fn glsl_mod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

// The shear-fold construction below (see `herringbone_value`'s own doc
// comment) only has two real geometric degrees of freedom: overall scale,
// and one length:width ratio — there's no third slot for "fill a fixed cell
// smaller, leaving a gap" without it actually being a gap. An earlier
// version tried exactly that (Plank Length/Width as fill fractions of an
// otherwise-fixed slot) and it did create visible gaps instead of the
// pattern simply re-tiling with more/fewer pieces. The fold's own `ceil()`
// step also visibly degenerates as the ratio approaches 1 (near-square
// planks) — confirmed by an earlier version that fed the raw, unclamped
// ratio in and broke into jumbled diamond shapes around ratio \u{2248} 1.28 —
// so the ratio actually used is clamped to a range confirmed safe.
const HERRINGBONE_ASPECT_MIN: f32 = 1.5;
const HERRINGBONE_ASPECT_MAX: f32 = 6.0;

// Herringbone, ported from a published reference implementation (GLSL,
// "GLSL\u{3067}\u{30d8}\u{30ea}\u{30f3}\u{30dc}\u{30fc}\u{30f3}\u{6a21}\u{69d8}",
// zenn.dev/aadebdeb/articles/glsl-herringbone-pattern) rather than derived
// from scratch. Its own technique: SHEAR every row by its own row index
// (`p.x -= floor(p.y)`) — the classic brick-stagger trick, just continuous
// instead of per-discrete-row — then tile a `2*aspect \u{d7} 1` repeat cell
// holding two `aspect \u{d7} 1` plank slots side by side; the second slot gets
// transposed (`p.x`/`p.y` folded via `ceil`/`mod`) so its own plank reads as
// rotated 90\u{b0} from the first — the perpendicular-pair structure real
// parquet herringbone is built from. `plank_length`/`plank_width` set
// `aspect` (clamped, see `HERRINGBONE_ASPECT_MIN`/`_MAX` above) — changing
// either re-tiles the whole pattern with more or fewer, differently-
// proportioned pieces that still meet edge to edge (no gaps), rather than
// shrinking a fixed-size plank within an unchanged slot.
//
// `diagonal` picks which of Herringbone's two standard orientations this
// is: false runs the shear directly on `uv` (Running Bond, axis-aligned);
// true first rotates the coordinate frame 45\u{b0} — written as the sum/
// difference coordinate `(ux - uy, ux + uy)` rather than an explicit
// sin/cos rotation, since at exactly 45\u{b0} that IS the rotation (cos =
// sin = 1/\u{221a}2) with the \u{221a}2 normalization folded in, so `n` keeps the
// same meaning (planks-per-tile-width) in both orientations. Geometry-wise
// this single shared function is correct and seamless for both — the
// construction was hand-calibrated against `wood.mcgraph`'s own pre-built-
// in-layout recipe and confirmed visually before Herringbone had a
// diagonal option at all.
//
// The one place the two orientations genuinely differ beyond the rotation
// itself: the diagonal frame's `q.y` (`(ux + uy) * n`) sweeps TWICE the raw
// range of the axis-aligned `q.y` (`uy * n`) for the exact same visual
// density, because `ux + uy` itself ranges over `[0, 2)` across the unit
// tile where a single axis only ranges over `[0, 1)` — confirmed by direct
// pixel measurement (rendering raw `hash2()` output showed one hash value
// on EXACTLY double the pixel area of every other, not assumed from the
// math alone). That means `row` genuinely takes twice as many distinct
// values for the diagonal case — there are really twice as many parallel
// diagonal plank-bands crossing the tile as there are horizontal rows for
// the same `n` — so wrapping `row` for the hash with the axis-aligned
// case's modulus (`n`) silently folds two real, different bands onto the
// same hash bucket (every plank on band `k` getting an identical jitter
// twin on band `k + n`, which is what the doubled pixel area was). The fix
// is exactly that: wrap `row` with `select(n, 2.0 * n, diagonal)` instead
// of a single shared modulus, giving every one of the diagonal case's own
// genuinely-distinct bands its own hash bucket. `cell_col`/`slot`'s own
// wrap periods need no such change — the shear-fold's `2*aspect`-wide
// column period is identical in both orientations.
fn herringbone_value(
    uv: vec2<f32>,
    columns: f32,
    plank_length: f32,
    plank_width: f32,
    gap: f32,
    bevel: f32,
    seed: u32,
    size_jitter: f32,
    luminance_jitter: f32,
    diagonal: bool,
) -> f32 {
    let raw_aspect = plank_length / max(plank_width, 0.0001);
    let aspect = clamp(raw_aspect, HERRINGBONE_ASPECT_MIN, HERRINGBONE_ASPECT_MAX);
    let period_divisor = aspect * 2.0;
    let periods = max(round(columns / period_divisor), 1.0);
    let n = periods * period_divisor;

    var q: vec2<f32>;
    if diagonal {
        q = vec2<f32>(uv.x - uv.y, uv.x + uv.y) * n;
    } else {
        q = uv * n;
    }

    let row = floor(q.y);
    let sheared_x = q.x - row;
    let cell_col = floor(sheared_x / (2.0 * aspect));

    var p = vec2<f32>(glsl_mod(sheared_x, 2.0 * aspect), glsl_mod(q.y, 1.0));
    var slot = 0;
    // A transposed (slot 1) plank is `aspect` units long, but its length
    // runs along the ROW axis once rotated 90\u{b0} — so, whenever `aspect` >
    // 1, ONE physical plank actually spans multiple consecutive `row`
    // values (one 1-unit-tall slice of it per row it crosses), not just
    // the row the current pixel happens to land on. Hashing by raw `row`
    // gave each of those slices its OWN jitter — visible as banding INSIDE
    // a single long plank, caught by rendering with aspect pushed well
    // above 1 and luminance jitter maxed out. `hash_row` corrects this:
    // as `row` advances by 1 across consecutive slices of the SAME
    // physical plank, `ceil(qx)` (which slice-along-the-length this is)
    // decreases by exactly 1 in lockstep (both track the same shear), so
    // `row + ceil(qx)` is the same value for every slice of one plank —
    // a stable per-plank identity instead of a per-slice one. Slot 0 never
    // spans multiple rows (its long side runs along the un-rotated column
    // axis), so it keeps using `row` directly.
    var hash_row = row;
    if p.x >= aspect {
        let qx = p.x;
        let qy = p.y;
        hash_row = row + ceil(qx);
        p = vec2<f32>((2.0 * aspect - ceil(qx)) + qy, glsl_mod(qx, 1.0));
        slot = 1;
    }

    // Each domino holds 2 plank slots (the un-transposed first half and the
    // folded second half) — `slot` gives each its own hash, rather than
    // sharing one between both, so every individual plank gets its own
    // jitter the same way every other node's "per-tile" jitter already
    // does, not just every domino-pair. `row`'s own wrap modulus doubles
    // for the diagonal case — see this function's own doc comment above
    // for why — confirmed clean (no more area-doubling) by direct pixel
    // measurement after this fix, the same methodology that caught the bug.
    let row_period = max(i32(select(n, 2.0 * n, diagonal)), 1);
    let wrapped_row = wrap_coord(i32(hash_row), row_period);
    let wrapped_col = wrap_coord(i32(cell_col), max(i32(periods), 1));
    let cell = vec2<i32>(wrapped_col * 2 + slot, wrapped_row);
    let size_scale = 1.0 - size_jitter * hash2(cell, seed);
    let luminance = 1.0 - luminance_jitter * hash2(cell, seed + 211u);

    // Box test fills the canonical `[0, aspect] x [0, 1]` plank slot
    // completely (minus Gap/jitter) — the slot's own size already reflects
    // the current `plank_length`/`plank_width` ratio via `aspect` above, so
    // there's no separate fill-fraction here that would otherwise leave a
    // gap. Same sdf-plus-bevel-falloff shape the Grid layout's own built-in
    // rect already uses (`d`/`outside`/`inside`/`bevel_amount` below) —
    // `gap`/`bevel` are fractions of one plank WIDTH here (the slot's own
    // short side, `1` unit), not a fraction of the whole tile.
    let half_x = max(aspect * 0.5 * size_scale - gap, 0.0001);
    let half_y = max(0.5 * size_scale - gap, 0.0001);
    let d = vec2<f32>(abs(p.x - aspect * 0.5) - half_x, abs(p.y - 0.5) - half_y);
    let outside = length(max(d, vec2<f32>(0.0)));
    let inside = min(max(d.x, d.y), 0.0);
    let sdf = outside + inside;
    if sdf >= 0.0 {
        return 0.0;
    }
    let bevel_amount = clamp(-sdf / max(bevel, 0.0001), 0.0, 1.0);
    return luminance * mix(0.5, 1.0, bevel_amount);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);

    // Herringbone (either orientation) has no use for the Grid path's
    // row/column cell setup below — it builds its own domino lattice
    // directly from `uv` (see `herringbone_value`'s own doc comment) — so
    // it's handled entirely separately, before any of that. Only applies
    // with no Shape connected, same restriction Gap/Bevel already have on
    // the Grid path. `offset < 0.25` picks the 45\u{b0} diagonal orientation,
    // `>= 0.25` the axis-aligned Running Bond one — see the `offset` param
    // doc comment above `TileGeneratorParams`.
    if params.has_shape == 0u && params.pattern_layout == LAYOUT_HERRINGBONE {
        let value = herringbone_value(
            uv,
            f32(params.columns),
            params.plank_length,
            params.plank_width,
            params.gap,
            params.bevel,
            params.seed,
            params.size_jitter,
            params.luminance_jitter,
            params.offset < 0.25,
        );
        textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
        return;
    }

    let rows_f = f32(params.rows);
    let columns_f = f32(params.columns);
    let columns_i = i32(params.columns);
    let is_horizontal_only = params.distribution == 1u;
    let is_vertical_only = params.distribution == 2u;
    let is_both = !is_horizontal_only && !is_vertical_only;

    // Horizontal Only means "turn this into vertical stripes": columns is
    // the one repeating axis (density/zoom controlled by Tile Count/Shape
    // Size, below), and the row axis isn't cropped to some strip thickness
    // at all — it always shows the FULL, uncropped tile height, so whatever
    // the connected Shape's own vertical extent is (including a
    // deliberately tall Scale Y stretch) comes through unchanged, not
    // squeezed into a `shape_size`-tall window. Vertical Only is the exact
    // mirror (rows repeat, columns show the full width uncropped).
    // `row_idx`/`col_idx` still need *some* fixed value on the restricted
    // axis purely for per-tile hashing (Size/Rotation/Luminance Jitter) —
    // which one doesn't matter since there's only ever one visible
    // row/column there.
    var row_idx: i32;
    var local_y: f32;
    if is_horizontal_only {
        local_y = uv.y - 0.5;
        row_idx = 0;
    } else {
        row_idx = i32(floor(uv.y * rows_f));
        local_y = fract(uv.y * rows_f) - 0.5;
    }

    // Stagger only has a coherent meaning across a full grid — applying it
    // to alternating rows of a single centered column would knock every
    // other tile sideways out of that column, so it's skipped entirely
    // outside Both. Herringbone never reaches this code at all (see the
    // early return above) — this is Grid layout's own stagger.
    var x = uv.x * columns_f;
    if is_both && (row_idx % 2) != 0 {
        x = x + params.offset;
    }

    var col_idx: i32;
    var local_x: f32;
    if is_vertical_only {
        local_x = uv.x - 0.5;
        col_idx = 0;
    } else {
        col_idx = wrap_coord(i32(floor(x)), columns_i);
        local_x = fract(x) - 0.5;
    }

    let cell = vec2<i32>(col_idx, row_idx);

    let h = hash2(cell, params.seed);
    let h2 = hash2(cell, params.seed + 101u);
    let h3 = hash2(cell, params.seed + 211u);

    let size_scale = 1.0 - params.size_jitter * h;
    // Rotation only has one consistent geometric meaning across a uniform
    // grid where both axes repeat at the same scale (Both). Horizontal/
    // Vertical Only mix one repeating (scaled) axis with one unscaled,
    // full-tile axis — rotating would mix two different units — so it's
    // skipped there (angle 0), the same "small cosmetic tilt, not a
    // precise fit" scope Rotation Jitter already documents for itself.
    let angle = select(0.0, (h2 - 0.5) * params.rotation_jitter * 0.6, is_both);
    let luminance = 1.0 - params.luminance_jitter * h3;

    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let rotated = vec2<f32>(
        local_x * cos_a - local_y * sin_a,
        local_x * sin_a + local_y * cos_a,
    );

    var value = 0.0;
    if params.has_shape != 0u {
        let size = i32(params.resolution);
        // `shape_size` is tile-absolute (see the param doc comment
        // above) — scale by `columns_f` (== `rows_f` here, since
        // `effective_grid` on the Rust side always forces a square
        // grid once a Shape is connected) to turn that absolute target
        // into the right per-cell zoom for the current Tile Count. Only
        // the repeating axis is scaled by it; Horizontal/Vertical
        // Only's other axis samples the Shape's own canvas 1:1,
        // uncropped, over the whole tile.
        let effective_scale =
            max(size_scale * max(params.shape_size * columns_f, 0.0001), 0.0001);
        var sample_pos: vec2<f32>;
        if is_horizontal_only {
            let sx = clamp(rotated.x / effective_scale + 0.5, 0.0, 1.0);
            sample_pos = vec2<f32>(sx, uv.y);
        } else if is_vertical_only {
            let sy = clamp(rotated.y / effective_scale + 0.5, 0.0, 1.0);
            sample_pos = vec2<f32>(uv.x, sy);
        } else {
            sample_pos = clamp(
                rotated / effective_scale + vec2<f32>(0.5, 0.5),
                vec2<f32>(0.0, 0.0),
                vec2<f32>(1.0, 1.0),
            );
        }
        let coord = vec2<i32>(
            clamp(i32(sample_pos.x * f32(size)), 0, size - 1),
            clamp(i32(sample_pos.y * f32(size)), 0, size - 1),
        );
        value = textureLoad(shape_tex, coord, 0).r * luminance;
    } else {
        let half_size = vec2<f32>(0.5 - params.gap) * size_scale;
        let d = abs(rotated) - half_size;
        let outside = length(max(d, vec2<f32>(0.0)));
        let inside = min(max(d.x, d.y), 0.0);
        let sdf = outside + inside;

        if sdf < 0.0 {
            let bevel_amount = clamp(-sdf / max(params.bevel, 0.0001), 0.0, 1.0);
            value = luminance * mix(0.5, 1.0, bevel_amount);
        }
    }
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
