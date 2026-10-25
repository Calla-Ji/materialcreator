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
//          `distribution_tex` — an optional greyscale Distribution Map:
//          when connected, biases which tiles survive at all (see
//          `distribution_survives`'s own doc comment) — brighter areas of
//          the map keep more tiles, darker areas drop more. `has_distribution_
//          map` says which case this dispatch is, same convention as
//          `has_shape` above (a throwaway texture is bound either way).
//          Grid layout only — Herringbone's own sheared/folded coordinate
//          system has no simple per-plank cell center to sample the map
//          at, a real, stated limitation, not an oversight.
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
//   position_jitter    — per-tile random offset from its own cell center,
//                        0..1 of the cell's own half-size (see
//                        `POSITION_JITTER_MAX`). Both distribution only,
//                        same reason/scope as Rotation Jitter above.
//   position_seed      — independent of `seed`, same idea as
//                        `luminance_seed`, for `position_jitter`'s own
//                        per-tile draw.
//   distribution_amount,
//   distribution_seed  — see `distribution_survives`'s own doc comment.
//                        Only matter once `distribution_tex` is connected.
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
    // Independent of `seed` — see `mc_nodes::tile_generator::
    // TileGeneratorParams::luminance_seed`'s own doc comment.
    luminance_seed: u32,
    // See `mc_nodes::tile_generator::OverlapBlendMode`'s own doc comment —
    // Grid layout only, only visible once Gap goes negative.
    overlap_blend_mode: u32,
    overlap_opacity: f32,
    // Per-tile random offset from its own cell center. `Both` distribution
    // only — same scope as rotation_jitter, same reason (see
    // `grid_rect_value`'s own doc comment).
    position_jitter: f32,
    position_seed: u32,
    // How strongly `distribution_tex` biases which tiles survive — see
    // `distribution_survives`'s own doc comment. Only has any effect once
    // `has_distribution_map` is nonzero.
    distribution_amount: f32,
    distribution_seed: u32,
    has_distribution_map: u32,
    // Chance each tile is independently mirrored per axis — only wired
    // into `shape_cell_value` (see its own doc comment on why mirroring
    // the built-in rect would always be a mathematical no-op).
    mirror_jitter: f32,
    mirror_seed: u32,
    // Pure padding — WGSL's uniform address space requires this struct's
    // own size to be a multiple of 16 bytes; the fields above land on 104,
    // 8 bytes short of the next multiple (112).
    _pad1: u32,
    _pad2: u32,
}

const OVERLAP_ADD: u32 = 0u;
const OVERLAP_MAX: u32 = 1u;
const OVERLAP_ADD_SUB: u32 = 2u;

@group(0) @binding(0) var<uniform> params: TileGeneratorParams;
@group(0) @binding(1) var shape_tex: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var distribution_tex: texture_2d<f32>;

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
    luminance_seed: u32,
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
    let luminance = 1.0 - luminance_jitter * hash2(cell, luminance_seed + 211u);

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

// How far Position Jitter can nudge a tile's own center, as a fraction of
// a full cell width/height, at jitter == 1 (so `(h - 0.5) * jitter * this`
// reaches at most half this value in either direction). 1.0 lets a maxed-
// out tile reach right up to its neighbor's own cell boundary — a real,
// clearly visible scatter, not just a subtle nudge — while still keeping
// `grid_rect_value`'s own ±1-cell negative-Gap overlap search (see its own
// doc comment) a reasonable approximation: beyond this, a jittered tile
// could land outside the neighborhood that search actually checks.
// Raised from an earlier 0.3 after the owner reported it read as "barely
// moved" even at jitter == 1 — confirmed by the numbers: 0.3 only ever let
// a tile wander ±0.15 of a cell, visually subtle by design but not what
// was asked for here.
const POSITION_JITTER_MAX: f32 = 1.0;

// Same idea as `POSITION_JITTER_MAX`, for Rotation Jitter's own max swing
// at jitter == 1: `(h2 - 0.5) * jitter * this` reaches at most half this
// value either way, so this constant being ~π means a maxed-out tile can
// swing a full ±90°. Raised from an earlier 0.6 (±17°) for the same
// reported reason as Position Jitter above — a "cosmetic small tilt" read
// as basically motionless at the slider's own max value.
const ROTATION_JITTER_MAX: f32 = 3.14159265;

// Size Jitter's own max GROW multiplier at jitter == 1 (the shrink side
// keeps reaching all the way down to 0, unchanged from before this
// existed — only the grow side is new). Deliberately smaller than a full
// symmetric ±1.0 would give: Gap's own most-negative value (-0.4) already
// grows a cell's half-size to 0.9 on its own (see Gap's own doc comment on
// why that's chosen specifically to stay "under 1.5 cells" — the ±1-cell
// neighbor search both `grid_rect_value` and `shape_cell_value` do relies
// on that bound). 0.9 x 1.5 = 1.35, safely under 1.5 even combined with
// Gap at its own most extreme — a plain symmetric ±1.0 grow (0.9 x 2.0 =
// 1.8) would have silently broken that assumption instead. Grid layout
// ("Both") only, same restriction as Rotation/Position Jitter and for the
// same reason: Horizontal/Vertical Only's own single-cell test (and a
// connected Shape's own version of it) has no neighbor search to fall
// back on, so growth there would just clip at the cell edge instead of
// spilling over cleanly.
const SIZE_JITTER_GROW_MAX: f32 = 1.5;

// Whether the tile at `cell` (sampled at `cell_uv`, its own center in the
// *output's* 0..1 uv space — the same space `distribution_tex` covers,
// since it's just another Gray input at the same resolution) survives the
// Distribution Map gate. `distribution_amount` mixes between "always
// survives" (0, map ignored) and "survives with probability exactly equal
// to the map's own brightness there" (1, white always keeps a tile, black
// always drops it) — a plain per-cell coin flip (`hash2`, its own
// independent seed) against that probability, not a hard per-pixel
// threshold, so the pattern thins out gradually rather than cutting a hard
// edge at some brightness value. Always true with nothing wired into
// Distribution Map (`has_distribution_map == 0u`) or Amount at 0 — the
// texture lookup itself is skipped in that case, not just discounted to
// zero effect, since nothing meaningful is bound there anyway (see
// `dispatch`'s own placeholder-texture doc comment).
fn distribution_survives(
    cell_uv: vec2<f32>,
    cell: vec2<i32>,
    has_distribution_map: u32,
    distribution_amount: f32,
    distribution_seed: u32,
) -> bool {
    if has_distribution_map == 0u || distribution_amount <= 0.0 {
        return true;
    }
    let size = i32(params.resolution);
    let coord = vec2<i32>(
        clamp(i32(cell_uv.x * f32(size)), 0, size - 1),
        clamp(i32(cell_uv.y * f32(size)), 0, size - 1),
    );
    let map_value = textureLoad(distribution_tex, coord, 0).r;
    let survive_probability = mix(1.0, map_value, distribution_amount);
    let roll = hash2(cell, distribution_seed + 521u);
    return roll < survive_probability;
}

// The built-in rect's own value at `local` (a pixel's position relative to
// one cell's own center, in cell-local units — can be outside [-0.5, 0.5]
// when testing whether a NEIGHBORING cell, grown past its own cell by a
// negative Gap, reaches this pixel). `-1.0` means "not covered by this
// cell at all" — distinguishes genuine non-coverage from a real but low
// luminance value, which plain `0.0` alone can't, and is also what a
// Distribution Map gate failure (or Position Jitter pushing the tile off
// this pixel) reports, the same "nothing here" signal either way.
fn grid_rect_value(
    local: vec2<f32>,
    cell: vec2<i32>,
    cell_uv: vec2<f32>,
    is_both: bool,
    gap: f32,
    bevel: f32,
    size_jitter: f32,
    rotation_jitter: f32,
    luminance_jitter: f32,
    position_jitter: f32,
    seed: u32,
    luminance_seed: u32,
    position_seed: u32,
    has_distribution_map: u32,
    distribution_amount: f32,
    distribution_seed: u32,
) -> f32 {
    if !distribution_survives(cell_uv, cell, has_distribution_map, distribution_amount, distribution_seed) {
        return -1.0;
    }
    let h = hash2(cell, seed);
    let h2 = hash2(cell, seed + 101u);
    let h3 = hash2(cell, luminance_seed + 211u);
    let h4 = hash2(cell, position_seed + 311u);
    let h5 = hash2(cell, position_seed + 433u);
    // Grid layout ("Both"): shrinks toward 0 or grows up to
    // `SIZE_JITTER_GROW_MAX`, `h` picking linearly between the two — see
    // that constant's own doc comment for why growth is capped lower than
    // the shrink side goes. Outside Both: shrink-only, unchanged, for the
    // same structural reason Rotation/Position Jitter are also restricted
    // there.
    let size_scale = select(
        1.0 - size_jitter * h,
        mix(1.0 - size_jitter, 1.0 + size_jitter * (SIZE_JITTER_GROW_MAX - 1.0), h),
        is_both,
    );
    let angle = select(0.0, (h2 - 0.5) * rotation_jitter * ROTATION_JITTER_MAX, is_both);
    let luminance = 1.0 - luminance_jitter * h3;
    let position_offset = select(
        vec2<f32>(0.0),
        vec2<f32>(h4 - 0.5, h5 - 0.5) * position_jitter * POSITION_JITTER_MAX,
        is_both,
    );
    let shifted = local - position_offset;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let rotated = vec2<f32>(
        shifted.x * cos_a - shifted.y * sin_a,
        shifted.x * sin_a + shifted.y * cos_a,
    );
    let half_size = vec2<f32>(0.5 - gap) * size_scale;
    let d = abs(rotated) - half_size;
    let outside = length(max(d, vec2<f32>(0.0)));
    let inside = min(max(d.x, d.y), 0.0);
    let sdf = outside + inside;
    if sdf >= 0.0 {
        return -1.0;
    }
    let bevel_amount = clamp(-sdf / max(bevel, 0.0001), 0.0, 1.0);
    return luminance * mix(0.5, 1.0, bevel_amount);
}

// Folds one more overlapping cell's own value `v` into the running `acc` —
// see `mc_nodes::tile_generator::OverlapBlendMode`'s own doc comment for
// what each mode means. `OVERLAP_MAX` is also the fallback for any
// out-of-range value, same convention as this file's other enum-as-u32
// params.
fn combine_overlap(mode: u32, acc: f32, v: f32) -> f32 {
    if mode == OVERLAP_ADD {
        return clamp(acc + v, 0.0, 1.0);
    } else if mode == OVERLAP_ADD_SUB {
        return clamp(acc + (v - 0.5) * 2.0, 0.0, 1.0);
    }
    return max(acc, v);
}

// What `cell`'s own (jittered) connected-Shape stamp shows at a pixel whose
// offset from THAT cell's own center is `local` — 0.0 if nothing (gated out
// by Distribution Map, or this cell's own jittered sample position falls
// outside the Shape's own canvas). `local` can be outside [-0.5, 0.5] when
// checking whether a NEIGHBORING cell's own tile, pushed by Position
// Jitter, reaches a pixel that geometrically belongs to a different cell —
// exactly mirroring why `grid_rect_value` accepts an out-of-range `local`
// for negative Gap's own neighbor search, for the same underlying reason:
// a tile is no longer guaranteed to stay within its own cell's pixel
// footprint. An earlier version only ever checked a pixel's own cell here,
// with no such search at all — so a tile jittered toward a neighbor got
// sliced off in a flat line exactly at the cell boundary once it needed to
// draw outside its own cell's pixel footprint to look right. A real
// reported bug: visible once Position Jitter's own range was widened
// enough for tiles to routinely shift that far (see
// `POSITION_JITTER_MAX`'s own doc comment for that change).
fn shape_cell_value(
    local: vec2<f32>,
    cell: vec2<i32>,
    cell_uv: vec2<f32>,
    is_both: bool,
    is_horizontal_only: bool,
    is_vertical_only: bool,
    uv: vec2<f32>,
    columns_f: f32,
    size: i32,
    seed: u32,
    luminance_seed: u32,
    position_seed: u32,
    mirror_seed: u32,
    size_jitter: f32,
    rotation_jitter: f32,
    luminance_jitter: f32,
    position_jitter: f32,
    mirror_jitter: f32,
    shape_size: f32,
    has_distribution_map: u32,
    distribution_amount: f32,
    distribution_seed: u32,
) -> f32 {
    if !distribution_survives(cell_uv, cell, has_distribution_map, distribution_amount, distribution_seed) {
        return 0.0;
    }
    let h = hash2(cell, seed);
    let h2 = hash2(cell, seed + 101u);
    let h3 = hash2(cell, luminance_seed + 211u);
    let h4 = hash2(cell, position_seed + 311u);
    let h5 = hash2(cell, position_seed + 433u);
    // Grid layout ("Both"): shrinks toward 0 or grows up to
    // `SIZE_JITTER_GROW_MAX`, `h` picking linearly between the two — see
    // that constant's own doc comment for why growth is capped lower than
    // the shrink side goes. Outside Both: shrink-only, unchanged, for the
    // same structural reason Rotation/Position Jitter are also restricted
    // there.
    let size_scale = select(
        1.0 - size_jitter * h,
        mix(1.0 - size_jitter, 1.0 + size_jitter * (SIZE_JITTER_GROW_MAX - 1.0), h),
        is_both,
    );
    let angle = select(0.0, (h2 - 0.5) * rotation_jitter * ROTATION_JITTER_MAX, is_both);
    let position_offset = select(
        vec2<f32>(0.0),
        vec2<f32>(h4 - 0.5, h5 - 0.5) * position_jitter * POSITION_JITTER_MAX,
        is_both,
    );
    let luminance = 1.0 - luminance_jitter * h3;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let offset_local = local - position_offset;
    let rotated = vec2<f32>(
        offset_local.x * cos_a - offset_local.y * sin_a,
        offset_local.x * sin_a + offset_local.y * cos_a,
    );
    let effective_scale = max(size_scale * max(shape_size * columns_f, 0.0001), 0.0001);
    var raw_sample_pos: vec2<f32>;
    var in_bounds: bool;
    if is_horizontal_only {
        let sx = rotated.x / effective_scale + 0.5;
        raw_sample_pos = vec2<f32>(sx, uv.y);
        in_bounds = sx >= 0.0 && sx <= 1.0;
    } else if is_vertical_only {
        let sy = rotated.y / effective_scale + 0.5;
        raw_sample_pos = vec2<f32>(uv.x, sy);
        in_bounds = sy >= 0.0 && sy <= 1.0;
    } else {
        raw_sample_pos = rotated / effective_scale + vec2<f32>(0.5, 0.5);
        in_bounds = all(raw_sample_pos >= vec2<f32>(0.0)) && all(raw_sample_pos <= vec2<f32>(1.0));
    }
    if !in_bounds {
        return 0.0;
    }
    // Mirroring the sample position (rather than the shape's own geometry
    // before sampling) is always safe regardless of distribution mode or
    // bounds — `1.0 - x` stays inside `[0, 1]` exactly when `x` already
    // was, so this can't turn an in-bounds sample out-of-bounds or vice
    // versa. No Both-only restriction needed here, unlike Size/Rotation/
    // Position Jitter above: a mirrored tile never grows past its own
    // sampling footprint, so there's nothing for a neighbor search to
    // catch. Independent per axis (two separate coin flips), each at
    // probability `mirror_jitter * 0.5` — 0.5, not 1.0, because at
    // `mirror_jitter == 1` the point is "half the tiles end up mirrored,
    // half don't" (maximum *variation*), not "every tile is mirrored"
    // (which would be a uniform transform, not jitter).
    var sample_pos = raw_sample_pos;
    let flip_probability = mirror_jitter * 0.5;
    if hash2(cell, mirror_seed + 541u) < flip_probability {
        sample_pos.x = 1.0 - sample_pos.x;
    }
    if hash2(cell, mirror_seed + 647u) < flip_probability {
        sample_pos.y = 1.0 - sample_pos.y;
    }
    let coord = vec2<i32>(
        clamp(i32(sample_pos.x * f32(size)), 0, size - 1),
        clamp(i32(sample_pos.y * f32(size)), 0, size - 1),
    );
    return textureLoad(shape_tex, coord, 0).r * luminance;
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
            params.luminance_seed,
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
    let h3 = hash2(cell, params.luminance_seed + 211u);
    // Same hash offsets `grid_rect_value` uses for its own, independent
    // Position Jitter — not load-bearing to match (each call site hashes
    // its own cell with its own seed either way), but keeping them equal
    // means the built-in rect path and a connected-Shape path nudge the
    // *same* cell by the *same* amount, which reads as more coherent.
    let h4 = hash2(cell, params.position_seed + 311u);
    let h5 = hash2(cell, params.position_seed + 433u);

    let size_scale = 1.0 - params.size_jitter * h;
    // Rotation/Position Jitter only have one consistent geometric meaning
    // across a uniform grid where both axes repeat at the same scale
    // (Both). Horizontal/Vertical Only mix one repeating (scaled) axis
    // with one unscaled, full-tile axis — rotating/offsetting would mix
    // two different units — so both are skipped there (angle 0, offset
    // zero), the same "small cosmetic tilt, not a precise fit" scope
    // Rotation Jitter already documents for itself.
    let angle = select(0.0, (h2 - 0.5) * params.rotation_jitter * ROTATION_JITTER_MAX, is_both);
    let position_offset = select(
        vec2<f32>(0.0),
        vec2<f32>(h4 - 0.5, h5 - 0.5) * params.position_jitter * POSITION_JITTER_MAX,
        is_both,
    );
    let luminance = 1.0 - params.luminance_jitter * h3;

    let cos_a = cos(angle);
    let sin_a = sin(angle);
    // Position Jitter shifts the tile BEFORE rotation, same order
    // `grid_rect_value` uses internally — applied here too so a connected
    // Shape (which samples `rotated` directly below, never going through
    // `grid_rect_value` at all) actually sees it. An earlier version only
    // wired Position Jitter into `grid_rect_value`, which the Shape-
    // connected branch never calls — a real reported bug (caught by the
    // owner's own hands-on testing with a Shape connected), the same class
    // of gap Distribution Map had for the same reason.
    let offset_local = vec2<f32>(local_x, local_y) - position_offset;
    let rotated = vec2<f32>(
        offset_local.x * cos_a - offset_local.y * sin_a,
        offset_local.x * sin_a + offset_local.y * cos_a,
    );

    // This cell's own center, in the output's 0..1 uv space — shared by
    // every branch below (the built-in rect path AND the connected-Shape
    // path) so the Distribution Map gate applies consistently regardless
    // of whether a custom Shape is wired in. An earlier version computed
    // this separately inside only two of the three branches and never
    // threaded it into the has_shape branch at all — Distribution Map
    // silently had zero effect whenever a Shape was also connected, a real
    // reported bug (caught by the owner's own hands-on testing), not a
    // hypothetical. `is_both` uses the cell's own true center; Horizontal/
    // Vertical Only use the UNCROPPED axis's own midline instead of the
    // current pixel's position there, so one tile's own survive/drop
    // decision stays a single clean yes/no rather than fading in/out along
    // an axis with no per-tile identity to begin with (see
    // `distribution_survives`'s own doc comment).
    let cell_uv = select(
        select(
            vec2<f32>(0.5, (f32(row_idx) + 0.5) / rows_f),
            vec2<f32>((f32(col_idx) + 0.5) / columns_f, 0.5),
            is_horizontal_only,
        ),
        (vec2<f32>(f32(col_idx), f32(row_idx)) + 0.5) / vec2<f32>(columns_f, rows_f),
        is_both,
    );
    let survives_distribution = distribution_survives(
        cell_uv, cell, params.has_distribution_map, params.distribution_amount, params.distribution_seed
    );

    var value = 0.0;
    if params.has_shape != 0u {
        let size = i32(params.resolution);
        // `shape_size` is tile-absolute (see the param doc comment above)
        // — scale by `columns_f` (== `rows_f` here, since `effective_grid`
        // on the Rust side always forces a square grid once a Shape is
        // connected) to turn that absolute target into the right per-cell
        // zoom for the current Tile Count.
        value = shape_cell_value(
            vec2<f32>(local_x, local_y),
            cell,
            cell_uv,
            is_both,
            is_horizontal_only,
            is_vertical_only,
            uv,
            columns_f,
            size,
            params.seed,
            params.luminance_seed,
            params.position_seed,
            params.mirror_seed,
            params.size_jitter,
            params.rotation_jitter,
            params.luminance_jitter,
            params.position_jitter,
            params.mirror_jitter,
            params.shape_size,
            params.has_distribution_map,
            params.distribution_amount,
            params.distribution_seed,
        );
        // Position Jitter can push a neighboring cell's own tile into THIS
        // pixel's own geometric footprint — see `shape_cell_value`'s own
        // doc comment for why this search exists at all. Same ±1-cell
        // neighborhood, stagger handling, and wrap convention as the
        // built-in rect path's own negative-Gap search just below; `max`
        // across every cell that reaches this pixel, same compositing
        // choice `combine_overlap`'s own default (`OVERLAP_MAX`) already
        // makes for that path, picking whichever tile actually has content
        // here rather than requiring an explicit z-order between stamps
        // that independently jittered toward the same pixel.
        if is_both {
            let rows_i = i32(rows_f);
            for (var dy = -1; dy <= 1; dy = dy + 1) {
                let wrapped_row_n = wrap_coord(row_idx + dy, rows_i);
                var x_n = uv.x * columns_f;
                if (wrapped_row_n % 2) != 0 {
                    x_n = x_n + params.offset;
                }
                let local_y_n = local_y - f32(dy);
                let base_col_n = i32(floor(x_n));
                let frac_x_n = fract(x_n) - 0.5;
                for (var dx = -1; dx <= 1; dx = dx + 1) {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let col_n = wrap_coord(base_col_n + dx, columns_i);
                    let local_x_n = frac_x_n - f32(dx);
                    let cell_n = vec2<i32>(col_n, wrapped_row_n);
                    let cell_uv_n =
                        (vec2<f32>(f32(col_n), f32(wrapped_row_n)) + 0.5) / vec2<f32>(columns_f, rows_f);
                    let v = shape_cell_value(
                        vec2<f32>(local_x_n, local_y_n),
                        cell_n,
                        cell_uv_n,
                        is_both,
                        is_horizontal_only,
                        is_vertical_only,
                        uv,
                        columns_f,
                        size,
                        params.seed,
                        params.luminance_seed,
                        params.position_seed,
                        params.mirror_seed,
                        params.size_jitter,
                        params.rotation_jitter,
                        params.luminance_jitter,
                        params.position_jitter,
                        params.mirror_jitter,
                        params.shape_size,
                        params.has_distribution_map,
                        params.distribution_amount,
                        params.distribution_seed,
                    );
                    value = max(value, v);
                }
            }
        }
    } else if is_both {
        // Negative Gap grows every tile past its own cell — test the
        // immediate 3x3 neighborhood (never further: Gap's own range,
        // -0.4..0.4, grows a cell's own half-size to at most 0.9, still
        // under 1.5 cells, so only directly-adjacent cells can ever reach
        // a given pixel) and blend any that also cover it, on top of this
        // cell's own value. With Gap >= 0 no tile ever grows past its own
        // cell, so no neighbor ever covers — `blended` stays exactly
        // `center_v`, a no-op regardless of mode/opacity (see
        // `OverlapBlendMode`'s own doc comment).
        let center_v = grid_rect_value(
            vec2<f32>(local_x, local_y),
            cell,
            cell_uv,
            is_both,
            params.gap,
            params.bevel,
            params.size_jitter,
            params.rotation_jitter,
            params.luminance_jitter,
            params.position_jitter,
            params.seed,
            params.luminance_seed,
            params.position_seed,
            params.has_distribution_map,
            params.distribution_amount,
            params.distribution_seed,
        );
        if center_v >= 0.0 {
            let rows_i = i32(rows_f);
            var blended = center_v;
            for (var dy = -1; dy <= 1; dy = dy + 1) {
                let wrapped_row_n = wrap_coord(row_idx + dy, rows_i);
                var x_n = uv.x * columns_f;
                if (wrapped_row_n % 2) != 0 {
                    x_n = x_n + params.offset;
                }
                let local_y_n = local_y - f32(dy);
                let base_col_n = i32(floor(x_n));
                let frac_x_n = fract(x_n) - 0.5;
                for (var dx = -1; dx <= 1; dx = dx + 1) {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let col_n = wrap_coord(base_col_n + dx, columns_i);
                    let local_x_n = frac_x_n - f32(dx);
                    let cell_n = vec2<i32>(col_n, wrapped_row_n);
                    let cell_uv_n =
                        (vec2<f32>(f32(col_n), f32(wrapped_row_n)) + 0.5) / vec2<f32>(columns_f, rows_f);
                    let v = grid_rect_value(
                        vec2<f32>(local_x_n, local_y_n),
                        cell_n,
                        cell_uv_n,
                        is_both,
                        params.gap,
                        params.bevel,
                        params.size_jitter,
                        params.rotation_jitter,
                        params.luminance_jitter,
                        params.position_jitter,
                        params.seed,
                        params.luminance_seed,
                        params.position_seed,
                        params.has_distribution_map,
                        params.distribution_amount,
                        params.distribution_seed,
                    );
                    if v >= 0.0 {
                        blended = combine_overlap(params.overlap_blend_mode, blended, v);
                    }
                }
            }
            value = mix(center_v, blended, params.overlap_opacity);
        }
    } else {
        // Horizontal/Vertical Only: a single, uncropped-axis strip has no
        // coherent "neighbor cell" to overlap with along that axis, so
        // this keeps the plain single-cell test instead. `survives_
        // distribution` (computed once, above, shared with the has_shape
        // branch) already used the UNCROPPED axis's own midline (0.5) for
        // this cell_uv, not the current pixel's own position there, so one
        // tile's own survive/drop decision stays a single clean yes/no
        // instead of fading in/out along an axis with no per-tile identity
        // to begin with.
        if survives_distribution {
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
    }
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
