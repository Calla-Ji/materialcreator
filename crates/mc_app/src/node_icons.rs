// SPDX-License-Identifier: MIT OR Apache-2.0
//! A Material Icons Round glyph per node kind, for the Node Library panel
//! (same bundled font as the 3D viewport toolbar — see ASSETS.md). These are
//! placeholder choices: picked for a reasonable visual match per node, not
//! bespoke art, and several nodes share a glyph where nothing better fits or
//! the visual distinction isn't important. Codepoints verified against the
//! font's own published codepoints file (fonts.google.com/icons), not
//! guessed.
//!
//! [`draw_icon`] is the real entry point callers should use — it covers
//! `icon_for_kind`'s plain-glyph case plus three hand-drawn exceptions:
//! Checker's literal checkerboard (no single glyph in the bundled font
//! actually alternates fill like a real one), Path to Mask's
//! horizontally-mirrored `star_half` (the owner's own request — egui's
//! text shape has no flip, only rotation, so this one's genuinely
//! hand-tessellated; see [`draw_mirrored_glyph`]), and every Output node,
//! which draws its own channel's letter(s) (`B`/`N`/`R`/`M`/`H`/`AO`/
//! `ORM`/`E`/`A`) in the UI's own body font rather than an icon glyph at
//! all — the owner's own request, since 9 similar-looking Output nodes
//! sitting in a row were harder to tell apart by icon than by just
//! reading the letter. All drawn at the same fixed size
//! ([`OUTPUT_LETTER_SCALE`]) regardless of length, so "ORM" doesn't read
//! smaller than "B" just because it has to fit the same box.

/// Output letters are drawn at `font_size * OUTPUT_LETTER_SCALE`, the one
/// size every Output node's label uses — see this module's own doc
/// comment. Picked so "ORM" (the longest label) actually fits inside the
/// Node Library panel's 32px icon square with real padding either side
/// (measured against the bundled Geist font at the panel's own 18pt icon
/// size: 12pt renders "ORM" ~27px wide — confirmed empirically, not
/// guessed, since Geist's own character widths aren't something to eyeball
/// correctly). Every other letter uses this exact same size too, even
/// though a single letter would comfortably fit larger — the owner's own
/// request, so the whole Output row reads as one consistent size rather
/// than each label being "as big as it can get away with."
const OUTPUT_LETTER_SCALE: f32 = 12.0 / 18.0;

pub fn draw_icon(
    painter: &egui::Painter,
    rect: egui::Rect,
    kind: &str,
    font_size: f32,
    color: egui::Color32,
) {
    if kind == "checker" {
        draw_checker_icon(painter, rect, font_size, color);
        return;
    }
    if kind == "path_to_mask" {
        draw_mirrored_glyph(painter, rect, '\u{e839}', font_size, color); // star_half
        return;
    }
    if kind == "path_to_sdf" {
        draw_path_to_sdf_icon(painter, rect, font_size, color);
        return;
    }
    if kind == "distance_field" {
        draw_distance_field_icon(painter, rect, font_size, color);
        return;
    }
    if kind == "bevel" || kind == "bevel_profile" {
        draw_bevel_icon(painter, rect, font_size, color);
        return;
    }
    if let Some(letter) = output_letter(kind) {
        // One fixed size for every Output node's own letter, regardless of
        // whether it's drawing "B" or "ORM" — letting a longer label run
        // wider rather than shrinking per letter to fit a fixed box, so
        // every letter actually reads at the same visual size instead of
        // "AO"/"ORM" looking smaller/more cramped next to the single-
        // letter ones.
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            letter,
            egui::FontId::new(
                font_size * OUTPUT_LETTER_SCALE,
                egui::FontFamily::Proportional,
            ),
            color,
        );
        return;
    }
    centered_glyph(painter, rect, icon_for_kind(kind), font_size, color);
}

/// Draws `glyph` centered in `rect`, mirrored left-to-right. egui's own
/// `TextShape` has no flip (only `angle`, a rotation), so there's no
/// built-in way to mirror a glyph — this instead tessellates the glyph
/// into a real `Mesh` via the same public `Tessellator` egui's own
/// renderer uses (reusing its actual font layout/hinting rather than
/// reimplementing it by hand), then negates every vertex's x position
/// around the glyph's own center. Winding order doesn't need correcting
/// afterward — epaint's own `Mesh::indices` doc comment: "egui is NOT
/// consistent with what winding order it uses, so turn off backface
/// culling," meaning nothing in this renderer cares which way a
/// mirror-flipped triangle now winds.
fn draw_mirrored_glyph(
    painter: &egui::Painter,
    rect: egui::Rect,
    glyph: char,
    font_size: f32,
    color: egui::Color32,
) {
    let ctx = painter.ctx();
    let font_id = egui::FontId::new(
        font_size,
        egui::FontFamily::Name("material_icons_round".into()),
    );
    let galley = ctx.fonts_mut(|f| f.layout_no_wrap(glyph.to_string(), font_id, color));
    let pos = rect.center() - galley.size() / 2.0;
    let shape = egui::Shape::Text(egui::epaint::TextShape::new(pos, galley, color));

    let font_tex_size = ctx.fonts(|f| f.font_image_size());
    let tessellation_options = ctx.tessellation_options(|o| *o);
    let mut tessellator = egui::epaint::Tessellator::new(
        ctx.pixels_per_point(),
        tessellation_options,
        font_tex_size,
        Vec::new(),
    );
    let mut mesh = egui::epaint::Mesh::default();
    tessellator.tessellate_shape(shape, &mut mesh);

    let center_x = rect.center().x;
    for vertex in &mut mesh.vertices {
        vertex.pos.x = 2.0 * center_x - vertex.pos.x;
    }

    painter.add(egui::Shape::Mesh(std::sync::Arc::new(mesh)));
}

/// Draws `glyph` centered in `rect` at `font_size`, plain (no mirroring or
/// other transform) — the shared boilerplate [`draw_path_to_sdf_icon`] and
/// [`draw_distance_field_icon`] both layer several glyphs/shapes with.
fn centered_glyph(
    painter: &egui::Painter,
    rect: egui::Rect,
    glyph: char,
    font_size: f32,
    color: egui::Color32,
) {
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        glyph,
        egui::FontId::new(
            font_size,
            egui::FontFamily::Name("material_icons_round".into()),
        ),
        color,
    );
}

/// Two concentric "star_border" outlines (a larger one drawn first, as a
/// second ring around the normal-size one) plus a small solid "star" at the
/// very center — the owner's own request. Path to SDF turns a Path into a
/// real signed distance field; concentric rings around a solid core read as
/// a natural nod to that (isolines around a shape), rather than a literal
/// SDF visualization.
fn draw_path_to_sdf_icon(
    painter: &egui::Painter,
    rect: egui::Rect,
    font_size: f32,
    color: egui::Color32,
) {
    centered_glyph(painter, rect, '\u{e83a}', font_size * 1.3, color); // star_border (outer)
    centered_glyph(painter, rect, '\u{e83a}', font_size, color); // star_border (inner)
    centered_glyph(painter, rect, '\u{e838}', font_size * 0.32, color); // star (solid)
}

/// "open_with"'s own 4 outward-radiating arrows already leave their center
/// empty — a small filled circle dropped there reads as "the point this
/// field measures distance from," distinguishing Distance Field from
/// Distance (`distance_spread`), which uses the same base glyph without the
/// circle. The owner's own request.
fn draw_distance_field_icon(
    painter: &egui::Painter,
    rect: egui::Rect,
    font_size: f32,
    color: egui::Color32,
) {
    centered_glyph(painter, rect, '\u{e89f}', font_size, color); // open_with
    painter.circle_filled(rect.center(), font_size * 0.15, color);
}

/// "rounded_corner"'s own curve plus a small, plain (unrounded) 90° angle
/// drawn beside it — the owner's own request: Bevel/Bevel Profile both turn
/// a sharp corner into a rounded one, so a sharp right-angle mark next to
/// the rounded glyph reads as "the corner this started from." Shared by
/// both nodes (same icon for both, per the owner's own request) — tucked
/// into the icon's lower-right corner so it doesn't compete with the
/// glyph's own curve, which reads top-left to bottom in Material's own
/// "rounded_corner" glyph.
fn draw_bevel_icon(
    painter: &egui::Painter,
    rect: egui::Rect,
    font_size: f32,
    color: egui::Color32,
) {
    centered_glyph(painter, rect, '\u{e920}', font_size, color); // rounded_corner

    let stroke = egui::Stroke::new(1.4, color);
    let corner = rect.right_bottom() - egui::vec2(font_size * 0.2, font_size * 0.35);
    let arm = font_size * 0.28;
    painter.line_segment([corner, corner - egui::vec2(0.0, arm)], stroke);
    painter.line_segment([corner, corner - egui::vec2(arm, 0.0)], stroke);
}

/// The letter(s) an Output node draws instead of a glyph — "AO"/"ORM" for
/// Ambient Occlusion/ORM, matching the abbreviations export filenames
/// already use; a single letter for everything else. `None` for every
/// non-Output kind.
fn output_letter(kind: &str) -> Option<&'static str> {
    match kind {
        "output_base_color" => Some("B"),
        "output_normal" => Some("N"),
        "output_roughness" => Some("R"),
        "output_metallic" => Some("M"),
        "output_height" => Some("H"),
        "output_ambient_occlusion" => Some("AO"),
        "output_orm" => Some("ORM"),
        "output_emissive" => Some("E"),
        "output_alpha" => Some("A"),
        _ => None,
    }
}

/// A literal 2x2 checkerboard — top-left and bottom-right filled solid,
/// top-right and bottom-left outlined only — centered in `rect` and sized
/// to roughly match a glyph drawn at `font_size` in the same spot.
fn draw_checker_icon(
    painter: &egui::Painter,
    rect: egui::Rect,
    font_size: f32,
    color: egui::Color32,
) {
    let board = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(font_size));
    let cell = board.size() / 2.0;
    let gap = 1.5;
    let stroke = egui::Stroke::new(1.2, color);
    for row in 0..2i32 {
        for col in 0..2i32 {
            let min = board.min + egui::vec2(col as f32 * cell.x, row as f32 * cell.y);
            let square = egui::Rect::from_min_size(min, cell).shrink(gap / 2.0);
            if (row + col) % 2 == 0 {
                painter.rect_filled(square, 1.0, color);
            } else {
                painter.rect_stroke(square, 1.0, stroke, egui::StrokeKind::Inside);
            }
        }
    }
}

pub fn icon_for_kind(kind: &str) -> char {
    match kind {
        "perlin_noise" => '\u{e3ea}',      // grain
        "simplex_noise" => '\u{f1b9}',     // multiple_stop — the owner's own pick
        "white_noise" => '\u{e5c3}',       // apps — the owner's own pick
        "clouds" => '\u{e2bd}',            // cloud
        "voronoi" => '\u{e268}',           // scatter_plot
        "fractal_sum" => '\u{e65f}',       // auto_awesome — the owner's own pick
        "anisotropic_noise" => '\u{e0b8}', // clear_all — the owner's own pick
        "gabor_noise" => '\u{e176}',       // waves — the owner's own pick
        "scattered_streaks" => '\u{eb66}', // ssid_chart — the owner's own pick
        "tile_generator" => '\u{e9b0}',    // grid_view
        "fabric_generator" => '\u{f016}',  // grid_4x4 — the owner's own pick
        "stitch" => '\u{e155}',            // gesture — the owner's own pick
        "knit" => '\u{eb3d}',              // all_inclusive — the owner's own pick
        "thread_shape" => '\u{e91a}',      // line_weight
        "image_import" => '\u{e43e}',      // add_photo_alternate
        "shape" => '\u{e574}',             // category
        "circular_pattern" => '\u{e918}',  // donut_small
        "type_text" => '\u{e264}',         // title
        "gradient_linear" => '\u{e3e9}',   // gradient
        "gradient_radial" => '\u{e3a2}',   // blur_circular
        "uniform_color" => '\u{e23a}',     // format_color_fill
        // "checker" isn't here — `draw_icon` below hand-draws a literal
        // checkerboard for it instead of using a single font glyph.
        "path" => '\u{eacd}',          // route — the owner's own pick
        "path_simplify" => '\u{e3c8}', // details — the owner's own pick
        // "path_to_mask" isn't here — `draw_icon` above hand-mirrors
        // "star_half" for it instead of using a plain glyph.
        "image_trace" => '\u{e5fa}',          // document_scanner
        "dither_pattern" => '\u{e3a3}',       // blur_linear — the owner's own pick
        "levels" => '\u{e429}',               // tune
        "invert" => '\u{e891}',               // invert_colors
        "clamp" => '\u{e94d}',                // compress
        "curve" => '\u{e6e1}',                // show_chart
        "hsl_adjust" => '\u{e40a}',           // palette
        "gradient_map" => '\u{e3e9}',         // gradient
        "grayscale_conversion" => '\u{e3db}', // filter_b_and_w
        "blend" => '\u{e53b}',                // layers
        "histogram_scan" => '\u{e915}',       // compare_arrows
        "threshold" => '\u{e3ca}',            // exposure
        "blur" => '\u{e3a5}',                 // blur_on
        // "bevel" and "bevel_profile" aren't here — `draw_icon` above
        // hand-draws a composite icon for both instead (see
        // `draw_bevel_icon`).
        "warp" => '\u{ef5e}',        // nightlight_round — the owner's own pick
        "transform2d" => '\u{e428}', // transform
        "directional_blur" => '\u{e3a3}', // blur_linear
        "directional_warp" => '\u{e80e}', // whatshot — the owner's own pick (Push Warp)
        "slope_blur" => '\u{efd8}',  // air
        "height_to_normal" => '\u{e564}', // terrain
        "height_to_ao" => '\u{e427}', // tonality
        "curvature_from_normal" => '\u{ea3b}', // architecture
        "channel_split" => '\u{e0b6}', // call_split
        "block_shift" => '\u{e3ad}', // broken_image
        "scanline_jitter" => '\u{e919}', // line_style
        "data_corrupt" => '\u{e868}', // bug_report
        "ordered_dither" => '\u{e421}', // texture
        "palette_dither" => '\u{e3b8}', // colorize
        "fasteners" => '\u{ea59}',   // hardware
        "color_match" => '\u{e23a}', // format_color_fill
        "panel_layout" => '\u{e871}', // dashboard
        "distance_spread" => '\u{e89f}', // open_with
        "channel_pack" => '\u{e1a0}', // workspaces
        "material_preset" => '\u{e02f}', // library_books
        "shape_boolean" => '\u{e87b}', // extension
        "edge_wear" => '\u{e90b}',   // all_out
        "redistance" => '\u{f8d0}',  // unfold_more_double
        // "path_to_sdf" and "distance_field" aren't here — `draw_icon`
        // above hand-draws a composite icon for each instead (see
        // `draw_path_to_sdf_icon`/`draw_distance_field_icon`).
        // No "output_*" entries — `draw_icon` above draws each Output
        // node's own letter instead (see `output_letter`).
        _ => '\u{e574}', // category (fallback)
    }
}
