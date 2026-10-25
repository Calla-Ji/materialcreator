// SPDX-License-Identifier: MIT OR Apache-2.0
//! A custom color picker — a polar hue/saturation wheel, a value slider,
//! R/G/B 0-255 fields, and a hex text box — replacing egui's own stock
//! color button's popup (confirmed by reading its actual source, not
//! assumed: that popup has drag-to-adjust R/G/B/A fields but no hex field
//! at all, and its own content isn't something a caller can append to from
//! outside). Built from scratch against egui's own public `epaint::Mesh`/
//! `Hsva` primitives — the same vertex-colored-mesh technique egui's own
//! square saturation/value box already uses internally
//! (`color_picker_hsvag_2d`), just mapped to polar coordinates instead of
//! a grid. Not modeled on any specific third-party color tool's own UI —
//! a wheel-plus-sliders-plus-hex layout is a generic, decades-old pattern
//! (Photoshop, Sketch, GIMP, Krita all have their own variant).

use egui::epaint::Hsva;
use egui::{Color32, Mesh, Response, Sense, Shape, Stroke, StrokeKind, Ui, Vec2};

const WHEEL_DIAMETER: f32 = 180.0;
/// Mesh density for the wheel's own vertex-colored fill — a `48 * 24` grid
/// of quads (`47` rings between the fully-desaturated center and the fully-
/// saturated rim, `48` hue steps around) is smooth enough that individual
/// facets aren't visible at `WHEEL_DIAMETER`'s own size, without generating
/// an excessive vertex count for a popup that's only open while actively
/// picking a color.
const WHEEL_RINGS: u32 = 24;
const WHEEL_SPOKES: u32 = 48;

/// Simple luminance-based black/white contrast pick for the wheel's own
/// selection marker outline — egui's own equivalent (`contrast_color` in
/// its `color_picker.rs`) isn't exported, so this is a small, self-
/// contained equivalent, not a copy (it's a standard luminance heuristic,
/// not expressive code worth attributing).
fn contrast_color(c: Color32) -> Color32 {
    let luminance = 0.299 * c.r() as f32 + 0.587 * c.g() as f32 + 0.114 * c.b() as f32;
    if luminance < 128.0 {
        Color32::WHITE
    } else {
        Color32::BLACK
    }
}

/// The wheel itself: angle = hue, radius = saturation, at a fixed `value`
/// (the slider alongside it in [`color_edit_button`] owns that third HSV
/// dimension — a single 2D control can only ever expose two of the three).
/// Returns `true` the frame a drag/click actually changed `hue`/`sat`.
fn hue_sat_wheel(ui: &mut Ui, hue: &mut f32, sat: &mut f32, value: f32) -> bool {
    let (rect, response) =
        ui.allocate_at_least(Vec2::splat(WHEEL_DIAMETER), Sense::click_and_drag());
    let center = rect.center();
    let radius = rect.width() / 2.0;

    let mut changed = false;
    if let Some(pos) = response.interact_pointer_pos() {
        let delta = pos - center;
        let r = delta.length().min(radius);
        *hue = (delta.y.atan2(delta.x) / std::f32::consts::TAU).rem_euclid(1.0);
        *sat = if radius > 0.0 { r / radius } else { 0.0 };
        changed = true;
    }

    if ui.is_rect_visible(rect) {
        let mut mesh = Mesh::default();
        let row = WHEEL_SPOKES + 1;
        for ring in 0..=WHEEL_RINGS {
            let s = ring as f32 / WHEEL_RINGS as f32;
            for spoke in 0..=WHEEL_SPOKES {
                let h = spoke as f32 / WHEEL_SPOKES as f32;
                let angle = h * std::f32::consts::TAU;
                let p = center + Vec2::new(angle.cos(), angle.sin()) * (s * radius);
                mesh.colored_vertex(p, Color32::from(Hsva::new(h, s, value, 1.0)));
                if ring < WHEEL_RINGS && spoke < WHEEL_SPOKES {
                    let tl = ring * row + spoke;
                    mesh.add_triangle(tl, tl + 1, tl + row);
                    mesh.add_triangle(tl + 1, tl + row, tl + row + 1);
                }
            }
        }
        ui.painter().add(Shape::mesh(mesh));
        ui.painter().circle_stroke(
            center,
            radius,
            ui.visuals().widgets.noninteractive.bg_stroke,
        );

        let marker_angle = *hue * std::f32::consts::TAU;
        let marker_radius = *sat * radius;
        let marker_pos = center + Vec2::new(marker_angle.cos(), marker_angle.sin()) * marker_radius;
        let marker_color = Color32::from(Hsva::new(*hue, *sat, value, 1.0));
        ui.painter().circle(
            marker_pos,
            5.0,
            marker_color,
            Stroke::new(1.5, contrast_color(marker_color)),
        );
    }

    changed
}

/// Parses `#RGB`, `#RRGGBB`, or a bare `RRGGBB`/`RGB` (case-insensitive,
/// leading `#` optional) into opaque sRGB bytes. Anything else (a typo, an
/// in-progress edit) returns `None` rather than guessing — the hex field
/// only ever commits the color on a fully valid parse, same convention
/// every other free-typed param in this panel already follows.
fn parse_hex(text: &str) -> Option<[u8; 3]> {
    let text = text.trim().strip_prefix('#').unwrap_or_else(|| text.trim());
    let hex_digit = |c: char| c.to_digit(16);
    let expand = |c: char| hex_digit(c).map(|d| (d * 16 + d) as u8);
    match text.len() {
        3 => {
            let mut chars = text.chars();
            Some([
                expand(chars.next()?)?,
                expand(chars.next()?)?,
                expand(chars.next()?)?,
            ])
        }
        6 => {
            let byte = |i: usize| u8::from_str_radix(&text[i..i + 2], 16).ok();
            Some([byte(0)?, byte(2)?, byte(4)?])
        }
        _ => None,
    }
}

/// [`color_edit_button`]'s own result — a plain `Response` plus whether the
/// popup's own eyedropper button was clicked this frame. Split out rather
/// than folded into `Response` itself (e.g. via a second `clicked()`-style
/// method) since only the *caller* knows which [`crate::eyedropper::
/// EyedropperTarget`] this particular color belongs to — this module has
/// no idea whether it's editing a node's solid color, a gradient stop, or
/// a Frame's color, so it can't start the eyedropper itself.
pub struct ColorEditResponse {
    pub response: Response,
    pub eyedropper_requested: bool,
}

/// A popup-based color editor: a swatch button that, when clicked, opens
/// the wheel above plus a Value slider, R/G/B 0-255 fields, a hex text
/// box, and an eyedropper button. `rgba_unmultiplied` is linear, same
/// convention every other `[f32; 4]` color value in this app already uses.
/// `.response`'s own `.changed()` is `true` the frame the color actually
/// changed, matching `Ui::color_edit_button_rgba_unmultiplied`'s own
/// convention (the stock widget this replaces, which had no hex field or
/// eyedropper at all).
pub fn color_edit_button(ui: &mut Ui, rgba_unmultiplied: &mut [f32; 4]) -> ColorEditResponse {
    let current = egui::Rgba::from_rgba_unmultiplied(
        rgba_unmultiplied[0],
        rgba_unmultiplied[1],
        rgba_unmultiplied[2],
        rgba_unmultiplied[3],
    );

    let popup_id = ui.auto_id_with("color_wheel_popup");
    let open = egui::Popup::is_id_open(ui.ctx(), popup_id);
    let size = ui.spacing().interact_size;
    let (rect, mut button_response) = ui.allocate_exact_size(size, Sense::click());
    if ui.is_rect_visible(rect) {
        egui::color_picker::show_color_at(ui.painter(), Color32::from(current), rect);
        let visuals = if open {
            &ui.visuals().widgets.open
        } else {
            ui.style().interact(&button_response)
        };
        ui.painter().rect_stroke(
            rect,
            visuals.corner_radius,
            visuals.bg_stroke,
            StrokeKind::Inside,
        );
    }
    if ui.style().explanation_tooltips {
        button_response = button_response.on_hover_text("Click to edit color");
    }

    let mut hsva = Hsva::from(current);
    let mut changed = false;
    let mut eyedropper_requested = false;

    egui::Popup::menu(&button_response)
        .id(popup_id)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.spacing_mut().slider_width = WHEEL_DIAMETER;

            ui.vertical_centered(|ui| {
                changed |= hue_sat_wheel(ui, &mut hsva.h, &mut hsva.s, hsva.v);
            });
            changed |= ui
                .add(egui::Slider::new(&mut hsva.v, 0.0..=1.0).text("Value"))
                .changed();

            let mut srgb = hsva.to_srgba_unmultiplied();
            ui.horizontal(|ui| {
                let mut field_changed = false;
                field_changed |= ui
                    .add(
                        egui::DragValue::new(&mut srgb[0])
                            .range(0..=255)
                            .prefix("R "),
                    )
                    .changed();
                field_changed |= ui
                    .add(
                        egui::DragValue::new(&mut srgb[1])
                            .range(0..=255)
                            .prefix("G "),
                    )
                    .changed();
                field_changed |= ui
                    .add(
                        egui::DragValue::new(&mut srgb[2])
                            .range(0..=255)
                            .prefix("B "),
                    )
                    .changed();
                if field_changed {
                    let alpha = hsva.a;
                    hsva = Hsva::from_srgba_unmultiplied([srgb[0], srgb[1], srgb[2], 255]);
                    hsva.a = alpha;
                    changed = true;
                }
            });
            changed |= ui
                .add(
                    egui::Slider::new(&mut hsva.a, 0.0..=1.0)
                        .text("Alpha")
                        .max_decimals(2),
                )
                .changed();

            // The hex field only ever reflects the *committed* color while
            // unfocused — re-deriving it from `srgb` every single frame
            // regardless would overwrite an in-progress edit (e.g. typing
            // "f" of a new "ff6600" over the old "000000") before the user
            // finished typing it.
            let hex_id = ui.id().with("hex");
            let focused = ui.memory(|m| m.has_focus(hex_id));
            let mut hex_text = if focused {
                ui.data(|d| d.get_temp::<String>(hex_id))
                    .unwrap_or_default()
            } else {
                format!("{:02X}{:02X}{:02X}", srgb[0], srgb[1], srgb[2])
            };
            ui.horizontal(|ui| {
                ui.label("Hex");
                let response = ui.add(
                    egui::TextEdit::singleline(&mut hex_text)
                        .id(hex_id)
                        .desired_width(70.0)
                        .hint_text("RRGGBB"),
                );
                if response.changed() {
                    ui.data_mut(|d| d.insert_temp(hex_id, hex_text.clone()));
                    if let Some([r, g, b]) = parse_hex(&hex_text) {
                        let alpha = hsva.a;
                        hsva = Hsva::from_srgba_unmultiplied([r, g, b, 255]);
                        hsva.a = alpha;
                        changed = true;
                    }
                }
                if crate::eyedropper::eyedropper_button(ui) {
                    eyedropper_requested = true;
                }
            });
        });

    if changed {
        let new_rgba = egui::Rgba::from(hsva).to_rgba_unmultiplied();
        *rgba_unmultiplied = new_rgba;
        button_response.mark_changed();
    }

    ColorEditResponse {
        response: button_response,
        eyedropper_requested,
    }
}

#[cfg(test)]
mod tests {
    use super::parse_hex;

    #[test]
    fn parse_hex_accepts_six_digit_with_or_without_hash() {
        assert_eq!(parse_hex("#FF6600"), Some([0xFF, 0x66, 0x00]));
        assert_eq!(parse_hex("ff6600"), Some([0xFF, 0x66, 0x00]));
    }

    #[test]
    fn parse_hex_accepts_three_digit_shorthand() {
        // #f60 expands each digit (f -> ff, 6 -> 66, 0 -> 00).
        assert_eq!(parse_hex("#f60"), Some([0xFF, 0x66, 0x00]));
    }

    #[test]
    fn parse_hex_rejects_anything_else() {
        assert_eq!(parse_hex(""), None);
        assert_eq!(parse_hex("#ff"), None);
        assert_eq!(parse_hex("not a color"), None);
        assert_eq!(parse_hex("#ff66001"), None, "7 digits is neither 3 nor 6");
        assert_eq!(parse_hex("#gggggg"), None, "g isn't a hex digit");
    }
}
