// SPDX-License-Identifier: MIT OR Apache-2.0
//! Screen-wide eyedropper: captures a screenshot, shows it full-screen in a
//! borderless overlay window, and reports back whichever pixel the artist
//! clicks on — anywhere on the real desktop, not just inside this app's own
//! window. Needs a new dependency (`xcap`), since neither egui nor winit can
//! read screen pixels on their own — added with the owner's own explicit
//! go-ahead (CLAUDE.md's "ask before adding dependencies").
//!
//! On macOS this triggers a one-time "Screen Recording" permission prompt
//! the first time it's used — a real OS requirement, not a bug; `start`
//! returns `None` rather than panicking if capture fails for any reason
//! (permission not yet granted, no monitor found, etc.), and the caller
//! just leaves the eyedropper un-started in that case.
//!
//! Single-monitor only for now: captures whichever monitor the app's own
//! window currently sits on. Multi-monitor spanning (dragging the picker
//! across a second display) is a real, stated limitation, not an
//! oversight — a reasonable follow-up once this is proven out, not
//! attempted here.

use egui::{Color32, ColorImage, Pos2, Rect, Vec2, ViewportBuilder, ViewportId};

/// Which color param an in-progress eyedropper pick should be written back
/// to once the artist clicks a pixel.
#[derive(Clone, Copy)]
pub enum EyedropperTarget {
    /// A node's own solid-`Color` param (`mc_nodes::defs::ParamSpecKind::
    /// Color`) — currently just Uniform Color's `"color"`.
    NodeColor {
        node: mc_core::NodeId,
        param: &'static str,
    },
    /// One stop within a node's `Gradient` param (Gradient Map, Palette
    /// Dither).
    GradientStop {
        node: mc_core::NodeId,
        param: &'static str,
        index: usize,
    },
    /// A Frame annotation's own color.
    FrameColor { frame: mc_core::FrameId },
}

/// An in-progress eyedropper pick — owned by `MaterialCreatorApp` as
/// `Option<EyedropperState>`, driven one frame at a time by [`show`].
pub struct EyedropperState {
    target: EyedropperTarget,
    /// Full *physical* pixel resolution (a Retina display captures more
    /// pixels than there are logical points) — kept at that resolution for
    /// pixel-exact sampling precision; `show` maps the cursor into it by
    /// *ratio* against whatever rect the overlay window's content area
    /// actually ends up being, not by assuming any particular pixels-per-
    /// point relationship.
    image: ColorImage,
    texture: Option<egui::TextureHandle>,
    /// The captured monitor's own bounds, in *logical points* (confirmed
    /// against xcap's own macOS source: `x()/y()/width()/height()` all come
    /// from `CGDisplayBounds`, which Apple documents as the global display
    /// coordinate space — points, not pixels; a Retina `capture_image()`
    /// returns more physical pixels than this covers, which is exactly why
    /// `image` above is sampled by *ratio*, not by a 1:1 pixel offset).
    /// Used to explicitly position and size the overlay window so it covers
    /// the *entire* monitor, menu bar included — `with_maximized(true)` was
    /// tried instead (letting the OS pick the geometry entirely) and still
    /// left a strip uncovered, showing the real, live main window next to
    /// the static screenshot: the actual cause of the "window in window"
    /// look, not a leftover from the geometry math this field now avoids by
    /// stating the target rect explicitly instead of leaving it to a
    /// same-named-but-different OS "maximize" gesture.
    monitor_rect: Rect,
    /// Whether `ViewportCommand::Focus` has already been sent for this
    /// pick — sent once, the first frame, not every frame (see `show`'s
    /// own comment on why the overlay needs to be explicitly focused, not
    /// just always-on-top, to avoid the real window staying frontmost —
    /// the actual cause of an earlier "thin sliver, wrong window in front"
    /// bug that a wrong theory about window *sizing* didn't fix, because
    /// sizing was never the real problem there).
    focus_requested: bool,
}

fn primary_monitor() -> Option<xcap::Monitor> {
    xcap::Monitor::all()
        .ok()?
        .into_iter()
        .find(|m| m.is_primary().unwrap_or(false))
}

impl EyedropperState {
    /// Captures the monitor the app's own window currently sits on (falls
    /// back to the primary monitor if that can't be determined — e.g. on
    /// Wayland, where `eframe` can't report a window position at all).
    /// Returns `None` on any failure rather than panicking — this touches
    /// real OS permissions and hardware outside the app's own control, so a
    /// failure here is an expected, not exceptional, outcome.
    pub fn start(ctx: &egui::Context, target: EyedropperTarget) -> Option<Self> {
        let window_pos = ctx.input(|i| i.viewport().outer_rect).map(|r| r.min);
        let monitor = window_pos
            .and_then(|pos| xcap::Monitor::from_point(pos.x as i32, pos.y as i32).ok())
            .or_else(primary_monitor)?;

        let monitor_rect = Rect::from_min_size(
            Pos2::new(monitor.x().ok()? as f32, monitor.y().ok()? as f32),
            Vec2::new(monitor.width().ok()? as f32, monitor.height().ok()? as f32),
        );

        let captured = monitor.capture_image().ok()?;
        let (width, height) = (captured.width() as usize, captured.height() as usize);
        log::debug!(
            "eyedropper: captured monitor {:?} at {width}x{height}px, bounds {monitor_rect:?} pt",
            monitor.name().ok()
        );
        let image = ColorImage::from_rgba_unmultiplied([width, height], captured.as_raw());

        Some(Self {
            target,
            image,
            texture: None,
            monitor_rect,
            focus_requested: false,
        })
    }

    pub fn target(&self) -> EyedropperTarget {
        self.target
    }
}

pub enum Outcome {
    /// Still open, nothing picked yet this frame.
    Pending,
    /// The artist clicked — linear `[r, g, b, a]`, same convention as every
    /// other color value in this app (`egui::Rgba`'s own sRGB-aware
    /// conversion does the gamma decode, not a hand-rolled one).
    Picked([f32; 4]),
    /// Esc, a click outside the captured monitor, or the OS asked the
    /// overlay window to close.
    Cancelled,
}

const LOUPE_SIZE: f32 = 80.0;
const LOUPE_OFFSET: Vec2 = Vec2::new(24.0, 24.0);
const LOUPE_SAMPLE_PX: usize = 9;

/// Draws the overlay window and handles its input for this frame. Call
/// every frame while `Some(EyedropperState)` is active; stop calling (and
/// drop the state) once this returns anything other than `Pending`.
///
/// This is a plain borderless window, explicitly positioned and sized to
/// `state.monitor_rect` — the captured monitor's own full bounds, menu bar
/// included. `with_maximized(true)` was tried as an alternative (handing
/// the geometry decision to the OS entirely, instead of computing it) but
/// left a strip of the real screen uncovered, showing the actual live main
/// window at the same time as the static screenshot underneath it — the
/// real cause of a "window in window" look, not a leftover geometry bug in
/// this explicit approach, which already covers the monitor's full bounds
/// (see `monitor_rect`'s own doc comment for why those bounds are already
/// in the right units). The cursor is still mapped into `state.image` by
/// *ratio* against whatever rect the window's content area actually ends
/// up being, not a precomputed scale factor, so sampling stays correct
/// even if the OS doesn't grant `monitor_rect` exactly as asked.
pub fn show(ctx: &egui::Context, state: &mut EyedropperState) -> Outcome {
    let texture = state.texture.get_or_insert_with(|| {
        ctx.load_texture(
            "eyedropper_capture",
            state.image.clone(),
            egui::TextureOptions::NEAREST,
        )
    });
    let texture_id = texture.id();
    let [img_w, img_h] = state.image.size;

    let mut outcome = Outcome::Pending;
    let viewport_id = ViewportId::from_hash_of("mc_eyedropper_overlay");

    ctx.show_viewport_immediate(
        viewport_id,
        ViewportBuilder::default()
            .with_title("Pick a color")
            .with_decorations(false)
            .with_resizable(false)
            .with_transparent(false)
            .with_always_on_top()
            .with_active(true)
            .with_position(state.monitor_rect.min)
            .with_inner_size(state.monitor_rect.size()),
        |ui, class| {
            if ui.ctx().input(|i| i.viewport().close_requested()) {
                let class = match class {
                    egui::ViewportClass::Root => "Root",
                    egui::ViewportClass::Immediate => "Immediate",
                    egui::ViewportClass::Deferred => "Deferred",
                    egui::ViewportClass::EmbeddedWindow => "EmbeddedWindow",
                };
                log::debug!("eyedropper: overlay close_requested, cancelling (class={class})");
                outcome = Outcome::Cancelled;
                return;
            }
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                log::debug!("eyedropper: Escape pressed, cancelling");
                outcome = Outcome::Cancelled;
                return;
            }
            // `with_active(true)` is a *hint* the OS doesn't always honor
            // for a freshly created window while another one already has
            // focus (which is exactly the situation here — the main app
            // window is what's handling input the instant this overlay is
            // requested) — an explicit `Focus` command the first frame is
            // the reliable way to actually bring it, and keep it, in front
            // instead of the real window staying frontmost with just the
            // overlay's own top edge peeking out from behind it.
            let rect = ui.max_rect();
            if !state.focus_requested {
                state.focus_requested = true;
                log::debug!("eyedropper: overlay first frame, content rect is {rect:?}");
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Focus);
            }

            // The OS doesn't always honor `with_position`/`with_inner_size`
            // exactly as asked — on macOS specifically, a borderless window
            // placed to cover the menu bar strip gets silently clamped to
            // the visible frame below it instead (confirmed from a real
            // debug session: a 982pt-tall request came back as 917pt — a
            // menu-bar-sized shortfall). Stretching the *whole* captured
            // image into whatever smaller area the OS actually granted
            // paints the captured image's own top strip (which already has
            // the real menu bar baked into it) just below the real, live
            // menu bar — a convincing-looking duplicate that isn't actually
            // a second window, just a wrongly-cropped one. Reading back
            // this viewport's own actual on-screen rect and only drawing
            // (and sampling) the matching slice of the captured image,
            // instead of assuming the OS granted the full monitor bounds,
            // fixes this regardless of exactly how much the OS clamps.
            let actual_rect_global = ui
                .ctx()
                .input(|i| i.viewport().outer_rect)
                .unwrap_or(state.monitor_rect);
            let uv_min = Pos2::new(
                (actual_rect_global.min.x - state.monitor_rect.min.x) / state.monitor_rect.width(),
                (actual_rect_global.min.y - state.monitor_rect.min.y) / state.monitor_rect.height(),
            );
            let uv_max = Pos2::new(
                (actual_rect_global.max.x - state.monitor_rect.min.x) / state.monitor_rect.width(),
                (actual_rect_global.max.y - state.monitor_rect.min.y) / state.monitor_rect.height(),
            );

            ui.painter().image(
                texture_id,
                rect,
                Rect::from_min_max(uv_min, uv_max),
                Color32::WHITE,
            );

            let Some(pos) = ui.ctx().pointer_latest_pos() else {
                return;
            };
            // Ratio of where the cursor sits within the overlay's own
            // *actual* rect (whatever the OS gave it, in whatever units
            // `ui.max_rect()` reports), mapped onto the *same* cropped UV
            // slice drawn above — not the full image — so a click lands on
            // the pixel the artist actually sees, even if the OS clamped
            // the window short of the full monitor.
            let frac_x = ((pos.x - rect.min.x) / rect.width().max(1.0)).clamp(0.0, 1.0);
            let frac_y = ((pos.y - rect.min.y) / rect.height().max(1.0)).clamp(0.0, 1.0);
            let u = uv_min.x + frac_x * (uv_max.x - uv_min.x);
            let v = uv_min.y + frac_y * (uv_max.y - uv_min.y);
            let px = ((u * img_w as f32) as usize).min(img_w - 1);
            let py = ((v * img_h as f32) as usize).min(img_h - 1);
            let sample = state.image[(px, py)];

            draw_loupe(ui, pos, &state.image, px, py);

            if ui.input(|i| i.pointer.primary_clicked()) {
                let linear = egui::Rgba::from(sample).to_rgba_unmultiplied();
                outcome = Outcome::Picked(linear);
            }
        },
    );

    outcome
}

/// A small fixed-zoom magnifier near the cursor, showing the
/// `LOUPE_SAMPLE_PX x LOUPE_SAMPLE_PX` neighborhood around the pixel about
/// to be picked plus its own center pixel's color as a swatch — precision
/// for picking a single texel out of a full screenshot is otherwise
/// difficult at native scale.
fn draw_loupe(ui: &egui::Ui, cursor: Pos2, image: &ColorImage, px: usize, py: usize) {
    let painter = ui.painter();
    let loupe_rect = Rect::from_min_size(cursor + LOUPE_OFFSET, Vec2::splat(LOUPE_SIZE));
    let cell = LOUPE_SIZE / LOUPE_SAMPLE_PX as f32;
    let half = (LOUPE_SAMPLE_PX / 2) as isize;

    for row in 0..LOUPE_SAMPLE_PX {
        for col in 0..LOUPE_SAMPLE_PX {
            let sx = px as isize + col as isize - half;
            let sy = py as isize + row as isize - half;
            let color = if sx >= 0
                && sy >= 0
                && (sx as usize) < image.width()
                && (sy as usize) < image.height()
            {
                image[(sx as usize, sy as usize)]
            } else {
                Color32::BLACK
            };
            let cell_rect = Rect::from_min_size(
                loupe_rect.min + Vec2::new(col as f32 * cell, row as f32 * cell),
                Vec2::splat(cell),
            );
            painter.rect_filled(cell_rect, 0.0, color);
        }
    }
    painter.rect_stroke(
        loupe_rect,
        0.0,
        egui::Stroke::new(2.0, Color32::WHITE),
        egui::StrokeKind::Outside,
    );
    // The exact pixel about to be picked, highlighted.
    let center_rect = Rect::from_min_size(
        loupe_rect.min + Vec2::splat(half as f32 * cell),
        Vec2::splat(cell),
    );
    painter.rect_stroke(
        center_rect,
        0.0,
        egui::Stroke::new(1.5, Color32::BLACK),
        egui::StrokeKind::Outside,
    );
}

/// A small eyedropper icon button — `colorize` (U+E3B8), verified against
/// Material Icons Round's own published codepoints file, same convention
/// `node_icons.rs`/`toast.rs` already use for every other glyph in this
/// app. Returns `true` the frame it's clicked; the caller (which alone
/// knows which specific color param this button belongs to) is
/// responsible for calling [`EyedropperState::start`] with the right
/// [`EyedropperTarget`].
pub fn eyedropper_button(ui: &mut egui::Ui) -> bool {
    ui.add(egui::Button::new(
        egui::RichText::new("\u{e3b8}")
            .family(egui::FontFamily::Name("material_icons_round".into())),
    ))
    .on_hover_text("Pick a color from anywhere on screen")
    .clicked()
}
