// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared custom window chrome for every modal popup in this app (Export,
//! SVG Import, the "Overwrite existing files?" and "Recover unsaved
//! work?" confirms, and anything added later) — Neutron Star fill,
//! Supernova Light border, a dock-panel-style header bar painted by hand
//! instead of `egui::Window`'s own default title bar, always
//! screen-centered. One place, so every popup actually stays visually
//! consistent instead of drifting the way the Export dialog and the two
//! confirm windows had before this existed (plain default egui chrome,
//! not even centered in the Export dialog's own case — the owner's own
//! report).
//!
//! `splash.rs`'s own window is the original template this recipe was
//! pulled from, but it stays its own hand-rolled implementation rather
//! than switching to this: it's also draggable and has no close button,
//! neither of which this generic version supports, and forcing both
//! shapes into one function would make it harder to read, not easier.

/// Every dialog in the app shares this width, for visual consistency —
/// override only with a real reason (so far, none has needed to).
pub const DEFAULT_WIDTH: f32 = 420.0;

const HEADER_HEIGHT: f32 = 28.0;
const HEADER_GAP: f32 = 8.0;
const CONTENT_PADDING: f32 = 12.0;
const CORNER_RADIUS: f32 = 8.0;
/// Also used by `splash.rs`'s own hand-rolled header, which needs this
/// to size the close button it positions itself (see that module's own
/// doc comment for why it doesn't use this file's `show` wrapper
/// wholesale).
pub(crate) const CLOSE_BUTTON_SIZE: f32 = 20.0;
/// "close", confirmed against Google's own
/// `MaterialIconsRound-Regular.codepoints` file — same convention every
/// other icon glyph constant in this app follows.
const CLOSE_GLYPH: char = '\u{e5cd}';

/// Square close button in the header's top-right corner, drawn from raw
/// primitives (same reasoning as `twod_toolbar.rs`'s own `icon_button`)
/// so it reads as a real app control rather than `egui::Window`'s own
/// default title-bar X, which `title_bar(false)` removes entirely.
/// `pub(crate)` so `splash.rs` can share the same glyph/hover styling
/// for its own header's close button instead of redrawing it by hand.
pub(crate) fn close_button(ui: &mut egui::Ui, id_source: &str, rect: egui::Rect) -> bool {
    let response = ui.interact(
        rect,
        ui.id().with((id_source, "close")),
        egui::Sense::click(),
    );
    let painter = ui.painter();
    if response.hovered() {
        painter.rect_filled(rect, 4.0, ui.visuals().widgets.hovered.weak_bg_fill);
    }
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        CLOSE_GLYPH.to_string(),
        egui::FontId::new(16.0, egui::FontFamily::Name("material_icons_round".into())),
        crate::theme::SUPERNOVA_LIGHT,
    );
    response.clicked()
}

/// Shows `title` in a centered, custom-chrome modal `width` points wide
/// (height auto-sized to `add_contents`). `open` is both input (nothing
/// renders at all while `false`) and output: `closable` dialogs set it
/// `false` themselves when their own header X is clicked; a `closable:
/// false` dialog (a hard confirm — Restore/Discard, Overwrite/Cancel)
/// has no X at all, and can only ever close from *inside*
/// `add_contents` (e.g. its own button handlers setting `*open = false`
/// directly) — deliberately, so a decision the artist must actually make
/// can't be dismissed by accident.
///
/// Returns `add_contents`'s own return value, or `None` if nothing was
/// shown this frame because `*open` was already `false` on entry.
pub fn show<R>(
    ctx: &egui::Context,
    id_source: &str,
    title: &str,
    open: &mut bool,
    closable: bool,
    width: f32,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> Option<R> {
    if !*open {
        return None;
    }
    let mut should_close = false;
    let mut result = None;

    egui::Window::new(id_source)
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .frame(
            egui::Frame::new()
                .fill(crate::theme::NEUTRON_STAR)
                .inner_margin(0)
                .corner_radius(CORNER_RADIUS),
        )
        .show(ctx, |ui| {
            ui.set_width(width);

            let (header_rect, _) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), HEADER_HEIGHT),
                egui::Sense::hover(),
            );
            ui.painter().text(
                header_rect.left_center() + egui::vec2(CONTENT_PADDING, 0.0),
                egui::Align2::LEFT_CENTER,
                title,
                egui::FontId::proportional(15.0),
                crate::theme::SUPERNOVA_LIGHT,
            );
            if closable {
                let close_rect = egui::Rect::from_center_size(
                    header_rect.right_center()
                        - egui::vec2(CONTENT_PADDING + CLOSE_BUTTON_SIZE / 2.0, 0.0),
                    egui::Vec2::splat(CLOSE_BUTTON_SIZE),
                );
                if close_button(ui, id_source, close_rect) {
                    should_close = true;
                }
            }
            ui.painter().hline(
                header_rect.x_range(),
                header_rect.bottom(),
                egui::Stroke::new(1.0, crate::theme::SUPERNOVA_LIGHT),
            );
            ui.add_space(HEADER_GAP);

            egui::Frame::new()
                .inner_margin(egui::Margin::symmetric(CONTENT_PADDING as i8, 0))
                .show(ui, |ui| {
                    // The window itself is `resizable(false)` and
                    // auto-sized to content — with no scroll area, shrinking
                    // the main app window below that content's natural
                    // height left egui constraining the dialog to fit the
                    // viewport by silently clipping its top and bottom, with
                    // no way to reach the clipped part (the owner's own
                    // report). Capping height to the actual available
                    // screen space and scrolling past that instead means
                    // the dialog always fits — a scrollbar only ever
                    // appears when it's genuinely needed.
                    let max_height = (ctx.input(|i| i.viewport_rect()).height()
                        - HEADER_HEIGHT
                        - HEADER_GAP
                        - CONTENT_PADDING * 2.0)
                        .max(120.0);
                    egui::ScrollArea::vertical()
                        .max_height(max_height)
                        .show(ui, |ui| {
                            result = Some(add_contents(ui));
                        });
                    ui.add_space(CONTENT_PADDING);
                });

            // Drawn last, against the bounding box of everything actually
            // laid out above (`ui.min_rect()`) — not a rect captured
            // *before* that content, which for an auto-sized window is
            // still an oversized guess on the very first frame, before
            // egui knows the real content height. Capturing it early (a
            // bug once fixed in the SVG Import dialog this was pulled
            // from) drew the border — and so the whole dialog's apparent
            // outline — stretched down over whatever was behind it.
            ui.painter().rect_stroke(
                ui.min_rect(),
                CORNER_RADIUS,
                egui::Stroke::new(1.0, crate::theme::SUPERNOVA_LIGHT),
                egui::StrokeKind::Inside,
            );
        });

    *open = *open && !should_close;
    result
}
