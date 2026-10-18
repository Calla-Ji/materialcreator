// SPDX-License-Identifier: MIT OR Apache-2.0
//! A resizable, titled rectangle on the node graph canvas, for visually
//! grouping a cluster of nodes — not a node itself: no kind, params, ports,
//! or evaluation, and dragging/resizing it never moves the nodes inside it
//! (a purely visual/organizational grouping, not a functional one). Added
//! via the canvas's own "Add Frame" button, sized to fit the current
//! selection if there is one; edited (title, color) through the same
//! Parameters panel a node's params would use.

use serde::{Deserialize, Serialize};

/// Large enough to usefully hold a handful of nodes (`NODE_WIDTH` in
/// `mc_app::canvas` is 170) without the artist immediately having to
/// resize it, small enough not to swallow the whole canvas.
pub const DEFAULT_FRAME_SIZE: [f32; 2] = [420.0, 320.0];

/// A muted slate blue, translucent enough that nodes placed over it (and
/// the dot grid underneath) both stay readable — `mc_core` is UI/theme-free
/// (CLAUDE.md §5), so this is a plain, independent default rather than
/// anything from `mc_app::theme`.
pub const DEFAULT_FRAME_COLOR: [f32; 4] = [0.30, 0.36, 0.48, 0.35];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    pub title: String,
    pub color: [f32; 4],
    /// Editor canvas position (top-left corner); purely cosmetic, same
    /// convention as `Node::position`.
    #[serde(default)]
    pub position: [f32; 2],
    #[serde(default = "default_size")]
    pub size: [f32; 2],
}

fn default_size() -> [f32; 2] {
    DEFAULT_FRAME_SIZE
}

impl Frame {
    pub fn new(title: impl Into<String>, position: [f32; 2], size: [f32; 2]) -> Self {
        Self {
            title: title.into(),
            color: DEFAULT_FRAME_COLOR,
            position,
            size,
        }
    }
}
