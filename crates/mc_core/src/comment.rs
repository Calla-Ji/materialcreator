// SPDX-License-Identifier: MIT OR Apache-2.0
//! A free-floating text annotation on the node graph canvas — not a node:
//! no kind, params, ports, or evaluation. Added via the canvas's own
//! "Add Comment" button; edited through the same Parameters panel a node's
//! params would use, rather than inline text editing on the canvas itself.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comment {
    pub text: String,
    /// Editor canvas position; purely cosmetic, same convention as
    /// `Node::position`.
    #[serde(default)]
    pub position: [f32; 2],
}

impl Comment {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            position: [0.0, 0.0],
        }
    }
}
