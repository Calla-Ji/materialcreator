// SPDX-License-Identifier: MIT OR Apache-2.0
//! Node identifiers.

use serde::{Deserialize, Serialize};

/// A stable identifier for a node within a [`crate::Graph`]. Assigned when
/// the node is added and never reused, even if the node is later removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

/// A stable identifier for a [`crate::Comment`] within a [`crate::Graph`].
/// Shares `Graph`'s own `next_id` counter with [`NodeId`]/[`FrameId`] (one
/// counter for every kind of id a graph hands out, rather than three
/// independent ones to keep in sync) — the counter never resets, so a
/// comment and a node could in principle share the same *number*, but
/// never the same *type*, and the two are never compared against each
/// other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CommentId(pub u64);

/// A stable identifier for a [`crate::Frame`] within a [`crate::Graph`] —
/// see [`CommentId`]'s own doc comment on sharing `next_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FrameId(pub u64);
