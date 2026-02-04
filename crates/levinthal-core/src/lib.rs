//! # Levinthal Core
//!
//! Core primitives for the Levinthal protein folding framework.
//!
//! This crate provides the fundamental building blocks for representing
//! protein states and trajectories through categorical state navigation:
//!
//! - **Partition Coordinates**: The (n, l, m, s) four-parameter system for
//!   specifying categorical states in bounded phase space
//! - **Ternary Representation**: Position-trajectory unified encoding where
//!   the address IS the path
//! - **S-Entropy Coordinates**: Three-dimensional encoding of amino acids
//!   through physicochemical properties (Sₖ, Sₜ, Sₑ)
//! - **Trajectory**: Ordered sequence of states representing folding pathways
//!
//! ## Core Insight
//!
//! Protein folding is not forward search through conformational space but
//! backward derivation through categorical space. The native structure
//! determines its own folding pathway through the geometry of partition
//! coordinates.
//!
//! ## Example
//!
//! ```rust
//! use levinthal_core::partition::PartitionState;
//! use levinthal_core::ternary::Trit;
//!
//! // Create a partition state
//! let state = PartitionState::new(2, 1, 0, 0.5).unwrap();
//!
//! // Check capacity at depth n=2
//! assert_eq!(PartitionState::capacity(2), 8); // 2n² = 8
//! ```

pub mod partition;
pub mod ternary;
pub mod sentropy;
pub mod trajectory;
pub mod amino_acid;
pub mod error;

pub use error::{LevinthalError, Result};
pub use partition::PartitionState;
pub use ternary::{Trit, Tryte};
pub use sentropy::SEntropyCoord;
pub use trajectory::Trajectory;
pub use amino_acid::AminoAcid;
