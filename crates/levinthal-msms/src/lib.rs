//! # Levinthal MS/MS
//!
//! MS/MS fragmentation and peptide identification using categorical trajectory completion.
//!
//! ## Overview
//!
//! This crate implements mass spectrometry analysis through the Levinthal framework:
//! - Peptide fragmentation patterns (b/y ions)
//! - Fragment mass calculation
//! - Spectrum matching via S-entropy coordinates
//! - Trajectory completion for peptide identification
//!
//! ## Key Insight
//!
//! MS/MS spectra encode trajectories through partition space. Fragment ions
//! represent intermediate states along the peptide's trajectory, and the
//! complete sequence can be derived through categorical completion.

pub mod fragment;
pub mod spectrum;
pub mod peptide;

pub use fragment::{Fragment, FragmentType};
pub use spectrum::Spectrum;
pub use peptide::Peptide;
