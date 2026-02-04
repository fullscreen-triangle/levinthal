//! # Levinthal Folding
//!
//! Phase-lock dynamics for protein folding using Kuramoto oscillators.
//!
//! ## Overview
//!
//! This crate implements the dynamical aspect of the Levinthal framework:
//! - Kuramoto oscillator networks modeling H-bond synchronization
//! - Phase coherence calculation (order parameter ⟨r⟩)
//! - GroEL resonance chamber simulation
//! - ATP-driven frequency modulation
//!
//! ## Key Insight
//!
//! Protein folding is phase synchronization. The hydrogen bond network
//! behaves as coupled oscillators that synchronize through phase-locking.
//! When coherence ⟨r⟩ exceeds the critical threshold (~0.8), the protein
//! has found its native state.
//!
//! ## GroEL Mechanism
//!
//! The GroEL chaperonin acts as a resonance chamber:
//! 1. ATP hydrolysis provides frequency modulation (±15% around 13.2 THz)
//! 2. The cavity geometry selects for coherent states
//! 3. 7 ATP molecules create standing wave patterns
//!
//! This crate provides simulation tools for studying these dynamics.

pub mod kuramoto;
pub mod coherence;
pub mod groel;

pub use kuramoto::{KuramotoOscillator, KuramotoNetwork};
pub use coherence::{OrderParameter, CoherenceMetrics};
pub use groel::GroELChamber;
