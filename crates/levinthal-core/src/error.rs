//! Error types for the Levinthal framework.

use thiserror::Error;

/// Result type alias for Levinthal operations.
pub type Result<T> = std::result::Result<T, LevinthalError>;

/// Errors that can occur in the Levinthal framework.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum LevinthalError {
    /// Invalid partition coordinate values.
    #[error("Invalid partition coordinates: {message}")]
    InvalidPartitionCoordinates { message: String },

    /// Selection rule violation in state transition.
    #[error("Selection rule violation: Δl={delta_l}, Δm={delta_m}, Δs={delta_s}")]
    SelectionRuleViolation {
        delta_l: i32,
        delta_m: i32,
        delta_s: i32,
    },

    /// Invalid trit value (must be 0, 1, or 2).
    #[error("Invalid trit value: {value} (must be 0, 1, or 2)")]
    InvalidTritValue { value: u8 },

    /// S-entropy coordinate out of range [0, 1].
    #[error("S-entropy coordinate out of range: {coord} = {value} (must be in [0, 1])")]
    SEntropyOutOfRange { coord: String, value: f64 },

    /// Unknown amino acid code.
    #[error("Unknown amino acid: {code}")]
    UnknownAminoAcid { code: String },

    /// Empty trajectory.
    #[error("Trajectory cannot be empty")]
    EmptyTrajectory,

    /// Trajectory discontinuity (states not adjacent).
    #[error("Trajectory discontinuity at index {index}")]
    TrajectoryDiscontinuity { index: usize },

    /// Phase coherence below threshold.
    #[error("Phase coherence {coherence:.4} below threshold {threshold:.4}")]
    InsufficientCoherence { coherence: f64, threshold: f64 },

    /// Computation did not converge.
    #[error("Computation did not converge after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },
}
