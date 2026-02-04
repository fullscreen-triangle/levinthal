//! S-Entropy Coordinate System
//!
//! Three-dimensional coordinate space for encoding amino acids and
//! molecular states through physicochemical properties.
//!
//! ## Coordinates
//!
//! - `Sₖ` (knowledge): Derived from hydrophobicity
//! - `Sₜ` (temporal): Derived from molecular volume (van der Waals)
//! - `Sₑ` (evolution): Derived from electrostatic properties
//!
//! ## Properties
//!
//! - All coordinates are normalized to [0, 1]
//! - The mapping preserves chemical relationships:
//!   - Hydrophobic residues → high Sₖ
//!   - Charged residues → high Sₑ
//!   - Small residues → low Sₜ

use serde::{Deserialize, Serialize};
use crate::error::{LevinthalError, Result};

/// S-entropy coordinates (Sₖ, Sₜ, Sₑ).
///
/// A point in the three-dimensional S-entropy space, where:
/// - Sₖ encodes knowledge/hydrophobicity
/// - Sₜ encodes temporal/volume
/// - Sₑ encodes evolution/electrostatic
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SEntropyCoord {
    /// Knowledge coordinate (hydrophobicity), in [0, 1]
    sk: f64,
    /// Temporal coordinate (molecular volume), in [0, 1]
    st: f64,
    /// Evolution coordinate (electrostatic), in [0, 1]
    se: f64,
}

impl SEntropyCoord {
    /// Create new S-entropy coordinates with validation.
    ///
    /// All coordinates must be in [0, 1].
    pub fn new(sk: f64, st: f64, se: f64) -> Result<Self> {
        if sk < 0.0 || sk > 1.0 {
            return Err(LevinthalError::SEntropyOutOfRange {
                coord: "Sₖ".to_string(),
                value: sk,
            });
        }
        if st < 0.0 || st > 1.0 {
            return Err(LevinthalError::SEntropyOutOfRange {
                coord: "Sₜ".to_string(),
                value: st,
            });
        }
        if se < 0.0 || se > 1.0 {
            return Err(LevinthalError::SEntropyOutOfRange {
                coord: "Sₑ".to_string(),
                value: se,
            });
        }
        Ok(Self { sk, st, se })
    }

    /// Create new S-entropy coordinates, clamping to [0, 1].
    pub fn new_clamped(sk: f64, st: f64, se: f64) -> Self {
        Self {
            sk: sk.clamp(0.0, 1.0),
            st: st.clamp(0.0, 1.0),
            se: se.clamp(0.0, 1.0),
        }
    }

    /// Create origin point (0, 0, 0).
    pub fn origin() -> Self {
        Self { sk: 0.0, st: 0.0, se: 0.0 }
    }

    /// Create center point (0.5, 0.5, 0.5).
    pub fn center() -> Self {
        Self { sk: 0.5, st: 0.5, se: 0.5 }
    }

    /// Get Sₖ (knowledge/hydrophobicity).
    pub fn sk(&self) -> f64 {
        self.sk
    }

    /// Get Sₜ (temporal/volume).
    pub fn st(&self) -> f64 {
        self.st
    }

    /// Get Sₑ (evolution/electrostatic).
    pub fn se(&self) -> f64 {
        self.se
    }

    /// Get coordinates as array [Sₖ, Sₜ, Sₑ].
    pub fn as_array(&self) -> [f64; 3] {
        [self.sk, self.st, self.se]
    }

    /// Euclidean distance to another point.
    pub fn distance(&self, other: &Self) -> f64 {
        let dk = self.sk - other.sk;
        let dt = self.st - other.st;
        let de = self.se - other.se;
        (dk * dk + dt * dt + de * de).sqrt()
    }

    /// Manhattan distance to another point.
    pub fn manhattan_distance(&self, other: &Self) -> f64 {
        (self.sk - other.sk).abs() + (self.st - other.st).abs() + (self.se - other.se).abs()
    }

    /// Linear interpolation between two points.
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self::new_clamped(
            self.sk + t * (other.sk - self.sk),
            self.st + t * (other.st - self.st),
            self.se + t * (other.se - self.se),
        )
    }

    /// Calculate the entropy of this coordinate (information content).
    ///
    /// Uses Shannon entropy: -Σ pᵢ log₂(pᵢ)
    /// Treats coordinates as a probability distribution after normalization.
    pub fn shannon_entropy(&self) -> f64 {
        let sum = self.sk + self.st + self.se;
        if sum < 1e-10 {
            return 0.0;
        }

        let probs = [self.sk / sum, self.st / sum, self.se / sum];
        let mut entropy = 0.0;
        for p in probs {
            if p > 1e-10 {
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Normalize coordinates to sum to 1 (convert to probability distribution).
    pub fn normalize(&self) -> Self {
        let sum = self.sk + self.st + self.se;
        if sum < 1e-10 {
            return Self::center();
        }
        Self {
            sk: self.sk / sum,
            st: self.st / sum,
            se: self.se / sum,
        }
    }

    /// Calculate variance of the three coordinates.
    pub fn variance(&self) -> f64 {
        let mean = (self.sk + self.st + self.se) / 3.0;
        let var = ((self.sk - mean).powi(2) + (self.st - mean).powi(2) + (self.se - mean).powi(2)) / 3.0;
        var
    }

    /// Check if this point is in the hydrophobic region (high Sₖ).
    pub fn is_hydrophobic(&self) -> bool {
        self.sk > 0.6
    }

    /// Check if this point is in the charged region (high Sₑ).
    pub fn is_charged(&self) -> bool {
        self.se > 0.6
    }

    /// Check if this point is in the small residue region (low Sₜ).
    pub fn is_small(&self) -> bool {
        self.st < 0.4
    }

    /// Classify into amino acid category based on coordinates.
    pub fn classify(&self) -> SEntropyCategory {
        if self.is_hydrophobic() {
            SEntropyCategory::Hydrophobic
        } else if self.is_charged() {
            SEntropyCategory::Charged
        } else if self.sk < 0.4 && self.se > 0.3 {
            SEntropyCategory::Polar
        } else {
            SEntropyCategory::Special
        }
    }
}

impl Default for SEntropyCoord {
    fn default() -> Self {
        Self::center()
    }
}

impl std::fmt::Display for SEntropyCoord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(Sₖ={:.3}, Sₜ={:.3}, Sₑ={:.3})", self.sk, self.st, self.se)
    }
}

impl From<[f64; 3]> for SEntropyCoord {
    fn from(arr: [f64; 3]) -> Self {
        Self::new_clamped(arr[0], arr[1], arr[2])
    }
}

impl From<SEntropyCoord> for [f64; 3] {
    fn from(coord: SEntropyCoord) -> Self {
        coord.as_array()
    }
}

/// Categories of amino acids based on S-entropy coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SEntropyCategory {
    /// High Sₖ: I, L, V, F, M, A, W
    Hydrophobic,
    /// High Sₑ: R, K, D, E
    Charged,
    /// Moderate Sₖ, moderate Sₑ: N, Q, S, T, H, Y, C
    Polar,
    /// Unique properties: G, P
    Special,
}

impl std::fmt::Display for SEntropyCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SEntropyCategory::Hydrophobic => write!(f, "Hydrophobic"),
            SEntropyCategory::Charged => write!(f, "Charged"),
            SEntropyCategory::Polar => write!(f, "Polar"),
            SEntropyCategory::Special => write!(f, "Special"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_valid_coordinates() {
        let coord = SEntropyCoord::new(0.5, 0.3, 0.8).unwrap();
        assert_relative_eq!(coord.sk(), 0.5);
        assert_relative_eq!(coord.st(), 0.3);
        assert_relative_eq!(coord.se(), 0.8);
    }

    #[test]
    fn test_invalid_coordinates() {
        assert!(SEntropyCoord::new(-0.1, 0.5, 0.5).is_err());
        assert!(SEntropyCoord::new(0.5, 1.1, 0.5).is_err());
        assert!(SEntropyCoord::new(0.5, 0.5, -0.01).is_err());
    }

    #[test]
    fn test_clamped() {
        let coord = SEntropyCoord::new_clamped(-0.5, 1.5, 0.5);
        assert_relative_eq!(coord.sk(), 0.0);
        assert_relative_eq!(coord.st(), 1.0);
        assert_relative_eq!(coord.se(), 0.5);
    }

    #[test]
    fn test_distance() {
        let a = SEntropyCoord::origin();
        let b = SEntropyCoord::new(1.0, 0.0, 0.0).unwrap();
        assert_relative_eq!(a.distance(&b), 1.0);

        let c = SEntropyCoord::new(1.0, 1.0, 1.0).unwrap();
        assert_relative_eq!(a.distance(&c), 3.0_f64.sqrt());
    }

    #[test]
    fn test_lerp() {
        let a = SEntropyCoord::origin();
        let b = SEntropyCoord::new(1.0, 1.0, 1.0).unwrap();

        let mid = a.lerp(&b, 0.5);
        assert_relative_eq!(mid.sk(), 0.5);
        assert_relative_eq!(mid.st(), 0.5);
        assert_relative_eq!(mid.se(), 0.5);
    }

    #[test]
    fn test_classification() {
        let hydro = SEntropyCoord::new(0.8, 0.5, 0.2).unwrap();
        assert_eq!(hydro.classify(), SEntropyCategory::Hydrophobic);

        let charged = SEntropyCoord::new(0.2, 0.5, 0.9).unwrap();
        assert_eq!(charged.classify(), SEntropyCategory::Charged);
    }

    #[test]
    fn test_shannon_entropy() {
        // Maximum entropy when all coordinates equal
        let uniform = SEntropyCoord::new(0.33, 0.33, 0.34).unwrap();
        let entropy = uniform.shannon_entropy();
        assert!(entropy > 1.5); // Close to log₂(3) ≈ 1.585

        // Lower entropy when one coordinate dominates
        let peaked = SEntropyCoord::new(0.9, 0.05, 0.05).unwrap();
        let entropy = peaked.shannon_entropy();
        assert!(entropy < 1.0);
    }
}
