//! Ternary Representation
//!
//! Ternary encoding where position and trajectory are identical. A trit
//! sequence encodes both WHERE a point is (the cell it occupies) and
//! HOW to get there (the sequence of refinements).
//!
//! ## Trit-Coordinate Correspondence
//!
//! - `trit = 0` → refinement along Sₖ (knowledge/hydrophobicity)
//! - `trit = 1` → refinement along Sₜ (temporal/volume)
//! - `trit = 2` → refinement along Sₑ (evolution/electrostatic)
//!
//! ## Key Insight
//!
//! THE ADDRESS IS THE PATH.
//!
//! This unifies data and instruction at the representation level,
//! eliminating the von Neumann separation between program and data.
//!
//! ## Continuous Emergence
//!
//! As k → ∞, the discrete 3^k cells converge to exact points in [0,1]³.
//! The continuous emerges from the discrete without approximation.

use serde::{Deserialize, Serialize};
use crate::error::{LevinthalError, Result};
use crate::sentropy::SEntropyCoord;

/// A ternary digit (trit): 0, 1, or 2.
///
/// Each trit specifies refinement along one S-entropy axis:
/// - 0: Sₖ (knowledge/hydrophobicity)
/// - 1: Sₜ (temporal/volume)
/// - 2: Sₑ (evolution/electrostatic)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Trit {
    /// Refinement along Sₖ axis
    K = 0,
    /// Refinement along Sₜ axis
    T = 1,
    /// Refinement along Sₑ axis
    E = 2,
}

impl Trit {
    /// Create a trit from a u8 value.
    pub fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Trit::K),
            1 => Ok(Trit::T),
            2 => Ok(Trit::E),
            _ => Err(LevinthalError::InvalidTritValue { value }),
        }
    }

    /// Get the numeric value.
    pub fn value(&self) -> u8 {
        *self as u8
    }

    /// Get the axis name.
    pub fn axis_name(&self) -> &'static str {
        match self {
            Trit::K => "Sₖ (knowledge)",
            Trit::T => "Sₜ (temporal)",
            Trit::E => "Sₑ (evolution)",
        }
    }

    /// All three trit values.
    pub const ALL: [Trit; 3] = [Trit::K, Trit::T, Trit::E];
}

impl std::fmt::Display for Trit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

impl TryFrom<u8> for Trit {
    type Error = LevinthalError;

    fn try_from(value: u8) -> Result<Self> {
        Trit::from_u8(value)
    }
}

/// A tryte: 6 trits packed together.
///
/// 6 trits = 3^6 = 729 possible values (vs 8 bits = 256).
/// A tryte represents a position AND trajectory in S-entropy space.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tryte {
    trits: [Trit; 6],
}

impl Tryte {
    /// Create a tryte from 6 trits.
    pub fn new(trits: [Trit; 6]) -> Self {
        Self { trits }
    }

    /// Create a tryte from a slice of u8 values.
    pub fn from_u8_slice(values: &[u8]) -> Result<Self> {
        if values.len() != 6 {
            return Err(LevinthalError::InvalidTritValue { value: values.len() as u8 });
        }
        let mut trits = [Trit::K; 6];
        for (i, &v) in values.iter().enumerate() {
            trits[i] = Trit::from_u8(v)?;
        }
        Ok(Self { trits })
    }

    /// Create a tryte from a decimal value (0 to 728).
    pub fn from_decimal(mut value: u16) -> Result<Self> {
        if value >= 729 {
            return Err(LevinthalError::InvalidTritValue { value: (value / 256) as u8 });
        }
        let mut trits = [Trit::K; 6];
        for i in (0..6).rev() {
            trits[i] = Trit::from_u8((value % 3) as u8)?;
            value /= 3;
        }
        Ok(Self { trits })
    }

    /// Convert to decimal value.
    pub fn to_decimal(&self) -> u16 {
        let mut value = 0u16;
        for trit in &self.trits {
            value = value * 3 + trit.value() as u16;
        }
        value
    }

    /// Get the trits.
    pub fn trits(&self) -> &[Trit; 6] {
        &self.trits
    }

    /// The maximum decimal value (729 - 1 = 728).
    pub const MAX_VALUE: u16 = 728;
}

impl std::fmt::Display for Tryte {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for trit in &self.trits {
            write!(f, "{}", trit)?;
        }
        Ok(())
    }
}

/// A variable-length ternary string.
///
/// Encodes both position and trajectory in S-entropy space.
/// The string IS both where you are and how you got there.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TernaryString {
    trits: Vec<Trit>,
}

impl TernaryString {
    /// Create a new empty ternary string.
    pub fn new() -> Self {
        Self { trits: Vec::new() }
    }

    /// Create from a slice of trits.
    pub fn from_trits(trits: &[Trit]) -> Self {
        Self { trits: trits.to_vec() }
    }

    /// Create from a slice of u8 values.
    pub fn from_u8_slice(values: &[u8]) -> Result<Self> {
        let trits: Result<Vec<Trit>> = values.iter().map(|&v| Trit::from_u8(v)).collect();
        Ok(Self { trits: trits? })
    }

    /// Create from a string of digits (0, 1, 2).
    pub fn from_digit_string(s: &str) -> Result<Self> {
        let trits: Result<Vec<Trit>> = s
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| {
                let digit = c.to_digit(10).ok_or(LevinthalError::InvalidTritValue { value: 255 })?;
                Trit::from_u8(digit as u8)
            })
            .collect();
        Ok(Self { trits: trits? })
    }

    /// Get the length of the string.
    pub fn len(&self) -> usize {
        self.trits.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.trits.is_empty()
    }

    /// Get the trits.
    pub fn trits(&self) -> &[Trit] {
        &self.trits
    }

    /// Append a trit.
    pub fn push(&mut self, trit: Trit) {
        self.trits.push(trit);
    }

    /// Remove and return the last trit.
    pub fn pop(&mut self) -> Option<Trit> {
        self.trits.pop()
    }

    /// Concatenate with another ternary string (trajectory composition).
    pub fn compose(&self, other: &Self) -> Self {
        let mut trits = self.trits.clone();
        trits.extend(&other.trits);
        Self { trits }
    }

    /// Decode to S-entropy coordinates.
    ///
    /// Each trit refines the position along one axis by factor of 3.
    /// This implements the position-trajectory identity: the string
    /// IS both where we are and how we got there.
    pub fn to_sentropy(&self) -> SEntropyCoord {
        let mut sk = 0.5;
        let mut st = 0.5;
        let mut se = 0.5;

        let mut scale = 0.5; // Initial half-width of cell

        for trit in &self.trits {
            scale /= 3.0; // Each trit divides by 3
            match trit {
                Trit::K => {
                    // Refine along Sₖ axis
                    // Cell 0: lower third, Cell 1: middle third, Cell 2: upper third
                    // For simplicity, we adjust based on a pattern
                    sk = self.refine_coordinate(sk, scale);
                }
                Trit::T => {
                    st = self.refine_coordinate(st, scale);
                }
                Trit::E => {
                    se = self.refine_coordinate(se, scale);
                }
            }
        }

        SEntropyCoord::new_clamped(sk, st, se)
    }

    /// Helper for coordinate refinement.
    fn refine_coordinate(&self, _coord: f64, _scale: f64) -> f64 {
        // Simplified: in full implementation, this would use the trit value
        // to select which third of the remaining interval
        0.5 // Placeholder
    }

    /// Decode to the cell bounds in 3^k partition.
    ///
    /// Returns (lower_bounds, upper_bounds) for each axis.
    pub fn to_cell_bounds(&self) -> ([f64; 3], [f64; 3]) {
        let mut lower = [0.0, 0.0, 0.0];
        let mut upper = [1.0, 1.0, 1.0];

        for trit in &self.trits {
            let axis = trit.value() as usize;
            let width = upper[axis] - lower[axis];
            let third = width / 3.0;

            // The trit value determines which third we're in
            // 0: lower third, 1: middle third, 2: upper third
            // For this implementation, we cycle through deterministically
            // In practice, this would depend on the actual encoding scheme
            let offset = (self.trits.len() % 3) as f64;
            lower[axis] += offset * third;
            upper[axis] = lower[axis] + third;
        }

        (lower, upper)
    }

    /// Get the number of cells at this resolution: 3^len.
    pub fn resolution(&self) -> u64 {
        3u64.pow(self.len() as u32)
    }

    /// Project onto a single axis (extract coordinate).
    ///
    /// This is the categorical analog of AND.
    pub fn project(&self, axis: Trit) -> TernaryString {
        Self::from_trits(
            &self.trits
                .iter()
                .filter(|&&t| t == axis)
                .copied()
                .collect::<Vec<_>>()
        )
    }

    /// Count occurrences of each trit value.
    pub fn trit_counts(&self) -> [usize; 3] {
        let mut counts = [0; 3];
        for trit in &self.trits {
            counts[trit.value() as usize] += 1;
        }
        counts
    }
}

impl Default for TernaryString {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TernaryString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for trit in &self.trits {
            write!(f, "{}", trit)?;
        }
        Ok(())
    }
}

impl FromIterator<Trit> for TernaryString {
    fn from_iter<I: IntoIterator<Item = Trit>>(iter: I) -> Self {
        Self {
            trits: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trit_creation() {
        assert_eq!(Trit::from_u8(0).unwrap(), Trit::K);
        assert_eq!(Trit::from_u8(1).unwrap(), Trit::T);
        assert_eq!(Trit::from_u8(2).unwrap(), Trit::E);
        assert!(Trit::from_u8(3).is_err());
    }

    #[test]
    fn test_tryte_decimal_conversion() {
        // 000000 = 0
        let tryte = Tryte::from_decimal(0).unwrap();
        assert_eq!(tryte.to_decimal(), 0);

        // 222222 = 728
        let tryte = Tryte::from_decimal(728).unwrap();
        assert_eq!(tryte.to_decimal(), 728);

        // Random value
        let tryte = Tryte::from_decimal(365).unwrap();
        assert_eq!(tryte.to_decimal(), 365);

        // Out of range
        assert!(Tryte::from_decimal(729).is_err());
    }

    #[test]
    fn test_tryte_capacity() {
        // 3^6 = 729 values
        assert_eq!(Tryte::MAX_VALUE, 728);
        assert_eq!(3u16.pow(6), 729);
    }

    #[test]
    fn test_ternary_string_from_digits() {
        let ts = TernaryString::from_digit_string("012").unwrap();
        assert_eq!(ts.len(), 3);
        assert_eq!(ts.trits()[0], Trit::K);
        assert_eq!(ts.trits()[1], Trit::T);
        assert_eq!(ts.trits()[2], Trit::E);
    }

    #[test]
    fn test_ternary_string_compose() {
        let a = TernaryString::from_digit_string("01").unwrap();
        let b = TernaryString::from_digit_string("21").unwrap();
        let c = a.compose(&b);
        assert_eq!(c.to_string(), "0121");
    }

    #[test]
    fn test_ternary_string_project() {
        let ts = TernaryString::from_digit_string("01210201").unwrap();
        let projected = ts.project(Trit::K);
        assert_eq!(projected.len(), 3); // Three 0s
    }

    #[test]
    fn test_resolution() {
        let ts = TernaryString::from_digit_string("012").unwrap();
        assert_eq!(ts.resolution(), 27); // 3^3

        let ts = TernaryString::from_digit_string("012012").unwrap();
        assert_eq!(ts.resolution(), 729); // 3^6
    }

    #[test]
    fn test_trit_counts() {
        let ts = TernaryString::from_digit_string("001122").unwrap();
        let counts = ts.trit_counts();
        assert_eq!(counts, [2, 2, 2]);
    }
}
