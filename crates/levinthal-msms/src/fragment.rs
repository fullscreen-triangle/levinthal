//! Fragment Ion Representation
//!
//! MS/MS fragmentation produces characteristic ion series.
//! In the Levinthal framework, these fragments correspond to
//! intermediate states along the folding trajectory.

use serde::{Deserialize, Serialize};
use levinthal_core::AminoAcid;

/// Type of fragment ion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FragmentType {
    /// b-ion (N-terminal fragment)
    B,
    /// y-ion (C-terminal fragment)
    Y,
    /// a-ion (b-ion minus CO)
    A,
    /// c-ion (b-ion plus NH)
    C,
    /// x-ion (y-ion plus CO)
    X,
    /// z-ion (y-ion minus NH)
    Z,
}

impl FragmentType {
    /// Check if this is an N-terminal ion.
    pub fn is_n_terminal(&self) -> bool {
        matches!(self, FragmentType::A | FragmentType::B | FragmentType::C)
    }

    /// Check if this is a C-terminal ion.
    pub fn is_c_terminal(&self) -> bool {
        matches!(self, FragmentType::X | FragmentType::Y | FragmentType::Z)
    }
}

impl std::fmt::Display for FragmentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FragmentType::A => write!(f, "a"),
            FragmentType::B => write!(f, "b"),
            FragmentType::C => write!(f, "c"),
            FragmentType::X => write!(f, "x"),
            FragmentType::Y => write!(f, "y"),
            FragmentType::Z => write!(f, "z"),
        }
    }
}

/// A fragment ion from MS/MS fragmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fragment {
    /// Fragment type (b, y, etc.)
    fragment_type: FragmentType,
    /// Position in the sequence (1-indexed)
    position: usize,
    /// Charge state
    charge: u8,
    /// Calculated m/z value
    mz: f64,
    /// Amino acids in this fragment
    residues: Vec<AminoAcid>,
}

impl Fragment {
    /// Create a new fragment.
    pub fn new(
        fragment_type: FragmentType,
        position: usize,
        charge: u8,
        mz: f64,
        residues: Vec<AminoAcid>,
    ) -> Self {
        Self {
            fragment_type,
            position,
            charge,
            mz,
            residues,
        }
    }

    /// Get fragment type.
    pub fn fragment_type(&self) -> FragmentType {
        self.fragment_type
    }

    /// Get position in sequence.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Get charge state.
    pub fn charge(&self) -> u8 {
        self.charge
    }

    /// Get m/z value.
    pub fn mz(&self) -> f64 {
        self.mz
    }

    /// Get residues in this fragment.
    pub fn residues(&self) -> &[AminoAcid] {
        &self.residues
    }

    /// Get fragment annotation string (e.g., "b3+")
    pub fn annotation(&self) -> String {
        let charge_str = if self.charge == 1 {
            String::new()
        } else {
            format!("+{}", self.charge)
        };
        format!("{}{}{}", self.fragment_type, self.position, charge_str)
    }

    /// Calculate neutral mass from m/z.
    pub fn neutral_mass(&self) -> f64 {
        const PROTON_MASS: f64 = 1.007276;
        (self.mz - PROTON_MASS) * self.charge as f64
    }
}

impl std::fmt::Display for Fragment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({:.4})", self.annotation(), self.mz)
    }
}

/// Constants for fragment mass calculation.
pub mod constants {
    /// Mass of a proton (Da)
    pub const PROTON_MASS: f64 = 1.007276;
    /// Mass of water (Da)
    pub const WATER_MASS: f64 = 18.010565;
    /// Mass of ammonia (Da)
    pub const AMMONIA_MASS: f64 = 17.026549;
    /// Mass of CO (Da)
    pub const CO_MASS: f64 = 27.994915;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fragment_type_terminal() {
        assert!(FragmentType::B.is_n_terminal());
        assert!(!FragmentType::B.is_c_terminal());
        assert!(!FragmentType::Y.is_n_terminal());
        assert!(FragmentType::Y.is_c_terminal());
    }

    #[test]
    fn test_fragment_annotation() {
        let frag = Fragment::new(
            FragmentType::B,
            3,
            1,
            350.5,
            vec![AminoAcid::Ala, AminoAcid::Gly, AminoAcid::Ser],
        );
        assert_eq!(frag.annotation(), "b3");

        let frag2 = Fragment::new(FragmentType::Y, 5, 2, 280.2, vec![]);
        assert_eq!(frag2.annotation(), "y5+2");
    }
}
