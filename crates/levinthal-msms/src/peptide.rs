//! Peptide Representation
//!
//! A peptide is a sequence of amino acids that can be analyzed through
//! trajectory completion in partition space.

use serde::{Deserialize, Serialize};
use levinthal_core::{AminoAcid, SEntropyCoord};
use crate::fragment::{Fragment, FragmentType, constants};

/// A peptide sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peptide {
    /// Amino acid sequence
    sequence: Vec<AminoAcid>,
}

impl Peptide {
    /// Create a new peptide from amino acid sequence.
    pub fn new(sequence: Vec<AminoAcid>) -> Self {
        Self { sequence }
    }

    /// Parse from single-letter sequence string.
    pub fn from_string(seq: &str) -> Result<Self, levinthal_core::LevinthalError> {
        let sequence = levinthal_core::amino_acid::parse_sequence(seq)?;
        Ok(Self { sequence })
    }

    /// Get the sequence.
    pub fn sequence(&self) -> &[AminoAcid] {
        &self.sequence
    }

    /// Get sequence length.
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Calculate monoisotopic mass.
    pub fn monoisotopic_mass(&self) -> f64 {
        let aa_mass: f64 = self.sequence.iter().map(|aa| aa.molecular_weight()).sum();
        // Add water for peptide bond formation
        aa_mass + constants::WATER_MASS
    }

    /// Calculate m/z for a given charge state.
    pub fn mz(&self, charge: u8) -> f64 {
        (self.monoisotopic_mass() + (charge as f64) * constants::PROTON_MASS) / (charge as f64)
    }

    /// Get S-entropy coordinates for each residue.
    pub fn sentropy_profile(&self) -> Vec<SEntropyCoord> {
        self.sequence.iter().map(|aa| aa.sentropy()).collect()
    }

    /// Calculate average S-entropy coordinates.
    pub fn average_sentropy(&self) -> SEntropyCoord {
        if self.sequence.is_empty() {
            return SEntropyCoord::center();
        }

        let profile = self.sentropy_profile();
        let n = profile.len() as f64;
        let sk: f64 = profile.iter().map(|s| s.sk()).sum::<f64>() / n;
        let st: f64 = profile.iter().map(|s| s.st()).sum::<f64>() / n;
        let se: f64 = profile.iter().map(|s| s.se()).sum::<f64>() / n;

        SEntropyCoord::new_clamped(sk, st, se)
    }

    /// Generate theoretical b-ions.
    pub fn b_ions(&self, charge: u8) -> Vec<Fragment> {
        let mut ions = Vec::new();
        let mut cumulative_mass = 0.0;

        for (i, &aa) in self.sequence.iter().enumerate() {
            if i == self.sequence.len() - 1 {
                break; // Skip last position (would be complete peptide)
            }

            cumulative_mass += aa.molecular_weight();
            let mz = (cumulative_mass + (charge as f64) * constants::PROTON_MASS) / (charge as f64);

            ions.push(Fragment::new(
                FragmentType::B,
                i + 1,
                charge,
                mz,
                self.sequence[..=i].to_vec(),
            ));
        }

        ions
    }

    /// Generate theoretical y-ions.
    pub fn y_ions(&self, charge: u8) -> Vec<Fragment> {
        let mut ions = Vec::new();
        let mut cumulative_mass = constants::WATER_MASS; // y-ions include C-terminal water

        for (i, &aa) in self.sequence.iter().rev().enumerate() {
            if i == self.sequence.len() - 1 {
                break; // Skip last position
            }

            cumulative_mass += aa.molecular_weight();
            let mz = (cumulative_mass + (charge as f64) * constants::PROTON_MASS) / (charge as f64);

            let start_idx = self.sequence.len() - i - 1;
            ions.push(Fragment::new(
                FragmentType::Y,
                i + 1,
                charge,
                mz,
                self.sequence[start_idx..].to_vec(),
            ));
        }

        ions
    }

    /// Generate all theoretical fragment ions.
    pub fn theoretical_fragments(&self, max_charge: u8) -> Vec<Fragment> {
        let mut fragments = Vec::new();

        for charge in 1..=max_charge {
            fragments.extend(self.b_ions(charge));
            fragments.extend(self.y_ions(charge));
        }

        fragments
    }

    /// Count hydrophobic residues.
    pub fn hydrophobic_count(&self) -> usize {
        self.sequence.iter().filter(|aa| aa.is_hydrophobic()).count()
    }

    /// Count charged residues.
    pub fn charged_count(&self) -> usize {
        self.sequence.iter().filter(|aa| aa.is_charged()).count()
    }

    /// Calculate net charge at pH 7.
    pub fn net_charge(&self) -> f64 {
        self.sequence.iter().map(|aa| aa.charge()).sum()
    }

    /// Get sequence as string.
    pub fn to_string(&self) -> String {
        levinthal_core::amino_acid::to_sequence_string(&self.sequence)
    }
}

impl std::fmt::Display for Peptide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_peptide_creation() {
        let peptide = Peptide::from_string("PEPTIDE").unwrap();
        assert_eq!(peptide.len(), 7);
    }

    #[test]
    fn test_mass_calculation() {
        let peptide = Peptide::from_string("GGG").unwrap();
        // 3 * Gly (75.07) + H2O (18.01) = 243.22
        assert_relative_eq!(peptide.monoisotopic_mass(), 243.22, epsilon = 0.1);
    }

    #[test]
    fn test_fragment_generation() {
        let peptide = Peptide::from_string("ACDE").unwrap();

        let b_ions = peptide.b_ions(1);
        assert_eq!(b_ions.len(), 3); // b1, b2, b3

        let y_ions = peptide.y_ions(1);
        assert_eq!(y_ions.len(), 3); // y1, y2, y3
    }

    #[test]
    fn test_sentropy_profile() {
        let peptide = Peptide::from_string("AKD").unwrap();
        let profile = peptide.sentropy_profile();

        assert_eq!(profile.len(), 3);
        // A is hydrophobic, K is positive, D is negative
        assert!(profile[0].sk() > profile[1].sk()); // Ala more hydrophobic than Lys
    }
}
