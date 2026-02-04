//! Amino Acid Representation
//!
//! Maps the 20 standard amino acids to S-entropy coordinates based on
//! their physicochemical properties.
//!
//! ## Property Mapping
//!
//! - Sₖ ← Hydrophobicity (Kyte-Doolittle scale, normalized)
//! - Sₜ ← Molecular volume (van der Waals, normalized)
//! - Sₑ ← Electrostatic properties (charge + polarity, normalized)
//!
//! ## Categories
//!
//! - **Hydrophobic**: I, L, V, F, M, A, W (high Sₖ)
//! - **Charged**: R, K, D, E (high Sₑ)
//! - **Polar**: N, Q, S, T, H, Y, C (moderate)
//! - **Special**: G, P (unique properties)

use serde::{Deserialize, Serialize};
use crate::error::{LevinthalError, Result};
use crate::sentropy::{SEntropyCoord, SEntropyCategory};

/// The 20 standard amino acids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AminoAcid {
    Ala, // A - Alanine
    Arg, // R - Arginine
    Asn, // N - Asparagine
    Asp, // D - Aspartic acid
    Cys, // C - Cysteine
    Gln, // Q - Glutamine
    Glu, // E - Glutamic acid
    Gly, // G - Glycine
    His, // H - Histidine
    Ile, // I - Isoleucine
    Leu, // L - Leucine
    Lys, // K - Lysine
    Met, // M - Methionine
    Phe, // F - Phenylalanine
    Pro, // P - Proline
    Ser, // S - Serine
    Thr, // T - Threonine
    Trp, // W - Tryptophan
    Tyr, // Y - Tyrosine
    Val, // V - Valine
}

impl AminoAcid {
    /// All 20 standard amino acids.
    pub const ALL: [AminoAcid; 20] = [
        AminoAcid::Ala, AminoAcid::Arg, AminoAcid::Asn, AminoAcid::Asp,
        AminoAcid::Cys, AminoAcid::Gln, AminoAcid::Glu, AminoAcid::Gly,
        AminoAcid::His, AminoAcid::Ile, AminoAcid::Leu, AminoAcid::Lys,
        AminoAcid::Met, AminoAcid::Phe, AminoAcid::Pro, AminoAcid::Ser,
        AminoAcid::Thr, AminoAcid::Trp, AminoAcid::Tyr, AminoAcid::Val,
    ];

    /// Create from single-letter code.
    pub fn from_code(code: char) -> Result<Self> {
        match code.to_ascii_uppercase() {
            'A' => Ok(AminoAcid::Ala),
            'R' => Ok(AminoAcid::Arg),
            'N' => Ok(AminoAcid::Asn),
            'D' => Ok(AminoAcid::Asp),
            'C' => Ok(AminoAcid::Cys),
            'Q' => Ok(AminoAcid::Gln),
            'E' => Ok(AminoAcid::Glu),
            'G' => Ok(AminoAcid::Gly),
            'H' => Ok(AminoAcid::His),
            'I' => Ok(AminoAcid::Ile),
            'L' => Ok(AminoAcid::Leu),
            'K' => Ok(AminoAcid::Lys),
            'M' => Ok(AminoAcid::Met),
            'F' => Ok(AminoAcid::Phe),
            'P' => Ok(AminoAcid::Pro),
            'S' => Ok(AminoAcid::Ser),
            'T' => Ok(AminoAcid::Thr),
            'W' => Ok(AminoAcid::Trp),
            'Y' => Ok(AminoAcid::Tyr),
            'V' => Ok(AminoAcid::Val),
            _ => Err(LevinthalError::UnknownAminoAcid {
                code: code.to_string(),
            }),
        }
    }

    /// Get single-letter code.
    pub fn code(&self) -> char {
        match self {
            AminoAcid::Ala => 'A',
            AminoAcid::Arg => 'R',
            AminoAcid::Asn => 'N',
            AminoAcid::Asp => 'D',
            AminoAcid::Cys => 'C',
            AminoAcid::Gln => 'Q',
            AminoAcid::Glu => 'E',
            AminoAcid::Gly => 'G',
            AminoAcid::His => 'H',
            AminoAcid::Ile => 'I',
            AminoAcid::Leu => 'L',
            AminoAcid::Lys => 'K',
            AminoAcid::Met => 'M',
            AminoAcid::Phe => 'F',
            AminoAcid::Pro => 'P',
            AminoAcid::Ser => 'S',
            AminoAcid::Thr => 'T',
            AminoAcid::Trp => 'W',
            AminoAcid::Tyr => 'Y',
            AminoAcid::Val => 'V',
        }
    }

    /// Get three-letter code.
    pub fn code3(&self) -> &'static str {
        match self {
            AminoAcid::Ala => "Ala",
            AminoAcid::Arg => "Arg",
            AminoAcid::Asn => "Asn",
            AminoAcid::Asp => "Asp",
            AminoAcid::Cys => "Cys",
            AminoAcid::Gln => "Gln",
            AminoAcid::Glu => "Glu",
            AminoAcid::Gly => "Gly",
            AminoAcid::His => "His",
            AminoAcid::Ile => "Ile",
            AminoAcid::Leu => "Leu",
            AminoAcid::Lys => "Lys",
            AminoAcid::Met => "Met",
            AminoAcid::Phe => "Phe",
            AminoAcid::Pro => "Pro",
            AminoAcid::Ser => "Ser",
            AminoAcid::Thr => "Thr",
            AminoAcid::Trp => "Trp",
            AminoAcid::Tyr => "Tyr",
            AminoAcid::Val => "Val",
        }
    }

    /// Get full name.
    pub fn name(&self) -> &'static str {
        match self {
            AminoAcid::Ala => "Alanine",
            AminoAcid::Arg => "Arginine",
            AminoAcid::Asn => "Asparagine",
            AminoAcid::Asp => "Aspartic acid",
            AminoAcid::Cys => "Cysteine",
            AminoAcid::Gln => "Glutamine",
            AminoAcid::Glu => "Glutamic acid",
            AminoAcid::Gly => "Glycine",
            AminoAcid::His => "Histidine",
            AminoAcid::Ile => "Isoleucine",
            AminoAcid::Leu => "Leucine",
            AminoAcid::Lys => "Lysine",
            AminoAcid::Met => "Methionine",
            AminoAcid::Phe => "Phenylalanine",
            AminoAcid::Pro => "Proline",
            AminoAcid::Ser => "Serine",
            AminoAcid::Thr => "Threonine",
            AminoAcid::Trp => "Tryptophan",
            AminoAcid::Tyr => "Tyrosine",
            AminoAcid::Val => "Valine",
        }
    }

    /// Get Kyte-Doolittle hydrophobicity value.
    ///
    /// Scale: -4.5 (most hydrophilic) to +4.5 (most hydrophobic)
    pub fn hydrophobicity(&self) -> f64 {
        match self {
            AminoAcid::Ala => 1.8,
            AminoAcid::Arg => -4.5,
            AminoAcid::Asn => -3.5,
            AminoAcid::Asp => -3.5,
            AminoAcid::Cys => 2.5,
            AminoAcid::Gln => -3.5,
            AminoAcid::Glu => -3.5,
            AminoAcid::Gly => -0.4,
            AminoAcid::His => -3.2,
            AminoAcid::Ile => 4.5,
            AminoAcid::Leu => 3.8,
            AminoAcid::Lys => -3.9,
            AminoAcid::Met => 1.9,
            AminoAcid::Phe => 2.8,
            AminoAcid::Pro => -1.6,
            AminoAcid::Ser => -0.8,
            AminoAcid::Thr => -0.7,
            AminoAcid::Trp => -0.9,
            AminoAcid::Tyr => -1.3,
            AminoAcid::Val => 4.2,
        }
    }

    /// Get molecular weight (Da).
    pub fn molecular_weight(&self) -> f64 {
        match self {
            AminoAcid::Ala => 89.09,
            AminoAcid::Arg => 174.20,
            AminoAcid::Asn => 132.12,
            AminoAcid::Asp => 133.10,
            AminoAcid::Cys => 121.15,
            AminoAcid::Gln => 146.15,
            AminoAcid::Glu => 147.13,
            AminoAcid::Gly => 75.07,
            AminoAcid::His => 155.16,
            AminoAcid::Ile => 131.17,
            AminoAcid::Leu => 131.17,
            AminoAcid::Lys => 146.19,
            AminoAcid::Met => 149.21,
            AminoAcid::Phe => 165.19,
            AminoAcid::Pro => 115.13,
            AminoAcid::Ser => 105.09,
            AminoAcid::Thr => 119.12,
            AminoAcid::Trp => 204.23,
            AminoAcid::Tyr => 181.19,
            AminoAcid::Val => 117.15,
        }
    }

    /// Get van der Waals volume (Å³).
    pub fn volume(&self) -> f64 {
        match self {
            AminoAcid::Ala => 88.6,
            AminoAcid::Arg => 173.4,
            AminoAcid::Asn => 114.1,
            AminoAcid::Asp => 111.1,
            AminoAcid::Cys => 108.5,
            AminoAcid::Gln => 143.8,
            AminoAcid::Glu => 138.4,
            AminoAcid::Gly => 60.1,
            AminoAcid::His => 153.2,
            AminoAcid::Ile => 166.7,
            AminoAcid::Leu => 166.7,
            AminoAcid::Lys => 168.6,
            AminoAcid::Met => 162.9,
            AminoAcid::Phe => 189.9,
            AminoAcid::Pro => 112.7,
            AminoAcid::Ser => 89.0,
            AminoAcid::Thr => 116.1,
            AminoAcid::Trp => 227.8,
            AminoAcid::Tyr => 193.6,
            AminoAcid::Val => 140.0,
        }
    }

    /// Get charge at pH 7.
    pub fn charge(&self) -> f64 {
        match self {
            AminoAcid::Arg | AminoAcid::Lys => 1.0,
            AminoAcid::His => 0.5, // Partially protonated at pH 7
            AminoAcid::Asp | AminoAcid::Glu => -1.0,
            _ => 0.0,
        }
    }

    /// Get S-entropy coordinates.
    ///
    /// Maps physicochemical properties to (Sₖ, Sₜ, Sₑ) ∈ [0, 1]³.
    pub fn sentropy(&self) -> SEntropyCoord {
        // Normalize hydrophobicity: [-4.5, 4.5] → [0, 1]
        let sk = (self.hydrophobicity() + 4.5) / 9.0;

        // Normalize volume: [60, 228] → [0, 1]
        let st = (self.volume() - 60.0) / 168.0;

        // Electrostatic: combine charge and polarity
        // Charged residues get high Se, polar moderate, hydrophobic low
        let se = match self {
            AminoAcid::Arg | AminoAcid::Lys => 0.95,
            AminoAcid::Asp | AminoAcid::Glu => 0.85,
            AminoAcid::His => 0.60,
            AminoAcid::Asn | AminoAcid::Gln => 0.47,
            AminoAcid::Ser | AminoAcid::Thr => 0.42,
            AminoAcid::Cys => 0.25,
            AminoAcid::Tyr => 0.28,
            AminoAcid::Trp => 0.20,
            AminoAcid::Gly => 0.35,
            AminoAcid::Pro => 0.32,
            _ => 0.15, // Hydrophobic residues
        };

        SEntropyCoord::new_clamped(sk, st, se)
    }

    /// Get the S-entropy category.
    pub fn category(&self) -> SEntropyCategory {
        match self {
            AminoAcid::Ile | AminoAcid::Leu | AminoAcid::Val |
            AminoAcid::Phe | AminoAcid::Met | AminoAcid::Ala |
            AminoAcid::Trp => SEntropyCategory::Hydrophobic,

            AminoAcid::Arg | AminoAcid::Lys |
            AminoAcid::Asp | AminoAcid::Glu => SEntropyCategory::Charged,

            AminoAcid::Asn | AminoAcid::Gln | AminoAcid::Ser |
            AminoAcid::Thr | AminoAcid::His | AminoAcid::Tyr |
            AminoAcid::Cys => SEntropyCategory::Polar,

            AminoAcid::Gly | AminoAcid::Pro => SEntropyCategory::Special,
        }
    }

    /// Check if this is a hydrophobic residue.
    pub fn is_hydrophobic(&self) -> bool {
        self.category() == SEntropyCategory::Hydrophobic
    }

    /// Check if this is a charged residue.
    pub fn is_charged(&self) -> bool {
        self.category() == SEntropyCategory::Charged
    }

    /// Check if this is a positively charged residue.
    pub fn is_positive(&self) -> bool {
        matches!(self, AminoAcid::Arg | AminoAcid::Lys | AminoAcid::His)
    }

    /// Check if this is a negatively charged residue.
    pub fn is_negative(&self) -> bool {
        matches!(self, AminoAcid::Asp | AminoAcid::Glu)
    }

    /// Check if this is a small residue.
    pub fn is_small(&self) -> bool {
        self.volume() < 100.0
    }

    /// Check if this is an aromatic residue.
    pub fn is_aromatic(&self) -> bool {
        matches!(self, AminoAcid::Phe | AminoAcid::Tyr | AminoAcid::Trp | AminoAcid::His)
    }
}

impl std::fmt::Display for AminoAcid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code())
    }
}

impl TryFrom<char> for AminoAcid {
    type Error = LevinthalError;

    fn try_from(c: char) -> Result<Self> {
        AminoAcid::from_code(c)
    }
}

/// Parse a protein sequence string into amino acids.
pub fn parse_sequence(seq: &str) -> Result<Vec<AminoAcid>> {
    seq.chars()
        .filter(|c| !c.is_whitespace())
        .map(AminoAcid::from_code)
        .collect()
}

/// Convert amino acids back to sequence string.
pub fn to_sequence_string(aas: &[AminoAcid]) -> String {
    aas.iter().map(|aa| aa.code()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_from_code() {
        assert_eq!(AminoAcid::from_code('A').unwrap(), AminoAcid::Ala);
        assert_eq!(AminoAcid::from_code('r').unwrap(), AminoAcid::Arg); // lowercase
        assert!(AminoAcid::from_code('X').is_err());
    }

    #[test]
    fn test_code_roundtrip() {
        for aa in AminoAcid::ALL {
            let code = aa.code();
            let recovered = AminoAcid::from_code(code).unwrap();
            assert_eq!(aa, recovered);
        }
    }

    #[test]
    fn test_hydrophobicity_range() {
        let hydros: Vec<f64> = AminoAcid::ALL.iter().map(|aa| aa.hydrophobicity()).collect();
        let min = hydros.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = hydros.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        assert_relative_eq!(min, -4.5); // Arg
        assert_relative_eq!(max, 4.5);  // Ile
    }

    #[test]
    fn test_sentropy_in_range() {
        for aa in AminoAcid::ALL {
            let s = aa.sentropy();
            assert!(s.sk() >= 0.0 && s.sk() <= 1.0, "{} Sk out of range", aa);
            assert!(s.st() >= 0.0 && s.st() <= 1.0, "{} St out of range", aa);
            assert!(s.se() >= 0.0 && s.se() <= 1.0, "{} Se out of range", aa);
        }
    }

    #[test]
    fn test_hydrophobic_high_sk() {
        let ile = AminoAcid::Ile.sentropy();
        let leu = AminoAcid::Leu.sentropy();
        let val = AminoAcid::Val.sentropy();

        assert!(ile.sk() > 0.9);
        assert!(leu.sk() > 0.9);
        assert!(val.sk() > 0.9);
    }

    #[test]
    fn test_charged_high_se() {
        let arg = AminoAcid::Arg.sentropy();
        let lys = AminoAcid::Lys.sentropy();
        let asp = AminoAcid::Asp.sentropy();
        let glu = AminoAcid::Glu.sentropy();

        assert!(arg.se() > 0.8);
        assert!(lys.se() > 0.8);
        assert!(asp.se() > 0.8);
        assert!(glu.se() > 0.8);
    }

    #[test]
    fn test_parse_sequence() {
        let seq = parse_sequence("MVLSPADKTNVKAAW").unwrap();
        assert_eq!(seq.len(), 15);
        assert_eq!(seq[0], AminoAcid::Met);
        assert_eq!(seq[1], AminoAcid::Val);
    }

    #[test]
    fn test_category_assignment() {
        assert_eq!(AminoAcid::Ile.category(), SEntropyCategory::Hydrophobic);
        assert_eq!(AminoAcid::Arg.category(), SEntropyCategory::Charged);
        assert_eq!(AminoAcid::Ser.category(), SEntropyCategory::Polar);
        assert_eq!(AminoAcid::Gly.category(), SEntropyCategory::Special);
    }
}
