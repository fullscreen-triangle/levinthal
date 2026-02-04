//! Mass Spectrum Representation
//!
//! Encodes observed peaks in a mass spectrum as categorical states.

use serde::{Deserialize, Serialize};

/// A peak in a mass spectrum.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Peak {
    /// Mass-to-charge ratio
    pub mz: f64,
    /// Intensity (arbitrary units)
    pub intensity: f64,
}

impl Peak {
    /// Create a new peak.
    pub fn new(mz: f64, intensity: f64) -> Self {
        Self { mz, intensity }
    }
}

/// A mass spectrum (collection of peaks).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spectrum {
    /// Precursor m/z
    precursor_mz: f64,
    /// Precursor charge
    precursor_charge: u8,
    /// Peaks in the spectrum
    peaks: Vec<Peak>,
}

impl Spectrum {
    /// Create a new spectrum.
    pub fn new(precursor_mz: f64, precursor_charge: u8) -> Self {
        Self {
            precursor_mz,
            precursor_charge,
            peaks: Vec::new(),
        }
    }

    /// Create with peaks.
    pub fn with_peaks(precursor_mz: f64, precursor_charge: u8, peaks: Vec<Peak>) -> Self {
        Self {
            precursor_mz,
            precursor_charge,
            peaks,
        }
    }

    /// Get precursor m/z.
    pub fn precursor_mz(&self) -> f64 {
        self.precursor_mz
    }

    /// Get precursor charge.
    pub fn precursor_charge(&self) -> u8 {
        self.precursor_charge
    }

    /// Get peaks.
    pub fn peaks(&self) -> &[Peak] {
        &self.peaks
    }

    /// Add a peak.
    pub fn add_peak(&mut self, mz: f64, intensity: f64) {
        self.peaks.push(Peak::new(mz, intensity));
    }

    /// Number of peaks.
    pub fn len(&self) -> usize {
        self.peaks.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.peaks.is_empty()
    }

    /// Calculate precursor neutral mass.
    pub fn precursor_mass(&self) -> f64 {
        const PROTON_MASS: f64 = 1.007276;
        (self.precursor_mz - PROTON_MASS) * self.precursor_charge as f64
    }

    /// Find peaks within tolerance of a target m/z.
    pub fn find_peaks(&self, target_mz: f64, tolerance_ppm: f64) -> Vec<&Peak> {
        let tolerance = target_mz * tolerance_ppm / 1_000_000.0;
        self.peaks
            .iter()
            .filter(|p| (p.mz - target_mz).abs() <= tolerance)
            .collect()
    }

    /// Get base peak (highest intensity).
    pub fn base_peak(&self) -> Option<&Peak> {
        self.peaks
            .iter()
            .max_by(|a, b| a.intensity.partial_cmp(&b.intensity).unwrap())
    }

    /// Get total ion current (sum of intensities).
    pub fn total_ion_current(&self) -> f64 {
        self.peaks.iter().map(|p| p.intensity).sum()
    }

    /// Normalize intensities to maximum.
    pub fn normalize(&mut self) {
        if let Some(max_intensity) = self.peaks.iter().map(|p| p.intensity).reduce(f64::max) {
            if max_intensity > 0.0 {
                for peak in &mut self.peaks {
                    peak.intensity /= max_intensity;
                }
            }
        }
    }

    /// Sort peaks by m/z.
    pub fn sort_by_mz(&mut self) {
        self.peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());
    }

    /// Sort peaks by intensity (descending).
    pub fn sort_by_intensity(&mut self) {
        self.peaks.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap());
    }
}

impl Default for Spectrum {
    fn default() -> Self {
        Self::new(0.0, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_spectrum_creation() {
        let mut spec = Spectrum::new(500.5, 2);
        spec.add_peak(200.1, 1000.0);
        spec.add_peak(300.2, 500.0);

        assert_eq!(spec.len(), 2);
        assert_relative_eq!(spec.precursor_mz(), 500.5);
    }

    #[test]
    fn test_find_peaks() {
        let spec = Spectrum::with_peaks(500.0, 2, vec![
            Peak::new(200.0, 100.0),
            Peak::new(200.01, 50.0),
            Peak::new(300.0, 200.0),
        ]);

        let found = spec.find_peaks(200.0, 100.0); // 100 ppm = 0.02 Da at 200
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_base_peak() {
        let spec = Spectrum::with_peaks(500.0, 2, vec![
            Peak::new(200.0, 100.0),
            Peak::new(300.0, 500.0),
            Peak::new(400.0, 200.0),
        ]);

        let base = spec.base_peak().unwrap();
        assert_relative_eq!(base.intensity, 500.0);
    }
}
