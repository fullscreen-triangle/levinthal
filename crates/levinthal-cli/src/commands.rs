//! CLI command implementations.

use levinthal_core::{
    AminoAcid, PartitionState, Trajectory,
    amino_acid::parse_sequence,
    ternary::TernaryString,
};
use levinthal_folding::groel::GroELChamber;

/// Analyze a protein sequence.
pub fn analyze(sequence: &str, format: &str) {
    match parse_sequence(sequence) {
        Ok(aas) => {
            if format == "json" {
                print_json_analysis(&aas);
            } else {
                print_text_analysis(&aas);
            }
        }
        Err(e) => {
            eprintln!("Error parsing sequence: {}", e);
        }
    }
}

fn print_text_analysis(aas: &[AminoAcid]) {
    println!("Sequence Analysis");
    println!("=================");
    println!("Length: {} residues", aas.len());
    println!();

    // Calculate statistics
    let hydrophobic_count = aas.iter().filter(|aa| aa.is_hydrophobic()).count();
    let charged_count = aas.iter().filter(|aa| aa.is_charged()).count();
    let aromatic_count = aas.iter().filter(|aa| aa.is_aromatic()).count();

    println!("Composition:");
    println!("  Hydrophobic: {} ({:.1}%)", hydrophobic_count,
             100.0 * hydrophobic_count as f64 / aas.len() as f64);
    println!("  Charged:     {} ({:.1}%)", charged_count,
             100.0 * charged_count as f64 / aas.len() as f64);
    println!("  Aromatic:    {} ({:.1}%)", aromatic_count,
             100.0 * aromatic_count as f64 / aas.len() as f64);

    // Calculate molecular weight
    let mw: f64 = aas.iter().map(|aa| aa.molecular_weight()).sum();
    println!();
    println!("Molecular Weight: {:.2} Da", mw + 18.015); // Add water

    // Calculate net charge
    let net_charge: f64 = aas.iter().map(|aa| aa.charge()).sum();
    println!("Net Charge (pH 7): {:.1}", net_charge);

    // S-entropy profile summary
    println!();
    println!("S-Entropy Summary:");
    let avg_sk: f64 = aas.iter().map(|aa| aa.sentropy().sk()).sum::<f64>() / aas.len() as f64;
    let avg_st: f64 = aas.iter().map(|aa| aa.sentropy().st()).sum::<f64>() / aas.len() as f64;
    let avg_se: f64 = aas.iter().map(|aa| aa.sentropy().se()).sum::<f64>() / aas.len() as f64;

    println!("  Mean Sₖ (hydrophobicity): {:.3}", avg_sk);
    println!("  Mean Sₜ (volume):         {:.3}", avg_st);
    println!("  Mean Sₑ (electrostatic):  {:.3}", avg_se);
}

fn print_json_analysis(aas: &[AminoAcid]) {
    let mw: f64 = aas.iter().map(|aa| aa.molecular_weight()).sum::<f64>() + 18.015;
    let net_charge: f64 = aas.iter().map(|aa| aa.charge()).sum();
    let avg_sk: f64 = aas.iter().map(|aa| aa.sentropy().sk()).sum::<f64>() / aas.len() as f64;
    let avg_st: f64 = aas.iter().map(|aa| aa.sentropy().st()).sum::<f64>() / aas.len() as f64;
    let avg_se: f64 = aas.iter().map(|aa| aa.sentropy().se()).sum::<f64>() / aas.len() as f64;

    println!("{{");
    println!("  \"length\": {},", aas.len());
    println!("  \"molecular_weight\": {:.2},", mw);
    println!("  \"net_charge\": {:.1},", net_charge);
    println!("  \"sentropy\": {{");
    println!("    \"sk\": {:.4},", avg_sk);
    println!("    \"st\": {:.4},", avg_st);
    println!("    \"se\": {:.4}", avg_se);
    println!("  }}");
    println!("}}");
}

/// Calculate partition states.
pub fn partition(sequence: &str, max_depth: u32) {
    match parse_sequence(sequence) {
        Ok(aas) => {
            println!("Partition State Enumeration");
            println!("===========================");
            println!("Sequence: {} ({} residues)", sequence, aas.len());
            println!();

            for n in 1..=max_depth {
                let capacity = PartitionState::capacity(n);
                println!("Depth n={}: C(n) = {} states", n, capacity);
            }

            println!();
            println!("Cumulative capacity at n={}: {}", max_depth,
                     PartitionState::cumulative_capacity(max_depth));
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}

/// Complete a trajectory from goal state.
pub fn complete(n: u32, l: u32, m: i32, s: f64) {
    match PartitionState::new(n, l, m, s) {
        Ok(goal) => {
            println!("Trajectory Completion");
            println!("=====================");
            println!("Goal state: {}", goal);
            println!();

            let trajectory = Trajectory::complete(goal);

            println!("Trajectory ({} states):", trajectory.len());
            for (i, state) in trajectory.iter().enumerate() {
                let _arrow = if i == 0 { "→" } else { "  →" };
                if i == 0 {
                    println!("  Origin: {}", state);
                } else if i == trajectory.len() - 1 {
                    println!("    Goal: {}", state);
                } else {
                    println!("       {}: {}", i, state);
                }
            }

            println!();
            println!("Coherence profile:");
            for (i, coherence) in trajectory.coherence_profile().iter().enumerate() {
                let bar_len = (coherence * 20.0) as usize;
                let bar: String = "█".repeat(bar_len);
                println!("  {}: {:5.3} |{}|", i, coherence, bar);
            }
        }
        Err(e) => {
            eprintln!("Invalid partition state: {}", e);
        }
    }
}

/// Simulate protein folding.
pub fn fold(residues: usize, coupling: f64, max_steps: usize, dt: f64) {
    println!("GroEL Folding Simulation");
    println!("========================");
    println!("Residues: {}", residues);
    println!("Coupling: {}", coupling);
    println!();

    let mut chamber = GroELChamber::new(residues, coupling);

    println!("Initial coherence: {:.4}", chamber.coherence());
    println!();
    println!("Simulating...");

    let (metrics, folded) = chamber.fold_with_metrics(dt, max_steps);

    println!();
    println!("Results:");
    println!("  Folded: {}", if folded { "Yes" } else { "No" });
    println!("  Final coherence: {:.4}", metrics.r_values().last().unwrap_or(&0.0));
    println!("  ATP cycles: {}", chamber.cycles());
    println!("  Mean coherence: {:.4}", metrics.mean_coherence());
    println!("  Max coherence: {:.4}", metrics.max_coherence());

    if let Some(t) = metrics.time_to_coherence(0.8) {
        println!("  Time to coherence (r > 0.8): step {}", t);
    }
}

/// Analyze MS/MS spectrum.
pub fn msms(input: &str, _output: Option<&str>) {
    println!("MS/MS Analysis");
    println!("==============");
    println!("Input: {}", input);
    println!();
    println!("Note: Full MS/MS file parsing not yet implemented.");
    println!("This will support mzML and MGF formats.");
}

/// Convert ternary string to coordinates.
pub fn ternary(string: &str) {
    match TernaryString::from_digit_string(string) {
        Ok(ts) => {
            println!("Ternary String Analysis");
            println!("=======================");
            println!("Input: {}", ts);
            println!("Length: {} trits", ts.len());
            println!("Resolution: 3^{} = {} cells", ts.len(), ts.resolution());

            let counts = ts.trit_counts();
            println!();
            println!("Trit Distribution:");
            println!("  0 (Sₖ axis): {}", counts[0]);
            println!("  1 (Sₜ axis): {}", counts[1]);
            println!("  2 (Sₑ axis): {}", counts[2]);

            let (lower, upper) = ts.to_cell_bounds();
            println!();
            println!("Cell Bounds:");
            println!("  Sₖ: [{:.4}, {:.4}]", lower[0], upper[0]);
            println!("  Sₜ: [{:.4}, {:.4}]", lower[1], upper[1]);
            println!("  Sₑ: [{:.4}, {:.4}]", lower[2], upper[2]);

            let sentropy = ts.to_sentropy();
            println!();
            println!("S-Entropy Coordinate: {}", sentropy);
        }
        Err(e) => {
            eprintln!("Invalid ternary string: {}", e);
        }
    }
}
