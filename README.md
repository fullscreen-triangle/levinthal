<h1 align="center">Levinthal</h1>
<p align="center"><em>Only Shakespear could write as Shakespear</em></p>

<p align="center">
  <img src="assets/img/monkey-typing.jpg" alt=" Logo" width="400"/>
</p>

A computational framework for protein folding based on categorical trajectory completion through partition space.

## Overview

This framework addresses Levinthal's paradox—the observation that proteins fold in milliseconds despite an astronomically large conformational search space (estimated at 3^N for N residues). The resolution lies not in accelerating forward search but in recognizing that folding operates through backward derivation in categorical space.

The native structure determines its own folding pathway. Given any goal state, the trajectory to reach it is uniquely determined by the geometry of partition coordinates. This converts the O(3^N) forward search problem into O(log₃ N) backward derivation.

## Theoretical Foundation

### Partition Coordinates

States in bounded phase space are specified by four parameters (n, l, m, s):

- **n ≥ 1**: Depth (nesting level in phase space)
- **l ∈ {0, 1, ..., n-1}**: Complexity (boundary shape)
- **m ∈ {-l, ..., +l}**: Orientation (angular position)
- **s ∈ {-½, +½}**: Chirality (handedness)

The capacity at depth n is C(n) = 2n², matching the electron shell structure of atomic orbitals. This is not coincidental—both arise from the geometry of bounded oscillatory systems.

### Selection Rules

Allowed transitions satisfy:

- Δl = ±1 (complexity changes by exactly one)
- |Δm| ≤ 1 (orientation changes by at most one)
- Δs = 0 (chirality is conserved)

These constraints reduce the effective dimensionality of trajectory space, making categorical completion tractable.

### Trajectory Completion

Given a goal state G, the trajectory T is derived by iteratively applying the partition operation:

```
T = [partition^k(G), ..., partition²(G), partition(G), G]
```

where partition^k(G) is the origin state. The trajectory is deterministic—there is exactly one path from any origin to any goal.

### S-Entropy Coordinates

Amino acids are mapped to a three-dimensional coordinate space (Sₖ, Sₜ, Sₑ) ∈ [0,1]³:

- **Sₖ**: Derived from hydrophobicity (Kyte-Doolittle scale)
- **Sₜ**: Derived from molecular volume (van der Waals)
- **Sₑ**: Derived from electrostatic properties

This mapping preserves chemical relationships: hydrophobic residues cluster at high Sₖ, charged residues at high Sₑ, and small residues at low Sₜ.

### Ternary Representation

Position and trajectory are encoded identically using ternary strings. Each trit (0, 1, 2) specifies refinement along one S-entropy axis:

- 0 → refinement along Sₖ
- 1 → refinement along Sₜ
- 2 → refinement along Sₑ

A ternary string of length k specifies a cell in the 3^k partition of [0,1]³. The string IS both where a point is (position) and how to reach it (trajectory). This unifies data and instruction at the representation level.

### Phase-Lock Dynamics

The hydrogen bond network behaves as coupled Kuramoto oscillators:

```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
```

Folding corresponds to phase synchronization. The order parameter r = |⟨e^(iθ)⟩| measures coherence:

- r → 0: Incoherent (unfolded)
- r → 1: Synchronized (native state)

The critical coherence threshold for native state formation is approximately r > 0.8.

### GroEL Mechanism

The GroEL chaperonin functions as a resonance chamber:

1. The barrel cavity creates geometric constraints selecting for coherent states
2. ATP hydrolysis (7 molecules per ring) provides frequency modulation at ±15% around 13.2 THz
3. Standing wave patterns destabilize incoherent configurations
4. Only trajectories satisfying selection rules persist through multiple ATP cycles

## Crate Structure

```
crates/
├── levinthal-core/      Core primitives
│   ├── partition.rs     Partition coordinates (n, l, m, s)
│   ├── trajectory.rs    Trajectory completion
│   ├── ternary.rs       Trit/Tryte/TernaryString
│   ├── sentropy.rs      S-entropy coordinates
│   └── amino_acid.rs    20 standard amino acids
│
├── levinthal-msms/      Mass spectrometry
│   ├── fragment.rs      b/y ion series
│   ├── spectrum.rs      Peak representation
│   └── peptide.rs       Peptide analysis
│
├── levinthal-folding/   Dynamics
│   ├── kuramoto.rs      Coupled oscillator networks
│   ├── coherence.rs     Order parameter metrics
│   └── groel.rs         Resonance chamber simulation
│
└── levinthal-cli/       Command-line interface
```

## Usage

### Analyze a Protein Sequence

```bash
levinthal analyze --sequence "MVLSPADKTNVKAAW"
```

Outputs composition statistics, molecular weight, net charge, and S-entropy profile.

### Complete a Trajectory

```bash
levinthal complete -n 3 -l 2 -m 1 -s 0.5
```

Derives the unique trajectory from origin to the specified goal state.

### Simulate Folding Dynamics

```bash
levinthal fold --residues 50 --coupling 2.0 --max-steps 10000
```

Runs Kuramoto dynamics with GroEL-like frequency modulation until coherence threshold is reached.

### Analyze Ternary Encoding

```bash
levinthal ternary --string "012102"
```

Converts ternary string to S-entropy coordinates and cell bounds.

## Building

```bash
cargo build --workspace
cargo test --workspace
```

Requires Rust 1.70 or later.

## References

The theoretical framework draws on:

- Levinthal, C. (1969). How to fold graciously. Mössbauer Spectroscopy in Biological Systems.
- Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence.
- Thirumalai, D. & Lorimer, G.H. (2001). Chaperonin-mediated protein folding. Annual Review of Biophysics.
- Kyte, J. & Doolittle, R.F. (1982). A simple method for displaying the hydropathic character of a protein. Journal of Molecular Biology.

## License

MIT
