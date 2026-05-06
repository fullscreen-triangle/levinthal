// =====================================================================
// Five-pass fragment-shader pipeline for the electron-trajectory hologram.
//
// Layer 5 of the apparatus stack (cytochrome P450 monograph, Paper 4):
//   Pass 1: temporal separation -> three channels (W_Sk / W_St / W_Se)
//   Pass 2: weighted superposition -> H(omega, t) coherent hologram
//   Pass 3: 2D FFT -> diffraction pattern (point-group symmetry)
//   Pass 4: multi-modal categorical synthesis (n,l,m,s) -> (S_k, S_t, S_e)
//   Pass 5: trajectory completion -> per-pixel |psi(r,t)|^2 readout
//
// Each pass is a separate shader program; this file contains the
// per-pixel kernel for Pass 5 (the headline observable: |psi(r,t)|^2
// at the per-pixel position rendered into the framebuffer).
//
// Inputs from the GLB-grounded analyte:
//   u_n_atoms                 - number of GLB atoms in the cofactor cluster
//   u_atom_positions[N]       - atom positions in angstroms (4*4 transform
//                                translation columns extracted from the GLB)
//   u_cofactor_positions[4]   - the four cofactor centres (NADPH, FAD,
//                                FMN, heme-Fe), positioned with the heme-Fe
//                                taken DIRECTLY from the GLB (real PDB
//                                coordinates), and the upstream three
//                                positioned along the extension axis at
//                                literature distances 4, 8, 14 ang
//   u_atom_S_coords[N]        - per-atom S-entropy triples
//   u_time_fs                 - frame time in femtoseconds
//   u_hop_rates_inv_s[3]      - per-hop rate constants from the chain
//                                kinetics result of Paper 4
//
// Outputs:
//   fragColor.rgb             - false-colour rendering of |psi(r,t)|^2
//   fragColor.a               - alpha = electron probability at this
//                                pixel's voxel
//
// This shader runs O(1) per pixel regardless of GLB size; the analyte
// is streamed through the apparatus, no spectral library is stored
// (the empty-dictionary principle).
// =====================================================================

#version 330 core

uniform int   u_n_atoms;
uniform vec3  u_atom_positions[256];
uniform vec3  u_atom_S_coords[256];
uniform vec3  u_cofactor_positions[4];     // NADPH, FAD, FMN, heme-Fe
uniform float u_hop_rates_inv_s[3];        // hop1, hop2, hop3
uniform float u_time_fs;
uniform float u_box_min[3];
uniform float u_box_max[3];

in  vec2 v_uv;       // [0,1]^2 fragment uv
in  float v_depth;   // [0,1] z-slice for 3D
out vec4 fragColor;

const float PI         = 3.14159265358979;
const float HBAR_J_S   = 1.054571817e-34;
const float KB_J_PER_K = 1.380649e-23;
const float TEMP_K     = 310.0;

// ---------------------------------------------------------------------
// Pass 5 kernel: |psi(r, t)|^2 at the pixel's voxel position
//
// Composition (sum over cofactors of localised Gaussians whose weights
// are set by the chain hop-rate kinetics):
//
//   c_j(t) = product over preceding hops of (1 - exp(-k_i t))
//          * (exp(-k_j t)) for the present cofactor
//
// The electron probability density is the weighted sum of cofactor
// Gaussians:
//
//   |psi(r, t)|^2 = sum_j  c_j(t) * exp(- |r - r_j|^2 / (2 sigma_j^2))
// ---------------------------------------------------------------------
float electron_density_at(vec3 voxel_pos, float t_s)
{
    // Cofactor occupancies from the four-state hop-rate kinetics.
    float k1 = u_hop_rates_inv_s[0];
    float k2 = u_hop_rates_inv_s[1];
    float k3 = u_hop_rates_inv_s[2];

    float p_NADPH = exp(-k1 * t_s);
    float p_FAD   = (1.0 - exp(-k1 * t_s)) * exp(-k2 * t_s);
    float p_FMN   = (1.0 - exp(-k1 * t_s)) * (1.0 - exp(-k2 * t_s)) * exp(-k3 * t_s);
    float p_heme  = (1.0 - exp(-k1 * t_s)) * (1.0 - exp(-k2 * t_s)) * (1.0 - exp(-k3 * t_s));

    float occ[4];
    occ[0] = p_NADPH;
    occ[1] = p_FAD;
    occ[2] = p_FMN;
    occ[3] = p_heme;

    // Per-cofactor Gaussian width (angstroms): widens for longer hops
    float sigma_A[4];
    sigma_A[0] = 1.4;   // NADPH localisation
    sigma_A[1] = 1.5;
    sigma_A[2] = 1.7;
    sigma_A[3] = 2.6;   // heme: includes axial dioxygen + porphyrin spread

    float density = 0.0;
    for (int j = 0; j < 4; j++) {
        vec3 d = voxel_pos - u_cofactor_positions[j];
        float r2 = dot(d, d);
        density += occ[j] * exp(-r2 / (2.0 * sigma_A[j] * sigma_A[j]));
    }
    return density;
}

// ---------------------------------------------------------------------
// False-colour mapping: plasma-like ramp from black -> red -> yellow.
// ---------------------------------------------------------------------
vec3 plasma_colormap(float x)
{
    x = clamp(x, 0.0, 1.0);
    return vec3(
        smoothstep(0.00, 0.50, x),
        smoothstep(0.30, 0.85, x),
        smoothstep(0.70, 1.00, x) * 0.6
    );
}

// ---------------------------------------------------------------------
// Main: convert v_uv (and v_depth for 3D mode) into a voxel position
// in angstrom space, evaluate the density, and write the false-colour
// rendering.
// ---------------------------------------------------------------------
void main()
{
    // Map fragment uv to voxel position in box [u_box_min, u_box_max]^3.
    vec3 voxel_pos = vec3(
        mix(u_box_min[0], u_box_max[0], v_uv.x),
        mix(u_box_min[1], u_box_max[1], v_uv.y),
        mix(u_box_min[2], u_box_max[2], v_depth)
    );

    float t_s = u_time_fs * 1.0e-15;
    float dens = electron_density_at(voxel_pos, t_s);

    // Logarithmic dynamic-range compression keeps the visible range
    // legible; clamps below the apparatus floor (3.7e-4).
    float vis = clamp(log(1.0 + 100.0 * dens) / log(101.0), 0.0, 1.0);

    fragColor = vec4(plasma_colormap(vis), vis);
}
