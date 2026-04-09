#version 300 es
precision highp float;

// Pass 6: Coupling Spectrum (Synthetic 2D-IR)
// Computes frequency-domain coupling from the coupling matrix.
// Each fragment (i,j) computes the spectral magnitude and phase
// of the coupling between residues i and j.
// This IS the 2D-IR observation -- rendering = measurement.

uniform sampler2D u_coupling;    // N x N coupling matrix from Pass 3
uniform sampler2D u_sentropy;    // 1 x N S-entropy from Pass 1
uniform int u_N;
uniform float u_time;            // animation time for phase evolution

in vec2 v_uv;
layout(location = 0) out vec4 fragColor;

const float TWO_PI = 6.28318530718;

void main() {
    int i = int(v_uv.x * float(u_N));
    int j = int(v_uv.y * float(u_N));
    if (i >= u_N || j >= u_N) { fragColor = vec4(0.0); return; }

    // Fetch coupling strength
    float K_ij = texelFetch(u_coupling, ivec2(i, j), 0).r;

    // Fetch natural frequencies (from Sk coordinate)
    float omega_i = texelFetch(u_sentropy, ivec2(i, 0), 0).r * 10.0 + 1.0;
    float omega_j = texelFetch(u_sentropy, ivec2(j, 0), 0).r * 10.0 + 1.0;

    // Phase evolution: each residue oscillates at its natural frequency
    float phase_i = omega_i * u_time;
    float phase_j = omega_j * u_time;
    float phase_diff = phase_i - phase_j;

    // Spectral magnitude: coupling modulated by phase coherence
    // At resonance (omega_i ~ omega_j), magnitude is maximum
    float detuning = abs(omega_i - omega_j);
    float lorentzian = 1.0 / (1.0 + detuning * detuning / 0.5);
    float magnitude = K_ij * lorentzian;

    // Phase angle: encodes relative orientation
    float phi = atan(sin(phase_diff), cos(phase_diff));

    // Cross-peak asymmetry (imaginary component of coupling)
    float asymmetry = K_ij * sin(phase_diff) * lorentzian;

    // R: spectral magnitude
    // G: phase angle (normalized to [0,1])
    // B: asymmetry (encodes directionality)
    // A: raw coupling for reference
    fragColor = vec4(
        magnitude,
        (phi + TWO_PI) / (TWO_PI * 2.0),
        abs(asymmetry),
        K_ij
    );
}
