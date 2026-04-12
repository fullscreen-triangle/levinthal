#version 300 es
precision highp float;

// Pass 3: Coupling Matrix (Harmonic Network)
// Computes K_ij between all residue pairs from S-entropy distance.
// Input: u_sentropy texture (1 x N) from Pass 1
// Output: N x N coupling matrix texture

uniform sampler2D u_sentropy;   // (Sk, St, Se) per residue
uniform int u_N;                // number of residues
uniform float u_K_scale;        // coupling strength scale

in vec2 v_uv;
layout(location = 0) out vec4 fragColor;

void main() {
    int i = int(v_uv.x * float(u_N));
    int j = int(v_uv.y * float(u_N));
    if (i >= u_N || j >= u_N) { fragColor = vec4(0.0); return; }
    if (i == j) { fragColor = vec4(0.0); return; }

    // Fetch S-entropy coordinates
    vec3 Si = texelFetch(u_sentropy, ivec2(i, 0), 0).rgb;
    vec3 Sj = texelFetch(u_sentropy, ivec2(j, 0), 0).rgb;

    // S-entropy distance
    float s_dist = length(Si - Sj);

    // Sequence separation
    float seq_sep = abs(float(i - j));

    // Backbone coupling: exponential decay with sequence distance
    float backbone = exp(-seq_sep / 2.0);

    // Helix coupling: strong at i,i+4 (hydrogen bonding pattern)
    float helix = 0.8 * exp(-(seq_sep - 4.0) * (seq_sep - 4.0) / 1.0)
                + 0.4 * exp(-(seq_sep - 3.0) * (seq_sep - 3.0) / 1.0);

    // Tertiary coupling: S-entropy similarity at long range
    float tertiary = exp(-s_dist / 0.2) * (1.0 - exp(-seq_sep / 8.0));

    // Cys-Cys bonus (Se ~ 0.1 for Cys)
    float cys_bonus = 0.0;
    if (abs(Si.z - 0.1) < 0.05 && abs(Sj.z - 0.1) < 0.05 &&
        abs(Si.x - 0.778) < 0.05 && abs(Sj.x - 0.778) < 0.05) {
        cys_bonus = 2.0 * exp(-s_dist / 0.5);
    }

    float K = u_K_scale * (0.3 * backbone + 0.35 * helix
            + 0.25 * tertiary + 0.1 * cys_bonus);

    // R: coupling strength
    // G: S-entropy distance (for visualization)
    // B: sequence separation normalized
    fragColor = vec4(K, s_dist, seq_sep / float(u_N), 1.0);
}
