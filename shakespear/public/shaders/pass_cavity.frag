#version 300 es
precision highp float;

// Pass: Harmonic Network and Virtual Cavity Detection
// Detects harmonic edges (omega_i/omega_j ≈ p/q) and identifies
// closed loops (virtual resonant cavities) in the coupling matrix.
// Each fragment (i,j) computes whether residues i and j are harmonically
// coupled, and whether they participate in a cavity.

uniform sampler2D u_sentropy;    // 1 x N S-entropy from Pass 1
uniform sampler2D u_coupling;    // N x N coupling matrix from Pass 3
uniform int u_N;
uniform float u_harmonic_tol;    // tolerance for rational ratio (default 0.05)

in vec2 v_uv;
layout(location = 0) out vec4 fragColor;

// Check if ratio is near a low-order rational p/q with max(p,q) <= 8
float harmonic_strength(float ratio) {
    float best = 1.0;
    for (int p = 1; p <= 8; p++) {
        for (int q = 1; q <= p; q++) {
            float target = float(p) / float(q);
            float dev = abs(ratio - target);
            if (dev < best) best = dev;
            // Also check inverse
            float inv_dev = abs(ratio - float(q) / float(p));
            if (inv_dev < best) best = inv_dev;
        }
    }
    // Return 1.0 for perfect harmonic, 0.0 for non-harmonic
    return smoothstep(u_harmonic_tol, 0.0, best);
}

void main() {
    int i = int(v_uv.x * float(u_N));
    int j = int(v_uv.y * float(u_N));
    if (i >= u_N || j >= u_N || i == j) {
        fragColor = vec4(0.0);
        return;
    }

    // Natural frequencies from Sk
    float omega_i = texelFetch(u_sentropy, ivec2(i, 0), 0).r * 10.0 + 1.0;
    float omega_j = texelFetch(u_sentropy, ivec2(j, 0), 0).r * 10.0 + 1.0;

    // Frequency ratio
    float ratio = max(omega_i, omega_j) / min(omega_i, omega_j);

    // Harmonic edge strength
    float harm = harmonic_strength(ratio);

    // Coupling strength
    float K = texelFetch(u_coupling, ivec2(i, j), 0).r;

    // Harmonic edge: strong coupling AND harmonic ratio
    float edge = harm * smoothstep(0.5, 2.0, K);

    // Cavity detection: check for triangles (3-loops)
    // If (i,j) is an edge AND there exists k such that (i,k) and (k,j) are also edges
    float cavity = 0.0;
    for (int k = 0; k < u_N; k++) {
        if (k == i || k == j) continue;
        float K_ik = texelFetch(u_coupling, ivec2(i, k), 0).r;
        float K_kj = texelFetch(u_coupling, ivec2(k, j), 0).r;

        float omega_k = texelFetch(u_sentropy, ivec2(k, 0), 0).r * 10.0 + 1.0;
        float ratio_ik = max(omega_i, omega_k) / min(omega_i, omega_k);
        float ratio_kj = max(omega_k, omega_j) / min(omega_k, omega_j);

        float harm_ik = harmonic_strength(ratio_ik);
        float harm_kj = harmonic_strength(ratio_kj);

        float edge_ik = harm_ik * smoothstep(0.5, 2.0, K_ik);
        float edge_kj = harm_kj * smoothstep(0.5, 2.0, K_kj);

        // Triangle exists if all three edges are strong
        float triangle = edge * edge_ik * edge_kj;
        cavity = max(cavity, triangle);
    }

    // R: harmonic edge strength
    // G: coupling strength (for reference)
    // B: cavity participation (part of a closed loop)
    // A: frequency ratio deviation from nearest rational
    fragColor = vec4(edge, K, cavity, ratio);
}
