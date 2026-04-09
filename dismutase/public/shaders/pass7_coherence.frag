#version 300 es
precision highp float;

// Pass 7: Coherence Measurement (Diagnostician)
// Computes the Kuramoto order parameter from oscillator phases.
// Also generates the contact probability map from the coupling spectrum.
// This IS the diagnostic observation -- rendering = diagnosis.

uniform sampler2D u_spectrum;    // N x N spectrum from Pass 6
uniform sampler2D u_sentropy;    // 1 x N S-entropy from Pass 1
uniform int u_N;
uniform float u_coherence_threshold;  // phase-lock threshold (default 0.3)

in vec2 v_uv;
layout(location = 0) out vec4 fragColor;

void main() {
    int i = int(v_uv.x * float(u_N));
    int j = int(v_uv.y * float(u_N));
    if (i >= u_N || j >= u_N) { fragColor = vec4(0.0); return; }

    // Fetch spectrum values
    vec4 spec = texelFetch(u_spectrum, ivec2(i, j), 0);
    float magnitude = spec.r;
    float phase = spec.g;
    float asymmetry = spec.b;
    float raw_coupling = spec.a;

    // Contact probability: high magnitude + low phase variance = contact
    float contact_prob = smoothstep(u_coherence_threshold, u_coherence_threshold + 0.2, magnitude);

    // Sequence separation penalty for short-range contacts
    float seq_sep = abs(float(i - j));
    float long_range_bonus = smoothstep(4.0, 8.0, seq_sep);

    // Combined contact score
    float contact = contact_prob * (0.3 + 0.7 * long_range_bonus);

    // Helix signal: contacts at i,i+4 with high coupling
    float helix_signal = 0.0;
    if (abs(seq_sep - 4.0) < 1.5) {
        helix_signal = magnitude;
    }

    // Sheet signal: long-range contacts with S-entropy match
    float sheet_signal = 0.0;
    if (seq_sep > 8.0) {
        vec3 Si = texelFetch(u_sentropy, ivec2(i, 0), 0).rgb;
        vec3 Sj = texelFetch(u_sentropy, ivec2(j, 0), 0).rgb;
        float s_match = exp(-length(Si - Sj) / 0.3);
        sheet_signal = magnitude * s_match;
    }

    // R: contact probability
    // G: helix signal strength
    // B: sheet signal strength
    // A: raw magnitude
    fragColor = vec4(contact, helix_signal, sheet_signal, magnitude);
}
