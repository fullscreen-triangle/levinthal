#version 300 es
precision highp float;

// Pass 1: S-Entropy Field
// Computes (Sk, St, Se) for each residue from amino acid properties.
// Input: u_sequence texture (1 x N, each texel = amino acid index 0-19)
// Output: RGBA texture (1 x N) with (Sk, St, Se, 1.0) per residue

uniform sampler2D u_sequence;   // amino acid indices
uniform int u_N;                // number of residues

in vec2 v_uv;
layout(location = 0) out vec4 fragColor;

// Kyte-Doolittle hydrophobicity, van der Waals volume, electrostatic index
// for 20 amino acids: A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V
// Indexed 0-19 in alphabetical 1-letter order: A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,W,Y,V
const float h[20] = float[20](
     1.8,  2.5, -3.5, -3.5,  2.8, -0.4, -3.2,  4.5, -3.9,  3.8,
     1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, -0.9, -1.3,  4.2
);
const float v[20] = float[20](
    88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7,
   162.9, 114.1, 122.7, 143.8, 173.4,  89.0, 116.1, 227.8, 193.6, 140.0
);
const float e[20] = float[20](
    0.00, 0.10, 1.00, 1.00, 0.00, 0.00, 0.60, 0.00, 1.00, 0.00,
    0.00, 0.50, 0.00, 0.50, 1.00, 0.30, 0.30, 0.10, 0.20, 0.00
);

const float H_MIN = -4.5;
const float H_MAX = 4.5;
const float V_MIN = 60.1;
const float V_MAX = 227.8;

void main() {
    int idx = int(v_uv.x * float(u_N));
    if (idx >= u_N) { fragColor = vec4(0.0); return; }

    // Read amino acid index from sequence texture
    int aa = int(texelFetch(u_sequence, ivec2(idx, 0), 0).r * 19.0 + 0.5);
    aa = clamp(aa, 0, 19);

    // Compute S-entropy coordinates
    float Sk = (h[aa] - H_MIN) / (H_MAX - H_MIN);
    float St = (v[aa] - V_MIN) / (V_MAX - V_MIN);
    float Se = e[aa];

    fragColor = vec4(Sk, St, Se, 1.0);
}
