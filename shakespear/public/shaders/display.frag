#version 300 es
precision highp float;

// Display shader: maps float observation textures to visible colors.
// Replaces CPU readback + putImageData with a single GPU pass.

uniform sampler2D u_observation;
uniform int u_mode; // 0=spectrum, 1=coupling, 2=contacts, 3=sentropy, 4=cavity

in vec2 v_uv;
layout(location = 0) out vec4 fragColor;

// Inferno-like colormap
vec3 inferno(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.0, 0.0, 0.015);
    vec3 c1 = vec3(0.5, 0.0, 0.5);
    vec3 c2 = vec3(0.9, 0.3, 0.0);
    vec3 c3 = vec3(1.0, 0.95, 0.7);
    if (t < 0.33) return mix(c0, c1, t / 0.33);
    if (t < 0.66) return mix(c1, c2, (t - 0.33) / 0.33);
    return mix(c2, c3, (t - 0.66) / 0.34);
}

// Viridis-like colormap
vec3 viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.267, 0.004, 0.329);
    vec3 c1 = vec3(0.128, 0.567, 0.551);
    vec3 c2 = vec3(0.993, 0.906, 0.144);
    if (t < 0.5) return mix(c0, c1, t * 2.0);
    return mix(c1, c2, (t - 0.5) * 2.0);
}

void main() {
    vec4 obs = texture(u_observation, v_uv);

    vec3 color;

    if (u_mode == 0) {
        // Spectrum: magnitude → inferno
        color = inferno(obs.r);
    }
    else if (u_mode == 1) {
        // Coupling: K_ij → inferno (scaled)
        color = inferno(obs.r / 5.0);
    }
    else if (u_mode == 2) {
        // Contacts: R=contact, G=helix, B=sheet
        float contact = obs.r;
        float helix = obs.g;
        float sheet = obs.b;
        // Helix = red, Sheet = blue, Contact = cyan
        color = vec3(
            min(1.0, helix * 3.0),
            min(1.0, sheet * 2.0 + contact * 0.3),
            min(1.0, contact * 1.5 + sheet * 2.0)
        );
    }
    else if (u_mode == 3) {
        // S-Entropy: (Sk, St, Se) direct as (R, G, B)
        color = obs.rgb;
    }
    else if (u_mode == 4) {
        // Cavity: highlight detected cavities
        float magnitude = obs.r;
        float phase = obs.g * 6.283185 * 2.0 - 3.141593;
        float cavity_signal = obs.b;
        // Cavity regions glow cyan, non-cavity dim
        vec3 base = inferno(magnitude);
        vec3 cavity_glow = vec3(0.345, 0.902, 0.851); // primaryDark
        color = mix(base, cavity_glow, cavity_signal * 0.7);
    }

    fragColor = vec4(color, 1.0);
}
