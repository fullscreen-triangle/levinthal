// Cofactor positions from CofactorPlacement, anchored to GLB Fe at
// (17.2587, 11.5273, 24.7629). Same values as ElectronTransferViewer.js.
// Plain JS — no Three.js dependency; safe to statically import in pages.

const NADPH = { name: "NADPH", pos: [32.548, 22.250, 36.393], color: "#4C72B0" };
const FAD   = { name: "FAD",   pos: [29.768, 20.300, 34.279], color: "#FFA500" };
const FMN   = { name: "FMN",   pos: [26.988, 18.351, 32.164], color: "#55A868" };
const HEME  = { name: "heme",  pos: [17.259, 11.527, 24.763], color: "#C44E52" };

export const CHAIN_MARKERS = [
  { ...NADPH, radius: 0.85, glow: 0.55 },
  { ...FAD,   radius: 0.85, glow: 0.55 },
  { ...FMN,   radius: 0.85, glow: 0.55 },
  { ...HEME,  radius: 0.85, glow: 0.55 },
];

export const HEME_MARKER = [
  { ...HEME, name: "heme · Fe", radius: 1.3, glow: 1.6 },
];

export const HEME_FMN_MARKERS = [
  { ...HEME, name: "heme · Fe", radius: 1.3, glow: 1.6 },
  { ...FMN,  name: "FMN",       radius: 1.0, glow: 0.9 },
];
