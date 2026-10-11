// Colours for the talk. Categorical slots are the dark-surface steps of the
// validated reference palette; text never wears a series colour.
export const C = {
  bg: "#121211",
  surface: "#1a1a19",
  panel: "#20201e",
  line: "#34332f",
  grid: "#2a2a27",
  ink: "#f2f1ec",
  ink2: "#c3c2b7",
  muted: "#8a8984",
  blue: "#3987e5",
  orange: "#d95926",
  aqua: "#199e70",
  yellow: "#c98500",
  magenta: "#d55181",
  violet: "#9085e9",
  red: "#e66767",
};

// Ledger labels used by both manuscripts: derived / supplied / checked.
export const LEDGER = {
  D: { label: "derived", color: C.blue },
  S: { label: "supplied", color: C.muted },
  C: { label: "checked", color: C.aqua },
  M: { label: "model value", color: C.yellow },
  L: { label: "literature", color: C.violet },
};

export const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "—");
export const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

// The stage is drawn at this logical size and scaled to the window.
export const STAGE = { w: 1600, h: 900 };
