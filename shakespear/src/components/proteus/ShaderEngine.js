/**
 * PROTEUS Shader Engine
 * ====================
 * Pure WebGL2 multi-pass fragment shader pipeline for protein observation.
 * No Python backend. The GPU IS the instrument.
 *
 * Pipeline:
 *   Pass 1: Sequence → S-Entropy field (1×N)
 *   Pass 3: S-Entropy → Coupling matrix (N×N)
 *   Pass 6: Coupling → Spectrum (N×N, animated)
 *   Pass 7: Spectrum → Coherence / Contact map (N×N)
 *
 * All computation happens in fragment shaders.
 * Rendering = Observation = Computation.
 */

// Amino acid one-letter to index mapping (alphabetical by 1-letter code)
// A=0, C=1, D=2, E=3, F=4, G=5, H=6, I=7, K=8, L=9,
// M=10, N=11, P=12, Q=13, R=14, S=15, T=16, W=17, Y=18, V=19
const AA_INDEX = {
  A: 0, C: 1, D: 2, E: 3, F: 4, G: 5, H: 6, I: 7, K: 8, L: 9,
  M: 10, N: 11, P: 12, Q: 13, R: 14, S: 15, T: 16, W: 17, Y: 18, V: 19,
};

// S-entropy coordinates for each amino acid (matching shader constants)
const AA_SENTROPY = [
  [0.700, 0.170, 0.000], // A
  [0.778, 0.289, 0.100], // C
  [0.111, 0.304, 1.000], // D
  [0.111, 0.467, 1.000], // E
  [0.811, 0.774, 0.000], // F
  [0.456, 0.000, 0.000], // G
  [0.144, 0.555, 0.600], // H
  [1.000, 0.636, 0.000], // I
  [0.067, 0.647, 1.000], // K
  [0.922, 0.636, 0.000], // L
  [0.711, 0.613, 0.000], // M
  [0.111, 0.322, 0.500], // N
  [0.322, 0.373, 0.000], // P
  [0.111, 0.499, 0.500], // Q
  [0.000, 0.676, 1.000], // R
  [0.411, 0.172, 0.300], // S
  [0.422, 0.334, 0.300], // T
  [0.400, 1.000, 0.100], // W
  [0.356, 0.796, 0.200], // Y
  [0.967, 0.476, 0.000], // V
];

export class ShaderEngine {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', {
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
    });
    if (!this.gl) throw new Error('WebGL2 not supported on this browser/device');

    const gl = this.gl;
    // Check float texture support -- critical for the pipeline
    this.hasFloat = !!gl.getExtension('EXT_color_buffer_float');
    if (!this.hasFloat) {
      // Try half-float fallback
      this.hasFloat = !!gl.getExtension('EXT_color_buffer_half_float');
    }
    // Also need float linear filtering
    gl.getExtension('OES_texture_float_linear');

    this.floatInternalFormat = this.hasFloat ? gl.RGBA32F : gl.RGBA16F;
    this.floatType = this.hasFloat ? gl.FLOAT : gl.HALF_FLOAT;

    this.programs = {};
    this.textures = {};
    this.framebuffers = {};
    this.quadVAO = null;
    this.sequence = '';
    this.N = 0;
    this.time = 0;
    this.error = null;
    this.initialized = false;

    this._initQuad();
  }

  _initQuad() {
    const gl = this.gl;
    const vertices = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.quadVAO = vao;
  }

  async init() {
    const vertSrc = await this._fetchShader('/shaders/quad.vert');
    const pass1Src = await this._fetchShader('/shaders/pass1_sentropy.frag');
    const pass3Src = await this._fetchShader('/shaders/pass3_coupling.frag');
    const pass6Src = await this._fetchShader('/shaders/pass6_spectrum.frag');
    const pass7Src = await this._fetchShader('/shaders/pass7_coherence.frag');
    const displaySrc = await this._fetchShader('/shaders/display.frag');
    const cavitySrc = await this._fetchShader('/shaders/pass_cavity.frag');

    this.programs.pass1 = this._createProgram(vertSrc, pass1Src);
    this.programs.pass3 = this._createProgram(vertSrc, pass3Src);
    this.programs.pass6 = this._createProgram(vertSrc, pass6Src);
    this.programs.pass7 = this._createProgram(vertSrc, pass7Src);
    this.programs.display = this._createProgram(vertSrc, displaySrc);
    this.programs.cavity = this._createProgram(vertSrc, cavitySrc);
    this.initialized = true;
    this.status = {
      hasFloat: this.hasFloat,
      programs: Object.keys(this.programs).length,
      renderer: this.gl.getParameter(this.gl.RENDERER),
      vendor: this.gl.getParameter(this.gl.VENDOR),
      version: this.gl.getParameter(this.gl.VERSION),
    };
    console.log('ShaderEngine initialized:', this.status);
  }

  async _fetchShader(path) {
    const res = await fetch(path);
    return await res.text();
  }

  _createShader(type, src) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error('Shader compile error: ' + log);
    }
    return shader;
  }

  _createProgram(vertSrc, fragSrc) {
    const gl = this.gl;
    const vs = this._createShader(gl.VERTEX_SHADER, vertSrc);
    const fs = this._createShader(gl.FRAGMENT_SHADER, fragSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.bindAttribLocation(prog, 0, 'a_position');
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('Program link error: ' + gl.getProgramInfoLog(prog));
    }
    return prog;
  }

  _createTexture(width, height, data) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (this.hasFloat) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0,
                    gl.RGBA, gl.FLOAT, data);
    } else {
      // Fallback: use RGBA8 (loses precision but works everywhere)
      const u8 = data ? new Uint8Array(data.length) : null;
      if (data) for (let i = 0; i < data.length; i++) u8[i] = Math.round(Math.max(0, Math.min(1, data[i])) * 255);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0,
                    gl.RGBA, gl.UNSIGNED_BYTE, u8);
    }
    return tex;
  }

  _createFBO(width, height) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (this.hasFloat) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0,
                    gl.RGBA, gl.FLOAT, null);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0,
                    gl.RGBA, gl.UNSIGNED_BYTE, null);
    }

    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                            gl.TEXTURE_2D, tex, 0);

    // Check framebuffer status
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.error(`FBO incomplete: ${status} (${width}x${height})`);
      this.error = `Framebuffer incomplete: ${status}`;
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { fbo, tex, width, height };
  }

  // Names of uniforms that are integers in GLSL (all others are float)
  static INT_UNIFORMS = new Set(['u_N', 'u_mode']);

  _renderPass(program, uniforms, output) {
    const gl = this.gl;
    gl.useProgram(program);

    let texUnit = 0;
    for (const [name, val] of Object.entries(uniforms)) {
      const loc = gl.getUniformLocation(program, name);
      if (loc === null) continue;
      if (val instanceof WebGLTexture) {
        gl.activeTexture(gl.TEXTURE0 + texUnit);
        gl.bindTexture(gl.TEXTURE_2D, val);
        gl.uniform1i(loc, texUnit);
        texUnit++;
      } else if (typeof val === 'number') {
        if (ShaderEngine.INT_UNIFORMS.has(name)) {
          gl.uniform1i(loc, val);
        } else {
          gl.uniform1f(loc, val);
        }
      }
    }

    if (output) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, output.fbo);
      gl.viewport(0, 0, output.width, output.height);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }

  setSequence(seq) {
    const gl = this.gl;
    this.sequence = seq.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '');
    this.N = this.sequence.length;
    if (this.N === 0) return;

    // Encode sequence as texture (1 x N, each texel = aa index / 19)
    const data = new Float32Array(this.N * 4);
    for (let i = 0; i < this.N; i++) {
      const idx = AA_INDEX[this.sequence[i]] || 0;
      data[i * 4] = idx / 19.0;
      data[i * 4 + 1] = 0;
      data[i * 4 + 2] = 0;
      data[i * 4 + 3] = 1;
    }
    this.textures.sequence = this._createTexture(this.N, 1, data);

    // Create FBOs for each pass
    this.framebuffers.sentropy = this._createFBO(this.N, 1);
    this.framebuffers.coupling = this._createFBO(this.N, this.N);
    this.framebuffers.spectrum = this._createFBO(this.N, this.N);
    this.framebuffers.coherence = this._createFBO(this.N, this.N);
    this.framebuffers.cavity = this._createFBO(this.N, this.N);

    this.time = 0;
  }

  /** Run the full pipeline and return observation textures. */
  observe(dt) {
    if (this.N === 0) return null;
    this.time += dt || 0.016;
    const debugOnce = !this._debugDone;
    if (debugOnce) this._debugDone = true;

    // Pass 1: Sequence → S-Entropy
    this._renderPass(this.programs.pass1, {
      u_sequence: this.textures.sequence,
      u_N: this.N,
    }, this.framebuffers.sentropy);

    if (debugOnce) {
      const s = this._readFBO(this.framebuffers.sentropy.fbo, this.N, 1);
      console.log('Pass1 sentropy [0]:', s[0], s[1], s[2], '(expect ~0.42, ~0.33, ~0.30 for T)');
    }

    // Pass 3: S-Entropy → Coupling Matrix
    this._renderPass(this.programs.pass3, {
      u_sentropy: this.framebuffers.sentropy.tex,
      u_N: this.N,
      u_K_scale: 5.0,
    }, this.framebuffers.coupling);

    if (debugOnce) {
      const c = this._readFBO(this.framebuffers.coupling.fbo, this.N, this.N);
      const mid = Math.floor(this.N / 2) * this.N * 4 + Math.floor(this.N / 2 + 1) * 4;
      console.log('Pass3 coupling [N/2, N/2+1]:', c[mid], c[mid+1], c[mid+2]);
    }

    // Pass 6: Coupling → Spectrum (animated)
    this._renderPass(this.programs.pass6, {
      u_coupling: this.framebuffers.coupling.tex,
      u_sentropy: this.framebuffers.sentropy.tex,
      u_N: this.N,
      u_time: this.time,
    }, this.framebuffers.spectrum);

    if (debugOnce) {
      const sp = this._readFBO(this.framebuffers.spectrum.fbo, this.N, this.N);
      const mid = Math.floor(this.N / 2) * this.N * 4 + Math.floor(this.N / 2 + 1) * 4;
      console.log('Pass6 spectrum [N/2, N/2+1]:', sp[mid], sp[mid+1], sp[mid+2]);
    }

    // Pass 7: Spectrum → Coherence / Contacts
    this._renderPass(this.programs.pass7, {
      u_spectrum: this.framebuffers.spectrum.tex,
      u_sentropy: this.framebuffers.sentropy.tex,
      u_N: this.N,
      u_coherence_threshold: 0.3,
    }, this.framebuffers.coherence);

    if (debugOnce) {
      const h = this._readFBO(this.framebuffers.coherence.fbo, this.N, this.N);
      const mid = Math.floor(this.N / 2) * this.N * 4 + Math.floor(this.N / 2 + 1) * 4;
      console.log('Pass7 coherence [N/2, N/2+1]:', h[mid], h[mid+1], h[mid+2]);
      console.log('GL error after pipeline:', this.gl.getError());
    }

    // Pass Cavity: Harmonic network + cavity detection
    this._renderPass(this.programs.cavity, {
      u_sentropy: this.framebuffers.sentropy.tex,
      u_coupling: this.framebuffers.coupling.tex,
      u_N: this.N,
      u_harmonic_tol: 0.05,
    }, this.framebuffers.cavity);

    return {
      sentropy: this.framebuffers.sentropy.tex,
      coupling: this.framebuffers.coupling.tex,
      spectrum: this.framebuffers.spectrum.tex,
      coherence: this.framebuffers.coherence.tex,
      cavity: this.framebuffers.cavity.tex,
    };
  }

  /** Read pixels from an FBO, handling both float and uint8 formats. */
  _readFBO(fbo, width, height) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    let pixels;
    if (this.hasFloat) {
      pixels = new Float32Array(width * height * 4);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, pixels);
    } else {
      const u8 = new Uint8Array(width * height * 4);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, u8);
      pixels = new Float32Array(u8.length);
      for (let i = 0; i < u8.length; i++) pixels[i] = u8[i] / 255;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return pixels;
  }

  /** Read back coherence data from GPU for scalar metrics. */
  readCoherence() {
    if (this.N === 0) return { eta: 0, contacts: 0 };
    const N = this.N;
    const pixels = this._readFBO(this.framebuffers.coherence.fbo, N, N);

    // Compute order parameter: mean of spectral magnitudes
    let sumReal = 0, sumImag = 0, count = 0, contacts = 0;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const idx = (i * N + j) * 4;
        const mag = pixels[idx + 3]; // raw magnitude
        const phase = pixels[idx + 1] * 2 * Math.PI * 2 - Math.PI;
        sumReal += mag * Math.cos(phase);
        sumImag += mag * Math.sin(phase);
        count++;
        if (pixels[idx] > 0.5 && Math.abs(i - j) > 4) contacts++;
      }
    }
    const eta = Math.sqrt(sumReal * sumReal + sumImag * sumImag) / (count || 1);
    return { eta: Math.min(eta * 50, 1.0), contacts: Math.floor(contacts / 2) };
  }

  /** Read back S-entropy for CPU-side analysis. */
  readSEntropy() {
    if (this.N === 0) return [];
    const pixels = this._readFBO(this.framebuffers.sentropy.fbo, this.N, 1);
    const result = [];
    for (let i = 0; i < this.N; i++) {
      result.push({
        residue: this.sequence[i],
        Sk: pixels[i * 4],
        St: pixels[i * 4 + 1],
        Se: pixels[i * 4 + 2],
      });
    }
    return result;
  }

  /** Render observation texture to canvas using GPU display shader. No CPU readback. */
  renderToCanvas(viewName) {
    const modeMap = { spectrum: 0, coupling: 1, contacts: 2, sentropy: 3, cavity: 4 };
    const fbKey = viewName === 'contacts' ? 'coherence' : viewName;
    const fb = this.framebuffers[fbKey];
    if (!fb) return;

    this._renderPass(this.programs.display, {
      u_observation: fb.tex,
      u_mode: modeMap[viewName] || 0,
    }, null); // null = render to screen
  }

  destroy() {
    const gl = this.gl;
    for (const p of Object.values(this.programs)) gl.deleteProgram(p);
    for (const t of Object.values(this.textures)) gl.deleteTexture(t);
    for (const f of Object.values(this.framebuffers)) {
      gl.deleteFramebuffer(f.fbo);
      gl.deleteTexture(f.tex);
    }
  }
}

export { AA_INDEX, AA_SENTROPY };
