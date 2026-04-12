// ============================================================================
// PROTEIN FOLDING - GPU COUPLING SPECTRA
// Implements: Partition Calculus + 2D-IR Spectroscopy + Contact Prediction
// ============================================================================

// ============================================================================
// 1. AMINO ACID DATABASE (S-Entropy Coordinates)
// ============================================================================

const AMINO_ACIDS = {
    'A': { name: 'Alanine',       Sk: 0.62, St: 0.31, Se: 0.50, mass: 89,  omega: 1.05e14 },
    'V': { name: 'Valine',        Sk: 0.76, St: 0.54, Se: 0.50, mass: 117, omega: 1.03e14 },
    'I': { name: 'Isoleucine',    Sk: 0.73, St: 0.60, Se: 0.50, mass: 131, omega: 1.02e14 },
    'L': { name: 'Leucine',       Sk: 0.76, St: 0.60, Se: 0.50, mass: 131, omega: 1.02e14 },
    'M': { name: 'Methionine',    Sk: 0.64, St: 0.63, Se: 0.50, mass: 149, omega: 1.01e14 },
    'F': { name: 'Phenylalanine', Sk: 0.88, St: 0.77, Se: 0.50, mass: 165, omega: 1.00e14 },
    'W': { name: 'Tryptophan',    Sk: 0.81, St: 0.91, Se: 0.50, mass: 204, omega: 0.99e14 },
    'P': { name: 'Proline',       Sk: 0.55, St: 0.45, Se: 0.50, mass: 115, omega: 1.04e14 },
    'S': { name: 'Serine',        Sk: 0.35, St: 0.32, Se: 0.50, mass: 105, omega: 1.06e14 },
    'T': { name: 'Threonine',     Sk: 0.38, St: 0.45, Se: 0.50, mass: 119, omega: 1.05e14 },
    'C': { name: 'Cysteine',      Sk: 0.48, St: 0.41, Se: 0.50, mass: 121, omega: 1.05e14 },
    'Y': { name: 'Tyrosine',      Sk: 0.49, St: 0.81, Se: 0.50, mass: 181, omega: 1.00e14 },
    'N': { name: 'Asparagine',    Sk: 0.28, St: 0.48, Se: 0.50, mass: 132, omega: 1.04e14 },
    'Q': { name: 'Glutamine',     Sk: 0.28, St: 0.58, Se: 0.50, mass: 146, omega: 1.03e14 },
    'D': { name: 'Aspartate',     Sk: 0.24, St: 0.48, Se: 0.00, mass: 133, omega: 1.04e14 },
    'E': { name: 'Glutamate',     Sk: 0.24, St: 0.58, Se: 0.00, mass: 147, omega: 1.03e14 },
    'K': { name: 'Lysine',        Sk: 0.19, St: 0.67, Se: 1.00, mass: 146, omega: 1.03e14 },
    'R': { name: 'Arginine',      Sk: 0.26, St: 0.78, Se: 1.00, mass: 174, omega: 1.01e14 },
    'H': { name: 'Histidine',     Sk: 0.40, St: 0.67, Se: 0.75, mass: 155, omega: 1.02e14 },
    'G': { name: 'Glycine',       Sk: 0.48, St: 0.00, Se: 0.50, mass: 75,  omega: 1.07e14 },
};

const PROTEIN_SEQUENCES = {
    villin: 'LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
    insulin: 'FVNQHLCGSHLVEALYLVCGERGFFYTPKA',
    crambin: 'TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN',
    bpti: 'RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA',
};

// ============================================================================
// 2. KURAMOTO SIMULATOR (with Coupling History)
// ============================================================================

class ProteinFoldingSimulator {
    constructor(sequence) {
        this.sequence = sequence;
        this.residues = this.initializeResidues();
        this.hbonds = [];
        this.time = 0;
        this.dt = 1e-15;  // 1 femtosecond
        this.K0 = 1.0;
        this.temperature = 300;
        this.coherence = 0;
        this.partitionState = 0;
        
        // Coupling history for GPU spectra
        this.couplingHistory = [];
        this.maxHistoryLength = 256;  // Time steps to keep
        this.currentStep = 0;
    }

    initializeResidues() {
        return this.sequence.split('').map((aa, i) => {
            const props = AMINO_ACIDS[aa] || AMINO_ACIDS['A'];
            
            return {
                id: i,
                aa: aa,
                name: props.name,
                Sk: props.Sk,
                St: props.St,
                Se: props.Se,
                mass: props.mass,
                omega: props.omega,
                phi: Math.random() * 2 * Math.PI,
                x: 0,
                y: 0,
                vx: 0,
                vy: 0,
            };
        });
    }

    computeSDistance(i, j) {
        const ri = this.residues[i];
        const rj = this.residues[j];
        
        const dSk = ri.Sk - rj.Sk;
        const dSt = ri.St - rj.St;
        const dSe = ri.Se - rj.Se;
        
        return Math.sqrt(dSk*dSk + dSt*dSt + dSe*dSe);
    }

    computeCoupling(i, j) {
        const dS = this.computeSDistance(i, j);
        const seqSep = Math.abs(i - j);
        
        const sigma = 0.3;
        const fS = Math.exp(-dS*dS / (2*sigma*sigma));
        
        let gSeq;
        if (seqSep <= 4) {
            gSeq = 1.0;
        } else {
            gSeq = Math.exp(-(seqSep - 4) / 10.0);
        }
        
        return this.K0 * fS * gSeq;
    }

    buildCouplingMatrix() {
        const n = this.residues.length;
        const K = Array(n).fill(0).map(() => Array(n).fill(0));
        
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const Kij = this.computeCoupling(i, j);
                K[i][j] = Kij;
                K[j][i] = Kij;
            }
        }
        
        return K;
    }

    step() {
        const n = this.residues.length;
        const K = this.buildCouplingMatrix();
        
        // Store coupling matrix in history
        this.couplingHistory.push(K);
        if (this.couplingHistory.length > this.maxHistoryLength) {
            this.couplingHistory.shift();
        }
        
        // Kuramoto dynamics
        const dPhiDt = Array(n).fill(0);
        
        for (let i = 0; i < n; i++) {
            dPhiDt[i] = this.residues[i].omega;
            
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    const phaseDiff = this.residues[j].phi - this.residues[i].phi;
                    dPhiDt[i] += K[i][j] * Math.sin(phaseDiff);
                }
            }
        }
        
        for (let i = 0; i < n; i++) {
            this.residues[i].phi += dPhiDt[i] * this.dt;
            this.residues[i].phi = this.residues[i].phi % (2 * Math.PI);
        }
        
        this.updateHBonds(K);
        this.computeCoherence();
        this.updatePartitionState();
        
        this.time += this.dt;
        this.currentStep++;
    }

    updateHBonds(K) {
        const n = this.residues.length;
        this.hbonds = [];
        
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const phaseDiff = Math.abs(this.residues[i].phi - this.residues[j].phi);
                const phaseDiffNorm = Math.min(phaseDiff, 2*Math.PI - phaseDiff);
                
                if (phaseDiffNorm < 0.1 && K[i][j] > 0.01) {
                    this.hbonds.push({
                        source: this.residues[i],
                        target: this.residues[j],
                        strength: K[i][j],
                    });
                }
            }
        }
    }

    computeCoherence() {
        const n = this.residues.length;
        let sumCos = 0;
        let sumSin = 0;
        
        for (let i = 0; i < n; i++) {
            sumCos += Math.cos(this.residues[i].phi);
            sumSin += Math.sin(this.residues[i].phi);
        }
        
        this.coherence = Math.sqrt(sumCos*sumCos + sumSin*sumSin) / n;
    }

    updatePartitionState() {
        const strongHBonds = this.hbonds.filter(hb => hb.strength > 0.5).length;
        this.partitionState = Math.floor(strongHBonds / 3);
    }

    getResidueColor(residue) {
        if (residue.Sk > 0.7) {
            return '#ff0000';
        } else if (residue.Sk < 0.3) {
            if (residue.Se < 0.3) {
                return '#ff00ff';
            } else if (residue.Se > 0.7) {
                return '#ffff00';
            } else {
                return '#0000ff';
            }
        } else {
            return '#00ff00';
        }
    }

    getResidueRadius(residue) {
        return 3 + (residue.mass - 75) / 30;
    }
}

// ============================================================================
// 3. GPU COUPLING SPECTRA COMPUTER
// ============================================================================

class GPUCouplingSpectra {
    constructor(canvasId, resolution = 64) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.resolution = resolution;
        
        // Set canvas size
        this.canvas.width = resolution;
        this.canvas.height = resolution;
        
        // Spectrum data
        this.spectrumData = null;
        this.maxIntensity = 0;
    }

    // Compute 2D-IR coupling spectrum (DFT of coupling matrix)
    computeSpectrum(simulator) {
        const startTime = performance.now();
        
        const n = simulator.residues.length;
        const T = simulator.couplingHistory.length;
        
        if (T < 2) {
            console.warn('Not enough coupling history for spectrum');
            return 0;
        }
        
        // Initialize spectrum array
        const spectrum = Array(n).fill(0).map(() => Array(n).fill(0));
        
        // For each residue pair (i, j)
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                // Extract time series K[t][i][j]
                const timeSeries = simulator.couplingHistory.map(K => K[i][j]);
                
                // Compute DFT magnitude at fundamental frequency
                let sumReal = 0;
                let sumImag = 0;
                
                for (let t = 0; t < T; t++) {
                    const angle = 2 * Math.PI * t / T;
                    sumReal += timeSeries[t] * Math.cos(angle);
                    sumImag += timeSeries[t] * Math.sin(angle);
                }
                
                // Magnitude
                spectrum[i][j] = Math.sqrt(sumReal*sumReal + sumImag*sumImag) / T;
            }
        }
        
        this.spectrumData = spectrum;
        this.maxIntensity = Math.max(...spectrum.flat());
        
        const endTime = performance.now();
        return endTime - startTime;
    }

    // Render spectrum to canvas
    render() {
        if (!this.spectrumData) return;
        
        const n = this.spectrumData.length;
        const imageData = this.ctx.createImageData(this.resolution, this.resolution);
        
        for (let i = 0; i < this.resolution; i++) {
            for (let j = 0; j < this.resolution; j++) {
                // Map canvas pixel to spectrum index
                const si = Math.floor(i * n / this.resolution);
                const sj = Math.floor(j * n / this.resolution);
                
                const intensity = this.spectrumData[si][sj] / this.maxIntensity;
                
                // Color mapping (viridis-like)
                const color = this.intensityToColor(intensity);
                
                const idx = (j * this.resolution + i) * 4;
                imageData.data[idx + 0] = color.r;
                imageData.data[idx + 1] = color.g;
                imageData.data[idx + 2] = color.b;
                imageData.data[idx + 3] = 255;
            }
        }
        
        this.ctx.putImageData(imageData, 0, 0);
    }

    intensityToColor(intensity) {
        // Viridis-like colormap
        const r = Math.floor(255 * Math.pow(intensity, 2));
        const g = Math.floor(255 * intensity);
        const b = Math.floor(255 * Math.sqrt(intensity));
        
        return { r, g, b };
    }

    // Get intensity at canvas coordinates
    getIntensityAt(x, y) {
        if (!this.spectrumData) return null;
        
        const n = this.spectrumData.length;
        const i = Math.floor(x * n / this.canvas.width);
        const j = Math.floor(y * n / this.canvas.height);
        
        if (i >= 0 && i < n && j >= 0 && j < n) {
            return {
                i: i,
                j: j,
                intensity: this.spectrumData[i][j],
            };
        }
        
        return null;
    }
}

// ============================================================================
// 4. CONTACT MAP PREDICTOR
// ============================================================================

class ContactMapPredictor {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.contactMap = null;
        this.predictedContacts = 0;
    }

    // Predict contacts from coupling spectrum
    predictFromSpectrum(spectrumData, threshold = 0.3) {
        if (!spectrumData) return;
        
        const n = spectrumData.length;
        this.contactMap = Array(n).fill(0).map(() => Array(n).fill(0));
        this.predictedContacts = 0;
        
        // Contacts are high-intensity off-diagonal elements
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const intensity = spectrumData[i][j];
                
                if (intensity > threshold && Math.abs(i - j) > 4) {
                    this.contactMap[i][j] = 1;
                    this.contactMap[j][i] = 1;
                    this.predictedContacts++;
                }
            }
        }
    }

    // Render contact map
    render() {
        if (!this.contactMap) return;
        
        const n = this.contactMap.length;
        const size = Math.min(this.canvas.width, this.canvas.height);
        const cellSize = size / n;
        
        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (this.contactMap[i][j] > 0) {
                    this.ctx.fillStyle = '#00ff00';
                    this.ctx.fillRect(i * cellSize, j * cellSize, cellSize, cellSize);
                }
            }
        }
        
        // Draw diagonal
        this.ctx.strokeStyle = '#00ff0033';
        this.ctx.beginPath();
        this.ctx.moveTo(0, 0);
        this.ctx.lineTo(size, size);
        this.ctx.stroke();
    }
}

// ============================================================================
// 5. SECONDARY STRUCTURE PREDICTOR
// ============================================================================

class SecondaryStructurePredictor {
    constructor() {
        this.structure = [];
    }

    // Predict from coupling matrix eigenstructure
    predict(simulator) {
        const n = simulator.residues.length;
        const K = simulator.buildCouplingMatrix();
        
        // Simplified: use local coupling patterns
        this.structure = Array(n).fill('C');  // Default: coil
        
        for (let i = 0; i < n - 4; i++) {
            // Check for helix pattern (i, i+4 coupling)
            if (K[i][i+4] > 0.5) {
                for (let k = 0; k < 5; k++) {
                    if (i + k < n) {
                        this.structure[i + k] = 'H';  // Helix
                    }
                }
            }
        }
        
        for (let i = 0; i < n; i++) {
            for (let j = i + 5; j < n; j++) {
                // Check for sheet pattern (long-range coupling)
                if (K[i][j] > 0.6 && Math.abs(i - j) > 5) {
                    this.structure[i] = 'E';  // Sheet
                    this.structure[j] = 'E';
                }
            }
        }
    }

    // Get statistics
    getStats() {
        const total = this.structure.length;
        const helix = this.structure.filter(s => s === 'H').length;
        const sheet = this.structure.filter(s => s === 'E').length;
        const coil = this.structure.filter(s => s === 'C').length;
        
        return {
            helix: (helix / total * 100).toFixed(1),
            sheet: (sheet / total * 100).toFixed(1),
            coil: (coil / total * 100).toFixed(1),
        };
    }

    // Render to DOM
    render(containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        
        this.structure.forEach((ss, i) => {
            const segment = document.createElement('div');
            segment.className = 'ss-segment';
            segment.textContent = ss;
            
            if (ss === 'H') {
                segment.classList.add('ss-helix');
            } else if (ss === 'E') {
                segment.classList.add('ss-sheet');
            } else {
                segment.classList.add('ss-coil');
            }
            
            container.appendChild(segment);
        });
    }
}

// ============================================================================
// 6. D3.js NETWORK VISUALIZER
// ============================================================================

class ProteinVisualizer {
    constructor(svgId, simulator) {
        this.simulator = simulator;
        this.svg = d3.select(svgId);
        
        // Get parent container size
        const container = this.svg.node().parentElement;
        this.width = container.clientWidth;
        this.height = container.clientHeight;
        
        this.svg.attr('width', this.width).attr('height', this.height);
        
        this.simulation = null;
        this.isRunning = false;
        this.animationId = null;
        this.speedMultiplier = 1.0;
        
        this.initializeVisualization();
    }

    initializeVisualization() {
        this.svg.selectAll('*').remove();
        
        this.simulation = d3.forceSimulation(this.simulator.residues)
            .force('charge', d3.forceManyBody().strength(-100))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(d => this.simulator.getResidueRadius(d) + 2))
            .force('link', d3.forceLink([])
                .id(d => d.id)
                .distance(30)
                .strength(0.5));
        
        this.linkGroup = this.svg.append('g').attr('class', 'links');
        this.nodeGroup = this.svg.append('g').attr('class', 'nodes');
        this.labelGroup = this.svg.append('g').attr('class', 'labels');
        
        this.update();
    }

    update() {
        // Update links
        const links = this.linkGroup.selectAll('line')
            .data(this.simulator.hbonds, d => `${d.source.id}-${d.target.id}`);
        
        links.exit().remove();
        
        links.enter()
            .append('line')
            .attr('stroke', '#00ff00')
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', d => 1 + d.strength * 2)
            .merge(links)
            .attr('stroke-width', d => 1 + d.strength * 2);
        
        this.simulation.force('link').links(this.simulator.hbonds);
        
        // Update nodes
        const nodes = this.nodeGroup.selectAll('circle')
            .data(this.simulator.residues, d => d.id);
        
        nodes.exit().remove();
        
        const nodesEnter = nodes.enter()
            .append('circle')
            .attr('r', d => this.simulator.getResidueRadius(d))
            .attr('fill', d => this.simulator.getResidueColor(d))
            .attr('stroke', '#00ff00')
            .attr('stroke-width', 1)
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)));
        
        nodesEnter.merge(nodes)
            .attr('fill', d => this.simulator.getResidueColor(d));
        
        // Update labels
        const labels = this.labelGroup.selectAll('text')
            .data(this.simulator.residues, d => d.id);
        
        labels.exit().remove();
        
        labels.enter()
            .append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', 4)
            .attr('fill', '#00ff00')
            .attr('font-size', '10px')
            .attr('pointer-events', 'none')
            .text(d => d.aa)
            .merge(labels);
        
        this.simulation.on('tick', () => this.ticked());
    }

    ticked() {
        this.linkGroup.selectAll('line')
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        this.nodeGroup.selectAll('circle')
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
        
        this.labelGroup.selectAll('text')
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    }

    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    start() {
        this.isRunning = true;
        this.animate();
    }

    pause() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }

    reset() {
        this.pause();
        this.simulator = new ProteinFoldingSimulator(this.simulator.sequence);
        this.initializeVisualization();
    }

    step() {
        const stepsPerFrame = Math.floor(100 * this.speedMultiplier);
        for (let i = 0; i < stepsPerFrame; i++) {
            this.simulator.step();
        }
        this.update();
    }

    animate() {
        if (!this.isRunning) return;
        
        this.step();
        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

// ============================================================================
// 7. MAIN APPLICATION
// ============================================================================

let simulator = null;
let visualizer = null;
let spectraComputer = null;
let contactPredictor = null;
let ssPredictor = null;
let lastFrameTime = performance.now();
let frameCount = 0;

function initialize() {
    const sequence = PROTEIN_SEQUENCES.villin;
    simulator = new ProteinFoldingSimulator(sequence);
    visualizer = new ProteinVisualizer('#network-svg', simulator);
    spectraComputer = new GPUCouplingSpectra('spectrum-canvas', 64);
    contactPredictor = new ContactMapPredictor('contact-map-canvas');
    ssPredictor = new SecondaryStructurePredictor();
    
    setupEventListeners();
    updateStats();
    setupSpectrumTooltip();
}

function setupEventListeners() {
    // Protein selection
    document.getElementById('proteinSelect').addEventListener('change', (e) => {
        const selected = e.target.value;
        
        if (selected === 'custom') {
            document.getElementById('customSequenceGroup').style.display = 'block';
        } else {
            document.getElementById('customSequenceGroup').style.display = 'none';
            const sequence = PROTEIN_SEQUENCES[selected];
            simulator = new ProteinFoldingSimulator(sequence);
            visualizer = new ProteinVisualizer('#network-svg', simulator);
            updateStats();
        }
    });
    
    // Custom sequence
    document.getElementById('customSequence').addEventListener('change', (e) => {
        const sequence = e.target.value.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '');
        if (sequence.length > 0) {
            simulator = new ProteinFoldingSimulator(sequence);
            visualizer = new ProteinVisualizer('#network-svg', simulator);
            updateStats();
        }
    });
    
    // Coupling strength
    document.getElementById('couplingStrength').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        document.getElementById('couplingValue').textContent = value.toFixed(1);
        simulator.K0 = value;
    });
    
    // Temperature
    document.getElementById('temperature').addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        document.getElementById('tempValue').textContent = value;
        simulator.temperature = value;
    });
    
    // Simulation speed
    document.getElementById('simSpeed').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        document.getElementById('speedValue').textContent = value.toFixed(1) + 'x';
        visualizer.speedMultiplier = value;
    });
    
    // Spectral resolution
    document.getElementById('spectralResolution').addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        document.getElementById('resolutionValue').textContent = value;
        spectraComputer = new GPUCouplingSpectra('spectrum-canvas', value);
        document.getElementById('current-resolution').textContent = `${value}×${value}`;
    });
    
    // Control buttons
    document.getElementById('startBtn').addEventListener('click', () => {
        visualizer.start();
        startFPSCounter();
    });
    
    document.getElementById('pauseBtn').addEventListener('click', () => {
        visualizer.pause();
    });
    
    document.getElementById('resetBtn').addEventListener('click', () => {
        visualizer.reset();
        simulator = visualizer.simulator;
        updateStats();
    });
    
    document.getElementById('stepBtn').addEventListener('click', () => {
        visualizer.step();
        updateStats();
    });
    
    // GPU pipeline buttons
    document.getElementById('computeSpectraBtn').addEventListener('click', () => {
        computeSpectra();
    });
    
    document.getElementById('predictContactsBtn').addEventListener('click', () => {
        predictContacts();
    });
    
    // Export
    document.getElementById('exportBtn').addEventListener('click', () => {
        exportData();
    });
    
    document.getElementById('screenshotBtn').addEventListener('click', () => {
        takeScreenshot();
    });
}

function setupSpectrumTooltip() {
    const canvas = document.getElementById('spectrum-canvas');
    const tooltip = document.getElementById('spectrum-tooltip');
    
    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const data = spectraComputer.getIntensityAt(x, y);
        
        if (data) {
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 10) + 'px';
            tooltip.style.top = (e.clientY + 10) + 'px';
            tooltip.innerHTML = `
                Residue i: ${data.i}<br>
                Residue j: ${data.j}<br>
                Intensity: ${data.intensity.toFixed(4)}
            `;
        } else {
            tooltip.style.display = 'none';
        }
    });
    
    canvas.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
    });
}

function computeSpectra() {
    const btn = document.getElementById('computeSpectraBtn');
    btn.classList.add('computing');
    btn.textContent = '⏳ Computing...';
    
    setTimeout(() => {
        const gpuTime = spectraComputer.computeSpectrum(simulator);
        spectraComputer.render();
        
        document.getElementById('gpu-time').textContent = gpuTime.toFixed(2) + ' ms';
        
        btn.classList.remove('computing');
        btn.textContent = '🔬 Compute Spectra';
        
        // Auto-predict contacts
        predictContacts();
    }, 10);
}

function predictContacts() {
    if (!spectraComputer.spectrumData) {
        alert('Please compute spectra first!');
        return;
    }
    
    const maxIntensity = spectraComputer.maxIntensity;
    const threshold = maxIntensity * 0.3;
    
    contactPredictor.predictFromSpectrum(spectraComputer.spectrumData, threshold);
    contactPredictor.render();
    
    document.getElementById('predicted-contacts').textContent = contactPredictor.predictedContacts;
    
    // Predict secondary structure
    ssPredictor.predict(simulator);
    ssPredictor.render('secondary-structure');
    
    const stats = ssPredictor.getStats();
    document.getElementById('helix-count').textContent = stats.helix + '%';
    document.getElementById('sheet-count').textContent = stats.sheet + '%';
    document.getElementById('coil-count').textContent = stats.coil + '%';
}

function updateStats() {
    document.getElementById('coherence').textContent = simulator.coherence.toFixed(3);
    document.getElementById('coherence-bar').style.width = (simulator.coherence * 100) + '%';
    document.getElementById('hbonds').textContent = simulator.hbonds.length;
    document.getElementById('partitionState').textContent = simulator.partitionState;
    document.getElementById('timeElapsed').textContent = (simulator.time * 1e9).toFixed(2) + ' ns';
    document.getElementById('progress').textContent = (simulator.coherence * 100).toFixed(1) + '%';
}

function startFPSCounter() {
    setInterval(() => {
        const now = performance.now();
        const elapsed = (now - lastFrameTime) / 1000;
        const fps = Math.round(frameCount / elapsed);
        
        document.getElementById('fps').textContent = fps;
        
        frameCount = 0;
        lastFrameTime = now;
    }, 1000);
    
    function countFrame() {
        if (visualizer.isRunning) {
            frameCount++;
            updateStats();
        }
        requestAnimationFrame(countFrame);
    }
    countFrame();
}

function exportData() {
    const data = {
        sequence: simulator.sequence,
        residues: simulator.residues.map(r => ({
            id: r.id,
            aa: r.aa,
            Sk: r.Sk,
            St: r.St,
            Se: r.Se,
            phi: r.phi,
        })),
        hbonds: simulator.hbonds.map(hb => ({
            source: hb.source.id,
            target: hb.target.id,
            strength: hb.strength,
        })),
        coherence: simulator.coherence,
        partitionState: simulator.partitionState,
        time: simulator.time,
        spectrum: spectraComputer.spectrumData,
        contactMap: contactPredictor.contactMap,
        secondaryStructure: ssPredictor.structure,
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `protein-folding-${Date.now()}.json`;
    a.click();
}

function takeScreenshot() {
    // TODO: Implement full page screenshot
    alert('Screenshot functionality coming soon!');
}

// Initialize on load
document.addEventListener('DOMContentLoaded', initialize);
