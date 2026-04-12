// ============================================================================
// PROTEIN FOLDING VISUALIZER - CATEGORICAL FRAMEWORK
// Based on: Partition Calculus + Kuramoto Dynamics
// ============================================================================

// ============================================================================
// 1. AMINO ACID DATABASE (S-Entropy Coordinates)
// ============================================================================

const AMINO_ACIDS = {
    // Hydrophobic
    'A': { name: 'Alanine',     Sk: 0.62, St: 0.31, Se: 0.50, mass: 89,  omega: 1.05e14 },
    'V': { name: 'Valine',      Sk: 0.76, St: 0.54, Se: 0.50, mass: 117, omega: 1.03e14 },
    'I': { name: 'Isoleucine',  Sk: 0.73, St: 0.60, Se: 0.50, mass: 131, omega: 1.02e14 },
    'L': { name: 'Leucine',     Sk: 0.76, St: 0.60, Se: 0.50, mass: 131, omega: 1.02e14 },
    'M': { name: 'Methionine',  Sk: 0.64, St: 0.63, Se: 0.50, mass: 149, omega: 1.01e14 },
    'F': { name: 'Phenylalanine', Sk: 0.88, St: 0.77, Se: 0.50, mass: 165, omega: 1.00e14 },
    'W': { name: 'Tryptophan',  Sk: 0.81, St: 0.91, Se: 0.50, mass: 204, omega: 0.99e14 },
    'P': { name: 'Proline',     Sk: 0.55, St: 0.45, Se: 0.50, mass: 115, omega: 1.04e14 },
    
    // Hydrophilic
    'S': { name: 'Serine',      Sk: 0.35, St: 0.32, Se: 0.50, mass: 105, omega: 1.06e14 },
    'T': { name: 'Threonine',   Sk: 0.38, St: 0.45, Se: 0.50, mass: 119, omega: 1.05e14 },
    'C': { name: 'Cysteine',    Sk: 0.48, St: 0.41, Se: 0.50, mass: 121, omega: 1.05e14 },
    'Y': { name: 'Tyrosine',    Sk: 0.49, St: 0.81, Se: 0.50, mass: 181, omega: 1.00e14 },
    'N': { name: 'Asparagine',  Sk: 0.28, St: 0.48, Se: 0.50, mass: 132, omega: 1.04e14 },
    'Q': { name: 'Glutamine',   Sk: 0.28, St: 0.58, Se: 0.50, mass: 146, omega: 1.03e14 },
    
    // Charged
    'D': { name: 'Aspartate',   Sk: 0.24, St: 0.48, Se: 0.00, mass: 133, omega: 1.04e14 },
    'E': { name: 'Glutamate',   Sk: 0.24, St: 0.58, Se: 0.00, mass: 147, omega: 1.03e14 },
    'K': { name: 'Lysine',      Sk: 0.19, St: 0.67, Se: 1.00, mass: 146, omega: 1.03e14 },
    'R': { name: 'Arginine',    Sk: 0.26, St: 0.78, Se: 1.00, mass: 174, omega: 1.01e14 },
    'H': { name: 'Histidine',   Sk: 0.40, St: 0.67, Se: 0.75, mass: 155, omega: 1.02e14 },
    
    // Special
    'G': { name: 'Glycine',     Sk: 0.48, St: 0.00, Se: 0.50, mass: 75,  omega: 1.07e14 },
};

// ============================================================================
// 2. PREDEFINED PROTEIN SEQUENCES
// ============================================================================

const PROTEIN_SEQUENCES = {
    villin: 'LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',  // 35 residues
    insulin: 'FVNQHLCGSHLVEALYLVCGERGFFYTPKA',      // 30 residues
    crambin: 'TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN', // 46 residues
};

// ============================================================================
// 3. PROTEIN FOLDING SIMULATOR (Kuramoto Dynamics)
// ============================================================================

class ProteinFoldingSimulator {
    constructor(sequence) {
        this.sequence = sequence;
        this.residues = this.initializeResidues();
        this.hbonds = [];
        this.time = 0;
        this.dt = 1e-15;  // 1 femtosecond timestep
        this.K0 = 1.0;    // Base coupling strength
        this.temperature = 300;  // Kelvin
        this.coherence = 0;
        this.partitionState = 0;
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
                phi: Math.random() * 2 * Math.PI,  // Initial phase (random)
                x: 0,  // Will be set by D3
                y: 0,
                vx: 0,
                vy: 0,
            };
        });
    }

    // Compute S-entropy distance between residues
    computeSDistance(i, j) {
        const ri = this.residues[i];
        const rj = this.residues[j];
        
        const dSk = ri.Sk - rj.Sk;
        const dSt = ri.St - rj.St;
        const dSe = ri.Se - rj.Se;
        
        return Math.sqrt(dSk*dSk + dSt*dSt + dSe*dSe);
    }

    // Compute coupling strength K_ij
    computeCoupling(i, j) {
        const dS = this.computeSDistance(i, j);
        const seqSep = Math.abs(i - j);
        
        // Gaussian coupling kernel
        const sigma = 0.3;
        const fS = Math.exp(-dS*dS / (2*sigma*sigma));
        
        // Sequence separation factor
        let gSeq;
        if (seqSep <= 4) {
            gSeq = 1.0;  // Local contacts
        } else {
            gSeq = Math.exp(-(seqSep - 4) / 10.0);  // Long-range decay
        }
        
        return this.K0 * fS * gSeq;
    }

    // Build coupling matrix
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

    // Kuramoto dynamics step
    step() {
        const n = this.residues.length;
        const K = this.buildCouplingMatrix();
        
        // Update phases (Kuramoto equation)
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
        
        // Integrate
        for (let i = 0; i < n; i++) {
            this.residues[i].phi += dPhiDt[i] * this.dt;
            this.residues[i].phi = this.residues[i].phi % (2 * Math.PI);
        }
        
        // Update H-bonds (phase-locked pairs)
        this.updateHBonds(K);
        
        // Compute phase coherence
        this.computeCoherence();
        
        // Update partition state
        this.updatePartitionState();
        
        this.time += this.dt;
    }

    // Detect H-bonds (phase-locked pairs)
    updateHBonds(K) {
        const n = this.residues.length;
        this.hbonds = [];
        
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const phaseDiff = Math.abs(this.residues[i].phi - this.residues[j].phi);
                const phaseDiffNorm = Math.min(phaseDiff, 2*Math.PI - phaseDiff);
                
                // Phase-lock criterion: |Δφ| < 0.1 rad
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

    // Compute Kuramoto order parameter (phase coherence)
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

    // Update partition state (ℓ from coupling matrix eigenstructure)
    updatePartitionState() {
        // Simplified: count number of strong H-bonds
        const strongHBonds = this.hbonds.filter(hb => hb.strength > 0.5).length;
        this.partitionState = Math.floor(strongHBonds / 3);
    }

    // Get residue color based on properties
    getResidueColor(residue) {
        if (residue.Sk > 0.7) {
            return '#ff0000';  // Hydrophobic (red)
        } else if (residue.Sk < 0.3) {
            if (residue.Se < 0.3) {
                return '#ff00ff';  // Negative charge (magenta)
            } else if (residue.Se > 0.7) {
                return '#ffff00';  // Positive charge (yellow)
            } else {
                return '#0000ff';  // Hydrophilic (blue)
            }
        } else {
            return '#00ff00';  // Neutral (green)
        }
    }

    // Get residue radius based on mass
    getResidueRadius(residue) {
        return 3 + (residue.mass - 75) / 30;  // 3-8 pixels
    }
}

// ============================================================================
// 4. D3.js VISUALIZATION
// ============================================================================

class ProteinVisualizer {
    constructor(containerId, simulator) {
        this.simulator = simulator;
        this.svg = d3.select(containerId);
        this.width = +this.svg.attr('width');
        this.height = +this.svg.attr('height');
        
        this.simulation = null;
        this.isRunning = false;
        this.animationId = null;
        
        this.initializeVisualization();
    }

    initializeVisualization() {
        // Clear SVG
        this.svg.selectAll('*').remove();
        
        // Create force simulation
        this.simulation = d3.forceSimulation(this.simulator.residues)
            .force('charge', d3.forceManyBody().strength(-100))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(d => this.simulator.getResidueRadius(d) + 2))
            .force('link', d3.forceLink([])
                .id(d => d.id)
                .distance(30)
                .strength(0.5));
        
        // Create link group
        this.linkGroup = this.svg.append('g').attr('class', 'links');
        
        // Create node group
        this.nodeGroup = this.svg.append('g').attr('class', 'nodes');
        
        // Create label group
        this.labelGroup = this.svg.append('g').attr('class', 'labels');
        
        this.update();
    }

    update() {
        // Update links (H-bonds)
        const links = this.linkGroup.selectAll('line')
            .data(this.simulator.hbonds, d => `${d.source.id}-${d.target.id}`);
        
        links.exit().remove();
        
        links.enter()
            .append('line')
            .attr('class', 'hbond-link')
            .attr('stroke', '#00ff00')
            .attr('stroke-width', d => 1 + d.strength * 2)
            .merge(links)
            .attr('stroke-width', d => 1 + d.strength * 2);
        
        // Update force simulation links
        this.simulation.force('link').links(this.simulator.hbonds);
        
        // Update nodes (residues)
        const nodes = this.nodeGroup.selectAll('circle')
            .data(this.simulator.residues, d => d.id);
        
        nodes.exit().remove();
        
        const nodesEnter = nodes.enter()
            .append('circle')
            .attr('class', 'residue-node')
            .attr('r', d => this.simulator.getResidueRadius(d))
            .attr('fill', d => this.simulator.getResidueColor(d))
            .attr('stroke', '#00ff00')
            .attr('stroke-width', 1)
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)))
            .on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mouseout', () => this.hideTooltip());
        
        nodesEnter.merge(nodes)
            .attr('fill', d => this.simulator.getResidueColor(d));
        
        // Update labels
        const labels = this.labelGroup.selectAll('text')
            .data(this.simulator.residues, d => d.id);
        
        labels.exit().remove();
        
        labels.enter()
            .append('text')
            .attr('class', 'residue-label')
            .attr('text-anchor', 'middle')
            .attr('dy', 4)
            .text(d => d.aa)
            .merge(labels);
        
        // Update positions on tick
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

    showTooltip(event, d) {
        // TODO: Implement tooltip
        console.log(`${d.name} (${d.aa}): Sk=${d.Sk.toFixed(2)}, St=${d.St.toFixed(2)}, Se=${d.Se.toFixed(2)}`);
    }

    hideTooltip() {
        // TODO: Implement tooltip hiding
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
        this.updateStats();
    }

    animate() {
        if (!this.isRunning) return;
        
        // Run multiple simulation steps per frame for speed
        const stepsPerFrame = 100;
        for (let i = 0; i < stepsPerFrame; i++) {
            this.simulator.step();
        }
        
        this.update();
        this.updateStats();
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }

    updateStats() {
        document.getElementById('coherence').textContent = this.simulator.coherence.toFixed(3);
        document.getElementById('hbonds').textContent = this.simulator.hbonds.length;
        document.getElementById('progress').textContent = (this.simulator.coherence * 100).toFixed(1) + '%';
        document.getElementById('partitionState').textContent = this.simulator.partitionState;
        document.getElementById('timeElapsed').textContent = (this.simulator.time * 1e9).toFixed(2) + ' ns';
    }
}

// ============================================================================
// 5. UI CONTROLS
// ============================================================================

let visualizer = null;

document.addEventListener('DOMContentLoaded', () => {
    // Initialize with default protein
    const sequence = PROTEIN_SEQUENCES.villin;
    const simulator = new ProteinFoldingSimulator(sequence);
    visualizer = new ProteinVisualizer('#visualization', simulator);
    
    // Protein selection
    document.getElementById('proteinSelect').addEventListener('change', (e) => {
        const selected = e.target.value;
        
        if (selected === 'custom') {
            document.getElementById('customSequenceGroup').style.display = 'block';
        } else {
            document.getElementById('customSequenceGroup').style.display = 'none';
            const sequence = PROTEIN_SEQUENCES[selected];
            const simulator = new ProteinFoldingSimulator(sequence);
            visualizer = new ProteinVisualizer('#visualization', simulator);
        }
    });
    
    // Custom sequence
    document.getElementById('customSequence').addEventListener('change', (e) => {
        const sequence = e.target.value.toUpperCase();
        const simulator = new ProteinFoldingSimulator(sequence);
        visualizer = new ProteinVisualizer('#visualization', simulator);
    });
    
    // Coupling strength
    document.getElementById('couplingStrength').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        document.getElementById('couplingValue').textContent = value.toFixed(1);
        visualizer.simulator.K0 = value;
    });
    
    // Temperature
    document.getElementById('temperature').addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        document.getElementById('tempValue').textContent = value;
        visualizer.simulator.temperature = value;
    });
    
    // Control buttons
    document.getElementById('startBtn').addEventListener('click', () => {
        visualizer.start();
    });
    
    document.getElementById('pauseBtn').addEventListener('click', () => {
        visualizer.pause();
    });
    
    document.getElementById('resetBtn').addEventListener('click', () => {
        visualizer.reset();
    });
    
    // Export
    document.getElementById('exportBtn').addEventListener('click', () => {
        const data = {
            sequence: visualizer.simulator.sequence,
            residues: visualizer.simulator.residues,
            hbonds: visualizer.simulator.hbonds,
            coherence: visualizer.simulator.coherence,
            time: visualizer.simulator.time,
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'protein-structure.json';
        a.click();
    });
    
    // Screenshot
    document.getElementById('screenshotBtn').addEventListener('click', () => {
        const svgData = new XMLSerializer().serializeToString(document.getElementById('visualization'));
        const canvas = document.createElement('canvas');
        canvas.width = 800;
        canvas.height = 600;
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0);
            canvas.toBlob(blob => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'protein-folding.png';
                a.click();
            });
        };
        img.src = 'data:image/svg+xml;base64,' + btoa(svgData);
    });
});
