import * as d3 from 'd3'

/**
 * Fetch JSON data from public/data/
 */
export async function fetchChartData(filename) {
  const res = await fetch(`/data/${filename}`)
  return res.json()
}

/**
 * Fetch CSV data from public/data/ and parse with d3
 */
export async function fetchCSVData(filename) {
  const res = await fetch(`/data/${filename}`)
  const text = await res.text()
  return d3.csvParse(text, d3.autoType)
}

/**
 * Load dynamics data
 */
export async function loadDockingTrajectory() {
  return fetchChartData('dynamics/docking_trajectory.json')
}

export async function loadDockingResults() {
  return fetchChartData('dynamics/docking_results.json')
}

export async function loadHelixMotion() {
  return fetchChartData('dynamics/helix_motion_results.json')
}

export async function loadAllExperiments() {
  return fetchChartData('dynamics/all_experiments.json')
}

/**
 * Load folding data
 */
export async function loadSEntropyTrajectories() {
  return fetchChartData('folding/chart1_sentropy_3d.json')
}

export async function loadPartitionCapacity() {
  return fetchChartData('folding/chart2_partition_capacity.json')
}

export async function loadOscillatorFrequencies() {
  return fetchChartData('folding/chart3_oscillator_frequencies.json')
}

export async function loadCoherenceEquation() {
  return fetchChartData('folding/chart4_coherence_equation.json')
}

export async function loadPhaselockSurface() {
  return fetchChartData('folding/chart7_phaselock_3d.json')
}

export async function loadFoldingDiagnostics() {
  return fetchChartData('folding/chart6_folding_diagnostics.json')
}

/**
 * Load catalysis/partition data
 */
export async function loadEnzymeEfficiency() {
  return fetchChartData('partition/panel5_enzyme_efficiency.json')
}

export async function loadPartitionCoordinates() {
  return fetchChartData('partition/panel1_partition_coordinates.json')
}

export async function loadPartitionDepth() {
  return fetchChartData('partition/panel2_partition_depth.json')
}

export async function loadActivationEnergy() {
  return fetchChartData('partition/panel3_activation_energy.json')
}

export async function loadDiseaseData() {
  return fetchChartData('partition/panel7_disease.json')
}

export async function loadMetabolism() {
  return fetchChartData('partition/panel6_metabolism.json')
}

/**
 * Load electron transfer data
 */
export async function loadCategoricalTrajectory() {
  return fetchCSVData('catalysis/categorical_trajectory.csv')
}

export async function loadSEntropyCoordinates() {
  return fetchCSVData('catalysis/s_entropy_coordinates.csv')
}

export async function loadBackactionMetrics() {
  return fetchCSVData('catalysis/backaction_metrics.csv')
}

/**
 * Load validation
 */
export async function loadGrandValidation() {
  return fetchChartData('validation/grand_validation.json')
}
