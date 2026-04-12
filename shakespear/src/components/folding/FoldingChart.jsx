import React, { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { loadPartitionCapacity, loadCoherenceEquation, loadFoldingDiagnostics, loadSEntropyTrajectories } from '../D3ChartHelpers'

/**
 * FoldingChart - Scroll-driven D3 chart for the Folding page.
 *
 * Step 0: Partition capacity bar chart (predicted = observed, C(n) = 2n²)
 * Step 1: S-entropy trajectories (ATP synthesis + Protein folding in Sk/St space)
 * Step 2: Coherence equation — enzyme eta values as horizontal bars
 * Step 3: Folding diagnostics — cellular states (healthy → critical)
 */
export default function FoldingChart({ activeStep }) {
  const svgRef = useRef(null)
  const [capacity, setCapacity] = useState(null)
  const [coherence, setCoherence] = useState(null)
  const [diagnostics, setDiagnostics] = useState(null)
  const [sentropy, setSentropy] = useState(null)

  useEffect(() => {
    loadPartitionCapacity().then(setCapacity)
    loadCoherenceEquation().then(setCoherence)
    loadFoldingDiagnostics().then(setDiagnostics)
    loadSEntropyTrajectories().then(setSentropy)
  }, [])

  useEffect(() => {
    if (!svgRef.current) return

    const svg = d3.select(svgRef.current)
    const width = 560
    const height = 480
    const margin = { top: 40, right: 40, bottom: 60, left: 70 }
    const w = width - margin.left - margin.right
    const h = height - margin.top - margin.bottom

    svg.selectAll('*').remove()
    svg.attr('viewBox', `0 0 ${width} ${height}`)

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const gold = '#f9d77e'
    const blue = '#82b1ff'
    const white = '#fff'
    const muted = '#808080'

    if ((activeStep === 0 || activeStep === undefined) && capacity) {
      // BAR CHART: Partition capacity C(n) = 2n²
      const data = capacity.data.n.map((n, i) => ({
        shell: capacity.data.shell_names[i],
        n,
        predicted: capacity.data.predicted_capacity[i],
        observed: capacity.data.observed_capacity[i]
      }))

      const xScale = d3.scaleBand()
        .domain(data.map(d => d.shell))
        .range([0, w]).padding(0.3)

      const yScale = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.predicted)])
        .range([h, 0]).nice()

      const subBands = d3.scaleBand()
        .domain(['predicted', 'observed'])
        .range([0, xScale.bandwidth()]).padding(0.1)

      // Bars
      data.forEach(d => {
        g.append('rect')
          .attr('x', xScale(d.shell) + subBands('predicted'))
          .attr('y', h).attr('width', subBands.bandwidth()).attr('height', 0)
          .attr('fill', gold).attr('rx', 2)
          .transition().duration(800)
          .attr('y', yScale(d.predicted)).attr('height', h - yScale(d.predicted))

        g.append('rect')
          .attr('x', xScale(d.shell) + subBands('observed'))
          .attr('y', h).attr('width', subBands.bandwidth()).attr('height', 0)
          .attr('fill', blue).attr('rx', 2).attr('opacity', 0.7)
          .transition().duration(800).delay(100)
          .attr('y', yScale(d.observed)).attr('height', h - yScale(d.observed))
      })

      // Legend
      g.append('rect').attr('x', w - 130).attr('y', 0).attr('width', 12).attr('height', 12).attr('fill', gold).attr('rx', 2)
      g.append('text').attr('x', w - 112).attr('y', 10).attr('fill', muted).style('font-size', '11px').text('Predicted')
      g.append('rect').attr('x', w - 130).attr('y', 20).attr('width', 12).attr('height', 12).attr('fill', blue).attr('rx', 2)
      g.append('text').attr('x', w - 112).attr('y', 30).attr('fill', muted).style('font-size', '11px').text('Observed')

      // Axes
      g.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('text').attr('x', w / 2).attr('y', h + 45)
        .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '12px')
        .text('Electron Shell')

      g.append('text').attr('x', -h / 2).attr('y', -50).attr('transform', 'rotate(-90)')
        .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '12px')
        .text('Electron Capacity')

      svg.append('text').attr('x', width / 2).attr('y', 24)
        .attr('text-anchor', 'middle').attr('fill', white)
        .style('font-size', '14px').style('font-family', 'Poppins')
        .text('Partition Capacity: C(n) = 2n²')

    } else if (activeStep === 1 && sentropy) {
      // S-ENTROPY TRAJECTORIES: 2D projection of Sk vs St
      const trajectories = sentropy.trajectories
      const colors = { ATP_Synthesis: gold, Protein_Folding: blue }

      // Collect all points for scaling
      let allSk = [], allSt = []
      Object.values(trajectories).forEach(traj => {
        allSk = allSk.concat(traj.Sk)
        allSt = allSt.concat(traj.St)
      })

      const xScale = d3.scaleLinear().domain(d3.extent(allSk)).range([0, w]).nice()
      const yScale = d3.scaleLinear().domain(d3.extent(allSt)).range([h, 0]).nice()

      // Grid
      g.append('g').selectAll('line').data(yScale.ticks(5)).join('line')
        .attr('x1', 0).attr('x2', w).attr('y1', d => yScale(d)).attr('y2', d => yScale(d))
        .attr('stroke', '#222').attr('stroke-dasharray', '2,4')

      // Draw each trajectory
      Object.entries(trajectories).forEach(([name, traj], ti) => {
        const points = traj.Sk.map((sk, i) => ({ sk, st: traj.St[i] }))
        const color = Object.values(colors)[ti] || gold

        // Line
        const line = d3.line().x(d => xScale(d.sk)).y(d => yScale(d.st)).curve(d3.curveCardinal)
        const path = g.append('path').datum(points).attr('d', line)
          .attr('fill', 'none').attr('stroke', color).attr('stroke-width', 2)

        const len = path.node().getTotalLength()
        path.attr('stroke-dasharray', len).attr('stroke-dashoffset', len)
          .transition().duration(2000).delay(ti * 500).ease(d3.easeCubicOut)
          .attr('stroke-dashoffset', 0)

        // Start/end markers
        g.append('circle').attr('cx', xScale(points[0].sk)).attr('cy', yScale(points[0].st))
          .attr('r', 4).attr('fill', color).attr('opacity', 0)
          .transition().delay(ti * 500).duration(300).attr('opacity', 1)

        g.append('circle').attr('cx', xScale(points[points.length - 1].sk)).attr('cy', yScale(points[points.length - 1].st))
          .attr('r', 4).attr('fill', color).attr('stroke', white).attr('stroke-width', 1.5)
          .attr('opacity', 0).transition().delay(ti * 500 + 1800).duration(300).attr('opacity', 1)
      })

      // Legend
      Object.entries(colors).forEach(([name, color], i) => {
        g.append('line').attr('x1', 10).attr('y1', i * 22 + 5).attr('x2', 30).attr('y2', i * 22 + 5)
          .attr('stroke', color).attr('stroke-width', 2)
        g.append('text').attr('x', 36).attr('y', i * 22 + 9).attr('fill', muted).style('font-size', '11px')
          .text(name.replace('_', ' '))
      })

      // Axes
      g.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(6))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('text').attr('x', w / 2).attr('y', h + 45)
        .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '12px').text('Sₖ (Kinetic Entropy)')
      g.append('text').attr('x', -h / 2).attr('y', -50).attr('transform', 'rotate(-90)')
        .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '12px').text('Sₜ (Thermal Entropy)')

      svg.append('text').attr('x', width / 2).attr('y', 24)
        .attr('text-anchor', 'middle').attr('fill', white)
        .style('font-size', '14px').style('font-family', 'Poppins')
        .text('S-Entropy Space Trajectories')

    } else if (activeStep === 2 && coherence) {
      // COHERENCE EQUATION: enzyme eta values
      const enzymes = coherence.oscillator_validations.Enzyme_Catalysis.enzyme_data
      const data = Object.entries(enzymes).map(([name, d]) => ({ name, eta: d.eta, kcat: d.kcat }))
        .sort((a, b) => b.eta - a.eta)

      const yScale = d3.scaleBand().domain(data.map(d => d.name)).range([0, h]).padding(0.3)
      const xScale = d3.scaleLinear().domain([0, 1]).range([0, w])

      // Bars
      data.forEach((d, i) => {
        g.append('rect')
          .attr('x', 0).attr('y', yScale(d.name))
          .attr('width', 0).attr('height', yScale.bandwidth())
          .attr('fill', d3.interpolateYlOrRd(1 - d.eta))
          .attr('rx', 3)
          .transition().duration(800).delay(i * 80)
          .attr('width', xScale(Math.max(0, d.eta)))

        // Value label
        g.append('text')
          .attr('x', xScale(Math.max(0, d.eta)) + 8).attr('y', yScale(d.name) + yScale.bandwidth() / 2 + 4)
          .attr('fill', muted).style('font-size', '11px').style('font-family', 'Courier New')
          .text(`η = ${d.eta.toFixed(2)}`)
          .attr('opacity', 0).transition().delay(i * 80 + 600).duration(200).attr('opacity', 1)
      })

      // Axes
      g.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(5).tickFormat(d => d.toFixed(1)))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('g').call(d3.axisLeft(yScale))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted).style('font-size', '10px'))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('text').attr('x', w / 2).attr('y', h + 45)
        .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '12px')
        .text('Coherence η')

      svg.append('text').attr('x', width / 2).attr('y', 24)
        .attr('text-anchor', 'middle').attr('fill', white)
        .style('font-size', '14px').style('font-family', 'Poppins')
        .text('Universal Coherence Equation')

    } else if (activeStep >= 3 && diagnostics) {
      // FOLDING DIAGNOSTICS: cellular states
      const states = diagnostics.cellular_states
      const data = Object.entries(states).map(([name, d]) => ({
        name, eta_mean: d.eta_mean, eta_std: d.eta_std, FEI: d.FEI
      }))

      const stateColors = { Healthy: '#4caf50', Stressed: gold, Diseased: '#ff7043', Critical: '#f44336' }

      const xScale = d3.scaleBand().domain(data.map(d => d.name)).range([0, w]).padding(0.35)
      const yScale = d3.scaleLinear().domain([-0.6, 1.1]).range([h, 0]).nice()

      // Zero line
      g.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', yScale(0)).attr('y2', yScale(0))
        .attr('stroke', '#444').attr('stroke-dasharray', '4,4')

      // Bars with error bars
      data.forEach((d, i) => {
        const color = stateColors[d.name] || muted
        const barY = d.eta_mean >= 0 ? yScale(d.eta_mean) : yScale(0)
        const barH = Math.abs(yScale(0) - yScale(d.eta_mean))

        g.append('rect')
          .attr('x', xScale(d.name)).attr('y', yScale(0))
          .attr('width', xScale.bandwidth()).attr('height', 0)
          .attr('fill', color).attr('rx', 4).attr('opacity', 0.85)
          .transition().duration(800).delay(i * 150)
          .attr('y', barY).attr('height', barH)

        // Error bar
        const cx = xScale(d.name) + xScale.bandwidth() / 2
        g.append('line')
          .attr('x1', cx).attr('x2', cx)
          .attr('y1', yScale(d.eta_mean - d.eta_std))
          .attr('y2', yScale(d.eta_mean + d.eta_std))
          .attr('stroke', white).attr('stroke-width', 1.5)
          .attr('opacity', 0).transition().delay(i * 150 + 600).duration(200).attr('opacity', 0.7)

        // Caps
        ;[d.eta_mean - d.eta_std, d.eta_mean + d.eta_std].forEach(val => {
          g.append('line')
            .attr('x1', cx - 6).attr('x2', cx + 6)
            .attr('y1', yScale(val)).attr('y2', yScale(val))
            .attr('stroke', white).attr('stroke-width', 1.5)
            .attr('opacity', 0).transition().delay(i * 150 + 600).duration(200).attr('opacity', 0.7)
        })

        // Value
        g.append('text')
          .attr('x', cx).attr('y', barY - 8)
          .attr('text-anchor', 'middle').attr('fill', color).style('font-size', '12px').style('font-weight', '600')
          .text(d.eta_mean.toFixed(2))
          .attr('opacity', 0).transition().delay(i * 150 + 400).duration(200).attr('opacity', 1)
      })

      // Axes
      g.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted).style('font-size', '12px'))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('g').call(d3.axisLeft(yScale).ticks(6))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('text').attr('x', -h / 2).attr('y', -50).attr('transform', 'rotate(-90)')
        .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '12px')
        .text('Coherence η')

      svg.append('text').attr('x', width / 2).attr('y', 24)
        .attr('text-anchor', 'middle').attr('fill', white)
        .style('font-size', '14px').style('font-family', 'Poppins')
        .text('Folding as Cellular Health Diagnostic')
    }

  }, [capacity, coherence, diagnostics, sentropy, activeStep])

  return (
    <div className="chart-container">
      <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />
    </div>
  )
}
