import React, { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { loadEnzymeEfficiency, loadPartitionCoordinates, loadCategoricalTrajectory, loadGrandValidation } from '../D3ChartHelpers'

/**
 * CatalysisChart - Scroll-driven D3 chart for the Catalysis page.
 *
 * Step 0: Enzyme efficiency scatter (log observed vs log predicted)
 * Step 1: Partition coordinates — shell capacity staircase
 * Step 2: Electron transfer trajectory (S-entropy over time)
 * Step 3: Grand validation summary — domain pass rates
 */
export default function CatalysisChart({ activeStep }) {
  const svgRef = useRef(null)
  const [enzymes, setEnzymes] = useState(null)
  const [partition, setPartition] = useState(null)
  const [trajectory, setTrajectory] = useState(null)
  const [validation, setValidation] = useState(null)

  useEffect(() => {
    loadEnzymeEfficiency().then(setEnzymes)
    loadPartitionCoordinates().then(setPartition)
    loadCategoricalTrajectory().then(setTrajectory)
    loadGrandValidation().then(setValidation)
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
    const green = '#4caf50'

    if ((activeStep === 0 || activeStep === undefined) && enzymes) {
      // SCATTER: log observed vs log predicted enzyme efficiency
      const data = enzymes.enzymes

      const xScale = d3.scaleLinear()
        .domain([4, 10]).range([0, w])
      const yScale = d3.scaleLinear()
        .domain([4, 10]).range([h, 0])

      // Perfect prediction line
      g.append('line')
        .attr('x1', xScale(4)).attr('y1', yScale(4))
        .attr('x2', xScale(10)).attr('y2', yScale(10))
        .attr('stroke', '#444').attr('stroke-dasharray', '6,4').attr('stroke-width', 1)

      // Grid
      g.append('g').selectAll('line').data(xScale.ticks(6)).join('line')
        .attr('x1', d => xScale(d)).attr('x2', d => xScale(d))
        .attr('y1', 0).attr('y2', h)
        .attr('stroke', '#1a1a1a')

      // Points
      data.forEach((d, i) => {
        g.append('circle')
          .attr('cx', xScale(d.log_predicted)).attr('cy', yScale(d.log_observed))
          .attr('r', 0).attr('fill', gold).attr('stroke', white).attr('stroke-width', 1)
          .transition().duration(400).delay(i * 60)
          .attr('r', 7)

        // Label
        g.append('text')
          .attr('x', xScale(d.log_predicted) + 10)
          .attr('y', yScale(d.log_observed) + 4)
          .attr('fill', muted).style('font-size', '9px')
          .text(d.name.length > 15 ? d.name.slice(0, 15) + '...' : d.name)
          .attr('opacity', 0).transition().delay(i * 60 + 300).duration(200).attr('opacity', 0.8)
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
        .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '12px')
        .text('log₁₀(k_cat/K_M) Predicted')
      g.append('text').attr('x', -h / 2).attr('y', -50).attr('transform', 'rotate(-90)')
        .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '12px')
        .text('log₁₀(k_cat/K_M) Observed')

      svg.append('text').attr('x', width / 2).attr('y', 24)
        .attr('text-anchor', 'middle').attr('fill', white)
        .style('font-size', '14px').style('font-family', 'Poppins')
        .text('Enzyme Catalytic Efficiency')

    } else if (activeStep === 1 && partition) {
      // STAIRCASE: partition coordinate capacities
      // panel1_partition_coordinates.json is row-oriented: shells is an
      // array of {n, predicted, observed, match, elements}. Guard the shape
      // so a regenerated payload degrades to an empty plot, not a crash that
      // unmounts the whole chart and takes the sibling tabs with it.
      const shells = Array.isArray(partition.shells) ? partition.shells : []
      const data = shells.map(s => ({
        n: s.n, shell: s.elements, capacity: s.predicted
      }))

      const xScale = d3.scaleLinear().domain([0.5, 7.5]).range([0, w])
      const yScale = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.capacity) ?? 100])
        .range([h, 0]).nice()

      // Staircase area
      const stepData = []
      data.forEach(d => {
        stepData.push({ x: d.n - 0.4, y: d.capacity })
        stepData.push({ x: d.n + 0.4, y: d.capacity })
      })

      const area = d3.area()
        .x(d => xScale(d.x)).y0(h).y1(d => yScale(d.y))
        .curve(d3.curveStepAfter)

      g.append('path').datum(stepData).attr('d', area)
        .attr('fill', 'rgba(249, 215, 126, 0.15)')

      // Points and labels
      data.forEach((d, i) => {
        g.append('circle')
          .attr('cx', xScale(d.n)).attr('cy', yScale(d.capacity))
          .attr('r', 0).attr('fill', gold).attr('stroke', white).attr('stroke-width', 1.5)
          .transition().duration(400).delay(i * 100).attr('r', 8)

        g.append('text')
          .attr('x', xScale(d.n)).attr('y', yScale(d.capacity) - 14)
          .attr('text-anchor', 'middle').attr('fill', white).style('font-size', '12px').style('font-weight', '600')
          .text(d.capacity)
          .attr('opacity', 0).transition().delay(i * 100 + 300).duration(200).attr('opacity', 1)

        g.append('text')
          .attr('x', xScale(d.n)).attr('y', yScale(d.capacity) + 24)
          .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '10px')
          .text(d.shell)
      })

      // C(n) = 2n² curve
      const curvePoints = d3.range(1, 7.1, 0.1).map(n => ({ n, c: 2 * n * n }))
      const line = d3.line().x(d => xScale(d.n)).y(d => yScale(d.c)).curve(d3.curveNatural)
      g.append('path').datum(curvePoints).attr('d', line)
        .attr('fill', 'none').attr('stroke', gold).attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4,4').attr('opacity', 0.5)

      // Axes
      g.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(7).tickFormat(d => d % 1 === 0 ? `n=${d}` : ''))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('g').call(d3.axisLeft(yScale).ticks(5))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('text').attr('x', -h / 2).attr('y', -50).attr('transform', 'rotate(-90)')
        .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '12px')
        .text('Capacity C(n)')

      svg.append('text').attr('x', width / 2).attr('y', 24)
        .attr('text-anchor', 'middle').attr('fill', white)
        .style('font-size', '14px').style('font-family', 'Poppins')
        .text('Partition Coordinate Capacity: C(n) = 2n²')

    } else if (activeStep === 2 && trajectory) {
      // ELECTRON TRANSFER: S-entropy coordinates over time
      const xScale = d3.scaleLinear()
        .domain(d3.extent(trajectory, d => d.time_fs))
        .range([0, w])

      const yScale = d3.scaleLinear()
        .domain([0, 1]).range([h, 0])

      const colors = { S_k: gold, S_t: blue, S_e: '#ff7043' }
      const labels = { S_k: 'Sₖ (Kinetic)', S_t: 'Sₜ (Thermal)', S_e: 'Sₑ (Electronic)' }

      Object.entries(colors).forEach(([key, color], ci) => {
        const line = d3.line()
          .x(d => xScale(d.time_fs))
          .y(d => yScale(d[key]))
          .curve(d3.curveMonotoneX)

        const path = g.append('path').datum(trajectory).attr('d', line)
          .attr('fill', 'none').attr('stroke', color).attr('stroke-width', 2)

        const len = path.node().getTotalLength()
        path.attr('stroke-dasharray', len).attr('stroke-dashoffset', len)
          .transition().duration(1500).delay(ci * 300).ease(d3.easeCubicOut)
          .attr('stroke-dashoffset', 0)

        // Legend
        g.append('line').attr('x1', 10).attr('y1', ci * 22 + 5).attr('x2', 30).attr('y2', ci * 22 + 5)
          .attr('stroke', color).attr('stroke-width', 2)
        g.append('text').attr('x', 36).attr('y', ci * 22 + 9)
          .attr('fill', muted).style('font-size', '11px').text(labels[key])
      })

      // Axes
      g.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(8))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('g').call(d3.axisLeft(yScale).ticks(5))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('text').attr('x', w / 2).attr('y', h + 45)
        .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '12px')
        .text('Time (fs)')
      g.append('text').attr('x', -h / 2).attr('y', -50).attr('transform', 'rotate(-90)')
        .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '12px')
        .text('S-Entropy Coordinate')

      svg.append('text').attr('x', width / 2).attr('y', 24)
        .attr('text-anchor', 'middle').attr('fill', white)
        .style('font-size', '14px').style('font-family', 'Poppins')
        .text('Electron Transfer: S-Entropy Evolution')

    } else if (activeStep >= 3 && validation) {
      // GRAND VALIDATION: domain pass rates
      // grand_validation.json keys domains by name:
      //   { "Atomic structure": { passed, total }, ... }
      // An object has no .length, so the previous `domains.length > 0` test
      // was always false and the chart silently drew hardcoded placeholders.
      // They happened to match the file, so stale numbers would have gone
      // unnoticed after any regeneration. Read the real values instead.
      const raw = validation.domains
      const data = Array.isArray(raw)
        ? raw
        : Object.entries(raw || {}).map(([name, v]) => ({
            name, passed: v.passed, total: v.total
          }))

      const xScale = d3.scaleBand().domain(data.map(d => d.name)).range([0, w]).padding(0.3)
      const yScale = d3.scaleLinear().domain([0, 100]).range([h, 0])

      data.forEach((d, i) => {
        const pct = (d.passed / d.total) * 100

        // Background bar (total)
        g.append('rect')
          .attr('x', xScale(d.name)).attr('y', yScale(100))
          .attr('width', xScale.bandwidth()).attr('height', h - yScale(100))
          .attr('fill', '#1a1a1a').attr('rx', 4)

        // Fill bar (passed)
        g.append('rect')
          .attr('x', xScale(d.name)).attr('y', h)
          .attr('width', xScale.bandwidth()).attr('height', 0)
          .attr('fill', pct === 100 ? green : gold).attr('rx', 4)
          .transition().duration(800).delay(i * 120)
          .attr('y', yScale(pct)).attr('height', h - yScale(pct))

        // Label
        g.append('text')
          .attr('x', xScale(d.name) + xScale.bandwidth() / 2)
          .attr('y', yScale(pct) - 10)
          .attr('text-anchor', 'middle').attr('fill', white)
          .style('font-size', '13px').style('font-weight', '600')
          .text(`${d.passed}/${d.total}`)
          .attr('opacity', 0).transition().delay(i * 120 + 600).duration(200).attr('opacity', 1)
      })

      // 94.4% overall label
      g.append('text').attr('x', w / 2).attr('y', h / 2 - 30)
        .attr('text-anchor', 'middle').attr('fill', gold)
        .style('font-size', '36px').style('font-weight', '700').style('font-family', 'Poppins')
        .text('94.4%')
        .attr('opacity', 0).transition().delay(800).duration(400).attr('opacity', 0.15)

      g.append('text').attr('x', w / 2).attr('y', h / 2)
        .attr('text-anchor', 'middle').attr('fill', gold)
        .style('font-size', '14px').style('font-family', 'Poppins')
        .text('34/36 tests passed')
        .attr('opacity', 0).transition().delay(1000).duration(400).attr('opacity', 0.2)

      // Axes
      g.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted).style('font-size', '9px'))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d => `${d}%`))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      svg.append('text').attr('x', width / 2).attr('y', 24)
        .attr('text-anchor', 'middle').attr('fill', white)
        .style('font-size', '14px').style('font-family', 'Poppins')
        .text('Grand Validation: 5 Domains')
    }

  }, [enzymes, partition, trajectory, validation, activeStep])

  return (
    <div className="chart-container">
      <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />
    </div>
  )
}
