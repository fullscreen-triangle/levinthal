import React, { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { loadDockingTrajectory, loadAllExperiments } from '../D3ChartHelpers'

/**
 * DynamicsChart - Scroll-driven D3 chart for the Dynamics page.
 *
 * Step 0: Docking trajectory line chart (ligand distance vs step)
 * Step 1: Distribution bar chart (ground/natural/excited at current step)
 * Step 2: Ternary encoding visualization
 * Step 3: Phase-lock convergence (all steps overlaid)
 */
export default function DynamicsChart({ activeStep }) {
  const svgRef = useRef(null)
  const [trajectory, setTrajectory] = useState(null)
  const [experiments, setExperiments] = useState(null)

  useEffect(() => {
    loadDockingTrajectory().then(setTrajectory)
    loadAllExperiments().then(setExperiments)
  }, [])

  useEffect(() => {
    if (!svgRef.current || !trajectory) return

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

    // Color palette
    const gold = '#f9d77e'
    const dimGold = 'rgba(249, 215, 126, 0.3)'
    const white = '#fff'
    const muted = '#808080'
    const categories = ['#666', '#82b1ff', gold]
    const catLabels = ['Ground', 'Natural', 'Excited']

    if (activeStep === 0 || activeStep === undefined) {
      // LINE CHART: ligand distance over docking steps
      const xScale = d3.scaleLinear()
        .domain([0, trajectory.length - 1])
        .range([0, w])

      const yScale = d3.scaleLinear()
        .domain([0, d3.max(trajectory, d => d.ligand_distance)])
        .range([h, 0])
        .nice()

      // Grid lines
      g.append('g').selectAll('line')
        .data(yScale.ticks(5))
        .join('line')
        .attr('x1', 0).attr('x2', w)
        .attr('y1', d => yScale(d)).attr('y2', d => yScale(d))
        .attr('stroke', '#222').attr('stroke-dasharray', '2,4')

      // Area fill
      const area = d3.area()
        .x(d => xScale(d.step))
        .y0(h)
        .y1(d => yScale(d.ligand_distance))
        .curve(d3.curveMonotoneX)

      g.append('path')
        .datum(trajectory)
        .attr('d', area)
        .attr('fill', dimGold)

      // Line
      const line = d3.line()
        .x(d => xScale(d.step))
        .y(d => yScale(d.ligand_distance))
        .curve(d3.curveMonotoneX)

      const path = g.append('path')
        .datum(trajectory)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', gold)
        .attr('stroke-width', 2.5)

      // Animate line drawing
      const totalLength = path.node().getTotalLength()
      path.attr('stroke-dasharray', totalLength)
        .attr('stroke-dashoffset', totalLength)
        .transition().duration(1500).ease(d3.easeCubicOut)
        .attr('stroke-dashoffset', 0)

      // End point
      const last = trajectory[trajectory.length - 1]
      g.append('circle')
        .attr('cx', xScale(last.step))
        .attr('cy', yScale(last.ligand_distance))
        .attr('r', 0)
        .attr('fill', gold)
        .transition().delay(1400).duration(300)
        .attr('r', 5)

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(10))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted).style('font-size', '11px'))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('g')
        .call(d3.axisLeft(yScale).ticks(5))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted).style('font-size', '11px'))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      // Labels
      g.append('text').attr('x', w / 2).attr('y', h + 45)
        .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '12px')
        .text('Docking Step')

      g.append('text').attr('x', -h / 2).attr('y', -50)
        .attr('transform', 'rotate(-90)')
        .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '12px')
        .text('Ligand Distance (Å)')

      // Title
      svg.append('text').attr('x', width / 2).attr('y', 24)
        .attr('text-anchor', 'middle').attr('fill', white)
        .style('font-size', '14px').style('font-family', 'Poppins')
        .text('Protein–Ligand Docking Trajectory')

    } else if (activeStep === 1) {
      // BAR CHART: Distribution at multiple steps
      const steps = [0, 25, 50, 75, 99].filter(s => s < trajectory.length)
      const data = steps.map(s => ({
        step: trajectory[s].step,
        ground: trajectory[s].distribution[0],
        natural: trajectory[s].distribution[1],
        excited: trajectory[s].distribution[2],
      }))

      const xScale = d3.scaleBand()
        .domain(data.map(d => `Step ${d.step}`))
        .range([0, w])
        .padding(0.3)

      const yMax = d3.max(data, d => Math.max(d.ground, d.natural, d.excited))
      const yScale = d3.scaleLinear()
        .domain([0, yMax]).range([h, 0]).nice()

      const subBands = d3.scaleBand()
        .domain([0, 1, 2])
        .range([0, xScale.bandwidth()])
        .padding(0.05)

      // Bars
      data.forEach(d => {
        [d.ground, d.natural, d.excited].forEach((val, ci) => {
          g.append('rect')
            .attr('x', xScale(`Step ${d.step}`) + subBands(ci))
            .attr('y', h)
            .attr('width', subBands.bandwidth())
            .attr('height', 0)
            .attr('fill', categories[ci])
            .attr('rx', 2)
            .transition().duration(800).delay(ci * 150)
            .attr('y', yScale(val))
            .attr('height', h - yScale(val))
        })
      })

      // Legend
      catLabels.forEach((label, i) => {
        const lx = w - 120
        g.append('rect').attr('x', lx).attr('y', i * 20)
          .attr('width', 12).attr('height', 12).attr('fill', categories[i]).attr('rx', 2)
        g.append('text').attr('x', lx + 18).attr('y', i * 20 + 10)
          .attr('fill', muted).style('font-size', '11px').text(label)
      })

      // Axes
      g.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted).style('font-size', '11px'))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('g').call(d3.axisLeft(yScale).ticks(5))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted).style('font-size', '11px'))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('text').attr('x', -h / 2).attr('y', -50)
        .attr('transform', 'rotate(-90)')
        .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '12px')
        .text('Atom Count')

      svg.append('text').attr('x', width / 2).attr('y', 24)
        .attr('text-anchor', 'middle').attr('fill', white)
        .style('font-size', '14px').style('font-family', 'Poppins')
        .text('Ternary State Distribution During Docking')

    } else if (activeStep === 2) {
      // TERNARY ENCODING: Show the ternary string as a heatmap strip
      const trits = trajectory.map(d => d.trit)
      const cellSize = Math.min(w / trits.length, 30)

      const tritColors = { 0: '#333', 1: '#82b1ff', 2: gold }
      const tritNames = { 0: 'Ground (0)', 1: 'Natural (1)', 2: 'Excited (2)' }

      // Heatmap
      const heatG = g.append('g')
        .attr('transform', `translate(0, ${h / 2 - cellSize * 3})`)

      trits.forEach((t, i) => {
        heatG.append('rect')
          .attr('x', i * (w / trits.length))
          .attr('y', 0)
          .attr('width', w / trits.length - 1)
          .attr('height', cellSize * 2)
          .attr('fill', tritColors[t])
          .attr('rx', 1)
          .attr('opacity', 0)
          .transition().duration(30).delay(i * 15)
          .attr('opacity', 1)
      })

      // Ternary string text
      const str = trits.join('')
      g.append('text')
        .attr('x', w / 2).attr('y', h / 2 + cellSize * 2 + 20)
        .attr('text-anchor', 'middle').attr('fill', gold)
        .style('font-size', '11px').style('font-family', 'Courier New')
        .text(str.length > 60 ? str.slice(0, 60) + '...' : str)

      // Label
      g.append('text')
        .attr('x', w / 2).attr('y', h / 2 - cellSize * 3 - 15)
        .attr('text-anchor', 'middle').attr('fill', white)
        .style('font-size', '13px')
        .text('Ternary Trajectory Encoding')

      // Legend
      Object.entries(tritNames).forEach(([key, label], i) => {
        const lx = w / 2 - 150 + i * 120
        const ly = h / 2 + cellSize * 2 + 50
        g.append('rect').attr('x', lx).attr('y', ly)
          .attr('width', 12).attr('height', 12).attr('fill', tritColors[key]).attr('rx', 2)
        g.append('text').attr('x', lx + 18).attr('y', ly + 10)
          .attr('fill', muted).style('font-size', '11px').text(label)
      })

      svg.append('text').attr('x', width / 2).attr('y', 24)
        .attr('text-anchor', 'middle').attr('fill', white)
        .style('font-size', '14px').style('font-family', 'Poppins')
        .text('Base-3 Trajectory Representation')

    } else if (activeStep >= 3) {
      // COMBINED: Distance + distribution overlay showing convergence
      const xScale = d3.scaleLinear()
        .domain([0, trajectory.length - 1]).range([0, w])

      const yDist = d3.scaleLinear()
        .domain([0, d3.max(trajectory, d => d.ligand_distance)])
        .range([h, 0]).nice()

      const yExcited = d3.scaleLinear()
        .domain([0, d3.max(trajectory, d => d.distribution[2])])
        .range([h, 0]).nice()

      // Distance line
      const distLine = d3.line()
        .x(d => xScale(d.step))
        .y(d => yDist(d.ligand_distance))
        .curve(d3.curveMonotoneX)

      g.append('path').datum(trajectory)
        .attr('d', distLine).attr('fill', 'none')
        .attr('stroke', gold).attr('stroke-width', 2)

      // Excited state line
      const excitedLine = d3.line()
        .x(d => xScale(d.step))
        .y(d => yExcited(d.distribution[2]))
        .curve(d3.curveMonotoneX)

      g.append('path').datum(trajectory)
        .attr('d', excitedLine).attr('fill', 'none')
        .attr('stroke', '#82b1ff').attr('stroke-width', 2)
        .attr('stroke-dasharray', '6,3')

      // Axes
      g.append('g').attr('transform', `translate(0,${h})`)
        .call(d3.axisBottom(xScale).ticks(10))
        .call(g => g.select('.domain').attr('stroke', muted))
        .call(g => g.selectAll('.tick text').attr('fill', muted))
        .call(g => g.selectAll('.tick line').attr('stroke', muted))

      g.append('g').call(d3.axisLeft(yDist).ticks(5))
        .call(g => g.select('.domain').attr('stroke', gold))
        .call(g => g.selectAll('.tick text').attr('fill', gold))
        .call(g => g.selectAll('.tick line').attr('stroke', gold))

      g.append('g').attr('transform', `translate(${w},0)`)
        .call(d3.axisRight(yExcited).ticks(5))
        .call(g => g.select('.domain').attr('stroke', '#82b1ff'))
        .call(g => g.selectAll('.tick text').attr('fill', '#82b1ff'))
        .call(g => g.selectAll('.tick line').attr('stroke', '#82b1ff'))

      // Legend
      g.append('line').attr('x1', 10).attr('y1', 10).attr('x2', 30).attr('y2', 10)
        .attr('stroke', gold).attr('stroke-width', 2)
      g.append('text').attr('x', 35).attr('y', 14).attr('fill', gold).style('font-size', '11px')
        .text('Ligand Distance (Å)')

      g.append('line').attr('x1', 10).attr('y1', 30).attr('x2', 30).attr('y2', 30)
        .attr('stroke', '#82b1ff').attr('stroke-width', 2).attr('stroke-dasharray', '6,3')
      g.append('text').attr('x', 35).attr('y', 34).attr('fill', '#82b1ff').style('font-size', '11px')
        .text('Excited State Count')

      svg.append('text').attr('x', width / 2).attr('y', 24)
        .attr('text-anchor', 'middle').attr('fill', white)
        .style('font-size', '14px').style('font-family', 'Poppins')
        .text('Convergence: Distance & Excited States')
    }

  }, [trajectory, experiments, activeStep])

  if (!trajectory) {
    return <div className="chart-container"><span style={{ color: '#808080' }}>Loading dynamics data...</span></div>
  }

  return (
    <div className="chart-container">
      <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />
    </div>
  )
}
