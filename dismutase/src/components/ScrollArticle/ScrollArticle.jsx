import React, { useEffect, useRef, useCallback } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/dist/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

/**
 * ScrollArticle - Scrollytelling layout component
 *
 * Follows the GSAP ScrollTrigger pin pattern from the vanilla JS example:
 * - Chart wrapper is pinned via ScrollTrigger.create({ pin: true })
 * - Text steps sit on the right, each with large bottom margins
 * - Each step triggers onEnter/onLeaveBack to change the chart
 * - Steps get toggleClass 'active' for opacity transitions
 *
 * Uses `scroller` option to target .cavani_tm_section (the overflow-y:scroll
 * container in this template) instead of window scroll.
 */
export default function ScrollArticle({ chartComponent, sections, onStepChange, activeStep = 0 }) {
  const chartWrapperRef = useRef(null)
  const containerRef = useRef(null)
  const stepRefs = useRef([])

  // Find the .cavani_tm_section ancestor (the overflow-y:scroll container)
  const findScroller = useCallback((el) => {
    let node = el
    while (node) {
      if (node.classList && node.classList.contains('cavani_tm_section')) return node
      node = node.parentElement
    }
    return null
  }, [])

  useEffect(() => {
    if (!containerRef.current || !chartWrapperRef.current) return

    const scroller = findScroller(containerRef.current)
    if (!scroller) return

    const lastStepEl = stepRefs.current[stepRefs.current.length - 1]
    if (!lastStepEl) return

    // Pin the chart wrapper — exactly like the vanilla example
    // but with scroller pointing to .cavani_tm_section
    const pinTrigger = ScrollTrigger.create({
      trigger: chartWrapperRef.current,
      endTrigger: lastStepEl,
      scroller: scroller,
      start: 'top top',
      end: () => {
        const scrollerHeight = scroller.clientHeight
        const chartHeight = chartWrapperRef.current.offsetHeight
        return `bottom ${chartHeight + (scrollerHeight - chartHeight) / 2}px`
      },
      pin: true,
      pinSpacing: false,
    })

    // Toggle active class on each step for opacity
    const stepTriggers = stepRefs.current.map((step, i) => {
      if (!step) return null

      return ScrollTrigger.create({
        trigger: step,
        scroller: scroller,
        start: 'top 80%',
        end: 'center top',
        toggleClass: { targets: step, className: 'scroll-article__step--active' },
        onEnter: () => onStepChange && onStepChange(i),
        onEnterBack: () => onStepChange && onStepChange(i),
      })
    })

    return () => {
      pinTrigger.kill()
      stepTriggers.forEach(t => t && t.kill())
    }
  }, [sections.length, onStepChange, findScroller])

  return (
    <div className="scroll-article" ref={containerRef}>
      <div className="scroll-article__chart-panel" ref={chartWrapperRef}>
        <div className="scroll-article__chart-inner">
          {chartComponent}
        </div>
      </div>
      <div className="scroll-article__text-panel">
        {sections.map((section, i) => (
          <div
            key={section.id || i}
            ref={el => stepRefs.current[i] = el}
            className="scroll-article__step"
          >
            {section.content}
          </div>
        ))}
      </div>
    </div>
  )
}
