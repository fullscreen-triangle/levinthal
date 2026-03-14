import React, { useEffect, useRef, useCallback } from 'react'

/**
 * ScrollArticle - Scrollytelling layout component
 *
 * Two-column layout: left chart (position:fixed) + right scrolling text.
 *
 * Uses IntersectionObserver rooted on the .cavani_tm_section scroll
 * container to detect which text step is in view, then fires
 * onStepChange to update the chart. No GSAP dependency — this is
 * simpler and works reliably inside overflow:scroll containers.
 *
 * The chart panel is position:fixed when the section is active,
 * manually positioned to fill the left half of the mainpart area.
 */
export default function ScrollArticle({ chartComponent, sections, onStepChange, activeStep = 0 }) {
  const containerRef = useRef(null)
  const chartPanelRef = useRef(null)
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
    if (!containerRef.current) return

    const scroller = findScroller(containerRef.current)
    if (!scroller) return

    // Use IntersectionObserver rooted on the scroll container
    // to detect which step is currently in view
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const idx = stepRefs.current.indexOf(entry.target)
            if (idx !== -1 && onStepChange) {
              onStepChange(idx)
            }
          }
        })
      },
      {
        root: scroller,
        // Trigger when a step crosses the middle 20% band of the viewport
        rootMargin: '-40% 0px -40% 0px',
        threshold: 0,
      }
    )

    stepRefs.current.forEach(el => {
      if (el) observer.observe(el)
    })

    return () => observer.disconnect()
  }, [sections.length, onStepChange, findScroller])

  // Position the chart panel as fixed, aligned to the left half of the layout
  useEffect(() => {
    if (!chartPanelRef.current || !containerRef.current) return

    const scroller = findScroller(containerRef.current)
    if (!scroller) return

    function updateChartPosition() {
      const panel = chartPanelRef.current
      if (!panel) return

      const scrollerRect = scroller.getBoundingClientRect()

      // Fixed position within the left half of the scroller
      panel.style.position = 'fixed'
      panel.style.top = scrollerRect.top + 'px'
      panel.style.left = scrollerRect.left + 'px'
      panel.style.width = (scrollerRect.width / 2) + 'px'
      panel.style.height = scrollerRect.height + 'px'
      panel.style.zIndex = '5'
    }

    updateChartPosition()

    // Update on resize
    window.addEventListener('resize', updateChartPosition)

    // Also update if the scroller repositions (e.g., nav changes)
    const resizeObserver = new ResizeObserver(updateChartPosition)
    resizeObserver.observe(scroller)

    return () => {
      window.removeEventListener('resize', updateChartPosition)
      resizeObserver.disconnect()
      if (chartPanelRef.current) {
        chartPanelRef.current.style.position = ''
      }
    }
  }, [findScroller])

  return (
    <div className="scroll-article" ref={containerRef}>
      <div className="scroll-article__chart-panel" ref={chartPanelRef}>
        <div className="scroll-article__chart-inner">
          {chartComponent}
        </div>
      </div>
      <div className="scroll-article__text-panel">
        {sections.map((section, i) => (
          <div
            key={section.id || i}
            ref={el => stepRefs.current[i] = el}
            className={`scroll-article__step ${i === activeStep ? 'scroll-article__step--active' : ''}`}
          >
            {section.content}
          </div>
        ))}
      </div>
    </div>
  )
}
