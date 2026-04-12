import React, { useEffect, useRef, useCallback } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/dist/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

/**
 * ScrollArticle - Scrollytelling layout component
 *
 * Left: chart panel (pinned via GSAP ScrollTrigger)
 * Right: scrolling text steps
 *
 * Uses window scroll (no custom scroller container needed).
 */
export default function ScrollArticle({ chartComponent, sections, onStepChange, activeStep = 0 }) {
  const chartWrapperRef = useRef(null)
  const containerRef = useRef(null)
  const stepRefs = useRef([])

  useEffect(() => {
    if (!containerRef.current || !chartWrapperRef.current) return
    if (!stepRefs.current.length) return

    const lastStepEl = stepRefs.current[stepRefs.current.length - 1]
    if (!lastStepEl) return

    // Pin the chart wrapper using window scroll
    const pinTrigger = ScrollTrigger.create({
      trigger: chartWrapperRef.current,
      endTrigger: lastStepEl,
      start: 'top 80px', // account for navbar height
      end: 'bottom center',
      pin: true,
      pinSpacing: false,
    })

    // Toggle active class on each step
    const stepTriggers = stepRefs.current.map((step, i) => {
      if (!step) return null
      return ScrollTrigger.create({
        trigger: step,
        start: 'top 70%',
        end: 'bottom 30%',
        toggleClass: { targets: step, className: 'scroll-article__step--active' },
        onEnter: () => onStepChange && onStepChange(i),
        onEnterBack: () => onStepChange && onStepChange(i),
      })
    })

    return () => {
      pinTrigger.kill()
      stepTriggers.forEach(t => t && t.kill())
    }
  }, [sections.length, onStepChange])

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
