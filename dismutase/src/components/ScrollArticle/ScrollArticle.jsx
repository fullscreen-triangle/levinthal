import React, { useEffect, useRef, useState } from 'react'

/**
 * ScrollArticle - Scrollytelling layout component
 *
 * Left side: sticky chart panel that stays in viewport
 * Right side: scrolling article sections
 *
 * GSAP ScrollTrigger fires onStepEnter callbacks to update chart state.
 *
 * Props:
 *   - chartComponent: React node rendered in the sticky left panel
 *   - sections: array of { id, content } where content is JSX
 *   - onStepChange: (stepIndex) => void — called when a new section scrolls into view
 *   - activeStep: current active step index (controlled from parent)
 */
export default function ScrollArticle({ chartComponent, sections, onStepChange, activeStep = 0 }) {
  const containerRef = useRef(null)
  const stepRefs = useRef([])

  useEffect(() => {
    let gsapModule, ScrollTriggerPlugin

    async function initScrollTrigger() {
      // Dynamic import to avoid SSR issues with Next.js 12
      gsapModule = (await import('gsap')).default
      ScrollTriggerPlugin = (await import('gsap/dist/ScrollTrigger')).default
      gsapModule.registerPlugin(ScrollTriggerPlugin)

      // Create a ScrollTrigger for each section
      const triggers = stepRefs.current.map((el, i) => {
        if (!el) return null
        return ScrollTriggerPlugin.create({
          trigger: el,
          start: 'top center',
          end: 'bottom center',
          onEnter: () => onStepChange && onStepChange(i),
          onEnterBack: () => onStepChange && onStepChange(i),
        })
      })

      return triggers
    }

    const triggersPromise = initScrollTrigger()

    return () => {
      triggersPromise.then(triggers => {
        if (triggers) triggers.forEach(t => t && t.kill())
      })
    }
  }, [sections.length, onStepChange])

  return (
    <div className="scroll-article" ref={containerRef}>
      <div className="scroll-article__chart-panel">
        <div className="scroll-article__chart-sticky">
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
