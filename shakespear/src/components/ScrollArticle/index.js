// ScrollArticle stub -- replace with full implementation
export default function ScrollArticle({ chartComponent, sections, onStepChange, activeStep }) {
  return (
    <div className="flex gap-8 flex-wrap xl:flex-nowrap">
      <div className="flex-1 min-w-[300px]">
        {chartComponent}
      </div>
      <div className="flex-1 min-w-[300px] space-y-12">
        {sections && sections.map((section, i) => (
          <div key={section.id || i}
            className={`transition-opacity duration-300 ${i === activeStep ? 'opacity-100' : 'opacity-50'}`}
            onClick={() => onStepChange && onStepChange(i)}>
            {section.content}
          </div>
        ))}
      </div>
    </div>
  )
}
