/**
 * Query Interface
 * ================
 * Natural language interface to the shader engine.
 * User types a question → parsed into operations → executed on GPU → result displayed.
 *
 * This is where the Purpose model plugs in.
 * For now: pattern matching. Later: LoRA-adapted LLM.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { parseQuery, executeOperation, formatResult } from './OperationExecutor';
import { compileWithModel, compileWithModelDirect } from './ModelCompiler';

const SUGGESTIONS = [
  'Will this protein fold?',
  'Show me the coupling spectrum',
  'Detect virtual cavities',
  'Mutate T4A',
  'Is this mutation pathogenic?',
  'Search database for similar proteins',
  'Predict activity',
  'Show coherence',
  'What are the binding sites?',
];

export default function QueryInterface({ engine, sequence, setSequence, setView, db, sar, fingerprint }) {
  const [query, setQuery] = useState('');
  const [history, setHistory] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [useModel, setUseModel] = useState(true);
  const [apiKey, setApiKey] = useState('');
  const [showKeyInput, setShowKeyInput] = useState(false);
  const [compiling, setCompiling] = useState(false);
  const inputRef = useRef(null);
  const historyRef = useRef(null);

  // Auto-scroll history
  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [history]);

  const handleSubmit = useCallback(async (q) => {
    const text = q || query;
    if (!text.trim() || compiling) return;

    let parsed = null;

    // Try model compilation first (if enabled)
    if (useModel) {
      setCompiling(true);
      try {
        // Try server-side route first (key in .env.local)
        let modelResult = await compileWithModel(text, sequence);

        // If server route fails, try direct with client key
        if (!modelResult && apiKey) {
          modelResult = await compileWithModelDirect(text, sequence, apiKey);
        }

        if (modelResult && modelResult.operation) {
          parsed = {
            operation: modelResult.operation,
            view: modelResult.view,
            args: modelResult.args,
            raw: text,
            source: 'model',
            explanation: modelResult.explanation,
          };
        }
      } catch (e) {
        // Fall through to pattern matching
      }
      setCompiling(false);
    }

    // Fallback to pattern matching
    if (!parsed) {
      parsed = parseQuery(text);
      parsed.source = 'pattern';
    }

    // Execute on engine
    const result = executeOperation(parsed, engine, sequence, db, sar);

    // Add model explanation if available
    if (parsed.explanation) {
      result.message = parsed.explanation;
    }

    // Switch view
    if (result.view) setView(result.view);

    // Handle mutation (updates sequence)
    if (result.newSequence) {
      setSequence(result.newSequence);
    }

    // Add to history
    const formatted = formatResult(result);
    const sourceTag = parsed.source === 'model' ? ' [model]' : ' [local]';
    setHistory(prev => [...prev, {
      query: text, result: formatted + sourceTag, timestamp: Date.now()
    }]);

    setQuery('');
    setShowSuggestions(false);
  }, [query, engine, sequence, setSequence, setView, db, sar, useModel, apiKey, compiling]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="mt-4">
      {/* History */}
      {history.length > 0 && (
        <div ref={historyRef}
          className="bg-dark/5 dark:bg-dark/80 border border-dark/10 dark:border-dark/20
                     rounded-lg p-3 mb-3 max-h-48 overflow-y-auto font-mono text-[11px]
                     leading-relaxed">
          {history.map((h, i) => (
            <div key={i} className="mb-2">
              <div className="text-primary dark:text-primaryDark">
                <span className="text-dark/30 dark:text-light/20">
                  {new Date(h.timestamp).toLocaleTimeString().slice(0, 5)}
                </span>
                {' '}$ {h.query}
              </div>
              <pre className="text-dark/60 dark:text-light/40 whitespace-pre-wrap ml-2">
                {h.result}
              </pre>
            </div>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="relative">
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-primary dark:text-primaryDark
                             font-mono text-sm pointer-events-none">$</span>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={e => { setQuery(e.target.value); setShowSuggestions(e.target.value.length === 0); }}
              onKeyDown={handleKeyDown}
              onFocus={() => query.length === 0 && setShowSuggestions(true)}
              onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
              className="w-full bg-light dark:bg-dark/80 border border-dark/10 dark:border-dark/30
                         rounded-lg py-2.5 pl-7 pr-3 font-mono text-sm text-dark dark:text-light
                         focus:outline-none focus:border-primaryDark placeholder:text-dark/30
                         dark:placeholder:text-light/20"
              placeholder="Ask about this protein..."
            />
          </div>
          <button onClick={() => handleSubmit()}
            className="px-4 py-2.5 bg-primary dark:bg-primaryDark text-light dark:text-dark
                       rounded-lg font-mono text-sm font-bold hover:opacity-90 transition-opacity">
            Run
          </button>
        </div>

        {/* Suggestions dropdown */}
        {showSuggestions && (
          <div className="absolute z-20 left-0 right-0 mt-1 bg-light dark:bg-dark border
                          border-dark/10 dark:border-dark/30 rounded-lg shadow-lg overflow-hidden">
            {SUGGESTIONS.map((s, i) => (
              <button key={i}
                onMouseDown={(e) => { e.preventDefault(); setQuery(s); handleSubmit(s); }}
                className="w-full text-left px-4 py-2 text-xs font-mono text-dark/60 dark:text-light/50
                           hover:bg-primaryDark/10 hover:text-primary dark:hover:text-primaryDark
                           transition-colors border-b border-dark/5 dark:border-dark/20 last:border-0">
                {s}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Quick action chips + model controls */}
      <div className="mt-2 flex flex-wrap gap-1 items-center">
        {['Fold?', 'Spectrum', 'Cavities', 'Coherence', 'Search DB', 'Predict'].map(chip => (
          <button key={chip}
            onClick={() => handleSubmit(chip)}
            className="px-2 py-0.5 text-[10px] rounded-full border border-dark/10 dark:border-dark/30
                       text-dark/40 dark:text-light/30 hover:border-primaryDark/50 hover:text-primaryDark
                       transition-colors font-mono">
            {chip}
          </button>
        ))}

        <span className="mx-1 text-dark/10 dark:text-light/10">|</span>

        {/* Model toggle */}
        <button onClick={() => setUseModel(!useModel)}
          className={`px-2 py-0.5 text-[10px] rounded-full border transition-colors font-mono
            ${useModel
              ? 'border-primaryDark/50 text-primaryDark bg-primaryDark/10'
              : 'border-dark/10 dark:border-dark/30 text-dark/30 dark:text-light/25'}`}>
          {useModel ? 'Model ON' : 'Model OFF'}
        </button>

        {/* API key button */}
        {useModel && (
          <button onClick={() => setShowKeyInput(!showKeyInput)}
            className={`px-2 py-0.5 text-[10px] rounded-full border transition-colors font-mono
              ${apiKey
                ? 'border-green-500/50 text-green-500'
                : 'border-dark/10 dark:border-dark/30 text-dark/30 dark:text-light/25'}`}>
            {apiKey ? 'Key Set' : 'Set API Key'}
          </button>
        )}

        {compiling && (
          <span className="text-[10px] text-primaryDark font-mono animate-pulse">compiling...</span>
        )}
      </div>

      {/* API Key Input (hidden by default) */}
      {showKeyInput && (
        <div className="mt-2 flex gap-2">
          <input
            type="password"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            className="flex-1 bg-light dark:bg-dark/80 border border-dark/10 dark:border-dark/30
                       rounded-lg py-1.5 px-3 font-mono text-[11px] text-dark dark:text-light
                       focus:outline-none focus:border-primaryDark
                       placeholder:text-dark/30 dark:placeholder:text-light/20"
            placeholder="sk-... (OpenAI API key, stored in session only)"
          />
          <button onClick={() => setShowKeyInput(false)}
            className="px-3 py-1.5 text-[10px] rounded-lg border border-dark/10 dark:border-dark/30
                       text-dark/40 dark:text-light/40 font-mono">
            Done
          </button>
        </div>
      )}
    </div>
  );
}
