/**
 * Model Compiler (Purpose-Trained Probe Interface)
 * ==================================================
 * Connects to OpenAI API to compile natural language queries
 * into operation sequences for the shader engine.
 *
 * This replaces the pattern-matching parseQuery() with an LLM
 * that understands protein science and the Partition Calculus.
 *
 * The model is the COMPILER. The shaders are the CPU.
 * The model produces operations. The GPU executes them.
 *
 * Model: gpt-4.1-mini (fast, cheap, good at structured output)
 * Endpoint: /api/compile (Next.js API route, keeps key server-side)
 */

const SYSTEM_PROMPT = `You are shakespear, a protein observation instrument compiler.
You translate natural language questions about proteins into executable operation sequences.

The instrument has these operations:
- probe: Compute S-entropy field from amino acid sequence
- couple: Build coupling matrix K_ij between all residue pairs
- observe: Generate coupling spectrum (synthetic 2D-IR)
- diagnose: Measure coherence order parameter η and predict contacts
- cavities: Detect virtual resonant cavities in harmonic network
- complete: Run full pipeline and extract cavity fingerprint
- mutate: Introduce a point mutation and measure coherence change
- compare: Compare two protein sequences
- predict: Predict activity (pIC50) from cavity fingerprint
- search: Search cavity database for similar proteins

For mutations, extract the position and amino acid change.
For comparisons, extract both sequences or identifiers.

Respond with ONLY a JSON object, no markdown, no explanation:
{
  "operation": "one of the operation names above",
  "view": "spectrum|coupling|contacts|cavity|sentropy",
  "args": { optional arguments like "pos", "from", "to" for mutations },
  "explanation": "one sentence explaining what will be computed"
}

Examples:
User: "Will this protein fold stably?"
{"operation":"diagnose","view":"contacts","explanation":"Measuring coherence order parameter to assess folding stability"}

User: "Mutate position 4 to alanine"
{"operation":"mutate","view":"contacts","args":{"pos":3,"to":"A"},"explanation":"Introducing T4A mutation and comparing coherence before/after"}

User: "Show me the 2D-IR spectrum"
{"operation":"observe","view":"spectrum","explanation":"Computing coupling spectrum between all residue pairs"}

User: "Find similar proteins in the database"
{"operation":"search","view":"cavity","explanation":"Extracting cavity fingerprint and searching database for matches"}

User: "Is the G93A mutation in SOD1 pathogenic?"
{"operation":"mutate","view":"contacts","args":{"pos":92,"to":"A"},"explanation":"Testing G93A SOD1 mutation for coherence disruption (ALS-associated variant)"}

User: "What are the binding sites?"
{"operation":"cavities","view":"cavity","explanation":"Detecting virtual resonant cavities that indicate potential binding pockets"}`;

/**
 * Compile a natural language query using the OpenAI API.
 * Calls the Next.js API route which holds the key server-side.
 */
export async function compileWithModel(query, sequence) {
  try {
    const res = await fetch('/api/compile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, sequence: sequence.substring(0, 100) }),
    });

    if (!res.ok) {
      console.warn('Model compile failed, falling back to pattern matching');
      return null;
    }

    const data = await res.json();
    return data;
  } catch (e) {
    console.warn('Model compile error:', e.message);
    return null;
  }
}

/** The system prompt for the API route to use. */
export { SYSTEM_PROMPT };

/**
 * Client-side fallback: direct API call (if no server route).
 * Only use this if the user explicitly provides their key in the UI.
 * Key is NOT stored -- held only in component state for the session.
 */
export async function compileWithModelDirect(query, sequence, apiKey) {
  if (!apiKey) return null;

  try {
    const res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'gpt-4.1-mini',
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          { role: 'user', content: `Protein sequence (first 100 residues): ${sequence.substring(0, 100)}\n\nQuestion: ${query}` },
        ],
        temperature: 0,
        max_tokens: 200,
        response_format: { type: 'json_object' },
      }),
    });

    if (!res.ok) return null;

    const data = await res.json();
    const content = data.choices?.[0]?.message?.content;
    if (!content) return null;

    return JSON.parse(content);
  } catch (e) {
    console.warn('Direct model compile error:', e.message);
    return null;
  }
}
