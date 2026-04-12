/**
 * /api/compile -- Server-side model compilation endpoint.
 * Calls OpenAI API with the key from environment variable.
 * Keeps the API key on the server, never exposed to client.
 *
 * Set OPENAI_API_KEY in .env.local:
 *   OPENAI_API_KEY=sk-...
 */

import { SYSTEM_PROMPT } from '@/components/proteus/ModelCompiler';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'OPENAI_API_KEY not configured' });
  }

  const { query, sequence } = req.body;
  if (!query) {
    return res.status(400).json({ error: 'Missing query' });
  }

  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'gpt-4.1-mini',
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          {
            role: 'user',
            content: `Protein sequence (first 100 residues): ${(sequence || '').substring(0, 100)}\n\nQuestion: ${query}`,
          },
        ],
        temperature: 0,
        max_tokens: 200,
        response_format: { type: 'json_object' },
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      return res.status(response.status).json({ error: err });
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content;
    if (!content) {
      return res.status(500).json({ error: 'Empty model response' });
    }

    const parsed = JSON.parse(content);
    return res.status(200).json(parsed);
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
}
