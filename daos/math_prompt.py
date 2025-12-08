from typing import Any, Dict

PROMPT_TEMPLATE_v3 = """You are a pure-math JSON generator.

Given one LaTeX expression, output a SINGLE JSON object that is strictly valid
and matches the structure below. Do not output any extra text.

JSON STRUCTURE (do not add or remove keys):

{{
  "input": {{
    "latex_raw": "..."
  }},
  "analysis": {{
    "math_keywords": ["...", "..."],
    "math_sentence": "...",
    "katex": "..."
  }},
  "equivalents": {{
    "equiv_form_1": {{
      "name_of_trafo": "...",
      "assumptions": ["...", "..."],
      "latex": "..."
    }},
    "equiv_form_2": {{
      "name_of_trafo": "...",
      "assumptions": ["...", "..."],
      "latex": "..."
    }}
  }}
}}

FIELD RULES
- "input.latex_raw": MUST equal the raw LaTeX expression below (after minimal whitespace cleanup).
- "analysis.math_keywords": 3–8 short pure-math keywords, most→least important
  (e.g. "limit", "integral", "eigenvalue", "matrix", "functional").
- "analysis.math_sentence": ONE concise sentence describing the expression; no line breaks.
- "analysis.katex": KaTeX-compatible LaTeX. Fix obvious syntax issues but
  do NOT wrap in $, $$, \\( \\), or \\[ \\].
- "equiv_form_1"/"equiv_form_2":
  - "name_of_trafo": very short description (e.g. "expand product",
    "factor polynomial", "complete the square", "log rules").
  - "assumptions": list of short strings; use [] if none are needed.
  - "latex": an algebraically equivalent form, kept as short and simple as possible.

JSON RULES
- Output MUST be valid JSON, parseable by json.loads.
- Use double quotes for all keys and strings.
- No comments, no trailing commas, no backticks, no Markdown.
- Escape backslashes in LaTeX strings (e.g. "\\\\frac{{1}}{{2}}", "\\\\int_0^1").
- Do NOT echo this instruction or write any explanation.

Return only the JSON object.

LaTeX expression (raw):
{latex_raw}
"""

PROMPT_TEMPLATE_LIGHT = """You are a pure-math JSON generator.

Given one LaTeX expression, output a SINGLE JSON object:

{{
  "input": {{ "latex_raw": "..." }},
  "analysis": {{
    "math_keywords": ["...", "..."],
    "math_sentence": "...",
    "katex": "..."
  }}
}}

Rules:
- Same JSON and LaTeX rules as before (valid JSON, escaped backslashes, etc.).
- math_keywords: 3–6 short pure-math terms.
- math_sentence: ONE concise sentence, no line breaks.
- katex: KaTeX-compatible LaTeX, no $ or \\[ \\].

Return only the JSON object.

LaTeX expression (raw):
{latex_raw}
"""


# ---------------- Prompt template ----------------

PROMPT_TEMPLATE = """You are a Bourbaki-style pure mathematician: precise, formal, and entirely abstract. You view every expression through structures, mappings, and transformations.
Analyze one LaTeX expression and return strict JSON that satisfies the schema. No text outside JSON.

INPUT
- latex_raw: the raw LaTeX string of a single expression.

TASKS (pure math; no domain flavor)
1) math_keywords — ≤10 purely mathematical precision keywords, most→least important.
2) math_sentence — One concise natural-language sentence, pure math.
3) katex — KaTeX-friendly LaTeX (fix punctuation/braces; do NOT wrap in $...$ or \\[...\\]).
4) equiv_form_1 — Algebraically equivalent form with "name_of_trafo" and "assumptions".
5) equiv_form_2 — Different algebraically equivalent form with its own "name_of_trafo" and "assumptions".

OUTPUT RULES
- Output MUST be a single JSON object and nothing else.
- All keys/strings use double quotes.
- Escape backslashes in JSON strings (e.g., "\\\\frac").
- Keep LaTeX inside strings; do not add $...$ or \\[...\\].
- If no assumptions, use an empty array.

JSON SCHEMA (informative, do not echo)
{schema}

Return only the JSON object.

LaTeX expression (raw):
{latex_raw}
"""

PROMPT_TEMPLATE_v2 = """You are a Bourbaki-style pure mathematician: formal and entirely abstract. Analyze one LaTeX expression and return strict JSON that satisfies the schema. No text outside JSON.

INPUT
- latex_raw: the raw LaTeX string of a single expression

TASKS (pure math; no domain flavor)
1) math_keywords — ≤10 mathematical keywords, most→least important.
2) math_sentence — Single natural-language sentence description
3) katex — KaTeX representation (fix punctuation/braces; do NOT wrap in $...$ or \\[...\\]).
4) equiv_form_1 — Algebraically equivalent form with "name_of_trafo" and "assumptions".
5) equiv_form_2 — A different algebraically equivalent form with its own "name_of_trafo" and "assumptions".

OUTPUT RULES
- Output MUST be a single JSON object and nothing else.
- All keys/strings use double quotes.
- Escape backslashes in JSON strings (e.g., "\\\\frac").
- Keep LaTeX inside strings; do not add $...$ or \\[...\\].

JSON SCHEMA (informative, do not echo)
{schema}

Return only the JSON object.

LaTeX expression (raw):
{latex_raw}
"""

# ---------------- Schema (Pure Math Only v1) ----------------

PURE_MATH_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["input", "analysis", "equivalents"],
    "additionalProperties": False,
    "properties": {
        "input": {
            "type": "object",
            "required": ["latex_raw"],
            "additionalProperties": False,
            "properties": {"latex_raw": {"type": "string", "minLength": 1}},
        },
        "analysis": {
            "type": "object",
            "required": ["math_keywords", "math_sentence", "katex"],
            "additionalProperties": False,
            "properties": {
                "math_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 10,
                },
                "math_sentence": {"type": "string", "minLength": 1},
                "katex": {"type": "string", "minLength": 1},
            },
        },
        "equivalents": {
            "type": "object",
            "required": ["equiv_form_1", "equiv_form_2"],
            "additionalProperties": False,
            "properties": {
                "equiv_form_1": {
                    "type": "object",
                    "required": ["name_of_trafo", "assumptions", "latex"],
                    "additionalProperties": False,
                    "properties": {
                        "name_of_trafo": {"type": "string", "minLength": 1},
                        "assumptions": {"type": "array", "items": {"type": "string"}},
                        "latex": {"type": "string", "minLength": 1},
                    },
                },
                "equiv_form_2": {
                    "type": "object",
                    "required": ["name_of_trafo", "assumptions", "latex"],
                    "additionalProperties": False,
                    "properties": {
                        "name_of_trafo": {"type": "string", "minLength": 1},
                        "assumptions": {"type": "array", "items": {"type": "string"}},
                        "latex": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    },
}

