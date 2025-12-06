from typing import Any, Dict

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
