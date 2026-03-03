from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from llm_client import DEFAULT_MODEL, get_client


TXTL_PARAMETERS = {
    "Mg_glutamate_mM": "Mg-glutamate concentration in mM (optimal: 4-6 mM additional)",
    "K_glutamate_mM": "K-glutamate concentration in mM (optimal: 60-80 mM)",
    "DTT_mM": "DTT concentration in mM (typical: 0-3 mM)",
    "NTP_multiplier": "NTP concentration as multiplier of standard 1mM ATP/GTP (e.g., 1, 2, 3)",
    "energy_source": "Energy regeneration system: '3-PGA', '3-PGA + maltose', '3-PGA + maltodextrin', or 'dialysis'",
    "PEG_8000_percent": "PEG-8000 crowding agent percentage (optimal: 2%)",
    "extract_type": "Extract strain: 'wild-type', 'KC6', or specific deletions",
    "temperature_C": "Reaction temperature in Celsius (25, 29-30, or 37)",
    "chaperones": "Chaperone addition: 'none', 'GroES/EL', 'DnaK/DnaJ/GrpE', or 'combined'",
    "plasmid_concentration_nM": "Plasmid DNA concentration in nM (optimal: 5-10 nM)",
    "reaction_mode": "Reaction mode: 'batch', 'semi-continuous', or 'microfluidic'",
}

FALLBACK_MODELS = [
    "google/gemini-2.5-flash",
    "deepseek/deepseek-v3.2-20251201",
    "openai/gpt-4o-mini",
]


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array found in response")

    bracket_positions = [i for i, c in enumerate(text) if c == "]"]
    for end_pos in reversed(bracket_positions):
        if end_pos <= start:
            continue
        candidate = text[start : end_pos + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    fixed = re.sub(r",\s*\]", "]", text[start:])
    fixed = re.sub(r",\s*\}", "}", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    raise ValueError("Could not parse JSON from response")


def _call_llm_for_hypotheses(messages: list[dict[str, str]], model: str, verbose: bool):
    response = get_client().chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=messages,
    )

    if verbose:
        print(f"[Model: {response.model}]")
        if response.choices[0].finish_reason != "stop":
            print(f"[Warning: finish_reason={response.choices[0].finish_reason}]")

    raw_text = response.choices[0].message.content
    return _extract_json_array(raw_text), raw_text


def generate_txtl_hypothesis(
    num_hypotheses: int = 3,
    context: str | None = None,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    system_prompt = """You are an expert in cell-free TX-TL optimization.
Return ONLY a valid JSON array. No markdown code blocks. No explanations."""

    user_prompt = f"""Generate {num_hypotheses} TX-TL hypotheses as a JSON array.

Required fields for each object:
- hypothesis_name (string, 3-5 words)
- Mg_glutamate_mM (number, 4-6 optimal)
- K_glutamate_mM (number, 60-80 optimal)
- DTT_mM (number, 0-3)
- NTP_multiplier (number, 1-3)
- energy_source (string: "3-PGA", "3-PGA + maltose", or "3-PGA + maltodextrin")
- PEG_8000_percent (number, 0-5)
- extract_type (string: "wild-type" or "KC6")
- temperature_C (number: 25, 30, or 37)
- chaperones (string: "none", "GroES/EL", "DnaK/DnaJ/GrpE", or "combined")
- plasmid_concentration_nM (number, 5-10 optimal)
- reaction_mode (string: "batch", "semi-continuous", or "microfluidic")
- estimated_productivity_mg_mL_hr (number, 0.1-0.8)
- rationale (string, 1 sentence)
{f"Context: {context}" if context else ""}

Return ONLY the JSON array:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]

    last_error = None
    last_raw_text = None
    for try_model in models_to_try:
        try:
            hypotheses, raw_text = _call_llm_for_hypotheses(messages, try_model, verbose)
            return hypotheses
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            last_raw_text = raw_text if "raw_text" in locals() else "N/A"
            if verbose:
                print(f"[Parse error with {try_model}, trying next model...]")

    print("All models failed to return valid JSON.")
    print(f"Last error: {last_error}")
    print("--- Last raw response ---")
    print(last_raw_text)
    raise ValueError("Could not generate valid JSON hypotheses from any model")


def hypotheses_to_dataframe(hypotheses: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(hypotheses)
