# Notebook Split

`llm_api_notebook.ipynb` logic is now split into standalone Python files:

- `llm_client.py`
  - OpenRouter client setup
  - model registry (`FREE_MODELS`, `MODELS`, `DEFAULT_MODEL`)
  - `call_llm`, `call_llm_structured`, `call_llm_json`
- `hypothesis_generator.py`
  - JSON extraction/parsing helpers
  - `generate_txtl_hypothesis`
  - `hypotheses_to_dataframe`
- `cross_check.py`
  - CSV experiment matching logic
  - oracle fallback prediction when no exact match
  - `cross_check_hypotheses`, `summarize_evidence`, `get_best_experimental_hypothesis`
- `iterative_generation.py`
  - `TXTLHypothesisIterator`
  - `run_hypothesis_iterations`
  - `benchmark_free_models`
- `txtl_pipeline.py`
  - Minimal end-to-end script for generation + cross-check (with oracle fallback)

## Quick Start

```bash
python3 txtl_pipeline.py
```

If you want to keep running from the notebook, import from these files instead of duplicating logic in cells.
