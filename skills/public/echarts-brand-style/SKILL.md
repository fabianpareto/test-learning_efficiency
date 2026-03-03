---
name: echarts-brand-style
description: Create Apache ECharts option JSON and Webflow embed snippets using the approved brand style for evaluation reporting. Use when asked to build or restyle grouped bar charts, enforce equal bar widths, keep zero-value series visible and labeled, remove distracting bar-slot backgrounds, or debug chart rendering in Webflow.
---

# Echarts Brand Style

## Overview

Create clean, reusable ECharts output for model-eval charts with consistent typography, spacing, color usage, and Webflow compatibility.
Prefer deterministic JSON output over ad hoc chart configuration.

## Workflow

1. Normalize the input data.
Input can be long format (`scenario, model, pass_rate`) or already pivoted.
Convert rates to percentages if needed.

2. Apply style rules from [references/style-guide.md](references/style-guide.md).
Use approved palettes, typography, axis styling, and spacing defaults.
Choose colors by semantic role, not by arbitrary ordering.

3. Enforce bar-readability defaults.
Use equal widths across series (`barWidth` and `barMaxWidth`).
Set `showBackground: false` unless the user explicitly asks for bar-slot backgrounds.
Use `barMinHeight` and explicit `0%` labels for series with many zero values.
Avoid bar borders unless requested.

4. Emit deliverables.
Write a pure JSON option file.
If asked for embed code, provide a Webflow-safe loader snippet with explicit height and error handling.

## Color Decision Rules

Use this decision sequence when assigning series colors:

1. Determine intent:
Comparison-only, ranking, highlight, or outlier emphasis.

2. Select palette mode:
- `purple` for unified, low-noise comparison.
- `multicolor` for stronger differentiation between model groups or roles.

3. Assign by role:
- Keep neutrals (`cream`, `white`, `off-black`) for backgrounds/text only.
- In `multicolor`, keep mint as highlight/accent, not default primary series color.
- In both palettes, use same-hue opacity for zero-heavy series.

4. Validate coherence:
Ensure adjacent series are distinguishable but still visually related.
Avoid mixing unrelated accents unless the chart has an explicit highlight story.

## Input Contract

For grouped bar charts, use one of:

- Long format CSV columns:
`prompt_id, model_name, pass_rate`

- Wide format table:
`scenario, <series_1>, <series_2>, ...`

Pass rates should be in `0..1` or `0..100`; convert to percentages in output.

## Output Contract

Return:

1. `option` JSON valid for ECharts v5.
2. Optional Webflow embed snippet if requested.
3. Explicit file path if writing to disk.

Keep JSON static-friendly:

- Prefer literal values over JS function formatters when possible.
- If formatter functions are necessary, note that CSP may block inline JS in some Webflow setups.

## Resources

- Style rules: [references/style-guide.md](references/style-guide.md)
- Base template: [assets/templates/grouped_bar_option_template.json](assets/templates/grouped_bar_option_template.json)
- Generator script: [scripts/build_grouped_bar_option.py](scripts/build_grouped_bar_option.py)

## Quick Commands

Generate from long-format CSV:

```bash
python3 scripts/build_grouped_bar_option.py \
  --input data.csv \
  --output chart.json \
  --title "Chart title" \
  --palette multicolor
```

Palette options:

- `multicolor` -> role-based selection from full multicolor range
- `purple` -> role-based selection from full purple range

Optional highlight command:

```bash
python3 scripts/build_grouped_bar_option.py \
  --input data.csv \
  --output chart.json \
  --title "Chart title" \
  --palette multicolor \
  --use-mint-highlight \
  --highlight-index 2
```
