# ECharts Style Guide

## Brand Colors

- `cream`: `#F8F6F0`
- `pareto_purple`: `#6369E5`
- `purple_2`: `#7777E8`
- `purple_3`: `#8985EC`
- `purple_4`: `#9A93EF`
- `purple_5`: `#AAA1F3`
- `purple_6`: `#B9B0F6`
- `purple_7`: `#C8BFF9`
- `purple_8`: `#D6CFFC`
- `purple_9`: `#E4DEFF`
- `light_purple`: `#D1D8FF`
- `brown`: `#5B2D2D`
- `mint`: `#EDFFD9`
- `white`: `#FFFFFF`
- `off_black`: `#141414`

## Full Palettes

- `purple_full`:
  `["#F8F6F0", "#6369E5", "#7777E8", "#8985EC", "#9A93EF", "#AAA1F3", "#B9B0F6", "#C8BFF9", "#D6CFFC", "#E4DEFF", "#FFFFFF", "#141414"]`
- `multicolor_full`:
  `["#F8F6F0", "#6369E5", "#D1D8FF", "#5B2D2D", "#EDFFD9", "#FFFFFF", "#141414"]`

## Palette Selection Rules

Use colors by role, not by raw list order.

- Reserve neutrals for non-series roles:
  - `#F8F6F0` (cream) -> tooltip/background only when requested
  - `#FFFFFF` (white) -> background only when requested
  - `#141414` (off-black) -> text/axes
- For `purple` palette series:
  - Use `#6369E5` as anchor for primary series
  - Use `#7777E8` ... `#E4DEFF` for additional series in ordered lightness
  - Do not use cream/white/off-black as bar fills
- For `multicolor` palette series:
  - First three series: `#6369E5`, `#5B2D2D`, `#D1D8FF`
  - Use mint (`#EDFFD9`) only as explicit highlight/accent, not default for primary comparison bars
  - For more than 3-4 series, extend with purple shades to stay coherent
- For zero-heavy series:
  - Use the same hue at lower opacity instead of swapping to unrelated colors
  - Keep explicit `0%` labels

## Global Chart Defaults

- `backgroundColor`: `"transparent"`
- `textStyle.color`: `#141414`
- `textStyle.fontFamily`: `"Avenir Next, Avenir, Segoe UI, sans-serif"`
- Title/legend/axis text should use normal weight unless user asks for bold.

## Grouped Bar Defaults

- Equal widths across all series:
  - `barWidth`: `44`
  - `barMaxWidth`: `44`
  - `barGap`: `"20%"`
  - `barCategoryGap`: `"35%"`
- Keep bar-slot backgrounds off:
  - `showBackground`: `false`
- Keep bar borders off unless requested:
  - no `borderWidth`, no `borderColor`
- Keep bars square for cleaner equal-width perception:
  - `borderRadius`: `0`

## Zero-Value Visibility

For a series with many `0%` values:

- Set `barMinHeight`: `5`
- Use faint fill of the same series hue (for example `rgba(91, 45, 45, 0.35)`)
- Add explicit `0%` labels at top for those bars

## Axis and Grid

- `xAxis.position`: `"bottom"`
- `xAxis.axisLabel.interval`: `0`
- `xAxis.axisLabel.margin`: `16`
- `yAxis.splitLine.lineStyle.type`: `"dashed"`
- `yAxis.splitLine.lineStyle.color`: `#D1D8FF`

## Webflow Embedding Notes

- Chart container needs explicit height.
- Use one ECharts script include globally, or a guarded loader if multiple embeds are isolated.
- Validate JSON URL returns raw JSON, not HTML.
