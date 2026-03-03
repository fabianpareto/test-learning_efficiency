#!/usr/bin/env python3
"""
Build an ECharts grouped-bar option JSON from long-format eval CSV data.

Expected columns:
- prompt_id
- model_name
- pass_rate   (0..1 or 0..100)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

BRAND = {
    "cream": "#F8F6F0",
    "pareto_purple": "#6369E5",
    "purple_2": "#7777E8",
    "purple_3": "#8985EC",
    "purple_4": "#9A93EF",
    "purple_5": "#AAA1F3",
    "purple_6": "#B9B0F6",
    "purple_7": "#C8BFF9",
    "purple_8": "#D6CFFC",
    "purple_9": "#E4DEFF",
    "light_purple": "#D1D8FF",
    "brown": "#5B2D2D",
    "mint": "#EDFFD9",
    "white": "#FFFFFF",
    "off_black": "#141414",
}

PALETTES_FULL = {
    "purple": [
        BRAND["cream"],
        BRAND["pareto_purple"],
        BRAND["purple_2"],
        BRAND["purple_3"],
        BRAND["purple_4"],
        BRAND["purple_5"],
        BRAND["purple_6"],
        BRAND["purple_7"],
        BRAND["purple_8"],
        BRAND["purple_9"],
        BRAND["white"],
        BRAND["off_black"],
    ],
    "multicolor": [
        BRAND["cream"],
        BRAND["pareto_purple"],
        BRAND["light_purple"],
        BRAND["brown"],
        BRAND["mint"],
        BRAND["white"],
        BRAND["off_black"],
    ],
}

SERIES_SEQUENCES = {
    # Keep neutrals out of default series usage.
    "purple": [
        BRAND["pareto_purple"],
        BRAND["purple_2"],
        BRAND["purple_3"],
        BRAND["purple_4"],
        BRAND["purple_5"],
        BRAND["purple_6"],
        BRAND["purple_7"],
        BRAND["purple_8"],
        BRAND["purple_9"],
    ],
    # For multicolor, use mint only as explicit highlight.
    "multicolor": [
        BRAND["pareto_purple"],
        BRAND["brown"],
        BRAND["light_purple"],
        BRAND["purple_2"],
        BRAND["purple_3"],
        BRAND["purple_4"],
        BRAND["purple_5"],
        BRAND["purple_6"],
        BRAND["purple_7"],
        BRAND["purple_8"],
        BRAND["purple_9"],
    ],
}


def _format_label(prompt_id: str) -> str:
    mapping = {
        "dizziness_panic_attack": "Dizziness &\nPanic Attack",
        "hyperthyroidism_weakness": "Muscle Weakness &\nHyperthyroidism",
        "palpitation_keto_flu": "Heart Palpitations &\nKeto Flu",
        "pulsatile_tinnitus_anemia": "Pulsatile Tinnitus &\nAnemia",
    }
    if prompt_id in mapping:
        return mapping[prompt_id]

    text = prompt_id.replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    words = [w.capitalize() for w in text.split(" ") if w]
    if len(words) > 2:
        mid = len(words) // 2
        return f"{' '.join(words[:mid])}\n{' '.join(words[mid:])}"
    return " ".join(words)


def _normalize_rates(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    # If values are mostly <=1, treat as proportions.
    if (numeric <= 1.0).all():
        return numeric * 100.0
    return numeric


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    text = hex_color.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color}")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


def _is_light(hex_color: str) -> bool:
    r, g, b = _hex_to_rgb(hex_color)
    # Relative luminance approximation.
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance > 190


def _choose_series_colors(
    palette_name: str,
    n_series: int,
    use_mint_highlight: bool = False,
    highlight_index: int | None = None,
) -> list[str]:
    base = SERIES_SEQUENCES[palette_name]
    colors = [base[i % len(base)] for i in range(n_series)]

    if palette_name == "multicolor" and use_mint_highlight and highlight_index is not None:
        if 0 <= highlight_index < n_series:
            colors[highlight_index] = BRAND["mint"]
    return colors


def _build_option(
    df: pd.DataFrame,
    title: str,
    x_axis_name: str,
    y_axis_name: str,
    palette_name: str,
    use_mint_highlight: bool = False,
    highlight_index: int | None = None,
) -> dict:
    required = {"prompt_id", "model_name", "pass_rate"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["pass_rate"] = _normalize_rates(df["pass_rate"])

    pivot = (
        df.pivot_table(index="prompt_id", columns="model_name", values="pass_rate", aggfunc="mean")
        .fillna(0.0)
    )

    categories = [_format_label(idx) for idx in pivot.index.tolist()]
    model_names = [str(col) for col in pivot.columns.tolist()]
    series_colors = _choose_series_colors(
        palette_name=palette_name,
        n_series=len(model_names),
        use_mint_highlight=use_mint_highlight,
        highlight_index=highlight_index,
    )

    series = []
    for i, model in enumerate(model_names):
        vals = [float(v) for v in pivot[model].tolist()]
        color = series_colors[i]

        series_item = {
            "name": model,
            "type": "bar",
            "barWidth": 44,
            "barMaxWidth": 44,
            "barGap": "20%",
            "barCategoryGap": "35%",
            "showBackground": False,
            "data": vals,
            "itemStyle": {
                "color": color,
                "borderRadius": 0,
            },
            "label": {"show": False},
        }

        # For all-zero series, show faint bars + explicit 0% labels.
        if all(v == 0 for v in vals):
            series_item["barMinHeight"] = 5
            series_item["itemStyle"]["color"] = _rgba(color, 0.40)
            label_color = BRAND["off_black"] if _is_light(color) else color
            series_item["data"] = [
                {
                    "value": 0,
                    "label": {
                        "show": True,
                        "formatter": "0%",
                        "position": "top",
                        "distance": 2,
                        "fontSize": 12,
                        "fontWeight": "normal",
                        "color": label_color,
                    },
                }
                for _ in vals
            ]

        series.append(series_item)

    option = {
        "backgroundColor": "transparent",
        # Keep full ranges available for manual extension while using role-based series colors.
        "color": series_colors,
        "textStyle": {
            "color": BRAND["off_black"],
            "fontFamily": "Avenir Next, Avenir, Segoe UI, sans-serif",
        },
        "title": {
            "text": title,
            "left": "center",
            "top": 8,
            "textStyle": {
                "fontSize": 20,
                "fontWeight": "normal",
                "color": BRAND["off_black"],
            },
        },
        "legend": {
            "top": 88,
            "left": "center",
            "itemWidth": 34,
            "itemHeight": 12,
            "textStyle": {
                "fontSize": 14,
                "fontWeight": "normal",
                "color": BRAND["off_black"],
            },
            "data": model_names,
        },
        "tooltip": {
            "trigger": "axis",
            "backgroundColor": BRAND["cream"],
            "borderColor": BRAND["light_purple"],
            "borderWidth": 1,
            "textStyle": {
                "color": BRAND["off_black"],
                "fontWeight": "normal",
            },
            "axisPointer": {
                "type": "shadow",
                "shadowStyle": {"color": _rgba(BRAND["light_purple"], 0.35)},
            },
        },
        "grid": {"left": 100, "right": 30, "top": 150, "bottom": 130},
        "xAxis": {
            "type": "category",
            "position": "bottom",
            "data": categories,
            "axisTick": {"alignWithLabel": True, "lineStyle": {"color": BRAND["brown"]}},
            "axisLine": {"lineStyle": {"color": BRAND["off_black"], "width": 1.25}},
            "axisLabel": {
                "show": True,
                "position": "bottom",
                "interval": 0,
                "fontSize": 13,
                "lineHeight": 16,
                "margin": 16,
                "hideOverlap": True,
                "fontWeight": "normal",
                "color": BRAND["off_black"],
            },
            "name": x_axis_name,
            "nameLocation": "middle",
            "nameGap": 98,
            "nameTextStyle": {
                "fontSize": 17,
                "color": BRAND["off_black"],
                "fontWeight": "normal",
            },
        },
        "yAxis": {
            "type": "value",
            "min": 0,
            "max": 108,
            "interval": 20,
            "axisLine": {"lineStyle": {"color": BRAND["off_black"], "width": 1.25}},
            "axisLabel": {"formatter": "{value}%", "fontWeight": "normal", "color": BRAND["off_black"]},
            "name": y_axis_name,
            "nameLocation": "middle",
            "nameGap": 70,
            "nameTextStyle": {
                "fontSize": 18,
                "color": BRAND["off_black"],
                "fontWeight": "normal",
            },
            "splitLine": {
                "lineStyle": {
                    "type": "dashed",
                    "color": BRAND["light_purple"],
                    "opacity": 0.75,
                }
            },
        },
        "series": series,
    }
    return option


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grouped-bar ECharts option JSON from CSV.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--title", required=True, help="Chart title.")
    parser.add_argument(
        "--x-axis-name",
        default="High-stakes Medical Scenario",
        help="X-axis name label.",
    )
    parser.add_argument(
        "--y-axis-name",
        default="Appropriate Uncertainty Expression Rate",
        help="Y-axis name label.",
    )
    parser.add_argument(
        "--palette",
        choices=sorted(PALETTES_FULL.keys()),
        default="multicolor",
        help="Color palette name.",
    )
    parser.add_argument(
        "--use-mint-highlight",
        action="store_true",
        help="Use mint as an explicit highlight color for one series (multicolor palette only).",
    )
    parser.add_argument(
        "--highlight-index",
        type=int,
        default=None,
        help="Zero-based series index to highlight when --use-mint-highlight is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    option = _build_option(
        df=df,
        title=args.title,
        x_axis_name=args.x_axis_name,
        y_axis_name=args.y_axis_name,
        palette_name=args.palette,
        use_mint_highlight=args.use_mint_highlight,
        highlight_index=args.highlight_index,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(option, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
