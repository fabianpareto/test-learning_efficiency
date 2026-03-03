from __future__ import annotations

import json

from cross_check import cross_check_hypotheses
from hypothesis_generator import generate_txtl_hypothesis


def main():
    hypotheses = generate_txtl_hypothesis(
        num_hypotheses=2,
        verbose=True,
    )

    print("\n=== Hypotheses ===")
    print(json.dumps(hypotheses, indent=2))

    print("\n=== Cross-check (with oracle fallback) ===")
    results = cross_check_hypotheses(hypotheses, verbose=True, use_oracle_fallback=True)

    print("\n=== Summary ===")
    for r in results:
        exp = r["experimental_productivity"]
        oracle = r.get("oracle_prediction")
        if exp.get("has_data"):
            print(f"- {r['hypothesis_name']}: experimental min={exp['min_productivity']:.3f} mg/mL/hr")
        elif oracle is not None:
            print(
                f"- {r['hypothesis_name']}: oracle={oracle['predicted_productivity_mg_mL_hr']:.3f} mg/mL/hr"
            )
        else:
            print(f"- {r['hypothesis_name']}: no experimental match, oracle unavailable")


if __name__ == "__main__":
    main()
