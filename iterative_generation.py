import json

import pandas as pd

from cross_check import cross_check_hypotheses
from hypothesis_generator import _extract_json_array
from llm_client import get_client


class TXTLHypothesisIterator:
    def __init__(self, model="google/gemini-2.5-flash", verbose=True, use_oracle_fallback=True):
        self.model = model
        self.verbose = verbose
        self.use_oracle_fallback = use_oracle_fallback
        self.conversation_history = []
        self.all_hypotheses = []
        self.iteration = 0

        self.system_prompt = """You are an expert in cell-free TX-TL optimization.
You are helping design experiments to maximize protein productivity (mg/mL/hr).

When generating hypotheses, return ONLY a valid JSON array with these fields:
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
- rationale (string, 1 sentence explaining why this might work better)

Learn from the experimental feedback provided and generate improved hypotheses."""

    def _format_feedback(self, hypotheses, cross_check_results):
        feedback_lines = ["Here are the results from checking your hypotheses against experimental data:\n"]

        for i, (hyp, result) in enumerate(zip(hypotheses, cross_check_results)):
            feedback_lines.append(f"## Hypothesis {i+1}: {hyp.get('hypothesis_name', 'Unnamed')}")

            oracle_pred = result.get("oracle_prediction")
            if oracle_pred is not None:
                feedback_lines.append(
                    f"Predicted productivity: "
                    f"{oracle_pred['predicted_productivity_mg_mL_hr']:.3f} mg/mL/hr"
                )
            else:
                feedback_lines.append("Productivity prediction unavailable.")

            feedback_lines.append("")

        return "\n".join(feedback_lines)

    def build_round1_messages(self, num_hypotheses=3, context=None):
        user_prompt = f"""Generate {num_hypotheses} diverse TX-TL optimization hypotheses.
{f"Context: {context}" if context else ""}

Return ONLY the JSON array:"""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def build_round2_user_prompt(self, hypotheses, cross_check_results, num_hypotheses):
        feedback = self._format_feedback(hypotheses, cross_check_results)
        return f"""{feedback}

Based on this experimental feedback, generate {num_hypotheses} NEW hypotheses that:
1. Avoid combinations that showed low productivity 
2. Are currently unknown but potentially have higher productivity 
3. Explain your reasoning in the rationale field

Return ONLY the JSON array:"""

    def build_round2_messages(self, hypotheses, cross_check_results, num_hypotheses=None):
        if num_hypotheses is None:
            num_hypotheses = len(hypotheses)
        if len(self.conversation_history) >= 2:
            round1_system = self.conversation_history[0]
            round1_user = self.conversation_history[1]
        else:
            round1_system, round1_user = self.build_round1_messages(num_hypotheses=num_hypotheses)

        return [
            round1_system,
            round1_user,
            {"role": "assistant", "content": json.dumps(hypotheses, indent=2)},
            {
                "role": "user",
                "content": self.build_round2_user_prompt(
                    hypotheses=hypotheses,
                    cross_check_results=cross_check_results,
                    num_hypotheses=num_hypotheses,
                ),
            },
        ]

    def generate_initial(self, num_hypotheses=3, context=None):
        self.iteration = 1
        self.conversation_history = self.build_round1_messages(
            num_hypotheses=num_hypotheses,
            context=context,
        )

        hypotheses = self._call_and_parse()
        self.all_hypotheses.extend(hypotheses)
        return hypotheses

    def generate_with_feedback(self, hypotheses, cross_check_results, num_hypotheses=3):
        self.iteration += 1
        round2_messages = self.build_round2_messages(
            hypotheses=hypotheses,
            cross_check_results=cross_check_results,
            num_hypotheses=num_hypotheses,
        )
        self.conversation_history = self.conversation_history + round2_messages[2:]

        new_hypotheses = self._call_and_parse()
        self.all_hypotheses.extend(new_hypotheses)
        return new_hypotheses

    def _call_and_parse(self):
        if self.verbose:
            print(f"[Iteration {self.iteration}, Model: {self.model}]")

        response = get_client().chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=self.conversation_history,
        )

        if self.verbose:
            print(f"[Actual model: {response.model}]")
            if response.choices[0].finish_reason != "stop":
                print(f"[Warning: finish_reason={response.choices[0].finish_reason}]")

        raw_text = response.choices[0].message.content
        try:
            return _extract_json_array(raw_text)
        except (json.JSONDecodeError, ValueError) as err:
            print(f"JSON parse error: {err}")
            print(f"Raw response:\n{raw_text[:1000]}")
            return []

    def get_history_summary(self):
        return {
            "total_iterations": self.iteration,
            "total_hypotheses": len(self.all_hypotheses),
            "model": self.model,
            "use_oracle_fallback": self.use_oracle_fallback,
            "conversation_turns": len(self.conversation_history),
        }

    def get_all_hypotheses_df(self):
        df = pd.DataFrame(self.all_hypotheses)
        df["iteration"] = [i // 3 + 1 for i in range(len(self.all_hypotheses))]
        return df


def _print_oracle_values(results, run_label):
    for result in results:
        oracle_pred = result.get("oracle_prediction")
        if oracle_pred is None:
            continue
        print(
            f"[{run_label}] {result.get('hypothesis_name', 'Unnamed')}: "
            f"oracle={oracle_pred['predicted_productivity_mg_mL_hr']:.3f} mg/mL/hr"
        )


def _best_available_productivity(results):
    best_idx = None
    best_prod = None
    oracle_count = 0

    for idx, result in enumerate(results):
        oracle_pred = result.get("oracle_prediction")
        if oracle_pred is not None and oracle_pred.get("predicted_productivity_mg_mL_hr") is not None:
            prod = oracle_pred["predicted_productivity_mg_mL_hr"]
            oracle_count += 1
            if best_prod is None or prod > best_prod:
                best_prod = prod
                best_idx = idx

    return best_idx, best_prod, "oracle", oracle_count


def run_hypothesis_iterations(
    num_iterations=3,
    hypotheses_per_iteration=3,
    model="google/gemini-2.5-flash",
    context=None,
    use_oracle_fallback=True,
):
    iterator = TXTLHypothesisIterator(
        model=model,
        use_oracle_fallback=use_oracle_fallback,
    )
    all_results = []

    print(f"{'='*60}")
    print(f"ITERATION 1 of {num_iterations}")
    print(f"{'='*60}")
    hypotheses = iterator.generate_initial(hypotheses_per_iteration, context)
    if not hypotheses:
        print("No valid JSON hypotheses in iteration 1; skipping run.")
        return iterator, all_results
    results = cross_check_hypotheses(
        hypotheses,
        verbose=True,
        use_oracle_fallback=iterator.use_oracle_fallback,
    )
    _print_oracle_values(results, "iteration 1")
    all_results.append((hypotheses, results))

    for i in range(2, num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {i} of {num_iterations}")
        print(f"{'='*60}")
        new_hypotheses = iterator.generate_with_feedback(hypotheses, results, hypotheses_per_iteration)
        if not new_hypotheses:
            print(f"Iteration {i}: no valid JSON hypotheses; skipping this iteration.")
            continue
        hypotheses = new_hypotheses
        new_results = cross_check_hypotheses(
            hypotheses,
            verbose=True,
            use_oracle_fallback=iterator.use_oracle_fallback,
        )
        _print_oracle_values(new_results, f"iteration {i}")
        results = new_results
        all_results.append((hypotheses, new_results))

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Total hypotheses generated: {len(iterator.all_hypotheses)}")
    print(f"Conversation history length: {len(iterator.conversation_history)} messages")

    return iterator, all_results


def benchmark_free_models(
    num_iterations=3,
    hypotheses_per_iteration=2,
    context=None,
    use_oracle_fallback=True,
):
    from llm_client import FREE_MODELS
    models_to_test = FREE_MODELS

    all_iterators = {}
    all_cross_checks = {}
    benchmark_results = []

    for model_name, model_id in models_to_test.items():
        print(f"\n{'#'*60}")
        print(f"# TESTING MODEL: {model_name}")
        print(f"{'#'*60}")

        try:
            iterator = TXTLHypothesisIterator(
                model=model_id,
                verbose=True,
                use_oracle_fallback=use_oracle_fallback,
            )
            model_cross_checks = []
            best_exp_prod = None
            best_exp_hyp_idx = None
            best_exp_iteration = None
            best_source = None
            total_hyp_with_oracle = 0
            exp_prods_by_iter = []

            hypotheses = iterator.generate_initial(hypotheses_per_iteration, context)
            if not hypotheses:
                print("[SKIP] No valid JSON hypotheses in iteration 1; moving to next model.")
                benchmark_results.append(
                    {
                        "model": model_name,
                        "status": "skipped: json parse",
                        "total_hypotheses": 0,
                        "hypotheses_with_csv_data": 0,
                        "hypotheses_with_oracle_estimate": 0,
                        "best_experimental_productivity": None,
                        "best_productivity_source": None,
                        "best_hypothesis_name": None,
                        "iteration_of_best": None,
                        "hypothesis_num_of_best": None,
                        "iter_best_productivity": [],
                    }
                )
                continue
            results = cross_check_hypotheses(
                hypotheses,
                verbose=False,
                use_oracle_fallback=iterator.use_oracle_fallback,
            )
            _print_oracle_values(results, f"{model_name} iter 1")
            model_cross_checks.append(results)

            best_idx, best_prod, source, oracle_count = _best_available_productivity(results)
            total_hyp_with_oracle += oracle_count
            exp_prods_by_iter.append(best_prod)

            if best_idx is not None and (best_exp_prod is None or best_prod > best_exp_prod):
                best_exp_prod = best_prod
                best_exp_hyp_idx = best_idx
                best_exp_iteration = 1
                best_source = source

            for iter_num in range(2, num_iterations + 1):
                new_hypotheses = iterator.generate_with_feedback(hypotheses, results, hypotheses_per_iteration)
                if not new_hypotheses:
                    print(f"[SKIP] {model_name} iteration {iter_num}: no valid JSON hypotheses; skipping iteration.")
                    exp_prods_by_iter.append(None)
                    continue
                hypotheses = new_hypotheses
                new_results = cross_check_hypotheses(
                    hypotheses,
                    verbose=False,
                    use_oracle_fallback=iterator.use_oracle_fallback,
                )
                _print_oracle_values(new_results, f"{model_name} iter {iter_num}")
                model_cross_checks.append(new_results)

                best_idx, best_prod, source, oracle_count = _best_available_productivity(new_results)
                total_hyp_with_oracle += oracle_count
                exp_prods_by_iter.append(best_prod)
                results = new_results

                if best_idx is not None and (best_exp_prod is None or best_prod > best_exp_prod):
                    best_exp_prod = best_prod
                    best_exp_hyp_idx = (iter_num - 1) * hypotheses_per_iteration + best_idx
                    best_exp_iteration = iter_num
                    best_source = source

            all_iterators[model_name] = iterator
            all_cross_checks[model_name] = model_cross_checks

            all_hyp = iterator.all_hypotheses
            best_hyp_name = all_hyp[best_exp_hyp_idx].get("hypothesis_name") if best_exp_hyp_idx is not None else None

            benchmark_results.append(
                {
                    "model": model_name,
                    "status": "success",
                    "total_hypotheses": len(all_hyp),
                    "hypotheses_with_oracle_estimate": total_hyp_with_oracle,
                    "best_experimental_productivity": best_exp_prod,
                    "best_productivity_source": best_source,
                    "best_hypothesis_name": best_hyp_name,
                    "iteration_of_best": best_exp_iteration,
                    "hypothesis_num_of_best": best_exp_hyp_idx + 1 if best_exp_hyp_idx is not None else None,
                    "iter_best_productivity": exp_prods_by_iter,
                }
            )
            for iter_idx, iter_best in enumerate(exp_prods_by_iter, start=1):
                benchmark_results[-1][f"iter_{iter_idx}_best"] = iter_best

            print(f"\n[SUCCESS] Best oracle productivity: {best_exp_prod} mg/mL/hr")
            print(
                f"          ({total_hyp_with_oracle}/{len(all_hyp)} oracle estimates)"
            )

        except Exception as err:
            print(f"\n[FAILED] Error: {str(err)[:100]}")
            benchmark_results.append(
                {
                    "model": model_name,
                    "status": f"failed: {str(err)[:50]}",
                    "total_hypotheses": 0,
                    "hypotheses_with_oracle_estimate": 0,
                    "best_experimental_productivity": None,
                    "best_productivity_source": None,
                    "best_hypothesis_name": None,
                    "iteration_of_best": None,
                    "hypothesis_num_of_best": None,
                    "iter_best_productivity": [],
                }
            )

    results_df = pd.DataFrame(benchmark_results)

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print("NOTE: all productivity values are oracle predictions.")
    print(f"{'='*60}")

    return results_df, all_iterators, all_cross_checks
