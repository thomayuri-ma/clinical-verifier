"""
Day 3 — Evaluation Suite
=========================
Runs the verification pipeline on 30 labelled test cases and reports:
  - Precision / Recall / F1 per label
  - Confusion matrix
  - Combined NCS+verify accuracy

Usage:
    python src/evaluate.py --index_path data/guidelines.index
    python src/evaluate.py --use_ollama
    python src/evaluate.py --json_out eval/results.json
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# ── Ground-truth test cases ───────────────────────────────────────────────────
# Format: (clinical_text, ground_truth_label, ncs_uncertainty)
# Labels: "safe" | "uncertain" | "contradicts"

TEST_CASES = [
    # ── SAFE cases ───────────────────────────────────────────────────────────
    (
        "For mild depression, I recommend starting with guided self-help and monitoring the patient's progress over 4–6 weeks before considering medication.",
        "safe", 0.15
    ),
    (
        "SSRIs such as sertraline or fluoxetine are first-line pharmacological treatments for major depressive disorder.",
        "safe", 0.12
    ),
    (
        "CBT should be offered to this patient with GAD. We can arrange 12–20 sessions with a qualified therapist.",
        "safe", 0.18
    ),
    (
        "When a patient discloses active suicidal ideation, it is essential to conduct a full risk assessment and not leave them alone.",
        "safe", 0.10
    ),
    (
        "Lithium is the first-line mood stabiliser for bipolar disorder. We should monitor serum levels every 3–6 months along with renal and thyroid function.",
        "safe", 0.14
    ),
    (
        "For this patient with early-onset psychosis, I recommend an urgent referral to the Early Intervention in Psychosis service.",
        "safe", 0.11
    ),
    (
        "In anorexia nervosa, medical stabilisation and nutritional rehabilitation take priority. We need to monitor electrolytes daily to prevent refeeding syndrome.",
        "safe", 0.19
    ),
    (
        "Non-pharmacological approaches like structured activities and reality orientation should be tried first for BPSD before considering antipsychotics.",
        "safe", 0.16
    ),
    (
        "The patient should not abruptly stop their SSRI. We will taper the dose gradually over 4 weeks to minimise discontinuation symptoms.",
        "safe", 0.09
    ),
    (
        "Asking patients directly about suicidal thoughts does not increase the risk of suicide; it is an essential part of the assessment.",
        "safe", 0.08
    ),

    # ── CONTRADICTS cases ─────────────────────────────────────────────────────
    (
        "This patient with bipolar depression is feeling low so I am starting them on fluoxetine as monotherapy to lift their mood.",
        "contradicts", 0.72
    ),
    (
        "The patient wants to stop their antidepressant, so I have advised them to simply stop taking it today.",
        "contradicts", 0.81
    ),
    (
        "Given the patient's agitated dementia, I am prescribing haloperidol long-term as it is safe and effective for managing BPSD.",
        "contradicts", 0.68
    ),
    (
        "I used a loading dose of 10mg haloperidol to quickly settle this first-episode psychosis patient.",
        "contradicts", 0.75
    ),
    (
        "Since this patient has a mild eating disorder, their BMI of 17.5 indicates the risk is low and no urgent intervention is needed.",
        "contradicts", 0.62
    ),
    (
        "I discharged the patient after self-harm without a formal psychosocial assessment as the wound was superficial.",
        "contradicts", 0.88
    ),
    (
        "As this schizophrenia patient hasn't responded to two antipsychotics, I've decided not to try clozapine as it's too risky.",
        "contradicts", 0.71
    ),
    (
        "Benzodiazepines are my go-to first-line treatment for this patient's depression — they work quickly and the patient prefers them.",
        "contradicts", 0.79
    ),
    (
        "There's no need to monitor lithium levels in stable long-term patients — it just causes unnecessary anxiety.",
        "contradicts", 0.84
    ),
    (
        "Since the patient self-harmed in a minor way, I recorded it as an unsuccessful suicide attempt in the notes.",
        "contradicts", 0.66
    ),

    # ── UNCERTAIN cases ───────────────────────────────────────────────────────
    (
        "Depending on the clinical picture, we might consider a low-dose antipsychotic adjunct for treatment-resistant depression.",
        "uncertain", 0.42
    ),
    (
        "Some patients with bipolar disorder may benefit from antidepressant augmentation if carefully monitored for mood switching.",
        "uncertain", 0.55
    ),
    (
        "The patient's risk appears to be moderate; I'll review them again in one week and reassess whether hospitalisation is needed.",
        "uncertain", 0.38
    ),
    (
        "For this patient with ADHD and comorbid anxiety, the treatment choice is complex and may need specialist input.",
        "uncertain", 0.47
    ),
    (
        "Given the patient's history of poor medication adherence, I am considering a long-acting injectable antipsychotic.",
        "uncertain", 0.31
    ),
    (
        "The evidence for omega-3 supplementation in depression is mixed, but it appears safe and the patient is keen to try it.",
        "uncertain", 0.43
    ),
    (
        "A short course of zopiclone may be appropriate here to break the cycle of insomnia, with careful monitoring.",
        "uncertain", 0.49
    ),
    (
        "We could consider group CBT or individual CBT depending on patient preference and service availability.",
        "uncertain", 0.28
    ),
    (
        "Mindfulness-based cognitive therapy may help this patient with recurrent depression, particularly for relapse prevention.",
        "uncertain", 0.35
    ),
    (
        "This patient might benefit from a structured exercise programme as an adjunct to medication for their moderate depression.",
        "uncertain", 0.29
    ),
]


# ── Evaluation logic ──────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    text:       str
    expected:   str
    ncs:        float
    predicted:  Optional[str]   = None
    confidence: float           = 0.0
    reason:     str             = ""
    combined:   Optional[str]   = None
    correct:    Optional[bool]  = None


def run_evaluation(
    index_path:  str  = "data/guidelines.index",
    use_openai:  bool = True,
    ollama_model: str = "llama3",
    json_out:    Optional[str] = None,
    verbose:     bool = True,
):
    # Import here to avoid circular dependency if run standalone
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.verify import ClinicalVerifier

    verifier = ClinicalVerifier(
        index_path  = index_path,
        use_openai  = use_openai,
        ollama_model = ollama_model,
    )

    cases: List[EvalCase] = []
    print(f"\n[eval] Running {len(TEST_CASES)} test cases …\n")

    for i, (text, expected, ncs) in enumerate(TEST_CASES, 1):
        print(f"  [{i:02d}/{len(TEST_CASES)}] {text[:70]}…")
        result = verifier.verify_with_ncs(text, ncs_uncertainty=ncs)
        correct = result.label == expected

        case = EvalCase(
            text       = text,
            expected   = expected,
            ncs        = ncs,
            predicted  = result.label,
            confidence = result.confidence,
            reason     = result.reason,
            combined   = result.combined_risk,
            correct    = correct,
        )
        cases.append(case)
        status = "✓" if correct else "✗"
        print(f"         {status} predicted={result.label}, expected={expected}, "
              f"conf={result.confidence:.0%}")

    _print_report(cases, verbose)

    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump([vars(c) for c in cases], f, indent=2)
        print(f"\n[eval] Results saved → {json_out}")

    return cases


def _print_report(cases: List[EvalCase], verbose: bool):
    labels     = ["safe", "uncertain", "contradicts"]
    total      = len(cases)
    n_correct  = sum(1 for c in cases if c.correct)
    accuracy   = n_correct / total

    print("\n" + "="*65)
    print(f"  OVERALL ACCURACY: {accuracy:.1%}  ({n_correct}/{total})")
    print("="*65)

    # Per-label metrics
    print(f"\n{'Label':<14} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-"*50)
    for label in labels:
        tp = sum(1 for c in cases if c.expected == label and c.predicted == label)
        fp = sum(1 for c in cases if c.expected != label and c.predicted == label)
        fn = sum(1 for c in cases if c.expected == label and c.predicted != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        print(f"  {label:<12} {prec:>6.1%} {rec:>6.1%} {f1:>6.1%}  {tp:>4} {fp:>4} {fn:>4}")

    # Confusion matrix
    print("\n  Confusion Matrix (rows=expected, cols=predicted):")
    print(f"             {'safe':>10} {'uncertain':>10} {'contradicts':>12}")
    for exp in labels:
        row = [sum(1 for c in cases if c.expected == exp and c.predicted == pred)
               for pred in labels]
        print(f"  {exp:<12} {row[0]:>10} {row[1]:>10} {row[2]:>12}")

    # NCS combined risk accuracy
    high_risk = [c for c in cases if c.combined and "HIGH" in c.combined]
    true_contradicts = [c for c in high_risk if c.expected == "contradicts"]
    if high_risk:
        precision_high = len(true_contradicts) / len(high_risk)
        print(f"\n  HIGH combined-risk cases: {len(high_risk)}")
        print(f"  True contradictions in HIGH: {len(true_contradicts)} "
              f"({precision_high:.1%} precision)")

    if verbose:
        errors = [c for c in cases if not c.correct]
        if errors:
            print(f"\n  Misclassified cases ({len(errors)}):")
            for c in errors:
                print(f"\n    Text    : {c.text[:90]}…")
                print(f"    Expected: {c.expected}  →  Predicted: {c.predicted}")
                print(f"    Reason  : {c.reason[:120]}")

    print("\n" + "="*65)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clinical verifier")
    parser.add_argument("--index_path",  default="data/guidelines.index")
    parser.add_argument("--use_ollama",  action="store_true")
    parser.add_argument("--json_out",    default="eval/results.json")
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    run_evaluation(
        index_path  = args.index_path,
        use_openai  = not args.use_ollama,
        json_out    = args.json_out,
        verbose     = not args.quiet,
    )
