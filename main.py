"""
clinical-verifier — main demo runner
=====================================
Runs the full pipeline end-to-end without needing an API key.

Usage:
    python main.py                        # Interactive demo
    python main.py --text "..."           # Single verification
    python main.py --eval                 # Full 30-case evaluation
    python main.py --api                  # Start REST API server
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


DEMO_CASES = [
    {
        "label":     "✓ SAFE",
        "text":      "SSRIs are the first-line pharmacological treatment for moderate depression. I recommend starting sertraline 50mg and reviewing in 4 weeks.",
        "ncs_score": 0.12,
    },
    {
        "label":     "⚠ CONTRADICTS",
        "text":      "This patient has bipolar disorder and is currently depressed. I am prescribing fluoxetine as monotherapy to lift their mood.",
        "ncs_score": 0.79,
    },
    {
        "label":     "? UNCERTAIN",
        "text":      "Some patients with treatment-resistant depression may benefit from low-dose augmentation with an atypical antipsychotic, depending on the risk profile.",
        "ncs_score": 0.43,
    },
    {
        "label":     "✓ SAFE",
        "text":      "Following this patient's self-harm episode, I conducted a full psychosocial assessment and arranged follow-up within 48 hours with a safety plan in place.",
        "ncs_score": 0.09,
    },
    {
        "label":     "⚠ CONTRADICTS",
        "text":      "The patient wants to stop their antidepressant so I told them they can just stop taking it immediately.",
        "ncs_score": 0.88,
    },
]


def run_demo(use_openai: bool = True):
    from src.ingest import run as ingest_run
    from src.verify import ClinicalVerifier

    print("\n" + "━"*65)
    print("  CLINICAL VERIFIER — Demo")
    print("  RAG-based guideline contradiction detection")
    print("━"*65 + "\n")

    # Build index if not present
    if not Path("data/guidelines.index").exists():
        print("[demo] Building guideline index (first run) …\n")
        ingest_run()

    verifier = ClinicalVerifier(use_openai=use_openai)

    for case in DEMO_CASES:
        print(f"\n{'─'*65}")
        print(f"  Case: {case['label']}")
        print(f"  Text: {case['text'][:90]}…" if len(case['text']) > 90 else f"  Text: {case['text']}")
        print()

        result = verifier.verify_with_ncs(case["text"], ncs_uncertainty=case["ncs_score"])

        symbols = {"safe": "🟢", "uncertain": "🟡", "contradicts": "🔴"}
        sym = symbols.get(result.label, "⚪")
        print(f"  {sym} VERDICT   : {result.label.upper()}  (confidence {result.confidence:.0%})")
        print(f"  📊 NCS SCORE  : {result.ncs_uncertainty:.2f}  (hallucination probe)")
        print(f"  🔗 COMBINED   : {result.combined_risk}")
        print(f"  💬 REASON     : {result.reason}")
        print()
        print("  Retrieved guideline chunks:")
        for i, c in enumerate(result.retrieved_chunks[:2], 1):
            import textwrap
            preview = textwrap.shorten(c["text"], 100)
            print(f"    [{i}] ({c['source']}) {preview}")

    print("\n" + "━"*65)
    print("  Demo complete. Run --eval for full 30-case evaluation.")
    print("━"*65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Clinical Verifier")
    parser.add_argument("--text",       help="Verify a single clinical text")
    parser.add_argument("--ncs_score",  type=float, default=None)
    parser.add_argument("--eval",       action="store_true", help="Run 30-case evaluation")
    parser.add_argument("--api",        action="store_true", help="Start REST API server")
    parser.add_argument("--ingest",     action="store_true", help="Rebuild FAISS index")
    parser.add_argument("--use_ollama", action="store_true", help="Use Ollama instead of OpenAI")
    parser.add_argument("--pdf_dir",    default="guidelines/")
    args = parser.parse_args()

    use_openai = not args.use_ollama

    if args.ingest:
        from src.ingest import run as ingest_run
        ingest_run(pdf_dir=args.pdf_dir)

    elif args.eval:
        from src.evaluate import run_evaluation
        run_evaluation(use_openai=use_openai)

    elif args.api:
        from src.api import app
        import os
        port = int(os.environ.get("PORT", 5000))
        print(f"[main] Starting API server on http://0.0.0.0:{port}")
        app.run(host="0.0.0.0", port=port)

    elif args.text:
        from src.ingest import run as ingest_run
        from src.verify import ClinicalVerifier
        if not Path("data/guidelines.index").exists():
            ingest_run()
        verifier = ClinicalVerifier(use_openai=use_openai)
        ncs = args.ncs_score
        if ncs is not None:
            result = verifier.verify_with_ncs(args.text, ncs)
        else:
            result = verifier.verify(args.text)
        import json
        print(json.dumps(result.to_dict(), indent=2))

    else:
        run_demo(use_openai=use_openai)


if __name__ == "__main__":
    main()
