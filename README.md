# Clinical Verifier
> RAG-based clinical guideline contradiction detection for LLM outputs

A system that takes an LLM's clinical output, retrieves the most relevant sections from NICE/WHO guidelines using vector search, then uses a judge LLM to determine whether the output contradicts those guidelines — returning a risk label of **safe**, **uncertain**, or **contradicts**.

Built as a research demo for the MSc AI in Healthcare (Manchester) programme.

---

## Motivation

Working at SerenMind AI, I noticed a model normalising a clinical risk indicator that contradicted NICE guidance — caught only by manual review. This project automates that catch.

It hits four themes central to the Manchester programme:
- **Retrieval-augmented reasoning** — FAISS over embedded NICE PDFs
- **Agentic AI** — multi-step orchestration: embed → retrieve → judge → fuse
- **Trustworthy biomedical AI** — structured risk labels with explanations
- **Tool-augmented LLMs** — LLM-as-judge with grounded context

---

## Architecture

```
Clinical LLM output
       │
       ▼
┌─────────────────────┐
│  sentence-transformers│  ← embed query
│  (all-MiniLM-L6-v2) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│     FAISS Index     │  ← top-3 guideline chunks
│  (NICE/WHO PDFs)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Judge LLM         │  ← GPT-3.5 or Llama3
│   (structured JSON) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   NCS Probe         │  ← hallucination uncertainty score
│   (Day-3 signal)    │
└─────────┬───────────┘
          │
          ▼
   { label, confidence, reason, combined_risk }
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the demo (no API key needed — uses heuristic judge + sample guidelines)
python main.py

# 3. Add your OpenAI key for the full LLM judge
export OPENAI_API_KEY=sk-...
python main.py

# 4. Verify a single clinical text
python main.py --text "Patients with bipolar disorder should take antidepressants alone."

# 5. Run the full 30-case evaluation
python main.py --eval

# 6. Use Ollama (Llama3) instead of OpenAI
ollama pull llama3
python main.py --use_ollama

# 7. Start the REST API
python main.py --api
curl -X POST http://localhost:5000/verify \
     -H 'Content-Type: application/json' \
     -d '{"text": "Stop antidepressants immediately.", "ncs_score": 0.75}'
```

---

## Day-by-Day Build Plan

### Day 1 — Ingest (`src/ingest.py`)
- Download 5–10 NICE mental health guideline PDFs to `guidelines/`
- Chunk → embed → store in FAISS index (`data/guidelines.index`)
- ~80 lines of Python; built-in sample guidelines for instant demo

```bash
# Download a NICE PDF, then:
python src/ingest.py --pdf_dir guidelines/ --index_path data/guidelines.index
```

### Day 2 — Verify (`src/verify.py`)
- Retrieve top-3 guideline chunks for any clinical text
- Judge LLM (GPT-3.5 or Llama3) returns JSON: `{label, confidence, reason}`
- Heuristic fallback works offline with no API key

```bash
python src/verify.py --text "Prescribe fluoxetine monotherapy for bipolar depression."
```

### Day 3 — NCS Probe + Evaluation (`src/ncs_probe.py`, `src/evaluate.py`)
- NCS (Neighbourhood Consistency Scoring): sample N responses, measure semantic variance
- High NCS + `contradicts` verdict = HIGH combined risk
- 30-case eval suite with precision/recall/F1 per label

```bash
python src/evaluate.py
```

---

## NICE Guidelines to Download

From [nice.org.uk](https://www.nice.org.uk/) — all free:

| Guideline | Code | Topic |
|-----------|------|-------|
| Depression in adults | CG90 | First-line treatment, antidepressants |
| Self-harm | NG185 | Assessment, discharge, safety planning |
| Psychosis & schizophrenia | CG178 | Antipsychotics, EIP referral |
| Bipolar disorder | NG116 | Lithium, antidepressant caution |
| Eating disorders | CG53 | Anorexia, refeeding syndrome |
| Dementia | CG42 | BPSD, antipsychotic caution |

Download PDFs, place in `guidelines/`, run `python main.py --ingest`.

---

## Project Structure

```
clinical-verifier/
├── main.py                  # Entry point / demo runner
├── requirements.txt
├── guidelines/              # Put NICE PDFs here
├── data/                    # FAISS index (auto-generated)
│   ├── guidelines.index
│   └── guidelines.meta.pkl
├── eval/                    # Evaluation results
│   └── results.json
└── src/
    ├── ingest.py            # Day 1: PDF → FAISS index
    ├── verify.py            # Day 2: RAG + judge LLM chain
    ├── ncs_probe.py         # Day 3: hallucination probe
    ├── evaluate.py          # Day 3: 30-case eval suite
    └── api.py               # REST API (Flask)
```

---

## Output Schema

```json
{
  "label":            "contradicts",
  "confidence":       0.87,
  "reason":           "Prescribing antidepressants as monotherapy for bipolar disorder risks precipitating a manic episode, which directly contradicts NICE NG116.",
  "retrieved_chunks": [
    {
      "source": "NICE_NG116_Bipolar.txt",
      "text":   "Do not offer antidepressants alone to people with bipolar disorder as this may precipitate mania.",
      "retrieval_score": 0.923
    }
  ],
  "ncs_uncertainty":  0.79,
  "combined_risk":    "HIGH — contradiction confirmed by both guideline check and NCS probe"
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI key for judge LLM |
| `JUDGE_MODEL` | `gpt-3.5-turbo` | OpenAI model to use |
| `USE_OLLAMA` | `0` | Set to `1` to use Ollama |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `INDEX_PATH` | `data/guidelines.index` | FAISS index location |
| `PORT` | `5000` | API server port |

---

## Limitations & Future Work

- **Not a medical device.** This is a research tool; verdicts should not be used for clinical decision-making without human oversight.
- **Retrieval quality** depends on chunk size and embedding model. Experiment with `CHUNK_SIZE` and `all-mpnet-base-v2`.
- **Judge bias** — LLM judges can be miscalibrated. Calibrate confidence scores on a held-out set.
- **NCS as hallucination proxy** — high variance ≠ hallucination; it can also reflect genuine clinical ambiguity.
- Future: fine-tune a dedicated verifier on clinical NLI datasets (MedNLI, NLI4CT).
```
