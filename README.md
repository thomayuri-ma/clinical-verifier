# ClinicalVerifier: repaired research prototype

This is a separate repair of the supplied source archive. Nothing has been pushed to GitHub, deployed, or published to Zenodo. The original files are preserved in `legacy/` for traceability. Do not run those files as the repaired application.

## What this version does

It retrieves reference passages, asks an explicitly selected language model to compare a candidate statement with them, and returns the verdict with citations, exact evidence quotations, the raw response and execution metadata. Model failure produces an unavailable assessment requiring review. There is no keyword fallback and no generated guideline corpus.

Labels are `supported`, `uncertain`, and `contradicts`. Supported means that the selected model found support in the supplied excerpts; it does not mean clinically safe. Citation validation checks that quoted text exists, not that the model interpreted it correctly.

The optional consistency component measures variation across actual model responses. It is separate from the activation probing project. It has no calibrated clinical interpretation. Preset scores are rejected by the evaluator and HTTP endpoint. A contradiction is never downgraded because the model is consistent.

## What was actually verified here

Run `python run_checks.py` from this folder. The included report records 43 passing software tests. Tests use scripted responses or mocked HTTP transports where necessary. They establish software behaviour, not language model quality or clinical accuracy.

An attempted Groq evaluation is saved under `reports/model_evaluation_attempt/`. No API credentials were configured. All 12 cases recorded backend errors; classification metrics were deliberately withheld. No external model request or paid inference occurred in that attempt. The requested model name was not checked against a provider catalogue.

Dense retrieval and live embedding based sampling were not executed in this environment because their optional dependencies and model weights were unavailable. Dense retrieval code remains provisional until integration testing with installed dependencies. Provider adapters were checked with mocked responses, not live accounts.

## Requirements

Python 3.10 or later. The lexical workflow, HTTP server, regression tests, and model HTTP clients use the standard library.

To use dense embeddings, create an isolated Python environment and install `requirements.txt`. These are compatibility ranges, not a tested exact dependency lock. Save the actual environment with `python -m pip freeze > reports/dependencies.txt` for any model experiment. A first embedding run may download model weights.

## Reference material

`data/references.json` contains two short NICE excerpts checked against official pages on 17 October 2026:

* CG178 recommendation 1.3.6.9, antipsychotic loading doses.
* NG225 recommendation 1.6.1, predictive use of risk scales.

The collection is deliberately small and is not comprehensive guidance. Full source URLs, locations, scope notes, check date and text provenance are included. Review NICE licensing and notice of rights before expanding or redistributing a reference corpus. This package does not assert that a URL was automatically fetched at runtime; `checked_on` records preparation of this snapshot.

Do not infer that absent evidence makes a statement acceptable. The current retrievers do not have a clinically validated relevance threshold. Positive lexical overlap can be irrelevant, and dense cosine similarity is not an entailment score.

To add reference material, curate passage objects with unique IDs, exact text and source metadata in the same JSON schema. Keep qualifications and exceptions intact. Revalidate using:

```bash
python ingest.py data/references.json
```

This replaces the old automatic PDF/sample ingestion with explicit curated input. Automatic PDF ingestion is not provided in this revision.

## Single assessment

Choose a model actually available to your account. Set credentials locally through an environment variable or a secrets manager. Never paste keys into chat or commit them to the repository.

For Groq, configure `GROQ_API_KEY`. For OpenAI, configure `OPENAI_API_KEY`. Ollama requires a running local server and an installed model.

```bash
python main.py --corpus data/references.json --provider groq --model YOUR_AVAILABLE_MODEL --text "Do not use a loading dose of antipsychotic medication."
```

For local inference:

```bash
python main.py --provider ollama --model YOUR_INSTALLED_MODEL --text "Do not use a loading dose of antipsychotic medication."
```

Add `--retrieval dense` only after installing and testing the optional stack. Lexical retrieval is an explicit diagnostic baseline, not equivalent to the earlier MiniLM and FAISS architecture.

## Reproducible model evaluation

`data/development.json` has 12 newly constructed cases, including negation, missing context and unsupported topics. Labels are developer assigned relations to two excerpts. They have not been independently clinically reviewed and this is not a held out test.

```bash
python evaluate.py --provider groq --model YOUR_AVAILABLE_MODEL --corpus data/references.json --dataset data/development.json --retrieval lexical --output reports/new_model_run
```

Each output folder must be new. The evaluator writes a manifest with hashes of active Python files, the dataset and corpus, plus provider, requested model and retrieval configuration. Predictions are saved and flushed after each case, including prompts, responses, citations, errors, backend metadata and measured elapsed time. A nonzero exit code of 2 means the run had assessment errors and performance metrics were withheld. A run completing without execution errors is still not clinical validation.

The retrieval hit calculation is diagnostic: with only two passages and top k equal to three, perfect passage inclusion is trivial. Do not quote it as retrieval effectiveness.

The old 30 examples remain in `data/legacy_cases_without_scores.json` for traceability. Their old labels are not independently validated. Their simulated scores are excluded. Do not compare a rerun against the historical claimed numbers without documenting all changes.

## Measured consistency

The default evaluation omits consistency. To use `--consistency measured`, every case must contain its actual `generation_prompt`, and embedding dependencies must be installed. Constructed clinical statements do not have original generation prompts, so the included development set cannot establish generator inconsistency.

The sampler records its prompt, sampled answers, provider responses and embedding model ID. It uses its documented JSON answering system prompt and temperature 0.8; it does not claim to reproduce an unknown original generation configuration. Five additional generation calls are made by default. Errors remain visible; no mock answers are substituted.

Compare the judge only baseline and measured consistency extension on independently reviewed examples. Do not choose thresholds using final test labels. Response agreement does not prove truth.

## Local research interface

```bash
python api.py --provider ollama --model YOUR_INSTALLED_MODEL --port 5000
```

Open http://127.0.0.1:5000. This is a local development server, not a public deployment. The interface uses text rendering for results. It accepts only a text field; it does not accept user supplied consistency scores or filesystem ingestion requests. No request text is logged by the server. Evaluation records intentionally contain submitted statements, so use constructed or appropriately deidentified material.

`GET /health` reports process readiness and configuration, not model connectivity. `POST /verify` returns 503 for backend failure and 502 for invalid model output. Unavailable or uncertain assessments require review.

## Remaining scientific work

1. Expand and independently review the reference collection.
2. Obtain independent annotations with explicit context and uncertainty rules.
3. Create separate development and held out evaluation sets.
4. Run and record actual models with exact dependency and weight revisions.
5. Compare retrieval, judge and optional consistency ablations.
6. Evaluate calibration, retrieval relevance, semantic evidence correctness, model failures and missing evidence.
7. Measure costs and latency under the intended configuration.

No stronger clinical performance claim can be restored from passing software tests alone. The old 73.3% accuracy and 95.2% F1 figures have not been reproduced.
