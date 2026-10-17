# Repair report

Prepared 17 October 2026 from the user supplied ClinicalVerifier archive.

## Changes made

* Replaced broken `src` imports with runnable root module entry points.
* Removed silent keyword fallback and canned sample responses from active code.
* Added explicit OpenAI, Groq and Ollama clients. No provider is silently substituted.
* Added evidence ID and quotation validation. Unsupported output becomes uncertain with an error status.
* Replaced `safe` with `supported` and separated uncertainty from contradiction review.
* Removed simulated scores from active datasets and prohibited client supplied scores in the API.
* Added actual sampling machinery with sample records and explicit embedding identity.
* Replaced incorrect built in clinical summaries with a small, source checked two excerpt collection. Old summaries are quarantined in `legacy/`.
* Added per case predictions, manifests, source and input hashes, explicit failures, and a metric withholding policy.
* Replaced the old deployment oriented demo with a local research interface and server.
* Added 43 regression tests covering parsing, citations, scores, import paths, failure propagation, metric arithmetic, HTTP validation and mocked provider transports.

## Results obtained

43 tests passed. See `reports/software_tests.json` and the complete text log.

Attempted development evaluation: 12 cases, 12 backend errors due to missing Groq credentials. No model calls occurred. Accuracy, precision, recall and F1 are not reported for this run. The pipeline correctly saved failure records and withheld classification metrics.

The reported retrieval inclusion of 1.0 covers 11 cases with expected evidence IDs. There are only two indexed excerpts and top k is three, making this inclusion uninformative about useful retrieval. It must not be presented as a scientific success result.

## Scope of verification

Parser and HTTP checks are software checks. Negation regression tests show that no old keyword heuristic overrides a supplied model verdict; they do not show that any real model understands negation.

The provider request formats and failure handling passed mocked transport tests. No live provider compatibility test was possible here. The dense FAISS pathway and live sentence embeddings have not been integration tested. The original source archive does not supply a reproducible historical experiment, and this package does not recreate one.

## Implications for the manuscript and CV

The code has changed, so manuscript methods must describe this version and its current limitations. Do not say that the old results were reproduced. Do not claim Groq inference was performed in this session. Do not call this component activation probing.

A defensible current project description is: Developed a research prototype combining reference retrieval and language model assessment to examine potential contradictions in generated mental health text. Added traceable evidence, explicit failure handling and a reproducible evaluation workflow. Software regression checks passed; independent model and clinical validation remain pending.

GitHub, the live demonstration and Zenodo have not been changed. Publishing this package would require replacing the hosted implementation and aligning the preprint methods with the actual model run when available.
