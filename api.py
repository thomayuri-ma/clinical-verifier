"""
API Server — exposes the clinical verifier as a REST endpoint.

Endpoints:
    POST /verify          { "text": "...", "ncs_score": 0.5 }  → VerificationResult
    POST /ingest          { "pdf_dir": "guidelines/" }          → rebuild index
    GET  /health                                                 → {"status": "ok"}

Usage:
    pip install flask
    python src/api.py

    # Or with gunicorn:
    gunicorn src.api:app --bind 0.0.0.0:8000
"""

import os
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request
from src.verify import ClinicalVerifier

app = Flask(__name__)

# Lazy-initialised singleton verifier
_verifier = None


def get_verifier() -> ClinicalVerifier:
    global _verifier
    if _verifier is None:
        use_openai = os.environ.get("USE_OLLAMA", "").lower() not in ("1", "true")
        _verifier  = ClinicalVerifier(
            index_path  = os.environ.get("INDEX_PATH", "data/guidelines.index"),
            use_openai  = use_openai,
            ollama_model = os.environ.get("OLLAMA_MODEL", "llama3"),
        )
    return _verifier


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "vectors": get_verifier().index.ntotal})


@app.route("/verify", methods=["POST"])
def verify():
    body = request.get_json(force=True, silent=True) or {}
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    ncs = body.get("ncs_score")
    try:
        ncs = float(ncs) if ncs is not None else None
    except (TypeError, ValueError):
        return jsonify({"error": "'ncs_score' must be a float"}), 400

    verifier = get_verifier()
    if ncs is not None:
        result = verifier.verify_with_ncs(text, ncs_uncertainty=ncs)
    else:
        result = verifier.verify(text)

    return jsonify(result.to_dict())


@app.route("/ingest", methods=["POST"])
def ingest():
    from src.ingest import run as ingest_run
    global _verifier
    body    = request.get_json(force=True, silent=True) or {}
    pdf_dir = body.get("pdf_dir", "guidelines/")
    idx_path = os.environ.get("INDEX_PATH", "data/guidelines.index")
    ingest_run(pdf_dir=pdf_dir, index_path=idx_path)
    _verifier = None          # Force reload on next request
    return jsonify({"status": "index rebuilt", "index_path": idx_path})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true")
    print(f"[api] Starting on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
