"""
Day 3 — NCS (Neighbourhood Consistency Scoring) Hallucination Probe
=====================================================================
Estimates how "uncertain" or "hallucinated" an LLM output is by measuring
semantic self-consistency across multiple re-samplings of the same prompt.

The intuition: if a model is confident about something true, regenerating
the response multiple times yields semantically similar outputs. If it's
hallucinating, the outputs scatter in embedding space.

NCS score = 1 - mean_pairwise_cosine_similarity of N samples
          → 0.0  = perfectly consistent (low hallucination risk)
          → 1.0  = maximally inconsistent (high hallucination risk)

Usage:
    from src.ncs_probe import NCSProbe
    probe  = NCSProbe()
    result = probe.score(
        system_prompt="You are a mental health assistant.",
        user_message="What is the first-line treatment for bipolar depression?",
        n_samples=5
    )
    print(result.ncs_score)   # e.g. 0.23 (low uncertainty)
"""

import os
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
N_SAMPLES_DEFAULT = 5


@dataclass
class NCSResult:
    ncs_score:       float           # 0 = consistent, 1 = inconsistent
    risk_label:      str             # "low" | "medium" | "high"
    samples:         List[str] = field(default_factory=list)
    pairwise_sims:   List[float] = field(default_factory=list)
    mean_similarity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ncs_score":       self.ncs_score,
            "risk_label":      self.risk_label,
            "mean_similarity": self.mean_similarity,
            "n_samples":       len(self.samples),
        }


class NCSProbe:
    """
    Neighbourhood Consistency Scoring probe.

    Works with OpenAI, Ollama, or a mock sampler (offline demo).
    """

    def __init__(
        self,
        embed_model: str  = EMBED_MODEL,
        use_openai: bool  = True,
        ollama_model: str = "llama3",
        judge_model: str  = "gpt-3.5-turbo",
        temperature: float = 0.8,     # Higher = more variance = more honest NCS
    ):
        self.use_openai   = use_openai
        self.ollama_model = ollama_model
        self.judge_model  = judge_model
        self.temperature  = temperature

        print("[ncs] Loading embedding model …")
        self.embedder = SentenceTransformer(embed_model)

    # ── Sampling ──────────────────────────────────────────────────────────

    def _sample_openai(self, system: str, user: str, n: int) -> List[str]:
        """Draw n independent samples from OpenAI."""
        try:
            from openai import OpenAI
            client  = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            samples = []
            for _ in range(n):
                resp = client.chat.completions.create(
                    model       = self.judge_model,
                    messages    = [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature = self.temperature,
                    max_tokens  = 200,
                )
                samples.append(resp.choices[0].message.content.strip())
            return samples
        except Exception as e:
            print(f"[ncs] OpenAI sampling failed: {e}. Using mock sampler.")
            return self._mock_samples(user, n)

    def _sample_ollama(self, system: str, user: str, n: int) -> List[str]:
        """Draw n independent samples from Ollama."""
        try:
            import requests
            samples = []
            prompt  = f"{system}\n\nUser: {user}\nAssistant:"
            for _ in range(n):
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": self.ollama_model, "prompt": prompt,
                          "stream": False, "options": {"temperature": self.temperature}},
                    timeout=60,
                )
                samples.append(resp.json()["response"].strip())
            return samples
        except Exception as e:
            print(f"[ncs] Ollama sampling failed: {e}. Using mock sampler.")
            return self._mock_samples(user, n)

    def _mock_samples(self, query: str, n: int) -> List[str]:
        """
        Offline mock: generates semantically similar responses for known
        high-confidence queries, scattered responses for uncertain ones.

        This is purely for demonstration without an API key.
        """
        KNOWN_SAFE = {
            "ssri": [
                "SSRIs are first-line antidepressants for depression. Common examples include fluoxetine and sertraline.",
                "For depression, SSRIs such as sertraline or fluoxetine are typically the first pharmacological choice.",
                "NICE recommends SSRIs as initial antidepressant therapy. Fluoxetine is commonly used.",
                "SSRIs are preferred first-line antidepressants for unipolar depression per NICE CG90.",
                "The standard first-line antidepressants are SSRIs — fluoxetine, sertraline, or citalopram.",
            ],
            "cbt": [
                "CBT is recommended for depression and anxiety disorders. It typically runs 12–20 sessions.",
                "Cognitive behavioural therapy (CBT) is an evidence-based psychological treatment for depression.",
                "CBT is NICE-recommended for depression. Sessions usually last 12–16 weeks.",
                "CBT for depression involves identifying and challenging negative thought patterns.",
                "NICE recommends CBT as a first-line psychological treatment for moderate depression.",
            ],
        }
        UNCERTAIN = [
            "This treatment approach may work for some patients depending on their profile and history.",
            "There are several options available; the choice depends on individual patient factors.",
            "Results can vary significantly based on the clinical context and comorbidities.",
            "Different guidelines offer different recommendations on this specific clinical question.",
            "Evidence is mixed; clinician judgment should guide the decision in this case.",
        ]

        query_lower = query.lower()
        for key, responses in KNOWN_SAFE.items():
            if key in query_lower:
                return responses[:n]

        # If query contains contradiction signals, return scattered responses
        contradiction_signals = ["stop abruptly", "no monitoring", "loading dose",
                                  "antidepressant alone bipolar", "antipsychotic routine dementia"]
        if any(s in query_lower for s in contradiction_signals):
            import random
            random.shuffle(UNCERTAIN)
            return UNCERTAIN[:n]

        return UNCERTAIN[:n]

    # ── NCS Calculation ───────────────────────────────────────────────────

    def _compute_ncs(self, samples: List[str]) -> tuple:
        """
        Embed samples, compute pairwise cosine similarity, return
        (ncs_score, mean_similarity, pairwise_sims).
        """
        embeddings = self.embedder.encode(samples, convert_to_numpy=True)
        # Normalise for cosine
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)

        n    = len(embeddings)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                sims.append(sim)

        mean_sim  = statistics.mean(sims) if sims else 1.0
        ncs_score = max(0.0, min(1.0, 1.0 - mean_sim))
        return ncs_score, mean_sim, sims

    def _risk_label(self, ncs: float) -> str:
        if ncs < 0.25:
            return "low"
        elif ncs < 0.55:
            return "medium"
        else:
            return "high"

    # ── Main API ──────────────────────────────────────────────────────────

    def score(
        self,
        user_message:  str,
        system_prompt: str = "You are a helpful mental health assistant. Be concise.",
        n_samples:     int = N_SAMPLES_DEFAULT,
    ) -> NCSResult:
        """
        Draw n_samples from the LLM for (system_prompt, user_message),
        compute NCS, return NCSResult.
        """
        print(f"[ncs] Drawing {n_samples} samples …")
        if self.use_openai:
            samples = self._sample_openai(system_prompt, user_message, n_samples)
        else:
            samples = self._sample_ollama(system_prompt, user_message, n_samples)

        ncs_score, mean_sim, pairwise_sims = self._compute_ncs(samples)

        return NCSResult(
            ncs_score       = ncs_score,
            risk_label      = self._risk_label(ncs_score),
            samples         = samples,
            pairwise_sims   = pairwise_sims,
            mean_similarity = mean_sim,
        )


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, textwrap

    parser = argparse.ArgumentParser(description="NCS hallucination probe")
    parser.add_argument("--query",      required=True, help="Clinical query to probe")
    parser.add_argument("--system",     default="You are a mental health clinical assistant.")
    parser.add_argument("--n_samples",  type=int, default=5)
    parser.add_argument("--use_ollama", action="store_true")
    args = parser.parse_args()

    probe  = NCSProbe(use_openai=not args.use_ollama)
    result = probe.score(args.query, args.system, args.n_samples)

    print("\n" + "="*60)
    print(f"  NCS SCORE  : {result.ncs_score:.3f}  ({result.risk_label.upper()} uncertainty)")
    print(f"  MEAN SIM   : {result.mean_similarity:.3f}")
    print("="*60)
    print("\nSampled responses:")
    for i, s in enumerate(result.samples, 1):
        print(f"\n  [{i}] {textwrap.shorten(s, 120)}")
