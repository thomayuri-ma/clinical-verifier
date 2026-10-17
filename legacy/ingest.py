"""
Day 1 — Guideline Ingestion Pipeline
=====================================
Downloads / reads NICE mental health guideline PDFs, chunks them into
paragraphs, embeds with sentence-transformers, and stores in a FAISS index.

Usage:
    python src/ingest.py --pdf_dir guidelines/ --index_path data/guidelines.index

Requirements:
    pip install sentence-transformers faiss-cpu pypdf langchain langchain-community
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"          # Fast, good quality, ~80 MB
CHUNK_SIZE   = 400                          # Characters per chunk
CHUNK_OVERLAP = 80
# ────────────────────────────────────────────────────────────────────────────


def load_pdfs(pdf_dir: str) -> List[Tuple[str, str]]:
    """
    Load all PDFs from *pdf_dir*.
    Returns list of (source_filename, page_text) tuples.
    """
    pdf_dir = Path(pdf_dir)
    docs: List[Tuple[str, str]] = []

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[ingest] No PDFs found in {pdf_dir}. "
              "Download NICE guidelines to that folder and re-run.")
        print("[ingest] Using built-in sample guidelines for demonstration.")
        return _sample_guidelines()

    for pdf_path in pdf_files:
        print(f"[ingest] Loading {pdf_path.name} …")
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        for page in pages:
            docs.append((pdf_path.name, page.page_content))

    print(f"[ingest] Loaded {len(docs)} pages from {len(pdf_files)} PDFs.")
    return docs


def _sample_guidelines() -> List[Tuple[str, str]]:
    """
    Built-in sample NICE guideline excerpts so the project runs
    immediately without downloading PDFs.
    """
    return [
        ("NICE_CG90_Depression.txt",
         "Depression in adults: recognition and management (CG90). "
         "Antidepressants are not recommended for the initial treatment of mild depression. "
         "For mild to moderate depression, consider: guided self-help, computerised CBT, "
         "structured group physical activity, or group problem-solving. "
         "If these are ineffective after 2–3 months, consider antidepressants."),

        ("NICE_CG90_Depression.txt",
         "Suicide risk assessment: Ask directly about suicidal ideation and intent. "
         "Do not avoid asking about suicide as this does not increase risk. "
         "Assess for hopelessness, previous attempts, substance misuse, and social isolation. "
         "Patients with active suicidal intent should be referred urgently to specialist services."),

        ("NICE_CG90_Depression.txt",
         "Antidepressant treatment: SSRIs are first-line pharmacological treatment for depression. "
         "Start at the lowest effective dose and titrate upward. "
         "Do not abruptly discontinue antidepressants; taper gradually to minimise withdrawal effects. "
         "Monitor for suicidal ideation in the first few weeks of treatment, especially in under-30s."),

        ("NICE_NG185_SelfHarm.txt",
         "Self-harm: assessment, management and preventing recurrence (NG185). "
         "Do not use the term 'unsuccessful suicide attempt' as it is stigmatising. "
         "Treat all episodes of self-harm with the same seriousness regardless of apparent intent. "
         "Conduct a full psychosocial assessment before discharge. "
         "Do not discharge patients without a safety plan."),

        ("NICE_NG185_SelfHarm.txt",
         "Following self-harm, offer a follow-up appointment within 48 hours. "
         "Means restriction counselling: advise on reducing access to methods of self-harm. "
         "Involve family or carers with patient consent. "
         "Consider DBT (Dialectical Behaviour Therapy) for recurrent self-harm."),

        ("NICE_CG178_Psychosis.txt",
         "Psychosis and schizophrenia in adults (CG178). "
         "Offer antipsychotic medication to people with an acute episode of psychosis. "
         "Do not use loading doses of antipsychotics. "
         "Clozapine should be considered for treatment-resistant schizophrenia after two adequate trials of other antipsychotics. "
         "Monitor metabolic parameters (weight, glucose, lipids) in all patients on antipsychotics."),

        ("NICE_CG178_Psychosis.txt",
         "Early intervention: Refer people with a first episode of psychosis urgently to an early intervention in psychosis (EIP) service. "
         "Do not delay treatment while waiting for a definitive diagnosis. "
         "Offer cognitive behavioural therapy (CBT) for psychosis to all patients. "
         "Family intervention should be offered to families of people with schizophrenia."),

        ("NICE_NG116_Bipolar.txt",
         "Bipolar disorder: assessment and management (NG116). "
         "Do not offer antidepressants alone to people with bipolar disorder as this may precipitate mania. "
         "Lithium is first-line for long-term treatment of bipolar disorder. "
         "Monitor lithium levels, renal function, and thyroid function every 6 months. "
         "Lithium has a narrow therapeutic index; toxicity can be fatal."),

        ("NICE_NG116_Bipolar.txt",
         "Mania: Offer an antipsychotic for acute mania. "
         "If the patient is taking an antidepressant, consider stopping it. "
         "Benzodiazepines may be used short-term for agitation or sleep disturbance. "
         "Hospitalisation may be required if the patient poses risk to self or others."),

        ("NICE_CG53_EatingDisorders.txt",
         "Eating disorders (CG53): recognition and treatment. "
         "For anorexia nervosa, medical stabilisation takes priority over psychological treatment. "
         "Do not use BMI alone as a measure of risk in anorexia; consider rate of weight loss and clinical signs. "
         "Refeeding syndrome is a serious risk; monitor electrolytes closely during nutritional rehabilitation. "
         "Compulsory treatment under the Mental Health Act may be necessary in life-threatening cases."),

        ("NICE_CG53_EatingDisorders.txt",
         "Bulimia nervosa: Offer CBT-ED as first-line treatment. "
         "Antidepressants (SSRIs, particularly fluoxetine at 60mg) may be used as adjunctive treatment. "
         "Do not routinely use antipsychotics in the treatment of bulimia nervosa. "
         "Monitor for dental erosion, electrolyte disturbances (especially hypokalaemia), and oesophageal tears."),

        ("NICE_CG42_Dementia.txt",
         "Dementia: diagnosis and assessment (CG42). "
         "Do not use antipsychotics routinely for behavioural and psychological symptoms of dementia (BPSD). "
         "If antipsychotics are necessary, use the lowest effective dose for the shortest possible time. "
         "Antipsychotic use in dementia is associated with increased risk of stroke and death. "
         "Non-pharmacological approaches should be tried first for BPSD."),

        ("WHO_mhGAP_2023.txt",
         "WHO mhGAP Intervention Guide 2023. "
         "Depression: Counsel on maintaining activities and social contacts. "
         "Avoid prescribing benzodiazepines as first-line treatment for depression. "
         "For moderate-severe depression, offer psychological treatment AND antidepressant if available. "
         "Follow up within 2 weeks of starting treatment."),

        ("WHO_mhGAP_2023.txt",
         "Psychosis: Do not leave a person with acute psychosis alone. "
         "Restrain only as a last resort and only with trained personnel. "
         "Avoid using antipsychotic polypharmacy without specialist review. "
         "Support the person to remain in the community where safe."),

        ("WHO_mhGAP_2023.txt",
         "Suicide and self-harm: Means restriction is one of the most effective suicide prevention strategies. "
         "Do not promise confidentiality when there is immediate risk to life. "
         "Conduct a risk assessment using a structured tool. "
         "Connect the person with community support and schedule follow-up within 72 hours."),
    ]


def chunk_documents(docs: List[Tuple[str, str]]) -> List[dict]:
    """Split raw text into overlapping chunks. Returns list of chunk dicts."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    for source, text in docs:
        for chunk in splitter.split_text(text):
            chunk = chunk.strip()
            if len(chunk) > 60:          # Skip trivially short fragments
                chunks.append({"source": source, "text": chunk})
    print(f"[ingest] Produced {len(chunks)} chunks.")
    return chunks


def embed_chunks(chunks: List[dict]) -> np.ndarray:
    """Embed all chunks with sentence-transformers. Returns (N, D) float32 array."""
    print(f"[ingest] Loading embedding model '{EMBED_MODEL}' …")
    model  = SentenceTransformer(EMBED_MODEL)
    texts  = [c["text"] for c in chunks]
    print(f"[ingest] Embedding {len(texts)} chunks …")
    embeds = model.encode(texts, batch_size=64, show_progress_bar=True,
                          convert_to_numpy=True)
    return embeds.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an inner-product (cosine after normalisation) FAISS index."""
    faiss.normalize_L2(embeddings)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[ingest] FAISS index built: {index.ntotal} vectors, dim={dim}.")
    return index


def save_artifacts(index: faiss.IndexFlatIP,
                   chunks: List[dict],
                   index_path: str) -> None:
    """Persist the FAISS index and chunk metadata to disk."""
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    meta_path = index_path.with_suffix(".meta.pkl")
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[ingest] Saved index → {index_path}")
    print(f"[ingest] Saved metadata → {meta_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def run(pdf_dir: str = "guidelines/", index_path: str = "data/guidelines.index"):
    docs   = load_pdfs(pdf_dir)
    chunks = chunk_documents(docs)
    embeds = embed_chunks(chunks)
    index  = build_faiss_index(embeds)
    save_artifacts(index, chunks, index_path)
    print("[ingest] ✓ Done — index ready for retrieval.")
    return index, chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest NICE guidelines into FAISS")
    parser.add_argument("--pdf_dir",    default="guidelines/")
    parser.add_argument("--index_path", default="data/guidelines.index")
    args = parser.parse_args()
    run(args.pdf_dir, args.index_path)
