"""Measured response variation, not a calibrated hallucination probability."""
import hashlib, json, math
from verify import digest

def cosine_variation(vectors):
    if len(vectors) < 2:
        raise ValueError('At least two samples required')
    dim = len(vectors[0])
    if not dim or any((len(v) != dim for v in vectors)):
        raise ValueError('Inconsistent embedding dimensions')
    units = []
    for v in vectors:
        if any((not math.isfinite(float(x)) for x in v)):
            raise ValueError('Nonfinite embedding')
        norm = math.sqrt(sum((float(x) ** 2 for x in v)))
        if norm == 0:
            raise ValueError('Zero embedding')
        units.append([float(x) / norm for x in v])
    sims = [max(-1.0, min(1.0, sum((x * y for x, y in zip(a, b))))) for i, a in enumerate(units) for b in units[i + 1:]]
    mean = sum(sims) / len(sims)
    return dict(score=max(0.0, min(1.0, 1.0 - mean)), mean_similarity=mean, pairwise_similarities=sims)

def measure_consistency(candidate, generation_prompt, client, n_samples=5, embedding_model='sentence-transformers/all-MiniLM-L6-v2', encoder=None):
    if type(n_samples) is not int or n_samples < 2:
        raise ValueError('At least two samples required')
    if not generation_prompt.strip():
        raise ValueError('Original generation prompt required')
    if encoder is None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(embedding_model)
    system = 'Respond to the research prompt. Return JSON with one field called answer containing your response.'
    samples, metadata = ([], [])
    for i in range(n_samples):
        raw, meta = client.complete(system, generation_prompt, temperature=0.8, seed=42 + i)
        answer = json.loads(raw).get('answer')
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('Invalid sample; no mock samples substituted')
        samples.append(answer)
        metadata.append(meta)
    return dict(cosine_variation(encoder.encode(samples)), origin='measured', n_samples=n_samples, samples=samples, candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(), generation_prompt=generation_prompt, generation_prompt_sha256=digest(generation_prompt), embedding_model=embedding_model, backends=metadata, interpretation='Response variation only; not factuality or clinical risk probability')
