"""Reproducible development evaluation with per-case records and visible errors."""
import argparse, hashlib, json, platform, statistics, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from verify import ClinicalVerifier, LABELS, PROMPT_VERSION, digest
from ncs_probe import measure_consistency

def metrics(records):
    matrix = {a: {b: 0 for b in LABELS} for a in LABELS}
    for r in records:
        matrix[r['expected']][r['result']['label']] += 1
    total = len(records)
    correct = sum((matrix[x][x] for x in LABELS))
    per_label = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum((matrix[x][label] for x in LABELS if x != label))
        fn = sum((matrix[label][x] for x in LABELS if x != label))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per_label[label] = dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0, support=sum(matrix[label].values()))
    return dict(n=total, accuracy=correct / total if total else None, confusion_matrix=matrix, per_label=per_label)

def validate_dataset(data):
    if data.get('split') != 'development' or data.get('label_provenance') not in ('developer_assigned', 'independent_review'):
        raise ValueError('Explicit split and label provenance required')
    ids = set()
    for c in data.get('cases', []):
        if not isinstance(c, dict) or not isinstance(c.get('id'), str) or (not isinstance(c.get('text'), str)) or (not c['text'].strip()) or (c.get('expected') not in LABELS):
            raise ValueError('Invalid case schema')
        if c['id'] in ids:
            raise ValueError('Duplicate case ID')
        ids.add(c['id'])
        if any((k in c for k in ('ncs', 'ncs_score', 'ncs_uncertainty', 'consistency_score'))):
            raise ValueError('Preset consistency scores prohibited')
    if not ids:
        raise ValueError('Dataset empty')

def run_evaluation(corpus, dataset, output, provider, model, retrieval='lexical', consistency_mode='none', verifier=None, allow_fixtures=False):
    data = json.loads(Path(dataset).read_text())
    validate_dataset(data)
    verifier = verifier or ClinicalVerifier(corpus, provider=provider, model=model, retrieval=retrieval, allow_fixtures=allow_fixtures)
    if verifier.store.kind == 'synthetic_workflow_fixture' and (not allow_fixtures):
        raise ValueError('Fixtures require explicit permission')
    if consistency_mode not in ('none', 'measured'):
        raise ValueError('Invalid consistency mode')
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py') if 'legacy' not in p.relative_to(root).parts}
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), python=sys.version, platform=platform.platform(), source_sha256=sources, corpus_sha256=verifier.store.sha256, dataset_sha256=digest(data), corpus_kind=verifier.store.kind, dataset_label_provenance=data['label_provenance'], split=data['split'], prompt_version=PROMPT_VERSION, provider=provider, model=model, retrieval=retrieval, retrieval_model=verifier.retriever.model_name if retrieval == 'dense' else None, top_k=verifier.top_k, consistency_mode=consistency_mode, clinical_validation=False)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    records = []
    with (folder / 'predictions.jsonl').open('w') as stream:
        for case in data['cases']:
            consistency = None
            consistency_error = None
            if consistency_mode == 'measured':
                try:
                    if not case.get('generation_prompt'):
                        raise ValueError('No original generation prompt recorded')
                    consistency = measure_consistency(case['text'], case['generation_prompt'], verifier.client)
                except Exception as e:
                    consistency_error = f'{type(e).__name__}: {e}'
            result = verifier.verify(case['text'], consistency=consistency).to_dict()
            record = dict(id=case['id'], text=case['text'], expected=case['expected'], expected_evidence_ids=case.get('evidence_ids', []), result=result, consistency_error=consistency_error)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            records.append(record)
    errors = [r for r in records if r['result']['status'] in ('backend_error', 'invalid_response') or r['consistency_error']]
    complete = not errors
    summary = dict(completed=complete, n_cases=len(records), status_counts=dict(Counter((r['result']['status'] for r in records))), consistency_failures=sum((bool(r['consistency_error']) for r in records)), clinical_validation=False, interpretation='Development only. Labels are not independent clinical ground truth. No calibrated safety claim.', classification_metrics=metrics(records) if complete else None, metrics_withheld_reason=None if complete else 'Model, response or consistency errors occurred; performance metrics withheld.', elapsed_seconds_total=sum((r['result']['elapsed_seconds'] for r in records)))
    eligible = [r for r in records if r['expected_evidence_ids']]
    summary['retrieval'] = {f'evidence_hit_at_{verifier.top_k}': sum((bool(set(r['expected_evidence_ids']) & {p['id'] for p in r['result']['retrieved_chunks']}) for r in eligible)) / len(eligible) if eligible else None, 'n_eligible': len(eligible), 'definition': 'At least one expected passage retrieved; not clinical correctness'}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', default='data/references.json')
    p.add_argument('--dataset', default='data/development.json')
    p.add_argument('--output', required=True)
    p.add_argument('--provider', choices=['openai', 'groq', 'ollama'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--retrieval', choices=['lexical', 'dense'], default='lexical')
    p.add_argument('--consistency', choices=['none', 'measured'], default='none')
    a = p.parse_args()
    s = run_evaluation(a.corpus, a.dataset, a.output, a.provider, a.model, a.retrieval, a.consistency)
    print(json.dumps(s, indent=2))
    return 0 if s['completed'] else 2
if __name__ == '__main__':
    sys.exit(main())
