import copy, json, math, os, subprocess, sys, tempfile, threading, unittest, urllib.error, urllib.request
from pathlib import Path
from http.server import HTTPServer
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify import ClinicalVerifier, ModelClient, ReferenceStore, Retriever, parse_verdict, review_priority, unit_score
from ncs_probe import cosine_variation, measure_consistency
from evaluate import metrics, validate_dataset, run_evaluation
from api import make_handler
CORPUS = ROOT / 'data/references.json'
PASSAGES = ReferenceStore(CORPUS).passages

class FakeClient:
    provider = 'test_double'
    model = 'scripted-response-not-a-language-model'

    def __init__(self, response=None, error=False):
        self.response = response
        self.error = error
        self.calls = 0

    def complete(self, system, user, **kwargs):
        self.calls += 1
        if self.error:
            raise RuntimeError('test backend unavailable')
        return (self.response, {'provider': self.provider, 'requested_model': self.model})

def reply(label='supported', quote=None, pid=None):
    p = PASSAGES[0]
    return json.dumps(dict(label=label, reason='Scripted test response, not clinical reasoning.', confidence=0.8, evidence=[] if label == 'uncertain' else [dict(passage_id=pid or p['id'], quote=quote or p['text'])]))

def verifier(client):
    return ClinicalVerifier(CORPUS, client=client)

class PipelineTests(unittest.TestCase):

    def test_valid_verdict(self):
        self.assertEqual(parse_verdict(reply(), PASSAGES)['label'], 'supported')

    def test_unknown_citation(self):
        with self.assertRaises(ValueError):
            parse_verdict(reply(pid='imagined'), PASSAGES)

    def test_invented_quote(self):
        with self.assertRaises(ValueError):
            parse_verdict(reply(quote='Invented quotation'), PASSAGES)

    def test_missing_evidence(self):
        d = json.loads(reply())
        d['evidence'] = []
        with self.assertRaises(ValueError):
            parse_verdict(json.dumps(d), PASSAGES)

    def test_safe_label_rejected(self):
        d = json.loads(reply())
        d['label'] = 'safe'
        with self.assertRaises(ValueError):
            parse_verdict(json.dumps(d), PASSAGES)

    def test_invalid_json(self):
        with self.assertRaises(ValueError):
            parse_verdict('not json', PASSAGES)

    def test_json_fences(self):
        self.assertEqual(parse_verdict('```json\n' + reply() + '\n```', PASSAGES)['label'], 'supported')

    def test_nonfinite_confidence(self):
        for value in [float('nan'), float('inf'), -1, 1.1, True, '0.9']:
            d = json.loads(reply())
            d['confidence'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_verdict(json.dumps(d), PASSAGES)

    def test_backend_error_requires_review(self):
        r = verifier(FakeClient(error=True)).verify('antipsychotic loading dose')
        self.assertEqual((r.label, r.status, r.review_priority), ('uncertain', 'backend_error', 'REVIEW_REQUIRED'))
        self.assertIsNone(r.confidence)

    def test_malformed_response_requires_review(self):
        r = verifier(FakeClient('broken')).verify('antipsychotic loading dose')
        self.assertEqual(r.status, 'invalid_response')
        self.assertEqual(r.raw_response, 'broken')

    def test_no_evidence_no_model_call(self):
        c = FakeClient(reply())
        r = verifier(c).verify('xyzqv zzzqq')
        self.assertEqual(r.status, 'insufficient_evidence')
        self.assertEqual(c.calls, 0)

    def test_full_provenance_preserved(self):
        r = verifier(FakeClient(reply())).verify('antipsychotic loading dose')
        self.assertIn('source_url', r.retrieved_chunks[0])
        self.assertIn('system', r.prompt)
        self.assertIsNotNone(r.raw_response)

    def test_no_cross_prompt_keyword_heuristic(self):
        self.assertEqual(verifier(FakeClient(reply('uncertain'))).verify('antipsychotic loading dose').label, 'uncertain')

    def test_negation_not_overridden(self):
        self.assertEqual(verifier(FakeClient(reply())).verify('Do not use a loading dose of antipsychotic medication.').label, 'supported')

    def test_contradiction_never_downgraded(self):
        for s in [0.0, 0.4, 0.9, None]:
            self.assertEqual(review_priority('contradicts', s), 'CONTRADICTION_REVIEW')

    def test_uncertainty_never_low(self):
        for s in [0.0, 0.4, 0.9, None]:
            self.assertEqual(review_priority('uncertain', s), 'REVIEW_REQUIRED')

    def test_invalid_consistency_scores(self):
        for s in [float('nan'), float('inf'), True, -1, 2]:
            with self.subTest(s=s), self.assertRaises(ValueError):
                unit_score(s)

    def test_preset_consistency_rejected(self):
        with self.assertRaises(ValueError):
            verifier(FakeClient(reply())).verify('text', dict(origin='simulated', score=0.8, n_samples=5))

    def test_wrong_candidate_consistency_rejected(self):
        with self.assertRaises(ValueError):
            verifier(FakeClient(reply())).verify('text', dict(origin='measured', score=0.2, n_samples=5, candidate_sha256='other'))

    def test_no_automatic_corpus_generation(self):
        with self.assertRaises(FileNotFoundError):
            ReferenceStore(ROOT / 'absent.json')

    def test_synthetic_corpus_requires_opt_in(self):
        d = copy.deepcopy(ReferenceStore(CORPUS).data)
        d['corpus_kind'] = 'synthetic_workflow_fixture'
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'corpus.json'
            p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                ReferenceStore(p)
            self.assertEqual(len(ReferenceStore(p, allow_fixtures=True).passages), 2)

    def test_duplicate_references_rejected(self):
        d = copy.deepcopy(ReferenceStore(CORPUS).data)
        d['passages'] *= 2
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'corpus.json'
            p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                ReferenceStore(p)

    def test_lexical_retrieval(self):
        self.assertEqual(Retriever(ReferenceStore(CORPUS)).search('antipsychotic loading dose', 1)[0]['id'], PASSAGES[0]['id'])

    def test_model_required(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            ModelClient('openai')

    def test_missing_credentials_no_network(self):
        with patch.dict(os.environ, {}, clear=True), patch('urllib.request.urlopen') as request:
            r = ClinicalVerifier(CORPUS, provider='openai', model='explicit-test-model').verify('antipsychotic dose')
            request.assert_not_called()
            self.assertEqual(r.status, 'backend_error')

    def test_consistency_math(self):
        self.assertAlmostEqual(cosine_variation([[1, 0], [1, 0]])['score'], 0)
        self.assertAlmostEqual(cosine_variation([[1, 0], [0, 1]])['score'], 1)

    def test_consistency_invalid_embeddings(self):
        for v in [[], [[1]], [[0], [1]], [[1], [1, 2]], [[math.nan], [1]]]:
            with self.subTest(v=v), self.assertRaises(ValueError):
                cosine_variation(v)

    def test_measured_samples(self):

        class Encoder:

            def encode(self, samples):
                return [[1, 0] for x in samples]
        c = FakeClient(json.dumps({'answer': 'A test sample'}))
        r = measure_consistency('candidate', 'original prompt', c, n_samples=3, encoder=Encoder())
        self.assertEqual(c.calls, 3)
        self.assertEqual(r['origin'], 'measured')
        self.assertEqual(len(r['samples']), 3)

    def test_sampler_failure_propagates(self):
        with self.assertRaises(RuntimeError):
            measure_consistency('a', 'b', FakeClient(error=True), encoder=object())

    def test_historical_metric_arithmetic_only(self):
        pairs = [('supported', 'supported')] * 10 + [('uncertain', 'supported')] * 7 + [('uncertain', 'uncertain')] * 2 + [('uncertain', 'contradicts')] + [('contradicts', 'contradicts')] * 10
        m = metrics([dict(expected=a, result={'label': b}) for a, b in pairs])
        self.assertAlmostEqual(m['accuracy'], 22 / 30)
        self.assertAlmostEqual(m['per_label']['contradicts']['f1'], 20 / 21)

    def test_dataset_scores_forbidden(self):
        d = json.loads((ROOT / 'data/development.json').read_text())
        d['cases'][0]['ncs_score'] = 0.9
        with self.assertRaises(ValueError):
            validate_dataset(d)

    def test_errors_withhold_metrics_and_save_records(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'run'
            s = run_evaluation(CORPUS, ROOT / 'data/development.json', out, 'test_double', 'fixture', verifier=verifier(FakeClient(error=True)))
            self.assertFalse(s['completed'])
            self.assertIsNone(s['classification_metrics'])
            self.assertEqual(len((out / 'predictions.jsonl').read_text().splitlines()), 12)
            self.assertTrue((out / 'manifest.json').exists())

    def test_cli_imports_without_optional_ml_packages(self):
        for name in ['main.py', 'api.py', 'evaluate.py', 'ingest.py']:
            p = subprocess.run([sys.executable, str(ROOT / name), '--help'], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)

class APITests(unittest.TestCase):

    def setUp(self):
        self.server = HTTPServer(('127.0.0.1', 0), make_handler(verifier(FakeClient(error=True))))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, data):
        req = urllib.request.Request(self.url + '/verify', data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req) as r:
                return (r.status, json.load(r))
        except urllib.error.HTTPError as e:
            return (e.code, json.load(e))

    def test_health_honest(self):
        with urllib.request.urlopen(self.url + '/health') as r:
            b = json.load(r)
        self.assertEqual(b['backend_connectivity'], 'not_checked')
        self.assertFalse(b['clinical_validation'])

    def test_api_invalid_types(self):
        for d in [[], {'text': 123}, {'text': ''}, {'text': None}]:
            self.assertEqual(self.post(d)[0], 400)

    def test_api_supplied_score_rejected(self):
        self.assertEqual(self.post({'text': 'a', 'ncs_score': 0.1})[0], 400)

    def test_api_backend_failure_503(self):
        s, d = self.post({'text': 'antipsychotic loading dose'})
        self.assertEqual(s, 503)
        self.assertEqual(d['review_priority'], 'REVIEW_REQUIRED')

    def test_ingest_endpoint_removed(self):
        req = urllib.request.Request(self.url + '/ingest', data=b'{}', headers={'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as c:
            urllib.request.urlopen(req)
        self.assertEqual(c.exception.code, 404)
if __name__ == '__main__':
    unittest.main(verbosity=2)
