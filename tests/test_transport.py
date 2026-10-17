import io, json, os, sys, unittest, urllib.error
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify import ModelClient

class TransportTests(unittest.TestCase):

    def exercise(self, provider, key):
        payload = {'model': 'returned-model', 'id': 'test-response', 'choices': [{'message': {'content': '{}'}}]}
        with patch.dict(os.environ, {key: 'test-placeholder'}, clear=True), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(payload).encode())) as call:
            raw, meta = ModelClient(provider, 'requested-model').complete('system instructions', 'user data')
            request = call.call_args.args[0]
            body = json.loads(request.data)
            self.assertEqual(raw, '{}')
            self.assertEqual(meta['returned_model'], 'returned-model')
            self.assertEqual(body['messages'][0]['role'], 'system')
            self.assertEqual(body['model'], 'requested-model')
            self.assertIn('Bearer', request.get_header('Authorization'))
            self.assertEqual(body['response_format'], {'type': 'json_object'})

    def test_openai_adapter(self):
        self.exercise('openai', 'OPENAI_API_KEY')

    def test_groq_adapter(self):
        self.exercise('groq', 'GROQ_API_KEY')

    def test_ollama_adapter(self):
        payload = {'model': 'local-test', 'message': {'content': '{}'}}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(payload).encode())) as call:
            raw, meta = ModelClient('ollama', 'local-test').complete('s', 'u', temperature=0.8, seed=43)
            body = json.loads(call.call_args.args[0].data)
            self.assertEqual(body['options'], {'temperature': 0.8, 'seed': 43})
            self.assertEqual(raw, '{}')

    def test_http_failure_visible(self):
        error = urllib.error.HTTPError('https://test.invalid', 401, 'Denied', {}, None)
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-placeholder'}), patch('urllib.request.urlopen', side_effect=error), self.assertRaisesRegex(RuntimeError, 'HTTP 401; no fallback'):
            ModelClient('groq', 'test').complete('s', 'u')

    def test_connection_failure_visible(self):
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError('unavailable')), self.assertRaisesRegex(RuntimeError, 'no fallback'):
            ModelClient('ollama', 'test').complete('s', 'u')
