"""Local research HTTP server. Start explicitly with a configured model."""
import argparse, json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from verify import ClinicalVerifier
MAX_BODY = 64000

def make_handler(verifier):

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, *args):
            pass

        def send(self, status, data, content_type='application/json'):
            body = json.dumps(data, allow_nan=False).encode() if content_type == 'application/json' else data
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                return self.send(200, dict(status='process_ready', provider=verifier.client.provider, model=verifier.client.model, references=len(verifier.store.passages), backend_connectivity='not_checked', clinical_validation=False))
            if self.path == '/':
                return self.send(200, Path(__file__).with_name('index.html').read_bytes(), 'text/html; charset=utf-8')
            self.send(404, dict(error='Not found'))

        def do_POST(self):
            if self.path != '/verify':
                return self.send(404, dict(error='Not found'))
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > MAX_BODY:
                    return self.send(413, dict(error='Invalid request size'))
                if self.headers.get_content_type() != 'application/json':
                    return self.send(415, dict(error='Use application/json'))
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or set(body) - {'text'}:
                    raise ValueError('Only the text field is accepted; supplied scores are prohibited')
                if not isinstance(body.get('text'), str) or not body['text'].strip():
                    raise ValueError('Nonempty text required')
                result = verifier.verify(body['text'])
                self.send(503 if result.status == 'backend_error' else 502 if result.status == 'invalid_response' else 200, result.to_dict())
            except (ValueError, TypeError, UnicodeError):
                self.send(400, dict(error='Invalid JSON or text payload'))
            except Exception:
                self.send(500, dict(error='Assessment unavailable'))
    return Handler

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider', choices=['openai', 'groq', 'ollama'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--corpus', default='data/references.json')
    p.add_argument('--retrieval', choices=['lexical', 'dense'], default='lexical')
    p.add_argument('--port', type=int, default=5000)
    a = p.parse_args()
    v = ClinicalVerifier(a.corpus, provider=a.provider, model=a.model, retrieval=a.retrieval)
    HTTPServer(('127.0.0.1', a.port), make_handler(v)).serve_forever()
if __name__ == '__main__':
    main()
