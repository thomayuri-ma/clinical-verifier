"""Single statement research runner. Never returns a clinical safety guarantee."""
import argparse, json
from verify import ClinicalVerifier

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text', required=True)
    p.add_argument('--corpus', default='data/references.json')
    p.add_argument('--provider', choices=['openai', 'groq', 'ollama'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--retrieval', choices=['lexical', 'dense'], default='lexical')
    a = p.parse_args()
    v = ClinicalVerifier(a.corpus, provider=a.provider, model=a.model, retrieval=a.retrieval)
    print(json.dumps(v.verify(a.text).to_dict(), indent=2))
if __name__ == '__main__':
    main()
