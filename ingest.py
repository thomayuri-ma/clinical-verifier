"""Validate curated JSON references. Missing data never generates a sample corpus."""
import argparse, json
from verify import ReferenceStore

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus')
    args = parser.parse_args()
    s = ReferenceStore(args.corpus)
    print(json.dumps(dict(passages=len(s.passages), corpus_sha256=s.sha256, kind=s.kind), indent=2))
if __name__ == '__main__':
    main()
