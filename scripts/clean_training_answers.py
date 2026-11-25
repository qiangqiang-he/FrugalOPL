"""Remove missing, null and blank answers from a JSON training list.

Usage: python scripts/clean_training_answers.py data/openthoughts_math30k.json
The input is replaced atomically; --output writes a separate cleaned file.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.eval import load_dataset_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with args.path.open(encoding='utf-8') as stream:
        raw = json.load(stream)
    if not isinstance(raw, list):
        parser.error('Training input must be a JSON list.')
    records = load_dataset_records({'path': str(args.path.resolve()), 'drop_missing_answer': True})
    destination = (args.output or args.path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.parent,
                                         prefix=destination.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(records, stream, ensure_ascii=False)
            stream.write('\n')
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({'original_records': len(raw), 'removed_records': len(raw)-len(records),
                      'retained_records': len(records)}))


if __name__ == '__main__':
    main()
