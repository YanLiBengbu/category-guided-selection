"""Merge the rec['lm'] feature dictionaries of candidate files that share records and candidates.

Usage: python merge_lm.py OUT.jsonl BASE.jsonl OTHER.jsonl [OTHER.jsonl ...]
"""
import sys

from common import read, write_lines


def main(dest, base, *others):
    records = read(base)
    for path in others:
        extra = {r['key']: r for r in read(path)}
        for rec in records:
            other = extra[rec['key']]
            assert other['candidates'] == rec['candidates'], rec['key']
            rec.setdefault('lm', {}).update(other.get('lm', {}))
    write_lines(dest, records)
    print('merged', sorted(records[0]['lm']), len(records))


if __name__ == '__main__':
    main(*sys.argv[1:])
