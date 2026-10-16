"""Apply selection rules to saved candidate files and report the six CAMC-JL metrics.

Usage:
  python evaluate.py test results/repro_test_candidates.jsonl
  python evaluate.py dev results/cv/<tag>/fold0.jsonl results/cv/<tag>/fold1.jsonl ...
"""
import sys
from collections import defaultdict

from common import dev_rows, test_rows, read, score, mean_scores, fmt, norm
from pipeline import choose_original, choose_nbest

RULES = dict(original=choose_original, nbest=choose_nbest,
             nbest_gen_only=lambda r: choose_nbest(r, include_compact=False))


def gold_for(split):
    if split == 'test':
        rows = test_rows()
        return {f'CAMCJL-T-{i + 1:04d}': r for i, r in enumerate(rows)}, lambda key: key.split(':')[1]
    rows = dev_rows()
    return {str(i): r for i, r in enumerate(rows)}, lambda key: key.split(':')[1]


def evaluate(split, records, rules=RULES, group_of=lambda key: key.split(':')[0]):
    gold, gid = gold_for(split)
    out = {}
    for name, rule in rules.items():
        groups = defaultdict(list)
        for rec in records:
            g = gold[gid(rec['key'])]
            choice = rule(rec)
            span = rec['source'] if choice else ''
            groups[group_of(rec['key'])].append(score(g, span, choice))
        per_group = {k: mean_scores(v) for k, v in groups.items()}
        out[name] = {m: sum(p[m] for p in per_group.values()) / len(per_group) for m in next(iter(per_group.values()))}
        out[name + '_groups'] = per_group
    return out


def oracle(split, records):
    gold, gid = gold_for(split)
    hit = top = 0
    for rec in records:
        g = gold[gid(rec['key'])]
        span_ok = (rec['selected_start'], rec['selected_end']) == (g['gold_span_start'], g['gold_span_end'])
        cands = [b['text'] for b in rec['beams'] if b['valid']] + [rec['compact']]
        hit += span_ok and any(norm(c) == norm(g['gold_revision']) for c in cands)
        top += span_ok
    return 100 * hit / len(records), 100 * top / len(records)


if __name__ == '__main__':
    split, files = sys.argv[1], sys.argv[2:]
    records = [r for f in files for r in read(f)]
    group = (lambda key: key.split(':')[0]) if split == 'test' else (lambda key: 'all')
    res = evaluate(split, records, group_of=group)
    for name in RULES:
        print(f'{name:16s}', fmt(res[name]))
        if split == 'test':
            for g, v in sorted(res[name + '_groups'].items()):
                print(f'  seed {g}', fmt(v))
    o, sm = oracle(split, records)
    print(f'oracle (any candidate exact) {o:.2f}  span-correct {sm:.2f}  n={len(records)}')
