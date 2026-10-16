"""Evaluate likelihood-ensemble selection on cross-scored candidate files.

Usage: python evaluate_ens.py dev|test results/....jsonl [subset of run indices, e.g. 0,1]
"""
import sys
from collections import defaultdict

from common import read, score, mean_scores, fmt, norm, valid
from evaluate import gold_for


def choose(rec, runs, include_compact=True, beams_from=None):
    beams_from = runs if beams_from is None else beams_from
    options = []
    if include_compact and rec['compact']:
        options.append(rec['compact'])
    for r in beams_from:
        options += [b['text'] for b in rec['beams'][r] if b['valid']]
    options = list(dict.fromkeys(options))
    if not options:
        return rec['compact']
    mean = lambda t: sum(rec['scores'][r][t] for r in runs) / len(runs)
    return max(options, key=mean)


def evaluate(split, records, runs, **kw):
    gold, gid = gold_for(split)
    groups = defaultdict(list)
    for rec in records:
        g = gold[gid(rec['key'])]
        y = choose(rec, runs, **kw)
        groups[rec['key'].split(':')[0] if split == 'test' else 'all'].append(score(g, rec['source'] if y else '', y))
    per = {k: mean_scores(v) for k, v in groups.items()}
    return {m: sum(p[m] for p in per.values()) / len(per) for m in next(iter(per.values()))}


if __name__ == '__main__':
    split, path = sys.argv[1], sys.argv[2]
    records = read(path)
    runs = sys.argv[3].split(',') if len(sys.argv) > 3 else list(records[0]['scores'])
    for r in runs:
        print(f'run {r} alone      ', fmt(evaluate(split, records, [r], beams_from=[r])))
    print('ensemble           ', fmt(evaluate(split, records, runs)))
    print('ensemble no compact', fmt(evaluate(split, records, runs, include_compact=False)))
