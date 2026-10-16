"""Seed-averaged test matrices for the systems of this study and paired comparisons.

Systems are built from saved candidate files (no GPU):
  single-adapter selection  results/final/<tag>/seed*.jsonl          (pipeline.choose_original / choose_nbest)
  likelihood ensemble       results/final/<cross>.jsonl              (evaluate_ens.choose)
  interval re-ranking       results/span/<tag>_test/seed*.jsonl      (rerank weights fitted on development)
Each seed pipeline uses its own frontend; we average the three pipelines within each example.
"""
import json
import sys
from collections import defaultdict

import numpy as np

from common import RESULTS, METRICS, read, score, test_rows, fmt
from stats import TEST_IDS, bootstrap, show

GOLD = {f'CAMCJL-T-{i + 1:04d}': r for i, r in enumerate(test_rows())}


def matrix(choices):
    """choices: {seed: {release_id: (span_text, revision)}} -> example x metric matrix (seed-averaged)."""
    per = {s: {rid: score(GOLD[rid], *c) for rid, c in d.items()} for s, d in choices.items()}
    return np.array([np.mean([[per[s][i][m] for m in METRICS] for s in per], axis=0) for i in TEST_IDS])


def single(tag, rule):
    out = defaultdict(dict)
    for seed in [20260927, 20260928, 20260929]:
        for rec in read(RESULTS / 'final' / tag / f'seed{seed}.jsonl'):
            y = rule(rec)
            out[seed][rec['key'].split(':')[1]] = (rec['source'] if y else '', y)
    return matrix(out)


def ensemble(path, runs, include_compact=True):
    from evaluate_ens import choose
    out = defaultdict(dict)
    for rec in read(path):
        seed, rid = rec['key'].split(':')
        y = choose(rec, runs, include_compact=include_compact)
        out[int(seed)][rid] = (rec['source'] if y else '', y)
    return matrix(out)


def summarize(name, mat):
    print(f'{name:40s}', fmt({m: v for m, v in zip(METRICS, 100 * mat.mean(0))}))


if __name__ == '__main__':
    from pipeline import choose_original, choose_nbest
    tag = sys.argv[1]
    for name, rule in [('original rule', choose_original), ('n-best', choose_nbest)]:
        summarize(f'{tag} {name}', single(tag, rule))
