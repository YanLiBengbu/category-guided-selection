"""Fit the candidate selector on all out-of-fold development pools and apply it once to the test set.

Usage: python apply_rerank_test.py CONFIG.json OUT_PREFIX
CONFIG (results/selector_final.json) fixes the development and test files, the generator runs,
the morphological-candidate switch, the fluency models, and the L2 penalty. Writes
OUT_PREFIX_choices.jsonl, OUT_PREFIX_matrix.npy (seed-averaged example x metric matrix in TEST_IDS
order), and OUT_PREFIX_weights.json, and prints the means.
"""
import json
import sys

import numpy as np

import select_rerank as sr
from common import ROOT, read, write_lines, score, METRICS, fmt
from stats import TEST_IDS


def configure(cfg, dev, test):
    sr.USE_MORPH[0] = cfg['morph']
    sr.DROP.clear()
    names = sorted(dev[0].get('lm', {}))
    assert names == sorted(test[0].get('lm', {})), (names, sorted(test[0].get('lm', {})))
    assert set(cfg['lm']) <= set(names), (cfg['lm'], names)
    sr.set_lm(names)
    sr.DROP.update('lm_' + n for n in names if n not in cfg['lm'])


def main(config, prefix):
    cfg = json.load(open(config, encoding='utf-8'))
    dev, test = read(ROOT / cfg['dev_file']), read(ROOT / cfg['test_file'])
    configure(cfg, dev, test)
    w = sr.fit_all_dev(dev, cfg['dev_runs'], cfg['l2'])
    items = sr.build('test', test, cfg['test_runs'])
    per_seed, choices = {}, []
    for i in items:
        seed, rid = i['rec']['key'].split(':')
        y = sr.pick(i, w)
        span = i['rec']['source'] if y else ''
        per_seed.setdefault(seed, {})[rid] = score(i['gold'], span, y)
        choices.append(dict(key=i['rec']['key'], span=span, revision=y))
    mat = np.array([np.mean([[per_seed[s][r][m] for m in METRICS] for s in per_seed], axis=0) for r in TEST_IDS])
    write_lines(f'{prefix}_choices.jsonl', choices)
    np.save(f'{prefix}_matrix.npy', mat)
    # select_rerank.features appends one likelihood margin per generator run after the named features.
    names = sr.NAMES + ([f'margin_generator{k + 1}' for k in range(len(cfg['test_runs']))]
                        if len(cfg['test_runs']) > 1 else [])
    assert len(names) == len(w), (len(names), len(w))
    json.dump(dict(weights=dict(zip(names, map(float, w))), config=cfg),
              open(f'{prefix}_weights.json', 'w', encoding='utf-8'), indent=1)
    print('test', fmt({m: v for m, v in zip(METRICS, 100 * mat.mean(0))}))
    for s in sorted(per_seed):
        print('  seed', s, fmt({m: 100 * np.mean([per_seed[s][r][m] for r in TEST_IDS]) for m in METRICS}))
    return mat


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
