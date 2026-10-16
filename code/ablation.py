"""Ablations of the frozen candidate selector on development (fold-wise) and test (fit on all development pools).

Removals from the final system:
  fluency: the base-model fluency features; consistency: the category marker, on-category,
  off-category, and register features; memory: the two development count features;
  guided: the guided-prompt generator; learned: the log-linear weights (the same pool ranked by the
  mean generator likelihood alone).
Additions to the final system (candidate sources that we tested and did not adopt):
  morph: the morphological variants; zero_shot: the zero-shot Qwen3.5 generator (beams and likelihood).
Usage: python ablation.py CONFIG.json OUT.json
"""
import json
import sys

import numpy as np

import select_rerank as sr
from apply_rerank_test import configure as configure_final
from common import ROOT, read, score, mean_scores, METRICS, fmt
from stats import TEST_IDS, bootstrap

ABLATIONS = ['full', 'fluency', 'consistency', 'memory', 'guided', 'learned', 'morph', 'zero_shot']


def configure(name, cfg, dev, test):
    configure_final(cfg, dev, test)
    dev_runs, test_runs = list(cfg['dev_runs']), list(cfg['test_runs'])
    if name == 'fluency':
        sr.DROP.add('lm')
    if name == 'consistency':
        sr.DROP.update({'marker', 'on_cat', 'off_cat', 'register'})
    if name == 'memory':
        sr.DROP.update({'mem_pair', 'mem_rev'})
    if name == 'guided':
        dev_runs = [r for r in dev_runs if 'guided' not in r]
        test_runs = [r for r in test_runs if 'guided' not in r]
    if name == 'morph':
        sr.USE_MORPH[0] = True
    if name == 'zero_shot':
        dev_runs.append('cv/qwen35_zs')
        test_runs.append('final/qwen35_zs')
    return dev_runs, test_runs


def available(name, test, test_runs):
    if name == 'morph':
        return any(r.get('morph') for r in test)
    return all(r in test[0]['scores'] for r in test_runs)


def test_matrix(items, w):
    per = {}
    for i in items:
        seed, rid = i['rec']['key'].split(':')
        y = sr.pick(i, w)
        per.setdefault(seed, {})[rid] = score(i['gold'], i['rec']['source'] if y else '', y)
    return np.array([np.mean([[per[s][r][m] for m in METRICS] for s in per], axis=0) for r in TEST_IDS])


def main(config, out):
    cfg = json.load(open(config, encoding='utf-8'))
    dev, test = read(ROOT / cfg['dev_file']), read(ROOT / cfg['test_file'])
    l2 = cfg['l2']
    results, mats = {}, {}
    for name in ABLATIONS:
        dev_runs, test_runs = configure(name, cfg, dev, test)
        if name in ('morph', 'zero_shot') and not available(name, test, test_runs):
            # The test pool holds only the frozen generators; tested additions are reported on development.
            dev_res, _, _ = sr.evaluate_dev(dev, dev_runs, l2)
            oracle = 100 * np.mean([i['label'] is not None for i in sr.build('dev', dev, dev_runs)])
            results[name] = dict(dev=dev_res, dev_pool_coverage=float(oracle), test=None)
            print(f'{name:12s} dev ', fmt(dev_res), f'coverage {oracle:.1f}', '(development only)', flush=True)
            continue
        if name == 'learned':
            items = sr.build('dev', dev, dev_runs)
            first = lambda i: np.eye(i['x'].shape[1])[0] if len(i['opts']) else None
            dev_res = mean_scores([score(i['gold'], i['rec']['source'] if (y := sr.pick(i, first(i))) else '', y)
                                   for i in items])
            titems = sr.build('test', test, test_runs)
            w = np.eye(titems[0]['x'].shape[1])[0]
        else:
            dev_res, _, _ = sr.evaluate_dev(dev, dev_runs, l2)
            w = sr.fit_all_dev(dev, dev_runs, l2)
            titems = sr.build('test', test, test_runs)
        mats[name] = test_matrix(titems, w)
        oracle = 100 * np.mean([i['label'] is not None for i in sr.build('dev', dev, dev_runs)])
        results[name] = dict(dev=dev_res, dev_pool_coverage=float(oracle),
                             test={m: float(v) for m, v in zip(METRICS, 100 * mats[name].mean(0))})
        print(f'{name:12s} dev ', fmt(dev_res), f'coverage {oracle:.1f}', flush=True)
        print(f'{"":12s} test', fmt(results[name]['test']), flush=True)
    sr.USE_MORPH[0] = True
    sr.DROP.clear()
    for name in mats:
        if name != 'full':
            results[name]['diff_vs_full'] = bootstrap(mats[name], mats['full'])
    json.dump(results, open(out, 'w', encoding='utf-8'), indent=1)
    np.save(out.replace('.json', '_full_matrix.npy'), mats['full'])
    np.save(out.replace('.json', '_learned_matrix.npy'), mats['learned'])  # same pool ranked by mean likelihood only


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
