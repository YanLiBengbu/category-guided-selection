"""Selector with a control generator in place of the guided generator (review round, 2026-10-12).

Controls (queue9.sh): orig_k0_b, a second base-prompt generator (training seed offset 1000); guided_k0, the guided
prompt without demonstrations; demos_k4, the guided prompt without the guidance sentence. Each control pool is the
compact candidate plus the beams of the base generator and the control generator. The selector keeps the frozen
features and L2 penalty: we refit its weights on the out-of-fold development pools and apply them once to the test
pools. Test differences are paired with the full selector (results/final_selector_matrix.npy).
Usage: python controls_eval.py -> results/controls.json
"""
import json

import numpy as np

import select_rerank as sr
from ablation import test_matrix
from apply_rerank_test import configure
from common import METRICS, ROOT, read
from stats import bootstrap

CONTROLS = {'orig_k0_b': 'Replace the guided generator with a second base generator',
            'guided_k0': 'Remove the demonstrations from the guided prompt',
            'demos_k4': 'Remove the guidance sentence from the guided prompt'}


def main():
    final = json.load(open(ROOT / 'results' / 'selector_final.json', encoding='utf-8'))
    full = np.load(ROOT / 'results' / 'final_selector_matrix.npy')
    out = {}
    for tag, label in CONTROLS.items():
        dev_path = ROOT / 'results' / f'cv_cross_q9_{tag}_lm2.jsonl'
        test_path = ROOT / 'results' / f'test_cross_q9_{tag}_lm2.jsonl'
        if not dev_path.exists():
            continue
        dev = read(dev_path)
        test = read(test_path) if test_path.exists() else dev
        configure(final, dev, test)
        dev_runs = ['cv/orig_k0', f'cv/{tag}']
        dev_res, _, _ = sr.evaluate_dev(dev, dev_runs, final['l2'])
        cov = 100 * float(np.mean([i['label'] is not None for i in sr.build('dev', dev, dev_runs)]))
        res = dict(label=label, dev=dev_res, dev_pool_coverage=cov, test=None)
        if test_path.exists():
            test_runs = ['final/orig_k0', f'final/{tag}']
            w = sr.fit_all_dev(dev, dev_runs, final['l2'])
            items = sr.build('test', test, test_runs)
            mat = test_matrix(items, w)
            np.save(ROOT / 'results' / f'controls_{tag}_matrix.npy', mat)
            res['test'] = {m: float(v) for m, v in zip(METRICS, 100 * mat.mean(0))}
            res['test_pool_coverage'] = 100 * float(np.mean([i['label'] is not None for i in items]))
            res['diff_vs_full'] = bootstrap(mat, full)
        out[tag] = res
        print(tag, 'dev Exact', round(dev_res['ExactMatch'], 2), 'cov', round(cov, 1),
              'test Exact', None if res['test'] is None else round(res['test']['ExactMatch'], 2), flush=True)
    json.dump(out, open(ROOT / 'results' / 'controls.json', 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
