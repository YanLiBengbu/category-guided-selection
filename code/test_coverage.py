"""Share of frozen test pools that contain the reference replacement (the ceiling of candidate selection).

Replicates the development 'Cov.' computation of ablation.py on the test pools of the frozen selector:
an item is covered when select_rerank.build() assigns it a label, i.e. the normalized reference
replacement is in the pool. Reports the full pool and the pool without the guided generator,
per seed pipeline and over all seed pipelines, with the mean pool size.
Output: results/test_pool_coverage.json (read by make_tables.py).
Usage: python test_coverage.py
"""
import json
from collections import defaultdict

import numpy as np

import select_rerank as sr
from apply_rerank_test import configure as configure_final
from common import ROOT, read


def main():
    cfg = json.load(open(ROOT / 'results' / 'selector_final.json', encoding='utf-8'))
    dev, test = read(ROOT / cfg['dev_file']), read(ROOT / cfg['test_file'])
    configure_final(cfg, dev, test)
    out = {}
    for name, runs in [('full', list(cfg['test_runs'])),
                       ('no_guided', [r for r in cfg['test_runs'] if 'guided' not in r])]:
        items = sr.build('test', test, runs)
        per_seed = defaultdict(list)
        for i in items:
            seed, _ = i['rec']['key'].split(':')
            per_seed[seed].append(i['label'] is not None)
        out[name] = dict(items=len(items),
                         coverage=100 * float(np.mean([i['label'] is not None for i in items])),
                         per_seed={s: 100 * float(np.mean(v)) for s, v in sorted(per_seed.items())},
                         mean_pool_size=float(np.mean([len(i['opts']) for i in items])))
        print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(ROOT / 'results' / 'test_pool_coverage.json', 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
