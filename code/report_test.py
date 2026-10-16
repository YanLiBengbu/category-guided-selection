"""Test report of the category-guided selector: main table, paired comparisons, subsets, and categories.

Inputs: the seed-averaged selector matrix from apply_rerank_test.py (TEST_IDS order), its per-seed choices
        (same prefix, _choices.jsonl), and results/ablation_learned_matrix.npy from ablation.py (the same
        two-generator pool ranked by the mean likelihood alone).
Outputs: results/report_test.json (all numbers quoted in the paper) and tables/main_table.tex.
Usage: python report_test.py SELECTOR_MATRIX.npy
"""
import json
import sys
from collections import defaultdict

import numpy as np

from common import METRICS, ROOT, read, score, test_rows
from final_eval import single
from pipeline import choose_original
from stats import PUB, TEST_IDS, bootstrap, hosted, per_example_csv, subsets
from tables import main_table

SELECTOR = 'Category-guided selection'
TOP_BEAM_4BIT = 'Earlier adapters, top-beam rule (4-bit)'
# Decomposition rows: the retrained generators of the selector pool with the top-beam rule, and the pool
# ranked by the mean likelihood alone (no learned weights).
BASE_TOP = 'Base generator, top-beam rule (4-bit)'
GUIDED_TOP = 'Guided generator, top-beam rule (4-bit)'
POOL_LIKELIHOOD = 'Two generators, mean likelihood (4-bit)'
# External baselines of the review round (rows appear when their result files exist).
EXTERNAL = [('External', 'Fine-tuned Japanese T5', 'ext/t5_matrix.npy', 'selector - t5'),
            ('External', 'DeBERTa localizer, base generator (4-bit)', 'ext/deberta_matrix.npy', 'selector - deberta'),
            ('Local LLM', 'Direct prompting, local Qwen3-4B', 'local_llm/direct_qwen3_matrix.npy', 'selector - local direct'),
            ('Local LLM', 'LGR-MAS, local Qwen3-4B', 'local_llm/lgr_qwen3_matrix.npy', 'selector - local lgr')]


def means(mat, mask=None):
    m = mat if mask is None else mat[mask]
    return {k: float(v) for k, v in zip(METRICS, 100 * m.mean(0))}


def per_seed_exact(choices_path):
    """ExactMatch of each seed pipeline from the selector choices (TEST_IDS order)."""
    rows = test_rows()  # TEST_IDS order, as in select_rerank.build
    acc = defaultdict(list)
    for c in read(choices_path):
        seed, rid = c['key'].split(':')
        gold = rows[int(rid.split('-')[-1]) - 1]
        acc[seed].append(score(gold, c['span'] if c['revision'] else '', c['revision'])['ExactMatch'])
    assert all(len(v) == len(TEST_IDS) for v in acc.values())
    return {s: 100 * float(np.mean(v)) for s, v in sorted(acc.items())}


def main(path):
    sel = np.load(path)
    base_top = single('orig_k0', choose_original)
    guided_top = single('guided_k4', choose_original)
    pool_ll = np.load(ROOT / 'results' / 'ablation_learned_matrix.npy')
    external = [(g, name, np.load(ROOT / 'results' / f), key) for g, name, f, key in EXTERNAL
                if (ROOT / 'results' / f).exists()]
    top4 = single('orig_published', choose_original)
    top16 = per_example_csv(PUB / 'historical_per_example.csv', 'qwen4b_aux16k_selected_primary')
    learner = per_example_csv(PUB / 'historical_per_example.csv', 'qwen4b_selected_primary')
    host = {f'{s}, {b}': hosted(s, b) for s in ['Direct LLM baseline', 'LGR-MAS', 'Evidence-gated LGR-MAS']
            for b in ['qwen3.7-max', 'gpt-5.4-low']}
    best_host = max(host, key=lambda k: host[k][:, 5].mean())
    sub = subsets()
    masks = dict(all=None, unseen_pair=~sub['pair_seen'], seen_pair=sub['pair_seen'],
                 no_shared_unit=sub['no_shared_unit'])
    out = dict(selector=means(sel), top_beam_4bit=means(top4), top_beam_bf16=means(top16), best_hosted=best_host,
               hosted={k: means(v) for k, v in host.items()}, comparisons={}, subsets={}, categories={},
               base_top=means(base_top), guided_top=means(guided_top), pool_likelihood=means(pool_ll),
               selector_seeds=per_seed_exact(str(path).replace('_matrix.npy', '_choices.jsonl')),
               external={name: means(mat) for _, name, mat, _ in external})
    pairs = {'selector - best hosted': (sel, host[best_host]),
             'selector - top-beam BF16': (sel, top16),
             'selector - top-beam 4-bit': (sel, top4),
             'selector - base top-beam': (sel, base_top),
             'selector - pool likelihood': (sel, pool_ll),
             'pool likelihood - base top-beam': (pool_ll, base_top),
             'selector - learner-only 4B': (sel, learner),
             'selector - evidence-gated GPT-5.4': (sel, host['Evidence-gated LGR-MAS, gpt-5.4-low'])}
    pairs.update({key: (sel, mat) for _, _, mat, key in external})
    for name, (a, b) in pairs.items():
        out['comparisons'][name] = bootstrap(a, b)
    for name, mask in masks.items():
        if mask is None:
            continue
        out['subsets'][name] = dict(n=int(mask.sum()), selector=means(sel, mask), best_hosted=means(host[best_host], mask),
                                    top_beam_bf16=means(top16, mask),
                                    diff_hosted=bootstrap(sel, host[best_host], mask=mask),
                                    diff_bf16=bootstrap(sel, top16, mask=mask))
    cats = np.array([r['error_type'] for r in test_rows()])
    for c in sorted(set(cats)):
        m = cats == c
        out['categories'][c] = dict(n=int(m.sum()), selector=means(sel, m), best_hosted=means(host[best_host], m),
                                    top_beam_bf16=means(top16, m))
    # span-correct examples where the selector and the best hosted system both found the interval
    both = (sel[:, 0] == 1) & (host[best_host][:, 0] == 1)
    out['both_span'] = dict(n=int(both.sum()), selector_rm=float(100 * sel[both, 2].mean()),
                            hosted_rm=float(100 * host[best_host][both, 2].mean()))
    json.dump(out, open(ROOT / 'results' / 'report_test.json', 'w', encoding='utf-8'), indent=1)
    tables = ROOT / 'tables'
    tables.mkdir(exist_ok=True)
    main_table([(g, name, mat) for g, name, mat, _ in external]
               + [('Ours', TOP_BEAM_4BIT, top4), ('Ours', BASE_TOP, base_top), ('Ours', GUIDED_TOP, guided_top),
                  ('Ours', POOL_LIKELIHOOD, pool_ll), ('Ours', SELECTOR, sel)], SELECTOR, tables / 'main_table.tex')
    print('best hosted', best_host, out['hosted'][best_host]['ExactMatch'])
    print('selector', {k: round(v, 2) for k, v in out['selector'].items()}, 'seeds', out['selector_seeds'])
    for name, res in out['comparisons'].items():
        e = res['ExactMatch']
        print(f"{name:36s} Exact {e['diff']:+.2f} sim [{e['sim'][0]:+.2f}, {e['sim'][1]:+.2f}]")
    for name, s in out['subsets'].items():
        e = s['diff_hosted']['ExactMatch']
        print(f"{name:16s} n={s['n']} sel {s['selector']['ExactMatch']:.2f} host {s['best_hosted']['ExactMatch']:.2f} "
              f"diff {e['diff']:+.2f} sim [{e['sim'][0]:+.2f}, {e['sim'][1]:+.2f}]")
    print('both span', out['both_span'])


if __name__ == '__main__':
    main(sys.argv[1])
