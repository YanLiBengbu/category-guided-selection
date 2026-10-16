"""Build the per-example score matrices of every compared system and the main results table.

All numbers are recomputed from per-example result files:
  published/baseline_per_example.csv     retrieval and linear baselines (deterministic)
  published/gru_v2_per_example.csv       character GRU (three seeds)
  published/historical_per_example.csv   fusion, adapted 1.7B/4B generators, original SLA-JEC (BF16)
  hosted_per_example_metrics.csv         prompted hosted LLM systems (CAMC-JL release of our earlier study)
  results/final/<tag>/seed*.jsonl        4-bit runs of this study (scored here)
"""
import csv
import json
from collections import defaultdict

import numpy as np

from common import DATA, RESULTS, METRICS, SEEDS, read, score, test_rows
from stats import TEST_IDS, PUB, per_example_csv, hosted


def simple_csv(path, method, key='method'):
    acc = defaultdict(list)
    for r in csv.DictReader(open(path, encoding='utf-8')):
        if r[key] == method:
            acc[r['release_id']].append([float(r[m]) for m in METRICS])
    assert len(acc) == 559, (method, len(acc))
    return np.array([np.mean(acc[i], axis=0) for i in TEST_IDS])


def run_matrix(records_by_seed, rule):
    """Seed-averaged example x metric matrix for a selection rule applied to saved candidates."""
    gold = {f'CAMCJL-T-{i + 1:04d}': r for i, r in enumerate(test_rows())}
    per_seed = {}
    for seed, recs in records_by_seed.items():
        rows = {}
        for rec in recs:
            rid = rec['key'].split(':')[1]
            y = rule(rec)
            rows[rid] = score(gold[rid], rec['source'] if y else '', y)
        per_seed[seed] = rows
    return np.array([np.mean([[per_seed[s][i][m] for m in METRICS] for s in per_seed], axis=0) for i in TEST_IDS])


def baselines():
    pe = PUB / 'historical_per_example.csv'
    rows = [
        ('Retrieval', 'Frequency memory', simple_csv(PUB / 'baseline_per_example.csv', 'frequency_memory')),
        ('Retrieval', 'TF--IDF retrieval', simple_csv(PUB / 'baseline_per_example.csv', 'tfidf_retrieval')),
        ('Retrieval', 'BM25 retrieval', simple_csv(PUB / 'baseline_per_example.csv', 'bm25_retrieval')),
        ('Supervised', 'Linear + nearest revision', simple_csv(PUB / 'baseline_per_example.csv', 'linear_nearest')),
        ('Supervised', 'Linear edit tagger', simple_csv(PUB / 'baseline_per_example.csv', 'linear_edit')),
        ('Supervised', 'Character GRU', simple_csv(PUB / 'gru_v2_per_example.csv', 'gru_v2_productive')),
        ('Supervised', 'GRU--linear fusion', per_example_csv(pe, 'fusion_productive')),
        ('Adapted generator', 'Auxiliary 1.7B', per_example_csv(pe, 'qwen17_aux16000_e10_fallback')),
        ('Adapted generator', 'Context-gated 1.7B', per_example_csv(pe, 'qwen17_context_fallback')),
        ('Adapted generator', 'Learner-only 4B', per_example_csv(pe, 'qwen4b_selected_primary')),
        ('Adapted generator', 'Auxiliary 4B, top-beam rule (BF16)', per_example_csv(pe, 'qwen4b_aux16k_selected_primary')),
    ]
    for backend, name in [('qwen3.7-max', 'Qwen3.7 Max'), ('gpt-5.4-low', 'GPT-5.4 Low')]:
        for system, label in [('Direct LLM baseline', 'Direct prompting'), ('LGR-MAS', 'LGR-MAS'),
                              ('Evidence-gated LGR-MAS', 'Evidence-gated LGR-MAS')]:
            rows.append(('Prompted LLM', f'{label}, {name}', hosted(system, backend)))
    return rows


def latex_rows(rows, bold=None):
    out, group = [], None
    for g, name, mat in rows:
        if g != group:
            if group is not None:
                out.append('\\midrule')
            group = g
        vals = 100 * mat.mean(0)
        cells = [f'{v:.2f}' for v in vals]
        if bold == name:
            cells = [f'\\textbf{{{c}}}' for c in cells]
            name = f'\\textbf{{{name}}}'
        out.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(out)


if __name__ == '__main__':
    for g, name, mat in baselines():
        print(f'{g:18s} {name:40s}', ' '.join(f'{v:6.2f}' for v in 100 * mat.mean(0)))


GROUP_LABEL = {'Retrieval': 'Retrieval', 'Supervised': 'Supervised', 'Adapted generator': 'Adapted generator',
               'Prompted LLM': 'Prompted LLM, hosted', 'External': 'Fine-tuned external model',
               'Local LLM': 'Prompted LLM, local', 'Ours': 'SLA-JEC'}


# Rows reported in the caption instead of the table body (weaker variants of a shown group).
CAPTION_ONLY = ['Frequency memory', 'TF--IDF retrieval', 'Linear + nearest revision', 'Direct prompting, GPT-5.4 Low',
                'LGR-MAS, GPT-5.4 Low']


def main_table(extra, bold, path):
    """Write the main results table (table*) with baselines plus extra (group, name, matrix) rows."""
    rows = baselines() + list(extra)
    aside = {name: 100 * mat.mean(0)[5] for _, name, mat in rows if name in CAPTION_ONLY}
    rows = [r for r in rows if r[1] not in CAPTION_ONLY]
    body, group = [], None
    for g, name, mat in rows:
        if g != group:
            if group is not None:
                body.append(r'\midrule')
            body.append(r'\multicolumn{7}{@{}l}{\textit{' + GROUP_LABEL.get(g, g) + r'}}\\')
            group = g
        vals = [f'{v:.2f}' for v in 100 * mat.mean(0)]
        label = name
        if name == bold:
            vals = [r'\textbf{' + v + '}' for v in vals]
            label = r'\textbf{' + label + '}'
        body.append(r'\quad ' + label + ' & ' + ' & '.join(vals) + r' \\')
    tex = (r'\begin{table*}[!t]\centering\footnotesize' + '\n'
           + r'\caption{Test Results on CAMC-JL (\%). Neural systems average three seeds; hosted systems report one run with Qwen3.7 Max, the best of five backends, and evidence-gated LGR-MAS also with GPT-5.4 Low, the second. '
           + 'Frequency memory, TF--IDF retrieval, the linear localizer with nearest-revision retrieval, and Direct prompting and LGR-MAS with GPT-5.4 Low had ExactMatch values of '
           + ', '.join(f'{aside[n]:.2f}' for n in CAPTION_ONLY[:-1]) + f', and {aside[CAPTION_ONLY[-1]]:.2f}, respectively. '
           + r'Soft and Exact denote SoftMatch and ExactMatch}' + '\n'
           + r'\label{tab:main}' + '\n' + r'\setlength{\tabcolsep}{6pt}' + '\n'
           + r'\begin{tabular}{@{}lrrrrrr@{}}\toprule' + '\n'
           + r'Method & SM & SO & RM & RS & Soft & Exact\\\midrule' + '\n' + '\n'.join(body)
           + '\n' + r'\bottomrule\end{tabular}' + '\n' + r'\end{table*}' + '\n')
    open(path, 'w', encoding='utf-8').write(tex)
    return tex
