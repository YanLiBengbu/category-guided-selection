"""Per-example result tables and paired source-unit cluster bootstrap.

We average seed outcomes within each test example, take paired differences, and resample the
194 available source-unit clusters with replacement (all examples of a drawn cluster are kept).
Simultaneous intervals use percentile endpoints alpha/(2*6) and 1-alpha/(2*6) over six metrics,
as in the original study (20,000 draws, seed 20260928).
"""
import csv
from collections import defaultdict

import numpy as np

from common import DATA, METRICS, test_rows

PUB = DATA / 'published'
TEST_IDS = [f'CAMCJL-T-{i + 1:04d}' for i in range(559)]


def clusters():
    rows = {r['release_id']: int(r['available_unit_rank']) for r in csv.DictReader(open(PUB / 'test_cluster_order.csv', encoding='utf-8'))}
    return np.array([rows[i] for i in TEST_IDS])


def subsets():
    rows = {r['release_id']: r for r in csv.DictReader(open(PUB / 'historical_subsets.csv', encoding='utf-8'))}
    return dict(pair_seen=np.array([rows[i]['pair_seen'] == '1' for i in TEST_IDS]),
                no_shared_unit=np.array([rows[i]['no_shared_available_unit'] == '1' for i in TEST_IDS]))


def per_example_csv(path, method, metrics=METRICS):
    """Example x metric matrix (seed-averaged) from a published per-example csv."""
    acc = defaultdict(list)
    for r in csv.DictReader(open(path, encoding='utf-8')):
        if r['method'] == method:
            acc[r['release_id']].append([float(r[m]) for m in metrics])
    assert len(acc) == 559, (method, len(acc))
    return np.array([np.mean(acc[i], axis=0) for i in TEST_IDS])


def hosted(system, backend, metrics=METRICS):
    acc = {}
    for r in csv.DictReader(open(DATA / 'hosted_per_example_metrics.csv', encoding='utf-8')):
        if r['system'] == system and r['backend'] == backend:
            acc[r['release_id']] = [float(r[m]) for m in metrics]
    return np.array([acc[i] for i in TEST_IDS])


def from_scores(score_rows_by_seed):
    """score_rows_by_seed: {seed: {release_id: metric dict}} -> seed-averaged matrix."""
    seeds = sorted(score_rows_by_seed)
    return np.array([np.mean([[score_rows_by_seed[s][i][m] for m in METRICS] for s in seeds], axis=0)
                     for i in TEST_IDS])


def bootstrap(a, b, draws=20000, seed=20260928, mask=None, families=6):
    """Paired cluster bootstrap of mean(a - b) in percentage points.

    Returns per metric: difference, marginal 95% interval, simultaneous 95% interval.
    """
    cl = clusters()
    diff = 100 * (a - b)
    if mask is not None:
        diff, cl = diff[mask], cl[mask]
    ids = np.unique(cl)
    index = {c: np.where(cl == c)[0] for c in ids}
    sums = np.array([diff[index[c]].sum(0) for c in ids])
    counts = np.array([len(index[c]) for c in ids])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(draws, len(ids)))
    means = sums[pick].sum(1) / counts[pick].sum(1)[:, None]
    lo, hi = 0.05 / 2, 1 - 0.05 / 2
    slo, shi = 0.05 / (2 * families), 1 - 0.05 / (2 * families)
    out = {}
    for j, m in enumerate(METRICS[:diff.shape[1]]):
        out[m] = dict(diff=float(diff[:, j].mean()),
                      ci=(float(np.quantile(means[:, j], lo)), float(np.quantile(means[:, j], hi))),
                      sim=(float(np.quantile(means[:, j], slo)), float(np.quantile(means[:, j], shi))))
    return out


def show(res, label=''):
    print(label)
    for m, v in res.items():
        print(f"  {m:20s} {v['diff']:+7.3f}  CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]  SIM [{v['sim'][0]:+.3f}, {v['sim'][1]:+.3f}]")


if __name__ == '__main__':
    pe = PUB / 'historical_per_example.csv'
    aux = per_example_csv(pe, 'qwen4b_aux16k_selected_primary')
    src = per_example_csv(pe, 'qwen4b_selected_primary')
    print('aux means', np.round(100 * aux.mean(0), 2), 'source-only means', np.round(100 * src.mean(0), 2))
    show(bootstrap(aux, src), 'aux16k minus source-only (published: Exact +5.069 SIM [1.179, 9.324])')
