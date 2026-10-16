"""Category-aware candidate re-ranking at a fixed interval (conditional logit over candidates).

Candidates: valid beams of one or more runs plus the compact candidate.
Features of candidate y for source span s and category c:
  mean normalized log likelihood (over the available adapters) and its gap to the best candidate,
  likelihood margin over copying s, normalized character edit distance to s, compact indicator,
  best beam rank, and category-marker agreement: whether y introduces the morphological marker of c
  that s lacks (passive, causative, benefactive, aspect, honorific), flips tense or a demonstrative
  series, or changes the connective or ending (style, sentence connection).
Weights are fitted on out-of-fold development candidates; development scores use fold-wise fitting.
"""
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

from common import dev_rows, test_rows, folds, read, score, mean_scores, fmt, norm, levenshtein, valid

MARKERS = dict(
    passive=['れる', 'れた', 'れて', 'れま', 'れな', 'られ', 'され'],
    causative=['させ', 'せる', 'せた', 'せて', 'せま', 'せな'],
    benefactive=['てくれ', 'でくれ', 'てもら', 'でもら', 'てあげ', 'であげ', 'てくださ', 'でくださ', 'ていただ', 'でいただ', 'てやる'],
    aspect=['てい', 'でい', 'てき', 'でき', 'てしま', 'でしま', 'てお', 'てあ', 'つつ', 'はじめ', '始め', 'だす', '出す'],
    honorific=['お', 'ご', '御', '様', '方', 'いらっしゃ', 'おっしゃ', '召し', 'いただ', 'くださ', '申し', '参', '致', 'おり', 'なさ'],
)
PAST = re.compile(r'(た|だ|でした|ました|かった|ませんでした)[、。！？!?」]*$')
DEMO = set('こそあど')
# Grammatical dimensions used to detect changes outside the error category (off_cat).
DIMENSIONS = dict(
    aspect=['てい', 'でい', 'てき', 'でき', 'てしま', 'でしま', 'てお', 'てあ'],
    passive=MARKERS['passive'], causative=MARKERS['causative'], benefactive=MARKERS['benefactive'],
    honorific=['いらっしゃ', 'おっしゃ', '召し', 'いただ', 'くださ', '申し', '致し', 'なさ'],
    negation=['ない', 'なかっ', 'ません', 'ず'],
)
# Categories whose corrections change inflection or auxiliaries rather than the content word.
GRAMMATICAL = {'tense', 'aspect', 'passive', 'causative', 'benefactive', 'transitivity'}
KANJI = re.compile(r'[一-鿿々]')
EXTRA = set()  # features tried and not adopted: 'negation' (polarity in off_cat), 'stem' (dropped kanji)
POLITE = ['ます', 'まし', 'ませ', 'です', 'でし']
POLITE_END = re.compile(r'(ます|ました|ません|です|でした|ましょう|でしょう)[」』）)]*[。！？!?]')
PLAIN_END = re.compile(r'(る|た|だ|い|ない|う|ろう)[」』）)]*[。！？!?]')
BASE_NAMES = ['ll', 'll_gap', 'margin', 'edit', 'compact', 'rank', 'marker', 'morph', 'mem_pair', 'mem_rev',
              'off_cat', 'register', 'on_cat']
LM_NAMES = []
NAMES = list(BASE_NAMES)
USE_MORPH = [True]
DROP = set()  # feature names zeroed out for ablations


def marker_feature(cat, src, y):
    s, t = norm(src), norm(y)
    if cat in MARKERS:
        gain = sum(t.count(m) for m in MARKERS[cat]) - sum(s.count(m) for m in MARKERS[cat])
        return float(np.clip(gain, -1, 1))
    if cat == 'tense':
        return float(bool(PAST.search(s)) != bool(PAST.search(t)))
    if cat == 'demonstrative':
        a = [ch for ch in s if ch in DEMO][:1]
        b = [ch for ch in t if ch in DEMO][:1]
        return float(a != b)
    if cat in ('style', 'sentence_connection'):
        return float(s[-2:] != t[-2:] or s[:2] != t[:2])
    return 0.0


def changed_dimensions(src, y):
    """Grammatical dimensions whose markers differ between the source span and the candidate."""
    s, t = norm(src), norm(y)
    out = {d for d, marks in DIMENSIONS.items() if (d != 'negation' or 'negation' in EXTRA)
           and sum(s.count(m) for m in marks) != sum(t.count(m) for m in marks)}
    if bool(PAST.search(s)) != bool(PAST.search(t)):
        out.add('tense')
    if [ch for ch in s if ch in DEMO][:1] != [ch for ch in t if ch in DEMO][:1]:
        out.add('demonstrative')
    return out


def off_category(cat, src, y):
    """Number of changed dimensions other than the error category (capped at two)."""
    return float(min(2, len(changed_dimensions(src, y) - {cat})))


def stem_change(cat, src, y):
    """Share of the source kanji that the candidate drops, for grammatical categories only."""
    if cat not in GRAMMATICAL:
        return 0.0
    kanji = set(KANJI.findall(src))
    return len(kanji - set(KANJI.findall(y))) / len(kanji) if kanji else 0.0


def politeness(text):
    """Share of polite sentence endings in a text (0.5 when it has no recognizable ending)."""
    polite, plain = len(POLITE_END.findall(text)), len(PLAIN_END.findall(text))
    return polite / (polite + plain) if polite + plain else 0.5


def register_shift(src, y, context_politeness):
    """Positive when the candidate moves the span toward the register of the surrounding text."""
    p = lambda t: float(any(m in norm(t) for m in POLITE))
    return (p(y) - p(src)) * (2 * context_politeness - 1)


def candidates(rec, runs):
    """Candidate list with features; rec may be a single-run record or a cross-scored record."""
    if 'candidates' in rec:  # cross-scored
        beams = [b for r in runs for b in rec['beams'][r]]
        scores = {t: np.mean([rec['scores'][r][t] for r in runs]) for t in rec['scores'][runs[0]]}
        per_run = [rec['scores'][r] for r in runs]
    else:
        beams = rec['beams']
        scores = rec['scores']
        per_run = [scores]
    ranks = {}
    for i, b in enumerate(beams):
        if b['valid']:
            ranks.setdefault(b['text'], i % 4)
    options = list(ranks)
    if rec['compact'] and rec['compact'] not in ranks and rec['compact'] in scores:
        options.append(rec['compact'])
        ranks[rec['compact']] = 4
    for t in (rec.get('morph', []) if USE_MORPH[0] else []):
        if t not in ranks and t in scores and valid(t, rec['source']):
            options.append(t)
            ranks[t] = 5
    return options, scores, ranks, per_run


def memory_counts(rows):
    pair, rev = defaultdict(int), defaultdict(int)
    for r in rows:
        span = norm(r['text'][r['gold_span_start']:r['gold_span_end']])
        pair[(r['error_type'], span, norm(r['gold_revision']))] += 1
        rev[(r['error_type'], norm(r['gold_revision']))] += 1
    return pair, rev


def features(rec, cat, runs, memory=None, text=None):
    options, scores, ranks, per_run = candidates(rec, runs)
    if not options:
        return [], np.zeros((0, len(NAMES)))
    src = rec['source']
    best = max(scores[t] for t in options)
    s, e = rec['selected_start'], rec['selected_end']
    context = politeness(text[:s] + text[e:]) if text is not None else 0.5
    rows = []
    for t in options:
        rows.append([scores[t], scores[t] - best, scores[t] - scores[src],
                     levenshtein(norm(src), norm(t)) / max(len(src), len(t), 1),
                     float(t == rec['compact'] and ranks[t] == 4), min(ranks[t], 4) / 4,
                     marker_feature(cat, src, t), float(ranks[t] == 5),
                     np.log1p(memory[0].get((cat, norm(src), norm(t)), 0)) if memory else 0.0,
                     np.log1p(memory[1].get((cat, norm(t)), 0)) if memory else 0.0,
                     off_category(cat, src, t), register_shift(src, t, context),
                     float(cat in changed_dimensions(src, t))]
                    + [rec['lm'][n].get(t, 0.0) / 10.0 for n in LM_NAMES]
                    + ([pr[t] - pr[src] for pr in per_run] if len(per_run) > 1 else [])
                    + ([stem_change(cat, src, t)] if 'stem' in EXTRA else []))
    x = np.array(rows)
    for k, name in enumerate(NAMES):
        if name in DROP or (name.startswith('lm_') and 'lm' in DROP):
            x[:, k] = 0.0
    return options, x


def fit(data, l2=1.0):
    def loss(w):
        total, grad = l2 * w @ w, 2 * l2 * w
        for x, g in data:
            z = x @ w
            z = z - z.max()
            p = np.exp(z) / np.exp(z).sum()
            total -= np.log(p[g] + 1e-12)
            grad -= x[g] - p @ x
        return total, grad
    w0 = np.zeros(data[0][0].shape[1])
    w0[0] = 1.0
    return minimize(loss, w0, jac=True, method='L-BFGS-B').x


def build(split, records, runs):
    gold = dev_rows() if split == 'dev' else test_rows()
    dev = dev_rows()
    if split == 'dev':
        mem = {f'{f}': memory_counts([dev[i] for i in fit]) for f, (fit, _) in enumerate(folds())}
    items = []
    for rec in records:
        idx = rec['key'].split(':')[1]
        g = gold[int(idx)] if split == 'dev' else gold[int(idx.split('-')[-1]) - 1]
        memory = mem[rec['key'].split(':')[0]] if split == 'dev' else memory_counts(dev)
        opts, x = features(rec, g['error_type'], runs, memory, g['text'])
        lab = [j for j, t in enumerate(opts) if norm(t) == norm(g['gold_revision'])]
        items.append(dict(rec=rec, gold=g, opts=opts, x=x, label=lab[0] if lab else None,
                          unit=rec['key'].split(':')[0]))
    return items


def set_lm(names):
    LM_NAMES[:] = list(names)
    NAMES[:] = BASE_NAMES + ['lm_' + n for n in names]


def pick(item, w):
    if not len(item['opts']):
        return item['rec']['compact']
    return item['opts'][int(np.argmax(item['x'] @ w))]


def evaluate_dev(records, runs, l2=1.0):
    items = build('dev', records, runs)
    out, base = [], []
    weights = {}
    for u in sorted({i['unit'] for i in items}):
        train = [(i['x'], i['label']) for i in items if i['unit'] != u and i['label'] is not None]
        w = fit(train, l2)
        weights[u] = w
        for i in items:
            if i['unit'] != u:
                continue
            y = pick(i, w)
            out.append(score(i['gold'], i['rec']['source'] if y else '', y))
            y0 = pick(i, np.eye(i['x'].shape[1])[0] if len(i['opts']) else None)
            base.append(score(i['gold'], i['rec']['source'] if y0 else '', y0))
    return mean_scores(out), mean_scores(base), weights


def fit_all_dev(records, runs, l2=1.0):
    items = build('dev', records, runs)
    return fit([(i['x'], i['label']) for i in items if i['label'] is not None], l2)


if __name__ == '__main__':
    files = sys.argv[1:]
    records = [r for f in files for r in read(f)]
    runs = list(records[0]['scores']) if 'candidates' in records[0] else None
    set_lm(sorted(records[0].get('lm', {})))
    for l2 in [0.3, 1.0, 3.0, 10.0]:
        res, base, w = evaluate_dev(records, runs, l2)
        print(f'l2={l2:5}', 'rerank', fmt(res))
        print('         likelihood only', fmt(base))
        print('         weights fold0', dict(zip(NAMES, np.round(w[sorted(w)[0]], 2))))
