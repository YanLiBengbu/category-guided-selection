"""Candidate generation and offline selection for fixed-interval replacement.

A row holds one input with its fixed interval and the compact candidate:
key, text, error_type, selected_start, selected_end, pred_span, pred_revision.
run_candidates keeps every beam and the normalized log likelihood of every
distinct candidate (beams, compact candidate and the unchanged source span).
"""
import os
import time

from common import generate_nbest, loglik, prompt_messages, valid


def messages_for(row, style, memory=None, k=4, exclude=None):
    s, e = row['selected_start'], row['selected_end']
    demos = memory.query(row['text'], row['error_type'], s, e, k=k, exclude=exclude) if memory and k else ()
    return prompt_messages(row['text'], row['error_type'], s, e, style=style, demos=demos)


def run_candidates(model, tok, rows, style='original', memory=None, k=4, log=print,
                   batch_size=int(os.environ.get('JGEC_GEN_BATCH', 4))):
    started = time.perf_counter()
    override = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gen_batch.txt')
    if os.path.exists(override):  # lets a running queue change the batch size of later steps
        batch_size = int(open(override).read().strip())
    msgs = [messages_for(r, style, memory, k) for r in rows]
    requests = [dict(messages=m, start=r['selected_start'], end=r['selected_end'],
                     source=r['text'][r['selected_start']:r['selected_end']]) for r, m in zip(rows, msgs)]
    beams = generate_nbest(model, tok, requests, batch_size=batch_size)
    log(f'generated {len(rows)} in {time.perf_counter() - started:.1f}s')
    pairs, owners = [], []
    for i, (r, b) in enumerate(zip(rows, beams)):
        source = requests[i]['source']
        strings = [x['text'] for x in b if x['valid']] + [r['pred_revision'], source]
        for text in dict.fromkeys(strings):
            pairs.append((msgs[i], text))
            owners.append((i, text))
    values = loglik(model, tok, pairs, batch_size=8)
    scores = [dict() for _ in rows]
    for (i, text), v in zip(owners, values):
        scores[i][text] = v
    log(f'scored {len(pairs)} candidates, total {time.perf_counter() - started:.1f}s')
    return [dict(key=r['key'], source=requests[i]['source'], compact=r['pred_revision'], beams=beams[i],
                 scores=scores[i], selected_start=r['selected_start'], selected_end=r['selected_end'])
            for i, r in enumerate(rows)]


# ------------------------------------------------------------------ selection rules
def choose_original(rec):
    """Original rule: top beam if valid and strictly more likely than the compact candidate."""
    top = rec['beams'][0]
    if top['valid'] and rec['scores'][top['text']] > rec['scores'][rec['compact']]:
        return top['text']
    return rec['compact']


def choose_nbest(rec, include_compact=True):
    """Most likely valid candidate among all beams and the compact candidate; ties keep the compact one."""
    options = [rec['compact']] if include_compact and rec['compact'] else []
    options += [b['text'] for b in rec['beams'] if b['valid']]
    if not options:
        return rec['compact']
    return max(options, key=lambda t: rec['scores'][t])


def ensemble_scores(records):
    """Average normalized log likelihood across models for candidates every model scored."""
    common = set(records[0]['scores'])
    for r in records[1:]:
        common &= set(r['scores'])
    return {t: sum(r['scores'][t] for r in records) / len(records) for t in common}


def choose_ensemble(records, include_compact=True):
    """Union of valid beams from all models, ranked by the mean likelihood across models."""
    scores = ensemble_scores(records)
    compact = records[0]['compact']
    options = [compact] if include_compact and compact and compact in scores else []
    for r in records:
        options += [b['text'] for b in r['beams'] if b['valid'] and b['text'] in scores]
    options = list(dict.fromkeys(options))
    if not options:
        return compact
    return max(options, key=lambda t: scores[t])


def is_valid(rec, text):
    return valid(text, rec['source'])
