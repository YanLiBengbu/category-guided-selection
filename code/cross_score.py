"""Score a union of candidates under several adapters (likelihood ensemble support).

For every example we take the union of the valid beams of the --union runs, the compact candidate,
and the source span. Each --score run (same base model per call) then adds the normalized log
likelihood of every union candidate under its own adapter and prompt. Scores and beams are keyed by
run directory name, so files from calls with different base models can be merged with merge_scores().

Development:  python cross_score.py dev OUT --score cv/orig_k0 cv/guided_k4 [--union ...]
Test:         python cross_score.py test OUT --score final/guided_k4 [--union ...]
(run directories are relative to results/)
"""
import argparse
import gc
import time
from pathlib import Path

from common import BASES, DATA, RESULTS, Memory, dev_rows, folds, read, write_lines, load, loglik, \
    tokenizer, load_model, torch_modules
from pipeline import messages_for


def run_info(name):
    d = RESULTS / name
    cfg = load(d / 'config.json')
    return dict(name=name, dir=d, style=cfg['style'], k=cfg['k'], model=cfg.get('model', 'qwen3'),
                zero_shot=cfg.get('zero_shot', False))


def units_for(split):
    dev = dev_rows()
    if split == 'dev':
        return [(f'fold{f}', f'adapter_fold{f}', [dev[i] for i in fit]) for f, (fit, _) in enumerate(folds())]
    return [(f'seed{s}', f'adapter_seed{s}', dev) for s in [20260927, 20260928, 20260929]]


def union_records(split, unit, union_runs, rows_by_key=None, morph=False):
    recs = [read(r['dir'] / f'{unit}.jsonl') for r in union_runs]
    if morph:
        from morph_candidates import candidates as morph_candidates
    merged = []
    for group in zip(*recs):
        assert len({g['key'] for g in group}) == 1
        cands = []
        for g in group:
            cands += [b['text'] for b in g['beams'] if b['valid']]
        cands += [group[0]['compact'], group[0]['source']]
        extra = []
        if morph:
            row = rows_by_key[group[0]['key']]
            extra = morph_candidates(row['text'], group[0]['selected_start'], group[0]['selected_end'], row['error_type'])
            cands += extra
        merged.append(dict(key=group[0]['key'], source=group[0]['source'], compact=group[0]['compact'],
                           selected_start=group[0]['selected_start'], selected_end=group[0]['selected_end'],
                           beams={r['name']: g['beams'] for r, g in zip(union_runs, group)},
                           morph=extra, candidates=list(dict.fromkeys(cands)), scores={},
                           own={r['name']: g.get('scores', {}) for r, g in zip(union_runs, group)}))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('split', choices=['dev', 'test'])
    ap.add_argument('out')
    ap.add_argument('--score', nargs='+', required=True)
    ap.add_argument('--union', nargs='+')
    ap.add_argument('--morph', action='store_true', help='add category-specific morphological candidates')
    ap.add_argument('--no-reuse', action='store_true',
                    help='rescore every candidate; by default the test split reuses the scores that each run '
                         'already computed for its own candidates with the same adapter, prompt, and loglik()')
    ap.add_argument('--reuse', help='earlier cross-scored file: reuse its score of the same run, key, and '
                                    'candidate (same adapter, prompt, and loglik(); only the batch composition differs)')
    a = ap.parse_args()
    torch, _, peft = torch_modules()
    previous = {r['key']: r['scores'] for r in read(a.reuse)} if a.reuse else {}
    scorers = [run_info(n) for n in a.score]
    union_runs = [run_info(n) for n in (a.union or a.score)]
    assert len({r['model'] for r in scorers}) == 1, 'one base model per call'
    path = BASES[scorers[0]['model']]
    tok = tokenizer(path)
    inputs = read(DATA / ('orig4b/inputs.jsonl' if a.split == 'dev' else 'orig4b_test/inputs.jsonl'))
    rows_by_key = {}
    for r in inputs:
        key = f"{r['fold']}:{r['parent_index']}:{r['view']}" if a.split == 'dev' else f"{r['seed']}:{r['release_id']}"
        rows_by_key[key] = r
    base = load_model(path)
    results = []
    for unit, adapter_name, memory_rows in units_for(a.split):
        merged = union_records(a.split, unit, union_runs, rows_by_key, a.morph)
        model = None
        for i, run in enumerate(scorers):
            t = time.perf_counter()
            if run['zero_shot']:
                assert len(scorers) == 1, 'score zero-shot runs in their own call'
                model = base
            else:
                adapter = str(run['dir'] / adapter_name)
                if model is None:
                    model = peft.PeftModel.from_pretrained(base, adapter, adapter_name=f'r{i}')
                else:
                    model.load_adapter(adapter, adapter_name=f'r{i}')
                model.set_adapter(f'r{i}')
            memory = Memory(memory_rows) if run['style'] in ('guided', 'demos') and run['k'] else None
            pairs, owners, reused = [], [], 0
            reuse = a.split == 'test' and not a.no_reuse
            for j, m in enumerate(merged):
                msgs = messages_for(rows_by_key[m['key']], run['style'], memory, run['k'])
                own = dict(previous.get(m['key'], {}).get(run['name'], {}))
                if reuse:
                    own.update(m['own'].get(run['name'], {}))
                for c in m['candidates']:
                    if c in own:
                        merged[j]['scores'].setdefault(run['name'], {})[c] = own[c]
                        reused += 1
                        continue
                    pairs.append((msgs, c))
                    owners.append((j, c))
            values = loglik(model, tok, pairs, batch_size=8)
            for (j, c), v in zip(owners, values):
                merged[j]['scores'].setdefault(run['name'], {})[c] = v
            print(unit, run['name'], len(pairs), 'pairs scored,', reused, 'reused',
                  round(time.perf_counter() - t, 1), 's', flush=True)
        if model is not base:
            base = model.unload()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        for m in merged:
            m.pop('own', None)
        results += merged
        write_lines(a.out, results)


def merge_scores(paths):
    """Merge cross-scored files that share the same union candidates (e.g. one file per base model)."""
    files = [read(p) for p in paths]
    out = []
    for group in zip(*files):
        assert len({g['key'] for g in group}) == 1
        rec = dict(group[0])
        rec['scores'] = {}
        rec['beams'] = {}
        for g in group:
            assert g['candidates'] == group[0]['candidates']
            rec['scores'].update(g['scores'])
            rec['beams'].update(g['beams'])
        out.append(rec)
    return out


if __name__ == '__main__':
    main()
