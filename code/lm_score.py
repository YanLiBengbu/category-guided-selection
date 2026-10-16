"""Fluency scores of candidate corrections under a base language model (no adapter, no prompt).

For each candidate y of a fixed interval [s, e), we insert y into a local window of the input
(W characters on each side) and compute the summed token log probability of the window text.
The feature is the difference from the same window with the original span, so the shared context
cancels. Scores are stored as rec['lm'][name][y]. Works on cross-scored candidate files.

Usage: python lm_score.py IN.jsonl OUT.jsonl qwen3|qwen35|sarashina [W] [REUSE.jsonl]
REUSE: an earlier scored file; its raw window scores of the same key and candidate are reused.
"""
import sys
import time

from common import BASES, DATA, read, write_lines, load_model, torch_modules
from transformers import AutoTokenizer


def windows(text, s, e, y, w):
    left, right = text[max(0, s - w):s], text[e:e + w]
    return ''.join((left + y + right).split())


def main(src, dest, name, w=24, reuse=None):
    torch, _, _ = torch_modules()
    path = BASES[name]
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = load_model(path)
    model.eval()
    records = read(src)
    rows = {}
    for f in [DATA / 'orig4b/inputs.jsonl', DATA / 'orig4b_test/inputs.jsonl']:
        for r in read(f):
            key = f"{r['fold']}:{r['parent_index']}:{r['view']}" if 'fold' in r else f"{r['seed']}:{r['release_id']}"
            rows[key] = r
    previous = {r['key']: r.get('lm_raw', {}).get(name, {}) for r in read(reuse)} if reuse else {}
    texts, owners = [], []
    for j, rec in enumerate(records):
        row = rows[rec['key']]
        s, e = rec['selected_start'], rec['selected_end']
        for y in rec['candidates']:
            if y in previous.get(rec['key'], {}):
                rec.setdefault('lm_raw', {}).setdefault(name, {})[y] = previous[rec['key']][y]
                continue
            texts.append(windows(row['text'], s, e, y, w))
            owners.append((j, y))
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    encoded = [[bos] + tok.encode(t, add_special_tokens=False) for t in texts]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    values = [0.0] * len(encoded)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, len(order), 16):
            idx = order[first:first + 16]
            width = max(len(encoded[i]) for i in idx)
            ids = torch.full((len(idx), width), tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for k, i in enumerate(idx):
                ids[k, :len(encoded[i])] = torch.tensor(encoded[i])
                mask[k, :len(encoded[i])] = 1
            ids, mask = ids.cuda(), mask.cuda()
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]
            lp = (lp * mask[:, 1:]).sum(1).cpu().tolist()
            for k, i in enumerate(idx):
                values[i] = lp[k]
            # Full-vocabulary logits of varying width fragment the cache; release them every batch.
            del logits, ids, mask
            torch.cuda.empty_cache()
    for (j, y), v in zip(owners, values):
        records[j].setdefault('lm_raw', {}).setdefault(name, {})[y] = v
    for rec in records:
        raw = rec['lm_raw'][name]
        rec.setdefault('lm', {})[name] = {y: raw[y] - raw[rec['source']] for y in raw}
    write_lines(dest, records)
    print(name, len(texts), 'windows', round(time.perf_counter() - t0, 1), 's')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 24,
         sys.argv[5] if len(sys.argv) > 5 else None)
