"""External baselines of the review round, trained on the same data as SLA-JEC.

deberta  Pretrained-encoder localizer: the character-level Japanese DeBERTa-v2 base model
         (ku-nlp/deberta-v2-base-japanese-char-wwm) fine-tuned as a category-conditioned BIO tagger on the
         200 development examples with a weighted cross-entropy (O 1, B 10, I 5). The number of epochs (20, 40, 60) is chosen by out-of-fold SpanMatch on the
         three grouped development folds; three seeds are then refit on all development examples. Intervals are
         decoded with the same interval score as the compact frontend. Replacements come from the base
         generator of the same seed pipeline (first valid beam of four) at the DeBERTa interval; beams are
         reused when the interval equals the frontend interval of that seed.
t5       Fine-tuned sequence-to-sequence model: the Japanese T5 (retrieva-jp/t5-base-long) maps
         "category: text" to "span => replacement". Stage one replays the 16,000 auxiliary pairs in the fixed
         order of the auxiliary stage (1,000 updates of 16); stage two fits the 200 development examples for
         10 epochs with three seeds. Decoding uses four beams. An emitted span is mapped back to the input
         ignoring whitespace and NFKC differences introduced by the tokenizer.

Outputs: results/ext/{deberta,t5}_test.jsonl and _matrix.npy (559 x 6, seed-averaged, TEST_IDS order);
results/ext/deberta_cv.json (epoch selection). Usage: python ext_baselines.py deberta|t5
"""
import json
import math
import random
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (CATEGORY_JA, DATA, METRICS, MODELS, RESULTS, SEEDS, dev_rows, folds, load_model, prompt_messages,
                    read, score, seed_all, test_rows, tokenizer, torch_modules, generate_nbest, write_lines)
from stats import TEST_IDS

OUT = RESULTS / 'ext'
DEBERTA = MODELS / 'deberta_v2_base_japanese_char_wwm'
T5 = MODELS / 't5_base_long_japanese'
MAX_SPAN = 40


# ------------------------------------------------------------------ interval decoding (frontend score)
def best_interval(logq):
    """argmax over 1..40-character intervals of log qB(s)/qO(s) + sum log qI(i)/qO(i); shorter, then earlier."""
    b = logq[:, 1] - logq[:, 0]
    i_ = logq[:, 2] - logq[:, 0]
    best, arg = -math.inf, None
    n = len(logq)
    for s in range(n):
        total = b[s]
        for e in range(s + 1, min(n, s + MAX_SPAN) + 1):
            if e > s + 1:
                total += i_[e - 1]
            if total > best + 1e-12:
                best, arg = total, (s, e)
    return arg


# ------------------------------------------------------------------ DeBERTa BIO localizer
def encode_bio(vocab, row, cat):
    """Character ids of category prefix and text; text positions map back to character offsets."""
    unk, cls, sep = vocab['[UNK]'], vocab['[CLS]'], vocab['[SEP]']
    prefix = [vocab.get(ch, unk) for ch in CATEGORY_JA.get(cat, cat)]
    ids, pos = [cls] + prefix + [sep], []
    for k, ch in enumerate(row['text']):
        if ch.isspace():
            continue
        pos.append((len(ids), k))
        ids.append(vocab.get(ch, unk))
    ids.append(sep)
    return ids, pos


def bio_labels(row, ids, pos):
    lab = [-100] * len(ids)
    s, e = row['gold_span_start'], row['gold_span_end']
    for t, k in pos:
        lab[t] = 1 if k == s else (2 if s < k < e else 0)
    return lab


def train_deberta(rows, epochs, seed, log):
    torch, transformers, _ = torch_modules()
    seed_all(seed)
    vocab = {l.rstrip('\n'): i for i, l in enumerate(open(DEBERTA / 'vocab.txt', encoding='utf-8'))}
    model = transformers.AutoModelForTokenClassification.from_pretrained(DEBERTA, num_labels=3,
                                                                       local_files_only=True).cuda()
    data = []
    for r in rows:
        ids, pos = encode_bio(vocab, r, r['error_type'])
        data.append((ids, bio_labels(r, ids, pos)))
    batch = 16
    steps = math.ceil(len(data) / batch) * epochs
    # As for the frontend, a weighted BIO cross-entropy offsets the rarity of B and I; the new head learns faster.
    weight = torch.tensor([1.0, 10.0, 5.0]).cuda()
    head = [p for n, p in model.named_parameters() if n.startswith('classifier')]
    body = [p for n, p in model.named_parameters() if not n.startswith('classifier')]
    opt = torch.optim.AdamW([dict(params=body, lr=5e-5), dict(params=head, lr=1e-3)], weight_decay=0.01)
    sched = transformers.get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    rng = random.Random(seed)
    model.train()
    step = 0
    for ep in range(epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for first in range(0, len(order), batch):
            chunk = [data[i] for i in order[first:first + batch]]
            width = max(len(x[0]) for x in chunk)
            ids = torch.tensor([x[0] + [0] * (width - len(x[0])) for x in chunk]).cuda()
            mask = torch.tensor([[1] * len(x[0]) + [0] * (width - len(x[0])) for x in chunk]).cuda()
            lab = torch.tensor([x[1] + [-100] * (width - len(x[1])) for x in chunk]).cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 3), lab.reshape(-1), weight=weight,
                                                     ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        if (ep + 1) % 10 == 0:
            log(f'deberta seed {seed} epoch {ep + 1} loss {float(loss.detach()):.4f}')
    model.eval()
    return model, vocab


def predict_intervals(model, vocab, rows):
    torch = torch_modules()[0]
    out = []
    with torch.inference_mode():
        for r in rows:
            ids, pos = encode_bio(vocab, r, r['error_type'])
            logits = model(input_ids=torch.tensor([ids]).cuda()).logits[0].float()
            lp = torch.log_softmax(logits, -1).cpu().numpy()
            logq = np.tile(np.log(np.array([1.0 - 2e-9, 1e-9, 1e-9])), (len(r['text']), 1))  # whitespace: outside
            for t, k in pos:
                logq[k] = lp[t]
            out.append(best_interval(logq))
    return out


def run_deberta(log):
    torch = torch_modules()[0]
    dev = dev_rows()
    cv_path = OUT / 'deberta_cv.json'
    if cv_path.exists():
        cv = json.loads(cv_path.read_text())
    else:
        cv = {}
        for epochs in (20, 40, 60):
            hits = []
            for f, (fit, held) in enumerate(folds()):
                model, vocab = train_deberta([dev[i] for i in fit], epochs, SEEDS[0] + f, log)
                pred = predict_intervals(model, vocab, [dev[i] for i in held])
                hits += [float(p == (dev[i]['gold_span_start'], dev[i]['gold_span_end'])) for p, i in zip(pred, held)]
                del model
                torch.cuda.empty_cache()
            cv[str(epochs)] = 100 * float(np.mean(hits))
            log(f'deberta cv epochs {epochs}: SpanMatch {cv[str(epochs)]:.2f}')
        cv_path.write_text(json.dumps(cv, indent=1))
    epochs = int(max(cv, key=lambda k: (cv[k], -int(k))))
    test = test_rows()
    ipath = OUT / 'deberta_intervals.json'
    intervals = json.loads(ipath.read_text()) if ipath.exists() else {}
    for seed in SEEDS:
        if str(seed) in intervals:
            continue
        model, vocab = train_deberta(dev, epochs, seed, log)
        intervals[str(seed)] = predict_intervals(model, vocab, test)
        ipath.write_text(json.dumps(intervals))
        del model
        torch.cuda.empty_cache()
    log(f'deberta: {epochs} epochs; generating replacements')
    # Replacements from the base generator of each seed pipeline at the DeBERTa intervals.
    _, _, peft = torch_modules()
    tok = tokenizer()
    choices = {}
    for seed in SEEDS:
        frontend = {r['key'].split(':')[1]: r for r in read(RESULTS / 'final' / 'orig_k0' / f'seed{seed}.jsonl')}
        todo, reqs = [], []
        for i, (rid, (s, e)) in enumerate(zip(TEST_IDS, intervals[str(seed)])):
            rec = frontend[rid]
            if (rec['selected_start'], rec['selected_end']) == (s, e):
                choices[(seed, rid)] = (rec['source'], next((b['text'] for b in rec['beams'] if b['valid']), ''))
                continue
            r = test[i]
            todo.append((rid, r['text'][s:e]))
            reqs.append(dict(messages=prompt_messages(r['text'], r['error_type'], s, e, 'original'), start=s, end=e,
                             source=r['text'][s:e]))
        if reqs:
            base = load_model()
            model = peft.PeftModel.from_pretrained(base, str(RESULTS / 'final' / 'orig_k0' / f'adapter_seed{seed}'))
            beams = generate_nbest(model, tok, reqs, batch_size=4)
            for (rid, src), b in zip(todo, beams):
                choices[(seed, rid)] = (src, next((x['text'] for x in b if x['valid']), ''))
            del model, base
            torch.cuda.empty_cache()
        log(f'deberta seed {seed}: {len(reqs)} new intervals generated, {559 - len(reqs)} reused')
    finish('deberta', choices, dict(epochs=epochs, cv=cv))


# ------------------------------------------------------------------ Japanese T5 seq2seq
SEP = ' => '


def t5_source(row):
    return f"{CATEGORY_JA.get(row['error_type'], row['error_type'])}: {row['text']}"


def t5_target(row):
    return row['text'][row['gold_span_start']:row['gold_span_end']] + SEP + row['gold_revision']


def t5_fit(model, tok, rows, updates, peak, seed, log, tag):
    torch, transformers, _ = torch_modules()
    opt = transformers.optimization.Adafactor(model.parameters(), lr=peak, scale_parameter=False,
                                              relative_step=False, warmup_init=False)
    sched = transformers.get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * updates)), updates)
    rng = random.Random(seed)
    order = []
    model.train()
    for step in range(1, updates + 1):
        if isinstance(rows, list) and not callable(rows):
            if len(order) < 16:
                perm = list(range(len(rows)))
                rng.shuffle(perm)
                order += perm
            chunk = [rows[i] for i in order[:16]]
            order = order[16:]
        else:
            chunk = rows(step)
        enc = tok([t5_source(r) for r in chunk], return_tensors='pt', padding=True, truncation=True, max_length=512)
        lab = tok([t5_target(r) for r in chunk], return_tensors='pt', padding=True, truncation=True, max_length=128)
        labels = lab['input_ids'].masked_fill(lab['attention_mask'] == 0, -100)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids=enc['input_ids'].cuda(), attention_mask=enc['attention_mask'].cuda(),
                         labels=labels.cuda()).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            log(f't5 {tag} step {step}/{updates} loss {float(loss):.4f}')


def recover_span(text, span):
    """Map an emitted span to an input substring, ignoring whitespace and NFKC differences."""
    if not span:
        return ''
    if span in text:
        return span
    key = lambda ch: unicodedata.normalize('NFKC', ch)
    chars = [(k, key(ch)) for k, ch in enumerate(text) if not ch.isspace()]
    flat = ''.join(c for _, c in chars)
    target = ''.join(ch for ch in unicodedata.normalize('NFKC', span) if not ch.isspace())
    if not target:
        return ''
    at = flat.find(target)
    if at < 0:
        return ''
    # map the normalized offsets back to the original characters
    acc, start, end = 0, None, None
    for k, c in chars:
        if start is None and acc >= at:
            start = k
        acc += len(c)
        if acc >= at + len(target):
            end = k + 1
            break
    return text[start:end] if start is not None and end is not None else ''


def run_t5(log):
    torch, transformers, _ = torch_modules()
    tok = transformers.AutoTokenizer.from_pretrained(T5, local_files_only=True)
    stage1 = OUT / 't5_stage1'
    if not (stage1 / 'config.json').exists():
        seed_all(20260928)
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(T5, local_files_only=True).cuda()
        pool = read(DATA / 'teacher_train.jsonl') + read(DATA / 'synthetic_train.jsonl')
        order = np.fromfile(DATA / 'orig4b/auxiliary_indices.i64', dtype='<i8')
        t5_fit(model, tok, lambda step: [pool[int(i)] for i in order[(step - 1) * 16:step * 16]], 1000, 5e-4,
               20260928, log, 'stage1')
        model.save_pretrained(str(stage1))
        del model
        torch.cuda.empty_cache()
    dev, test = dev_rows(), test_rows()
    choices = {}
    for seed in SEEDS:
        seed_all(seed)
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(stage1, local_files_only=True).cuda()
        t5_fit(model, tok, dev, math.ceil(len(dev) / 16) * 10, 3e-4, seed, log, f'seed {seed}')
        model.eval()
        with torch.inference_mode():
            for first in range(0, len(test), 16):
                chunk = test[first:first + 16]
                enc = tok([t5_source(r) for r in chunk], return_tensors='pt', padding=True, truncation=True,
                          max_length=512)
                gen = model.generate(input_ids=enc['input_ids'].cuda(), attention_mask=enc['attention_mask'].cuda(),
                                     num_beams=4, max_new_tokens=96)
                for r, rid, g in zip(chunk, TEST_IDS[first:first + 16], tok.batch_decode(gen, skip_special_tokens=True)):
                    span, _, rev = g.partition(SEP.strip())
                    span = recover_span(r['text'], span.strip())
                    rev = rev.strip()
                    choices[(seed, rid)] = (span if rev else '', rev if span else '')
        del model
        torch.cuda.empty_cache()
        log(f't5 seed {seed} decoded')
    finish('t5', choices, {})


def finish(name, choices, meta):
    test = test_rows()
    per = defaultdict(dict)
    rows = []
    for (seed, rid), (span, rev) in sorted(choices.items()):
        g = test[int(rid.split('-')[-1]) - 1]
        per[seed][rid] = score(g, span, rev)
        rows.append(dict(key=f'{seed}:{rid}', span=span, revision=rev))
    mat = np.array([np.mean([[per[s][r][m] for m in METRICS] for s in per], axis=0) for r in TEST_IDS])
    write_lines(OUT / f'{name}_test.jsonl', rows)
    np.save(OUT / f'{name}_matrix.npy', mat)
    seeds = {s: 100 * float(np.mean([per[s][r]['ExactMatch'] for r in TEST_IDS])) for s in per}
    (OUT / f'{name}_summary.json').write_text(json.dumps(
        dict(meta, means={m: 100 * float(v) for m, v in zip(METRICS, mat.mean(0))}, exact_by_seed=seeds), indent=1))
    print(name, {m: round(100 * v, 2) for m, v in zip(METRICS, mat.mean(0))}, 'seeds', seeds, flush=True)


def main(which):
    OUT.mkdir(exist_ok=True)

    def log(msg):
        print(time.strftime('%H:%M'), msg, flush=True)

    {'deberta': run_deberta, 't5': run_t5}[which](log)


if __name__ == '__main__':
    main(sys.argv[1])
