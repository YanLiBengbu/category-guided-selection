"""Shared data, scoring, prompt, model and decoding code for the SLA-JEC IEEE study.

The scorer reproduces the CAMC-JL metrics of the original study
(research/scorer/metrics.py). The original prompt reproduces
research/scripts/train_qwen_baseline.py. Generation and likelihood scoring
follow research/scripts/inference_stability_gpu.py, extended to keep the
complete beam list and to score any candidate set.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
MODELS = Path(os.environ.get('JGEC_MODELS', ROOT / 'models'))  # local copies of the base models
QWEN3_4B = MODELS / 'qwen3_4b_instruct_2507'
QWEN35_4B = MODELS / 'qwen3_5_4b'
SARASHINA_3B = MODELS / 'sarashina2_2_3b_instruct'
BASES = dict(qwen3=QWEN3_4B, qwen35=QWEN35_4B, sarashina=SARASHINA_3B)
SEEDS = [20260927, 20260928, 20260929]
METRICS = ['SpanMatch', 'SpanOverlapScore', 'RevisionMatch', 'RevisionSimilarity', 'SoftMatch', 'ExactMatch']
SHORT = dict(SpanMatch='SM', SpanOverlapScore='SO', RevisionMatch='RM', RevisionSimilarity='RS',
             SoftMatch='Soft', ExactMatch='Exact')


# ----------------------------------------------------------------------------- io
def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_lines(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dev_rows():
    rows = read(DATA / 'dev.jsonl')
    assert len(rows) == 200
    return rows


def test_rows():
    rows = read(DATA / 'test.jsonl')
    assert len(rows) == 559
    return rows


def folds():
    """Grouped development folds of the original study: list of (fit, heldout) index lists."""
    rows = list(csv.DictReader((DATA / 'development_folds.csv').open(encoding='utf-8')))
    lookup = {int(r['release_id'].split('-')[-1]) - 1: r for r in rows}
    result = []
    for fold in range(3):
        fit = [i for i in range(200) if int(lookup[i]['fold']) != fold]
        held = [i for i in range(200) if int(lookup[i]['fold']) == fold]
        assert not {lookup[i]['component'] for i in fit} & {lookup[i]['component'] for i in held}
        result.append((fit, held))
    return result


# ------------------------------------------------------------------------ scoring
def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def similarity(a, b):
    if not a and not b:
        return 1.0
    d = max(len(a), len(b))
    return max(0.0, 1.0 - levenshtein(a, b) / d) if d else 0.0


def align(gold, pred_span):
    """Original scorer alignment: locate the emitted span text in the input."""
    if not pred_span:
        return None
    text = gold['text']
    gs, ge = gold['gold_span_start'], gold['gold_span_end']
    matches = []
    start = text.find(pred_span)
    while start != -1:
        matches.append((start, start + len(pred_span)))
        start = text.find(pred_span, start + 1)
    if not matches:
        return None
    return max(matches, key=lambda s: (max(0, min(s[1], ge) - max(s[0], gs)), -abs(s[0] - gs)))


def score(gold, pred_span, pred_revision):
    """Six CAMC-JL metrics for one prediction (values in [0, 1])."""
    span = align(gold, pred_span)
    gs, ge = gold['gold_span_start'], gold['gold_span_end']
    if span is None:
        sm = so = 0.0
    else:
        sm = float(span == (gs, ge))
        inter = max(0, min(span[1], ge) - max(span[0], gs))
        union = max(span[1], ge) - min(span[0], gs)
        so = inter / union if union > 0 else 0.0
    p, g = norm(pred_revision), norm(gold['gold_revision'])
    rm = float(p == g)
    rs = similarity(p, g)
    return dict(SpanMatch=sm, SpanOverlapScore=so, RevisionMatch=rm, RevisionSimilarity=rs,
                SoftMatch=so * rs, ExactMatch=sm * rm)


def mean_scores(score_rows):
    n = len(score_rows)
    return {m: 100 * sum(r[m] for r in score_rows) / n for m in METRICS}


def fmt(means):
    return ' '.join(f"{SHORT[m]} {means[m]:6.2f}" for m in METRICS)


# ------------------------------------------------------------------------ prompts
SYSTEM = 'You correct Japanese grammar. Replace only the marked error span. Return only the replacement text, with no explanation.'

CATEGORY_JA = dict(
    aspect='アスペクト', tense='テンス', style='文体', passive='受身', transitivity='自他',
    honorific='敬語', benefactive='授受', demonstrative='指示詞', sentence_connection='文の接続',
    causative='使役', generic='一般', external_typo='誤字')

# Guidance sentences reuse the fixed type-level rules of the earlier LGR-MAS system (code/src/lgr/rules.py).
GUIDANCE = dict(
    demonstrative='Check local demonstrative choice in the こ/そ/あ series and related demonstrative phrases. Prefer the smallest replacement that matches the discourse distance implied by the sentence.',
    style='Check sentence ending style, register consistency, connectives, and local connective forms. Keep the sentence structure and replace only the expression needed to match the target style.',
    tense='Check past versus non-past choice in the local predicate or temporal expression. Preserve aspect and polarity unless the local tense form requires the minimal change.',
    aspect='Check aspectual forms such as ている, ていた, た, and てある. Choose the local form that best matches state, completion, continuation, or resultative meaning.',
    passive='Check active versus passive morphology and local argument perspective. Prefer the smallest predicate-form replacement.',
    causative='Check non-causative versus causative morphology. Replace the local predicate form without rewriting surrounding arguments.',
    benefactive='Check giving and receiving viewpoint expressions such as てくれる, てもらう, and てあげる. Preserve the described event and only adjust the local benefactive expression.',
    transitivity='Check intransitive versus transitive predicate choice.',
    honorific='Check honorific, humble, beautification, and address-level forms.',
    sentence_connection='Check connective particles, discourse connectives, and punctuation used to connect clauses or sentences. Replace the smallest local connection expression.',
    generic='Correct the marked expression with the smallest local change.',
    external_typo='Correct the typing error in the marked expression.',
)


def marked(text, start, end):
    assert 0 <= start < end <= len(text)
    return text[:start] + '<error>' + text[start:end] + '</error>' + text[end:]


def demo_line(row, width=12):
    t, s, e = row['text'], row['gold_span_start'], row['gold_span_end']
    left = ''.join(t[max(0, s - width):s].split())
    right = ''.join(t[e:e + width].split())
    return f"{left}<error>{t[s:e]}</error>{right} → {row['gold_revision']}"


def bigrams(text):
    text = ''.join(text.split())
    return {text[i:i + 2] for i in range(max(1, len(text) - 1))}


class Memory:
    """Same-category development examples ranked by local character-bigram overlap."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.keys = []
        for r in self.rows:
            s, e = r['gold_span_start'], r['gold_span_end']
            self.keys.append((bigrams(r['text'][s:e]), bigrams(r['text'][max(0, s - 6):e + 6])))

    def query(self, text, typ, start, end, k=4, exclude=None):
        span, local = bigrams(text[start:end]), bigrams(text[max(0, start - 6):end + 6])
        ranked = []
        for i, (r, (ks, kl)) in enumerate(zip(self.rows, self.keys)):
            if r['error_type'] != typ or (exclude is not None and r.get('id') == exclude):
                continue
            a = len(span & ks) / max(1, len(span | ks))
            b = len(local & kl) / max(1, len(local | kl))
            ranked.append((-(a + 0.5 * b), i))
        ranked.sort()
        return [self.rows[i] for _, i in ranked[:k]]


def prompt_messages(text, typ, start, end, style='original', demos=()):
    if style == 'original':
        return [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': f'Error type: {typ}\nSentence: {marked(text, start, end)}'}]
    assert style in ('guided', 'demos')  # 'demos': the guided prompt without the guidance sentence (control)
    lines = [f"Error type: {typ} ({CATEGORY_JA.get(typ, typ)})"]
    if style == 'guided':
        lines.append(f"Guidance: {GUIDANCE.get(typ, GUIDANCE['generic'])}")
    if demos:
        lines.append('Annotated corrections of this type:')
        lines.extend('- ' + demo_line(d) for d in demos)
    lines.append(f'Sentence: {marked(text, start, end)}')
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': '\n'.join(lines)}]


# ------------------------------------------------------------------------- models
def torch_modules():
    import torch
    import transformers
    import peft
    return torch, transformers, peft


def tokenizer(path=QWEN3_4B):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    assert tok.pad_token_id is not None and tok.eos_token_id is not None
    return tok


def prompt_ids(tok, messages):
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if isinstance(ids, dict) or hasattr(ids, 'keys'):
        ids = ids['input_ids']
    return list(ids)


def response_ids(tok, revision):
    ids = tok.encode(revision, add_special_tokens=False)
    return ids + [tok.eos_token_id]


def load_model(path=QWEN3_4B, adapter=None, trainable=False, new_lora=False, four_bit=True):
    torch, transformers, peft = torch_modules()
    kwargs = dict(local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
    if four_bit:
        kwargs['quantization_config'] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs['device_map'] = {'': 0}
    if Path(path) == QWEN35_4B:
        # The multimodal checkpoint stores the text model under model.language_model.
        kwargs['key_mapping'] = {r'^model\.language_model\.': 'model.'}
    model = transformers.AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.config.use_cache = False
    if adapter is not None:
        model = peft.PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    elif new_lora:
        model = peft.get_peft_model(model, peft.LoraConfig(
            task_type='CAUSAL_LM', r=8, lora_alpha=16, lora_dropout=.05, bias='none',
            target_modules=lora_targets(path)))
    if trainable or new_lora:
        for n, p in model.named_parameters():
            p.requires_grad = 'lora_' in n
            if 'lora_' in n:
                p.data = p.data.float()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    return model


def lora_targets(path):
    targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    if Path(path) == QWEN35_4B:
        targets += ['in_proj_qkv', 'in_proj_z', 'out_proj']
    return targets


def seed_all(value):
    import numpy as np
    torch, _, _ = torch_modules()
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def lr_at(step, total, peak):
    """Schedule of the original 4B study: 5% linear warm-up, cosine decay to 0.1 of peak."""
    warm = max(1, int(.05 * total))
    if step <= warm:
        return peak * step / warm
    return peak * (.1 + .45 * (1 + math.cos(math.pi * (step - warm) / (total - warm))))


def collate_left(tok, rows, device='cuda'):
    """Left-pad prompt+response rows; return model inputs and the response targets of the last positions."""
    torch, _, _ = torch_modules()
    width = max(len(r['ids']) for r in rows)
    keep = max(r['response_length'] for r in rows) + 1
    ids = torch.full((len(rows), width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    targets = torch.full((len(rows), keep - 1), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r['ids'])
        ids[i, width - n:] = torch.tensor(r['ids'])
        mask[i, width - n:] = 1
        resp = r['ids'][n - r['response_length']:]
        targets[i, keep - 1 - len(resp):] = torch.tensor(resp)
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    batch = dict(input_ids=ids.to(device), attention_mask=mask.to(device), position_ids=positions.to(device))
    return batch, targets.to(device), keep


def response_nll(model, tok, rows):
    """Per-row summed negative log likelihood of the response tokens and their counts."""
    torch, _, _ = torch_modules()
    batch, targets, keep = collate_left(tok, rows)
    logits = model(**batch, use_cache=False, logits_to_keep=keep).logits[:, :-1].float()
    nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100, reduction='none')
    return nll.sum(1), targets.ne(-100).sum(1)


def encode_example(tok, messages, revision):
    p = prompt_ids(tok, messages)
    r = response_ids(tok, revision)
    return dict(ids=p + r, labels=[-100] * len(p) + r, response_length=len(r))


TRAIN_TOKEN_BUDGET = 800  # padded tokens per backward pass on a 6 GB GPU


def train_adapter(model, tok, examples, updates, horizon, peak, batch=16, micro=4, seed=20260927, log=None):
    """Response-token cross-entropy, token-weighted accumulation, AdamW, clipping at 1."""
    torch, _, _ = torch_modules()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=peak, weight_decay=.01)
    rng = random.Random(seed)
    order = []
    step = 0
    history = []
    model.train()
    while step < updates:
        if len(order) < batch:
            perm = list(range(len(examples)))
            rng.shuffle(perm)
            order.extend(perm)
        rows = [examples[i] for i in order[:batch]]
        del order[:batch]
        step += 1
        for g in opt.param_groups:
            g['lr'] = lr_at(step, horizon, peak)
        opt.zero_grad(set_to_none=True)
        denom = sum(r['response_length'] for r in rows)
        total = 0.0
        parts = [rows[first:first + micro] for first in range(0, len(rows), micro)]
        while parts:
            part = parts.pop(0)
            # Long guided prompts: halve a micro-batch whose padded length exceeds the token budget.
            # The loss is normalized by the batch token count, so the gradient is unchanged.
            if len(part) > 1 and len(part) * max(len(r['ids']) for r in part) > TRAIN_TOKEN_BUDGET:
                half = len(part) // 2
                parts[:0] = [part[:half], part[half:]]
                continue
            nll, count = response_nll(model, tok, part)
            loss = nll.sum() / denom
            loss.backward()
            total += float(loss.detach())
        norm_ = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        torch.cuda.empty_cache()  # long runs otherwise fragment the 6 GB of VRAM until allocation fails
        history.append(dict(step=step, loss=total, grad_norm=float(norm_), lr=lr_at(step, horizon, peak)))
        if log is not None and (step % 10 == 0 or step == updates):
            log(f'step {step}/{updates} loss {total:.4f} lr {lr_at(step, horizon, peak):.2e}')
    model.eval()
    return history


def generation_limit(start, end):
    return min(128, max(16, 4 * (end - start) + 8))


def valid(candidate, source):
    return bool(candidate) and candidate != source and '\n' not in candidate and '\r' not in candidate \
        and len(candidate) <= max(44, 4 * len(source))


def generate_nbest(model, tok, requests, beams=4, batch_size=4):
    """Four-beam deterministic decoding that returns every beam.

    requests: list of dicts with 'messages', 'start', 'end', 'source'.
    Returns, per request, the list of beams as dicts (text, completed, valid).
    """
    torch, transformers, _ = torch_modules()
    model.eval()
    results = [None] * len(requests)
    groups = {}
    for i, r in enumerate(requests):
        groups.setdefault(generation_limit(r['start'], r['end']), []).append(i)
    lengths = {i: len(prompt_ids(tok, r['messages'])) for i, r in enumerate(requests)}
    with torch.inference_mode():
        for limit, indices in sorted(groups.items()):
            indices = sorted(indices, key=lambda i: lengths[i])
            first = 0
            while first < len(indices):
                # Long prompts use smaller batches so that the beam cache stays inside 6 GB of VRAM.
                size = batch_size if lengths[indices[first]] < 300 else max(1, batch_size // 2)
                chosen = indices[first:first + size]
                first += size
                prompts = [prompt_ids(tok, requests[i]['messages']) for i in chosen]
                width = max(map(len, prompts))
                ids = torch.full((len(prompts), width), tok.pad_token_id, dtype=torch.long, device='cuda')
                mask = torch.zeros_like(ids)
                for j, p in enumerate(prompts):
                    ids[j, -len(p):] = torch.tensor(p, device='cuda')
                    mask[j, -len(p):] = 1
                config = transformers.GenerationConfig(
                    do_sample=False, num_beams=beams, num_return_sequences=beams, length_penalty=.6,
                    early_stopping=True, max_new_tokens=limit, eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id, use_cache=True)
                out = model.generate(input_ids=ids, attention_mask=mask, generation_config=config,
                                     temperature=None, top_k=None, top_p=None)[:, width:].cpu().tolist()
                for j, i in enumerate(chosen):
                    beams_out = []
                    for tokens in out[j * beams:(j + 1) * beams]:
                        complete = tok.eos_token_id in tokens
                        if complete:
                            tokens = tokens[:tokens.index(tok.eos_token_id)]
                        tokens = [t for t in tokens if t != tok.pad_token_id]
                        raw = tok.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                        text = raw.strip()
                        beams_out.append(dict(text=text, completed=complete,
                                              valid=complete and valid(text, requests[i]['source'])))
                    results[i] = beams_out
                # Prompts of varying length fragment the cache; on Windows the driver then spills to host memory.
                torch.cuda.empty_cache()
    return results


def loglik(model, tok, pairs, batch_size=8):
    """Length-normalized response log likelihood m^-0.6 * sum log p, as in the original study.

    pairs: list of (messages, revision). Returns a list of floats.
    """
    torch, _, _ = torch_modules()
    model.eval()
    rows = [encode_example(tok, m, y) for m, y in pairs]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]['ids']))
    values = [0.0] * len(rows)
    budget = int(os.environ.get('JGEC_SCORE_BUDGET', 2400))  # padded tokens per batch (6 GB of VRAM)
    with torch.inference_mode():
        first = 0
        while first < len(order):
            size = max(1, min(batch_size, budget // max(1, len(rows[order[min(len(order) - 1, first + batch_size - 1)]]['ids']))))
            idx = order[first:first + size]
            first += size
            nll, count = response_nll(model, tok, [rows[i] for i in idx])
            for i, v in zip(idx, (-nll / count.float().pow(.6)).cpu().tolist()):
                values[i] = v
            if first % (20 * size) < size:
                torch.cuda.empty_cache()  # bound the fragmentation from varying batch shapes
    return values


def loglik_prefix(model, tok, pairs, chunk=8):
    """loglik() with each shared prompt encoded once; its responses are scored from the copied KV cache.

    Pairs that share a prompt must share the same messages object. Only for full-attention models
    (Qwen3); the hybrid cache of Qwen3.5 is not copied this way.
    """
    import copy
    torch, _, _ = torch_modules()
    model.eval()
    groups = {}
    for i, (m, _) in enumerate(pairs):
        groups.setdefault(id(m), (m, []))[1].append(i)
    values = [0.0] * len(pairs)
    with torch.inference_mode():
        for m, idx in groups.values():
            prompt = torch.tensor([prompt_ids(tok, m)], device='cuda')
            out = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
            first = torch.log_softmax(out.logits[0, -1].float(), -1)
            cache, plen = out.past_key_values, prompt.shape[1]
            for c0 in range(0, len(idx), chunk):
                part = idx[c0:c0 + chunk]
                resp = [response_ids(tok, pairs[i][1]) for i in part]
                total = [first[r[0]].item() for r in resp]
                width = max(len(r) for r in resp) - 1
                if width > 0:
                    ids = torch.full((len(part), width), tok.pad_token_id, dtype=torch.long)
                    mask = torch.ones((len(part), plen + width), dtype=torch.long)
                    for b, r in enumerate(resp):
                        ids[b, :len(r) - 1] = torch.tensor(r[:-1])
                        mask[b, plen + len(r) - 1:] = 0
                    pos = (plen + torch.arange(width)).expand(len(part), width)
                    past = copy.deepcopy(cache)
                    past.batch_repeat_interleave(len(part))
                    logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), position_ids=pos.cuda(),
                                   past_key_values=past, use_cache=True).logits.float()
                    lp = torch.log_softmax(logits, -1)
                    for b, r in enumerate(resp):
                        if len(r) > 1:
                            tgt = torch.tensor(r[1:], device='cuda')
                            total[b] += lp[b, torch.arange(len(r) - 1, device='cuda'), tgt].sum().item()
                    del past, logits, lp
                for b, i in enumerate(part):
                    values[i] = total[b] / len(resp[b]) ** .6
            del out, cache
    return values


def loglik_checked(model, tok, pairs, check=48, tol=0.05, log=print):
    """Use loglik_prefix() after it reproduces loglik() on the first prompts; otherwise fall back to loglik()."""
    seen, head = set(), []
    for i, (m, _) in enumerate(pairs):
        if id(m) not in seen and len(head) >= check:
            break
        seen.add(id(m))
        head.append(i)
    sub = [pairs[i] for i in head]
    slow, fast = loglik(model, tok, sub), loglik_prefix(model, tok, sub)
    diff = max(abs(a - b) for a, b in zip(slow, fast))
    log(f'prefix-cache check on {len(sub)} pairs: max |diff| = {diff:.4f}')
    if diff > tol:
        log('prefix-cache scores differ; using loglik()')
        return loglik(model, tok, pairs)
    return loglik_prefix(model, tok, pairs)
