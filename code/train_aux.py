"""Auxiliary stage for a new base model, replaying the fixed 16,000-pair order of the original study.

The pool is teacher_train + synthetic_train (59,767 pairs); auxiliary_indices.i64 lists the
16,000 sampled pairs in visiting order (16 per update, 1,000 updates, peak learning rate 2e-4,
5% warm-up and cosine decay to 0.1 over 1,000 updates).
Usage: python train_aux.py --model qwen35 --tag qwen35_aux [--style original|guided]
"""
import argparse
import time

import numpy as np

from common import (BASES, DATA, RESULTS, read, dump, tokenizer, load_model, lora_targets, seed_all, lr_at,
                    encode_example, prompt_messages, response_nll, torch_modules)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(BASES))
    ap.add_argument('--tag', required=True)
    ap.add_argument('--style', default='original', choices=['original', 'guided'])
    ap.add_argument('--updates', type=int, default=1000)
    ap.add_argument('--micro', type=int, default=8)
    a = ap.parse_args()
    torch, _, peft = torch_modules()
    out = RESULTS / 'aux' / a.tag
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'config.json', vars(a))
    path = BASES[a.model]
    tok = tokenizer(path)
    pool = read(DATA / 'teacher_train.jsonl') + read(DATA / 'synthetic_train.jsonl')
    assert len(pool) == 59767
    order = np.fromfile(DATA / 'orig4b/auxiliary_indices.i64', dtype='<i8')
    assert len(order) == 16000
    seed_all(20260928)
    base = load_model(path)
    model = peft.get_peft_model(base, peft.LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=16, lora_dropout=.05,
                                                      bias='none', target_modules=lora_targets(path)))
    for n, p in model.named_parameters():
        p.requires_grad = 'lora_' in n
        if 'lora_' in n:
            p.data = p.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=.01)
    log, t0 = [], time.perf_counter()
    model.train()
    for step in range(1, a.updates + 1):
        rows = []
        for i in order[(step - 1) * 16:step * 16]:
            r = pool[int(i)]
            msgs = prompt_messages(r['text'], r['error_type'], r['gold_span_start'], r['gold_span_end'], style=a.style)
            rows.append(encode_example(tok, msgs, r['gold_revision']))
        for g in opt.param_groups:
            g['lr'] = lr_at(step, a.updates, 2e-4)
        opt.zero_grad(set_to_none=True)
        denom = sum(r['response_length'] for r in rows)
        total = 0.0
        for first in range(0, 16, a.micro):
            nll, _ = response_nll(model, tok, rows[first:first + a.micro])
            loss = nll.sum() / denom
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        log.append(dict(step=step, loss=total))
        if step % 25 == 0:
            msg = f'aux step {step} loss {sum(x["loss"] for x in log[-25:]) / 25:.4f} {time.perf_counter() - t0:.0f}s'
            print(msg, flush=True)
            with open(out / 'log.txt', 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
    model.save_pretrained(str(out / f'aux_step{a.updates}'))
    dump(out / 'train.json', log)


if __name__ == '__main__':
    main()
