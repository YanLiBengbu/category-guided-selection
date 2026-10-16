"""Grouped three-fold development study of the source stage.

For each fold, we start from the auxiliary-stage adapter, fit the source stage on
the fitting folds and generate candidates for the held-out original views at the
saved out-of-fold frontend intervals. Test data are never read.
"""
import argparse
import math
import time
from pathlib import Path

from common import (BASES, DATA, RESULTS, QWEN3_4B, QWEN35_4B, Memory, dev_rows, folds, read, write_lines, dump,
                    tokenizer, load_model, lora_targets, seed_all, train_adapter, encode_example, prompt_messages,
                    torch_modules)
from pipeline import run_candidates

FIELDS = ['text', 'error_type', 'selected_start', 'selected_end', 'pred_span', 'pred_revision']


def training_examples(tok, rows, style, k):
    memory = Memory(rows) if style in ('guided', 'demos') and k else None
    out = []
    for r in rows:
        s, e = r['gold_span_start'], r['gold_span_end']
        demos = memory.query(r['text'], r['error_type'], s, e, k=k, exclude=r['id']) if memory else ()
        msgs = prompt_messages(r['text'], r['error_type'], s, e, style=style, demos=demos)
        ex = encode_example(tok, msgs, r['gold_revision'])
        if len(ex['ids']) <= 768:
            out.append(ex)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--style', default='original', choices=['original', 'guided', 'demos'])
    ap.add_argument('--k', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20260927)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--peak', type=float, default=1e-4)
    ap.add_argument('--model', default='qwen3', choices=['qwen3', 'qwen35', 'sarashina'])
    ap.add_argument('--init', default=str(DATA / 'orig4b/aux_step1000'))
    ap.add_argument('--folds', default='0,1,2')
    ap.add_argument('--train-offset', type=int, default=0,
                    help='added to the training seed only (a second generator with the same recipe)')
    a = ap.parse_args()
    _, _, peft = torch_modules()
    path = BASES[a.model]
    out_dir = RESULTS / 'cv' / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(out_dir / 'config.json', vars(a))
    dev = dev_rows()
    saved = [r for r in read(DATA / 'orig4b/inputs.jsonl') if r['view'] == 'original']
    tok = tokenizer(path)
    log_path = out_dir / 'log.txt'

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    for fold in [int(x) for x in a.folds.split(',')]:
        dest = out_dir / f'fold{fold}.jsonl'
        if dest.exists():
            log(f'fold {fold} exists, skipped')
            continue
        fit, held = folds()[fold]
        fit_rows = [dev[i] for i in fit]
        seed_all(a.seed + fold + a.train_offset)
        examples = training_examples(tok, fit_rows, a.style, a.k)
        updates = math.ceil(len(examples) / 16) * a.epochs
        base = load_model(path)
        saved_adapter = out_dir / f'adapter_fold{fold}'
        if saved_adapter.exists():
            # Training finished earlier: reuse the saved fold adapter and only run inference.
            model = peft.PeftModel.from_pretrained(base, str(saved_adapter))
            log(f'fold {fold}: reusing saved adapter')
            rows = [dict(key=f"{fold}:{r['parent_index']}:original", **{k: r[k] for k in FIELDS})
                    for r in saved if r['fold'] == fold]
            memory = Memory(fit_rows) if a.style in ('guided', 'demos') and a.k else None
            write_lines(dest, run_candidates(model, tok, rows, a.style, memory, a.k, log=log))
            del model, base
            import gc
            gc.collect()
            torch_modules()[0].cuda.empty_cache()
            continue
        if a.init and a.init != 'none':
            model = peft.PeftModel.from_pretrained(base, a.init, is_trainable=True)
        else:
            model = peft.get_peft_model(base, peft.LoraConfig(
                task_type='CAUSAL_LM', r=8, lora_alpha=16, lora_dropout=.05, bias='none',
                target_modules=lora_targets(path)))
        for n, p in model.named_parameters():
            p.requires_grad = 'lora_' in n
            if 'lora_' in n:
                p.data = p.data.float()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        t = time.perf_counter()
        log(f'fold {fold}: {len(examples)} examples, {updates} updates')
        history = train_adapter(model, tok, examples, updates, 3 * updates, a.peak, seed=a.seed + fold + a.train_offset, log=log)
        log(f'fold {fold}: trained in {time.perf_counter() - t:.0f}s')
        model.save_pretrained(str(out_dir / f'adapter_fold{fold}'))
        dump(out_dir / f'train_fold{fold}.json', history)
        model.gradient_checkpointing_disable()
        torch_modules()[0].cuda.empty_cache()
        rows = [dict(key=f"{fold}:{r['parent_index']}:original", **{k: r[k] for k in FIELDS})
                for r in saved if r['fold'] == fold]
        assert sorted(int(r['key'].split(':')[1]) for r in rows) == sorted(held)
        memory = Memory(fit_rows) if a.style in ('guided', 'demos') and a.k else None
        recs = run_candidates(model, tok, rows, a.style, memory, a.k, log=log)
        write_lines(dest, recs)
        del model, base
        import gc
        gc.collect()
        torch_modules()[0].cuda.empty_cache()


if __name__ == '__main__':
    main()
