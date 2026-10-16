"""Final refits on all 200 development examples and one test pass per seed.

Run only after the development study has fixed the configuration.
Each seed starts from the auxiliary-stage adapter, fits the source stage on all
development examples (130 updates, 390-update schedule horizon) and generates
candidates at the saved test intervals of the same seed's frontend.
"""
import argparse
import gc
import math
import time

from common import (BASES, DATA, RESULTS, QWEN3_4B, QWEN35_4B, SEEDS, Memory, dev_rows, read, write_lines, dump,
                    tokenizer, load_model, lora_targets, seed_all, train_adapter, torch_modules)
from cv_source import training_examples, FIELDS
from pipeline import run_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--style', default='original', choices=['original', 'guided', 'demos'])
    ap.add_argument('--k', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--peak', type=float, default=1e-4)
    ap.add_argument('--model', default='qwen3', choices=['qwen3', 'qwen35', 'sarashina'])
    ap.add_argument('--init', default=str(DATA / 'orig4b/aux_step1000'))
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    ap.add_argument('--train-offset', type=int, default=0,
                    help='added to the training seed only; outputs keep the pipeline seed keys')
    a = ap.parse_args()
    torch, _, peft = torch_modules()
    path = BASES[a.model]
    out = RESULTS / 'final' / a.tag
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'config.json', vars(a))
    dev = dev_rows()
    inputs = read(DATA / 'orig4b_test/inputs.jsonl')
    tok = tokenizer(path)

    def log(msg):
        print(msg, flush=True)
        with open(out / 'log.txt', 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    for seed in [int(s) for s in a.seeds.split(',')]:
        dest = out / f'seed{seed}.jsonl'
        if dest.exists():
            continue
        seed_all(seed + a.train_offset)
        saved = out / f'adapter_seed{seed}'
        if (saved / 'adapter_model.safetensors').exists():
            # Training finished earlier (inference was interrupted): reuse the saved adapter.
            base = load_model(path)
            model = peft.PeftModel.from_pretrained(base, str(saved))
            log(f'seed {seed}: reusing saved adapter')
            infer(model, tok, inputs, dev, a, seed, dest, log)
            del model, base
            gc.collect()
            torch.cuda.empty_cache()
            continue
        examples = training_examples(tok, dev, a.style, a.k)
        updates = math.ceil(len(examples) / 16) * a.epochs
        base = load_model(path)
        if a.init != 'none':
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
        log(f'seed {seed}: {len(examples)} examples, {updates} updates')
        history = train_adapter(model, tok, examples, updates, 3 * updates, a.peak, seed=seed + a.train_offset, log=log)
        log(f'seed {seed}: trained in {time.perf_counter() - t:.0f}s')
        model.save_pretrained(str(out / f'adapter_seed{seed}'))
        dump(out / f'train_seed{seed}.json', history)
        # Release the training state completely and reload base model and adapter for inference; on a 6 GB
        # GPU the allocator otherwise keeps about 1 GB of training residue and long prompts spill to host memory.
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
        base = load_model(path)
        model = peft.PeftModel.from_pretrained(base, str(out / f'adapter_seed{seed}'))
        infer(model, tok, inputs, dev, a, seed, dest, log)
        del model, base
        gc.collect()
        torch.cuda.empty_cache()


def infer(model, tok, inputs, dev, a, seed, dest, log):
    """Candidates and scores at the saved test intervals of the same seed's frontend."""
    rows = [dict(key=f"{seed}:{r['release_id']}", **{k: r[k] for k in FIELDS})
            for r in inputs if r['seed'] == seed]
    memory = Memory(dev) if a.style in ('guided', 'demos') and a.k else None
    t = time.perf_counter()
    recs = run_candidates(model, tok, rows, a.style, memory, a.k, log=log)
    log(f'seed {seed}: test inference {time.perf_counter() - t:.0f}s')
    write_lines(dest, recs)


if __name__ == '__main__':
    main()
