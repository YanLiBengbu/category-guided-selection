"""Prompted baselines with a local backend: Direct prompting and LGR-MAS of our earlier study on Qwen3-4B.

The prompts, agents, memory tool, rule guidance, and parsing are imported unchanged from the original LGR-MAS
code (transfer package, code/src); only the backend differs. A batched Hugging Face backend serves the
generate() calls of several concurrent examples: greedy decoding, temperature 0, at most 256 new tokens, the
prompt sent as one user message, and the base model in 4-bit NF4 as in the selection experiments. The
CorrectionMemoryTool holds the 200 development examples. No data leave this machine.

Outputs: results/local_llm/{direct,lgr}_{model}.jsonl (prediction, call count, parse status per example) and
results/local_llm/{direct,lgr}_{model}_matrix.npy (559 x 6 metric matrix in TEST_IDS order).
Usage: python local_llm_baselines.py direct|lgr [qwen3|qwen35] [WORKERS]
"""
import json
import os
import queue
import sys
import threading
import time
import types
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import BASES, METRICS, RESULTS, dev_rows, load_model, score, test_rows, tokenizer, torch_modules

LGR_SRC = Path(os.environ['LGR_MAS_SRC'])  # the src/ folder of the LGR-MAS code of our earlier study
sys.modules.setdefault('httpx', types.ModuleType('httpx'))  # only the hosted backends use it
sys.path.insert(0, str(LGR_SRC))
from camc_dataset.schema import CamcExample, CorpusSource, ErrorType  # noqa: E402
from llm.backend import LLMResponse  # noqa: E402
from lgr.memory import CorrectionMemoryTool  # noqa: E402
from lgr.context import LGRContext  # noqa: E402
from lgr.system import build_lgr_mas  # noqa: E402
from baselines.direct import run_direct_prediction  # noqa: E402

MAX_NEW = 256
TOKEN_BUDGET = 9000  # padded prompt tokens per generation batch on the 6 GB GPU


class BatchedLocalBackend:
    """generate() blocks the calling thread; one GPU thread batches the pending requests."""

    def __init__(self, name):
        self.model = f'local-{name}'
        self.tok = tokenizer(BASES[name])
        self.tok.padding_side = 'left'
        self.net = load_model(BASES[name])
        self.net.config.use_cache = True
        self.net.eval()
        self.requests = queue.Queue()
        self.calls = 0
        self.stop = False
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()

    def generate(self, prompt, *, temperature=0.0, max_output_tokens=None):
        fut = Future()
        self.requests.put((prompt, min(MAX_NEW, max_output_tokens or MAX_NEW), fut))
        return fut.result()

    def encode(self, prompt):
        text = self.tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                            add_generation_prompt=True)
        return self.tok(text, add_special_tokens=False)['input_ids']

    def loop(self):
        torch = torch_modules()[0]
        while not self.stop:
            try:
                first = self.requests.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = [first]
            time.sleep(0.3)  # let concurrent examples submit their calls
            ids = [self.encode(first[0])]
            while True:
                try:
                    item = self.requests.get_nowait()
                except queue.Empty:
                    break
                enc = self.encode(item[0])
                width = max(max(len(x) for x in ids), len(enc))
                if width * (len(batch) + 1) > TOKEN_BUDGET:
                    self.requests.put(item)
                    break
                batch.append(item)
                ids.append(enc)
            try:
                width = max(len(x) for x in ids)
                pad = self.tok.pad_token_id
                input_ids = torch.tensor([[pad] * (width - len(x)) + x for x in ids]).cuda()
                mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids]).cuda()
                with torch.inference_mode():
                    out = self.net.generate(input_ids=input_ids, attention_mask=mask, do_sample=False,
                                            max_new_tokens=max(b[1] for b in batch), pad_token_id=pad,
                                            temperature=None, top_p=None, top_k=None)
                for k, (prompt, limit, fut) in enumerate(batch):
                    new = out[k, width:].tolist()
                    if self.tok.eos_token_id in new:
                        new = new[:new.index(self.tok.eos_token_id)]
                    new = new[:limit]
                    text = self.tok.decode(new, skip_special_tokens=True)
                    fut.set_result(LLMResponse(text=text, input_tokens=len(ids[k]), output_tokens=len(new)))
                self.calls += len(batch)
                del out, input_ids, mask
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                for _, _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)


def example(row):
    return CamcExample(id=row['id'], text=row['text'], error_type=ErrorType(row['error_type']),
                       source=CorpusSource(row['source']), gold_span_start=row['gold_span_start'],
                       gold_span_end=row['gold_span_end'], gold_revision=row['gold_revision'])


def main(system, name='qwen3', workers=8):
    out_dir = RESULTS / 'local_llm'
    out_dir.mkdir(exist_ok=True)
    dest = out_dir / f'{system}_{name}.jsonl'
    rows = test_rows()
    done = {}
    if dest.exists():
        done = {json.loads(l)['index']: json.loads(l) for l in dest.read_text(encoding='utf-8').splitlines() if l}
    backend = BatchedLocalBackend(name)
    memory = CorrectionMemoryTool([example(r) for r in dev_rows()])
    lock = threading.Lock()
    t0 = time.perf_counter()

    def run(i):
        ex = example(rows[i])
        calls = []
        if system == 'direct':
            pred = run_direct_prediction(example=ex, backend=backend, calls=calls, temperature=0.0,
                                         max_output_tokens=MAX_NEW)
        else:
            coordinator = build_lgr_mas(backend=backend, memory_tool=memory, temperature=0.0,
                                        max_output_tokens=MAX_NEW)
            context = LGRContext(example=ex)
            try:
                pred = coordinator.run(context)
            except Exception as exc:  # noqa: BLE001  (an agent failure yields an empty prediction)
                print('example', i, 'failed:', exc, flush=True)
                pred = None
            calls = context.llm_calls
        rec = dict(index=i, pred_span=pred.pred_span if pred else '', pred_revision=pred.pred_revision if pred else '',
                   calls=len(calls), output_tokens=sum(c.output_tokens or 0 for c in calls))
        with lock:
            with open(dest, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            done[i] = rec
            if len(done) % 50 == 0:
                print(f'{len(done)}/559 done, {time.perf_counter() - t0:.0f}s, {backend.calls} calls', flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, [i for i in range(len(rows)) if i not in done]))
    backend.stop = True
    mat = np.array([[score(rows[i], done[i]['pred_span'], done[i]['pred_revision'])[m] for m in METRICS]
                    for i in range(len(rows))])
    np.save(out_dir / f'{system}_{name}_matrix.npy', mat)
    empty = sum(1 for i in range(len(rows)) if not done[i]['pred_revision'])
    print(system, name, {m: round(100 * v, 2) for m, v in zip(METRICS, mat.mean(0))}, 'empty', empty,
          'calls', sum(d['calls'] for d in done.values()), f'{time.perf_counter() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'qwen3', int(sys.argv[3]) if len(sys.argv) > 3 else 8)
