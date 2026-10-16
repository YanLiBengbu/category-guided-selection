#!/bin/bash
# GPU queue 8 (replaces queue 7 while another project shares the GPU): every step waits for 3.9 GB of free
# VRAM; finished seeds and saved adapters are reused. Then the frozen test pool and its fluency scores.
PY=${PY:-python}
export PYTHONIOENCODING=utf-8 TRANSFORMERS_VERBOSITY=error JGEC_GEN_BATCH=2 JGEC_SCORE_BUDGET=1200
cd "$(dirname "$0")"
R=../results
gate() { $PY gpu_gate.py 3900 3; }
done_all() { for s in 20260927 20260928 20260929; do [ -f $R/final/$1/seed$s.jsonl ] || return 1; done; }
for attempt in 1 2 3 4 5; do
  done_all orig_k0 || { gate; $PY final_test.py --tag orig_k0 --style original --k 0 >> $R/final_orig2.log 2>&1; }
  done_all guided_k4 || { gate; $PY final_test.py --tag guided_k4 --style guided --k 4 >> $R/final_guided2.log 2>&1; }
  done_all orig_k0 && done_all guided_k4 && break
  echo "attempt $attempt incomplete"; sleep 60
done
done_all orig_k0 && done_all guided_k4 || { echo QUEUE8_FAILED_REFITS; exit 1; }
for attempt in 1 2 3; do
  gate; $PY cross_score.py test $R/test_cross2_q3.jsonl --score final/orig_k0 final/guided_k4 > $R/cross2_test.log 2>&1 && break
  echo "cross attempt $attempt failed"; sleep 60
done
gate; $PY lm_score.py $R/test_cross2_q3.jsonl $R/test_cross2_lm1.jsonl qwen3 > $R/lm_test.log 2>&1
gate; $PY lm_score.py $R/test_cross2_lm1.jsonl $R/test_cross2_lm2.jsonl sarashina >> $R/lm_test.log 2>&1
echo QUEUE8_DONE
