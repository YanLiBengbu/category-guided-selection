#!/bin/bash
# GPU queue 10 (review round, 2026-10-12): external baselines, slotted in while queue 9 waits at its next gate.
#   local Direct prompting and LGR-MAS of our earlier study on Qwen3-4B (local_llm_baselines.py)
#   character DeBERTa BIO localizer with the base generator, and the fine-tuned Japanese T5 (ext_baselines.py)
# gpu_hold.txt holds queue 9 at its next gate; this queue ignores the hold and removes it at the end.
PY=${PY:-python}
export PYTHONIOENCODING=utf-8 TRANSFORMERS_VERBOSITY=error JGEC_IGNORE_HOLD=1
cd "$(dirname "$0")"
R=../results
touch gpu_hold.txt
until grep -q "held by gpu_hold" $R/queue9.log || grep -q "Q9_DONE" $R/queue9.log; do sleep 60; done
echo "$(date +%H:%M) queue 9 held"
gate() { $PY gpu_gate.py 3900 3; }
gate; $PY local_llm_baselines.py direct qwen3 8 > $R/q10_direct.log 2>&1; echo "$(date +%H:%M) direct exit $?"
gate; $PY ext_baselines.py deberta > $R/q10_deberta.log 2>&1; echo "$(date +%H:%M) deberta exit $?"
gate; $PY ext_baselines.py t5 > $R/q10_t5.log 2>&1; echo "$(date +%H:%M) t5 exit $?"
gate; $PY local_llm_baselines.py lgr qwen3 8 > $R/q10_lgr.log 2>&1; echo "$(date +%H:%M) lgr exit $?"
rm -f gpu_hold.txt
echo Q10_DONE
