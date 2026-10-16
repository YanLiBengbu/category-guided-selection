#!/bin/bash
# GPU queue 9 (review round, 2026-10-12): control generators for the guided generator of the selector.
#   orig_k0_b  : a second base-prompt generator (same recipe, training seed offset 1000)
#   guided_k0  : the guided prompt without demonstrations (category guidance only)
#   demos_k4   : the guided prompt without the guidance sentence (four demonstrations only)
# Development first (three folds), then the cross-scored development pools with fluency scores, then the
# test refits (three seeds) and the test pools. Finished folds, seeds and files are reused on a rerun.
PY=${PY:-python}
export PYTHONIOENCODING=utf-8 TRANSFORMERS_VERBOSITY=error JGEC_SCORE_BUDGET=2400
cd "$(dirname "$0")"
R=../results
gate() { $PY gpu_gate.py 3900 3; }
CONTROLS="orig_k0_b guided_k0 demos_k4"
args() {
  case $1 in
    orig_k0_b) echo "--style original --k 0 --train-offset 1000" ;;
    guided_k0) echo "--style guided --k 0" ;;
    demos_k4) echo "--style demos --k 4" ;;
  esac
}
cv_done() { for f in 0 1 2; do [ -f $R/cv/$1/fold$f.jsonl ] || return 1; done; }
final_done() { for s in 20260927 20260928 20260929; do [ -f $R/final/$1/seed$s.jsonl ] || return 1; done; }
pool() {  # split control prefix reuse
  local split=$1 c=$2 prefix=$3 reuse=$4 base=$5
  [ -f $R/${prefix}_${c}_lm2.jsonl ] && return 0
  [ -f $R/${prefix}_${c}.jsonl ] || { gate; $PY cross_score.py $split $R/${prefix}_${c}.jsonl --score $base/orig_k0 $base/$c --reuse $R/$reuse >> $R/q9_cross.log 2>&1; }
  gate; $PY lm_score.py $R/${prefix}_${c}.jsonl $R/${prefix}_${c}_lm1.jsonl qwen3 24 $R/$reuse >> $R/q9_lm.log 2>&1
  gate; $PY lm_score.py $R/${prefix}_${c}_lm1.jsonl $R/${prefix}_${c}_lm2.jsonl sarashina 24 $R/$reuse >> $R/q9_lm.log 2>&1
}
for c in $CONTROLS; do
  for attempt in 1 2 3; do
    cv_done $c && break
    gate; $PY cv_source.py --tag $c $(args $c) >> $R/q9_cv_$c.log 2>&1
  done
  cv_done $c || echo "Q9 CV FAILED $c"
done
for c in $CONTROLS; do cv_done $c && pool dev $c cv_cross_q9 cv_cross3_lm2.jsonl cv; done
echo Q9_DEV_DONE
for c in $CONTROLS; do
  for attempt in 1 2 3; do
    final_done $c && break
    gate; $PY final_test.py --tag $c $(args $c) >> $R/q9_final_$c.log 2>&1
  done
  final_done $c || echo "Q9 FINAL FAILED $c"
  final_done $c && pool test $c test_cross_q9 test_cross2_lm2.jsonl final
done
echo Q9_DONE
