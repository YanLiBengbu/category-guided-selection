"""Wait until the shared GPU has enough free memory for the next step (other projects may be using it).

Usage: python gpu_gate.py NEED_MIB [CHECKS]
Returns once nvidia-smi reports at least NEED_MIB free memory in CHECKS consecutive checks 30 s apart.
"""
import os
import subprocess
import sys
import time
from pathlib import Path


def free_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    total, used = (int(x) for x in out.split(','))
    return total - used


def main(need, checks=3):
    # gpu_gate_need.txt lets a running queue use a new threshold: "NEED [MAX_WAIT_MIN FALLBACK_NEED]".
    # After MAX_WAIT_MIN minutes the gate accepts FALLBACK_NEED, so another long-running job cannot block it forever.
    max_wait, fallback = None, None
    override = Path(__file__).with_name('gpu_gate_need.txt')
    if override.exists():
        values = [int(v) for v in override.read_text().split()]
        need = values[0]
        if len(values) == 3:
            max_wait, fallback = values[1], values[2]
    ok, waited, t0 = 0, False, time.time()
    # Any gpu_hold*.txt pauses every queue at its next gate, except callers that set JGEC_IGNORE_HOLD=1 (used to
    # slot another queue in between the steps of a long one without killing it).
    here = Path(__file__).parent
    while any(here.glob('gpu_hold*.txt')) and os.environ.get('JGEC_IGNORE_HOLD') != '1':
        if not waited:
            print(time.strftime('%H:%M'), 'gpu gate: held by gpu_hold.txt', flush=True)
            waited = True
        time.sleep(30)
    while ok < checks:
        if max_wait is not None and time.time() - t0 > 60 * max_wait:
            need = fallback
        free = free_mib()
        ok = ok + 1 if free >= need else 0
        if ok < checks:
            if not waited or int(time.time() - t0) % 1800 < 30:
                print(time.strftime('%H:%M'), f'gpu gate: {free} MiB free, need {need}', flush=True)
            waited = True
            time.sleep(30)
    print(time.strftime('%H:%M'), f'gpu gate open after {round((time.time() - t0) / 60)} min', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 3)
