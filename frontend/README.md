# Compact frontend

The compact frontend fixes one error interval of a supplied category. It holds three character GRUs (one per seed),
the sparse localizer and sparse editor, the crop localizer trained on cropped contexts, and the context gate:
4,234,774 stored values. The GRU, sparse, and gate files are release assets; `python scripts/download_weights.py`
places them in this folder and checks their SHA-256.

```bash
python predict.py --list-error-types
python predict.py --text '日本語の入力文章' --error-type tense
```

The API is `Corrector(directory, seed).predict(text, category)`. The default seed is 20260927; all three seeds
are included. Prediction needs NumPy and PyTorch and runs on a CPU.
