# Repository scope

This is a reproducibility archive for DPA4 synthesizability classification.

- Never add training arrays, mixed background data, original pretraining weights, machine.json, passwords or API tokens to Git.
- This repository only exposes the later native classification + cross-entropy workflow. Do not add earlier property/one-hot training comparisons.
- Use native `dp --pt train` and `dp --pt test` commands. Do not add training hooks or call internal test functions when the public CLI suffices; leave redundant defaults to DeepMD.
- Treat vendor/, original experiment configs/models, and metadata hash indices as immutable historical artifacts. New adaptations belong in scripts/ and workflow/. Change historical bytes only when explicitly creating and documenting a new version.
- Run all actual model training and inference on Bohrium, not on the local workspace. Pure NumPy metrics, hashing, format conversion and static checks may run locally.
- Train/valid/test membership is grouped by recipe. Training samples and checkpoint selection metrics are structure-level. Select on full valid macro-F1, then balanced accuracy, lower probability MAE, earlier step. Test does not select checkpoints or tune thresholds.
- Runtime data paths are external and must exist inside the Bohrium container. Preparing a code payload must not package external datasets.
- Read credentials only from the caller's external --machine path. Real logs must be both Task.outlog and backward_files. Use grouped=True and reuse one task group ID for a batch.
- Keep root README in clear Chinese and preserve the upstream provenance/license.

Before committing code changes, run:

```bash
ruff check .
ruff format --check .
python scripts/reproduce.py verify
python -m unittest discover -s tests -v
```
