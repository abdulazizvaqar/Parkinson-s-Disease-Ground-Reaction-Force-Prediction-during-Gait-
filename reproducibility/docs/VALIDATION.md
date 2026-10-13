# Validation of this prepared code package

- All 42 Python files parse, including the preserved original scripts.
- Five Colab notebooks contain cleared outputs; Python code cells parse after standard notebook magic handling.
- Copied original/revision source hashes match the recorded source manifest.
- Revised manifests contain 1,638 task-specific cycle IDs, 117 outer splits and 2,166 inner splits; participant separation checks pass.
- Complete-family FDR calculations reproduce the recorded BH counts: Pearson 11/396, Spearman 0/396, gait-stage 3/19.
- Available selection records and five-seed summaries are internally consistent and correctly labelled as one-participant sensitivity results.
- 26 pipeline CPU tests pass (including baseline, leakage, metadata, frozen external-evaluation and harmonization guards).
- Simulated checkpoint tests pass after interruption immediately following a saved epoch; changed plans and corrupted prediction caches are rejected; incomplete seed sets do not enter balanced reports.

These checks do not execute GPU training and do not reconstruct the original 3,600 runs. TensorFlow/GPU training must be run in the compatible Colab environment. Synthetic software tests are not scientific results. Raw-data-dependent agreement/contact recomputation requires the separately supplied original prediction workbooks.
