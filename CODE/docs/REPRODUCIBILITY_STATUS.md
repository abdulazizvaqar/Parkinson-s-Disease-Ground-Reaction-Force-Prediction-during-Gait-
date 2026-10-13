# Reproducibility audit

| Reviewer request | Included evidence | Remaining gap |
|---|---|---|
| Training/analysis code | Original scripts; revised complete, coordinate and seed workflows; statistical analyses; Colab notebooks | New code is not a retroactive reconstruction of the original training execution |
| Preprocessing scripts | Original data assembly/plot scripts; revised paired Raw loader and external importer | Raw-marker event detection, filtering and full original source-to-normalized-cycle pipeline are not supplied |
| Dataset split identifiers | 1,638 task-specific cycle records; 117 revised outer splits; 2,166 planned inner splits | These deterministic revised splits do not establish historical cycle-level validation membership for the original run |
| Selected hyperparameters | AP-left/SUB01 LSTM and GRU configurations, selected epochs and source-record hashes | Other original/revised fold selections and original per-trial logs missing |
| Architecture details | Original builders and revised generic builder; exact available widths/losses/settings | Full original 3,600 instantiated model configurations unavailable |
| Stopping rules | Original early stopping/search logic; revised inner-LOSO epoch rule; frozen seed budget/checkpoint policy | Original actual stopped epoch and selected checkpoints per trial/fold unavailable |
| Software versions | Exact completed revised-seed/source software; local analysis software | Original study's complete environment/hardware export unavailable |
| External harmonization | Original inference script; revised mapping, sign, unit, phase interpolation and readiness checks | Original exact transformations, verified identities, visits, units and axes not fully recovered; revised external evaluation remains blocked |
| Repeated training | 10 completed fits, five seeds, two fixed AP-left/SUB01 configurations | Full cohort/component/Conv1D-LSTM repeats missing |
| Code availability | Reviewable repository package | GitHub publication URL and immutable release/commit not yet created |

The 3,600 count is a manuscript claim. This package does not contain a ledger verifying that count or enabling bitwise reproduction. Do not replace that missing ledger with the planned revised runs. Trial count, model fits within inner folds, final refits and repeated seeds are different counts.

## Completed results

- FDR: 396 Pearson tests, 11 BH findings; 396 Spearman tests, 0 BH findings; 19 gait-stage Kruskal–Wallis tests, 3 BH findings. BY sensitivity retains 1 Pearson finding and 0 gait findings. Additional 418 spatiotemporal-clinical tests form a separate family, with 75 BH findings.
- Internal agreement: 18 combinations, left n=20 / 283 cycles and right n=19 / 263 cycles. Participant summary bias/limits/CIs and phase-specific results calculated; raw prediction workbooks required for regeneration.
- Contact/peak: full/contact errors for all saved predictions; 260 left and 215 right measured-two-peak cycles across 18 participants each. No actual per-cycle duration or clinical threshold supplied.
- Seeds: LSTM RMSE 0.339888 ± 0.126806 and GRU 0.397021 ± 0.015115 across five seeds, but one test participant only.

## Needed for the original training audit

Supply an original run ledger with task, side, outer participant, candidate ID, full parameters, train/validation/test cycle IDs, selected checkpoint, epoch counts, objective values, success/failure state, random seeds if recorded, and software/hardware details. Supply original tuner directories and preprocessing/harmonization scripts or records. A seed not recorded at original execution cannot be recovered reliably from final predictions.
