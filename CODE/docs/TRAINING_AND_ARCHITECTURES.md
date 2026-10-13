# Architectures and stopping rules

## Original scripts

Authoritative builders are in `legacy/`; their bytes and filenames are retained in `metadata/source_manifest.json`. LSTM uses encoder, RepeatVector, decoder, additional recurrent layers and a time-distributed dense output. Conv1D-LSTM adds a 64-filter, kernel-size-3 ReLU convolution and max pooling before the recurrent encoder/decoder. GRU uses a sequence-preserving recurrent stack and dense scalar output. Both initial widths and additional widths vary by builder/script version; consult the corresponding builder rather than assume one architecture applies to every original run. The supplied alternative LSTM script is archived separately; its presence does not determine which version produced each reported fold.

Original Bayesian tuning uses validation MAE, up to 10 trials, up to 1,000 epochs and a 10% cycle-level validation split. EarlyStopping monitors val_mae, patience=5, mode=min and restores best weights. Learning-rate candidates include 0.01, 0.001 and 0.0001; optimizers include Adam, SGD and RMSprop; loss choices vary by family. No reliable fixed-seed/repeated-seed original record is available. Exact original trial outcomes are absent.

## Revised participant-wise pipeline

`revised/complete/models.py`, `revision_models.py` and `protocol.py` define builders/search and training. Every outer test participant is excluded from all selection. Each inner validation fold holds out a complete development participant. Candidate score is equal-participant mean validation MAE. The selected epoch budget is the ceiling of the median best epoch across the winning complete inner LOSO candidate. Refit then uses all outer-development participants with no outer-test-driven early stopping. CLI flags/config determine patience/search budget/profile; they must be saved and never changed on resume.

Raw loader uses paired `in`/`out` sheets, rejects nonfinite values, preserves cycle ordering and tracks provenance. It does not apply the legacy silent zero filling. Optional standardization is fit on training inputs only; target GRF values retain original units. This change in protocol is intentional and does not imply exact reproduction of the original pipeline.

## Available selected models

AP-left, SUB01 held out, 19 development participants, 7 test cycles. Both batch size 32, activation tanh, optimizer Adam, learning rate 0.001.

- LSTM: encoder 100; decoder 128; 3 additional recurrent layers of 260, 388 and 452 units; time-distributed Dense(1); MAE loss; fixed refit budget 16 epochs.
- GRU: initial sequence-preserving 32-unit GRU; 1 additional 32-unit GRU; Dense(1); MSE loss; fixed refit budget 6 epochs.

Exact JSON and source selection hashes are in `metadata/available_selected_configurations.json`. These are completed revision configurations, not the missing full original fold archive.

Five master seeds: 11,22,33,44,55. Actual per-fold initialization/per-epoch permutation seeds are SHA256-derived from master seed, task, outer participant, model and phase labels. Fixed epoch budgets are retained. Complete-epoch Keras checkpoints include optimizer state; optimizer iterations determine the resume epoch. TensorFlow deterministic operations are enabled. Cross-device bitwise identity is not promised.

## Baselines and coordinate controls

Revised models include equal-participant mean training waveform, phase-only Fourier basis, ridge regression and a one-hidden-layer MLP. Input-dependent baseline scaling uses only development inputs. Translation normalization subtracts the initial ankle position within each cycle. Additional pelvis/alignment/body/rotation transformations require genuine anatomical data and verified axes/units; scalar translation tests do not prove independence from laboratory-specific information.
