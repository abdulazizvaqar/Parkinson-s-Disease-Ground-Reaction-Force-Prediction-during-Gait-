# Ankle Kinematics Driven Prediction of Ground Reaction Forces During Gait Assessment in Parkinson’s Disease Patients: A Deep Learning Approach

Ground reaction force (GRF) analysis is essential for characterizing gait impairments in Parkinson’s disease (PD), where kinetic alterations reflect disease severity and progression. However, conventional GRF assessment relies on expensive and laboratory-bound force platforms, limiting routine clinical use. This study proposes a deep learning-based framework for predicting three-dimensional GRF components antero-posterior (AP), vertical (vGRF), and medio-lateral (ML) using only ankle position data over a complete gait cycle, substantially reducing instrumentation requirements. Motion data from 26 individuals with PD were analyzed and three sequence-to-sequence architectures Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and one-dimensional convolution (Conv1D-LSTM ) were trained to estimate GRF profiles. Correlation coefficients exceeded 0.95 for AP and 0.96 for vGRFs. External validation showed component-dependent performance, with relatively smaller degradation for AP and ML GRFs but substantially increased errors for vGRF, indicating limited cross-dataset generalization for the vertical component. After false-discovery-rate correction, 11 of 396 Pearson associations between prediction error and clinical measures remained significant. These associations were exploratory and were not retained in the corresponding Spearman analysisCritically, model prediction errors correlated significantly with disease severity: 62 of 396 clinical correlations were significant (p<0.05), with motor dysfunction (UPDRS-III) as dominant correlation (r=0.66). Associations between prediction error and clinical characteristics were interpreted as exploratory model-performance findings rather than evidence of disease sensitivity. The results support the feasibility of reconstructing GRF waveforms from laboratory motion-capture-derived ankle trajectories.

Code and available audit records for the Parkinson's ankle-trajectory GRF manuscript and its methodological revision. This package includes original scripts, revised participant-wise nested LOSO code, Colab notebooks, baseline/coordinate controls, FDR correction, participant-level agreement, force-defined contact/peak analyses, and repeated-seed training.

## Layout

- `legacy/`: 13 original user-supplied Python scripts, copied byte-for-byte, including preprocessing/assembly, LSTM, GRU, Conv1D-LSTM, external inference and figures/statistics. Some legacy statistical procedures have known limitations; preservation is for provenance, not endorsement.
- `revised/complete/`: nested LOSO and complete revised pipeline, baseline models, final-refit and guarded external import/evaluation.
- `revised/coordinates/`: frozen-setting coordinate checks and saved-model evaluation; preserves the source versions needed by the coordinate notebook.
- `revised/seeds/`: fixed-setting repeated-seed orchestration and optimizer checkpoints.
- `analysis/`: standalone agreement, contact/peak and FDR calculations.
- `notebooks/`: self-contained Colab entry points; outputs cleared, code unchanged.
- `metadata/`: cycle IDs, planned inner/outer split IDs, available selected settings, software records and source hashes.
- `templates/external/`: source-workbook audit and incomplete mapping templates. Blank/unconfirmed fields are deliberate and prevent unsupported external evaluation.
- `results/`: aggregate audited outputs; no raw ankle/force waveforms, trained weights or participant clinical records are distributed.
- `requirements/`: separate verified revision-training and local-analysis environments. These are not claimed to recreate the original study environment.

## Start with Colab

Upload the appropriate notebook in `notebooks/` to Colab. Mount the same Drive project, select GPU for training, check the configuration paths, and run cells in order. Package changes require restarting the runtime once. Training checkpoints remain in Drive.

1. `Nested_LOSO_GRF_Complete_Run.ipynb`: complete revised workflow, subject-wise selection, baselines and guarded external analysis. Full execution remains incomplete.
2. `Coordinate_Robustness_Updated_Colab.ipynb`: frozen-setting normalization/perturbation sensitivity or fresh nested coordinate re-selection. Only scalar translation analyses were completed in the supplied run.
3. `Repeated_Seed_Training_GRF_Colab.ipynb`: five-seed refits of completed configurations, without new tuning. Defaults discover available selections; missing settings are not replaced.
4. `Test_Current_Trained_Models_Colab.ipynb`: audit/test saved models and predictions.
5. `Shida_FDR_Clinical_Gait_Colab.ipynb`: complete-family multiple-testing correction.

Default revision Drive project: `MyDrive/Shida_Nested_LOSO`. Internal ZIP: `datasets/Shida-LOSO-ANKLE.zip`. Selection experiments: `complete_studies/reviewer_complete_v1/<task>/absolute/`. The seed notebook also requires completed original-selection/refit metadata. `Internal.rar` contains predictions, not ankle inputs or complete training settings.

## Local commands

Use a separate environment for analysis versus GPU training; see requirements files. Python 3.13.15 was recorded for the completed Colab revision/seed runs. Local analyses were prepared under the Python version in `metadata/local_analysis_software.json`.

```bash
python -m pip install -r requirements/analysis.txt
python analysis/run_fdr.py --clinical results/fdr/inputs/01_Correlation_Results_Complete_All_Metrics.csv --gait results/fdr/inputs/ST_by_HY_Stage_Statistics.csv --spatiotemporal-clinical results/fdr/inputs/01_Spatiotemporal_vs_Clinical_All.csv --output work/fdr
python analysis/recalculate_agreement.py --input /path/to/extracted/Internal --output work/agreement --bootstrap 10000
python analysis/recalculate_stance.py --input /path/to/extracted/Internal --output work/stance
python analysis/plot_agreement_summary.py --input results/agreement/Participant_Summary_Agreement.csv --output work/Participant_Level_Agreement_Summary.png
```

The original `Internal.rar` must be extracted using a RAR-capable utility; scripts read its original Excel sheets. Cycle pairing and participant IDs are verified before computing metrics.

Audit revised raw inputs without training:

```bash
python revised/complete/nested_loso.py --data /path/to/Shida-LOSO-ANKLE.zip --mode audit --output work/audit
```

The CLI nested implementation and the complete-pipeline notebook are separate entry points and use distinct metadata layouts. Do not substitute one output directory for the other.

## Data sources

The manuscript cites Shida et al. (2023), DOI https://doi.org/10.3389/fnins.2023.992585, and Viglialoro et al. (2025), DOI https://doi.org/10.1038/s41597-025-06209-9. Consult the authors' published repositories and applicable data terms. Supplied ZIP/workbook snapshots are excluded from this code release. Metadata hashes identify the analyzed snapshots; source columns preserve original cycle ordering, not independently verified trial identities.

## Statistical scope

Participants are the unit of summary/inference. Repeated cycles, phases, architectures and seeds are not additional participants. Agreement limits apply to participant-average summaries; phase intervals are pointwise. Contact masks are force-derived exploratory stance proxies. Cycle durations are unavailable, so phase-integrated AP areas are not physical impulses and loading rates are not recovered. Clinical acceptability thresholds have not been validated.

FDR is applied across all tests, not the significant-only subsets. Pearson, Spearman and gait-stage families are handled separately. Optional Benjamini–Yekutieli outputs are included as an arbitrary-dependence sensitivity check; they are more conservative than BH. The former count of 62 combined Pearson/Spearman nominal findings is not a count of 62 Pearson findings. The supplied freezer stratification had incorrect full-cohort results in the freezer column and is not included as validated subgroup evidence.

## Validation and publishing

Run `python tools/validate_release.py` and `python tools/validate_seed_restart.py`. The latter uses a simulated backend to test interruption/resume, not TensorFlow training. The pipeline CPU tests run with `PYTHONPATH=revised/complete python -m unittest discover -s revised/complete/tests -p "test_*.py"`. Analysis/source syntax, split separation, metadata and reference FDR counts can be checked without TensorFlow. GPU training is not performed by this validation. Publish the package contents to the authors' chosen GitHub repository; record a commit/tag in the manuscript, rather than only linking the changing default branch. This prepared package has no live GitHub URL yet. License and author/citation metadata must reflect the authors' intended release; none are invented here.
