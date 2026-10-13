# Internal GRF agreement reanalysis

This analysis recalculates agreement from the saved internal predictions in Internal.rar. It does not train models or evaluate the external cohort. It addresses the serial-correlation concern by using participant-level summaries and by evaluating phase-specific differences across participants, never treating 100 waveform points as 100 independent observations.

## Data and checks

18 workbooks cover AP, vertical (labelled LONG in the originals), and ML force, left/right, and Conv1D-LSTM, LSTM and GRU. Each left-side combination contains 20 participants and 283 cycles; each right-side combination contains 19 participants and 263 cycles. These are the same participants and cycles evaluated by different models, not independent model replicates. All 100-point paired waveforms are finite. No cycles were excluded in this reanalysis. Original study exclusions cannot be recovered from this archive. Cycle identifiers are unique within each workbook; participant IDs are recovered from SUBxx in the Name sheet. Saved cycle RMSE and NRMSE reproduce from paired measured/predicted arrays, and measured arrays match by cycle identifier across models. The archive does not independently prove the provenance or held-out status of predictions; internal LOSO provenance is assumed from the supplied workflow.

## Participant summary analysis

For each cycle, calculate the arithmetic mean of the 100 force samples, maximum force, and minimum force. For each participant, average each of these quantities across that participant's cycles, separately for measured and predicted signals. Each participant then contributes one paired summary for each component, side, model and summary type. Maxima and minima are calculated per cycle before averaging; they are not extrema of the averaged waveform. These full-cycle extrema are not separately annotated stance peaks or first/second vertical peaks.

Difference = predicted minus measured. Bias = arithmetic mean of the participant differences. Lower/upper normal-based limits of agreement = bias ± 1.96 × sample SD of participant differences (ddof=1). Each participant has equal weight regardless of cycle count. Limits describe agreement of participant-average summaries, not individual cycles or instantaneous waveform errors.

95% percentile bootstrap confidence intervals for bias and each limit use 10,000 participant resamples with replacement, retaining paired summaries, and seed 20261012. Empirical 2.5th/97.5th percentile limits and their bootstrap confidence intervals are supplied as distributional sensitivity estimates. With only 19–20 participants, extreme percentiles and bootstrap uncertainty remain imprecise. Shapiro–Wilk p-values are exploratory diagnostics, not another confirmatory hypothesis family. Non-normal distributions, particularly vertical differences, limit interpretation of normal-based limits as bounds containing 95% of future differences. Neither bootstrap CIs nor empirical limits establish clinical acceptability.

## Phase-specific analysis

Average all measured cycles and all predicted cycles within each participant, giving one paired mean waveform per participant. At each phase index separately, compute bias and normal-based limits across participant differences. Bootstrap draws select complete participant waveforms, preserving serial dependence. The resulting confidence intervals are pointwise, not simultaneous bands, and no inferential tests treating phases as independent are performed. These limits apply to participant-mean waveforms, not individual cycles. Phase-dependent mean and variance are shown rather than pooled into one waveform-point SD.

## Important interpretation

A small full-cycle mean bias can result from cancellation of positive and negative errors, especially for AP and ML. Maximum/minimum and phase-specific results must accompany mean-force results. Models show larger amplitude discrepancies than the small signed means alone suggest. For example AP-left LSTM mean-force bias is -0.003038, while its maximum-force bias is -0.269070. Vertical-left LSTM maximum-force bias is -1.218029, with bootstrap bias CI [-1.933793, -0.618025]. Its normal-based limits are [-4.222048, 1.785990], but the difference distribution is non-normal and those limits require caution.

Units are retained from the original workbooks; the manuscript identifies GRF units as N/kg. There is no independent calibration or unit label in these sheets, so this analysis does not verify that normalization anew. Uncertainty is conditional on the saved model predictions and available participants; it does not include refitting, hyperparameter selection or dependence introduced by overlapping LOSO training sets. No prespecified clinical acceptability threshold is available, so interchangeability cannot be claimed.

## Files

- Participant_Summary_Agreement.csv: 54 rows (18 combinations × 3 summaries), with bias, limits, 95% CIs, normality diagnostics and empirical sensitivity limits.
- Participant_Summary_Values.csv: 1,053 participant summary records, including paired values and cycle counts.
- Phase_Specific_Agreement.csv: 1,800 phase-specific estimates with pointwise 95% CIs.
- Data_Audit.json: input filenames, SHA256 hashes and integrity checks.
- Agreement_Figures.pdf: 12 pages of participant-summary and phase-specific plots. PNG copies are also supplied.
- recalculate_agreement.py: reproducible analysis code. Run against the extracted original Excel workbooks; dependencies are numpy, pandas, openpyxl, scipy and matplotlib.

## Suggested reviewer response

We thank the reviewer for identifying the dependence between time-normalized waveform points. We have recalculated internal agreement using participant-level summaries of cycle-mean, maximum and minimum GRF, averaging cycle summaries within each participant. Each participant contributes one paired measured/predicted value per analysis (20 participants for left-side models and 19 for right-side models). We report mean bias, lower and upper limits of agreement, and their 95% confidence intervals from 10,000 participant bootstrap resamples. In an additional phase-specific analysis, we compare participant-mean waveforms at each gait phase and resample complete participant waveforms to preserve temporal dependence; confidence intervals are pointwise. We include empirical percentile limits as a sensitivity analysis and acknowledge that non-normal differences limit the interpretation of normal-based agreement limits. The new estimates concern participant-average summaries and waveforms and do not establish agreement for individual cycles or clinical interchangeability.

The calculation is complete. Manuscript replacement text, supplementary tables/figures and the response letter have not been edited by this analysis. Do not state those documents have been updated until the replacements are made.

## Reference

Bland JM, Altman DG. Agreement between methods of measurement with multiple observations per individual. Journal of Biopharmaceutical Statistics. 2007;17:571–582. doi:10.1080/10543400701329422. https://www-users.york.ac.uk/~mb55/meas/bland2007.pdf
