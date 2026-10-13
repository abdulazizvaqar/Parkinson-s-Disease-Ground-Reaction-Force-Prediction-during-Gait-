# Internal force-defined contact and vertical peak reanalysis

## Scope

Recalculation from the 18 saved internal workbooks in Internal.rar; no model retraining and no external data. The same cycles are evaluated by three architectures, so architecture outputs do not increase the participant sample size. Left results: 20 participants / 283 cycles; right: 19 participants / 263 cycles. Per-workbook saved RMSE is reproduced to floating-point tolerance. Cycle identifiers are recovered from SUBxx_tr_model_output_index and matched between component files within each side; measured vertical targets match exactly across architectures. Archive provenance does not independently verify that predictions were held out; their internal LOSO provenance is assumed from the original workflow.

## Force-defined contact instead of annotated stance

No explicit cycle-level heel-strike/toe-off events are present. For each cycle, force-defined contact is measured vertical force > 1% of that cycle's measured vertical maximum. The same measured-force mask is used for all architectures and GRF components. Masks contain one continuous segment per cycle. This is an exploratory contact proxy, not a validated stance-event annotation. Thresholds are retrospective analysis rules and were not prespecified in the original study.

Sensitivity masks use 2% and 5%. The 1% and 2% masks are identical for all 546 side-specific cycles. The 5% masks change in 104 of 283 left cycles and 105 of 263 right cycles. All three sets of contact errors are supplied. Model settings and predictions are unchanged by threshold choice.

Compute per-cycle full-cycle RMSE and force-defined-contact RMSE and MAE. NRMSE is divided by the measured full-cycle range for both full and contact estimates to retain a common denominator; exported NRMSE columns are percentages. Then average cycle metrics within participant. Cohort summaries give equal weight to participants and include 95% percentile bootstrap confidence intervals from 10,000 participant resamples, seed 20261012. This bootstraps performance summaries conditional on saved predictions; it does not account for model refitting, hyperparameter selection, overlapping LOSO training sets, or annotation uncertainty.

## Vertical peak amplitude and timing

Overall vertical maximum is assessed within force-defined contact for all cycles. Peak amplitude error is predicted minus measured; timing is the difference in peak sample indices multiplied by 100/99, in percentage points of a complete gait cycle. It is not time in seconds. Resolution is approximately 1.01 percentage points per sample.

For exploratory first/second peak analysis, measured local peaks have prominence >= 5% of the cycle measured vertical maximum and a minimum separation of ceil(10% of contact samples). Zero padding is used at contact boundaries. If more than two peaks qualify, select the two with greatest prominence and order them in time. There are 260 eligible left cycles and 215 eligible right cycles, each across 18 participants. The remaining 23 left and 48 right cycles have no measured qualifying double peak and are excluded only from first/second peak measures, not full/contact errors or overall peak measures.

Divide the measured contact segment at the midpoint between the two selected measured peaks. Predicted maxima in the first/second reference windows provide the corresponding predicted amplitudes and timing. These window maxima can lie on a monotonic shoulder or a boundary; they must not be described as proof that two local peaks were reconstructed. A separate topology table reports how often predictions contain at least two qualifying local peaks, using the same measured-force prominence scale and contact interval. This is reference-window peak matching, not independently annotated early/late stance kinetics. Prominence and separation rules have not undergone additional sensitivity testing and require validation.

## Braking and propulsive waveform areas

Within measured-force contact, trapezoidally integrate max(-AP,0) for braking magnitude and max(AP,0) for propulsive magnitude. Integration uses normalized full-cycle phase (dx=1/99). Columns are explicitly named phase_area. Their units are GRF units × unit-normalized gait phase. They are NOT physical impulses, because cycle durations are absent. If cycle duration T seconds is later verified for each matched cycle, physical impulse equals phase area × T (in N s/kg if GRF is verified N/kg). Participant-average durations cannot replace cycle-specific durations. Conversion to force in N also requires verified mass normalization and participant body mass.

Physical loading rates are not calculated. Actual per-cycle time intervals and a justified, noise-robust loading-rate definition are required. The source motion-capture/force-platform sampling frequency does not reconstruct duration after each cycle has been resampled to 100 points.

## Main findings

Force-defined contact RMSE exceeds full-cycle RMSE for every combination. Relative increases in cohort means range from 8.12% to 24.48%. This supports the concern that including swing reduces full-cycle error. Example LSTM results:

| Component / side | Full-cycle mean participant RMSE | Contact RMSE | 95% CI for contact RMSE |
|---|---:|---:|---:|
| AP left | 0.3261 | 0.4057 | 0.3303 to 0.4962 |
| Vertical left | 1.4917 | 1.7088 | 1.2698 to 2.2653 |
| ML left | 0.1987 | 0.2420 | 0.1982 to 0.2890 |

For left vertical LSTM, first/second reference-window peak amplitude absolute errors average 1.3923 and 1.3522 in original GRF units, across 18 participants with 260 eligible cycles. Absolute peak timing errors average 5.51 and 6.77 percentage points of the gait cycle. Predicted waveforms contain qualifying double peaks in only 110/260 eligible cycles for this combination. Topology failures should be reported alongside amplitude/timing summaries.

Units are retained from the original workbooks. The manuscript states N/kg, but these worksheet exports do not independently label or validate the unit conversion. No new calibration has been performed.

## Clinical acceptability

No clinically validated numerical error threshold is established by these retrospective results. A threshold selected from the observed errors is not a prespecified criterion and may be optimistic. The present data support laboratory feasibility and exploratory performance evaluation, not interchangeability with force plates or clinical deployment.

For future prospective validation, specify the exact clinical decision (e.g. detecting a rehabilitation-related change in propulsive impulse); use independent force-platform repeatability and the smallest decision-relevant change to set tolerances for amplitude, timing and impulse before examining validation predictions. Evaluate the confidence bounds and participant-level error distributions against these fixed tolerances. Those clinical repeatability/change data are absent here, so numerical tolerances cannot be responsibly supplied. The numerical clinical-acceptability and physical impulse/loading-rate portions of the reviewer comment remain incomplete.

## Files

- Cycle_Level_Stance_and_Peak_Metrics.csv: 4,914 records, comprising the same side-specific cycles across 18 combinations. Not 4,914 independent participants.
- Participant_Level_Metric_Values.csv: 6,012 participant metric records, with valid-cycle counts.
- Participant_Level_Metric_Summaries.csv: 312 summaries across metrics/thresholds, including analysis-specific N and confidence intervals.
- Peak_Eligibility_and_Topology.csv: all-cycle denominators and prediction double-peak counts among measured-eligible cycles.
- Audit.json: hashes and contact/peak eligibility checks.
- Stance_and_Peak_Figures.pdf and PNGs: plots of full/contact RMSE and reference-window vertical peak amplitude errors.
- recalculate_stance.py: reproducible Python calculation; dependencies numpy, pandas, openpyxl, scipy and matplotlib. Run with --input pointing to extracted original Excel files.

## Suggested response to the agreement comment

We thank the reviewer for highlighting the serial dependence of time-normalized waveform points. We have recalculated internal agreement using participant-level summaries of mean, maximum and minimum force, with cycle summaries averaged within each participant. Each analysis uses one paired measured/predicted value per participant (20 left-side and 19 right-side participants). We report mean bias, lower and upper limits of agreement, and their 95% confidence intervals from 10,000 participant bootstrap resamples. Additional phase-specific analyses compare participant-mean waveforms and resample entire participant waveforms, preserving temporal dependence. Their confidence intervals are pointwise. Empirical percentile limits are also provided because some difference distributions are non-normal. These analyses assess participant-average summaries and waveforms, rather than treating waveform samples as independent observations or claiming clinical interchangeability.

## Suggested response to the stance/kinetic-feature comment

We thank the reviewer for highlighting the influence of swing-phase samples. We have recalculated errors within a force-defined contact interval using measured vertical force above 1% of each cycle's maximum, with 2% and 5% thresholds as sensitivity analyses. This is an exploratory proxy for stance because explicit event annotations were unavailable. Cycle errors were averaged within participants and cohort estimates include participant-bootstrap confidence intervals. Contact-only RMSE exceeded full-cycle RMSE for all combinations, with increases of approximately 8–24% in cohort means. We also calculated vertical reference-window first/second peak amplitude and timing errors where measured cycles exhibited two qualifying peaks, and separately reported the frequency with which predictions reproduced two local peaks. Peak timing is expressed in percentage points of normalized gait phase. Braking and propulsive normalized-phase areas were calculated, but physical impulses and loading rates could not be recovered without matched cycle durations. No clinical acceptance threshold had been prespecified; consequently, these retrospective results support feasibility assessment and do not establish clinical acceptability. Prospective thresholds require an intended clinical decision and independent repeatability or clinically meaningful-change data.

Both responses describe calculations now completed. The manuscript and response-letter documents have not been edited by this calculation. Add Methods, Results, limitations and supplementary tables/figures before claiming document revisions are complete.
