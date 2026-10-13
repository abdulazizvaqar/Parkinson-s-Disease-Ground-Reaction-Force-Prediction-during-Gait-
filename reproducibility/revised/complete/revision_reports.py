"""Participant-level errors, trial-cluster intervals, and domain-shift accounting."""
from pathlib import Path
import numpy as np
import pandas as pd
from protocol import cycle_metrics, stable_seed, write_json

METRICS = ('rmse', 'nrmse', 'mae')

def prediction_table(ds, prediction, model):
    metrics = cycle_metrics(ds.y, prediction)
    table = pd.concat([ds.cycles.reset_index(drop=True), metrics], axis=1)
    table['model'] = model
    table['task'] = ds.task
    if 'participant_id' not in table:
        table['participant_id'] = ds.groups
    if 'visit' not in table:
        table['visit'] = 'internal'
    if 'trial_id' not in table:
        table['trial_id'] = 'unknown'
    # Pearson r is descriptive only; it does not replace error/baseline comparisons.
    centered_y = ds.y.reshape(len(ds.y), -1).astype(float)
    centered_p = prediction.reshape(len(prediction), -1).astype(float)
    centered_y -= centered_y.mean(axis=1, keepdims=True)
    centered_p -= centered_p.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(centered_y, axis=1) * np.linalg.norm(centered_p, axis=1)
    table['waveform_r'] = np.divide((centered_y * centered_p).sum(axis=1), denom,
                                   out=np.full(len(denom), np.nan), where=denom > 1e-12)
    return table

def bootstrap_interval(values, seed, repeats, statistic=np.median):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    estimates = statistic(rng.choice(values, size=(repeats, len(values)), replace=True), axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]))

def participant_table(cycles, repeats=2000, seed=2026):
    keys = ['task', 'model', 'participant_id', 'visit']
    if cycles.duplicated(keys + ['cycle_id']).any():
        raise ValueError('Duplicate predictions for the same participant/cycle/model: do not stack model replicates')
    rows = []
    for key, frame in cycles.groupby(keys, sort=True):
        known_trials = ~frame.trial_id.isin(['', 'unknown'])
        trial_ids = frame.trial_id.unique() if known_trials.all() else []
        row = dict(zip(keys, key))
        row.update(n_cycles=len(frame), n_trials=len(trial_ids) if len(trial_ids) else None,
                   undefined_nrmse_cycles=int(frame.nrmse.isna().sum()),
                   ci_method='trial-cluster percentile bootstrap; conditional on fitted model',
                   ci_status='estimated' if len(trial_ids) >= 2 else 'unavailable: fewer than two known trials')
        row['waveform_r'] = frame.waveform_r.mean() if 'waveform_r' in frame else np.nan
        for metric in METRICS:
            row[metric] = frame[metric].mean()
            row[f'{metric}_ci95_low'] = row[f'{metric}_ci95_high'] = np.nan
            if len(trial_ids) < 2:
                continue
            # Cluster resampling preserves all cycles within each sampled trial.
            groups = [frame.loc[frame.trial_id == t, metric].dropna().to_numpy() for t in trial_ids]
            rng = np.random.default_rng(stable_seed(seed, *key, metric))
            estimates = []
            for _ in range(repeats):
                draw = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                if len(draw):
                    estimates.append(draw.mean())
            if estimates:
                row[f'{metric}_ci95_low'], row[f'{metric}_ci95_high'] = np.quantile(estimates, [.025, .975])
        rows.append(row)
    return pd.DataFrame(rows)

def cohort_table(participants, repeats=2000, seed=2026):
    rows = []
    for key, frame in participants.groupby(['task', 'model', 'visit'], sort=True):
        for metric in METRICS:
            v = frame[metric].dropna().to_numpy()
            if not len(v):
                continue
            lo, hi = bootstrap_interval(v, stable_seed(seed, *key, metric), repeats)
            rows.append(dict(task=key[0], model=key[1], visit=key[2], metric=metric,
                n_participants=len(v), mean=float(v.mean()), median=float(np.median(v)),
                q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)),
                median_ci95_low=lo, median_ci95_high=hi,
                ci_method='participant percentile bootstrap; conditional on fitted pipeline'))
    return pd.DataFrame(rows)

def paired_comparisons(participants, reference='selected_deep', repeats=2000, seed=2026):
    rows = []
    base = participants[participants.model == reference]
    keys = ['task', 'visit', 'participant_id']
    for other in sorted(set(participants.model) - {reference}):
        joined = base.merge(participants[participants.model == other], on=keys, suffixes=('_deep', '_baseline'), validate='one_to_one')
        for (task, visit), frame in joined.groupby(['task', 'visit']):
            for metric in METRICS:
                # Positive = lower error in the deep model. Correlation is not the endpoint.
                delta = (frame[f'{metric}_baseline'] - frame[f'{metric}_deep']).dropna().to_numpy()
                lo, hi = bootstrap_interval(delta, stable_seed(seed, task, visit, other, metric), repeats)
                rows.append(dict(task=task, visit=visit, deep_model=reference, comparator=other, metric=metric,
                    n_paired_participants=len(delta), mean_improvement=float(delta.mean()) if len(delta) else np.nan,
                    median_improvement=float(np.median(delta)) if len(delta) else np.nan,
                    median_improvement_ci95_low=lo, median_improvement_ci95_high=hi,
                    interpretation='positive favors deep; descriptive intervals, no multiplicity-adjusted superiority claim'))
    return pd.DataFrame(rows)

def stratified_table(participants, people, columns, minimum=5, repeats=2000, seed=2026):
    if people.participant_id.duplicated().any():
        raise ValueError('One metadata row per participant is required')
    data = participants.merge(people, on='participant_id', how='left', validate='many_to_one')
    rows = []
    for col in columns:
        if col not in data:
            rows.append(dict(characteristic=col, stratum='unknown', status='metadata unavailable'))
            continue
        data[col] = data[col].replace('', 'unknown').fillna('unknown')
        for key, group in data.groupby(['task', 'model', 'visit', col], dropna=False):
            n = group.participant_id.nunique()
            status = 'reported' if n >= minimum and str(key[3]).lower() not in ('unknown', 'nan') else 'suppressed: sparse or unknown'
            for metric in METRICS:
                v = group[metric].dropna().to_numpy()
                sufficient = status == 'reported' and len(v) >= minimum
                lo, hi = bootstrap_interval(v, stable_seed(seed, *key, col, metric), repeats) if sufficient else (np.nan, np.nan)
                rows.append(dict(task=key[0], model=key[1], visit=key[2], characteristic=col, stratum=key[3],
                    n_participants=n, n_metric_defined=len(v), metric=metric, status=status if sufficient else 'suppressed: sparse or unknown',
                    median=float(np.median(v)) if sufficient else np.nan,
                    median_ci95_low=lo, median_ci95_high=hi))
    return pd.DataFrame(rows)

def save_reports(cycles, directory, cfg, people=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cycles.to_csv(directory / 'cycle_metrics.csv', index=False)
    participants = participant_table(cycles, cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    participants.to_csv(directory / 'participant_metrics.csv', index=False)
    cohort_table(participants, cfg['bootstrap_resamples'], cfg['bootstrap_seed']).to_csv(directory / 'cohort_summary.csv', index=False)
    paired_comparisons(participants, repeats=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed']).to_csv(directory / 'paired_baseline_comparisons.csv', index=False)
    if people is not None:
        metadata = people.copy()
        if 'age' in metadata:
            ages = pd.to_numeric(metadata.age, errors='coerce')
            # Boundaries are in the locked prespecification, not chosen from errors.
            metadata['age_band'] = pd.cut(ages, bins=cfg['age_bin_edges'], right=False).astype(str)
        stratified_table(participants, metadata, cfg['strata'], cfg['minimum_stratum_n'],
                         cfg['bootstrap_resamples'], cfg['bootstrap_seed']).to_csv(directory / 'stratified_results.csv', index=False)
    return participants

def audit_external(cohort, visits, directory, internal_context):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ledger = cohort.ledger.copy()
    ledger['analysis_included'] = ledger.include & ledger.visit.isin(visits)
    ledger['analysis_exclusion_reason'] = np.where(~ledger.include, ledger.exclusion_reason,
                                                  np.where(~ledger.visit.isin(visits), 'visit excluded by prespecified analysis policy', ''))
    ledger.to_csv(directory / 'cycle_flow.csv', index=False)
    trials = cohort.trials.copy()
    keys = ['participant_id', 'trial_id', 'visit', 'side']
    counts = ledger[ledger.analysis_included].groupby(keys).size().rename('valid_cycles').reset_index()
    trials = trials.merge(counts, on=keys, how='left')
    trials['valid_cycles'] = trials.valid_cycles.fillna(0).astype(int)
    trials['analysis_included'] = trials.include & trials.visit.isin(visits) & trials.valid_cycles.gt(0)
    trials['analysis_exclusion_reason'] = np.where(~trials.include, trials.exclusion_reason,
        np.where(~trials.visit.isin(visits), 'visit excluded by prespecified analysis policy',
        np.where(trials.valid_cycles.eq(0), 'no valid analyzed cycles', '')))
    trials.to_csv(directory / 'trial_flow.csv', index=False)
    roster = cohort.participants.copy()
    pc = ledger[ledger.analysis_included].groupby('participant_id').size()
    roster['valid_cycles'] = roster.participant_id.map(pc).fillna(0).astype(int)
    roster['analysis_exclusion_reason'] = np.where(roster.valid_cycles.eq(0),
        'no analyzed eligible cycles; see trial/cycle exclusion ledgers', '')
    roster.to_csv(directory / 'participant_flow.csv', index=False)
    for name, frame in [('cycle', ledger), ('trial', trials)]:
        frame.loc[~frame.analysis_included].groupby(['visit', 'analysis_exclusion_reason']).size().rename('count').reset_index().to_csv(directory / f'{name}_exclusion_reasons.csv', index=False)
    rows = []
    for visit in sorted(set(trials.visit) | set(ledger.visit)):
        l, t = ledger[ledger.visit == visit], trials[trials.visit == visit]
        rows.append(dict(visit=visit, n_participants_with_trials=t.participant_id.nunique(),
            n_trials_recorded=len(t), n_candidate_cycles_recorded=len(l),
            n_participants_analyzed=l.loc[l.analysis_included, 'participant_id'].nunique(),
            n_trials_analyzed=int(t.analysis_included.sum()), n_valid_cycles=int(l.analysis_included.sum())))
    pd.DataFrame(rows).to_csv(directory / 'cohort_flow_by_visit.csv', index=False)
    write_json(directory / 'cohort_flow_totals.json', {'n_participants_in_roster': len(roster),
        'n_participants_analyzed': int(roster.valid_cycles.gt(0).sum()),
        'n_participants_without_analyzed_cycles': int(roster.valid_cycles.eq(0).sum()),
        'n_trials_recorded': len(trials), 'n_trials_analyzed': int(trials.analysis_included.sum()),
        'n_valid_cycles': int(ledger.analysis_included.sum()), 'visit_policy': visits})
    keys = sorted(set(internal_context) | set(cohort.context))
    pd.DataFrame([{'characteristic': k, 'internal': str(internal_context.get(k, 'unknown')),
                   'external': str(cohort.context.get(k, 'unknown'))} for k in keys]).to_csv(directory / 'domain_shift_context.csv', index=False)
