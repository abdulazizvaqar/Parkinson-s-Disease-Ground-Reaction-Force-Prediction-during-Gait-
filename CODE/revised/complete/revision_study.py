"""Extended nested study -> full-cohort refit -> locked external domain-shift test."""
import argparse
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
import json
import importlib.metadata
import platform
import numpy as np
import pandas as pd
from protocol import write_json, read_json, parameter_hash, stable_seed
from revision_data import (legacy_cohort, load_cohort, coordinate_dataset, digest_file, MODES, harmonize_input_units)
from revision_models import BASELINES, DEEP, tune_family, fit_frozen_model, predict_saved, mean_ensemble
from revision_reports import save_reports, prediction_table, audit_external, bootstrap_interval

def default_config():
    return dict(seed=42, search_profile='original', max_trials=10, epochs=1000,
        patience=5, batch_size=32, verbose=0, scale_input=False, smoke=False,
        families=list(BASELINES + DEEP), coordinate_mode='absolute',
        architecture_rule='minimum complete inner-LOSO participant MAE; ties lexical family name',
        final_epoch_rule='ceiling of median best epoch from selected full-internal LOSO candidate',
        external_visits=['baseline'], external_status='previously_evaluated',
        include_loso_ensemble=False, ensemble_rule='equal-weight arithmetic mean of all outer-selected deep waveforms',
        bootstrap_resamples=2000, bootstrap_seed=2026, minimum_stratum_n=5,
        strata=['sex', 'medication_state', 'HY_stage', 'age_band', 'intervention_group'],
        age_bin_edges=[0, 65, 75, 120],
        translation_offsets_m=[[0.25, 0., 0.], [-0.25, 0., 0.], [0., 0.25, 0.], [0., -0.25, 0.]],
        rotation_angles_deg=[-15., 15.], run_perturbations=False,
        scope='one side/component per experiment; external visit summaries remain separate')

def validate_config(cfg):
    if set(default_config()) - set(cfg):
        raise ValueError('Incomplete configuration')
    if cfg['coordinate_mode'] not in MODES or cfg['external_status'] not in ('untouched_new_holdout', 'previously_evaluated'):
        raise ValueError('Invalid coordinates/external status')
    if not cfg['external_visits'] or set(cfg['external_visits']) - {'baseline', 'post_intervention'}:
        raise ValueError('Prespecify baseline and/or post_intervention; reports never pool visits')
    if not cfg['families'] or set(cfg['families']) - set(BASELINES + DEEP) or len(set(cfg['families'])) != len(cfg['families']):
        raise ValueError('Unknown or duplicate model family')
    if min(cfg['max_trials'], cfg['epochs'], cfg['batch_size']) < 1 or cfg['patience'] < 0:
        raise ValueError('Invalid training settings')
    if cfg['bootstrap_resamples'] < 100 or cfg['minimum_stratum_n'] < 2:
        raise ValueError('Use >=100 bootstrap draws and >=2 participants per reported stratum')
    if cfg['seed'] < 0 or cfg['seed'] >= 2**31:
        raise ValueError('Invalid seed')
    if np.any(np.diff(cfg['age_bin_edges']) <= 0):
        raise ValueError('Age bin edges must be strictly increasing')

def software():
    info = {'python': platform.python_version()}
    for name in ['numpy', 'pandas', 'scikit-learn', 'tensorflow', 'keras', 'keras-tuner', 'joblib']:
        try:
            info[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info[name] = None
    return info

def source_hashes():
    root = Path(__file__).parent
    return {p.name: digest_file(p) for p in [root / name for name in
        ['data.py', 'protocol.py', 'models.py', 'revision_data.py', 'revision_models.py', 'revision_reports.py', 'revision_study.py']]}

def bind_experiment(root, ds, cfg):
    validate_config(cfg)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    signature = {'config': cfg, 'task': ds.task, 'data_fingerprint': ds.fingerprint,
                 'subjects': ds.subjects, 'sources': source_hashes(), 'software': software()}
    path = root / 'prespecification.json'
    if path.exists():
        if read_json(path) != signature:
            raise ValueError('Prespecification, data, code, or software changed. Use the original settings or a new experiment.')
    else:
        if any(root.iterdir()):
            raise ValueError('Nonempty directory without a prespecification')
        write_json(path, signature)
    return signature

def choose_deep(selections):
    choices = [s for s in selections if s['family'] in DEEP]
    if not choices:
        raise ValueError('No deep family was evaluated')
    return min(choices, key=lambda s: (s['inner_subject_mae'], s['family']))

def run_nested(cohort, cfg, root, outer_subjects=None):
    root = Path(root)
    ds = coordinate_dataset(cohort, cfg['coordinate_mode'])
    bind_experiment(root, ds, cfg)
    selected = outer_subjects or ds.subjects
    if set(selected) - set(ds.subjects) or len(ds.subjects) < 3:
        raise ValueError('Unknown outer participant or fewer than three participants')
    for outer in selected:
        print(f'OUTER TEST = {outer}', flush=True)
        tr_ds = ds.subset([s for s in ds.subjects if s != outer])
        te_ds = ds.subset([outer])
        outer_dir = root / 'nested' / outer
        selections = []
        for family in cfg['families']:
            directory = outer_dir / family
            chosen = tune_family(tr_ds, family, cfg, directory / 'selection')
            selections.append(chosen)
            marker = directory / 'complete.json'
            if not marker.exists():
                model_dir = directory / 'refit'
                fit_frozen_model(tr_ds, chosen, cfg, model_dir)
                prediction = predict_saved(model_dir, te_ds.x)
                prediction_table(te_ds, prediction, family).to_csv(directory / 'cycles.csv', index=False)
                np.savez_compressed(directory / 'predictions.npz', cycle_ids=te_ds.cycles.cycle_id.to_numpy(str),
                                    measured=te_ds.y, predicted=prediction)
                write_json(marker, {'outer_subject': outer, 'family': family, 'inner_subject_mae': chosen['inner_subject_mae']})
        if set(cfg['families']) & set(DEEP):
            winner = choose_deep(selections)
            source = pd.read_csv(outer_dir / winner['family'] / 'cycles.csv', dtype={'participant_id': str, 'trial_id': str, 'cycle_id': str})
            source['model'] = 'selected_deep'
            source.to_csv(outer_dir / 'selected_deep.csv', index=False)
            write_json(outer_dir / 'winner.json', {'family': winner['family'], 'inner_subject_mae': winner['inner_subject_mae'],
                'refit': str((outer_dir / winner['family'] / 'refit').relative_to(root)),
                'rule': cfg['architecture_rule'], 'outer_subject': outer})
        write_json(outer_dir / 'complete.json', {'families': cfg['families'], 'outer_subject': outer})
    tables = [pd.read_csv(p, dtype={'participant_id': str, 'trial_id': str, 'cycle_id': str})
              for p in sorted((root / 'nested').glob('*/*/cycles.csv'))]
    tables += [pd.read_csv(p, dtype={'participant_id': str, 'trial_id': str, 'cycle_id': str})
               for p in sorted((root / 'nested').glob('*/selected_deep.csv'))]
    save_reports(pd.concat(tables, ignore_index=True), root / 'internal_reports', cfg)
    complete = [p.parent.name for p in (root / 'nested').glob('*/complete.json')]
    write_json(root / 'nested_coverage.json', {'available_subjects': ds.subjects, 'completed_subjects': sorted(complete),
                                             'complete': set(complete) == set(ds.subjects)})
    return root

def artifact_hashes(directory):
    return {str(p.relative_to(directory)): digest_file(p) for p in Path(directory).rglob('*') if p.is_file()}

def final_refit(cohort, cfg, root):
    root = Path(root)
    ds = coordinate_dataset(cohort, cfg['coordinate_mode'])
    bind_experiment(root, ds, cfg)
    if cfg['smoke']:
        raise ValueError('A smoke test cannot produce a research external-evaluation bundle')
    if set(cfg['families']) != set(BASELINES + DEEP):
        raise ValueError('Final release requires all four baselines and all three deep families')
    coverage = read_json(root / 'nested_coverage.json')
    if not coverage['complete'] or set(coverage['completed_subjects']) != set(ds.subjects):
        raise ValueError('Complete every internal outer participant before the final refit')
    freeze_path = root / 'frozen_external_plan.json'
    if freeze_path.exists():
        check_frozen(root)
        print('Frozen full-cohort models already exist; no retraining.', flush=True)
        return root
    selections = [tune_family(ds, family, cfg, root / 'full_internal_selection' / family) for family in cfg['families']]
    winner = choose_deep(selections)
    write_json(root / 'full_internal_selection' / 'architecture_selection.json',
               {'family': winner['family'], 'inner_subject_mae': winner['inner_subject_mae'],
                'note': 'Training selection score, not an unbiased performance estimate; use nested outer results for that.'})
    model_dirs = {}
    for chosen in selections:
        if chosen['family'] in DEEP and chosen['family'] != winner['family']:
            continue
        label = 'selected_deep' if chosen['family'] in DEEP else chosen['family']
        directory = root / 'final_models' / label
        fit_frozen_model(ds, chosen, cfg, directory)
        model_dirs[label] = str(directory.relative_to(root))
    ensemble = []
    if cfg['include_loso_ensemble']:
        ensemble = [read_json(root / 'nested' / s / 'winner.json')['refit'] for s in ds.subjects]
    files = artifact_hashes(root / 'final_models')
    hashes = {str(Path('final_models') / k): v for k, v in files.items()}
    for member in ensemble:
        hashes.update({str(Path(member) / k): v for k, v in artifact_hashes(root / member).items()})
    # External data/labels are never arguments to this function.
    frozen = {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
        'internal_context': cohort.context, 'internal_subjects': ds.subjects,
        'internal_global_ids': cohort.participants.get('global_subject_id', pd.Series(dtype=str)).replace('', np.nan).dropna().tolist(),
        'internal_fingerprint': ds.fingerprint, 'task': ds.task, 'model_directories': model_dirs,
        'selected_architecture': winner['family'], 'ensemble_members': ensemble, 'model_hashes': hashes,
        'source_hashes': source_hashes(), 'software': software(),
        'primary_model': 'selected_deep', 'inference_unit': 'participant; never model-times-participant',
        'external_status': cfg['external_status']}
    write_json(freeze_path, frozen)
    return root

def check_frozen(root):
    root = Path(root)
    frozen = read_json(root / 'frozen_external_plan.json')
    for name, expected in frozen['model_hashes'].items():
        if digest_file(root / name) != expected:
            raise ValueError('Frozen model changed: ' + name)
    if source_hashes() != frozen['source_hashes'] or software() != frozen['software']:
        raise ValueError('Source/environment changed since the model freeze')
    return frozen

def predict_frozen(root, frozen, x):
    output = {name: predict_saved(Path(root) / folder, x) for name, folder in frozen['model_directories'].items()}
    if frozen['ensemble_members']:
        output['loso_ensemble'] = mean_ensemble([predict_saved(Path(root) / p, x) for p in frozen['ensemble_members']])
    return output

def external_evaluation(root, external_directory):
    root = Path(root)
    frozen = check_frozen(root)  # Must exist BEFORE external files are opened.
    cfg = frozen['config']
    cohort = load_cohort(external_directory, frozen['task'], cfg['external_visits'], role='external')
    if not cohort.context.get('participant_disjointness_verified', False):
        raise ValueError('Verify that external participants are disjoint from the development cohort')
    external_global = cohort.participants.get('global_subject_id', pd.Series(dtype=str)).replace('', np.nan).dropna().tolist()
    if set(external_global) & set(frozen['internal_global_ids']):
        raise ValueError('Internal/external global participant identity overlap')
    if not cohort.context.get('units_verified', False) or not cohort.context.get('harmonization_verified', False):
        raise ValueError('Verify units, axis signs, gait events, and internal/external harmonization before evaluation')
    input_cohort = harmonize_input_units(cohort, frozen['internal_context']['ankle_units'])
    ds = coordinate_dataset(input_cohort, cfg['coordinate_mode'])
    output = root / 'external_evaluation'
    output.mkdir(parents=True, exist_ok=True)
    identity = {'frozen_plan_sha256': digest_file(root / 'frozen_external_plan.json'),
                'external_fingerprint': ds.fingerprint}
    lock = output / 'evaluation_identity.json'
    if lock.exists() and read_json(lock) != identity:
        raise ValueError('External evaluation was already bound to different data/models. Do not retune or replace the holdout.')
    if not lock.exists():
        write_json(lock, identity)
    if (output / 'complete.json').exists():
        if digest_file(output / 'predictions.npz') != read_json(output / 'complete.json')['prediction_sha256']:
            raise ValueError('Saved external predictions changed')
        print('External evaluation already completed. Returning the saved report without another prediction pass.', flush=True)
        return output
    audit_external(cohort, cfg['external_visits'], output / 'domain_shift_audit', frozen['internal_context'])
    cache = output / 'predictions.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as f:
            predictions = {name: f[name] for name in frozen['model_directories']}
            if frozen['ensemble_members']:
                predictions['loso_ensemble'] = f['loso_ensemble']
    else:
        predictions = predict_frozen(root, frozen, ds.x)
        temporary = output / 'predictions.tmp.npz'
        np.savez_compressed(temporary, **predictions, measured=ds.y,
                            cycle_ids=ds.cycles.cycle_id.to_numpy(str), participant_ids=ds.groups)
        temporary.replace(cache)
    tables = [prediction_table(ds, p, name) for name, p in predictions.items()]
    save_reports(pd.concat(tables, ignore_index=True), output, cfg, cohort.participants)
    write_json(output / 'interpretation.json', {'evaluation_type': 'domain-shift evaluation',
        'external_status': cfg['external_status'], 'primary_model': 'selected_deep',
        'prior_examination_note': 'Previously reported data cannot be relabeled untouched by rerunning a new protocol.'
           if cfg['external_status'] == 'previously_evaluated' else 'Untouched status is a user provenance declaration, not provable by code.',
        'participant_ci': 'Resamples whole trials within participants; unavailable with fewer than two known trials.',
        'cohort_ci': 'Resamples participants; visits remain separate and fitted-model uncertainty is not included.',
        'strata': 'Descriptive, prespecified, suppressed below minimum sample size; no causal explanation of domain shift.',
        'ensemble_rule': cfg['ensemble_rule'] if frozen['ensemble_members'] else 'No ensemble; one full-internal-cohort deep model',
        'external_adaptation': 'none; no external tuning, early stopping, or calibration'})
    write_json(output / 'complete.json', {**identity, 'prediction_sha256': digest_file(cache), 'n_unique_participants': len(ds.subjects)})
    return output

def compare_coordinates(experiment_root, modes, directory, cfg):
    """Paired comparisons on identical held-out participant/cycle membership."""
    rows, cohort_rows = [], []
    reference_cycles = None
    for mode in modes:
        root = Path(experiment_root) / mode
        if not read_json(root / 'nested_coverage.json')['complete']:
            raise ValueError(f'Incomplete outer coverage for {mode}')
        cycles = pd.read_csv(root / 'internal_reports' / 'cycle_metrics.csv', dtype={'participant_id': str, 'cycle_id': str})
        cycles = cycles[cycles.model == 'selected_deep']
        membership = set(zip(cycles.participant_id, cycles.cycle_id, cycles.visit))
        if reference_cycles is not None and membership != reference_cycles:
            raise ValueError('Coordinate comparisons require the same valid participant/cycle set')
        reference_cycles = membership
        table = pd.read_csv(root / 'internal_reports' / 'participant_metrics.csv', dtype={'participant_id': str})
        table = table[table.model == 'selected_deep'].copy()
        table['coordinate_mode'] = mode
        rows.append(table)
    all_rows = pd.concat(rows, ignore_index=True)
    if 'absolute' not in modes:
        raise ValueError('Include absolute coordinates as the reference')
    ref = all_rows[all_rows.coordinate_mode == 'absolute']
    for mode in modes:
        if mode == 'absolute':
            continue
        joined = all_rows[all_rows.coordinate_mode == mode].merge(ref, on=['task', 'participant_id', 'visit'], suffixes=('_mode', '_absolute'), validate='one_to_one')
        for metric in ['rmse', 'nrmse', 'mae']:
            delta = (joined[f'{metric}_absolute'] - joined[f'{metric}_mode']).dropna().to_numpy()
            lo, hi = bootstrap_interval(delta, stable_seed(cfg['bootstrap_seed'], mode, metric), cfg['bootstrap_resamples'])
            cohort_rows.append(dict(coordinate_mode=mode, metric=metric, n_participants=len(delta),
                median_improvement=float(np.median(delta)) if len(delta) else np.nan,
                median_improvement_ci95_low=lo, median_improvement_ci95_high=hi,
                interpretation='positive favors normalized coordinates; descriptive paired participant bootstrap'))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(directory / 'participant_coordinate_metrics.csv', index=False)
    pd.DataFrame(cohort_rows).to_csv(directory / 'paired_coordinate_comparisons.csv', index=False)
    return directory

def perturb_internal(cohort, cfg, root):
    """Prespecified input nuisance tests on held-out INTERNAL cycles; no refitting."""
    root = Path(root)
    ds = coordinate_dataset(cohort, cfg['coordinate_mode'])
    bind_experiment(root, ds, cfg)
    if not cfg['run_perturbations']:
        raise ValueError('Enable perturbations in the prespecification before running the study')
    if not read_json(root / 'nested_coverage.json')['complete']:
        raise ValueError('Finish internal outer folds first')
    scenarios = [('unperturbed', None, 0.)]
    scenarios += [(f'translation_{i}', t, 0.) for i, t in enumerate(cfg['translation_offsets_m'])]
    if 'ankle_xyz' in cohort.arrays:
        scenarios += [(f'yaw_{angle:g}', None, angle) for angle in cfg['rotation_angles_deg']]
    elif cfg['rotation_angles_deg']:
        raise ValueError('Rotation tests require verified 3D data; do not rotate a scalar trace')
    rows = []
    for name, translation, angle in scenarios:
        transformed = coordinate_dataset(cohort, cfg['coordinate_mode'], translation, angle)
        for outer in ds.subjects:
            te = transformed.subset([outer])
            winner = read_json(root / 'nested' / outer / 'winner.json')
            prediction = predict_saved(root / winner['refit'], te.x)
            table = prediction_table(te, prediction, 'selected_deep')
            table['scenario'] = name
            rows.append(table)
    cycles = pd.concat(rows, ignore_index=True)
    output = root / 'perturbations'
    output.mkdir(parents=True, exist_ok=True)
    cycles.to_csv(output / 'cycle_metrics.csv', index=False)
    subjects = cycles.groupby(['task', 'model', 'participant_id', 'scenario'])[['rmse', 'nrmse', 'mae']].mean().reset_index()
    reference = subjects[subjects.scenario == 'unperturbed']
    delta = subjects.merge(reference.drop(columns='scenario'), on=['task', 'model', 'participant_id'], suffixes=('', '_unperturbed'), validate='many_to_one')
    for metric in ['rmse', 'nrmse', 'mae']:
        delta[f'{metric}_change'] = delta[metric] - delta[f'{metric}_unperturbed']
    delta.to_csv(output / 'participant_error_changes.csv', index=False)
    write_json(output / 'protocol.json', {'scenarios': scenarios,
        'definition': 'Rigid nuisance transform of ankle and pelvis positions together; direction rotates with them; gait-frame targets fixed.',
        'interpretation': 'Sensitivity evidence, not proof of biomechanical causality. No augmentation or parameter tuning on these tests.'})
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['nested', 'final', 'external', 'perturb'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--internal-zip')
    parser.add_argument('--internal-bundle')
    parser.add_argument('--external-bundle')
    parser.add_argument('--outer-subjects', nargs='+')
    args = parser.parse_args()
    cfg = read_json(args.config)
    if args.stage == 'external':
        if not args.external_bundle:
            parser.error('--external-bundle is required')
        frozen = check_frozen(args.output)
        if frozen['config'] != cfg or frozen['task'] != args.task:
            raise ValueError('External settings differ from the frozen protocol')
        external_evaluation(args.output, args.external_bundle)
        return
    if bool(args.internal_zip) == bool(args.internal_bundle):
        parser.error('Provide exactly one of --internal-zip or --internal-bundle')
    cohort = legacy_cohort(args.internal_zip, args.task) if args.internal_zip else load_cohort(args.internal_bundle, args.task, role='internal')
    if cfg['smoke']:
        cohort = replace(cohort, ds=cohort.ds.subset(cohort.ds.subjects[:3])) if not cohort.arrays else cohort
        if cohort.arrays:
            raise ValueError('Smoke test uses the original scalar ZIP; use normal mode for enriched bundles')
    if set(cfg['families']) & set(DEEP):
        from models import configure_tensorflow
        configure_tensorflow(cfg['seed'])
    if args.stage == 'nested':
        run_nested(cohort, cfg, args.output, args.outer_subjects or (cohort.ds.subjects[:1] if cfg['smoke'] else None))
    elif args.stage == 'final':
        final_refit(cohort, cfg, args.output)
    else:
        perturb_internal(cohort, cfg, args.output)

if __name__ == '__main__':
    main()
