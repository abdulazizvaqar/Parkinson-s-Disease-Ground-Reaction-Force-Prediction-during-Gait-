"""All-task study controller. Completed stages resume; missing data stay explicit."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import traceback
import numpy as np
import pandas as pd
from data import TASK_ORDER
from protocol import write_json, read_json
from revision_data import legacy_cohort, load_cohort, coordinate_dataset, digest_file, MODES
from revision_models import BASELINES, DEEP, baseline_grid
from revision_study import (default_config, validate_config, run_nested, final_refit,
    perturb_internal, compare_coordinates, external_evaluation)
from external_workbook import readiness, prepare_external_bundles

def load_internal(plan, task):
    if plan['internal_source'] == 'original_zip':
        return legacy_cohort(plan['internal_zip'], task)
    if plan['internal_source'] == 'enriched_bundles':
        return load_cohort(Path(plan['internal_bundle_root']) / task, task, role='internal')
    raise ValueError('Unknown internal_source')

def configurations(plan, mode):
    cfg = {**default_config(), **plan['training'], 'smoke': False, 'coordinate_mode': mode}
    cfg['families'] = list(BASELINES + DEEP)
    validate_config(cfg)
    return cfg

def inspect_plan(plan):
    if not plan['tasks'] or len(set(plan['tasks'])) != len(plan['tasks']) or set(plan['tasks']) - set(TASK_ORDER):
        raise ValueError('Unknown/duplicate task')
    if not plan['coordinate_modes'] or set(plan['coordinate_modes']) - set(MODES):
        raise ValueError('Unknown coordinate mode')
    if plan['primary_coordinates'] not in plan['coordinate_modes']:
        raise ValueError('Primary coordinate mode must be prespecified in the requested modes')
    rows = []
    for task in plan['tasks']:
        try:
            cohort = load_internal(plan, task)
        except (ValueError, FileNotFoundError) as exc:
            rows.append(dict(task=task, coordinate='all', stage='input', status='blocked', reason=str(exc), fit_upper_bound=0))
            continue
        for mode in plan['coordinate_modes']:
            cfg = configurations(plan, mode)
            try:
                ds = coordinate_dataset(cohort, mode)
                n = len(ds.subjects)
                candidates = 3 * cfg['max_trials'] + sum(len(baseline_grid(f)) for f in BASELINES)
                fits = n * (candidates * (n - 1) + 7)
                rows.append(dict(task=task, coordinate=mode, stage='nested', status='ready', reason='',
                    n_participants=n, n_cycles=len(ds.x), fit_upper_bound=fits))
                if mode == plan['primary_coordinates']:
                    rows.append(dict(task=task, coordinate=mode, stage='final', status='after nested completion', reason='',
                        n_participants=n, n_cycles=len(ds.x), fit_upper_bound=n * candidates + 5))
                if cfg['run_perturbations']:
                    try:
                        for t in cfg['translation_offsets_m']:
                            coordinate_dataset(cohort, mode, translation_m=t)
                        for angle in cfg['rotation_angles_deg']:
                            coordinate_dataset(cohort, mode, rotation_deg=angle)
                        state, reason = 'after nested completion', ''
                    except ValueError as exc:
                        state, reason = 'blocked', str(exc)
                    rows.append(dict(task=task, coordinate=mode, stage='perturb', status=state, reason=reason, fit_upper_bound=0))
            except ValueError as exc:
                rows.append(dict(task=task, coordinate=mode, stage='nested', status='blocked', reason=str(exc), fit_upper_bound=0))
    return pd.DataFrame(rows)

def lock_plan(plan, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Runtime location of uploaded files can change; content is checked by the study
    # and external loaders. Scientific settings and adapter source must not change.
    signature = {k: plan[k] for k in ['tasks', 'coordinate_modes', 'primary_coordinates', 'internal_source', 'training']}
    signature['controller_sha256'] = digest_file(Path(__file__))
    signature['external_adapter_sha256'] = digest_file(Path(__file__).with_name('external_workbook.py'))
    path = root / 'complete_run_plan.json'
    if path.exists() and read_json(path) != signature:
        raise ValueError('The complete-run scientific plan changed. Restore it or use a new experiment name.')
    if not path.exists():
        write_json(path, signature)

def run_batch(plan):
    root = Path(plan['output_root'])
    lock_plan(plan, root)
    preflight = inspect_plan(plan)
    preflight.to_csv(root / 'preflight.csv', index=False)
    print(preflight[['task', 'coordinate', 'stage', 'status', 'fit_upper_bound']].to_string(index=False), flush=True)
    print(f'Total planned model-fit upper bound: {int(preflight.fit_upper_bound.sum()):,}', flush=True)
    states = read_json(root / 'progress.json') if (root / 'progress.json').exists() else {}

    def record(task, mode, stage, status, reason=''):
        key = f'{task}/{mode}/{stage}'
        states[key] = dict(task=task, coordinate=mode, stage=stage, status=status, reason=reason,
                           updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(root / 'progress.json', states)
        pd.DataFrame(states.values()).to_csv(root / 'progress.csv', index=False)
        print(f'[{status.upper()}] {key}' + (f': {reason}' if reason else ''), flush=True)

    def execute(task, mode, stage, function):
        record(task, mode, stage, 'running')
        try:
            function()
        except Exception as exc:
            record(task, mode, stage, 'failed', str(exc))
            raise
        record(task, mode, stage, 'completed')

    for row in preflight.itertuples():
        if row.status == 'blocked':
            record(row.task, row.coordinate, row.stage, 'blocked', row.reason)
    if not (preflight.status == 'ready').any():
        print('No internal analyses are ready. Resolve the listed input issues.', flush=True)
        return
    from models import configure_tensorflow
    hardware = configure_tensorflow(plan['training']['seed'])
    write_json(root / 'runtime_hardware.json', hardware)
    for task in plan['tasks']:
        subset = preflight[preflight.task == task]
        ready_modes = subset.loc[(subset.stage == 'nested') & (subset.status == 'ready'), 'coordinate'].tolist()
        if not ready_modes:
            continue
        cohort = load_internal(plan, task)
        for mode in ready_modes:
            cfg = configurations(plan, mode)
            experiment = root / task / mode
            execute(task, mode, 'nested', lambda: run_nested(cohort, cfg, experiment))
            if ((subset.coordinate == mode) & (subset.stage == 'perturb') & (subset.status != 'blocked')).any():
                execute(task, mode, 'perturb', lambda: perturb_internal(cohort, cfg, experiment))
        if 'absolute' in ready_modes and len(ready_modes) > 1:
            execute(task, 'comparison', 'coordinates', lambda: compare_coordinates(root / task, ready_modes,
                root / task / 'coordinate_comparison', configurations(plan, 'absolute')))
        primary = plan['primary_coordinates']
        if primary not in ready_modes:
            record(task, primary, 'final', 'blocked', 'Prespecified primary coordinates unavailable; no automatic substitution')
            record(task, primary, 'external', 'blocked', 'Final primary model unavailable')
            continue
        cfg, experiment = configurations(plan, primary), root / task / primary
        execute(task, primary, 'final', lambda: final_refit(cohort, cfg, experiment))
        # Only NOW, after the final model freeze, import external targets.
        external_folder = Path(plan['external_bundle_root']) / task
        files = ['waveforms.npz', 'participants.csv', 'trials.csv', 'cycles.csv', 'cohort.json']
        if not all((external_folder / p).is_file() for p in files):
            if not plan.get('external_workbook') or not Path(plan['external_workbook']).is_file():
                record(task, primary, 'external', 'blocked', 'External workbook or prepared cohort bundle missing')
                continue
            reasons = readiness(plan['external_workbook'], plan['external_metadata_dir'])
            if reasons:
                record(task, primary, 'external', 'blocked', '; '.join(reasons))
                continue
            prepare_external_bundles(plan['external_workbook'], plan['external_metadata_dir'], plan['external_bundle_root'])
        context = read_json(external_folder / 'cohort.json')
        missing_flags = [name for name in ['units_verified', 'harmonization_verified', 'participant_disjointness_verified']
                         if context.get(name) is not True]
        if missing_flags:
            record(task, primary, 'external', 'blocked', 'Complete source verification: ' + ', '.join(missing_flags))
            continue
        execute(task, primary, 'external', lambda: external_evaluation(experiment, external_folder))
    frames = []
    for task in plan['tasks']:
        for mode in plan['coordinate_modes']:
            p = root / task / mode / 'internal_reports' / 'cohort_summary.csv'
            if p.exists():
                frame = pd.read_csv(p)
                frame['coordinate_mode'] = mode
                frames.append(frame)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(root / 'all_internal_cohort_summaries.csv', index=False)
    counts = pd.Series([r['status'] for r in states.values()]).value_counts().to_dict()
    write_json(root / 'completion_status.json', {'stage_counts': counts,
        'all_requested_stages_complete': all(r['status'] == 'completed' for r in states.values()),
        'note': 'Completed means the stage ran; blocked stages require the stated data. Missing source inputs are never imputed.'})
    print('Available work finished. Read progress.csv for completed and blocked stages.', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    plan = read_json(args.plan)
    if args.preflight_only:
        table = inspect_plan(plan)
        table.to_csv(Path(args.plan).with_name('preflight_preview.csv'), index=False)
        print(table.to_string(index=False))
        print('FIT UPPER BOUND:', int(table.fit_upper_bound.sum()))
    else:
        run_batch(plan)
