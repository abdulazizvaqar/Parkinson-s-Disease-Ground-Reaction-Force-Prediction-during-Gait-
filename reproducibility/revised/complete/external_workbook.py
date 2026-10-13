"""Read the supplied anonymous external workbook without inventing identities."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from protocol import write_json, read_json
from revision_data import load_cohort, bool_column, digest_file

SHEETS = {f'{component}_{side}': (f'{side[0]}-{force}', f'{side[0]}ANK')
          for side in ['Left', 'Right']
          for component, force in [('AP', 'AP'), ('Vertical', 'VERT'), ('ML', 'ML')]}

def audit_workbook(path):
    tables = pd.read_excel(path, sheet_name=None, header=None, engine='openpyxl')
    rows = []
    for name, frame in tables.items():
        if frame.empty:
            rows.append(dict(sheet=name, samples=0, columns=0, missing_values=0, status='empty'))
            continue
        try:
            x = frame.to_numpy(dtype=float)
        except ValueError:
            raise ValueError(f'{name}: expected an all-numeric sheet with no header row')
        rows.append(dict(sheet=name, samples=x.shape[0], columns=x.shape[1],
                         missing_values=int((~np.isfinite(x)).sum()), minimum=float(np.nanmin(x)),
                         maximum=float(np.nanmax(x)), status='numeric waveform sheet'))
    return tables, pd.DataFrame(rows)

def make_mapping_templates(workbook, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tables, audit = audit_workbook(workbook)
    audit.to_csv(directory / 'external_workbook_audit.csv', index=False)
    mapping = []
    for side in ['Left', 'Right']:
        frame = tables[f'{side[0]}-VERT']
        for column in range(1, frame.shape[1] + 1):
            mapping.append(dict(side=side.lower(), column_1based=column,
                cycle_id=f'EXT_{side.upper()}_COL{column:03d}', participant_id='', trial_id='',
                visit='', include='', exclusion_reason='', body_mass_kg='',
                age='', sex='', medication_state='', HY_stage='', intervention_group='', global_subject_id=''))
    # Never overwrite metadata the user has filled in.
    if not (directory / 'external_column_map.csv').exists():
        pd.DataFrame(mapping).to_csv(directory / 'external_column_map.csv', index=False)
    spec = dict(workbook_sha256=digest_file(workbook), ankle_units='UNCONFIRMED', force_units='UNCONFIRMED',
        ankle_axis={'AP': 'UNCONFIRMED', 'Vertical': 'UNCONFIRMED', 'ML': 'UNCONFIRMED'},
        ankle_sign={'AP': None, 'Vertical': None, 'ML': None},
        force_sign={'AP': None, 'Vertical': None, 'ML': None},
        phase_grid_0_to_100_confirmed=False, column_pairing_verified=False,
        internal_external_axes_and_signs_verified=False, participant_disjointness_verified=False,
        visit_mapping_verified=False, identity_mapping_verified=False,
        coordinate_definition='UNCONFIRMED', motion_capture_system='unknown', force_platforms='unknown',
        kinematics_sampling_hz='unknown', force_sampling_hz='unknown', walking_protocol='unknown',
        cycle_definition='UNCONFIRMED', harmonization_description='UNCONFIRMED',
        resampling='linear 101-to-100 normalized-phase interpolation; preserve both endpoints',
        source_population_scope='Provided workbook columns only; source recruitment/exclusion history unavailable')
    if not (directory / 'external_import_settings.json').exists():
        write_json(directory / 'external_import_settings.json', spec)
    return audit

def readiness(workbook, directory):
    directory = Path(directory)
    reasons = []
    needed = ['external_column_map.csv', 'external_import_settings.json']
    if any(not (directory / name).exists() for name in needed):
        return ['External column map or import settings not supplied']
    spec = read_json(directory / needed[1])
    if spec.get('workbook_sha256') != digest_file(workbook):
        reasons.append('External mapping/settings refer to a different workbook')
    if spec.get('ankle_units') not in ('mm', 'm'):
        reasons.append('Confirm ankle units: mm or m')
    if spec.get('force_units') not in ('N', 'N/kg'):
        reasons.append('Confirm force units: N or N/kg')
    for flag in ['phase_grid_0_to_100_confirmed', 'column_pairing_verified',
                 'internal_external_axes_and_signs_verified', 'participant_disjointness_verified',
                 'visit_mapping_verified', 'identity_mapping_verified']:
        if spec.get(flag) is not True:
            reasons.append('Source verification needed: ' + flag)
    for component in ['AP', 'Vertical', 'ML']:
        if spec.get('ankle_axis', {}).get(component) not in ('X', 'Y', 'Z'):
            reasons.append('Confirm ankle axis for ' + component)
        for item in ['ankle_sign', 'force_sign']:
            if spec.get(item, {}).get(component) not in (-1, 1):
                reasons.append(f'Confirm {item} for {component}: +1 or -1')
    for field in ['coordinate_definition', 'cycle_definition', 'harmonization_description']:
        if spec.get(field) in (None, '', 'UNCONFIRMED'):
            reasons.append('Complete ' + field)
    mapping = pd.read_csv(directory / needed[0], dtype=str, keep_default_na=False)
    fields = ['participant_id', 'trial_id', 'visit', 'include']
    for field in fields:
        if field not in mapping or mapping[field].str.strip().eq('').any():
            reasons.append('Complete external_column_map.csv: ' + field)
    if not reasons:
        included = bool_column(mapping.include)
        if not included.any():
            reasons.append('No included external waveform columns')
        if spec['force_units'] == 'N':
            mass = pd.to_numeric(mapping.loc[included, 'body_mass_kg'], errors='coerce')
            if mass.isna().any() or (mass <= 0).any():
                reasons.append('Positive source body_mass_kg is required to convert N to N/kg')
    return reasons

def prepare_external_bundles(workbook, metadata_dir, output_root):
    """Deterministic import; called after internal model freeze, never tunes using labels."""
    reasons = readiness(workbook, metadata_dir)
    if reasons:
        raise ValueError('; '.join(reasons))
    metadata_dir, output_root = Path(metadata_dir), Path(output_root)
    settings = read_json(metadata_dir / 'external_import_settings.json')
    mapping = pd.read_csv(metadata_dir / 'external_column_map.csv', dtype=str, keep_default_na=False)
    mapping['column_1based'] = pd.to_numeric(mapping.column_1based, errors='raise').astype(int)
    mapping['include'] = bool_column(mapping.include)
    if mapping.duplicated(['side', 'column_1based']).any() or mapping.cycle_id.duplicated().any():
        raise ValueError('Duplicate source column or cycle ID')
    tables, audit = audit_workbook(workbook)
    person_cols = ['age', 'sex', 'medication_state', 'HY_stage', 'intervention_group', 'global_subject_id']
    for _, group in mapping.groupby('participant_id'):
        for col in [*person_cols, 'body_mass_kg']:
            values = set(group[col]) - {''}
            if len(values) > 1:
                raise ValueError(f'{col} differs within a participant; use a visit-specific metadata design before import')
    people = mapping.groupby('participant_id', sort=True)[person_cols].agg(lambda s: next((x for x in s if x != ''), '')).reset_index()
    outputs = {}
    for task, (force_sheet, ankle_prefix) in SHEETS.items():
        component, side = task.split('_')
        rows = mapping[mapping.side == side.lower()].sort_values('column_1based').copy()
        n_columns = tables[force_sheet].shape[1]
        if set(rows.column_1based) != set(range(1, n_columns + 1)):
            raise ValueError(f'{side}: map every source column exactly once, including exclusions')
        force = tables[force_sheet].to_numpy(float)
        ankle = tables[ankle_prefix + settings['ankle_axis'][component]].to_numpy(float)
        if force.shape != (101, n_columns) or ankle.shape != force.shape:
            raise ValueError('Expected matched 101-sample column arrays; first row is data, not a header')
        kept = rows[rows.include]
        if kept.empty:
            continue
        idx = kept.column_1based.to_numpy() - 1
        x = ankle[:, idx].T * settings['ankle_sign'][component]
        y = force[:, idx].T * settings['force_sign'][component]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Missing/nonfinite included waveforms; document exclusions or repair source data')
        if settings['force_units'] == 'N':
            mass = pd.to_numeric(kept.body_mass_kg, errors='raise').to_numpy(float)
            y /= mass[:, None]
        phase_from, phase_to = np.linspace(0, 100, 101), np.linspace(0, 100, 100)
        x = np.stack([np.interp(phase_to, phase_from, v) for v in x])[:, :, None].astype('float32')
        y = np.stack([np.interp(phase_to, phase_from, v) for v in y])[:, :, None].astype('float32')
        folder = output_root / task
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'waveforms.npz', x=x, y=y, cycle_ids=kept.cycle_id.to_numpy(str))
        rows.to_csv(folder / 'cycles.csv', index=False)
        # Trial status derived only for trials represented by workbook columns.
        trial_keys = ['participant_id', 'trial_id', 'visit', 'side']
        trials = rows.groupby(trial_keys).agg(include=('include', 'any'),
            exclusion_reason=('exclusion_reason', lambda s: '; '.join(sorted(set(s) - {''})))).reset_index()
        trials.loc[trials.include, 'exclusion_reason'] = ''
        trials.to_csv(folder / 'trials.csv', index=False)
        people.to_csv(folder / 'participants.csv', index=False)
        context = {key: settings[key] for key in ['ankle_units', 'coordinate_definition', 'motion_capture_system',
            'force_platforms', 'kinematics_sampling_hz', 'force_sampling_hz', 'walking_protocol',
            'cycle_definition', 'harmonization_description', 'source_population_scope']}
        context.update(cohort_id='Viglialoro2025_external_workbook', dataset_role='external', task=task,
            grf_units='N/kg', units_verified=True, harmonization_verified=True,
            participant_disjointness_verified=True, source_workbook_sha256=digest_file(workbook),
            import_settings_sha256=digest_file(metadata_dir / 'external_import_settings.json'),
            column_map_sha256=digest_file(metadata_dir / 'external_column_map.csv'),
            force_source_units=settings['force_units'], time_normalization=settings['resampling'],
            exclusions_scope='Only columns and represented trials in this workbook; source exclusions unavailable')
        write_json(folder / 'cohort.json', context)
        load_cohort(folder, task, role='external')
        outputs[task] = str(folder)
    return outputs
