"""Strict cohort contract and coordinate controls; no guessed anatomical data."""
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import pandas as pd
from data import Dataset, load_datasets
from protocol import write_json, read_json

MODES = ('absolute', 'translation', 'pelvis', 'aligned', 'body', 'combined')
AXIS = {'AP': 0, 'ML': 1, 'Vertical': 2}
REQUIRED_CONTEXT = ['cohort_id', 'dataset_role', 'ankle_units', 'grf_units',
                    'coordinate_definition', 'motion_capture_system', 'force_platforms',
                    'kinematics_sampling_hz', 'force_sampling_hz', 'walking_protocol',
                    'harmonization_description', 'cycle_definition']

def digest_file(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def bool_column(series):
    mapping = {'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False}
    v = series.astype(str).str.lower().str.strip().map(mapping)
    if v.isna().any():
        raise ValueError('include must contain explicit true/false values')
    return v.astype(bool)

def read_table(path, columns):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if set(columns) - set(df):
        raise ValueError(f'{path}: missing columns {set(columns) - set(df)}')
    for col in set(columns) - {'exclusion_reason'}:
        if df[col].str.strip().eq('').any():
            raise ValueError(f'{path}: blank required {col}')
    return df

@dataclass
class Cohort:
    ds: Dataset
    context: dict
    participants: pd.DataFrame
    trials: pd.DataFrame
    ledger: pd.DataFrame
    arrays: dict

def legacy_cohort(archive, task):
    ds = load_datasets(archive, [task])[task]
    meta = ds.cycles.copy()
    meta['participant_id'] = meta.subject
    meta['trial_id'] = 'unknown'  # Never invent trial IDs from cycle positions.
    meta['visit'] = 'internal'
    meta['side'] = task.split('_')[-1].lower()
    meta['include'] = True
    meta['exclusion_reason'] = ''
    ds = replace(ds, cycles=meta)
    context = {k: 'unknown' for k in REQUIRED_CONTEXT}
    context.update(cohort_id='Shida_internal', dataset_role='internal', ankle_units='mm',
                   grf_units='N/kg', units_verified=False,
                   coordinate_definition='Original single component; physical frame unverified',
                   source='Paired Raw sheets; manuscript units assumed, not independently verified')
    people = pd.DataFrame({'participant_id': ds.subjects})
    return Cohort(ds, context, people, pd.DataFrame(), meta, {})

def load_cohort(directory, task, visits=None, role=None):
    """Read one side/component cohort bundle. Excluded cycles need no waveform."""
    root = Path(directory)
    context = read_json(root / 'cohort.json')
    if any(k not in context or context[k] is None for k in REQUIRED_CONTEXT):
        raise ValueError('Complete cohort.json; use "unknown" explicitly for unavailable descriptive fields')
    if role and context['dataset_role'] != role:
        raise ValueError('Wrong dataset_role')
    if context['ankle_units'] not in ('m', 'mm') or context['grf_units'] != 'N/kg':
        raise ValueError('Convert ankle units to m/mm and GRF to N/kg before making the bundle')
    if context.get('task') != task:
        raise ValueError('cohort.json task differs from requested component/side')
    people = read_table(root / 'participants.csv', ['participant_id'])
    trials = read_table(root / 'trials.csv', ['participant_id', 'trial_id', 'visit', 'side', 'include', 'exclusion_reason'])
    ledger = read_table(root / 'cycles.csv', ['cycle_id', 'participant_id', 'trial_id', 'visit', 'side', 'include', 'exclusion_reason'])
    if people.participant_id.duplicated().any() or ledger.cycle_id.duplicated().any():
        raise ValueError('Participant and cycle IDs must be unique within this cohort bundle')
    if not people.participant_id.str.fullmatch(r'[A-Za-z0-9_-]+').all():
        raise ValueError('Use letters, digits, underscores, or hyphens in participant IDs')
    keys = ['participant_id', 'trial_id', 'visit', 'side']
    if trials.duplicated(keys).any():
        raise ValueError('Duplicate trial record')
    for frame in (trials, ledger):
        frame['include'] = bool_column(frame.include)
        if frame.loc[~frame.include, 'exclusion_reason'].str.strip().eq('').any():
            raise ValueError('Every excluded trial/cycle requires a reason')
        if set(frame.participant_id) - set(people.participant_id):
            raise ValueError('Participant missing from participants.csv')
        if not set(frame.visit).issubset({'baseline', 'post_intervention', 'internal'}):
            raise ValueError('Use baseline, post_intervention, or internal visit labels')
        if set(frame.side) != {task.split('_')[-1].lower()}:
            raise ValueError('Each bundle must describe just its selected side')
    joined = ledger.merge(trials[keys + ['include']], on=keys, how='left', suffixes=('', '_trial'), validate='many_to_one')
    if joined.include_trial.isna().any() or (joined.include & ~joined.include_trial.astype(bool)).any():
        raise ValueError('Every cycle must link to a trial; an excluded trial cannot have an included cycle')
    with np.load(root / 'waveforms.npz', allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    ids = arrays.pop('cycle_ids').astype(str)
    valid = ledger[ledger.include].copy().set_index('cycle_id', drop=False)
    if len(set(ids)) != len(ids) or set(ids) != set(valid.index):
        raise ValueError('NPZ cycle_ids must match ALL included cycles, one-to-one, before visit filtering')
    meta = valid.loc[ids].reset_index(drop=True)
    n = len(ids)
    for k, arr in arrays.items():
        if len(arr) != n or not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
            raise ValueError(f'{k}: wrong cycle count or nonfinite/nonnumeric values')
    if 'ankle_xyz' in arrays:
        xyz = arrays['ankle_xyz']
        if xyz.shape != (n, 100, 3) or arrays.get('grf_xyz', np.empty(0)).shape != xyz.shape:
            raise ValueError('ankle_xyz and grf_xyz must both be [N,100,3]')
        if context.get('canonical_axis_order') != ['forward', 'left', 'up']:
            raise ValueError('Map both kinematics and GRFs into the same right-handed [forward,left,up] laboratory axes')
        if arrays.get('direction_xyz', np.empty(0)).shape != (n, 3):
            raise ValueError('Provide a force-independent horizontal walking direction for every cycle')
        basis = gait_basis(arrays['direction_xyz'])
        axis = AXIS[task.split('_')[0]]
        x = xyz[:, :, axis:axis + 1]
        # SAME gait-frame targets for ALL coordinate variants, enabling fair comparisons.
        y = np.einsum('nti,nij->ntj', arrays['grf_xyz'], basis)[:, :, axis:axis + 1]
    else:
        x, y = arrays.get('x'), arrays.get('y')
        if x is None or y is None or x.shape != (n, 100, 1) or y.shape != x.shape:
            raise ValueError('Scalar bundles require x and y, each [N,100,1]')
    take = np.ones(n, bool) if visits is None else meta.visit.isin(visits).to_numpy()
    if not take.any():
        raise ValueError('No valid cycles for the prespecified visit policy')
    meta = meta.loc[take].reset_index(drop=True)
    meta['subject'] = meta.participant_id
    fingerprint = sha256(''.join(digest_file(root / f) for f in
                         ['cohort.json', 'participants.csv', 'trials.csv', 'cycles.csv', 'waveforms.npz']).encode()).hexdigest()
    fingerprint = sha256((fingerprint + repr(visits)).encode()).hexdigest()
    ds = Dataset(task, x[take].astype('float32'), y[take].astype('float32'),
                 meta.participant_id.to_numpy(str), meta, fingerprint)
    return Cohort(ds, context, people, trials, ledger, {k: v[take] for k, v in arrays.items()})

def gait_basis(direction):
    forward = np.array(direction, float, copy=True)
    if forward.ndim != 2 or forward.shape[1] != 3 or not np.isfinite(forward).all():
        raise ValueError('Invalid direction vectors')
    if np.any(np.abs(forward[:, 2]) > 1e-6):
        raise ValueError('Walking directions must be projected onto the horizontal plane')
    length = np.linalg.norm(forward, axis=1)
    if np.any(length < 1e-8):
        raise ValueError('Zero walking direction; supply verified direction from pelvis progression/protocol')
    forward /= length[:, None]
    up = np.tile([0., 0., 1.], (len(forward), 1))
    left = np.cross(up, forward)
    return np.stack([forward, left, up], axis=2)

def coordinate_dataset(cohort, mode='absolute', translation_m=None, rotation_deg=0):
    if mode not in MODES:
        raise ValueError(f'Unknown coordinate mode {mode}')
    ds, a = cohort.ds, cohort.arrays
    units = cohort.context['ankle_units']
    factor = 1000. if units == 'mm' else 1.
    axis = AXIS[ds.task.split('_')[0]]
    if 'ankle_xyz' not in a:
        if mode not in ('absolute', 'translation') or rotation_deg:
            raise ValueError('This analysis requires verified 3D/pelvis/body-scale data; scalar inputs cannot supply it')
        x = ds.x.astype(float).copy()
        if translation_m is not None:
            if not cohort.context.get('units_verified', False):
                raise ValueError('Verify physical units before metre-based perturbations')
            x += np.asarray(translation_m)[axis] * factor
        if mode == 'translation':
            x -= x[:, :1, :]
    else:
        x = a['ankle_xyz'].astype(float).copy()
        pelvis = a.get('pelvis_xyz')
        pelvis = None if pelvis is None else pelvis.astype(float).copy()
        direction = a['direction_xyz'].astype(float).copy()
        if rotation_deg:
            theta = np.deg2rad(rotation_deg)
            R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            x = x @ R.T
            direction = direction @ R.T
            if pelvis is not None:
                pelvis = pelvis @ R.T
        if translation_m is not None:
            if not cohort.context.get('units_verified', False):
                raise ValueError('Verify physical units before metre-based perturbations')
            offset = np.asarray(translation_m, float) * factor
            if offset.shape != (3,):
                raise ValueError('Translation requires three metre offsets')
            x += offset
            if pelvis is not None:
                pelvis += offset
        if mode in ('pelvis', 'combined'):
            if pelvis is None or pelvis.shape != x.shape:
                raise ValueError('Synchronized pelvis_xyz [N,100,3] is required')
            x -= pelvis
        elif mode in ('translation', 'aligned', 'body'):
            x -= x[:, :1, :]
        if mode in ('aligned', 'combined'):
            x = np.einsum('nti,nij->ntj', x, gait_basis(direction))
        if mode in ('body', 'combined'):
            if not cohort.context.get('units_verified', False) or not cohort.context.get('body_scale_definition'):
                raise ValueError('Verify coordinate units and prespecify height or leg-length definition')
            height = a.get('body_scale_m')
            if height is None or height.shape != (len(x),) or np.any(height <= 0):
                raise ValueError('Positive body_scale_m [N], with a fixed definition, is required')
            x /= height[:, None, None] * factor
        x = x[:, :, axis:axis + 1]
    fingerprint = sha256((ds.fingerprint + repr((mode, translation_m, rotation_deg))).encode()).hexdigest()
    return replace(ds, x=x.astype('float32'), fingerprint=fingerprint)

def write_templates(directory):
    """Blank data contract, not fabricated external observations."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for name, columns in {
        'participants.csv': ['participant_id', 'global_subject_id', 'age', 'sex', 'medication_state', 'HY_stage', 'intervention_group'],
        'trials.csv': ['participant_id', 'trial_id', 'visit', 'side', 'include', 'exclusion_reason'],
        'cycles.csv': ['cycle_id', 'participant_id', 'trial_id', 'visit', 'side', 'include', 'exclusion_reason'],
    }.items():
        if not (root / name).exists():
            pd.DataFrame(columns=columns).to_csv(root / name, index=False)
    context = {k: None for k in REQUIRED_CONTEXT}
    context.update(task=None, units_verified=False, harmonization_verified=False, participant_disjointness_verified=False,
                   canonical_axis_order=['forward', 'left', 'up'], body_scale_definition=None)
    if not (root / 'cohort.json').exists():
        write_json(root / 'cohort.json', context)
    return root

def harmonize_input_units(cohort, training_units):
    """Deterministic physical-unit conversion, never fitted on external data."""
    units = cohort.context['ankle_units']
    if units not in ('m', 'mm') or training_units not in ('m', 'mm'):
        raise ValueError('Known internal and external length units are required')
    if units == training_units:
        return cohort
    multiplier = 1000. if units == 'm' else .001
    arrays = dict(cohort.arrays)
    for key in ['ankle_xyz', 'pelvis_xyz']:
        if key in arrays:
            arrays[key] = arrays[key] * multiplier
    ds = replace(cohort.ds, x=cohort.ds.x * multiplier)
    return replace(cohort, ds=ds, arrays=arrays,
                   context={**cohort.context, 'ankle_units': training_units})

def pack_scalar_workbooks(input_xlsx, target_xlsx, cycle_map_csv, participants_csv,
                          trials_csv, cohort_json, output):
    """Convert explicitly mapped old-format Excel cycles; never infer participants."""
    import shutil
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    ledger = read_table(cycle_map_csv, ['cycle_id', 'participant_id', 'trial_id', 'visit', 'side',
                                       'include', 'exclusion_reason'])
    if {'input_column', 'target_column'} - set(ledger):
        raise ValueError('Explicit input_column and target_column mappings are required')
    accepted = ledger[bool_column(ledger.include)]
    inputs = pd.read_excel(input_xlsx, engine='openpyxl')
    targets = pd.read_excel(target_xlsx, engine='openpyxl')
    if inputs.shape[0] != 100 or targets.shape[0] != 100:
        raise ValueError('Prepare matched complete cycles at 100 normalized gait phases before packing; no padding/resampling is guessed')
    xs, ys = [], []
    for row in accepted.itertuples():
        x = inputs[row.input_column].to_numpy(dtype='float32')
        y = targets[row.target_column].to_numpy(dtype='float32')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f'{row.cycle_id}: missing values; document exclusion or fix the source, never zero-fill')
        xs.append(x[:, None]); ys.append(y[:, None])
    if not xs:
        raise ValueError('No included cycles')
    np.savez_compressed(root / 'waveforms.npz', cycle_ids=accepted.cycle_id.to_numpy(str), x=np.stack(xs), y=np.stack(ys))
    ledger.to_csv(root / 'cycles.csv', index=False)
    for source, filename in [(participants_csv, 'participants.csv'), (trials_csv, 'trials.csv'), (cohort_json, 'cohort.json')]:
        if Path(source).resolve() != (root / filename).resolve():
            shutil.copy2(source, root / filename)
    context = read_json(root / 'cohort.json')
    load_cohort(root, context['task'], role=context['dataset_role'])
    return root
