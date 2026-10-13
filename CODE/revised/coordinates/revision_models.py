"""Subject-wise model-family selection and four explicit comparator families."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from protocol import cycle_metrics, stable_seed, scale_from_training, write_json, read_json, parameter_hash

BASELINES = ('mean_waveform', 'ridge', 'phase_only', 'shallow_mlp')
DEEP = ('lstm', 'gru', 'conv1d_lstm')

def participant_mean(y, groups):
    return np.mean([y[groups == g].mean(axis=0) for g in np.unique(groups)], axis=0)

def fourier_basis(timesteps, harmonics):
    # Endpoints include 0 and 100% consistently with the supplied waveforms.
    phase = np.linspace(0, 1, timesteps)
    return np.column_stack([np.ones(timesteps)] +
        [f(2 * np.pi * k * phase) for k in range(1, harmonics + 1) for f in (np.sin, np.cos)])

def baseline_grid(family, smoke=False):
    if family == 'mean_waveform':
        return [{}]
    if family == 'ridge':
        return [{'alpha': v} for v in ([1.] if smoke else [0.01, 0.1, 1., 10., 100.])]
    if family == 'phase_only':
        return [{'harmonics': v} for v in ([3] if smoke else [1, 3, 5, 10])]
    if family == 'shallow_mlp':
        return [{'units': u, 'alpha': a, 'max_iter': 30 if smoke else 1000}
                for u in ([4] if smoke else [16, 32, 64])
                for a in ([0.01] if smoke else [0.0001, 0.01])]
    raise ValueError(family)

def fit_baseline(family, x, y, groups, parameters, seed):
    obj = {'family': family, 'shape': list(y.shape[1:]), 'parameters': parameters}
    if family == 'mean_waveform':
        obj['waveform'] = participant_mean(y, groups)
    elif family == 'phase_only':
        basis = fourier_basis(y.shape[1], parameters['harmonics'])
        # Equal participant weighting. Uses phase alone, never ankle positions.
        coef = np.linalg.lstsq(basis, participant_mean(y, groups), rcond=None)[0]
        obj['waveform'] = basis @ coef
    else:
        flat = x.reshape(len(x), -1)
        scaler = StandardScaler().fit(flat)  # Training rows only.
        z = scaler.transform(flat)
        target = y.reshape(len(y), -1)
        if family == 'ridge':
            model = Ridge(alpha=parameters['alpha'])
            weights = np.array([1. / np.sum(groups == g) for g in groups])
            weights *= len(weights) / weights.sum()
            model.fit(z, target, sample_weight=weights)
            obj['fit_weighting'] = 'equal participants'
        elif family == 'shallow_mlp':
            # One hidden layer; training-loss optimizer convergence, NO random validation split.
            model = MLPRegressor(hidden_layer_sizes=(parameters['units'],), activation='relu',
                                 solver='lbfgs', alpha=parameters['alpha'], max_iter=parameters['max_iter'],
                                 max_fun=50000, early_stopping=False, random_state=seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(z, target)
            obj['convergence_warnings'] = [str(w.message) for w in caught]
            obj['fit_weighting'] = 'equal cycles; validation score remains equal participants'
        else:
            raise ValueError(family)
        obj.update(model=model, scaler=scaler)
    return obj

def predict_baseline(obj, x):
    if 'waveform' in obj:
        return np.repeat(obj['waveform'][None], len(x), axis=0)
    z = obj['scaler'].transform(x.reshape(len(x), -1))
    return obj['model'].predict(z).reshape((len(x), *obj['shape']))

def backend(family, ds, cfg):
    from models import TensorFlowBackend
    return TensorFlowBackend(family, ds.x.shape[1], cfg['search_profile'], cfg['epochs'],
                             cfg['patience'], cfg['batch_size'], cfg['verbose'])

def cv_candidate(ds, family, parameters, cfg, directory):
    """Only the development Dataset is passed; outer/external arrays are absent."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    signature = parameter_hash(parameters)
    rows = []
    for val in ds.subjects:
        path = directory / f'validation_{val}.json'
        if path.exists():
            row = read_json(path)
            if row['parameter_hash'] != signature or row['data_fingerprint'] != ds.fingerprint:
                raise ValueError('Candidate cache changed')
            rows.append(row)
            continue
        tr, va = ds.groups != val, ds.groups == val
        if not tr.any():
            raise ValueError('At least two development participants are required')
        seed = stable_seed(cfg['seed'], ds.task, family, val, 'inner')
        print(f'    {family}: validation {val}', flush=True)
        if family in DEEP:
            b = backend(family, ds, cfg)
            (xt, xv), _ = scale_from_training(ds.x[tr], ds.x[va], enabled=cfg['scale_input'])
            fit = b.fit_inner(xt, ds.y[tr], xv, ds.y[va], parameters, seed, directory / f'history_{val}.csv')
            prediction, epoch = fit['prediction'], fit['best_epoch']
            convergence = []
        else:
            model = fit_baseline(family, ds.x[tr], ds.y[tr], ds.groups[tr], parameters, seed)
            prediction, epoch = predict_baseline(model, ds.x[va]), None
            convergence = model.get('convergence_warnings', [])
        metric = cycle_metrics(ds.y[va], np.asarray(prediction))
        row = {'validation_subject': val, 'training_subjects': sorted(set(ds.groups[tr])),
               'n_validation_cycles': int(va.sum()), 'parameter_hash': signature,
               'data_fingerprint': ds.fingerprint, 'mae': float(metric.mae.mean()),
               'rmse': float(metric.rmse.mean()), 'best_epoch': epoch, 'convergence_warnings': convergence}
        write_json(path, row)
        rows.append(row)
    score = float(np.mean([r['mae'] for r in rows]))
    if not np.isfinite(score):
        raise ValueError('Nonfinite full subject-wise validation objective')
    result = {'family': family, 'parameters': parameters, 'inner_subject_mae': score, 'folds': rows,
              'epochs': max(1, int(np.ceil(np.median([r['best_epoch'] for r in rows])))) if family in DEEP else None}
    write_json(directory / 'summary.json', result)
    return result

def tune_family(ds, family, cfg, directory):
    directory = Path(directory)
    done = directory / 'selected.json'
    if done.exists():
        return read_json(done)
    directory.mkdir(parents=True, exist_ok=True)
    if family not in DEEP:
        candidates = [cv_candidate(ds, family, p, cfg, directory / f'candidate_{i:03d}')
                      for i, p in enumerate(baseline_grid(family, cfg.get('smoke', False)))]
        selected = min(candidates, key=lambda r: (r['inner_subject_mae'], parameter_hash(r['parameters'])))
    else:
        from models import training_dependencies, search_space
        tf, kt = training_dependencies()
        class SubjectTuner(kt.Tuner):
            def run_trial(self, trial, *args, **kwargs):
                try:
                    result = cv_candidate(ds, family, dict(trial.hyperparameters.values), cfg,
                                          directory / 'candidates' / trial.trial_id)
                except (ValueError, tf.errors.ResourceExhaustedError) as exc:
                    write_json(directory / 'candidates' / trial.trial_id / 'failure.json', {'error': str(exc)})
                    raise kt.errors.FailedTrialError(str(exc)) from exc
                return {'inner_subject_mae': result['inner_subject_mae']}
        hp = search_space(family, cfg['search_profile'])
        oracle = kt.oracles.BayesianOptimizationOracle(objective=kt.Objective('inner_subject_mae', 'min'),
            max_trials=cfg['max_trials'], num_initial_points=min(3, cfg['max_trials']),
            hyperparameters=hp, allow_new_entries=False, tune_new_entries=False,
            seed=stable_seed(cfg['seed'], ds.task, family, 'oracle'), max_consecutive_failed_trials=3)
        tuner = SubjectTuner(oracle=oracle, hypermodel=None, directory=str(directory), project_name='tuner', overwrite=False)
        tuner.search()
        best = tuner.oracle.get_best_trials(1)
        if not best or best[0].score is None or not np.isfinite(best[0].score):
            raise ValueError(f'No complete successful candidate for {family}')
        selected = read_json(directory / 'candidates' / best[0].trial_id / 'summary.json')
    write_json(done, selected)
    return selected

def fit_frozen_model(ds, selected, cfg, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = directory / 'model.json'
    if descriptor.exists():
        return read_json(descriptor)
    family = selected['family']
    seed = stable_seed(cfg['seed'], ds.task, family, 'final_refit')
    if family in DEEP:
        b = backend(family, ds, cfg)
        (x,), scaler = scale_from_training(ds.x, enabled=cfg['scale_input'])
        model = b.fit_final(x, ds.y, selected['parameters'], seed, selected['epochs'], directory / 'history.csv')
        b.save(model, directory / 'model.keras')
        b.clear()
        result = {'family': family, 'file': 'model.keras', 'input_scaler': scaler}
    else:
        model = fit_baseline(family, ds.x, ds.y, ds.groups, selected['parameters'], seed)
        joblib.dump(model, directory / 'model.joblib')
        result = {'family': family, 'file': 'model.joblib',
                  'convergence_warnings': model.get('convergence_warnings', [])}
    result.update(parameters=selected['parameters'], epochs=selected['epochs'], seed=seed,
                  n_training_participants=len(ds.subjects), n_training_cycles=len(ds.x),
                  training_participants=ds.subjects, data_fingerprint=ds.fingerprint)
    write_json(descriptor, result)
    return result

def predict_saved(directory, x):
    directory = Path(directory)
    info = read_json(directory / 'model.json')
    if info['family'] in DEEP:
        from models import training_dependencies
        tf, _ = training_dependencies()
        scaler = info['input_scaler']
        z = (x - np.array(scaler['mean'])[None, None]) / np.array(scaler['std'])[None, None]
        model = tf.keras.models.load_model(directory / info['file'], compile=False)
        try:
            return np.asarray(model.predict(z, verbose=0))
        finally:
            del model
            tf.keras.backend.clear_session()
    # Only load models created by this notebook, whose file hashes are frozen.
    return np.asarray(predict_baseline(joblib.load(directory / info['file']), x))

def mean_ensemble(predictions):
    """Average predictions BEFORE errors: models are not independent participants."""
    values = [np.asarray(p) for p in predictions]
    if not values or any(v.shape != values[0].shape or not np.isfinite(v).all() for v in values):
        raise ValueError('Ensemble members require identical finite cycle/time/component arrays')
    return np.mean(np.stack(values), axis=0)
