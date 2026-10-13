"""Backend-independent nested LOSO protocol, metrics, and resumable records."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd

from data import Dataset, nested_indices


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_seed(seed, *labels):
    text = "|".join(map(str, (seed, *labels)))
    return int.from_bytes(sha256(text.encode()).digest()[:4], "big") % (2**31 - 1)


def parameter_hash(parameters):
    return sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()


def scale_from_training(train, *others, enabled=False):
    """Only training inputs determine scale. Targets retain their original units."""
    if enabled:
        mean = np.mean(train, axis=(0, 1), keepdims=True, dtype=np.float64)
        std = np.std(train, axis=(0, 1), keepdims=True, dtype=np.float64)
        std = np.where(std < 1e-8, 1.0, std)
    else:
        mean = np.zeros((1, 1, train.shape[-1]))
        std = np.ones_like(mean)
    arrays = [((arr - mean) / std).astype(np.float32) for arr in (train,) + others]
    return arrays, {"enabled": enabled, "mean": mean.reshape(-1).tolist(),
                    "std": std.reshape(-1).tolist()}


def cycle_metrics(truth, prediction):
    if truth.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction shape or nonfinite predictions")
    error = truth.astype(np.float64) - prediction.astype(np.float64)
    rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
    mae = np.mean(np.abs(error), axis=(1, 2))
    span = np.ptp(truth.astype(np.float64), axis=1).reshape(-1)
    nrmse = np.divide(rmse, span, out=np.full_like(rmse, np.nan), where=span > 1e-12)
    return pd.DataFrame({"rmse": rmse, "nrmse": nrmse, "mae": mae,
                         "target_range": span, "nrmse_defined": span > 1e-12})


def evaluate_candidate(ds, outer_subject, parameters, backend, seed, directory, scale_input=False):
    """Score one candidate equally across inner validation participants.

    The backend only receives inner train/validation data. Outer test arrays
    are never supplied to tuning, early stopping, or input scaling.
    """
    directory = Path(directory)
    signature = parameter_hash(parameters)
    records = []
    for validation_subject, tr, va, _ in nested_indices(ds.groups, outer_subject):
        path = directory / f"inner_{validation_subject}.json"
        if path.exists():
            record = read_json(path)
            if record["parameter_hash"] != signature:
                raise ValueError(f"Cached trial has different parameters: {path}")
            records.append(record)
            continue
        (xtr, xva), scaler = scale_from_training(ds.x[tr], ds.x[va], enabled=scale_input)
        fit_seed = stable_seed(seed, ds.task, outer_subject, validation_subject, "inner")
        fit = backend.fit_inner(xtr, ds.y[tr], xva, ds.y[va], parameters, fit_seed,
                                directory / f"history_{validation_subject}.csv")
        metric = cycle_metrics(ds.y[va], fit["prediction"])
        record = {"outer_subject": outer_subject, "validation_subject": validation_subject,
                  "training_subjects": sorted(set(ds.groups[tr])), "n_training_cycles": len(tr),
                  "n_validation_cycles": len(va), "seed": fit_seed,
                  "parameter_hash": signature, "scaler": scaler,
                  "best_epoch": int(fit["best_epoch"]), "epochs_run": int(fit["epochs_run"]),
                  "validation_mae": float(metric.mae.mean()),
                  "validation_rmse": float(metric.rmse.mean())}
        write_json(path, record)
        records.append(record)
    score = float(np.mean([r["validation_mae"] for r in records]))
    if not np.isfinite(score):
        raise ValueError("Nonfinite inner objective")
    summary = {"parameters": parameters, "inner_subject_mae": score, "folds": records}
    write_json(directory / "inner_summary.json", summary)
    return summary


def refit_and_evaluate(ds, outer_subject, parameters, inner_summary, backend, seed,
                       directory, architecture, scale_input=False):
    """Refit on ALL outer-training participants for an inner-derived epoch budget."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tr = np.flatnonzero(ds.groups != outer_subject)
    te = np.flatnonzero(ds.groups == outer_subject)
    if len(te) == 0 or set(ds.groups[tr]) & set(ds.groups[te]):
        raise ValueError("Invalid outer subject separation")
    if set(r["validation_subject"] for r in inner_summary["folds"]) != set(ds.groups[tr]):
        raise ValueError("The winning trial did not validate every outer-training participant")
    if any(r["outer_subject"] != outer_subject for r in inner_summary["folds"]):
        raise ValueError("Inner summary belongs to a different outer fold")
    if parameter_hash(inner_summary["parameters"]) != parameter_hash(parameters):
        raise ValueError("Refit parameters differ from the selected inner trial")
    # Predefined rule: ceiling of median inner best epoch; outer test never selects epochs.
    epochs = max(1, int(np.ceil(np.median([r["best_epoch"] for r in inner_summary["folds"]]))))
    (xtr, xte), scaler = scale_from_training(ds.x[tr], ds.x[te], enabled=scale_input)
    refit_seed = stable_seed(seed, ds.task, outer_subject, "refit")
    model = backend.fit_final(xtr, ds.y[tr], parameters, refit_seed, epochs,
                              directory / "final_history.csv")
    # This is the first point where the fitted outer model sees test inputs.
    prediction = backend.predict(model, xte)
    metrics = cycle_metrics(ds.y[te], prediction)
    metadata = ds.cycles.iloc[te].reset_index(drop=True)
    per_cycle = pd.concat([metadata, metrics], axis=1)
    per_cycle.insert(0, "task", ds.task)
    per_cycle.insert(1, "architecture", architecture)
    per_cycle.insert(2, "seed", seed)
    per_cycle.to_csv(directory / "cycle_metrics.csv", index=False)
    np.savez_compressed(directory / "predictions.npz", measured=ds.y[te], predicted=prediction,
                        cycle_ids=metadata.cycle_id.to_numpy(dtype=str),
                        subject_ids=ds.groups[te], gait_phase=np.linspace(0, 100, ds.y.shape[1]))
    backend.save(model, directory / "model.keras")
    backend.clear()
    summary = {"task": ds.task, "architecture": architecture, "seed": seed,
               "subject": outer_subject, "n_cycles": len(te), "n_train_cycles": len(tr),
               "n_train_subjects": len(set(ds.groups[tr])), "epochs": epochs,
               "refit_seed": refit_seed, "rmse": float(metrics.rmse.mean()),
               "nrmse": float(metrics.nrmse.mean()) if metrics.nrmse.notna().any() else None,
               "mae": float(metrics.mae.mean()),
               "undefined_nrmse_cycles": int((~metrics.nrmse_defined).sum()),
               "inner_subject_mae": inner_summary["inner_subject_mae"],
               "parameters": parameters, "input_scaler": scaler,
               "epoch_rule": "ceiling of median best epoch across winning inner LOSO fits"}
    # Written last: a completed outer fold can be safely skipped on resume.
    write_json(directory / "result.json", summary)
    return summary


def summarize_subject_results(records, directory, bootstrap_seed=2026, resamples=2000):
    """Bootstrap participants; repeated seeds are averaged within participant first."""
    directory = Path(directory)
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))}
                       for r in records])
    if df.empty:
        return
    df.to_csv(directory / "outer_subject_metrics_by_seed.csv", index=False)
    averaged = df.groupby(["task", "architecture", "subject"], as_index=False).agg(
        rmse=("rmse", "mean"), nrmse=("nrmse", "mean"), mae=("mae", "mean"),
        n_seeds=("seed", "nunique"), rmse_seed_sd=("rmse", "std"),
        nrmse_seed_sd=("nrmse", "std"), mae_seed_sd=("mae", "std"))
    averaged.to_csv(directory / "outer_subject_metrics_seed_average.csv", index=False)
    rows = []
    for (task, arch), group in averaged.groupby(["task", "architecture"]):
        for metric in ["rmse", "nrmse", "mae"]:
            values = group[metric].dropna().to_numpy()
            if not len(values):
                continue
            rng = np.random.default_rng(stable_seed(bootstrap_seed, task, arch, metric))
            if len(values) >= 2:
                boot = np.median(rng.choice(values, size=(resamples, len(values)), replace=True), axis=1)
                low, high = np.quantile(boot, [0.025, 0.975])
            else:
                low = high = np.nan
            rows.append({"task": task, "architecture": arch, "metric": metric,
                         "n_participants": len(values), "min_seeds_per_subject": int(group.n_seeds.min()),
                         "max_seeds_per_subject": int(group.n_seeds.max()),
                         "median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
                         "q75": float(np.quantile(values, .75)),
                         "median_ci95_lower": low, "median_ci95_upper": high,
                         "ci_method": "participant percentile bootstrap after averaging seeds"})
    pd.DataFrame(rows).to_csv(directory / "cohort_summary.csv", index=False)
