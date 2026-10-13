#!/usr/bin/env python3
"""Nested subject-wise LOSO for the uploaded Shida ankle/GRF archive.

Start with --mode audit (the default). See README.md before a full training run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import pandas as pd

from data import TASK_ORDER, audit_datasets, load_datasets
from protocol import read_json, refit_and_evaluate, summarize_subject_results, write_json


def software_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ["numpy", "pandas", "openpyxl", "tensorflow", "keras", "keras-tuner", "scikit-learn"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def code_fingerprint():
    digest = sha256()
    for name in ["data.py", "models.py", "protocol.py", "nested_loso.py"]:
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def ensure_configuration(directory, config, resume):
    directory = Path(directory)
    path = directory / "experiment_config.json"
    if path.exists():
        if not resume:
            raise ValueError(f"Existing experiment: {directory}. Use --resume or a new --output directory.")
        if read_json(path) != config:
            raise ValueError(f"Data/code/settings/software changed in {directory}; use a new output directory")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Nonempty experiment without a configuration: {directory}")
        write_json(path, config)


def write_audit(audit, directory, args):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "dataset_audit.json", {
        "tasks": audit["tasks"], "source": str(Path(args.data).resolve()),
        "paired_by": "same column position in a participant workbook's in/out sheets",
        "preassembled_train_files_used": False,
        "zero_filling_used": False, "smoke_test": args.smoke_test})
    pd.DataFrame(audit["participants"]).to_csv(directory / "participants.csv", index=False)
    pd.DataFrame(audit["cycles"]).to_csv(directory / "cycles.csv", index=False)
    write_json(directory / "nested_splits.json", audit["splits"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Shida-LOSO-ANKLE.zip or extracted dataset/Raw directory")
    parser.add_argument("--output", default="nested_loso_results")
    parser.add_argument("--mode", choices=["audit", "train"], default="audit")
    parser.add_argument("--tasks", nargs="+", choices=TASK_ORDER, default=TASK_ORDER)
    parser.add_argument("--architectures", nargs="+", choices=["lstm", "gru", "conv1d_lstm"],
                        default=["lstm", "gru", "conv1d_lstm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--search-profile", choices=["original", "compact"], default="original")
    parser.add_argument("--scale-input", action="store_true", help="Optional training-only standardization; off by default")
    parser.add_argument("--outer-subjects", nargs="+", help="Run selected outer folds while retaining ALL inner subjects")
    parser.add_argument("--resume", action="store_true", help="Resume only if data, code, settings and versions match")
    parser.add_argument("--smoke-test", action="store_true", help="One outer fold, three subjects, tiny networks, one epoch; NOT research results")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--threads", type=int, default=0, help="TensorFlow CPU threads; 0 keeps its defaults")
    args = parser.parse_args(argv)
    if min(args.max_trials, args.epochs, args.batch_size) < 1 or args.patience < 0 or args.threads < 0:
        parser.error("Trials, epochs and batch size must be positive; patience/threads must be nonnegative")
    if any(s < 0 or s >= 2**31 for s in args.seeds):
        parser.error("Seeds must be in [0, 2**31)")
    for attr in ["tasks", "architectures", "seeds"]:
        if len(getattr(args, attr)) != len(set(getattr(args, attr))):
            parser.error(f"Duplicate entries in --{attr}")
    if args.outer_subjects:
        args.outer_subjects = [s.upper() for s in args.outer_subjects]
    if args.smoke_test:
        args.mode, args.tasks, args.seeds = "train", ["AP_Left"], [args.seeds[0]]
        args.search_profile, args.max_trials, args.epochs, args.patience = "smoke", 1, 1, 0
        args.outer_subjects = None
    return args


def main(argv=None):
    args = parse_args(argv)
    # Configure environment BEFORE TensorFlow is imported.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    datasets = load_datasets(args.data, args.tasks)
    if args.smoke_test:
        datasets = {task: ds.subset(ds.subjects[:3]) for task, ds in datasets.items()}
    audit = audit_datasets(datasets)
    outer_selection = {}
    for task, ds in datasets.items():
        selected = ds.subjects[:1] if args.smoke_test else (args.outer_subjects or ds.subjects)
        missing = set(selected) - set(ds.subjects)
        if missing:
            raise ValueError(f"{task}: requested outer participants absent: {sorted(missing)}")
        outer_selection[task] = selected
        print(f"{task}: {len(ds.subjects)} participants, {len(ds.x)} cycles, "
              f"{ds.x.shape[1]} time points; {len(selected)} outer folds", flush=True)
    fit_count = sum(len(outer_selection[t]) * (args.max_trials * (len(ds.subjects) - 1) + 1)
                    for t, ds in datasets.items()) * len(args.architectures) * len(args.seeds)
    print(f"Planned upper bound: {fit_count:,} model fits (inner trials plus outer refits).", flush=True)
    output = Path(args.output).expanduser().resolve()
    write_audit(audit, output / "audit", args)
    if args.mode == "audit":
        print(f"Audit passed. No training performed. Reports: {output / 'audit'}")
        return 0

    from models import TensorFlowBackend, configure_tensorflow, select_hyperparameters
    hardware = configure_tensorflow(args.seeds[0], args.threads)
    versions = software_versions()
    settings = {key: getattr(args, key) for key in ["max_trials", "epochs", "patience", "batch_size",
                                                   "search_profile", "scale_input", "smoke_test", "threads"]}
    run_info = {"utc_started": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                "software": versions, "hardware": hardware, "code_fingerprint": code_fingerprint(),
                "expected_fit_upper_bound": fit_count}
    write_json(output / "run_metadata.json", run_info)
    experiment_dirs = []
    # Validate every requested configuration before spending time training.
    for task, ds in datasets.items():
        for architecture in args.architectures:
            for seed in args.seeds:
                directory = output / task / architecture / f"seed_{seed}"
                config = {"task": task, "data_fingerprint": ds.fingerprint,
                          "architecture": architecture, "seed": seed, "settings": settings,
                          "software": {k: v for k, v in versions.items() if k != "platform"},
                          "code_fingerprint": run_info["code_fingerprint"]}
                ensure_configuration(directory, config, args.resume)
                experiment_dirs.append(directory)
    for task, ds in datasets.items():
        for architecture in args.architectures:
            backend = TensorFlowBackend(architecture, ds.x.shape[1], args.search_profile,
                                        args.epochs, args.patience, args.batch_size, args.verbose)
            for seed in args.seeds:
                experiment = output / task / architecture / f"seed_{seed}"
                for outer_subject in outer_selection[task]:
                    directory = experiment / f"outer_{outer_subject}"
                    if (directory / "result.json").exists():
                        print(f"Completed fold retained: {task}/{architecture}/{seed}/{outer_subject}", flush=True)
                        continue
                    print(f"Training {task}/{architecture}/seed={seed}/outer={outer_subject}", flush=True)
                    params, inner_summary = select_hyperparameters(
                        ds, outer_subject, architecture, backend, seed, directory,
                        args.search_profile, args.max_trials, args.scale_input, args.resume)
                    refit_and_evaluate(ds, outer_subject, params, inner_summary, backend, seed,
                                       directory, architecture, args.scale_input)
    # Scope summary to the task/architecture/seed combinations requested in this invocation.
    records = [read_json(p) for directory in experiment_dirs
               for p in sorted(directory.glob("outer_*/result.json"))]
    summarize_subject_results(records, output)
    coverage = [{"task": task, "architecture": arch, "seed": seed,
                 "available_subjects": ds.subjects,
                 "completed_subjects": sorted(r["subject"] for r in records
                                              if r["task"] == task and r["architecture"] == arch and r["seed"] == seed)}
                for task, ds in datasets.items() for arch in args.architectures for seed in args.seeds]
    write_json(output / "coverage.json", {"smoke_test": args.smoke_test, "experiments": coverage})
    print(f"Finished. Participant-level metrics and coverage report: {output}", flush=True)
    if args.smoke_test:
        print("SMOKE TEST ONLY: these results are not suitable for the manuscript.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
