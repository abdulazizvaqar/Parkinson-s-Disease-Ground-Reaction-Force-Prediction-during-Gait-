"""Read the paired per-participant Raw workbooks in Shida-LOSO-ANKLE."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import pandas as pd


TASKS = {
    "apleft": "AP_Left", "apright": "AP_Right",
    "longlef": "Vertical_Left", "longleft": "Vertical_Left",
    "longrig": "Vertical_Right", "longright": "Vertical_Right",
    "verticalleft": "Vertical_Left", "verticalright": "Vertical_Right",
    "mlleft": "ML_Left", "mlright": "ML_Right",
}
TASK_ORDER = ["AP_Left", "AP_Right", "Vertical_Left", "Vertical_Right", "ML_Left", "ML_Right"]
RAW_PATTERN = re.compile(r"(?:^|/)Raw/([^/]+)/([^/]+\.xlsx)$", re.I)
SUBJECT_PATTERN = re.compile(r"(?:SUB|UB)(\d+)", re.I)


@dataclass
class Dataset:
    task: str
    x: np.ndarray                       # [cycles, 100 time points, 1 feature]
    y: np.ndarray
    groups: np.ndarray                  # participant ID for EVERY cycle
    cycles: pd.DataFrame                # source and cycle-position provenance
    fingerprint: str

    @property
    def subjects(self):
        return sorted(np.unique(self.groups).tolist())

    def subset(self, subjects):
        mask = np.isin(self.groups, subjects)
        if not mask.any():
            raise ValueError("Empty participant subset")
        fingerprint = sha256((self.fingerprint + repr(sorted(subjects))).encode()).hexdigest()
        return Dataset(self.task, self.x[mask], self.y[mask], self.groups[mask],
                       self.cycles.loc[mask].reset_index(drop=True), fingerprint)


def canonical_subject(text):
    matches = SUBJECT_PATTERN.findall(str(text))
    ids = {f"SUB{int(value):02d}" for value in matches}
    if len(ids) != 1:
        raise ValueError(f"Expected one unambiguous participant in {text!r}; got {ids}")
    return ids.pop()


def workbook_arrays(payload, filename, subject, expected_timesteps=100):
    """Keep the original in/out column order. GRF headers lack trial identifiers."""
    with pd.ExcelFile(BytesIO(payload), engine="openpyxl") as book:
        if not {"in", "out"}.issubset(book.sheet_names):
            raise ValueError(f"{filename}: paired 'in' and 'out' sheets are required")
        inputs = pd.read_excel(book, sheet_name="in", header=0)
        targets = pd.read_excel(book, sheet_name="out", header=0)
    if inputs.shape != targets.shape or inputs.shape[0] != expected_timesteps or not inputs.shape[1]:
        raise ValueError(f"{filename}: expected matched ({expected_timesteps}, cycles) sheets; "
                         f"got in={inputs.shape}, out={targets.shape}")
    for header in inputs.columns:
        if canonical_subject(header) != subject:
            raise ValueError(f"{filename}: input column belongs to a different participant: {header}")
    try:
        x = inputs.to_numpy(dtype=np.float32).T[:, :, None]
        y = targets.to_numpy(dtype=np.float32).T[:, :, None]
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{filename}: nonnumeric waveform value") from exc
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{filename}: missing/nonfinite values; no silent zero filling or exclusion is allowed")
    metadata = [{"subject": subject, "cycle_id": f"{subject}:C{i+1:03d}",
                 "column_position_1based": i + 1, "input_header": str(inputs.columns[i]),
                 "target_header": str(targets.columns[i]), "source_workbook": filename}
                for i in range(len(x))]
    return x, y, metadata


def load_datasets(source, requested_tasks=None):
    """Accept the uploaded ZIP, its extracted root, or the Raw directory itself.

    Never load the preassembled Train files: they repeat subjects across folds
    and some supplied input matrices were concatenated along the wrong axis.
    """
    source = Path(source).expanduser().resolve()
    requested = set(requested_tasks or TASK_ORDER)
    unknown = requested - set(TASK_ORDER)
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    archive = None
    if source.is_file() and source.suffix.lower() == ".zip":
        archive = ZipFile(source)
        names = sorted(n for n in archive.namelist() if RAW_PATTERN.search(n))
        reader = archive.read
    elif source.is_dir():
        paths = {p.as_posix(): p for p in source.rglob("*.xlsx") if RAW_PATTERN.search(p.as_posix())}
        names = sorted(paths)
        reader = lambda name: paths[name].read_bytes()
    else:
        raise ValueError(f"Expected an existing ZIP or directory: {source}")
    accum, seen, fingerprints = {}, set(), {}
    try:
        for name in names:
            match = RAW_PATTERN.search(name)
            task = TASKS.get(re.sub(r"[^a-z]", "", match.group(1).lower()))
            if task not in requested:
                continue
            subject = canonical_subject(match.group(2))
            if (task, subject) in seen:
                raise ValueError(f"Duplicate Raw workbook for {task}/{subject}; provide one dataset root")
            seen.add((task, subject))
            payload = reader(name)
            relative_name = "Raw/" + name.split("/")[-2] + "/" + name.split("/")[-1]
            x, y, metadata = workbook_arrays(payload, relative_name, subject)
            accum.setdefault(task, []).append((subject, x, y, metadata))
            fingerprints.setdefault(task, sha256()).update(relative_name.encode() + sha256(payload).digest())
    finally:
        if archive is not None:
            archive.close()
    absent = requested - set(accum)
    if absent:
        raise ValueError(f"No Raw workbooks found for {sorted(absent)} in {source}")
    result = {}
    for task in TASK_ORDER:
        if task not in accum:
            continue
        records = sorted(accum[task], key=lambda item: item[0])
        if len(records) < 3:
            raise ValueError(f"{task}: at least three participants are required for nested LOSO")
        result[task] = Dataset(
            task, np.concatenate([r[1] for r in records]), np.concatenate([r[2] for r in records]),
            np.concatenate([np.repeat(r[0], len(r[1])) for r in records]),
            pd.DataFrame([m for r in records for m in r[3]]), fingerprints[task].hexdigest())
    return result


def nested_indices(groups, outer_subject):
    """Yield train, validation, test indices; one entire subject in each holdout."""
    groups = np.asarray(groups)
    subjects = sorted(np.unique(groups).tolist())
    if outer_subject not in subjects or len(subjects) < 3:
        raise ValueError("Unknown outer participant or fewer than three participants")
    test = np.flatnonzero(groups == outer_subject)
    for validation_subject in subjects:
        if validation_subject == outer_subject:
            continue
        validation = np.flatnonzero(groups == validation_subject)
        train = np.flatnonzero((groups != outer_subject) & (groups != validation_subject))
        assert_subject_separation(groups, train, validation, test)
        yield validation_subject, train, validation, test


def assert_subject_separation(groups, train, validation, test):
    combined = np.concatenate([train, validation, test])
    if not np.array_equal(np.sort(combined), np.arange(len(groups))):
        raise ValueError("Each cycle must appear exactly once across the three partitions")
    sets = [set(np.asarray(groups)[idx]) for idx in [train, validation, test]]
    if any(not s for s in sets) or sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Participant overlap or an empty train/validation/test partition")


def audit_datasets(datasets):
    tasks, participants, cycles, splits = [], [], [], []
    for task, ds in datasets.items():
        tasks.append({"task": task, "n_participants": len(ds.subjects), "n_cycles": len(ds.x),
                      "timepoints": ds.x.shape[1], "features": ds.x.shape[2],
                      "missing_input_values": int((~np.isfinite(ds.x)).sum()),
                      "missing_target_values": int((~np.isfinite(ds.y)).sum()),
                      "subjects": ds.subjects, "fingerprint": ds.fingerprint})
        cycles.extend(ds.cycles.assign(task=task).to_dict("records"))
        for subject in ds.subjects:
            participants.append({"task": task, "subject": subject,
                                 "n_cycles": int(np.sum(ds.groups == subject))})
            for val_subject, tr, va, te in nested_indices(ds.groups, subject):
                splits.append({"task": task, "outer_test_subject": subject,
                               "inner_validation_subject": val_subject,
                               "inner_training_subjects": sorted(set(ds.groups[tr])),
                               "n_train_cycles": len(tr), "n_validation_cycles": len(va),
                               "n_test_cycles": len(te)})
    return {"tasks": tasks, "participants": participants, "cycles": cycles, "splits": splits}
