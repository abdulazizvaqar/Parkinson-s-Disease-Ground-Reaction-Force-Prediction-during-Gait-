"""Leakage and aggregation tests. No TensorFlow dependency and no model claims."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import unittest
import numpy as np
import pandas as pd

from data import Dataset, assert_subject_separation, canonical_subject, nested_indices
from protocol import (cycle_metrics, evaluate_candidate, read_json, refit_and_evaluate,
                      scale_from_training, summarize_subject_results)


def fixture():
    # Unequal cycle counts detect accidental cycle weighting of the inner score.
    groups = np.array(["SUB01"] * 2 + ["SUB02"] * 3 + ["SUB03"] * 1 + ["SUB04"] * 4)
    participant_value = np.array([int(s[-2:]) for s in groups], dtype=np.float32)
    x = np.repeat(participant_value[:, None, None], 8, axis=1)
    y = x + np.linspace(0, 1, 8, dtype=np.float32)[None, :, None]
    metadata = pd.DataFrame({"subject": groups, "cycle_id": [f"c{i}" for i in range(len(groups))]})
    return Dataset("AP_Left", x, y, groups, metadata, "synthetic-test-only")


class RecordingBackend:
    """A deterministic test double; it does not train or approximate a neural net."""
    def __init__(self):
        self.inner_calls = []
        self.final_calls = []

    def fit_inner(self, xtr, ytr, xva, yva, parameters, seed, path):
        self.inner_calls.append((xtr.copy(), ytr.copy(), xva.copy(), yva.copy()))
        subject = int(xva[0, 0, 0])
        return {"prediction": yva + subject, "best_epoch": subject, "epochs_run": subject + 1}

    def fit_final(self, xtr, ytr, parameters, seed, epochs, path):
        self.final_calls.append((xtr.copy(), ytr.copy(), epochs))
        return np.mean(ytr, axis=0, keepdims=True)

    def predict(self, model, inputs):
        return np.repeat(model, len(inputs), axis=0)

    def save(self, model, path):
        Path(path).write_text("TEST DOUBLE ONLY", encoding="utf-8")

    def clear(self):
        pass


class ProtocolTests(unittest.TestCase):
    def test_all_nested_folds_keep_whole_subjects(self):
        ds = fixture()
        for outer in ds.subjects:
            inner_subjects = []
            for val, tr, va, te in nested_indices(ds.groups, outer):
                inner_subjects.append(val)
                self.assertEqual(set(ds.groups[te]), {outer})
                self.assertEqual(set(ds.groups[va]), {val})
                self.assertEqual(set(ds.groups[tr]), set(ds.subjects) - {outer, val})
                self.assertEqual(sorted(np.concatenate([tr, va, te]).tolist()), list(range(len(ds.x))))
            self.assertEqual(set(inner_subjects), set(ds.subjects) - {outer})

    def test_overlap_is_rejected(self):
        ds = fixture()
        with self.assertRaises(ValueError):
            assert_subject_separation(ds.groups, np.array([0, 2, 5, 6, 7, 8, 9]),
                                      np.array([1]), np.array([3, 4]))

    def test_subject_tokens_do_not_confuse_sub01_with_sub10(self):
        self.assertEqual(canonical_subject("SUB01_off_UB01_walk"), "SUB01")
        self.assertEqual(canonical_subject("SUB10_walk"), "SUB10")
        with self.assertRaises(ValueError):
            canonical_subject("SUB01_UB10_walk")

    def test_scaler_never_uses_validation_or_test_values(self):
        tr = np.array([0., 2.], dtype=np.float32).reshape(1, 2, 1)
        va = np.full((1, 2, 1), 1000., dtype=np.float32)
        (scaled_tr, scaled_va), record = scale_from_training(tr, va, enabled=True)
        self.assertEqual(record["mean"], [1.])
        self.assertEqual(record["std"], [1.])
        np.testing.assert_allclose(scaled_tr.reshape(-1), [-1., 1.])
        np.testing.assert_allclose(scaled_va, 999.)

    def test_candidate_equal_subject_weight_and_outer_exclusion(self):
        ds, backend = fixture(), RecordingBackend()
        with tempfile.TemporaryDirectory() as tmp:
            summary = evaluate_candidate(ds, "SUB01", {}, backend, 42, tmp)
        self.assertEqual(len(backend.inner_calls), 3)
        self.assertAlmostEqual(summary["inner_subject_mae"], 3.)  # mean(2,3,4), not cycle-weighted
        for xtr, ytr, xva, yva in backend.inner_calls:
            self.assertNotIn(1., xtr)
            self.assertNotIn(1., xva)
            self.assertFalse(set(xtr[:, 0, 0]) & set(xva[:, 0, 0]))

    def test_poisoning_outer_data_cannot_change_inner_selection(self):
        original, changed = fixture(), fixture()
        changed.x[changed.groups == "SUB01"] = 1e7
        changed.y[changed.groups == "SUB01"] = -1e7
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            one = evaluate_candidate(original, "SUB01", {}, RecordingBackend(), 42, a)
            two = evaluate_candidate(changed, "SUB01", {}, RecordingBackend(), 42, b)
        self.assertEqual(one, two)

    def test_completed_inner_fits_resume_without_retraining(self):
        ds, backend = fixture(), RecordingBackend()
        with tempfile.TemporaryDirectory() as tmp:
            first = evaluate_candidate(ds, "SUB01", {"a": 1}, backend, 42, tmp)
            again = evaluate_candidate(ds, "SUB01", {"a": 1}, backend, 42, tmp)
            self.assertEqual(first, again)
            self.assertEqual(len(backend.inner_calls), 3)
            with self.assertRaises(ValueError):
                evaluate_candidate(ds, "SUB01", {"a": 2}, backend, 42, tmp)

    def test_outer_refit_uses_all_development_subjects_and_inner_epochs(self):
        ds, backend = fixture(), RecordingBackend()
        with tempfile.TemporaryDirectory() as tmp:
            selected = evaluate_candidate(ds, "SUB01", {}, backend, 42, Path(tmp) / "inner")
            result = refit_and_evaluate(ds, "SUB01", {}, selected, backend, 42, Path(tmp) / "outer", "test")
            xtr, ytr, epochs = backend.final_calls[0]
            self.assertEqual(set(xtr[:, 0, 0]), {2., 3., 4.})
            self.assertEqual(len(xtr), 8)
            self.assertEqual(epochs, 3)
            self.assertEqual(result["n_cycles"], 2)
            saved = np.load(Path(tmp) / "outer" / "predictions.npz")
            self.assertEqual(set(saved["subject_ids"]), {"SUB01"})
            self.assertEqual(saved["measured"].shape, (2, 8, 1))
            self.assertEqual(read_json(Path(tmp) / "outer" / "result.json")["epochs"], 3)

    def test_undefined_nrmse_is_not_replaced_by_arbitrary_denominator(self):
        truth = np.array([[[2.], [2.]], [[0.], [2.]]])
        result = cycle_metrics(truth, truth + 1)
        self.assertTrue(np.isnan(result.nrmse.iloc[0]))
        self.assertEqual(result.nrmse.iloc[1], .5)
        self.assertEqual(result.rmse.iloc[0], 1.)

    def test_repeated_seeds_do_not_inflate_participant_count(self):
        records = [{"task": "AP_Left", "architecture": "lstm", "subject": subject,
                    "seed": seed, "rmse": value, "nrmse": value / 10, "mae": value}
                   for subject, value in [("SUB01", 1.), ("SUB02", 3.)] for seed in [1, 2, 3]]
        with tempfile.TemporaryDirectory() as tmp:
            summarize_subject_results(records, tmp, resamples=30)
            summary = pd.read_csv(Path(tmp) / "cohort_summary.csv")
            self.assertTrue((summary.n_participants == 2).all())
            self.assertTrue((summary.min_seeds_per_subject == 3).all())
            self.assertEqual(summary.loc[summary.metric == "rmse", "median"].iloc[0], 2.)


if __name__ == "__main__":
    unittest.main()
