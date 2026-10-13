import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from data import Dataset
from protocol import read_json, write_json
from revision_data import Cohort, coordinate_dataset, load_cohort, REQUIRED_CONTEXT, harmonize_input_units
from revision_models import (participant_mean, fit_baseline, predict_baseline, mean_ensemble,
                             cv_candidate, BASELINES, DEEP)
from revision_reports import prediction_table, participant_table, save_reports, audit_external
import revision_study as study


def sample_cohort():
    rng = np.random.default_rng(5)
    groups = np.repeat(['P01', 'P02', 'P03'], 3)
    x = rng.normal(size=(9, 100, 1)).astype('float32')
    y = (0.2 * x + np.sin(np.linspace(0, 2 * np.pi, 100))[None, :, None]).astype('float32')
    meta = pd.DataFrame({'cycle_id': [f'C{i}' for i in range(9)], 'subject': groups, 'participant_id': groups,
        'trial_id': ['T0', 'T1', 'T1'] * 3, 'visit': 'internal', 'side': 'left', 'include': True, 'exclusion_reason': ''})
    ds = Dataset('AP_Left', x, y, groups, meta, 'synthetic-fixture')
    people = pd.DataFrame({'participant_id': ['P01', 'P02', 'P03'], 'sex': ['F', 'M', 'F'], 'age': [60, 70, 80]})
    context = {k: 'unknown' for k in REQUIRED_CONTEXT}
    context.update(cohort_id='test', dataset_role='internal', ankle_units='m', grf_units='N/kg',
                   units_verified=True, body_scale_definition='measured height in m')
    return Cohort(ds, context, people, pd.DataFrame(), meta, {})


def make_external_bundle(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    c = sample_cohort()
    context = dict(c.context, cohort_id='external_fixture', dataset_role='external', task='AP_Left',
                   harmonization_verified=True, participant_disjointness_verified=True)
    write_json(root / 'cohort.json', context)
    c.participants.to_csv(root / 'participants.csv', index=False)
    meta = c.ledger.copy()
    meta['visit'] = ['baseline', 'baseline', 'post_intervention'] * 3
    # T1 at baseline and T1 post are different trials, keyed by visit.
    trials = meta[['participant_id', 'trial_id', 'visit', 'side', 'include', 'exclusion_reason']].drop_duplicates()
    excluded = dict(participant_id='P01', trial_id='T99', visit='baseline', side='left', include=False, exclusion_reason='no force-plate contact')
    trials = pd.concat([trials, pd.DataFrame([excluded])], ignore_index=True)
    extra = {**excluded, 'cycle_id': 'excluded_cycle', 'subject': 'P01'}
    ledger = pd.concat([meta, pd.DataFrame([extra])], ignore_index=True)
    trials.to_csv(root / 'trials.csv', index=False)
    ledger.to_csv(root / 'cycles.csv', index=False)
    # Reverse waveform order to verify explicit IDs, never positional metadata pairing.
    np.savez_compressed(root / 'waveforms.npz', cycle_ids=meta.cycle_id.to_numpy(str)[::-1], x=c.ds.x[::-1], y=c.ds.y[::-1])
    return root


class RevisionTests(unittest.TestCase):
    def test_participant_weighted_mean_and_phase_ignore_ankle(self):
        y = np.zeros((4, 100, 1)); y[-1] = 10
        groups = np.array(['A', 'A', 'A', 'B'])
        self.assertTrue(np.allclose(participant_mean(y, groups), 5))
        x = np.zeros_like(y)
        for family, p in [('mean_waveform', {}), ('phase_only', {'harmonics': 3})]:
            model = fit_baseline(family, x, y, groups, p, 42)
            np.testing.assert_allclose(predict_baseline(model, x), predict_baseline(model, x + 1000))

    def test_mlp_is_single_hidden_layer_without_random_validation(self):
        c = sample_cohort()
        m = fit_baseline('shallow_mlp', c.ds.x, c.ds.y, c.ds.groups, {'units': 4, 'alpha': .01, 'max_iter': 5}, 42)
        self.assertFalse(m['model'].early_stopping)
        self.assertEqual(m['model'].hidden_layer_sizes, (4,))
        self.assertEqual(predict_baseline(m, c.ds.x).shape, c.ds.y.shape)

    def test_cv_separates_subjects_and_uses_equal_subject_score(self):
        c = sample_cohort(); ds = c.ds.subset(['P01', 'P02'])
        cfg = study.default_config()
        with tempfile.TemporaryDirectory() as tmp:
            result = cv_candidate(ds, 'ridge', {'alpha': 1}, cfg, tmp)
            for fold in result['folds']:
                self.assertNotIn(fold['validation_subject'], fold['training_subjects'])
                self.assertNotIn('P03', fold['training_subjects'])
            self.assertAlmostEqual(result['inner_subject_mae'], np.mean([f['mae'] for f in result['folds']]))

    def test_scalar_coordinates_cannot_fake_pelvis_or_rotation(self):
        c = sample_cohort()
        for mode in ['pelvis', 'combined', 'aligned', 'body']:
            with self.assertRaises(ValueError):
                coordinate_dataset(c, mode)
        with self.assertRaises(ValueError):
            coordinate_dataset(c, 'absolute', rotation_deg=15)
        a = coordinate_dataset(c, 'translation')
        b = coordinate_dataset(c, 'translation', [1, 0, 0])
        np.testing.assert_allclose(a.x, b.x, atol=1e-6)

    def test_physical_unit_conversion_does_not_fit_external_distribution(self):
        c = sample_cohort()
        converted = harmonize_input_units(c, 'mm')
        np.testing.assert_allclose(converted.ds.x, 1000 * c.ds.x)
        np.testing.assert_array_equal(converted.ds.y, c.ds.y)
        self.assertEqual(converted.context['ankle_units'], 'mm')

    def test_3d_coordinate_variants_keep_same_gait_force_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_external_bundle(tmp)
            context = read_json(root / 'cohort.json')
            context['canonical_axis_order'] = ['forward', 'left', 'up']
            write_json(root / 'cohort.json', context)
            with np.load(root / 'waveforms.npz') as f:
                ids, x = f['cycle_ids'], f['x']
            xyz = np.repeat(x, 3, axis=2)
            forces = np.broadcast_to(np.array([1., 2., 3.]), xyz.shape)
            np.savez_compressed(root / 'waveforms.npz', cycle_ids=ids, ankle_xyz=xyz, grf_xyz=forces,
                direction_xyz=np.tile([0., 1., 0.], (len(x), 1)), pelvis_xyz=xyz / 2,
                body_scale_m=np.repeat(1.7, len(x)))
            c = load_cohort(root, 'AP_Left', ['baseline'])
            for mode in ['absolute', 'translation', 'pelvis', 'aligned', 'body', 'combined']:
                np.testing.assert_allclose(coordinate_dataset(c, mode).y, 2.)

    def test_combined_coordinates_are_translation_and_yaw_invariant(self):
        c = sample_cohort(); rng = np.random.default_rng(99)
        xyz = rng.normal(size=(9, 100, 3)); pelvis = rng.normal(size=xyz.shape)
        c = replace(c, arrays={'ankle_xyz': xyz, 'pelvis_xyz': pelvis, 'direction_xyz': np.tile([1., 0., 0.], (9, 1)), 'body_scale_m': np.repeat(1.7, 9)})
        original = coordinate_dataset(c, 'combined')
        nuisance = coordinate_dataset(c, 'combined', [1.2, -2.3, .5], 35)
        np.testing.assert_allclose(original.x, nuisance.x, atol=1e-6)
        np.testing.assert_array_equal(original.y, nuisance.y)

    def test_ensemble_averages_before_error(self):
        a, b = np.ones((2, 100, 1)), -np.ones((2, 100, 1))
        np.testing.assert_array_equal(mean_ensemble([a, b]), np.zeros_like(a))
        with self.assertRaises(ValueError):
            mean_ensemble([a, b[:1]])

    def test_trial_intervals_and_duplicate_model_rows(self):
        c = sample_cohort()
        cycles = prediction_table(c.ds, np.zeros_like(c.ds.y), 'selected_deep')
        rows = participant_table(cycles, repeats=100)
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows.rmse_ci95_low.notna().all())
        cycles['trial_id'] = 'single_trial'
        rows = participant_table(cycles, repeats=100)
        self.assertTrue(rows.rmse_ci95_low.isna().all())
        with self.assertRaises(ValueError):
            participant_table(pd.concat([cycles, cycles]), repeats=100)

    def test_external_contract_preserves_visits_and_exclusion_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_external_bundle(Path(tmp) / 'data')
            c = load_cohort(root, 'AP_Left', ['baseline'], role='external')
            self.assertEqual(len(c.ds.x), 6)
            self.assertEqual(c.ds.cycles.cycle_id.iloc[0], 'C7')
            audit_external(c, ['baseline'], Path(tmp) / 'audit', {})
            flow = read_json(Path(tmp) / 'audit' / 'cohort_flow_totals.json')
            self.assertEqual(flow['n_valid_cycles'], 6)
            self.assertEqual(flow['n_trials_recorded'], 10)
            excluded = pd.read_csv(Path(tmp) / 'audit' / 'cycle_exclusion_reasons.csv')
            self.assertIn('no force-plate contact', excluded.analysis_exclusion_reason.tolist())

    def test_nested_baseline_integration_and_resume(self):
        c = sample_cohort(); cfg = study.default_config()
        cfg.update(families=list(BASELINES), smoke=True, bootstrap_resamples=100)
        with tempfile.TemporaryDirectory() as tmp:
            study.run_nested(c, cfg, tmp)
            self.assertTrue(read_json(Path(tmp) / 'nested_coverage.json')['complete'])
            rows = pd.read_csv(Path(tmp) / 'internal_reports' / 'participant_metrics.csv')
            self.assertEqual(len(rows), 12)
            with patch('revision_study.predict_saved', side_effect=AssertionError('Should resume')):
                study.run_nested(c, cfg, tmp)

    def test_full_refit_freezes_internal_only_and_external_runs_once(self):
        c = sample_cohort(); cfg = study.default_config(); cfg['bootstrap_resamples'] = 100
        cfg['include_loso_ensemble'] = True
        trained_subjects = []
        def fake_tune(ds, family, cfg, directory):
            scores = {'lstm': 3., 'gru': 1., 'conv1d_lstm': 2.}
            return dict(family=family, parameters={}, epochs=1, inner_subject_mae=scores.get(family, 4.), folds=[])
        def fake_fit(ds, selected, cfg, directory):
            # Recording test double, saved as a real mean model for inference tests.
            trained_subjects.append(tuple(ds.subjects))
            actual = dict(family='mean_waveform', parameters={}, epochs=None)
            return real_fit(ds, actual, cfg, directory)
        real_fit = study.fit_frozen_model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'experiment'
            with patch('revision_study.tune_family', fake_tune), patch('revision_study.fit_frozen_model', fake_fit):
                study.run_nested(c, cfg, root)
                study.final_refit(c, cfg, root)
            self.assertEqual(trained_subjects[-5:], [tuple(c.ds.subjects)] * 5)
            frozen = read_json(root / 'frozen_external_plan.json')
            self.assertEqual(frozen['selected_architecture'], 'gru')
            self.assertEqual(len(frozen['ensemble_members']), 3)
            external = make_external_bundle(Path(tmp) / 'external')
            output = study.external_evaluation(root, external)
            rows = pd.read_csv(output / 'participant_metrics.csv')
            self.assertEqual(len(rows), 3 * 6)  # Three people, six predictors; not 18 people.
            with patch('revision_study.predict_frozen', side_effect=AssertionError('Second external pass')):
                study.external_evaluation(root, external)
            frozen['config']['external_visits'] = ['post_intervention']
            write_json(root / 'frozen_external_plan.json', frozen)
            with self.assertRaises(ValueError):
                study.external_evaluation(root, external)

    def test_final_model_rejects_incomplete_outer_validation(self):
        c = sample_cohort(); cfg = study.default_config()
        with tempfile.TemporaryDirectory() as tmp:
            study.bind_experiment(tmp, coordinate_dataset(c, cfg['coordinate_mode']), cfg)
            write_json(Path(tmp) / 'nested_coverage.json', {'complete': False, 'completed_subjects': ['P01']})
            with self.assertRaises(ValueError):
                study.final_refit(c, cfg, tmp)

if __name__ == '__main__':
    unittest.main()
