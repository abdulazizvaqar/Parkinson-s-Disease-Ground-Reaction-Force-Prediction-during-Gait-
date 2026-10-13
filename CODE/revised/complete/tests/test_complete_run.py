import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from external_workbook import make_mapping_templates, readiness, prepare_external_bundles
from protocol import read_json, write_json
from revision_study import default_config
from test_revision import sample_cohort
from complete_study import inspect_plan, run_batch

class CompleteRunTests(unittest.TestCase):
    def tables(self):
        base = np.column_stack([np.linspace(10, 110, 101), np.linspace(20, 220, 101)])
        tables = {}
        for side in ['L', 'R']:
            for force in ['AP', 'VERT', 'ML']:
                tables[f'{side}-{force}'] = pd.DataFrame(-base * 10)
            for axis in ['X', 'Y', 'Z']:
                tables[f'{side}ANK{axis}'] = pd.DataFrame(base)
        return tables

    def test_external_unknown_identity_and_units_block_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); book = root / 'fixture.xlsx'; book.write_bytes(b'fixture')
            with patch('external_workbook.pd.read_excel', return_value=self.tables()):
                make_mapping_templates(book, root / 'metadata')
            why = readiness(book, root / 'metadata')
            self.assertTrue(any('participant_id' in s for s in why))
            self.assertTrue(any('force units' in s for s in why))
            with self.assertRaises(ValueError):
                prepare_external_bundles(book, root / 'metadata', root / 'bundles')

    def test_direct_import_uses_first_sample_and_physical_mass_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); book = root / 'fixture.xlsx'; book.write_bytes(b'fixture')
            meta = root / 'metadata'
            with patch('external_workbook.pd.read_excel', return_value=self.tables()):
                make_mapping_templates(book, meta)
            df = pd.read_csv(meta / 'external_column_map.csv', keep_default_na=False)
            df['participant_id'] = ['TEST01', 'TEST02', 'TEST01', 'TEST02']
            df['trial_id'] = 'trial1'; df['visit'] = 'baseline'; df['include'] = True
            df['body_mass_kg'] = 50
            df.to_csv(meta / 'external_column_map.csv', index=False)
            settings = read_json(meta / 'external_import_settings.json')
            for k in ['phase_grid_0_to_100_confirmed', 'column_pairing_verified',
                      'internal_external_axes_and_signs_verified', 'participant_disjointness_verified',
                      'visit_mapping_verified', 'identity_mapping_verified']:
                settings[k] = True
            settings.update(ankle_units='mm', force_units='N', ankle_axis={'AP':'Y','Vertical':'Z','ML':'X'},
                ankle_sign={'AP':1,'Vertical':1,'ML':1}, force_sign={'AP':-1,'Vertical':-1,'ML':-1},
                coordinate_definition='TEST fixture', cycle_definition='TEST full cycle', harmonization_description='TEST mapping')
            write_json(meta / 'external_import_settings.json', settings)
            with patch('external_workbook.pd.read_excel', return_value=self.tables()):
                result = prepare_external_bundles(book, meta, root / 'bundles')
            self.assertEqual(len(result), 6)
            with np.load(Path(result['AP_Left']) / 'waveforms.npz') as f:
                self.assertEqual(f['x'].shape, (2,100,1))
                self.assertAlmostEqual(float(f['x'][0,0,0]), 10)
                self.assertAlmostEqual(float(f['x'][0,-1,0]), 110)
                self.assertAlmostEqual(float(f['y'][0,0,0]), 2)

    def test_batch_runs_all_tasks_and_keeps_missing_stages_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = default_config(); cfg['run_perturbations'] = True
            plan = dict(tasks=['AP_Left','AP_Right','Vertical_Left','Vertical_Right','ML_Left','ML_Right'],
                coordinate_modes=['absolute','translation','pelvis'], primary_coordinates='absolute',
                internal_source='original_zip', training=cfg, output_root=str(Path(tmp)/'study'),
                external_bundle_root=str(Path(tmp)/'external'), external_workbook='', external_metadata_dir='')
            calls = []
            def nested(cohort,cfg,root): calls.append(('nested',str(root)))
            def final(cohort,cfg,root): calls.append(('final',str(root)))
            # Scalar fixture can translate but cannot support pelvis or yaw.
            with patch('complete_study.load_internal', return_value=sample_cohort()), \
                 patch('complete_study.run_nested', side_effect=nested), \
                 patch('complete_study.final_refit', side_effect=final), \
                 patch('complete_study.compare_coordinates'), \
                 patch('models.configure_tensorflow', return_value={'test':'no real neural training'}):
                run_batch(plan)
            self.assertEqual(sum(t=='nested' for t,_ in calls), 12)
            self.assertEqual(sum(t=='final' for t,_ in calls), 6)
            progress = read_json(Path(tmp)/'study'/'progress.json')
            self.assertEqual(sum(v['stage']=='external' and v['status']=='blocked' for v in progress.values()), 6)
            self.assertFalse(read_json(Path(tmp)/'study'/'completion_status.json')['all_requested_stages_complete'])

if __name__ == '__main__':
    unittest.main()
