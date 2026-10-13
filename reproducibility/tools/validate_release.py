"""Validate code package and saved metadata without GPU training or raw data."""
from pathlib import Path
import ast,json,hashlib,subprocess,sys
import pandas as pd,numpy as np
root=Path(__file__).resolve().parents[1]
issues=[];count=0
for p in root.rglob('*.py'):
 if '__pycache__' in p.parts:continue
 try:ast.parse(p.read_text(encoding='utf-8-sig'));count+=1
 except Exception as e:issues.append((str(p.relative_to(root)),str(e)))
assert not issues,issues
for p in (root/'notebooks').glob('*.ipynb'):
 n=json.loads(p.read_text());assert n['nbformat']==4
 for c in n['cells']:
  if c['cell_type']!='code':continue
  assert not c.get('outputs') and c.get('execution_count') is None
  s=c['source'];s=''.join(s) if isinstance(s,list) else s
  if s.lstrip().startswith('%%'):continue
  s='\n'.join('' if l.lstrip().startswith(('%','!')) else l for l in s.splitlines())
  ast.parse(s)
for r in json.loads((root/'metadata/source_manifest.json').read_text()):
 p=root/r['repository_path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['repository_path']
cycles=pd.read_csv(root/'metadata/internal_cycle_identifiers.csv');assert len(cycles)==1638 and not cycles.duplicated(['task','cycle_id']).any()
outer=pd.read_csv(root/'metadata/revised_outer_splits.csv');inner=pd.read_csv(root/'metadata/revised_inner_splits.csv');assert len(outer)==117 and len(inner)==2166
for _,r in outer.iterrows():
 train=set(json.loads(r.development_participants_json));people=set(cycles[cycles.task==r.task].participant_id)
 assert train==people-{r.outer_test_participant}
for _,r in inner.iterrows():
 tr=set(json.loads(r.inner_training_participants_json));people=set(cycles[cycles.task==r.task].participant_id)
 assert tr==people-{r.outer_test_participant,r.inner_validation_participant} and r.outer_test_participant!=r.inner_validation_participant
f=json.loads((root/'results/fdr/summary.json').read_text());assert f['Pearson']['bh_q_lt_0_05']==11 and f['Spearman']['bh_q_lt_0_05']==0 and f['gait']['bh_q_lt_0_05']==3
for kind,n in [('Clinical_Pearson_FDR.csv',396),('Clinical_Spearman_FDR.csv',396),('Gait_Kruskal_Wallis_FDR.csv',19)]:
 d=pd.read_csv(root/'results/fdr'/kind);assert len(d)==n and d.q_bh.between(0,1).all()
sel=json.loads((root/'metadata/available_selected_configurations.json').read_text());assert len(sel)==2 and {r['family'] for r in sel}=={'lstm','gru'}
s=pd.read_csv(root/'results/seeds/seed_stability_summary.csv');assert set(s.n_seeds)=={5} and set(s.n_participants)=={1} and set(s.scope)=={'limited available folds'}
print(f'PASS: {count} Python files; clean notebooks; source hashes; 1638 cycle IDs; 117 outer/2166 inner planned splits; FDR and seed metadata.')
print('No GPU training or historical 3600-run reconstruction was performed by this check.')
