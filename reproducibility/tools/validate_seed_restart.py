from pathlib import Path
import sys,json,ast,tempfile,types,pickle,math
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'revised/seeds'),str(root/'revised/coordinates')]
import seed_stability as s
from data import Dataset
from revision_data import legacy_cohort,coordinate_dataset
import pandas as pd
nb=json.loads((root/'notebooks/Repeated_Seed_Training_GRF_Colab.ipynb').read_text());assert nb['nbformat']==4
for c in nb['cells']:
 if c['cell_type']=='code':ast.parse(c['source'])
for p in (root/'revised/coordinates').glob('*.py'):ast.parse(p.read_text())
print('Notebook syntax passed; raw-input loader requires separately supplied data.')
class Counter:
 def __init__(self,n=0):self.n=n
 def numpy(self):return self.n
class FakeModel:
 def __init__(self):self.optimizer=types.SimpleNamespace(iterations=Counter());self.w=np.random.normal()
 def fit(self,x,y,batch_size,**kwargs):
  for start in range(0,len(x),batch_size):
   xx=x[start:start+batch_size];yy=y[start:start+batch_size];self.w-=.001*float(np.mean((self.w*xx-yy)*xx));self.optimizer.iterations.n+=1
  return types.SimpleNamespace(history={'loss':[float(np.mean((self.w*x-y)**2))]})
 def predict(self,x,**kwargs):return self.w*x
 def save(self,p):
  with open(p,'wb') as f:pickle.dump(self,f)
def load(p):
 with open(p,'rb') as f:return pickle.load(f)
fake=types.SimpleNamespace(keras=types.SimpleNamespace(backend=types.SimpleNamespace(clear_session=lambda:None),utils=types.SimpleNamespace(set_random_seed=np.random.seed),models=types.SimpleNamespace(load_model=load),callbacks=types.SimpleNamespace(TerminateOnNaN=lambda:None)))
sys.modules['tensorflow']=fake;sys.modules['models']=types.SimpleNamespace(build_model=lambda *args:FakeModel())
x=np.linspace(.1,1.2,800,dtype=np.float32).reshape(8,100,1);groups=np.repeat(['SUB01','SUB02','SUB03','SUB04'],2);meta=pd.DataFrame({'cycle_id':[f'C{i}' for i in range(8)]})
ds=Dataset('AP_Left',x,.7*x,groups,meta,'fixture');job=dict(task='AP_Left',outer_subject='SUB01',family='lstm',parameters={},epochs=4,batch_size=2,scale_input=False,search_profile='original')
with tempfile.TemporaryDirectory() as tmp:
 out1=Path(tmp)/'full';out2=Path(tmp)/'resume'
 full=s.train_job(ds,job,11,out1,'signature');realwrite=s.write;interrupted=[False]
 def failonce(p,o):
  if Path(p).name=='training_progress.json' and not interrupted[0]:interrupted[0]=True;raise RuntimeError('Simulated process stop immediately after epoch checkpoint')
  realwrite(p,o)
 s.write=failonce
 try:s.train_job(ds,job,11,out2,'signature')
 except RuntimeError:pass
 finally:s.write=realwrite
 resumed=s.train_job(ds,job,11,out2,'signature')
 assert full['rmse']==resumed['rmse'] and full['predictions_sha256']==resumed['predictions_sha256']
 # Resume completed run; corrupt predictions must not be accepted.
 assert s.train_job(ds,job,11,out2,'signature')['rmse']==full['rmse']
 try:s.train_job(ds,job,11,out2,'changed');raise AssertionError('Changed plan was accepted')
 except ValueError:pass
 summary=s.summarize(out1,[job],[11,22,33],{'AP_Left':ds});assert summary is None
 for seed in [22,33]:s.train_job(ds,job,seed,out1,'signature')
 summary=s.summarize(out1,[job],[11,22,33],{'AP_Left':ds});assert set(summary.scope)=={'limited available folds'} and set(summary.n_participants)=={1}
 assert summary[summary.metric=='rmse'].sd.iloc[0]>0
 (out2/'runs/AP_Left/SUB01/lstm/seed_11/predictions.npz').write_bytes(b'corrupt')
 try:s.train_job(ds,job,11,out2,'signature');raise AssertionError('Corrupt result was accepted')
 except ValueError:pass
print('Checkpoint interruption/resume, changed-plan guard, corrupted-cache guard, balanced seeds and limited-coverage reporting passed with a simulated backend.')
print('Actual TensorFlow/GPU training was not executed locally.')
