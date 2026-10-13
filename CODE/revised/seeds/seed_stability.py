"""Frozen-setting repeated-seed LOSO sensitivity study. No hyperparameter search."""
from pathlib import Path
import hashlib,json,math,time,traceback,importlib.metadata,platform
import numpy as np,pandas as pd
from data import TASK_ORDER
from revision_data import legacy_cohort,coordinate_dataset
from protocol import cycle_metrics,scale_from_training
from evaluate_saved import check_metadata

FAMILIES=('lstm','gru','conv1d_lstm')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hashobj(o):return hashlib.sha256(json.dumps(o,sort_keys=True,allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(o,indent=2,allow_nan=False));tmp.replace(p)
def integer_seed(master,*labels):return int(hashlib.sha256((str(master)+'|'+'|'.join(map(str,labels))).encode()).hexdigest()[:8],16)%(2**31-1)
def versions():
 d={'python':platform.python_version()}
 for name in ['tensorflow','keras','keras-tuner','numpy','pandas','openpyxl']:
  try:d[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:d[name]=None
 return d

def inventory(data_zip,source_root,tasks,families,outer,code_root):
 cohorts={};jobs=[];rows=[]
 for task in tasks:
  cohort=legacy_cohort(data_zip,task);ds=coordinate_dataset(cohort,'absolute');cohorts[task]=ds
  root=Path(source_root)/task/'absolute';sp=root/'prespecification.json'
  if not sp.is_file():
   rows.append(dict(task=task,outer_subject='all',family='all',status='unavailable',reason='Source prespecification missing'));continue
  spec=read(sp)
  if spec['task']!=task or spec['data_fingerprint']!=ds.fingerprint or sorted(spec['subjects'])!=ds.subjects or spec['config'].get('smoke'):
   raise ValueError(f'{task}: source data or research profile mismatch')
  for file in ['data.py','protocol.py','models.py','revision_data.py','revision_models.py']:
   if digest(Path(code_root)/file)!=spec['sources'].get(file):raise ValueError(f'{task}: source-code mismatch for {file}')
  for sub in ds.subjects:
   if outer!='all' and sub not in outer:continue
   for family in families:
    folder=root/'nested'/sub/family;sel=folder/'selection/selected.json';inf=folder/'refit/model.json';done=folder/'complete.json'
    if not all(p.is_file() for p in [sel,inf,done]):
     rows.append(dict(task=task,outer_subject=sub,family=family,status='unavailable',reason='Completed original selection/refit/outer marker missing'));continue
    selected,info=read(sel),read(inf);check_metadata(spec,info,selected,family,sub,ds)
    expected=[s for s in ds.subjects if s!=sub];folds=selected.get('folds',[])
    if sorted(r['validation_subject'] for r in folds)!=expected:raise ValueError('Incomplete inner subject-wise selection')
    train=ds.subset(expected)
    for r in folds:
     if sorted(r['training_subjects'])!=[s for s in expected if s!=r['validation_subject']] or r['data_fingerprint']!=train.fingerprint or not np.isfinite(r['mae']):raise ValueError('Invalid inner fold data/separation')
    epoch=max(1,int(np.ceil(np.median([r['best_epoch'] for r in folds]))))
    if epoch!=selected['epochs'] or epoch!=info['epochs']:raise ValueError('Epoch budget is not the fixed inner-derived budget')
    job=dict(task=task,outer_subject=sub,family=family,parameters=selected['parameters'],epochs=epoch,batch_size=int(spec['config']['batch_size']),scale_input=bool(spec['config']['scale_input']),search_profile=spec['config']['search_profile'],source_selection_sha256=digest(sel),source_descriptor_sha256=digest(inf),source_spec_sha256=digest(sp),source_software=spec['software'],data_fingerprint=ds.fingerprint)
    rows.append(dict(task=task,outer_subject=sub,family=family,status='available',reason='',epochs=epoch));jobs.append(job)
 return cohorts,jobs,pd.DataFrame(rows)

def bind_plan(output,settings,jobs,code_root):
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 signature={'settings':settings,'jobs':jobs,'software':versions(),'code_sha256':{p.name:digest(p) for p in Path(code_root).glob('*.py')}}
 p=out/'frozen_seed_plan.json'
 if p.exists() and read(p)!=signature:raise ValueError('Data, settings, selected configurations, software or code changed. Use a NEW RUN_ID; do not mix results.')
 if not p.exists():write(p,signature)
 return hashobj(signature)

def train_job(ds,job,master_seed,output,plan_hash,verbose=0):
 import tensorflow as tf
 from models import build_model
 sub=job['outer_subject'];fam=job['family'];folder=Path(output)/'runs'/ds.task/sub/fam/f'seed_{master_seed}';folder.mkdir(parents=True,exist_ok=True)
 completed=folder/'result.json';cache=folder/'predictions.npz'
 if completed.exists():
  result=read(completed)
  if result['plan_hash']!=plan_hash or result['master_seed']!=master_seed or not cache.exists() or result['predictions_sha256']!=digest(cache):raise ValueError('Completed result/cache mismatch')
  return result
 actual_seed=integer_seed(master_seed,ds.task,sub,fam,'initialization');tr=np.flatnonzero(ds.groups!=sub);te=np.flatnonzero(ds.groups==sub)
 assert len(tr)>0 and len(te)>0 and not set(ds.groups[tr])&set(ds.groups[te])
 (xtr,xte),scaler=scale_from_training(ds.x[tr],ds.x[te],enabled=job['scale_input']);ytr=ds.y[tr]
 steps=math.ceil(len(tr)/job['batch_size']);checkpoint=folder/'checkpoint.keras';start=time.monotonic()
 tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(actual_seed)
 if checkpoint.exists():
  model=tf.keras.models.load_model(checkpoint)
  iters=int(model.optimizer.iterations.numpy())
  if iters%steps:raise ValueError('Checkpoint does not represent a complete epoch')
  epoch=iters//steps
 else:model=build_model(fam,ds.x.shape[1],job['parameters'],job['search_profile']);epoch=0
 if epoch>job['epochs']:raise ValueError('Checkpoint exceeded frozen epoch budget')
 print(f'{ds.task}/{sub}/{fam}/seed {master_seed}: {epoch}/{job["epochs"]} epochs',flush=True)
 try:
  while epoch<job['epochs']:
   e_seed=integer_seed(master_seed,ds.task,sub,fam,'epoch',epoch+1);tf.keras.utils.set_random_seed(e_seed)
   order=np.random.default_rng(e_seed).permutation(len(tr));before=int(model.optimizer.iterations.numpy())
   h=model.fit(xtr[order],ytr[order],batch_size=job['batch_size'],epochs=1,shuffle=False,verbose=verbose,callbacks=[tf.keras.callbacks.TerminateOnNaN()])
   loss=float(h.history['loss'][-1])
   if not np.isfinite(loss) or int(model.optimizer.iterations.numpy())!=before+steps:raise ValueError('Nonfinite or incomplete training epoch; last good checkpoint retained')
   # Save weights and optimizer state every epoch. Optimizer iterations are authoritative.
   tmp=folder/'checkpoint.tmp.keras';model.save(tmp);tmp.replace(checkpoint);epoch+=1
   write(folder/'training_progress.json',dict(master_seed=master_seed,actual_seed=actual_seed,completed_epochs=epoch,total_epochs=job['epochs'],last_loss=loss,plan_hash=plan_hash))
   with (folder/'history.csv').open('a') as stream:
    if stream.tell()==0:stream.write('epoch,loss,epoch_seed\n')
    stream.write(f'{epoch},{loss},{e_seed}\n')
   if epoch==1 or epoch%10==0 or epoch==job['epochs']:print(f'  epoch {epoch}/{job["epochs"]}; loss {loss:.6g}',flush=True)
  pred=np.asarray(model.predict(xte,batch_size=job['batch_size'],verbose=0));metric=cycle_metrics(ds.y[te],pred)
  if not np.isfinite(metric.rmse).all():raise ValueError('Nonfinite held-out metric')
  ids=ds.cycles.iloc[te].cycle_id.to_numpy(str);np.savez_compressed(cache,measured=ds.y[te],predicted=pred,cycle_ids=ids,subject_ids=ds.groups[te].astype(str),gait_phase=np.linspace(0,100,ds.y.shape[1]))
  metric.insert(0,'cycle_id',ids);metric.to_csv(folder/'cycle_metrics.csv',index=False)
  result={**job,'master_seed':master_seed,'actual_seed':actual_seed,'participant':sub,'n_cycles':len(te),'n_training_cycles':len(tr),'n_training_participants':len(set(ds.groups[tr])),'rmse':float(metric.rmse.mean()),'nrmse':float(metric.nrmse.mean()) if metric.nrmse.notna().any() else None,'mae':float(metric.mae.mean()),'undefined_nrmse_cycles':int((~metric.nrmse_defined).sum()),'input_scaler':scaler,'elapsed_this_session_seconds':time.monotonic()-start,'plan_hash':plan_hash,'predictions_sha256':digest(cache)}
  write(completed,result);return result
 finally:
  del model;tf.keras.backend.clear_session()

def summarize(output,jobs,seeds,cohorts):
 out=Path(output);rows=[];coverage=[]
 for job in jobs:
  complete=[]
  for seed in seeds:
   p=out/'runs'/job['task']/job['outer_subject']/job['family']/f'seed_{seed}'/'result.json'
   if p.exists():
    r=read(p);complete.append(seed);rows.append({k:r[k] for k in ['task','family','participant','master_seed','actual_seed','epochs','n_cycles','rmse','nrmse','mae','undefined_nrmse_cycles']})
  coverage.append(dict(task=job['task'],family=job['family'],participant=job['outer_subject'],completed_seeds=len(complete),expected_seeds=len(seeds),all_seeds_complete=set(complete)==set(seeds)))
 cv=pd.DataFrame(coverage);cv.to_csv(out/'coverage.csv',index=False)
 if not rows:return None
 df=pd.DataFrame(rows);df.to_csv(out/'participant_seed_metrics.csv',index=False)
 # Balanced reporting: only participants with every requested seed enter summaries.
 good=cv[cv.all_seeds_complete];balanced=df.merge(good[['task','family','participant']],on=['task','family','participant'])
 if balanced.empty:return None
 participant=[];seedrows=[];stability=[]
 for (task,family,sub),g in balanced.groupby(['task','family','participant']):
  for metric in ['rmse','nrmse','mae']:
   a=g[metric].dropna().to_numpy();participant.append(dict(task=task,family=family,participant=sub,metric=metric,n_seeds=len(a),mean=a.mean() if len(a) else np.nan,sd=a.std(ddof=1) if len(a)>1 else np.nan,minimum=a.min() if len(a) else np.nan,maximum=a.max() if len(a) else np.nan))
 pd.DataFrame(participant).to_csv(out/'participant_seed_variation.csv',index=False)
 for (task,family),g in balanced.groupby(['task','family']):
  folds=g.participant.nunique();total=len(cohorts[task].subjects)
  for seed,gg in g.groupby('master_seed'):
   seedrows.append(dict(task=task,family=family,master_seed=seed,n_participants=folds,expected_cohort_participants=total,full_cohort_coverage=folds==total,rmse=gg.rmse.mean(),nrmse=gg.nrmse.mean(),mae=gg.mae.mean()))
 sd=pd.DataFrame(seedrows);sd.to_csv(out/'cohort_metrics_per_seed.csv',index=False)
 for (task,family),g in sd.groupby(['task','family']):
  for metric in ['rmse','nrmse','mae']:
   a=g[metric].dropna().to_numpy();stability.append(dict(task=task,family=family,metric=metric,n_seeds=len(a),n_participants=int(g.n_participants.iloc[0]),expected_cohort_participants=int(g.expected_cohort_participants.iloc[0]),scope='full cohort' if g.full_cohort_coverage.all() else 'limited available folds',mean=a.mean() if len(a) else np.nan,sd=a.std(ddof=1) if len(a)>1 else np.nan,minimum=a.min() if len(a) else np.nan,maximum=a.max() if len(a) else np.nan))
 pd.DataFrame(stability).to_csv(out/'seed_stability_summary.csv',index=False);return pd.DataFrame(stability)
