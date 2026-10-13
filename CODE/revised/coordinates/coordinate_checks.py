"""Coordinate ablations and held-out nuisance tests, isolated from the original run."""
from dataclasses import replace
from pathlib import Path
import hashlib
import importlib.metadata
import json
import traceback
import numpy as np
import pandas as pd
from protocol import write_json, read_json, cycle_metrics
from revision_data import coordinate_dataset, gait_basis, AXIS
from evaluate_saved import check_metadata, read_prediction_cache, predict_weights, digest

MODES = ('absolute','translation','pelvis','aligned','body','combined')
FAMILIES = ('mean_waveform','ridge','phase_only','shallow_mlp','lstm','gru','conv1d_lstm')
TRAINING_KEYS = ('seed','search_profile','max_trials','epochs','patience','batch_size','verbose','scale_input','smoke')

def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()

def versions():
    out={}
    for name in ['numpy','pandas','scikit-learn','tensorflow','keras','keras-tuner','joblib']:
        try: out[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: out[name]=None
    return out

def require_versions(reference, family):
    names=['numpy','pandas']+(['tensorflow','keras','keras-tuner'] if family in FAMILIES[-3:] else ['scikit-learn','joblib'])
    for name in names:
        if not reference.get(name) or versions().get(name)!=reference[name]:
            raise ValueError(f'{name}: use training version {reference.get(name)}; found {versions().get(name)}. Run the setup cell and restart if requested.')

def verify_folds(selected, train):
    folds=selected.get('folds',[])
    names=[r['validation_subject'] for r in folds]
    if sorted(names)!=train.subjects or len(names)!=len(set(names)):
        raise ValueError('Selection does not contain one complete inner fold for every training participant')
    for row in folds:
        expected=set(train.subjects)-{row['validation_subject']}
        if set(row['training_subjects'])!=expected or len(row['training_subjects'])!=len(expected):
            raise ValueError('Invalid or overlapping inner training/validation participants')
        if row['data_fingerprint']!=train.fingerprint or not np.isfinite(row['mae']):
            raise ValueError('Invalid inner fold fingerprint or score')

def original_model(source, ds, outer, family):
    source=Path(source)
    spec=read_json(source/'prespecification.json')
    folder=source/'nested'/outer/family
    info=read_json(folder/'refit/model.json')
    selected=read_json(folder/'selection/selected.json')
    check_metadata(spec,info,selected,family,outer,ds)
    verify_folds(selected,ds.subset([s for s in ds.subjects if s!=outer]))
    return folder,info,selected,spec

def verify_original_source(source, cohort, code_dir):
    spec=read_json(Path(source)/'prespecification.json')
    ds=coordinate_dataset(cohort,'absolute')
    if spec['data_fingerprint']!=ds.fingerprint or spec['task']!=ds.task:
        raise ValueError('Original data fingerprint/task differs from the saved experiment')
    if sorted(spec['subjects'])!=ds.subjects or spec['config'].get('smoke'):
        raise ValueError('Saved participant list or research/smoke setting is invalid')
    for name in ['data.py','protocol.py','models.py','revision_data.py','revision_models.py']:
        if digest(Path(code_dir)/name)!=spec['sources'].get(name):
            raise ValueError(f'Original source mismatch: {name}')
    return spec

def make_templates(cohort, folder):
    """Create missing templates, never overwrite user-supplied metadata."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    paths={}
    paths['cycle_id_reference.csv']=cohort.ds.cycles[['cycle_id','participant_id']]
    paths['participant_body_scale.csv']=pd.DataFrame({'participant_id':cohort.ds.subjects,
        'body_scale_m':['']*len(cohort.ds.subjects)})
    for name,frame in paths.items():
        if not (folder/name).exists():frame.to_csv(folder/name,index=False)
    meta=dict(task=cohort.ds.task,input_length_unit='unverified',units_verified=False,
        canonical_axis_order=['AP','ML','Vertical'],frame_and_axes_verified=False,
        cycle_alignment_verified=False,targets_already_in_gait_frame_verified=False,
        body_scale_definition='',body_scale_source='',direction_source='',
        direction_independent_of_grf_verified=False,
        notes='Do not change verification flags unless supported by the source data/protocol.')
    if not (folder/'coordinate_metadata.json').exists():write_json(folder/'coordinate_metadata.json',meta)
    text='''OPTIONAL ANATOMICAL INPUTS - DO NOT INVENT MISSING ARRAYS

All cycles must match cycle_id_reference.csv exactly by ID. The loader reorders by ID.
Create coordinate_inputs.npz (numpy.savez_compressed) with cycle_ids: Unicode [N].
Optional real, synchronized arrays:
  ankle_xyz: [N,100,3], in original laboratory length units and AP,ML,Vertical order.
  pelvis_xyz: [N,100,3], same time points, frame, units and origin as ankles.
  direction_xyz: [N,3], nonzero horizontal forward vectors, derived without GRF.
Do not join separate component files by column number without verifying cycle pairing.
The ankle component is checked against the original internal ZIP before use.
Pelvis-only arrays can support component-wise pelvis centering without ankle_xyz.

Body normalization can run on scalar ankle trajectories: complete participant_body_scale.csv
with real height OR leg length in metres and one prespecified definition for all participants.
Set its source/definition and verified original input unit (m/mm) in coordinate_metadata.json.
Do not fill missing anthropometry with cohort means or estimates.

For alignment/rotation tests, original scalar GRF targets must already represent gait-frame
components; verify this in coordinate_metadata.json. If they are laboratory-frame components,
this companion cannot reuse the original scalar targets: rebuild a consistently harmonized
3D kinematics/force cohort before that analysis. This notebook does not rotate targets by guess.

Translation SD tests require no assumed physical unit: offsets use only outer-training inputs.
External data must not be used to populate or select settings for this internal analysis.
'''
    if not (folder/'INPUT_GUIDE.txt').exists():(folder/'INPUT_GUIDE.txt').write_text(text)

def seed_known_heights(folder, heights_m, provenance, confirmed_units='unverified', unit_evidence=''):
    """Use supplied, explicitly labelled clinical heights; never replace user-entered scales."""
    folder=Path(folder);meta=read_json(folder/'coordinate_metadata.json')
    body=pd.read_csv(folder/'participant_body_scale.csv',dtype=str,keep_default_na=False)
    if body.body_scale_m.str.strip().eq('').all() and not meta.get('body_scale_definition') and not meta.get('body_scale_source'):
        if set(body.participant_id).issubset(heights_m):
            values=body.participant_id.map(heights_m).astype(float)
            if not np.isfinite(values).all() or (values<=0).any():raise ValueError('Invalid supplied clinical height')
            body['body_scale_m']=values
            body.to_csv(folder/'participant_body_scale.csv',index=False)
            meta['body_scale_definition']='Participant height: explicitly labelled Height (cm), converted to metres'
            meta['body_scale_source']=provenance
    if confirmed_units not in ('unverified','m','mm'):raise ValueError('Use unverified, m or mm for CONFIRMED_ANKLE_UNITS')
    if confirmed_units!='unverified':
        if not str(unit_evidence).strip():raise ValueError('Document ANKLE_UNITS_EVIDENCE before confirming physical units')
        if meta.get('units_verified') and meta.get('input_length_unit')!=confirmed_units:
            raise ValueError('Confirmed units conflict with previously supplied coordinate metadata')
        meta.update(input_length_unit=confirmed_units,units_verified=True,unit_evidence=str(unit_evidence))
    write_json(folder/'coordinate_metadata.json',meta)

class CoordinateInputs:
    def __init__(self, cohort, folder):
        self.cohort=cohort; self.ds=cohort.ds; self.folder=Path(folder)
        self.meta=read_json(self.folder/'coordinate_metadata.json');self.arrays={}
        if self.meta.get('task')!=self.ds.task:raise ValueError('Coordinate metadata task differs')
        self.body=None
        files=['coordinate_metadata.json']
        path=self.folder/'coordinate_inputs.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as a:
                raw={k:a[k] for k in a.files}
            ids=raw.pop('cycle_ids').astype(str)
            expected=self.ds.cycles.cycle_id.to_numpy(str)
            if len(set(ids))!=len(ids) or set(ids)!=set(expected):
                raise ValueError('Anatomical NPZ must contain exactly all original cycle IDs')
            order=pd.Index(ids).get_indexer(expected)
            for key,arr in raw.items():
                if key not in ('ankle_xyz','pelvis_xyz','direction_xyz'):raise ValueError(f'Unknown NPZ field: {key}')
                shape=(len(ids),3) if key=='direction_xyz' else (len(ids),100,3)
                if arr.shape!=shape or not np.issubdtype(arr.dtype,np.number) or not np.isfinite(arr).all():
                    raise ValueError(f'Invalid {key}: expected finite {shape}')
                self.arrays[key]=arr[order].astype(float)
            files.append(path.name)
        path=self.folder/'participant_body_scale.csv'
        if path.exists():
            body=pd.read_csv(path,dtype={'participant_id':str},keep_default_na=False)
            if set(['participant_id','body_scale_m'])-set(body):raise ValueError('Body-scale CSV columns missing')
            if body.participant_id.duplicated().any() or set(body.participant_id)!=set(self.ds.subjects):
                raise ValueError('Body-scale participant IDs must match this task exactly')
            vals=pd.to_numeric(body.body_scale_m,errors='coerce')
            if np.isfinite(vals).all() and (vals>0).all():
                mapping=dict(zip(body.participant_id,vals));self.body=np.array([mapping[g] for g in self.ds.groups])
            files.append(path.name)
        self.fingerprint=canonical_hash({name:digest(self.folder/name) for name in files})

    def check_frame(self):
        if self.meta.get('canonical_axis_order')!=['AP','ML','Vertical'] or not self.meta.get('frame_and_axes_verified') or not self.meta.get('cycle_alignment_verified'):
            raise ValueError('Verify anatomical axes, common frame and cycle/time alignment in coordinate_metadata.json')
        if 'ankle_xyz' in self.arrays:
            axis=AXIS[self.ds.task.split('_')[0]]
            if not np.allclose(self.arrays['ankle_xyz'][:,:,axis:axis+1],self.ds.x,rtol=1e-5,atol=1e-5):
                raise ValueError('Anatomical ankle component differs from original inputs: frame, units or cycle pairing mismatch')

    def require(self,mode,rotation=False):
        if mode not in MODES:raise ValueError('Unknown coordinate mode')
        if mode in ('pelvis','aligned','combined') or rotation:
            self.check_frame()
        if mode in ('pelvis','combined') and 'pelvis_xyz' not in self.arrays:
            raise ValueError('Synchronized pelvis_xyz is missing')
        if mode in ('aligned','combined') or rotation:
            if 'ankle_xyz' not in self.arrays or 'direction_xyz' not in self.arrays:
                raise ValueError('Verified ankle_xyz and force-independent direction_xyz are required')
            if not self.meta.get('direction_independent_of_grf_verified') or not self.meta.get('direction_source'):
                raise ValueError('Document and verify the force-independent walking-direction source')
            gait_basis(self.arrays['direction_xyz'])
            if not self.meta.get('targets_already_in_gait_frame_verified'):
                raise ValueError('Verify that unchanged GRF targets are already gait-frame components before alignment or rotation tests')
        if mode in ('body','combined'):
            if self.body is None:raise ValueError('Complete verified participant-linked body_scale_m values')
            if not self.meta.get('units_verified') or self.meta.get('input_length_unit') not in ('m','mm'):
                raise ValueError('Verify original input length units (m/mm) before body normalization')
            if not self.meta.get('body_scale_definition') or not self.meta.get('body_scale_source'):
                raise ValueError('Prespecify and document a common height or leg-length definition')

    def dataset(self,mode,offset=None,yaw=0.):
        self.require(mode,rotation=bool(yaw))
        if mode=='absolute' and offset is None and yaw==0:
            return coordinate_dataset(self.cohort,'absolute')
        axis=AXIS[self.ds.task.split('_')[0]]
        xyz=self.arrays.get('ankle_xyz');xyz=None if xyz is None else xyz.copy()
        pelvis=self.arrays.get('pelvis_xyz');pelvis=None if pelvis is None else pelvis.copy()
        direction=self.arrays.get('direction_xyz');direction=None if direction is None else direction.copy()
        # Unperturbed scalar features retain the exact original values.
        x=self.ds.x.astype(float).copy()
        if yaw:
            theta=np.deg2rad(yaw);c,s=np.cos(theta),np.sin(theta)
            rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
            xyz=xyz@rotation.T;direction=direction@rotation.T
            if pelvis is not None:pelvis=pelvis@rotation.T
            x=xyz[:,:,axis:axis+1]
        if offset is not None:
            offset=np.asarray(offset,float)
            if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('Invalid translation vector')
            x=x+offset[axis]
            if xyz is not None:xyz=xyz+offset
            if pelvis is not None:pelvis=pelvis+offset
        if mode in ('translation','body'):
            x=x-x[:,:1,:]
        elif mode=='pelvis':x=x-pelvis[:,:,axis:axis+1]
        elif mode in ('aligned','combined'):
            centered=xyz-pelvis if mode=='combined' else xyz-xyz[:,:1,:]
            x=np.einsum('nti,nij->ntj',centered,gait_basis(direction))[:,:,axis:axis+1]
        if mode in ('body','combined'):
            factor=1000. if self.meta['input_length_unit']=='mm' else 1.
            x=x/(self.body[:,None,None]*factor)
        fp=canonical_hash({'original':self.ds.fingerprint,'supplement':self.fingerprint,'mode':mode,
                           'offset':None if offset is None else offset.tolist(),'yaw':float(yaw)})
        return replace(self.ds,x=x.astype('float32'),fingerprint=fp)

def metric_tables(ds,pred,outer,mode,family,scenario='unperturbed',reference=None):
    t=cycle_metrics(ds.y,pred)
    a=ds.y.reshape(len(ds.y),-1).astype(float);b=pred.reshape(len(pred),-1).astype(float)
    a=a-a.mean(axis=1,keepdims=True);b=b-b.mean(axis=1,keepdims=True)
    d=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    t['waveform_r']=np.divide(np.sum(a*b,axis=1),d,out=np.full(len(a),np.nan),where=d>1e-12)
    t['nrmse_percent']=100*t.nrmse
    for key,value in dict(task=ds.task,participant_id=outer,coordinate=mode,model=family,scenario=scenario).items():t[key]=value
    t['cycle_id']=ds.cycles.cycle_id.to_numpy(str)
    if reference is not None:
        ref=cycle_metrics(ds.y,reference)
        t['rmse_change']=t.rmse-ref.rmse
        t['prediction_change_rmse']=np.sqrt(np.mean((pred.astype(float)-reference.astype(float))**2,axis=(1,2)))
    keys=['task','participant_id','coordinate','model','scenario']
    measures=['rmse','mae','nrmse_percent','waveform_r']+(['rmse_change','prediction_change_rmse'] if reference is not None else [])
    summary=t.groupby(keys,dropna=False)[measures].mean().reset_index()
    summary['n_cycles']=len(t);summary['undefined_nrmse_cycles']=int(t.nrmse.isna().sum())
    return t,summary

def bind(folder,signature):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'case_manifest.json'
    if path.exists() and read_json(path)!=signature:
        raise ValueError('Case settings/data/code/software changed; choose a new RUN_ID. Existing results were not overwritten.')
    if not path.exists():write_json(path,signature)

def case_key(task,outer,mode,family,stage,scenario='unperturbed'):
    return '/'.join([task,outer,mode,family,stage,scenario])

def record(root,key,status,reason=''):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/'status.json';items=read_json(path) if path.exists() else {}
    items[key]={'case':key,'status':status,'reason':str(reason)}
    write_json(path,items);pd.DataFrame(items.values()).to_csv(root/'status.csv',index=False)
    print(status.upper(),key, str(reason)[:240],flush=True)

def current_sources():
    folder=Path(__file__).parent
    return {p.name:digest(p) for p in folder.glob('*.py')}

def execute_case(inputs,source,root,outer,mode,family,strategy,cfg):
    """Return saved model location, validated prediction, case folder and software."""
    from coordinate_models import tune_family,fit_frozen_model,predict_saved
    ds=inputs.dataset(mode);te=ds.subset([outer]);train=ds.subset([s for s in ds.subjects if s!=outer])
    if len(train.subjects)<2:raise ValueError('Need at least three participants for nested LOSO')
    case=Path(root)/'cases'/ds.task/outer/mode/family
    absds=inputs.dataset('absolute')
    original=None
    if mode=='absolute' or strategy=='frozen_selection':
        try:original=original_model(source,absds,outer,family)
        except FileNotFoundError:
            if strategy=='frozen_selection':raise ValueError('No completed original inner selection/refit for this model and outer participant; use nested_reselect to train it')
    signature={'dataset':ds.fingerprint,'outer':outer,'mode':mode,'family':family,'strategy':strategy,
               'configuration':{k:cfg[k] for k in TRAINING_KEYS},'sources':current_sources(),'software':versions()}
    if original:
        folder,info,selection,spec=original
        signature['source_model_metadata']=digest(folder/'refit/model.json')
        signature['source_selection']=digest(folder/'selection/selected.json')
        signature['source_prespecification']=digest(Path(source)/'prespecification.json')
        if (folder/'predictions.npz').exists():signature['source_predictions']=digest(folder/'predictions.npz')
    bind(case,signature)
    if mode=='absolute' and original:
        folder,info,selection,spec=original
        marker=read_json(folder/'complete.json')
        if marker.get('outer_subject')!=outer or marker.get('family')!=family:raise ValueError('Original completion marker mismatch')
        pred=read_prediction_cache(folder/'predictions.npz',te)
        model_path=folder/'refit';software=spec['software'];origin='original completed outer test'
    else:
        model_path=case/'refit';software=versions()
        if (case/'complete.json').exists():
            if read_json(case/'complete.json')['prediction_sha256']!=digest(case/'predictions.npz'):
                raise ValueError('Saved coordinate predictions changed after completion')
            pred=read_prediction_cache(case/'predictions.npz',te)
            info=read_json(model_path/'model.json')
            chosen=read_json(case/'selection/selected.json')
        else:
            if strategy=='frozen_selection':
                folder,info,selection,spec=original
                require_versions(spec['software'],family)
                chosen={'family':family,'parameters':selection['parameters'],'epochs':selection['epochs'],
                        'selection_strategy':'fixed from original absolute-coordinate inner LOSO',
                        'source_absolute_inner_subject_mae':selection['inner_subject_mae'],
                        'source_selection_sha256':digest(folder/'selection/selected.json')}
                write_json(case/'selection/selected.json',chosen)
            else:
                chosen=tune_family(train,family,cfg,case/'selection')
                verify_folds(chosen,train)
            info=fit_frozen_model(train,chosen,cfg,model_path)
            pred=predict_saved(model_path,te.x)
        if info['training_participants']!=train.subjects or outer in info['training_participants'] or info['data_fingerprint']!=train.fingerprint:
            raise ValueError('Refitted model training participants/data mismatch')
        if info['parameters']!=chosen['parameters'] or info['epochs']!=chosen['epochs']:
            raise ValueError('Refitted model differs from frozen selection')
        origin=strategy
    cycles,summary=metric_tables(te,pred,outer,mode,family)
    cycles.to_csv(case/'cycle_metrics.csv',index=False);summary['selection_strategy']=origin
    summary.to_csv(case/'participant_metrics.csv',index=False)
    np.savez_compressed(case/'predictions.npz',cycle_ids=te.cycles.cycle_id.to_numpy(str),measured=te.y,predicted=pred)
    write_json(case/'complete.json',{'model':family,'outer':outer,'coordinate':mode,'strategy':origin,
        'prediction_sha256':digest(case/'predictions.npz'),'model_directory':str(model_path)})
    return model_path,info,pred,case,software

def perturb_case(inputs,root,outer,mode,family,model_path,info,pred,case,software,options):
    """Shift/rotate held-out inputs only; no fitting, no test-driven choice of perturbations."""
    ds=inputs.dataset(mode);te=ds.subset([outer]);trmask=inputs.ds.groups!=outer
    axis=AXIS[ds.task.split('_')[0]]
    scenarios=[]
    if options['translation']:
        sd=float(np.std(inputs.ds.x[trmask].astype(float)))
        if not np.isfinite(sd) or sd<=1e-12:raise ValueError('Outer-training input SD is zero; no defined SD perturbation')
        for multiple in options['translation_sd_multipliers']:
            vector=np.zeros(3);vector[axis]=float(multiple)*sd
            scenarios.append((f'translation_sd_{multiple:+g}',vector,0.,{'translation_sd':float(multiple),'source_unit_offset':float(vector[axis])}))
    if options['rotation']:
        try:inputs.require(mode,rotation=True)
        except ValueError as e:
            record(root,case_key(ds.task,outer,mode,family,'perturbation','rotation'),'blocked',str(e))
        else:
            for angle in options['yaw_degrees']:
                scenarios.append((f'yaw_{angle:+g}',None,float(angle),{'yaw_degrees':float(angle)}))
    if not scenarios:return
    model_file=Path(model_path)/info['file']
    context={'model_sha256':digest(model_file),'original_case':digest(case/'case_manifest.json'),
             'metadata':inputs.fingerprint,'options':options,'software':versions()}
    fresh=None
    for name,offset,yaw,fields in scenarios:
        key=case_key(ds.task,outer,mode,family,'perturbation',name)
        folder=case/'perturbations'/name
        try:
            transformed=inputs.dataset(mode,offset=offset,yaw=yaw).subset([outer])
            bind(folder,{**context,'scenario':name,'fields':fields})
            if (folder/'complete.json').exists():
                if read_json(folder/'complete.json')['prediction_sha256']!=digest(folder/'predictions.npz'):
                    raise ValueError('Saved perturbation predictions changed after completion')
                shifted=read_prediction_cache(folder/'predictions.npz',transformed)
            else:
                if fresh is None:
                    fresh=predict_weights(model_path,info,te.x,software)
                    if not np.allclose(fresh,pred,rtol=1e-4,atol=1e-5):
                        raise ValueError('Saved weights do not reproduce the unperturbed prediction cache')
                shifted=predict_weights(model_path,info,transformed.x,software)
            cycle,summary=metric_tables(transformed,shifted,outer,mode,family,name,reference=pred)
            for col,val in fields.items():summary[col]=val;cycle[col]=val
            summary['translation_definition']='Constant shift along the original task axis, in multiples of pooled outer-training ankle-input SD'
            cycle.to_csv(folder/'cycle_metrics.csv',index=False);summary.to_csv(folder/'participant_metrics.csv',index=False)
            np.savez_compressed(folder/'predictions.npz',cycle_ids=transformed.cycles.cycle_id.to_numpy(str),measured=transformed.y,predicted=shifted)
            write_json(folder/'complete.json',{'scenario':name,'prediction_sha256':digest(folder/'predictions.npz')})
            record(root,key,'completed')
        except Exception as e:
            record(root,key,'failed',str(e));(case/'last_perturbation_error.txt').write_text(traceback.format_exc())

def run_task(cohort,source,root,supplement,outer_subjects,modes,families,strategy,options,cfg):
    if strategy not in ('frozen_selection','nested_reselect'):raise ValueError('Unknown selection strategy')
    root=Path(root);source=Path(source)
    if root.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(root.resolve()):
        raise ValueError('New output and original training directories must be separate')
    inputs=CoordinateInputs(cohort,supplement)
    outers=cohort.ds.subjects if outer_subjects=='all' else list(outer_subjects)
    if set(outers)-set(cohort.ds.subjects) or len(outers)!=len(set(outers)):raise ValueError('Unknown/duplicate outer participant')
    if set(modes)-set(MODES) or set(families)-set(FAMILIES):raise ValueError('Unknown model or coordinate mode')
    bind(root,{'strategy':strategy,'coordinate_modes':modes,'families':families,'perturbations':options,
               'configuration':{k:cfg[k] for k in TRAINING_KEYS},'software':versions(),'sources':current_sources()})
    bind(root/'input_manifests'/cohort.ds.task,{'dataset':cohort.ds.fingerprint,'supplement':inputs.fingerprint})
    write_json(root/f'{cohort.ds.task}_requested_scope.json',dict(task=cohort.ds.task,outer_participants=outers,
        coordinate_modes=modes,families=families,selection_strategy=strategy,
        note='Default single-outer sensitivity run is not a complete cohort nested-LOSO study.'))
    # Do all saved absolute-model tests FIRST, so a long normalized refit cannot hide these outputs.
    for mode in modes:
        try:inputs.require(mode)
        except ValueError as e:
            record(root,case_key(cohort.ds.task,'all_requested',mode,'all_requested','coordinate'),'blocked',str(e));continue
        for outer in outers:
            for family in families:
                key=case_key(cohort.ds.task,outer,mode,family,'coordinate')
                record(root,key,'running')
                try:
                    model_path,info,pred,case,software=execute_case(inputs,source,root,outer,mode,family,strategy,cfg)
                    record(root,key,'completed')
                except Exception as e:
                    record(root,key,'failed_or_unavailable',str(e));continue
                try:perturb_case(inputs,root,outer,mode,family,model_path,info,pred,case,software,options)
                except Exception as e:record(root,case_key(cohort.ds.task,outer,mode,family,'perturbation'),'failed',str(e))
                summarize(root)
    return summarize(root)

def summarize(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rows=[pd.read_csv(p) for p in (root/'cases').glob('*/*/*/*/participant_metrics.csv')]
    metrics=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    if len(metrics):
        metrics.to_csv(root/'coordinate_participant_metrics.csv',index=False)
        baseline=metrics.loc[metrics.coordinate=='absolute',['task','participant_id','model','rmse','nrmse_percent']]
        paired=metrics.loc[metrics.coordinate!='absolute'].merge(baseline,on=['task','participant_id','model'],suffixes=('','_absolute'),validate='many_to_one')
        paired['rmse_improvement']=paired.rmse_absolute-paired.rmse
        paired['nrmse_percentage_point_improvement']=paired.nrmse_percent_absolute-paired.nrmse_percent
        paired.to_csv(root/'paired_coordinate_comparisons.csv',index=False)
        counts=metrics.groupby(['task','coordinate','model']).agg(n_participants=('participant_id','nunique'),n_cycles=('n_cycles','sum'),mean_participant_rmse=('rmse','mean'),mean_participant_nrmse_percent=('nrmse_percent','mean')).reset_index()
        counts.to_csv(root/'coordinate_summary.csv',index=False)
    rows=[pd.read_csv(p) for p in (root/'cases').glob('*/*/*/*/perturbations/*/participant_metrics.csv')]
    perturb=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    if len(perturb):perturb.to_csv(root/'perturbation_participant_metrics.csv',index=False)
    return metrics,perturb

def plots(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);metrics,perturb=summarize(root);paths=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    if len(metrics):
        for task,frame in metrics.groupby('task'):
            agg=frame.groupby(['model','coordinate']).rmse.mean().unstack()
            fig,ax=plt.subplots(figsize=(12,6));agg.plot.bar(ax=ax,width=.82)
            ax.set_ylabel('Mean participant RMSE (target units)');ax.set_xlabel('Model')
            ax.set_title(f'{task}: coordinate sensitivity | {frame.participant_id.nunique()} evaluated participant(s)')
            ax.tick_params(axis='x',rotation=20);ax.legend(title='Coordinates',bbox_to_anchor=(1.01,1))
            fig.text(.02,.02,'Descriptive results. Check participant counts and paired comparisons CSV; incomplete models/modes are omitted, never zero-filled.',fontsize=9)
            fig.tight_layout(rect=(0,.05,1,1));path=root/f'{task}_coordinate_RMSE.png';fig.savefig(path,dpi=180);fig.savefig(path.with_suffix('.pdf'));plt.close(fig);paths.append(path)
    if len(perturb) and 'translation_sd' in perturb:
        for (task,mode),frame in perturb.dropna(subset=['translation_sd']).groupby(['task','coordinate']):
            fig,axes=plt.subplots(1,2,figsize=(12,5))
            for family,part in frame.groupby('model'):
                for ax,measure in zip(axes,['rmse_change','prediction_change_rmse']):
                    values=part.groupby('translation_sd')[measure].mean().sort_index()
                    values=pd.concat([values,pd.Series({0.:0.})]).sort_index()
                    ax.plot(values.index,values.to_numpy(),marker='o',label=family)
            for ax in axes:ax.axhline(0,color='black',lw=.6);ax.set_xlabel('Translation / outer-training input SD');ax.grid(alpha=.15)
            axes[0].set_ylabel('Change in participant RMSE (positive = worse)')
            axes[1].set_ylabel('Prediction change RMSE (target units)')
            axes[1].legend(fontsize=8)
            fig.suptitle(f'{task} / {mode}: translation sensitivity | {frame.participant_id.nunique()} participant(s)')
            fig.text(.02,.02,'Fixed saved models; no refitting under perturbation. Normalized-coordinate invariance can follow directly from preprocessing.',fontsize=9)
            fig.tight_layout(rect=(0,.07,1,.94));path=root/f'{task}_{mode}_translation_sensitivity.png';fig.savefig(path,dpi=180);fig.savefig(path.with_suffix('.pdf'));plt.close(fig);paths.append(path)
    if len(perturb) and 'yaw_degrees' in perturb:
        for (task,mode),frame in perturb.dropna(subset=['yaw_degrees']).groupby(['task','coordinate']):
            matrix=frame.groupby(['model','yaw_degrees']).rmse_change.mean().unstack()
            fig,ax=plt.subplots(figsize=(10,5));matrix.plot.bar(ax=ax);ax.axhline(0,color='black',lw=.7)
            ax.set_ylabel('Change in participant RMSE');ax.set_title(f'{task} / {mode}: coordinate-frame rotation sensitivity')
            ax.tick_params(axis='x',rotation=20);fig.tight_layout();path=root/f'{task}_{mode}_rotation_sensitivity.png';fig.savefig(path,dpi=180);fig.savefig(path.with_suffix('.pdf'));plt.close(fig);paths.append(path)
    return paths
