"""Read-only held-out evaluation. This module never trains or tunes a model."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import numpy as np
import pandas as pd
from protocol import cycle_metrics

FAMILIES = ('mean_waveform', 'ridge', 'phase_only', 'shallow_mlp', 'lstm', 'gru', 'conv1d_lstm')

def read_json(path):
    return json.loads(Path(path).read_text())

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def check_metadata(spec, info, selected, family, outer, ds):
    if spec['config'].get('smoke', False):
        raise ValueError('This evaluator requires research outputs, not smoke-test fixtures')
    if spec['task'] != ds.task or spec['data_fingerprint'] != ds.fingerprint:
        raise ValueError('Internal data or preprocessing differs from the saved experiment')
    if sorted(spec['subjects']) != ds.subjects or outer not in ds.subjects:
        raise ValueError('Participant list differs from the saved experiment')
    if info['family'] != family or selected['family'] != family:
        raise ValueError('Saved family mismatch')
    training = info.get('training_participants', [])
    expected = [s for s in ds.subjects if s != outer]
    if outer in training or sorted(training) != expected or len(training) != len(set(training)):
        raise ValueError('Test participant overlaps training, or training cohort is incomplete')
    if info.get('n_training_participants') != len(expected):
        raise ValueError('Saved training-participant count differs')
    if info.get('data_fingerprint') != ds.subset(expected).fingerprint:
        raise ValueError('Model training-data fingerprint does not match')
    if info.get('parameters') != selected.get('parameters'):
        raise ValueError('Saved refit parameters differ from the frozen selection')
    if family in ('lstm', 'gru', 'conv1d_lstm') and info.get('epochs') != selected.get('epochs'):
        raise ValueError('Saved refit epochs differ from the frozen selection')

def read_prediction_cache(path, te):
    with np.load(path, allow_pickle=False) as cache:
        ids = cache['cycle_ids'].astype(str)
        truth = cache['measured']
        pred = cache['predicted']
    expected = te.cycles.cycle_id.to_numpy(str)
    if not np.array_equal(ids, expected):
        raise ValueError('Saved predictions have different cycle identities/order')
    if truth.shape != te.y.shape or pred.shape != te.y.shape:
        raise ValueError('Saved prediction or target shape differs from test data')
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError('Nonfinite cached targets or predictions')
    if not np.allclose(truth, te.y, rtol=1e-6, atol=1e-7):
        raise ValueError('Cached targets differ from the original held-out measurements')
    return pred

def predict_weights(directory, info, x, software):
    """Inference only, using the input scaler saved with the fitted model."""
    path = Path(directory)/info['file']
    if info['family'] in ('lstm','gru','conv1d_lstm'):
        for name in ['tensorflow','keras']:
            actual = importlib.metadata.version(name)
            if actual != software.get(name):
                raise ValueError(f'{name}: inference needs the saved version {software.get(name)}, found {actual}. Cached predictions do not require TensorFlow.')
        import tensorflow as tf
        scaler = info['input_scaler']
        std = np.asarray(scaler['std'], float)
        if not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError('Invalid saved scaler')
        z = (x - np.asarray(scaler['mean'])[None,None,:]) / std[None,None,:]
        model = tf.keras.models.load_model(path, compile=False)
        try:
            return np.asarray(model.predict(z, verbose=0))
        finally:
            del model
            tf.keras.backend.clear_session()
    for name in ['scikit-learn','joblib']:
        actual = importlib.metadata.version(name)
        if actual != software.get(name):
            raise ValueError(f'{name}: saved-model version is {software.get(name)}, current version is {actual}. Use cached predictions or the original package versions.')
    import joblib
    # Only load trusted model files created by the user's own training notebook.
    obj = joblib.load(path)
    if obj['family'] != info['family']:
        raise ValueError('Baseline family differs inside saved model')
    if 'waveform' in obj:
        return np.repeat(obj['waveform'][None], len(x), axis=0)
    z = obj['scaler'].transform(x.reshape(len(x),-1))
    return np.asarray(obj['model'].predict(z)).reshape((len(x),*obj['shape']))

def evaluate_available(root, ds, outer, output, mode='cached_only'):
    if mode not in ('cached_only','inference_if_needed','fresh_inference'):
        raise ValueError('Unknown evaluation mode')
    root, output = Path(root), Path(output)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Write test reports outside the training experiment')
    output.mkdir(parents=True,exist_ok=True)
    spec = read_json(root/'prespecification.json')
    te = ds.subset([outer])
    inventory, tables, summary, predictions, hashes = [], [], [], {}, {}
    for family in FAMILIES:
        folder = root/'nested'/outer/family
        descriptor, selection = folder/'refit/model.json', folder/'selection/selected.json'
        if not descriptor.exists() or not selection.exists():
            inventory.append(dict(model=family,status='unavailable',reason='No completed frozen selection/refitted model'))
            continue
        try:
            info, chosen = read_json(descriptor), read_json(selection)
            check_metadata(spec,info,chosen,family,outer,ds)
            marker, cache = folder/'complete.json', folder/'predictions.npz'
            use_cache = mode != 'fresh_inference' and cache.exists() and marker.exists()
            if use_cache:
                done = read_json(marker)
                if done.get('outer_subject') != outer or done.get('family') != family:
                    raise ValueError('Outer-evaluation completion marker differs')
                pred = read_prediction_cache(cache,te)
                kind = 'saved held-out predictions'
                hashes[family] = {'predictions_sha256':digest(cache)}
            else:
                if mode == 'cached_only':
                    inventory.append(dict(model=family,status='unavailable',reason='Completed prediction cache absent; inference mode can use a saved refit'))
                    continue
                weights = folder/'refit'/info['file']
                if not weights.is_file():
                    raise ValueError('Refitted weights missing')
                pred = predict_weights(folder/'refit',info,te.x,spec['software'])
                kind = 'fresh inference from saved weights'
                hashes[family] = {'model_sha256':digest(weights)}
            metrics = cycle_metrics(te.y,pred)
            a = te.y.reshape(len(te.y),-1).astype(float)
            b = pred.reshape(len(pred),-1).astype(float)
            a -= a.mean(axis=1,keepdims=True)
            b -= b.mean(axis=1,keepdims=True)
            denominator = np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
            metrics['waveform_r'] = np.divide((a*b).sum(axis=1),denominator,
                out=np.full(len(a),np.nan),where=denominator>1e-12)
            metrics.insert(0,'cycle_id',te.cycles.cycle_id.to_numpy(str))
            metrics.insert(0,'participant_id',outer)
            metrics.insert(0,'model',family)
            metrics.insert(0,'task',te.task)
            metrics['prediction_source'] = kind
            tables.append(metrics)
            summary.append(dict(task=te.task,participant_id=outer,model=family,n_participants=1,
                n_cycles=len(te.y),mae=float(metrics.mae.mean()),rmse=float(metrics.rmse.mean()),
                nrmse=float(metrics.nrmse.mean()),nrmse_percent=float(metrics.nrmse.mean()*100),
                mean_cycle_waveform_r=float(metrics.waveform_r.mean()),
                undefined_nrmse_cycles=int(metrics.nrmse.isna().sum()),prediction_source=kind,
                uncertainty='No participant-level confidence interval from one participant'))
            predictions[family] = pred
            hashes[family].update(model_metadata_sha256=digest(descriptor),selection_sha256=digest(selection))
            np.savez_compressed(output/f'{family}_heldout_predictions.npz',
                cycle_ids=te.cycles.cycle_id.to_numpy(str),measured=te.y,predicted=pred)
            inventory.append(dict(model=family,status='evaluated',reason=kind))
        except (ValueError,KeyError,OSError,importlib.metadata.PackageNotFoundError) as error:
            inventory.append(dict(model=family,status='not evaluated',reason=str(error)))
    inventory = pd.DataFrame(inventory)
    inventory.to_csv(output/'model_availability.csv',index=False)
    result = pd.DataFrame(summary)
    if tables:
        pd.concat(tables,ignore_index=True).to_csv(output/'heldout_cycle_metrics.csv',index=False)
        result.to_csv(output/'heldout_participant_metrics.csv',index=False)
        if 'mean_waveform' in set(result.model):
            reference = result.set_index('model').loc['mean_waveform']
            comparisons=[]
            for _, row in result.iterrows():
                if row.model == 'mean_waveform': continue
                comparisons.append(dict(model=row.model,reference='mean_waveform',n_participants=1,
                    rmse_improvement=reference.rmse-row.rmse,
                    nrmse_percentage_point_improvement=reference.nrmse_percent-row.nrmse_percent,
                    interpretation='Positive favors model; descriptive result for one held-out participant'))
            pd.DataFrame(comparisons).to_csv(output/'descriptive_baseline_comparisons.csv',index=False)
    (output/'evaluation_provenance.json').write_text(json.dumps({
        'experiment':str(root),'task':ds.task,'held_out_participant':outer,
        'n_test_participants':1,'test_cycle_count':len(te.y),'mode':mode,
        'prespecification_sha256':digest(root/'prespecification.json'),'artifacts':hashes,
        'notes':['No training or tuning performed','No external cohort loaded',
                 'No selected_deep winner declared from these test results',
                 'NRMSE = cycle RMSE / measured full-cycle range; participant result = mean of defined cycle values',
                 'Cycles are repeated observations within one participant; no cohort inference or significance tests']},indent=2))
    return inventory,result,predictions,te

def make_plots(summary, predictions, te, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from math import ceil
    output=Path(output)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    labels={'mean_waveform':'Mean waveform','ridge':'Ridge','phase_only':'Phase only',
            'shallow_mlp':'Shallow MLP','lstm':'LSTM','gru':'GRU','conv1d_lstm':'Conv1D-LSTM'}
    order=[m for m in FAMILIES if m in predictions]
    if not order: return []
    data=summary.set_index('model').loc[order]
    colors=['#6C7A89' if m not in ('lstm','gru','conv1d_lstm') else {'lstm':'#0072B2','gru':'#D58A00','conv1d_lstm':'#009E73'}[m] for m in order]
    subtitle=f'{te.task} | held-out participant {te.subjects[0]} | {len(te.y)} cycles | n = 1 participant'
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,metric,title in zip(axes,['mae','rmse','nrmse_percent'],['Mean cycle MAE','Mean cycle RMSE','Mean cycle NRMSE (%)']):
        values=data[metric].to_numpy()
        ax.barh(range(len(order)),values,color=colors)
        ax.set_yticks(range(len(order)),[labels[m] for m in order]);ax.invert_yaxis()
        ax.set_xlabel('Target units' if metric!='nrmse_percent' else 'Percent of measured cycle range')
        ax.set_title(title);ax.set_xlim(left=0);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Saved-model held-out errors\n'+subtitle,fontsize=14)
    fig.text(.02,.015,'Descriptive participant-level results. No cohort confidence intervals. Physical target units require source verification.',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.89))
    fig.savefig(output/'Heldout_Model_Errors.png',dpi=180)
    fig.savefig(output/'Heldout_Model_Errors.pdf');plt.close(fig)
    phase=np.linspace(0,100,te.y.shape[1]);truth=te.y[:,:,0]
    fig,axes=plt.subplots(ceil(len(order)/3),3,figsize=(15,3.5*ceil(len(order)/3)),squeeze=False,sharex=True,sharey=True)
    for ax,family,color in zip(axes.flat,order,colors):
        low,high=np.quantile(truth,[.25,.75],axis=0)
        ax.fill_between(phase,low,high,color='#ADB5BD',alpha=.25,label='Measured cycle IQR')
        ax.plot(phase,truth.mean(axis=0),color='black',lw=1.8,label='Measured mean')
        ax.plot(phase,predictions[family][:,:,0].mean(axis=0),color=color,lw=1.8,label='Predicted mean')
        ax.set_title(labels[family]);ax.set_xlabel('Normalized gait phase (%)');ax.set_ylabel('GRF (target units)')
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(order):]:ax.set_visible(False)
    handles,legend=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Measured and predicted waveforms\n'+subtitle,fontsize=14)
    fig.tight_layout(rect=(0,.07,1,.90))
    fig.savefig(output/'Heldout_Waveforms.png',dpi=180)
    fig.savefig(output/'Heldout_Waveforms.pdf');plt.close(fig)
    k=min(3,len(truth));fig,axes=plt.subplots(k,1,figsize=(11,3*k),squeeze=False,sharex=True)
    for i,ax in enumerate(axes[:,0]):
        ax.plot(phase,truth[i],color='black',lw=2.2,label='Measured')
        for j,(family,color) in enumerate(zip(order,colors)):
            ax.plot(phase,predictions[family][i,:,0],color=color,ls=['--',':','-.','--','-','-','-'][FAMILIES.index(family)],lw=1.4,label=labels[family])
        ax.set_title(str(te.cycles.cycle_id.iloc[i]));ax.set_ylabel('GRF (target units)');ax.grid(alpha=.15)
    axes[-1,0].set_xlabel('Normalized gait phase (%)')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.suptitle('First available cycles in source order; no selection by prediction quality\n'+subtitle,fontsize=12)
    fig.tight_layout(rect=(0,.10,1,.91))
    fig.savefig(output/'Heldout_Example_Cycles.png',dpi=180);plt.close(fig)
    return [output/name for name in ['Heldout_Model_Errors.png','Heldout_Waveforms.png','Heldout_Example_Cycles.png']]
