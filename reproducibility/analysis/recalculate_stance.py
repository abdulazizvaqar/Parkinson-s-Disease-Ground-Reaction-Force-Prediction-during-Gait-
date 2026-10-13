"""Retrospective force-defined contact and peak sensitivity analysis.
Does not train models. Uses measured vertical force to define contact for all components.
Physical impulses/loading rates are unavailable without cycle durations.
"""
import argparse,re,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def key(name):
 m=re.search(r'(SUB\d+)_tr_model_output_(\d+)',name);assert m,name;return (m[1],int(m[2]))
def read(f):
 x=pd.ExcelFile(f);m=pd.read_excel(x,'Measured').iloc[:,1:].to_numpy(float).T;p=pd.read_excel(x,'Predicted').iloc[:,1:].to_numpy(float).T
 names=pd.read_excel(x,'Name').iloc[0,1:].astype(str).to_numpy();keys=[key(n) for n in names]
 assert len(set(keys))==len(keys) and m.shape==p.shape==(len(keys),100) and np.isfinite(m).all() and np.isfinite(p).all()
 saved=pd.read_excel(x,'RMSE').iloc[0,1:].to_numpy(float);assert np.allclose(saved,np.sqrt(np.mean((p-m)**2,axis=1)),atol=1e-10,rtol=1e-10)
 return {k:(m[i],p[i],names[i]) for i,k in enumerate(keys)}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--input',default='agreement_reanalysis/raw/Internal');parser.add_argument('--output',default='stance_reanalysis/results');args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 files=sorted(Path(args.input).glob('AnalysisGRF_*_STAT_.xlsx'));datasets={};hashes={}
 for f in files:
  z=re.fullmatch(r'AnalysisGRF_(AP|LONG|ML)_(LEFT|RIGHT)_SEQ2SEQ_(.*?)_STAT_\.xlsx',f.name);assert z
  c,side,model=z.groups();c={'LONG':'Vertical'}.get(c,c);model={'1D CONVO':'Conv1D-LSTM'}.get(model,model);datasets[c,side,model]=read(f);hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
 # Contact target mapping must be identical across architectures and match cycle IDs.
 refs={side:datasets['Vertical',side,'Conv1D-LSTM'] for side in ['LEFT','RIGHT']}
 for side in refs:
  for model in ['Conv1D-LSTM','LSTM','GRU']:
   d=datasets['Vertical',side,model];assert refs[side].keys()==d.keys()
   assert all(np.array_equal(refs[side][k][0],d[k][0]) for k in d)
 rows=[];checks=[];dx=1/99
 for (c,side,model),data in datasets.items():
  assert data.keys()==refs[side].keys()
  for k,(m,p,name) in data.items():
   v=refs[side][k][0];mask=v>.01*v.max();ii=np.flatnonzero(mask);assert len(ii)>1 and np.all(np.diff(ii)==1)
   span=np.ptp(m);assert span>0;err=p-m
   r={'Component':c,'Side':side,'Model':model,'Participant':k[0],'Cycle_index':k[1],'Original_name':name,'Contact_points_1pct':len(ii),'Contact_first_index':ii[0],'Contact_last_index':ii[-1],
      'Full_RMSE':np.sqrt(np.mean(err**2)),'Full_NRMSE_full_range_pct':100*np.sqrt(np.mean(err**2))/span}
   for threshold,label in [(.01,'1pct'),(.02,'2pct'),(.05,'5pct')]:
    mask_t=v>threshold*v.max();assert mask_t.any()
    r['Contact_RMSE_'+label]=np.sqrt(np.mean(err[mask_t]**2));r['Contact_NRMSE_full_range_pct_'+label]=100*r['Contact_RMSE_'+label]/span
    r['Contact_MAE_'+label]=np.mean(np.abs(err[mask_t]));r['Contact_points_'+label]=int(mask_t.sum())
   r['Contact_RMSE_to_full_ratio']=r['Contact_RMSE_1pct']/r['Full_RMSE']
   if c=='AP':
    for event,fun in [('Braking',lambda w:np.maximum(-w,0)),('Propulsive',lambda w:np.maximum(w,0))]:
     # Integrate only from the first through last force-defined contact sample.
     area_m=np.trapezoid(fun(m[ii]),dx=dx);area_p=np.trapezoid(fun(p[ii]),dx=dx)
     r[event+'_phase_area_measured']=area_m;r[event+'_phase_area_predicted']=area_p
     r[event+'_phase_area_error']=area_p-area_m;r[event+'_phase_area_absolute_error']=abs(area_p-area_m)
   if c=='Vertical':
    im=int(ii[np.argmax(m[ii])]);ip=int(ii[np.argmax(p[ii])]);r['Global_peak_amplitude_error']=p[ip]-m[im];r['Global_peak_amplitude_absolute_error']=abs(r['Global_peak_amplitude_error'])
    r['Global_peak_timing_error_pct_cycle']=100*(ip-im)/99;r['Global_peak_timing_absolute_error_pct_cycle']=abs(r['Global_peak_timing_error_pct_cycle'])
    seg=m[ii];peaks,props=find_peaks(np.r_[0,seg,0],prominence=.05*m.max(),distance=max(1,int(np.ceil(.1*len(seg)))))
    peaks=peaks-1;valid=(peaks>=0)&(peaks<len(seg));peaks=peaks[valid];prom=props['prominences'][valid]
    r['Measured_two_peak_eligible']=len(peaks)>=2;r['Measured_qualifying_peak_count']=len(peaks)
    if len(peaks)>=2:
     chosen=sorted(peaks[np.argsort(prom)[-2:]]);first,second=[int(ii[j]) for j in chosen];mid=(first+second)//2
     for label,mp,window in [('First',first,ii[ii<=mid]),('Second',second,ii[ii>mid])]:
      pp=int(window[np.argmax(p[window])]);r[label+'_peak_measured']=m[mp];r[label+'_peak_predicted']=p[pp]
      r[label+'_peak_amplitude_error']=p[pp]-m[mp];r[label+'_peak_amplitude_absolute_error']=abs(p[pp]-m[mp])
      r[label+'_peak_timing_error_pct_cycle']=100*(pp-mp)/99;r[label+'_peak_timing_absolute_error_pct_cycle']=abs(r[label+'_peak_timing_error_pct_cycle'])
     # Report topology separately: amplitude/timing do not prove two predicted local peaks exist.
     pq,_=find_peaks(np.r_[0,p[ii],0],prominence=.05*m.max(),distance=max(1,int(np.ceil(.1*len(ii)))))
     r['Predicted_two_peak_detected']=len(pq)>=2
   rows.append(r)
  print(c,side,model,'completed',flush=True)
 cycles=pd.DataFrame(rows);cycles.to_csv(out/'Cycle_Level_Stance_and_Peak_Metrics.csv',index=False)
 metrics=[x for x in cycles if x.startswith(('Full_','Contact_RMSE_','Contact_NRMSE_','Contact_MAE_')) and not x.endswith(('measured','predicted'))]
 metrics+=['Braking_phase_area_error','Braking_phase_area_absolute_error','Propulsive_phase_area_error','Propulsive_phase_area_absolute_error']
 metrics+=[x for x in cycles if re.match(r'(Global|First|Second)_peak_(amplitude|timing).*error',x)]
 metrics=list(dict.fromkeys(metrics));persons=[];summaries=[];rng=np.random.default_rng(20261012)
 for (c,side,model),g in cycles.groupby(['Component','Side','Model'],sort=True):
  for metric in metrics:
   gg=g[['Participant',metric]].dropna()
   if gg.empty:continue
   values=gg.groupby('Participant')[metric].agg(['mean','count']);n=len(values);arr=values['mean'].to_numpy()
   for sub,r in values.iterrows():persons.append({'Component':c,'Side':side,'Model':model,'Metric':metric,'Participant':sub,'N_valid_cycles':int(r['count']),'Participant_mean':r['mean']})
   samples=arr[rng.integers(0,n,size=(10000,n))].mean(axis=1);ci=np.quantile(samples,[.025,.975]);summaries.append({'Component':c,'Side':side,'Model':model,'Metric':metric,'N_participants':n,'N_valid_cycles':len(gg),'Participant_mean':arr.mean(),'Mean_CI_low':ci[0],'Mean_CI_high':ci[1],'Participant_median':np.median(arr),'Participant_Q1':np.quantile(arr,.25),'Participant_Q3':np.quantile(arr,.75)})
 summary=pd.DataFrame(summaries);summary.to_csv(out/'Participant_Level_Metric_Summaries.csv',index=False);pd.DataFrame(persons).to_csv(out/'Participant_Level_Metric_Values.csv',index=False)
 audit=[]
 for side in ['LEFT','RIGHT']:
  g=cycles[(cycles.Component=='Vertical')&(cycles.Side==side)&(cycles.Model=='Conv1D-LSTM')]
  d={'side':side,'cycles':len(g),'participants':g.Participant.nunique(),'measured_two_peak_cycles':int(g.Measured_two_peak_eligible.sum()),'non_two_peak_cycles':int((g.Measured_two_peak_eligible == False).sum()),'contact_points_min':int(g.Contact_points_1pct.min()),'contact_points_max':int(g.Contact_points_1pct.max()),'cycles_contact_mask_changes_2pct':int((g.Contact_points_1pct!=g.Contact_points_2pct).sum()),'cycles_contact_mask_changes_5pct':int((g.Contact_points_1pct!=g.Contact_points_5pct).sum())};audit.append(d)
 eligibility=cycles[cycles.Component=='Vertical'].groupby(['Side','Model']).agg(N_cycles=('Measured_two_peak_eligible','size'),N_measured_two_peak_cycles=('Measured_two_peak_eligible','sum'),N_predicted_two_peak_cycles_among_eligible=('Predicted_two_peak_detected','sum'));eligibility.to_csv(out/'Peak_Eligibility_and_Topology.csv')
 (out/'Audit.json').write_text(json.dumps({'files_sha256':hashes,'contact_and_peak_checks':audit,'B':10000,'seed':20261012},indent=2))
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 with PdfPages(out/'Stance_and_Peak_Figures.pdf') as pdf:
  fig,axs=plt.subplots(3,2,figsize=(10,10),layout='constrained')
  for i,c in enumerate(['AP','Vertical','ML']):
   for j,side in enumerate(['LEFT','RIGHT']):
    ax=axs[i,j];g=summary[(summary.Component==c)&(summary.Side==side)];models=['Conv1D-LSTM','LSTM','GRU'];xx=np.arange(3)
    for metric,offset,label,col in [('Full_RMSE',-.15,'Full cycle','#758b9b'),('Contact_RMSE_1pct',.15,'Force-defined contact','#176b8b')]:
     z=g[g.Metric==metric].set_index('Model').loc[models];yy=z.Participant_mean.to_numpy();ax.bar(xx+offset,yy,width=.28,color=col,label=label);ax.errorbar(xx+offset,yy,yerr=np.vstack([yy-z.Mean_CI_low,z.Mean_CI_high-yy]),fmt='none',color='black',capsize=3)
    ax.set_xticks(xx,models);ax.set_ylabel('Mean participant RMSE (original GRF units)');ax.set_title(c+' '+side.lower());ax.legend(fontsize=8)
  fig.suptitle('Full-cycle and force-defined contact errors\n95% participant-bootstrap confidence intervals',fontsize=14);pdf.savefig(fig);fig.savefig(out/'Contact_vs_Full_RMSE.png',dpi=150);plt.close(fig)
  fig,axs=plt.subplots(2,2,figsize=(10,8),layout='constrained')
  for i,metric in enumerate(['First_peak_amplitude_absolute_error','Second_peak_amplitude_absolute_error']):
   for j,side in enumerate(['LEFT','RIGHT']):
    ax=axs[i,j];g=summary[(summary.Component=='Vertical')&(summary.Side==side)&(summary.Metric==metric)].set_index('Model').loc[models];yy=g.Participant_mean.to_numpy();ax.bar(xx,yy,color='#176b8b');ax.errorbar(xx,yy,yerr=np.vstack([yy-g.Mean_CI_low,g.Mean_CI_high-yy]),fmt='none',color='black',capsize=3);ax.set_xticks(xx,models);ax.set_title(metric.split('_')[0]+' peak '+side.lower());ax.set_ylabel('Mean participant absolute amplitude error')
  fig.suptitle('Vertical peak errors in measured two-peak cycles\nReference-window maxima; 95% participant-bootstrap confidence intervals',fontsize=13);pdf.savefig(fig);fig.savefig(out/'Vertical_Peak_Amplitude_Errors.png',dpi=150);plt.close(fig)
 print('AUDIT',json.dumps(audit));print(summary[summary.Metric.isin(['Full_RMSE','Contact_RMSE_1pct'])][['Component','Side','Model','Metric','Participant_mean']].to_string(index=False))
if __name__=='__main__':main()
