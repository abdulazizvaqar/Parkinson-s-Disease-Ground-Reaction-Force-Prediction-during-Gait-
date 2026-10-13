"""Participant-summary agreement; bootstrap resamples participants, never waveform points.
Run: python recalculate_agreement.py --input DIR --output DIR --bootstrap 10000
Input DIR holds original AnalysisGRF_*_STAT_.xlsx files from Internal.rar.
"""
import argparse,re,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import shapiro
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def estimates(d):
 return np.stack([d.mean(axis=1),d.mean(axis=1)-1.96*d.std(axis=1,ddof=1),d.mean(axis=1)+1.96*d.std(axis=1,ddof=1)],axis=-1)
def main():
 a=argparse.ArgumentParser();a.add_argument('--input',default='agreement_reanalysis/raw/Internal');a.add_argument('--output',default='agreement_reanalysis/results');a.add_argument('--bootstrap',type=int,default=10000);args=a.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(20261012)
 records=[];persons=[];phases=[];audits=[];plotdata={};phaseplots={};targets={}
 for f in sorted(Path(args.input).glob('AnalysisGRF_*_STAT_.xlsx')):
  match=re.fullmatch(r'AnalysisGRF_(AP|LONG|ML)_(LEFT|RIGHT)_SEQ2SEQ_(.*?)_STAT_\.xlsx',f.name);assert match,f.name
  c,side,model=match.groups();c={'LONG':'Vertical'}.get(c,c);model={'1D CONVO':'Conv1D-LSTM'}.get(model,model)
  x=pd.ExcelFile(f);m=pd.read_excel(x,'Measured').iloc[:,1:].to_numpy(float).T;p=pd.read_excel(x,'Predicted').iloc[:,1:].to_numpy(float).T
  names=pd.read_excel(x,'Name').iloc[0,1:].astype(str).to_numpy();ids=np.array([re.search(r'SUB\d+',n).group() for n in names]);subs=np.unique(ids);n=len(subs)
  assert m.shape==p.shape==(len(names),100) and len(set(names))==len(names) and np.isfinite(m).all() and np.isfinite(p).all()
  saved=pd.read_excel(x,'RMSE').iloc[0,1:].to_numpy(float);rmse=np.sqrt(np.mean((p-m)**2,axis=1));assert np.allclose(saved,rmse,atol=1e-10,rtol=1e-10)
  savednr=pd.read_excel(x,'nRMSE').iloc[0,1:].to_numpy(float);nr=rmse/np.ptp(m,axis=1);assert np.allclose(savednr,nr,atol=1e-10,rtol=1e-10)
  key=(c,side);target={name:row for name,row in zip(names,m)}
  if key in targets:assert targets[key].keys()==target.keys() and all(np.array_equal(target[k],targets[key][k]) for k in target)
  else:targets[key]=target
  audit={'file':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'component':c,'side':side,'model':model,'participants':n,'cycles':len(names),'points_per_cycle':100,'excluded_cycles':0,'max_saved_rmse_difference':float(np.max(abs(saved-rmse)))};audits.append(audit)
  draw=rng.integers(0,n,size=(args.bootstrap,n))
  for metric,fun in [('Mean_force',lambda v:v.mean(axis=1)),('Maximum_force',lambda v:v.max(axis=1)),('Minimum_force',lambda v:v.min(axis=1))]:
   cm,cp=fun(m),fun(p);sm=np.array([cm[ids==s].mean() for s in subs]);sp=np.array([cp[ids==s].mean() for s in subs]);d=sp-sm;pair=(sp+sm)/2
   est=estimates(d[None,:])[0];boots=estimates(d[draw]);ci=np.quantile(boots,[.025,.975],axis=0);assert ci.shape==(2,3)
   row={'Component':c,'Side':side,'Model':model,'Summary':metric,'N_participants':n,'N_cycles':len(names),'Units':'original GRF units; manuscript identifies N/kg','Bias':est[0],'Bias_CI_low':ci[0,0],'Bias_CI_high':ci[1,0],'Lower_LoA':est[1],'Lower_LoA_CI_low':ci[0,1],'Lower_LoA_CI_high':ci[1,1],'Upper_LoA':est[2],'Upper_LoA_CI_low':ci[0,2],'Upper_LoA_CI_high':ci[1,2],'Difference_SD':np.std(d,ddof=1),'Shapiro_Wilk_p_descriptive':shapiro(d).pvalue,'Empirical_difference_2_5_percentile':np.quantile(d,.025),'Empirical_difference_97_5_percentile':np.quantile(d,.975)};emp_boot=np.quantile(d[draw],[.025,.975],axis=1).T
   emp_ci=np.quantile(emp_boot,[.025,.975],axis=0)
   row.update({'Empirical_lower_CI_low':emp_ci[0,0],'Empirical_lower_CI_high':emp_ci[1,0],'Empirical_upper_CI_low':emp_ci[0,1],'Empirical_upper_CI_high':emp_ci[1,1]})
   records.append(row)
   for i,s in enumerate(subs):persons.append({'Component':c,'Side':side,'Model':model,'Summary':metric,'Participant':s,'N_cycles':sum(ids==s),'Measured_summary':sm[i],'Predicted_summary':sp[i],'Pair_mean':pair[i],'Difference_predicted_minus_measured':d[i]})
   plotdata[c,metric,side,model]=(pair,d,row)
  # One averaged waveform per participant: never pool phase points to obtain an SD.
  sm=np.array([m[ids==s].mean(axis=0) for s in subs]);sp=np.array([p[ids==s].mean(axis=0) for s in subs]);dp=sp-sm;est=estimates(dp[None,:,:])[0]
  boots=np.empty((args.bootstrap,100,3))
  for j in range(0,args.bootstrap,250):boots[j:j+250]=estimates(dp[draw[j:j+250]])
  ci=np.quantile(boots,[.025,.975],axis=0)
  for j in range(100):
   row={'Component':c,'Side':side,'Model':model,'N_participants':n,'N_cycles':len(names),'Phase_index':j,'Phase_percent':j*100/99,'Bias':est[j,0],'Bias_CI_low':ci[0,j,0],'Bias_CI_high':ci[1,j,0],'Lower_LoA':est[j,1],'Lower_LoA_CI_low':ci[0,j,1],'Lower_LoA_CI_high':ci[1,j,1],'Upper_LoA':est[j,2],'Upper_LoA_CI_low':ci[0,j,2],'Upper_LoA_CI_high':ci[1,j,2]};phases.append(row)
  phaseplots[c,side,model]=(est,ci,n)
  print(c,side,model,'verified and calculated',flush=True)
 pd.DataFrame(records).to_csv(out/'Participant_Summary_Agreement.csv',index=False)
 pd.DataFrame(persons).to_csv(out/'Participant_Summary_Values.csv',index=False)
 pd.DataFrame(phases).to_csv(out/'Phase_Specific_Agreement.csv',index=False)
 (out/'Data_Audit.json').write_text(json.dumps(audits,indent=2))
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 with PdfPages(out/'Agreement_Figures.pdf') as pdf:
  for c in ['AP','Vertical','ML']:
   for metric in ['Mean_force','Maximum_force','Minimum_force']:
    fig,axs=plt.subplots(3,2,figsize=(10,11),layout='constrained')
    for row,model in enumerate(['Conv1D-LSTM','LSTM','GRU']):
     for col,side in enumerate(['LEFT','RIGHT']):
      ax=axs[row,col];xx,d,v=plotdata[c,metric,side,model];ax.scatter(xx,d,s=23,color='#175676');ax.axhline(0,color='#aaa',lw=.6)
      for name,color in [('Bias','#165a9d'),('Lower_LoA','#ba533c'),('Upper_LoA','#ba533c')]:
       ax.axhline(v[name],color=color,lw=1.2);ax.axhspan(v[name+'_CI_low'],v[name+'_CI_high'],color=color,alpha=.09)
      ax.set_title(f'{model} {side.lower()}   n={v["N_participants"]}');ax.set_xlabel('Mean of predicted and measured summary');ax.set_ylabel('Predicted − measured')
      ax.text(.02,.98,f'Bias {v["Bias"]:.3f}; LoA [{v["Lower_LoA"]:.3f}, {v["Upper_LoA"]:.3f}]',transform=ax.transAxes,va='top',fontsize=8)
    fig.suptitle(f'{c}: participant-level {metric.replace("_"," ").lower()}\nOne point per participant; shaded bands are 95% bootstrap confidence intervals',fontsize=13)
    pdf.savefig(fig);fig.savefig(out/f'{c}_{metric}_agreement.png',dpi=150);plt.close(fig)
   fig,axs=plt.subplots(3,2,figsize=(10,11),layout='constrained');phase=np.linspace(0,100,100)
   for row,model in enumerate(['Conv1D-LSTM','LSTM','GRU']):
    for col,side in enumerate(['LEFT','RIGHT']):
     ax=axs[row,col];e,ci,n=phaseplots[c,side,model]
     for k,color,label in [(0,'#165a9d','Bias'),(1,'#ba533c','Lower LoA'),(2,'#ba533c','Upper LoA')]:
      ax.plot(phase,e[:,k],color=color,label=label,lw=1.2);ax.fill_between(phase,ci[0,:,k],ci[1,:,k],color=color,alpha=.12)
     ax.axhline(0,color='#aaa',lw=.6);ax.set_title(f'{model} {side.lower()}   n={n}');ax.set_xlabel('Normalized gait phase (%)');ax.set_ylabel('Predicted − measured')
   fig.suptitle(f'{c}: agreement at each phase across participant-mean waveforms\nShaded confidence intervals are pointwise, not simultaneous',fontsize=13);pdf.savefig(fig);fig.savefig(out/f'{c}_phase_agreement.png',dpi=150);plt.close(fig)
 summary=pd.DataFrame(records);print(summary[summary.Summary=='Mean_force'][['Component','Side','Model','Bias','Lower_LoA','Upper_LoA']].to_string(index=False))
if __name__=='__main__':main()
