from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import argparse
parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
d=pd.read_csv(args.input)
assert len(d)==54
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.labelsize':11,'axes.titlesize':13,'xtick.labelsize':10,'ytick.labelsize':10})
fig,axs=plt.subplots(3,3,figsize=(16,12))
fig.subplots_adjust(left=.145,right=.985,bottom=.16,top=.865,hspace=.35,wspace=.23)
models=['Conv1D-LSTM','LSTM','GRU'];sides=['LEFT','RIGHT'];colors={'LEFT':'#166B99','RIGHT':'#B35D22'}
ypos=[5.2,4.4,3.2,2.4,1.2,.4]
labels=[f'{m}  {s.lower()}' for m in models for s in sides]
for i,comp in enumerate(['AP','Vertical','ML']):
 for j,metric in enumerate(['Mean_force','Maximum_force','Minimum_force']):
  ax=axs[i,j];g=d[(d.Component==comp)&(d.Summary==metric)]
  for y,(model,side) in zip(ypos,[(m,s) for m in models for s in sides]):
   r=g[(g.Model==model)&(g.Side==side)].iloc[0];color=colors[side]
   # Thick segment is the LoA interval, not the confidence interval for bias.
   ax.plot([r.Lower_LoA,r.Upper_LoA],[y,y],color=color,lw=2.4,alpha=.6,zorder=2)
   for name in ['Lower_LoA','Upper_LoA']:
    ax.plot([r[name+'_CI_low'],r[name+'_CI_high']],[y,y],color=color,lw=.8,zorder=3)
    ax.plot([r[name+'_CI_low'],r[name+'_CI_low']],[y-.09,y+.09],color=color,lw=.8)
    ax.plot([r[name+'_CI_high'],r[name+'_CI_high']],[y-.09,y+.09],color=color,lw=.8)
    ax.scatter(r[name],y,s=28,marker='|',color=color,zorder=4)
   ax.plot([r.Bias_CI_low,r.Bias_CI_high],[y,y],color=color,lw=1.2,zorder=5)
   ax.scatter(r.Bias,y,s=38,color=color,edgecolor='white',linewidth=.6,zorder=6)
  ax.axvline(0,color='#626262',ls=(0,(3,3)),lw=.9,zorder=1)
  ax.set_ylim(-.15,5.8);ax.set_yticks(ypos,labels if j==0 else ['']*6);ax.tick_params(axis='y',length=0,pad=9)
  ax.set_title(f'{chr(65+i*3+j)}   {comp} · {metric.replace("_"," ").lower()}',loc='left',pad=12,fontweight='bold')
  ax.grid(axis='x',color='#e7e7e7',lw=.7,zorder=0)
  # Limits include all endpoint confidence intervals.
  lo=g.Lower_LoA_CI_low.min();hi=g.Upper_LoA_CI_high.max();pad=(hi-lo)*.07;ax.set_xlim(lo-pad,hi+pad)
  ax.set_xlabel('Predicted − measured (reported N/kg)',labelpad=7)
fig.text(.145,.955,'Participant-level GRF agreement',fontsize=23,fontweight='bold',ha='left')
fig.text(.145,.924,'Mean, maximum and minimum force averaged across cycles within each participant',fontsize=13,ha='left',color='#414141')
fig.text(.145,.897,'Left: 20 participants / 283 cycles   •   Right: 19 participants / 263 cycles',fontsize=12,ha='left',color='#414141')
handles=[Line2D([0],[0],marker='o',lw=0,color=colors['LEFT'],label='Left'),Line2D([0],[0],marker='o',lw=0,color=colors['RIGHT'],label='Right'),Line2D([0],[0],lw=2.4,color='#666',label='Bias ± 1.96 SD (limits of agreement)'),Line2D([0],[0],lw=.8,marker='|',color='#666',label='Endpoint whiskers: 95% CI')]
fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.14,.095),ncol=4,frameon=False,fontsize=11,handlelength=2.5,columnspacing=1.6)
fig.text(.145,.070,'Dots show mean bias; the short line around each dot shows its 95% CI. All CIs use 10,000 participant bootstrap resamples.',fontsize=10.5,color='#414141')
fig.text(.145,.045,'Normal-based limits require caution for non-normal differences. These limits concern participant averages, not individual cycles.',fontsize=10.5,color='#414141')
fig.text(.145,.020,'Axis ranges differ between panels. No threshold for clinical interchangeability was established.',fontsize=10.5,color='#414141')
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=300,facecolor='white');plt.close(fig);print(out)
