"""Recalculate FDR from the supplied complete aggregate statistical tables."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import false_discovery_control
from fdr_core import correct_family,summarize

def main():
 a=argparse.ArgumentParser();a.add_argument('--clinical',required=True);a.add_argument('--gait',required=True);a.add_argument('--spatiotemporal-clinical');a.add_argument('--output',required=True);args=a.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);clinical=pd.read_csv(args.clinical);gait=pd.read_csv(args.gait);report={}
 for kind in ['Pearson','Spearman']:
  f=clinical.copy();f['test_id']=f.nRMSE_Metric.astype(str)+' | '+f.Severity_Metric.astype(str);f['error_metric']=f.nRMSE_Metric;f['clinical_measure']=f.Severity_Metric;f['p_raw']=f[kind+'_p'];f['r']=f[kind+'_r'];f['n_participants']=f.N
  r=correct_family(f,'clinical_correlations');r.to_csv(out/f'Clinical_{kind}_FDR.csv',index=False);report[kind]=summarize(r)
 f=gait.copy();f['test_id']=f.Parameter;f['gait_parameter']=f.Clean_Label;f['p_raw']=f.Kruskal_Wallis_p;f['n_participants']=f.N_Subjects;r=correct_family(f,'gait_comparisons');r.to_csv(out/'Gait_Kruskal_Wallis_FDR.csv',index=False);report['gait']=summarize(r)
 report['nominal_either_method_count']=int(((clinical.Pearson_p<.05)|(clinical.Spearman_p<.05)).sum())
 if args.spatiotemporal_clinical:
  f=pd.read_csv(args.spatiotemporal_clinical)
  if len(f)!=418 or f.duplicated(['Spatiotemporal','Clinical']).any():raise ValueError('Supply all 418 unique spatiotemporal-clinical tests')
  p=f.p.to_numpy(float)
  if not np.isfinite(p).all() or np.any((p<=0)|(p>1)):raise ValueError('Invalid complete p-values')
  f['q_bh']=false_discovery_control(p,method='bh');f['q_by']=false_discovery_control(p,method='by');f.to_csv(out/'Spatiotemporal_Clinical_FDR.csv',index=False)
  report['spatiotemporal_clinical']={'n_tests':418,'nominal':int((p<.05).sum()),'bh':int((f.q_bh<.05).sum())}
 (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
