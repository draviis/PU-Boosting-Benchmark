"""Descriptive labels from the audited Methods registry; no manual ID guesses."""
from pathlib import Path
import re,csv
ROOT=Path(__file__).resolve().parents[3];HOME=Path(__file__).resolve().parents[1]
NL=['Additive sinusoidal','Radial / concentric','Pure XOR interaction','Threshold interaction',r'Sinusoidal + 497 noise predictors ($p=500$)','Nonlinear heterogeneous covariance']
REG=list(csv.DictReader((ROOT/'mlj_revision/supplement_methods/tables/csv/tab_s_core_scenario_registry.csv').open(encoding='utf-8-sig')))
def core_label(i,design):
 r=REG[i-1];fam=r['scenario_family'];ts=design=='two_sample'
 pi=float(r['ts_pi' if ts else 'os_pi']);c=float(r['os_c']);np0=int(r['ts_nP'])
 if fam=='reference':return 'Reference'
 if fam=='prevalence':return rf'Prevalence $\pi={pi:.2f}$'
 if fam=='positive-label availability':return rf'Labeled positives $n_P={np0}$' if ts else rf'Labeling probability $c={c:.2f}$'
 if fam=='class separation':return rf'Class separation $\delta={float(r["delta"]):.2f}$'
 if fam=='dimensionality':return rf'High dimension ($p={int(r["p"])}$)'
 if fam=='predictor correlation':return rf'Correlation $\rho={float(r["rho"]):.2f}$'
 if fam=='joint prevalence-label availability':return rf'Prevalence $\pi={pi:.2f}$, '+(rf'$n_P={np0}$' if ts else rf'$c={c:.2f}$')
 return fam[0].upper()+fam[1:]
def label(x,design='one_sample'):
 s=str(x)
 for old,new in [('credit_default','Credit Default'),('dry_bean','Dry Bean'),('online_shoppers','Online Shoppers'),('optical_digits','Optical Digits')]:s=s.replace(old,new)
 s=s.replace("Task 1C's pilot rule",'a pilot rule')
 s=re.sub(r'\bNL0([1-6])\b',lambda m:NL[int(m[1])-1],s)
 s=re.sub(r'\b(TS|S)(\d{3})\b',lambda m:core_label(int(m[2]),'two_sample' if m[1]=='TS' else design),s)
 s=s.replace('Task-2','original benchmark').replace('Task 2','original benchmark').replace('Task 6','estimated-prior experiment').replace('Task-4','fixed-parameter experiment').replace('Task-7B','unique-score analysis')
 for old,new in [('credit default','Credit Default'),('dry bean','Dry Bean'),('htru2','HTRU2'),('online shoppers','Online Shoppers'),('optical digits','Optical Digits'),('phishing','Phishing Websites'),('rice','Rice'),('spambase','Spambase'),('c005','budget 0.05'),('c020','budget 0.20')]:s=re.sub(r'\b'+old+r'\b',new,s)
 return s.replace('A_same_model quantile','Same-model quantile').replace('A_same_model','Same-model quantile')
def texescape(s):
 # Protect math while escaping prose.
 parts=re.split(r'(\$[^$]*\$)',str(s))
 for i in range(0,len(parts),2):
  for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:parts[i]=parts[i].replace(a,b)
 return ''.join(parts)
def write_map():
 rows=[]
 for des in ['one_sample','two_sample']:
  for i,r in enumerate(REG,1):
   setting=r['two_sample_varied_setting' if des=='two_sample' else 'one_sample_varied_setting']
   rows.append(dict(study='core',design=des,scenario_id=f'{"TS" if des=="two_sample" else "S"}{i:03}',scenario_family=r['scenario_family'],parameter_name=r['scenario_family'],parameter_value=setting,short_display_label=core_label(i,des),long_display_label=core_label(i,des)))
  for i,n in enumerate(NL,1):rows.append(dict(study='nonlinear',design=des,scenario_id=f'NL{i:02}',scenario_family=['sinusoidal','radial','XOR','threshold interaction','sinusoidal','heterogeneous covariance'][i-1],parameter_name='frozen nonlinear mechanism',parameter_value='unchanged approved calibration',short_display_label=n,long_display_label=n))
 with (HOME/'source_data/scenario_display_labels.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

