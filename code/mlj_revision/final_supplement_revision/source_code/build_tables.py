"""Table authoring from archived numeric displays. No fitting."""
import sys,re,shutil,json,subprocess
import pandas as pd,numpy as np
from labels import *
OLD=ROOT/'mlj_revision/display_package'
sys.path.insert(0,str(OLD))
import display_tables as dt
from display_specs import DISPLAYS,METHODS,DESIGNS,DNAME,REPS
def table_tex(stem,t,caption,note):
 cols=[c for c in t.columns if c not in ['Panel','scenario_id']]
 total=7.0-(len(cols)-1)*6/72.27
 if stem=='tab_s_bayes_regret':widths=[2.05]+[(total-2.05)/(len(cols)-1)]*(len(cols)-1)
 elif stem=='tab_s_rn_mechanism':widths=[2.00]+[(total-2)/(len(cols)-1)]*(len(cols)-1)
 elif stem=='tab_s_estimated_prior':widths=[1.65,1.30]+[(total-2.95)/(len(cols)-2)]*(len(cols)-2)
 elif cols[0]=='Strategy':widths=[1.35,1.0]+[(total-2.35)/(len(cols)-2)]*(len(cols)-2)
 else:widths=[2.05]+[(total-2.05)/(len(cols)-1)]*(len(cols)-1)
 spec='@{}'+''.join(r'>{\raggedright\arraybackslash}p{'+f'{w:.3f}in'+'}' for w in widths)+'@{}'
 titlemap={'Logistic probability correction':'Logistic correction','PU gradient boosting':'PU gradient boosting','TN retained / available':'TN retained / available','n / requested':'n / target'}
 head=' & '.join(texescape(titlemap.get(c,c)) for c in cols)+r' \\'
 groups=list(t.groupby('Panel',sort=False,dropna=False)) if 'Panel'in t else [('',t)]
 lines=[r'\begingroup',r'\small',r'\setlength{\tabcolsep}{3pt}',r'\setlength{\LTpre}{4pt}\setlength{\LTpost}{8pt}',r'\renewcommand{\arraystretch}{'+('1.0' if stem=='tab_s_rn_mechanism' else '1.08')+'}']
 for j,(panel,g) in enumerate(groups):
  if stem=='tab_s_computational_time':
   names=[c if c=='Method' else c+(' (ms)' if 'Incremental' in panel else ' (min)' if '$p=500$' in c else ' (s)') for c in cols]
   head=' & '.join(texescape(c) for c in names)+r' \\'
  if (stem=='tab_s_probability_metrics' and j==2) or (stem=='tab_s_nonlinear_secondary_metrics' and j in [4,8]) or (stem=='tab_s_rn_selector_sensitivity' and j==5) or (stem=='tab_s_computational_time' and panel.endswith('Native PU boosting')) or (stem=='tab_s_supplied_prior_sensitivity' and j and j%6==0):lines.append(r'\clearpage')
  lines.append(r'\begin{longtable}{'+spec+'}')
  if j==0:lines.append(r'\caption{'+texescape(caption)+r'}\label{tab:'+stem.removeprefix('tab_').replace('_','-')+r'}\\')
  sec=r'\multicolumn{'+str(len(cols))+r'}{@{}p{7in}}{\textbf{'+texescape(panel)+r'}}\\' if panel else ''
  lines.extend([r'\toprule',sec,head,r'\midrule',r'\endfirsthead',r'\toprule',sec,head,r'\midrule',r'\endhead',r'\bottomrule',r'\endfoot'])
  for _,r in g.iterrows():lines.append(' & '.join(texescape(r[c]) for c in cols)+r' \\')
  lines.append(r'\end{longtable}')
 lines.extend([r'\noindent\textit{Note.} '+texescape(note),r'\par\endgroup'])
 return '\n'.join(lines)+'\n'
def save(stem,t,caption,note,compile=True):
 t.to_csv(HOME/'revised_tables'/f'{stem}.csv',index=False)
 (HOME/'revised_tables'/f'{stem}.tex').write_text(table_tex(stem,t,caption,note),encoding='utf8')
 if compile:compile_table(stem)
def compile_table(stem):
 build=HOME/'audits/tex_build';build.mkdir(exist_ok=True)
 wrapper=r'\documentclass[11pt,letterpaper]{article}'+'\n'+r'\usepackage[margin=0.65in]{geometry}\usepackage{booktabs,array,longtable,amsmath,textcomp,tabularx,caption,multirow}\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{microtype}'+'\n'+r'\begin{document}'+'\n'+r'\input{'+(HOME/'revised_tables'/f'{stem}.tex').as_posix()+'}\n'+r'\end{document}'
 p=build/f'{stem}.tex';p.write_text(wrapper,encoding='utf8')
 for _ in range(2):
  r=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory',str(build),str(p)],capture_output=True,text=True)
  if r.returncode:raise RuntimeError(r.stdout[-5000:])
 log=(build/f'{stem}.log').read_text(errors='replace')
 over=re.findall(r'Overfull[^\n]+',log)
 shutil.copy2(build/f'{stem}.pdf',HOME/'revised_tables'/f'{stem}.pdf')
 print(stem,'overflow',len(over),flush=True)
 return over
def main():
 inventory=[]
 for ident,spec in DISPLAYS.items():
  if not ident.startswith('tab_s_'):continue
  p=OLD/'supplement/tables'/f'{ident}.csv'
  t=pd.read_csv(p,dtype=str,keep_default_na=False)
  t=t.rename(columns={'A_same_model':'Same-model quantile'})
  for i,r in t.iterrows():
   des='two_sample' if 'Two-sample' in r.get('Panel','') or 'two_sample' in r.get('Panel','') else 'one_sample'
   for col in t:t.at[i,col]=label(r[col],des)
  if 'Panel' in t:
   t['Panel']=t.Panel.str.replace('main, , ','simulations / ',regex=False).str.replace('real, c005, ','real data, budget 0.05 / ',regex=False).str.replace('real, c020, ','real data, budget 0.20 / ',regex=False).str.replace(', c=',', budget ',regex=False)
  # Use authoritative design-specific labels for the generic few-label shorthand.
  if ident=='tab_s_neural_simulation':
   t['Panel']=t['Panel'].map(lambda s:s.replace('Few labels',core_label(12,'two_sample' if s.startswith('Two') else 'one_sample')).replace('500 predictors',core_label(26,'one_sample')).replace(' - Curved',' - Curved boundary').replace(' - Interaction',' - Feature interaction').replace(' - Unequal covariance',' - Unequal class covariance'))
  if 'rn_selector' in ident:
   t['Panel']=t['Panel'].str.replace('500 predictors',core_label(26,'one_sample'),regex=False).str.replace(' - Curved',' - Curved boundary',regex=False).str.replace(' - Interaction',' - Feature interaction',regex=False).str.replace(' - Unequal covariance',' - Unequal class covariance',regex=False)
  # Existing saved notes are read from their immutable TeX source.
  text=(p.with_suffix('.tex')).read_text(encoding='utf8')
  note=text.split(r'\noindent\textit{Note.} ')[-1].split(r'\par\endgroup')[0].strip()
  note=note.replace(r'\_','_').replace(r'\%','%').replace(r'\&','&')
  note=label(note)
  note=note.replace('Sxxx identifies the corresponding original setting (TSxxx in two-sample). ','').replace('Scenario identifiers correspond to Task 2. ','')
  note=note.replace('Task-2','original benchmark').replace('Task 2','original benchmark').replace('Task-7B','saved unique-score').replace('Task-4','fixed-parameter')
  save(ident,t,label(spec['caption']),note)
  src=OLD/'audit/source_checks'/f'{ident}_numeric.csv'
  if src.exists():shutil.copy2(src,HOME/'source_data'/src.name)
  inventory.append(dict(display=ident,action='Regenerated from existing displayed values; descriptive labels; numeric source preserved'))
 # Full nonlinear secondary table.
 p=OLD/'supplement/tables/tab_s_nonlinear_secondary_metrics.csv'
 d=pd.read_csv(p);assert len(d)==72 and all(d[c].eq(100).all() for c in ['BA_n','F1_n','MCC_n'])
 d['scenario_id']=d.scenario;d['scenario_description']=d.scenario.map(label)
 d.to_csv(HOME/'source_data/nonlinear_secondary_full_precision.csv',index=False)
 rows=[]
 for (des,sc),g in d.groupby(['design','scenario'],sort=False):
  for _,r in g.iterrows():rows.append({'Panel':DNAME[des]+' - '+label(sc),'Method':r.method_name.replace('CS CatBoost','CS-CatBoost').replace('RN CatBoost','RN-CatBoost'),'Balanced accuracy':f'{r.BA:.4f}','F1':f'{r.F1:.4f}','MCC':f'{r.MCC:.4f}','n':'100','scenario_id':sc})
 caption='Secondary threshold-based performance in the nonlinear simulation extension. Values are means over 100 repetitions under the common same-model prevalence-quantile decision rule. Balanced accuracy, F1, and Matthews correlation coefficient (MCC) are secondary threshold-dependent measures. In the one-sample design the threshold is estimated from the full validation population; in the two-sample design it is estimated from validation U only. The selected training-sample model is used both to determine the validation threshold and to score the test sample.'
 save('tab_s_nonlinear_secondary_metrics',pd.DataFrame(rows),caption,'The six methods are prespecified fixed representatives. Corrected methods use raw source-membership scores. No numerical threshold is transferred to a refitted model. All 7,200 displayed method/repetition records are available; unavailable RN final refits do not affect these selected-model hard metrics.')
 pd.DataFrame(inventory).to_csv(HOME/'audits/TABLE_REGENERATION.csv',index=False)
 (HOME/'audits/SECONDARY_METRIC_PUBLICATION_DECISION.md').write_text('# Secondary-metric publication decision\n\nRetain the complete nonlinear table: 6 DGPs x 2 designs x 100 repetitions x 6 prespecified methods, with BA/F1/MCC available throughout. The final same-model rule is verified in the existing secondary-metric audit and Task-7 records.\n\nOmit the original-core pilot from the publication displays. The three original core secondary CSVs provide only 400 verified practical-PU method/case results out of 200,000 requested (8 settings per design, one repetition, 25 methods), not a complete 40-setting or ten-family analysis. No full-family BA/F1/MCC mean is computed. The sources and old pilot remain unchanged. The threshold-rule diagnostic remains explicitly a one-repetition sensitivity analysis, not a core benchmark estimate.\n',encoding='utf8')
if __name__=='__main__':main()



