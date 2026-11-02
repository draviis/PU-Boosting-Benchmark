"""Regenerate figures from saved numeric display sources, preserving every value."""
import sys,re,json,shutil,textwrap,hashlib,os
from pathlib import Path
from labels import *
os.environ['MPLCONFIGDIR']=str(HOME/'audits/matplotlib')
import pandas as pd,numpy as np
OLD=ROOT/'mlj_revision/display_package'
sys.path.insert(0,str(OLD))
import display_figures as fg
from display_specs import DISPLAYS,DESIGNS,DNAME,REPS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.text import Text
fg.SCENARIOS={i:core_label(i,'one_sample') for i in range(1,41)}
def wrapped(s):
 s=s.replace('Sinusoidal + 497 noise predictors', 'Sinusoidal + 497 noise\npredictors').replace('Nonlinear heterogeneous covariance','Nonlinear heterogeneous\ncovariance')
 if len(s)>35 and '$' not in s:s='\n'.join(textwrap.wrap(s,30))
 return s
def fix(fig,ident,subtitle):
 design='two_sample' if 'Two-sample' in subtitle else 'one_sample'
 for ax in fig.axes:
  des='two_sample' if 'Two-sample' in ax.get_title(loc='left') else design
  ticks=ax.get_yticklabels()
  for t in ticks:
   s=label(t.get_text(),des)
   if ident=='fig_s_neural_simulation' and design=='two_sample':s=s.replace(core_label(12,'one_sample'),core_label(12,'two_sample'))
   t.set_text(wrapped(s))
  # FixedFormatter update needed: artist text alone is replaced by formatter on redraw.
  if ticks:ax.set_yticks(ax.get_yticks(),[t.get_text() for t in ticks])
  for loc in ['left','center','right']:
   ax.set_title(label(ax.get_title(loc=loc),des),loc=loc)
 for t in fig.findobj(Text):
  t.set_text(label(t.get_text(),design))
 if ident=='fig_nonlinear_decomposition':
  fig.set_size_inches(7.2,8.4)
  fig.subplots_adjust(left=.405,right=.98,top=.95,bottom=.14,hspace=.55,wspace=.28)
  for i,ax in enumerate(fig.axes):
   ax.set_yticks(range(6),[wrapped(n) for n in NL] if i%2==0 else ['']*6)
   ax.set_title(ax.get_title(loc='left').replace(' AUROC','\nAUROC').replace(' AP','\nAP'),loc='left')
   ax.xaxis.set_major_locator(fg.MaxNLocator(4))
 elif ident=='fig_s_nonlinear_full':
  fig.set_size_inches(7.2,8.2)
  for i,ax in enumerate(fig.axes[:2]):
   ax.set_position([.405,.65-.45*i,.49,.24])
   ax.set_yticks(np.arange(6)+.5,[wrapped(n) for n in NL])
  fig.axes[-1].set_position([.905,.20,.014,.69])
  for t in fig.texts:t.set_position((.405,.975))
 elif ident in ['fig_s_rn_selector_sensitivity','fig_s_neural_simulation']:
  fig.subplots_adjust(left=.31,wspace=.34)
  fig.set_figheight(5.8)
  for t in fig.texts:t.set_x(.31)
 elif ident=='fig_s_estimated_prior':
  fig.subplots_adjust(left=.31,right=.99,wspace=.42)
  for t in fig.texts:t.set_x(.31)
  fig.set_figheight(6.2)
 for ax in fig.axes:
  lo,hi=sorted(ax.get_xlim()); ax.set_xticks([x for x in ax.get_xticks() if lo<=x<=hi]); ax.set_xlim(lo,hi)
 for t in fig.findobj(Text):
  if t.get_visible() and t.get_text():t.set_fontsize(max(9,t.get_fontsize()))
def bayes(d):
 vmin=min(0,d.regret.min());vmax=d.regret.max()
 for des in DESIGNS:
  for start in [1,21]:
   fig=plt.figure(figsize=(7.2,8.7));ax=fig.add_axes([.43,.17,.44,.75])
   g=d[d.design==des];rows=list(range(start,start+20))
   m=g.pivot(index='scenario_index',columns='method_number',values='regret').reindex(index=rows,columns=REPS)
   im=fg.heat(ax,m.to_numpy(),[core_label(i,des) for i in rows],[fg.short(i) for i in REPS],vmin,vmax,'magma_r')
   fg.title(ax,'a',DNAME[des])
   fg.cb(fig,im,fig.add_axes([.895,.17,.014,.75]),'Bayes AUROC - method AUROC')
   yield fig,DNAME[des]+f' / settings {start}--{start+19}','All settings retain their validated analytic or exact-likelihood-ratio Bayes reference; lower regret is better.'
def build(ident):
 src=OLD/'audit/source_checks'/f'{ident}_numeric.csv'
 d=pd.read_csv(src,low_memory=False)
 for col in ['budget','dataset']:
  if col in d:d[col]=d[col].fillna('')
 shutil.copy2(src,HOME/'source_data'/src.name)
 d.to_csv(HOME/'revised_figures'/f'{ident}.csv',index=False)
 dest=HOME/'revised_figures'/f'{ident}.pdf'
 checks=[];tex=[]
 generator=bayes(d) if ident=='fig_s_bayes_regret' else fg.figures(ident,d)
 with PdfPages(dest) as pdf:
  for page,(fig,subtitle,note) in enumerate(generator,1):
   fix(fig,ident,subtitle);fig.canvas.draw()
   bad=[];renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
   for t in fig.findobj(Text):
    if not t.get_visible() or not t.get_text():continue
    b=t.get_window_extent(renderer)
    if b.x0 < -1 or b.y0 < -1 or b.x1>width+1 or b.y1>height+1:bad.append(dict(text=t.get_text(),bounds=list(b.bounds)))
   checks.append(dict(page=page,outside=bad,min_font=min(t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()),text=[t.get_text() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]))
   pdf.savefig(fig)
   fig.savefig(HOME/'revised_figures'/(ident+('' if page==1 else f'_page{page:02}')+'.png'),dpi=160)
   subtitle=subtitle.replace('main /','Simulation /').replace('real /','Real data /').rstrip(' /')
   caption=label(DISPLAYS[ident]['caption']+' '+subtitle+'. '+(note or ''),'two_sample' if 'Two-sample' in subtitle else 'one_sample')
   tex.extend([r'\begin{figure*}[p]',r'\centering',rf'\includegraphics[page={page},width=\textwidth]{{revised_figures/{ident}.pdf}}',r'\caption{'+texescape(caption)+'}',r'\label{fig:'+ident.replace('fig_','').replace('_','-')+f'-{page}'+'}',r'\end{figure*}'])
   plt.close(fig)
 (HOME/'revised_figures'/f'{ident}.tex').write_text('\n'.join(tex)+'\n',encoding='utf8')
 (HOME/'audits'/f'{ident}_layout.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
 print(ident,'pages',len(checks),'outside',sum(len(c['outside']) for c in checks),flush=True)
 if ident=='fig_nonlinear_decomposition' and '--main-only' in sys.argv:
  for ext in ['pdf','png']:shutil.copy2(HOME/'revised_figures'/f'{ident}.{ext}',OLD/'main/figures'/f'{ident}.{ext}')
if __name__=='__main__':
 write_map()
 ids=['fig_nonlinear_decomposition'] if '--main-only' in sys.argv else ['fig_nonlinear_decomposition']+[k for k,v in DISPLAYS.items() if k.startswith('fig_s_')]
 for ident in ids:build(ident)



