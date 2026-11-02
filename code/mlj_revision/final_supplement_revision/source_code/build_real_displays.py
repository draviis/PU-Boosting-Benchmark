"""Regenerate real-data supplement with its existing descriptive dataset labels."""
import sys,shutil,json,os
from pathlib import Path
from labels import ROOT,HOME
os.environ['MPLCONFIGDIR']=str(HOME/'audits/matplotlib')
sys.path.insert(0,str(ROOT/'mlj_revision/real_data'))
import real_data_figures as f,real_data_tables as t
stage=HOME/'audits/real_generation'
for name in ['supplement/figures','supplement/figure_tex','supplement/tables','supplement/data','audit/logs']:(stage/name).mkdir(parents=True,exist_ok=True)
f.HOME=stage;t.HOME=stage
original_layout=f.check_layout
def readable_layout(fig,stem,page):
 from matplotlib.text import Text
 for artist in fig.findobj(Text):
  if artist.get_visible() and artist.get_text():artist.set_fontsize(max(9,artist.get_fontsize()))
 original_layout(fig,stem,page)
f.check_layout=readable_layout
f.fixed_methods();f.neural_figure()
for fn in [t.by_dataset,t.all_methods,t.budget_table,t.agreement_table,t.neural_table,t.rn_table]:fn()
for p in (stage/'supplement/figures').iterdir():shutil.copy2(p,HOME/'revised_figures'/p.name)
for p in (stage/'supplement/data').glob('*.csv'):shutil.copy2(p,HOME/'revised_figures'/p.name)
for p in (stage/'supplement/figure_tex').glob('*.tex'):(HOME/'revised_figures'/p.name).write_text(p.read_text().replace('supplement/figures/','revised_figures/').replace('Task 3','the reliable-negative mechanism audit').replace('Task 4','the selector sensitivity analysis'),encoding='utf8')
from build_tables import compile_table
for p in (stage/'supplement/tables').iterdir():
 if p.suffix=='.tex':(HOME/'revised_tables'/p.name).write_text(p.read_text().replace('Task 3','the reliable-negative mechanism audit').replace('Task 4','the selector sensitivity analysis'),encoding='utf8')
 else:shutil.copy2(p,HOME/'revised_tables'/p.name)
for p in (stage/'supplement/tables').glob('*.tex'):
 if p.stem=='tab_s_real_fixed_methods_by_dataset':
  target=HOME/'revised_tables'/p.name
  tex=target.read_text().replace(r'\small',r'\fontsize{9.5}{11}\selectfont').replace(r'\arraystretch}{1.08}',r'\arraystretch}{1.0}')
  target.write_text(tex,encoding='utf8')
 compile_table(p.stem)

