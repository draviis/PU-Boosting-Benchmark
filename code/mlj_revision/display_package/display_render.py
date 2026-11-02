"""Output layer. Writes only new files within the display package."""
import json,hashlib,os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.text import Text
from display_specs import *
from display_data import data_for
from display_tables import table_values,latex,esc
from display_figures import figures
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def check_sources(id):
 m=pd.read_csv(HOME/"SOURCE_MANIFEST.csv",keep_default_na=False)
 rows=m[m.display_id==id]
 assert len(rows)>0
 missing=[]
 for _,r in rows.iterrows():
  p=ROOT/r.source_file
  if not p.is_file():missing.append(r.source_file);continue
  assert sha(p)==r.sha256,f"Source changed after manifest: {p}"
 if missing:raise FileNotFoundError("; ".join(missing))
def newwrite(p,text):
 with p.open("x",encoding="utf-8",newline="") as f:f.write(text)
def csv(d,p):
 with p.open("x",encoding="utf-8",newline="") as f:d.to_csv(f,index=False,float_format="%.17g",na_rep="")
def figcheck(fig):
 # Remove locator ticks outside the visible range, which are not drawn but have text objects.
 for ax in fig.axes:
  xl=ax.get_xlim();yl=ax.get_ylim()
  ax.set_xticks([v for v in ax.get_xticks() if min(xl)-1e-9 <= v <= max(xl)+1e-9])
  ax.set_yticks([v for v in ax.get_yticks() if min(yl)-1e-9 <= v <= max(yl)+1e-9])
  ax.set_xlim(xl);ax.set_ylim(yl)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
 texts=[];outside=[];small=[]
 for t in fig.findobj(Text):
  if not t.get_visible() or not t.get_text():continue
  if t.get_fontsize()<8.5-1e-7:small.append(t.get_text())
  b=t.get_window_extent(renderer)
  if b.x0<-.5 or b.y0<-.5 or b.x1>bounds.x1+.5 or b.y1>bounds.y1+.5:outside.append(dict(text=t.get_text(),bbox=list(b.bounds)))
  texts.append(dict(text=t.get_text(),fontsize=t.get_fontsize(),bbox=list(b.bounds)))
 # Cross-axis ticks are not checked by text extraction; retain all boxes for independent QA.
 return dict(width_inches=fig.get_figwidth(),height_inches=fig.get_figheight(),text_below_8_5pt=small,text_outside_canvas=outside,texts=texts)
def run(id):
 spec=DISPLAYS[id];check_sources(id)
 loc=HOME/spec["location"];isfig=id.startswith("fig_")
 primary=loc/("figures" if isfig else "tables")/(id+(".pdf" if isfig else ".tex"))
 if primary.exists():raise FileExistsError(f"Refusing to overwrite {primary}. Use a fresh --output-root.")
 d=data_for(id);print(f"DISPLAY {id}: {len(d)} post-processed rows",flush=True)
 numeric=HOME/"audit/source_checks"/(id+"_numeric.csv");csv(d,numeric)
 assets=[];checks=[]
 if isfig:
  plotcsv=loc/"figures"/(id+".csv");csv(d,plotcsv);assets.append(plotcsv)
  wrappers=[];pages=0
  with PdfPages(primary,metadata={"Title":id,"Author":"PU Benchmark manuscript display package","Creator":"Read-only matplotlib post-processing","CreationDate":None,"ModDate":None}) as pdf:
   for pages,(fig,subtitle,extra) in enumerate(figures(id,d),1):
    checks.append(figcheck(fig));pdf.savefig(fig)
    preview=loc/"figures"/(id+("" if pages==1 else f"_page{pages:02}")+".png")
    if preview.exists():raise FileExistsError(preview)
    fig.savefig(preview,dpi=300);assets.append(preview);plt.close(fig)
    caption=spec["caption"]+" "+subtitle+". "+(extra or "")
    rel=primary.relative_to(ROOT).as_posix()
    wrappers.append(r"\begin{figure*}[t]"+"\n"+r"\centering"+"\n"+r"\includegraphics[page="+str(pages)+r",width=\textwidth]{"+rel+"}\n"+r"\caption{"+esc(caption)+"}\n"+r"\label{fig:"+id.replace("_","-")+f"-page{pages}"+"}\n"+r"\end{figure*}"+"\n")
  wrap=loc/"figure_tex"/(id+".tex");newwrite(wrap,"\n".join(wrappers));assets.extend([primary,wrap])
 else:
  t,note=table_values(id,d);shown=loc/"tables"/(id+".csv");csv(t,shown)
  text=latex(id,t,note,spec["caption"]);newwrite(primary,text);assets.extend([shown,primary])
  assert r"\scriptsize" not in text and r"\resizebox" not in text
  reread=pd.read_csv(shown,keep_default_na=False,dtype=str)
  assert reread.equals(t.astype(str))
  checks=[dict(displayed_rows=len(t),displayed_columns=list(t),exact_display_csv_roundtrip=True)]
  pages=0
 record=dict(display_id=id,status="generated",numeric_rows=len(d),pages=pages,assets=[dict(path=p.relative_to(HOME).as_posix(),sha256=sha(p)) for p in assets],checks=checks)
 newwrite(HOME/"audit/source_checks"/(id+"_validation.json"),json.dumps(record,indent=2))
 print(f"CREATED {primary.relative_to(ROOT)} ({pages} figure pages)",flush=True)
