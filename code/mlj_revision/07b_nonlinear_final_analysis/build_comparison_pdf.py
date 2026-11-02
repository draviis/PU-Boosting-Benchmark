"""Publication tables from existing analysis JSON; no simulation or model execution."""
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.pagesizes import landscape,A4
from reportlab.pdfbase.pdfmetrics import stringWidth
H=Path(__file__).resolve().parent
T=json.loads((H/'analysis_tables.json').read_text())
W,HT=landscape(A4);L=40;RW=W-80
LABELS={'001':'Gaussian reference','026':'Gaussian, p = 500','038':'Curved boundary','039':'Interaction + main effect','040':'Unequal covariance',
'NL01':'Sinusoidal','NL02':'Radial','NL03':'Pure XOR','NL04':'Threshold interaction','NL05':'Sinusoidal, p = 500','NL06':'Nonlinear heterogeneity'}
METHODS=['corrected_logistic','nnPU_logistic','naive_CatBoost','CS_CatBoost','PU_gradient_boosting']
def title(c,txt,sub,page):
 c.setFillColor(black);c.setFont('Helvetica-Bold',16);c.drawString(L,HT-34,txt)
 c.setFont('Helvetica',9.5);c.drawString(L,HT-51,sub)
 c.setFont('Helvetica',8);c.setFillColor(HexColor('#555555'));c.drawRightString(W-L,20,str(page))
def foot(c,lines):
 c.setFillColor(HexColor('#444444'));c.setFont('Helvetica',8)
 for i,line in enumerate(lines):c.drawString(L,42-11*i,line)
def draw_table(c,headers,rows,widths,top,dy=16,break_after=5,highlight_from=None,body_font=9.5):
 assert abs(sum(widths)-RW)<.01
 xs=[L]
 for w in widths:xs.append(xs[-1]+w)
 headerheight=29
 c.setStrokeColor(black);c.setLineWidth(.7);c.line(L,top,W-L,top)
 c.setFont('Helvetica-Bold',9);c.setFillColor(black)
 for j,head in enumerate(headers):
  lines=head.split('\n')
  for k,line in enumerate(lines):
   assert stringWidth(line,'Helvetica-Bold',9)<widths[j]-6,(line,widths[j])
   y=top-11-k*10
   if j==0:c.drawString(xs[j]+4,y,line)
   else:c.drawCentredString((xs[j]+xs[j+1])/2,y,line)
 c.setLineWidth(.45);c.line(L,top-headerheight,W-L,top-headerheight)
 for i,row in enumerate(rows):
  yt=top-headerheight-i*dy
  if highlight_from is not None:
   c.setFillColor(HexColor('#F0F3F7'));c.rect(xs[highlight_from],yt-dy,xs[-1]-xs[highlight_from],dy,fill=1,stroke=0)
  if break_after is not None and i==break_after:
   c.setStrokeColor(HexColor('#777777'));c.setLineWidth(.45);c.line(L,yt,W-L,yt)
  for j,v in enumerate(row):
   font='Helvetica-Bold' if highlight_from is not None and j>=highlight_from else 'Helvetica'
   c.setFont(font,body_font);c.setFillColor(black)
   assert stringWidth(str(v),font,body_font)<widths[j]-8,(v,widths[j])
   if j==0:c.drawString(xs[j]+4,yt-dy+4,str(v))
   else:c.drawRightString(xs[j+1]-7,yt-dy+4,str(v))
 c.setLineWidth(.6);c.line(L,top-headerheight-len(rows)*dy,W-L,top-headerheight-len(rows)*dy)
def label(r):
 sc=r['scenario_id'];return sc+'  '+LABELS[sc if sc.startswith('NL') else sc[-3:]]
def main():
 p=H/'original_vs_nonlinear_comparison.pdf';assert not p.exists()
 c=canvas.Canvas(str(p),pagesize=(W,HT),pageCompression=1)
 c.setTitle('Original and nonlinear PU simulations: fixed-method comparison')
 c.setAuthor('PU Boosting Benchmark')
 heads=['DGP','Bayes\nAUROC','Corrected\nlogistic','nnPU\nlogistic','Naive\nCatBoost','CS\nCatBoost','PU gradient\nboosting']
 for page,metric in enumerate(['AUROC','AP','regret'],1):
  isap=metric=='AP';isreg=metric=='regret'
  title(c,'Original and nonlinear simulations' if not isreg else 'AUROC regret relative to Bayes',
        'Average precision (AP); final train+validation refits.' if isap else 'Bayes AUROC minus method AUROC, in percentage points; lower is better.' if isreg else 'AUROC; final train+validation refits. Each method mean uses 100 repetitions.',page)
  for des,top in [('one_sample',HT-83),('two_sample',HT-326)]:
   c.setFillColor(black);c.setFont('Helvetica-Bold',11);c.drawString(L,top+10,des.replace('_',' ').capitalize())
   rows=[]
   for r in T['original_vs_nonlinear_comparison.csv']:
    if r['design']!=des:continue
    row=[label(r)]+([] if isap else [f"{r['Bayes_AUROC']:.4f}"])
    row += [f"{100*r[k+'_AUROC_regret']:.2f}" if isreg else f"{r[k+'_'+metric]:.3f}" for k in METHODS]
    rows.append(row)
   nh=heads[:1]+heads[2:] if isap else heads
   widths=[205]+[(RW-205)/(len(nh)-1)]*(len(nh)-1)
   draw_table(c,nh,rows,widths,top)
  foot(c,['First five rows: original contexts. Next six: nonlinear extension. Fixed methods throughout; no group envelopes.',
          'Sources: existing Task 2 and Task 7 outputs. Bayes Monte Carlo uncertainty and method MCSEs are retained in the CSV.'])
  c.showPage()
 title(c,'Nonlinear extension: headline AUROC results','Fixed methods and paired CatBoost learner/strategy differences. Bold columns are AUROC percentage points.',4)
 headers=['DGP','Bayes','Corrected\nlogistic','Naive\nCatBoost','CS\nCatBoost','RN\nCatBoost','PU gradient\nboosting','Learner\ngain (pp)','CS gain\n(pp)','RN gain\n(pp)']
 widths=[175]+[(RW-175)/9]*9
 for des,top in [('one_sample',HT-92),('two_sample',HT-295)]:
  c.setFont('Helvetica-Bold',11);c.setFillColor(black);c.drawString(L,top+12,des.replace('_',' ').capitalize())
  rows=[]
  for r in T['nonlinear_headline_results.csv']:
   if r['design']!=des:continue
   vals=[label(r),f"{r['Bayes_AUROC']:.3f}"]+[f"{r[k+'_AUROC']:.3f}"+('*' if k=='RN_CatBoost' and r['RN_CatBoost_n']==99 else '') for k in ['corrected_logistic','naive_CatBoost','CS_CatBoost','RN_CatBoost','PU_gradient_boosting']]
   vals += [f"{r[k+'_mean_difference_pp']:+.2f}" for k in ['learner_gain','CS_additional_gain','RN_additional_gain']]
   rows.append(vals)
  draw_table(c,headers,rows,widths,top,dy=21,break_after=None,highlight_from=7,body_font=9)
 foot(c,['Learner gain = naive CatBoost - corrected logistic; CS/RN gain = strategy CatBoost - naive CatBoost.',
         '* 99 successful refits; paired RN change uses those same 99 pairs. No imputation. All other displayed means use 100.'])
 c.showPage();c.save();print(p)
if __name__=='__main__':main()
