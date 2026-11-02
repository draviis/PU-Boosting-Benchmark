"""Scientific fixed-method figure; all individual DGP cells, no envelopes."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
H=Path(__file__).resolve().parent
def main():
    data=json.loads((H/'analysis_tables.json').read_text())['nonlinear_scenario_summary.csv']
    methods=[f'M{i:03d}' for i in range(3,28)]+['M001','M002']
    names={r['method_id']:r['method'].replace('Block0 | ','').replace('Block1 | ','').replace('Block2 | ','').replace('Block3 | ','').replace(' repaired','').replace('Oracle ','Supervised ').replace('CaseControl density-ratio known-pi','Corrected logistic (known pi)').replace('Elkan-Noto known-c','Corrected logistic (known c)') for r in data if r['design']=='one_sample'}
    names['M003']='Corrected logistic'
    titles={'NL01':'NL01\nSinusoidal','NL02':'NL02\nRadial','NL03':'NL03\nXOR','NL04':'NL04\nThreshold','NL05':'NL05\nSinusoidal\np=500','NL06':'NL06\nHetero.\ncovariance'}
    scs=list(titles);pdf=H/'nonlinear_fixed_method_comparison.pdf';assert not pdf.exists()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    with PdfPages(pdf) as pages:
        for design in ['one_sample','two_sample']:
            records={(r['method_id'],r['scenario_id']):r for r in data if r['design']==design}
            fig,axes=plt.subplots(1,3,figsize=(16.5,12),sharey=True)
            fig.subplots_adjust(left=.25,right=.985,bottom=.13,top=.86,wspace=.10)
            for ax,metric,title,limits in zip(axes,['AUROC','AP','BA'],['AUROC - final refit','AP - final refit','BA - selected train-only'],[(.45,.9),(.15,.8),(.45,.8)]):
                matrix=np.array([[records[(m,s)][metric+'_mean'] for s in scs] for m in methods])
                im=ax.imshow(matrix,cmap='viridis',aspect='auto',vmin=limits[0],vmax=limits[1])
                ax.set_xticks(range(6),[titles[s] for s in scs],fontsize=8);ax.tick_params(top=False,bottom=True,labelbottom=True)
                ax.set_yticks(range(len(methods)),[names[m] for m in methods]);ax.set_title(title,pad=12,fontsize=12,weight='bold')
                ax.axhline(24.5,color='white',lw=3)
                for row in range(len(methods)):
                    for col in range(6):
                        color='black' if (matrix[row,col]-limits[0])/(limits[1]-limits[0])>.65 else 'white'
                        suffix='*' if records[(methods[row],scs[col])][metric+'_n']<100 else ''
                        ax.text(col,row,f'{matrix[row,col]:.3f}'+suffix,ha='center',va='center',fontsize=7.3,color=color)
                fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.027,pad=.09)
                for spine in ax.spines.values():spine.set_visible(False)
            fig.suptitle('Nonlinear extension | '+design.replace('_',' ')+' | 100 repetitions per DGP',x=.25,ha='left',y=.96,fontsize=17,weight='bold')
            fig.text(.25,.915,'25 fixed PU methods; two supervised references below the white divider.',fontsize=11)
            fig.text(.25,.89,'Individual DGP means are primary. Naive/corrected pairs share fitted ranking scores.',fontsize=10,color='#374151')
            fig.text(.02,.035,'BA uses validation prevalence-quantile thresholds and test scores from the same selected train-only model.\nAll validation observations in one-sample; validation U only in two-sample. Supervised references are excluded from PU ranks.\n* Ranking mean uses 99 successful refits; the empty RN refit is unavailable. All BA cells use 100 selected models.\nMCSEs, paired differences and both aggregate weighting schemes are supplied in the accompanying CSV tables.',fontsize=9,color='#374151')
            pages.savefig(fig);fig.savefig(H/(design+'_comparison_preview.png'),dpi=100);plt.close(fig)
    print(pdf)
if __name__=='__main__':main()
