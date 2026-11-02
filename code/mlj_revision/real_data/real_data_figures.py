"""Matplotlib-only publication figures from archived real-data results."""
from real_data_common import *
import os
os.environ.setdefault("MPLCONFIGDIR",str(HOME/"audit/logs/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from real_data_tables import neural_summary
COLORS=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#6B6B6B"]
MARKERS=["o","s","^","D","v","P"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":11,"axes.labelsize":10,
                     "xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":9,
                     "pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,
                     "savefig.dpi":220})

def wrapper(stem,caption,label,location,pages=1):
    lines=[]
    for p in range(1,pages+1):
        lines += [r"\begin{figure*}[t]",r"\centering"]
        if p>1:lines.append(r"\ContinuedFloat")
        lines += [r"\includegraphics[width=\textwidth"+(f",page={p}" if pages>1 else "")+r"]{"+location+"/figures/"+stem+r".pdf}",
                  r"\caption{"+caption+(f" Page {p} of {pages}." if pages>1 else "")+"}",
                  r"\label{"+label+("" if p==1 else f"-p{p}")+"}",r"\end{figure*}"]
        if p<pages:lines.append(r"\clearpage")
    (HOME/location/"figure_tex"/(stem+".tex")).write_text("\n".join(lines)+"\n",encoding="utf-8")

def style_axis(ax,zero=False):
    ax.grid(axis="x",color="#D9D9D9",linewidth=.5)
    ax.set_axisbelow(True)
    if zero:ax.axvline(0,color="#444444",linewidth=.85,zorder=1)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis="y",length=0)

def check_layout(fig,stem,page):
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bad=[];sizes=[]
    # Locators construct ticks outside the view limits, but Axis.draw omits them.
    undrawn_tick_labels=set()
    for ax in fig.axes:
        for axis in (ax.xaxis,ax.yaxis):
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not lo-1e-10<=tick.get_loc()<=hi+1e-10:
                    undrawn_tick_labels.update((tick.label1,tick.label2))
    for artist in fig.findobj(match=Text):
        if artist in undrawn_tick_labels or not artist.get_visible() or not artist.get_text().strip(): continue
        sizes.append(artist.get_fontsize())
        box=artist.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
            bad.append(artist.get_text())
    assert sizes and min(sizes)>=8.5,(stem,min(sizes))
    assert not bad,(stem,bad)
    js(dict(figure=stem,page=page,width_inches=float(fig.get_figwidth()),minimum_font_pt=min(sizes),
            text_outside_canvas=bad),HOME/"audit/logs"/f"{stem}_layout_page{page}.json")

def save_single(fig,stem,location):
    check_layout(fig,stem,1)
    target=HOME/location/"figures"
    fig.savefig(target/(stem+".pdf"),metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(target/(stem+".png"))
    plt.close(fig)

def decomposition():
    require_audit();d=paired_dataset(paired(core()))
    csv(d,HOME/"main/data/fig_real_decomposition.csv")
    fig,axes=plt.subplots(2,2,figsize=(7.2,7.5),sharey=True)
    fig.subplots_adjust(left=.225,right=.985,bottom=.18,top=.89,wspace=.13,hspace=.35)
    labels=list(DATASETS.values());y=np.arange(8)
    for row,metric in enumerate(METRICS):
        subset=d[d.metric==metric]
        limit=math.ceil(max(abs(subset.split_p10_pp).max(),abs(subset.split_p90_pp).max(),abs(subset.difference_pp).max())/5)*5
        for col,design in enumerate(DESIGNS):
            ax=axes[row,col];style_axis(ax,True)
            for k,contrast in enumerate(CONTRASTS):
                z=subset[(subset.design==design)&(subset.contrast==contrast)].set_index("dataset_id").loc[list(DATASETS)]
                ys=y+(k-1)*.19
                ax.hlines(ys,z.split_p10_pp,z.split_p90_pp,color=COLORS[k],linewidth=1.0,alpha=.65)
                ax.scatter(z.difference_pp,ys,color=COLORS[k],marker=MARKERS[k],s=21,zorder=3,edgecolors="white",linewidths=.35)
            ax.set_yticks(y,labels);ax.set_ylim(7.6,-.6);ax.set_xlim(-limit,limit)
            ax.set_title(f"({chr(97+row*2+col)}) {'One-sample' if col==0 else 'Two-sample'} {metric}",loc="left")
            ax.set_xlabel("Difference (percentage points)")
    handles=[Line2D([0],[0],color=COLORS[k],marker=MARKERS[k],linestyle="-",markersize=4,label=c) for k,c in enumerate(CONTRASTS)]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.59,.975),ncol=3,frameon=False,columnspacing=1.2)
    fig.text(.225,.045,"Points: paired means, equal budgets. Bars: 10th-90th percentile split spread.\nRepeated splits overlap; bars are not confidence intervals.",fontsize=8.5,ha="left")
    save_single(fig,"fig_real_decomposition","main")
    wrapper("fig_real_decomposition","Real-data learner-versus-strategy decomposition. Learner gain is naive CatBoost minus logistic probability correction; CS and RN additions compare the respective PU strategy with naive CatBoost. Each dataset point averages 20 paired repeated splits within each label budget and then the two budgets equally. Bars show the empirical 10th--90th percentile of the 40 paired split contrasts, a descriptive balanced-budget mixture. Repeated splits are not independent datasets; these bars are not confidence intervals. Dataset order is fixed, and no winner or group maximum is selected.","fig:real-decomposition","main")
    print(f"Display fig_real_decomposition: CSV rows={len(d)}; pages=1")

def fixed_methods():
    require_audit();d=condition_means(core());d=d[d.number.isin(REPS)].copy()
    csv(d,HOME/"supplement/data/fig_s_real_fixed_methods.csv")
    dest=HOME/"supplement/figures";stem="fig_s_real_fixed_methods"
    pages=[]
    with PdfPages(dest/(stem+".pdf"),metadata={"CreationDate":None,"ModDate":None}) as pdf:
        for page,(metric,budget) in enumerate([(m,b) for m in METRICS for b in BUDGETS],1):
            fig,axes=plt.subplots(1,2,figsize=(7.2,5.7),sharey=True)
            fig.subplots_adjust(left=.225,right=.985,bottom=.17,top=.76,wspace=.13)
            for j,design in enumerate(DESIGNS):
                ax=axes[j];style_axis(ax)
                for k,num in enumerate(REPS):
                    z=d[(d.design==design)&(d.budget==budget)&(d.number==num)].set_index("dataset_id").loc[list(DATASETS)]
                    ax.scatter(z[metric],np.arange(8)+(k-2.5)*.105,color=COLORS[k],marker=MARKERS[k],s=21)
                ax.set_yticks(np.arange(8),list(DATASETS.values()));ax.set_ylim(7.6,-.6);ax.set_xlim(0,1)
                ax.set_title(("One-sample" if j==0 else "Two-sample")+f"; budget {budget:.2f}",loc="left")
                ax.set_xlabel(metric)
            handles=[Line2D([0],[0],color=COLORS[k],marker=MARKERS[k],linestyle="",markersize=4,label=METHODS[num]) for k,num in enumerate(REPS)]
            fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.54,.985),ncol=2,frameon=False,columnspacing=1.1,handletextpad=.45)
            fig.text(.225,.045,f"{metric}: final-refit means over 20 splits. Absolute scale 0 to 1.",fontsize=8.5)
            check_layout(fig,stem,page)
            pdf.savefig(fig)
            png=dest/(stem+(f"_page{page:02d}" if page>1 else "")+".png");fig.savefig(png)
            pages.append(png.name);plt.close(fig)
    wrapper(stem,"Fixed-method real-data ranking performance, retaining each dataset, design and budget. Points are means over 20 repeated splits for six conceptually selected configurations, using final train+validation refits. AUROC and AP are shown on absolute 0--1 scales. Dataset order is fixed; supervised references and group maxima are excluded.","fig:s-real-fixed-methods","supplement",4)
    print(f"Display {stem}: CSV rows={len(d)}; pages=4; PNGs={pages}")

def neural_figure():
    require_audit();c,d=neural_summary()
    csv(d,HOME/"supplement/data/fig_s_real_neural.csv")
    fig,axes=plt.subplots(2,2,figsize=(7.2,7.5),sharey=True)
    fig.subplots_adjust(left=.225,right=.985,bottom=.18,top=.9,wspace=.13,hspace=.36)
    for row,metric in enumerate(METRICS):
        limit=max(1,math.ceil(d[d.metric==metric].difference_pp.abs().max()))
        for col,selection in enumerate(["auc","risk"]):
            ax=axes[row,col];style_axis(ax,True)
            for k,risk in enumerate(["upu","nnpu"]):
                z=d[(d.metric==metric)&(d.selection==selection)&(d.risk==risk)].set_index("dataset_id").loc[list(DATASETS)]
                ax.scatter(z.difference_pp,np.arange(8)+(k-.5)*.22,color=COLORS[k],marker=MARKERS[k],s=25)
            ax.set_yticks(np.arange(8),list(DATASETS.values()));ax.set_ylim(7.6,-.6);ax.set_xlim(-limit,limit)
            ax.set_title(f"({chr(97+row*2+col)}) {metric}: "+("label-AUC selection" if col==0 else "PU-risk selection"),loc="left",fontsize=11)
            ax.set_xlabel("Neural - linear (percentage points)")
    handles=[Line2D([0],[0],color=COLORS[k],marker=MARKERS[k],linestyle="",label=risk,markersize=5) for k,risk in enumerate(["uPU","nnPU"])]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.59,.98),ncol=2,frameon=False)
    fig.text(.225,.045,"Equal average of four design-budget conditions within each dataset.\nFixed-budget extension; training-cap limitations retained.",fontsize=8.5)
    save_single(fig,"fig_s_real_neural","supplement")
    wrapper("fig_s_real_neural","Matched neural-minus-linear uPU/nnPU ranking performance. Paired differences are first averaged over 20 splits within a condition and then over the four design-budget conditions within each dataset. Label-AUC selection uses observed validation P/U labels; PU-risk selection uses empirical PU risk. These are the frozen extension baselines, not the core catalog's linear results. Repeated splits overlap and no population-inference intervals are shown. The existing fixed-budget limitation remains: 522/5,120 selected fits reached 3,000 epochs; verification of outputs does not establish convergence.","fig:s-real-neural","supplement")
    print(f"Display fig_s_real_neural: CSV rows={len(d)}; pages=1")

FUNCTIONS={"fig_real_decomposition":decomposition,"fig_s_real_fixed_methods":fixed_methods,"fig_s_real_neural":neural_figure}
