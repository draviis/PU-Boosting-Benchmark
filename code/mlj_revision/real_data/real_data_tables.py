"""Manuscript-style real-data tables; all aggregation is within dataset first."""
from real_data_common import *
SHORT={3:"Logistic probability correction",5:"nnPU logistic",10:"Naive CatBoost",25:"CS CatBoost",20:"RN CatBoost",27:"PU gradient boosting"}
CONDITIONS=[(d,b) for d in DESIGNS for b in BUDGETS]
def fmt(v,pp=False):return ("--" if pd.isna(v) else f"{v:+.3f}" if pp else f"{v:.4f}")
def start(caption,label):
    return [r"\begin{table*}[t]",r"\centering",r"\caption{"+caption+"}",r"\label{"+label+"}",r"\small",
            r"\setlength{\tabcolsep}{3.5pt}",r"\renewcommand{\arraystretch}{1.08}"]
def finish(note):
    return [r"\par\medskip",r"\begin{minipage}{\textwidth}\small",note,r"\end{minipage}",r"\end{table*}"]
def save(stem,d,tex=None,location="supplement"):
    csv(d,HOME/location/"tables"/(stem+".csv"))
    if tex is not None:(HOME/location/"tables"/(stem+".tex")).write_text("\n".join(tex)+"\n",encoding="utf-8")
    print(f"Display {stem}: CSV rows={len(d)}")
def fixed_table():
    require_audit();d=condition_means(core());d=d[d.number.isin(REPS)]
    z=d.groupby(["design","budget","number","method_id","method_name"],as_index=False).agg(
        AUROC=("AUROC","mean"),AP=("AP","mean"),n_datasets=("dataset_id","nunique"))
    assert z.n_datasets.eq(8).all()
    rows=[]
    for metric in METRICS:
        for r in z.itertuples():
            best=z[(z.design==r.design)&(z.budget==r.budget)][metric].max()
            rows.append(dict(design=r.design,budget=r.budget,method_id=r.method_id,method_name=r.method_name,number=r.number,
                             metric=metric,value=getattr(r,metric),display=fmt(getattr(r,metric)),
                             bold=getattr(r,metric)==best,n_datasets=8,repetitions_per_dataset=20))
    out=pd.DataFrame(rows)
    lines=start("Fixed-method real-data ranking performance. Repeated splits are first averaged within dataset; each of the eight datasets then receives equal weight. Representative methods were selected for conceptual coverage, not by outcome. Bold marks the highest displayed fixed-method value in each column at full precision and does not indicate significance.","tab:real-fixed-methods")
    lines += [r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xrrrr}",r"\toprule"]
    for metric in METRICS:
        if metric=="AP":lines.append(r"\midrule")
        lines += [r"\multicolumn{5}{l}{\textbf{Panel "+("A: AUROC" if metric=="AUROC" else "B: AP")+r"}} \\",
                  r"Method & OS 0.05 & OS 0.20 & TS 0.05 & TS 0.20 \\",r"\midrule"]
        for num in REPS:
            cells=[]
            for design,budget in CONDITIONS:
                r=out[(out.metric==metric)&(out.number==num)&(out.design==design)&(out.budget==budget)].iloc[0]
                cells.append(r"\textbf{"+r.display+"}" if r.bold else r.display)
            lines.append(SHORT[num]+" & "+" & ".join(cells)+r" \\")
    lines += [r"\bottomrule",r"\end{tabularx}"]
    lines += finish("OS and TS denote one- and two-sample sampling. Entries use final train+validation refits, 20 repeated splits per dataset/condition, and the original supplied training-background prior. Supervised references are reported separately. AP denotes average precision.")
    save("tab_real_fixed_methods",out,lines,"main")

def decomposition_table():
    require_audit();p=paired(core());c=paired_condition(p)
    z=c.groupby(["design","budget","metric","contrast"],sort=False,as_index=False).agg(difference_pp=("difference_pp","mean"),n_datasets=("dataset_id","nunique"),n_pairs=("n_pairs","sum"))
    assert z.n_datasets.eq(8).all()
    z["display"]=z.difference_pp.map(lambda x:fmt(x,True))
    lines=start("Real-data learner and strategy decomposition. Entries are equal-dataset mean paired differences in percentage points; the 20 paired split differences are first averaged within each dataset and condition. These are descriptive pipeline contrasts, not population inference.","tab:real-decomposition-summary")
    lines += [r"\begin{tabularx}{\textwidth}{llr*{3}{>{\centering\arraybackslash}X}}",r"\toprule",
              r"Metric & Design & Budget & Learner gain & CS addition & RN addition \\",r"\midrule"]
    for metric in METRICS:
        for design,budget in CONDITIONS:
            cells=[z[(z.metric==metric)&(z.design==design)&(z.budget==budget)&(z.contrast==contrast)].iloc[0].display for contrast in CONTRASTS]
            lines.append(f"{metric} & {'OS' if design=='one_sample' else 'TS'} & {budget:.2f} & "+" & ".join(cells)+r" \\")
        if metric=="AUROC":lines.append(r"\midrule")
    lines += [r"\bottomrule",r"\end{tabularx}"]
    lines += finish("Learner gain: naive CatBoost minus logistic probability correction. CS/RN additions: the respective CatBoost strategy minus naive CatBoost. Positive values favour the first named method. Strategy-specific tuning is retained; no test-selected representative or group maximum is used.")
    save("tab_real_decomposition_summary",z,lines,"main")

def by_dataset():
    require_audit();d=condition_means(core());d=d[d.number.isin(REPS)].copy()
    lines=[r"\begingroup",r"\small",r"\setlength{\tabcolsep}{3.5pt}",r"\renewcommand{\arraystretch}{1.08}",
           r"\begin{longtable}{@{}p{0.65in}rp{2.85in}rrr@{}}",
           r"\caption{Fixed-method AUROC/AP by dataset, design and positive-label budget. Each entry averages 20 repeated splits using final refitted models; no dataset or method is selected according to performance.}\label{tab:s-real-fixed-methods}\\",
           r"\toprule",r"Design & Budget & Method & AUROC & AP & $n$ \\",r"\midrule",r"\endfirsthead",
           r"\multicolumn{6}{l}{\textit{Fixed-method real-data results (continued)}}\\",r"\toprule",
           r"Design & Budget & Method & AUROC & AP & $n$ \\",r"\midrule",r"\endhead",
           r"\midrule\multicolumn{6}{r}{Continued on next page}\\\endfoot",r"\bottomrule\endlastfoot"]
    for dataset_index,(ds,name) in enumerate(DATASETS.items()):
        if dataset_index in [2,4,6]: lines.append(r"\pagebreak")
        lines += [r"\multicolumn{6}{l}{\textbf{"+name+r"}}\\*"]
        for design,budget in CONDITIONS:
            for num in REPS:
                r=d[(d.dataset_id==ds)&(d.design==design)&(d.budget==budget)&(d.number==num)].iloc[0]
                lines.append(f"{'OS' if design=='one_sample' else 'TS'} & {budget:.2f} & {SHORT[num]} & {fmt(r.AUROC)} & {fmt(r.AP)} & {r.n_repetitions}"+r" \\")
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\end{longtable}",r"\endgroup"]
    for metric in METRICS:d[metric+"_display"]=d[metric].map(fmt)
    save("tab_s_real_fixed_methods_by_dataset",d,lines)

def all_methods():
    require_audit();d=condition_means(core());d=d[d.number>=3].copy()
    cols=["dataset","design","budget","method_id","method_name","AUROC","AP","n_repetitions","AUROC_n","AP_n"]
    save("tab_s_real_all_methods",d[cols])
    first=d.groupby(["dataset_id","design","number","method_id","method_name"],as_index=False)[METRICS].mean()
    z=first.groupby(["design","number","method_id","method_name"],as_index=False).agg(AUROC=("AUROC","mean"),AP=("AP","mean"),n_datasets=("dataset_id","nunique"))
    assert z.n_datasets.eq(8).all()
    for metric in METRICS:z[metric+"_display"]=z[metric].map(fmt)
    lines=start("All 25 PU configurations: equal-dataset mean ranking performance. Repetitions are averaged within each condition, the two budgets are averaged within each dataset, and the eight datasets are weighted equally.","tab:s-real-all-methods-equal-dataset")
    lines += [r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xrrrr}",r"\toprule",r"Method & AUROC OS & AUROC TS & AP OS & AP TS \\",r"\midrule"]
    for num in range(3,28):
        cells=[fmt(z[(z.number==num)&(z.design==design)].iloc[0][metric]) for metric in METRICS for design in DESIGNS]
        lines.append(METHODS[num]+" & "+" & ".join(cells)+r" \\")
    lines += [r"\bottomrule",r"\end{tabularx}"]
    lines += finish("Naive/corrected same-engine ranking aliases are retained as separate original configurations for audit; their differences are at machine precision and do not represent independent evidence. No method average or group envelope is computed. Supervised references are excluded and supplied separately.")
    save("tab_s_real_all_methods_equal_dataset",z,lines)

def budget_data():
    d=condition_means(core());d=d[d.number.isin(REPS)]
    keys=["dataset_id","dataset","design","number","method_id","method_name"]
    a=d[d.budget==.20].set_index(keys);b=d[d.budget==.05].set_index(keys)
    assert set(a.index)==set(b.index)
    rows=[]
    for metric in METRICS:
        delta=(100*(a[metric]-b[metric])).rename("difference_pp").reset_index()
        delta["metric"]=metric;delta["record_type"]="dataset";delta["low_n"]=20;delta["high_n"]=20
        rows.append(delta)
    z=pd.concat(rows,ignore_index=True)
    mean=z.groupby(["design","number","method_id","method_name","metric"],as_index=False).agg(
        difference_pp=("difference_pp","mean"),n_datasets=("dataset_id","nunique"),
        n_positive=("difference_pp",lambda x:int((x>1e-12).sum())),n_negative=("difference_pp",lambda x:int((x< -1e-12).sum())),
        n_zero=("difference_pp",lambda x:int((abs(x)<=1e-12).sum())))
    mean["record_type"]="equal_dataset";mean["dataset"]="Equal-dataset mean"
    return pd.concat([z,mean],ignore_index=True)

def budget_table():
    require_audit();z=budget_data();m=z[z.record_type=="equal_dataset"]
    z["display"]=z.difference_pp.map(lambda x:fmt(x,True))
    lines=start("Effect of additional positive-label information: budget 0.20 minus 0.05, in percentage points. Differences are computed between within-dataset split means before averaging equally across eight datasets.","tab:s-real-budget-effects")
    lines += [r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xrrrr}",r"\toprule",
              r"Method & AUROC OS & AUROC TS & AP OS & AP TS \\",r"\midrule"]
    for num in REPS:
        cells=[]
        for metric in METRICS:
            for design in DESIGNS:
                r=m[(m.number==num)&(m.metric==metric)&(m.design==design)].iloc[0]
                cells.append(fmt(r.difference_pp,True)+f" ({int(r.n_positive)}/8)")
        lines.append(SHORT[num]+" & "+" & ".join(cells)+r" \\")
    lines += [r"\bottomrule",r"\end{tabularx}"]
    lines += finish("Parentheses give the number of datasets with a positive change. Each budget has 20 archived splits per condition; differences compare budget-specific means and are not assumed to pair identical observations across budgets. Dataset-specific values and negative/zero counts are retained in the CSV. No population-level inference is attached.")
    save("tab_s_real_budget_effects",z,lines)

def agreement_data():
    d=condition_means(core());d=d[d.number.isin(REPS)]
    p=paired_condition(paired(core()));rows=[]
    for ds,name in DATASETS.items():
        for budget in BUDGETS:
            os=d[(d.dataset_id==ds)&(d.design=="one_sample")&(d.budget==budget)].set_index("number").loc[REPS]
            ts=d[(d.dataset_id==ds)&(d.design=="two_sample")&(d.budget==budget)].set_index("number").loc[REPS]
            for metric in METRICS:
                r=dict(dataset_id=ds,dataset=name,budget=budget,metric=metric,pearson_correlation=os[metric].corr(ts[metric]),n_fixed_methods=6)
                for contrast in CONTRASTS:
                    values=[]
                    for design in DESIGNS:
                        v=p[(p.dataset_id==ds)&(p.design==design)&(p.budget==budget)&(p.metric==metric)&(p.contrast==contrast)].iloc[0].difference_pp
                        sign=0 if abs(v)<=1e-12 else 1 if v>0 else -1
                        r[contrast+"_"+design+"_pp"]=v;r[contrast+"_"+design+"_sign"]=sign;values.append(sign)
                    r[contrast+"_sign_agrees"]=values[0]==values[1]
                rows.append(r)
    return pd.DataFrame(rows)

def agreement_table():
    require_audit();z=agreement_data()
    lines=start("Agreement of fixed-method performance patterns between sampling designs. Pearson correlations compare the six prespecified method means; sign agreement compares each prespecified paired contrast. Results are descriptive and do not identify a causal effect of sampling design.","tab:s-real-design-agreement")
    lines += [r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xrrrrr}",r"\toprule"]
    for metric in METRICS:
        if metric=="AP":lines.append(r"\midrule")
        lines += [r"\multicolumn{6}{l}{\textbf{"+metric+r"}}\\",r"Dataset & Budget & Correlation & Learner & CS & RN \\",r"\midrule"]
        for ds,name in DATASETS.items():
            for budget in BUDGETS:
                r=z[(z.dataset_id==ds)&(z.budget==budget)&(z.metric==metric)].iloc[0]
                flags=["Yes" if r[c+"_sign_agrees"] else "No" for c in CONTRASTS]
                lines.append(f"{name} & {budget:.2f} & {r.pearson_correlation:.3f} & "+" & ".join(flags)+r" \\")
    lines += [r"\bottomrule",r"\end{tabularx}"]
    lines += finish("Learner, CS and RN columns indicate whether the respective contrast has the same sign in both designs. Values within $10^{-12}$ percentage points of zero are numerical ties; two zero signs agree. CSV files retain the signed contrasts. Each method mean uses 20 splits; the comparison unit for this diagnostic is the six fixed configurations, not independent datasets.")
    save("tab_s_real_design_agreement",z,lines)

def neural_summary():
    p=neural_pairs()
    c=p.groupby(["dataset_id","dataset","design","budget","risk","selection","metric"],as_index=False).agg(
        difference_pp=("difference_pp","mean"),n_pairs=("difference_pp","count"))
    assert c.n_pairs.eq(20).all()
    z=c.groupby(["dataset_id","dataset","risk","selection","metric"],as_index=False).agg(
        difference_pp=("difference_pp","mean"),n_conditions=("budget","size"),n_pairs=("n_pairs","sum"))
    assert z.n_conditions.eq(4).all()
    return c,z

def neural_table():
    require_audit();c,z=neural_summary()
    c["record_type"]="condition";z["record_type"]="dataset_equal_four_conditions"
    summary=z.groupby(["risk","selection","metric"],as_index=False).agg(difference_pp=("difference_pp","mean"),n_datasets=("dataset_id","nunique"))
    summary["dataset"]="Equal-dataset mean";summary["record_type"]="equal_dataset"
    out=pd.concat([c,z,summary],ignore_index=True);out["display"]=out.difference_pp.map(lambda x:fmt(x,True))
    lines=start("Matched neural-minus-linear real-data ranking differences, in percentage points. For each dataset, 20 paired split differences are averaged within each condition, then the four design-budget conditions are weighted equally. The final row weights the eight datasets equally.","tab:s-real-neural")
    lines += [r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xrrrr}",r"\toprule"]
    for risk in ["upu","nnpu"]:
        if risk=="nnpu":lines.append(r"\midrule")
        lines += [r"\multicolumn{5}{l}{\textbf{"+("uPU" if risk=="upu" else "nnPU")+r"}}\\",
                  r"& \multicolumn{2}{c}{AUROC difference} & \multicolumn{2}{c}{AP difference}\\",
                  r"Dataset & Label-AUC & PU-risk & Label-AUC & PU-risk\\",r"\midrule"]
        for name in [*DATASETS.values(),"Equal-dataset mean"]:
            g=summary if name=="Equal-dataset mean" else z[z.dataset==name]
            vals=[fmt(g[(g.risk==risk)&(g.metric==metric)&(g.selection==sel)].iloc[0].difference_pp,True) for metric in METRICS for sel in ["auc","risk"]]
            lines.append(name+" & "+" & ".join(vals)+r" \\")
    lines += [r"\bottomrule",r"\end{tabularx}"]
    lines += finish("Label-AUC selects on observed validation P/U labels, not hidden true Y. PU-risk selects on the empirical PU risk. These are matched extension linear baselines, not substitutions for the core catalog. Repeated splits overlap; no population-inference interval is shown. This is the preserved fixed-budget study: 522/5,120 selections reached 3,000 epochs, and the original numerical cap gate was not passed; execution/metric checks passed. No convergence claim is made.")
    save("tab_s_real_neural",out,lines)

def rn_table():
    require_audit();d=rn_data()
    cols=["Delta_AUROC","Delta_AP","RN_precision","RN_negative_recall","RN_selection_fraction"]
    c=d.groupby(["dataset_id","dataset","design","budget"],as_index=False)[cols].mean()
    z=c.groupby(["dataset_id","dataset","design"],as_index=False)[cols].mean()
    z["AUROC_difference_pp"]=100*z.Delta_AUROC;z["AP_difference_pp"]=100*z.Delta_AP
    for k in ["RN_precision","RN_negative_recall","RN_selection_fraction"]:z[k+"_percent"]=100*z[k]
    z["n_budgets"]=2;z["n_splits_per_budget"]=20;z["source_file"]=SOURCES["rn"]
    lines=start("RN-CatBoost real-data diagnostic from Task 3. Ranking differences are RN minus naive CatBoost; each dataset/design value averages 20 splits within each budget and then the two budgets. Selector diagnostics refer to the final train+validation RN set.","tab:s-real-rn-diagnostic")
    lines += [r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xrrrrr}",r"\toprule"]
    for design in DESIGNS:
        if design=="two_sample":lines.append(r"\midrule")
        lines += [r"\multicolumn{6}{l}{\textbf{"+("One-sample" if design=="one_sample" else "Two-sample")+r"}}\\",
                  r"Dataset & AUROC pp & AP pp & Purity \% & Neg. recall \% & Selected U \%\\",r"\midrule"]
        for ds,name in DATASETS.items():
            r=z[(z.dataset_id==ds)&(z.design==design)].iloc[0]
            vals=[fmt(r.AUROC_difference_pp,True),fmt(r.AP_difference_pp,True),f"{100*r.RN_precision:.2f}",f"{100*r.RN_negative_recall:.2f}",f"{100*r.RN_selection_fraction:.2f}"]
            lines.append(name+" & "+" & ".join(vals)+r" \\")
    lines += [r"\bottomrule",r"\end{tabularx}"]
    lines += finish("Purity is the true-negative fraction among selected U; negative recall is retained true negatives divided by available true negatives; selected U is the fraction of U retained. RN sets were previously reconstructed from exact archived inputs/code because original indices were unavailable. No selectors are rerun here. These associations do not show that purity causes a performance difference; Task 4 alternative selectors are not substituted for original RN.")
    save("tab_s_real_rn_diagnostic",z,lines)

FUNCTIONS={"tab_real_fixed_methods":fixed_table,"tab_real_decomposition_summary":decomposition_table,
"tab_s_real_fixed_methods_by_dataset":by_dataset,"tab_s_real_all_methods":all_methods,
"tab_s_real_budget_effects":budget_table,"tab_s_real_design_agreement":agreement_table,
"tab_s_real_neural":neural_table,"tab_s_real_rn_diagnostic":rn_table}
