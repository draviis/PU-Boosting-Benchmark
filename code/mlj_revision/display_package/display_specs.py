"""Shared specifications. All paths are relative to the repository root."""
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "mlj_revision/02_fixed_method_and_family_analysis").is_dir())
HOME = Path(__file__).resolve().parent
TASK = {
 "1B":"mlj_revision/01b_threshold_completion_and_refit_audit",
 "1C":"mlj_revision/01c_pu_balanced_accuracy_threshold",
 "2":"mlj_revision/02_fixed_method_and_family_analysis",
 "3":"mlj_revision/03_rn_mechanism",
 "4":"mlj_revision/04_rn_selector_sensitivity",
 "6":"mlj_revision/06_estimated_prior",
 "7":"mlj_revision/07_nonlinear_simulations",
 "7B":"mlj_revision/07b_nonlinear_final_analysis",
}
SOURCES = {
 "core":("2","fixed_method_scenario_results.csv"),
 "family":("2","family_balanced_results.csv"),
 "bayes":("2","bayes_ceiling_results.csv"),
 "core_ranks":("2","fixed_method_ranks.csv"),
 "alias_original":("7B","original_unique_score_model_ranks.csv"),
 "alias_audit":("7B","ranking_alias_verification.csv"),
 "alias_exceptions":("7B","ranking_alias_exceptions.csv"),
 "decomp":("7B","nonlinear_learner_strategy_decomposition.csv"),
 "headline":("7B","nonlinear_headline_results.csv"),
 "nlrank":("7B","nonlinear_unique_score_model_ranks.csv"),
 "failures":("7B","unavailable_RN_refits.csv"),
 "nl":("7","nonlinear_scenario_summary.csv"),
 "nlraw":("7","nonlinear_results.csv"),
 "nlcal":("7","nonlinear_dgp_calibration.csv"),
 "nl_original_rank":("7","nonlinear_fixed_method_ranks.csv"),
 "threshold":("1C","pu_ba_threshold_results.csv"),
 "thresholdA":("1B","policy_A_same_model_results.csv"),
 "thresholdB":("1B","policy_B_same_model_oracle_results.csv"),
 "rn":("3","rn_selection_quality_all.csv"),
 "rnfamily":("3","rn_family_balanced_summary.csv"),
 "rnpaired":("3","rn_paired_mc_uncertainty.csv"),
 "selector":("4","rn_selector_performance.csv"),
 "selector_mc":("4","rn_selector_mc_uncertainty.csv"),
 "priorcover":("6","prior_sensitivity_coverage.csv"),
 "est_sim":("6","estimated_prior_mc_uncertainty.csv"),
 "est_real":("6","estimated_prior_real_summary.csv"),
 "neural":("existing neural","publication_preparation/epoch_check_analysis/paired_model_summary.csv"),
 "supplied_os":("supplied-prior source","phase9_one_sample_prior_misspecification_confirmatory_100reps_8_scenarios/results/prior_misspecification_analysis/paired_changes_raw.csv"),
 "supplied_ts":("supplied-prior source","phase7_two_sample_prior_misspecification_confirmatory_100reps_8_scenarios/results/prior_misspecification_analysis/paired_changes_raw.csv"),
}
def source_path(key):
 task, name=SOURCES[key]
 return Path(TASK[task])/name if task in TASK else Path(name)

ENGINES=["AdaBoost","GBM","XGBoost","LightGBM","CatBoost"]
METHODS={3:"Logistic probability correction",4:"uPU logistic",5:"nnPU logistic",26:"AdaPU",27:"PU gradient boosting"}
for start,strategy in [(6,"Naive"),(11,"Corrected"),(16,"RN"),(21,"CS")]:
 for i,engine in enumerate(ENGINES): METHODS[start+i]=f"{strategy} {engine}"
REPS=[3,5,10,25,20,27]
SHORT={3:"Logistic\ncorrection",4:"uPU\nlogistic",5:"nnPU\nlogistic",10:"Naive\nCatBoost",25:"CS\nCatBoost",20:"RN\nCatBoost",27:"PU gradient\nboosting"}
BLOCKS=[("Nonboosting and native PU",[3,4,5,26,27]),("Naive boosting",list(range(6,11))),("Corrected boosting",list(range(11,16))),("RN boosting",list(range(16,21))),("CS boosting",list(range(21,26)))]
DESIGNS=["one_sample","two_sample"]
DNAME={"one_sample":"One-sample","two_sample":"Two-sample"}
FAMILIES=["Reference","Positive-class prevalence","Positive-label availability","Class separation","Dimensionality","Predictor correlation","Joint prevalence / labelling","Curved boundary","Feature interaction","Unequal class covariance"]
FSHORT=["Reference","Prevalence","Label availability","Separation","Dimensionality","Correlation","Joint prevalence / labels","Curved","Interaction","Unequal covariance"]
SCENARIOS={1:"Reference",2:"Rare positives",3:"Prevalence 0.05",11:"Prevalence 0.90",12:"Few labels",26:"500 predictors",38:"Curved",39:"Interaction",40:"Unequal covariance"}
NLNAMES=["NL01 sinusoidal","NL02 radial","NL03 pure XOR","NL04 threshold interaction","NL05 sinusoidal + noise (p=500)","NL06 heterogeneous nonlinear"]
DISPLAYS={}
def add(id,location,section,metric,methods,keys,purpose,caption=None):
 DISPLAYS[id]=dict(location=location,manuscript_section=section,metric=metric,methods=methods,source_keys=keys,purpose=purpose,caption=caption or purpose)
add("fig_core_fixed_methods","main","Controlled simulations","AUROC; AP","M003 reference; M005 M010 M025 M020 M027",["core","family"],"Compare fixed configurations across ten design-defined families.",
 "Fixed configurations in the original controlled simulations. Values are percentage-point differences from logistic probability correction. Settings are averaged within each family; each scenario family receives equal weight. No best-of-group selection is used. AUROC and AP use the archived final refitted models. OS and TS denote one- and two-sample designs.")
add("tab_same_engine_family_balanced","main","Controlled simulations","AUROC; AP","Correction/RN/CS x five engines",["core","family"],"Hold the learner family fixed while comparing PU strategies.",
 "Family-balanced same-engine differences relative to naive training. Positive values favour the PU strategy. The learner family is held fixed, although strategy-specific tuning can select different hyperparameter settings.")
add("tab_core_bayes_reference","main","Controlled simulations","AUROC; regret","Six prespecified representatives",["core","bayes"],"Place five representative original settings against their validated Bayes references.")
add("fig_nonlinear_decomposition","main","Nonlinear extension","AUROC; AP","M010-M003; M025-M010; M020-M010",["decomp"],"Separate learner flexibility from CatBoost-specific CS and RN changes.",
 "Prespecified CatBoost contrasts for each nonlinear DGP: learner gain is naive CatBoost minus logistic probability correction; CS and RN additions hold CatBoost fixed. Points and bars are paired mean differences and 95% Monte Carlo intervals (mean +/- 1.96 MCSE), in percentage points. Individual DGPs are primary; no DGP average or group maximum is shown. NL02 two-sample RN uses 99 pairs because one final refit is unavailable; all other displayed contrasts use 100.")
add("tab_nonlinear_headline","main","Nonlinear extension","AUROC","M003 M010 M025 M020 M027",["headline","failures","nlcal"],"Report absolute performance for the prespecified nonlinear representatives.")
add("fig_s_core_full_fixed_methods","supplement","Full controlled simulations","AUROC; AP","All 25 PU configurations",["core","family"],"Retain every PU configuration in readable method blocks.")
add("tab_s_core_family_balanced_all_methods","supplement","Full controlled simulations","AUROC; AP","All 25 PU configurations",["family"],"Report equal-ten-family mean performance for every PU configuration.")
add("tab_s_same_engine_40setting_sensitivity","supplement","Weighting sensitivity","AUROC; AP","Correction/RN/CS x five engines",["core","family"],"Retain the unweighted 40-setting sensitivity summary.")
for kind in ["fig","tab"]:
 add(f"{kind}_s_bayes_regret","supplement","Bayes reference","AUROC regret","Six prespecified representatives",["core","bayes"],"Show Bayes AUROC minus fixed-method AUROC in all 40 validated settings; lower regret is better.")
 add(f"{kind}_s_threshold_sensitivity","supplement","Threshold audit","Balanced accuracy","All 25 in table; six representatives in figure",["threshold","thresholdA","thresholdB"],"Compare same-model quantile, PU-estimated BA and oracle BA thresholds on the limited pilot.",
 "Threshold pilot only: eight simulation settings x one repetition per design, plus eight real datasets x two label budgets x one split per design in the table. All three rules apply to the same selected train-only model. A_same_model uses the population-representative validation quantile (all validation observations for one-sample; validation U only for two-sample). PU-estimated BA is Task 1C's pilot rule; oracle BA uses true validation labels and is diagnostic only. Test labels never select thresholds. No historical mixed or refit-transfer rule is presented. The figure shows the six conceptual representatives on simulation cases; the table retains all 25 methods and real pilot cases. Six real-data method/case replays failed validation verification and are marked unavailable, leaving 1,194 verified method/case results per rule.")
 add(f"{kind}_s_rn_mechanism","supplement","RN mechanism","Purity; recall; retention; AUROC/AP change","RN CatBoost vs naive CatBoost",["rn","rnfamily","rnpaired"],"Connect negative-set quality with fixed CatBoost performance, including interaction.",
 "RN masks were reconstructed from exact archived data/code because selected indices were not originally stored. CatBoost results use the final train+validation RN selection stage. RN purity is true negatives / selected U; true-negative recall is selected true negatives / available true negatives (also the fraction of available true negatives retained). U retention is selected U / available U and is shown separately. The figure averages settings within each of ten families; the table retains all 40 settings per design. Interaction is explicitly labelled. AUROC/AP changes are RN CatBoost minus naive CatBoost; no broad BA comparison is made.")
 add(f"{kind}_s_rn_selector_sensitivity","supplement","RN selector diagnostic","AUROC; AP","Fixed CatBoost; five selectors/controls",["selector","selector_mc"],"Compare spy selection with matched-size and oracle controls at fixed CatBoost parameters.",
 "Diagnostic fixed-parameter CatBoost analysis in five prespecified scenarios. Original logistic spy, full CatBoost nonlinear spy, size-matched nonlinear spy, random size-matched and oracle true-negative size-matched controls are compared. Random/oracle draws are first averaged within each repetition; repetitions are then equally weighted (100 per setting/design). Oracle controls use hidden truth and are not practical PU methods. Nonlinear-spy RN is not added to the benchmark catalog. Plotted absolute AUROC/AP scales are 0 to 1.")
 add(f"{kind}_s_supplied_prior_sensitivity","supplement","Supplied-prior sensitivity","AUROC; AP","M004 M005 M026 M027",["supplied_os","supplied_ts"],"Show sensitivity of four prior-dependent fitted models to supplied-prior perturbations.",
 "Supplied-minus-correct-prior AUROC/AP changes for uPU logistic, nnPU logistic, AdaPU and PU gradient boosting. Eight prespecified settings are averaged equally; each setting has 100 paired repetitions. Table rows retain each setting and pointwise 95% Monte Carlo intervals. The ratio-one point compares each archived correct-prior result with itself, so its difference is identically zero; it is not an imputation. One-sample labeling probability remains as in the archived sensitivity protocol.")
 add(f"{kind}_s_estimated_prior","supplement","Estimated-prior diagnostic","Prior ratio; AUROC/AP change","M004 M005 M026 M027",["priorcover","est_sim","est_real"],"Relate KM2 prior error to changes from the correct-supplied-prior comparator.",
 "KM2 estimated-prior diagnostic, specific to the estimator and implementation audited in Task 6. Prior panels show the mean estimated/true prior ratio and empirical 5th--95th percentiles across repetitions; performance panels show paired mean changes from the correct supplied prior, in percentage points. The table retains every audited setting and real-data budget for the four directly prior-dependent fitted methods. Simulation cases use 100 repetitions; real-data cases use 20 repeated splits, whose uncertainty is conditional resampling uncertainty rather than independent-population Monte Carlo uncertainty. KM2 findings are not generalized to other prior estimators.")
add("fig_s_nonlinear_full","supplement","Full nonlinear extension","AUROC; AP","All 25 PU configurations",["nl","nlraw","failures","nlcal"],"Retain every nonlinear PU configuration with explicit coverage.")
add("tab_s_nonlinear_all_methods","supplement","Full nonlinear extension","AUROC; AP; metric-specific n","All 25 PU configurations",["nl","nlraw","failures"],"Report all nonlinear means and metric-specific counts; disclose six unavailable refits.")
add("tab_s_nonlinear_unique_score_ranks","supplement","Nonlinear ranking audit","AUROC/AP ranks","20 verified ranking entries",["nlrank","alias_audit","alias_exceptions","alias_original","core_ranks","nl_original_rank"],"Use Task 7B ranking aliases without altering the original 25-configuration rank files.")
for kind in ["fig","tab"]:
 add(f"{kind}_s_neural_simulation","supplement","Exploratory learner capacity","AUROC; AP","Matched uPU/nnPU linear-neural, two selection rules",["neural"],"Compare neural and matched linear learners in the exploratory ten-repetition extension.",
 "Exploratory linear--neural simulation extension: 10 paired repetitions per setting and design. Neural minus matched linear uPU/nnPU is shown separately for risk-based and proxy-AUROC hyperparameter selection. Bars retain the saved pointwise paired t intervals (9 degrees of freedom), not Gaussian large-sample intervals. No BA headline is drawn; these capacity comparisons do not isolate boosting.")

assert len(DISPLAYS)==25

# Publication captions are descriptive; purpose strings remain README planning metadata.
DISPLAYS["tab_core_bayes_reference"]["caption"]="Bayes reference and fixed-method AUROC in five representative original simulations."
DISPLAYS["tab_nonlinear_headline"]["caption"]="Nonlinear-extension AUROC for prespecified representative PU configurations."
DISPLAYS["fig_s_core_full_fixed_methods"]["caption"]="Absolute AUROC and AP for all 25 PU configurations, split into prespecified method blocks. Values average settings within each of the ten families. Scales are identical (0 to 1) across designs and blocks; no group maximum is taken."
DISPLAYS["tab_s_core_family_balanced_all_methods"]["caption"]="Equal-family mean AUROC and AP for all 25 core PU configurations."
DISPLAYS["tab_s_same_engine_40setting_sensitivity"]["caption"]="Unweighted 40-setting sensitivity summary. Same-engine PU-minus-naive differences in percentage points."
DISPLAYS["fig_s_bayes_regret"]["caption"]="Bayes AUROC minus fixed-method AUROC for the six conceptual representatives in all 40 original settings with validated Bayes references. Lower regret is better."
DISPLAYS["tab_s_bayes_regret"]["caption"]=DISPLAYS["fig_s_bayes_regret"]["caption"]
DISPLAYS["fig_s_nonlinear_full"]["caption"]="All 25 nonlinear-extension PU configurations, separated into prespecified method blocks. AUROC and AP use final train+validation refits. Individual DGPs are primary, and scales are fixed at 0 to 1. Six unavailable RN refits remain missing; asterisks mark means based on 99 rather than 100 successful refits."
DISPLAYS["tab_s_nonlinear_all_methods"]["caption"]="Complete nonlinear-extension AUROC and AP results, with metric-specific coverage."
DISPLAYS["tab_s_nonlinear_unique_score_ranks"]["caption"]="Nonlinear ranking results in the verified 20-entry unique-score universe."
