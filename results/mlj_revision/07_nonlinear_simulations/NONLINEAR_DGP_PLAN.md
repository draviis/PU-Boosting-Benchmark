# Nonlinear simulation extension: DGP plan

Status: proposed plan, written before calibration. Only DGP calibration and validation are authorized at this stage. The 100-repetition learner experiments await the requested plan check/approval. Cluster access does not require another approval; no cluster job is needed for the initial calibration.

This is a separate six-scenario extension, identified as NL01–NL06. The original S001–S040 scenarios, results and manuscript remain unchanged.

## Scientific purpose and proposed difficulty

The design isolates six interpretable structures, with a controlled low-/high-dimensional pair. Difficulty is chosen using the TRUE likelihood ratio, never a fitted benchmark learner.

| ID | Structure | p | Calibrated parameter | Target Bayes AUROC |
|---|---|---:|---|---:|
| NL01 | Additive sinusoidal density signal | 10 | concentration theta | .78 |
| NL02 | Radial/concentric class structure | 10 | scale ratio r>1 | .80 |
| NL03 | Pure, stronger XOR Gaussian mixture | 10 | component displacement a>=.67 | .84 |
| NL04 | Piecewise threshold interaction | 10 | conditional mean magnitude theta | .76 |
| NL05 | NL01 signal plus 497 independent noise variables | 500 | same theta as NL01 | .78 |
| NL06 | Heterogeneous conditional covariance and nonlinear means | 10 | mean multiplier theta | .86 |

Targets span useful discrimination levels rather than nearly perfect classification. The acceptance band is [.70,.90], with each independent confirmation estimate within .015 of its target and its 95% Monte Carlo interval inside the band. NL05 inherits NL01's parameter; it is not separately tuned. If confirmation fails, retain the failed record and revise the proposal before approval, with a new independent confirmation stream. Do not tune to learner rankings.

## Common population conventions

Y ~ Bernoulli(pi), pi=.20. Write s_y=2y-1. All variables not explicitly informative are independent N(0,1) noise, independent of Y and the signal coordinates. All likelihood ratios below are f_1(x)/f_0(x), without prior odds.

Let L(x)=log[f_1(x)/f_0(x)]. The Bayes AUROC uses L as a ranking score. For balanced accuracy under equal class weighting, the Bayes decision is L>=0, regardless of pi. The population posterior is expit(logit(pi)+L); its .5 cutoff instead minimizes prevalence-weighted classification error. These rules are not interchangeable.

Noise densities cancel from L. The calibration confirmation still generates the full p-dimensional population in chunks, including NL05's 497 noise coordinates; it does not pretend to fit or approximate a 500-dimensional Bayes classifier.

### NL01: additive sinusoidal signal

For j=1,2,3, set weights w=(1,.7,.5) and frequencies k=(1,2,1). Conditional on Y=y, the coordinates are independent on [-pi_math,pi_math):

f_y(x_j) = exp[s_y*theta*w_j*sin(k_j*x_j)] / [2*pi_math*I_0(theta*w_j)].

Here pi_math is the mathematical constant and I_0 is the modified Bessel function. Coordinates 4–10 are independent standard-normal noise.

The normalizers are the same for both classes, so

L(x) = 2*theta*sum_{j=1}^3 w_j*sin(k_j*x_j).

This gives an additive but nonmonotone periodic boundary, including a two-cycle coordinate. It is not a Gaussian location-shift model in the observed predictors.

Exact sampling: draw V_j from von Mises with mode s_y*pi_math/2 and concentration theta*w_j; draw K_j uniformly from {0,...,k_j-1}; set X_j=(V_j+2*pi_math*K_j)/k_j and wrap to [-pi_math,pi_math). This covers all k_j periods with the correct density. The von Mises density and sampler are documented by [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.vonmises.html). Concentration is selected in [0,3] using Bayes-AUROC calibration.

### NL02: radial/concentric structure

For the informative pair T=(X_1,X_2),

T | Y=0 ~ N_2(0,I_2),
T | Y=1 ~ N_2(0,r^2 I_2), r>1.

Coordinates 3–10 are standard-normal noise. The decision boundary is a circle:

L(x) = -2*log(r) + (1-r^(-2))*||T||^2/2.

This isolates a radial variance signal without a mean shift. Both class distributions have overlapping support; no class is confined to a deterministic ring.

Because independent squared radii are scaled chi-square_2 variables,

Bayes AUROC = r^2/(1+r^2).

Thus r=sqrt(.80/.20)=2 gives the target exactly. For equal-weight BA define
q=4*log(r)/(1-r^(-2)).
Then TPR=exp[-q/(2*r^2)], TNR=1-exp(-q/2), and Bayes BA=(TPR+TNR)/2.
Independent Monte Carlo confirmation must agree with both analytic results within its sampling error.

### NL03: stronger, pure XOR interaction

Let C be an independent Rademacher sign, P(C=+1)=P(C=-1)=1/2. For E~N_2(0,I_2),

T=(X_1,X_2) = (a*C, s_y*a*C)+E.

Coordinates 3–10 are standard-normal noise. Each single informative coordinate has the SAME marginal distribution in both classes; class information lies in the joint sign/interaction structure. There is no additional linear main-effect coordinate.

The exact mixture likelihood ratio is

L(x) = log cosh[a*(X_1+X_2)] - log cosh[a*(X_1-X_2)].

Evaluate log cosh stably. This is a full two-component likelihood calculation, not a surrogate product score. Calibrate a in [.67,2] to AUROC .84. The lower bound is the original interaction component displacement; removing the original main-effect shift and increasing interaction separation makes the intended mechanism explicit.

### NL04: piecewise threshold interaction

Z_1,Z_2 are independent Uniform(-1,1), independent of Y. Define

h(Z)=sign(Z_1)*sign(Z_2-.25),
T | (Y=y,Z) ~ N(s_y*theta*h(Z),1).

Take (X_1,X_2,X_3)=(Z_1,Z_2,T); remaining coordinates are standard-normal noise. Use sign(u)=+1 for u>=0 and -1 otherwise; boundary events have probability zero.

The exact score is

L(x)=2*theta*h(Z)*T.

The class direction of T flips across the two covariate thresholds. This probes piecewise interaction handling, distinct from the smooth XOR mixture.

Since h(Z)^2=1, conditional score distributions reduce to a normal comparison:
Bayes AUROC=Phi(sqrt(2)*theta), and Bayes BA=Phi(theta).
Set theta=Phi^(-1)(.76)/sqrt(2) and independently confirm these formulas.

### NL05: sinusoidal signal with many noise variables

Use exactly the NL01 conditional distribution for X_1,X_2,X_3 and the identical calibrated theta, weights and frequencies. Add independent N(0,1) coordinates X_4,...,X_500.

The likelihood ratio is exactly the NL01 expression and does not depend on noise coordinates. Its population Bayes AUROC and BA must therefore equal NL01's. Use an independently generated confirmation population for NL05; compare its estimates with NL01 using the MCSE of the difference. Do not alter theta to compensate for the behavior of a learner in 500 dimensions.

This pair isolates irrelevant-dimension sensitivity at fixed signal information.

### NL06: heterogeneous covariance and nonlinear conditional means

Let Z=(Z_1,Z_2) have independent Uniform(-1,1) coordinates, independent of Y. Define

h_1(Z)=sin(pi_math*Z_1) + .5*(Z_2^2-1/3),
h_2(Z)=cos(pi_math*Z_2) + .5*Z_1*Z_2.

For T=(T_1,T_2),

T | (Y=y,Z) ~ N_2(s_y*theta*h(Z), Sigma_y),

Sigma_0 = [[1,.30],[.30,1]],
Sigma_1 = [[1.30,-.25],[-.25,.70]].

Both covariance matrices are positive definite. Use X=(Z_1,Z_2,T_1,T_2, six independent standard-normal noise variables).

The exact log likelihood ratio is

L(x) = -.5*log(det(Sigma_1)/det(Sigma_0))
       -.5*(T-theta*h)' Sigma_1^(-1) (T-theta*h)
       +.5*(T+theta*h)' Sigma_0^(-1) (T+theta*h).

This combines class-specific dispersion/orientation with smooth nonlinear conditional means and interactions. The distributions of Z cancel. The two mean functions integrate to zero, so the population class means are equal; discrimination is not a disguised global mean shift. Covariances stay fixed while theta in [0,3] is calibrated to AUROC .86.

## Calibration and independent checks

1. Write and hash this plan before calibration.
2. Implement conditional generators, explicit log densities, and closed-form likelihood-ratio scores. Verify L=log f_1-log f_0 independently, including mixture normalizers and noise cancellation. Check the analytic radial and threshold-interaction formulas.
3. For NL01, NL03 and NL06, use 50,000 independent draws per class per parameter evaluation, within the stated brackets. Use 12 bisection iterations and select the evaluated parameter closest to the fixed target. Search seeds are fixed in advance. NL02/NL04 use the analytic inverse formulas. NL05 inherits NL01.
4. Freeze the selected parameters BEFORE independent confirmation. For every scenario generate a fresh population of 1,000,000 observations with Y~Bernoulli(.2), in chunks of 10,000. Store exact L and Y, seed information and a streaming hash of the full X. No calibration observations are reused for benchmark training, validation or testing.
5. Estimate AUROC with half credit for ties. Estimate its MCSE from the two independent class-conditional influence components: Var(V_P)/n_P + Var(V_N)/n_N, where V_P is a positive observation's fraction of lower-scoring negatives and V_N is a negative observation's fraction of higher-scoring positives. Use mean +/-1.96*MCSE. This quantifies Monte Carlo integration uncertainty, not the future learner's repetition uncertainty.
6. Estimate equal-weight Bayes BA from L>=0 and report its binomial-component MCSE. Also record class counts, score ties, target error and acceptance flags.
7. Confirm NL02/NL04 against analytic AUROC and BA, and confirm NL01/NL05 agree within four MCSEs of their difference. The four-MCSE check is an implementation diagnostic, not a performance hypothesis test.
8. Validate the PU samplers independently: no labeled negative observations in one-sample data, correct P/U sizes and independent sources in two-sample data, reproducible/disjoint RNG streams, correct dimensions, and true Y absent from model-facing inputs.

RNG namespace: NumPy PCG64 via SeedSequence([20261017,7,scenario_index,phase,...]). Phase 11 is parameter search, 22 independent confirmation, 33 unit checks, and 44 future benchmark repetitions. The confirmation stream and every train/validation/test stream are disjoint. Record package versions and code hashes.

## Main-run protocol, pending plan approval

Both sampling designs use the frozen population DGP, pi=.20 and SCAR c=.30 where applicable.

- One-sample: independent population samples of 5,000 training, 5,000 validation and 10,000 test observations. On training/validation, S=Y*Bernoulli(.30); labeled positives and U partition each population sample.
- Two-sample: independent positive samples of 300 P and population-mixture samples of 4,700 U for each of training and validation. The population test sample has 10,000 observations. P and U are sampled independently; U is not filtered using hidden Y.
- 100 repetitions for every NL01–NL06 x design cell, paired across methods. Original scenario identifiers and results remain untouched.
- This primary extension retains the supplied-prior information condition for compatibility. It does not claim to remove Task 6's estimated-prior limitations; do not silently replace supplied pi/c by an estimator.
- Use the frozen catalog: original 25 PU configurations, the four Task 9 first-order risk methods, and its two matched naive-FG controls: 31 PU configurations, plus two supervised references reported separately. This implies 37,200 PU method-result rows and 2,400 supervised rows before failures/aliases; these are output records, not the number of tuning fits.
- No approved BaggingPU implementation or catalog decision was found in the existing project audit. No new BaggingPU method is invented here. An approved implementation identified during plan review must be included and frozen before any main run. Preserve its precise objective, base learner, bagging size, tuning budget and selection protocol.
- Existing methods retain their frozen selection rules/grids. New Task 9 methods must use matching, predeclared first-order controls and a common finite tuning budget finalized in the execution manifest before fitting. No parameters are selected from hidden validation Y or test outcomes.
- Pin one engine version across all same-engine comparators. Task 9 validated XGBoost 3.4.0/LightGBM 4.7.0; any different execution version must pass its small derivative/native-interface gate first.
- Use Task 1 common threshold policy A for every PU method. Selected train-only model validation scores set the empirical linear-interpolated (1-pi_tilde) quantile. Use all one-sample PU validation observations or only two-sample validation U. Corrected methods use raw q. Retain the threshold after final refitting, as specified in Task 1. Record the model-stage transfer explicitly.
- Save selected-model and final-refit validation/test raw ranking scores and corrected probabilities separately, plus both model identities. Preserve same-model BA as an explicitly labeled supplementary check so threshold-transfer effects can be audited without further refits. Never use test labels to select thresholds, candidates or stopping rules.
- Retain AUROC/AP, common-threshold BA, F1 and MCC, threshold ties and predicted-positive fractions. Brier/log loss can be saved under their existing probability definitions without replacing ranking scores.
- Compare fixed methods. Keep supervised/Bayes references outside practical PU ranks. Show each scenario and the NL01/NL05 pair; give the five signal mechanisms equal weight in any preferred aggregate, first averaging NL01 and NL05 within the sinusoidal mechanism. Also display the unweighted six-setting mean. No group maxima and no hypothesis tests treating the designed scenarios as random draws.
- For predeclared paired method contrasts report repetition-level mean differences, MCSE=sd(differences)/sqrt(n_complete_pairs), and +/-1.96*MCSE intervals, with intended n=100 and explicit failure/pair denominators. Do not discard failures silently.

## Approval gate and deliverables

The plan, DGP code, calibration CSV and checks will be delivered for the user's plan review. The instruction “do not run full simulations until that plan is checked” is the scientific gate. The separate authorization to use ravidaya's Augsburg cluster is already granted and will not trigger another cluster-access permission question.

Before approval: NONLINEAR_DGP_PLAN.md, nonlinear_dgp_calibration.csv, frozen parameter file, exact-score/sampler tests, calibration evidence and REPORT.md stating no learner results yet.

After approval: run the two designs x 100 repetitions with the frozen catalog, then create nonlinear_results.csv and nonlinear_fixed_method_comparison.pdf and extend REPORT.md with results. Do not fabricate empty/full-result artifacts or label a Bayes calibration figure as a learned-method comparison.

All writes stay in mlj_revision/07_nonlinear_simulations/. No manuscript changes.
