# Framework specification

Formal definitions behind `mdp_diagnostics`. Code references point to `src/mdp_diagnostics/`.

## 1. Objects

**Observational process.** For trajectory (patient stay) $i$ we observe decision rows
$(t_{ij}, X_{ij}, A_{ij}, R_{ij})_{j=0}^{n_i}$ with irregular times $t_{ij}$, partially missing
measurements $X_{ij}$, recorded actions $A_{ij}$ and optional rewards; plus an optional trajectory
outcome $Y_i$. The last row is a terminal observation (no action used).
History $H_{ij} = (X_{i0}, A_{i0}, \dots, X_{ij})$.

**Formulation.** An MDP formulation is a tuple $\mathcal M = (\phi, \Delta t, g_A, r, T)$:
state map $\phi(H) \in \mathbb R^d$ (`state_cols`), decision grid $\Delta t$ (`dt`, rows if `None`),
action map $g_A$ (`action_bins`, declared `action_levels`), reward $r$ (`reward_col` or
`terminal_reward`), horizon $T$. Every diagnostic is a property of $(\mathcal M, \mathcal D)$, never of
the clinical problem alone.

**Transitions** (`data/schema.py::Transitions`). Complete-case tuples
$(s_t, a_t, r_t, s_{t+1}, h_t)$ with $h_t = (s_{t-1..t-k}, a_{t-1..t-k})$. Rows without a full $k$-step
history are dropped so that all compared models use identical rows; missing values are never imputed
inside the toolkit (the count of dropped rows is reported).

## 2. Two questions, five levels

| Level | Question | Evidence | Typical status source |
|---|---|---|---|
| A Clinical problem suitability | **A**: is MDP modelling appropriate? | clinical metadata + data proxies | expert judgement, action multiplicity, delayed effects |
| B Formulation validity | **A** | data | state/reward sufficiency, alignment, stationarity, estimability, decision interval |
| C Dataset suitability | **B**: does the dataset support offline RL? | raw data | missingness, sampling, timestamps, observation age, informative observation |
| D Offline RL feasibility | **B** | data (+ untestable assumptions flagged) | overlap, coverage, IS reliability, behaviour history dependence, confounding = not assessable |
| E Deployment readiness | outside data diagnostics | metadata only | expert judgement |

Question A can be answered positively while Question B is answered negatively (adequate model, but data
that cannot support evaluating a *different* policy). The framework never merges the two.

## 3. Statuses (`framework/criteria.py`)

`no_concern_detected` (strongest positive statement; *not* "valid" or "safe"), `concern`,
`serious_concern`, `inconclusive`, `not_assessable` (untestable from data), `expert_judgement_required`.
There is **no aggregate score**: dimensions are incommensurable and an average would hide the binding
constraint.

## 4. Diagnostics

### 4.1 Information-gain family (`metrics/markov.py`)
For base features $Z$, extra features $E$, target $Y$ and trajectory-level $K$-fold cross-fitting:
$$G(E \to Y \mid Z) = \frac1n\sum_{i} \big[\log \hat p_{-f(i)}(Y_i \mid Z_i, E_i) - \log \hat p_{-f(i)}(Y_i \mid Z_i)\big].$$
Continuous $Y$: Gaussian working likelihood with pooled out-of-fold residual variance per dimension,
so $G = \sum_j \tfrac12 \log(\hat\sigma^2_{j,Z}/\hat\sigma^2_{j,ZE})$ — an estimate of the Gaussian
conditional mutual information $I(Y;E\mid Z)$ (exact for linear-Gaussian systems). Discrete $Y$:
out-of-fold log-loss difference of a probabilistic classifier. CI: trajectory-level bootstrap of
per-transition contributions. Model classes: ridge/logistic (`linear`) or gradient boosting (`gbm`).

| Diagnostic | $Y$ | $Z$ | $E$ | Assumption probed |
|---|---|---|---|---|
| State sufficiency | $s_{t+1}$ | $(s_t,a_t)$ | $h_t$ | $S_{t+1} \perp H_{t-1} \mid S_t, A_t$ (w.r.t. recorded history) |
| Lag profile | $s_{t+1}$ | $(s_t,a_t)$ | first $k$ lags | order of observable memory |
| Reward sufficiency | $r_t$ | $(s_t,a_t,s_{t+1})$ | $h_t$ | $R_t \perp H_{t-1} \mid S_t,A_t,S_{t+1}$ |
| Time homogeneity | $s_{t+1}$ | $(s_t,a_t)$ | (step, time) | stationary kernel |
| Action alignment | $s_{t+1}$ | $s_t$ | $a_t$ | recorded action acts within the interval |
| Delayed effects | $s_{t+1}$ | $(s_t,a_t)$ | $a_{t-1..t-k}$ | past treatment matters (supports sequentiality) |
| Behaviour history dependence | $a_t$ | $s_t$ | $h_t$ | clinicians use information beyond $s_t$ (confounding *symptom*) |
| Informative observation | $a_t$ | LOCF state | missingness indicators | measurement process informs treatment |

**GCM test** (Shah & Peters 2020): out-of-fold residuals $\varepsilon^Y = Y-\hat E[Y\mid Z]$,
$\varepsilon^E = E-\hat E[E\mid Z]$; per pair $(j,m)$ the trajectory-clustered statistic
$T_{jm} = \sum_g S_{g,jm} / \sqrt{\sum_g (S_{g,jm} - n_g \bar R_{jm})^2}$ with $S_{g,jm}$ the sum of
residual products in trajectory $g$; test statistic $\max_{jm}|T_{jm}|$ with a Gaussian multiplier
bootstrap over trajectories.

**Decision rule** (equivalence logic, `classify_gain`), with relevance threshold $\delta$
(`min_gain_nats`, default 0.01, a convention reported with sensitivity):
- CI upper bound $< \delta$ → `no_concern_detected` (relevant dependence excluded, even if significant);
- GCM rejects at $\alpha$ and CI lower bound $\ge \delta$ → `concern`;
- otherwise `inconclusive` (or `concern` if significant with point estimate $\ge \delta$).

### 4.2 Overlap (`metrics/overlap.py`)
Cross-fitted behaviour policy $\hat\pi_b(a\mid s)$ over the *declared* action space.
- Support-violation rate $\mathrm{SVR}_\epsilon = \frac1n\sum_t \mathbb 1[\min_a \hat\pi_b(a\mid s_t) < \epsilon]$, with a sensitivity curve over $\epsilon$ and per-action shares.
- Normalised entropy $\bar H = \frac1n\sum_t H(\hat\pi_b(\cdot\mid s_t))/\log|\mathcal A|$; effective number of actions $e^{H}$.
- Lower quantiles of $\hat\pi_b(a_t\mid s_t)$ (sources of extreme weights).
- Calibration gate: expected calibration error of $\hat\pi_b$ (one-vs-rest, out-of-fold).
- Model-free check: share of decisions whose $k$ nearest neighbours from *other* trajectories all took one action.
- AOI (candidate, comparison only): $\frac1n\sum_t \frac1{|\mathcal A|}\sum_a \mathbb 1[\hat\pi_b(a\mid s_t)>\epsilon] = 1-$ mean share of unsupported actions.

### 4.3 Coverage and estimability (`metrics/coverage.py`)
- Occupancy-weighted support: with K-means state clusters $c(s)$ and counts $N(c,a)$, $\frac1n\sum_t \frac1{|\mathcal A|}\sum_a \mathbb 1[N(c(s_t),a)\ge N_{\min}]$ plus the share of decisions in under-supported own cells.
- ESAC (candidate, comparison only): unweighted share of supported cells.
- Transition epistemic ratio: mean over dimensions of (SD of bootstrap-refitted ridge predictions) / (residual SD).
- kNN distance ratio for a given set of query states (distribution shift; needs target-policy states, e.g. from a simulator).

### 4.4 IS-based OPE reliability (`metrics/ope_reliability.py`)
Trajectory weights $w_i = \prod_t \pi_e(a_t\mid s_t)/\hat\pi_b(a_t\mid s_t)$ for a target policy; without a
clinical target, the reference family $\pi_\lambda = (1-\lambda)\hat\pi_b + \lambda/|\mathcal A|$.
Reported: Kish ESS share $(\sum w)^2/(n\sum w^2)$, Pareto-$\hat k$ of $w$ (Zhang–Stephens fit as in PSIS;
$\hat k>0.7$ = unreliable, Vehtari et al.), max-weight share, ESS share by horizon, optional WIS
bootstrap CI width. These need a target policy but no trained model.

### 4.5 Temporal indicators (`metrics/temporal.py`), on raw rows
Missingness (per variable, per trajectory); interval CV and gap rate (gap $>\gamma\times$ median
interval); timestamp heaping shares; observation age (time since last measurement, per variable);
share of action changes at rows without any observation; share of grid steps aggregating $>1$ distinct
raw action (decision interval coarser than decisions); informative missingness (outcome AUC from
measurement patterns; action gain from missingness indicators).

## 5. What is deliberately not assessable
Sequential exchangeability (no unmeasured confounding), consistency, clinical meaningfulness of
$\phi$, $g_A$, $r$, reward validity, and anything about latent variables that leave no trace in recorded
history. These are reported as `not_assessable` or `expert_judgement_required`, with recommended
sensitivity analyses.

## 6. Recommended remedies (encoded as `next_step`)
| Diagnosed pattern | Remedy |
|---|---|
| State-sufficiency concern, gain concentrated in lag $k$ | add lags/aggregates (cumulative dose), belief state or recurrent representation |
| Decision-interval concern / action-alignment concern | finer grid, event-driven epochs (semi-MDP), fix treatment timestamps |
| Stationarity concern | add time/phase to the state or finite-horizon time-indexed MDP |
| Overlap concern | restrict candidate policies to supported actions, coarsen actions, pessimistic offline RL |
| IS-reliability concern | model-based/FQE with uncertainty, limit deviation from practice, shorter horizons |
| Behaviour-history concern | richer state for propensity models, confounding sensitivity analysis |
| Informative missingness | measurement indicators/times in the state, MNAR sensitivity analysis |
