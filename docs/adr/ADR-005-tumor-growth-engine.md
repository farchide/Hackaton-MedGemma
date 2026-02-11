# ADR-005: Tumor Growth and Counterfactual Simulation Engine

| Field       | Value                                      |
|-------------|--------------------------------------------|
| **Status**  | Accepted                                   |
| **Date**    | 2026-02-11                                 |
| **Authors** | Digital Twin Tumor Team                    |
| **Deciders**| Full team                                  |

---

## Context

The core value proposition of the Digital Twin Tumor project is the ability to
fit patient-specific tumor growth models to longitudinal imaging data and then
run counterfactual simulations: "What if treatment had started two weeks
earlier?", "What if the patient switched to regimen B after cycle 3?", "What
trajectory would we expect under treatment resistance?"

This requires:

1. A **state-space formulation** that separates latent true tumor burden from
   noisy imaging observations.
2. A **growth model** (or ensemble of models) that captures the biological
   dynamics of tumor progression and treatment response.
3. A **treatment effect model** that maps therapy logs (drug, dose, timing) to
   a continuous modifier on tumor growth rate.
4. A **parameter fitting engine** that estimates patient-specific parameters
   from sparse, noisy longitudinal measurements (typically 3-8 timepoints per
   patient).
5. A **counterfactual engine** that modifies the treatment effect function and
   re-simulates forward to produce alternative trajectories with uncertainty
   bands.

The hackathon timeline demands that the primary engine be implementable in pure
Python with scipy/numpy, without requiring MCMC sampling or GPU-based inference.
A Bayesian extension is planned for post-hackathon.

---

## Decision

We implement an **ensemble of three parametric growth models** with a shared
treatment effect formulation, fit via multi-start nonlinear optimization, and
produce counterfactual simulations by modifying the treatment effect function.

---

### Mathematical Formulation

#### State-Space Model

Let V_i(t) denote the latent true tumor volume for patient i at continuous
time t. We observe discrete measurements y_{i,k} at timepoints t_k:

```
y_{i,k} = V_i(t_k) + epsilon_{i,k}

epsilon_{i,k} ~ N(0, sigma_obs^2)
```

where sigma_obs represents imaging measurement noise, which can be calibrated
from same-day repeat scans (RIDER dataset, see ADR-006).

#### Growth Models

**Model 1: Gompertz (Primary)**

The Gompertz model captures the deceleration of growth as tumors approach a
carrying capacity. It is the most widely used model in oncological growth
fitting.

```
dV/dt = r_eff(t) * V * ln(K / V)

r_eff(t) = r - s * E(t)
```

Where:
- `r` is the intrinsic growth rate (day^-1)
- `K` is the carrying capacity (mm^3)
- `s` is the treatment sensitivity coefficient (day^-1)
- `E(t)` is the treatment effect function (dimensionless, 0 to 1)

Analytical solution (constant E):

```
V(t) = K * exp(ln(V0/K) * exp(-(r - s*E) * t))
```

**Model 2: Exponential**

The simplest growth model, appropriate for early-stage tumors far from carrying
capacity.

```
dV/dt = r_eff(t) * V

r_eff(t) = r - s * E(t)
```

Analytical solution (constant E):

```
V(t) = V0 * exp((r - s*E) * t)
```

**Model 3: Logistic**

An alternative saturating model with a different deceleration profile than
Gompertz.

```
dV/dt = r_eff(t) * V * (1 - V/K)

r_eff(t) = r - s * E(t)
```

Analytical solution (constant E, r_eff > 0):

```
V(t) = K / (1 + ((K - V0) / V0) * exp(-r_eff * t))
```

#### Treatment Effect Function E(t)

The treatment effect is modeled as a **piecewise constant function** derived
from the patient's therapy log:

```
E(t) = sum_j  e_j * I[t in [t_start_j, t_end_j]]
```

Where:
- `e_j` is the effect magnitude of treatment episode j (0 = no effect, 1 = max)
- `I[.]` is an indicator function for the time interval
- Treatment episodes are extracted from the therapy log (drug name, start date,
  end date, dose)

For the hackathon, `e_j` is a fitted parameter per treatment episode. For
post-hackathon, `e_j` can be decomposed by drug class using pharmacokinetic
models.

An optional **exponential decay** variant models resistance:

```
E_decay(t) = e_j * exp(-lambda * (t - t_start_j))
```

Where lambda is a resistance rate parameter.

---

### Parameter Fitting

#### Optimization Strategy

For each patient and each growth model, parameters are estimated by minimizing
the negative log-likelihood:

```
NLL(theta) = (n/2)*ln(2*pi*sigma_obs^2) + (1/(2*sigma_obs^2)) * sum_k (y_{i,k} - V(t_k; theta))^2
```

Where theta = {V0, r, K, s, e_1, ..., e_J, sigma_obs} (K omitted for
exponential model).

**Multi-start optimization** is used to mitigate local minima:

1. Generate N_starts = 50 initial parameter vectors by Latin Hypercube Sampling
   within biologically plausible bounds.
2. Run `scipy.optimize.minimize` with L-BFGS-B (for box constraints) from each
   starting point.
3. Select the solution with the lowest NLL.

**Parameter bounds:**

| Parameter   | Lower       | Upper       | Units       |
|-------------|-------------|-------------|-------------|
| V0          | 10          | 100000      | mm^3        |
| r           | 0.0001      | 0.1         | day^-1      |
| K           | 100         | 1000000     | mm^3        |
| s           | 0.0         | 0.2         | day^-1      |
| e_j         | 0.0         | 1.0         | dimensionless|
| sigma_obs   | 1.0         | 5000        | mm^3        |

#### Uncertainty Quantification

**Bootstrap resampling** (N_boot = 1000) of the residuals provides confidence
intervals on all parameters and on the predicted trajectory:

1. Fit the model to original data, obtain residuals.
2. For each bootstrap iteration: resample residuals with replacement, add to
   fitted values, re-fit the model.
3. Compute 2.5th and 97.5th percentiles of predicted V(t) at each evaluation
   timepoint.

For post-hackathon, this is replaced by full Bayesian posterior sampling.

#### Model Selection and Ensemble Weighting

Each model's fit is scored by **AIC** (Akaike Information Criterion) and
**BIC** (Bayesian Information Criterion):

```
AIC = 2*k + n*ln(RSS/n)
BIC = k*ln(n) + n*ln(RSS/n)
```

Where k is the number of parameters, n is the number of observations, and RSS
is the residual sum of squares.

**Ensemble weights** are computed via Akaike weights:

```
delta_i = AIC_i - AIC_min
w_i = exp(-delta_i / 2) / sum_j exp(-delta_j / 2)
```

The ensemble prediction at each timepoint is:

```
V_ensemble(t) = sum_i w_i * V_i(t)
```

Ensemble uncertainty bands combine within-model bootstrap uncertainty and
between-model disagreement.

---

### Counterfactual Simulation Engine

The counterfactual engine re-uses the fitted patient-specific parameters but
modifies E(t) to represent alternative treatment scenarios:

#### Supported Scenarios

1. **Earlier/later treatment start:** shift t_start_j by delta_t days.
2. **Regimen switch:** replace e_j with a different value for a specific
   treatment episode (e.g., "What if we had used a more effective drug?").
3. **Treatment resistance:** apply the exponential decay variant E_decay(t)
   with a fitted or hypothesized lambda.
4. **Treatment holiday:** set E(t) = 0 for a specified interval.
5. **No treatment (natural history):** set E(t) = 0 for all t.

#### Simulation Procedure

1. Load the fitted parameters for the patient from the selected model (or
   ensemble).
2. Construct the modified E_cf(t) for the counterfactual scenario.
3. Integrate the ODE forward using `scipy.integrate.solve_ivp` with the RK45
   method (adaptive step size).
4. Propagate parameter uncertainty by running the integration for each
   bootstrap parameter set, producing uncertainty bands on the counterfactual
   trajectory.
5. Return both the factual and counterfactual trajectories for overlay
   visualization.

For piecewise-constant E(t), the analytical solutions can be chained across
intervals for computational speed, with `solve_ivp` reserved for more complex
E(t) profiles (e.g., PK-based).

---

### Implementation Structure

```
src/growth/
    models.py          # GompertzModel, ExponentialModel, LogisticModel classes
    treatment.py       # TreatmentEffect, PiecewiseConstant, ExponentialDecay
    fitting.py         # MultiStartFitter, BootstrapUQ, EnsembleWeighter
    counterfactual.py  # CounterfactualEngine, ScenarioBuilder
    state_space.py     # StateSpaceModel, ObservationModel
```

Each model class implements a common interface:

```python
class GrowthModel(Protocol):
    def predict(self, t: np.ndarray, params: dict) -> np.ndarray: ...
    def ode_rhs(self, t: float, V: float, params: dict) -> float: ...
    def fit(self, t_obs: np.ndarray, y_obs: np.ndarray, ...) -> FitResult: ...
    def param_names(self) -> list[str]: ...
    def param_bounds(self) -> dict[str, tuple[float, float]]: ...
```

---

### RuVector-Enhanced Growth Intelligence

The growth engine's outputs are persisted and indexed in the RuVector+PostgreSQL
database tier to enable cross-patient learning and structured querying.

**Structured storage in PostgreSQL.** Fitted model parameters (V0, r, K, s,
e_j, sigma_obs), AIC/BIC scores, ensemble weights, and simulation results are
stored in PostgreSQL tables. This enables structured queries such as:
- "Retrieve all patients whose Gompertz growth rate `r` exceeds 0.05 day^-1"
- "Find patients where the treatment sensitivity `s` for SRS exceeds the
  cohort median"
- "Compare ensemble weights across the PROTEAS cohort"

These queries support the MedGemma reasoning layer (ADR-003) by providing
population-level context for individual patient narratives.

**Growth trajectory vectors in RuVector.** Each patient's observed growth
trajectory is encoded as a fixed-length vector (e.g., resampled to N
uniformly-spaced timepoints, concatenated with fitted parameter values) and
indexed in RuVector's HNSW store. This enables semantic search for "patients
with similar growth patterns":

```python
from ruvector import VectorDB, VectorEntry, SearchQuery

trajectory_db = VectorDB(dimensions=64)  # trajectory embedding dimension

# Encode and store a patient's growth trajectory
trajectory_vector = encode_trajectory(
    observed_volumes, fitted_params, timepoints
)
trajectory_db.insert(VectorEntry(
    id=f"{patient_id}_{lesion_id}",
    vector=trajectory_vector.tolist(),
    metadata={
        "patient_id": patient_id,
        "lesion_id": lesion_id,
        "model_form": best_model_name,
        "growth_rate": fitted_params["r"],
        "response_category": response_category,
    }
))

# Find patients with similar growth patterns
similar_patients = trajectory_db.search(SearchQuery(
    vector=query_trajectory.tolist(), k=10, include_vectors=False
))
```

**GNN-enhanced cross-patient learning.** RuVector's GNN layer
(MultiHeadAttention, GRUCell, LayerNorm) learns relationships between growth
trajectories as the corpus grows. Over time, the index refines its notion of
"similar growth pattern" beyond raw vector distance, incorporating learned
attention over trajectory features. This can improve counterfactual predictions
by identifying patients whose treatment response trajectories best match the
current patient, even when the raw growth parameters differ.

**Temporal hypergraph for treatment-response relationships.** RuVector's
temporal hypergraph structures encode treatment-response relationships as
time-aware graph edges. For example:

- `(Patient A) -[:TREATED_WITH {week: 4}]-> (SRS)`
- `(SRS on Patient A) -[:CAUSED]-> (growth_rate_decrease: 0.03 day^-1)`
- `(growth_rate_decrease) -[:CLASSIFIED_AS]-> (Partial Response)`

These relationships are queryable via Cypher:

```cypher
MATCH (p:Patient)-[:TREATED_WITH]->(d:Drug)
WHERE d.name = 'SRS'
RETURN p.patient_id, p.growth_rate, p.response_category
ORDER BY p.growth_rate ASC
```

This enables graph-based analysis of treatment effectiveness across the cohort,
providing context for the counterfactual engine: "Among patients treated with
SRS who had similar baseline growth rates, what was the typical response?"

---

## Python Libraries

| Library      | Version   | Purpose                                          |
|--------------|-----------|--------------------------------------------------|
| `scipy`      | >=1.11    | optimize.minimize (L-BFGS-B), integrate.solve_ivp|
| `numpy`      | >=1.24    | Array operations, random sampling                |
| `lmfit`      | >=1.2     | Parameter objects with bounds, alternative fitter |
| `matplotlib` | >=3.7     | Trajectory and counterfactual visualization      |
| `arviz`      | >=0.16    | Posterior diagnostics (post-hackathon Bayesian)  |
| `pymc`       | >=5.0     | Hierarchical Bayesian fitting (post-hackathon)   |
| `numpyro`    | >=0.13    | JAX-based MCMC alternative (post-hackathon)      |
| `pandas`     | >=2.0     | Therapy log and observation data management      |
| `psycopg`    | >=3.1     | Store fitted parameters and simulation results in PostgreSQL |
| `ruvector`   | >=0.1     | Vector indexing for growth trajectory similarity search |

---

## Consequences

### Positive

- **Interpretable.** All three growth models have well-understood biological
  interpretations. Parameters like growth rate and carrying capacity are
  meaningful to clinicians.
- **Fast.** Multi-start optimization with analytical solutions runs in seconds
  per patient. The full PROTEAS cohort (40 patients) can be fit in under a
  minute on a CPU-only Kaggle notebook.
- **Extensible.** The model interface allows adding new growth models (e.g.,
  von Bertalanffy, generalized logistic) without changing the fitting or
  counterfactual engine.
- **Uncertainty-aware.** Bootstrap uncertainty bands and ensemble weighting
  prevent overconfident predictions from any single model.
- **Counterfactual storytelling.** The ability to overlay "what-if" trajectories
  is the key differentiator for the MedGemma challenge narrative.
- **Cross-patient learning.** By indexing growth trajectories in RuVector and
  storing treatment-response relationships in the temporal hypergraph, the
  system improves its predictions over time. New patients benefit from the
  accumulated growth pattern corpus, enabling better initial parameter
  estimates and more informed counterfactual scenarios. RuVector's GNN layers
  automatically refine the notion of "similar growth pattern" as more cases
  are processed.

### Negative

- **Sparse data.** With only 3-8 observations per patient, complex models may
  be overparameterized. The exponential model (2-3 parameters) may be the only
  identifiable model for patients with very few timepoints.
- **Piecewise-constant treatment effect.** This is a simplification; real drug
  effects have pharmacokinetic dynamics. Acceptable for the hackathon but
  should be extended.
- **Bootstrap vs. Bayesian.** Bootstrap uncertainty can underestimate true
  posterior uncertainty, especially with small samples. The post-hackathon
  Bayesian extension addresses this.

---

## Alternatives Considered

### 1. Neural ODE / Physics-Informed Neural Network

Rejected for the hackathon phase. Neural ODEs (via `torchdiffeq`) can learn
flexible growth dynamics but require more data per patient than is available
(3-8 points). They also sacrifice interpretability -- a core requirement for
the clinical narrative. Considered for post-hackathon with population-level
pre-training.

### 2. Pure Bayesian from the Start (PyMC)

Rejected for timing reasons. Full MCMC sampling with PyMC/NumPyro adds
significant implementation and debugging overhead. The multi-start optimization
+ bootstrap approach delivers credible uncertainty estimates with a fraction of
the implementation effort. The code is structured to accept a Bayesian fitting
backend later.

### 3. Single Growth Model (Gompertz Only)

Rejected because model misspecification risk is high with a single model.
Some tumors exhibit exponential growth (far from carrying capacity), and the
Gompertz model can produce poor fits in those cases. The ensemble approach
is cheap (three fits per patient) and provides robustness.

### 4. Discrete-Time Difference Equations

Rejected because the treatment effect function E(t) is naturally continuous,
and the observations are irregularly spaced. Continuous-time ODEs with
analytical solutions for piecewise-constant E(t) are both more natural and
more efficient than discrete-time models that require interpolation.

### 5. Population-Level Mixed-Effects Model (NLME)

Considered as a complement to patient-specific fitting. In a nonlinear
mixed-effects framework, parameters are drawn from a population distribution,
which provides regularization for patients with few observations. Rejected for
the hackathon phase due to implementation complexity (would require `nlmixr2`
via R bridge or a custom PyMC hierarchical model). Planned for post-hackathon
as an extension of the Bayesian approach.

### 6. RuVector for Cross-Patient Growth Pattern Learning

Use RuVector's HNSW index and temporal hypergraph to store growth trajectories
as vectors and treatment-response relationships as graph edges, enabling
semantic search for similar growth patterns and graph-based treatment
effectiveness analysis.

- **Accepted** because:
  - **Similar-patient retrieval**: Encoding growth trajectories as vectors and
    indexing them in RuVector's HNSW store enables instant retrieval of patients
    with similar growth dynamics. This provides population-level context for
    individual patient predictions and informs counterfactual scenarios with
    empirical data from similar cases.
  - **Treatment-response graph**: The temporal hypergraph encodes causal
    relationships between treatments and growth rate changes, queryable via
    Cypher. This enables structured queries like "What was the typical growth
    rate response to SRS among patients with similar baseline tumors?"
  - **Self-improving retrieval**: RuVector's GNN layers refine the similarity
    metric as more cases are processed. Early in the cohort, retrieval relies
    on raw vector distance; as the GNN learns from the accumulating data, it
    discovers latent structure in growth patterns that improves match quality.
  - **Durable storage**: PostgreSQL provides durable, queryable storage for
    fitted parameters and simulation results, supporting multi-session
    workflows and reproducible analysis.
  - **Zero GPU cost**: RuVector runs on CPU, so trajectory indexing and
    similarity search do not compete with scipy optimization or downstream
    MedGemma inference for computational resources.
  - **Alternatives rejected**: FAISS provides HNSW but lacks graph queries and
    GNN self-improvement. A standalone Neo4j instance would add deployment
    complexity. Storing trajectories as JSON blobs in PostgreSQL would lack
    semantic search capability.

---

## References

- Benzekry, S., et al. "Classical Mathematical Models for Description and
  Prediction of Experimental Tumor Growth." PLoS Computational Biology, 2014.
- Gerlee, P. "The Model Muddle: In Search of Tumor Growth Laws." Cancer
  Research, 2013.
- Burnham, K.P. & Anderson, D.R. "Model Selection and Multimodel Inference."
  Springer, 2002.
- Ribba, B., et al. "A Tumor Growth Inhibition Model for Low-Grade Glioma
  Treated with Chemotherapy or Radiotherapy." Clinical Cancer Research, 2012.
