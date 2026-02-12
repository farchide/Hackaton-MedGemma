# ADR-008: Uncertainty Quantification Framework

## Status

Accepted

## Date

2026-02-11

## Context

A digital twin that produces point estimates without uncertainty is not a digital twin -- it
is a curve fit. The project specification identifies uncertainty quantification as the single
most important differentiator for the Digital Twin Tumor system: "Judges reward systems that
know when they don't know."

The system must quantify and communicate uncertainty at multiple levels:

1. **Measurement level**: How reliable is a single diameter or volume measurement? This is
   driven by segmentation variability, scan-rescan noise, and human annotation variance.
2. **Model level**: How confident is the growth model in its parameter estimates and
   trajectory predictions? This is epistemic uncertainty driven by limited data and model
   form assumptions.
3. **Scenario level**: How sensitive are projections to assumptions about therapy effect
   size, patient adherence, and future imaging intervals?

The target datasets provide specific calibration opportunities:

- **RIDER Lung CT**: same-day repeat scans enabling empirical measurement of scan-rescan
  variability. This dataset provides the ground truth for Tier 1 calibration.
- **PROTEAS brain metastases**: longitudinal follow-ups enabling forecasting backtests
  where predicted intervals can be evaluated against actual measurements.

The key evaluation metric is **coverage probability**: predicted 80% credible intervals
should contain the true measurement approximately 80% of the time. Miscalibrated
uncertainty (either overconfident or underconfident) undermines trust.

## Decision

We implement a three-tier uncertainty framework where each tier feeds into the next,
producing composed uncertainty estimates that propagate through the entire simulation
pipeline.

### Tier 1: Measurement Uncertainty

Measurement uncertainty quantifies the noise floor: even if the tumor has not changed,
how much would our measurement vary due to technical and human factors?

#### Composition Formula

For lesion `i` at timepoint `k`, the measurement uncertainty variance is:

```
sigma^2_{i,k} = sigma^2_manual + sigma^2_auto + sigma^2_scan
```

Where:

- `sigma^2_manual`: **Manual annotation variance**. Estimated from inter-observer
  variability studies or, within our system, from the difference between user-edited
  and auto-suggested contours. Default prior: 1.5mm standard deviation for longest
  diameter measurement (literature-consistent for radiologist inter-observer variability
  on lesion diameter).

- `sigma^2_auto`: **Auto-segmentation variance**. Computed from deep ensemble
  disagreement (see Tier 2) on the segmentation mask. Specifically, we compute the
  longest diameter on each of N ensemble member masks and take the variance:
  ```
  sigma^2_auto = Var([d_1, d_2, ..., d_N])
  ```
  where `d_j` is the longest diameter derived from ensemble member `j`.

- `sigma^2_scan`: **Scan-rescan variability**. Calibrated empirically from the RIDER
  Lung CT dataset. For each lesion in RIDER, we compute the absolute difference in
  diameter between the two same-day scans and fit a heteroscedastic noise model:
  ```
  sigma_scan(d) = a + b * d
  ```
  where `d` is the lesion diameter, and `a`, `b` are fitted coefficients. This captures
  the empirical observation that larger lesions have proportionally larger scan-rescan
  variability. Default fallback (if RIDER calibration unavailable): `sigma_scan = 1.0mm`.

#### Implementation

```python
@dataclass
class MeasurementUncertainty:
    sigma_manual_mm: float       # Standard deviation from annotation variability
    sigma_auto_mm: float         # Standard deviation from segmentation ensemble
    sigma_scan_mm: float         # Standard deviation from scan-rescan noise
    total_sigma_mm: float        # Composed: sqrt(sum of variances)
    confidence_flag: str         # "high" | "medium" | "low" based on total_sigma

    @staticmethod
    def compose(manual: float, auto: float, scan: float) -> "MeasurementUncertainty":
        total = math.sqrt(manual**2 + auto**2 + scan**2)
        flag = "high" if total < 2.0 else ("medium" if total < 4.0 else "low")
        return MeasurementUncertainty(manual, auto, scan, total, flag)
```

The confidence flag thresholds (2mm, 4mm) are chosen based on RECIST significance: a 4mm
uncertainty on a 20mm lesion is a 20% relative error, which approaches the RECIST threshold
for progression (20% increase). Measurements with "low" confidence trigger a UI warning.

### Tier 2: Model / Epistemic Uncertainty

Model uncertainty captures what the model does not know due to limited training data,
model architecture choices, and growth model form assumptions.

#### Segmentation Model Uncertainty (MC Dropout + Deep Ensembles)

For segmentation masks, we use two complementary approaches:

1. **MC Dropout** (lightweight, fast): At inference time, keep dropout layers active and
   run T=10 forward passes. Compute per-voxel variance across passes:
   ```
   p_mean(x) = (1/T) * sum_{t=1}^{T} p_t(x)
   sigma^2_epistemic(x) = (1/T) * sum_{t=1}^{T} (p_t(x) - p_mean(x))^2
   ```
   This produces a voxel-level uncertainty map overlaid on the segmentation.

2. **Deep Ensembles** (higher quality, higher cost): Train or use M=3 segmentation model
   checkpoints (different random seeds or training stages). Compute mask agreement:
   ```
   agreement_ratio = |intersection of M masks| / |union of M masks|
   ```
   Low agreement ratio (< 0.7) flags uncertain segmentation boundaries.

The ensemble disagreement directly feeds into `sigma^2_auto` in Tier 1.

#### Growth Model Epistemic Uncertainty (Ensemble + Bootstrap)

For the digital twin growth models, epistemic uncertainty comes from two sources:

1. **Model form uncertainty**: We fit three growth model families (Exponential, Logistic,
   Gompertz) to each lesion's trajectory and retain all three. The spread across model
   predictions at future timepoints quantifies structural uncertainty:
   ```
   sigma^2_model_form(t) = Var([V_exp(t), V_logistic(t), V_gompertz(t)])
   ```

2. **Parameter uncertainty** (within each model): Bootstrap resampling of the observed
   timepoints. For each of B=100 bootstrap resamples:
   - Resample the K observed measurements with replacement
   - Add noise consistent with Tier 1 `sigma_{i,k}` to each resampled point
   - Re-fit model parameters
   - Generate trajectory prediction

   The resulting B trajectories per model provide parameter uncertainty:
   ```
   sigma^2_param(t) = (1/B) * sum_{b=1}^{B} (V_b(t) - V_mean(t))^2
   ```

3. **Composed model uncertainty**:
   ```
   sigma^2_epistemic(t) = sigma^2_model_form(t) + mean_over_models(sigma^2_param(t))
   ```

#### Implementation

```python
@dataclass
class ModelUncertainty:
    model_predictions: Dict[str, np.ndarray]  # model_name -> trajectory array
    bootstrap_trajectories: np.ndarray         # Shape: (B, T_future)
    sigma_model_form: np.ndarray               # Per-timepoint model form uncertainty
    sigma_param: np.ndarray                    # Per-timepoint parameter uncertainty
    sigma_total: np.ndarray                    # Composed epistemic uncertainty

    def prediction_interval(self, alpha: float = 0.80) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) bounds for the alpha prediction interval."""
        low_q = (1 - alpha) / 2
        high_q = 1 - low_q
        all_trajectories = self.bootstrap_trajectories  # (B, T)
        lower = np.quantile(all_trajectories, low_q, axis=0)
        upper = np.quantile(all_trajectories, high_q, axis=0)
        return lower, upper
```

### Tier 3: Scenario Uncertainty

Scenario uncertainty captures factors outside the model that the user controls or
that are inherently unknowable: therapy effect magnitude, patient adherence,
imaging schedule variability.

#### User-Controllable Parameters (UI Sliders)

| Parameter | Default Range | UI Control | Effect on Simulation |
|---|---|---|---|
| Therapy effect size | [0.5x, 2.0x] of fitted `s_i` | Slider | Scales growth rate reduction |
| Therapy start offset | [-4, +4] weeks | Slider | Shifts `E(t)` onset |
| Adherence factor | [0.5, 1.0] | Slider | Multiplicative on `E(t)` |
| Imaging interval | [4, 12] weeks | Slider | Changes observation density |
| Resistance onset | [never, 4, 8, 12 weeks] | Dropdown | Gradual `s_i` decay |

#### Propagation Through Simulation

When a user adjusts a slider, the system:
1. Modifies the relevant parameter in the growth ODE.
2. Re-runs the ensemble of B bootstrap trajectories with the modified parameter.
3. Recomputes prediction intervals.
4. Updates the visualization in real-time (target: < 500ms for 100 bootstrap samples).

The scenario uncertainty is displayed as **shaded bands** that widen or narrow based on
slider positions:
```
sigma^2_total(t) = sigma^2_measurement(t_k) + sigma^2_epistemic(t) + sigma^2_scenario(t)
```

Where `sigma^2_scenario(t)` is the variance across the user-specified parameter range,
computed by evaluating the model at the slider endpoints and taking the spread.

### Calibration Protocol

Calibration ensures that stated uncertainty intervals are neither overconfident nor
underconfident.

#### Method: Forecasting Backtest on PROTEAS

1. For each patient with K >= 3 timepoints, fit the model on timepoints 1...(K-1).
2. Predict timepoint K with 80% prediction interval.
3. Check whether the actual measurement at K falls within the interval.
4. Aggregate coverage across all patients and lesions.

**Target**: 80% interval coverage should be 75-85% (allowing for small-sample noise).

If coverage is too low (overconfident): inflate `sigma` by a calibration factor `c`:
```
sigma_calibrated = c * sigma_raw, where c = sqrt(target_coverage / observed_coverage)
```

If coverage is too high (underconfident): deflate similarly.

#### Reliability Score

Each projection receives a composite reliability score displayed in the UI:

```python
def reliability_score(n_timepoints: int, total_sigma_mm: float,
                      model_agreement: float, coverage_history: float) -> str:
    score = 0.0
    score += min(n_timepoints / 6, 1.0) * 0.3    # More data = more reliable
    score += max(1 - total_sigma_mm / 10, 0) * 0.2  # Lower noise = more reliable
    score += model_agreement * 0.25                # Model consensus
    score += coverage_history * 0.25               # Historical calibration
    if score >= 0.7:
        return "HIGH"
    elif score >= 0.4:
        return "MODERATE"
    else:
        return "LOW"
```

### RuVector-Enhanced Uncertainty Calibration

Measurement uncertainty estimates are strengthened by leveraging population-level data
from similar lesions, retrieved via RuVector's vector similarity search. This approach
uses the principle that lesions with similar morphology (as captured by their MedSigLIP
embeddings) tend to exhibit similar measurement uncertainty profiles.

**Process:**

1. When a new lesion is measured, RuVector searches for the k=20 most similar historical
   lesions by embedding cosine similarity.
2. For each retrieved similar lesion, the system loads its historical measurement
   uncertainty profile: `sigma_manual`, `sigma_auto`, `sigma_scan`, and the composed
   `total_sigma`.
3. The population-level uncertainty distribution from these similar cases serves as an
   **informative prior** for the new case's `sigma_obs` parameter in the state-space
   growth model (ADR-005).
4. Specifically, the prior for `sigma_obs` is set as the median of the similar cases'
   `total_sigma` values, weighted by embedding similarity:
   ```python
   similar_cases = ruvector_client.search(
       query_vector=current_lesion_embedding,
       k=20,
       filter={"modality": current_modality},
   )
   weights = np.array([case.similarity for case in similar_cases])
   sigmas = np.array([case.metadata["total_sigma_mm"] for case in similar_cases])
   sigma_prior = np.average(sigmas, weights=weights)
   ```
5. For cases with limited data (fewer than 3 timepoints), this similarity-informed
   prior substantially reduces uncertainty compared to using a generic population mean.

**GNN-Enhanced Calibration:**

RuVector's GNN layer learns which embedding features best predict measurement
uncertainty over time. As more lesions are measured and their uncertainty profiles
recorded, the GNN refines its internal representation to improve calibration accuracy.
This creates a virtuous cycle: each new measurement improves the system's ability to
estimate uncertainty for future similar lesions.

**Fallback:** When RuVector is unavailable (e.g., Kaggle notebook environment), the
system falls back to the static calibration protocol described above using RIDER-derived
estimates.

### Visualization Strategy

1. **Trajectory plots**: Central prediction line with shaded bands.
   - Inner band (darker): Tier 1 measurement uncertainty only.
   - Middle band (medium): Tier 1 + Tier 2 (measurement + model).
   - Outer band (lightest): Tier 1 + Tier 2 + Tier 3 (full composed uncertainty).

2. **Confidence flags on measurements**: Each observed measurement point displays a
   vertical error bar representing `+/- 2 * sigma_{i,k}` (95% measurement CI).

3. **Reliability badge**: Each lesion projection shows a colored badge:
   - Green: HIGH reliability (>= 4 timepoints, low noise, models agree).
   - Yellow: MODERATE reliability (2-3 timepoints, moderate noise).
   - Red: LOW reliability (< 2 timepoints, high noise, models disagree).

4. **Calibration dashboard** (evaluation notebook): Coverage probability plot showing
   predicted vs. observed coverage across confidence levels (20%, 40%, 60%, 80%, 95%).

### Python Library Stack

| Library | Purpose |
|---|---|
| `numpy` | Array operations, bootstrap resampling |
| `scipy.stats` | Distribution fitting, quantile functions, statistical tests |
| `torch` | MC Dropout inference on segmentation models |
| `matplotlib` | Static uncertainty band plots, calibration curves |
| `plotly` | Interactive trajectory plots with hover-over uncertainty details |
| `ruvector` | Similarity-based calibration data retrieval from historical lesions |

## Consequences

### Positive

- **Trust through transparency**: Users can see exactly why a projection is uncertain
  and which component dominates. This is the key differentiator for hackathon judges.
- **Calibrated intervals**: The backtest calibration protocol ensures stated intervals
  are meaningful, not arbitrary. This directly addresses the evaluation criterion
  "uncertainty quality: interval coverage."
- **User agency**: Tier 3 sliders give users the ability to explore "what if" scenarios
  and see how uncertainty responds, making the demo interactive and engaging.
- **Composable design**: Each tier is independent and can be improved without affecting
  the others. Tier 1 can be improved with better calibration data; Tier 2 with more
  ensemble members; Tier 3 with more scenario parameters.
- **Honest communication**: The reliability score prevents overconfident presentation of
  projections based on insufficient data.
- **Population-level uncertainty intelligence**: RuVector-based similarity search enables
  uncertainty priors informed by similar historical cases, reducing calibration uncertainty
  for new patients and improving the reliability of predictions for cases with limited data.

### Negative

- **Computational cost**: MC Dropout (T=10 passes) and bootstrap (B=100 resamples)
  multiply inference time. Mitigated by: (a) MC Dropout runs on GPU in batch, (b)
  growth model bootstrap is lightweight (ODE fits, not neural network inference).
- **Complexity for users**: Three-tier uncertainty may confuse non-technical users.
  Mitigated by defaulting to a single "total uncertainty" band with optional expansion.
- **Bootstrap limitations**: With very few timepoints (K=2), bootstrap resamples provide
  limited diversity. The reliability score honestly flags this as LOW confidence.
- **Calibration requires held-out data**: The PROTEAS dataset has limited patients (40);
  leave-one-out cross-validation may be necessary, reducing calibration sample size.

### Risks

- MC Dropout may underestimate true epistemic uncertainty compared to proper Bayesian
  inference. This is a known limitation accepted for hackathon practicality. Post-hackathon,
  hierarchical Bayesian inference (as described in the spec) would replace this.
- Scenario uncertainty ranges are subjective (chosen by developers, adjusted by users).
  Mitigation: document the rationale for default ranges and allow full user override.

## Alternatives Considered

### 1. Full Bayesian Inference (MCMC / Variational Inference)

Rejected for hackathon timeline. Hierarchical Bayesian inference with MCMC (e.g., via
PyMC or NumPyro) would provide theoretically superior posterior distributions but requires
significant implementation and debugging time. The bootstrap + ensemble approach provides
80% of the benefit at 20% of the implementation cost. Bayesian inference is planned for
the post-hackathon extended roadmap.

### 2. Conformal Prediction for Distribution-Free Intervals

Considered as an alternative calibration method. Conformal prediction provides
distribution-free coverage guarantees but requires an exchangeability assumption that may
not hold for longitudinal data (measurements within a patient are correlated). We retain
conformal prediction as a post-hackathon enhancement but use empirical calibration for now.

### 3. Single Uncertainty Number (No Tiers)

Rejected. Collapsing all uncertainty into a single number hides the source of uncertainty
and prevents users from understanding whether improving the scan quality, adding more
timepoints, or changing therapy assumptions would most reduce uncertainty. The three-tier
decomposition is essential for interpretability and aligns with the spec's emphasis on
transparent uncertainty communication.

### 4. Deterministic Error Bounds Instead of Probabilistic Intervals

Rejected. Worst-case deterministic bounds (e.g., "measurement is between X-3mm and X+3mm")
are overly conservative and not actionable. Probabilistic intervals with calibrated
coverage provide more useful information and align with modern uncertainty quantification
best practices in medical imaging.

### 5. Dropout-Only Uncertainty (No Ensembles)

Rejected. MC Dropout alone is known to underestimate uncertainty for out-of-distribution
inputs. The combination with deep ensembles (for segmentation) and model-form ensembles
(for growth models) provides more robust uncertainty estimates. The additional computational
cost is acceptable given the small number of lesions per patient (typically 1-5 target
lesions per RECIST).
