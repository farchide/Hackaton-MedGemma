# ADR-013: Evaluation & Metrics Framework

## Status

Accepted

## Date

2026-02-11

## Context

The Digital Twin Tumor system combines multiple subsystems -- measurement extraction,
lesion tracking across timepoints, growth simulation, and MedGemma-driven reasoning --
each of which requires its own evaluation methodology. A hackathon project carries an
elevated risk of overstating results, so we must embed honest reporting practices into
the evaluation framework from day one.

Key constraints that shape this decision:

1. **No large private clinical dataset.** We evaluate primarily on the PROTEAS public
   longitudinal CT dataset and on synthetic time-series we generate ourselves.
2. **Multiple output modalities.** The system produces segmentation masks, scalar
   measurements, categorical RECIST responses, narrative text, and probabilistic
   growth forecasts. A single metric cannot capture quality across all of these.
3. **Kaggle judging expectations.** Judges want reproducible notebooks with clearly
   stated metrics, confidence intervals, and acknowledged limitations.
4. **Runtime matters.** A digital twin that takes 30 minutes per scan is not viable
   for interactive clinical use. We must benchmark computational cost alongside
   accuracy.

We considered a flat list of metrics but found that grouping them by evaluation
layer -- measurement, tracking, simulation -- keeps the evaluation notebook
navigable and makes it clear which subsystem each metric targets.

## Decision

We adopt a **three-layer evaluation framework** with an additional cross-cutting
product-metrics layer and a response-categorization layer. All metrics are computed
in a single reproducible Jupyter notebook (`notebooks/evaluation.ipynb`) that can be
re-run end-to-end with one command.

### Layer 1: Measurement Evaluation

This layer evaluates how accurately the system extracts tumor measurements from
individual scans, independent of longitudinal tracking.

| Metric | Target | Notes |
|--------|--------|-------|
| Dice coefficient | >= 0.75 on PROTEAS held-out | Overlap between predicted and ground-truth segmentation masks |
| IoU (Jaccard) | >= 0.60 on PROTEAS held-out | Stricter than Dice; reported alongside for completeness |
| Hausdorff distance (95th percentile) | <= 8 mm | Captures worst-case boundary errors; 95th percentile avoids outlier domination |
| Longest-axis diameter MAE | <= 3 mm | Compared to radiologist-drawn RECIST diameters |
| Volume MAE | <= 15% relative | Percentage error on lesion volume in cubic millimeters |
| Short-axis diameter MAE | <= 3 mm | For lymph-node targets measured on short axis per RECIST 1.1 |

**Evaluation protocol:**

- Split PROTEAS into 70/15/15 train/val/test by patient ID (not by scan, to avoid
  data leakage from the same patient appearing in multiple splits).
- Report mean +/- standard deviation across patients in the test split.
- Report per-lesion-size strata: small (< 15 mm), medium (15-40 mm), large (> 40 mm).
- For Dice and IoU, compute both per-slice (2D) and per-volume (3D) variants.

### Layer 2: Tracking Evaluation

This layer evaluates the system's ability to maintain correct lesion identity across
longitudinal timepoints.

| Metric | Definition | Target |
|--------|------------|--------|
| Lesion match accuracy | Fraction of ground-truth lesion pairs correctly linked across consecutive timepoints | >= 0.90 |
| ID switches per patient | Number of times a lesion's identity is incorrectly swapped with another lesion | <= 0.5 mean |
| New lesion detection precision | Of flagged "new" lesions, fraction that are truly new | >= 0.80 |
| New lesion detection recall | Of truly new lesions, fraction that are flagged | >= 0.70 |
| Fragmentation rate | Fraction of ground-truth lesions split into two or more tracked IDs | Report only |
| Merge rate | Fraction of distinct ground-truth lesions collapsed into one tracked ID | Report only |

**Evaluation protocol:**

- Use PROTEAS patients with >= 3 timepoints.
- Ground truth is manually curated lesion correspondence annotations (created as part
  of data preparation; see ADR-011).
- Report metrics per-patient and aggregate.
- Stratify by number of target lesions per patient (1-2 vs 3-5).

### Layer 3: Simulation Evaluation

This layer evaluates the growth-model forecasting and uncertainty quantification
components of the twin engine.

| Metric | Definition | Target |
|--------|------------|--------|
| Forecast MAE (diameter) | Fit model on timepoints 1..n-1, predict timepoint n, measure absolute error in mm | <= 5 mm |
| Forecast MAE (volume) | Same protocol, volume in % relative error | <= 20% |
| Coverage probability (80% PI) | Fraction of held-out observations falling within the 80% prediction interval | 72-88% (nominal 80%) |
| Coverage probability (95% PI) | Same for 95% interval | 90-99% (nominal 95%) |
| Interval sharpness | Mean width of the 80% prediction interval in mm | Report only (narrower is better given adequate coverage) |
| Calibration slope | Slope of observed-vs-predicted quantile regression | 0.85-1.15 |

**Evaluation protocol:**

- Leave-last-timepoint-out backtest for every patient with >= 3 timepoints.
- For patients with >= 5 timepoints, also perform leave-last-two-out to evaluate
  multi-step forecasting.
- Generate calibration plots: predicted quantile vs observed frequency (PIT histogram).
- Report separate results for each growth model family (exponential, logistic,
  Gompertz, linear) and for the ensemble/model-averaged prediction.

### Product Metrics (Cross-Cutting)

These metrics ensure the system is usable in an interactive clinical workflow.

| Metric | Measurement Method | Target |
|--------|--------------------|--------|
| Runtime per scan (preprocessing + measurement) | `time.perf_counter` around pipeline | <= 30 s on T4 GPU |
| GPU VRAM peak | `torch.cuda.max_memory_allocated()` | <= 12 GB (fits single T4) |
| MedGemma query latency (p50, p95) | Timed wrapper around API/model call | p50 <= 3 s, p95 <= 8 s |
| Gradio UI time-to-interactive | Measured from scan upload to first displayed slice | <= 10 s |
| Total memory footprint (CPU RAM) | `psutil.Process().memory_info().rss` | <= 8 GB |

### Database & Vector Search Metrics

These metrics evaluate the performance of the RuVector+PostgreSQL persistence and
vector intelligence layer to ensure it meets interactive latency requirements and
demonstrates production readiness.

| Metric | Measurement Method | Target |
|--------|-------------------|--------|
| RuVector search latency (p50, p95) | Timed wrapper around vector search | p50 < 5ms, p95 < 20ms |
| RuVector batch insert throughput | Time 1000 vector inserts | > 5000 vectors/sec |
| PostgreSQL query latency | Timed SQL queries | p50 < 10ms |
| GNN improvement over baseline | Compare match accuracy with/without GNN | Report improvement % |
| Hybrid search precision@k | Fraction of top-k results relevant | >= 0.80 at k=10 |
| Similar case retrieval quality | Manual review of top-5 similar cases | >= 4/5 relevant |

**Evaluation protocol:**

- RuVector search latency is measured over 100 repeated queries with a warm cache,
  using realistic lesion embedding vectors (768-dimensional).
- Batch insert throughput is measured by inserting 1000 embedding vectors with
  associated metadata in a single batch operation.
- PostgreSQL query latency is measured on indexed audit log queries (by session,
  by action type, by timestamp range).
- GNN improvement is measured by comparing lesion match accuracy (Layer 2 metrics)
  with and without GNN-enhanced retrieval, using the same test set.
- Hybrid search precision@k is measured by combining vector similarity with graph
  relationship queries and manually evaluating the relevance of top-10 results.

### Response Categorization (MedGemma Reasoning)

Evaluate the quality of MedGemma-generated RECIST classifications and narratives.

| Metric | Definition |
|--------|------------|
| Agreement with RECIST rules | Fraction of cases where MedGemma's stated response category matches the mathematically derived category from measurements |
| Confusion matrix | 4x4 matrix over CR/PR/SD/PD categories |
| Cohen's kappa | Inter-rater agreement between MedGemma and rule-based RECIST classification |
| Narrative factual accuracy | Manual spot-check on 20 cases: count of factual errors per narrative |

**Evaluation protocol:**

- Compute rule-based RECIST 1.1 classification from measurements as ground truth.
- Run MedGemma reasoning module on same measurements.
- Report confusion matrix, accuracy, and Cohen's kappa with 95% bootstrap CI.
- For narrative evaluation, two team members independently rate 20 narratives for
  factual errors and inter-rater agreement is reported.

### Reporting Standards

1. **Honest limitations.** Every results section must include a "Limitations" subsection
   that states sample size, dataset bias, and generalizability caveats.
2. **No inflated clinical claims.** We do not claim the system is "clinically validated"
   or "ready for deployment." We describe it as a research prototype and proof of concept.
3. **Confidence intervals.** All aggregate metrics are reported with 95% bootstrap
   confidence intervals (1000 resamples).
4. **Effect of human edits.** We separately report metrics before and after human-in-the-loop
   corrections to quantify the value of interactive refinement.
5. **Failure case gallery.** The evaluation notebook includes a section showing the 5
   worst-performing cases per layer with qualitative analysis.

### Evaluation Notebook Structure

The notebook `notebooks/evaluation.ipynb` is organized as follows:

```
1. Setup & Data Loading
2. Layer 1: Measurement Metrics
   2.1 Segmentation (Dice, IoU, Hausdorff)
   2.2 Diameter and Volume Accuracy
   2.3 Per-Size-Stratum Breakdown
3. Layer 2: Tracking Metrics
   3.1 Lesion Match Accuracy
   3.2 ID Switches and Fragmentation
   3.3 New Lesion Detection
4. Layer 3: Simulation Metrics
   4.1 Leave-Last-Out Backtest
   4.2 Coverage and Calibration Plots
   4.3 Per-Model-Family Results
5. Product Metrics
   5.1 Runtime Benchmarks
   5.2 Memory Profiling
   5.3 Database Performance
       5.3.1 RuVector Search Latency
       5.3.2 Batch Insert Throughput
       5.3.3 PostgreSQL Query Performance
       5.3.4 GNN Learning Curve
6. Response Categorization
   6.1 RECIST Agreement
   6.2 Confusion Matrix and Kappa
7. Summary Dashboard
8. Limitations
9. Failure Case Gallery
```

### Python Libraries

| Library | Purpose | Version Constraint |
|---------|---------|-------------------|
| scikit-learn | Classification metrics, confusion matrix, bootstrap CI | >= 1.3 |
| monai | Medical image metrics (Dice, Hausdorff, surface distance) | >= 1.3 |
| matplotlib | Static plots, calibration curves | >= 3.8 |
| seaborn | Statistical visualizations, heatmaps | >= 0.13 |
| jupyter / ipykernel | Notebook execution | >= 1.0 |
| pandas | Tabular metric aggregation and reporting | >= 2.1 |
| numpy | Numerical computation | >= 1.26 |
| scipy | Statistical tests, bootstrap | >= 1.11 |
| ruvector | Vector search benchmarking, batch insert timing | >= 0.1 |

## Consequences

### Positive

- **Structured accountability.** Each subsystem has named metrics with explicit targets,
  preventing hand-waving about "good" results.
- **Reproducibility.** A single notebook with deterministic seeds produces all figures
  and tables for the submission.
- **Honest framing.** Built-in limitation reporting and failure-case galleries reduce
  the risk of overstating the prototype's capabilities.
- **Actionable during development.** Developers can run individual notebook sections to
  get fast feedback on the subsystem they are working on.
- **Database benchmarks demonstrate production readiness.** Quantified RuVector search
  latency, batch throughput, and PostgreSQL query performance provide evidence that the
  persistence layer meets interactive latency requirements and can scale beyond the
  hackathon prototype.

### Negative

- **Evaluation overhead.** Maintaining a comprehensive evaluation notebook takes
  significant effort for a hackathon timeline.
- **Target inflation risk.** Stated targets may be too ambitious for the available data;
  we mitigate this by reporting actuals honestly even if targets are missed.
- **Narrative evaluation is subjective.** Manual spot-checks of MedGemma narratives
  introduce inter-rater variability. We mitigate by reporting inter-rater agreement.

### Risks

- If PROTEAS has fewer usable patients than expected, confidence intervals will be wide
  and per-stratum analysis may be underpowered. Mitigation: supplement with synthetic
  data and clearly label which results use synthetic vs real data.
- Calibration assessment requires a reasonable number of forecasting instances. With
  few multi-timepoint patients, PIT histograms may be noisy. Mitigation: pool across
  lesions (not just patients) for calibration analysis, noting the non-independence.

## Alternatives Considered

### 1. Single Composite Score

We considered computing a single weighted score across all layers. Rejected because
composite scores obscure which subsystem is underperforming and make debugging harder.
We report per-layer metrics and let reviewers form their own assessment.

### 2. Leaderboard-Style Evaluation Only

We considered using only Kaggle-standard leaderboard metrics (e.g., mean Dice).
Rejected because our system has multiple output types that a single leaderboard
metric cannot capture. The three-layer approach covers all output modalities.

### 3. Clinical Trial-Style Evaluation

We considered a rigorous multi-reader multi-case (MRMC) study design. Rejected as
infeasible within hackathon scope and without access to practicing radiologists.
Acknowledged as future work in the limitations section.

### 4. Automated LLM-as-Judge for Narratives

We considered using a second LLM to evaluate MedGemma narrative quality. Rejected
due to the circular reasoning risk (LLM evaluating LLM). Manual spot-check with
explicit criteria is more defensible for a medical application.
