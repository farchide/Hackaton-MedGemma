Building a Maximalist Digital Twin Tumor for the MedGemma Hackathon
Framing, constraints, and what “maximalist” should mean in a hackathon
A maximalist Digital Twin Tumor project is best defined as a patient-specific, continuously updatable model that integrates longitudinal imaging measurements with treatment context to (a) summarize what changed, (b) quantify uncertainty, and (c) run counterfactual “what-if” simulations—while clearly communicating limitations and avoiding clinical overreach. This aligns with the core “digital twin” concept in oncology as a virtual simulation customized with real-world data to generate actionable insight about alternative treatment pathways. 

Your chosen competition (MedGemma Impact Challenge) is a Kaggle hackathon where submissions are judged on criteria including (weights shown on the Kaggle page snippet): Effective use of HAI‑DEF models (20%), Problem domain (15%), Impact potential (15%), Product feasibility (20%), and Execution & communication (30%). Use of at least one HAI‑DEF model (e.g., MedGemma) is mandatory. 
 The submission format is constrained: a video (≤3 minutes) plus write-up (≤3 pages) in a single submission package (per Kaggle overview snippet), and a single entry can also be considered for one special technology award category. 
 Kaggle’s submission instructions also emphasize that the submission must be a Kaggle Writeup and can be un-submitted/edited/re-submitted. 

Unknown constraints you explicitly flagged: team size and access to real clinical DICOM. This report assumes (a) access to at least public imaging datasets and (b) GPU available, but it treats clinical data access as unknown and designs a path that remains credible without it.

MedGemma’s role should be central and defensible: it is a developer model intended to be validated and adapted for specific use cases. Official guidance stresses it is not clinical-grade and not intended to directly inform diagnosis or patient-management decisions without appropriate validation and independent verification. 

End-to-end architecture for a Digital Twin Tumor system
A maximalist build works best if you explicitly separate the pipeline into measurement, state estimation, and simulation, with MedGemma orchestrating the “reasoning + narrative + safety” layer.

Architectural layers and responsibilities
Ingestion & governance layer

Accept imaging as DICOM or NIfTI, with optional prior reports and an explicit “therapy log” (start dates, regimen labels, dose changes).
If you accept any non-public DICOM, include a hard gate: PHI scanning + de-identification workflow. TCIA emphasizes DICOM as the primary radiology format and that archived datasets are de-identified. 
 For general HIPAA de-identification methods (Safe Harbor / Expert Determination), use HHS guidance as your north star. 
Preprocessing layer

CT: resample, intensity windowing; MRI: normalize sequences; ensure consistent orientation; for hackathon practicality, implement a “slice/patch selection” policy rather than full 3D on day one.
MedGemma 1.5 explicitly expands support to 3D volume representations of CT/MRI, and Google’s docs note that CT/MRI and whole-slide imaging require preprocessing with provided examples/notebooks. 
Human-in-the-loop measurement layer (core differentiator)

A clinician-like workflow: pick target lesions (or accept defaults), interactively segment or mark endpoints, and confirm lesion identity across time.
This layer drives credibility more than “fully automated RECIST.” (Automation can be partial; reliability and auditability wins in regulated contexts.)
Lesion tracking & identity graph

Represent lesions as trackable entities across time (a simple graph: nodes=timepoint lesions; edges=matches).
Detect “new lesions” and label them separately, aligning with iRECIST/RECIST thinking about new lesions and progression confirmation timing (see iRECIST definitions and confirmatory scan guidance). 
Digital Twin engine

Fits a patient-specific dynamical model per lesion and/or for total tumor burden.
Produces: growth velocity/acceleration, heterogeneity metrics, and counterfactual trajectories under alternative therapy schedules or sensitivity assumptions.
The “bidirectional update” principle—recalibrating with new data and quantifying uncertainty—is consistent with state-of-the-art digital twin framing and is explicitly discussed in TumorTwin (a modular digital twin framework enabling updating and uncertainty quantification). 
MedGemma reasoning layer

Converts structured measurements + uncertainty + evidence slices into tumor-board-ready narratives with safety constraints and refusal behaviors.
MedGemma is designed for medical text and image comprehension (built on Gemma 3) and can be adapted via prompt engineering, fine-tuning, or agentic orchestration. 
Why this architecture scores well on judging criteria
It makes MedGemma “non-optional” in your value proposition: not merely describing images, but grounding interpretation in measured evidence, then generating a transparent counterfactual explanation (what changed, why you think it changed, and what uncertainty remains). This directly supports “effective use of HAI-DEF models” and “product feasibility” criteria. 

Model stack choices and how to justify them with primary sources
A maximalist system will use multiple models, but your write-up should clearly separate: (a) measurement models and (b) reasoning models.

MedGemma and related HAI‑DEF models
MedGemma is Google’s open model collection for medical text and image comprehension, built on Gemma 3, with versions including MedGemma 1.5 4B multimodal and MedGemma 1 variants (including 4B multimodal and 27B text-only and multimodal). 
 The MedGemma 1.5 model card highlights expanded support for CT/MRI 3D, longitudinal imaging, and localization. 

A useful design pattern from Google’s model card: if your task is primarily image-based and you don’t need generation, MedSigLIP is recommended for image-based applications such as classification or retrieval. 
 In a maximalist system, this supports “evidence retrieval”: find the most similar prior slice/lesion view to compare, or cluster lesions by appearance as an interpretability aid.

Segmentation/tracking models: recommended options and tradeoffs
You can mix “foundation” interactive segmentation with robust medical baselines:

Component	Model choice	Why it fits a maximalist Digital Twin	Key risks / mitigations
Interactive lesion segmentation	MedSAM (medical adaptation of SAM)	Designed for broad medical segmentation; foundation-model style segmentation makes annotation fast and demo-friendly. 
May fail on modality/contrast edge cases; require human correction + confidence flags.
Auto-segmentation baseline	nnU-Net	Strong “workhorse” segmentation framework that self-adapts per dataset and is widely used; good for reproducible baselines. 
Needs task-specific training; mitigate by starting with pre-trained weights or limiting to a narrow tumor type.
General foundation segmentation (generic)	SAM	Highly promptable; strong interactive UX. 
Direct medical performance may be weaker than medical-adapted variants; prefer MedSAM for medical domains. 

A strong hackathon posture is: human-in-loop first, automation second. You can still optionally add “auto-suggest” segmentation, but you always show an editable contour and record whether the user accepted/edited it (auditability).

Tumor growth / response models to power counterfactuals
Counterfactual simulation is your “moonshot” feature. The trick is selecting growth models that are:

plausible enough to be meaningful,
simple enough to implement and explain,
and uncertainty-aware.
A robust approach is to run an ensemble of growth models and treat model-form as part of uncertainty. Classical tumor growth models frequently referenced include exponential, logistic, and Gompertz; comparative analyses often find Gompertz captures growth deceleration better in many settings. 

Model family	Form (conceptual)	Strengths	Weaknesses	Use in your system
Exponential	(V(t)=V_0 e^{rt})	Simple; good local approximation over short intervals	Unrealistic long-term; no saturation	Baseline sanity check and short-interval trend model
Logistic	(dV/dt=rV(1-V/K))	Saturating growth	K hard to identify with few timepoints	Alternate model for longer horizons
Gompertz	(dV/dt=rV \ln(K/V))	Empirically strong in many tumor growth settings; captures slowing growth 
Parameters can be correlated; still simplistic biologically	Primary mechanistic backbone model
Semi-parametric	GAM / GP	Flexible curves; captures nonlinearity	Risk of overfit with few timepoints	Use only when you have ≥5–6 timepoints per lesion

To be “digital twin” rather than “curve fitting,” you also need a treatment effect term (see next section) and a data assimilation/update step as new imaging arrives, consistent with digital twin framing. 

Human-in-the-loop measurement, data needs, and minimal viable dataset strategy
What data you need (minimum viable)
For a credible demo, you need longitudinal cases (≥3 timepoints is ideal, ≥2 minimum) plus either:

lesion segmentations (best), or
at minimum, slices on which you can annotate lesions.
Also capture a minimal “therapy log” per case (start date, therapy label, “no therapy change” markers) to drive counterfactual sliders.

Minimal viable dataset strategy: a tiered plan
A strong strategy is to build around public, longitudinal, annotated datasets and supplement with synthetic longitudinal augmentation clearly labeled as synthetic.

Tier A: Real longitudinal oncology dataset with follow-ups + segmentations (recommended core) A standout choice for longitudinal modeling is a brain metastases MRI dataset with tumor segmentations and follow-ups described in Scientific Data (Nature). It includes standard MRI sequences, detailed tumor subregion annotations, and post-treatment follow-ups at predefined intervals (6 weeks, 3/6/9/12 months). 
 The paper states the dataset includes 40 patients and hundreds of scans, plus clinical/treatment data and provides baseline and follow-up tumor masks. 
 This is unusually well-suited for “digital twin” style temporal modeling because you get repeated imaging and explicitly longitudinal ground truth.

Why it’s ideal for your hackathon:

multiple follow-ups per patient enable real “update the twin” demos, not just baseline vs follow-up,
masks allow objective evaluation (segmentation/volume/diameter),
treatment context is included (radiosurgery details etc.), enabling therapy timeline overlays. 
Tier B: Measurement variability / uncertainty calibration dataset Use TCIA’s RIDER Lung CT (“coffee-break” repeats) to calibrate measurement uncertainty: it provides same-day repeat CT scans in NSCLC patients and includes annotated lesion contours as a reference standard. 
 Even though it’s not therapy response, it’s excellent for quantifying how much measurements vary due to imaging and reconstruction settings—critical for honest uncertainty.

Tier C: Synthetic longitudinal augmentation (explicitly labeled) If your real longitudinal therapy dataset is small or modality-specific, generate synthetic follow-ups:

take a real lesion mask,
apply controlled morphological dilation/erosion + intensity perturbations to simulate growth/shrink,
optionally inject “new lesion” masks at later timepoints to test new-lesion logic. You must label these cases “synthetic trajectories” in the UI and write-up to avoid misleading clinical implications.
Human-in-loop measurement workflow mapped to RECIST/iRECIST principles
Even if you do not claim “RECIST-certified,” anchoring to RECIST 1.1 measurement logic instantly makes your system legible to clinicians and judges.

Key RECIST 1.1 essentials you can implement:

Target lesions: up to 2 per organ and 5 total. 
Lesions are measured by longest diameter (non-nodal) and short axis for nodal lesions, with nodal measurability thresholds (e.g., malignant node measurable if ≥15 mm short axis). 
Response cutoffs for target lesions: PR requires ≥30% decrease in sum of diameters; PD requires ≥20% increase from nadir plus ≥5 mm absolute increase; SD is neither. 
For immunotherapy-aware logic, iRECIST introduces unconfirmed progression (iUPD) vs confirmed progression (iCPD) and recommends confirmatory re-imaging to differentiate pseudoprogression; confirmatory scans are typically ≥4 and ≤8 weeks after iUPD. 
 Importantly, iRECIST explicitly states the guidelines are not intended to guide clinical practice decisions, which you can mirror as a safety statement in your app. 

Practical HITL UI actions:

“Select target lesions” (enforce 2-per-organ/5-total guardrails by default, with override for research).
“Measure diameter” (endpoint-click tool) + “Segment lesion” (MedSAM/nnU-Net suggestion + edit).
“Lock identity” (confirm lesion match across time or reassign).
“Flag new lesions” (user confirm).
This makes your digital twin defensible because your simulation is only as good as your measurement provenance.

Counterfactual simulation math and uncertainty quantification
Your uniqueness prize lives or dies here. The goal is not perfect biology; it’s transparent, uncertainty-aware, user-steerable simulation grounded in observed measurements.

A workable mathematical core: state-space + treatment effect
Define each lesion (i) with a latent “true burden” (V_i(t)) (volume or diameter-derived proxy). Observations are (y_{i,k}) at scan times (t_k).

Observation model (measurement uncertainty) [ y_{i,k} = V_i(t_k) + \epsilon_{i,k}, \quad \epsilon_{i,k}\sim \mathcal{N}(0,\sigma^2_{i,k}) ] (\sigma_{i,k}) can be lesion-specific and time-specific, composed of:

manual measurement error (endpoint placement),
segmentation variability (if using auto masks),
scan variability (calibrated from repeat-scan datasets like RIDER Lung CT). 
Growth dynamics with therapy effect Pick a base growth model; Gompertz is a strong default in the modeling literature. 

Example Gompertz per lesion: [ \frac{dV_i}{dt} = r_i V_i \ln\left(\frac{K_i}{V_i}\right) ]

Introduce treatment effect (E(t)) as a multiplicative or subtractive modifier on growth rate: [ \frac{dV_i}{dt} = \left(r_i - s_i,E(t)\right) V_i \ln\left(\frac{K_i}{V_i}\right) ] Where:

(r_i) = intrinsic growth parameter (lesion aggressiveness),
(K_i) = carrying capacity-like parameter (often weakly identified with few timepoints),
(s_i) = therapy sensitivity (lesion-specific, enabling heterogeneity),
(E(t)) = therapy exposure/effect function derived from the therapy log (piecewise constant or delayed).
This structure gives you the magic trick:

counterfactuals are just changes to (E(t)) (start date shifts, intensity changes, therapy swap coefficients).
Counterfactual engine: what you actually simulate
Counterfactual queries you can safely support in a demo:

“What if therapy effect started 4 weeks earlier?”
“What if therapy effect is 30% weaker (possible resistance)?”
“What if we switch to regimen B at week 12 (assumed higher sensitivity)?”
Your UI should label these outputs as hypothesis projections, not treatment recommendations.

Parameter fitting: fast, realistic approaches for a 10–14 day build
You need a method that runs quickly on a GPU workstation and is explainable.

Option that wins hackathons: multi-start optimization + bootstrap

Fit parameters (\theta={r_i,K_i,s_i}) by minimizing weighted squared error: [ \min_\theta \sum_k \frac{(y_{i,k}-\hat{V}i(t_k;\theta))^2}{\sigma^2{i,k}} ]
Estimate uncertainty by bootstrap resampling of timepoints and/or adding noise consistent with (\sigma_{i,k}).
Repeat for multiple model forms (exponential/logistic/Gompertz) and ensemble them.
This yields “uncertainty bands” without heavy MCMC.

Extended plan: hierarchical Bayesian digital twin A more “true digital twin” approach is to model lesion parameters hierarchically: [ r_i \sim \mathcal{N}(\mu_r,\tau_r^2),\quad s_i \sim \mathcal{N}(\mu_s,\tau_s^2) ] and infer posterior distributions over (\theta). This aligns with how digital twin frameworks describe recalibration and uncertainty quantification. 

Uncertainty quantification: what to implement and how to score it
In a maximalist system, you should separate uncertainty types and expose them in the UI.

Measurement uncertainty (must-have)

Driven by segmentation/edit history and known repeatability.
Use RIDER repeat scans to empirically bound “noise floors” for size/volume measurement variability. 
Model (epistemic) uncertainty (should-have) Use standard deep learning UQ methods:

MC Dropout (dropout at inference as approximate Bayesian inference). 
Deep ensembles (train multiple models / multiple parameter fits; strong practical uncertainty). 
In your system, deep ensembles can apply either to:

segmentation masks (multiple seeds / checkpoints),
growth model forms (model averaging),
parameter fit bootstraps.
Scenario uncertainty (must-have for honesty)

Uncertainty about therapy effect size, adherence, imaging interval variability.
Implement as “slider ranges” and show how projections change.
Visualization outputs your digital twin must have
Per lesion and total tumor burden curves with “therapy overlays.”
Heterogeneity view: multiple lesion trajectories showing divergence (mixed response).
“Breakpoint” markers: detect slope changes after therapy initiation (simple change-point detection).
Counterfactual panel: compare trajectories under user-defined therapy timing/intensity.
Evaluation protocol, metrics, and what “rigor” looks like in a hackathon
Because this is a hackathon demo, your evaluation must be component-level and honest. Judges tend to reward clarity over inflated claims.

Evaluation design: three layers
Measurement layer evaluation (objective) If you use datasets with segmentations, evaluate segmentation accuracy and measurement accuracy:

Dice coefficient / IoU for masks (if you do segmentation).
Absolute error in diameter and volume (if ground truth masks exist, diameter can be derived). The PROTEAS brain metastases dataset explicitly provides baseline and follow-up tumor segmentations and is designed to support training/validation of longitudinal models—ideal for objective measurement evaluation. 
Tracking layer evaluation

Lesion identity accuracy across time: % of lesions correctly matched to the same lesion on follow-up (manual adjudication on a small sample).
New lesion detection: precision/recall on synthetic injected lesions or annotated datasets.
Twin/simulation layer evaluation (semi-objective) You rarely have ground truth for “counterfactual,” so do:

Forecasting backtest: fit on first (n) timepoints, predict the next one; report MAE and coverage of uncertainty intervals.
Calibration: do predicted 80% intervals contain true measurements ~80% of the time? (Coverage probability.)
Clinical plausibility audit: a small rubric-based review (even 1–2 domain experts if available; if not, do structured self-audit with clear limitations).
Minimal but powerful metric tables
Recommended metrics checklist

Subsystem	Primary metric	Secondary metrics	Why it matters
Segmentation	Dice / IoU	Hausdorff distance, boundary F-score	Ensures lesion geometry is credible for downstream modeling
Measurement	Diameter MAE (mm)	Volume MAE (%), repeatability error	Directly tied to RECIST-style interpretation 
Tracking	Match accuracy	ID switches per patient	Stability across time is core to longitudinal reasoning
Response categorization	Agreement with RECIST-like rules	Confusion matrix; Cohen’s κ	If you implement PR/SD/PD rules, your output must align with definitions 
Uncertainty quality	Interval coverage	ECE / Brier (if probabilistic), AUROC for “error detection”	Judges reward systems that know when they don’t know 
Product feasibility	Runtime (sec/scan)	GPU VRAM peak, latency per query	Helps justify deployment choices

Evaluation dataset selection table (pragmatic)
Dataset	Modality	Longitudinal?	Labels available	What you use it for
PROTEAS longitudinal brain metastases dataset (Scientific Data)	MRI + follow-ups	Yes (multiple predefined follow-ups)	Tumor subregion segmentations + clinical/treatment context	End-to-end longitudinal twin demo; segmentation/measurement eval 
TCIA RIDER Lung CT	CT repeat scans	Repeat (short interval)	Annotated lesion contours	Measurement variability / uncertainty calibration 
TCIA QIN-LungCT-Seg analysis result	CT	Not necessarily therapy longitudinal	Tumor segmentations	Additional segmentation benchmarking 

Reproducible demo plan, safety/ethics/limitations, compute, and timelines
Reproducible demo plan aligned to Kaggle deliverables
Kaggle expects a tight narrative with evidence of real engineering and reproduction. Your minimal reproducible package should include:

A public code repo + a runnable demo app (Streamlit/Gradio).
A 3-minute video showing end-to-end usage.
A 3-page write-up following the competition template and explicitly answering judging criteria. 
Demo storyboard (3 minutes)

“Problem”: longitudinal imaging interpretation is hard; response patterns can be complex; uncertainty matters.
Upload a longitudinal case (from a public dataset).
Show lesion selection + interactive segmentation (fast).
Show tumor trajectories + uncertainty bands + heterogeneity.
Use counterfactual slider (therapy start shift or sensitivity change).
Generate tumor board summary with MedGemma, explicitly grounded in measurements + safety language.
Show audit log: what was manual vs automated; confidence flags.
Safety, ethics, and limitations to state explicitly
Non-clinical use statement (must be prominent) MedGemma’s own documentation warns outputs are not intended to directly inform clinical diagnosis/patient management and require independent verification. 
 Mirror this in your UI.

Response criteria are for trials, not practice iRECIST explicitly notes its guidelines are for consistent clinical trial data handling and not to guide clinical treatment decisions. 

AI ethics and governance If you include a “responsible deployment” section, anchor it in recognized guidance:

WHO’s guidance identifies ethical risks and principles for AI in health. 
FDA and partner regulators publish “Good Machine Learning Practice” guiding principles for AI/ML medical device development. 
Data privacy

Use only openly licensed, de-identified datasets for the demo.
If supporting user-provided DICOM: reference HIPAA de-identification guidance and implement a “do not upload PHI” gate. 
Key limitations (be explicit)

Imaging-only digital twin cannot infer biology/genomics; it’s a limited proxy.
Counterfactuals are not causal without randomized evidence—label as hypothesis projections.
Parameter identifiability is weak with few timepoints; show this via wide uncertainty bands.
Tumor response criteria vary by disease (RECIST vs RANO, etc.); you are implementing a focused subset.
Compute plan and practical GPU constraints
MedGemma is built on Gemma 3 variants and commonly used via Transformers; the MedGemma 1.5 model card notes Gemma 3 support in Transformers starting at a specific version. 
 For memory planning, Google’s Gemma 3 docs provide approximate memory to load models by size and quantization level, with an explicit caution that these estimates exclude prompt/KV cache and other overhead. 

Practical hackathon compute strategy:

Use MedGemma 1.5 4B for multimodal reasoning; reserve 27B for optional offline batch summarization only if you have the hardware/time. 
Use quantization where needed; keep context modest (avoid giant context windows for image-heavy prompts).
Preprocess 3D volumes into a controlled number of slices/patches before passing to MedGemma; this matches official notes that CT/MRI require preprocessing and example notebooks are provided. 
Timelines with prioritized implementation steps
Below is a 10–14 day maximalist-but-feasible path, followed by an extended plan.

Ten to fourteen day plan (ship a real twin + counterfactuals) Day 1–2: Repo + app skeleton; dataset loader; timeline viewer; case registry with provenance.
Day 3–4: Human-in-loop lesion selection + measurement tooling (click endpoints; basic mask editing). Implement “2 per organ / 5 total” default logic. 

Day 5–6: Integrate interactive segmentation (MedSAM) or baseline nnU-Net inference; always keep manual override. 

Day 7: Lesion identity graph + simple registration/nearest-slice matching; new lesion flagging.
Day 8–9: Digital twin engine v1: fit growth model(s), compute derivatives, detect breakpoints. Ground model choice in tumor growth literature (Gompertz + alternatives). 

Day 10: Counterfactual simulator + uncertainty bands (bootstrap + model ensemble). Add UI sliders.
Day 11: MedGemma narrative layer: “tumor board summary” grounded in your measurement table + uncertainty + safety disclaimers. 

Day 12: Evaluation notebook: segmentation/measurement metrics on a small held-out set; forecasting backtests + coverage.
Day 13–14: Video polish + write-up compression to 3 pages; finalize Kaggle Writeup submission packaging requirements. 

Extended plan (post-hackathon, make it a real research artifact)

Add hierarchical Bayesian inference for lesion-specific sensitivity (true “digital twin” update cycles). 
Add iRECIST workflow logic for immunotherapy response confirmation and iUPD/iCPD tracking. 
Expand to multi-modal digital twin inputs (radiomics, clinical covariates, labs) and document governance per GMLP. 
Priority order (what matters most for a uniqueness prize)
If you must ruthlessly prioritize for winning:

Counterfactual sliders + uncertainty bands (the “holy ****” moment).
Evidence-grounded narrative with MedGemma (measurement table → cited statements → caveats).
Human-in-loop lesion measurement (trust and auditability).
Heterogeneity/mixed response visualization (multiple lesions diverging).
Minimal, honest evaluation (backtests + coverage, not inflated clinical claims).
This combination is rare in hackathons because it merges “product” with “scientific rigor,” which is exactly what uniqueness awards tend to reward.
