# Digital Twin: Tumor Response Assessment -- Video Script (3 minutes)

> **Target:** MedGemma Impact Challenge, Main Track ($30K first place) + Agentic Workflow Prize ($5K)
>
> **Judging weights:** Execution & Communication (30%), Product Feasibility (20%), HAI-DEF Model Use (20%), Problem Domain (15%), Impact Potential (15%)

---

## Pre-recording Setup

- Browser open to Gradio app at `localhost:7860` (dark theme auto-applied)
- Demo Patient 1 ("Classic Responder -- Brain Metastases") pre-loaded in the database
- Demo Patient 3 ("Mixed Responder -- Heterogeneous") available for optional B-roll
- Screen recording at 1080p minimum (1440p preferred), 30fps
- OBS Studio with window capture (crop out OS chrome)
- Microphone: quiet room, pop filter, consistent distance
- Tab order visible in the UI: Dashboard, Upload & Preprocess, Annotate & Measure, Track Lesions, Simulate, Narrate
- All plots should be pre-warmed (load patient once before recording to cache any initial delays)
- Clear browser console to avoid any error badges

---

## [0:00-0:20] Opening -- The Problem (20 seconds)

**Screen:** Fade in from black to a title card:

```
DIGITAL TWIN: TUMOR RESPONSE ASSESSMENT
Longitudinal Imaging + Counterfactual Simulation + MedGemma Reasoning
```

Hold for 3 seconds, then crossfade to a split-screen showing two brain MRI slices side by side (Week 0 and Week 12 from Demo Patient 1), with a subtle cyan outline around each lesion.

**Voiceover:**

> "A patient with brain metastases receives six cycles of immunotherapy. Their oncologist reviews five scans over twelve months -- each one with multiple lesions, each lesion potentially responding differently. Did the tumor shrink enough? Is it still shrinking? What would have happened if treatment started a month earlier? Today, answering these questions means manually measuring lesions scan by scan, with no uncertainty quantification and no way to explore alternatives. We built a system that changes that."

**Key message:** Longitudinal tumor assessment is hard, subjective, and lacks uncertainty-awareness. This is a real, high-magnitude clinical problem.

**Judging criteria addressed:**
- **Problem Domain (15%):** Establishes unmet need, defines the user (oncologist/radiologist), articulates the magnitude (millions of cancer patients undergo serial imaging).
- **Execution & Communication (30%):** Strong narrative hook. Judges should feel the problem before seeing the solution.

**Timing notes:** Title card appears at 0:00, crossfade to MRI comparison at 0:03, voiceover runs continuously. The two MRI slices provide immediate visual credibility -- this is real medical imaging data.

---

## [0:20-0:45] Dashboard -- Loading a Patient (25 seconds)

**Screen actions (in order):**
1. (0:20) Show the full Dashboard tab. The cyan gradient header "Digital Twin: Tumor Response Assessment" and the yellow disclaimer banner ("DISCLAIMER -- This tool is for research and educational purposes only...") are visible at the top.
2. (0:22) Mouse moves to the "Select Demo Patient" radio buttons. Click "Patient 1 -- Classic Responder (Brain Metastases, Immunotherapy)".
3. (0:24) Click the "Load Patient" button (cyan gradient, glows on hover).
4. (0:25) The dashboard populates: patient profile HTML card appears (cancer type, stage, age/sex, histology, ECOG PS), followed by a therapy timeline visualization showing treatment windows.
5. (0:28) Slowly scroll down to reveal the three dashboard plots loading simultaneously: RECIST Waterfall chart (left), Lesion Tracking graph (right), and Growth Trajectory plot (bottom).
6. (0:33) Pause on the Growth Trajectory plot -- point out the observed data points (white markers with cyan borders), the fitted curve (cyan line), and the 95% confidence band (translucent cyan fill).
7. (0:38) Open the "Volume Viewer" accordion. Select "Week 0" from the timepoint dropdown. An axial brain MRI slice appears. Drag the slice slider slowly from slice 20 to slice 40 to show lesion cross-sections.

**Voiceover:**

> "Our system loads longitudinal oncology cases from public datasets -- here, a brain metastases patient with five imaging timepoints over twelve months of immunotherapy. With one click, we populate a full clinical dashboard: patient profile, treatment history, and three interactive views. The RECIST waterfall shows response at each timepoint. The identity graph tracks individual lesions across scans. And the growth trajectory fits a mathematical model with uncertainty bands to every measurement. Everything starts from the data."

**Key message:** One-click loading of a complete longitudinal case. The system is data-driven, not a toy. Disclaimer is visible (safety-conscious design).

**Judging criteria addressed:**
- **Product Feasibility (20%):** Demonstrates a real, working application with professional UI. Shows the Gradio stack is deployable. Dark glassmorphism theme communicates production quality.
- **Execution & Communication (30%):** Clean, fast interaction. No loading spinners or errors. The three plots appearing simultaneously is visually impressive.
- **Problem Domain (15%):** Uses a real public dataset (PROTEAS brain metastases with longitudinal follow-ups), not synthetic toy data.

**Timing notes:** The "one click to full dashboard" moment at 0:25-0:28 should feel effortless. Rehearse this transition to ensure it is smooth. If there is any lag, pre-load and use a screen recording edit to tighten the gap.

---

## [0:45-1:15] Measurements & RECIST Classification (30 seconds)

**Screen actions (in order):**
1. (0:45) Click the "Annotate & Measure" tab. The HITL annotation interface appears: annotated slice viewer on the left, controls on the right.
2. (0:48) The slice viewer shows an axial MRI with a lesion visible. Type a bounding box in the "BBox (x0,y0,x1,y1)" field: `80,60,160,140`. Click "Segment".
3. (0:52) The segmentation overlay appears: a semi-transparent green mask over the lesion. The "Volume (mm^3)" field populates with the computed value. The info textbox shows voxel count and estimated volume.
4. (0:56) Type `18.5` in the "Manual Diameter (mm)" field. Click "Save Diameter". A toast notification confirms "Diameter: 18.5 mm".
5. (0:59) The Measurements table at the bottom updates with one row: Lesion ID, diameter, volume, method ("Semi-auto"), timepoint.
6. (1:02) Click "Classify RECIST". The RECIST result appears below: "**PR** | Sum: 18.5 mm | Baseline: -32.1% | Nadir: -32.1%". The waterfall reference lines for PR (-30%) and PD (+20%) provide context.
7. (1:06) Brief pause on the measurement table, then transition.

**Voiceover:**

> "The human-in-the-loop measurement workflow is the foundation of trust. A clinician draws a bounding box, the system segments the lesion -- here using a medical-adapted segmentation model -- and the user confirms or edits the result. Diameters are recorded manually, matching the clinical workflow of RECIST 1.1 where the longest diameter is the standard measure. The system then classifies response automatically: this patient achieves a partial response, a 32% decrease from baseline. Every measurement records its method -- auto, semi-auto, or manual -- because auditability matters."

**Key message:** Human-in-the-loop design with full provenance tracking. Not black-box AI -- the clinician is in control. RECIST 1.1 anchoring makes it clinically legible.

**Judging criteria addressed:**
- **Effective Use of HAI-DEF Models (20%):** Demonstrates the measurement pipeline that feeds into MedGemma reasoning. The segmentation model (MedSAM-adapted) is part of the HAI-DEF ecosystem. Shows the system is not just "call an API" but a thoughtful pipeline.
- **Product Feasibility (20%):** Realistic clinical workflow. The semi-auto approach with manual override is how real radiology software works. RECIST 1.1 compliance is a regulatory-aware design choice.
- **Impact Potential (15%):** Every oncology patient undergoing treatment monitoring could benefit from standardized, auditable measurements.

**Timing notes:** The segmentation result appearing at 0:52 is a key credibility moment. If the green mask overlay looks clean and follows the lesion boundary, it signals engineering quality. Ensure the bounding box values produce a visually satisfying segmentation for the demo case.

---

## [1:15-1:45] Lesion Tracking & Longitudinal View (30 seconds)

**Screen actions (in order):**
1. (1:15) Click the "Track Lesions" tab. Click "Build Graph".
2. (1:17) The Lesion Identity Graph appears: a DAG visualization with nodes (cyan circles) representing lesion observations at each timepoint, connected by edges showing identity matches. Timepoints run left to right. Multiple lesion rows show different lesions tracked independently.
3. (1:20) Hover over a node to see the tooltip: "L:abc123 V:1450" (lesion ID and volume). The summary textbox shows "Nodes: 15 | Edges: 12 | Lesions: 3".
4. (1:24) Scroll down to the RECIST Waterfall chart. The bars are color-coded: green for PR (partial response), yellow for SD (stable disease), red for PD (progressive disease). Dotted reference lines at -30% (PR threshold) and +20% (PD threshold) are visible. The trajectory shows response deepening over time.
5. (1:28) Return to the Dashboard tab briefly. Point out the Growth Trajectory plot. Multiple lesion curves are visible, showing heterogeneous response: one lesion shrinking (responder), one stable, one growing slightly (mixed response).
6. (1:35) Hover over the uncertainty band on the growth plot. The translucent cyan fill around the fitted curve widens at later timepoints where fewer measurements constrain the model.

**Voiceover:**

> "Lesions are tracked as entities across time using an identity graph -- each node is a lesion observation, each edge a confirmed match. This is not just 'biggest lesion at each scan'; it is true longitudinal tracking. The RECIST waterfall summarizes response at each timepoint, but the real clinical insight is in the heterogeneity view: three lesions, three different trajectories. One is responding, one is stable, one is growing. Mixed response like this is common in immunotherapy, and traditional RECIST can miss it entirely. Our growth model captures this divergence, and the uncertainty bands widen honestly where data is sparse."

**Key message:** Individual lesion tracking reveals heterogeneity that aggregate measures hide. Uncertainty bands are honest -- they widen where evidence is thin. This is the scientific rigor that distinguishes a digital twin from a dashboard.

**Judging criteria addressed:**
- **Problem Domain (15%):** Heterogeneous/mixed response is a well-documented clinical challenge, especially in immunotherapy. Judges who are clinically aware will recognize this as a real problem.
- **Product Feasibility (20%):** The identity graph, waterfall, and growth plots are all interactive Plotly visualizations. This is a deployable product, not a Jupyter notebook.
- **Execution & Communication (30%):** The visual storytelling of "three lesions, three stories" is memorable. The uncertainty bands widening is a subtle but powerful detail.

**Timing notes:** The heterogeneity view (1:28-1:35) is the setup for the next section's "wow moment." Linger on it. Let the audience absorb that different lesions behave differently before showing the counterfactual simulation.

---

## [1:45-2:15] Simulation & Counterfactuals -- The "Wow" Moment (30 seconds)

**THIS IS THE SECTION THAT WINS OR LOSES THE COMPETITION.**

**Screen actions (in order):**
1. (1:45) Click the "Simulate" tab. The Digital Twin Growth Simulation interface appears: scenario dropdown, three sliders (Therapy Shift, Effect Multiplier, Resistance Onset), and a large growth trajectory plot on the right.
2. (1:48) The dropdown shows "Natural history (no treatment)" selected. Click "Run Simulation". The plot updates: the observed data points remain, the fitted curve remains, and a new dashed orange line appears showing projected growth without any treatment. It diverges sharply upward. The uncertainty band around the counterfactual is wider than the observed trajectory.
3. (1:53) Change the dropdown to "Earlier treatment". Drag the "Therapy Shift" slider from 0 to -4 (weeks). Set "Effect Multiplier" to 0.7. Click "Run Simulation". A new dashed line appears showing a faster, deeper response -- the tumor shrinks earlier and further.
4. (1:58) Change to "Treatment resistance". Drag "Resistance Onset" slider to week 8. Click "Run Simulation". The plot now shows three scenarios stacked: (a) observed reality, (b) earlier treatment, (c) resistance at week 8. The resistance scenario shows initial response followed by regrowth -- a clinically terrifying but realistic pattern.
5. (2:03) **The key visual moment:** Slowly drag the "Therapy Shift" slider from -4 back through 0 to +4. As the slider moves, the simulated curve shifts in real time (or with quick re-simulation clicks), showing how delayed treatment leads to worse outcomes. The uncertainty bands widen as the counterfactual diverges further from observed data.
6. (2:08) Pause. Let the three overlapping trajectories (natural history, earlier treatment, resistance) sit on screen for 3 seconds. The Plotly legend labels each scenario clearly.

**Voiceover:**

> "Now -- the digital twin. We fit an ensemble of growth models -- exponential, logistic, and Gompertz -- and select the best fit using Akaike weights. Then we ask: what if there had been no treatment? The orange line shows unchecked growth -- this is the natural history counterfactual. What if treatment started four weeks earlier? The response deepens. What if resistance develops at week eight? Initial response, then regrowth. These are not predictions -- they are hypothesis projections, clearly labeled, with uncertainty bands that widen as we extrapolate further from observed data. The sliders let a clinician explore the parameter space interactively: therapy timing, sensitivity, resistance onset. This is the 'what if' engine that makes a digital twin more than a dashboard."

**Key message:** Counterfactual simulation with interactive sliders and honest uncertainty bands. This is the feature that no other hackathon entry will have. The combination of mathematical rigor (model ensembles, Akaike weights) with intuitive UX (sliders) is the differentiator.

**Judging criteria addressed:**
- **Effective Use of HAI-DEF Models (20%):** The simulation engine generates the structured evidence that MedGemma will reason over in the next section. This is the pipeline, not just a standalone model call.
- **Impact Potential (15%):** Counterfactual exploration could fundamentally change how tumor boards discuss treatment options. Instead of "did it work?" the question becomes "what else could work?"
- **Product Feasibility (20%):** The growth models are mathematically grounded (Gompertz, exponential, logistic -- standard oncology models from the literature). The UI sliders make complex math accessible.
- **Execution & Communication (30%):** The slider interaction is the "holy sh*t" moment. Judges will remember it.

**Timing notes:** This section must be rehearsed extensively. The slider drag at 2:03-2:08 is the single most important visual in the entire video. Practice it until it feels natural. If real-time re-simulation is too slow, pre-record each scenario and cut between them, or click "Run Simulation" after each slider change. The three overlapping trajectories at 2:08 should be visually clear -- ensure the colors (cyan observed, orange natural history, red resistance) are distinct.

---

## [2:15-2:40] MedGemma Narrative Generation (25 seconds)

**Screen actions (in order):**
1. (2:15) Click the "Narrate" tab. The yellow disclaimer banner appears again at the top: "This tool is for research and educational purposes only." Below it: "AI Narrative -- Not for clinical use."
2. (2:17) Click the "Generate" button (cyan gradient).
3. (2:18-2:28) The narrative renders in real time as Markdown. Show the text streaming in (or appearing in sections if using the template fallback):

   ```
   ## Digital Twin Tumor Assessment Report

   > DISCLAIMER: AI-generated for research purposes only.
   > NOT a substitute for clinical judgement.

   ### Patient Profile
   - Scenario: Classic Responder
   - Cancer Type: Brain Metastases (NSCLC)
   - Stage: IV
   - Age/Sex: 58/M

   ### Treatment History
   - Immunotherapy: Pembrolizumab (Week 0 to ongoing)

   ### Longitudinal Assessment
   | Week | RECIST | Sum (mm) | Change (%) | Status |
   | ---  | ---    | ---      | ---        | ---    |
   | 0    | BL     | 45.2     | +0.0%      | Pre-treatment |
   | 6    | PR     | 35.1     | -22.3%     | On treatment  |
   | 12   | PR     | 28.4     | -37.2%     | On treatment  |
   ...

   ### Growth Model Ensemble
   - Gompertz: AIC=12.3, weight=0.621
   - Logistic: AIC=14.1, weight=0.258
   - Exponential: AIC=16.8, weight=0.121

   RECIST Trajectory: BL -> PR -> PR -> PR -> SD
   ```

4. (2:28) Scroll down through the narrative to show the evidence grounding: every number in the text traces back to a measurement. The "Evidence" textbox at the bottom shows "MedGemma: Active | Grounding: OK".
5. (2:32) Briefly highlight the growth model ensemble section -- Gompertz model dominates with 62% weight, consistent with tumor growth literature.
6. (2:35) Click "Export" to show the downloadable report file.

**Voiceover:**

> "MedGemma is the reasoning layer. It takes every measurement, every RECIST classification, every growth model fit, and every counterfactual simulation -- and synthesizes a tumor board-ready narrative. Every claim is grounded in measured evidence. Every number is traceable. The disclaimer is not an afterthought -- it is structurally embedded. The model does not hallucinate treatment recommendations because the prompt architecture constrains it to summarize evidence with explicit uncertainty language. This is what 'effective use' of a medical AI model looks like: not replacing the clinician, but giving them a structured, auditable summary they can trust and verify."

**Key message:** MedGemma is not a gimmick. It is the reasoning layer that converts structured measurements into human-readable, evidence-grounded narrative. Safety constraints (disclaimer, grounding check, prohibited-phrase filtering) are architectural, not cosmetic.

**Judging criteria addressed:**
- **Effective Use of HAI-DEF Models (20%):** This is the payoff. MedGemma is used to its fullest: multimodal medical text comprehension, structured prompt engineering, safety-constrained generation. The grounding check (every claim cites a measurement) is a differentiator.
- **Product Feasibility (20%):** The narrative is exportable as Markdown. The safety pipeline (sanitize_narrative with grounding_check and safety_check) is auditable. This is how real medical AI systems must work.
- **Execution & Communication (30%):** The streaming narrative is visually satisfying. The evidence trail is compelling.

**Timing notes:** The narrative rendering at 2:18-2:28 should feel like watching intelligence at work. If using the template fallback (no GPU), the Markdown still appears in structured sections. Either way, scroll slowly so judges can read key lines. Pause on "RECIST Trajectory: BL -> PR -> PR -> PR -> SD" -- it is the punchline of the patient's clinical story.

---

## [2:40-2:55] Agentic Workflow & Audit Trail (15 seconds)

**Screen actions (in order):**
1. (2:40) Briefly show the system architecture as a diagram overlay (pre-made graphic) or scroll back to the Dashboard to show all the interconnected components:
   - Ingestion -> Preprocessing -> Measurement -> Tracking -> Twin Engine -> Simulation -> Narrative
2. (2:43) Show the audit trail concept. If an Agentic Workflow tab or audit log viewer is available, display it. Otherwise, show the JSONL audit log file briefly in a terminal or code editor:

   ```json
   {"event_id": "a1b2c3", "timestamp": "2026-02-12T10:15:32",
    "action_type": "measurement_recorded", "lesion_id": "L001",
    "after_state": {"diameter_mm": 18.5, "method": "semi-auto"},
    "metadata": {"method": "semi-auto"}}
   {"event_id": "d4e5f6", "timestamp": "2026-02-12T10:15:45",
    "action_type": "recist_confirmation", "after_state": {"category": "PR",
    "confirmed": true}, "metadata": {"confirmed": true}}
   {"event_id": "g7h8i9", "timestamp": "2026-02-12T10:16:01",
    "action_type": "human_override", "before_state": {"diameter_mm": 19.2},
    "after_state": {"diameter_mm": 18.5},
    "metadata": {"reason": "Corrected for irregular margin"}}
   ```

3. (2:48) Highlight one audit entry showing a "human_override" event: before_state and after_state clearly recorded, with a clinical reason. This is the auditability moment.
4. (2:50) Flash a quick summary overlay (pre-designed graphic):

   ```
   Pipeline: 7 modules | 3 growth models | 7 simulation scenarios
   Safety: Grounding check + prohibited phrase filter + disclaimer injection
   Audit: Every action logged with UUID, timestamp, before/after state
   ```

**Voiceover:**

> "The system is built as an agentic pipeline: seven modules, each responsible for one stage, from ingestion through narrative generation. Every action is audit-logged -- measurements, overrides, confirmations -- with before-and-after state and clinical rationale. When a clinician corrects an AI-generated measurement, that override is permanently recorded. This is not just a demo. This is the architecture that regulatory review requires: traceable, reproducible, and honest about what was automated and what was human."

**Key message:** Agentic architecture with full audit trail. The system is designed for regulatory scrutiny, not just hackathon demos. Every human correction is a data point for future improvement.

**Judging criteria addressed:**
- **Product Feasibility (20%):** Audit logging with UUID-stamped events, before/after state, and clinical rationale is a regulatory requirement for medical AI. Showing this signals product maturity.
- **Impact Potential (15%):** The audit trail enables post-hoc analysis of AI vs. human agreement, which is the path to clinical validation.
- **Effective Use of HAI-DEF Models (20%):** The agentic pipeline (eligible for Agentic Workflow Prize) shows MedGemma as one component in a larger orchestrated system, not a standalone API call.

**Timing notes:** This section is fast -- 15 seconds. The audit log JSON should flash on screen for just long enough to read "human_override" and "before_state / after_state." Judges do not need to read every field; they need to see that provenance tracking exists. The summary overlay at 2:50 is the capstone.

---

## [2:55-3:00] Closing (5 seconds)

**Screen:** Fade to a closing card:

```
DIGITAL TWIN: TUMOR RESPONSE ASSESSMENT

Longitudinal Tracking | Counterfactual Simulation | MedGemma Reasoning
Uncertainty-Aware | Auditable | Human-in-the-Loop

github.com/farchide/Hackaton-MedGemma

Built with MedGemma + Gradio + Plotly
For research use only. Not a medical device.
```

**Voiceover:**

> "Digital Twin Tumor: because every patient deserves a second opinion from their own data."

**Key message:** Memorable closing line that encapsulates the product vision. The "second opinion from their own data" framing reframes the digital twin concept for a non-technical audience.

**Judging criteria addressed:**
- **Execution & Communication (30%):** A strong closing line lands the narrative. The repo link and tech stack confirm reproducibility.

**Timing notes:** The closing line should be delivered with a brief pause before "their own data" for emphasis. Total runtime should land at 2:58-3:00. Do not exceed 3:00 under any circumstances.

---

## Production Notes

### Recording

- **Resolution:** 1920x1080 minimum; 2560x1440 preferred for crisp text
- **Frame rate:** 30fps
- **Format:** MP4 (H.264) for maximum compatibility
- **Browser zoom:** Set to 100% (or 110% if text is too small at 1080p)
- **Mouse cursor:** Use a visible but not oversized cursor; consider a cursor highlighter tool that shows click ripples
- **Screen transitions:** Use crossfades (0.3s) between sections, not hard cuts

### Audio

- **Voiceover:** Professional, measured tone. Not rushed, not monotone. The speaker should sound like a senior engineer presenting to a board, not a YouTuber. Target pace: 150-160 words per minute.
- **Total word count:** The script above contains approximately 850 words of voiceover. At 155 WPM, this fits within 5.5 minutes of speaking time -- but the video is 3 minutes with overlapping screen actions. Rehearse to confirm timing.
- **Background music:** Optional. If used, choose a subtle, low-volume ambient track. Medical/clinical contexts benefit from understated audio. No upbeat tech-demo music.
- **Audio levels:** Voiceover at -14 LUFS; music (if any) at -28 LUFS minimum

### Visual Polish

- **Pre-warm the app:** Load a patient once before recording. This ensures all cached assets are ready and there are no first-load delays.
- **Dark theme advantage:** The dark glassmorphism theme photographs well in screen recordings. The cyan accents pop against the dark background. Avoid showing any browser chrome that breaks the dark aesthetic.
- **Plot hover states:** When hovering over Plotly charts, the tooltips add visual richness. Practice hover paths that reveal useful information without looking random.
- **Disclaimer visibility:** The yellow disclaimer banner should appear in at least three shots (Dashboard load, Narrate tab, closing card). Judges will notice if safety messaging is prominent vs. hidden.

### Key Visual Moments (in priority order)

1. **[2:03-2:08] Counterfactual slider drag** -- The single most memorable visual. Three overlapping trajectories with uncertainty bands diverging. This is the image judges will remember.
2. **[0:25-0:28] One-click dashboard population** -- Three plots appearing simultaneously signals engineering depth.
3. **[2:18-2:28] MedGemma narrative rendering** -- Structured medical text appearing in real time signals AI integration quality.
4. **[0:52] Segmentation overlay** -- Green mask on lesion signals medical imaging competence.
5. **[1:28-1:35] Heterogeneous response view** -- Multiple diverging lesion curves signal clinical awareness.

### Judging Criteria Scoreboard

| Section | Duration | HAI-DEF (20%) | Problem (15%) | Impact (15%) | Feasibility (20%) | Execution (30%) |
| --- | --- | --- | --- | --- | --- | --- |
| Opening | 20s | - | HIGH | HIGH | - | HIGH |
| Dashboard | 25s | LOW | MEDIUM | - | HIGH | HIGH |
| Measurements | 30s | MEDIUM | - | MEDIUM | HIGH | HIGH |
| Tracking | 30s | LOW | HIGH | MEDIUM | HIGH | HIGH |
| Simulation | 30s | MEDIUM | - | HIGH | HIGH | HIGH |
| MedGemma | 25s | HIGH | - | MEDIUM | HIGH | HIGH |
| Audit | 15s | MEDIUM | - | MEDIUM | HIGH | MEDIUM |
| Closing | 5s | - | - | MEDIUM | - | HIGH |

Every section scores on at least three criteria. No dead time.

### Contingency Plans

- **If MedGemma GPU is unavailable:** The FallbackClient generates a structured template narrative with the same sections. The script still works -- adjust voiceover from "MedGemma generates" to "the reasoning layer synthesizes."
- **If segmentation is slow:** Pre-compute the segmentation mask and load it from cache. The demo still shows the workflow.
- **If the video runs long:** Cut the audit trail section (2:40-2:55) to 8 seconds. The closing line can absorb the remaining time.
- **If the video runs short:** Add a 3-second pause at 2:08 to let the three counterfactual trajectories breathe on screen. Judges need processing time for complex visuals.

### Rehearsal Checklist

- [ ] Full run-through with screen recording and voiceover: confirm total time is 2:55-3:00
- [ ] Verify all button clicks produce expected results on the demo patient
- [ ] Verify the counterfactual slider interaction is smooth (no jank, no error toasts)
- [ ] Verify the narrative generation completes within the allotted 10-second window
- [ ] Verify the audit log JSON is readable at recording resolution
- [ ] Watch the recording on a laptop screen (not just a monitor) to confirm text legibility
- [ ] Have a non-technical person watch and confirm they understand the problem and the "wow" moment
