---
title: Digital Twin Tumor Response Assessment
emoji: "\U0001F52C"
colorFrom: cyan
colorTo: blue
sdk: gradio
sdk_version: "5.0"
python_version: "3.11"
app_file: app.py
pinned: true
license: apache-2.0
tags:
  - medgemma
  - medical-imaging
  - digital-twin
  - oncology
  - hai-def
---

# Digital Twin: Tumor Response Assessment

AI-powered longitudinal tumor monitoring with **MedGemma** integration for the
[Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge).

## Overview

A seven-tab Gradio application that builds a *digital twin* of a patient's
tumor over time, combining medical imaging, RECIST 1.1 classification, growth
modelling, counterfactual simulation, and MedGemma-powered clinical narratives.

### Tabs

| # | Tab | Description |
|---|-----|-------------|
| 0 | **Dashboard** | Load demo patients, view timelines, RECIST waterfall, growth plots |
| 1 | **Upload & Preprocess** | Ingest DICOM / NIfTI scans, normalize, view slices |
| 2 | **Annotate & Measure** | Semi-automatic segmentation, diameter/volume measurement |
| 3 | **Track Lesions** | Longitudinal lesion identity graph across timepoints |
| 4 | **Simulate** | Digital twin growth models with counterfactual scenarios |
| 5 | **Narrate** | MedGemma-generated clinical narratives with safety disclaimers |
| 6 | **Agentic Workflow** | Autonomous 6-stage oncology pipeline with audit trail |

## MedGemma Integration

- **MedGemma-4B-IT** (Gemma 3 + MedSigLIP 400M image encoder) for multimodal
  imaging analysis and clinical text generation
- Template-based fallback when GPU is unavailable
- ZeroGPU-compatible lazy model loading
- Safety disclaimers on all AI-generated content

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

The app generates synthetic demo data on first launch and starts on port 7860.

## Disclaimer

This tool is for **research and educational purposes only**. It is NOT a
medical device and must NOT be used for clinical decision-making.
