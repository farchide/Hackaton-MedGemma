# ADR-002: Python Framework and Web UI

| Field       | Value                                              |
|-------------|----------------------------------------------------|
| **Status**  | Accepted                                           |
| **Date**    | 2026-02-11                                         |
| **Authors** | Digital Twin Tumor Team                            |
| **Scope**   | Runtime, dependency management, UI framework, and  |
|             | demo application structure                         |

---

## Context

The Digital Twin Tumor project requires a web-based demo application that can:

1. Accept medical imaging uploads (DICOM, NIfTI)
2. Display and annotate medical images interactively
3. Visualize longitudinal tumor trajectories with uncertainty bands
4. Provide interactive counterfactual simulation sliders
5. Present AI-generated tumor board narratives
6. Run reproducibly on Kaggle or a reviewer's machine

The competition awards bonus points for a "public interactive live demo app"
(Kaggle submission instructions). The demo storyboard from our spec defines a
five-step flow: Upload --> Annotate --> Track --> Simulate --> Narrate.

We need a framework that supports medical imaging display natively, integrates
well with the HuggingFace/PyTorch ecosystem (since MedGemma is distributed via
HuggingFace), and can be deployed with minimal infrastructure. The UI must be
compelling enough for a 3-minute video demo that scores well on the "Execution
and communication" criterion (30% of total score).

We also need to decide on Python version, dependency management strategy, and
the core library set that all six architecture layers depend on.

---

## Decision

### UI Framework: Gradio

We adopt **Gradio** (version 5.x) as the demo application framework. The
application entry point is `src/app.py`, which composes Gradio Blocks into a
tabbed interface mapping to the demo storyboard.

### Python Runtime: 3.11+

We target **Python 3.11 or later** for:
- Native `tomllib` support (read pyproject.toml without extra deps)
- Improved error messages with precise tracebacks
- Performance improvements in the interpreter (10-60% faster than 3.10)
- `typing.Self`, `StrEnum`, and `ExceptionGroup` for cleaner code

### Dependency Management: pyproject.toml + uv

We use **`pyproject.toml`** as the single source of truth for project metadata
and dependencies, with **`uv`** as the package installer and virtual environment
manager.

```toml
[project]
name = "digital-twin-tumor"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # UI
    "gradio>=5.0",

    # ML / MedGemma
    "transformers>=4.50",
    "torch>=2.1",
    "accelerate>=0.25",

    # Medical imaging
    "pydicom>=2.4",
    "nibabel>=5.0",
    "SimpleITK>=2.3",
    "opencv-python>=4.8",

    # Scientific computing
    "numpy>=1.26",
    "scipy>=1.12",
    "pandas>=2.1",
    "networkx>=3.2",

    # Visualization
    "matplotlib>=3.8",
    "plotly>=5.18",

    # Utilities
    "pillow>=10.0",
    "pydantic>=2.5",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.5",
    "mypy>=1.8",
]

segmentation = [
    "segment-anything>=1.0",
]
```

### Core Library Selection

| Category            | Library           | Version  | Purpose in Our System                          |
|---------------------|-------------------|----------|------------------------------------------------|
| **UI**              | `gradio`          | >=5.0    | Demo application, image annotation, sliders    |
| **ML Runtime**      | `torch`           | >=2.1    | MedGemma inference, MedSAM inference           |
| **Model Loading**   | `transformers`    | >=4.50   | Load MedGemma 1.5 4B; tokenizer + generation   |
| **Acceleration**    | `accelerate`      | >=0.25   | Device mapping, mixed precision for MedGemma   |
| **DICOM**           | `pydicom`         | >=2.4    | Read DICOM files, extract headers, PHI scan    |
| **NIfTI**           | `nibabel`         | >=5.0    | Read NIfTI volumes, affine transforms          |
| **Image Processing**| `SimpleITK`       | >=2.3    | Resampling, registration, intensity windowing  |
| **CV Operations**   | `opencv-python`   | >=4.8    | Contour extraction, morphological ops          |
| **Arrays**          | `numpy`           | >=1.26   | Core numerical operations everywhere           |
| **Optimization**    | `scipy`           | >=1.12   | Gompertz fitting, ODE integration, spatial ops |
| **DataFrames**      | `pandas`          | >=2.1    | Therapy logs, measurement tables, CSV export   |
| **Graphs**          | `networkx`        | >=3.2    | Lesion identity graph across timepoints        |
| **Static Plots**    | `matplotlib`      | >=3.8    | Publication-quality trajectory plots           |
| **Interactive Plots**| `plotly`          | >=5.18   | Interactive tumor burden curves in Gradio      |
| **Validation**      | `pydantic`        | >=2.5    | Input validation at system boundaries          |
| **Images**          | `pillow`          | >=10.0   | Image format conversion, thumbnail generation  |

### Why Gradio Over Alternatives

**1. Native medical imaging support**

Gradio's `gr.Image` component supports multiple input types (numpy arrays, PIL
images, file paths) with built-in display. More importantly, `gr.ImageEditor`
provides annotation capabilities (drawing, erasing, brush tools) that map
directly to our lesion measurement workflow. Streamlit's `st.image` is
display-only; annotation requires third-party components that are less mature.

**2. HuggingFace ecosystem integration**

MedGemma is distributed via HuggingFace. Gradio is built by HuggingFace and
has first-class integration:
- `gr.load()` can wrap HuggingFace models directly
- HuggingFace Spaces provides free hosting for Gradio apps (bonus demo points)
- The `transformers` pipeline API connects seamlessly to Gradio interfaces
- Sharing a Gradio app generates a public URL with `share=True`

**3. Tabbed multi-step workflows**

Gradio Blocks with `gr.Tab` components naturally express our five-step demo
storyboard. Each tab corresponds to a pipeline stage, and state flows between
tabs via `gr.State`. Streamlit's page model is less natural for guided workflows
with shared state.

**4. Interactive controls for counterfactual simulation**

Gradio provides `gr.Slider`, `gr.Dropdown`, and `gr.Number` components that
directly map to our counterfactual controls (therapy start shift, sensitivity
multiplier, regimen selection). These trigger Python callbacks that re-run the
twin engine and update Plotly charts in real time.

**5. Deployment simplicity**

A Gradio app is a single Python file. Deployment options:
- Local: `python src/app.py`
- HuggingFace Spaces: Push repo, auto-deploy (free GPU available)
- Docker: Gradio apps work in containers with no configuration
- Kaggle: Gradio can run inside Kaggle notebooks

### UI Structure Mapped to Demo Storyboard

```
+------------------------------------------------------------------+
|  Digital Twin Tumor                                [Non-Clinical] |
+------------------------------------------------------------------+
| [Upload] | [Annotate] | [Track] | [Simulate] | [Narrate]        |
+------------------------------------------------------------------+

Tab 1: UPLOAD (Ingestion Layer)
  +---------------------------+  +-----------------------------+
  | gr.File                   |  | Case Summary                |
  | "Upload DICOM/NIfTI"      |  | - Patient ID (deidentified) |
  |                           |  | - Timepoints found: 5       |
  | gr.File                   |  | - Modality: MRI T1+C        |
  | "Upload Therapy Log"      |  | - Therapy: SRS (3 sessions) |
  +---------------------------+  +-----------------------------+

Tab 2: ANNOTATE (Measurement Layer)
  +---------------------------+  +-----------------------------+
  | gr.ImageEditor            |  | Lesion Table                |
  | [Axial slice with overlay]|  | ID | Diam | Vol  | Source  |
  |                           |  | L1 | 12mm | 904  | Manual  |
  | Slice slider (gr.Slider)  |  | L2 | 8mm  | 268  | MedSAM  |
  | Timepoint selector        |  | L3 | 15mm | 1767 | Edited  |
  +---------------------------+  +-----------------------------+

Tab 3: TRACK (Tracking Layer)
  +---------------------------+  +-----------------------------+
  | gr.Plot (Plotly)          |  | Identity Graph              |
  | [Lesion match timeline]   |  | gr.Plot (NetworkX viz)      |
  |  L1 ---o---o---o---o     |  |                             |
  |  L2 ---o---o---x (lost)  |  | New lesions detected: 1     |
  |  L3 -------o---o---o     |  | Unmatched: 0                |
  +---------------------------+  +-----------------------------+

Tab 4: SIMULATE (Twin Engine Layer)
  +---------------------------+  +-----------------------------+
  | gr.Plot (Plotly)          |  | Counterfactual Controls     |
  | [Trajectories + bands]    |  | Therapy start: gr.Slider    |
  |  -- Observed              |  |   [-8w ... +8w]             |
  |  -- Predicted             |  | Sensitivity: gr.Slider      |
  |  -- Counterfactual        |  |   [0.0 ... 2.0]             |
  |  == Uncertainty band      |  | Model: gr.Dropdown          |
  |                           |  |   [Gompertz|Logistic|Exp]   |
  | Therapy timeline overlay  |  | gr.Button "Run Simulation"  |
  +---------------------------+  +-----------------------------+

Tab 5: NARRATE (MedGemma Reasoning Layer)
  +---------------------------+  +-----------------------------+
  | gr.Textbox (output)       |  | Evidence Panel              |
  | "Tumor Board Summary"     |  | gr.Gallery                  |
  |                           |  | [Selected evidence slices]  |
  | - Response assessment     |  |                             |
  | - Key measurements        |  | Confidence: 0.82           |
  | - Uncertainty caveats     |  | Model: MedGemma 1.5 4B     |
  | - Safety disclaimers      |  |                             |
  |                           |  | gr.Button "Regenerate"      |
  | [NON-CLINICAL USE ONLY]   |  | gr.Button "Export PDF"      |
  +---------------------------+  +-----------------------------+
```

### Application Entry Point

```python
# src/app.py (simplified structure)
import gradio as gr
from src.ingestion import ingest_study
from src.preprocessing import preprocess_volume
from src.measurement import MeasurementUI
from src.tracking import track_lesions
from src.twin_engine import run_simulation
from src.reasoning import generate_narrative

DISCLAIMER = (
    "FOR RESEARCH AND DEVELOPMENT ONLY. NOT FOR CLINICAL USE. "
    "This tool is not intended to inform diagnosis or patient management. "
    "All outputs require independent clinical verification."
)

with gr.Blocks(title="Digital Twin Tumor", theme=gr.themes.Soft()) as app:
    gr.Markdown(f"# Digital Twin Tumor\n> {DISCLAIMER}")
    state = gr.State({})

    with gr.Tab("Upload"):
        # ... ingestion UI components
        pass

    with gr.Tab("Annotate"):
        # ... measurement UI components
        pass

    with gr.Tab("Track"):
        # ... tracking visualization
        pass

    with gr.Tab("Simulate"):
        # ... twin engine controls + plots
        pass

    with gr.Tab("Narrate"):
        # ... MedGemma narrative output
        pass

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
```

### Development Workflow

```bash
# Environment setup
uv venv --python 3.11
uv pip install -e ".[dev]"

# Run the application
uv run python src/app.py

# Run tests
uv run pytest tests/ -v --cov=src

# Lint and type check
uv run ruff check src/
uv run mypy src/
```

---

## Consequences

### Positive

- **Fast prototyping**: Gradio Blocks allow building interactive UIs in hours,
  not days. The declarative component model means UI changes are quick.
- **Ecosystem alignment**: Using HuggingFace's own UI framework alongside their
  model hub creates a seamless integration path for MedGemma.
- **Free deployment**: HuggingFace Spaces provides free GPU-enabled hosting for
  Gradio apps, giving us a live demo URL for the submission (bonus points).
- **Reproducibility**: `pyproject.toml` + `uv` ensures deterministic installs.
  Reviewers can clone the repo and run with two commands.
- **Medical imaging UX**: `gr.ImageEditor` with drawing tools provides a
  credible lesion annotation experience without building custom canvas code.
- **Modern Python**: Python 3.11+ gives us performance, better typing, and
  cleaner error messages during development.
- **Dependency management**: `uv` is 10-100x faster than pip for resolution
  and installation, reducing CI time and developer friction.

### Negative

- **Gradio version churn**: Gradio's API changes between major versions. We
  pin to >=5.0 and test against a specific version in CI. If a breaking change
  lands during development, we pin to the exact known-good version.
- **Limited custom visualization**: Complex medical imaging viewers (MPR,
  windowing sliders, DICOM-native display) are harder in Gradio than in
  dedicated frameworks like OHIF. Mitigated by using matplotlib/plotly for
  custom renders passed as images to Gradio components.
- **Single-user model**: Gradio's default mode runs one inference at a time.
  Acceptable for a demo; we document concurrent-user limitations.
- **Large dependency tree**: PyTorch + transformers + SimpleITK + Gradio is a
  multi-GB install. Mitigated by Docker image caching and `uv`'s speed.

### Trade-offs Accepted

| Trade-off                        | Accepted Because                                   |
|----------------------------------|----------------------------------------------------|
| Gradio over custom React UI     | 10x faster to build; good enough for demo          |
| uv over pip/poetry              | Faster installs; growing ecosystem support          |
| Plotly over D3.js               | Python-native; interactive in Gradio without JS     |
| SimpleITK over ITK Python       | Simpler API for our preprocessing needs             |
| pydantic over manual validation | Consistent validation; auto-generates schemas       |

---

## Alternatives Considered

### 1. Streamlit

Streamlit is the most common choice for ML demos.

- **Rejected because**:
  - No native image annotation component. `streamlit-drawable-canvas` is a
    third-party component with limited medical imaging features.
  - Top-to-bottom rerun model is awkward for stateful multi-step workflows.
    Our lesion tracking requires persistent state across interactions.
  - No built-in HuggingFace Spaces integration (Streamlit Community Cloud is
    separate and does not provide GPU).
  - Streamlit's session state management is more complex for our five-tab
    workflow where each tab builds on the previous tab's output.

### 2. Panel (HoloViz)

Panel provides powerful dashboarding with Bokeh.

- **Rejected because**:
  - Smaller community; fewer examples for medical imaging use cases
  - Heavier learning curve for the team compared to Gradio
  - No HuggingFace Spaces integration for free GPU hosting
  - Better suited for data dashboards than interactive annotation workflows

### 3. FastAPI + React Frontend

A proper API backend with a custom frontend.

- **Rejected because**:
  - Requires JavaScript/TypeScript development alongside Python
  - At least 3-5 extra days of development for a custom frontend
  - Two codebases to maintain during a 13-day hackathon
  - Overkill for a demo; appropriate for production but not for competition

### 4. Jupyter Widgets (ipywidgets + Voila)

Interactive widgets inside notebooks, served via Voila.

- **Rejected because**:
  - Limited interactivity compared to Gradio Blocks
  - Voila deployment is less polished than Gradio/HF Spaces
  - Notebook-based development makes version control harder
  - Judges expect a "real application" for the Product Feasibility criterion

### 5. pip + requirements.txt (Dependency Management)

Traditional dependency management.

- **Rejected because**:
  - No lock file by default (non-deterministic installs)
  - Slow resolution for our complex dependency tree (torch + transformers)
  - `pyproject.toml` is the modern Python standard (PEP 621)
  - `uv` provides 10-100x faster installs and built-in venv management

---

## References

- Gradio Documentation: https://www.gradio.app/docs
- HuggingFace Spaces GPU: Free T4 GPU tier for Gradio apps
- PEP 621: Storing project metadata in pyproject.toml
- uv: https://github.com/astral-sh/uv
- MedGemma on HuggingFace: google/medgemma-4b-it (requires Transformers >=4.50)
- Kaggle Submission Instructions: Bonus for "public interactive live demo app"
