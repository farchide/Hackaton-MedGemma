# ADR-014: Python Project Structure & Dependencies

## Status

Accepted

## Date

2026-02-11

## Context

The Digital Twin Tumor project is a Python application that spans medical image
processing, statistical modeling, LLM integration, and interactive UI. It must run
in two primary environments:

1. **Kaggle notebooks** -- limited to pre-installed packages plus `pip install` at
   runtime, with T4 GPU and 13 GB RAM.
2. **Local development** -- where developers need linting, type checking, testing, and
   fast iteration cycles.

The codebase must be navigable by hackathon teammates who may join mid-sprint, so
clear module boundaries and typed interfaces are essential. We also need to support
both GPU and CPU-only execution paths (for CI and for developers without GPUs).

Key tensions:

- **Flat scripts vs structured package.** Kaggle culture favors monolithic notebooks.
  But our system has six distinct subsystems that will be developed in parallel by
  multiple contributors, making a structured package necessary.
- **Minimal dependencies vs full tooling.** Medical imaging libraries are large. We
  must be deliberate about what we include to stay within Kaggle's install-time budget
  (roughly 5 minutes for `pip install`).
- **Strict typing vs rapid prototyping.** Full typing slows initial development but
  prevents integration bugs when multiple people work on different modules. Given the
  complexity of our data flow, typing wins.

## Decision

We adopt a **clean Python package structure** under `src/` using modern `pyproject.toml`
for dependency management, with Protocol-based typed interfaces for cross-module
communication and dataclasses for all domain objects.

### Directory Structure

```
Hackaton-MedGemma/
|-- pyproject.toml              # Build system, dependencies, tool config
|-- README.md                   # Project overview (already exists)
|-- src/
|   |-- digital_twin_tumor/
|       |-- __init__.py         # Package version, top-level exports
|       |-- __main__.py         # Entry point: python -m digital_twin_tumor
|       |
|       |-- ingestion/
|       |   |-- __init__.py
|       |   |-- dicom_loader.py     # DICOM file reading, metadata extraction
|       |   |-- nifti_loader.py     # NIfTI file reading for pre-converted data
|       |   |-- phi_gate.py         # PHI detection and stripping before processing
|       |   |-- protocols.py        # ImageLoader protocol, ScanMetadata type
|       |
|       |-- preprocessing/
|       |   |-- __init__.py
|       |   |-- normalize.py        # Window/level, intensity normalization
|       |   |-- resample.py         # Isotropic resampling, spacing standardization
|       |   |-- slice_selector.py   # Axial slice selection for 2D display
|       |   |-- protocols.py        # Preprocessor protocol
|       |
|       |-- measurement/
|       |   |-- __init__.py
|       |   |-- diameter.py         # Longest-axis and short-axis diameter computation
|       |   |-- segmentation.py     # MedSAM-based segmentation with prompt interface
|       |   |-- volume.py           # Volumetric computation from contiguous masks
|       |   |-- recist.py           # RECIST 1.1 classification logic
|       |   |-- protocols.py        # Measurer protocol, MeasurementResult type
|       |
|       |-- tracking/
|       |   |-- __init__.py
|       |   |-- identity_graph.py   # Lesion identity graph across timepoints
|       |   |-- registration.py     # Lightweight rigid/affine registration
|       |   |-- matcher.py          # Lesion matching (centroid + appearance)
|       |   |-- protocols.py        # Tracker protocol, LesionMatch type
|       |
|       |-- twin_engine/
|       |   |-- __init__.py
|       |   |-- growth_models.py    # Exponential, logistic, Gompertz, linear models
|       |   |-- model_selection.py  # AIC/BIC model comparison, ensemble weighting
|       |   |-- simulation.py       # Forward simulation with counterfactuals
|       |   |-- uncertainty.py      # Prediction intervals, bootstrap UQ
|       |   |-- protocols.py        # GrowthModel protocol, SimulationResult type
|       |
|       |-- reasoning/
|       |   |-- __init__.py
|       |   |-- medgemma_client.py  # MedGemma API/model wrapper with retry logic
|       |   |-- prompt_templates.py # Structured prompts for RECIST reasoning
|       |   |-- narrative.py        # Natural-language report generation
|       |   |-- protocols.py        # Reasoner protocol, NarrativeResult type
|       |
|       |-- ui/
|       |   |-- __init__.py
|       |   |-- app.py              # Gradio Blocks application (entry point)
|       |   |-- components.py       # Reusable Gradio component factories
|       |   |-- measurement_overlay.py  # Canvas overlay for diameter drawing
|       |   |-- timeline.py         # Longitudinal timeline navigator widget
|       |   |-- state.py            # Session state management
|       |
|       |-- common/
|           |-- __init__.py
|           |-- types.py            # Core dataclasses: Patient, Lesion, TimePoint, etc.
|           |-- protocols.py        # Shared Protocol definitions
|           |-- constants.py        # Magic numbers, RECIST thresholds, config defaults
|           |-- logging.py          # Structured logging setup
|           |-- exceptions.py       # Domain-specific exception hierarchy
|
|-- tests/
|   |-- conftest.py                 # Shared fixtures: sample DICOM, NIfTI, patients
|   |-- fixtures/                   # Small binary test fixtures (synthetic DICOM, etc.)
|   |-- test_ingestion/
|   |-- test_preprocessing/
|   |-- test_measurement/
|   |-- test_tracking/
|   |-- test_twin_engine/
|   |-- test_reasoning/
|   |-- test_ui/
|   |-- test_common/
|
|-- notebooks/
|   |-- evaluation.ipynb            # Reproducible metrics (see ADR-013)
|   |-- demo.ipynb                  # Interactive demo notebook
|   |-- exploration/                # Scratch notebooks (not tracked in CI)
|
|-- data/
|   |-- raw/                        # Original DICOM/NIfTI (gitignored)
|   |-- processed/                  # Preprocessed arrays (gitignored)
|   |-- synthetic/                  # Generated test data (committed if small)
|
|-- scripts/
|   |-- download_proteas.py         # Script to fetch PROTEAS dataset
|   |-- generate_synthetic.py       # Script to create synthetic test data
|   |-- run_evaluation.py           # CLI wrapper to execute evaluation notebook
|
|-- config/
|   |-- default.yaml                # Default configuration values
|   |-- kaggle.yaml                 # Kaggle-specific overrides
|
|-- docs/
|   |-- adr/                        # Architecture Decision Records
```

### Domain Dataclasses

All domain objects are defined as frozen dataclasses in `common/types.py` with full
type annotations. Key types:

```python
@dataclass(frozen=True)
class Patient:
    patient_id: str
    timepoints: tuple[TimePoint, ...]
    metadata: dict[str, Any]

@dataclass(frozen=True)
class TimePoint:
    timepoint_id: str
    scan_date: date
    scan_path: Path
    treatment_context: str | None
    lesions: tuple[Lesion, ...]

@dataclass(frozen=True)
class Lesion:
    lesion_id: str
    organ: str
    is_target: bool
    measurements: tuple[Measurement, ...]
    mask: np.ndarray | None       # None if not yet segmented

@dataclass(frozen=True)
class Measurement:
    longest_diameter_mm: float
    short_axis_mm: float | None   # Only for lymph nodes
    volume_mm3: float | None
    method: Literal["manual", "semi_auto", "auto"]
    confidence: float             # 0.0 - 1.0
    timestamp: datetime

@dataclass(frozen=True)
class GrowthModel:
    model_type: Literal["exponential", "logistic", "gompertz", "linear"]
    parameters: dict[str, float]
    aic: float
    bic: float

@dataclass(frozen=True)
class Simulation:
    lesion_id: str
    predicted_diameters: list[float]
    prediction_intervals: list[tuple[float, float]]
    time_horizon_days: list[int]
    model_used: GrowthModel
```

### Protocol-Based Interfaces

Each module exposes a Protocol class that defines its public interface. Downstream
modules depend on the Protocol, not the concrete implementation, enabling testing
with lightweight fakes.

```python
# Example: measurement/protocols.py
class Measurer(Protocol):
    def measure_diameter(
        self, image: np.ndarray, points: tuple[tuple[int, int], tuple[int, int]]
    ) -> Measurement: ...

    def segment_lesion(
        self, image: np.ndarray, prompt: SegmentationPrompt
    ) -> np.ndarray: ...
```

### Dependency Management

The `pyproject.toml` defines four dependency groups:

```toml
[project]
name = "digital-twin-tumor"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "pandas>=2.1",
    "scikit-learn>=1.3",
    "pillow>=10.0",
    "pydicom>=2.4",
    "nibabel>=5.2",
    "SimpleITK>=2.3",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
gpu = [
    "torch>=2.1",
    "monai>=1.3",
    "transformers>=4.36",
    "segment-anything-2>=0.1",
    "gradio>=4.10",
]
cpu = [
    "torch>=2.1",
    "monai>=1.3",
    "transformers>=4.36",
    "gradio>=4.10",
]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.3",
    "mypy>=1.8",
    "black>=24.1",
    "ipykernel>=6.28",
    "pre-commit>=3.6",
]
```

Install commands:

```bash
# Full GPU development
pip install -e ".[gpu,dev]"

# CPU-only development
pip install -e ".[cpu,dev]"

# Kaggle (minimal, GPU assumed)
pip install -e ".[gpu]"
```

### Entry Points

| Method | Command | Use Case |
|--------|---------|----------|
| Module execution | `python -m digital_twin_tumor` | CLI pipeline run |
| Gradio app | `gradio src/digital_twin_tumor/ui/app.py` or `python src/digital_twin_tumor/ui/app.py` | Interactive UI |
| Notebook import | `from digital_twin_tumor.measurement import ...` | Kaggle notebook usage |

### Code Quality Tools

| Tool | Purpose | Configuration Location |
|------|---------|----------------------|
| ruff | Linting and import sorting | `[tool.ruff]` in pyproject.toml |
| black | Code formatting (line length 99) | `[tool.black]` in pyproject.toml |
| mypy | Static type checking (strict mode) | `[tool.mypy]` in pyproject.toml |
| pytest | Test execution with coverage | `[tool.pytest.ini_options]` in pyproject.toml |
| pre-commit | Git hooks for lint/format/type check | `.pre-commit-config.yaml` |

### Testing Strategy

- **Unit tests** for every module, using Protocol-based fakes for dependencies.
- **Fixtures** in `tests/conftest.py` provide small synthetic DICOM files (generated
  via `pydicom.Dataset`), NIfTI volumes (generated via `nibabel`), and sample
  `Patient`/`Lesion` dataclass instances.
- **No real patient data in tests.** All test data is synthetic or from PROTEAS with
  explicit license confirmation.
- **Coverage target:** 70% line coverage across `src/`, enforced in CI.
- Test directory structure mirrors `src/` exactly for discoverability.

### Configuration

Application configuration is loaded from YAML files via a simple loader in
`common/constants.py`. The resolution order is:

1. `config/default.yaml` (committed, contains sane defaults)
2. `config/kaggle.yaml` (committed, overrides for Kaggle environment)
3. Environment variables prefixed with `DTT_` (e.g., `DTT_MEDGEMMA_MODEL`)
4. CLI arguments (for `__main__.py` entry point)

Secrets (API keys) are loaded exclusively from environment variables and are never
written to configuration files.

## Consequences

### Positive

- **Parallel development.** Module boundaries with typed Protocols let team members
  work on different subsystems without merge conflicts or integration surprises.
- **Kaggle compatibility.** The `src/` layout with `pip install -e .` works in Kaggle
  notebooks after a cell with `!pip install -e .`.
- **Test isolation.** Protocol-based fakes mean unit tests run in milliseconds without
  GPU or real data.
- **Discoverability.** New team members can navigate the codebase by reading the
  `protocols.py` in each module to understand its contract.
- **Type safety.** mypy in strict mode catches None-safety issues and interface
  mismatches before runtime.

### Negative

- **Boilerplate overhead.** Protocol files, `__init__.py` files, and frozen dataclasses
  add ceremony. Accepted as a worthwhile trade for a multi-person project.
- **Learning curve.** Contributors unfamiliar with Protocol-based design may need a
  brief orientation. Mitigated by clear docstrings and a CONTRIBUTING section.
- **Frozen dataclass rigidity.** Frozen dataclasses require creating new instances for
  mutations. This is intentional (immutability prevents subtle state bugs) but can
  feel awkward for developers used to mutable objects.

### Risks

- If the Kaggle `pip install` step takes too long (> 5 min), we may need to reduce
  dependencies or pre-build a wheel. Mitigation: profile install time early and trim
  if needed.
- mypy strict mode may flag issues in third-party libraries with incomplete stubs.
  Mitigation: per-module `# type: ignore` with comments explaining why.

## Alternatives Considered

### 1. Monolithic Notebook

A single Kaggle notebook containing all code. Rejected because it makes parallel
development impossible, has no type checking, and produces unmaintainable code.
We use notebooks only for evaluation and demos, not for production logic.

### 2. Flat Script Directory

All Python files in a single directory without package structure. Rejected because
it leads to import path confusion, makes testing harder, and does not scale to
six subsystems with cross-cutting concerns.

### 3. Poetry for Dependency Management

Poetry provides lockfiles and virtual environment management. Rejected in favor of
`pyproject.toml` with pip because Kaggle environments do not support Poetry natively,
and pip is universally available. We lose deterministic lockfiles but gain
compatibility.

### 4. Pydantic Instead of Dataclasses

Pydantic provides runtime validation and serialization. Rejected because it adds a
significant dependency and runtime overhead for object construction. Our domain
objects are constructed in validated code paths (ingestion layer), so the extra
validation at every construction site is unnecessary. We use Pydantic only if we
later add a REST API layer.

### 5. Mutable Dataclasses

Using non-frozen dataclasses for easier mutation. Rejected because mutable state
shared across modules is a major source of bugs, especially when lesion measurements
are passed between tracking, simulation, and reasoning modules. Immutability makes
the data flow explicit.
