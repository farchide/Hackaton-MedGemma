# ADR-012: Compute & GPU Deployment Strategy

## Status

Accepted

## Date

2026-02-11

## Context

The Digital Twin Tumor system integrates multiple deep learning models that must run
within the constrained GPU environments typical of hackathon and research settings.
The primary deployment targets are:

- **Kaggle Notebooks**: NVIDIA T4 GPU (16GB VRAM), 13GB system RAM, session time limits
- **Google Colab**: T4 (free tier, 15GB VRAM) or A100 (Colab Pro, 40/80GB VRAM)
- **Local workstation**: Consumer GPUs (RTX 3080/3090/4090, 10-24GB VRAM)

The models in the pipeline and their approximate VRAM requirements:

| Model              | Parameters | FP16 VRAM | INT4 VRAM | Purpose                     |
|--------------------|------------|-----------|-----------|------------------------------|
| MedGemma 1.5 4B   | 4B         | ~8-10GB   | ~3-4GB    | Multimodal narrative gen     |
| MedGemma 27B       | 27B        | ~54GB     | ~16GB     | Higher quality narrative     |
| MedSAM             | ~93M       | ~1-2GB    | N/A       | Tumor segmentation           |
| nnU-Net (3D)       | ~30M       | ~2-4GB    | N/A       | Volumetric segmentation      |
| MedSigLIP (encoder)| ~400M     | ~1-2GB    | N/A       | Lesion appearance embeddings |

The combined VRAM requirement for running MedGemma 4B + MedSAM simultaneously in FP16
is approximately 10-12GB, which fits within a T4's 16GB with margin for CUDA overhead
and batch data. Running MedGemma 27B requires multi-GPU or A100 hardware.

The system must handle the full pipeline (preprocessing, segmentation, growth modeling,
narrative generation) within interactive latency targets to provide a responsive demo
experience.

## Decision

We adopt an **optimized single-GPU deployment strategy with graceful degradation**
across model precision, model selection, and compute fallback paths.

### 1. Primary Configuration: MedGemma 4B on Single GPU

The default and recommended configuration targets MedGemma 1.5 4B multimodal as the
primary language model, running on a single 16-24GB GPU.

**Precision Strategy**:

| VRAM Available | Precision       | Library          | Expected VRAM |
|----------------|-----------------|------------------|---------------|
| >= 16GB        | bfloat16        | transformers     | ~8-10GB       |
| 12-16GB        | float16         | transformers     | ~8-10GB       |
| < 12GB         | int4 (NF4)      | bitsandbytes     | ~3-4GB        |
| < 8GB          | CPU offload     | accelerate       | ~2GB GPU      |

Precision selection is automatic at startup:

```python
import torch

def select_precision():
    if not torch.cuda.is_available():
        return "cpu_offload"

    vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)

    if vram_gb >= 16:
        # Prefer bfloat16 on Ampere+; fall back to float16
        if torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"
    elif vram_gb >= 12:
        return "float16"
    elif vram_gb >= 8:
        return "int4_nf4"
    else:
        return "cpu_offload"
```

**BitsAndBytes 4-bit Quantization Config**:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,  # Nested quantization for memory savings
)
```

Double quantization reduces VRAM by an additional ~0.4GB with negligible quality loss.

### 2. Model Co-residency Strategy

On a single T4 (16GB), models cannot all reside in VRAM simultaneously. The strategy
is sequential loading with explicit memory management:

**Phase 1 -- Preprocessing & Segmentation** (GPU-resident: MedSAM or nnU-Net):
- Load MedSAM (~2GB VRAM) for interactive segmentation
- Process all lesions across all timepoints
- Cache segmentation masks to disk as compressed NumPy arrays
- Unload MedSAM, call `torch.cuda.empty_cache()`

**Phase 2 -- Embedding Extraction** (GPU-resident: MedSigLIP encoder):
- Load MedSigLIP vision encoder (~1.5GB VRAM)
- Extract embeddings for all lesion ROI patches
- Cache embeddings to disk as NumPy arrays
- Unload MedSigLIP, call `torch.cuda.empty_cache()`

**Phase 3 -- Narrative Generation** (GPU-resident: MedGemma 4B):
- Load MedGemma 4B (~8-10GB VRAM in float16, ~3-4GB in int4)
- Generate narratives for each timepoint comparison
- Keep model loaded for interactive counterfactual queries
- Unload only when switching back to segmentation

**Memory Management**:

```python
def unload_model(model):
    """Explicitly free GPU memory after model use."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import gc
    gc.collect()
```

### 2a. RuVector+PostgreSQL Deployment

The `ruvnet/ruvector-postgres` Docker image provides PostgreSQL and RuVector combined
in a single container, running alongside the application as a database tier. This
container handles all persistent storage (audit logs, measurement history, patient
metadata) and vector operations (lesion embeddings, similar case retrieval, GNN-enhanced
search).

**Docker Compose Configuration:**

```yaml
services:
  ruvector-postgres:
    image: ruvnet/ruvector-postgres:latest
    ports:
      - "5432:5432"
      - "8080:8080"  # RuVector API
    environment:
      POSTGRES_DB: digital_twin_tumor
      POSTGRES_USER: dtt
      POSTGRES_PASSWORD: ${DTT_DB_PASSWORD}
    volumes:
      - ruvector_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          memory: 2G

  digital-twin-tumor:
    build: .
    ports:
      - "7860:7860"
    environment:
      DATABASE_URL: postgresql://dtt:${DTT_DB_PASSWORD}@ruvector-postgres:5432/digital_twin_tumor
      RUVECTOR_URL: http://ruvector-postgres:8080
    depends_on:
      - ruvector-postgres

volumes:
  ruvector_data:
```

**Resource Characteristics:**

- RuVector runs entirely on CPU (no GPU required), using approximately 500MB-2GB RAM
  depending on the size of the vector index and the number of stored embeddings.
- The database tier does NOT affect GPU memory budget -- it runs as a separate CPU
  process. GPU VRAM remains fully available for MedGemma, MedSAM, and other deep
  learning models.
- For Kaggle notebooks, where Docker is not available, RuVector can run as an
  in-process library using Python bindings (no separate container needed). This mode
  uses SQLite as the backing store and provides the same vector search API.
- For Colab notebooks on the free tier, the in-process mode is also recommended to
  avoid Docker overhead. Colab Pro and local environments use the full Docker deployment.

### 3. MedSAM Configuration

MedSAM runs in inference mode with the following settings:

- Input: 2D slices extracted from 3D volumes at lesion centroid locations
- Prompt: bounding box prompt derived from prior segmentation or user click
- Resolution: images resized to 1024x1024 as required by SAM architecture
- VRAM: ~1.5-2GB during inference
- Batch size: 1 slice at a time (interactive mode) or 4-8 slices (batch mode)

For 3D volumetric segmentation, MedSAM is applied slice-by-slice with post-processing
to enforce 3D consistency (connected component analysis, morphological smoothing).

### 4. nnU-Net Configuration

nnU-Net provides an alternative segmentation path for volumetric analysis:

- **GPU mode**: Standard nnU-Net inference, ~2-4GB VRAM depending on patch size
- **CPU fallback**: Enabled via `device=torch.device('cpu')`, slower but functional
  for cases where GPU VRAM is fully allocated to MedGemma
- **Pre-trained weights**: Use published nnU-Net models for liver/lung tumor
  segmentation from the Medical Segmentation Decathlon
- **Patch-based inference**: Sliding window with overlap for memory efficiency

CPU fallback performance is approximately 5-10x slower than GPU but remains acceptable
for batch preprocessing where results are cached.

### 5. KV Cache Management for MedGemma

MedGemma's KV cache grows linearly with context length and consumes significant VRAM
for long prompts. Management strategy:

- **Maximum context**: Limit input context to 2048 tokens for routine reports; allow
  up to 4096 tokens for comprehensive multi-timepoint summaries.
- **Image tokens**: MedGemma 4B multimodal encodes images as ~256 tokens each. Limit
  to 2-3 images per inference call (current scan + 1-2 prior scans).
- **Structured prompts**: Use concise, structured prompts with measurement tables
  rather than verbose natural language to minimize token usage.
- **KV cache clearing**: Explicitly clear KV cache between independent inference calls
  by deleting past_key_values.
- **No chat history accumulation**: Each inference call is independent; do not maintain
  conversational KV cache across calls.

Example prompt structure for efficient token usage:

```
[System prompt: ~200 tokens]
[Image: current scan, ~256 tokens]
[Image: prior scan, ~256 tokens]
[Measurement table: ~100 tokens]
[Query: ~50 tokens]
Total: ~862 tokens input
```

### 6. MedGemma 27B: Optional High-Quality Path

MedGemma 27B is supported as an optional upgrade path for environments with sufficient
resources:

| Environment       | Feasibility | Configuration                          |
|-------------------|-------------|----------------------------------------|
| Kaggle T4 (16GB)  | No          | Not feasible even with INT4            |
| Colab A100 (40GB) | Yes         | INT4 quantization (~16GB), fits with margin |
| Colab A100 (80GB) | Yes         | FP16 (~54GB), comfortable fit          |
| Multi-GPU (2xT4)  | Possible    | Device map with accelerate             |
| Local RTX 4090    | Yes         | INT4 quantization (~16GB)              |

When MedGemma 27B is available, the system uses it for:
- Final report generation (batch mode, not interactive)
- Complex multi-timepoint narrative synthesis
- Detailed differential description of treatment response patterns

The 4B model remains the default for interactive queries due to lower latency.

**Device Map for Multi-GPU**:

```python
from accelerate import infer_auto_device_map, init_empty_weights

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_name)

device_map = infer_auto_device_map(
    model,
    max_memory={0: "14GiB", 1: "14GiB", "cpu": "24GiB"},
    no_split_module_classes=["GemmaDecoderLayer"],
)
```

### 7. Batch Preprocessing Pipeline

To minimize interactive latency, heavy computation is front-loaded:

**Offline Preprocessing** (runs once per dataset load):

| Step                          | Target Time  | Output Cached As           |
|-------------------------------|-------------|----------------------------|
| DICOM to NIfTI conversion     | <10s/scan   | `.nii.gz` files            |
| Resampling to isotropic       | <5s/scan    | Resampled `.nii.gz`        |
| Windowing & normalization     | <2s/scan    | Preprocessed tensors `.pt` |
| MedSAM segmentation           | <5s/scan    | Mask `.nii.gz` + `.npy`    |
| Lesion measurement extraction | <1s/scan    | JSON measurements          |
| MedSigLIP embedding extraction| <2s/scan    | Embedding `.npy` files     |
| Growth model fitting          | <2s total   | Model parameters JSON      |

**Total offline preprocessing**: <30s per scan (target), <3 minutes for a typical
5-timepoint longitudinal case.

**Cached Tensor Format**:

```python
# Preprocessed tensors saved for rapid loading
torch.save({
    'volume': preprocessed_tensor,      # [1, D, H, W] float32
    'spacing': original_spacing,         # [3] float64
    'origin': original_origin,           # [3] float64
    'direction': original_direction,     # [9] float64
    'window_center': window_center,      # float
    'window_width': window_width,        # float
}, f"data/processed/{patient_id}/{timepoint_id}/preprocessed.pt")
```

### 8. Runtime Performance Targets

Interactive operations (after preprocessing):

| Operation                    | Target Latency | Notes                        |
|------------------------------|---------------|------------------------------|
| Slice rendering              | <100ms        | Pre-loaded NumPy array       |
| Lesion selection & highlight | <200ms        | Cached mask overlay          |
| Measurement display          | <50ms         | Pre-computed JSON            |
| Growth curve update          | <500ms        | Lightweight scipy curve_fit  |
| Counterfactual projection    | <1s           | Analytical model, no GPU     |
| MedGemma narrative (4B)      | <10s          | ~200 output tokens at ~20 tok/s |
| MedGemma narrative (27B)     | <30s          | ~200 output tokens at ~7 tok/s  |

### 9. Environment Detection & Configuration

The system auto-detects the runtime environment at startup:

```python
def detect_environment():
    """Detect compute environment and set configuration."""
    env = {
        'platform': 'unknown',
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': None,
        'gpu_vram_gb': 0,
        'cpu_cores': os.cpu_count(),
        'ram_gb': psutil.virtual_memory().total / (1024**3),
    }

    if env['gpu_available']:
        props = torch.cuda.get_device_properties(0)
        env['gpu_name'] = props.name
        env['gpu_vram_gb'] = props.total_mem / (1024**3)

    # Detect platform
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        env['platform'] = 'kaggle'
    elif 'google.colab' in str(get_ipython()):
        env['platform'] = 'colab'
    else:
        env['platform'] = 'local'

    return env
```

Configuration is then selected from a predefined profile:

| Profile         | MedGemma  | Precision | MedSAM | nnU-Net | Batch Size | Database                        |
|-----------------|-----------|-----------|--------|---------|------------|---------------------------------|
| `kaggle_t4`     | 4B        | float16   | GPU    | CPU     | 1          | In-process RuVector (no Docker) |
| `colab_t4`      | 4B        | float16   | GPU    | CPU     | 1          | In-process RuVector (no Docker) |
| `colab_a100`    | 27B       | float16   | GPU    | GPU     | 4          | Docker RuVector-Postgres        |
| `local_16gb`    | 4B        | float16   | GPU    | GPU     | 2          | Docker RuVector-Postgres        |
| `local_24gb`    | 4B        | bfloat16  | GPU    | GPU     | 4          | Docker RuVector-Postgres        |
| `cpu_only`      | 4B        | int4+offload | CPU | CPU     | 1          | In-process RuVector (no Docker) |

### 10. Python Dependencies

| Library         | Version     | Purpose                                      |
|-----------------|-------------|----------------------------------------------|
| `torch`         | >=2.1       | Core deep learning framework                 |
| `transformers`  | >=4.40      | MedGemma model loading and inference         |
| `accelerate`    | >=0.27      | Device mapping, CPU offload, multi-GPU       |
| `bitsandbytes`  | >=0.42      | 4-bit NF4 quantization                       |
| `safetensors`   | >=0.4       | Fast, safe model weight loading              |
| `psutil`        | >=5.9       | System resource detection                    |
| `SimpleITK`     | >=2.3       | Medical image I/O and preprocessing          |
| `nibabel`       | >=5.0       | NIfTI file I/O                               |
| `pydicom`       | >=2.4       | DICOM file I/O                               |
| `numpy`         | >=1.24      | Array operations                             |
| `scipy`         | >=1.10      | Growth model curve fitting                   |
| `psycopg[binary]` | >=3.1    | PostgreSQL async client                       |
| `ruvector`      | >=0.1       | Vector database client (in-process or remote) |

## Consequences

### Positive

- The system runs on the most common free-tier GPU (T4 16GB) without modification,
  maximizing accessibility for hackathon judges and reviewers.
- Automatic precision selection means users never need to manually configure VRAM
  management; the system adapts to available hardware.
- Sequential model loading with explicit cache clearing prevents OOM errors that would
  crash the notebook and lose state.
- Batch preprocessing with cached results ensures interactive operations meet latency
  targets regardless of dataset size.
- The graceful degradation path (float16 -> int4 -> CPU offload) means the system
  produces results on any hardware, trading speed for accessibility.
- MedGemma 27B support provides a quality upgrade path without requiring code changes.
- The RuVector+PostgreSQL database tier runs entirely on CPU and does not consume any
  GPU VRAM, keeping the full GPU memory budget available for deep learning models.

### Negative

- Sequential model loading means the system cannot run segmentation and narrative
  generation simultaneously. Users must wait for phase transitions (~5-10s for model
  swap). This is acceptable for an interactive demo.
- 4-bit quantization introduces small quality degradation in MedGemma outputs compared
  to full precision. Empirical testing shows this is minimal for structured medical
  reporting tasks.
- CPU fallback for nnU-Net is 5-10x slower, making batch preprocessing for large
  datasets impractical without GPU. This is documented as a known limitation.
- The preprocessing cache consumes disk space (~50-100MB per scan for preprocessed
  tensors, masks, and embeddings). Kaggle provides sufficient disk for typical cases.

### Risks

- CUDA OOM errors during MedGemma inference if KV cache grows unexpectedly. Mitigation:
  hard context length limits and try/except with automatic fallback to smaller batch
  or int4 precision.
- Model download times on Kaggle/Colab can be significant (~5-10 minutes for MedGemma
  4B). Mitigation: document expected setup time; consider pre-cached model weights in
  Kaggle datasets.
- BitsAndBytes library may have compatibility issues with specific CUDA versions.
  Mitigation: pin library versions and document tested CUDA configurations.
- T4 GPU may be unavailable during peak Kaggle usage. Mitigation: CPU-only fallback
  path ensures the system still functions, albeit slowly.

## Alternatives Considered

### 1. Cloud API-Based Inference (Vertex AI)

Use Google Cloud's MedGemma API instead of local GPU inference. Rejected because:
requires API keys and billing setup, adds network latency, cannot be self-contained
in a Kaggle notebook, and introduces a dependency on external service availability.

### 2. ONNX Runtime Optimization

Convert models to ONNX format for optimized inference. Rejected because: MedGemma's
multimodal architecture has limited ONNX export support, the optimization gain is
marginal compared to native PyTorch with float16, and it adds build complexity.

### 3. TensorRT Quantization

Use NVIDIA TensorRT for INT8 inference. Rejected because: TensorRT compilation is
time-consuming and environment-specific, not suitable for notebook-based demos where
the model must be loaded fresh each session.

### 4. Model Sharding Across CPU+GPU

Shard MedGemma across CPU RAM and GPU VRAM using accelerate's device_map. This is
included as the CPU offload fallback but not the primary strategy because it
significantly increases inference latency (~3-5x slower than full GPU inference).

### 5. Smaller Model Only (MedGemma 4B INT4 Always)

Always use 4-bit quantization regardless of available VRAM. Rejected because: when
sufficient VRAM is available, full float16 precision provides measurably better output
quality. The automatic precision selection captures the best of both approaches.
