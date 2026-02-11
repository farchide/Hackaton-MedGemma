# ADR-010: Lesion Tracking & Identity Graph

## Status

Accepted

## Date

2026-02-11

## Context

The Digital Twin Tumor system must track individual lesions across multiple imaging
timepoints for a single patient. Oncology longitudinal follow-up requires that each
measurable lesion observed at baseline be unambiguously linked to the same lesion at
subsequent scans, even when lesions change size, shape, or position due to treatment
response, progression, or anatomical shifts between scans.

Key challenges include:

1. **Identity persistence**: A lesion at timepoint T0 must maintain its identity through
   T1, T2, ... TN even as its morphology changes dramatically (partial response, growth,
   cavitation, necrosis).
2. **New lesion detection**: iRECIST and RECIST 1.1 both require identification of new
   lesions, which changes the overall response category. The data structure must cleanly
   represent lesions that appear at later timepoints with no prior identity.
3. **Complete response**: Lesions may disappear entirely. The system must represent this
   as a terminal state rather than data loss.
4. **Split and merge events**: In rare cases a lesion may fragment into multiple distinct
   masses or adjacent lesions may coalesce. The tracking model must support these
   topological changes.
5. **Multi-method matching**: Matching may be performed automatically (registration-based
   centroid proximity, appearance embeddings) or manually by a radiologist. Both sources
   of identity assignment must coexist with clear provenance.
6. **Serialization**: The tracking structure must be serializable for storage, reproducible
   analysis, and integration with the rest of the Digital Twin pipeline.

Alternative flat-table approaches (e.g., a DataFrame with `lesion_id` and `timepoint`
columns) were considered but rejected because they cannot naturally represent split/merge
events, do not encode match confidence on the identity link itself, and make graph
queries (ancestors, descendants, connected components) cumbersome.

## Decision

We adopt a **directed acyclic graph (DAG)** data structure for lesion identity tracking.
The graph is implemented using NetworkX and serialized to JSON per patient.

### Graph Structure

**Nodes** represent a specific lesion observation at a specific timepoint. Each node is
uniquely keyed as `{lesion_id}@{timepoint_id}`.

Node attributes:

| Attribute              | Type       | Description                                              |
|------------------------|------------|----------------------------------------------------------|
| `lesion_id`            | `str`      | Unique identifier for the lesion lineage                 |
| `timepoint_id`         | `str`      | Identifier for the imaging timepoint (e.g., ISO date)    |
| `timepoint_index`      | `int`      | Ordinal index of the timepoint (0-based)                 |
| `centroid_world`       | `float[3]` | Centroid in world coordinates (mm), from image header     |
| `centroid_voxel`       | `int[3]`   | Centroid in voxel indices                                |
| `volume_mm3`           | `float`    | Lesion volume in cubic millimeters                       |
| `longest_diameter_mm`  | `float`    | RECIST longest axial diameter in millimeters             |
| `short_axis_mm`        | `float`    | Short axis diameter (relevant for lymph nodes)           |
| `segmentation_mask_ref`| `str`      | File path or key to the binary segmentation mask         |
| `confidence`           | `float`    | Segmentation confidence score (0.0 - 1.0)               |
| `anatomy_label`        | `str`      | Anatomical location label (e.g., "liver", "lung_right")  |
| `is_target`            | `bool`     | Whether this is a RECIST target lesion                   |
| `is_new`               | `bool`     | Whether this lesion was first detected at this timepoint |
| `metadata`             | `dict`     | Extensible dict for additional attributes                |

**Edges** represent identity links between observations across timepoints. An edge from
node A to node B means "the lesion observed at A is the same lesion observed at B" (or a
descendant in split scenarios).

Edge attributes:

| Attribute               | Type    | Description                                            |
|-------------------------|---------|--------------------------------------------------------|
| `match_confidence`      | `float` | Confidence of the identity match (0.0 - 1.0)          |
| `match_method`          | `str`   | One of: `manual`, `auto_centroid`, `auto_embedding`, `auto_hybrid` |
| `registration_transform`| `str`   | Serialized reference to the spatial transform used     |
| `registation_method`    | `str`   | Registration algorithm (e.g., `rigid`, `affine`)       |
| `user_override`         | `bool`  | Whether a human manually set or corrected this edge    |
| `timestamp`             | `str`   | ISO 8601 timestamp of when this edge was created       |

### Topological Semantics

- **Tracked lesion**: A node at timepoint T>0 with at least one incoming edge from T-1.
- **New lesion**: A node at timepoint T>0 with zero incoming edges. The `is_new` flag is
  set to `True`. This is critical for iRECIST: new lesions trigger "iUPD" status.
- **Complete response (disappeared)**: A node at timepoint T with zero outgoing edges to
  T+1, and the lesion is not found at T+1. Represented by the absence of a successor
  node rather than a sentinel value.
- **Split**: A node at timepoint T with two or more outgoing edges to distinct nodes at
  T+1. Each descendant receives a new `lesion_id` suffixed with a split index (e.g.,
  `L001_a`, `L001_b`) while preserving the lineage via the edge.
- **Merge**: Two or more nodes at timepoint T with edges converging to a single node at
  T+1. The merged node receives a new `lesion_id` and both ancestors are recorded.

### Auto-Matching Pipeline

Automatic identity matching proceeds in three stages:

1. **Rigid/Affine Registration** (SimpleITK): Register scan at T+1 to scan at T using
   `sitk.ImageRegistrationMethod` with mutual information metric and regular step
   gradient descent. Apply the resulting transform to all lesion centroids at T to map
   them into the coordinate space of T+1.

2. **Centroid Proximity Matching**: After registration, compute pairwise Euclidean
   distances between transformed centroids from T and raw centroids at T+1. Apply the
   Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) with a maximum distance
   threshold of 20mm (configurable). Pairs within threshold become candidate matches.

3. **Appearance Similarity Refinement**: For candidate matches with centroid distance
   between 10mm and 20mm (ambiguous zone), compute cosine similarity of MedSigLIP
   embeddings extracted from cropped ROI patches around each lesion. Reject matches
   below a similarity threshold of 0.7 (configurable).

Unmatched lesions at T+1 are flagged as new lesion candidates. Unmatched lesions at T
are flagged as potentially disappeared.

### Human Override

The system always allows a human operator to:

- **Reassign**: Change which prior lesion a current observation maps to.
- **Split**: Declare that a single prior lesion has become multiple observations.
- **Merge**: Declare that multiple prior lesions have coalesced.
- **Create**: Manually add a new lesion node not found by segmentation.
- **Delete**: Remove a spurious node (false positive segmentation).

All human overrides set `user_override=True` on affected edges and log the action to
the audit trail (see ADR-011).

### Serialization

The graph is serialized using `networkx.node_link_data()` which produces a JSON-
compatible dictionary. One JSON file per patient is stored at:

```
data/processed/{patient_id}/lesion_graph.json
```

The JSON structure follows the NetworkX node-link format:

```json
{
  "directed": true,
  "multigraph": false,
  "graph": {
    "patient_id": "PATIENT_001",
    "created_at": "2026-02-11T10:00:00Z",
    "schema_version": "1.0"
  },
  "nodes": [
    {
      "id": "L001@T0",
      "lesion_id": "L001",
      "timepoint_id": "2025-01-15",
      "centroid_world": [102.3, 45.1, -220.5],
      "volume_mm3": 1250.0,
      "longest_diameter_mm": 15.2,
      "confidence": 0.94
    }
  ],
  "links": [
    {
      "source": "L001@T0",
      "target": "L001@T1",
      "match_confidence": 0.97,
      "match_method": "auto_hybrid",
      "user_override": false
    }
  ]
}
```

### Query Utilities

The following graph queries are supported via utility functions:

- `get_lesion_trajectory(graph, lesion_id)` -- returns the ordered list of nodes for a
  lesion across all timepoints.
- `get_new_lesions(graph, timepoint_id)` -- returns all nodes at a timepoint with no
  incoming edges.
- `get_disappeared_lesions(graph, timepoint_id)` -- returns all nodes at the prior
  timepoint with no outgoing edges to the given timepoint.
- `get_active_lesions(graph, timepoint_id)` -- returns all nodes at a timepoint.
- `get_lineage(graph, node_id)` -- returns all ancestors and descendants of a node.
- `validate_dag(graph)` -- asserts the graph is a valid DAG (no cycles).

### Python Dependencies

| Library     | Version   | Purpose                                      |
|-------------|-----------|----------------------------------------------|
| `networkx`  | >=3.0     | Graph data structure, DAG operations, JSON I/O |
| `SimpleITK` | >=2.3     | Image registration (rigid, affine)           |
| `numpy`     | >=1.24    | Coordinate transforms, distance computation  |
| `scipy`     | >=1.10    | Hungarian algorithm for matching             |

## Consequences

### Positive

- The DAG naturally models all lesion lifecycle events: appearance, tracking, split,
  merge, and disappearance without sentinel values or nullable foreign keys.
- Match provenance (auto vs. manual, confidence scores) is encoded directly on edges,
  enabling downstream quality filtering and uncertainty propagation.
- NetworkX provides a mature, well-tested graph library with built-in DAG validation,
  topological sorting, and path-finding algorithms.
- JSON serialization is human-readable, version-controllable, and easily loaded in
  notebooks for analysis.
- The registration-then-proximity-then-embedding pipeline provides robust matching even
  with significant patient repositioning between scans.

### Negative

- NetworkX stores graphs in memory; for patients with hundreds of lesions across dozens
  of timepoints, memory usage could become significant (mitigated: oncology cases
  rarely exceed 20 target + non-target lesions across 10 timepoints).
- JSON serialization is verbose compared to binary formats; for the expected data sizes
  this is acceptable.
- The three-stage auto-matching pipeline requires both SimpleITK registration and
  MedSigLIP inference, adding computational overhead per timepoint pair.
- Graph-based structures are less familiar to data scientists accustomed to tabular
  DataFrames; utility functions and clear documentation are required.

### Risks

- Registration failures (e.g., due to large anatomical changes post-surgery) could
  cascade into incorrect auto-matches. Mitigation: always present auto-matches for
  human review before finalizing.
- Schema evolution of node/edge attributes requires migration logic for existing JSON
  files. Mitigation: `schema_version` field enables forward-compatible loading.

## Alternatives Considered

### 1. Flat DataFrame with lesion_id Column

A pandas DataFrame with columns `(patient_id, lesion_id, timepoint, diameter, volume)`.
Rejected because: cannot represent split/merge, cannot store per-link confidence, and
graph queries require complex groupby/merge operations.

### 2. Relational Database Tables

SQL tables for `lesions`, `observations`, and `identity_links`. Rejected because: adds
infrastructure complexity (database server) inappropriate for a Kaggle notebook
environment, and the data volume does not justify a relational engine.

### 3. SimpleITK Label Maps Only

Track lesions purely through connected-component labels in registered segmentation masks.
Rejected because: registration errors directly corrupt identity, no mechanism for
confidence scoring, and no support for human override.

### 4. Neo4j or Graph Database

A dedicated graph database for lesion tracking. Rejected because: excessive infrastructure
for the expected data volume (tens of nodes per patient), adds deployment complexity
incompatible with Kaggle/Colab targets, and NetworkX provides sufficient functionality.
