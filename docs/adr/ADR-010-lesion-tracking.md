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

### RuVector-Enhanced Lesion Matching

For production and multi-patient deployments, RuVector's native graph and vector
capabilities augment the NetworkX-based pipeline with persistent storage, GNN-enhanced
matching, and expressive Cypher graph queries.

#### Graph Storage in RuVector

The lesion identity graph is stored in RuVector's graph engine, which provides:

- **Persistent, queryable storage**: unlike the in-memory NetworkX graph, the RuVector
  graph persists across sessions in the `ruvector-postgres` Docker container.
- **Cypher query language**: all graph traversal operations are expressed as Cypher
  queries, enabling complex clinical questions to be answered declaratively.
- **Scalability**: RuVector's Raft consensus protocol supports horizontal scaling for
  multi-institution deployments with hundreds of patients.

#### GNN-Based Matching

RuVector's GNN layer replaces the static centroid+embedding pipeline with a learned
matching model that improves over time:

- **Self-improving index**: the GNN layer learns from confirmed matches (both automatic
  and human-verified) to improve future matching accuracy. Each confirmation is a
  training signal that refines the model's internal representation.
- **Contextual embeddings**: each lesion node has an embedding vector that the GNN
  refines based on neighbor context (prior observations, co-occurring lesions, anatomical
  location). This captures information that isolated centroid/appearance features miss.
- **Multi-head attention**: the GNN uses MultiHeadAttention with 4 heads to weigh
  spatial features (centroid proximity after registration), temporal features (time
  interval between observations), and appearance features (MedSigLIP embedding
  similarity). The attention weights are interpretable and can be inspected to understand
  why a particular match was made.

#### Hybrid Search for Matching

RuVector's hybrid search combines vector similarity with graph constraints in a single
query, replacing the sequential centroid-then-embedding pipeline:

```python
# RuVector hybrid search for lesion matching
matches = ruvector_client.hybrid_search(HybridQuery(
    vector=current_lesion_embedding,
    cypher_filter="""
        MATCH (l:Lesion)-[:OBSERVED_AT]->(t:Timepoint)
        WHERE t.date < $current_date AND l.organ = $organ
    """,
    k=5,
))
```

This approach is more expressive than the sequential pipeline because:

- Graph constraints (same organ, prior timepoint) are applied simultaneously with
  vector similarity, not as separate filtering stages.
- The Cypher filter can encode arbitrarily complex clinical logic (e.g., exclude
  lesions that have already been matched, restrict to lesions within a specific
  anatomical region).

#### Cypher Queries for Graph Operations

All graph operations are expressed as Cypher queries:

```cypher
-- Get lesion trajectory
MATCH (l:Lesion {id: 'L001'})-[:TRACKED_AS*]->(obs:Observation)
RETURN obs ORDER BY obs.timepoint

-- Find new lesions at a timepoint
MATCH (obs:Observation {timepoint: '2025-06-15'})
WHERE NOT (obs)<-[:TRACKED_AS]-()
RETURN obs

-- Get lineage with split/merge
MATCH path = (start:Observation)-[:TRACKED_AS|SPLIT_FROM|MERGED_INTO*]->(end:Observation)
WHERE start.lesion_id = 'L001'
RETURN path

-- Count active lesions per patient across all timepoints
MATCH (p:Patient)-[:HAS_LESION]->(l:Lesion)-[:OBSERVED_AT]->(t:Timepoint)
RETURN p.id, t.date, count(l) as active_count
ORDER BY p.id, t.date
```

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
compatible dictionary. While JSON serialization is retained for portability and
notebook environments, the primary storage for production deployments is RuVector's
graph engine, which provides persistent, queryable storage via the `ruvector-postgres`
Docker container. The JSON export serves as a snapshot/backup format and enables
offline analysis without a running RuVector instance.

One JSON file per patient is stored at:

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

### RuVector Graph Query Utilities

When RuVector is available, the utility functions above map to Cypher queries for
persistent, indexed execution. The mapping is:

| NetworkX Utility                  | RuVector Cypher Equivalent                                      |
|-----------------------------------|-----------------------------------------------------------------|
| `get_lesion_trajectory(g, lid)`   | `MATCH (l:Lesion {id: $lid})-[:TRACKED_AS*]->(o) RETURN o ORDER BY o.timepoint` |
| `get_new_lesions(g, tp)`          | `MATCH (o:Observation {timepoint: $tp}) WHERE NOT (o)<-[:TRACKED_AS]-() RETURN o` |
| `get_disappeared_lesions(g, tp)`  | `MATCH (o:Observation {timepoint: $prev_tp}) WHERE NOT (o)-[:TRACKED_AS]->() RETURN o` |
| `get_active_lesions(g, tp)`       | `MATCH (o:Observation {timepoint: $tp}) RETURN o`               |
| `get_lineage(g, nid)`             | `MATCH path = (n {id: $nid})-[:TRACKED_AS|SPLIT_FROM|MERGED_INTO*]-(m) RETURN path` |
| `validate_dag(g)`                 | `MATCH (n)-[:TRACKED_AS*]->(n) RETURN count(n) = 0 as is_dag`  |

A `LesionGraphBackend` abstraction layer provides a unified interface that delegates to
either NetworkX (in-memory, Kaggle/notebook mode) or RuVector (persistent, production
mode) based on configuration:

```python
class LesionGraphBackend(Protocol):
    def get_trajectory(self, lesion_id: str) -> List[Observation]: ...
    def get_new_lesions(self, timepoint: str) -> List[Observation]: ...
    def find_matches(self, lesion_embedding: np.ndarray, constraints: dict) -> List[Match]: ...
```

### Python Dependencies

| Library     | Version   | Purpose                                      |
|-------------|-----------|----------------------------------------------|
| `networkx`  | >=3.0     | Graph data structure, DAG operations, JSON I/O |
| `SimpleITK` | >=2.3     | Image registration (rigid, affine)           |
| `numpy`     | >=1.24    | Coordinate transforms, distance computation  |
| `scipy`     | >=1.10    | Hungarian algorithm for matching             |
| `ruvector`  | >=0.1     | Graph storage, GNN-enhanced matching, Cypher queries, hybrid search |
| `psycopg`   | >=3.1     | PostgreSQL client for structured lesion metadata |

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
- GNN-enhanced matching improves over time with each confirmed match, creating a virtuous
  cycle where the system becomes more accurate as more data is processed.
- Cypher queries provide expressive graph traversal for complex clinical questions (e.g.,
  "find all patients where a specific lesion split during treatment") without custom
  traversal code.
- Hybrid search combining spatial, temporal, and appearance features in a single RuVector
  query replaces the sequential multi-stage matching pipeline with a more natural and
  efficient approach.

### Negative

- NetworkX stores graphs in memory; for patients with hundreds of lesions across dozens
  of timepoints, memory usage could become significant. Note: RuVector's persistent
  graph storage eliminates this concern for large-scale deployments, as the graph is
  stored on disk with indexed access rather than held entirely in memory. NetworkX
  remains the fallback for notebook environments where the data volume is small.
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

A dedicated graph database for lesion tracking. Originally rejected because of excessive
infrastructure for the expected data volume and deployment complexity incompatible with
Kaggle/Colab targets.

**Partially addressed by RuVector:** RuVector provides equivalent graph query capabilities
(full Cypher query language) without the infrastructure overhead of a separate Neo4j
instance, since it is bundled with PostgreSQL in the `ruvector-postgres` Docker image.
This gives the system Neo4j-class graph queries (Cypher), GNN-enhanced indexing, and
vector similarity search in a single container, eliminating the primary objection to a
dedicated graph database. NetworkX remains the in-memory fallback for notebook
environments where Docker is unavailable.
