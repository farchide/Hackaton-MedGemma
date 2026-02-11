# RuVector Examples - Comprehensive Catalog

## Repository Overview

**Repository**: https://github.com/ruvnet/ruvector
**Language**: Rust (with Node.js bindings, WASM support, TypeScript SDK)
**License**: MIT OR Apache-2.0
**Description**: A distributed vector database with self-learning capabilities -- "the vector database that gets smarter the more you use it."

**Core Features**: Vector search, graph queries (Cypher), machine learning (GNN), distributed systems (Raft consensus), 40+ attention mechanisms, HNSW indexing, ONNX embedding generation, spiking neural networks, hyperdimensional computing.

---

## EXAMPLES DIRECTORY STRUCTURE (33 items)

```
examples/
  bounded_instance_demo.rs          # Standalone Rust file
  README.md
  rust/                             # 6 core Rust examples
  graph/                            # 4 graph database examples
  nodejs/                           # 2 Node.js examples
  onnx-embeddings/                  # ONNX embedding library (full crate)
  onnx-embeddings-wasm/             # ONNX in WASM
  wasm/                             # WASM iOS
  wasm-react/                       # React + WASM app
  wasm-vanilla/                     # Vanilla HTML + WASM
  edge/                             # Edge AI swarm framework
  edge-net/                         # Collective AI computing network
  edge-full/                        # Full edge deployment
  agentic-jujutsu/                  # Quantum-resistant version control for AI agents
  exo-ai-2025/                      # Consciousness/cognition research platform
  meta-cognition-spiking-neural-network/  # JS-based SNN + attention demos
  spiking-network/                  # Rust spiking neural network crate
  neural-trader/                    # Financial trading system
  scipix/                           # Rust OCR engine for scientific documents
  prime-radiant/                    # Universal coherence engine (sheaf math)
  delta-behavior/                   # Coherence-first state management
  refrag-pipeline/                  # ~30x RAG latency reduction
  ruvLLM/                           # Local LLM inference (incl. ESP32)
  vibecast-7sense/                  # Bioacoustic intelligence platform
  mincut/                           # Self-organizing network examples
  subpolynomial-time/               # Subpolynomial min-cut algorithm
  ultra-low-latency-sim/            # Ultra-low latency simulation
  google-cloud/                     # GPU-accelerated Cloud Run deployment
  vwm-viewer/                       # Vector working memory viewer
  apify/                            # Web scraping integration
  data/                             # Data files
  benchmarks/                       # Performance benchmarks
  docs/                             # Documentation
```

---

## CATEGORY 1: CORE RUST EXAMPLES (`examples/rust/`)

### 1.1 basic_usage.rs
**Purpose**: Getting started with RuVector -- database creation, vector insertion, search.
**APIs Demonstrated**:
- `VectorDB::new(DbOptions)` -- create database with dimensions and storage path
- `VectorEntry { id, vector, metadata }` -- vector data structure
- `db.insert(entry)` -- single vector insertion
- `db.insert_batch(entries)` -- bulk insertion of 100 vectors
- `SearchQuery { vector, k, filter, include_vectors }` -- similarity search
- `db.search(&query)` -- returns results with id and distance
- `db.count()` -- database statistics

**Key Patterns**:
```rust
let mut options = DbOptions::default();
options.dimensions = 128;
options.storage_path = "./examples_basic.db".to_string();
let db = VectorDB::new(options)?;

let query = SearchQuery {
    vector: vec![0.15; 128],
    k: 5,
    filter: None,
    include_vectors: false,
};
let results = db.search(&query)?;
```

**Medical Imaging Application**: Foundation for storing and searching MedGemma image embeddings. Create a 128/384/768-dimension database of tumor image embeddings, insert embeddings from processed MRI/CT scans, and search for similar tumors by embedding distance.

---

### 1.2 advanced_features.rs
**Purpose**: Demonstrates six advanced subsystems: hypergraph indexing, temporal hypergraphs, causal memory, learned indexes, neural hashing, and topological analysis.
**APIs Demonstrated**:
- `HypergraphIndex::new()` -- hypergraph for multi-node relationships
- `TemporalHyperedge::new()` -- time-stamped hyperedges for event tracking
- `CausalMemory::new()` -- weighted utility scoring for agent reasoning
  - Formula: `0.7*similarity + 0.2*causal_uplift - 0.1*latency`
- `RecursiveModelIndex::new()` -- learned index structures
- `NeuralHasher::new()` -- LSH-based vector compression
- Topological analysis with quality reports

**Key Patterns**:
```rust
// Hypergraph: model relationships between multiple entities
let mut hg = HypergraphIndex::new();
hg.add_hyperedge(HyperedgeData {
    node_ids: vec![...],
    weight: 1.0,
    metadata: Some(serde_json::json!({...})),
});

// Causal memory with weighted utility scoring
let utility = 0.7 * similarity + 0.2 * causal_uplift - 0.1 * latency;
```

**Medical Imaging Application**:
- **Hypergraphs**: Model relationships between tumors, patients, treatment protocols, and outcomes where a single treatment affects multiple tumors
- **Temporal hypergraphs**: Track tumor progression over time across multiple scan dates
- **Causal memory**: Store and retrieve diagnostic reasoning chains -- "similar tumor appearance + known treatment outcome - diagnostic delay"
- **Neural hashing**: Compress high-dimensional MedGemma embeddings for efficient storage of millions of scan images
- **Topological analysis**: Detect anomalies in embedding quality, flag scans with unusual feature distributions

---

### 1.3 agenticdb_demo.rs
**Purpose**: Demonstrates the AgenticDB subsystem -- five interconnected tables for AI agent memory and learning.
**APIs Demonstrated**:
- `AgenticDB::new()` -- create agent memory database
- **Reflexion Episodes**: `db.store_episode()` -- store experiences with success/failure and embedding
- **Skill Library**: `db.create_skill()` -- store reusable capabilities with vector embeddings
- **Causal Memory**: `db.add_causal_edge()` -- hypergraph relationships with utility scoring
- **World Model**: Maintain agent's understanding of environment state
- **RL Sessions**: `db.start_session()` -- reinforcement learning session management

**Key Patterns**:
```rust
// Store diagnostic episode
db.store_episode(Episode {
    task: "diagnose_tumor",
    outcome: "success",
    embedding: tumor_embedding,
    metadata: json!({"confidence": 0.95}),
})?;

// Create skill for tumor classification
db.create_skill(Skill {
    name: "classify_tumor_grade",
    embedding: skill_embedding,
    success_rate: 0.92,
})?;

// Track causal relationship
db.add_causal_edge(cause_id, effect_id, weight)?;
```

**Medical Imaging Application**: Build an AI radiologist agent that remembers past diagnoses (reflexion), learns reusable diagnostic skills (skill library), tracks cause-and-effect in treatment outcomes (causal memory), maintains a model of patient history (world model), and improves through reinforcement learning sessions. Critical for a MedGemma-based tumor tracking system where the AI agent needs persistent memory across patient encounters.

---

### 1.4 batch_operations.rs
**Purpose**: High-throughput batch processing -- 10,000 vector insertions and 1,000 search queries with performance measurement.
**APIs Demonstrated**:
- Bulk `insert_batch()` with 10,000 random 128-dimensional vectors
- Search benchmarking with throughput measurement
- `Instant::now()` and `elapsed()` for performance timing

**Key Patterns**:
```rust
let entries: Vec<VectorEntry> = (0..10_000)
    .map(|i| {
        let vector: Vec<f32> = (0..128).map(|_| rng.gen::<f32>()).collect();
        VectorEntry { id: Some(format!("vec_{:05}", i)), vector, metadata: None }
    })
    .collect();

let start = Instant::now();
let ids = db.insert_batch(entries)?;
let throughput = ids.len() as f64 / duration.as_secs_f64();
```

**Medical Imaging Application**: Ingest large datasets of medical images in batch. A hospital might process thousands of CT slices per day -- this pattern shows how to efficiently load them all into the vector database. Essential for initial dataset loading (e.g., importing 50,000 historical tumor scans from PACS systems).

---

### 1.5 gnn_example.rs
**Purpose**: Graph Neural Network layer usage -- forward pass, individual components.
**APIs Demonstrated**:
- `RuvectorLayer::new(input_dim, hidden_dim, heads, dropout)` -- GNN layer creation
- `gnn_layer.forward(&node_embedding, &neighbor_embeddings, &edge_weights)` -- message passing
- `Linear::new(in, out)` -- linear transformation layer
- `LayerNorm::new(dim, eps)` -- layer normalization
- `MultiHeadAttention::new(dim, heads)` -- multi-head attention mechanism
- `GRUCell::new(input, hidden)` -- gated recurrent unit for state updates

**Key Patterns**:
```rust
let gnn_layer = RuvectorLayer::new(128, 256, 4, 0.1);
let node_embedding = vec![0.5; 128];
let neighbor_embeddings = vec![vec![0.3; 128], vec![0.7; 128], vec![0.5; 128]];
let edge_weights = vec![0.8, 0.6, 0.4];
let updated = gnn_layer.forward(&node_embedding, &neighbor_embeddings, &edge_weights);
```

**Medical Imaging Application**: DIRECTLY APPLICABLE to tumor tracking. Build a graph where:
- **Nodes** = individual tumors (with MedGemma embeddings as features)
- **Edges** = spatial proximity, same-patient relationships, similar morphology
- **Edge weights** = inverse distance between tumors, temporal correlation
- GNN forward pass aggregates information from neighboring tumors to produce enriched representations that capture tumor context. This enables:
  - Predicting tumor behavior based on neighboring tissue characteristics
  - Identifying tumor clusters that share similar progression patterns
  - Multi-scale analysis combining local tumor features with regional context

---

### 1.6 rag_pipeline.rs
**Purpose**: Complete Retrieval-Augmented Generation pipeline -- document ingestion, embedding, retrieval, prompt construction.
**APIs Demonstrated**:
- Document ingestion with metadata (text, doc_id, timestamp)
- 384-dimensional embeddings (matching sentence-transformers/all-MiniLM-L6-v2)
- Semantic search with metadata retrieval
- RAG prompt construction from retrieved context
- `serde_json::json!()` for metadata

**Key Patterns**:
```rust
let entry = VectorEntry {
    id: Some(format!("doc_{}", i)),
    vector: embedding,
    metadata: Some(metadata),  // HashMap with text, doc_id, timestamp
};

// Construct RAG prompt
fn construct_rag_prompt(query: &str, context: &[&str]) -> String {
    format!("Context:\n{}\n\nUser Question: {}\n\nAnswer:", context_text, query)
}
```

**Medical Imaging Application**: Build a medical knowledge retrieval system where:
- Ingest radiology reports, clinical guidelines, and case studies as documents
- Generate embeddings for each document chunk
- When analyzing a new tumor image, retrieve the most relevant prior reports and guidelines
- Construct a prompt for MedGemma that includes retrieved clinical context
- This pattern directly enables "given this tumor image, find similar cases and their outcomes"

---

## CATEGORY 2: GRAPH DATABASE EXAMPLES (`examples/graph/`)

### 2.1 basic_graph.rs
**Purpose**: Template for fundamental graph operations -- node/relationship CRUD.
**Demonstrated Concepts** (template with commented-out API):
- `GraphDatabase::open()` -- create graph database
- Node creation with labels and properties (Person, age, name)
- Relationship creation (FRIENDS_WITH, since)
- Pattern matching queries
- Traversal queries
- Property updates and deletion
- Graph statistics

**Medical Imaging Application**: Model the medical knowledge graph:
- **Nodes**: Patient, Tumor, Scan, Treatment, Physician, Hospital
- **Relationships**: HAS_TUMOR, SCANNED_ON, TREATED_WITH, DIAGNOSED_BY
- **Properties**: tumor_grade, scan_date, treatment_protocol, confidence_score

---

### 2.2 cypher_queries.rs
**Purpose**: Neo4j-compatible Cypher query patterns -- 10 query types.
**Demonstrated Queries**:
1. CREATE -- `CREATE (n:Person {name: 'Charlie', age: 28}) RETURN n`
2. MATCH + WHERE -- `MATCH (p:Person) WHERE p.age > 25 RETURN p.name`
3. Relationship creation -- `CREATE (a)-[r:KNOWS {since: 2023}]->(b)`
4. Variable-length traversal -- `MATCH (start)-[:KNOWS*1..3]->(end)`
5. Aggregation -- `count(p), avg(p.age), min(), max()`
6. Shortest path -- `shortestPath((a)-[:KNOWS*]-(b))`
7. Pattern comprehension -- list comprehension in RETURN
8. Multi-pattern matching -- JOIN across multiple patterns
9. Property updates -- `SET p.age = p.age + 1`
10. MERGE (upsert) -- `ON CREATE SET ... ON MATCH SET ...`

**Medical Imaging Application**: Query the tumor tracking database:
```cypher
// Find all tumors that responded to a specific treatment
MATCH (t:Tumor)-[:TREATED_WITH]->(rx:Treatment {name: 'immunotherapy'})
WHERE t.response = 'positive'
RETURN t.grade, t.location, count(t) as responders

// Track tumor progression over time
MATCH (p:Patient {id: 'P001'})-[:HAS_TUMOR]->(t:Tumor)-[:SCANNED_ON]->(s:Scan)
RETURN t.id, s.date, s.volume ORDER BY s.date

// Find similar tumors by treatment pathway
MATCH path = (t1:Tumor)-[:SIMILAR_TO*1..3]->(t2:Tumor)
WHERE t2.outcome = 'remission'
RETURN path, length(path) as similarity_distance
```

---

### 2.3 distributed_cluster.rs
**Purpose**: Multi-node distributed clustering with Raft consensus.
**Demonstrated Concepts**:
- Cluster configuration for 3 nodes (ports 7001-7003)
- Raft consensus establishment and leader election
- Distributed data insertion across shards (1000 nodes)
- Cross-shard query execution
- Data replication verification
- Node failure simulation and failover
- Node recovery and rejoin
- Performance metrics collection

**Medical Imaging Application**: Deploy tumor database across multiple hospital sites:
- Each hospital runs a node with local patient data
- Raft consensus ensures consistent diagnostic protocols
- Cross-shard queries enable multi-institution tumor studies
- Failover ensures 24/7 availability for emergency diagnostics
- Federated model: each site contributes to the shared knowledge base

---

### 2.4 hybrid_search.rs
**Purpose**: Combining vector similarity search with graph traversal.
**Demonstrated Concepts**:
- Node creation with vector embeddings attached
- Pure semantic/vector similarity search
- Graph-constrained vector search (search within graph neighborhood)
- Combined scoring: `alpha * vector_score + (1-alpha) * graph_score`
- Multi-hop semantic traversal
- Performance comparison of vector-only vs. hybrid approaches

**Medical Imaging Application**: The most powerful pattern for tumor tracking:
- Attach MedGemma embeddings to tumor nodes in the graph
- Search by visual similarity (vector) AND clinical context (graph)
- Example: "Find tumors that LOOK like this one AND are connected to patients with similar demographics"
- Combined ranking ensures both visual and clinical relevance
- Multi-hop: starting from a tumor, traverse treatment relationships, then find visually similar tumors at the endpoints

---

## CATEGORY 3: NODE.JS EXAMPLES (`examples/nodejs/`)

### 3.1 basic_usage.js
**Purpose**: Getting started with the JavaScript SDK.
**APIs Demonstrated**:
```javascript
const { VectorDB } = require('ruvector');
const db = new VectorDB({
    dimensions: 128,
    storagePath: './examples_basic_node.db',
    distanceMetric: 'cosine'
});
await db.insert({ id: 'doc_001', vector: vector, metadata: { text: 'Example' } });
await db.insertBatch(entries);
await db.search({ vector: queryVector, k: 5, includeMetadata: true });
db.count();
```

**Medical Imaging Application**: Build a web-based tumor search interface. Frontend sends embeddings from MedGemma, Node.js backend queries RuVector for similar cases, returns results with metadata (patient age, tumor grade, treatment outcome).

---

### 3.2 semantic_search.js
**Purpose**: Full semantic search system with HNSW configuration and filtered search.
**APIs Demonstrated**:
- HNSW configuration: `{ m: 32, efConstruction: 200, efSearch: 100 }`
- Document indexing with category metadata
- Multi-query semantic search
- Filtered search by metadata field: `filter: { category: 'technology' }`
- Similarity scoring: `1 - result.distance`

**Key Patterns**:
```javascript
const db = new VectorDB({
    dimensions: 384,
    distanceMetric: 'cosine',
    hnsw: { m: 32, efConstruction: 200, efSearch: 100 }
});

// Filtered search
const results = await db.search({
    vector: queryEmbedding,
    k: 3,
    filter: { category: 'technology' },
    includeMetadata: true
});
```

**Medical Imaging Application**: Build a filtered medical image search:
- Filter by tumor type: `filter: { tumor_type: 'glioblastoma' }`
- Filter by body region: `filter: { region: 'brain' }`
- Filter by scan modality: `filter: { modality: 'MRI_T2' }`
- HNSW parameters can be tuned for medical dataset sizes

---

## CATEGORY 4: ONNX EMBEDDINGS (`examples/onnx-embeddings/`)

### Full Embedding Library
**Purpose**: Production-ready ONNX-based embedding generation in pure Rust.
**Key Features**:
- 8 pretrained models: AllMiniLmL6V2 (fastest), AllMpnetBaseV2 (highest quality), E5/BGE variants
- GPU acceleration: CUDA, TensorRT, CoreML
- 6 pooling strategies: Mean, CLS, Max, MeanSqrtLen, LastToken, WeightedMean
- Batch processing with configurable parallelism
- Performance: 188-5,120 embeddings/second depending on hardware

**Key APIs**:
```rust
let embedder = EmbedderBuilder::new()
    .model(Model::AllMiniLmL6V2)
    .batch_size(32)
    .max_length(512)
    .pooling(PoolingStrategy::Mean)
    .normalize(true)
    .build()?;

let embedding = embedder.embed("text")?;
let batch = embedder.embed_batch(&["text1", "text2"])?;
let similarity = embedder.cosine_similarity(&emb1, &emb2)?;
```

**Medical Imaging Application**: While this processes text, it is essential for:
- Embedding radiology reports to pair with image embeddings
- Building a multimodal search (text report + image embedding)
- Processing clinical notes for the RAG pipeline
- Cross-referencing MedGemma image embeddings with text-based clinical descriptions

---

## CATEGORY 5: WASM EXAMPLES

### 5.1 wasm-react/ (App.jsx)
**Purpose**: React application with WASM vector database, Web Workers, and IndexedDB persistence.
**APIs Demonstrated**:
- `WorkerPool` -- multi-threaded WASM execution via Web Workers
- `IndexedDBPersistence` -- browser-local vector storage
- `pool.insertBatch(entries)` -- parallel vector insertion
- `pool.search(query, k, filter)` -- similarity search from browser
- Performance benchmarking (ops/sec measurement)
- Save/load to IndexedDB for offline capability

**Medical Imaging Application**: Build a browser-based tumor analysis tool:
- Clinicians upload scans directly in the browser
- WASM processes MedGemma embeddings client-side (no server round-trip)
- IndexedDB persists the local tumor database for offline use
- Web Workers ensure the UI remains responsive during heavy computation
- Useful for field clinics or low-connectivity environments

### 5.2 wasm-vanilla/ (index.html)
**Purpose**: Minimal vanilla HTML/JS WASM integration without frameworks.

### 5.3 wasm/ios/
**Purpose**: iOS WebAssembly deployment.

---

## CATEGORY 6: EDGE COMPUTING EXAMPLES

### 6.1 edge/
**Purpose**: Lightweight WASM framework for distributed AI agent swarms on edge devices.
**Key Specs**:
- 364KB WASM binary
- 150x faster vector search than brute force
- 50,000 signing operations/second
- Ed25519 + AES-256-GCM encryption
- Raft consensus + CRDT for distributed coordination
- Spiking neural networks
- Post-quantum hybrid cryptography

**Medical Imaging Application**: Deploy tumor analysis directly on hospital edge devices (radiology workstations). Each workstation maintains a local vector index synchronized with the hospital network via Raft consensus. Enables real-time tumor comparison without cloud dependency, with cryptographic security for PHI/HIPAA compliance.

### 6.2 edge-net/
**Purpose**: Collective AI computing network -- browser-based resource sharing for AI workloads.
**Key Features**:
- MicroLoRA adapters for task specialization (2,236+ ops/sec)
- SONA self-optimizing neural architecture (instant/background/deep learning loops)
- Federated learning with 90% gradient compression
- ReasoningBank for storing successful diagnostic patterns
- 4 attention mechanisms: neural, DAG, graph, state-space

**Medical Imaging Application**: Create a federated tumor analysis network where:
- Multiple hospitals contribute compute resources
- Federated learning trains shared tumor classification models without sharing patient data
- MicroLoRA specializes models per tumor type on each site
- ReasoningBank stores and retrieves successful diagnostic patterns across the network
- SONA continuously improves the system as more cases are processed

### 6.3 edge-full/
**Purpose**: Complete edge deployment package with compiled WASM.

---

## CATEGORY 7: SPECIALIZED ML EXAMPLES

### 7.1 spiking-network/
**Purpose**: Rust spiking neural network crate with SIMD optimization.
**Dependencies**: ruvector-core, ruvector-gnn, ndarray, rayon (parallelism), bitvec (spike encoding)
**Features**: SIMD optimization, WASM compatibility, visualization

**Medical Imaging Application**: Spiking neural networks are energy-efficient and excel at temporal pattern recognition. For tumor tracking:
- Encode tumor progression as spike trains (growth rate changes as spikes)
- Temporal pattern detection for identifying rapidly growing tumors
- Energy-efficient processing for continuous monitoring systems

### 7.2 meta-cognition-spiking-neural-network/
**Purpose**: JavaScript-based demos combining SNN with attention mechanisms and cognitive exploration.
**Key Demos**:
- `attention/all-mechanisms.js` -- all attention mechanism demonstrations
- `attention/hyperbolic-deep-dive.js` -- Poincare disk attention
- `snn/examples/pattern-recognition.js` -- SNN pattern recognition
- `snn/examples/benchmark.js` -- SNN performance benchmarks
- `vector-search/semantic-search.js` -- semantic search with SNN
- `optimization/simd-optimized-ops.js` -- SIMD performance optimization
- `self-discovery/cognitive-explorer.js` -- self-aware system exploration

**Medical Imaging Application**: Combine SNN with hyperbolic attention for hierarchical tumor classification. Hyperbolic space naturally represents hierarchies (organ > region > tissue type > tumor grade), making it ideal for organizing medical taxonomies.

### 7.3 scipix/
**Purpose**: Rust OCR engine for scientific documents and math equations.
**Structure**: Full crate with src/, tests/, benches/, WASM support.

**Medical Imaging Application**: Extract structured data from radiology reports, medical literature, and clinical trial documents. Parse measurement values, tumor dimensions, and grading criteria from scanned reports for automated database population.

### 7.4 neural-trader/
**Purpose**: Comprehensive financial neural trading system.
**Structure**: 15 subdirectories covering accounting, risk, portfolio, strategies, neural models, MCP integration, production deployment, and testing.

**Medical Imaging Application**: The risk modeling and portfolio management patterns apply to treatment planning:
- Risk assessment for treatment options (analogous to portfolio risk)
- Strategy selection based on historical outcomes
- Neural prediction models for treatment response
- MCP integration pattern for connecting with clinical systems

---

## CATEGORY 8: MATHEMATICAL FRAMEWORK EXAMPLES

### 8.1 prime-radiant/
**Purpose**: Universal coherence engine using sheaf Laplacian mathematics.
**Key Concepts**:
- Global incoherence measurement: `E(S) = sum(w_e * ||r_e||^2)`
- Six mathematical directions: sheaf cohomology, category theory, homotopy type theory, spectral invariants, causal abstraction, quantum topology
- Compute ladder: Reflex (<1ms), Retrieval (~10ms), Heavy (~100ms), Human review
- Six domains: AI agents, finance, medical, robotics, security, science

**Medical Imaging Application**: DIRECTLY APPLICABLE to medical coherence checking:
- Detect when a diagnosis is inconsistent with imaging features (hallucination detection)
- Verify clinical reasoning chains are logically coherent
- Flag when treatment plans contradict imaging evidence
- The "medical clinical disagreement" domain is explicitly supported
- Tiered compute: instant alerts for critical inconsistencies, deeper analysis for complex cases

### 8.2 delta-behavior/
**Purpose**: Coherence-first state management -- "change is permitted but collapse is not."
**Key Features**:
- Three-layer enforcement: energy costs, scheduling constraints, memory gating
- DeltaEnforcer with Allowed/Throttled/Blocked states
- 11 exotic applications including self-limiting reasoning and artificial homeostasis
- O(n) algorithms with SIMD acceleration
- WASM + TypeScript SDK

**Key API**:
```rust
let enforcer = DeltaEnforcer::new(config);
match enforcer.check(current, predicted) {
    EnforcementResult::Allowed => { /* proceed */ }
    EnforcementResult::Throttled(delay) => { /* wait */ }
    EnforcementResult::Blocked(reason) => { /* reject */ }
}
```

**Medical Imaging Application**: Safety guardrails for AI diagnostic systems:
- Prevent catastrophic misdiagnosis by blocking incoherent state transitions
- Throttle rapid changes in tumor classification (require human review)
- Memory gating ensures the system maintains consistent patient records
- Self-limiting reasoning: AI scales diagnostic confidence with evidence strength
- Artificial homeostasis: system self-monitors for diagnostic drift

### 8.3 mincut/
**Purpose**: Self-organizing network examples using minimum cut analysis.
**Sub-examples** (10 directories):
1. `temporal_attractors/` -- networks evolving toward stability
2. `strange_loop/` -- self-aware, self-repairing swarm systems
3. `causal_discovery/` -- automatic root cause analysis
4. `time_crystal/` -- self-sustaining periodic coordination
5. `morphogenetic/` -- bio-inspired organic network growth
6. `neural_optimizer/` -- learned graph optimization
7. `snn_integration/` -- spiking neural network integration
8. `temporal_hypergraph/` -- time-aware graph structures
9. `federated_loops/` -- distributed coordination patterns
10. `benchmarks/` -- performance measurements

**Performance**: ~50 microsecond updates for 1,000-node networks (20,000+ recalculations/sec).

**Medical Imaging Application**:
- **Causal discovery**: Identify root causes of treatment failures from graph analysis
- **Temporal hypergraphs**: Track tumor network evolution over time
- **Neural optimizer**: Optimize the tumor knowledge graph structure
- **Federated loops**: Coordinate multi-site tumor databases
- **Morphogenetic networks**: Model tumor growth patterns as self-organizing networks
- Min-cut analysis identifies critical weaknesses in diagnostic pathways

### 8.4 subpolynomial-time/
**Purpose**: Subpolynomial-time minimum cut algorithm -- December 2025 breakthrough.
**Key Feature**: Maintains network state for instant queries and rapid updates.

**Medical Imaging Application**: Analyze connectivity and vulnerability in medical knowledge graphs at scale with sub-polynomial time complexity.

---

## CATEGORY 9: AI PLATFORM EXAMPLES

### 9.1 exo-ai-2025/
**Purpose**: Research platform for computational consciousness, memory, and cognition.
**Scale**: 9 interconnected Rust crates, ~15,800+ lines of research-grade code.
**Crates**:
1. exo-core -- Integrated Information Theory (IIT) consciousness measurement
2. exo-temporal -- Memory with causal tracking and consolidation
3. exo-hypergraph -- Topological analysis using persistent homology
4. exo-manifold -- SIREN neural networks + SIMD retrieval (8-54x speedup)
5. exo-exotic -- 10 cognitive experiments (dreams, free energy, emergence)
6. exo-federation -- Post-quantum distributed cognitive systems
7. exo-backend-classical -- SIMD compute (AVX2/AVX-512/NEON)
8. exo-wasm -- Browser/edge deployment
9. exo-node -- JavaScript bindings

**Medical Imaging Application**:
- **Temporal memory**: Track tumor evolution with causal tracking -- why did this tumor respond to treatment?
- **Topological analysis**: Persistent homology can detect shape features in tumor morphology that are invariant to scale
- **SIREN neural networks**: Continuous implicit representations of tumor shapes for precise volume tracking
- **SIMD-accelerated retrieval**: 54x speedup for comparing tumor embeddings at scale
- **Cognitive architecture**: Build a self-reflective diagnostic system that monitors its own reasoning quality

### 9.2 agentic-jujutsu/
**Purpose**: Quantum-resistant, self-learning version control for AI agents.
**Files**:
- `basic-usage.ts` -- Repository init, commits, branches, diffs
- `learning-workflow.ts` -- Learning trajectory management
- `multi-agent-coordination.ts` -- Concurrent multi-agent workflows
- `quantum-security.ts` -- Post-quantum cryptographic operations

**Key Patterns**:
```typescript
// Multi-agent coordination
const backend = new JjWrapper();
backend.startTrajectory("Implement REST API");
await backend.branchCreate("feature/api");
await backend.newCommit("Add API endpoints");
backend.addToTrajectory();
backend.finalizeTrajectory(0.9, "API complete");

// AI suggestions based on learned patterns
const suggestion = JSON.parse(tester.getSuggestion("Test API and UI"));
```

**Medical Imaging Application**: Manage multiple AI diagnostic agents working concurrently:
- One agent analyzes MRI sequences, another processes CT scans, a third reviews prior reports
- Each agent maintains its own trajectory with conflict-free coordination
- Shared learning: insights from one modality inform the others
- Version control for diagnostic models and feature extraction pipelines

### 9.3 ruvLLM/
**Purpose**: Local LLM inference engine, including ESP32 embedded support.
**Structure**: Rust crate with modules for plans/specs, ESP32 flash deployment, benchmarks.

**Medical Imaging Application**: Run MedGemma-style inference locally on hospital hardware without cloud dependencies. ESP32 support suggests ultra-edge deployment possibilities (embedded medical devices).

---

## CATEGORY 10: DOMAIN APPLICATION EXAMPLES

### 10.1 vibecast-7sense/
**Purpose**: Bioacoustic intelligence platform -- bird song analysis through vector embeddings.
**Architecture**: 9 Rust crates handling audio ingestion, ONNX embedding generation, HNSW vector indexing, GNN learning, acoustic clustering, and API serving.
**Performance**: <50ms query latency, 150x search speedup, >100 segments/sec, <6GB per 1M vectors.

**Medical Imaging Application**: DIRECT ARCHITECTURAL TEMPLATE for MedGemma:

| 7sense Component | Medical Equivalent |
|---|---|
| sevensense-audio (WAV/MP3 ingestion) | medgemma-imaging (DICOM/NIfTI ingestion) |
| sevensense-embedding (ONNX Perch model) | medgemma-embedding (ONNX MedGemma model) |
| sevensense-vector (HNSW indexing) | medgemma-vector (HNSW tumor indexing) |
| sevensense-learning (GNN online learning) | medgemma-learning (GNN tumor pattern learning) |
| sevensense-analysis (acoustic clustering) | medgemma-analysis (tumor clustering) |
| sevensense-interpretation (confidence scoring) | medgemma-interpretation (diagnostic confidence) |
| sevensense-api (GraphQL/REST/WebSocket) | medgemma-api (FHIR/REST/WebSocket) |

This is the single best architectural reference for the MedGemma project.

### 10.2 google-cloud/
**Purpose**: GPU-accelerated deployment on Google Cloud Run with NVIDIA L4 GPUs.
**Capabilities**:
- Single-node and multi-node benchmarking
- Attention/GNN inference with 16GB memory
- Raft consensus clustering (3+ instances)
- Primary-replica replication (async/sync)
- INT8 and product quantization compression
- SIMD auto-detection (AVX-512, AVX2, NEON)

**Medical Imaging Application**: Production deployment template for the MedGemma system:
- GPU-accelerated embedding generation for real-time tumor analysis
- Multi-node deployment for hospital-wide or multi-hospital installations
- Raft consensus for consistent diagnostic state across servers
- Quantization for efficient storage of millions of embeddings
- Cloud Run serverless scaling: auto-scale during peak radiology hours

### 10.3 vwm-viewer/
**Purpose**: Vector Working Memory viewer -- HTML canvas visualization.
**Files**: Dockerfile, canvas-viewer.html, football.html, nginx.conf

**Medical Imaging Application**: Template for building a web-based tumor visualization dashboard.

---

## CATEGORY 11: STANDALONE EXAMPLE

### bounded_instance_demo.rs
**Purpose**: Demonstrate BoundedInstance with DeterministicLocalKCut for dynamic minimum cut queries.
**APIs Demonstrated**:
```rust
use ruvector_mincut::prelude::*;
let graph = DynamicGraph::new();
let mut instance = BoundedInstance::init(&graph, 1, 5);  // bounds [1, 5]
instance.apply_inserts(&[(0, 0, 1), (1, 1, 2)]);  // add edges
instance.apply_deletes(&[(1, 1, 2)]);  // remove edges
match instance.query() {
    InstanceResult::ValueInRange { value, witness } => { ... }
    InstanceResult::AboveRange => { ... }
}
let cert = instance.certificate();
```

**Medical Imaging Application**: Analyze network resilience in medical knowledge graphs. Identify single points of failure in diagnostic pathways -- if one data source becomes unavailable, does the diagnostic graph remain connected?

---

## CATEGORY 12: TEST FILES (Usage Patterns)

### tests/graph_integration.rs
**Complete working test suite** demonstrating:
- Full workflow: create nodes with labels/properties, create edges, verify
- Social network scenario (10 users, follow chains)
- Movie database scenario (movies, actors, ACTED_IN relationships)
- Knowledge graph scenario (ML/AI/DL/NN concept hierarchy with IS_A/USES relationships)
- Batch import (100 nodes)
- Graph transformation (add reverse edges for bidirectional traversal)
- Error handling (missing nodes, duplicate IDs)
- Data integrity (referential integrity checks)
- Performance test (1000 nodes, 5000 edges)

**Key Working API**:
```rust
use ruvector_graph::{GraphDB, Node, Edge, Label, RelationType, Properties, PropertyValue};
let db = GraphDB::new();
db.create_node(Node::new("id", vec![Label { name: "Person" }], props)).unwrap();
db.create_edge(Edge::new("eid", "from", "to", RelationType { name: "KNOWS" }, props)).unwrap();
let node = db.get_node("id");
let edge = db.get_edge("eid");
```

### tests/test_agenticdb.rs
**Complete test suite** for AgenticDB:
- Reflexion memory: episode storage/retrieval with semantic similarity
- Skill library: skill creation, searching, auto-consolidation
- Causal memory: hypergraph relationships with utility scoring
- Learning sessions: reinforcement learning algorithm support
- Integration: cross-table queries, persistence, concurrent operations

### tests/advanced_tests.rs
Integration tests for hypergraph, temporal, causal memory, learned index, neural hash, and topological features.

---

## CATEGORY 13: BENCHMARK FILES (Performance Patterns)

### benches/attention_latency.rs
**Benchmarks 5 attention mechanisms at 100 tokens**:
1. Multi-head attention (O(n^2))
2. Mamba SSM (O(n) selective scan)
3. RWKV attention (linear attention)
4. Flash attention approximation (tiled computation)
5. Hyperbolic attention (Poincare distance-weighted)

Plus scaling benchmarks (32-512 sequence lengths) and memory efficiency comparison (standard O(n^2) vs. memory-efficient O(n)).

### benches/neuromorphic_benchmarks.rs
**Benchmarks 6 neuromorphic components**:
1. HDC operations: bundle, bind, permute, similarity (1K-10K dimensions)
2. HDC encoding: continuous value to hypervector encoding
3. BTSP: forward pass, eligibility trace update, behavioral weight update
4. Spiking neurons: LIF neuron simulation, network step (100-1000 neurons)
5. STDP: spike-timing dependent plasticity weight updates (10K synapses)
6. Reservoir computing: 500-neuron reservoir with 100-step sequences

### benches/learning_performance.rs
Learning system performance benchmarks.

### benches/plaid_performance.rs
PLAID (Pre-computed Late Interaction over BERT) performance benchmarks.

---

## SUMMARY: TOP PATTERNS FOR MEDGEMMA TUMOR TRACKING

### Tier 1: Directly Applicable (use immediately)
1. **vibecast-7sense architecture** -- Copy the 9-crate modular architecture, replacing audio with DICOM imaging
2. **gnn_example.rs** -- GNN for tumor graph with neighbor aggregation
3. **hybrid_search.rs** -- Combined vector + graph search for clinical context
4. **rag_pipeline.rs** -- RAG for retrieving similar cases and clinical guidelines
5. **agenticdb_demo.rs** -- Agent memory for persistent diagnostic learning

### Tier 2: High Value (adapt for medical domain)
6. **advanced_features.rs** -- Temporal hypergraphs for tumor progression tracking
7. **prime-radiant** -- Coherence checking for diagnostic consistency
8. **delta-behavior** -- Safety guardrails preventing catastrophic misdiagnosis
9. **edge/edge-net** -- Federated learning across hospital sites
10. **google-cloud** -- Production GPU deployment template

### Tier 3: Specialized Enhancement
11. **onnx-embeddings** -- Text embedding for radiology reports
12. **spiking-network** -- Temporal pattern detection in tumor growth
13. **exo-ai-2025** -- Topological analysis of tumor morphology
14. **mincut/causal_discovery** -- Root cause analysis for treatment failures
15. **wasm-react** -- Browser-based clinical tool with offline capability
16. **batch_operations.rs** -- High-throughput ingestion of historical scan archives
17. **distributed_cluster.rs** -- Multi-hospital Raft consensus deployment
18. **refrag-pipeline** -- 30x RAG latency reduction for real-time diagnosis
