# Algorithm Analysis Report

This document provides a detailed technical analysis of the algorithms implemented in the HNSW-Flash repository. The focus is on the novelty of the Flash optimization techniques and guidance on which parts of the code are most important for understanding the implementation.

## Table of Contents

1. [Overview](#overview)
2. [Core Algorithms](#core-algorithms)
   - [HNSW (Hierarchical Navigable Small World)](#hnsw-hierarchical-navigable-small-world)
   - [NSG (Navigating Spreading-out Graph)](#nsg-navigating-spreading-out-graph)
   - [Tau-MG (Tau-Monotonic Graph)](#tau-mg-tau-monotonic-graph)
3. [Vector Quantization Techniques](#vector-quantization-techniques)
   - [Product Quantization (PQ)](#product-quantization-pq)
   - [Scalar Quantization (SQ)](#scalar-quantization-sq)
   - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
4. [Distance Computation Methods](#distance-computation-methods)
   - [SDC (Symmetric Distance Computation)](#sdc-symmetric-distance-computation)
   - [ADC (Asymmetric Distance Computation)](#adc-asymmetric-distance-computation)
5. [Flash Optimization Techniques (Novel Contributions)](#flash-optimization-techniques-novel-contributions)
   - [PQLINK_STORE: Neighbor Vector Storage](#pqlink_store-neighbor-vector-storage)
   - [PQLINK_CALC: SIMD-Optimized Batch Distance Computation](#pqlink_calc-simd-optimized-batch-distance-computation)
   - [4-bit Packed Encoding](#4-bit-packed-encoding)
   - [Memory-Aligned Data Layout](#memory-aligned-data-layout)
6. [Additional Optimizations](#additional-optimizations)
   - [ADSampling (Adaptive Dimensional Sampling)](#adsampling-adaptive-dimensional-sampling)
   - [VBase Window-based Early Termination](#vbase-window-based-early-termination)
   - [Reranking Strategy](#reranking-strategy)
7. [Key Code Sections for Learning](#key-code-sections-for-learning)
8. [Summary](#summary)

---

## Overview

HNSW-Flash is a high-performance implementation for Approximate Nearest Neighbor Search (ANNS) that builds upon the standard HNSW algorithm with novel optimizations designed for modern CPUs. The key innovation is the **Flash** technique, which combines:

1. **Product Quantization** for vector compression
2. **SIMD-optimized batch distance computation** for parallel processing
3. **Cache-friendly memory layouts** for reduced memory access latency
4. **4-bit packed encoding** for efficient storage and processing

The repository implements multiple strategies that can be selected based on the use case:

| Strategy | Description |
|----------|-------------|
| `flash` | Full Flash optimization with PQ, SIMD, and neighbor data storage |
| `hnsw` | Standard HNSW baseline |
| `nsg` | Navigating Spreading-out Graph |
| `nsg-flash` | NSG with Flash optimizations |
| `pq-adc` | HNSW with Product Quantization (Asymmetric Distance Computation) |
| `pq-sdc` | HNSW with Product Quantization (Symmetric Distance Computation) |
| `sq-adc` | HNSW with Scalar Quantization (Asymmetric Distance Computation) |
| `sq-sdc` | HNSW with Scalar Quantization (Symmetric Distance Computation) |
| `pca-sdc` | HNSW with PCA dimensionality reduction |
| `taumg` | Tau-Monotonic Graph |
| `taumg-flash` | Tau-MG with Flash optimizations |

---

## Core Algorithms

### HNSW (Hierarchical Navigable Small World)

**Location:** `include/strategy/hnsw_strategy.h`, `third_party/hnswlib/hnswalg.h`

HNSW is a graph-based approximate nearest neighbor search algorithm that organizes data points in a multi-layer graph structure. Each layer is a proximity graph where nodes are connected to their approximate nearest neighbors.

**Key Characteristics:**
- **Hierarchical Structure:** Higher layers contain fewer nodes, enabling fast navigation; the base layer (level 0) contains all nodes
- **Small World Property:** Long-range connections in upper layers enable O(log N) search complexity
- **Parameters:**
  - `M`: Maximum number of connections per node (default: 32)
  - `ef_construction`: Size of dynamic candidate list during index construction (default: 1024)
  - `ef_search`: Size of dynamic candidate list during search (default: 64)

**Algorithm Flow:**
1. **Index Construction:** For each new point, search for nearest neighbors starting from entry point, then connect to M nearest neighbors using a heuristic
2. **Search:** Start from entry point, greedily descend through layers, perform beam search at base layer

**Code Focus Areas:**
- `addPoint()` function: Shows how nodes are inserted into the graph
- `searchBaseLayerST()` function: Core search logic with priority queue management
- `getNeighborsByHeuristic2()` function: Neighbor selection heuristic

### NSG (Navigating Spreading-out Graph)

**Location:** `include/strategy/nsg_strategy.h`, `third_party/hnswlib/nsg.h`

NSG is a graph-based index that aims to build a more structured graph by using a navigating-spreading construction method.

**Key Characteristics:**
- **Single-Layer Graph:** Unlike HNSW, NSG uses a single-layer graph structure
- **Centroid Navigation:** Uses the dataset centroid as a navigation starting point
- **Edge Pruning:** Applies pruning to ensure the graph has good navigability properties

**Algorithm Flow:**
1. Build initial NNG (Nearest Neighbor Graph)
2. Compute dataset centroid
3. Apply NSG pruning to optimize graph structure

**Code Focus Areas:**
- `nsgPrune()` function: The pruning algorithm that differentiates NSG from basic NNG
- `setCentroids()` function: Centroid computation for navigation entry point

### Tau-MG (Tau-Monotonic Graph)

**Location:** `include/strategy/taumg_strategy.h`, `third_party/hnswlib/taumg.h`

Tau-MG introduces monotonicity constraints to ensure each step during search always makes progress toward the query.

**Key Characteristics:**
- **Monotonicity Parameter (tau):** Controls the trade-off between graph quality and construction time
- **Improved Convergence:** The monotonicity constraint helps avoid local minima during search

**Code Focus Areas:**
- `taumgPrune()` function: Implements the tau-monotonic pruning criterion
- `TAU` constant in `core.h`: The tau parameter (default: 8)

---

## Vector Quantization Techniques

### Product Quantization (PQ)

**Location:** `include/strategy/pq_sdc_strategy.h`, `include/strategy/flash_strategy.h`

Product Quantization divides vectors into subvectors and quantizes each subspace independently.

**Key Characteristics:**
- **Subvector Division:** Original D-dimensional vectors are split into M subvectors
- **Codebook Learning:** K-means clustering generates K centroids for each subspace
- **Compact Encoding:** Each subvector is represented by its nearest centroid index

**Parameters:**
- `SUBVECTOR_NUM`: Number of subvectors (default: 16)
- `CLUSTER_NUM`: Number of clusters per subspace (default: 16, enabling 4-bit encoding)
- `MAX_ITERATIONS`: K-means iterations (default: 12)
- `SAMPLE_NUM`: Sample size for codebook training (default: 100,000)

**Code Focus Areas:**
```cpp
// include/strategy/pq_sdc_strategy.h
void generate_codebooks(size_t sample_num);  // Codebook generation via k-means
std::vector<data_t> pqEncode(const std::vector<float>& vec);  // Vector encoding

// include/strategy/flash_strategy.h
void pqEncode(float *data, uint8_t *encoded_vector, data_t *dist_table, int is_query = 1);
```

### Scalar Quantization (SQ)

**Location:** `include/strategy/sq_sdc_strategy.h`, `include/space/space_sq.h`

Scalar Quantization maps each dimension independently to a fixed range.

**Key Characteristics:**
- **Per-Dimension Quantization:** Each dimension is scaled to [0, max_data_t]
- **Simpler Codebooks:** No clustering required, just min/max statistics
- **Lower Precision:** Uses INT8/INT16/INT32 instead of float32

**Code Focus Areas:**
```cpp
// include/space/space_sq.h
static std::vector<data_t> sqEncode(const std::vector<float>& vec);  // Encoding
static std::vector<float> sqDecode(const data_t* vec, size_t dim);   // Decoding
void train_quantizer(const std::vector<std::vector<float>>& data_set); // Parameter learning
```

### Principal Component Analysis (PCA)

**Location:** `include/strategy/pca_sdc_strategy.h`, `include/strategy/flash_strategy.h`

PCA reduces dimensionality while preserving maximum variance.

**Key Characteristics:**
- **Dimensionality Reduction:** Reduces from original dimension to `PRINCIPAL_DIM` (default: 32)
- **Variance-Based Selection:** Eigenvectors with largest eigenvalues are retained
- **Optional Optimal Grouping:** `USE_PCA_OPTIMAL` flag enables variance-based subvector length allocation

**Code Focus Areas:**
```cpp
// include/strategy/flash_strategy.h
void generate_matrix(std::vector<std::vector<float>>& data_set, size_t sample_num);
void pcaEncode(std::vector<std::vector<float>>& data_set);
```

---

## Distance Computation Methods

### SDC (Symmetric Distance Computation)

**Concept:** Both query and database vectors are quantized; distance is computed between quantized representations.

**Characteristics:**
- **Same Treatment:** Query and data points undergo identical encoding
- **Precomputed Tables:** Distance between cluster centroids can be precomputed
- **Lower Accuracy:** Both vectors introduce quantization error

**Implementation:**
```cpp
// include/space/space_pq.h
static float PqSdcL2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
```

### ADC (Asymmetric Distance Computation)

**Concept:** Only database vectors are quantized; the raw query is compared against quantized data.

**Characteristics:**
- **Query Preserved:** Full precision query vector
- **On-the-fly Computation:** Query-to-centroid distances computed per query
- **Higher Accuracy:** Only one source of quantization error

**Implementation:**
```cpp
// include/space/space_pq.h
static float PqAdcL2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
```

---

## Flash Optimization Techniques (Novel Contributions)

The Flash techniques represent the core novelty of this work. They are designed to maximize throughput on modern CPUs through careful memory layout and SIMD vectorization.

### PQLINK_STORE: Neighbor Vector Storage

**Location:** `third_party/hnswlib/hnswalg_flash.h` (lines 154-167, 267-273)

**Novelty:** Instead of storing only neighbor IDs in the adjacency list, Flash stores the encoded vectors of all neighbors directly alongside the links.

**Benefit:**
- **Eliminates Random Memory Access:** No need to fetch neighbor data from separate memory locations
- **Cache-Friendly:** Neighbor data is co-located with graph structure
- **Enables Batch Processing:** All neighbor vectors available contiguously

**Memory Layout:**
```
Level 0: [link_count][neighbor_id_0]...[neighbor_id_M0][neighbor_vec_0]...[neighbor_vec_M0]
Level i: [link_count][neighbor_id_0]...[neighbor_id_M][neighbor_vec_0]...[neighbor_vec_M]
```

**Code Focus:**
```cpp
// Memory layout calculation
#if defined(PQLINK_STORE)
    size_links_level0_ = maxM0_ * sizeof(tableint) + maxM0_ * data_size_ + sizeof(linklistsizeint);
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
    offsetLinklistData0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
#endif

// Setting neighbor data
inline void setLinksData(const void *data, size_t idx, tableint neighbor_id, int level) const;
```

### PQLINK_CALC: SIMD-Optimized Batch Distance Computation

**Location:** `third_party/hnswlib/hnswalg_flash.h` (lines 1766-1906)

**Novelty:** The `PqLinkL2Sqr` function computes distances to multiple neighbors simultaneously using SIMD instructions, exploiting the PSHUFB (Packed Shuffle Bytes) instruction for efficient table lookups.

**Key Innovation - Data Rearrangement:**
The encoded neighbor data is rearranged to enable efficient SIMD processing:

```text
Standard Layout (per neighbor):
[subvec_0|subvec_1][subvec_2|subvec_3]...[subvec_14|subvec_15]

Flash Layout (16 neighbors interleaved):
Block 0: [n0_sub01][n1_sub01]...[n15_sub01]
Block 1: [n0_sub23][n1_sub23]...[n15_sub23]
...
```

**SIMD Implementation (AVX2):**
```cpp
const __m256i low_mask = _mm256_set1_epi8(0x0F);
const __m256i* qdists = reinterpret_cast<const __m256i*>(pVect1v);
const __m256i* part = reinterpret_cast<const __m256i*>(pVect2v);

for (size_t j = 0; j < tmp; j += 2) {
    comps = _mm256_loadu_si256(part);
    twolane_sum = _mm256_add_epi8(twolane_sum, _mm256_add_epi8(
        _mm256_shuffle_epi8(qdists[j | 1], _mm256_and_si256(comps, low_mask)),
        _mm256_shuffle_epi8(qdists[j], _mm256_and_si256(_mm256_srli_epi64(comps, 4), low_mask))
    ));
    ++part;
}
```

**Why PSHUFB Works:**
- With 16 clusters, each subvector index is 4 bits (0-15)
- PSHUFB shuffles bytes within 128-bit lanes based on index
- Distance table has 16 entries fitting in 128 bits (16 × 8-bit = 128 bits)
- Single PSHUFB instruction performs 16 table lookups simultaneously

### 4-bit Packed Encoding

**Location:** `include/strategy/flash_strategy.h` (lines 419-429, 454-464)

**Novelty:** With `CLUSTER_NUM = 16`, two subvector indices can be packed into a single byte, halving storage requirements.

```cpp
if (CLUSTER_NUM == 16) {
    size_t index = (i / (BATCH << 1) * BATCH) + i % BATCH;
    if (i % (BATCH << 1) >= BATCH) {
        encoded_vector[index] = (encoded_vector[index] & 0xF0) | best_index;
    } else {
        encoded_vector[index] = (encoded_vector[index] & 0x0F) | (best_index << 4);
    }
}
```

**Special Index Rearrangement for SIMD:**
The subvector order is rearranged (e.g., [0, 2, 1, 3] for AVX) to match the SIMD shuffle constraints where shuffling only works within 128-bit lanes.

### Memory-Aligned Data Layout

**Location:** `include/strategy/flash_strategy.h` (lines 38-44)

**Novelty:** Distance tables and encoded vectors are aligned to SIMD register widths.

```cpp
#if defined(RUN_WITH_AVX)
    thread_encoded_vector[i] = (uint8_t *)aligned_alloc(32, ...);  // 256-bit alignment
#elif defined(RUN_WITH_AVX512)
    thread_encoded_vector[i] = (uint8_t *)aligned_alloc(64, ...);  // 512-bit alignment
#endif
```

---

## Additional Optimizations

### ADSampling (Adaptive Dimensional Sampling)

**Location:** `third_party/adsampling.h`, `third_party/hnswlib/hnswalg_flash.h`

**Concept:** Instead of computing full distance, progressively sample dimensions and use statistical hypothesis testing to early-reject non-promising candidates.

**Key Equation:**
```cpp
// Check if candidate can be rejected
if (res >= dis * ratio(D, i)) {
    return -res * D / i;  // Negative indicates rejection
}

// Ratio function
inline float ratio(const int &D, const int &i) {
    return 1.0 * i / D * (1.0 + epsilon0 / std::sqrt(i)) * (1.0 + epsilon0 / std::sqrt(i));
}
```

**Parameters:**
- `ADSAMPLING_EPSILON`: Confidence parameter (default: 2.1)
- `ADSAMPLING_DELTA_D`: Sampling interval (default: 4 dimensions)

### VBase Window-based Early Termination

**Location:** `third_party/hnswlib/hnswalg_flash.h` (lines 463-507, 549-559)

**Concept:** Track recent candidate distances in a sliding window and use the median as a robust estimate for early termination.

**Implementation:**
```cpp
#ifdef VBASE
std::queue<dist_t> pace;
std::multiset<dist_t, std::greater<dist_t>> max_heap;
std::multiset<dist_t, std::less<dist_t>> min_heap;
// Median calculation using two heaps
#endif
```

### Reranking Strategy

**Location:** All strategy files with `#if defined(RERANK)` blocks

**Concept:** Search for K×100 candidates using approximate distance, then rerank using exact L2 distance on original vectors.

```cpp
#if defined(RERANK)
std::priority_queue<...> tmp = hnsw->searchKnn(encoded_query, K * 100);
// Recompute exact distances
while (!tmp.empty()) {
    float res = 0;
    for (int j = 0; j < ori_dim; ++j) {
        float t = org_data_set_[a][j] - org_query_set_[i][j];
        res += t * t;
    }
    result.emplace(res, a);
    tmp.pop();
}
#endif
```

---

## Key Code Sections for Learning

For someone learning the codebase, here are the most important sections to study in order:

### 1. Configuration and Constants (`include/core.h`)
- **Lines 29-67:** All tunable parameters with their meanings
- **Lines 69-85:** Data type definitions based on compile flags
- **Lines 87-94:** SIMD batch size configuration

### 2. Base Strategy Pattern (`include/strategy/solve_strategy.h`)
- **Lines 7-25:** Constructor and data loading
- **Line 26:** Virtual `solve()` method that each strategy implements
- **Lines 37-71:** `recall()` function for evaluation

### 3. PQ Encoding Process (`include/strategy/flash_strategy.h`)
- **Lines 237-306:** `generate_codebooks()` - K-means clustering and distance table computation
- **Lines 315-366:** `kMeans()` - Standard k-means implementation
- **Lines 378-478:** `pqEncode()` - Vector encoding with SQ on distance tables

### 4. PCA Integration (`include/strategy/flash_strategy.h`)
- **Lines 487-560:** `generate_matrix()` - PCA matrix computation with optional optimal subvector grouping
- **Lines 566-587:** `pcaEncode()` - PCA transformation application

### 5. Flash Distance Computation (`third_party/hnswlib/hnswalg_flash.h`)
- **Lines 1766-1906:** `PqLinkL2Sqr()` - **THE CORE INNOVATION**
  - Lines 1797-1847: AVX2 implementation with detailed comments
  - Lines 1770-1796: SSE implementation
  - Lines 1848-1881: AVX512 implementation
  - Lines 1883-1905: Fallback scalar implementation

### 6. Graph Construction and Search (`third_party/hnswlib/hnswalg_flash.h`)
- **Lines 329-441:** `searchBaseLayer()` - Construction-time search
- **Lines 444-659:** `searchBaseLayerST()` - Query-time search with all optimizations
- **Lines 662-722:** `getNeighborsByHeuristic2()` - RNG-based neighbor selection
- **Lines 745-901:** `mutuallyConnectNewElement()` - Bidirectional edge insertion

### 7. Space Implementations (`include/space/`)
- **`space_flash.h`:** Flash-specific distance with 4-bit lookups
- **`space_pq.h`:** PQ SDC/ADC distance functions
- **`space_sq.h`:** SQ encoding/decoding and distance

---

## Summary

The HNSW-Flash repository presents a highly optimized implementation of approximate nearest neighbor search. The key innovations are:

1. **PQLINK_STORE:** Co-locating neighbor vectors with graph structure eliminates random memory accesses
2. **PQLINK_CALC:** SIMD-optimized batch distance computation using PSHUFB for parallel table lookups
3. **4-bit Encoding:** With 16 clusters, two indices fit in one byte, enabling efficient SIMD processing
4. **Memory Alignment:** Proper alignment ensures maximum SIMD throughput

These optimizations work synergistically:
- PQ reduces vector dimensions and enables integer arithmetic
- 4-bit encoding with 16 clusters matches SIMD lane constraints perfectly
- PQLINK_STORE ensures data is available for batch processing
- PQLINK_CALC exploits modern CPU SIMD units to the fullest

For those learning the codebase, the recommended path is:
1. Understand the parameters in `core.h`
2. Study the base strategy pattern
3. Learn PQ/SQ/PCA encoding
4. Deep dive into `PqLinkL2Sqr()` for the SIMD magic
5. Trace through the search and construction algorithms

The combination of algorithmic techniques (HNSW graph, PQ compression, early termination) with systems-level optimizations (SIMD, memory layout, cache efficiency) makes this an excellent example of high-performance computing applied to machine learning infrastructure.
