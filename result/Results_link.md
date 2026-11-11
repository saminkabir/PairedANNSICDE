# README

## Overview

This artifact provides the complete experimental results used in the paper, enabling full reproducibility of all analyses and figures. The results are organized in a structured format within the following Google Sheets document:

**Results Spreadsheet:**  
https://docs.google.com/spreadsheets/d/1P3xHtJ30q7GMMyMTdHv-V47996UrmS94eHRgxqbqx9I/edit?usp=sharing

The spreadsheet contains aggregated and per-configuration performance metrics for all evaluated Approximate Nearest Neighbor Search (ANNS) methods, including both standalone and paired configurations.

---

## Contents

The spreadsheet includes:

### Datasets
Results across 20 real-world datasets:
- 15 in-distribution (ID) datasets (query and data vectors follow the same distribution)
- 5 out-of-distribution (OOD) datasets (queries follow a different distribution than indexed data)

### Methods Evaluated
Multiple ANNS methods spanning different families:
- Graph-based methods (e.g., HNSW, NSG, FLATNAV)
- Quantization-based methods (e.g., VAQ)
- Partition-based methods (e.g., SCANN)
- Hash-based methods (e.g., DB-LSH)

### Configurations
- Standalone methods  
- Paired configurations (parallel execution of two indices with result merging and re-ranking)

### Performance Metrics
- Recall (primary accuracy metric)
- Query latency / throughput (QPS)
- Index construction time
- Memory usage (when applicable)
- Distance computations (for compute-normalized analysis)

### Experimental Settings
- Constraint profiles (e.g., `p_c = 0.25`, `p_c = 0.5`)
- Dataset scales (e.g., 100K, 500K, 1M)
- Varying dimensionalities
- Query difficulty levels via Relative Contrast (RC)

---

## Usage

The spreadsheet supports the following analyses:

### 1. Reproducing Results for Figures and Tables
Results associated with all figures and tables reported in the paper can be regenerated directly from the provided data.

### 2. Standalone vs. Paired Comparison
Compare recall–efficiency trade-offs between standalone methods and paired configurations.

### 3. Multi-Objective Evaluation
Analyze performance across multiple objectives:
- Query latency
- Construction cost

### 4. Hardness-Aware Analysis
Filter results based on query difficulty (RC) to evaluate performance under controlled hardness levels.

### 5. ID vs. OOD Analysis
Perform separate analyses for:
- In-distribution datasets
- Out-of-distribution datasets

---

## Notes on Paired Configurations

For paired configurations:
- Each query is issued to both indices independently.
- Each index returns its top-`k` candidates.
- Candidates are merged (up to `2k`, often fewer due to overlap).
- Final top-`k` results are obtained via exact distance re-ranking.

### Fair Comparison Protocol
- Standalone methods are also evaluated using `2k` candidate retrieval followed by re-ranking.

---

## Reproducibility

This artifact ensures:

- **Transparency:** All reported results are directly accessible.  
- **Reproducibility:** Analyses can be recomputed without rerunning indexing/search pipelines.  
- **Extensibility:** Researchers can reuse the data to evaluate new methods or pairing strategies.  

---

## Contact

For questions regarding the artifact or experimental setup, please refer to the paper or contact the authors.
