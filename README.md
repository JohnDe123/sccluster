# scCluster

**Single-cell RNA-seq Clustering Pipeline**

A hybrid analysis pipeline integrating **scVI** (deep learning), **Leiden** algorithm (graph clustering), and **Wilcoxon rank-sum test** (non-parametric statistics) for high-precision single-cell clustering and marker gene identification.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

scCluster processes an expression matrix (cells × genes) through a multi-step pipeline:

```
A[原始表达矩阵<br/>(细胞 × 基因)]
    │
    ▼
B[步骤1: 预处理<br/>质控 → 归一化 → 高变基因选择]
    │
    ▼
C[步骤2: 深度学习降维<br/>scVI 模型训练<br/>(学习非线性、去噪、去批次的隐空间)]
    │
    ▼
D[步骤3: 图聚类<br/>基于scVI隐空间构建邻接图<br/>→ Leiden算法聚类]
    │
    ▼
E[步骤4: 特征基因提取<br/>非参数检验(Wilcoxon秩和检验)<br/>识别每群的高表达特异性基因]
    │
    ▼
F[步骤5: 结果验证与可视化<br/>UMAP图 / 差异基因火山图 / 热图]
```

### Why This Combination?

| Component | Role | Advantage |
|-----------|------|-----------|
| **scVI** | Non-linear dimensionality reduction | Handles dropout, batch effects, and noise; learns biological latent space |
| **Leiden** | Graph-based community detection | Guarantees well-connected communities; faster and better than Louvain |
| **Wilcoxon** | Marker gene identification | Non-parametric; no normality assumption; robust to outliers in count data |

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (GPU recommended for scVI training)

### From source

```bash
git clone https://github.com/user/sccluster.git
cd sccluster
pip install -e .
```

### Dependencies

```bash
pip install scanpy scvi-tools anndata numpy scipy pandas \
    scikit-learn matplotlib seaborn leidenalg umap-learn \
    torch pynndescent tqdm joblib adjustText
```

Full list in [`requirements.txt`](requirements.txt).

---

## Quick Start

### Command Line

```bash
# Run full pipeline on an H5AD file
sccluster run data.h5ad -o ./results

# High-resolution mode for rare subpopulations
sccluster run data.h5ad -o ./results --preset high_resolution --batch-key sample_id

# Quick test with fewer parameters
sccluster run data.h5ad -o ./results --preset quick

# From a CSV matrix (transpose if genes are rows)
sccluster run expression.csv -o ./results --format csv --transpose

# Generate synthetic test data
sccluster synthetic -o test.h5ad --n-cells 2000 --n-clusters 6
```

### Python API

```python
import scanpy as sc
from sccluster import SCClusterPipeline

# Load your data
adata = sc.read_h5ad("thyroid_epithelial.h5ad")

# Create and run pipeline
pipeline = SCClusterPipeline(
    adata,
    output_dir="./thyroid_results",
    random_seed=42,
)

# Run with preset configuration
pipeline.run(preset="default")

# Access results
print(f"Found {adata.obs['leiden'].nunique()} clusters")
print(adata.obs['leiden'].value_counts())

# Get markers for cluster 0
markers = pipeline.get_cluster_markers("0", n=10)
print(markers)
```

### Step-by-Step (Manual Control)

```python
from sccluster import (
    preprocess_pipeline,
    scvi_pipeline,
    cluster_pipeline,
    marker_pipeline,
    generate_all_figures,
)
import scanpy as sc

adata = sc.read_h5ad("data.h5ad")

# Step 1: Preprocessing
adata = preprocess_pipeline(adata)

# Step 2: scVI dimensionality reduction
model = scvi_pipeline(adata)

# Step 3: Leiden clustering
adata = cluster_pipeline(adata)

# Step 4: Marker gene identification
adata, filtered_markers, top_markers = marker_pipeline(adata)

# Step 5: Visualization
generate_all_figures(adata, output_dir="./figures")
```

---

## Pipeline Details

### Step 1: Preprocessing

- **Quality Control**: Filter cells by min genes (default: 200), filter genes by min cells (default: 3), remove cells with high mitochondrial read percentage (default: <20%)
- **Normalization**: Library-size normalization to 10,000 counts per cell, then log1p transformation
- **HVG Selection**: Select top 2,000 highly variable genes using Seurat v3 method

### Step 2: scVI Training

scVI (single-cell Variational Inference) is a deep generative model:

```
z_n ~ N(0, I)           ... latent cell representation (learned)
x_ng | z_n, s_n ~ ZINB  ... observed gene expression
```

Default architecture: 2 hidden layers × 128 nodes, 30-dimensional latent space, ZINB likelihood.

### Step 3: Leiden Clustering

1. Build kNN graph from scVI latent space (default: k=15)
2. Run Leiden community detection with configurable resolution
3. Compute UMAP for 2D visualization

### Step 4: Marker Gene Identification

Wilcoxon rank-sum test for each gene in each cluster vs. all others:
- Non-parametric: no normality assumption
- Multiple testing correction: Benjamini-Hochberg (FDR)
- Filters: adjusted p-value, log2 fold-change, expression fraction

### Step 5: Visualization

Outputs include:
- **UMAP** plot with cluster labels
- **Volcano plots** per cluster (logFC vs. -log10 p-value)
- **Heatmap** of top marker genes
- **Dotplot** showing expression patterns
- **Violin plots** for individual marker genes
- **QC metrics** by cluster
- **Cluster composition** bar chart

---

## Configuration Presets

| Preset | Latent Dim | HVGs | Resolution | Max Epochs | Use Case |
|--------|-----------|------|------------|------------|----------|
| `default` | 30 | 2,000 | 1.0 | 400 | Standard analysis |
| `high_resolution` | 50 | 3,000 | 2.0 | 500 | Rare subpopulations |
| `quick` | 10 | 1,000 | 0.8 | 100 | Fast exploration |

Custom configurations can override any preset parameter:

```python
pipeline.run(
    preset="default",
    custom_config={
        "scvi": {"model_kwargs": {"n_latent": 40, "n_hidden": 256}},
        "clustering": {"leiden_kwargs": {"resolution": 1.5}},
    },
)
```

---

## Output Structure

```
sccluster_results/
├── processed_adata.h5ad          # Full AnnData with all results
├── top_markers.csv               # Top marker genes (flat format)
├── markers_per_cluster.xlsx      # Per-cluster marker sheets
├── run_config.json               # Run parameters and timing
├── models/
│   └── scvi_model/               # Trained scVI model
├── figures/
│   ├── umap_clusters.png         # UMAP colored by cluster
│   ├── cluster_composition.png   # Cluster size distribution
│   ├── qc_by_cluster.png         # QC metrics per cluster
│   ├── marker_dotplot.png        # Dotplot of marker genes
│   ├── marker_heatmap.png        # Heatmap of marker expression
│   ├── marker_violins.png        # Violin plots of key markers
│   ├── volcano/                  # Per-cluster volcano plots
│   │   ├── volcano_cluster_0.png
│   │   ├── volcano_cluster_1.png
│   │   └── ...
│   └── umap_genes/               # UMAP colored by gene expression
│       ├── umap_GENE1.png
│       └── ...
```

---

## Input Formats

Supports multiple input formats:

| Format | Extension | Description |
|--------|-----------|-------------|
| H5AD | `.h5ad` | AnnData native format (recommended) |
| 10X HDF5 | `.h5` | 10X Genomics output |
| CSV | `.csv` | Comma-separated matrix |
| TSV | `.tsv` | Tab-separated matrix |
| MTX | `.mtx` | 10X sparse matrix directory |
| Loom | `.loom` | Loom format |

For CSV/TSV inputs, use `--transpose` if your file has genes as rows and cells as columns.

---

## API Reference

### Preprocessing

| Function | Description |
|----------|-------------|
| `filter_cells_genes()` | QC filter cells and genes |
| `normalize_and_log()` | Library-size normalize + log1p |
| `select_highly_variable_genes()` | HVG selection |
| `preprocess_pipeline()` | Full preprocessing workflow |

### scVI (Dimensionality Reduction)

| Function | Description |
|----------|-------------|
| `setup_scvi()` | Register AnnData for scVI |
| `train_scvi_model()` | Train scVI model |
| `get_latent_representation()` | Extract latent embedding |
| `scvi_pipeline()` | Full scVI workflow |

### Clustering

| Function | Description |
|----------|-------------|
| `build_neighbors_graph()` | kNN graph construction |
| `leiden_clustering()` | Leiden community detection |
| `compute_umap()` | UMAP embedding |
| `cluster_pipeline()` | Full clustering workflow |

### Markers

| Function | Description |
|----------|-------------|
| `rank_marker_genes()` | Wilcoxon rank-sum test |
| `filter_significant_markers()` | Filter by p-value, logFC |
| `extract_top_markers()` | Top N markers per cluster |
| `marker_pipeline()` | Full marker workflow |

### Visualization

| Function | Description |
|----------|-------------|
| `plot_umap()` | UMAP scatter plot |
| `plot_volcano()` | Volcano plot |
| `plot_heatmap()` | Expression heatmap |
| `plot_marker_dotplot()` | Dotplot of markers |
| `plot_marker_violin()` | Violin plots |
| `plot_cluster_composition()` | Cluster composition bar chart |
| `generate_all_figures()` | All standard figures |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

This pipeline builds on:

- **scVI**: Lopez, R., et al. "Deep generative modeling for single-cell transcriptomics." *Nature Methods* (2018).
- **Leiden**: Traag, V.A., et al. "From Louvain to Leiden: guaranteeing well-connected communities." *Scientific Reports* (2019).
- **Scanpy**: Wolf, F.A., et al. "SCANPY: large-scale single-cell gene expression data analysis." *Genome Biology* (2018).
