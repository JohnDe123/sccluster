"""
scCluster: Single-cell RNA-seq Clustering Pipeline
===================================================

A hybrid analysis pipeline integrating:
  - scVI (deep learning) for non-linear, denoised, batch-corrected representation learning
  - Leiden algorithm (graph clustering) for high-resolution community detection
  - Wilcoxon rank-sum test (non-parametric statistics) for marker gene identification

Core workflow:
  1. Preprocessing: QC → Normalization → HVG selection
  2. scVI: Deep latent variable model training
  3. Clustering: kNN graph → Leiden community detection
  4. Markers: Differential expression via Wilcoxon test
  5. Visualization: UMAP, volcano plots, heatmaps
"""

__version__ = "1.0.0"
__author__ = "scCluster Team"

from sccluster.preprocessing import (
    filter_cells_genes,
    normalize_and_log,
    select_highly_variable_genes,
    preprocess_pipeline,
)
from sccluster.dim_reduction import (
    setup_scvi,
    train_scvi_model,
    get_latent_representation,
    scvi_pipeline,
)
from sccluster.clustering import (
    build_neighbors_graph,
    leiden_clustering,
    cluster_pipeline,
)
from sccluster.markers import (
    rank_marker_genes,
    filter_significant_markers,
    extract_top_markers,
    marker_pipeline,
)
from sccluster.visualization import (
    plot_umap,
    plot_volcano,
    plot_heatmap,
    plot_marker_dotplot,
    plot_marker_violin,
    plot_cluster_composition,
    generate_all_figures,
)
from sccluster.pipeline import SCClusterPipeline
from sccluster.utils import (
    setup_logging,
    save_results,
    load_data,
    summarize_clusters,
    export_markers_to_csv,
    export_markers_to_excel,
)

__all__ = [
    # Preprocessing
    "filter_cells_genes",
    "normalize_and_log",
    "select_highly_variable_genes",
    "preprocess_pipeline",
    # Dimensionality reduction (scVI)
    "setup_scvi",
    "train_scvi_model",
    "get_latent_representation",
    "scvi_pipeline",
    # Clustering (Leiden)
    "build_neighbors_graph",
    "leiden_clustering",
    "cluster_pipeline",
    # Marker genes
    "rank_marker_genes",
    "filter_significant_markers",
    "extract_top_markers",
    "marker_pipeline",
    # Visualization
    "plot_umap",
    "plot_volcano",
    "plot_heatmap",
    "plot_marker_dotplot",
    "plot_marker_violin",
    "plot_cluster_composition",
    "generate_all_figures",
    # Pipeline
    "SCClusterPipeline",
    # Utilities
    "setup_logging",
    "save_results",
    "load_data",
    "summarize_clusters",
    "export_markers_to_csv",
    "export_markers_to_excel",
]
