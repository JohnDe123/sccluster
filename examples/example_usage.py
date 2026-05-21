#!/usr/bin/env python3
"""
Example: scCluster Pipeline for Thyroid Epithelial Cell Clustering
===================================================================

This script demonstrates the complete workflow:
  1. Generate or load expression matrix (cells × genes)
  2. Preprocessing (QC, normalization, HVG selection)
  3. scVI deep learning dimensionality reduction
  4. Leiden graph-based clustering
  5. Wilcoxon marker gene identification
  6. Visualization and result export

Usage:
    python examples/example_usage.py
    python examples/example_usage.py --input your_data.h5ad
    python examples/example_usage.py --synthetic --n-cells 2000
"""

import argparse
import sys
import os

# Add package to path if running from source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")


def run_example(
    input_path: str = None,
    use_synthetic: bool = True,
    n_cells: int = 1000,
    n_genes: int = 5000,
    output_dir: str = "./example_results",
    preset: str = "default",
):
    """Run the full scCluster pipeline demonstration."""

    # ── Load or generate data ──────────────────────────────
    if input_path is not None:
        print(f"[Example] Loading data from: {input_path}")
        from sccluster.utils import load_data
        adata = load_data(input_path)
    elif use_synthetic:
        print("[Example] Generating synthetic thyroid-like data...")
        from sccluster.utils import generate_synthetic_data
        adata = generate_synthetic_data(
            n_cells=n_cells,
            n_genes=n_genes,
            n_clusters=6,
            n_batches=2,
            random_seed=42,
        )
        # Rename clusters to simulate thyroid cell types
        cluster_map = {
            "Type_0": "Follicular_1",
            "Type_1": "Follicular_2",
            "Type_2": "Parafollicular_C",
            "Type_3": "Hurthle",
            "Type_4": "Undifferentiated",
            "Type_5": "Progenitor",
        }
        adata.obs["cell_type"] = adata.obs["true_label"].map(cluster_map)
        print(f"[Example] Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")
    else:
        raise ValueError("Either --input or --synthetic must be specified.")

    print(f"[Example] Data shape: {adata.n_obs} cells × {adata.n_vars} genes")

    # ── Run pipeline ──────────────────────────────────────
    from sccluster.pipeline import SCClusterPipeline

    pipeline = SCClusterPipeline(
        adata,
        output_dir=output_dir,
        random_seed=42,
    )

    # Custom configuration for thyroid data
    custom_config = {
        "preprocessing": {
            "filter_kwargs": {
                "min_genes": 200,
                "min_cells": 3,
                "pct_mt_threshold": 20.0,
                "mt_prefix": "MT-",
            },
            "hvg_kwargs": {
                "n_top_genes": 2000,
                "flavor": "seurat_v3",
            },
        },
        "scvi": {
            "setup_kwargs": {"batch_key": "batch"},
            "model_kwargs": {
                "n_latent": 30,
                "n_layers": 2,
                "n_hidden": 128,
                "max_epochs": 300,
            },
        },
        "clustering": {
            "neighbors_kwargs": {"n_neighbors": 15},
            "leiden_kwargs": {"resolution": 1.0},
        },
        "markers": {
            "rank_kwargs": {
                "method": "wilcoxon",
                "corr_method": "benjamini-hochberg",
            },
        },
    }

    # Execute pipeline
    adata = pipeline.run(
        preset=preset,
        custom_config=custom_config,
    )

    # ── Display results ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    n_clusters = adata.obs["leiden"].nunique()
    print(f"\n  Number of clusters found: {n_clusters}")
    print(f"\n  Cluster sizes:")
    for cluster, size in adata.obs["leiden"].value_counts().sort_index().items():
        pct = size / adata.n_obs * 100
        bar = "█" * int(pct / 2)
        print(f"    Cluster {cluster}: {size:5d} cells ({pct:5.1f}%) {bar}")

    if pipeline.top_markers is not None and len(pipeline.top_markers) > 0:
        print(f"\n  Top marker genes (per cluster):")
        for cluster in sorted(pipeline.top_markers["cluster"].unique(), key=lambda x: int(x) if x.isdigit() else x):
            cluster_markers = pipeline.top_markers[
                pipeline.top_markers["cluster"] == cluster
            ].head(5)
            gene_str = ", ".join(
                f"{row['gene']} (logFC={row['logFC']:.1f})"
                for _, row in cluster_markers.iterrows()
            )
            print(f"    Cluster {cluster}: {gene_str}")

    print(f"\n  Output files saved to: {output_dir}/")
    print(f"    - {output_dir}/processed_adata.h5ad")
    print(f"    - {output_dir}/top_markers.csv")
    print(f"    - {output_dir}/figures/ (all visualizations)")
    print(f"    - {output_dir}/models/scvi_model/ (trained model)")
    print()

    return adata, pipeline


def main():
    parser = argparse.ArgumentParser(
        description="scCluster Example: Thyroid Epithelial Cell Clustering"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input data file (h5ad, csv, tsv, h5, etc.)"
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="Generate and use synthetic test data"
    )
    parser.add_argument(
        "--n-cells", type=int, default=1000,
        help="Number of cells for synthetic data"
    )
    parser.add_argument(
        "--n-genes", type=int, default=5000,
        help="Number of genes for synthetic data"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./example_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--preset", type=str, default="default",
        choices=["default", "high_resolution", "quick"],
        help="Analysis preset"
    )

    args = parser.parse_args()

    adata, pipeline = run_example(
        input_path=args.input,
        use_synthetic=args.synthetic,
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        output_dir=args.output_dir,
        preset=args.preset,
    )

    print("Example complete!")
    return 0


if __name__ == "__main__":
    main()
