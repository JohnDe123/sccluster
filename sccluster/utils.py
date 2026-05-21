"""
Utility functions for data loading, saving, and cluster summary.
"""

from __future__ import annotations

import os
import logging
import warnings
from typing import Optional, Dict, List, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    module_name: str = "sccluster",
) -> logging.Logger:
    """
    Configure logging for the scCluster pipeline.

    Parameters
    ----------
    level : str
        Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
    log_file : str, optional
        Path to a log file.
    module_name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def load_data(
    path: str,
    format: Optional[str] = None,
    transpose: bool = False,
    var_names: str = "gene_symbols",
    **kwargs,
) -> AnnData:
    """
    Load single-cell expression data from various file formats.

    Parameters
    ----------
    path : str
        Path to the input file.
    format : str, optional
        File format: 'h5ad', 'h5', '10x-h5', 'csv', 'tsv', 'mtx', 'loom'.
        Auto-detected from extension if None.
    transpose : bool
        Transpose the data after loading (useful for CSV/TSV).
    var_names : str
        Column name for gene annotation when reading from 10X.
    **kwargs
        Additional arguments passed to scanpy reading functions.

    Returns
    -------
    AnnData
        Expression matrix.
    """
    path_lower = path.lower()
    format = format or (
        "h5ad" if path_lower.endswith(".h5ad") else
        "10x-h5" if path_lower.endswith(".h5") else
        "csv" if path_lower.endswith(".csv") else
        "tsv" if path_lower.endswith(".tsv") else
        "mtx" if path_lower.endswith(".mtx") else
        "loom" if path_lower.endswith(".loom") else
        None
    )

    if format is None:
        raise ValueError(
            f"Cannot auto-detect format for: {path}. "
            f"Please specify format='h5ad', 'csv', '10x-h5', etc."
        )

    print(f"[Load] Reading {format} from: {path}")

    if format == "h5ad":
        adata = sc.read_h5ad(path, **kwargs)
    elif format in ("10x-h5", "h5"):
        adata = sc.read_10x_h5(path, **kwargs)
    elif format == "csv":
        df = pd.read_csv(path, index_col=0)
        if transpose:
            df = df.T
        adata = AnnData(df.values, obs=pd.DataFrame(index=df.index), var=pd.DataFrame(index=df.columns))
    elif format == "tsv":
        df = pd.read_csv(path, index_col=0, sep="\t")
        if transpose:
            df = df.T
        adata = AnnData(df.values, obs=pd.DataFrame(index=df.index), var=pd.DataFrame(index=df.columns))
    elif format == "mtx":
        adata = sc.read_10x_mtx(os.path.dirname(path), var_names=var_names, **kwargs)
    elif format == "loom":
        adata = sc.read_loom(path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Ensure unique gene names
    if not adata.var_names.is_unique:
        n_dup = adata.var_names.duplicated().sum()
        warnings.warn(f"{n_dup} duplicate gene names found. Making unique.")
        adata.var_names_make_unique()

    # Ensure unique cell barcodes
    if not adata.obs_names.is_unique:
        n_dup = adata.obs_names.duplicated().sum()
        warnings.warn(f"{n_dup} duplicate cell barcodes found. Making unique.")
        adata.obs_names_make_unique()

    # Basic validation
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"Empty data loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    print(f"[Load] Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"[Load] Sparsity: {(adata.X == 0).mean() if hasattr(adata.X, '__array__') else (adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1])):.2%} non-zero")

    return adata


def save_results(
    adata: AnnData,
    output_dir: str,
    save_adata: bool = True,
    save_markers: bool = True,
    save_clusters: bool = True,
    compress: bool = True,
) -> Dict[str, str]:
    """
    Save analysis results to disk.

    Parameters
    ----------
    adata : AnnData
        Processed AnnData.
    output_dir : str
        Output directory.
    save_adata : bool
        Save full AnnData as .h5ad.
    save_markers : bool
        Save marker genes as CSV.
    save_clusters : bool
        Save cluster assignments as CSV.
    compress : bool
        Compress H5AD output.

    Returns
    -------
    dict
        Paths of saved files.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = {}

    if save_adata:
        path = os.path.join(output_dir, "processed_adata.h5ad")
        adata.write(path, compression="gzip" if compress else None)
        saved["adata"] = path
        print(f"[Save] AnnData → {path}")

    if save_clusters and "leiden" in adata.obs.columns:
        path = os.path.join(output_dir, "cluster_assignments.csv")
        cluster_df = pd.DataFrame({
            "cell_barcode": adata.obs_names,
            "cluster": adata.obs["leiden"].values,
        })
        cluster_df.to_csv(path, index=False)
        saved["clusters"] = path
        print(f"[Save] Cluster assignments ({adata.obs['leiden'].nunique()} clusters) → {path}")

    if save_markers and "rank_genes_groups" in adata.uns:
        path = os.path.join(output_dir, "marker_genes.csv")
        result = adata.uns["rank_genes_groups"]
        rows = []
        for cluster in result["names"].dtype.names:
            for i in range(min(100, len(result["names"][cluster]))):
                rows.append({
                    "cluster": cluster,
                    "rank": i + 1,
                    "gene": result["names"][cluster][i],
                    "score": result["scores"][cluster][i],
                    "logFC": result["logfoldchanges"][cluster][i],
                    "pval": result["pvals"][cluster][i],
                    "pval_adj": result["pvals_adj"][cluster][i],
                })
        pd.DataFrame(rows).to_csv(path, index=False)
        saved["markers"] = path
        print(f"[Save] Marker genes → {path}")

    return saved


def summarize_clusters(
    adata: AnnData,
    groupby: str = "leiden",
) -> pd.DataFrame:
    """
    Generate a summary DataFrame of cluster statistics.

    Parameters
    ----------
    adata : AnnData
        Data with cluster labels.
    groupby : str
        Key in adata.obs for cluster labels.

    Returns
    -------
    DataFrame
        Per-cluster statistics.
    """
    clusters = sorted(adata.obs[groupby].unique().tolist(), key=lambda x: int(x) if x.isdigit() else x)
    n_clusters = len(clusters)

    summary_rows = []
    for cluster in clusters:
        mask = adata.obs[groupby] == cluster
        n_cells = mask.sum()
        pct = n_cells / adata.n_obs * 100

        row = {
            "cluster": cluster,
            "n_cells": n_cells,
            "pct_cells": f"{pct:.1f}%",
        }

        if "n_genes" in adata.obs.columns:
            row["median_genes"] = adata.obs.loc[mask, "n_genes"].median()
        if "n_counts" in adata.obs.columns:
            row["median_counts"] = adata.obs.loc[mask, "n_counts"].median()
        if "pct_mt" in adata.obs.columns:
            row["mean_pct_mt"] = f"{adata.obs.loc[mask, 'pct_mt'].mean():.1f}%"

        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)

    # Print formatted summary
    print(f"\n{'='*70}")
    print(f"  CLUSTER SUMMARY ({n_clusters} clusters, {adata.n_obs} cells)")
    print(f"{'='*70}")
    print(f"  {'Cluster':<10} {'Cells':<10} {'%':<8}", end="")
    if "median_genes" in df.columns:
        print(f" {'Median Genes':<14}", end="")
    if "mean_pct_mt" in df.columns:
        print(f" {'Mean %MT':<10}", end="")
    print()
    print(f"  {'-'*60}")

    for _, row in df.iterrows():
        print(f"  {str(row['cluster']):<10} {row['n_cells']:<10} {row['pct_cells']:<8}", end="")
        if "median_genes" in row:
            print(f" {int(row['median_genes']):<14}", end="")
        if "mean_pct_mt" in row:
            print(f" {row['mean_pct_mt']:<10}", end="")
        print()
    print(f"{'='*70}\n")

    return df


def export_markers_to_csv(
    markers_df: pd.DataFrame,
    path: str,
    **kwargs,
) -> str:
    """
    Export marker genes DataFrame to CSV.

    Parameters
    ----------
    markers_df : DataFrame
        Markers table from extract_top_markers().
    path : str
        Output path.
    **kwargs
        Additional arguments for pandas.to_csv().

    Returns
    -------
    str
        Output path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    markers_df.to_csv(path, index=False, **kwargs)
    print(f"[Export] {len(markers_df)} markers → {path}")
    return path


def export_markers_to_excel(
    markers_dict: Dict[str, pd.DataFrame],
    path: str,
) -> str:
    """
    Export per-cluster markers to an Excel file (one sheet per cluster).

    Parameters
    ----------
    markers_dict : dict
        Dictionary mapping cluster labels to marker DataFrames.
    path : str
        Output .xlsx path.

    Returns
    -------
    str
        Output path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for cluster, df in markers_dict.items():
            sheet_name = f"Cluster_{cluster}"[:31]  # Excel 31-char limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"[Export] Markers per cluster → {path}")
    return path


def generate_synthetic_data(
    n_cells: int = 1000,
    n_genes: int = 5000,
    n_clusters: int = 5,
    n_batches: int = 2,
    random_seed: int = 42,
) -> AnnData:
    """
    Generate synthetic scRNA-seq data for testing.

    Creates a ground-truth dataset with known clusters and batch effects,
    useful for testing the pipeline.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_genes : int
        Number of genes.
    n_clusters : int
        Number of true cell types.
    n_batches : int
        Number of batches (adds batch effects).
    random_seed : int
        Random seed.

    Returns
    -------
    AnnData
        Synthetic data with ground truth cluster labels in .obs['true_label']
        and batch in .obs['batch'].
    """
    rng = np.random.default_rng(random_seed)

    # Gene-level parameters
    gene_means = rng.lognormal(mean=0, sigma=1.5, size=n_genes)
    gene_dispersions = rng.gamma(shape=2, scale=0.5, size=n_genes)

    # Cluster-specific expression profiles (log-fold changes)
    cluster_profiles = np.zeros((n_clusters, n_genes))
    n_marker_genes_per_cluster = max(10, n_genes // (n_clusters * 5))
    for k in range(n_clusters):
        marker_idx = rng.choice(n_genes, size=n_marker_genes_per_cluster, replace=False)
        cluster_profiles[k, marker_idx] = rng.lognormal(mean=1.5, sigma=0.5, size=n_marker_genes_per_cluster)

    # Batch effects
    batch_effects = rng.normal(0, 0.5, size=(n_batches, n_genes))

    # Generate cells
    counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    true_labels = np.zeros(n_cells, dtype=int)
    batch_labels = np.zeros(n_cells, dtype=int)
    cells_per_cluster = n_cells // n_clusters

    for k in range(n_clusters):
        start = k * cells_per_cluster
        end = start + cells_per_cluster if k < n_clusters - 1 else n_cells
        n = end - start
        true_labels[start:end] = k

        # Assign random batches
        batch_labels[start:end] = rng.choice(n_batches, size=n)

        # Base rates
        lib_size = rng.lognormal(mean=8.5, sigma=0.5, size=n)  # ~5k-10k UMIs
        base_rates = gene_means[np.newaxis, :] * np.exp(cluster_profiles[k])[np.newaxis, :]

        for i, idx in enumerate(range(start, end)):
            rate = lib_size[i] * base_rates[i % 1] * np.exp(batch_effects[batch_labels[idx]])
            rate = rate / rate.sum() * lib_size[i]
            counts[idx] = rng.negative_binomial(
                n=gene_dispersions,
                p=gene_dispersions / (gene_dispersions + rate),
            )

    # Build AnnData
    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    cell_names = [f"CELL_{i:04d}" for i in range(n_cells)]

    adata = AnnData(
        X=counts,
        obs=pd.DataFrame({
            "true_label": [f"Type_{k}" for k in true_labels],
            "batch": [f"Batch_{int(b)}" for b in batch_labels],
        }, index=cell_names),
        var=pd.DataFrame(index=gene_names),
    )

    # Add MT-like genes for QC
    n_mt = n_genes // 50
    for i in range(n_mt):
        mt_idx = rng.integers(0, n_genes)
        counts[:, mt_idx] *= rng.uniform(0.5, 2.0, size=n_cells)
    adata.X = counts

    print(f"[Synthetic] Generated: {n_cells} cells × {n_genes} genes")
    print(f"[Synthetic] {n_clusters} true cell types, {n_batches} batches")

    return adata
