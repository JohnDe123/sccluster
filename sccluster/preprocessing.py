"""
Preprocessing module for single-cell RNA-seq data.

Performs quality control, normalization, log-transformation,
and highly variable gene (HVG) selection using scanpy.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union, List

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse


def filter_cells_genes(
    adata: AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    max_genes: Optional[int] = None,
    max_counts: Optional[float] = None,
    min_counts: Optional[float] = None,
    pct_mt_threshold: float = 20.0,
    mt_prefix: str = "MT-",
    filter_ribo: bool = False,
    ribo_prefix: Tuple[str, ...] = ("RPS", "RPL"),
    max_ribo_pct: float = 50.0,
    copy: bool = False,
) -> AnnData:
    """
    Filter low-quality cells and genes from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Input expression matrix (cells x genes).
    min_genes : int
        Minimum number of genes a cell must express to be kept.
    min_cells : int
        Minimum number of cells a gene must be expressed in to be kept.
    max_genes : int, optional
        Maximum number of genes a cell can express (to filter doublets).
    max_counts : float, optional
        Maximum total counts per cell.
    min_counts : float, optional
        Minimum total counts per cell.
    pct_mt_threshold : float
        Maximum percentage of mitochondrial reads allowed per cell.
    mt_prefix : str
        Prefix for mitochondrial gene names.
    filter_ribo : bool
        Whether to filter cells with excessive ribosomal gene expression.
    ribo_prefix : tuple of str
        Prefixes for ribosomal protein gene names.
    max_ribo_pct : float
        Maximum percentage of ribosomal reads allowed per cell.
    copy : bool
        Whether to return a copy of the data.

    Returns
    -------
    AnnData
        Filtered AnnData object with QC metrics stored in .obs.
    """
    adata = adata.copy() if copy else adata

    # Store raw counts for QC before filtering
    adata.obs["n_genes"] = (adata.X > 0).sum(axis=1).A1 if issparse(adata.X) else (adata.X > 0).sum(axis=1)
    adata.obs["n_counts"] = adata.X.sum(axis=1).A1 if issparse(adata.X) else adata.X.sum(axis=1)

    # Detect mitochondrial genes
    mt_mask = [g.startswith(mt_prefix) for g in adata.var_names]
    n_mt = mt_mask.count(True)
    if n_mt > 0:
        adata.obs["pct_mt"] = (
            adata.X[:, mt_mask].sum(axis=1).A1 / adata.obs["n_counts"].values * 100
            if issparse(adata.X)
            else adata.X[:, mt_mask].sum(axis=1) / adata.obs["n_counts"].values * 100
        )
    else:
        adata.obs["pct_mt"] = 0.0
        warnings.warn(
            f"No mitochondrial genes found with prefix '{mt_prefix}'. "
            f"Skipping MT-based filtering. Available gene prefixes: "
            f"{set(g[:3] for g in adata.var_names[:20])}..."
        )

    # Detect ribosomal genes
    if filter_ribo:
        ribo_mask = np.any(
            [adata.var_names.str.startswith(p) for p in ribo_prefix], axis=0
        )
        n_ribo = ribo_mask.sum()
        if n_ribo > 0:
            adata.obs["pct_ribo"] = (
                adata.X[:, ribo_mask].sum(axis=1).A1 / adata.obs["n_counts"].values * 100
                if issparse(adata.X)
                else adata.X[:, ribo_mask].sum(axis=1) / adata.obs["n_counts"].values * 100
            )
        else:
            adata.obs["pct_ribo"] = 0.0

    # Log pre-filtering statistics
    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars
    print(f"[QC] Before filtering: {n_cells_before} cells × {n_genes_before} genes")

    # Apply cell-level filters
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if max_genes is not None:
        keep = adata.obs["n_genes"] <= max_genes
        adata = adata[keep, :].copy()

    if max_counts is not None:
        keep = adata.obs["n_counts"] <= max_counts
        adata = adata[keep, :].copy()

    if min_counts is not None:
        keep = adata.obs["n_counts"] >= min_counts
        adata = adata[keep, :].copy()

    # Filter by mitochondrial percentage
    keep_mt = adata.obs["pct_mt"] <= pct_mt_threshold
    adata = adata[keep_mt, :].copy()

    # Filter by ribosomal percentage
    if filter_ribo:
        keep_ribo = adata.obs["pct_ribo"] <= max_ribo_pct
        adata = adata[keep_ribo, :].copy()

    print(
        f"[QC] After filtering: {adata.n_obs} cells × {adata.n_vars} genes "
        f"(removed {n_cells_before - adata.n_obs} cells, "
        f"{n_genes_before - adata.n_vars} genes)"
    )

    # Print QC summary
    if "pct_mt" in adata.obs.columns:
        print(f"[QC] MT%: {adata.obs['pct_mt'].mean():.1f}% ± {adata.obs['pct_mt'].std():.1f}%")

    return adata


def normalize_and_log(
    adata: AnnData,
    target_sum: float = 1e4,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    key_added: Optional[str] = None,
    copy: bool = False,
) -> AnnData:
    """
    Normalize library size and log-transform the data.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object with raw counts.
    target_sum : float
        Target sum for library-size normalization (default: 10,000).
    exclude_highly_expressed : bool
        Whether to exclude highly expressed genes during normalization.
    max_fraction : float
        Maximum fraction of total counts a gene can represent
        before being excluded from normalization size factor calculation.
    key_added : str, optional
        Key in .layers to store normalized (pre-log) values.
    copy : bool
        Whether to return a copy.

    Returns
    -------
    AnnData
        Normalized and log-transformed AnnData.
    """
    adata = adata.copy() if copy else adata

    # Store raw counts
    adata.raw = adata.copy()

    # Normalize to target sum (library size correction)
    sc.pp.normalize_total(
        adata,
        target_sum=target_sum,
        exclude_highly_expressed=exclude_highly_expressed,
        max_fraction=max_fraction,
        key_added=key_added,
    )

    # Check for zero counts after normalization
    if issparse(adata.X):
        nnz = adata.X.nnz
        total = np.prod(adata.X.shape)
        sparsity = 1 - (nnz / total)
    else:
        sparsity = float((adata.X == 0).mean())

    print(
        f"[Normalize] Library-size normalization to {target_sum:.0e} counts/cell "
        f"(data sparsity: {sparsity:.1%})"
    )

    # Log1p transform
    sc.pp.log1p(adata)

    # Store log-normalized data in a layer for reference
    adata.layers["log_norm"] = adata.X.copy()

    print(f"[Normalize] log1p transformation applied")

    return adata


def select_highly_variable_genes(
    adata: AnnData,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    batch_key: Optional[str] = None,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    n_bins: int = 20,
    span: float = 0.3,
    subset: bool = True,
    inplace: bool = True,
) -> AnnData:
    """
    Select highly variable genes (HVGs) for downstream analysis.

    Supports 'seurat_v3' (default, count-based dispersion),
    'seurat' (log-normalized variance), and 'cell_ranger' flavors.

    Parameters
    ----------
    adata : AnnData
        Normalized expression data with raw counts in .raw.
    n_top_genes : int
        Number of top HVGs to select.
    flavor : str
        Method for HVG selection: 'seurat_v3', 'seurat', or 'cell_ranger'.
    batch_key : str, optional
        Key in .obs for batch-aware HVG selection.
    min_mean : float
        Minimum mean expression for HVG candidates.
    max_mean : float
        Maximum mean expression for HVG candidates.
    min_disp : float
        Minimum dispersion for HVG candidates.
    n_bins : int
        Number of bins for mean-expression binning (seurat flavors).
    span : float
        Loess span for variance-mean trend fitting (seurat_v3).
    subset : bool
        Whether to subset the AnnData to HVGs only.
    inplace : bool
        Whether to modify the AnnData in place.

    Returns
    -------
    AnnData
        AnnData with HVG annotation in .var and optionally subsetted.
    """
    if not inplace:
        adata = adata.copy()

    n_genes_before = adata.n_vars

    # Ensure raw counts are available for seurat_v3
    if flavor == "seurat_v3" and adata.raw is not None:
        # seurat_v3 expects raw counts
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor=flavor,
            batch_key=batch_key,
            span=span,
            n_bins=n_bins,
            subset=False,
            inplace=True,
        )
    else:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor=flavor,
            batch_key=batch_key,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp,
            n_bins=n_bins,
            subset=False,
            inplace=True,
        )

    n_hvg = adata.var["highly_variable"].sum()
    pct_hvg = n_hvg / n_genes_before * 100

    print(
        f"[HVG] Selected {n_hvg} highly variable genes "
        f"({pct_hvg:.1f}% of {n_genes_before})"
    )

    # Show distribution of HVG stats
    hvg_vars = adata.var.loc[adata.var["highly_variable"]]
    if "dispersions_norm" in hvg_vars.columns:
        print(f"[HVG] Normalized dispersion: "
              f"{hvg_vars['dispersions_norm'].min():.3f} - "
              f"{hvg_vars['dispersions_norm'].max():.3f}")

    if subset:
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"[HVG] Subset data to HVGs: {adata.n_obs} cells × {adata.n_vars} genes")

    return adata


def detect_doublets(
    adata: AnnData,
    method: str = "scrublet",
    expected_doublet_rate: float = 0.075,
    random_state: int = 42,
    sim_doublet_ratio: int = 2,
    n_neighbors: Optional[int] = None,
    copy: bool = False,
) -> AnnData:
    """
    Detect potential doublets in the data using Scrublet.

    Parameters
    ----------
    adata : AnnData
        Raw count expression matrix.
    method : str
        Doublet detection method. Currently supports 'scrublet'.
    expected_doublet_rate : float
        Expected doublet rate for the experiment.
    random_state : int
        Random seed for reproducibility.
    sim_doublet_ratio : int
        Number of simulated doublets per real cell.
    n_neighbors : int, optional
        Number of neighbors. Default: round(log10(n_cells) * 20).

    Returns
    -------
    AnnData
        AnnData with doublet scores and predictions in .obs.
    """
    adata = adata.copy() if copy else adata

    if method == "scrublet":
        sc.pp.scrublet(
            adata,
            expected_doublet_rate=expected_doublet_rate,
            sim_doublet_ratio=sim_doublet_ratio,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        n_pred = adata.obs["predicted_doublet"].sum() if "predicted_doublet" in adata.obs else 0
        print(
            f"[Doublet] Scrublet: predicted {n_pred} doublets "
            f"({n_pred / adata.n_obs * 100:.1f}%) "
            f"(expected rate: {expected_doublet_rate:.1%})"
        )

    return adata


def preprocess_pipeline(
    adata: AnnData,
    filter_kwargs: Optional[dict] = None,
    doublet_kwargs: Optional[dict] = None,
    normalize_kwargs: Optional[dict] = None,
    hvg_kwargs: Optional[dict] = None,
    detect_doublets_flag: bool = False,
    verbose: bool = True,
) -> AnnData:
    """
    Run the full preprocessing pipeline.

    Steps:
    1. QC filtering (cells, genes, MT%, etc.)
    2. Doublet detection (optional)
    3. Library-size normalization + log1p
    4. Highly variable gene selection

    Parameters
    ----------
    adata : AnnData
        Raw expression matrix.
    filter_kwargs : dict, optional
        Keyword arguments for filter_cells_genes().
    doublet_kwargs : dict, optional
        Keyword arguments for detect_doublets().
    normalize_kwargs : dict, optional
        Keyword arguments for normalize_and_log().
    hvg_kwargs : dict, optional
        Keyword arguments for select_highly_variable_genes().
    detect_doublets_flag : bool
        Whether to run doublet detection.
    verbose : bool
        Print progress messages.

    Returns
    -------
    AnnData
        Preprocessed AnnData ready for scVI training.
    """
    if verbose:
        print("=" * 60)
        print("  STEP 1: PREPROCESSING")
        print("=" * 60)

    # Step 1.1: QC filtering
    if verbose:
        print("\n--- 1.1 Quality Control ---")
    fkwargs = filter_kwargs or {}
    adata = filter_cells_genes(adata, **fkwargs)

    # Step 1.2: Doublet detection (optional)
    if detect_doublets_flag:
        if verbose:
            print("\n--- 1.2 Doublet Detection ---")
        dkwargs = doublet_kwargs or {}
        adata = detect_doublets(adata, **dkwargs)

    # Step 1.3: Normalization and log-transform
    if verbose:
        print("\n--- 1.3 Normalization ---")
    nkwargs = normalize_kwargs or {}
    adata = normalize_and_log(adata, **nkwargs)

    # Step 1.4: Highly variable gene selection
    if verbose:
        print("\n--- 1.4 HVG Selection ---")
    hkwargs = hvg_kwargs or {}
    adata = select_highly_variable_genes(adata, **hkwargs)

    if verbose:
        print(f"\n[Preprocessing] Complete: {adata.n_obs} cells × {adata.n_vars} genes")

    return adata
