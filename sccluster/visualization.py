"""
Visualization module for scRNA-seq clustering results.

Produces publication-quality figures:
  - UMAP plots colored by cluster, expression, and metadata
  - Volcano plots for differential expression
  - Heatmaps of marker gene expression
  - Dotplots showing marker expression patterns
  - Violin plots for gene expression distributions
  - Cluster composition bar charts
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
from anndata import AnnData
import scanpy as sc


# ─────────────────────────────────────────────
# Color palette utilities
# ─────────────────────────────────────────────

def _get_cluster_colors(n_clusters: int, palette: str = "tab20") -> np.ndarray:
    """Generate distinct colors for cluster visualization."""
    if n_clusters <= 10:
        cmap = plt.cm.get_cmap("tab10")
    elif n_clusters <= 20:
        cmap = plt.cm.get_cmap("tab20")
    else:
        cmap = plt.cm.get_cmap("gist_rainbow")
    return cmap(np.linspace(0, 1, n_clusters))


def _default_colors() -> List[str]:
    """Default color cycle for sequential clusters."""
    return [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]


def _setup_figure(figsize: Tuple[float, float] = (8, 6), dpi: int = 150):
    """Set up matplotlib figure with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax


# ─────────────────────────────────────────────
# UMAP Visualization
# ─────────────────────────────────────────────

def plot_umap(
    adata: AnnData,
    color: Union[str, List[str]] = "leiden",
    layer: Optional[str] = None,
    palette: Optional[str] = None,
    size: float = 5.0,
    alpha: float = 0.8,
    legend_loc: str = "right margin",
    legend_fontsize: int = 10,
    title: Optional[str] = None,
    frameon: bool = True,
    show: bool = False,
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot UMAP embedding colored by cluster labels or gene expression.

    Parameters
    ----------
    adata : AnnData
        Data with UMAP coordinates in .obsm['X_umap'].
    color : str or list of str
        Key(s) in adata.obs for coloring, or gene name(s) for expression.
    layer : str, optional
        Layer to use for gene expression coloring.
    palette : str, optional
        Color palette name. Auto-selected if None.
    size : float
        Point size for scatter plot.
    alpha : float
        Opacity of points.
    legend_loc : str
        Legend position: 'right margin', 'on data', 'best', etc.
    legend_fontsize : int
        Font size for legend labels.
    title : str, optional
        Figure title. Auto-generated if None.
    frameon : bool
        Whether to draw the plot frame.
    show : bool
        Whether to call plt.show().
    save_path : str, optional
        Path to save the figure.
    **kwargs
        Additional arguments for sc.pl.umap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    print(f"[Plot] UMAP colored by: {color}")

    # Handle single color key
    color_key = color if isinstance(color, list) else [color]

    fig = sc.pl.umap(
        adata,
        color=color_key,
        layer=layer,
        palette=palette,
        size=size,
        alpha=alpha,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
        title=title,
        frameon=frameon,
        show=show,
        return_fig=True,
        **kwargs,
    )

    # Handle list-of-axes or single-axes case
    if hasattr(fig, "savefig"):
        pass  # single axes figure
    elif isinstance(fig, list):
        fig = fig[0] if len(fig) > 0 else plt.gcf()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Plot] UMAP saved to: {save_path}")

    return fig


def plot_umap_clusters(
    adata: AnnData,
    cluster_key: str = "leiden",
    palette: Optional[List[str]] = None,
    size: float = 8.0,
    show: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Convenience function: UMAP with cluster labels and centered legend."""
    n_clusters = adata.obs[cluster_key].nunique()
    if palette is None:
        palette = _default_colors()[:n_clusters]

    print(f"[Plot] UMAP clusters: {n_clusters} clusters, key='{cluster_key}'")

    fig, ax = _setup_figure((10, 8))

    sc.pl.umap(
        adata,
        color=cluster_key,
        ax=ax,
        palette=palette,
        size=size,
        legend_loc="right margin",
        legend_fontsize=9,
        frameon=False,
        show=False,
    )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(f"scVI + Leiden Clustering ({n_clusters} clusters)", fontsize=14, fontweight="bold")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Plot] UMAP clusters saved to: {save_path}")

    return fig


# ─────────────────────────────────────────────
# Volcano Plot
# ─────────────────────────────────────────────

def plot_volcano(
    adata: AnnData,
    cluster: str,
    key: str = "rank_genes_groups",
    groupby: str = "leiden",
    pval_cutoff: float = 0.05,
    logfc_cutoff: float = 1.0,
    n_label: int = 15,
    figsize: Tuple[float, float] = (10, 8),
    show: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Volcano plot showing differentially expressed genes for a cluster.

    Parameters
    ----------
    adata : AnnData
        Data with rank_genes_groups results in .uns.
    cluster : str
        Cluster label to plot.
    key : str
        Key in adata.uns for marker results.
    groupby : str
        Key for cluster labels.
    pval_cutoff : float
        Adjusted p-value threshold for significance (horizontal line).
    logfc_cutoff : float
        Log2 fold-change threshold (vertical lines).
    n_label : int
        Number of top genes to label.
    figsize : tuple
        Figure size.
    show : bool
        Whether to call plt.show().
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    result = adata.uns[key]
    cluster_names = result["names"].dtype.names

    if cluster not in cluster_names:
        raise ValueError(
            f"Cluster '{cluster}' not found. Available: {cluster_names}"
        )

    names = result["names"][cluster]
    scores = result["scores"][cluster]
    logfc = result["logfoldchanges"][cluster]
    pvals_adj = result["pvals_adj"][cluster]

    # Build DataFrame
    df = pd.DataFrame({
        "gene": names,
        "logFC": logfc,
        "pval_adj": pvals_adj,
        "score": scores,
    })
    df["neg_log10_padj"] = -np.log10(df["pval_adj"].clip(lower=1e-300))

    # Significance categories
    df["is_significant"] = (df["pval_adj"] < pval_cutoff) & (df["logFC"].abs() > logfc_cutoff)
    df["direction"] = "NS"
    df.loc[df["is_significant"] & (df["logFC"] > 0), "direction"] = "Up"
    df.loc[df["is_significant"] & (df["logFC"] < 0), "direction"] = "Down"

    # Counts
    n_up = (df["direction"] == "Up").sum()
    n_down = (df["direction"] == "Down").sum()

    print(f"[Volcano] Cluster {cluster}: {n_up} upregulated, {n_down} downregulated "
          f"(p_adj < {pval_cutoff}, |logFC| > {logfc_cutoff})")

    # Plot
    fig, ax = _setup_figure(figsize)

    colors = {"Up": "#d62728", "Down": "#1f77b4", "NS": "#cccccc"}
    for direction, color in colors.items():
        subset = df[df["direction"] == direction]
        ax.scatter(
            subset["logFC"], subset["neg_log10_padj"],
            c=color, s=3, alpha=0.6, label=f"{direction} ({len(subset)})",
            rasterized=True,
        )

    # Threshold lines
    ax.axhline(-np.log10(pval_cutoff), color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axvline(logfc_cutoff, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axvline(-logfc_cutoff, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)

    # Label top genes
    top_genes = df.nlargest(n_label, "neg_log10_padj")
    for _, row in top_genes.iterrows():
        if row["is_significant"]:
            ax.annotate(
                row["gene"],
                (row["logFC"], row["neg_log10_padj"]),
                fontsize=7,
                ha="center",
                va="bottom",
                alpha=0.9,
                fontweight="bold" if row["neg_log10_padj"] > 10 else "normal",
            )

    ax.set_xlabel("Log2 Fold Change", fontsize=13)
    ax.set_ylabel("-Log10 Adjusted P-value", fontsize=13)
    ax.set_title(f"Cluster {cluster} vs. Rest (Wilcoxon)", fontsize=14, fontweight="bold")

    ax.legend(
        loc="upper right", fontsize=9,
        title="Direction", title_fontsize=10,
    )

    # Add stats annotation
    ax.text(
        0.98, 0.95,
        f"Total DEGs: {n_up + n_down}\nUp: {n_up}\nDown: {n_down}",
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    sns.despine()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Volcano] Saved to: {save_path}")

    return fig


# ─────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────

def plot_heatmap(
    adata: AnnData,
    var_names: Union[List[str], Dict[str, List[str]]],
    groupby: str = "leiden",
    n_top: int = 5,
    layer: Optional[str] = None,
    use_raw: bool = True,
    cmap: str = "RdBu_r",
    standard_scale: str = "var",
    dendrogram: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    show_gene_labels: bool = True,
    show: bool = False,
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Heatmap of marker gene expression across clusters.

    Parameters
    ----------
    adata : AnnData
        Expression data with cluster labels.
    var_names : list or dict
        List of gene names, or dict mapping clusters to gene lists.
    groupby : str
        Key in adata.obs for cluster labels.
    n_top : int
        If var_names is not provided, use top n markers per cluster.
    layer : str, optional
        Layer to use for expression data.
    use_raw : bool
        Use raw expression data for the heatmap.
    cmap : str
        Colormap name.
    standard_scale : str
        Standardization: 'var' (genes) or 'group' (clusters).
    dendrogram : bool
        Whether to add dendrograms (hierarchical clustering of rows/cols).
    figsize : tuple, optional
        Figure size (width, height).
    show_gene_labels : bool
        Show gene names along the x-axis.
    show : bool
        Whether to call plt.show().
    save_path : str, optional
        Path to save the figure.
    **kwargs
        Additional arguments for sc.pl.heatmap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Extract top markers if not provided
    if var_names is None:
        markers_dict = {}
        if "rank_genes_groups" in adata.uns:
            result = adata.uns["rank_genes_groups"]
            for cluster in result["names"].dtype.names:
                markers_dict[cluster] = result["names"][cluster][:n_top].tolist()
        var_names = []
        for genes in markers_dict.values():
            var_names.extend(genes)
        # Remove duplicates keeping order
        seen = set()
        var_names = [g for g in var_names if not (g in seen or seen.add(g))]

    if isinstance(var_names, dict):
        all_genes = []
        for genes in var_names.values():
            all_genes.extend(genes)
        seen = set()
        var_names_list = [g for g in all_genes if not (g in seen or seen.add(g))]
    else:
        var_names_list = var_names

    print(f"[Heatmap] {len(var_names_list)} genes across clusters")

    # Compute auto figsize
    if figsize is None:
        n_genes = len(var_names_list)
        w = max(8, n_genes * 0.25)
        h = max(6, adata.obs[groupby].nunique() * 0.6)
        figsize = (min(w, 20), min(h, 16))

    fig = sc.pl.heatmap(
        adata,
        var_names=var_names_list,
        groupby=groupby,
        layer=layer,
        use_raw=use_raw,
        cmap=cmap,
        standard_scale=standard_scale,
        dendrogram=dendrogram,
        figsize=figsize,
        show_gene_labels=show_gene_labels,
        show=show,
        **kwargs,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        if hasattr(fig, "savefig"):
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        elif isinstance(fig, dict):
            fig.get("grouper", plt.gcf()).savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Heatmap] Saved to: {save_path}")

    return fig


# ─────────────────────────────────────────────
# Dotplot
# ─────────────────────────────────────────────

def plot_marker_dotplot(
    adata: AnnData,
    var_names: Optional[List[str]] = None,
    groupby: str = "leiden",
    n_genes: int = 5,
    layer: Optional[str] = None,
    use_raw: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    dendrogram: bool = False,
    show: bool = False,
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Dotplot showing marker gene expression (dot size = % expressing,
    color = mean expression) per cluster.

    Parameters
    ----------
    adata : AnnData
        Expression data.
    var_names : list of str, optional
        Genes to plot. Auto-selects top markers if None.
    groupby : str
        Cluster labels.
    n_genes : int
        Number of top markers per cluster (used if var_names is None).
    layer : str, optional
        Data layer for expression.
    use_raw : bool
        Use raw expression data.
    figsize : tuple, optional
        Figure size.
    dendrogram : bool
        Add dendrogram from hierarchical clustering.
    show : bool
        Call plt.show().
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Auto-select marker genes
    if var_names is None:
        markers = set()
        if "rank_genes_groups" in adata.uns:
            result = adata.uns["rank_genes_groups"]
            for cluster in result["names"].dtype.names:
                top = result["names"][cluster][:n_genes]
                markers.update(top)
        var_names = sorted(markers)

    n_genes_plot = len(var_names)
    n_clusters = adata.obs[groupby].nunique()

    if figsize is None:
        w = max(8, n_genes_plot * 0.4)
        h = max(5, n_clusters * 0.35)
        figsize = (min(w, 22), min(h, 14))

    print(f"[Dotplot] {n_genes_plot} genes × {n_clusters} clusters")

    fig = sc.pl.dotplot(
        adata,
        var_names=var_names,
        groupby=groupby,
        layer=layer,
        use_raw=use_raw,
        dendrogram=dendrogram,
        figsize=figsize,
        show=show,
        return_fig=True,
        **kwargs,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Dotplot] Saved to: {save_path}")

    return fig


# ─────────────────────────────────────────────
# Violin Plot
# ─────────────────────────────────────────────

def plot_marker_violin(
    adata: AnnData,
    genes: List[str],
    groupby: str = "leiden",
    layer: Optional[str] = None,
    use_raw: bool = True,
    rotation: int = 45,
    n_cols: int = 3,
    figsize_per_gene: Tuple[float, float] = (4, 3),
    show: bool = False,
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Violin plots showing expression distribution of marker genes per cluster.

    Parameters
    ----------
    adata : AnnData
        Expression data.
    genes : list of str
        Genes to plot.
    groupby : str
        Cluster labels.
    layer : str, optional
        Data layer.
    use_raw : bool
        Use raw expression.
    rotation : int
        Rotation angle for gene labels.
    n_cols : int
        Number of columns in the subplot grid.
    figsize_per_gene : tuple
        Figure size per gene subplot.
    show : bool
        Call plt.show().
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_genes = len(genes)
    print(f"[Violin] {n_genes} genes, grouped by '{groupby}'")

    fig = sc.pl.violin(
        adata,
        keys=genes,
        groupby=groupby,
        layer=layer,
        use_raw=use_raw,
        rotation=rotation,
        ncols=n_cols,
        size=2,
        show=show,
        return_fig=True,
        **kwargs,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        if isinstance(fig, list):
            for i, f in enumerate(fig):
                f.savefig(
                    save_path.replace(".png", f"_panel{i}.png"),
                    dpi=300, bbox_inches="tight", facecolor="white"
                )
        else:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Violin] Saved to: {save_path}")

    return fig


# ─────────────────────────────────────────────
# Cluster Composition Plot
# ─────────────────────────────────────────────

def plot_cluster_composition(
    adata: AnnData,
    groupby: str = "leiden",
    split_by: Optional[str] = None,
    normalize: bool = True,
    palette: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    show: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of cluster sizes / composition.

    Parameters
    ----------
    adata : AnnData
        Data with cluster labels.
    groupby : str
        Cluster labels.
    split_by : str, optional
        Additional grouping variable for stacked bars.
    normalize : bool
        Normalize to proportion if split_by is provided.
    palette : list, optional
        Color palette.
    figsize : tuple
        Figure size.
    show : bool
        Call plt.show().
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _setup_figure(figsize)

    if split_by is not None:
        # Stacked bar chart
        ct = pd.crosstab(adata.obs[groupby], adata.obs[split_by])
        if normalize:
            ct = ct.div(ct.sum(axis=1), axis=0)
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", edgecolor="white", linewidth=0.3)
        ax.set_ylabel("Proportion" if normalize else "Count", fontsize=12)
        ax.legend(title=split_by, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    else:
        # Simple bar chart
        counts = adata.obs[groupby].value_counts().sort_index()
        n_clusters = len(counts)
        if palette is None:
            palette = _default_colors()[:n_clusters]
        bars = ax.bar(
            range(n_clusters), counts.values,
            color=palette[:n_clusters], edgecolor="white", linewidth=0.5,
        )
        # Add count labels on bars
        for bar, count in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts.values) * 0.01,
                str(count), ha="center", va="bottom", fontsize=8,
            )
        ax.set_ylabel("Number of Cells", fontsize=12)

    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_title(f"Cluster Composition ({adata.n_obs} cells, {n_clusters} clusters)", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)

    sns.despine()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Composition] Saved to: {save_path}")

    return fig


# ─────────────────────────────────────────────
# QC Summary Plot
# ─────────────────────────────────────────────

def plot_qc_summary(
    adata: AnnData,
    metrics: Optional[List[str]] = None,
    groupby: str = "leiden",
    ncols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    QC metrics violin plots by cluster to assess cluster quality.

    Parameters
    ----------
    adata : AnnData
        Data with QC metrics in .obs.
    metrics : list of str, optional
        QC metric keys to plot. Default: ['n_genes', 'n_counts', 'pct_mt'].
    groupby : str
        Cluster labels.
    ncols : int
        Number of columns in subplot grid.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if metrics is None:
        metrics = ["n_genes", "n_counts"]
        if "pct_mt" in adata.obs.columns:
            metrics.append("pct_mt")
        if "pct_ribo" in adata.obs.columns:
            metrics.append("pct_ribo")

    print(f"[QC Plot] Metrics: {metrics}")

    n_metrics = len(metrics)
    nrows = int(np.ceil(n_metrics / ncols))

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig = sc.pl.violin(
        adata,
        keys=metrics,
        groupby=groupby,
        ncols=ncols,
        size=2,
        show=False,
        return_fig=True,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        if isinstance(fig, list):
            for i, f in enumerate(fig):
                f.savefig(
                    save_path.replace(".png", f"_panel{i}.png"),
                    dpi=300, bbox_inches="tight", facecolor="white"
                )
        else:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[QC Plot] Saved to: {save_path}")

    return fig


# ─────────────────────────────────────────────
# Batch-corrected UMAP (before/after comparison)
# ─────────────────────────────────────────────

def plot_batch_comparison(
    adata_raw: AnnData,
    adata_scvi: AnnData,
    batch_key: str,
    figsize: Tuple[float, float] = (16, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side UMAP comparison: raw PCA vs. scVI latent space,
    to visualize batch correction effectiveness.

    Parameters
    ----------
    adata_raw : AnnData
        Data with PCA embedding (before batch correction).
    adata_scvi : AnnData
        Data with scVI-corrected UMAP.
    batch_key : str
        Batch annotation key.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    print(f"[Batch] Comparing before/after batch correction")

    # Compute PCA-based UMAP for "before" if not already done
    if "X_umap" not in adata_raw.obsm:
        sc.pp.pca(adata_raw)
        sc.pp.neighbors(adata_raw)
        sc.tl.umap(adata_raw)

    # Before (PCA-based)
    sc.pl.umap(
        adata_raw, color=batch_key, ax=axes[0],
        palette="tab10", title=f"Before scVI (PCA)\ncolored by {batch_key}",
        show=False, frameon=False,
    )

    # After (scVI-based)
    sc.pl.umap(
        adata_scvi, color=batch_key, ax=axes[1],
        palette="tab10", title=f"After scVI (Latent Space)\ncolored by {batch_key}",
        show=False, frameon=False,
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Batch] Comparison saved to: {save_path}")

    return fig


# ─────────────────────────────────────────────
# Generate All Figures
# ─────────────────────────────────────────────

def generate_all_figures(
    adata: AnnData,
    output_dir: str = "./figures",
    cluster_key: str = "leiden",
    marker_key: str = "rank_genes_groups",
    n_markers_per_cluster: int = 5,
    selected_markers: Optional[List[str]] = None,
    batch_key: Optional[str] = None,
    adata_raw: Optional[AnnData] = None,
    show: bool = False,
) -> Dict[str, str]:
    """
    Generate all standard visualizations for a clustering result.

    Parameters
    ----------
    adata : AnnData
        AnnData with full pipeline results.
    output_dir : str
        Directory to save figures.
    cluster_key : str
        Key for cluster labels in adata.obs.
    marker_key : str
        Key for marker gene results in adata.uns.
    n_markers_per_cluster : int
        Number of top marker genes to display per cluster.
    selected_markers : list of str, optional
        Specific marker genes to focus on. Auto-selected if None.
    batch_key : str, optional
        Batch annotation key for batch correction visualization.
    adata_raw : AnnData, optional
        Raw (pre-scVI) AnnData for batch comparison.
    show : bool
        Call plt.show() on each figure.

    Returns
    -------
    dict
        Mapping of plot name to file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    print("\n" + "=" * 60)
    print("  STEP 5: VISUALIZATION")
    print("=" * 60)

    # 1. UMAP with cluster labels
    print("\n--- 5.1 UMAP Plot ---")
    fig = plot_umap_clusters(
        adata, cluster_key=cluster_key,
        save_path=f"{output_dir}/umap_clusters.png",
        show=show,
    )
    saved_paths["umap_clusters"] = f"{output_dir}/umap_clusters.png"
    plt.close(fig)

    # 2. Cluster composition
    print("\n--- 5.2 Cluster Composition ---")
    fig = plot_cluster_composition(
        adata, groupby=cluster_key,
        save_path=f"{output_dir}/cluster_composition.png",
        show=show,
    )
    saved_paths["cluster_composition"] = f"{output_dir}/cluster_composition.png"
    plt.close(fig)

    # 3. QC metrics by cluster
    print("\n--- 5.3 QC Metrics ---")
    fig = plot_qc_summary(
        adata, groupby=cluster_key,
        save_path=f"{output_dir}/qc_by_cluster.png",
    )
    saved_paths["qc_by_cluster"] = f"{output_dir}/qc_by_cluster.png"
    plt.close("all")

    # 4. Volcano plot for each cluster
    if marker_key in adata.uns:
        print(f"\n--- 5.4 Volcano Plots ---")
        os.makedirs(f"{output_dir}/volcano", exist_ok=True)
        result = adata.uns[marker_key]
        for cluster in result["names"].dtype.names:
            fig = plot_volcano(
                adata, cluster=cluster, key=marker_key,
                groupby=cluster_key,
                save_path=f"{output_dir}/volcano/volcano_cluster_{cluster}.png",
                show=show,
            )
            plt.close(fig)
        saved_paths["volcano"] = f"{output_dir}/volcano/"

        # 5. Dotplot
        print("\n--- 5.5 Dotplot ---")
        fig = plot_marker_dotplot(
            adata, groupby=cluster_key, n_genes=n_markers_per_cluster,
            save_path=f"{output_dir}/marker_dotplot.png",
            show=show,
        )
        saved_paths["marker_dotplot"] = f"{output_dir}/marker_dotplot.png"
        plt.close("all")

        # 6. Heatmap
        print("\n--- 5.6 Heatmap ---")
        fig = plot_heatmap(
            adata, var_names=None, groupby=cluster_key, n_top=n_markers_per_cluster,
            save_path=f"{output_dir}/marker_heatmap.png",
            show=show,
        )
        saved_paths["marker_heatmap"] = f"{output_dir}/marker_heatmap.png"
        plt.close("all")

        # 7. Violin plots for selected markers
        print("\n--- 5.7 Violin Plots ---")
        if selected_markers is None:
            selected_markers = []
            result = adata.uns[marker_key]
            for cluster in result["names"].dtype.names[:4]:
                top = result["names"][cluster][:2]
                selected_markers.extend(top)
            selected_markers = list(dict.fromkeys(selected_markers))  # deduplicate

        if selected_markers:
            fig = plot_marker_violin(
                adata, genes=selected_markers, groupby=cluster_key,
                save_path=f"{output_dir}/marker_violins.png",
                show=show,
            )
            saved_paths["marker_violins"] = f"{output_dir}/marker_violins.png"
            plt.close("all")
    else:
        print(f"\n[Viz] Skipping marker-based plots: '{marker_key}' not found in adata.uns")

    # 8. Batch comparison (if batch_key provided)
    if batch_key is not None and adata_raw is not None:
        print("\n--- 5.8 Batch Correction Comparison ---")
        fig = plot_batch_comparison(
            adata_raw, adata, batch_key=batch_key,
            save_path=f"{output_dir}/batch_comparison.png",
        )
        saved_paths["batch_comparison"] = f"{output_dir}/batch_comparison.png"
        plt.close(fig)

    # 9. UMAP colored by expression of top markers
    if marker_key in adata.uns and selected_markers:
        print("\n--- 5.9 UMAP Expression Maps ---")
        os.makedirs(f"{output_dir}/umap_genes", exist_ok=True)
        for gene in selected_markers[:8]:  # limit to 8 genes
            if gene in adata.var_names or (adata.raw is not None and gene in adata.raw.var_names):
                fig = plot_umap(
                    adata, color=gene,
                    save_path=f"{output_dir}/umap_genes/umap_{gene}.png",
                    show=show,
                )
                plt.close(fig)
        saved_paths["umap_genes"] = f"{output_dir}/umap_genes/"

    print(f"\n[Viz] Generated {len(saved_paths)} figure sets in '{output_dir}/'")
    for name, path in saved_paths.items():
        print(f"    {name}: {path}")

    plt.close("all")
    return saved_paths
