"""
Marker gene identification module.

Uses Wilcoxon rank-sum test (non-parametric) to identify cluster-specific
marker genes. The Wilcoxon test is robust to outliers and does not assume
normality, making it well-suited for scRNA-seq count data.
"""

from __future__ import annotations

import warnings
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.stats import wilcoxon


def rank_marker_genes(
    adata: AnnData,
    groupby: str = "leiden",
    groups: str = "all",
    reference: str = "rest",
    method: str = "wilcoxon",
    n_genes: Optional[int] = None,
    layer: Optional[str] = None,
    use_raw: bool = True,
    tie_correct: bool = True,
    corr_method: str = "benjamini-hochberg",
    pts: bool = True,
    key_added: str = "rank_genes_groups",
    copy: bool = False,
    **kwargs,
) -> AnnData:
    """
    Rank genes by differential expression using Wilcoxon rank-sum test.

    For each cluster, tests H0: gene expression in cluster = gene expression
    in rest of cells (or specified reference). The Wilcoxon test ranks all
    cells by gene expression and tests whether the ranks are systematically
    higher in the target cluster.

    Parameters
    ----------
    adata : AnnData
        Log-normalized expression data.
    groupby : str
        Key in adata.obs for cluster labels.
    groups : str or list
        Cluster(s) to test. 'all' tests every cluster.
    reference : str
        Reference group: 'rest' (all other cells) or a specific cluster label.
    method : str
        Statistical test: 'wilcoxon', 't-test', 't-test_overestim_var',
        'logreg'.
    n_genes : int, optional
        Number of top genes to store per group. Store all if None.
    layer : str, optional
        Use data from adata.layers[layer] for testing.
    use_raw : bool
        Use adata.raw for testing (if available).
    tie_correct : bool
        Apply tie correction to the Wilcoxon test.
    corr_method : str
        Multiple testing correction: 'benjamini-hochberg', 'bonferroni', etc.
    pts : bool
        Compute fraction of cells expressing each gene per group.
    key_added : str
        Key under which results are stored in adata.uns.
    copy : bool
        Whether to return a copy.
    **kwargs
        Additional arguments passed to sc.tl.rank_genes_groups.

    Returns
    -------
    AnnData
        AnnData with ranked genes stored in .uns[key_added].
    """
    adata = adata.copy() if copy else adata

    n_clusters = adata.obs[groupby].nunique()
    print(f"[Markers] Ranking genes per cluster using {method.upper()} test")
    print(f"[Markers] groupby='{groupby}', n_clusters={n_clusters}, reference='{reference}'")

    # Check input data
    if use_raw and adata.raw is not None:
        data_source = "adata.raw"
    elif layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers. Available: {list(adata.layers.keys())}")
        data_source = f"adata.layers['{layer}']"
    else:
        data_source = "adata.X"

    print(f"[Markers] Data source: {data_source}")

    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        groups=groups,
        reference=reference,
        method=method,
        n_genes=n_genes,
        layer=layer,
        use_raw=use_raw,
        tie_correct=tie_correct,
        corr_method=corr_method,
        pts=pts,
        key_added=key_added,
        **kwargs,
    )

    # Summarize top markers per cluster
    result = adata.uns[key_added]
    print(f"\n[Markers] Top 5 markers per cluster:")
    for cluster in result["names"].dtype.names:
        top5 = result["names"][cluster][:5]
        scores = result["scores"][cluster][:5]
        marker_str = ", ".join(
            f"{g}({s:.1f})" for g, s in zip(top5, scores)
        )
        print(f"    Cluster {cluster}: {marker_str}")

    return adata


def filter_significant_markers(
    adata: AnnData,
    key: str = "rank_genes_groups",
    groupby: str = "leiden",
    pval_cutoff: float = 0.05,
    logfc_cutoff: float = 1.0,
    min_pct_in_group: float = 0.25,
    min_pct_out_group: Optional[float] = None,
    n_top: int = 50,
) -> Dict[str, pd.DataFrame]:
    """
    Filter marker genes by significance, fold-change, and expression thresholds.

    Parameters
    ----------
    adata : AnnData
        AnnData with rank_genes_groups results in .uns.
    key : str
        Key in adata.uns for rank_genes_groups results.
    groupby : str
        Key in adata.obs for cluster labels.
    pval_cutoff : float
        Maximum adjusted p-value threshold.
    logfc_cutoff : float
        Minimum absolute log2 fold-change threshold.
    min_pct_in_group : float
        Minimum fraction of cells in the cluster that must express the gene.
    min_pct_out_group : float, optional
        Maximum fraction of cells outside the cluster expressing the gene.
    n_top : int
        Maximum number of markers to return per cluster.

    Returns
    -------
    dict of DataFrame
        Dictionary mapping cluster labels to DataFrames of filtered markers,
        with columns: ['names', 'scores', 'logfoldchanges', 'pvals',
        'pvals_adj', 'pct_in_group', 'pct_out_group'].
    """
    result = adata.uns[key]
    clusters = result["names"].dtype.names
    filtered_markers = {}

    print(f"[Filter] Filtering markers: p_adj < {pval_cutoff}, "
          f"|logFC| > {logfc_cutoff}")

    for cluster in clusters:
        df = pd.DataFrame({
            "gene": result["names"][cluster],
            "scores": result["scores"][cluster],
            "logfoldchanges": result["logfoldchanges"][cluster],
            "pvals": result["pvals"][cluster],
            "pvals_adj": result["pvals_adj"][cluster],
        })

        # Add expression percentages if available
        if "pts" in result:
            pts = result["pts"][cluster]
            df["pct_in_group"] = pts[:, 0] / 100.0
            df["pct_out_group"] = pts[:, 1] / 100.0

        # Apply filters
        mask = (
            (df["pvals_adj"] < pval_cutoff) &
            (df["logfoldchanges"].abs() > logfc_cutoff)
        )
        if "pct_in_group" in df.columns:
            mask = mask & (df["pct_in_group"] >= min_pct_in_group)
        if min_pct_out_group is not None and "pct_out_group" in df.columns:
            mask = mask & (df["pct_out_group"] <= min_pct_out_group)

        df_filtered = df[mask].head(n_top).copy()
        filtered_markers[cluster] = df_filtered
        print(f"    Cluster {cluster}: {len(df_filtered)} significant markers "
              f"(from {len(df)} tested)")

    return filtered_markers


def extract_top_markers(
    adata: AnnData,
    key: str = "rank_genes_groups",
    n_per_cluster: int = 10,
    min_logfc: float = 0.5,
    min_pval: float = 0.05,
    sort_by: str = "scores",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Extract the top N marker genes per cluster as a flat DataFrame.

    Parameters
    ----------
    adata : AnnData
        AnnData with rank_genes_groups results.
    key : str
        Key in adata.uns for marker results.
    n_per_cluster : int
        Number of top markers per cluster.
    min_logfc : float
        Minimum log2 fold-change for inclusion.
    min_pval : float
        Maximum adjusted p-value for inclusion.
    sort_by : str
        Column to sort by: 'scores', 'logfoldchanges', or 'pvals_adj'.
    ascending : bool
        Whether to sort in ascending order.

    Returns
    -------
    DataFrame
        Long-format DataFrame with columns [cluster, gene, score, logFC,
        pval, pval_adj].
    """
    result = adata.uns[key]
    clusters = result["names"].dtype.names
    all_markers = []

    for cluster in clusters:
        names = result["names"][cluster]
        scores = result["scores"][cluster]
        logfc = result["logfoldchanges"][cluster]
        pvals = result["pvals"][cluster]
        pvals_adj = result["pvals_adj"][cluster]

        for i in range(min(n_per_cluster * 2, len(names))):
            if abs(logfc[i]) >= min_logfc and pvals_adj[i] < min_pval:
                all_markers.append({
                    "cluster": cluster,
                    "gene": names[i],
                    "score": scores[i],
                    "logFC": logfc[i],
                    "pval": pvals[i],
                    "pval_adj": pvals_adj[i],
                })

    df = pd.DataFrame(all_markers)

    if len(df) == 0:
        print("[Markers] Warning: No markers pass the thresholds.")
        return df

    # Sort and take top N per cluster
    sort_asc = sort_by in ("pvals_adj", "pval")
    df = df.sort_values(["cluster", sort_by], ascending=[True, sort_asc])
    df = df.groupby("cluster").head(n_per_cluster).reset_index(drop=True)

    print(f"[Markers] Extracted {len(df)} markers across {df['cluster'].nunique()} clusters")
    return df


def pairwise_marker_comparison(
    adata: AnnData,
    gene: str,
    groupby: str = "leiden",
    use_raw: bool = True,
) -> pd.DataFrame:
    """
    Perform pairwise Wilcoxon comparison for a single gene across all clusters.

    Useful for detailed examination of a candidate marker gene.

    Parameters
    ----------
    adata : AnnData
        Expression data.
    gene : str
        Gene name to test.
    groupby : str
        Cluster labels in adata.obs.
    use_raw : bool
        Whether to use raw expression data.

    Returns
    -------
    DataFrame
        Pairwise comparison matrix (p-values) between all cluster pairs.
    """
    clusters = sorted(adata.obs[groupby].unique().tolist())
    n_clusters = len(clusters)

    if use_raw and adata.raw is not None:
        expr = adata.raw[:, gene].X.toarray().flatten() if hasattr(adata.raw[:, gene].X, "toarray") else adata.raw[:, gene].X.flatten()
    else:
        if gene not in adata.var_names:
            raise ValueError(f"Gene '{gene}' not found in adata.var_names")
        expr = adata[:, gene].X.toarray().flatten() if hasattr(adata[:, gene].X, "toarray") else adata[:, gene].X.flatten()

    pval_matrix = pd.DataFrame(
        np.ones((n_clusters, n_clusters)),
        index=clusters, columns=clusters
    )

    mean_expr = {}
    for c in clusters:
        mask = (adata.obs[groupby].values == c)
        mean_expr[c] = expr[mask].mean()

    print(f"[Pairwise] Gene: {gene}")
    for i, ci in enumerate(clusters):
        expr_i = expr[adata.obs[groupby].values == ci]
        for j, cj in enumerate(clusters):
            if i >= j:
                continue
            expr_j = expr[adata.obs[groupby].values == cj]
            try:
                _, pval = wilcoxon(expr_i, expr_j, zero_method="wilcox")
                pval_matrix.loc[ci, cj] = pval
                pval_matrix.loc[cj, ci] = pval
            except Exception:
                pval_matrix.loc[ci, cj] = 1.0
                pval_matrix.loc[cj, ci] = 1.0

    print(f"  Mean expression: " + ", ".join(
        f"c{c}: {mean_expr[c]:.2f}" for c in clusters
    ))

    return pval_matrix


def marker_pipeline(
    adata: AnnData,
    groupby: str = "leiden",
    rank_kwargs: Optional[Dict[str, Any]] = None,
    filter_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[AnnData, Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Run the full marker gene identification pipeline.

    Parameters
    ----------
    adata : AnnData
        AnnData with cluster labels and log-normalized expression.
    groupby : str
        Key in adata.obs for cluster labels.
    rank_kwargs : dict, optional
        Keyword arguments for rank_marker_genes().
    filter_kwargs : dict, optional
        Keyword arguments for filter_significant_markers().
    verbose : bool
        Print progress messages.

    Returns
    -------
    adata : AnnData
        AnnData with marker results stored internally.
    filtered : dict of DataFrame
        Filtered markers per cluster.
    top_markers : DataFrame
        Top markers in flat format for easy export.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 4: MARKER GENE IDENTIFICATION (Wilcoxon)")
        print("=" * 60)

    # Rank genes
    if verbose:
        print("\n--- 4.1 Rank Genes ---")
    rkwargs = {"groupby": groupby}
    rkwargs.update(rank_kwargs or {})
    adata = rank_marker_genes(adata, **rkwargs)

    # Filter significant markers
    if verbose:
        print("\n--- 4.2 Filter Markers ---")
    fkwargs = filter_kwargs or {}
    filtered = filter_significant_markers(adata, **fkwargs)

    # Extract top markers
    if verbose:
        print("\n--- 4.3 Extract Top Markers ---")
    top_markers = extract_top_markers(adata)

    if verbose:
        print(f"\n[Markers] Complete: {len(top_markers)} top markers identified")

    return adata, filtered, top_markers
