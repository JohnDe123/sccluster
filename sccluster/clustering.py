"""
Graph-based clustering module.

Builds a k-nearest neighbor (kNN) graph on the scVI latent space
and performs Leiden community detection for cell clustering.

Leiden algorithm guarantees well-connected communities and is more
robust than Louvain, finding better partitions with comparable speed.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData


def _validate_latent_key(adata: AnnData, latent_key: str) -> None:
    """Validate that the latent representation key exists in adata.obsm."""
    if latent_key not in adata.obsm:
        available = list(adata.obsm.keys())
        raise KeyError(
            f"Latent key '{latent_key}' not found in adata.obsm. "
            f"Available keys: {available}. "
            f"Run scvi_pipeline() or get_latent_representation() first."
        )


def build_neighbors_graph(
    adata: AnnData,
    latent_key: str = "X_scVI",
    n_neighbors: int = 15,
    n_pcs: int = 0,  # 0 = use all dimensions
    metric: str = "euclidean",
    method: str = "umap",
    random_state: int = 42,
    copy: bool = False,
) -> AnnData:
    """
    Build a k-nearest neighbor graph from the scVI latent representation.

    The graph topology captures the manifold structure of the latent space,
    which preserves biological similarity between cells.

    Parameters
    ----------
    adata : AnnData
        AnnData with latent representation in .obsm[latent_key].
    latent_key : str
        Key in adata.obsm for the latent representation.
    n_neighbors : int
        Number of neighbors for the kNN graph.
        Lower values (5-10): finer resolution, more noise sensitivity.
        Higher values (20-50): coarser, smoother manifold.
    n_pcs : int
        Number of PCs to use from latent space. 0 = use all dimensions.
    metric : str
        Distance metric: 'euclidean', 'cosine', 'manhattan', etc.
    method : str
        kNN computation method: 'umap' (default, pynndescent) or 'gauss'.
    random_state : int
        Random seed for reproducibility.
    copy : bool
        Whether to return a copy.

    Returns
    -------
    AnnData
        AnnData with neighbor graph in .obsp and neighbors in .uns['neighbors'].
    """
    _validate_latent_key(adata, latent_key)
    adata = adata.copy() if copy else adata

    n_latent_dims = adata.obsm[latent_key].shape[1]
    print(f"[Neighbors] Building kNN graph from {n_latent_dims}-dimensional latent space")
    print(f"[Neighbors] n_neighbors={n_neighbors}, metric='{metric}', method='{method}'")

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=latent_key,
        metric=metric,
        method=method,
        random_state=random_state,
    )

    n_edges = (
        adata.obsp["connectivities"].nnz
        if hasattr(adata.obsp["connectivities"], "nnz")
        else np.count_nonzero(adata.obsp["connectivities"])
    )
    print(f"[Neighbors] Graph constructed: {n_edges} edges (sparse)")

    return adata


def leiden_clustering(
    adata: AnnData,
    resolution: float = 1.0,
    n_iterations: int = 2,
    key_added: str = "leiden",
    adjacency: Optional[Any] = None,
    use_weights: bool = True,
    random_state: int = 42,
    partition_type: Optional[Any] = None,
    copy: bool = False,
) -> AnnData:
    """
    Perform Leiden clustering on the neighborhood graph.

    The Leiden algorithm iteratively refines partitions:
    1. Local moving: greedily optimize modularity
    2. Refinement: split communities to ensure well-connectedness
    3. Aggregation: build reduced graph, then repeat

    This guarantees that all communities are:
    - Well-connected (internally)
    - Subpartition γ-separated (locally optimal)
    - Subset optimal (no subset can be moved to improve modularity)

    Parameters
    ----------
    adata : AnnData
        AnnData with neighbor graph (from build_neighbors_graph).
    resolution : float
        Resolution parameter for the Leiden algorithm.
        Higher values → more clusters, finer granularity.
        Lower values → fewer clusters, coarser granularity.
        Typical range: 0.5 - 2.0.
    n_iterations : int
        Number of iterations of the Leiden algorithm (-1 = run until convergence).
    key_added : str
        Key under which to store the clustering result in adata.obs.
    adjacency : sparse matrix, optional
        Custom adjacency matrix. Uses connectivities from neighbors if None.
    use_weights : bool
        Whether to use edge weights from the kNN graph.
    random_state : int
        Random seed for reproducible results.
    partition_type : optional
        Partition type for leidenalg. None uses default RBConfigurationVertexPartition.
    copy : bool
        Whether to return a copy.

    Returns
    -------
    AnnData
        AnnData with cluster labels in .obs[key_added].
    """
    adata = adata.copy() if copy else adata

    if adjacency is None and "connectivities" not in adata.obsp:
        raise ValueError(
            "No neighbor graph found. Run build_neighbors_graph() first."
        )

    print(f"[Leiden] Running Leiden clustering with resolution={resolution}")
    print(f"[Leiden] n_iterations={n_iterations}")

    sc.tl.leiden(
        adata,
        resolution=resolution,
        n_iterations=n_iterations,
        key_added=key_added,
        adjacency=adjacency,
        use_weights=use_weights,
        random_state=random_state,
        partition_type=partition_type,
    )

    n_clusters = adata.obs[key_added].nunique()
    cluster_sizes = adata.obs[key_added].value_counts().sort_index()

    print(f"[Leiden] Found {n_clusters} clusters")
    print(f"[Leiden] Cluster sizes: min={cluster_sizes.min()}, "
          f"median={cluster_sizes.median():.0f}, max={cluster_sizes.max()}")

    return adata


def multi_resolution_leiden(
    adata: AnnData,
    resolutions: List[float],
    key_prefix: str = "leiden_r",
    n_iterations: int = 2,
    random_state: int = 42,
    copy: bool = False,
) -> AnnData:
    """
    Run Leiden clustering at multiple resolutions for comparison.

    Useful for choosing the best resolution by examining how cluster
    assignments change across the resolution spectrum.

    Parameters
    ----------
    adata : AnnData
        AnnData with neighbor graph.
    resolutions : list of float
        List of resolution values to try.
    key_prefix : str
        Prefix for storing results in adata.obs (e.g., 'leiden_r1.0').
    n_iterations : int
        Number of Leiden iterations.
    random_state : int
        Random seed.
    copy : bool
        Whether to return a copy.

    Returns
    -------
    AnnData
        AnnData with multiple clustering results.
    """
    adata = adata.copy() if copy else adata

    n_clusters_list = []
    for res in resolutions:
        key = f"{key_prefix}{res}"
        adata = leiden_clustering(
            adata,
            resolution=res,
            key_added=key,
            n_iterations=n_iterations,
            random_state=random_state,
            copy=False,
        )
        n_clusters_list.append(adata.obs[key].nunique())

    print(f"\n[Leiden multi-res] Resolution → Clusters:")
    for res, nc in zip(resolutions, n_clusters_list):
        print(f"    res={res:.1f} → {nc} clusters")

    return adata


def compute_umap(
    adata: AnnData,
    latent_key: str = "X_scVI",
    n_components: int = 2,
    min_dist: float = 0.3,
    spread: float = 1.0,
    n_neighbors: Optional[int] = None,
    metric: str = "euclidean",
    random_state: int = 42,
    copy: bool = False,
) -> AnnData:
    """
    Compute UMAP embedding from the scVI latent space for visualization.

    Parameters
    ----------
    adata : AnnData
        AnnData with latent representation and/or neighbor graph.
    latent_key : str
        Key in adata.obsm for the input representation.
    n_components : int
        Number of UMAP dimensions (typically 2 for visualization).
    min_dist : float
        Minimum distance between points in the UMAP embedding.
        Lower values → tighter local clustering.
    spread : float
        Effective scale of the embedding.
    n_neighbors : int, optional
        Number of neighbors. Uses existing connectivities if available.
    metric : str
        Distance metric for UMAP.
    random_state : int
        Random seed.
    copy : bool
        Whether to return a copy.

    Returns
    -------
    AnnData
        AnnData with UMAP coordinates in .obsm['X_umap'].
    """
    _validate_latent_key(adata, latent_key)
    adata = adata.copy() if copy else adata

    print(f"[UMAP] Computing UMAP from '{latent_key}'")
    print(f"[UMAP] n_components={n_components}, min_dist={min_dist}, spread={spread}")

    sc.tl.umap(
        adata,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )

    print(f"[UMAP] UMAP coordinates stored in adata.obsm['X_umap']")
    return adata


def cluster_pipeline(
    adata: AnnData,
    latent_key: str = "X_scVI",
    neighbors_kwargs: Optional[Dict[str, Any]] = None,
    leiden_kwargs: Optional[Dict[str, Any]] = None,
    umap_kwargs: Optional[Dict[str, Any]] = None,
    multi_res: Optional[List[float]] = None,
    verbose: bool = True,
) -> AnnData:
    """
    Run the full clustering pipeline: kNN graph → Leiden → UMAP.

    Parameters
    ----------
    adata : AnnData
        AnnData with latent representation in .obsm[latent_key].
    latent_key : str
        Key in adata.obsm for the latent representation.
    neighbors_kwargs : dict, optional
        Keyword arguments for build_neighbors_graph().
    leiden_kwargs : dict, optional
        Keyword arguments for leiden_clustering().
    umap_kwargs : dict, optional
        Keyword arguments for compute_umap().
    multi_res : list of float, optional
        If provided, run clustering at multiple resolutions.
    verbose : bool
        Print progress messages.

    Returns
    -------
    AnnData
        AnnData with clustering and UMAP results.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 3: GRAPH-BASED CLUSTERING (Leiden)")
        print("=" * 60)

    # Build kNN graph
    if verbose:
        print("\n--- 3.1 kNN Graph Construction ---")
    nkwargs = neighbors_kwargs or {}
    nkwargs.setdefault("latent_key", latent_key)
    adata = build_neighbors_graph(adata, **nkwargs)

    # Leiden clustering
    if verbose:
        print("\n--- 3.2 Leiden Community Detection ---")
    lkwargs = leiden_kwargs or {}
    adata = leiden_clustering(adata, **lkwargs)

    # Multi-resolution
    if multi_res is not None:
        if verbose:
            print("\n--- 3.3 Multi-Resolution Clustering ---")
        adata = multi_resolution_leiden(adata, resolutions=multi_res)

    # UMAP for visualization
    if verbose:
        print("\n--- 3.4 UMAP Embedding ---")
    ukwargs = umap_kwargs or {}
    ukwargs.setdefault("latent_key", latent_key)
    adata = compute_umap(adata, **ukwargs)

    if verbose:
        print("\n[Clustering] Complete")

    return adata
