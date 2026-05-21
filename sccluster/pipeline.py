"""
End-to-end analysis pipeline orchestrator.

SCClusterPipeline encapsulates the complete workflow:
  preprocessing → scVI → clustering → markers → visualization
"""

from __future__ import annotations

import os
import json
import time
import warnings
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from sccluster.preprocessing import preprocess_pipeline
from sccluster.dim_reduction import scvi_pipeline
from sccluster.clustering import cluster_pipeline
from sccluster.markers import marker_pipeline
from sccluster.visualization import generate_all_figures
from sccluster.utils import (
    setup_logging,
    save_results,
    summarize_clusters,
    export_markers_to_csv,
)


class SCClusterPipeline:
    """
    Single-cell RNA-seq clustering pipeline using scVI + Leiden + Wilcoxon.

    Workflow:
      1. Preprocessing: QC → Normalization → HVG selection
      2. scVI: Deep latent variable model training
      3. Clustering: kNN graph → Leiden community detection
      4. Markers: Wilcoxon rank-sum test for cluster-specific genes
      5. Visualization: UMAP, volcano, heatmap, dotplot, etc.

    Parameters
    ----------
    adata : AnnData
        Raw expression matrix (cells x genes).
    output_dir : str
        Directory for output files, figures, and saved models.
    random_seed : int
        Random seed for reproducibility.

    Examples
    --------
    >>> from sccluster import SCClusterPipeline
    >>> pipeline = SCClusterPipeline(adata, output_dir="./results")
    >>> pipeline.run()
    """

    def __init__(
        self,
        adata: AnnData,
        output_dir: str = "./sccluster_results",
        random_seed: int = 42,
        verbose: bool = True,
    ):
        self.adata = adata
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        self.verbose = verbose

        # Internal state
        self.model = None
        self.top_markers = None
        self.filtered_markers = None

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir = self.output_dir / "figures"
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.output_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Set random seeds
        np.random.seed(random_seed)
        import torch
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    # ── Configuration profiles ──────────────────────────────

    def _get_config(
        self,
        preset: Optional[str] = None,
        custom_config: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Dict]:
        """Build configuration from preset and/or custom overrides."""
        presets = {
            "default": {
                "preprocessing": {
                    "filter_kwargs": {
                        "min_genes": 200,
                        "min_cells": 3,
                        "pct_mt_threshold": 20.0,
                    },
                    "normalize_kwargs": {"target_sum": 1e4},
                    "hvg_kwargs": {
                        "n_top_genes": 2000,
                        "flavor": "seurat_v3",
                    },
                },
                "scvi": {
                    "model_kwargs": {
                        "n_latent": 30,
                        "n_layers": 2,
                        "n_hidden": 128,
                        "dropout_rate": 0.1,
                        "dispersion": "gene",
                        "gene_likelihood": "zinb",
                        "max_epochs": 400,
                        "early_stopping": True,
                        "early_stopping_patience": 45,
                    },
                },
                "clustering": {
                    "neighbors_kwargs": {"n_neighbors": 15},
                    "leiden_kwargs": {"resolution": 1.0, "n_iterations": 2},
                    "umap_kwargs": {"min_dist": 0.3, "spread": 1.0},
                },
                "markers": {
                    "rank_kwargs": {
                        "method": "wilcoxon",
                        "tie_correct": True,
                        "corr_method": "benjamini-hochberg",
                    },
                },
            },
            "high_resolution": {
                "preprocessing": {
                    "filter_kwargs": {
                        "min_genes": 500,
                        "min_cells": 5,
                        "pct_mt_threshold": 10.0,
                    },
                    "hvg_kwargs": {"n_top_genes": 3000, "flavor": "seurat_v3"},
                },
                "scvi": {
                    "model_kwargs": {
                        "n_latent": 50,
                        "n_layers": 3,
                        "n_hidden": 256,
                        "dropout_rate": 0.2,
                        "max_epochs": 500,
                        "early_stopping_patience": 60,
                    },
                },
                "clustering": {
                    "neighbors_kwargs": {"n_neighbors": 30},
                    "leiden_kwargs": {"resolution": 2.0, "n_iterations": -1},
                    "umap_kwargs": {"min_dist": 0.1, "spread": 1.5},
                },
                "markers": {
                    "rank_kwargs": {
                        "method": "wilcoxon",
                        "tie_correct": True,
                        "corr_method": "benjamini-hochberg",
                    },
                },
            },
            "quick": {
                "preprocessing": {
                    "hvg_kwargs": {"n_top_genes": 1000, "flavor": "seurat"},
                },
                "scvi": {
                    "model_kwargs": {
                        "n_latent": 10,
                        "n_layers": 1,
                        "n_hidden": 64,
                        "max_epochs": 100,
                        "early_stopping_patience": 20,
                    },
                },
                "clustering": {
                    "neighbors_kwargs": {"n_neighbors": 10},
                    "leiden_kwargs": {"resolution": 0.8, "n_iterations": 1},
                    "umap_kwargs": {"min_dist": 0.5},
                },
                "markers": {
                    "rank_kwargs": {"method": "t-test"},
                },
            },
        }

        if preset is not None and preset not in presets:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {list(presets.keys())}"
            )

        config = presets.get(preset, presets["default"]).copy()

        # Deep merge custom config
        if custom_config:
            for section in custom_config:
                if section in config:
                    config[section].update(custom_config[section])
                else:
                    config[section] = custom_config[section]

        return config

    # ── Core pipeline steps ─────────────────────────────────

    def run(
        self,
        preset: Optional[str] = "default",
        custom_config: Optional[Dict[str, Dict]] = None,
        skip_preprocessing: bool = False,
        skip_scvi: bool = False,
        skip_clustering: bool = False,
        skip_markers: bool = False,
        skip_viz: bool = False,
        save_adata: bool = True,
    ) -> AnnData:
        """
        Execute the complete analysis pipeline.

        Parameters
        ----------
        preset : str, optional
            Configuration preset: 'default', 'high_resolution', or 'quick'.
        custom_config : dict, optional
            Custom configuration overrides (deep-merged with preset).
        skip_preprocessing : bool
            Skip preprocessing (use if already preprocessed).
        skip_scvi : bool
            Skip scVI training (use if model already trained).
        skip_clustering : bool
            Skip clustering (use if Leiden already run).
        skip_markers : bool
            Skip marker gene identification.
        skip_viz : bool
            Skip visualization generation.
        save_adata : bool
            Save the final AnnData object to disk.

        Returns
        -------
        AnnData
            Processed AnnData with all analysis results.
        """
        start_time = time.time()
        config = self._get_config(preset, custom_config)

        if self.verbose:
            print("=" * 70)
            print("  scCluster: scRNA-seq Analysis Pipeline")
            print("  scVI + Leiden + Wilcoxon")
            print("=" * 70)
            print(f"  Cells: {self.adata.n_obs:,}  |  Genes: {self.adata.n_vars:,}")
            print(f"  Preset: {preset}  |  Seed: {self.random_seed}")
            print(f"  Output: {self.output_dir}")
            print("=" * 70)

        # ── Step 1: Preprocessing ──
        if not skip_preprocessing:
            preprocess_start = time.time()
            self.adata = preprocess_pipeline(
                self.adata,
                **config.get("preprocessing", {}),
                verbose=self.verbose,
            )
            if self.verbose:
                print(f"[Timing] Preprocessing: {time.time() - preprocess_start:.1f}s")
        else:
            if self.verbose:
                print("[Pipeline] Skipping preprocessing (skip_preprocessing=True)")

        # ── Step 2: scVI ──
        if not skip_scvi:
            scvi_start = time.time()
            scvi_cfg = config.get("scvi", {})
            scvi_cfg.setdefault("save_model_path", str(self.model_dir / "scvi_model"))
            self.model = scvi_pipeline(
                self.adata,
                verbose=self.verbose,
                **scvi_cfg,
            )
            if self.verbose:
                print(f"[Timing] scVI: {time.time() - scvi_start:.1f}s")
        else:
            if self.verbose:
                print("[Pipeline] Skipping scVI (skip_scvi=True)")

        # ── Step 3: Clustering ──
        if not skip_clustering:
            cluster_start = time.time()
            self.adata = cluster_pipeline(
                self.adata,
                verbose=self.verbose,
                **config.get("clustering", {}),
            )
            if self.verbose:
                print(f"[Timing] Clustering: {time.time() - cluster_start:.1f}s")
        else:
            if self.verbose:
                print("[Pipeline] Skipping clustering (skip_clustering=True)")

        # ── Step 4: Markers ──
        if not skip_markers:
            markers_start = time.time()
            self.adata, self.filtered_markers, self.top_markers = marker_pipeline(
                self.adata,
                verbose=self.verbose,
                **config.get("markers", {}),
            )
            if self.verbose:
                print(f"[Timing] Markers: {time.time() - markers_start:.1f}s")
        else:
            if self.verbose:
                print("[Pipeline] Skipping markers (skip_markers=True)")

        # ── Step 5: Visualization ──
        if not skip_viz:
            viz_start = time.time()
            generate_all_figures(
                self.adata,
                output_dir=str(self.figure_dir),
                cluster_key="leiden",
            )
            if self.verbose:
                print(f"[Timing] Visualization: {time.time() - viz_start:.1f}s")
        else:
            if self.verbose:
                print("[Pipeline] Skipping visualization (skip_viz=True)")

        # ── Export results ──
        if save_adata:
            adata_path = self.output_dir / "processed_adata.h5ad"
            self.adata.write(adata_path)
            if self.verbose:
                print(f"\n[Pipeline] AnnData saved to: {adata_path}")

        if self.top_markers is not None and len(self.top_markers) > 0:
            marker_csv = self.output_dir / "top_markers.csv"
            export_markers_to_csv(self.top_markers, str(marker_csv))
            marker_excel = self.output_dir / "markers_per_cluster.xlsx"
            try:
                from sccluster.utils import export_markers_to_excel
                export_markers_to_excel(self.filtered_markers, str(marker_excel))
            except Exception:
                pass

        # ── Summary ──
        total_time = time.time() - start_time
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"  Pipeline complete! Total time: {total_time:.1f}s "
                  f"({total_time/60:.1f} min)")
            print("=" * 70)
            self.summary()

        # Save run config
        config_path = self.output_dir / "run_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "preset": preset,
                "config": {k: v for k, v in config.items() if k in ("preprocessing", "scvi", "clustering", "markers")},
                "random_seed": self.random_seed,
                "total_time_seconds": total_time,
                "n_cells": int(self.adata.n_obs),
                "n_genes": int(self.adata.n_vars),
                "n_clusters": int(self.adata.obs.get("leiden", pd.Series()).nunique()) if "leiden" in self.adata.obs else 0,
            }, f, indent=2, default=str)

        return self.adata

    def summary(self) -> None:
        """Print a summary of the analysis results."""
        if "leiden" not in self.adata.obs.columns:
            print("[Summary] No clustering results found. Run the pipeline first.")
            return
        summarize_clusters(self.adata, groupby="leiden")

    def get_cluster_markers(self, cluster: str, n: int = 20) -> pd.DataFrame:
        """Get marker genes for a specific cluster."""
        if self.filtered_markers is None:
            raise RuntimeError("No marker results. Run the pipeline first.")
        if cluster not in self.filtered_markers:
            raise ValueError(f"Cluster '{cluster}' not found.")
        return self.filtered_markers[cluster].head(n)

    def save_state(self, path: Optional[str] = None) -> str:
        """Save the full AnnData object to disk."""
        if path is None:
            path = str(self.output_dir / "final_adata.h5ad")
        self.adata.write(path)
        print(f"[Pipeline] State saved to: {path}")
        return path
