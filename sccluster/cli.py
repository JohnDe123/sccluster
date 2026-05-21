"""
Command-line interface for the scCluster pipeline.

Usage:
    sccluster run input.h5ad -o ./results
    sccluster run input.h5ad -o ./results --preset high_resolution
    sccluster run input.csv -o ./results --format csv --transpose
    sccluster synthetic -o ./test_data.h5ad --n-cells 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for CLI commands."""
    parser = argparse.ArgumentParser(
        prog="sccluster",
        description="scCluster: scRNA-seq clustering pipeline (scVI + Leiden + Wilcoxon)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on H5AD file
  sccluster run data.h5ad -o ./results

  # High-resolution mode
  sccluster run data.h5ad -o ./results --preset high_resolution

  # Quick test
  sccluster run data.h5ad -o ./results --preset quick

  # From CSV (genes x cells, transpose needed)
  sccluster run matrix.csv -o ./results --format csv --transpose

  # Generate synthetic test data
  sccluster synthetic -o test_data.h5ad --n-cells 2000

  # Resume (skip finished steps)
  sccluster run data.h5ad -o ./results --skip-preprocessing --skip-scvi
        """,
    )

    subparsers = parser.add_subparsers(dest="command", title="Commands")

    # ── `run` command ──────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Run the full clustering pipeline")
    run_parser.add_argument("input", help="Path to input file (h5ad, csv, tsv, h5, mtx, loom)")
    run_parser.add_argument("-o", "--output-dir", default="./sccluster_results",
                            help="Output directory (default: ./sccluster_results)")
    run_parser.add_argument("--format", choices=["h5ad", "csv", "tsv", "10x-h5", "mtx", "loom"],
                            help="Input file format (auto-detected if omitted)")
    run_parser.add_argument("--transpose", action="store_true",
                            help="Transpose input matrix (useful for CSV with genes as rows)")
    run_parser.add_argument("--preset", choices=["default", "high_resolution", "quick"],
                            default="default", help="Analysis preset (default: default)")
    run_parser.add_argument("--batch-key", default=None,
                            help="Key in .obs for batch labels (enables batch correction)")
    run_parser.add_argument("--n-latent", type=int, default=30,
                            help="scVI latent dimension (default: 30)")
    run_parser.add_argument("--n-hvg", type=int, default=2000,
                            help="Number of highly variable genes (default: 2000)")
    run_parser.add_argument("--resolution", type=float, default=1.0,
                            help="Leiden resolution (default: 1.0)")
    run_parser.add_argument("--max-epochs", type=int, default=400,
                            help="Max scVI training epochs (default: 400)")
    run_parser.add_argument("--seed", type=int, default=42,
                            help="Random seed (default: 42)")
    run_parser.add_argument("--gpu", action="store_true", default=True,
                            help="Use GPU if available (default: True)")
    run_parser.add_argument("--no-gpu", action="store_false", dest="gpu",
                            help="Force CPU even if GPU available")

    # Skip flags
    skip_group = run_parser.add_argument_group("Skip options")
    skip_group.add_argument("--skip-preprocessing", action="store_true",
                            help="Skip preprocessing step")
    skip_group.add_argument("--skip-scvi", action="store_true",
                            help="Skip scVI training step")
    skip_group.add_argument("--skip-clustering", action="store_true",
                            help="Skip clustering step")
    skip_group.add_argument("--skip-markers", action="store_true",
                            help="Skip marker gene step")
    skip_group.add_argument("--skip-viz", action="store_true",
                            help="Skip visualization step")

    # ── `synthetic` command ─────────────────────────────────
    syn_parser = subparsers.add_parser("synthetic", help="Generate synthetic test data")
    syn_parser.add_argument("-o", "--output", default="synthetic_data.h5ad",
                            help="Output path (default: synthetic_data.h5ad)")
    syn_parser.add_argument("--n-cells", type=int, default=1000,
                            help="Number of cells (default: 1000)")
    syn_parser.add_argument("--n-genes", type=int, default=5000,
                            help="Number of genes (default: 5000)")
    syn_parser.add_argument("--n-clusters", type=int, default=5,
                            help="Number of true cell types (default: 5)")
    syn_parser.add_argument("--n-batches", type=int, default=2,
                            help="Number of batches (default: 2)")
    syn_parser.add_argument("--seed", type=int, default=42,
                            help="Random seed (default: 42)")

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the `run` command."""
    from sccluster.utils import load_data
    from sccluster.pipeline import SCClusterPipeline

    print(f"scCluster v1.0.0")
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")

    # Load data
    adata = load_data(
        args.input,
        format=args.format,
        transpose=args.transpose,
    )

    # Build custom config from CLI args
    custom_config = {
        "preprocessing": {
            "hvg_kwargs": {"n_top_genes": args.n_hvg},
        },
        "scvi": {
            "model_kwargs": {
                "n_latent": args.n_latent,
                "max_epochs": args.max_epochs,
                "use_cuda": args.gpu,
            },
            "setup_kwargs": {
                "batch_key": args.batch_key,
            } if args.batch_key else {},
        },
        "clustering": {
            "leiden_kwargs": {"resolution": args.resolution},
        },
    }

    # Run pipeline
    pipeline = SCClusterPipeline(
        adata,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )

    pipeline.run(
        preset=args.preset,
        custom_config=custom_config,
        skip_preprocessing=args.skip_preprocessing,
        skip_scvi=args.skip_scvi,
        skip_clustering=args.skip_clustering,
        skip_markers=args.skip_markers,
        skip_viz=args.skip_viz,
    )

    return 0


def _cmd_synthetic(args: argparse.Namespace) -> int:
    """Execute the `synthetic` command."""
    from sccluster.utils import generate_synthetic_data

    print(f"Generating synthetic scRNA-seq data...")
    print(f"  n_cells={args.n_cells}, n_genes={args.n_genes}")
    print(f"  n_clusters={args.n_clusters}, n_batches={args.n_batches}")

    adata = generate_synthetic_data(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_clusters=args.n_clusters,
        n_batches=args.n_batches,
        random_seed=args.seed,
    )

    adata.write(args.output)
    print(f"Synthetic data saved to: {args.output}")
    return 0


def main(args=None) -> int:
    """Entry point for the sccluster CLI."""
    parser = _create_parser()

    if args is None:
        args = sys.argv[1:]

    if len(args) == 0:
        parser.print_help()
        return 0

    parsed = parser.parse_args(args)

    if parsed.command == "run":
        return _cmd_run(parsed)
    elif parsed.command == "synthetic":
        return _cmd_synthetic(parsed)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
