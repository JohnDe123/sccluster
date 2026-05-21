"""
Dimensionality reduction module using scVI (single-cell Variational Inference).

scVI is a deep generative model that learns a non-linear, batch-corrected,
denoised latent representation of scRNA-seq data. The latent space captures
biological variability while removing technical noise and batch effects.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Dict, Any

import numpy as np
import scanpy as sc
from anndata import AnnData

import scvi
from scvi.model import SCVI


def setup_scvi(
    adata: AnnData,
    batch_key: Optional[str] = None,
    labels_key: Optional[str] = None,
    categorical_covariate_keys: Optional[list] = None,
    continuous_covariate_keys: Optional[list] = None,
    layer: Optional[str] = None,
    copy: bool = False,
) -> AnnData:
    """
    Register an AnnData object for scVI training.

    Configures which annotations to use as batch labels, covariates,
    and where raw count data resides.

    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData with raw counts accessible via .raw or a layer.
    batch_key : str, optional
        Key in adata.obs for batch annotation.
        Critical for batch correction. If None, no batch correction is applied.
    labels_key : str, optional
        Key in adata.obs for cell-type labels (if available).
    categorical_covariate_keys : list of str, optional
        Keys for additional categorical covariates.
    continuous_covariate_keys : list of str, optional
        Keys for additional continuous covariates.
    layer : str, optional
        Key in adata.layers containing raw counts.
        If None, uses adata.raw.X (recommended) or adata.X.
    copy : bool
        Whether to return a copy.

    Returns
    -------
    AnnData
        AnnData with scVI setup completed.
    """
    adata = adata.copy() if copy else adata

    if layer is None and adata.raw is not None:
        print("[scVI setup] Using adata.raw.X as count data source")
    elif layer is not None:
        print(f"[scVI setup] Using adata.layers['{layer}'] as count data source")
    else:
        print("[scVI setup] Using adata.X as count data source")

    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key=batch_key,
        labels_key=labels_key,
        categorical_covariate_keys=categorical_covariate_keys,
        continuous_covariate_keys=continuous_covariate_keys,
        layer=layer,
    )

    # Log setup info
    print(f"[scVI setup] n_cells={adata.n_obs}, n_genes={adata.n_vars}")
    print(f"[scVI setup] batch_key='{batch_key}', labels_key='{labels_key}'")
    print(f"[scVI setup] categorical covariates: {categorical_covariate_keys or 'none'}")
    print(f"[scVI setup] continuous covariates: {continuous_covariate_keys or 'none'}")

    return adata


def train_scvi_model(
    adata: AnnData,
    n_latent: int = 30,
    n_layers: int = 2,
    n_hidden: int = 128,
    dropout_rate: float = 0.1,
    dispersion: str = "gene",
    gene_likelihood: str = "zinb",
    max_epochs: int = 400,
    early_stopping: bool = True,
    early_stopping_patience: int = 45,
    early_stopping_min_delta: float = 0.001,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-6,
    use_cuda: bool = True,
    random_seed: int = 42,
    train_kwargs: Optional[dict] = None,
    verbose: bool = True,
) -> SCVI:
    """
    Initialize and train a scVI model.

    scVI learns a latent variable model:
      z_n ~ N(0, I)           ... latent cell representation
      x_ng | z_n, s_n ~ ZINB  ... observed gene counts

    where z_n is the low-dimensional embedding and s_n captures
    library size and batch effects.

    Parameters
    ----------
    adata : AnnData
        AnnData with scVI setup already called.
    n_latent : int
        Dimensionality of the latent space (default: 30).
    n_layers : int
        Number of hidden layers in encoder/decoder networks.
    n_hidden : int
        Number of nodes per hidden layer.
    dropout_rate : float
        Dropout rate for regularization during training.
    dispersion : str
        Dispersion parameter specification: "gene", "gene-batch", "gene-label",
        or "gene-cell" (trade-off: more params vs. more flexibility).
    gene_likelihood : str
        Likelihood model: "zinb" (zero-inflated negative binomial),
        "nb" (negative binomial), or "poisson".
    max_epochs : int
        Maximum number of training epochs.
    early_stopping : bool
        Whether to use early stopping on validation loss.
    early_stopping_patience : int
        Number of epochs without improvement before stopping.
    early_stopping_min_delta : float
        Minimum change in loss to qualify as improvement.
    batch_size : int
        Minibatch size for training.
    learning_rate : float
        Learning rate for the Adam optimizer.
    weight_decay : float
        Weight decay (L2 regularization).
    use_cuda : bool
        Whether to use GPU if available.
    random_seed : int
        Random seed for reproducibility.
    train_kwargs : dict, optional
        Additional keyword arguments for model.train().
    verbose : bool
        Print training progress.

    Returns
    -------
    SCVI
        Trained scVI model object.
    """
    if train_kwargs is None:
        train_kwargs = {}

    # Determine device
    if use_cuda:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("[scVI] CUDA requested but not available. Using CPU.")
    else:
        device = "cpu"

    print(f"[scVI train] Device: {device}")
    print(f"[scVI train] Model architecture:")
    print(f"    n_latent={n_latent}, n_layers={n_layers}, n_hidden={n_hidden}")
    print(f"    dropout={dropout_rate}, dispersion={dispersion}")
    print(f"    gene_likelihood={gene_likelihood}")
    print(f"[scVI train] Training config:")
    print(f"    max_epochs={max_epochs}, batch_size={batch_size}")
    print(f"    lr={learning_rate}, weight_decay={weight_decay}")
    print(f"    early_stopping={early_stopping}, patience={early_stopping_patience}")

    # Initialize model
    model = SCVI(
        adata,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        dropout_rate=dropout_rate,
        dispersion=dispersion,
        gene_likelihood=gene_likelihood,
        use_layer_norm="both",
        use_batch_norm="none",
        encode_covariates=True,
        deeply_inject_covariates=False,
    )

    # Train
    train_defaults = dict(
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        batch_size=batch_size,
        train_size=0.9,
        check_val_every_n_epoch=1,
        plan_kwargs=dict(
            lr=learning_rate,
            weight_decay=weight_decay,
            n_epochs_kl_warmup=50,
        ),
    )
    train_defaults.update(train_kwargs)

    model.train(**train_defaults)

    # Log training history
    history = model.history
    if history is not None:
        n_epochs_trained = len(history["elbo_train"])
        final_train_loss = history["elbo_train"].iloc[-1]
        final_val_loss = (
            history["elbo_validation"].iloc[-1]
            if "elbo_validation" in history
            else None
        )
        print(f"[scVI train] Completed {n_epochs_trained} epochs")
        print(f"[scVI train] Final ELBO (train): {final_train_loss:.2f}")
        if final_val_loss is not None:
            print(f"[scVI train] Final ELBO (val):   {final_val_loss:.2f}")

    return model


def get_latent_representation(
    model: SCVI,
    adata: Optional[AnnData] = None,
    give_mean: bool = True,
    n_samples: int = 1,
    batch_size: Optional[int] = None,
    copy: bool = False,
) -> AnnData:
    """
    Extract the latent representation from a trained scVI model.

    The latent space captures the biological variability with:
    - Batch effects removed (if batch_key was set during setup)
    - Technical noise reduced
    - Non-linear gene-gene relationships preserved

    Parameters
    ----------
    model : SCVI
        Trained scVI model.
    adata : AnnData, optional
        AnnData to store latent representation in. Uses model.adata if None.
    give_mean : bool
        If True, return the mean of the variational posterior.
        If False, sample from the posterior (useful for uncertainty quantification).
    n_samples : int
        Number of posterior samples (only when give_mean=False).
    batch_size : int, optional
        Batch size for computing latent representation.
    copy : bool
        Whether to return a copy.

    Returns
    -------
    AnnData
        AnnData with latent representation stored in .obsm["X_scVI"].
    """
    if adata is None:
        adata = model.adata
    adata = adata.copy() if copy else adata

    latent = model.get_latent_representation(
        adata,
        give_mean=give_mean,
        n_samples=n_samples,
        batch_size=batch_size,
    )

    adata.obsm["X_scVI"] = latent
    adata.obsm["X_latent"] = latent  # generic alias

    n_latent = latent.shape[1]
    print(f"[scVI latent] Extracted {n_latent}-dimensional latent representation")
    print(f"[scVI latent] Shape: {latent.shape} (cells × latent_dims)")

    # Basic stats
    print(f"[scVI latent] Mean: {latent.mean():.4f}, Std: {latent.std():.4f}")
    print(f"[scVI latent] Min: {latent.min():.4f}, Max: {latent.max():.4f}")
    print(f"[scVI latent] Stored in adata.obsm['X_scVI'] and adata.obsm['X_latent']")

    return adata


def scvi_pipeline(
    adata: AnnData,
    setup_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    train_kwargs: Optional[Dict[str, Any]] = None,
    latent_kwargs: Optional[Dict[str, Any]] = None,
    save_model_path: Optional[str] = None,
    verbose: bool = True,
) -> SCVI:
    """
    Run the full scVI pipeline: setup → train → extract latent representation.

    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData ready for scVI.
    setup_kwargs : dict, optional
        Keyword arguments for setup_scvi().
    model_kwargs : dict, optional
        Keyword arguments for model initialization in train_scvi_model().
    train_kwargs : dict, optional
        Keyword arguments for model training.
    latent_kwargs : dict, optional
        Keyword arguments for get_latent_representation().
    save_model_path : str, optional
        Path to save the trained model.
    verbose : bool
        Print progress messages.

    Returns
    -------
    SCVI
        Trained scVI model.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  STEP 2: scVI DEEP LATENT VARIABLE MODEL")
        print("=" * 60)

    # Setup
    if verbose:
        print("\n--- 2.1 scVI Setup ---")
    skwargs = setup_kwargs or {}
    adata = setup_scvi(adata, **skwargs)

    # Training
    if verbose:
        print("\n--- 2.2 scVI Training ---")
    mkwargs = model_kwargs or {}
    model = train_scvi_model(adata, **mkwargs, **(train_kwargs or {}))

    # Latent extraction
    if verbose:
        print("\n--- 2.3 Latent Representation ---")
    lkwargs = latent_kwargs or {}
    get_latent_representation(model, **lkwargs)

    # Save model
    if save_model_path is not None:
        os.makedirs(os.path.dirname(save_model_path) if os.path.dirname(save_model_path) else ".", exist_ok=True)
        model.save(save_model_path, overwrite=True)
        print(f"[scVI] Model saved to: {save_model_path}")

    return model
