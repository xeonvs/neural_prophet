import logging
import math
import os
from collections.abc import Mapping
from contextlib import contextmanager
from numbers import Number
from types import MethodType
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.tuner.lr_finder import _LRCallback

from neuralprophet.configure import Train
from neuralprophet.logger import ProgressBar

log = logging.getLogger("NP.utils_lightning")
MAX_LR_PROBE_WINDOWS = 1024
MAX_LR_PROBE_BYTES = 64 * 1024 * 1024


def _observed_lr_loss(callback, trainer, pl_module, outputs, batch, batch_idx):
    """Instance-local LR callback using observed updates, not optimizer global_step.

    Lightning's global_step is already one after TimeNet's first manual step;
    its additional +1 underestimates the first loss and the divergence baseline.
    The probe callback owns the count of actual finite observations.
    """
    if (trainer.fit_loop.batch_idx + 1) % trainer.accumulate_grad_batches != 0:
        return
    if outputs is None or isinstance(outputs, dict) and not outputs:
        callback.losses.append(float("nan"))
        return
    value = outputs if isinstance(outputs, torch.Tensor) else outputs.get("loss")
    if value is None:
        callback.losses.append(float("nan"))
        return
    value = float(value.item())
    if callback.progress_bar:
        callback.progress_bar.update()
    if not math.isfinite(value):
        callback.losses.append(float("nan"))
        trainer.should_stop = True
    else:
        callback._np_observed_updates += 1
        callback.avg_loss = callback.beta * callback.avg_loss + (1 - callback.beta) * value
        smoothed = callback.avg_loss / (1 - callback.beta**callback._np_observed_updates)
        if callback._np_observed_updates > 1 and smoothed > callback.early_stop_threshold * callback.best_loss:
            trainer.should_stop = True
        callback.best_loss = min(callback.best_loss, smoothed)
        callback.losses.append(smoothed)
    trainer.should_stop = trainer.strategy.broadcast(trainer.should_stop)
    if trainer.should_stop and callback.progress_bar:
        callback.progress_bar.close()


@contextmanager
def _lr_observation_count(model, trainer):
    """Correct only this probe's callback; restore exact instance attributes."""
    missing = object()
    original_hook_attr = vars(model).get("on_train_start", missing)
    original_hook = getattr(model, "on_train_start", None)
    original_callbacks = getattr(trainer, "callbacks", missing)
    attached = []

    def on_train_start(instance):
        if original_hook is not None:
            original_hook()
        matches = [callback for callback in trainer.callbacks if isinstance(callback, _LRCallback)]
        if len(matches) != 1:
            raise ValueError("auto_lr_unreliable")
        callback = matches[0]
        if not (
            math.isfinite(callback.beta)
            and 0 < callback.beta < 1
            and callback.early_stop_threshold is not None
            and math.isfinite(callback.early_stop_threshold)
            and callback.early_stop_threshold > 0
        ):
            raise ValueError("auto_lr_unreliable")
        originals = {
            name: vars(callback).get(name, missing)
            for name in ("on_train_batch_end", "_np_observed_updates", "avg_loss", "best_loss")
        }
        attached.append((callback, originals))
        callback._np_observed_updates = 0
        callback.avg_loss = 0.0
        callback.best_loss = math.inf
        callback.on_train_batch_end = MethodType(_observed_lr_loss, callback)

    model.on_train_start = MethodType(on_train_start, model)
    try:
        yield
    finally:
        for callback, originals in attached:
            for name, value in originals.items():
                if value is missing:
                    vars(callback).pop(name, None)
                else:
                    setattr(callback, name, value)
        if original_hook_attr is missing:
            vars(model).pop("on_train_start", None)
        else:
            model.on_train_start = original_hook_attr
        if original_callbacks is not missing:
            trainer.callbacks = original_callbacks


def _lr_sample_bytes(value):
    """Estimate standard collate tensor storage without stacking a large batch."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("auto_lr_unreliable")
        return value.nbytes
    if isinstance(value, Mapping):
        return sum(_lr_sample_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_lr_sample_bytes(item) for item in value)
    if isinstance(value, Number):
        return 16 if isinstance(value, complex) else 8
    if isinstance(value, (str, bytes)):
        return 0  # Dataset ID metadata is not stacked into tensors.
    raise ValueError("auto_lr_unreliable")


def _lr_probe_loader(loader):
    """Compare the same bounded training windows at every LR, not different minibatches.

    Use all windows when small; otherwise choose evenly spaced indices, without
    inspecting loss/target values. The byte cap estimates stacked sample tensor
    storage, not total process memory or allocations made by a custom collator.
    The ordinary training dataset, sampler, batch size and collator stay intact.
    """
    size = len(loader.dataset)
    if size <= 0:
        raise ValueError("auto_lr_unreliable")
    indices = np.linspace(0, size - 1, min(size, MAX_LR_PROBE_WINDOWS), dtype=int)
    # Inspect one item at a time, never preallocate a possibly oversized stack.
    maximum_bytes = max(_lr_sample_bytes(loader.dataset[int(index)]) for index in indices)
    if maximum_bytes <= 0:
        raise ValueError("auto_lr_unreliable")
    count = min(len(indices), MAX_LR_PROBE_BYTES // maximum_bytes)
    if count < 1:
        raise ValueError("auto_lr_unreliable")
    positions = np.linspace(0, len(indices) - 1, count, dtype=int)
    selected = [int(indices[position]) for position in positions]
    dataset = loader.dataset if len(selected) == size else Subset(loader.dataset, selected)
    probe = DataLoader(
        dataset, batch_size=len(selected), shuffle=False, collate_fn=loader.collate_fn, num_workers=0, drop_last=False
    )
    log.info(
        "Learning rate probe: %s/%s fixed training windows, estimated tensor bytes %s",
        len(selected),
        size,
        len(selected) * maximum_bytes,
    )
    return probe


def smooth_loss_and_suggest(lr_finder, window=10):
    """
    Select from a supported, training-only loss descent before divergence.

    At least three decreasing observed points must reach a new trough. The
    selector smooths only through the first such trough, never an exploded
    tail, and rejects an unsupported curve instead of inventing a default LR.

    Parameters
    ----------
        lr_finder
            Lightning range-test results containing learning rates and observed losses.
        window : int
            Hamming smoothing window. This does not relax observed-decrease checks.

    Returns
    -------
        loss : np.array
            Original losses with historical edge padding retained for tuple compatibility.
        lr: np.array
            Learning rate values
        suggested_lr: float
            Suggested rate from the supported decreasing region; otherwise ValueError("auto_lr_unreliable").
    """
    lr_finder_results = lr_finder.results
    lr = lr_finder_results["lr"]
    observed_loss = np.asarray(lr_finder_results["loss"], dtype=float)
    rates = np.asarray(lr, dtype=float)
    if (
        observed_loss.ndim != 1
        or rates.ndim != 1
        or len(rates) != len(observed_loss)
        or len(rates) < 3
        or not np.isfinite(window)
        or window <= 0
    ):
        raise ValueError("auto_lr_unreliable")
    # Stop before the first nonfinite or diverged observation. Smoothing an
    # exploded tail can create a false negative gradient at an unsafe rate.
    end = len(rates)
    best = math.inf
    for index, (rate, value) in enumerate(zip(rates, observed_loss)):
        if not np.isfinite(rate) or rate <= 0 or not np.isfinite(value) or value > 4 * best:
            end = index
            break
        best = min(best, value)
    rates_valid, loss_valid = rates[:end], observed_loss[:end]
    if len(rates_valid) < 3 or np.any(np.diff(rates_valid) <= 0):
        raise ValueError("auto_lr_unreliable")
    tolerance = np.finfo(float).eps * max(np.max(np.abs(loss_valid)), np.finfo(float).tiny) * 16
    # At least three observed points (two decreases), reaching a new trough
    # below earlier observations, must support the region. In particular, a
    # curve whose first probe remains best does not justify any larger rate.
    run_start = 0
    supported = None
    for index in range(1, len(loss_valid)):
        if loss_valid[index] >= loss_valid[index - 1] - tolerance:
            if index - run_start >= 3 and loss_valid[index - 1] < np.min(loss_valid[: run_start + 1]) - tolerance:
                supported = (run_start, index - 1)
                break
            run_start = index
    if supported is None and len(loss_valid) - run_start >= 3:
        if loss_valid[-1] < np.min(loss_valid[: run_start + 1]) - tolerance:
            supported = (run_start, len(loss_valid) - 1)
    if supported is None:
        raise ValueError("auto_lr_unreliable")
    start, trough = supported
    rates_valid, loss_valid = rates_valid[: trough + 1], loss_valid[: trough + 1]
    half_window = math.ceil(window / 2)
    hamming_window = np.hamming(2 * half_window)
    loss_smooth = np.convolve(
        hamming_window / hamming_window.sum(), np.pad(loss_valid, half_window, mode="edge"), mode="valid"
    )[1:]
    gradients = np.gradient(loss_smooth, np.log(rates_valid))
    candidates = np.arange(start + 1, trough)
    candidates = candidates[np.isfinite(gradients[candidates]) & (gradients[candidates] < -tolerance)]
    if not len(candidates):
        raise ValueError("auto_lr_unreliable")
    selected = int(candidates[np.argmin(gradients[candidates])])
    lr_suggestion = float(rates_valid[selected])
    log.info("Learning rate finder selected %s from supported observations %s-%s", lr_suggestion, start, trough)
    # Preserve the existing three-tuple and padded first-array compatibility.
    # Selection does not depend on this historical mismatched padding shape.
    loss = np.pad(observed_loss, pad_width=half_window, mode="edge")
    return loss, lr, lr_suggestion


def _smooth_loss(loss, beta=0.9):
    smoothed_loss = np.zeros_like(loss)
    smoothed_loss[0] = loss[0]
    for i in range(1, len(loss)):
        smoothed_loss[i] = smoothed_loss[i - 1] * beta + (1 - beta) * loss[i]
    return smoothed_loss


def configure_trainer(
    config_train: Train,
    metrics_logger,
    early_stopping_target: str = "Loss",
    accelerator: Optional[str] = None,
    progress_bar_enabled: bool = True,
    metrics_enabled: bool = False,
    checkpointing_enabled: bool = False,
    num_batches_per_epoch: int = 100,
    deterministic: bool = False,
):
    """
    Configures the PyTorch Lightning trainer.

    Parameters
    ----------
        config_train : Dict
            dictionary containing the overall training configuration.
        metrics_logger : MetricsLogger
            MetricsLogger object to log metrics to.
        early_stopping_target : str
            Target metric to use for early stopping.
        accelerator : str
            Accelerator to use for training.
        progress_bar_enabled : bool
            If False, no progress bar is shown.
        metrics_enabled : bool
            If False, no metrics are logged. Calculating metrics is computationally expensive and reduces the training
            speed.
        checkpointing_enabled : bool
            If False, no checkpointing is performed. Checkpointing reduces the training speed.
        num_batches_per_epoch : int
            Number of batches per epoch.

    Returns
    -------
        pl.Trainer
            PyTorch Lightning trainer
        checkpoint_callback
            PyTorch Lightning checkpoint callback to load the best model
    """
    if config_train.pl_trainer_config is None:
        config_train.pl_trainer_config = {}

    pl_trainer_config = config_train.pl_trainer_config
    # pl_trainer_config = pl_trainer_config.copy()

    # Set max number of epochs
    assert hasattr(config_train, "epochs") and config_train.epochs is not None
    pl_trainer_config["max_epochs"] = config_train.epochs

    # Configure the Ligthing-logs directory
    if "default_root_dir" not in pl_trainer_config.keys():
        pl_trainer_config["default_root_dir"] = os.getcwd()

    # Accelerator
    if isinstance(accelerator, str):
        if (accelerator == "auto" and torch.cuda.is_available()) or accelerator == "gpu":
            pl_trainer_config["accelerator"] = "gpu"
            pl_trainer_config["devices"] = -1
        elif (accelerator == "auto" and hasattr(torch.backends, "mps")) or accelerator == "mps":
            if torch.backends.mps.is_available():
                pl_trainer_config["accelerator"] = "mps"
                pl_trainer_config["devices"] = 1
        elif accelerator != "auto":
            pl_trainer_config["accelerator"] = accelerator
            pl_trainer_config["devices"] = 1

    if "accelerator" in pl_trainer_config:
        log.info(f"Using accelerator {pl_trainer_config['accelerator']} with {pl_trainer_config['devices']} device(s).")
    elif accelerator == "auto":
        log.info("No accelerator available. Using CPU for training.")

    # Configure metrics
    if metrics_enabled:
        pl_trainer_config["logger"] = metrics_logger
    else:
        pl_trainer_config["logger"] = False

    pl_trainer_config["deterministic"] = deterministic

    # Configure callbacks
    callbacks = []
    has_custom_callbacks = True if "callbacks" in pl_trainer_config else False

    # Configure checkpointing
    has_modelcheckpoint_callback = (
        True
        if has_custom_callbacks
        and any(isinstance(callback, pl.callbacks.ModelCheckpoint) for callback in pl_trainer_config["callbacks"])
        else False
    )
    if has_modelcheckpoint_callback and not checkpointing_enabled:
        raise ValueError(
            "Checkpointing is disabled but a ModelCheckpoint callback is provided. Please enable checkpointing or "
            "remove the callback."
        )
    if checkpointing_enabled:
        if not has_modelcheckpoint_callback:
            # Callback to access both the last and best model
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor=early_stopping_target, mode="min", save_top_k=1, save_last=True
            )
            callbacks.append(checkpoint_callback)
        else:
            checkpoint_callback = next(
                callback
                for callback in pl_trainer_config["callbacks"]
                if isinstance(callback, pl.callbacks.ModelCheckpoint)
            )
    else:
        pl_trainer_config["enable_checkpointing"] = False
        checkpoint_callback = None

    # Configure the progress bar, refresh every epoch
    has_progressbar_callback = (
        True
        if has_custom_callbacks
        and any(isinstance(callback, pl.callbacks.ProgressBar) for callback in pl_trainer_config["callbacks"])
        else False
    )
    if has_progressbar_callback and not progress_bar_enabled:
        raise ValueError(
            "Progress bar is disabled but a ProgressBar callback is provided. Please enable the progress bar or remove"
            " the callback."
        )
    if progress_bar_enabled:
        if not has_progressbar_callback:
            prog_bar_callback = ProgressBar(refresh_rate=num_batches_per_epoch, epochs=config_train.epochs)
            callbacks.append(prog_bar_callback)
    else:
        pl_trainer_config["enable_progress_bar"] = False

    # Early stopping monitor
    has_earlystopping_callback = (
        True
        if has_custom_callbacks
        and any(isinstance(callback, pl.callbacks.EarlyStopping) for callback in pl_trainer_config["callbacks"])
        else False
    )
    if has_earlystopping_callback and not config_train.early_stopping:
        raise ValueError(
            "Early stopping is disabled but an EarlyStopping callback is provided. Please enable early stopping or "
            "remove the callback."
        )
    if config_train.early_stopping:
        if not metrics_enabled:
            raise ValueError("Early stopping requires metrics to be enabled.")
        if not has_earlystopping_callback:
            early_stop_callback = pl.callbacks.EarlyStopping(
                monitor=early_stopping_target, mode="min", patience=20, divergence_threshold=5.0
            )
            callbacks.append(early_stop_callback)

    if has_custom_callbacks:
        pl_trainer_config["callbacks"].extend(callbacks)
    else:
        pl_trainer_config["callbacks"] = callbacks
    pl_trainer_config["num_sanity_val_steps"] = 0
    pl_trainer_config["enable_model_summary"] = False
    # TODO: Disabling sampler_ddp brings a good speedup in performance, however, check whether this is a good idea
    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#replace-sampler-ddp
    # config["replace_sampler_ddp"] = False

    return pl.Trainer(**pl_trainer_config), checkpoint_callback


def find_learning_rate(model, loader, trainer, train_epochs):
    log.info("No Learning Rate provided. Activating learning rate finder")

    # Configure the learning rate finder args
    batches_per_epoch = len(loader)
    main_training_total_steps = train_epochs * batches_per_epoch
    # main_training_total_steps is around 1e3 to 1e6 -> num_training 100 to 200
    num_training = 100 + int(np.log10(1 + main_training_total_steps / 1000) * 30)
    if batches_per_epoch < num_training:
        log.warning(
            f"Learning rate finder: The number of batches per epoch ({batches_per_epoch}) is too small than the required number \
                for the learning rate finder ({num_training}). The results might not be optimal."
        )
        # num_training = num_batches
    lr_finder_args = {
        "min_lr": 1e-6,
        "max_lr": 10.0,
        "num_training": num_training,
        "early_stop_threshold": 4.0,
        "mode": "exponential",
    }
    log.info(f"Learning rate finder ---- ARGs: {lr_finder_args}")

    # Execute the learning rate range finder
    tuner = Tuner(trainer)
    probe_loader = _lr_probe_loader(loader)
    original_max_epochs = trainer.fit_loop.max_epochs
    original_finding_lr = getattr(model, "finding_lr", False)
    original_learning_rate = getattr(model, "learning_rate", None)
    model.finding_lr = True
    # model.train_loader = loader
    try:
        # Lightning bounds max_steps for its range test but leaves max_epochs
        # unchanged. One fixed probe batch otherwise stops after a short main
        # training epoch budget, before the requested range can be observed.
        if original_max_epochs is not None and 0 <= original_max_epochs < num_training:
            trainer.fit_loop.max_epochs = num_training
        with _lr_observation_count(model, trainer), patch.dict(os.environ, {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}):
            lr_finder = tuner.lr_find(
                model=model,
                train_dataloaders=probe_loader,
                update_attr=False,
                # No held-out prices participate in automatic learning-rate selection.
                **lr_finder_args,
            )
    finally:
        trainer.fit_loop.max_epochs = original_max_epochs
        model.finding_lr = original_finding_lr
        model.learning_rate = original_learning_rate

    # Estimate the optimal learning rate from the loss curve
    if lr_finder is None:
        raise ValueError("auto_lr_unreliable")
    loss_list, lr_list, lr_suggested = smooth_loss_and_suggest(lr_finder)
    # TimeNet.configure_optimizers reads this attribute, not only the outer
    # forecaster's config. Never leave Lightning's default/probe LR in place.
    model.learning_rate = lr_suggested
    log.info(f"Learning rate finder suggested learning rate: {lr_suggested}")
    return lr_suggested
