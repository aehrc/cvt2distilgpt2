from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from ray_lightning import HorovodRayPlugin, RayPlugin
from ray_lightning.tune import TuneReportCallback
from transmodal.logger import CSVLogger
from transmodal.utils import get_ckpt_path
from typing import Optional
import os
import pathlib
import torch
import warnings

def create_trainer(
    monitor: str,
    monitor_mode: str,
    ver: str = "<VERSION_NAME>",
    gradient_clip_val: float = 0.0,  # 0.5 is the standard value.
    hyper: bool = False,
    gpus_per_trial: Optional[int] = None,
    early_stopping: bool = False,
    patience: int = 0,
    min_delta: float = 0.0,
    divergence_threshold: Optional[float] = None,
    max_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    exp_dir: Optional[str] = None,
    log_every_n_steps: int = 50,
    log_gpu_memory: Optional[str] = "min_max",
    task: Optional[str] = None,
    resumable: bool = True,
    resume_training: Optional[bool] = None,
    resume_epoch: Optional[int] = None,
    ckpt_path: Optional[str] = None,
    sched_inter: Optional[str] = None,  # "step", "epoch", or None.
    half_precision: bool = False,
    save_top_k: int = 1,
    sharded: bool = False,
    debug: bool = False,
    train_set_fraction: float = 1.0,
    val_set_fraction: float = 1.0,
    test_set_fraction: float = 1.0,
    num_nodes: int = 1,
    num_gpus: Optional[int] = None,
    accelerator: Optional[str] = "ddp",
    refresh_rate: int = 50,
    **kwargs,
) -> Trainer:
    """
    Returns a pytorch_lightning Trainer.

    Note: resuming training has not been implemented yet. Ensure to save the
    state of the optimiser and learning rate scheduler when implementing this
    feature. Need to use Trainer(resume_from_checkpoint...) for this.

    Argument/s:
        monitor - metric to monitor for EarlyStopping and ModelCheckpoint.
        monitor_mode - whether the metric to be monitored is to be maximised or
        minimised.
        ver - model version.
        gradient_clip_val - gradient clipping value.
        hyper - if hyperparameter optimisation is to be performed.
        gpus_per_trial - Number of GPUs to use for each hyperparameter search trial.
        early_stopping - stop training when a monitored metric has stopped
            improving.
        patience - no. of epochs with no improvement after which training will
            be stopped.
        min_delta - minimum change in the monitored quantity to qualify as an
            improvement.
        max_epochs - maximum number of epochs.
        max_steps - maximum number of training steps.
        exp_dir - directory where the files for the experiment are saved.
        log_every_n_steps - how often to log within steps (defaults to every 50 steps).
        log_gpu_memory - log gpu memory usage
        task - name of the task.
        resumable - whether the last completed epoch is saved to enable resumable training.
        resume_training - resume training from the last epoch.
        resume_epoch - the epoch to resume training from.
        ckpt_path - resume training from the specified checkpoint.
        sched_inter - learning rate scheduler interval ("step" or "epoch").
        half_precision - use half precision (16-bit floating point).
        save_top_k - best k models saved according to the monitored metric. If
            0, no models are saved. If -1, all models are saved.
        sharded - use dpp_sharded to partition large models across multiple GPUs.
        debug - training, validation, and testing are completed using one mini-batch.
        train_set_fraction - fraction of training set to use (for debugging).
        val_set_fraction - fraction of validation set to use (for debugging).
        test_set_fraction - fraction of test set to use (for debugging).
            (used for debugging).
        num_nodes - number of nodes for the job.
        num_gpus - number of GPUs per node.
        accelerator - the distributed backend.
        refresh_rate - refresh rate of progress bar in terms of steps.
        kwargs - keyword arguments.
    """
    loggers = []
    callbacks = [RichProgressBar()]
    plugins = []
    accelerator = accelerator if num_gpus > 0 else None

    # Add callbacks and plugins for hyperparameter optimisation
    if hyper:
        callbacks.append(TuneReportCallback(metrics=monitor, on="validation_end"))
        plugins.append(RayPlugin(num_workers=gpus_per_trial, use_gpu=True))
        accelerator, num_gpus = None, None
        num_nodes, refresh_rate, exp_dir = 1, 0, pathlib.Path().resolve()
        resumable = False

    # CSV loggers
    loggers.append(CSVLogger(exp_dir, name="", version=""))

    # Model checkpointing
    callbacks.append(
        ModelCheckpoint(
            dirpath=None if hyper else exp_dir,  # Need to make sure this works with Tune.
            monitor=monitor,
            mode=monitor_mode,
            save_top_k=save_top_k,
            filename="{epoch:d}-{" + monitor + ":f}",
            save_last=resumable,
        )
    )

    # Early stopping
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=monitor_mode,
                min_delta=min_delta,
                patience=patience,
                divergence_threshold=divergence_threshold,
                verbose=False,
            )
        )

    # Learning rate monitor
    if sched_inter is not None:
        callbacks.append(LearningRateMonitor())

    # Load checkpoint before training
    if resumable:
        ckpt_path = os.path.join(exp_dir, "last.ckpt")
        if os.path.isfile(ckpt_path):
            print("Resuming training from {}.".format(ckpt_path))
        else:
            ckpt_path = None
            warnings.warn("last.ckpt does not exist, starting training from epoch 0 unless ckpt_path is specified.")
    elif resume_epoch > 0:
        ckpt_path = get_ckpt_path(exp_dir, load_epoch=resume_epoch)
        print("Resuming training from {}.".format(ckpt_path))

    # Distribute model across GPUs
    if sharded:
        raise ValueError(
            "Not using sharded at this stage, as it cannot be resumed from when using ddp."
        )

    # PyTorch Lightning Trainer
    return Trainer(
        logger=loggers,
        callbacks=callbacks,
        plugins=plugins,
        max_epochs=max_epochs,
        max_steps=max_steps,
        resume_from_checkpoint=ckpt_path,
        gradient_clip_val=gradient_clip_val,
        weights_summary="top",
        num_sanity_val_steps=0,
        fast_dev_run=debug,
        accelerator=accelerator,  # if num_gpus > 1 or num_nodes > 1 else None,
        gpus=num_gpus,
        num_nodes=num_nodes,
        precision=16 if half_precision and torch.cuda.is_available() else 32,
        log_gpu_memory=log_gpu_memory if torch.cuda.is_available() else False,
        # plugins="ddp_sharded" if torch.cuda.device_count() > 1 and sharded else None,
        limit_train_batches=train_set_fraction,
        limit_val_batches=val_set_fraction,
        limit_test_batches=test_set_fraction,
        progress_bar_refresh_rate=refresh_rate,
        log_every_n_steps=log_every_n_steps,
    )
