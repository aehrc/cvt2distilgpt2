from pytorch_lightning import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule
from typing import Any, Dict, Tuple
import functools
import gc
import getpass
import glob
import GPUtil
import importlib
import json
import numpy as np
import os
import re
import socket
import time
import torch
import warnings
import yaml


def get_workstation_config(num_gpus):
    """
    Get the configuration of the workstation.

    Argument/s:
        num_gpus (int) - the number of GPUs for the job.

    Returns:
        Dictionary of the workstation configuration.
    """
    with open("workstations.yaml") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    config = None
    for i in configs.keys():
        if i in socket.gethostname():
            config = configs[i]
            break
    if not isinstance(config, dict):
        warnings.warn(f"Workstation configuration for {socket.gethostname()} does not exist. Using default "
                      f"configuration: num_workers=3, total_gpus=1, total_memory=16")
    else:
        assert num_gpus <= config["total_gpus"], f"Too many GPUs requested for workstation {socket.gethostname()}."

    if config == None:
        config = {"num_workers": 3, "total_gpus": 1, "total_memory": 16}

    config["memory"] = str(int((config["total_memory"] / config["total_gpus"])) * num_gpus) + "GB"
    return config


def get_paths(task):
    """
    Reads from the paths directory of the task and returns the paths to the experiment,
    dataset, checkpoint, directories, and the path to the virtualenv.

        task (str) - the task.

    Returns:
        Dictionary of paths.
    """
    with open(os.path.join("task", task, "paths.yaml")) as f:
        paths = yaml.load(f, Loader=yaml.FullLoader)
    return paths

def get_dataset(config: Dict[str, Any]) -> Tuple[LightningDataModule, Dict[str, Any]]:
    """
    Returns the dataset for the task and potentially adds variables from
    dataset.setup() to the configuration.

    Argument/s:
        config - dictionary containing the configuration.

    Returns:
        Dataset and the configuration.
    """
    dataset_module = importlib.import_module(
        ".".join(["task", config["task"], "datamodule"])
    )
    TaskDataset = getattr(dataset_module, "TaskDataModule")
    dataset = TaskDataset(**config)

    if hasattr(dataset, "get_dataset_var"):
        dataset.prepare_data()
        dataset.setup(stage="fit")
        config = {**config, **dataset.get_dataset_var()}

    return dataset, config


def get_transmodal(model_config: Dict[str, Any]) -> LightningModule:
    """
    Create an instance of the task specific or standard multimodal model class. Handles
    loading model checkpoints.

    Argument/s:
        config - dictionary containing the model configuration.

    Returns:
        Multimodal model instance.
    """
    if os.path.isfile(os.path.join("task", model_config["task"], "model.py")):
        Transmodal = getattr(
            importlib.import_module(".".join(["task", model_config["task"], "model"])),
            "TaskModel",
        )
    else:
        from transmodal.model import Transmodal

    if "pre_trained_ckpt_path" in model_config:
        ckpt_path = model_config["pre_trained_ckpt_path"]
        # raise NotImplementedError

        # if isinstance(model_config["pre_trained_ckpt"], dict):
        #     ckpt_path = get_best_ckpt(
        #         {
        #             "ckpt_dir": model_config["ckpt_dir"],
        #             **model_config["pre_trained_ckpt"],
        #         }
        #     )
        # else:
        #     ckpt_path = os.path.join(
        #         model_config["ckpt_dir"], model_config["pre_trained_ckpt"]
        #     )
        print("Initialising from pre-trained checkpoint {}.".format(ckpt_path))
        transmodal = Transmodal.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            **model_config,
        )
    else:
        transmodal = Transmodal(**model_config)

    return transmodal


def get_ckpt_path(config: Dict[str, Any], load_epoch: int) -> str:
    """
    Get the epoch's checkpoint.

    Argument/s:
        config - dictionary containing the configuration.
        load_epoch - epoch to load.

    Returns:
        Path to the epoch's checkpoint.
    """
    raise NotImplementedError
    # try:
    #     ckpt_path = glob.glob(
    #         os.path.join(
    #             config["ckpt_dir"],
    #             config["task"],
    #             config["ver"],
    #             "checkpoints",
    #             "epoch=" + str(load_epoch) + "*.ckpt",
    #         )
    #     )[0]
    # except:
    #     raise ValueError(
    #         "Epoch {} is not in the checkpoint directory.".format(str(load_epoch))
    #     )
    # return ckpt_path

def get_trials(clargs_base, jargs):
    """
    Get the first and last trial number.

    Argument/s:
        clargs_base (Namespace) - command line arguments.
        jargs (Dict) - job arguments.

    Returns:
        First and last trial.
    """
    if clargs_base.trial >= 0:
        first_trial, last_trial = clargs_base.trial, clargs_base.trial + 1
    elif 'trial' in jargs:
        first_trial, last_trial = jargs['trial'], jargs['trial'] + 1
    elif 'trials' in jargs:
        first_trial, last_trial = 0, jargs['trials']
        jargs.pop('trials', None)
    else:
        first_trial, last_trial = -1, 0

    return first_trial, last_trial


def get_best_ckpt(exp_dir: str, monitor_mode: str) -> str:
    """
    Get the best epoch from the checkpoint directory.

    Argument/s:
        exp_dir - Experiment directory (where the checkpoints are saved).
        monitor_mode - Metric motitoring mode, either "min" or "max".

    Returns:
        Path to the epoch's checkpoint.
    """

    ckpt_list = glob.glob(os.path.join(exp_dir, "epoch*.ckpt"))

    if not ckpt_list:
        raise ValueError(f"No checkpoint exist in {exp_dir}.")

    scores = [
        re.findall(r"[-+]?\d*\.\d+|\d+", i.rsplit("=", 1)[1])[0] for i in ckpt_list
    ]

    if monitor_mode == "max":
        ckpt_path = ckpt_list[np.argmax(scores)]
    elif monitor_mode == "min":
        ckpt_path = ckpt_list[np.argmin(scores)]
    else:
        raise ValueError("'monitor_mode' must be max or min, not {}.".format(monitor_mode))
    return ckpt_path


def write_test_ckpt_path(ckpt_path: str, exp_dir: str):
    """
    Write ckpt_path used for testing to a text file.

    Argument/s:
        ckpt_path - path to the checkpoint of the epoch that scored
            highest for the given validation metric.
        exp_dir - path to the experiment directory.
    """
    with open(os.path.join(exp_dir, "test_ckpt_path.txt"), "a") as f:
        f.write(ckpt_path + "\n")


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def trial_dirname_string(trial):
    return str(trial)


def checkpoint_exists(directory):
    if not os.path.exists(directory):
        return False
    return any(
        (i.startswith("experiment_state") and i.endswith(".json"))
        for i in os.listdir(directory))

def gpu_clean_up(target_util=0.01, delay_s=5, verbose=False):

    requires_cleaning = 0
    for gpu in GPUtil.getGPUs():
        if verbose:
            print(f"GPU cleanup: Checking {gpu.id}...")
        if gpu.memoryUtil > target_util:
            if verbose:
                print(f"GPU cleanup: Utilisation on GPU {gpu.id} is {gpu.memoryUtil:0.3f}. "
                      f"Attempting to reach {target_util}. ")
            requires_cleaning = True
        else:
            if verbose:
                print(f"GPU cleanup: Utilisation on GPU {gpu.id} is {gpu.memoryUtil:0.3f}. "
                      f"(Less than {target_util}). ")

    if requires_cleaning:
        stt = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        if verbose:
            for gpu in GPUtil.getGPUs():
                print(f"GPU cleanup: Utilisation on GPU {gpu.id} is now {gpu.memoryUtil:0.3f}. "
                      f"Time taken was {(time.time() - stt)} seconds.")
        time.sleep(delay_s)