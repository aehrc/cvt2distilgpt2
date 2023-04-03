from argparse import Namespace
from transformers import AutoTokenizer
from typing import Any, Dict, Optional
import importlib
import inspect
import json
# import optuna
import os
import warnings


def load_config(clargs: Namespace) -> Dict[str, Any]:
    # The model config can be contained in a .py file (needed for hyperparameter optimisation)
    # or a .json file

    assert isinstance(clargs.task, str), f'clargs.task is not a string: {clargs.task}, type:{type(clargs.task)}'
    assert isinstance(clargs.ver, str), f'clargs.ver is not a string: {clargs.ver}, type:{type(clargs.ver)}'

    python_file = os.path.join("task", clargs.task, "config", clargs.ver + ".py")
    json_file = os.path.join("task", clargs.task, "config", clargs.ver + ".json")
    if os.path.isfile(python_file) and os.path.isfile(json_file):
        raise FileExistsError("Both a .json and a .py file exist for {} {} "
                              "(only one can exist).".format(clargs.task, clargs.ver))
    if os.path.isfile(python_file):
        module = ".".join(["task", clargs.task, "config", *clargs.ver.split("/")])
        config = getattr(importlib.import_module(module), "config")()
    else:
        with open(json_file) as f:
            config = json.load(f)
    return config

def get_config(clargs: Namespace) -> Dict[str, Any]:
    """
    The command line arguments stored in 'clargs' are used to determine which
    model configuration is used.

    The following is added to the model's configuration:
        - The contents of the .json file for the model configuration found under the task's
            'config' directory.
        - The command line arguments contained in 'clargs'.
        - The contents of the .json file found under the task's 'paths' directory.
        - A tokenizer.
        - The sequence generation search configuration.
        - The 'shared.json' configuration for the task.

    Argument/s:
        clargs - the job configuration created using command line arguments.

    Returns:
        Dictionary containing the model's configuration.
    """

    # Load the model config from a .json or .py file
    config = load_config(clargs)

    # Add command line arguments to config
    config = {**vars(clargs), **config}
    # print("Version: {}, task: {}.".format(config["ver"], config["task"]))

    # config["exp_dir"] = os.path.join(config["exp_dir"], config["task"], config["ver"])
    # if clargs.trial > -1:
    #     config["exp_dir"] = os.path.join(config["exp_dir"], str(clargs.trial))

    # config["exp_dir"] = os.path.join("task", config["task"], "log", config["ver"])
    # config["best_trial_path"] = os.path.join(config["ckpt_save_dir"], config["task"], config["ver"],
    #                                                "best_trial.pkl")

    config["study_name"] = "_".join(config["task"].split("/") + config["ver"].split("/"))


    # # For optuna
    # if config["monitor_mode"] == "max":
    #     config["direction"] = "maximize"
    # elif config["monitor_mode"] == "min":
    #     config["direction"] = "minimize"
    # else:
    #     raise ValueError("'monitor_mode' must either be min or max.")


    if "tokenizer_init" in config:
        assert ("encoder_tokenizer_init" not in config) or (
                "decoder_tokenizer_init" not in config
        ), (
            "Only tokenizer_init or encoder_tokenizer_init and "
            "decoder_tokenizer_init can be specified, not both.",
        )

        assert isinstance(config["ckpt_zoo_dir"], str), f'ckpt_zoo_dir must be a string, got: ' \
                                                        f'{type(config["ckpt_zoo_dir"])} instead.'
        assert isinstance(config["tokenizer_init"], str), f'tokenizer_init must be a string, got: ' \
                                                          f'{type(config["tokenizer_init"])} instead.'

        config["tokenizer"] = AutoTokenizer.from_pretrained(
            os.path.join(config["ckpt_zoo_dir"], config["tokenizer_init"]),
            local_files_only=True,
        )

        if "special_tokens" in config:
            config["tokenizer"].add_special_tokens(config["special_tokens"])

    # elif "encoder_tokenizer_init" in config and "decoder_tokenizer_init" in config:
    #     config["encoder_tokenizer"] = AutoTokenizer.from_pretrained(
    #         config["encoder_tokenizer_init"], cache_dir=config["ckpt_zoo_dir"]
    #     )
    #     config["decoder_tokenizer"] = AutoTokenizer.from_pretrained(
    #         config["decoder_tokenizer_init"], cache_dir=config["ckpt_zoo_dir"]
    #     )

        if config["tokenizer"].pad_token is None:
            config["tokenizer"].add_special_tokens({'pad_token': '[PAD]'})

        if "loss_kwargs" not in config:
            config["loss_kwargs"] = {}

        # for i, j in zip(config['tokenizer'].all_special_tokens, config['tokenizer'].all_special_ids):
        #     print(f"{i}: {j}")

        config["loss_kwargs"]["ignore_index"] = config["tokenizer"].pad_token_id

    file_path = os.path.join("task", config["task"], "config", "search.json")

    if os.path.isfile(file_path):
        with open(file_path) as f:
            config["search_config"] = json.load(f)
    else:
        warnings.warn("Search configuration file (search.json) does not exist.")

    if "tokenizer" in config:

        # [BOS] token
        if config["tokenizer"].cls_token_id is not None:  # BERT.
            config["search_config"]["bos_token_id"] = config["tokenizer"].cls_token_id
        elif config["tokenizer"].bos_token_id is not None:  # GPT2.
            config["search_config"]["bos_token_id"] = config["tokenizer"].bos_token_id
        else:
            raise ValueError(f"'bos_token_id' cannot be specified (check config.py).")

        # [EOS] token
        if config["tokenizer"].sep_token_id is not None:  # BERT.
            config["search_config"]["eos_token_id"] = config["tokenizer"].sep_token_id
        elif config["tokenizer"].eos_token_id is not None:  # GPT2.
            config["search_config"]["eos_token_id"] = config["tokenizer"].eos_token_id
        else:
            raise ValueError(f"'eos_token_id' cannot be specified (check config.py).")

        config["search_config"]["pad_token_id"] = config["tokenizer"].pad_token_id

    # elif "decoder_tokenizer" in config:
    #     config["search_config"]["bos_token_id"] = config[
    #         "decoder_tokenizer"
    #     ].cls_token_id
    #     config["search_config"]["eos_token_id"] = config[
    #         "decoder_tokenizer"
    #     ].sep_token_id
    #     config["search_config"]["pad_token_id"] = config[
    #         "decoder_tokenizer"
    #     ].pad_token_id
    else:
        warnings.warn("No tokenizer in config.")

    file_path = os.path.join("task", config["task"], "config", "shared.json")

    if os.path.isfile(file_path):
        with open(file_path) as f:
            shared_config = json.load(f)
        config = {**config, **shared_config}
    else:
        warnings.warn("Shared configuration file (shared.json) does not exist.")

    # Delete the checkpoints and logs from previous sessions
    # if clargs.delete_previous:
    #     shutil.rmtree(config["exp_dir"], ignore_errors=True)
    #     shutil.rmtree(os.path.join(config["ckpt_zoo_dir"], config["task"], config["ver"]), ignore_errors=True)

    return config
