from config.cvt_21_to_distilgpt2_chexbert import config as external_config
import os
import yaml

def config():
    updated_config = external_config(multi_image_input=False)

    updated_config["self_critical"] = True

    with open(os.path.join('task', 'mimic_cxr_jpg_chen', 'paths.yaml')) as f:
        paths = yaml.load(f, Loader=yaml.FullLoader)

    updated_config["reward"] = {
        "module": "transmodal.rewards.cider",
        "definition": "ChenCOCOCIDErReward",
        "kwargs": {"labels_file_path": os.path.join(paths['dataset_dir'], "mimic_cxr_chen", "annotation.json")}
    }
    updated_config["pre_trained_ckpt_path"] = os.path.join(
        paths['exp_dir'],
        "mimic_cxr_jpg_chen",
        "cvt_21_to_distilgpt2",
        "epoch=8-val_chen_cider=0.425092.ckpt",
    )

    lr = 1e-5
    lr_nl = lr
    lr_i = 1e-6
    opt = "SGD"

    updated_config["mbatch_size"] = 1
    updated_config["early_stopping"] = False
    updated_config["max_epochs"] = 3

    updated_config["opt"] = {
        "module": "torch.optim",
        "definition": opt,
        "kwargs": {},
        "param_groups": {
            "group_1": {
                "modules": {
                    "encoder_projection": {},
                    "decoder.decoder.transformer": {"include": ["crossattention"]},
                },
                "kwargs": {"lr": lr},
            },
            "group_2": {
                "modules": {"encoder": {}},
                "kwargs": {"lr": lr_i},
            },
            "group_3": {
                "modules": {"decoder.decoder": {"exclude": ["crossattention"]}},
                "kwargs": {"lr": lr_nl},
            },
        },
    }

    return updated_config