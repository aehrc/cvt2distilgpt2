from config.cvt_21_to_distilgpt2_chexbert import config as external_config
import os
import yaml

def config():
    updated_config = external_config(multi_image_input=True)

    updated_config["self_critical"] = True

    with open(os.path.join('task', 'iu_x_ray_chen', 'paths.yaml')) as f:
        paths = yaml.load(f, Loader=yaml.FullLoader)

    updated_config["reward"] = {
        "module": "transmodal.rewards.cider",
        "definition": "ChenCOCOCIDErReward",
        "kwargs": {"labels_file_path": os.path.join(paths['dataset_dir'], "iu_x-ray_chen", "annotation.json")}
    }
    updated_config["pre_trained_ckpt_path"] = os.path.join(
        paths['exp_dir'],
        "iu_x_ray_chen",
        "cvt_21_to_distilgpt2",
        "epoch=10-val_chen_cider=0.475024.ckpt",
    )

    lr = 1e-5
    lr_nl = lr
    lr_i = 1e-6
    opt = "SGD"

    updated_config["mbatch_size"] = 1
    updated_config["early_stopping"] = False
    updated_config["max_epochs"] = 10

    updated_config["opt"] = {
        "module": "torch.optim",
        "definition": opt,
        "kwargs": {},
        "param_groups": {
            "group_1": {
                "modules": {
                    "encoder_projection": {},
                    "decoder.encoder_decoder.decoder.transformer": {"include": ["crossattention"]},
                },
                "kwargs": {"lr": lr},
            },
            "group_2": {
                "modules": {"encoder": {}},
                "kwargs": {"lr": lr_i},
            },
            "group_3": {
                "modules": {"decoder.encoder_decoder.decoder": {"exclude": ["crossattention"]}},
                "kwargs": {"lr": lr_nl},
            },
        },
    }

    return updated_config