from medicap_21.config.final.cvt_21_to_distilgpt2 import config as external_config

def config():
    updated_config = external_config(multi_image_input=True)

    updated_config["self_critical"] = True

    updated_config["reward"] = {
        "module": "transmodal.rewards.cider",
        "definition": "ChenCOCOCIDErReward",
        "kwargs": {"labels_file_path": "/datasets/work/hb-mlaifsp-mm/source/Datasets/iu_x-ray_chen/annotation.json"}
    }
    updated_config["pre_trained_ckpt_path"] = "/datasets/work/hb-mlaifsp-mm/work/Experiments/iu_x_ray_chen/medicap_21/final/cvt_21_to_distilgpt2/1/epoch=10-val_chen_cider=0.475024.ckpt"

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