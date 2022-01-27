from medicap_21.config.final.xcit_s_24_p8_to_distilgpt2 import config as external_config

def config():
    updated_config = external_config(multi_image_input=True)


    updated_config["self_critical"] = True

    updated_config["reward"] = {
        "module": "transmodal.rewards.bleu",
        "class": "ChenCOCOCIDErReward",
        "kwargs": {"labels_file_path": "/datasets/work/hb-mlaifsp-mm/source/Datasets/iu_x-ray_chen/annotation.json"}
    }
    updated_config["pre_trained_ckpt_path"] = "/scratch1/nic261/Experiments/iu_x_ray_chen/medicap_21/final/xcit_s_24_p8_to_distilgpt2/0/epoch=22-val_chen_cider=0.567060.ckpt"
    return updated_config