from medicap_21.config.final.cvt_21_to_distilgpt2 import config as external_config

def config():
    updated_config = external_config(multi_image_input=True)

    updated_config["half_precision"] = False
    updated_config["mbatch_size"] = 1

    return updated_config
