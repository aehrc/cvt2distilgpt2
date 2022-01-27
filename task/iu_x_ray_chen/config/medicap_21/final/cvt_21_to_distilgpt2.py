from medicap_21.config.final.cvt_21_to_distilgpt2 import config as external_config

def config():
    updated_config = external_config(multi_image_input=True)
    return updated_config
