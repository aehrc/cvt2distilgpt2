from medicap_21.config.final.xcit_s_24_p8_to_distilgpt2 import config as external_config

def config():
    updated_config = external_config(multi_image_input=True)
    return updated_config
