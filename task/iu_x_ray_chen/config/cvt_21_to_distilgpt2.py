from config.cvt_21_to_distilgpt2_chexbert import config as external_config

def config():
    updated_config = external_config(multi_image_input=True)

    return updated_config
