from config.cvt_21_to_distilgpt2 import config as external_config

def config():
    updated_config = external_config(multi_image_input=False)

    # updated_config["num_samples"] = 50000
    #
    # decoder_module = "transmodal.network.gpt2decoder_redux"
    # decoder_definition = "GPT2Decoder"
    # decoder_ckpt_name = "distilgpt2"
    # decoder_warm_start = True
    #
    # updated_config["networks"]["decoder"] = {
    #     "module": decoder_module,
    #     "definition": decoder_definition,
    #     "inputs": {
    #         "last_hidden_state": "encoder_hidden_states",
    #         "decoder_input_ids": "decoder_input_ids",
    #         "decoder_attention_mask": "decoder_attention_mask"
    #     },
    #     "outputs": {"logits": "logits"},
    #     "generate_inputs": {
    #         "last_hidden_state": "encoder_hidden_states"
    #     },
    #     "generate_outputs": {"sequences": "predictions"},
    #     "self_critical_outputs": {"samples": "samples", "log_probs": "log_probs"},
    #     "kwargs": {
    #         "ckpt_name": decoder_ckpt_name,
    #         "warm_start": decoder_warm_start,
    #     },
    # }
    #
    # updated_config["opt"]["param_groups"]["group_1"]["modules"] = {
    #     "encoder_projection": {},
    #     "decoder.decoder.decoder.transformer": {"include": ["crossattention"]},
    # }
    #
    # updated_config["opt"] = {
    #     "module": "torch.optim",
    #     "definition": 'AdamW',
    #     "kwargs": {"lr": 1e-5},
    # }


    return updated_config
