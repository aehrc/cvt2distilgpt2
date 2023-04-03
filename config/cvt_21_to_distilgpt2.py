from transmodal.network.cvt import spatial_position_feature_size

# TODO: Ensure this is working for training. The optimizer is likely not setup for the new decoder.

def config(
        multi_image_input=True,
        image_size=384,
        encoder_module="transmodal.network.cvt",
        encoder_definition="CvT",
        encoder_ckpt_name="CvT-21-384x384-IN-22k",
        encoder_model_config="cvt-21-384x384",
        encoder_warm_start=True,
        permute_encoder_last_hidden_state=[0, 2, 1],
        decoder_module="transmodal.network.gpt2decoder",
        decoder_definition="GPT2Decoder",
        decoder_ckpt_name="distilgpt2",
        decoder_warm_start=True,
):

    resize_size = image_size + 64
    decoder_max_len = 128
    opt = "AdamW"

    # Same as Chen et al.
    lr = 1e-4
    lr_nl = lr
    lr_i = 1e-5

    monitor = "val_chen_cider"
    monitor_mode = "max"
    val_metrics = {
        "nlg_metrics": {
            "module": "transmodal.metrics.chen",
            "definition": "ChenCaption",
            "kwargs": {"metrics": ["bleu", "cider", "meteor", "rouge"]},
        }
    }
    test_metrics = {
        "nlg_metrics": {
            "module": "transmodal.metrics.chen",
            "definition": "ChenCaption",
            "kwargs": {
                "metrics": ["bleu", "cider", "meteor", "rouge"],
                "save": True,
                "save_individual_scores": True,
            },
        }
    }
    config_dict = {
        "networks": {
            "multi_image_input": {
                "module": "transmodal.network.multi_image",
                "definition": "MultiImageInput",
                "inputs": {
                    "encoder_images": "images"
                },
                "outputs": {"images": "encoder_images", "images_per_example": "images_per_example"},
                "kwargs": {},
            },
            "encoder": {
                "module": encoder_module,
                "definition": encoder_definition,
                "inputs": {"encoder_images": "images"},
                "outputs": {"last_hidden_state": "last_hidden_state"},
                "kwargs": {
                    "is_encoder": True,
                    "ckpt_name": encoder_ckpt_name,
                    "warm_start": encoder_warm_start,
                    "model_config": encoder_model_config,
                },
            },
            "encoder_projection": {
                "module": "transmodal.network.encoder_projection",
                "definition": "EncoderPermuteProject",
                "inputs": {"last_hidden_state": "encoder_last_hidden_state"},
                "outputs": {"projected_encoder_last_hidden_state": "last_hidden_state"},
                "kwargs": {
                    "permute_encoder_last_hidden_state": permute_encoder_last_hidden_state,
                    "encoder_last_hidden_state_size": spatial_position_feature_size(encoder_model_config),
                    "decoder_hidden_state_size": 768,
                },
            },
            "multi_image_output": {
                "module": "transmodal.network.multi_image",
                "definition": "MultiImageOutput",
                "inputs": {
                    "last_hidden_state": "last_hidden_state", "images_per_example": "images_per_example"
                },
                "outputs": {"last_hidden_state": "last_hidden_state"},
                "kwargs": {},
            },
            "decoder": {
                "module": decoder_module,
                "definition": decoder_definition,
                "inputs": {
                    "last_hidden_state": "encoder_hidden_states",
                    "decoder_input_ids": "decoder_input_ids",
                    "decoder_attention_mask": "decoder_attention_mask"
                },
                "outputs": {"logits": "logits"},
                "generate_inputs": {
                    "last_hidden_state": "encoder_hidden_states"
                },
                "generate_outputs": {"sequences": "predictions"},
                "self_critical_outputs": {"samples": "samples", "log_probs": "log_probs"},
                "kwargs": {
                    "ckpt_name": decoder_ckpt_name,
                    "warm_start": decoder_warm_start,
                },
            }
        },
        "add_bos_eos_manually": True,
        "special_tokens": {"bos_token": "[BOS]"},
        "monitor": monitor,
        "monitor_mode": monitor_mode,
        "early_stopping": True,
        "patience": 10,
        "min_delta": 1e-4,
        "mbatch_size": 4,
        "max_epochs": 512,
        "save_top_k": 1,
        "loss": "CrossEntropyLoss",
        "opt": {
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
        },
        "tokenizer_init": decoder_ckpt_name,
        "permute_outp": [0, 2, 1],
        "half_precision": True,
        "print_model": False,
        "colour_space": "RGB",
        "search_config": {
            "max_length": decoder_max_len,
            "num_beams": 4,
        },
        "train_transforms": {
            "Resize": {"size": resize_size},
            "RandomCrop": {
                "size": [image_size, image_size],
                "pad_if_needed": True
            },
            "RandomRotation": {"degrees": 5.0},
            "ToTensor": {},
            "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        },
        "test_transforms": {
            "Resize": {"size": resize_size},
            "CenterCrop": {"size": [image_size, image_size]},
            "ToTensor": {},
            "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        },
        "decoder_max_len": decoder_max_len,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    if not multi_image_input:
        config_dict["networks"].pop("multi_image_input", None)
        config_dict["networks"].pop("multi_image_output", None)
    return config_dict
