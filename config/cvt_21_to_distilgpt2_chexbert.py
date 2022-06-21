from config.cvt_21_to_distilgpt2 import config as external_config
# TODO: Ensure this is working for training. The optimizer is likely not setup for the new decoder.

def config(multi_image_input=True):
    config_dict = external_config(multi_image_input=multi_image_input)
    config_dict['val_metrics'] = {
        "nlg_metrics": {
            "module": "transmodal.metrics.chen",
            "definition": "ChenCaption",
            "kwargs": {"metrics": ["bleu", "cider", "meteor", "rouge"]},
        },
        'ce_metrics': {
            'module': 'transmodal.metrics.chexbert',
            'definition': 'ClinicalEfficacy',
            'kwargs': {
                'bert_path': 'bert-base-uncased',
                'checkpoint_path': 'stanford/chexbert/chexbert.pth',
            },
        },
    }
    config_dict['test_metrics'] = {
        "nlg_metrics": {
            "module": "transmodal.metrics.chen",
            "definition": "ChenCaption",
            "kwargs": {
                "metrics": ["bleu", "cider", "meteor", "rouge"],
                "save": True,
                "save_individual_scores": True,
            },
        },
        'ce_metrics': {
            'module': 'transmodal.metrics.chexbert',
            'definition': 'ClinicalEfficacy',
            'kwargs': {
                'bert_path': 'bert-base-uncased',
                'checkpoint_path': 'stanford/chexbert/chexbert.pth',
                'save_class_scores': True,
                'save_outputs': True,
            },
        },
    }

    return config_dict
