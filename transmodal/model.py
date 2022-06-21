from pytorch_lightning import LightningModule
from torch.nn.functional import softmax
from transformers.tokenization_utils import PreTrainedTokenizer
from transmodal.utils import rgetattr
from typing import Any, Dict, Optional, Union
import importlib
import inspect
import pandas as pd
import torch
import torch.nn as nn
import warnings


class Transmodal(LightningModule):
    """
    Multimodal deep learning model.
    """

    test_val_set = False  # Set if the validation set is used in test_step().

    def __init__(
            self,
            networks: Dict[str, Any],
            # num_classes: Optional[int] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            opt: Optional[Dict[str, any]] = None,
            loss: Optional[str] = None,  # convert to: loss: Optional[Dict[str, any]] = None,
            loss_kwargs: Dict[str, any] = {},
            sched: Optional[Dict[str, any]] = None,
            reduce_on_plateau: Optional[bool] = False,
            monitor: Optional[bool] = None,
            self_critical: Optional[bool] = False,
            baseline: Optional[bool] = True,
            reward: Optional[Dict[str, any]] = None,
            val_metrics: Dict[str, any] = {},
            test_metrics: Dict[str, any] = {},
            forward_output_keys: list = ["logits"],
            generate_output_keys: list = ["predictions"],
            skip_special_tokens: bool = True,
            autoregressive: Optional[bool] = False,
            self_critical_output_keys: list = ["samples", "log_probs"],
            labels_key: Union[str, int] = "labels",
            labels_gen_key: Optional[str] = None,
            search_config: Optional[Dict[str, any]] = None,
            permute_outp: Optional[list] = None,
            softmax_metrics: Optional[bool] = False,
            float_labels: Optional[bool] = False,
            ckpt_zoo_dir: Optional[str] = None,
            ver: str = "tmp",
            exp_dir: Optional[str] = None,
            print_model: bool = False,
            dataset_dir: Optional[str] = None,
            cuis: Optional[list] = None,
            coco_metrics: Optional[list] = [],  # remove after medicap_21.
            accelerator: Optional[str] = "ddp",
            **kwargs,
    ):
        """
        Argument/s:
            networks - networks.
            # num_classes - number of classes.
            tokenizer - tokenizer.
            opt - optimiser.
            loss - loss function.
            loss_kwargs - loss function keyword arguments.
            sched - learning rate scheduler.
            reduce_on_plateau - if ReduceLROnPlateau is used as the learning rate scheduler.
            monitor - metric to be monitored by ReduceLROnPlateau.
            self_critical - perform self-critical sequence training (SCST).
            baseline - use a reference reward or baseline with SCST.
            reward - reward for SCST.
            val_metrics - validation metrics.
            test_metrics - test metrics.
            forward_output_keys - keys to output at the end of forward.
            generate_output_keys - keys to output at the end of generate.
            skip_special_tokens - skip special tokens when decoding a generated sentence.
            autoregressive - if the model is autoregressive and needs to generate
                sequences during validation and test steps.
            self_critical_output_keys - keys to output at the end of generate for
                self-critical sequence training.
            labels_key - mini-batch dictionary key name for the labels.
            labels_gen_key - mini-batch dictionary key name for the labels used
                during generation. These may be diferent to labels_key.
            search_config - configuration for the search for the output sequence.
            permute_outp - permute the output of the model.
            # softmax_metrics - give the softmax of the logits to the metrics.
            float_labels - convert labels to floating point.
            ckpt_zoo_dir - directory containing the zoo of model checkpoints.
            ver - model version.
            exp_dir - experiment directory.
            print_model - print the model layers to stdout.
            dataset_dir - dataset directory.
            cuis - list of UMLS CUIs to be added to the CUIs considered by the model.
            coco_metrics - list of COCO metrics to be computed.
            accelerator - the distributed backend.
            kwargs - keyword arguments.
        """
        super(Transmodal, self).__init__()
        self.save_hyperparameters()

        self.networks = networks
        # self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.opt = opt
        self.loss = loss
        self.loss_kwargs = loss_kwargs
        self.sched = sched
        self.reduce_on_plateau = reduce_on_plateau
        self.monitor = monitor
        self.self_critical = self_critical
        self.baseline = baseline
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.forward_output_keys = forward_output_keys
        self.generate_output_keys = generate_output_keys
        self.skip_special_tokens = skip_special_tokens
        self.autoregressive = autoregressive
        self.self_critical_output_keys = self_critical_output_keys
        self.labels_key = labels_key
        self.labels_gen_key = labels_key if labels_gen_key is None else labels_gen_key
        self.search_config = search_config
        self.permute_outp = permute_outp
        # self.softmax_metrics = softmax_metrics
        self.float_labels = float_labels
        self.ckpt_zoo_dir = ckpt_zoo_dir
        self.dataset_dir = dataset_dir
        self.ver = ver
        self.exp_dir = exp_dir
        self.cuis = cuis
        self.coco_metrics = coco_metrics
        self.accelerator = accelerator

        # Create UMLS CUI & judgement to index dataframe
        # if self.cuis:
        #     if not hasattr(self, "cui_judgement_to_idx"):
        #         self.cui_judgement_to_idx = pd.DataFrame(
        #             columns=["Negative", "Uncertain", "Positive"]
        #         )
        #         self.cui_judgement_to_idx.index.name = "CUI"
        #
        #     if not self.cuis:
        #         warnings.warn(
        #             "The list of CUIs is empty for rank {}.".format(self.global_rank)
        #         )
        #
        #     for cui in self.cuis:
        #         if cui not in self.cui_judgement_to_idx.index:
        #             negative_index = len(self.cui_judgement_to_idx.index) * 3
        #             self.cui_judgement_to_idx = self.cui_judgement_to_idx.append(
        #                 pd.Series(
        #                     {
        #                         "Negative": negative_index,
        #                         "Uncertain": negative_index + 1,
        #                         "Positive": negative_index + 2,
        #                     },
        #                     name=cui,
        #                 ),
        #             )
        #
        #     self.num_classes = self.cui_judgement_to_idx.size
        #     print(
        #         "Number of classes for concept judgement: {}".format(self.num_classes)
        #     )

        # Networks
        for (i, j) in self.networks.items():
            Network = getattr(importlib.import_module(j["module"]), j["definition"])
            setattr(
                self,
                i,
                Network(ckpt_dir=self.ckpt_zoo_dir, **j["kwargs"]), # num_classes=self.num_classes,
            )
            if print_model:
                print(getattr(self, i))

        # Loss function (see https://pytorch.org/docs/stable/nn.html#loss-functions)
        if self.loss is None:
            warnings.warn("A loss function has not been defined.")
        elif isinstance(self.loss, str):
            self.loss = getattr(nn, self.loss)(**self.loss_kwargs)
        else:
            for k, v in self.loss.items():
                self.loss[k]["fnc"] = getattr(nn, v["definition"])(**v["kwargs"])
                if "output_permutation" not in self.loss[k]:
                    self.loss[k]["output_permutation"] = False
                if "float_labels" not in self.loss[k]:
                    self.loss[k]["float_labels"] = False

        # Reward for self-critical sequence training
        if reward:
            self.reward = getattr(importlib.import_module(reward["module"]), reward["definition"])(**reward["kwargs"])

    def init_metrics(self, metrics_config, set):
        """
        Initialises a dict of metrics.

        Argument/s:
            metrics_config - dictionary with metric class names as keys and
                keyword arguments (i.e. dicts) as values.

        Returns:
            metrics - a dict of metrics.
        """
        for k, v in metrics_config.items():
            metric = getattr(importlib.import_module(v["module"]), v["definition"])
            # if "num_classes" in inspect.signature(metric).parameters:
            #     v["kwargs"]["num_classes"] = self.num_classes
            if "exp_dir" in inspect.signature(metric).parameters:
                v["kwargs"]["exp_dir"] = self.exp_dir
            if "softmax" not in metrics_config[k]:
                metrics_config[k]["softmax"] = False
            if 'ckpt_dir' in inspect.signature(metric).parameters:
                v['kwargs']['ckpt_dir'] = self.ckpt_zoo_dir
            if 'device' in inspect.signature(metric).parameters:
                v['kwargs']['device'] = self.device
            setattr(self, set + "_" + k.lower(), metric(**v["kwargs"]))
            getattr(self, set + "_" + k.lower()).to(self.device)
            if hasattr(getattr(self, set + "_" + k.lower()), "requires_normaliser"):
                getattr(
                    self, set + "_" + k.lower()
                ).normaliser = self.tokenizer._tokenizer.normalizer

    def metrics_step(
        self,
        stage: str,
        y_hat,
        y,
        id,
        metrics: dict = {},
        on_step: bool = False,
        on_epoch: bool = True,
    ):
        """
        Scoring for each metric during the step of the specified stage.

        Argument/s:
            stage - 'train', 'val', or 'test'.
            y_hat - predictions.
            y - labels.
            id (str or int) - identification for the example.
            metrics - a dict that stores the scores for each metric.
            on_step - log scores during each step.
            on_epoch - log scores for the epoch.
        """
        for k, v in getattr(self, f"{stage}_metrics").items():
            metric = stage + "_" + k.lower()
            if isinstance(y_hat, dict):
                y_hat_task = y_hat[v["predictions"]]
                y_task = y[v["labels"]]
                if v["softmax"]:  # Apply softmax to y_hat for metrics.
                    y_hat = softmax(y_hat, dim=-1)
                scores = getattr(self, metric)(y_hat_task, y_task, id)
            else:
                if v["softmax"]:  # Apply softmax to y_hat for metrics.
                    y_hat = softmax(y_hat, dim=-1)
                scores = getattr(self, metric)(y_hat, y, id)
            if isinstance(scores, dict):
                metrics.update({f"{stage}_{k}": v for k, v in scores.items()})
            elif scores is not None:
                metrics[metric] = scores
        self.log_dict(metrics, on_step=on_step, on_epoch=on_epoch)

    def metrics_epoch_end(
        self,
        stage: str,
        metrics: dict = {},
        on_step: bool = False,
        on_epoch: bool = True,
    ):
        """
        Scoring for each metric at the end of the epoch of the specified stage.

        Argument/s:
            stage - 'train', 'val', or 'test'.
            metrics - a dict that stores the scores for each metric.
            on_step - log scores during each step.
            on_epoch - log scores for the epoch.
        """
        for k, v in getattr(self, f"{stage}_metrics").items():
            metric = stage + "_" + k.lower()
            if not getattr(self, metric).compute_on_step:
                scores = getattr(self, metric).compute()
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                getattr(self, metric).reset()  # PTL does not seem to reset the states if compute_on_step=False.
                if isinstance(scores, dict):
                    metrics.update({f"{stage}_{k}": v for k, v in scores.items()})
                elif scores is not None:
                    metrics[metric] = scores
        self.log_dict(metrics, on_step=on_step, on_epoch=on_epoch)

    def forward(self, **kwargs):
        """
        Forward propagation.

        Argument/s:
            kwargs - keyword arguments.

        Returns:
            Logits from the output of the last network resulting from forward
                propagation.
        """

        # Iterate through the networks
        for (i, j) in self.networks.items():
            outputs = getattr(self, i).forward(**{v: kwargs[k] for k, v in j["inputs"].items()})
            kwargs = {**kwargs, **{v: outputs[k] for k, v in j["outputs"].items()}}
        return {k: kwargs[k] for k in self.forward_output_keys}

    def generate(self, sample=False, log_probs=False, greedy_search=False, **kwargs):
        """
        Autoregresively generate a prediction.

        Argument/s:
            sample - perform sampling instead.
            log_probs - return the log-probabilities used to generate the sample.
            greedy_search (bool) - sets num_beams to one.
            kwargs - keyword arguments.

        Returns:
            Indices of the tokens for the predicted sequence.
        """

        # Iterate through the networks
        for (i, j) in self.networks.items():
            if hasattr(getattr(self, i), "is_decoder"):
                outputs = getattr(self, i).generate(
                    sample=sample,
                    log_probs=log_probs,
                    device=self.device,
                    greedy_search=greedy_search,
                    **{v: kwargs[k] for k, v in j["generate_inputs"].items()},
                    **self.search_config,
                )
                output_keys = (
                    j["self_critical_outputs"].items()
                    if log_probs
                    else j["generate_outputs"].items()
                )
                kwargs = {**kwargs, **{v: outputs[k] for k, v in output_keys}}
            else:
                outputs = getattr(self, i).forward(
                    **{v: kwargs[k] for k, v in j["inputs"].items()},
                )
                kwargs = {**kwargs, **{v: outputs[k] for k, v in j["outputs"].items()}}
        output_keys = (
            self.self_critical_output_keys if log_probs else self.generate_output_keys
        )
        return {k: kwargs[k] for k in output_keys}

    def compute_multi_task_loss(self, y_hat, y):
        """
        Computes the sum of the weighted losses for multiple tasks.

        Argument/s:
            y_hat - dictionary of predictions.
            y - dictionary of labels.
        """
        losses = []
        for k, v in self.loss.items():
            y_hat_task = y_hat[self.loss[k]["predictions"]]
            if self.loss[k]["output_permutation"]:
                y_hat_task = y_hat_task.permute(*self.loss[k]["output_permutation"])
            y_task = y[self.loss[k]["labels"]].float() if self.loss[k]["float_labels"] else y[self.loss[k]["labels"]]
            task_loss = self.loss[k]["fnc"](y_hat_task, y_task)
            task_loss = self.loss[k]["weight"] * task_loss
            losses.append(task_loss)
        return sum(losses)

    def training_step(self, batch, batch_idx):
        """
        Training step (the training loss needs to be returned).

        Argument/s:
            batch - mini-batch from the training set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.

        Returns:
            loss - training loss for the mini-batch.
        """

        # Self-critical sequence training step
        if self.self_critical:
            return self.self_critical_step(batch, batch_idx)

        # Inference
        y_hat = self(**batch)

        # Loss
        if isinstance(self.loss, dict):
            train_loss = self.compute_multi_task_loss(y_hat, batch)

        # Old loss
        else:
            y = batch[self.labels_key]
            if self.permute_outp:
                y_hat = y_hat["logits"].permute(*self.permute_outp)
            train_loss = self.loss(y_hat, y.float() if self.float_labels else y)

        # Log training loss
        self.log_dict({"train_loss": train_loss}, on_step=False, on_epoch=True)

        # Update and log scores for each validation metric.
        return train_loss

    def on_validation_start(self):
        """
        Called at the beginning of validation.
        """
        self.init_metrics(self.val_metrics, "val")

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Argument/s:
            batch - mini-batch from the validation set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.
        """

        # If the model needs to generate its output autoregresively
        if self.autoregressive:
            return self.validation_generate_step(batch, batch_idx)

        # Labels
        # y = batch[self.labels_key]

        # Inference
        y_hat = self(**batch) # ["logits"]

        # Permute
        # if self.permute_outp:
        #     y_hat = y_hat.permute(*self.permute_outp)

        # Loss
        if isinstance(self.loss, dict):
            val_loss = self.compute_multi_task_loss(y_hat, batch)

        # Old loss
        else:
            y = batch[self.labels_key]
            if self.permute_outp:
                y_hat = y_hat["logits"].permute(*self.permute_outp)
            val_loss = self.loss(y_hat, y.float() if self.float_labels else y)


        # val_loss = self.loss(y_hat, y.float() if self.float_labels else y)

        # Apply softmax to y_hat for metrics
        # if self.softmax_metrics:
        #     y_hat = softmax(y_hat, dim=-1)

        # Update and log scores for each validation metric.
        # if self.accelerator == "dp":
        #     return {"val_loss": val_loss, "y_hat": y_hat, "y": y}
        # else:
        self.metrics_step("val", y_hat, y, {"val_loss": val_loss})

    def validation_generate_step(self, batch, batch_idx):
        """
        Validation generate step.

        Argument/s:
            batch - mini-batch from the validation set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.
        """

        # Generate outputs autoregresively
        y_hat = self.generate(
            encoder_images=batch["encoder_images"],
        )["predictions"]

        # All unnecessary tokens are removed.
        y_hat = self.tokenizer.batch_decode(y_hat, skip_special_tokens=self.skip_special_tokens)

        # Update and log scores for each validation metric.
        # if self.accelerator == "dp":
        #     return {"y_hat": y_hat, "y": batch[self.labels_gen_key]}
        # else:
        self.metrics_step("val", y_hat, batch[self.labels_gen_key], batch["id"])

    # def validation_step_end(self, outputs):
    #     """
    #     Validation step end.
    #
    #     Argument/s:
    #         outputs - Operate on all the outputs of the validation step.
    #     """
    #     if self.accelerator == "dp":
    #         self.metrics_step(
    #             "val", outputs["y_hat"], outputs["y"], {"val_loss": outputs["val_loss"]},
    #         )

    def validation_epoch_end(self, outputs):
        """
        Operate on all of the outputs of each validation step.

        Argument/s:
            outputs - outputs returned from each test step.
        """
        self.metrics_epoch_end("val")

    def on_test_start(self):
        """
        Called at the beginning of testing.
        """
        self.init_metrics(self.test_metrics, "test")

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Argument/s:
            batch - mini-batch from the test set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.
        """

        # If the model needs to generate its output autoregressively
        if self.autoregressive:
            return self.test_generate_step(batch, batch_idx)

        # Labels
        y = batch[self.labels_key]

        # Inference
        y_hat = self(**batch)["logits"]

        # Permute
        if self.permute_outp:
            y_hat = y_hat.permute(*self.permute_outp)

        # Apply softmax to y_hat for metrics
        # if self.softmax_metrics:
        #     y_hat = softmax(y_hat, dim=-1)

        # Update and log scores for each test metric.
        # if self.accelerator == "dp":
        #     return {"y_hat": y_hat, "y": y}
        # else:
        self.metrics_step("test", y_hat, y)

    def test_generate_step(self, batch, batch_idx):
        """
        Test generate step.

        Argument/s:
            batch - mini-batch from the test set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.
        """

        # Generate outputs autoregresively
        y_hat = self.generate(
            search_config=self.search_config,
            greedy_search=False,
            encoder_images=batch["encoder_images"],
        )["predictions"]

        # All unnecessary tokens are removed.
        y_hat = self.tokenizer.batch_decode(y_hat, skip_special_tokens=self.skip_special_tokens)

        # Update and log scores for each test metric.
        # if self.accelerator == "dp":
        #     return {"y_hat": y_hat, "y": batch[self.labels_gen_key]}
        # else:
        self.metrics_step("test", y_hat, batch[self.labels_gen_key], batch["id"])

    # def test_step_end(self, outputs):
    #     """
    #     Test step end.
    #
    #     Argument/s:
    #         outputs - Operate on all the outputs of the test step.
    #     """
    #     if self.accelerator == "dp":
    #         self.metrics_step("test", outputs["y_hat"], outputs["y"])

    def test_epoch_end(self, outputs):
        """
        Operate on all of the outputs of each test step.

        Argument/s:
            outputs - outputs returned from each test step.
        """
        self.metrics_epoch_end("test")

    def self_critical_step(self, batch, batch_idx):
        """
        Self-critical sequence training step (the training loss needs to be returned).

        Argument/s:
            batch - mini-batch from the training set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.

        Returns:
            loss - training loss for the mini-batch.
        """

        # Sample
        sample = self.generate(
            search_config=self.search_config,
            greedy_search=True,
            sample=True,
            log_probs=True,
            **batch,
        )

        # Baseline
        if self.baseline:
            baseline = self.generate(
                search_config=self.search_config,
                greedy_search=True,
                **batch,
            )["predictions"]

        # Convert token indices into strings for the reward function
        sample_str = self.tokenizer.batch_decode(
            sample["samples"], skip_special_tokens=True
        )
        baseline_str = (
            self.tokenizer.batch_decode(baseline, skip_special_tokens=True)
            if self.baseline
            else None
        )

        # Reward
        reward = self.get_reward(batch, sample_str, baseline_str)

        # Loss
        nll = -sample["log_probs"]
        mask = sample["samples"] == self.search_config["pad_token_id"]
        nll = nll.masked_fill(mask[:, :-1], 0.0)
        train_loss = reward.unsqueeze(-1) * nll
        seq_len = torch.sum(torch.logical_not(mask), dim=-1).float()
        train_loss = torch.mean(torch.sum(train_loss) / seq_len)

        # Update and log scores for each metric
        metrics = {"train_loss": train_loss, "reward": reward}
        self.log_dict(metrics, on_step=False, on_epoch=True)

        return train_loss

    def get_reward(self, batch, sample, baseline):
        """
        Reward for training step of self-critical sequence training.

        Argument/s:
            batch - all elements of the mini-batch (e.g. labels corpus).
            sample - samples.
            baseline - baselines.

        Returns:
            reward - reward which is potentially referenced to a baseline.
        """

        # Sample reward
        reward = self.reward(sample, **batch).to(self.device)  # batch contains the labels.

        # Baseline reward
        if self.baseline:
            baseline = self.reward(baseline, **batch).to(self.device)  # batch contains the labels.
            reward = reward - baseline

        return reward

    def configure_optimizers(self):
        """
        Define optimisers and learning rate schedulers. This has not been set
        up for multiple optimisers and learning rate schedulers. as good source
        for understanding how the pytorch learning rate schedulers work can be
        found here:
        https://www.programmersought.com/article/12164650026/

        Returns:
            See configure_optimizers() for return options at the following link:
            https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html.
        """

        if "param_groups" in self.opt:
            params = []
            for i in self.opt["param_groups"].values():
                group_params = []  # Parameters for the group.
                for k, v in i["modules"].items():
                    named_params = rgetattr(self, k).named_parameters()
                    if "exclude" in v:
                        assert isinstance(v["exclude"], list), "'exclude' must be a list of strings."
                        named_params = list(filter(lambda x: any(s not in x[0] for s in v["exclude"]), named_params))
                    elif "include" in v:
                        assert isinstance(v["include"], list), "'include' must be a list of strings."
                        named_params = list(filter(lambda x: any(s in x[0] for s in v["include"]), named_params))
                    elif ("exclude" in v) and ("include" in v):
                        raise ValueError("Can only exclude or include parameters from a module, not both.")
                    group_params = group_params + list(map(lambda x: x[1], named_params))
                params.append({"params": group_params, **i["kwargs"]})
        else:
            params = self.parameters()

        optimiser = {
            "optimizer": getattr(
                importlib.import_module(self.opt["module"]), self.opt["definition"]
            )(params, **self.opt["kwargs"])
        }

        if self.sched:
            scheduler = getattr(importlib.import_module(self.sched["module"]), self.sched["definition"])(
                optimiser["optimizer"], **self.sched["kwargs"],
            )
            optimiser["lr_scheduler"] = {
                "scheduler": scheduler,  # the LR schduler.
                "interval": self.sched["inter"],  # the unit of the scheduler's step size.
                "frequency": 1,  # the frequency of the scheduler.
                "reduce_on_plateau": self.reduce_on_plateau,  # for ReduceLROnPlateau scheduler.
                "monitor": self.monitor,  # metric for ReduceLROnPlateau to monitor.
                "strict": True,  # whether to crash the training if `monitor` is not found.
            }

        return optimiser

    def cuis_judgements_to_multi_label_vector(self, cuis_judgements):
        """
        Converts the UMLS CUIs & their judgements for each example into a set of multi-label vectors.

        Argument/s:
            cuis_judgements - a list containing dictionaries where the keys are CUIs and the values are judgements.

        Returns:
            Batch of multi-label vectors
        """
        multi_label_vector = torch.zeros(
            [len(cuis_judgements), self.num_classes], dtype=torch.int32
        ).to(self.device)
        x, y = [], []
        for i, j in enumerate(cuis_judgements):
            x.extend([i] * len(j))
            y.extend([self.cui_judgement_to_idx.loc[k, j] for k, j in j.items()])
        multi_label_vector[(x, y)] = 1.0
        return multi_label_vector
