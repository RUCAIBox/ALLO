import re
import json
import logging
import random
import os
import math

import sys
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

from tqdm import tqdm

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import is_sagemaker_mp_enabled

# from flash_attn.flash_attn_triton import flash_attn_func

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    set_seed,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import IntervalStrategy
from transformers.trainer_callback import TrainerCallback
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention, apply_rotary_pos_emb

from trl import DPOTrainer

from torch.optim import Optimizer

import deepspeed.comm as dist

from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the dataset to use (via the datasets library)."}
    )
    mask_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the dataset to use (via the datasets library)."}
    )
    eval_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    only_test: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    num_pair: Optional[int] = field(
        default=3,
        metadata={
            "help": (
                "The number of pairs to train the reward model."
            )
        },
    )
    beta: Optional[float] = field(
        default=0.1,
        metadata={
            "help": (
                "Beta in DPO"
            )
        },
    )
    epsilon: Optional[float] = field(
        default=0.1,
        metadata={
            "help": (
                "Epsilon in DPO"
            )
        },
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    source_suffix: Optional[str] = field(
        default="", metadata={"help": "A suffix to add after every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    max_source_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    result_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )


class EvalEpochIntervalCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)

        if (epoch % 5 == 0):
            control.should_save = True
        else:
            control.should_save = False

        if (args.logging_strategy == IntervalStrategy.EPOCH):
            control.should_log = True
        
        control.should_evaluate = True

        return control


class MaskAdamW_HF(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

        self.print_details()
        self.world_size = dist.get_world_size()
        self.cur_rank = dist.get_rank()

    def print_details(self):
        print('WARNING: In print details!!!')
        print('Total Group: ', len(self.param_groups))
        print('Keys in group: ', self.param_groups[0].keys())
        sum_params = 0
        for group in self.param_groups:
            sum_params = sum_params + len(group['params'])
        print('sum params: ', sum_params)


    @torch.no_grad()
    def step(self, closure = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        sum_params = 0
        for group in self.param_groups:
            sum_params = sum_params + len(group['params'])
        # print('sum params: ', sum_params)

        for group in self.param_groups:
            if (len(group['params']) == 0):
                continue
            p = group['params'][0]
            mask = group['mask']
            num_neural = mask.size(-1)
            assert(num_neural % self.world_size == 0)
            delta = num_neural // self.world_size
            st_idx = delta * self.cur_rank
            en_idx = delta * (self.cur_rank + 1)
            mask = mask[st_idx: en_idx].to(p.device)

            # print(p.size(), mask.size())
                
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = 0
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p)
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            beta1, beta2 = group["betas"]

            state["step"] += 1

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().add_(group["eps"])

            step_size = group["lr"]
            if group["correct_bias"]:  # No bias correction for Bert
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
            
            p.addcdiv_(exp_avg * mask, denom, value=-step_size)

            if group["weight_decay"] > 0.0:
                p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


class MaskDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model = None,
        ref_model = None,
        beta = 0.1,
        label_smoothing = 0,
        loss_type = "sigmoid",
        args = None,
        data_collator = None,
        label_pad_token_id = -100,
        padding_value = None,
        truncation_mode = "keep_end",
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        callbacks = None,
        optimizers = (None, None),
        preprocess_logits_for_metrics = None,
        max_length = None,
        max_prompt_length = None,
        max_target_length = None,
        peft_config = None,
        is_encoder_decoder = None,
        disable_dropout = True,
        generate_during_eval = False,
        compute_metrics = None,
        precompute_ref_log_probs = False,
        dataset_num_proc = None,
        model_init_kwargs = None,
        ref_model_init_kwargs = None,
        model_adapter_name = None,
        ref_adapter_name = None,
        reference_free = False,
        force_use_ref_model = False,
        params_mask = None,
        epsilon = None,
    ):
        super().__init__(
            model = model,
            ref_model = ref_model,
            beta = beta,
            label_smoothing = label_smoothing,
            loss_type = loss_type,
            args = args,
            data_collator = data_collator,
            label_pad_token_id = label_pad_token_id,
            padding_value = padding_value,
            truncation_mode = truncation_mode,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            tokenizer = tokenizer,
            model_init = model_init,
            callbacks = callbacks,
            optimizers = optimizers,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
            max_length = max_length,
            max_prompt_length = max_prompt_length,
            max_target_length = max_target_length,
            peft_config = peft_config,
            is_encoder_decoder = is_encoder_decoder,
            disable_dropout = disable_dropout,
            generate_during_eval = generate_during_eval,
            compute_metrics = compute_metrics,
            precompute_ref_log_probs = precompute_ref_log_probs,
            dataset_num_proc = dataset_num_proc,
            model_init_kwargs = model_init_kwargs,
            ref_model_init_kwargs = ref_model_init_kwargs,
            model_adapter_name = model_adapter_name,
            ref_adapter_name = ref_adapter_name,
            reference_free = reference_free,
            force_use_ref_model = force_use_ref_model,
        )

        self.params_mask = params_mask
        self.epsilon = epsilon

    def create_optimizer(self):
        

        print('is_sagemaker_mp_enabled: ', is_sagemaker_mp_enabled())
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = []
            optimizer_grouped_parameters_mask = []

            for n, p in opt_model.named_parameters():
                if (p.requires_grad == False):
                    continue
                if (n in decay_parameters):
                    weight_decay = self.args.weight_decay
                else:
                    weight_decay = 0.0
                # print(n, p.size(), params_mask[n[6:]].size())
                optimizer_grouped_parameters.append({
                    "params": [p],
                    # "params": [p, params_mask[n]],
                    "weight_decay": weight_decay,
                    "mask": self.params_mask[n].view(-1),
                })
                # optimizer_grouped_parameters_mask.append({
                #     "params": [params_mask[n]],
                #     "weight_decay": weight_decay,
                # })

            print(
                'Total Training Parameters: ', 
                len(optimizer_grouped_parameters), 
                len(optimizer_grouped_parameters[0]['params']), 
                optimizer_grouped_parameters[0]['params'][0].size(),
                optimizer_grouped_parameters[0].keys(),
            )

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)
            print(optimizer_kwargs)
            optimizer_cls = MaskAdamW_HF

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            # self.optimizer_mask = optimizer_cls(optimizer_grouped_parameters_mask, **optimizer_kwargs)
            # self.optimizer = optimizer_cls(opt_model.parameters(), **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        manager.register_module_override(module, "weight", {"optim_bits": 32})

        return self.optimizer

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
        # reward = None,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
            # reward_mask = reward_mask[:, 1:]
        loss_mask = labels != label_pad_token_id

        
        # print(reward_mask.size(), loss_mask.size())

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        with torch.no_grad():
            def gen_mask(epsilon, t, c):
                num_selected = int(t.size(-1) * epsilon)
                mask = torch.zeros(t.size(0), t.size(1))
                for i in range(t.size(0)):
                    cand_tensor = t[i, :] * c
                    cand_tensor = torch.where(cand_tensor == 0, -1e10, cand_tensor)
                    tk = cand_tensor.topk(num_selected).values[-1]
                    mask[i, :] = torch.where(cand_tensor >= tk, 1.0, 0.0)
                return mask.to(t.device)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        
        mask = gen_mask(self.epsilon, rejected_logratios, -1)
        rejected_logratios = rejected_logratios * mask


        chosen_logratios = chosen_logratios.sum(-1)
        rejected_logratios = rejected_logratios.sum(-1)

        chosen_logratios = chosen_logratios.to(self.accelerator.device)
        rejected_logratios = rejected_logratios.to(self.accelerator.device)
        logits = chosen_logratios - rejected_logratios


        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards


    def concatenated_forward(self, model, batch):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            # reward=batch['reward'] + batch['rejected_reward'],
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # with open(data_args.mask_path, 'r') as fin:
    #     params_mask = json.load(fin)
    params_mask = torch.load(data_args.mask_path, map_location='cpu')

    # Set seed before initializing model.
    set_seed(training_args.seed)
    random.seed(training_args.seed)

    def load_jsonl_data(data_path):
        files = data_path.split(',')
        raw_dataset = []
        for f in files:
            with open(f, 'r') as fin:
                tmp_dataset = fin.readlines()
                raw_dataset = raw_dataset + tmp_dataset
        raw_dataset = [json.loads(d) for d in raw_dataset]
        dataset = {}
        for data in raw_dataset:
            for k in data.keys():
                if (k not in dataset):
                    dataset[k] = []
                dataset[k].append(data[k])
        return Dataset.from_dict(dataset)

    train_dataset = load_jsonl_data(data_args.dataset_path)
    logger.info(train_dataset)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    config.pad_token_id = tokenizer.unk_token_id


    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    ref_model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    logger.info(model)
    print(model.device, ref_model.device)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    # Initialize our Trainer
    trainer = MaskDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        beta=data_args.beta,
        epsilon=data_args.epsilon,
        max_length=data_args.max_length,
        max_target_length=data_args.max_target_length,
        max_prompt_length=data_args.max_source_length,
        tokenizer=tokenizer,
        params_mask=params_mask,
    )

    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    logger.info(train_result)


if __name__ == "__main__":
    main()