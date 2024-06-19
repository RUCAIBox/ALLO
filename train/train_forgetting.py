import os
import copy
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import random
from typing import List, Optional, Tuple, Union
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention, apply_rotary_pos_emb
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from tqdm import tqdm, trange

from transformers.utils import is_sagemaker_mp_enabled

from torch.optim import Optimizer

import deepspeed.comm as dist


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "[UNK]"
PROMPT_DICT = {
    "alpaca_format": (
        "{input}"
    ),
}

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    flash_attention: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    mask_path: str = field(default=None, metadata={"help": "Path to the mask of the parameters."})
    prompt_type: Optional[str] = field(default="instruction")
    dailog_augmentation: Optional[bool] = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, prompt_type: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        prompt_simple_inference = PROMPT_DICT["alpaca_format"]
        self.sources, self.targets, self.reward, self.ref_prob = [], [], [], []
        for path in data_path.split(','):
            with open(path, 'r') as f:
                for line in tqdm(f.readlines()):
                # for i in trange(10000):
                    # line = f.readline()
                    try:
                        c = json.loads(line)
                    except:
                        print(path)
                        print(line)
                        raise ValueError

                    input_text = c['input']
                    source = prompt_simple_inference.format_map(dict(input=input_text))
                    self.sources.append(source.strip())

                    output_text = c['output']
                    self.targets.append(source + output_text + tokenizer.eos_token)

                    self.reward.append(c['reward'])
                    self.ref_prob.append(c['ref_prob'])

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.sources[i], 
            labels=self.targets[i],
            reward=self.reward[i],
            ref_prob=self.ref_prob[i],
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    data_args: DataArguments
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            text=[instance['labels'] for instance in instances],
            text_target=[instance['input_ids'] for instance in instances],
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        labels = copy.deepcopy(inputs['input_ids'])
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        labels[torch.where(inputs['labels'] != self.tokenizer.pad_token_id)] = IGNORE_INDEX
        inputs['labels'] = labels
        
        cur_len = inputs['input_ids'].size(1)
        
        reward = [[0] * (cur_len - len(instance['reward'])) + instance['reward'] for instance in instances]
        reward = torch.tensor(reward, dtype=torch.bfloat16)
        inputs['reward'] = reward
        
        ref_prob = [[0] * (cur_len - len(instance['ref_prob'])) + instance['ref_prob'] for instance in instances]
        ref_prob = torch.tensor(ref_prob, dtype=torch.bfloat16)
        inputs['ref_prob'] = ref_prob
        
        return inputs


class LlamaForFGNPO(LlamaForCausalLM):
    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        reward,
        ref_prob,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        reward = reward[..., 1:].contiguous()
        reward = reward.view(-1).contiguous()
        
        ref_prob = ref_prob[..., 1:].contiguous()
        ref_prob = ref_prob.view(-1).contiguous()
        log_ref_prob = ref_prob
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        CEFunc = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        log_policy_prob = -CEFunc(shift_logits, shift_labels)
        
        beta=0.1
        num_label = torch.sum(torch.ne(reward, 0.0))
        diff = -beta * (log_policy_prob - log_ref_prob)
        loss = -reward * F.logsigmoid(diff)
        loss = loss.sum() / num_label
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class LlamaForMaskNPO(torch.nn.Module):
    def __init__(self, model, mask_model):
        super().__init__()
        self.model = model
        self.mask_model = mask_model.eval()
        self.config = self.model.config
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        reward,
        ref_prob,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        logits = logits.float()

        reward = reward[..., 1:].contiguous()
        reward = reward.view(-1).contiguous()
        
        ref_prob = ref_prob[..., 1:].contiguous()
        ref_prob = ref_prob.view(-1).contiguous()
        log_ref_prob = ref_prob
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        CEFunc = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        log_policy_prob = -CEFunc(shift_logits, shift_labels)
        
        beta=0.1
        num_label = torch.sum(torch.ne(reward, 0.0))
        diff = -beta * (log_policy_prob - log_ref_prob)
        loss = -reward * F.logsigmoid(diff)
        loss = loss.sum() / num_label
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



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



class MaskTrainer(Trainer):
    def __init__(
        self,
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None, None),
        preprocess_logits_for_metrics = None,
        params_mask = None
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.params_mask = params_mask

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
    

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, prompt_type=data_args.prompt_type, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(data_args=data_args, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = LlamaForFGNPO.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        # device_map='auto'
    )
    print('Finish loading model.')

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
    )
    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    assert num_new_tokens == 0
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    params_mask = torch.load(data_args.mask_path, map_location='cpu')

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = MaskTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        params_mask=params_mask,
        **data_module
    )
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
