import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
import json
import random

import numpy as np
import multiprocessing as mp

random.seed(42)

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM


ref_model_path = ""
rw_model_path = ""
tokenizer = AutoTokenizer.from_pretrained(ref_model_path)

def get_prob(input_text, output_text, model, is_return_log_prob=True):
    with torch.no_grad():
        label_ids = tokenizer(output_text + tokenizer.eos_token, add_special_tokens=False)['input_ids']
        input_ids = tokenizer(input_text + output_text + tokenizer.eos_token, add_special_tokens=False)['input_ids']
        device = model.device
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

        outputs = model(input_ids)
        
        logits = outputs.logits
        labels = input_ids
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        CEFunc = CrossEntropyLoss(reduction='none')
        vocab_size = shift_logits.size(-1)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)

        log_prob = -CEFunc(shift_logits, shift_labels)
        log_prob = log_prob[-len(label_ids):]
        if (is_return_log_prob == False):
            result = torch.exp(log_prob)
            # result = torch.where(result < 0.5, 1.0, 0.0)
        else:
            result = log_prob

        return result.tolist()

src_data_path = ""
tgt_folder_path = ""
if (os.path.exists(tgt_folder_path) == False):
    os.mkdir(tgt_folder_path)

pattern = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{}\n\n"
    "### Response: Let's think step by step."
)

REWARD_PROMPT = (
    "Given the problem and the correct solution, you need to correct the mistakes in the prediction to get the correct answer. "
    "You should write down the correct prediction and use 'The answer is: ' (without quotation mark) to identify the final answer. "
    "You should make minimal modifications. "
    "You should not copy the problem.\n\n"
    "### Problem:\n{}\n\n"
    "### Correct solution:\n{}\n\n"
    "### Prediction:\n{}\n\n"
    "### Correct prediction:\n"
)


def get_reward_prompt(d):
    input_text = REWARD_PROMPT.format(
        d['question'], 
        d['solution'], 
        d['prediction'],
    )
    output_text = d['prediction']
    return input_text, output_text


def label_data(dataset, id):
    tgt_data_path = os.path.join(tgt_folder_path, 'part-{}.jsonl'.format(id))
    fout = open(tgt_data_path, 'w')
    print(tgt_data_path)

    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_path, 
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    ).to('cuda:{}'.format(id)).eval()

    rw_model = AutoModelForCausalLM.from_pretrained(
        rw_model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    ).to('cuda:{}'.format(id)).eval()
    num_too_long = 0
    num_wrong_data = 0
    for data in tqdm(dataset):
        if (data['is_correct'] == True):
            continue
        
        num_wrong_data = num_wrong_data + 1

        problem = pattern.format(data['question'].strip())
        input_ids = tokenizer(problem + data['prediction'] + tokenizer.eos_token)['input_ids']
        if (len(input_ids) > 2048): 
            num_too_long = num_too_long + 1
            continue
        ref_prob = get_prob(problem, data['prediction'], ref_model)
        
        input_text, output_text = get_reward_prompt(data)
        reward = get_prob(input_text, output_text, model=rw_model, is_return_log_prob=False)
        
        new_data = {
            'input': problem,
            'output': data['prediction'],
            'reward': reward,
            'ref_prob': ref_prob,
        }
        fout.write(json.dumps(new_data) + '\n')
        fout.flush()

    fout.close()
    print(id, num_wrong_data, num_too_long)


if __name__ == '__main__':
    with open(src_data_path, 'r') as fin:
        dataset = fin.readlines()
        dataset = [json.loads(d) for d in dataset]

    part_num = 4
    slice_idx = np.linspace(0, len(dataset), part_num + 1).astype('int')
    p = mp.Pool(part_num)
    for start_id in range(part_num):
        start, end = slice_idx[start_id], slice_idx[start_id + 1]
        new_lines = dataset[start:end]
        print("start process %s" % start_id)
        p.apply_async(label_data, args=(new_lines, start_id))
    p.close()
    p.join()
    print("All of the child processes over!")
