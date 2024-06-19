import argparse
import os
import json
import random

from tqdm import tqdm

import torch


random.seed(42)

parser = argparse.ArgumentParser()
    
parser.add_argument("--ref_model_path", type=str)
parser.add_argument("--trained_model_path", type=str)
parser.add_argument("--result_path", type=str)
parser.add_argument("--l_per", type=int)
parser.add_argument("--r_per", type=int)
parser.add_argument("--is_neg", type=int)

args = parser.parse_args()

num_param = 0
nam2param = {}
nam2idx = {}
params = []
with torch.no_grad():
    ori_state_dict = torch.load(args.ref_model_path)
    state_dict = torch.load(args.trained_model_path)
    
    for nam, param in tqdm(state_dict.items()):
        abs_para = torch.abs(param - ori_state_dict[nam]) * args.is_neg
        param = abs_para.view(-1).to(torch.bfloat16).to('cuda:0')

        num_param = num_param + len(param)
        params.append(torch.topk(param, len(param) * args.r_per * 2 // 100).values)

    print('Total Parameters: {}'.format(num_param))

    params = torch.cat(params, dim=0)

    def get_threshold(per):
        if (per == 0): return 100000000
        topk = num_param * per // 100
        topk_output = torch.topk(params, topk)
        threshold = topk_output.values[-1]
        topk_output.values.to('cpu')
        return threshold

    params.to('cuda:1')
    l_threshold = get_threshold(args.l_per)
    params.to('cuda:2')
    r_threshold = get_threshold(args.r_per)
    print('Threshold: {} {}'.format(l_threshold, r_threshold))

    results = {}
    total_mask = 0
    with torch.no_grad():
        ori_state_dict = torch.load(args.ref_model_path)
        state_dict = torch.load(args.trained_model_path)
        
        for nam, param in tqdm(state_dict.items()):
            abs_para = (torch.abs(param - ori_state_dict[nam]) * args.is_neg).to('cuda')
            mask = torch.where(abs_para >= l_threshold, -100000000.0, abs_para)
            mask = torch.where(mask >= r_threshold, 1.0, 0.0)
            mask = mask.type(torch.bfloat16)
            total_mask = total_mask + mask.sum().item()
            results[nam] = mask

    torch.save(results, args.result_path)
    print('Mask Ratio: {}%'.format(round(total_mask / num_param * 100, 2)))
