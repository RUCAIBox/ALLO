import os
import json
import random

from tqdm import tqdm


random.seed(42)

pattern = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{}\n\n"
    "### Response: Let's think step by step."
)

src_path = ''
gen_folder = ''

with open(src_path, 'r') as fin:
    src_dataset = fin.readlines()
    src_dataset = [json.loads(d) for d in src_dataset]
prob2answer = {}
for data in src_dataset:
    prob2answer[data['input'].strip()] = data['output']

gen_dataset = []
for fp in os.listdir(gen_folder):
    gen_path = os.path.join(gen_folder, fp)
    with open(gen_path, 'r') as fin:
        tmp_gen_dataset = fin.readlines()
        gen_dataset = gen_dataset + tmp_gen_dataset
gen_dataset = [json.loads(d) for d in gen_dataset]
print('Total Data: ', len(gen_dataset))

gen2answer = {}
pred = {}
num_diff_data = 0
num_correct = 0
num_wrong = 0
for data in tqdm(gen_dataset):
    data['question'] = data['question'].strip()
    try:
        assert(data['question'].strip() in prob2answer)
    except AssertionError:
        print(data['question'])
        continue

    if (data['question'] not in gen2answer):
        gen2answer[data['question']] = {}
    if (data['prediction'] in gen2answer[data['question']]):
        continue
    gen2answer[data['question']][data['prediction']] = 1
    num_diff_data = num_diff_data + 1

    pred_ans = data['prediction'].split('The answer is')[-1].strip()
    real_ans = prob2answer[data['question']].split('The answer is')[-1].strip()
    if (data['question'] not in pred):
        pred[data['question']] = {
            'neg': [],
            'pos': [],
        }
    if (pred_ans.lower() == real_ans.lower()):
        pred[data['question']]['pos'].append(data['prediction'])
        num_correct = num_correct + 1
    else:
        pred[data['question']]['neg'].append(data['prediction'])
        num_wrong = num_wrong + 1

print('Num Valid Data: ', num_diff_data)
print('Num Correct Data: ', num_correct)
print('Num Wrong Data: ', num_wrong)

tgt_path = ''
fout = open(tgt_path, 'w')
num_train_data = 0
for problem, preds in pred.items():
    cand_data = []
    for pos in preds['pos']:
        for neg in preds['neg']:
            new_data = {
                'prompt': pattern.format(problem.strip()),
                'chosen': pos,
                'rejected': neg,
            }
            cand_data.append(new_data)
    num_selected = min(5, len(cand_data))
    selected_data = random.sample(cand_data, num_selected)
    for data in selected_data:
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')
        num_train_data = num_train_data + 1
fout.close()
print('Num Train Data: ', num_train_data)