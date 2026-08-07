import dill
import os
from pathlib import Path

INPUT_RES_DIR = Path('./input/detection_responses')
INPUT_TRUTH_DIR = Path('./input/ground_truth/responses_generate_ground_truth_description_latex_gpt_54.dill')
OUTPUT_PATH = Path('./input_for_judge_gpt_54_all.dill')

FILTER_LIST = None #Path('./annotated_ids.txt')
FILTER_MODELS = None

if OUTPUT_PATH.is_file():
    choice = input("Input file already exists: Make sure to store all detection results before continuing. Are you sure you want to continue? (Y/n) ")
    if choice == 'n':
        print('Aborting...')
        exit()

model_responses = {}
for filename in os.listdir(INPUT_RES_DIR):
    with open(INPUT_RES_DIR / filename, 'rb') as fp:
        model_res = dill.load(fp)
        for paper_id, error_id, res in model_res:
            meta_id = f'{paper_id}___{error_id}'
            if meta_id in model_responses:
                if 'model_responses' in model_responses[meta_id] and 'model_responses' in res:
                    model_responses[meta_id]['model_responses'].extend(res['model_responses'])
            else:
                model_responses[meta_id] = res

with open(INPUT_TRUTH_DIR, 'rb') as fp:
    ground_truth = dill.load(fp)
    for paper_id, error_id, res in ground_truth:
        meta_id = f'{paper_id}___{error_id}'
        if 'ground_truth' in res:
            model_responses[meta_id]['ground_truth'] = res['ground_truth']

final_responses = []
for meta_id, res in model_responses.items():
    paper_id, error_id = meta_id.split('___')
    final_responses.append((paper_id, error_id, res))

# filter responses if needed
filter_ids = []
if FILTER_LIST is not None:
    with open(FILTER_LIST, 'r') as fp:
        for line in fp:
            meta_id = line
            #'{paper_id}___{error_id}___{model_id}___{task_id}___{input_type}'
            filter_ids.append(meta_id.strip())

if len(filter_ids) > 0:
    filtered_responses = []
    for paper_id, error_id, res in final_responses:
        filtered_model_responses = []
        if 'model_responses' not in res:
            continue

        for mr in res['model_responses']:
            model_id = mr['model_id']
            task_id = mr['task_id']
            input_type = mr['input_type']
            
            meta_id = f'{paper_id}___{error_id}___{model_id}___{task_id}___{input_type}'

            if meta_id in filter_ids:
                filtered_model_responses.append(mr)

        if len(filtered_model_responses) == 0:
            continue

        res['model_responses'] = filtered_model_responses
        filtered_responses.append((paper_id, error_id, res))
else:
    filtered_responses = final_responses

if FILTER_MODELS is not None and len(FILTER_MODELS) > 0:
    filtered_responses = []
    for paper_id, error_id, res in final_responses:
        filtered_model_responses = []

        if 'model_responses' not in res:
            continue

        for mr in res['model_responses']:
            model_id = mr['model_id']
            if model_id in FILTER_MODELS:
                filtered_model_responses.append(mr)

        if len(filtered_model_responses) == 0:
            continue

        res['model_responses'] = filtered_model_responses
        filtered_responses.append((paper_id, error_id, res))
        
print(f'Total manuscript/error pairs: {len(filtered_responses)}')

print(f'Total valid manuscript/error pairs: {len([res for res in filtered_responses if res[2].get("valid")])}')

num_model_responses = sum([len(res[2]['model_responses']) for res in filtered_responses if res[2].get("valid") and 'model_responses' in res[2]])
print(f'Total model responses: {num_model_responses}')
unique_model_ids = set([mr['model_id'] for res in filtered_responses if res[2].get("valid") and 'model_responses' in res[2] for mr in res[2]['model_responses']])
print(f'Models: {unique_model_ids}')

with open(OUTPUT_PATH, 'wb') as fp:
    dill.dump(filtered_responses, fp)