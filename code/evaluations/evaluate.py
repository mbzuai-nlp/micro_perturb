import json
import random
import base64
import os
import shutil
import subprocess
import dill
import nltk
import itertools
import io
import math
import textwrap
import taxonomy
import time
import numpy as np
import evaluate
import argparse

from PIL import Image
from pathlib import Path
from collections import Counter
from pydantic import BaseModel
from scipy import stats
import numpy as np

HUMAN_ANNOTATIONS_FILE_PATH = Path('./aj_main.json')

#MODEL_RESPONSES_FILE_PATH = Path('responses_judge_opus_5_gpt_54_only.dill')
#MODEL_RESPONSES_FILE_PATH = Path('responses_judge_qwen_3_vl_30b_a3b_thinking_gpt_54_only.dill')

input_types = [
    'images', 
    'latex', 
    'images_and_latex'
]

input_types_to_name = {
    'images': 'Images',
    'latex': 'Latex',
    'images_and_latex': r'Images \& Latex',
}

error_categories = [
    'contradiction_with_established_knowledge', 
    'logically_invalid_interpretation', 
    'methodology_result_conflict', 
    'semantic_misinterpretation', 
    'suboptimal_methodology'
]

error_cat_to_acronym = {
    'contradiction_with_established_knowledge': 'ConEK',
    'logically_invalid_interpretation': 'LogII',
    'methodology_result_conflict': 'MetRC',
    'semantic_misinterpretation': 'SemM',
    'suboptimal_methodology': 'SubM'
}

error_category_names = [taxonomy.TAXONOMY_DATA[cat]['name'] for cat in error_categories]
error_category_acronyms = [error_cat_to_acronym[cat] for cat in error_categories]

model_ids = [
    'gpt_54', 
    # 'gpt_54_mini',
    # 'gpt_54_mini_low',
    # 'gpt_54_nano',
    # 'gpt_54_nano_low', 
    # 'gemini_25_flash', 
    # 'gemini_25_flash_lite', 
    # 'gemini_31_flash_lite', 
    # 'qwen_3_vl_30b_a3b_thinking', 
    # 'qwen_3_vl_8b_thinking', 
    # 'qwen_3_vl_8b_instruct'
]

model_id_to_name = {
    'gpt_54': 'GPT 5.4 (high)', 
    'gpt_54_mini': 'GPT 5.4-mini (high)',
    'gpt_54_nano': 'GPT 5.4-nano (high)',
    'gpt_54_mini_low': 'GPT 5.4-mini (low)',
    'gpt_54_nano_low': 'GPT 5.4-nano (low)',
    'gemini_25_flash': 'Gemini 2.5 Flash (dynamic)',
    'gemini_25_flash_lite': 'Gemini 2.5 Flash-lite (none)',
    'gemini_31_flash_lite': 'Gemini 3.1 Flash-lite (high)',
    'qwen_3_vl_30b_a3b_thinking': 'Qwen3-VL 30BA3B Thinking',
    'qwen_3_vl_8b_thinking': 'Qwen3-VL 8B Thinking',
    'qwen_3_vl_8b_instruct': 'Qwen3-VL 8B Instruct',
}

class Score(BaseModel):
    final_score: int

def get_choice_value(annotation):
    annotations = annotation['annotations']
    for ann in annotations:
        if ann['was_cancelled']:
            return None
        else:
            for res in ann['result']:
                if res['type'] == 'choices' and res['from_name'] == 'edit_correct':
                    choice = res['value']['choices'][0]
                    if choice == 'yes':
                        return 1
                    else:
                        return 0

def print_merged_table(metric1_scores, metric1_name, metric2_scores, metric2_name, precision=1):
    print(r'\begin{tabular}{c|ccccc|ccccc}')
    print(r'    \toprule')
    #print(r'    Model & \multicolumn{5}{|c|}{images} & \multicolumn{5}{|c|}{latex} & \multicolumn{5}{|c|}{images_and_latex} \\')
    print(r' & \multicolumn{5}{c}{'+f'{metric1_name}'+r'} & \multicolumn{5}{c}{'+f'{metric2_name}'+r'} \\')
    print(r'    Model & ' + ' & '.join(error_category_acronyms) + ' & ' + ' & '.join(error_category_acronyms) + r' \\')
    print(r'    \midrule')

    for model_id in model_ids:
        model_name = model_id_to_name[model_id]
        model_row = f'   {model_name} & ' + '& ' * (len(error_category_acronyms)-1) + r' \\'
        print(model_row)
        print(r'    \midrule')
        for input_type in input_types:
            input_name = input_types_to_name[input_type]
            outp = f'   {input_name} & '
            for metric_scores in [metric1_scores, metric2_scores]:
                scores = metric_scores
                for error_cat in error_categories:
                    error_cat_b64 = base64.urlsafe_b64encode(error_cat.encode()).decode()
                    score_array = scores[f'{model_id}___{input_type}___{error_cat_b64}']
                    score = np.mean(score_array)
                    if precision == 1:
                        outp += f'{score:.1f} & '
                    else:
                        outp += f'{score:.2f} & '
            print(outp[:-2] + r'\\' )

        print(r'    \midrule')
    print(r'\bottomrule')
    print(r'\end{tabular}')

def load_results(file_path):
    with open(file_path, 'rb') as fp:
        responses = dill.load(fp)

    # convert judgements into json if required
    skip_count = {}
    for paper_id, error_id, res in responses:
        if 'model_responses' in res:
            for model_response in res['model_responses']:
                # if never got judge response, just skip that instance 
                if 'judgement' not in model_response or model_response['judgement'] is None:
                    model_id = model_response['model_id']
                    skip_count[model_id] = skip_count.get(model_id, 0) + 1
                    continue

                if type(model_response['judgement']) == Score:
                    judgement = model_response['judgement'].final_score
                else:
                    try:
                        judgement = model_response['judgement']['final_score']
                    except:
                        judgement = json.loads(model_response['judgement'])
                        model_response['judgement'] = judgement

    print(skip_count)

    results = [] # all scores with model_id, error_id, paper_id so it can be grouped later
    counts = {} #count of valid responses by error id - but using this as a denominator is wrong when the judgement is missing for a specific model?
    for paper_id, error_id, res in responses:
        if res['valid']:
            counts['all'] = counts.get('all', 0) + 1
            counts[error_id] = counts.get(error_id, 0) + 1

        if 'model_responses' in res:
            for model_response in res['model_responses']:
                if 'judgement' not in model_response or model_response['judgement'] is None: #do I need to adjust counts for specific models here???
                    continue

                model_id = model_response['model_id']
                #add task_id here if we add other tasks
                input_type = model_response['input_type']
                try:
                    score = model_response['judgement']['final_score']
                except:
                    score = model_response['judgement'].final_score

                results.append((model_id, input_type, error_id, paper_id, score))


    # valid responses
    print('Number of valid responses by error category:')
    for k,v in counts.items():
        if k == 'all':
            print(f'total: {v}')
        else:
            print(f'{base64.b64decode(k).decode():50}: {v}')

    return results

def results_to_scores(results):
    scores = {}
    #Group scores by (model, input_type, error_id). So basically all scores for papers for each of those key combinations.
    for model, input_type, error_id, paper_id, score in results:
        meta_id = f'{model}___{input_type}___{error_id}'
        scores[meta_id] = scores.get(meta_id, []) + [score]

    return scores
    
def print_scores_list(scores):
    print('All grouped scores:')

    for meta_id, score_array in sorted(scores.items(), key=lambda x: (x[0].split('___')[0], x[0].split('___')[1], x[0].split('___')[2])):
        avg_score = np.mean(score_array)
        model, input_type, error_id = meta_id.split('___')
        error_category = base64.b64decode(error_id).decode()
        print(f'{model:30} {input_type:20} {error_category:50} {avg_score:3.2}')

def print_scores_table(scores):
    print(r'\begin{tabular}{c|ccccc}')
    print(r'    \toprule')
    #print(r'    Model & \multicolumn{5}{|c|}{images} & \multicolumn{5}{|c|}{latex} & \multicolumn{5}{|c|}{images_and_latex} \\')
    print(r'    Model & ' + ' & '.join(error_category_acronyms) + r' \\')
    print(r'    \midrule')

    for model_id in model_ids:
        model_name = model_id_to_name[model_id]
        model_row = f'   {model_name} & ' + '& ' * (len(error_category_acronyms)-1) + r' \\'
        print(model_row)
        print(r'    \midrule')
        for input_type in input_types:
            input_name = input_types_to_name[input_type]
            outp = f'   {input_name} & '
            for error_cat in error_categories:
                error_cat_b64 = base64.urlsafe_b64encode(error_cat.encode()).decode()
                score_array = scores[f'{model_id}___{input_type}___{error_cat_b64}']
                score = np.mean(score_array)
                outp += f'{score:.2} & '
            print(outp[:-2] + r'\\' )

        print(r'    \midrule')
    print(r'\bottomrule')
    print(r'\end{tabular}')

def print_merged_flipped_table(metric1_scores, metric1_name, metric2_scores, metric2_name, precision=1):
    print(r'\begin{tabular}{ll*{' + str(len(model_ids)) + r'}{c}||*{' + str(len(model_ids)) + r'}{c}}')
    print(r'& & \multicolumn{' + str(len(model_ids)) + r'}{c}{' + metric1_name + r'} & \multicolumn{' + str(len(model_ids)) + r'}{c}{' + metric2_name + r'} \\')
    print(r'Input & Weakness')

    outp = ''
    for i in range(2):
        for model_id in model_ids:
            model_name = model_id_to_name[model_id]
            outp += r'& \rotatebox{90}{' + model_name + r'}' + '\n'

    outp = outp[:-1] + r' \\'
    print(outp)
    print(r'\midrule')

    input_types_to_name = {
        'images': 'Images',
        'latex': 'Latex',
        'images_and_latex': r'Combined',
    }
    
    score_matrix = np.empty((len(input_types)*len(error_categories), 2*len(model_ids)), dtype=object)

    for i, input_type in enumerate(input_types):
        for j, error_cat in enumerate(error_categories):
            error_cat_b64 = base64.urlsafe_b64encode(error_cat.encode()).decode()
            for k, scores in enumerate([metric1_scores, metric2_scores]):
                for l, model_id in enumerate(model_ids):
                    score_array = scores[f'{model_id}___{input_type}___{error_cat_b64}']
                    #score = np.mean(score_array)
                    score_matrix[i*len(error_categories)+j, k*len(model_ids)+l] = score_array

    for i, input_type in enumerate(input_types):
        input_name = input_types_to_name[input_type]
        print(r'\multirow{5}{*}{\rotatebox{90}{' + input_name +r'}}')
        for j, error_cat in enumerate(error_categories):
            error_cat_b64 = base64.urlsafe_b64encode(error_cat.encode()).decode()
            error_cat_acronym = error_cat_to_acronym[error_cat]

            outp = ''
            outp += r'& ' + error_cat_acronym
            for k, scores in enumerate([metric1_scores, metric2_scores]):
                for l, model_id in enumerate(model_ids):
                    score = np.mean(score_matrix[i*len(error_categories)+j, k*len(model_ids)+l])

                    max_other = 0
                    for x in range(len(input_types)*len(error_categories)):
                        if x == i*len(error_categories)+j:
                            continue
                        oscore_array = score_matrix[x, k*len(model_ids)+l]
                        oscore = np.mean(oscore_array)
                        max_other = max(oscore, max_other)

                    score = float(f'{score:.1f}')
                    max_other = float(f'{max_other:.1f}')

                    if score == max_other:
                        outp+= (r' & \underline{' + f'{score:.1f}' + r'}')
                    elif score > max_other:
                        outp+= (r' & \textbf{' + f'{score:.1f}' + r'}')
                    else:
                        outp+= (r' & ' + f'{score:.1f}')

            outp += r' \\'

            print(outp)
        print(r'\midrule')

    print(r'\bottomrule')
    print(r'\end{tabular}')

def load_human_validation_results():
    with open(HUMAN_ANNOTATIONS_FILE_PATH, 'r') as fp:
        annotations = json.load(fp)

    human_validation = {}
    for annotation in annotations:
        paper_id = annotation['data']['id']
        error_id = annotation['data']['error_id']
        meta_id = f'{paper_id}___{error_id}'
        annotation_value = get_choice_value(annotation)
        error_name = base64.urlsafe_b64decode(error_id).decode()
        human_validation[meta_id] = annotation_value

    return human_validation

def results_to_human_valid_scores(results):
    human_validation = load_human_validation_results()

    human_valid_scores = {}
    for model, input_type, error_id, paper_id, score in results:
        meta_id = f'{model}___{input_type}___{error_id}'
        human_valid = human_validation.get(f'{paper_id}___{error_id}', None)
        if human_valid == 1:
            human_valid_scores[meta_id] = human_valid_scores.get(meta_id, []) + [score]

    return human_valid_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Analyze',
        description='Produces average score from LLM-as-judge scores',
    )

    parser.add_argument('judge_results_path')
    args = parser.parse_args()

    judge_results_path = Path(args.judge_results_path)
    results = load_results(judge_results_path)
    scores = results_to_scores(results)
    human_valid_scores = results_to_human_valid_scores(results)

    assert len(scores) == len(human_valid_scores.keys()) # The number of models, input_type, error_type combinations should be the same.Basically 15 per model (3x inputs, 5x types).

    print_merged_table(scores, 'Full Dataset', human_valid_scores, 'Human Validated Subset')