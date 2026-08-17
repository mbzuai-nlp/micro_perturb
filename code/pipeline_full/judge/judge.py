import json
import random
import base64
import os
import shutil
import subprocess
import dill
import nltk
import config
import itertools
import io
import math
import textwrap
import shared.taxonomy as taxonomy
import time
import importlib
import argparse

from PIL import Image
from pathlib import Path
from collections import Counter
from pdf2image import convert_from_bytes

from shared.utils.DataHandlers import LatexDataHandler
from shared.utils.ModelClient import ModelClient
from shared import config
from .structured_outputs import Score

def get_system_prompt():
    system_prompt = textwrap.dedent("""
    Compare the following: a generated review of weaknesses in a scientific paper and a ground truth weakness. Assess how well the generated review captures the ground truth weakness in terms of relatedness and specificity. Do not penalize the generated review for mentioning additional weaknesses or details not contained in the ground truth.
                                
    Rating Criteria:
    - 5 points: The generated review covers all key points from the ground truth weakness with a similar level of detail.
    - 4 points: The generated review covers all key points from the ground truth weakness, but it is less detailed than the ground truth.
    - 3 points: The generated review is related to the ground truth, but does not cover all key points.
    - 2 points: The generated review is only loosely related to the ground truth.
    - 1 point: There is no connection between the generated review and the ground truth.
                                    
    Assign a rating (1-5).
    """[1:]
    )

    return system_prompt

SYSTEM_PROMPT = get_system_prompt()

MESSAGE_TEMPLATE = """
Ground-truth weakness: 
{0}

Generated review of weaknesses:
{1}
"""

def manuscript_iterator(project_data, sampling_details):
    for paper_id, error_id, data in project_data:
        meta_id = f'{paper_id}___{error_id}' #combined paper + error id
        ground_truth = None
        for gr in data['ground_truth']:
            if gr['task_id'] == GROUND_TRUTH_TYPE:
                ground_truth = gr['response_text']
                break

        for model_res in data['model_responses']:
            model_id = model_res['model_id'] 
            task_id = model_res['task_id']
            input_type = model_res['input_type']
            response = model_res['response_text']

            meta_id = f'{paper_id}___{error_id}___{model_id}___{task_id}___{input_type}'

            message = MESSAGE_TEMPLATE.format(
                ground_truth,
                response,
            )

            req_data = {
                'system_prompt': SYSTEM_PROMPT,
                'message': message,
                'reasoning_effort': "high", 
                'structured_output_format': Score,
            }

            if sampling_details:
                req_data = req_data | sampling_details

            yield (meta_id, req_data)

def judge(config, model_id, input_file, output_file_path, cache_path, structured_output_path=None):
    task_id = config["TASK_ID"]
    model_info = config["TASK_MODELS_INFO"][model_id]
    model_client = ModelClient(*model_info[0])
    sampling_details = model_info[1]

    project_data = []
    with open(input_file, 'rb') as fp:
        compiled_data = dill.load(fp)

        for project_id, error_id, compiled in compiled_data:
            if compiled['valid'] and compiled["edited_pdf_path"] is not None and 'model_responses' in compiled:
                project_data.append((project_id, error_id, compiled))

    for meta_id, res in model_client.iterative_query(manuscript_iterator(project_data, sampling_details), cache_file_path=cache_path, batch=False, retry_cached_empty_response=False, structured_output_path=structured_output_path):
        paper_id, error_id, model_id, task_id, input_type  = meta_id.split('___')
        print(meta_id)

        for compiled in compiled_data:
            if compiled[0] == paper_id and compiled[1] == error_id and 'model_responses' in compiled[2]:
                model_responses = compiled[2]['model_responses']
                for model_res in model_responses:
                    if model_res['model_id'] == model_id and model_res['task_id'] == task_id and model_res['input_type'] == input_type:
                        model_res['judgement'] = res

    with open(output_file_path, 'wb') as fp:
        dill.dump(compiled_data, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Judge responses against ground-truth',
        description='Score responses from 1-5 based on relatedness and specificity with ground-truth',
    )

    parser.add_argument('config_filepath')
    parser.add_argument('model_id')
    parser.add_argument('input_file')

    args = parser.parse_args()

    config = importlib.import_module(args.config_filepath).CONFIG["JUDGE_CONFIG"]
    model_id = args.model_id #model used to generate perturbations; model info must be in config for this task
    input_file = args.input_file #path to responses from perturbation step - don't actually use this info, just manuscript info

    judge(config, model_id, input_file)
