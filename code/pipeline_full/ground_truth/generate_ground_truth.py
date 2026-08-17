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
import time
import shared.taxonomy as taxonomy
import argparse
import importlib

from PIL import Image
from shared.utils.DataHandlers import LatexDataHandler
from shared.utils.ModelClient import ModelClient
from pathlib import Path
from collections import Counter
from pdf2image import convert_from_bytes

def get_system_prompt():
    system_prompt = textwrap.dedent("""
        # Identity
        You are a helpful assistant that describes specific errors and weaknesses that are introduced into a scientific manuscript after editing.

        # Task
        Your task is to generate a description of a specified table-text error, weakness, or discrepancy that is introduced into a scientific manuscript.
                                    
        You will be provided with:
        * The latex of the original scientific manuscript.
        * The table-text error or weakness type that was introduced into the manuscript and its description.
        * A list of edits that were performed on tables to introduce the weakness into the manuscript. An edit consists of two parts: a unique span of text from the original manuscript, and a second span of text that replaces the first.
        * A list of sentences from the manuscript that demonstrate the existence of the weakness in the manuscript after the edits are performed.
        * An explanation of why the edit introduces the specified type of weakness into the manuscript.
        
        The edited manuscript can be derived from the original latex and the list of edits - it is not directly provided. You should provide a detailed description of the error or weakness and describe where in the manuscript it occurs.
    """[1:]
    )

    req = textwrap.dedent("""
        # Requirements
        * You should focus only on the error or weakness introduced by the edits and clarified by the provided explanation - do not mention other errors or weaknesses
        * You should be as specific as possible; try to provide the specific sentences or tables in the edited manuscript that are relevant.
        * The description should be derivable solely from the edited manuscript. Use the provided information to support your reasoning, but do not make any references to the specific edits that were performed or the original table values.
        * You should only provide the description. Do not ask follow-up questions or provide irrelevant information. 
        """[1:]
    )

    return system_prompt + req

SYSTEM_PROMPT = get_system_prompt()

MESSAGE_TEMPLATE = """
Manuscript Latex:
{0}
Weakness Type: 
{1}
Weakness Description: 
{2}
Edits Introduced:
{3}
Sentences Demonstrating Weakness:
{4}
Explanation of Edits:
{5}

Provide a detailed description of the weakness or error that is introduced into the manuscript by the edits.
"""

def manuscript_iterator(project_data, sampling_details=None):
    for paper_id, error_id, data in project_data:
        meta_id = f'{paper_id}___{error_id}' #combined paper + error id
        latex_path = DATA_FOLDER_PATH / paper_id / 'flat_main.tex'
        #paper_path = data['edited_latex_path']

        with open(latex_path, 'r') as fp:
            manuscript_latex = fp.read()
        
        weakness_type = base64.urlsafe_b64decode(error_id).decode()
        weakness_description = taxonomy.TAXONOMY_DATA[weakness_type]
        sentences = data['sentences']
        edit_list = data['edit_list']
        edit_reasoning = data['reasoning']

        message = MESSAGE_TEMPLATE.format(
            manuscript_latex,
            weakness_type,
            weakness_description,
            edit_list,
            sentences,
            edit_reasoning,
        )

        req_data = {
            'system_prompt': SYSTEM_PROMPT,
            'message': message,
        }

        if sampling_details:
            req_data = req_data | sampling_details

        yield (meta_id, req_data)

def generate_ground_truth(config, model_id, input_file, output_file, cache_path):
    task_id = config["TASK_ID"]
    model_info = config["TASK_MODELS_INFO"][model_id]

    model_client = ModelClient(*model_info[0])
    sampling_details = model_info[1]

    with open(input_file, 'rb') as fp:
        compiled_data = dill.load(fp)

        project_data = []
        for project_id, error_id, compiled in compiled_data:
            if compiled['valid'] and compiled["edited_pdf_path"] is not None:
                project_data.append((project_id, error_id, compiled))

    for meta_id, res in model_client.iterative_query(manuscript_iterator(project_data), cache_file_path=cache_path, batch=False):
        paper_id, error_id = meta_id.split('___')
        error_id = meta_id.split('___')[1]
        print(meta_id)
        res_dict = {
            'model_id': model_client.model_prefix_name,
            'task_id': task_id,
            'response_text': res
        }

        for compiled in compiled_data:
            if compiled[0] == paper_id and compiled[1] == error_id:
                if 'ground_truth' in compiled[2]:
                    compiled[2]['ground_truth'].append(res_dict)
                else:
                    compiled[2]['ground_truth'] = [res_dict]
                break

    with open(output_file, 'wb') as fp:
        dill.dump(compiled_data, fp)


if __name__== '__main__':
    parser = argparse.ArgumentParser(
        prog='Generate Ground-Truth Description',
        description='Converts edits and model-generated explanation for the weakness into a detailed ground-truth description from a peer-reviewer perspective (i.e, not containing details that cannot be deduced from the edited manuscript).',
    )

    parser.add_argument('config_filepath')
    parser.add_argument('model_id')
    parser.add_argument('input_file')

    args = parser.parse_args()

    config = importlib.import_module(args.config_filepath).CONFIG["GENERATE_GROUND_TRUTH_CONFIG"]
    model_id = args.model_id #model used to generate perturbations; model info must be in config for this task
    input_file = args.input_file #path to responses from perturbation step

    generate_ground_truth(config, model_id, input_file)