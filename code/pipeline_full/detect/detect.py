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
from shared.utils.DataHandlers import LatexDataHandler
from shared.utils.ModelClient import ModelClient
from pathlib import Path
from collections import Counter
from pdf2image import convert_from_bytes

def get_system_prompt():
    system_prompt = textwrap.dedent("""
        # Identity
        You are a peer-review assistant that identifies errors and weaknesses in scientific manuscripts.

        # Task
        You will be provided with a scientific manuscript. Your task is to identify and locate any errors, weaknesses, or inconsistencies between the claims made in the text and the values presented in the tables. You should pay particular attention to the types of errors and weaknesses described below.
                                    
        """[1:]
    )

    error_extension = textwrap.dedent("""
        # Error or Weakness
        This section contains a list of error or weakness types and their corresponding definitions.

        """[1:]
    )

    for error_category in taxonomy.TAXONOMY_DATA.keys():
        description = taxonomy.TAXONOMY_DATA[error_category]['description']
        error_name = taxonomy.TAXONOMY_DATA[error_category]['name']
        error_extension += f"## {error_name}\n{description}\n\n"

    req = textwrap.dedent("""
        # Requirements
        * You should be as specific as possible; try to provide the specific sentences or tables in the manuscript that are relevant to the identified error or weakness.
        * You should be as comprehensive as possible; identify all weaknesses or errors in the manuscript, even if the source of the issue is the same.
        * If more than one of the categories described above is relevant to an error or weakness, you should explain and attempt to localize it for each relevant category.
        * You should focus on identifying weaknesses and errors. Do not abstain from responding, do not include irrelevant information even if it may be helpful, and do not ask follow-up questions.
        """[1:]
    )

    return system_prompt + error_extension + req

SYSTEM_PROMPT = get_system_prompt()

MESSAGE_TEMPLATE_IMAGES = """Identify any inconsistencies between the text and tables in the provided manuscript. You are provided with images of the pages of the manuscript."""

MESSAGE_TEMPLATE_LATEX = """Identify any inconsistencies between the text and tables in the provided manuscript. You are provided with the latex source of the manuscript.

# Manuscript
{0}
"""

MESSAGE_TEMPLATE_IMAGES_AND_LATEX = """# Manuscript
{0}

Identify any inconsistencies between the text and tables in the provided manuscript. You are provided with the images of the pages of the manuscript and the latex source of the manuscript.
"""

# from prismm-bench code in llms-eval: https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/prismm_bench
def concat_images(images, column_num=3):
    if not images:
        return None

    if column_num == 1:
        total_height = images[0].height * len(images)
        total_width = images[0].width
    else:
        rows = math.ceil(len(images) / column_num)
        total_height = images[0].height * rows
        total_width = images[0].width * column_num

    concatenated_image = Image.new("RGB", (total_width, total_height), "white")

    x_offset, y_offset = 0, 0
    for cnt, image in enumerate(images):
        concatenated_image.paste(image, (x_offset, y_offset))
        x_offset += image.width
        if (cnt + 1) % column_num == 0:
            y_offset += image.height
            x_offset = 0

    return concatenated_image

def get_pdf_bytes(doc):
    pdf_data = doc.get("pdf")
    if pdf_data is None:
        raise ValueError("PDF data not found in document")
    return pdf_data.stream.getvalue()

# modified from prismm-bench code in llms-eval: https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/prismm_bench
def paper_to_img_grid(paper_path):
    visuals = []
    # pdf_bytes = get_pdf_bytes(doc)
    
    with open(paper_path, 'rb') as pdf:
        pdf_bytes = pdf.read()

    if not pdf_bytes:
        return visuals

    try:
        images = convert_from_bytes(pdf_bytes, dpi=144, last_page=50)  # safeguard for very large documents
        max_images = 5
        column_num = 3

        if not images:
            return visuals

        pages_per_image = math.ceil(len(images) / max_images)

        for i in range(0, len(images), pages_per_image):
            batch_pages = images[i : i + pages_per_image]
            concatenated_image = concat_images(batch_pages, column_num)
            if concatenated_image:
                visuals.append(concatenated_image.convert("RGB"))
            if len(visuals) >= max_images:
                break
    except Exception as e:
        print(e)
        pass

    return visuals

def prepare_image_req_data(paper_path):
    if f'{paper_path.stem}_visual_grid.dill' in os.listdir(paper_path.parent):
        with open(paper_path.parent / f'{paper_path.stem}_visual_grid.dill', 'rb') as fp:
            img_data = dill.load(fp)
    else:
        visuals = paper_to_img_grid(paper_path)

        img_data = []
        for vis in visuals:
            img_byte_arr = io.BytesIO()
            vis.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            base64_img = base64.b64encode(img_byte_arr).decode('utf-8')
            img_data.append(base64_img)

        with open(paper_path.parent / f'{paper_path.stem}_visual_grid.dill', 'wb') as fp:
            dill.dump(img_data, fp)

    req_data = {
        'system_prompt': SYSTEM_PROMPT,
        'message': MESSAGE_TEMPLATE_IMAGES,
        'img_data': img_data,
    }

    return req_data

def prepare_latex_req_data(paper_path):
    with open(paper_path, 'r') as fp:
        latex = fp.read()    

    req_data = {
        'system_prompt': SYSTEM_PROMPT,
        'message': MESSAGE_TEMPLATE_LATEX.format(latex),
    }

    return req_data

def prepare_images_and_latex_req_data(paper_path_images, paper_path_latex):
    req_data = prepare_image_req_data(paper_path_images)
    with open(paper_path_latex, 'r') as fp:
        latex = fp.read()    

    req_data['message'] = MESSAGE_TEMPLATE_IMAGES_AND_LATEX.format(latex)
    return req_data

def manuscript_iterator(project_data, sampling_details):
    for paper_id, error_id, data in project_data:
        meta_id = f'{paper_id}___{error_id}' #combined paper + error id
        for input_type in ['images', 'images_and_latex', 'latex']:
            meta_id_input = meta_id + '___' + input_type
            if input_type == 'images':
                paper_path = data['edited_pdf_path']
                req_data = prepare_image_req_data(paper_path)
            elif input_type == 'latex':
                paper_path = data['edited_latex_path']
                req_data = prepare_latex_req_data(paper_path)
            elif input_type == 'images_and_latex':
                paper_path_images = data['edited_pdf_path']
                paper_path_latex = data['edited_latex_path']
                req_data = prepare_images_and_latex_req_data(paper_path_images, paper_path_latex)

            if sampling_details:
                req_data = req_data | sampling_details
                
            yield (meta_id_input, req_data)

def detect_weaknesses(config, model_id, input_file, output_file, cache_path):
    #for model_client in DETECTION_CLIENTS:
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
    
    for meta_id, res in model_client.iterative_query(manuscript_iterator(project_data, sampling_details), cache_file_path=cache_path, batch=False, retry_cached_empty_response=True):
        paper_id, error_id, input_type = meta_id.split('___')
        error_id = meta_id.split('___')[1]
        print(meta_id)
        res_dict = {
            'model_id': model_client.model_prefix_name,
            'task_id': task_id,
            'input_type': input_type,
            'response_text': res
        }

        for compiled in compiled_data:
            if compiled[0] == paper_id and compiled[1] == error_id:
                if 'model_responses' in compiled[2]:
                    compiled[2]['model_responses'].append(res_dict)
                else:
                    compiled[2]['model_responses'] = [res_dict]
                break

    with open(output_file, 'wb') as fp:
        dill.dump(compiled_data, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Identify table-text weaknesses in manuscripts',
        description='Identify table-text weaknesses in a manuscript using the specified model',
    )

    parser.add_argument('config_filepath')
    parser.add_argument('model_id')
    parser.add_argument('input_file')

    args = parser.parse_args()

    config = importlib.import_module(args.config_filepath).CONFIG["DETECT_CONFIG"]
    model_id = args.model_id #model used to generate perturbations; model info must be in config for this task
    input_file = args.input_file #path to responses from perturbation step - don't actually use this info, just manuscript info

    detect_weaknesses(config, model_id, input_file)
