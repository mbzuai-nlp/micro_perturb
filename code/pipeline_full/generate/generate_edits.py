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
import textwrap
import math
import io
import argparse
import importlib

import shared.taxonomy as taxonomy
from shared.utils.DataHandlers import LatexDataHandler
from shared.utils.ModelClient import ModelClient
from pathlib import Path
from collections import Counter
from pdf2image import convert_from_bytes
from PIL import Image
from pylatexenc.latexwalker import LatexWalker, LatexMacroNode, LatexGroupNode
from pylatexenc.latex2text import LatexNodes2Text
from .structured_outputs import Edit, Flaw


def get_system_prompt():
    system_prompt= textwrap.dedent("""\
        # Identity
        You are a latex editing assistant. Based on the task, you will provide suggestions on how to edit a latex file to meet the required criteria.

        # Task
        ## Input
        You will be provided with a scientific manuscript in latex format, images of the pages of the compiled output pdf, and a description for a type of error or weakness. You will also be provided with examples in JSON format containing a table from a manuscript, some extracted sentences, and a list of edits that introduce the required type of error.

        ## Task Description        
        Your task is to suggest edits to the tables of the manuscript so that the described error or weakness is introduced into the manuscript. Before attempting the task, you should determine whether the specified type of error can be introduced into the provided manuscript. If the task is not possible, you should refuse.
                                   
        You may update values in existing tables, update the structure of tables by adding or removing rows and columns, add new tables, or delete tables. Captions are not considered part of the table and should not be edited.
        
        ## Constraints
        You must also ensure the following:
        - That the edits you introduce are over tables: that is, that they edit, add or delete tables.
        - That you do not include edits that are not required.
        - That you do not include sentences that are not required to verify that the error has been introduced into the manuscript.
        - That you do not extract sentences out of context. The sentence should be related to error and the tables to which the edits are performed.
        - That any edits you introduce into a table do not change its internal consistency; that is, it should not be possible to detect the existence of an error or weakness from the table alone. You can assume that the manuscript is originally correct.
        - That the edits do not introduce any of the other defined categories of errors or weaknesses.
        - That you only introduce one error at a time.
        - That the values you edit do not appear elsewhere in the paper, including other tables, figures, or the text of the manuscript. This is only acceptable if it aligns with the definition of the error or weakness type.
        
        ## Refusal Criteria
        You should refuse to edit the manuscript if it does not contain the type of textual claims required for the error. If you are unsure, you should choose to refuse rather than attempting to insert the error or weakness.
                                   
        You should also refuse if the requested type of error or weakness cannot be introduced into the manuscript without also introducing one of the other defined errors or weaknesses. If you are unsure, you should choose to refuse rather than attempting to insert the error or weakness.
                                   
        Finally, you should refuse if completing the task requires violating any of the other defined constraints, such as the constraint on editing, adding, or deleting tables only.
                                   
        # Output Format
        Your response should be in valid JSON format, with the following schema:
        {
            "refuse": bool,
            "sentences": list[str],
            "edit_list": list[list[str]],
            "reasoning": str,
        }
                                   
        The "refuse" field should be a boolean that is true if you are unable to suggest edits that would introduce the weakness or error into the manuscript, and false otherwise. If you choose to refuse, you must also provide your reasoning for refusing the request.
                                   
        The "sentences" field should include a minimal set of sentences extracted verbatim from the manuscript that are essential for detecting the error after any edits. These are sentences that were originally correct, but may be weakly supported or contradicted by the table after the edits. You may include up to three sentences, but provide fewer when possible.

        The "edit_list" field should include a list of edits, where each edit is a list of two strings. The first string in the list is a text span from the original manuscript that you suggest to replace, and the second string is the new text span that you suggest to introduce into the manuscript. The text spans in the "edit_list" should be chosen such that when the suggested edits are applied to the original manuscript, the resulting manuscript would contain the described flaw. You may add new tables by selecting a portion of an existing table, and replacing it with the same portion appended with the new table. You may delete a table by selecting the full table text and replacing it with the empty string.
                                   
        Finally, the "reasoning" field should include either a short explanation for why the edits introduce the specified error or weakness into the manuscript. If you refused to provide edits, the "reasoning" field should include an explanation for why the specified error or weakness could not be introduced into the manuscript.
        """
    )
    return system_prompt

def get_system_prompt_with_error_definitions():
    system_prompt= get_system_prompt()

    error_extension = textwrap.dedent(\
        """
        # Error or Weakness
        This section contains a list of error or weakness types and their corresponding definitions. While there may be some overlap in definitions, when introducing edits you should aim to introduce only one of these specified types of errors or weaknesses at a time.


        """
    )

    for error_category in taxonomy.TAXONOMY_DATA.keys():
        description = taxonomy.TAXONOMY_DATA[error_category]['description']
        error_name = taxonomy.TAXONOMY_DATA[error_category]['name']
        error_extension += f"## {error_name}\n{description}\n\n"

    return system_prompt + error_extension


#=======IMG req==========
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

def prepare_image_data(paper_path):
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

    return img_data
#=================================

def get_message(latex, error):
    MESSAGE_TEMPLATE = textwrap.dedent("""\
        # Manuscript
        {0}

        # Error or Weakness Description
        {1}
        """
    )
    name = error['name']
    description = error['description']
    out_str = f"{name}: {description}"

    message = MESSAGE_TEMPLATE.format(latex, out_str)
    examples = error['examples']
    if len(examples) > 0:
        example_str = "# Examples\n"
        for i, example in enumerate(examples):
            example_str += f"## Example {i+1}\n"
            example_str += json.dumps(example)
        message = message + '\n' + example_str

    return message

def prepare_text_req(latex, error):
    message = get_message(latex, error)
    req_data = {
        'system_prompt': get_system_prompt_with_error_definitions(),
        'message': message,
    }

    return req_data

def edit_data_iterator(data_handler, manuscript_ids, sampling_details=None):
    for manuscript_id in manuscript_ids:
        latex = data_handler.get_latex_file_by_id(manuscript_id)
        meta = data_handler.get_meta_by_id(manuscript_id)
        print(meta['id'])
        main_pdf_path = data_handler.get_path(meta, 'flattened_pdf_path')
        for error_name, error in taxonomy.TAXONOMY_DATA.items():
            error_id = error['id']
            req_id = f"{manuscript_id}___{error_id}"
            req_data = prepare_text_req(latex, error)
            img_data = prepare_image_data(main_pdf_path)
            req_data['img_data'] = img_data

            #add sampling details
            req_data['structured_output_format'] = Flaw

            if sampling_details:
                req_data = req_data | sampling_details

            yield (req_id, req_data)

def generate_edits(config, model_id, input_path, output_path, cache_path, structured_output_path=None):
    data_handler = LatexDataHandler(config["DATA_FOLDER_PATH"])

    manuscript_ids = []
    with open(input_path, 'r') as fp:
        for line in fp:
            manuscript_ids.append(line.strip())

    model_info = config["TASK_MODELS_INFO"][model_id]

    model_client = ModelClient(*model_info[0])
    sampling_details = model_info[1]

    req_iterator = edit_data_iterator(data_handler, manuscript_ids, sampling_details)
    responses = []
    for req_id, res in model_client.iterative_query(req_iterator, cache_file_path=cache_path, batch=False, retry_cached_empty_response=False, structured_output_path=structured_output_path):
        meta_id, error_id = req_id.split('___')

        if res is None:
            responses.append((meta_id, error_id, res))
            continue

        res_dict = {
            'sentences': res.sentences,
            'edit_list': [
                {
                    'text_span_to_replace': edit.text_span_to_replace, 
                    'new_text_span': edit.new_text_span
                } for edit in res.edits
            ],
            'refuse': res.refuse,
            'reasoning': res.reasoning,
        }
        
        responses.append((meta_id, error_id, res_dict))

    with open(output_path, 'wb') as fp:
        dill.dump(responses, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Generate Perturbations',
        description='Generates a list of edits, relevant sentences, and reasoning for each input latex manuscript and weakness type',
    )

    parser.add_argument('config_filepath')
    parser.add_argument('model_id')
    parser.add_argument('input_path', help="Path to the input file, containing one manuscript ID per line")
    parser.add_argument('output_path')
    parser.add_argument('-cache_path')
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if args.cache_path:
        cache_path = Path(args.cache_path)
    else:
        cache_path = output_path.parent / (str(input_path.stem) + '_cache.dill')

    config = importlib.import_module(args.config_filepath).CONFIG["GENERATE_CONFIG"]
    model_id = args.model_id #model used to generate perturbations; model info must be in config for this task

    generate_edits(config, model_id, input_path, output_path, cache_path)