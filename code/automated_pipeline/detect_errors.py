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

from utils.DataHandlers import LatexDataHandler
from utils.ModelClient import ModelClient
from pathlib import Path
from collections import Counter
from pylatexenc.latexwalker import LatexWalker, LatexMacroNode, LatexGroupNode
from pylatexenc.latex2text import LatexNodes2Text
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from pydantic import BaseModel
from openai.lib._pydantic import to_strict_json_schema

EDIT_TYPE = config.EDIT_TYPE
DATA_FOLDER_PATH = config.DATA_FOLDER_PATH


SYSTEM_PROMPT = """# Identify
You are a peer-review assistant. You will be provided with a scientific manuscript as a PDF file. Your task is to identify any limitations or errors between the claims made in the text and the values presented in tables.

# Output Format
Your output should be in valid JSON format, with the following schema:

{
    "discrepancies": [
        [str, str, str],
        ...
    ]
}

For each discrepancy that you identify, you should return:
* a string identifying the table (e.g. "Table 3")
* the sentence that contains the false claim, extracted verbatim from the manuscript
* a short explanation for why the sentence contains a false claim

Ensure that you include all sentences that contain claims contradicted by data in the tables.
"""

MESSAGE_TEMPLATE = """Identify any discrepancies between the text and tables in the provided manuscript."""


def compile_latex(latex_path):
    cmd = [
        'latexmk',
        '-pdf',
        #"-cd",  # run from source dir (helps relative paths)
        #f"-outdir={str(build_dir)}",
        latex_path.stem,
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(latex_path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,  # 10 minutes per paper
        )

        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        print(e)
        return True

def load_cached():
    responses = []

    if not os.path.isfile(f'output/responses_detect_error_{model_client.model_prefix_name}.jsonl'):
        return responses
    
    with open(f'output/responses_detect_error_{model_client.model_prefix_name}.jsonl', 'r') as res_fp:
        project_responses = []
        project_response_count = [len(v) for v in validated if len(v) > 0][len(responses)]
        for line in res_fp:
            res_dict = json.loads(line)
            project_responses.append(res_dict['res'])

            project_response_count -= 1
            if project_response_count == 0:
                responses.append(project_responses)
                project_responses = []
                project_response_count = [len(v) for v in validated if len(v) > 0][len(responses)]

    return responses

def project_compiled_latex_iterator(src_dir, dst_dir):
    num_files = len(os.listdir(src_dir))
    #do not iterate directly: iterate in order of number of edits
    for edit_num in range(num_files):
        edited_latex = f'edit_{edit_num}.tex'
        shutil.copy(src_dir / edited_latex, dst_dir / edited_latex)
        success = compile_latex(dst_dir / edited_latex)

        #compilation failed
        if not success:
            project_responses.append('')
            res_dict = {'res': ''}
            res_fp.write(json.dumps(res_dict) + '\n')
            res_fp.flush()
            continue

        paper_path = dst_dir / (edited_latex[:-3] + 'pdf')
        with open(paper_path, 'rb') as fp:
            data = fp.read()

        file_data = base64.b64encode(data).decode("utf-8")

        yield (SYSTEM_PROMPT, MESSAGE_TEMPLATE, file_data)

def compile_all_latex(src_dir, dst_dir):
    # compile all latex files
    tex_files = [f for f in os.listdir(src_dir) if f[-4:] == '.tex']
    manuscript_file_paths = []
    #do not iterate directly: iterate in order of number of edits so responses are in same order
    for edited_latex in tex_files:
        shutil.copy(src_dir / edited_latex, dst_dir / edited_latex)
        success = compile_latex(dst_dir / edited_latex)

        if not success:
            if EDIT_TYPE == 'SEQUENTIAL':
                #stop if there is a compilation error: otherwise later files will also have issues
                break
            else:
                #otherwise just ignore: we assume they're independent so order doesn't matter
                continue

        paper_path = dst_dir / (edited_latex[:-3] + 'pdf')

        # compilation still failed
        if not os.path.isfile(paper_path):
            continue

        manuscript_file_paths.append(paper_path)

    return manuscript_file_paths

def manuscript_iterator(file_paths):
    for paper_path in file_paths:
        with open(paper_path, 'rb') as fp:
            data = fp.read()

        file_data = base64.b64encode(data).decode("utf-8")

        meta_id = paper_path.stem[5:] #combined paper + error id
        yield (meta_id, SYSTEM_PROMPT, MESSAGE_TEMPLATE, file_data)

def save_responses(responses):
    with open(f'output/responses_detect_error_{model_client.model_prefix_name}.dill', 'wb') as fp:
        dill.dump(responses, fp)


if __name__ == "__main__":
    os.makedirs('workspace/curr_latex_folder/', exist_ok=True)

    # DETECTION_CLIENTS = [
    #     # ModelClient("openai", "gpt-5.2-2025-12-11", "gpt_52", "output/detect_gpt52_workspace"),
    #     ModelClient("openai", "gpt-5-mini-2025-08-07", "gpt_52_mini", "output/detect_gpt52_mini_workspace"),
    #     ModelClient("openrouter", "qwen/qwen3.5-397b-a17b", "qwen_35_397b", "output/detect_qwen_35_397b_workspace"),
    #     ModelClient("openrouter", "qwen/qwen3.5-35b-a3b", "qwen_35_35b", "output/detect_qwen_35_35b_workspace"),
    #     #ModelClient("openrouter", "google/gemini-2.5-flash-lite", "gemini_25_flash", "output/detect_gemini_25_flash_workspace"),
    #     #ModelClient("openrouter", "nvidia/nemotron-3-nano-30b-a3b", "nemotron_3_30b", "output/detect_nemotron_3_30b_workspace"),
    #     #ModelClient("openrouter", "qwen/qwen3.5-9b", "qwen_35_9b", "output/detect_qwen_35_9b_workspace"),
    # ]

    DETECTION_CLIENTS = []
    
    for dmi in config.DETECTION_MODELS_INFO:
        DETECTION_CLIENTS.append(ModelClient(*dmi))

    for model_client in DETECTION_CLIENTS:
        print(model_client.model_prefix_name)

        responses = []
        if os.path.isfile(f'output/responses_detect_error_{model_client.model_prefix_name}.dill'):
            with open(f'output/responses_detect_error_{model_client.model_prefix_name}.dill', 'rb') as fp:
                responses = dill.load(fp)

        remaining_project_ids = list(os.listdir('workspace/edited_latex_files/'))[len(responses):]

        with open(f'output/responses_detect_error_{model_client.model_prefix_name}.jsonl', 'a') as res_fp:
            for project_id in remaining_project_ids:
                print(project_id)
                shutil.copytree(DATA_FOLDER_PATH / project_id, 'workspace/curr_latex_folder/', dirs_exist_ok=True)

                src_dir = Path('workspace/edited_latex_files') / project_id
                dst_dir = Path('workspace/curr_latex_folder/')

                manuscript_file_paths = compile_all_latex(src_dir, dst_dir)

                project_responses = []
                for meta_id, res in model_client.iterative_query(manuscript_iterator(manuscript_file_paths)):
                    paper_id = meta_id.split('___')[0]
                    error_id = meta_id.split('___')[1]
                    project_responses.append((paper_id, error_id, res))

                    res_dict = {
                        'meta_id': meta_id,
                        'error_id': error_id,
                        'response': res
                    }
                    res_fp.write(json.dumps(res_dict) + '\n')
                    res_fp.flush()

                responses.append(project_responses)

                shutil.rmtree('workspace/curr_latex_folder/')
                save_responses(responses)