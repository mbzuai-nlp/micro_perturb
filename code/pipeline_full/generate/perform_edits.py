

import json
import random
import base64
import os
import shutil
import subprocess
import dill
import nltk
import itertools
import config
import argparse
import importlib
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from pydantic import BaseModel
from openai.lib._pydantic import to_strict_json_schema
from shared.utils.DataHandlers import LatexDataHandler
from shared.utils.ModelClient import ModelClient
from pathlib import Path
from collections import Counter
from pylatexenc.latexwalker import LatexWalker, LatexMacroNode, LatexGroupNode
from pylatexenc.latex2text import LatexNodes2Text
from string2string.alignment import SmithWaterman

def create_edited_files(validated, data_handler):
    os.makedirs(f'workspace/edited_latex_files', exist_ok=True)
    for res_data in validated:
        manuscript_id = res_data[0]
        error_id = res_data[1]
        sentence_edits = res_data[2] #sentences, edit_list, refuse

        if sentence_edits['refuse'] or not sentence_edits['valid']:
            sentence_edits['edited_latex_path'] = None
            continue

        ind = -1
        for i, meta in enumerate(data_handler.metadata):
            if meta['id'] == manuscript_id:
                ind = i
                break

        latex = data_handler.get_latex_file(ind)

        meta = data_handler.metadata[ind]

        project_id = meta['id']
        tmp_project_path = Path(f'workspace/edited_latex_files/{project_id}')
        os.makedirs(tmp_project_path, exist_ok=True)

        edit_list = sentence_edits['edit_list']

        new_latex = latex
        for edit in edit_list:
            new_latex = new_latex.replace(edit['text_span_to_replace'], edit['new_text_span'])

        output_path = tmp_project_path / f'edit_{manuscript_id}___{error_id}.tex'
        with open(output_path, 'w') as fp:
            fp.write(new_latex)

        sentence_edits['edited_latex_path'] = output_path

        # if EDIT_TYPE == 'SEQUENTIAL':
        #     latex = new_latex

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
    
def compile_edited_files(validated, dataset_path):
    for i, data in enumerate(validated):
        project_id, error_id, res_data = data
        edited_latex_path = res_data['edited_latex_path']

        if edited_latex_path is None:
            validated[i][2]['edited_pdf_path'] = None
            continue

        dst_dir = dataset_path / project_id
        # copy original project folder to dataset folder in workspace
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copytree(DATA_FOLDER_PATH / project_id, dst_dir, dirs_exist_ok=True)

        #copy specific file to project latex folder
        target_path = dst_dir / edited_latex_path.name
        shutil.copy(edited_latex_path, target_path)

        #compile
        success = compile_latex(target_path)

        paper_path = dst_dir / (edited_latex_path.stem + '.pdf')
        if not success or not os.path.isfile(paper_path):
            validated[i][2]['edited_pdf_path'] = None
            continue

        validated[i][2]['edited_pdf_path'] = paper_path

    return validated


def edit_and_compile(validated, data_handler, dataset_path):
    os.makedirs(dataset_path, exist_ok=True)
    for i, res_data in enumerate(validated):
        manuscript_id = res_data[0]
        error_id = res_data[1]
        sentence_edits = res_data[2] #sentences, edit_list, refuse

        if sentence_edits['refuse'] or not sentence_edits['valid']:
            sentence_edits['edited_latex_path'] = None
            continue

        meta = data_handler.get_meta_by_id(manuscript_id)
        latex = data_handler.get_latex_file_by_id(manuscript_id)

        project_path = dataset_path / manuscript_id
        shutil.copytree(data_handler.latex_dir / manuscript_id, project_path, dirs_exist_ok=True)

        edit_list = sentence_edits['edit_list']

        new_latex = latex
        for edit in edit_list:
            new_latex = new_latex.replace(edit['text_span_to_replace'], edit['new_text_span'])

        output_path = project_path / f'edit_{manuscript_id}___{error_id}.tex'
        with open(output_path, 'w') as fp:
            fp.write(new_latex)

        sentence_edits['edited_latex_path'] = output_path

        success = compile_latex(output_path)

        paper_path = project_path / (output_path.stem + '.pdf')
        if not success or not os.path.isfile(paper_path):
            validated[i][2]['edited_pdf_path'] = None
            continue

        validated[i][2]['edited_pdf_path'] = paper_path


def perform_edits(config, input_file, output_file, dataset_path):
    data_handler = LatexDataHandler(config["DATA_FOLDER_PATH"], config["METADATA_OUTPUT_PATH"])
    # data_handler.flatten_latex(force_flatten=True)

    assert os.path.isfile(input_file)

    with open(input_file, 'rb') as fp:
        validated = dill.load(fp)

    edit_and_compile(validated, data_handler, dataset_path)
    # create_edited_files(validated, data_handler)

    # with open(output_file, 'wb') as fp:
    #     dill.dump(validated, fp)

    # compile_edited_files(validated, dataset_path)

    with open(output_file, 'wb') as fp:
        dill.dump(validated, fp)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        prog='Perform edits',
        description='Perform validated edits on the files and compile them',
    )

    parser.add_argument('config_filepath')
    parser.add_argument('validated_filepath')
    parser.add_argument('output_path')

    args = parser.parse_args()

    config = importlib.import_module(args.config_filepath).CONFIG["GENERATE_CONFIG"]

    perform_edits(config, args.validated_filepath, args.output_path)