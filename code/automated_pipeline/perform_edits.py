from utils.DataHandlers import LatexDataHandler
from utils.ModelClient import ModelClient
from pathlib import Path
from collections import Counter
from pylatexenc.latexwalker import LatexWalker, LatexMacroNode, LatexGroupNode
from pylatexenc.latex2text import LatexNodes2Text

import json
import random
import base64
import os
import shutil
import subprocess
import dill
import nltk
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from pydantic import BaseModel
from openai.lib._pydantic import to_strict_json_schema
import itertools

MODEL_IDENTIFIER = "gpt_52"

def calculate_distance(sentence1, sentence2):
    tokens1 = sentence1.split()
    tokens2 = sentence2.split()

    sw = SmithWaterman()
    aligned_seq1, aligned_seq2 = sw.get_alignment(tokens1, tokens2)

    # if we drop too many tokens in the alignment, just return a large number: sentences are too different
    if len(aligned_seq1) < 0.8 * len(tokens1) or len(aligned_seq2) < 0.8 * len(tokens2):
        return len(tokens1) + len(tokens2)
    else:
        distance = nltk.edit_distance(aligned_seq1, aligned_seq2)
        return distance
    
def validate_responses(responses):
    validated = []

    # validate that sentence is in latex
    for i, latex in enumerate(data_handler.iterate_flat()):
        if i == NUM_SAMPLES:
            break

        latex_sentences = sent_tokenize(latex)

        edit_data = responses[i]['edits']
        
        for sentence_edits in edit_data:
            sentence = sentence_edits['sentence']

            min_dist = 100
            best_match = ''
            for ls in latex_sentences:
                dist = calculate_distance(ls, sentence)
                if dist < min_dist:
                    min_dist = dist
                    best_match = ls
            
            acceptable_range = 0.1 * len(sentence.split())
            if min_dist > 0.1 * len(sentence.split()):
                print(sentence)
                print(best_match)
                print(acceptable_range)
                raise Exception("No match found for sentence")
    
    #check that all edits are valid: that is, each edit is in a unique table
    for i in range(NUM_SAMPLES):
        meta = data_handler.metadata[i]
        tables = meta['tables']

        edit_data = responses[i]['edits']
        validated_edits = []
        for sentence_edits in edit_data:
            edit_list = sentence_edits['edit_list']
            
            valid = True
            for edit in edit_list:
                original = edit[0]
                num_tables = 0

                for table in data_handler.metadata[i]['tables']:
                    if original in table[2]:
                        num_tables += 1

                if num_tables != 1:
                    valid = False
            
            if valid:
                validated_edits.append(sentence_edits)
        
        validated.append(validated_edits)

    return validated

def create_edited_files(validated):
    os.makedirs(f'workspace/edited_latex_files', exist_ok=True)
    for i in range(NUM_SAMPLES):
        latex = data_handler.get_latex_file(i)

        sentence_edits = validated[i]
        if len(sentence_edits) == 0:
            continue

        meta = data_handler.metadata[i]
        project_id = meta['id']
        tmp_project_path = Path(f'workspace/edited_latex_files/{project_id}')
        os.makedirs(tmp_project_path, exist_ok=True)

        for i, sentence_edit in enumerate(sentence_edits):
            edit_list = sentence_edit['edit_list']

            new_latex = latex
            for edit in edit_list:
                new_latex = new_latex.replace(edit[0], edit[1])

            with open(tmp_project_path / f'edit_{i}.tex', 'w') as fp:
                fp.write(new_latex)

            if EDIT_TYPE == 'SEQUENTIAL':
                latex = new_latex


if __name__=="__main__":
    assert os.path.isfile(f'output/responses_edit_latex_{MODEL_IDENTIFIER}.dill')

    with open(f'output/responses_edit_latex_{MODEL_IDENTIFIER}.dill', 'rb') as fp:
        responses = dill.load(fp)

    responses_dicts = []
    for res in responses:
        try:
            res_json = json.loads(res)
        except:
            res_json_compatible = res.replace('(', '[').replace(')', ']')
            res_json = json.loads(res_json_compatible)
        
    responses_dicts.append(res_json)

    validated = validate_responses(responses_dicts)

    with open('output/responses_edit_latex_gpt_52_validated.dill', 'wb') as fp:
        dill.dump(validated, fp)

    create_edited_files(validated)