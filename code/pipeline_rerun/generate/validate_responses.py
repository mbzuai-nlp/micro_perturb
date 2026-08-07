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

MODEL_IDENTIFIER = config.EDIT_MODEL_INFO[2]
EDIT_TYPE = config.EDIT_TYPE
DATA_FOLDER_PATH = config.DATA_FOLDER_PATH

def calculate_distance(sentence1, sentence2):
    # tokens1 = sentence1.split()
    # tokens2 = sentence2.split()

    # distance = nltk.edit_distance(tokens1, tokens2)
    # return distance
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

def validate_change(latex, tables, sentences, edit_list):
    latex_sentences = sent_tokenize(latex)
    valid = True
        
    for sentence in sentences:
        # check that all sentences exist in manuscript
        min_dist = 100
        best_match = ''
        for ls in latex_sentences:
            dist = calculate_distance(ls, sentence)
            if dist < min_dist:
                min_dist = dist
                best_match = ls
        
        acceptable_range = 0.1 * len(sentence.split())
        if min_dist > 0.1 * len(sentence.split()):
            valid = False
        
    # check that edits are valid for tables
    for edit in edit_list:
        original = edit['text_span_to_replace']
        num_tables = 0

        for table in tables:
            if original in table[2]:
                num_tables += 1

        if num_tables != 1:
            valid = False

    return valid
     
def validate_responses(responses):
    new_responses = []
    for res_data in responses:
        manuscript_id = res_data[0]
        error_id = res_data[1]
        sentence_edits = res_data[2] #sentences, edit_list, refuse

        if sentence_edits is None:
            sentence_edits = {
                'refuse': True,
                'valid': False,
            }

            new_responses.append((manuscript_id, error_id, sentence_edits))
            continue

        if sentence_edits['refuse']:
            sentence_edits['valid'] = False
            new_responses.append((manuscript_id, error_id, sentence_edits))
            continue

        ind = -1
        for i, meta in enumerate(data_handler.metadata):
            if meta['id'] == manuscript_id:
                ind = i
                break

        meta = data_handler.metadata[ind]
        tables = meta['tables']

        latex = data_handler.get_latex_file(ind)
        valid = validate_change(latex, tables, sentence_edits['sentences'], sentence_edits['edit_list'])
        sentence_edits['valid'] = valid

        new_responses.append((manuscript_id, error_id, sentence_edits))

    return new_responses

if __name__=="__main__":
    data_handler = LatexDataHandler(DATA_FOLDER_PATH)
    assert os.path.isfile(f'output/responses_edit_latex_{MODEL_IDENTIFIER}.dill')

    with open(f'output/responses_edit_latex_{MODEL_IDENTIFIER}.dill', 'rb') as fp:
        responses = dill.load(fp)

    responses = validate_responses(responses)

    with open(f'output/responses_edit_latex_{MODEL_IDENTIFIER}_validated.dill', 'wb') as fp:
        dill.dump(responses, fp)

    # with open('output/responses_edit_latex_gpt_52_validated.jsonl', 'w') as fp:
    #     for item in validated:
    #         fp.write(json.dumps(item) + '\n')