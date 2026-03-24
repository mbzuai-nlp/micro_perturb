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
import config
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from pydantic import BaseModel
from openai.lib._pydantic import to_strict_json_schema
import itertools

NUM_SAMPLES = config.NUM_SAMPLES
EDIT_TYPE = config.EDIT_TYPE
DATA_FOLDER_PATH = config.DATA_FOLDER_PATH


SYSTEM_PROMPT="""# Identity
You are a latex editing assistant. Based on the task, you will provide suggestions on how to edit a latex file to meet the required criteria.

# Task
You will be provided with a scientific manuscript in latex format. Assume that all claims in the manuscript are true. Your task is to suggest edits to the tables of the manuscript so that one or more of the claims in the manuscript becomes false.

# Output Format
Your response should be in valid JSON format, with the following schema:
{
    "edits": [
        {
            "sentence": str,
            "edit_list": [
                (str, str),
                ...
            ]
        },
        ...
    ]
}

Each dictionary in the edits list represents a collection of edits that must be made so that one or more of the claims in the sentence is false. Each edit is represented by a tuple of two strings: the first string represents a unique text span in the latex document that should be replaced by the second text span.

# Example
This is an example of the output format. We demonstrate this example for a single collection of edits for a particular table and sentence, while your actual input will be a full manuscript and you can suggest edits for different sentences and claims.

## Example 1
Input:
\\begin{table}
    \\centering
     \\resizebox{0.49\\textwidth}{!}{%
    \\begin{tabular}{lccc}
    \\toprule
     & \\begin{tabular}[c]{@{}c@{}}Agreement (\\%) ($\\uparrow$)\\end{tabular}& \\begin{tabular}[c]{@{}c@{}}Krippendorff's $\\alpha$ ($\\uparrow$)\\end{tabular} & \\begin{tabular}[c]{@{}c@{}}Accuracy (\\%) ($\\uparrow$)\\end{tabular}\\\\\\midrule
    \\texttt{Simplicity Level} & 57.17                              & 0.68                                     &48.11                             \\\\
    \\texttt{Characterization} & 60.24                              & 0.55                                     & 58.05                             \\\\
    \\texttt{Legal Meaning Preservation}  & 25.96                              & 0.70                                     & 18.48  \\\\
    \\midrule
    Average & 47.74& 0.28 & 42.84\\\\
    \\bottomrule                          
    \\end{tabular}
    }
    \\caption{Annotators inter-agreements metrics per annotation task and average. $\\uparrow$ means higher is better.}
    \\label{tab:annotators_res}
\\end{table}

We can see that annotators have a high agreement over the agreement score and accuracy for the \\texttt{simplicity level} and \\texttt{characterization}.

Output:
{
    "edits": [
        "sentence": "We can see that annotators have a high agreement over the agreement score and accuracy for the \\texttt{simplicity level} and \\texttt{characterization}."
        "edits": [
            ("0.68", "0.18"),
            ("58.05", "15.03"),
        ]
    ]
}
"""

MESSAGE_TEMPLATE = """Suggest edits to the following manuscript:
{0}"""

def edit_data_iterator():
    for i in range(NUM_SAMPLES):
        latex = data_handler.get_latex_file(i)
        message = MESSAGE_TEMPLATE.format(latex)
        yield (SYSTEM_PROMPT, message)


if __name__ == "__main__":
    data_handler = LatexDataHandler(DATA_FOLDER_PATH)
    model_client = ModelClient(*config.EDIT_MODEL_INFO)

    responses = []
    with open(f'output/responses_edit_latex_{model_client.model_prefix_name}.json', 'w') as fp:
        for res in model_client.iterative_query(edit_data_iterator()):
            responses.append(res)
            res_dict = {
                'response': res
            }

            res_json = json.dumps(res_dict)

            fp.write(res_json + '\n')

    with open(f'output/responses_edit_latex_{model_client.model_prefix_name}.dill', 'wb') as fp:
        dill.dump(responses, fp)