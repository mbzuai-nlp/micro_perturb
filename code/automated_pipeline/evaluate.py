import json
import random
import base64
import os
import shutil
import subprocess
import dill
import nltk
import matplotlib.pyplot as plt
import itertools
import config
import pandas as pd
import taxonomy
import numpy as np

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


def find_nodes(node_list, node_type, node_value):
    found = []
    for node in node_list:
        if node.isNodeType(node_type) and (getattr(node, node_value[0]) == node_value[1] or getattr(node, node_value[0]) == node_value[1]+'*'):
            found.append(node)

        # check children in nodelist
        if hasattr(node, "nodelist") and node.nodelist:
            found.extend(find_nodes(node.nodelist, node_type, node_value))

        # check macro arguments
        if hasattr(node, "nodeargd") and node.nodeargd:
            for arg in node.nodeargd.argnlist:
                if arg and hasattr(arg, "nodelist"):
                    found.extend(find_nodes(arg.nodelist, node_type, node_value))

    return found

def count_macros_in_latex(sentence: str) -> Counter:
    walker = LatexWalker(sentence)
    nodes, pos, length = walker.get_latex_nodes()

    counts = Counter()

    def visit(nodelist):
        for node in nodelist:
            if node.isNodeType(LatexMacroNode):
                counts[node.macroname] += 1

                # recurse into macro arguments, if any
                if getattr(node, "nodeargd", None) and node.nodeargd.argnlist:
                    for arg in node.nodeargd.argnlist:
                        if arg is not None and hasattr(arg, "nodelist"):
                            visit(arg.nodelist)

            elif hasattr(node, "nodelist") and node.nodelist is not None:
                visit(node.nodelist)

    visit(nodes)
    return counts

def calculate_distance(sentence1, sentence2):
    tokens1 = sentence1.split()
    tokens2 = sentence2.split()

    distance = nltk.edit_distance(tokens1, tokens2)
    return distance

    # sw = SmithWaterman()
    # aligned_seq1, aligned_seq2 = sw.get_alignment(tokens1, tokens2)

    # # if we drop too many tokens in the alignment, just return a large number: sentences are too different
    # if len(aligned_seq1) < 0.8 * len(tokens1) or len(aligned_seq2) < 0.8 * len(tokens2):
    #     return len(tokens1) + len(tokens2)
    # else:
    #     distance = nltk.edit_distance(aligned_seq1, aligned_seq2)
    #     return distance

# Given an edit (mainly the identified false sentence) and the model response, is the error detected?
def is_error_detected(edit, res_obj):
    sentence = edit['sentence']
    sentence_text = LatexNodes2Text().latex_to_text(sentence)

    discrepancies = res_obj['discrepancies']
    # for table_identifier, manuscript_sentence, reasoning in discrepancies:
    #     if manuscript_sentence == sentence or manuscript_sentence == sentence_text:
    #         print(manuscript_sentence)
    #         print(sentence)
    #         return True
    # return False


    min_distance = len(sentence)
    matched_sentence = ''
    for table_identifier, manuscript_sentence, reasoning in discrepancies:
        distance = min(calculate_distance(sentence, manuscript_sentence), calculate_distance(sentence_text, manuscript_sentence))

        if distance < min_distance:
            min_distance = distance
            matched_sentence = manuscript_sentence

    
    acceptable_range = 0.1 * len(sentence_text.split()) + sum(count_macros_in_latex(sentence).values())
    #acceptable_range = sum(count_macros_in_latex(sentence).values())
    if min_distance < acceptable_range:
        return True
    else:
        return False

def compute_project_scores(project_edit, project_res):
    project_results = []

    responses_by_edit = {}
    for res in project_res:
        err_id = res[1]
        responses_by_edit[err_id] = res

    for edit in project_edit:
        edit_err_id = edit['error_id']
        res_data = responses_by_edit.get(edit_err_id, None)
        if res_data is None:
            project_results.append(None)
            continue

        res = res_data[2]
        edit_list = edit['edit_list']

        detected = True
        try:
            res_obj = json.loads(res[2])
            # validate object
            for table_identifier, manuscript_sentence, reasoning in res_obj['discrepancies']:
                if len(table_identifier) == 0 or len(manuscript_sentence) == 0 or len(reasoning) == 0:
                    raise Exception
        except:
            detected = None
            project_results.append(detected)
            continue
        
        detected = is_error_detected(edit, res)
        project_results.append(int(detected))

    return project_results

# def compute_project_scores(project_edit, project_res):
#     project_results = []

#     if config.EDIT_TYPE == 'SEQUENTIAL':
#         #all edits, including those before, must be captured
#         edit_lists = []
#         for edit_num in range(len(project_edit)):
#             edit_lists.append(project_edit[:edit_num+1])
#     else:
#         #only look for current edit, so we create lists of size one
#         edit_lists = []
#         for edit_num in range(len(project_edit)):
#             edit_lists.append(project_edit[edit_num:edit_num+1])
    
#     responses_by_edit = {}
#     for res in project_res:
#         err_id = res[1]
#         responses_by_edit[err_id] = res
        
#     for edit_list in edit_lists, project_res:
#         res = responses_by_edit[edit_list[-1]['error_id']][2]
#         all_detected = True
#         for edit in edit_list:
#             # register match as None if json doesn't load
#             try:
#                 res_obj = json.loads(res)
#                 # validate object
#                 for table_identifier, manuscript_sentence, reasoning in res_obj['discrepancies']:
#                     if len(table_identifier) == 0 or len(manuscript_sentence) == 0 or len(reasoning) == 0:
#                         raise Exception
#             except:
#                 all_detected = None
#                 break

#             detected = is_error_detected(edit, res_obj)

#             if not detected:
#                 all_detected = False
#                 break
        
#         if all_detected is None:
#             project_results.append(None)
#         else:
#             project_results.append(int(all_detected))

#     return project_results

def compute_json_stats(responses):
    valid_json = 0
    total_res = 0

    for project_res in responses:
        for res in project_res:
            try:
                res_obj = json.loads(res)       
                valid_json += 1
            except:
                try:
                    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', res)
                    json.loads(fixed)
                    valid_json += 1
                except:
                    pass
                pass

            total_res += 1

    print(f'Total model responses: {total_res}, Total valid JSON: {valid_json}')

def match(edit_model_prefix, detection_model_prefix):
    with open(f'output/responses_edit_latex_{edit_model_prefix}_validated.dill', 'rb') as fp:
        validated = dill.load(fp)

    with open(f'output/responses_detect_error_{detection_model_prefix}.dill', 'rb') as fp:
        responses = dill.load(fp)

    validated_edits = {}
    for v in validated:
        meta_id = v['meta_id']
        error_id = v['error_id']
        edit_list = v['edit_list']

        if len(edit_list) == 0:
            continue

        edit = edit_list[0]

        if meta_id in validated_edits:
            validated_edits[meta_id][error_id] = edit
        else:
            validated_edits[meta_id] = {error_id: edit}

    results = {}
    for meta_id in validated_edits.keys():
        for err_id in validated_edits[meta_id].keys():
            results[meta_id] = {err_id: None}

    for project_res in responses:
        if len(project_res) == 0:
            continue

        for meta_id, err_id, res in project_res:
            try:
                if res[0:8] == '```json\n':
                    res = res[8:-4]
                res_obj = json.loads(res)
                # validate object
                for table_identifier, manuscript_sentence, reasoning in res_obj['discrepancies']:
                    if len(table_identifier) == 0 or len(manuscript_sentence) == 0 or len(reasoning) == 0:
                        raise Exception
            except:
                continue

            detected = is_error_detected(validated_edits[meta_id][err_id], res_obj)
            results[meta_id][err_id] = int(detected)

    return results

# Loads suggested edits and corresponding responses from detection model. Returns 1 (error is detected),0 (error was not detected),or None (response did not load correctly)
# def match(edit_model_prefix, detection_model_prefix):
#     with open(f'output/responses_edit_latex_{edit_model_prefix}_validated.dill', 'rb') as fp:
#         validated = dill.load(fp)

#     validated_edits = [v for v in validated if len(v) > 0]

#     with open(f'output/responses_detect_error_{detection_model_prefix}.dill', 'rb') as fp:
#         responses = dill.load(fp)

#     validated_edits = {}
#     for v in validated:
#         edit_list = v['edit_list']
#         if len(edit_list) == 0:
#             continue

#         if v['meta_id'] in validated_edits:
#             validated_edits[v['meta_id']].append(v)
#         else:
#             validated_edits[v['meta_id']] = [v]

#     results = []
#     for project_res in responses:
#         if len(project_res) == 0:
#             continue

#         project_meta_id = project_res[0][0]
#         project_edit = validated_edits.get(project_meta_id, [])
#         project_results = compute_project_scores(project_edit, project_res)
#         results.append(project_results)

#     print(results[0:5])
#     print(len(list(itertools.chain.from_iterable(responses))))
#     print(len(list(itertools.chain.from_iterable(results))))
#     assert len(list(itertools.chain.from_iterable(responses))) == len(list(itertools.chain.from_iterable(results)))
#     return results

# Compute results for responses identified by the mask
# def get_scores(results, masks):
#     score = 0
#     total = sum(itertools.chain.from_iterable(masks))
#     for res, mask in zip(itertools.chain.from_iterable(results), itertools.chain.from_iterable(masks)):
#         if mask:
#             score += res

#     max_err = max(len(r) for r in results)

#     score_by_errors = []

#     for res, mask in zip(itertools.zip_longest(*results, fillvalue=None), itertools.zip_longest(*masks, fillvalue=0)):
#         s = 0
#         t = sum(mask)
#         for r, m in zip(res, mask):
#             if m:
#                 s += r

#         f = s/t if t > 0 else 0
#         score_by_errors.append((s, t, f))

#     return score/total, score_by_errors
    # for i in range(max_err):
    #     s = 0
    #     mask = [m[i] for m in masks]
    #     res = [r[i] for r in results]

    #     for r, m in zip(res, mask):
    #         if m:
    #             s += r
        
    #     t = sum(mask)
    #     err_score = s/t if t > 0 else 0
    #     score_by_errors.append(err_score)

    # return (score/total), score_by_errors

    # for res, mask in zip(results, masks):
    #     for r, m in zip(res, mask):
    #         if m :
    #             score += r

    # return score/sum(ite)
    # max_err = max(len(r) for r in results)

    # score_by_errors = []
    # for i in range(max_err):
    #     valid = 0
    #     total = 0
    #     for r in results:
    #         if i < len(r):
    #             if r[i] is not None:
    #                 valid += r[i]
    #                 total += 1

    #     score = valid/total if total > 0 else 0
    #     score_by_errors.append((valid, total, score))
    # return score_by_errors


def plot_by_errors(model_scores):
    categories = list(taxonomy.TAXONOMY_DATA.keys())

    x = np.arange(len(model_scores.keys()))  # the label locations
    width = 0.125  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    error_scores = {}

    for i, category in enumerate(categories):
        error_score = []
        for model_name, score_data in model_scores.items():
            error_score.append(score_data[i][2])

        error_scores[category] = error_score
    
    for category, scores in error_scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, scores, width, label=category)
        #ax.bar_label(rects, padding=3)
        multiplier += 1


    # for attribute, measurement in penguin_means.items():
    # for model_name, score_data in model_scores.items():
    #     offset = width * multiplier
    #     scores = [s[2] for s in score_data]
    #     rects = ax.bar(x + offset, scores, width, label=model_name)
    #     #ax.bar_label(rects, padding=3)
    #     multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    #ax.set_ylabel('Length (mm)')
    #ax.set_title('Penguin attributes by species')
    #ax.set_ylim(0, 250)
    ax.set_xticks(x + width, model_scores.keys())
    #ax.legend(loc='outside right', ncols=1)

    ax.set(ylim=(0, 1), xlabel='Model', ylabel='Pass Rate', title='Performance by Error Category')
    # ax.set_xticks(range(1, MAX_ERRORS, 1))
    #ax.grid() # Adds a grid
    ax.legend() # Adds a legend
    ax.spines[['right', 'top']].set_visible(False)

    # 5. Display the plot
    #plt.show()
    plt.savefig('results/plot_by_error.png')

    plt.show()


# def plot_by_errors(model_scores):
#     fig, ax = plt.subplots()

#     for model_name, scores in model_scores.items():
#         #ax.plot(x, y, linewidth=2.0, label='A simple line')
#         y = [ms[2] for ms in model_scores[model_name]]
#         x = list(taxonomy.TAXONOMY_DATA.keys())
#         ax.bar(x,y, label=model_name)

#     ax.set(ylim=(0, 1), xlabel='Error category', ylabel='Pass Rate', title='Performance as introduced errors is increased')
#     # ax.set_xticks(range(1, MAX_ERRORS, 1))
#     ax.grid() # Adds a grid
#     ax.legend() # Adds a legend

#     # 5. Display the plot
#     #plt.show()
#     plt.savefig('results/plot_by_error.png')

def plot_overall_avg(model_scores_all):
    fig, ax = plt.subplots()

    MAX_ERRORS = 7
    for model_name, score in model_scores_all.items():
        #ax.plot(x, y, linewidth=2.0, label='A simple line')
        #y = [ms[2] for ms in model_scores[model_name]][:MAX_ERRORS]
        #x = list(range(1, len(y)+1))
        ax.bar(model_name, score)
        #ax.plot(x,y, label=model_name)

    ax.set(ylim=(0, 1), xlabel='Model', ylabel='Pass Rate', title='Overall Performance')
    ax.spines[['right', 'top']].set_visible(False)
    #ax.set_xticks(range(1, MAX_ERRORS, 1))
    #ax.grid() # Adds a grid
    #ax.legend()

    # 5. Display the plot
    #plt.show()
    plt.savefig('results/plot_overall.png')

def print_scores(model_scores, model_scores_all):
    for model_name, scores in model_scores.items():
        print(model_name)

        print(f"{'Errors':6}    {'Correct':6}    {'Responded':10}  {'Accuracy':10}")


        corr = sum([s[0] for s in scores])
        resp = sum([s[1] for s in scores])

        print(f"{'all':6} {corr: 8} {resp: 10} {model_scores_all[model_name]:10.2f}")
        for i, score in enumerate(scores):
            print(f'{i+1:6} {score[0]:8} {score[1]:10} {score[2]:10.2f}')

def save_scores(model_scores, model_scores_all):
    data = []
    for model_name, scores in model_scores.items():
        corr = sum([s[0] for s in scores])
        resp = sum([s[1] for s in scores])

        data.append((model_name, 'all', corr, resp, model_scores_all[model_name]))
        for i, score in enumerate(scores):
            data.append((model_name, i, score[0], score[1], score[2]))

    df = pd.DataFrame(data, columns=["model", "subset", "correct", "responses", "accuracy"])
    df.to_csv("results/results.csv", index=False)

def get_scores(results, masks):
    score = 0
    total_responses = 0
    err_scores = {}
    for meta_id in masks.keys():
        for err_id in masks[meta_id].keys():
            if err_id not in err_scores:
                err_scores[err_id] = []

            mask = masks[meta_id][err_id]
            if mask == 1:
                res = results[meta_id][err_id]
                score += res
                total_responses += 1
                err_scores[err_id].append(res)

    score_by_errors = []
    for err_cat in taxonomy.TAXONOMY_DATA.keys():
        err_id = taxonomy.TAXONOMY_DATA[err_cat]['id']
        scores = err_scores.get(err_id, [])
        s = sum(scores)
        t = len(scores)
        f = s/t if t > 0 else 0
        score_by_errors.append((s, t, f))
    # for err_id, scores in err_scores.items():
    #     s = sum(scores)
    #     t = len(scores)
    #     f = s/t if t > 0 else 0
    #     score_by_errors.append((s, t, f))

    return score/total_responses, score_by_errors

if __name__ == "__main__":
    model_matches = {}

    for dmi in config.DETECTION_MODELS_INFO:
        model_prefix = dmi[2]
        model_match = match('gpt_52', model_prefix)
        model_matches[model_prefix] = model_match

    with open('output/evaluation_matches.dill', 'wb') as fp:
        dill.dump(model_matches, fp)

    minimum_response_set = {}
    for model_name in model_matches.keys():
        for meta_id in model_matches[model_name].keys():
            minimum_response_set[meta_id] = {}
            for err_id in model_matches[model_name][meta_id].keys():
                minimum_response_set[meta_id][err_id] = 1
    
    for meta_id in minimum_response_set.keys():
        for model_name in model_matches.keys():
            if meta_id not in model_matches[model_name]:
                for err_id in minimum_response_set[meta_id].keys():
                    minimum_response_set[meta_id][err_id] = 0

        for err_id in minimum_response_set[meta_id].keys():
            for model_name in model_matches.keys():
                if meta_id not in model_matches[model_name]:
                    continue

                if err_id not in model_matches[model_name][meta_id]:
                     minimum_response_set[meta_id][err_id] = 0
                     continue

                if model_matches[model_name][meta_id][err_id] is None:
                    minimum_response_set[meta_id][err_id] = 0

    # for model_name in model_matches.keys():
    #     print(f'==================================={model_name}===================================')
    #     print(model_matches[model_name])
        
    #     for meta_id in model_matches[model_name].keys():
    #         for err_id in model_matches[model_name][meta_id].keys():
    #             if model_matches[model_name][meta_id][err_id] is None:
    #                 minimum_response_set[meta_id][err_id] = 0

    # minimum_response_set = []
    # for i, project_res in enumerate(model_matches[model_name]):
    #     minimum_response_set.append([1]*len(project_res))

    # for model_name in model_matches.keys():
    #     for i, project_res in enumerate(model_matches[model_name]):
    #         for j, res in enumerate(project_res):
    #             if res is None:
    #                     minimum_response_set[i][j] = 0

    # print(f'Total responses: {len(list(itertools.chain.from_iterable(minimum_response_set)))}')
    # print(f'Minimum valid responses: {sum(itertools.chain.from_iterable(minimum_response_set))}')

    model_scores = {}
    model_scores_all = {}
    for model_prefix, model_match in model_matches.items():
        model_match = model_matches[model_prefix]
        all_scores, score_by_errors = get_scores(model_match, minimum_response_set)
        model_scores_all[model_prefix] = all_scores
        model_scores[model_prefix] = score_by_errors

    print_scores(model_scores, model_scores_all)
    os.makedirs('results', exist_ok=True)
    save_scores(model_scores, model_scores_all)
    plot_by_errors(model_scores)
    plot_overall_avg(model_scores_all)
