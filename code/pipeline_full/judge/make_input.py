import dill
import os
from pathlib import Path

def make_input(input_res_files, input_truth_dir, output_path, filter_list, filter_models):
    model_responses = {}
    for filepath in input_res_files:
        with open(filepath, 'rb') as fp:
            model_res = dill.load(fp)
            for paper_id, error_id, res in model_res:
                meta_id = f'{paper_id}___{error_id}'
                if meta_id in model_responses:
                    if 'model_responses' in model_responses[meta_id] and 'model_responses' in res:
                        model_responses[meta_id]['model_responses'].extend(res['model_responses'])
                else:
                    model_responses[meta_id] = res

    with open(input_truth_dir, 'rb') as fp:
        ground_truth = dill.load(fp)
        for paper_id, error_id, res in ground_truth:
            meta_id = f'{paper_id}___{error_id}'
            if 'ground_truth' in res:
                model_responses[meta_id]['ground_truth'] = res['ground_truth']

    final_responses = []
    for meta_id, res in model_responses.items():
        paper_id, error_id = meta_id.split('___')
        final_responses.append((paper_id, error_id, res))

    # filter responses if needed
    filter_ids = []
    if filter_list is not None:
        with open(filter_list, 'r') as fp:
            for line in fp:
                meta_id = line
                #'{paper_id}___{error_id}___{model_id}___{task_id}___{input_type}'
                filter_ids.append(meta_id.strip())

    filtered_responses = final_responses
    if len(filter_ids) > 0:
        filtered_responses = []
        for paper_id, error_id, res in final_responses:
            filtered_model_responses = []
            if 'model_responses' not in res:
                continue

            for mr in res['model_responses']:
                model_id = mr['model_id']
                task_id = mr['task_id']
                input_type = mr['input_type']
                
                meta_id = f'{paper_id}___{error_id}___{model_id}___{task_id}___{input_type}'

                if meta_id in filter_ids:
                    filtered_model_responses.append(mr)

            if len(filtered_model_responses) == 0:
                continue

            res['model_responses'] = filtered_model_responses
            filtered_responses.append((paper_id, error_id, res))

    if filter_models is not None and len(filter_models) > 0:
        filtered_responses = []
        for paper_id, error_id, res in final_responses:
            filtered_model_responses = []

            if 'model_responses' not in res:
                continue

            for mr in res['model_responses']:
                model_id = mr['model_id']
                if model_id in filter_models:
                    filtered_model_responses.append(mr)

            if len(filtered_model_responses) == 0:
                continue

            res['model_responses'] = filtered_model_responses
            filtered_responses.append((paper_id, error_id, res))

    with open(output_path, 'wb') as fp:
        dill.dump(filtered_responses, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Make Judge Input File',
        description='Make a consolidated input file for the judge',
    )

    parser.add_argument('res_dir')
    parser.add_argument('truth_dir')
    parser.add_argument('output_path')
    parser.add_argument('filter_list')
    parser.add_argument('filter_models')

    args = parser.parse_args()

    if args.output_path.is_file():
        choice = input("Input file already exists: Make sure to store all detection results before continuing. Are you sure you want to continue? (Y/n) ")
        if choice == 'n':
            print('Aborting...')
            exit()

    res_files = []
    for filename in os.listdir(args.res_dir):
        res_files.append(Path(args.res_dir) / filename)

    make_input(res_files, args.truth_dir, args.output_path, args.filter_list, args.filter_models)

    with open(args.output_path, 'rb') as fp:
        filtered_responses = dill.load(fp)
        print(f'Total manuscript/error pairs: {len(filtered_responses)}')
        print(f'Total valid manuscript/error pairs: {len([res for res in filtered_responses if res[2].get("valid")])}')
        num_model_responses = sum([len(res[2]['model_responses']) for res in filtered_responses if res[2].get("valid")])
        print(f'Total model responses: {num_model_responses}')
        unique_model_ids = set([mr['model_id'] for res in filtered_responses if res[2].get("valid") for mr in res[2]['model_responses']])
        print(f'Models: {unique_model_ids}')