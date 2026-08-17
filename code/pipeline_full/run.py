import os
import shutil
import config
import argparse
import importlib
from shared.utils.DataHandlers import LatexDataHandler
from shared.utils.ModelClient import ModelClient
from pathlib import Path
from generate.generate_edits import generate_edits
from generate.perform_edits import perform_edits
from generate.validate_responses import validate_responses
from ground_truth.generate_ground_truth import generate_ground_truth
from detect.detect import detect_weaknesses
from judge.make_input import make_input
from judge.judge import judge

TASK_IDS = [
    "edit_latex",
    "generate_ground_truth_description",
    "identification_and_localization",
    "judge"
]

class Artifact:
    path: Path
    cache_path: Path
    stage_info: dict

    def __init__(self, filepath):
        if type(filepath) is str:
            filepath = Path(filepath)

        self.path = filepath
        self.cache_path = filepath.parent / (str(filepath.stem) + '_cache.dill')
        filename = str(filepath.stem)
        filemeta = filename[filename.find('_')+1:] #anything before first underscore is just a prefix

        all_stage_metadata = filemeta.split('___')

        stage_info = {}
        for stage_metadata in all_stage_metadata:
            for task_name in TASK_IDS:
                if stage_metadata.startswith(task_name):
                    model_id = stage_metadata.split(task_name)[1][1:]
                    stage_info[task_name] = model_id

        self.stage_info = stage_info

# Load paths to all artifacts relevant to CURRENT run
class ArtifactStore:
    def __init__(self, config):
        self.artifacts = {}

    def add_artifact(self, task, artifact):
        self.artifacts[task] = self.artifacts.get(task, []).append(artifact)

    def add_artifacts(self, task, artifacts_list):
        self.artifacts[task] = artifacts_list

    def get_artifacts(self, task):
        return self.artifacts[task]

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.artifact_store = ArtifactStore(config)
        self.stages = []

    def add_stage(self, stage):
        self.stages.append(stage)

    def run(self):
        for stage in self.stages:
            stage.prepare_input(self.artifact_store)
            stage.run()
            stage.postprocess()
            output = stage.output_artifacts
            self.artifact_store.add_artifacts(stage.stage_name, output)

class Stage:
    def __init__(self, config, stage_config):
        self.config = config
        self.stage_name = config['TASK_ID']
        self.stage_config = stage_config

        #if no model id specified, do not run anything for this step - only cached output is used
        if stage_config:
            model_ids = stage_config.model_ids
            if model_ids is None:
                self.task_models = self.config['TASK_MODELS_INFO'].keys()
                self.skip = True
            else:
                self.task_models = model_ids
                self.skip = False

    def prepare_input(self, artifact_store):
        pass

    def run(self):
        pass

    def postprocess(self):
        pass

class PreprocessingStage(Stage):
    def run(self):
        self.output_artifacts = []
        #when data handler is instantiated it will load files in the directory, flatten them, compile them, save metadata, etc.
        data_handler = LatexDataHandler(self.config["DATA_FOLDER_PATH"])
        start = self.config["SKIP_SAMPLES"]
        end = self.config["SKIP_SAMPLES"]+self.config["NUM_SAMPLES"]

        output_path = self.config["OUTPUT_PATH"]
        with open(output_path, 'w') as fp:
            for i in range(start, end):
                meta_id = data_handler.metadata[i]['id']
                fp.write(meta_id + '\n')

        artifact = Artifact(output_path)
        self.output_artifacts.append(artifact)

class GenerationStage(Stage):
    def prepare_input(self, artifact_store):
        self.input_artifacts = artifact_store.get_artifacts('preprocessing')
    
    def run(self):
        self.output_artifacts = []
        output_folder = self.config['OUTPUT_FOLDER_PATH']

        structured_output_path = None
        if "STRUCTURED_OUTPUT_PATH" in self.config:
            structured_output_path = self.config["STRUCTURED_OUTPUT_PATH"]
        # should usually only have one input artifact - just a list of manuscript ids
        for artifact in self.input_artifacts:
            input_file = artifact.path
            for model_id in self.task_models:
                cache_path = f'{output_folder}responses_{self.stage_name}_{model_id}_cache.dill'
                output_path = f'{output_folder}responses_{self.stage_name}_{model_id}.dill'

                # already have output, skip this step
                if os.path.exists(output_path):
                    print(f"Using cached results for generation: {model_id}")
                    artifact = Artifact(output_path)
                    self.output_artifacts.append(artifact)
                    continue

                if self.skip:
                    print(f'Skipping edit generation for {model_id} (pipeline config)')
                    continue

                print(f'Generating edits for {input_file} using {model_id}, storing in {output_path}, caching in {cache_path}')
                generate_edits(self.config, model_id, input_file, output_path, cache_path, structured_output_path)
                artifact = Artifact(output_path)
                self.output_artifacts.append(artifact)

    def postprocess(self):
        for artifact in self.output_artifacts:
            input_file = artifact.path
            output_file = input_file.parent / (str(input_file.stem) + '_validated.dill')
            if os.path.exists(output_file):
                continue

            print(f'Validating: {input_file} -> {output_file}')
            validate_responses(self.config, input_file, output_file)

        for artifact in self.output_artifacts:
            input_file = artifact.path.parent / (str(artifact.path.stem) + '_validated.dill')
            output_file = artifact.path.parent / (str(artifact.path.stem) + '_compiled.dill')
            if os.path.exists(output_file):
                continue

            print(f'Editing: {input_file} -> {output_file}')
            # make separate datasets for each generation - the model name can be retrieved from the artifact
            generation_model = artifact.stage_info[self.stage_name]
            dataset_path = Path(self.config["EDITED_OUTPUT_DIR"]) / f'dataset_{generation_model}'
            perform_edits(self.config, input_file, output_file, dataset_path)
            move_path = artifact.path.parent / (str(artifact.path.stem) + '_precompilation.dill')
            print(f'Moving: {artifact.path} -> {move_path}')
            shutil.move(artifact.path, move_path)
            print(f'Copying: {output_file} -> {artifact.path}')
            shutil.copy(output_file, artifact.path)

class GroundTruthStage(Stage):
    def prepare_input(self, artifact_store):
        self.input_artifacts = artifact_store.get_artifacts('edit_latex')

    def run(self):
        self.output_artifacts = []
        output_folder = self.config['OUTPUT_FOLDER_PATH']
        task_id = self.config['TASK_ID']

        for input_artifact in self.input_artifacts:
            input_path = input_artifact.path
            
            for model_id in self.task_models:
                output_path = Path(output_folder) / (str(input_path.stem) + f'___{task_id}_{model_id}.dill')
                cache_path = Path(output_folder) / (str(input_path.stem) + f'___{task_id}_{model_id}_cache.dill')
                # already have output, skip this step
                if os.path.exists(output_path):
                    print(f"Using cached results for ground truth generation: {model_id}")
                    artifact = Artifact(output_path)
                    self.output_artifacts.append(artifact)
                    continue

                if self.skip:
                    print(f'Skipping ground truth generation for {model_id} (pipeline config)')
                    continue

                print(f'Generating ground-truth for {input_path} using {model_id}, storing in {output_path}, caching in {cache_path}')
                generate_ground_truth(self.config, model_id, input_path, output_path, cache_path)
                artifact = Artifact(output_path)
                self.output_artifacts.append(artifact)

class DetectionStage(Stage):
    def prepare_input(self, artifact_store):
        self.input_artifacts = artifact_store.get_artifacts('edit_latex')

    def run(self):
        self.output_artifacts = []
        output_folder = self.config['OUTPUT_FOLDER_PATH']
        task_id = self.config['TASK_ID']

        for input_artifact in self.input_artifacts:
            input_path = input_artifact.path
            for model_id in self.task_models:
                output_path = Path(output_folder) / (str(input_path.stem) + f'___{task_id}_{model_id}.dill')
                cache_path = Path(output_folder) / (str(input_path.stem) + f'___{task_id}_{model_id}_cache.dill')
                if os.path.exists(output_path):
                    print(f"Using cached results for detection: {model_id}")
                    artifact = Artifact(output_path)
                    self.output_artifacts.append(artifact)
                    continue

                if self.skip:
                    print(f'Skipping detection for {model_id} (pipeline config)')
                    continue

                print(f'Generating detection results for {input_path} using {model_id}, storing in {output_path}, caching in {cache_path}')
                detect_weaknesses(self.config, model_id, input_path, output_path, cache_path)
                artifact = Artifact(output_path)
                self.output_artifacts.append(artifact)

class JudgeStage(Stage):
    def prepare_input(self, artifact_store):
        input_artifacts = []
        ground_truth_artifacts = artifact_store.get_artifacts("generate_ground_truth_description")
        response_artifacts = artifact_store.get_artifacts("identification_and_localization")
        response_paths = [res.path for res in response_artifacts]

        suffix = self.stage_config.run_settings['output_suffix']
        filter_list = self.stage_config.input_settings.get('filter_list', None)
        filter_models = self.stage_config.input_settings.get('filter_models', None) 
    
        # create input file for every ground truth artifact (should usually only be one ground truth file per run)
        for ground_truth_artifact in ground_truth_artifacts:
            ground_truth_path = ground_truth_artifact.path
            ground_truth_filename = str(ground_truth_path.stem)
            ground_truth_suffix = ground_truth_filename[ground_truth_filename.find('_')+1:] #everything after response_ 
            file_output_path = Path(self.config['OUTPUT_FOLDER_PATH']) / f'input_{ground_truth_suffix}_{suffix}.dill' #responses
            print(f'Making judge input, filtering by filter list ({filter_list}), filter models ({filter_models})')
            make_input(response_paths, ground_truth_path, file_output_path, filter_list, filter_models) # don't really need filter models here
            input_artifacts.append(Artifact(file_output_path))

        self.input_artifacts = input_artifacts

    def run(self):
        self.output_artifacts = []
        output_folder = self.config['OUTPUT_FOLDER_PATH']
        task_id = self.config['TASK_ID']
        suffix = self.stage_config.run_settings['output_suffix']

        structured_output_path = None
        if "STRUCTURED_OUTPUT_PATH" in self.config:
            structured_output_path = self.config["STRUCTURED_OUTPUT_PATH"]

        for input_artifact in self.input_artifacts:
            input_path = input_artifact.path
            output_path_prefix = 'responses_' + str(input_path.stem)[len('input_'):-(len(suffix)+1)]

            for model_id in self.task_models:
                output_path = Path(output_folder) / (output_path_prefix + f'___{task_id}_{model_id}_{suffix}.dill')
                cache_path = Path(output_folder) / (output_path_prefix + f'___{task_id}_{model_id}_cache.dill') #suffix doesn't affect cache: it's just filtered, but same result
                if os.path.exists(output_path):
                    print(f"Using cached results for judging: {model_id}")
                    artifact = Artifact(output_path)
                    self.output_artifacts.append(artifact)
                    continue

                if self.skip:
                    print(f'Skipping judging for {model_id} (pipeline config)')
                    continue

                print(f'Generating judge results for {input_path} using {model_id}, storing in {output_path}, caching in {cache_path}')
                judge(self.config, model_id, input_path, output_path, cache_path, structured_output_path)
                artifact = Artifact(output_path)
                self.output_artifacts.append(artifact)


PIPELINE_CONFIG = config.MAIN_PIPELINE_CONFIG
# Each stage: prepare input, run, postprocess
pipeline = Pipeline(config.CONFIG)

preprocess_stage = PreprocessingStage(
    config.CONFIG["PREPROCESSING_CONFIG"], 
    None
)
pipeline.add_stage(preprocess_stage)

generate_stage = GenerationStage(
    config.CONFIG["GENERATE_CONFIG"],
    PIPELINE_CONFIG["edit_latex"],
)
pipeline.add_stage(generate_stage)

ground_truth_stage = GroundTruthStage(
    config.CONFIG["GENERATE_GROUND_TRUTH_CONFIG"],
    PIPELINE_CONFIG["generate_ground_truth_description"]
)
pipeline.add_stage(ground_truth_stage)

detect_stage = DetectionStage(
    config.CONFIG["DETECT_CONFIG"],
    PIPELINE_CONFIG["identification_and_localization"]
)
pipeline.add_stage(detect_stage)

judge_stage = JudgeStage(
    config.CONFIG["JUDGE_CONFIG"],
    PIPELINE_CONFIG["judge"]
)
pipeline.add_stage(judge_stage)

pipeline.run()