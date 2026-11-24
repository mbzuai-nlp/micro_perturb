# Each step produces output in the form {input_file_name}___{task_id}_{model_id}
# so as long as previous step model is different or current step is different it won't overwrite

CONFIG = {}

CONFIG["PREPROCESSING_CONFIG"] = {
    "TASK_ID": "preprocessing",
    # Specify number of manuscripts to skip before starting generation (e.g. if used during development)
    "SKIP_SAMPLES": 21, #19 #16 #6 used for examples and ICL, 10 for IAA round 1
    # Number of manuscripts to use for generation
    "NUM_SAMPLES": 60,
    "DATA_FOLDER_PATH": '/Users/aadi.joshi/Desktop/scientific_fact_checking/tabular_scientific_fact_checking/datasets/arxiv_sources',
    "OUTPUT_PATH": 'output/input_manuscript_ids.txt',
    "METADATA_OUTPUT_PATH": 'output/metadata/metadata.dill'
}

CONFIG["GENERATE_CONFIG"] = {
    "TASK_ID": "edit_latex",
    # Path to arxiv dataset, where each folder is the latex source for the project
    "DATA_FOLDER_PATH": '/Users/aadi.joshi/Desktop/scientific_fact_checking/tabular_scientific_fact_checking/datasets/arxiv_sources',
    # Output path for edits
    "OUTPUT_FOLDER_PATH": 'output/generate/',
    "METADATA_OUTPUT_PATH": 'output/metadata/metadata.dill',
    "STRUCTURED_OUTPUT_PATH": 'generate/structured_outputs.py',
    # where to store edited latex and pdf files
    "EDITED_OUTPUT_DIR": 'output/datasets/',
    "TASK_MODELS_INFO": {
        "gpt_54": (("openai", "gpt-5.4", "gpt_54", "output/judge_gpt54_workspace"), {"reasoning_effort": "high"}),
        "opus_5": (("openrouter", "anthropic/claude-opus-5", "opus_5", "output/opus_5_workspace"), {"reasoning_effort": "high"})
    }
}

CONFIG["GENERATE_GROUND_TRUTH_CONFIG"] = {
    "TASK_ID": 'generate_ground_truth_description',
    "TASK_MODELS_INFO": {
        "gpt_54": (("openai", "gpt-5.4", "gpt_54", "output/judge_gpt54_workspace"), {"reasoning_effort": "high"}),
        "opus_5": (("openrouter", "anthropic/claude-opus-5", "opus_5", "output/opus_5_workspace"), {"reasoning_effort": "high"})
    },
    "OUTPUT_FOLDER_PATH": 'output/ground_truth/',
}

CONFIG["DETECT_CONFIG"] = {
    "TASK_ID": 'identification_and_localization',
    "TASK_MODELS_INFO": {
        "gpt_54_nano": (("openai", "gpt-5.4-nano-2026-03-17", "gpt_54_nano", "output/detect_gpt54_nano_workspace"), {"reasoning_effort": "high"}),
        "gpt_54": (("openai", "gpt-5.4-2026-03-05", "gpt_54", "output/detect_gpt54_workspace"), {"reasoning_effort": "high"}),
        "gpt_54_mini": (("openai", "gpt-5.4-mini-2026-03-17", "gpt_54_mini", "output/detect_gpt54_mini_workspace"), {"reasoning_effort": "high"}),
        "gpt_54_nano_low": (("openai", "gpt-5.4-nano-2026-03-17", "gpt_54_nano_low", "output/detect_gpt54_nano_workspace"), {"reasoning_effort": "low"}),
        "gpt_54_mini_low": (("openai", "gpt-5.4-mini-2026-03-17", "gpt_54_mini_low", "output/detect_gpt54_mini_workspace"), {"reasoning_effort": "low"}),
        "gemini_25_flash": (("gemini", "gemini-2.5-flash", "gemini_25_flash", "output/detect_gemini_25_flash_workspace"), {"reasoning_effort": None}), #dynamic reasoning by default
        "gemini_25_flash_lite": (("gemini", "gemini-2.5-flash-lite", "gemini_25_flash_lite", "output/detect_gemini_25_flash_lite_workspace"), {"reasoning_effort": None}), #no reasoning by default
        "gemini_25_flash_high": (("gemini", "gemini-2.5-flash", "gemini_25_flash_high", "output/detect_gemini_25_flash_high_workspace"), {"reasoning_effort": "high"}),
        "gemini_25_flash_lite_high": (("gemini", "gemini-2.5-flash-lite", "gemini_25_flash_lite_high", "output/detect_gemini_25_flash_lite_high_workspace"), {"reasoning_effort": "high"}),
        "gemini_25_flash_none": (("gemini", "gemini-2.5-flash", "gemini_25_flash_none", "output/detect_gemini_25_flash_none_workspace"), {"reasoning_effort": "none"}),
        "qwen_3_vl_30b_a3b_thinking": (("openrouter", "qwen/qwen3-vl-30b-a3b-thinking", "qwen_3_vl_30b_a3b_thinking", "output/qwen_3_vl_30b_a3b_thinking"), None),
        "qwen_3_vl_8b_thinking": (("openrouter", "qwen/qwen3-vl-8b-thinking", "qwen_3_vl_8b_thinking", "output/qwen_3_vl_8b_thinking"), None),
        "qwen_3_vl_8b_instruct": (("openrouter", "qwen/qwen3-vl-8b-instruct", "qwen_3_vl_8b_instruct", "output/qwen_3_vl_8b_instruct"), None),
        # THIS DOESN'T ACTUALLY WORK - JUST MAKE SURE THAT THESE ARE ALREADY IN THE OUTPUT FOLDER SO THAT IT JUST SKIPS RUNNING
        "qwen3_vl_4b_local": (("vllm", "qwen/qwen3-vl-4b-thinking", "qwen3_vl_4b_local", "output/qwen_3_vl_4b_thinking_local"), None),
        "qwen3_vl_8b_local": (("vllm", "qwen/qwen3-vl-4b-thinking", "qwen3_vl_4b_local", "output/qwen_3_vl_8b_thinking_local"), None),
        "qwen3_vl_32b_local": (("vllm", "qwen/qwen3-vl-4b-thinking", "qwen3_vl_4b_local", "output/qwen_3_vl_32b_thinking_local"), None),
    },
    "OUTPUT_FOLDER_PATH": 'output/detect/',
}

CONFIG["JUDGE_CONFIG"] = {
    "TASK_ID": "judge",
    "TASK_MODELS_INFO": {
        'gpt_54': (("openai", "gpt-5.4", "gpt_54", "output/judge_gpt54_workspace"), {"reasoning_effort": "high"}),
        "opus_5": (("openrouter", "anthropic/claude-opus-5", "opus_5", "output/opus_5_workspace"), {"reasoning_effort": "high"}),
        'qwen_3_vl_30b_a3b_thinking': (("openrouter", "qwen/qwen3-vl-30b-a3b-thinking", "qwen_3_vl_30b_a3b_thinking", "output/qwen_3_vl_30b_a3b_thinking"), None),
    },
    "OUTPUT_FOLDER_PATH": 'output/judge/',
    "STRUCTURED_OUTPUT_PATH": 'judge/structured_outputs.py',
}

# =========FULL PIPELINE CONFIG==========
#task ids to models to use in this run, None to skip stage
class PipelineStageConfig:
    def __init__(self, model_ids, input_settings=None, run_settings=None, postprocess_settings=None):
        self.model_ids = model_ids
        # setttings that affect individual steps of the stage
        self.input_settings = input_settings
        self.run_settings = run_settings
        self.postprocess_settings = postprocess_settings

# SETTING FOR MAIN RESULTS
MAIN_PIPELINE_CONFIG = {
    "edit_latex": PipelineStageConfig(["gpt_54"]),
    "generate_ground_truth_description": PipelineStageConfig(["gpt_54"]),
    "identification_and_localization": PipelineStageConfig(CONFIG["DETECT_CONFIG"]["TASK_MODELS_INFO"].keys()),
    "judge": PipelineStageConfig(
        ["gpt_54"],
        {
            "filter_list": None,
            "filter_models": None,
        },
        {
            "output_suffix": "all",
        }
    ),
}

# SETTING FOR ALTERNATIVE GENERATION
ALT_PIPELINE_CONFIG = {
    "edit_latex": PipelineStageConfig(["opus_5"]),
    "generate_ground_truth_description": PipelineStageConfig(["opus_5"]),
    "identification_and_localization": PipelineStageConfig(["gpt_54", "gpt_54_mini", "gpt_54_nano"]),
    "judge": PipelineStageConfig(
        ["gpt_54"],
        {
            "filter_list": None,
            "filter_models": None,
        },
        {
            "output_suffix": "gpt_54_family",
        }
    ),
}

# SETTING FOR JUDGEMENT CORRELATION STUDY
JUDGE_CORRELATION_PIPELINE_CONFIG = {
    "edit_latex": PipelineStageConfig(["gpt_54"]),
    "generate_ground_truth_description": PipelineStageConfig(["gpt_54"]),
    "identification_and_localization": PipelineStageConfig(["gpt_54", "gemini_25_flash", "qwen_3_vl_30b_a3b_thinking"]),
    "judge": PipelineStageConfig(
        ["gpt_54", "opus_5", "qwen_3_vl_30b_a3b_thinking"],
        {
            "filter_list": 'supplementary_input/annotated_ids.txt',
            "filter_models": None,
        },
        {
            "output_suffix": "annotated",
        }
    )
}

# SETTING FOR MODEL JUDGEMENT BIAS STUDY
JUDGE_BIAS_PIPELINE_CONFIG = {
    "edit_latex": PipelineStageConfig(["gpt_54"]),
    "generate_ground_truth_description": PipelineStageConfig(["gpt_54"]),
    "identification_and_localization": PipelineStageConfig(["gpt_54"]),
    "judge": PipelineStageConfig(
        ["opus_5", "qwen_3_vl_30b_a3b_thinking"], #don't need to run gpt_54 again
        {
            "filter_list": None,
            "filter_models": ["gpt_54"],
        },
        {
            "output_suffix": "gpt_54_only",
        }
    ),
}
# NUM_SAMPLES = 60
# SKIP_SAMPLES = 21 #19 #16 #6 used for examples and ICL, 10 for IAA round 1
# #SKIP_SAMPLES = 3
# HUMAN_ANNOTATIONS = True
# DATA_FOLDER_PATH = Path('/Users/aadi.joshi/Desktop/scientific_fact_checking/tabular_scientific_fact_checking/datasets/arxiv_sources')

# #EDIT_MODEL_INFO = ("openai", "gpt-5.2-2025-12-11", "gpt_52", "output/gpt52_workspace")
# #EDIT_MODEL_INFO = ("openai", "gpt-5.4", "gpt_54", "output/gpt54_workspace")
# # EDIT_MODEL_INFO = ("openrouter", "anthropic/claude-opus-5", "opus_5", "output/opus_5_workspace")

# DETECTION_MODELS_INFO = [
#     #("vllm", "Qwen/Qwen3.5-4B", "qwen_35_4b", "output/detect_qwen_35_4b_workspace"),
#     #("openai", "gpt-5.2-2025-12-11", "gpt_52", "output/detect_gpt52_workspace"),
#     ("openai", "gpt-5-mini-2025-08-07", "gpt_52_mini", "output/detect_gpt52_mini_workspace"),
#     #("openrouter", "qwen/qwen3.5-397b-a17b", "qwen_35_397b", "output/detect_qwen_35_397b_workspace"),
#     #("openrouter", "qwen/qwen3.5-35b-a3b", "qwen_35_35b", "output/detect_qwen_35_35b_workspace"),
#     # ("openrouter", "google/gemini-2.5-flash-lite", "gemini_25_flash", "output/detect_gemini_25_flash_workspace"),
#     # ("openrouter", "google/gemma-4-26b-a4b-it", "gemma_4_26b_a4b_it", "output/detect_gemma_4_26b_a4b_it_workspace"),
#     #ModelClient("openrouter", "google/gemini-2.5-flash-lite", "gemini_25_flash", "output/detect_gemini_25_flash_workspace"),
#     #ModelClient("openrouter", "nvidia/nemotron-3-nano-30b-a3b", "nemotron_3_30b", "output/detect_nemotron_3_30b_workspace"),
#     #ModelClient("openrouter", "qwen/qwen3.5-9b", "qwen_35_9b", "output/detect_qwen_35_9b_workspace"),
# ]

# JUDGE_MODEL_INFO = ("openai", "gpt-5.4", "gpt_54", "output/judge_gpt54_workspace")

# OUTPUT_FOLDER_PATH = 'output/'
# WORKSPACE_FOLDER_PATH = 'workspace/'
# RESULTS_FOLDER_PATH = 'results/'

# COMPILATION_TYPE = 'PRECOMPILE' #PRECOMPILE OR JUST_IN_TIME. JIT will compile files with edits just once then delete. Precompile will store all files.