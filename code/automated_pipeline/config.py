from pathlib import Path

NUM_SAMPLES = 60
SKIP_SAMPLES = 21 #19 #16 #6 used for examples and ICL, 10 for IAA round 1
#SKIP_SAMPLES = 3
HUMAN_ANNOTATIONS = True
EDIT_TYPE = 'INDEPENDENT' #SEQUENTIAL or INDEPENDENT. SEQUENTIAL will combine the edits so later files have all of the previous errors too.
DATA_FOLDER_PATH = Path('/Users/aadi.joshi/Desktop/scientific_fact_checking/tabular_scientific_fact_checking/datasets/arxiv_sources')

#EDIT_MODEL_INFO = ("openai", "gpt-5.2-2025-12-11", "gpt_52", "output/gpt52_workspace")
EDIT_MODEL_INFO = ("openai", "gpt-5.4", "gpt_54", "output/gpt54_workspace")

DETECTION_MODELS_INFO = [
    #("vllm", "Qwen/Qwen3.5-4B", "qwen_35_4b", "output/detect_qwen_35_4b_workspace"),
    #("openai", "gpt-5.2-2025-12-11", "gpt_52", "output/detect_gpt52_workspace"),
    ("openai", "gpt-5-mini-2025-08-07", "gpt_52_mini", "output/detect_gpt52_mini_workspace"),
    #("openrouter", "qwen/qwen3.5-397b-a17b", "qwen_35_397b", "output/detect_qwen_35_397b_workspace"),
    #("openrouter", "qwen/qwen3.5-35b-a3b", "qwen_35_35b", "output/detect_qwen_35_35b_workspace"),
    # ("openrouter", "google/gemini-2.5-flash-lite", "gemini_25_flash", "output/detect_gemini_25_flash_workspace"),
    # ("openrouter", "google/gemma-4-26b-a4b-it", "gemma_4_26b_a4b_it", "output/detect_gemma_4_26b_a4b_it_workspace"),
    #ModelClient("openrouter", "google/gemini-2.5-flash-lite", "gemini_25_flash", "output/detect_gemini_25_flash_workspace"),
    #ModelClient("openrouter", "nvidia/nemotron-3-nano-30b-a3b", "nemotron_3_30b", "output/detect_nemotron_3_30b_workspace"),
    #ModelClient("openrouter", "qwen/qwen3.5-9b", "qwen_35_9b", "output/detect_qwen_35_9b_workspace"),
]

JUDGE_MODEL_INFO = ("openai", "gpt-5.4", "gpt_54", "output/judge_gpt54_workspace")

OUTPUT_FOLDER_PATH = 'output/'
WORKSPACE_FOLDER_PATH = 'workspace/'
RESULTS_FOLDER_PATH = 'results/'

COMPILATION_TYPE = 'PRECOMPILE' #PRECOMPILE OR JUST_IN_TIME. JIT will compile files with edits just once then delete. Precompile will store all files.