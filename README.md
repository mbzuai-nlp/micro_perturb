# Micro-Perturb: An Automated Perturbation-Based Framework for Evaluating LLMs on Table-Text Weakness Detection

## Overview

Companion repository to the paper "Micro-Perturb: An Automated Perturbation-Based Framework for Evaluating LLMs on Table-Text Weakness Detection".

## Abstract

Scholarly peer-reviewing is a critical but time-consuming task. Existing work has studied how language models can assist in the review process, with scientific errors and weaknesses detection emerging as a key application. Motivated by the existing work on scientific claim verification, we introduce the task of \emph{table--text weakness detection} in scientific manuscripts. Through an inductive qualitative analysis of peer-review data, we identify five main categories of table--text weaknesses. We then introduce a guided perturbation pipeline for introducing such weaknesses into scientific manuscripts, and evaluate models on their ability to detect them. Extensive human validation demonstrates that our pipeline reliably generates realistic weaknesses across most categories. Finally, we benchmark various vision-language models on both the full generated dataset and a human-validated subset. Our results show that the models are able to reliably detect explicit logical inconsistencies, but struggle with more subtle semantic, contextual, and methodological weaknesses.

## Usage
Install requirements.

`pip install -r requirements.txt`

Also install the shared package containing utility code:

`cd pipeline_full`

`pip install -e .`

The full pipeline can be found in `pipeline_full`. The file `pipeline_full/config.py` can be edited to change the configuration of the pipeline run. It contains preconfigured pipelines that were used in the paper.

To run the pipeline, run the `pipeline_full/run.py` script. All outputs will be stored in the `output` folder by default unless the configuration is changed.

For models run locally using VLLM, move the output files into the detection output folder (by default this is `pipeline_full/output/detect/`) and run the pipeline as usual.
