#!/bin/bash

python metadata_collector.py -e emnlp_2025 -u https://aclanthology.org/events/emnlp-2025/ -mc 1810 -fc 1406
#python latex_s3_downloader.py -e emnlp_2025 -t ./target_list.txt -m ./emnlp_2025/arxiv_metadata.json -o /Volumes/Data Backup/arxiv/
#python latex_s3_downloader.py -e emnlp_2025 -t ./required_list.txt -m ./emnlp_2025/arxiv_metadata.json --output_dir="/Volumes/Data Backup/arxiv/"
python latex_s3_downloader.py -e emnlp_2025 -t ./required_list.txt -m ./emnlp_2025/arxiv_metadata.json --output_dir="./workspace/arxiv_sources/"