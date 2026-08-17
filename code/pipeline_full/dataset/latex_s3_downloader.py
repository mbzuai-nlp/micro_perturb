import boto3
import json
import argparse
import os
import tarfile
import shutil
import subprocess
import re
from pathlib import Path
from lxml import etree, objectify

def download_arxiv_source_manifest(out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(
        Bucket="arxiv",
        Key="src/arXiv_src_manifest.xml",
        Filename=str(out_path),
        ExtraArgs={"RequestPayer": "requester"},
    )

def download_arxiv_source_chunk(chunk, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    s3.download_file(
        Bucket="arxiv",
        Key=chunk,
        Filename=str(dst),
        ExtraArgs={"RequestPayer": "requester"},
    )

def metadata_to_chunks(metadata, arxiv_manifest):
    arxiv_manifest = objectify.fromstring(arxiv_manifest)
    boundaries = {}
    for file_ele in arxiv_manifest.file:
        filename = file_ele.filename.text
        first_item = file_ele.first_item
        last_item = file_ele.last_item

        try:
            first_identifier = first_item.text.split(".")[0]
            second_identifier = last_item.text.split(".")[0]

            first_index = int(first_item.text.split(".")[1])
            second_index = int(last_item.text.split(".")[1])
        except Exception as e:
            continue

        # this only seems to be the case for the old entries: new ones seem to have the same identifier for first and last item in a batch
        if first_identifier != second_identifier:
            continue

        if first_identifier in boundaries:
            boundaries[first_identifier].append((first_index, second_index, filename))
        else:
            boundaries[first_identifier] = [(first_index, second_index, filename)]
    
    for meta in metadata.values():
        if 'arxiv_metadata' not in meta:
            continue

        arxiv_id = meta['arxiv_metadata']['arxiv_id']
        identifier = arxiv_id.split(".")[0]
        index = int(arxiv_id.split(".")[1])

        if identifier not in boundaries:
            raise Exception(f"Identifier {identifier} not found in boundaries.")

        found_chunk = False
        for first_index, last_index, filename in boundaries[identifier]:
            if first_index <= index and index <= last_index:
                meta['arxiv_metadata']['chunk_filename'] = filename
                found_chunk = True
                break
        
        if not found_chunk:
            raise Exception(f"ArXiv ID {arxiv_id} not found in any chunk.")
        
    return metadata

def extract_and_move(metadata, chunk_dir, target_ids, output_dir):
    for meta in metadata.values():
        if 'arxiv_metadata' not in meta:
            continue

        if len(target_ids) > 0 and meta['arxiv_metadata']['arxiv_id'] not in target_ids:
            continue

        if meta['status'] == 'downloaded':
            continue

        arxiv_id = meta['arxiv_metadata']['arxiv_id']
        chunk_filename = meta['arxiv_metadata']['chunk_filename']
        chunk_path = Path(chunk_dir)/chunk_filename
        
        chunk_workspace = Path(chunk_dir) / "workspace" / Path(chunk_filename).stem
        os.makedirs(chunk_workspace, exist_ok=True)

        if not chunk_path.exists():
            raise Exception(f"Chunk file {chunk_path} does not exist.")

        with tarfile.open(chunk_path, "r") as tar:
            tar.extractall(path=chunk_workspace)

        src_filepath = chunk_workspace / arxiv_id.split(".")[0] / (arxiv_id+'.gz')
        if not src_filepath.exists():
            raise Exception(f"Source file {src_filepath} does not exist in chunk {chunk_filename}.")
        
        with tarfile.open(src_filepath, "r") as tar:
            tar.extractall(path=str(Path(output_dir)/arxiv_id))

        if not chunk_workspace.exists():
            print('Does not exist')

        shutil.rmtree(chunk_workspace, ignore_errors=True)

        meta['status'] = 'downloaded'
        meta['latex_dir'] = str(Path(output_dir)/arxiv_id)

def save_paper_status(paper_status, status_file):
    with open(status_file, 'w') as f:
        json.dump(paper_status, f)

def detect_main_tex(project_root):
    #adaptation of VimTex implementation, just using the simple stuff: https://github.com/lervag/vimtex/blob/3daee2934dfa858bbabde2053893bd46b89b4ec7/autoload/vimtex/state.vim
    # 1. Search for root specifier
    # 2. Look for \documentclass followed by \begin{document} somewhere below
    # 3. Look for common main file names
    ROOT_RE = re.compile(r'^\s*%\s*!TeX\s+root\s*=\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)
    DOCCLASS_RE = re.compile(r'\\documentclass(?:\[[^\]]*\])?\{[^}]+\}')
    BEGIN_DOC_RE = re.compile(r'\\begin\{document\}')

    COMMON_MAIN_NAMES = [
        'acl_latex.tex',
        'main.tex',
    ]

    root = Path(project_root).resolve()
    tex_files = sorted(p.resolve() for p in root.rglob("*.tex"))

    for tex_file in tex_files:
        if tex_file.name in COMMON_MAIN_NAMES:
            return tex_file
    
    main_tex_file = None
    for tex_file in tex_files:
        if tex_file.name == 'flat_main.tex':
            continue

        with open(tex_file, 'r') as fp:
            latex = fp.read()
        matches = ROOT_RE.search(latex)
        has_docclass = DOCCLASS_RE.search(latex)
        has_begin_doc = BEGIN_DOC_RE.search(latex)

        if matches:
            main_tex_file = root / matches.group(1).strip()
            return main_tex_file
        elif has_docclass and has_begin_doc:
            return tex_file
    
    return main_tex_file

def compile_latex(main_tex_path):
    project_path = os.path.dirname(main_tex_path)

    cmd = [
        'latexmk',
        '-pdf',
        main_tex_path.stem,
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=project_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,  # 10 minutes per paper
        )

        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        print(e)
        return True

def compile_all_latex(metadata):
    for meta in metadata.values():
        if meta['status'] != 'downloaded' or meta['status'] == 'compiled':
            continue
        
        project_path = meta['latex_dir']
        main_tex_path = detect_main_tex(project_path)
        compile_latex(main_tex_path)

        compiled_path = os.path.splitext(main_tex_path)[0]+'.pdf'
        if os.path.exists(compiled_path):
            meta['status'] = 'compiled'
        else:
            print(f'Failed compilation for: {meta["arxiv_data"]["arxiv_id"]}')
            meta['status'] = 'failed_compilation'

        meta['main_latex_path'] = str(main_tex_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Latex Downloader',
        description='Downloads latex source code given the arxiv metadata.',
    )

    parser.add_argument('-e', '--event_name')
    parser.add_argument('-t', '--target_list')
    parser.add_argument('-m', '--metadata_file')
    parser.add_argument('-o', '--output_dir')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / 'chunks', exist_ok=True)

    manifest_path = output_dir / "arXiv_src_manifest.xml"
    #download_arxiv_source_manifest(manifest_path)
    event_name = args.event_name
    target_list = Path(args.target_list)
    metadata_filepath = Path(args.metadata_file)

    with open(metadata_filepath, 'r') as f:
        metadata = json.load(f)

    with open(manifest_path, 'r') as f:
        arxiv_manifest = f.read()

    metadata = metadata_to_chunks(metadata, arxiv_manifest)

    target_ids = []
    with open(target_list, 'r') as f:
        target_ids = [line.strip() for line in f.readlines()]

    chunk_counts = {}
    for meta in metadata.values():
        if 'arxiv_metadata' not in meta:
            continue

        arxiv_id = meta['arxiv_metadata']['arxiv_id']
        if len(target_ids) > 0 and arxiv_id not in target_ids:
            continue

        chunk_filename = meta['arxiv_metadata']['chunk_filename']
        chunk_counts[chunk_filename] = chunk_counts.get(chunk_filename, 0) + 1

    chunk_download_list = list(chunk_counts.keys())
    print(f'Downloading {len(chunk_download_list)} chunks for {len(target_ids)} target IDs.')
    for chunk_filename in chunk_download_list:
        chunk_path = Path(chunk_filename)
        filename = chunk_path.name
        filepath = chunk_path.parent

        if filename in os.listdir(output_dir / 'chunks' / filepath):
            print(f'Chunk {chunk_filename} already exists, skipping download.')
            continue

        download_arxiv_source_chunk(chunk_filename, output_dir / 'chunks' / chunk_filename)
        print(f'Downloaded chunk {filename}')

    extract_and_move(metadata, output_dir / 'chunks', target_ids, Path(event_name) / 'arxiv_sources')
    save_paper_status(metadata, metadata_filepath)

    compile_all_latex(metadata)
    save_paper_status(metadata, metadata_filepath)

