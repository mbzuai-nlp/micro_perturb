import os
import re
import subprocess
import json
import dill
from pathlib import Path
from collections import defaultdict
from TexSoup import TexSoup
from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode

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

def detect_main_tex(project_root):
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

def compile_latex_file(latex_path):
    cmd = [
        'latexmk',
        '-pdf',
        #"-cd",  # run from source dir (helps relative paths)
        #f"-outdir={str(build_dir)}",
        latex_path.stem,
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(latex_path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,  # 2 minutes per paper
        )

        return True
    # except subprocess.TimeoutExpired:
    #     return False
    except Exception as e:
        print(e)
        return True

class LatexDataHandler:
    def __init__(self, latex_dir, output_path):
        self.latex_dir = Path(latex_dir)
        self.output_path = Path(output_path) #where metadata is stored
        self.load_metadata()
        self.filter_metadata()


    def filter_metadata(self):
        new_metadata = []
        for i, meta in enumerate(self.metadata):
            # empty latex
            data = self.get_latex_file(i)
            if len(data) < 100:
                continue

            #flattened compile failed
            if len(str(meta['flattened_pdf_path'])) == 0:
                continue

            parent_folder = self.latex_dir / meta['folder_path']
            if not os.path.isfile(parent_folder / meta['flattened_pdf_path']):
                continue

            new_metadata.append(meta)

        self.metadata = new_metadata

    def load_metadata(self):
        metadata_path = self.output_path
        self.metadata = []

        if metadata_path.is_file():
            with open(self.output_path, 'rb') as fp:
                self.metadata = dill.load(fp)

        else:
            self.load_data_paths()
            self.detect_main_tex()
            self.flatten_latex()
            self.extract_tables()
            self.compile_latex()
            self.save_metadata()
                
    def save_metadata(self):
        with open(self.output_path, 'wb') as fp:
            dill.dump(self.metadata, fp)

    # loads all path information for important files (we save this relative to initial provided path - so if directory moves, it stays consistent)
    def load_data_paths(self):
        for latex_project_folder in os.listdir(self.latex_dir):
            #skip hidden files
            if latex_project_folder[0] == '.':
                continue

            self.metadata.append({
                'id': latex_project_folder,
                'folder_path': Path(latex_project_folder)
            })
    
    def detect_main_tex(self):
        for meta in self.metadata:
            latex_project_folder_path = meta['folder_path']
            main_file = detect_main_tex(self.latex_dir / latex_project_folder_path)
            if main_file is None:
                meta['main_file_path'] = ""
            else:
                meta['main_file_path'] = Path(main_file)

    def flatten_file(self, meta, force_flatten=False):
        main_file = meta['main_file_path']
        parent_folder = self.latex_dir / meta['folder_path']
        input_file = parent_folder / main_file
        output_file = parent_folder / 'flat_main.tex'

        if force_flatten or not output_file.is_file():
            with output_file.open("w") as f:
                subprocess.run(
                    ["latexpand", str(input_file)],
                    cwd=parent_folder,
                    stdout=f,
                    check=True
                )

        meta['flattened_file_path'] = Path('flat_main.tex')
    
    def compile_latex(self):
        print("compiling...")
        for i, meta in enumerate(self.metadata):
            print(f"Compiling {i}: {meta['id']}")
            parent_folder = self.latex_dir / meta['folder_path']
            flattened_file_path = parent_folder / meta['flattened_file_path']
            compile_latex_file(flattened_file_path)

            compiled_pdf_path = Path(str(meta['flattened_file_path'].stem) + '.pdf')
            if os.path.isfile(parent_folder / compiled_pdf_path):
                meta['flattened_pdf_path'] = compiled_pdf_path
                continue

            # just do the same thing again but try force flattening the file first
            self.flatten_file(meta, force_flatten=True)
            compile_latex_file(flattened_file_path)
            if os.path.isfile(parent_folder / compiled_pdf_path):
                meta['flattened_pdf_path'] = compiled_pdf_path
                continue
            
            meta['flattened_pdf_path'] = ""

    def flatten_latex(self, force_flatten=False):
        for meta in self.metadata:
            self.flatten_file(meta, force_flatten=force_flatten)

    def find_nodes(self, node_list, node_type, node_value):
        found = []

        for node in node_list:
            if node.isNodeType(node_type) and (getattr(node, node_value[0]) == node_value[1] or getattr(node, node_value[0]) == node_value[1]+'*'):
                found.append(node)

            # check children in nodelist
            if hasattr(node, "nodelist") and node.nodelist:
                found.extend(self.find_nodes(node.nodelist, node_type, node_value))

            # check macro arguments
            if hasattr(node, "nodeargd") and node.nodeargd:
                for arg in node.nodeargd.argnlist:
                    if arg and hasattr(arg, "nodelist"):
                        found.extend(self.find_nodes(arg.nodelist, node_type, node_value))

        return found

    def extract_tables(self):
        for i, tex in enumerate(self.iterate_flat()):
            walker = LatexWalker(tex)
            nodes, _, _ = walker.get_latex_nodes()
            found = self.find_nodes(nodes, LatexEnvironmentNode, ('environmentname', 'table'))

            tables = []
            for node in found:
                pos = node.pos
                pos_end = pos + node.len
                org_text = tex[pos:pos_end]

                tables.append((pos, pos_end, org_text))

            self.metadata[i]['tables'] = tables

    def load_file(self, file_path):
        with open(file_path, 'r') as fp:
            return fp.read()
        
    def iterate_metadata(self):
        for meta in self.metadata:
            yield meta

    def iterate_flat(self):
        for meta in self.metadata:
            parent_folder = self.latex_dir / meta['folder_path']
            data = self.load_file(parent_folder / meta['flattened_file_path'])
            yield data

    def get_latex_file(self, i):
        meta = self.metadata[i]
        parent_folder = self.latex_dir / meta['folder_path']
        data = self.load_file(parent_folder / meta['flattened_file_path'])
        return data

    def get_meta_by_id(self, meta_id):
        for meta in self.iterate_metadata():
            if meta['id'] == meta_id:
                return meta
                    
    def get_latex_file_by_id(self, meta_id):
        for meta in self.iterate_metadata():
            if meta['id'] == meta_id:
                parent_folder = self.latex_dir / meta['folder_path']
                data = self.load_file(parent_folder / meta['flattened_file_path'])
                return data

    def get_path(self, meta, key):
        parent_folder = self.latex_dir / meta['folder_path']
        return parent_folder / meta[key]