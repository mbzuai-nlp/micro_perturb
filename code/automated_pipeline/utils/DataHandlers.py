import os
import re
import subprocess
import json
import dill
from pathlib import Path
from collections import defaultdict
from TexSoup import TexSoup
from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode

ROOT_RE = re.compile(r'^\s*%\s*!TeX\s+root\s*=\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)
INPUT_RE = re.compile(r'\\(?:input|include)\{([^}]+)\}')
DOCCLASS_RE = re.compile(r'\\documentclass(?:\[[^\]]*\])?\{[^}]+\}')
BEGIN_DOC_RE = re.compile(r'\\begin\{document\}')
SUBFILES_CLASS_RE = re.compile(r'\\documentclass(?:\[[^\]]*\])?\{subfiles\}')
SUBFILES_USE_RE = re.compile(r'\\usepackage(?:\[[^\]]*\])?\{subfiles\}')

# def strip_comments(text: str) -> str:
#     lines = []
#     for line in text.splitlines():
#         # remove unescaped %
#         parts = re.split(r'(?<!\\)%', line, maxsplit=1)
#         lines.append(parts[0])
#     return "\n".join(lines)

# def normalize_tex_ref(base: Path, ref: str) -> Path:
#     ref = ref.strip()
#     p = (base / ref)
#     if p.suffix == "":
#         p = p.with_suffix(".tex")
#     return p.resolve()

# def read_text_safe(path: Path) -> str:
#     try:
#         return path.read_text(encoding="utf-8")
#     except UnicodeDecodeError:
#         return path.read_text(encoding="latin-1")

# def find_tex_files(root: Path):
#     return sorted(p.resolve() for p in root.rglob("*.tex"))

# def extract_root_magic(path: Path):
#     text = read_text_safe(path)
#     m = ROOT_RE.search(text)
#     if not m:
#         return None
#     target = m.group(1).strip()
#     return normalize_tex_ref(path.parent, target)

# def extract_includes(path: Path):
#     text = strip_comments(read_text_safe(path))
#     refs = INPUT_RE.findall(text)
#     return [normalize_tex_ref(path.parent, ref) for ref in refs]

# def file_features(path: Path):
#     text = strip_comments(read_text_safe(path))
#     return {
#         "has_documentclass": bool(DOCCLASS_RE.search(text)),
#         "has_begin_document": bool(BEGIN_DOC_RE.search(text)),
#         "is_subfiles_class": bool(SUBFILES_CLASS_RE.search(text)),
#         "uses_subfiles_pkg": bool(SUBFILES_USE_RE.search(text)),
#     }

# def detect_main_tex(project_root: str):
#     root = Path(project_root).resolve()
#     tex_files = find_tex_files(root)
#     if not tex_files:
#         return None, "No .tex files found"

#     # 1) explicit magic comments
#     magic_targets = defaultdict(list)
#     for f in tex_files:
#         target = extract_root_magic(f)
#         if target:
#             magic_targets[target].append(f)

#     existing_magic_targets = {k: v for k, v in magic_targets.items() if k.exists()}
#     if existing_magic_targets:
#         # choose target referenced by the most files
#         best_target = max(existing_magic_targets.items(), key=lambda kv: len(kv[1]))[0]
#         return best_target, "Chosen from % !TeX root directives"

#     # 2) build include graph
#     includes = {}
#     included_by = defaultdict(set)
#     for f in tex_files:
#         children = [c for c in extract_includes(f) if c.exists()]
#         includes[f] = children
#         for c in children:
#             included_by[c].add(f)

#     # 3) score candidates
#     scores = {}
#     reasons = {}

#     for f in tex_files:
#         feats = file_features(f)
#         score = 0
#         why = []

#         if feats["has_documentclass"]:
#             score += 50
#             why.append("has \\documentclass")
#         if feats["has_begin_document"]:
#             score += 50
#             why.append("has \\begin{document}")
#         if not included_by[f]:
#             score += 20
#             why.append("not included by another file")
#         if includes[f]:
#             score += min(len(includes[f]) * 5, 20)
#             why.append(f"includes {len(includes[f])} file(s)")
#         if feats["uses_subfiles_pkg"]:
#             score += 5
#             why.append("uses subfiles package")
#         if feats["is_subfiles_class"]:
#             score -= 20
#             why.append("is a subfiles child document")

#         scores[f] = score
#         reasons[f] = why

#     best = max(scores, key=scores.get)
#     return best, "; ".join(reasons[best])


# function! s:file_is_main(file) abort " {{{1
#   if !filereadable(a:file) | return 0 | endif

#   let l:preamble = vimtex#parser#preamble(a:file, {
#         \ 'inclusive' : 1,
#         \ 'root' : fnamemodify(a:file, ':p:h'),
#         \})

#   " Check if a:file is a main file by looking for the \documentclass command,
#   " but ignore the following:
#   " * \documentclass[...]{subfiles}
#   " * \documentclass[...]{standalone}
#   let l:lines = copy(l:preamble)
#   call filter(l:lines, 'v:val =~# ''^\s*\\documentclass\_\s*[\[{]''')
#   call filter(l:lines, 'v:val !~# ''{subfiles}''')
#   call filter(l:lines, 'v:val !~# ''{standalone}''')
#   if len(l:lines) == 0 | return 0 | endif

#   " A main file must also contain `\begin{document}`
#   let l:lines = copy(l:preamble)
#   call filter(l:lines, 'v:val =~# ''^\s*\\begin\s*{document}''')
#   return len(l:lines) > 0
# endfunction

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

# class LatexManuscript:
#     def __init__(self, project_id, latex_file_path, metadata=None):
#         self.project_id = project_id
#         self.latex_path = latex_file_path
#         self.metadata = metadata or {}

#     def load(self):
#         with open(self.latex_path, 'r') as fp:
#             return fp.read()

#     def compile_latex(self):
#         cmd = [
#             'latexmk',
#             '-pdf',
#             self.latex_path.stem,
#         ]

#         try:
#             proc = subprocess.run(
#                 cmd,
#                 cwd=str(self.latex_path.parent),
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.STDOUT,
#                 text=True,
#                 timeout=600,  # 10 minutes per paper
#             )

#             return True
#         except subprocess.TimeoutExpired:
#             return False
#         except Exception as e:
#             print(e)
#             return True

#     def find_nodes(self, node_list, node_type, node_value):
#         found = []

#         for node in node_list:
#             if node.isNodeType(node_type) and (getattr(node, node_value[0]) == node_value[1] or getattr(node, node_value[0]) == node_value[1]+'*'):
#                 found.append(node)

#             # check children in nodelist
#             if hasattr(node, "nodelist") and node.nodelist:
#                 found.extend(self.find_nodes(node.nodelist, node_type, node_value))

#             # check macro arguments
#             if hasattr(node, "nodeargd") and node.nodeargd:
#                 for arg in node.nodeargd.argnlist:
#                     if arg and hasattr(arg, "nodelist"):
#                         found.extend(self.find_nodes(arg.nodelist, node_type, node_value))

#         return found

#     def extract_tables(self):
#         tex = self.load()
#         walker = LatexWalker(tex)
#         nodes, _, _ = walker.get_latex_nodes()
#         found = self.find_nodes(nodes, LatexEnvironmentNode, ('environmentname', 'table'))

#         tables = []
#         for node in found:
#             pos = node.pos
#             pos_end = pos + node.len
#             org_text = tex[pos:pos_end]

#             tables.append((pos, pos_end, org_text))

#         self.tables = tables

# class LatexProject:
#     def __init__(self, project_id, project_folder_path):
#         self.project_id = project_id
#         self.project_folder_path = project_folder_path
#         self.latex_files = []

#     def detect_main_tex(self):
#         latex_project_folder_path = meta['folder_path']
#         main_file, reason = detect_main_tex(latex_project_folder_path)
#         meta['main_file_path'] = latex_project_folder_path.parent / main_file

#     def add_latex_file(self, latex_file_path):
#         self.latex_files.append(LatexManuscript(self.project_id, latex_file_path))

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
    def __init__(self, latex_dir):
        self.latex_dir = Path(latex_dir)
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

            if not os.path.isfile(meta['flattened_pdf_path']):
                continue

            new_metadata.append(meta)

        self.metadata = new_metadata

    def load_metadata(self):
        metadata_path = Path('metadata.dill')
        self.metadata = []

        if metadata_path.is_file():
            with open('metadata.dill', 'rb') as fp:
                self.metadata = dill.load(fp)
                # for line in fp:
                #     data = json.loads(line)
                #     self.metadata.append(data)
        else:
            self.load_data_paths()
            self.detect_main_tex()
            self.flatten_latex()
            self.extract_tables()
            self.compile_latex()
            self.save_metadata()
                
    def save_metadata(self):
        with open('metadata.dill', 'wb') as fp:
            dill.dump(self.metadata, fp)
        #     for meta in self.metadata:
        #         meta_json = json.dumps(meta)
        #         fp.write(meta_json + '\n')

    def load_data_paths(self):
        for latex_project_folder in os.listdir(self.latex_dir):
            #skip hidden files
            if latex_project_folder[0] == '.':
                continue

            self.metadata.append({
                'id': latex_project_folder,
                'folder_path': self.latex_dir / latex_project_folder
            })
    
    def detect_main_tex(self):
        for meta in self.metadata:
            latex_project_folder_path = meta['folder_path']
            main_file = detect_main_tex(latex_project_folder_path)
            if main_file is None:
                meta['main_file_path'] = ""
            else:
                meta['main_file_path'] = latex_project_folder_path.parent / main_file

    def flatten_file(self, meta, force_flatten=False):
        main_file = meta['main_file_path']
        output_file = main_file.parent / 'flat_main.tex'

        if force_flatten or not output_file.is_file():
            with output_file.open("w") as f:
                subprocess.run(
                    ["latexpand", str(main_file)],
                    cwd=main_file.parent,
                    stdout=f,
                    check=True
                )

        meta['flattened_file_path'] = output_file
    
    def compile_latex(self):
        print("compiling...")
        for i, meta in enumerate(self.metadata):
            print(f"Compiling {i}: {meta['id']}")
            compile_latex_file(meta['flattened_file_path'])
            compiled_pdf_path = Path(os.path.splitext(meta['flattened_file_path'])[0] + '.pdf')
            if os.path.isfile(compiled_pdf_path):
                meta['flattened_pdf_path'] = compiled_pdf_path
                continue
            
            self.flatten_file(meta, force_flatten=True)
            compile_latex_file(meta['flattened_file_path'])
            if os.path.isfile(compiled_pdf_path):
                meta['flattened_pdf_path'] = compiled_pdf_path
                continue
            
            meta['flattened_pdf_path'] = ""
            # if success:
            #     meta['flattened_pdf_path'] = Path(os.path.splitext(meta['flattened_file_path'])[0] + '.pdf')
            # else:
            #     self.flatten_file(meta, force_flatten=True)
            #     success = compile_latex_file(meta['flattened_file_path'])
            #     if success:
            #         meta['flattened_pdf_path'] = Path(os.path.splitext(meta['flattened_file_path'])[0] + '.pdf')
            #     else:
            #         meta['flattened_pdf_path'] = ""

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
            data = self.load_file(meta['flattened_file_path'])
            yield data

    def get_latex_file(self, i):
        meta = self.metadata[i]
        data = self.load_file(meta['flattened_file_path'])
        return data