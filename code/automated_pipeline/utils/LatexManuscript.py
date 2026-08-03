import os
import re
import subprocess
import json
import dill
from pathlib import Path
from collections import defaultdict

class LatexManuscript:
    def __init__(self, project_id, latex_file_path, metadata=None):
        self.project_id = project_id
        self.latex_path = latex_file_path
        self.metadata = metadata or {}

    def load(self):
        with open(self.latex_path, 'r') as fp:
            return fp.read()

    def compile_latex(self):
        cmd = [
            'latexmk',
            '-pdf',
            self.latex_path.stem,
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.latex_path.parent),
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
        tex = self.load()
        walker = LatexWalker(tex)
        nodes, _, _ = walker.get_latex_nodes()
        found = self.find_nodes(nodes, LatexEnvironmentNode, ('environmentname', 'table'))

        tables = []
        for node in found:
            pos = node.pos
            pos_end = pos + node.len
            org_text = tex[pos:pos_end]

            tables.append((pos, pos_end, org_text))

        self.tables = tables