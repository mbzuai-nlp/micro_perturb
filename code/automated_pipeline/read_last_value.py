import dill
from pathlib import Path
from pydantic import BaseModel

class Edit(BaseModel):
    text_span_to_replace: str
    new_text_span: str

class Flaw(BaseModel):
    refuse: bool
    reasoning: str
    sentences: list[str]
    edits: list[Edit]

def load_from_cache(filename):
    print(f"Loading cached responses from {filename}")
    with open(filename, 'rb') as f:
        while True:
            try:
                # Load the next object in the stream
                obj = dill.load(f)
                yield obj
            except EOFError:
                # End of file reached
                break

file_path = Path(f'output/responses_edit_latex_gpt_54_cache.dill')
data = list(load_from_cache(file_path))
counts = {}
paper_counts = {}
for d in data:
    meta = d['meta_id']
    counts[meta] = counts.get(meta, 0) + 1

    paper_id = meta.split('___')[0]
    paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1

repeated = False
for k,v in counts.items():
    if v > 1:
        repeated = True
        print(f'Repeated manuscript + error: {k}')
        break

for k,v in paper_counts.items():
    if v > 5:
        repeated = True
        print(f'Repeated manuscript {v} times: {k}')
        break

if not repeated:
    print("No repeats of manuscripts + errors")

print(len(data))
print(data[-1])