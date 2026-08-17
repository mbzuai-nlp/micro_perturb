from pydantic import BaseModel

class Edit(BaseModel):
    text_span_to_replace: str
    new_text_span: str

class Flaw(BaseModel):
    refuse: bool
    reasoning: str
    sentences: list[str]
    edits: list[Edit]