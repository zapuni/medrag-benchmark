from pydantic import BaseModel


class Query(BaseModel):
    text: str


class RewrittenQuery(BaseModel):
    original: str
    rewritten: str
