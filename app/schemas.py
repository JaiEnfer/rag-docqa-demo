from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
    top_k: int = 4

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
