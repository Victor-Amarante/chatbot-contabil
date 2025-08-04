from pydantic import BaseModel

class Question(BaseModel):
  question: str

class UploadResponse(BaseModel):
    message: str
    filename: str