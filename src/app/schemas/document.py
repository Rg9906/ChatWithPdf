from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class DocumentBase(BaseModel):
    filename: str
    file_size_bytes: int
    page_count: int

class DocumentCreate(DocumentBase):
    file_path: str
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = "uploaded"

class DocumentResponse(DocumentBase):
    id: str
    upload_timestamp: datetime
    status: str

    class Config:
        from_attributes = True  # Allows Pydantic to read data from class attributes (ORM mode in v2)
