from typing import Optional, List, Any
from pydantic import BaseModel, Field

class Pagination(BaseModel):
    total: int
    page: int
    size: int
    pages: int

class PaginatedResponse(BaseModel):
    data: List[Any]
    pagination: Pagination

class ErrorResponse(BaseModel):
    error: str
    message: str
