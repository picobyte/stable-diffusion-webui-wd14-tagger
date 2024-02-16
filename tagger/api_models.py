"""Purpose: Pydantic models for the API."""
from typing import List, Dict

from modules.api import models as sd_models  # pylint: disable=E0401
from pydantic import BaseModel, Field


class TaggerInterrogateRequest(sd_models.ReqInterrogate):
    """Interrogate request model"""
    model: str = Field(
        title='Model',
        description='The interrogate model used.',
    )
    threshold: float = Field(
        title='Threshold',
        description='The threshold used for the interrogate model.',
        default=0.0,
    )
    queue: str = Field(
        title='Queue',
        description='name of queue; leave empty for single response',
        default='',
    )
    name_in_queue: str = Field(
        title='Name',
        description='name to queue image as or use <sha256>. leave empty to '
                    'retrieve the final response',
        default='',
    )


class TaggerInterrogateResponse(BaseModel):
    """Interrogate response model"""
    caption: Dict[str, Dict[str, float]] = Field(
        title='Caption',
        description='The generated captions for the image.'
    )


class TaggerInterrogatorsResponse(BaseModel):
    """Interrogators response model"""
    models: List[str] = Field(
        title='Models',
        description=''
    )
