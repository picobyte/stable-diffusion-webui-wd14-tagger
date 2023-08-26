"""Purpose: Pydantic models for the API."""
from typing import List, Dict

from modules.api import models as sd_models  # pylint: disable=E0401
from pydantic import BaseModel, Field


class TaggerInterrogateRequest(sd_models.InterrogateRequest):
    """Interrogate request model"""
    model: str = Field(
        title='Model',
        description='The interrogate model used.',
    )
    threshold: float = Field(
        title='Threshold',
        description='The threshold used for the interrogate model.',
    )


class TaggerQueueImageRequest(sd_models.InterrogateRequest):
    name: str = Field(
        title='Name',
        description='Only queue the image, under this name.',
    )


class TaggerBatchRequest(sd_models.InterrogateRequest):
    """Batch request model"""
    model: str = Field(
        title='Model',
        description='The interrogate model used.',
    )
    name: str = Field(
        title='Name',
        description='name of the last image',
    )
    threshold: float = Field(
        title='Threshold',
        description='The threshold used for the interrogate model.',
    )


class TaggerInterrogateResponse(BaseModel):
    """Interrogate response model"""
    caption: Dict[str, float] = Field(
        title='Caption',
        description='The generated captions for the image.'
    )


class TaggerInterrogatorsResponse(BaseModel):
    """Interrogators response model"""
    models: List[str] = Field(
        title='Models',
        description=''
    )


class TaggerQueueImageResponse(BaseModel):
    """Queue image response model"""
    pass


class TaggerBatchResponse(BaseModel):
    """Batch response model"""
    captions: Dict[str, Dict[str, float]] = Field(
        title='Captions',
        description='The generated captions for the images.'
    )
