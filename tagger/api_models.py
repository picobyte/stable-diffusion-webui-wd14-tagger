"""Purpose: Pydantic models for the API."""
from typing import List, Dict

from modules.api import models as sd_models  # pylint: disable=E0401
from pydantic import BaseModel, Field


class TaggerInterrogateRequest(sd_models.InterrogateRequest):
    """Interrogate request model"""
    model: str = Field(
        title='Model',
        description='The interrogate model(s) used. Comma separated.',
    )

    threshold: float = Field(
        default=0.35,
        title='Threshold',
        description='',
        ge=0,
        le=1
    )
    tag_frac_threshold: float = Field(
        default=0.05,
        title='Amongst interrogations tag fraction threshold',
        description='',
        ge=0,
        le=1
    )
    count_threshold: float = Field(
        default=100,
        title='Count threshold',
        description='',
        ge=1,
        le=1000000
    )
    auto_unload: bool = Field(
        default=True,
        title='Auto unload',
        description='Unload each model after interrogation.'
    )


class TaggerInterrogateResponse(BaseModel):
    """Interrogate response model"""
    caption: Dict[str, float] = Field(
        title='Caption',
        description='The generated caption for the image.'
    )


class InterrogatorsResponse(BaseModel):
    """Interrogators response model"""
    models: List[str] = Field(
        title='Models',
        description=''
    )
