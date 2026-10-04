from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from helm import UserProfile


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=2000)
    profile: UserProfile = UserProfile.END_USER
    model: Literal["demo", "xlmr", "qwen", "camembert"] = "demo"

    @field_validator("text")
    @classmethod
    def meaningful_text(cls, value):
        value = value.strip()
        if not any(char.isalnum() for char in value):
            raise ValueError("Saisissez un texte contenant des mots.")
        return value


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str
    rating: int = Field(ge=1, le=5, strict=True)
