from pydantic import BaseModel, Field


class Iris(BaseModel):
    sepallength: float = Field(..., gt=0, description="Must be > 0")
    sepalwidth: float = Field(..., gt=0, description="Must be > 0")
    petallength: float = Field(..., gt=0, description="Must be > 0")
    petalwidth: float = Field(..., gt=0, description="Must be > 0")


FEATURE_NAMES = list(Iris.__fields__.keys())