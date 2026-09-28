from pydantic import BaseModel, Field


class ClinicalExtraction(BaseModel):
    conditions: list[str] = Field(default_factory=list)
    drugs: list[str] = Field(default_factory=list)
    dosages: list[str] = Field(default_factory=list)
    procedures: list[str] = Field(default_factory=list)