"""Schémas Pydantic de validation des entrées utilisateur."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ConversionAssumptions(BaseModel):
    """Hypothèses SIMULÉES de conversion lecture -> impressions (pas des paramètres TF1+)."""

    ad_opportunities_per_view: float = Field(2.0, gt=0, le=50)
    eligibility_rate: float = Field(0.85, ge=0, le=1)
