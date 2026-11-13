"""Contrôles de qualité des données : colonnes, types, manquants, doublons, valeurs impossibles."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class ValidationReport:
    name: str
    n_rows: int
    n_cols: int
    missing_columns: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    negative_counts: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)   # avertissements
    errors: list[str] = field(default_factory=list)   # bloquants

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


class DataValidationError(ValueError):
    pass


def validate_dataframe(
    df: pd.DataFrame,
    required: list[str],
    non_negative: list[str] | tuple[str, ...] = (),
    name: str = "dataset",
    max_null_frac: float = 0.2,
) -> ValidationReport:
    rep = ValidationReport(name=name, n_rows=len(df), n_cols=df.shape[1])
    if len(df) == 0:
        rep.errors.append("Le jeu de données est vide.")
        return rep
    rep.missing_columns = [c for c in required if c not in df.columns]
    if rep.missing_columns:
        rep.errors.append(f"Colonnes obligatoires absentes : {rep.missing_columns}")
    nulls = df.isna().sum()
    rep.null_counts = {c: int(n) for c, n in nulls.items() if n > 0}
    for c, n in rep.null_counts.items():
        frac = n / len(df)
        if c in required and frac > max_null_frac:
            rep.errors.append(f"« {c} » : {frac:.0%} de valeurs manquantes (> {max_null_frac:.0%}).")
        elif frac > 0:
            rep.issues.append(f"« {c} » : {n} valeurs manquantes ({frac:.1%}).")
    rep.duplicate_rows = int(df.duplicated().sum())
    if rep.duplicate_rows:
        rep.issues.append(f"{rep.duplicate_rows} lignes dupliquées.")
    for c in non_negative:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            n = int((df[c] < 0).sum())
            if n:
                rep.negative_counts[c] = n
                rep.issues.append(f"« {c} » : {n} valeurs négatives (impossibles).")
    return rep


def raise_if_invalid(report: ValidationReport) -> None:
    if not report.ok:
        raise DataValidationError("; ".join(report.errors))
