"""Téléchargement des datasets publics (avec repli documenté si la source est inaccessible)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

from ..common.config import Paths, default_paths, load_config
from ..common.logging import get_logger

log = get_logger()

MANUAL_INSTRUCTIONS = {
    "internos": (
        "Téléchargement manuel : ouvrir https://data.mendeley.com/datasets/69g55s83mn/1 "
        "(licence CC BY 4.0), télécharger le fichier de données et le placer dans "
        "data/raw/ sous le nom « {name} » (CSV ; un XLSX est aussi accepté en le renommant "
        "internos.xlsx)."
    ),
    "criteo": (
        "Téléchargement manuel : https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/ "
        "ou https://huggingface.co/datasets/criteo/criteo-attribution-dataset (CC BY-NC-SA 4.0). "
        "Placer le fichier TSV (.tsv.gz) dans data/raw/ sous le nom « {name} »."
    ),
}

# URL tentée automatiquement (peut changer côté éditeur ; en cas d'échec -> instructions manuelles)
CRITEO_URL = (
    "https://huggingface.co/datasets/criteo/criteo-attribution-dataset/resolve/main/"
    "criteo_attribution_dataset.tsv.gz"
)


def download_file(url: str, dest: Path, timeout: int = 60) -> tuple[bool, str]:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - toute erreur réseau => repli manuel
        return False, f"{type(exc).__name__}: {exc}"


def download_all(paths: Paths | None = None) -> dict[str, dict]:
    """Retourne un statut par dataset ; ne fabrique jamais de donnée en cas d'échec."""
    paths = (paths or default_paths()).ensure()
    cfg = load_config("app", paths)["datasets"]
    status: dict[str, dict] = {}

    internos = paths.data_raw / cfg["internos_file"]
    if internos.exists():
        status["internos"] = {"status": "present", "path": str(internos)}
    else:
        status["internos"] = {
            "status": "manuel",
            "message": MANUAL_INSTRUCTIONS["internos"].format(name=cfg["internos_file"]),
        }

    criteo = paths.data_raw / cfg["criteo_file"]
    if criteo.exists():
        status["criteo"] = {"status": "present", "path": str(criteo)}
    else:
        ok, msg = download_file(CRITEO_URL, criteo)
        status["criteo"] = (
            {"status": "telecharge", "path": str(criteo)} if ok else
            {"status": "manuel", "error": msg,
             "message": MANUAL_INSTRUCTIONS["criteo"].format(name=cfg["criteo_file"])}
        )
    for name, st in status.items():
        log.info("%s -> %s %s", name, st["status"], st.get("message", ""))
    return status
