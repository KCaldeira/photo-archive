"""Load config.yaml and expose resolved paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    takeout_dir: Path
    work_dir: Path
    workers: int = 14
    thumb_size: int = 512
    analysis_size: int = 1024
    rank: dict = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.work_dir / "photos.sqlite"

    @property
    def thumbs_dir(self) -> Path:
        return self.work_dir / "thumbs"

    @property
    def sheets_dir(self) -> Path:
        return self.work_dir / "sheets"

    def thumb_path(self, sha1: str) -> Path:
        return self.thumbs_dir / sha1[:2] / f"{sha1}.jpg"


def load_config(path: str | os.PathLike | None = None, **overrides) -> Config:
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text()) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    for key in ("takeout_dir", "work_dir"):
        raw[key] = Path(os.path.expanduser(str(raw[key])))
    cfg = Config(**raw)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    return cfg
