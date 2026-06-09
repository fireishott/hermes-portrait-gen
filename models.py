"""Data classes for portrait-gen."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json
import time


@dataclass
class FaceIdentity:
    """A known person with their face embedding."""
    name: str
    embedding: list[float]
    ref_count: int = 0
    best_ref: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FaceIdentity":
        return cls(**d)


@dataclass
class RefPhoto:
    """Metadata about a reference photo."""
    path: str
    person: str
    quality: float = 0.0
    added_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FaceMatch:
    """Result of matching a face against known identities."""
    person: Optional[str]
    confidence: float
    is_new: bool
    bbox: list[float] = field(default_factory=list)


@dataclass
class PortraitRequest:
    """Parameters for a portrait generation."""
    person: str
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg: float = 3.5
    seed: int = -1
    pulid_weight: float = 0.8


@dataclass
class PortraitResult:
    """Result of a portrait generation."""
    success: bool
    output_path: Optional[str] = None
    person: Optional[str] = None
    prompt: Optional[str] = None
    seed: Optional[int] = None
    error: Optional[str] = None
    generation_time_s: Optional[float] = None


class EmbeddingsDB:
    """Simple JSON-backed store for face embeddings."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, FaceIdentity] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                self._data = {
                    k: FaceIdentity.from_dict(v) for k, v in raw.items()
                }
            except (json.JSONDecodeError, TypeError):
                self._data = {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in self._data.items()}
        self.path.write_text(json.dumps(data, indent=2))

    def get(self, name: str) -> Optional[FaceIdentity]:
        return self._data.get(name)

    def put(self, identity: FaceIdentity):
        self._data[identity.name] = identity
        self._save()

    def all(self) -> dict[str, FaceIdentity]:
        return dict(self._data)

    def names(self) -> list[str]:
        return list(self._data.keys())

    def delete(self, name: str):
        self._data.pop(name, None)
        self._save()
