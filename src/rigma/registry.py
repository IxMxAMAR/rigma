from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path

from .models import Combo, ModelSpec


class Registry:
    def __init__(self, gpus: list[dict], models: dict[str, ModelSpec],
                 combos: dict[str, Combo]):
        self.gpus, self.models, self.combos = gpus, models, combos

    @classmethod
    def load(cls, path: Path | None = None) -> "Registry":
        if path is None and os.environ.get("RIGMA_REGISTRY_DIR"):
            path = Path(os.environ["RIGMA_REGISTRY_DIR"])
        if path is None:
            path = Path(str(resources.files("rigma").joinpath("data/registry")))
        gpus = json.loads((path / "gpus.json").read_text(encoding="utf-8"))
        models = {}
        for f in sorted((path / "models").glob("*.json")):
            spec = ModelSpec.model_validate_json(f.read_text(encoding="utf-8"))
            models[spec.slug] = spec
        combos = {}
        for f in sorted((path / "combos").rglob("*.json")):
            rel = f.relative_to(path / "combos").as_posix()
            combos[rel] = Combo.model_validate_json(f.read_text(encoding="utf-8"))
        return cls(gpus, models, combos)

    def find_combo(self, vendor: str, gpu_slug: str, vram_gb: int, ram_gb: int,
                   use_case: str) -> tuple[Combo, str] | None:
        candidates = [
            f"{vendor}/{gpu_slug}/ram-{ram_gb}/{use_case}.json",
            f"{vendor}/{gpu_slug}/ram-{ram_gb}/general.json",
            f"_class/vram-{vram_gb}/ram-{ram_gb}/{use_case}.json",
            f"_class/vram-{vram_gb}/ram-{ram_gb}/general.json",
        ]
        for rel in candidates:
            if rel in self.combos:
                return self.combos[rel], rel
        return None
