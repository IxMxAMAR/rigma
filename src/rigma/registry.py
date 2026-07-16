from __future__ import annotations

import io
import json
import os
import shutil
import zipfile
from importlib import resources
from pathlib import Path

import httpx

from .models import Combo, ModelSpec, UseCase

DEFAULT_REGISTRY_ZIP = (
    "https://codeload.github.com/IxMxAMAR/rigma-registry/zip/refs/heads/master")


def _fetch_bytes(url: str) -> bytes:
    r = httpx.get(url, follow_redirects=True, timeout=120)
    r.raise_for_status()
    return r.content


def _registry_cache_dir() -> Path:
    from .runtime import rigma_home
    return rigma_home() / "registry"


def _custom_dir() -> Path:
    """Hangar's custom-model specs (seam: tests isolate the user's real
    installs here without touching the registry cache)."""
    from .runtime import rigma_home
    return rigma_home() / "custom" / "models"


def update_registry(url: str = DEFAULT_REGISTRY_ZIP) -> Path:
    dest = _registry_cache_dir()
    tmp = dest.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(io.BytesIO(_fetch_bytes(url))) as z:
        z.extractall(tmp)
    inner = next(p for p in tmp.iterdir() if p.is_dir())
    if not (inner / "gpus.json").exists():
        shutil.rmtree(tmp)
        raise RuntimeError("downloaded registry is missing gpus.json")
    if dest.exists():
        shutil.rmtree(dest)
    inner.rename(dest)
    shutil.rmtree(tmp, ignore_errors=True)
    return dest


class Registry:
    def __init__(self, gpus: list[dict], models: dict[str, ModelSpec],
                 combos: dict[str, Combo],
                 use_cases: dict[str, UseCase] | None = None):
        self.gpus, self.models, self.combos = gpus, models, combos
        self.use_cases = use_cases or {}

    @classmethod
    def load(cls, path: Path | None = None) -> "Registry":
        if path is None and os.environ.get("RIGMA_REGISTRY_DIR"):
            path = Path(os.environ["RIGMA_REGISTRY_DIR"])
        if path is None:
            cache = _registry_cache_dir()
            if (cache / "gpus.json").exists():
                path = cache
        if path is None:
            path = Path(str(resources.files("rigma").joinpath("data/registry")))
        gpus = json.loads((path / "gpus.json").read_text(encoding="utf-8"))
        models = {}
        for f in sorted((path / "models").glob("*.json")):
            spec = ModelSpec.model_validate_json(f.read_text(encoding="utf-8"))
            models[spec.slug] = spec
        # user-installed models (Hangar); registry wins slug collisions
        custom = _custom_dir()
        if custom.is_dir():
            for f in sorted(custom.glob("*.json")):
                try:
                    spec = ModelSpec.model_validate_json(
                        f.read_text(encoding="utf-8"))
                except Exception:
                    continue   # a broken custom spec must not brick startup
                if spec.slug not in models:
                    models[spec.slug] = spec.model_copy(
                        update={"custom": True})
        combos = {}
        for f in sorted((path / "combos").rglob("*.json")):
            rel = f.relative_to(path / "combos").as_posix()
            combos[rel] = Combo.model_validate_json(f.read_text(encoding="utf-8"))
        use_cases = {}
        uc_dir = path / "use_cases"
        if uc_dir.is_dir():
            for f in sorted(uc_dir.glob("*.json")):
                uc = UseCase.model_validate_json(f.read_text(encoding="utf-8"))
                use_cases[uc.name] = uc
        return cls(gpus, models, combos, use_cases)

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
