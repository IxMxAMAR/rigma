from __future__ import annotations

import io
import json
import os
import shutil
import uuid
import zipfile
from importlib import resources
from pathlib import Path

import httpx

from .models import STANDARD_GB, Combo, ModelSpec, UseCase

DEFAULT_REGISTRY_ZIP = (
    "https://codeload.github.com/IxMxAMAR/rigma-registry/zip/refs/heads/master")


def _ram_tiers(ram_gb: int) -> list[int]:
    """The machine's own RAM tier first, then every lower standard tier.

    AUDIT F15-4: a combo is keyed to the RAM tier it was measured at. The
    reference box measures 31.4 GB (tier 32) while the verified RX 9070 XT combo
    is filed at ram-16, so an exact-tier-only lookup missed on the very machine
    it was measured on and fell through to the fit calculator. Never fall UP: a
    combo budgeted for more RAM than the machine has would overcommit exactly
    the resource it was sized against.
    """
    return [t for t in sorted(STANDARD_GB, reverse=True) if t <= ram_gb]


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
    # R3-STORE-10: this used a FIXED `<dest>.tmp`. Two concurrent `rigma update`
    # runs — or a CLI update racing the server's own — then shared one staging
    # directory: the second `rmtree` deleted the first's extraction mid-flight and
    # the rename below failed on a half-extracted tree. A unique staging name per
    # call, removed on every exit path.
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(_fetch_bytes(url))) as z:
        z.extractall(tmp)
    # AUDIT 06R3-5: a registry re-packaged flat — no `rigma-registry-master/`
    # wrapper — made this `next()` raise a bare StopIteration, which nothing
    # catches (cli.py catches ResolveError) and which escaped before the rmtree
    # below, leaving registry.tmp behind. Named the same way as the sibling
    # malformed case so `rigma update` reports rather than tracebacks.
    inner = next((p for p in tmp.iterdir() if p.is_dir()), None)
    if inner is None:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(
            "downloaded registry has no top-level directory — the archive "
            "layout is not the expected <repo>/gpus.json")
    if not (inner / "gpus.json").exists():
        shutil.rmtree(tmp)
        raise RuntimeError("downloaded registry is missing gpus.json")
    # swap-in via backup: on Windows rmtree(dest) often fails on a locked file,
    # leaving the registry half-deleted. Move the old aside first, put the new
    # in place, then delete the old (best-effort).
    backup = dest.with_suffix(".old")
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
    if dest.exists():
        dest.rename(backup)
    try:
        inner.rename(dest)
    except OSError:
        if backup.exists():        # restore the old registry on failure
            backup.rename(dest)
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(backup, ignore_errors=True)
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
                    # Heal HERE, not on the Models page: geometry feeds the
                    # launcher, the fit calculator and the bench as well as the
                    # UI, and a page that healed only its own view would show
                    # numbers the engine never used.
                    spec = spec.model_copy(update={"custom": True})
                    try:
                        from .hangar import heal_spec
                        spec = heal_spec(spec)
                    except Exception:
                        pass       # healing is a repair, never a load barrier
                    models[spec.slug] = spec
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
        # AUDIT F15-4: the exact tier first, then the nearest lower tiers — see
        # _ram_tiers. Within a tier the preference is unchanged: this card's own
        # combo before the class fallback, use-case before general.
        for tier in _ram_tiers(ram_gb):
            candidates = [
                f"{vendor}/{gpu_slug}/ram-{tier}/{use_case}.json",
                f"{vendor}/{gpu_slug}/ram-{tier}/general.json",
                f"_class/vram-{vram_gb}/ram-{tier}/{use_case}.json",
                f"_class/vram-{vram_gb}/ram-{tier}/general.json",
            ]
            for rel in candidates:
                if rel in self.combos:
                    return self.combos[rel], rel
        return None
