from __future__ import annotations

import hashlib
import json
import os
import tarfile
import zipfile
from importlib import resources
from pathlib import Path

import httpx


def rigma_home() -> Path:
    return Path(os.environ.get("RIGMA_HOME", str(Path.home() / ".rigma")))


def _engines_manifest() -> dict:
    return json.loads(resources.files("rigma").joinpath("data/engines.json")
                      .read_text(encoding="utf-8"))


def _fetch(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)


def _extract(archive: Path, dest: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")


def ensure_engine(backend: str, os_name: str) -> Path:
    man = _engines_manifest()
    key = f"{os_name}/{backend}"
    if key not in man["assets"]:
        raise RuntimeError(f"no pinned engine build for {key}")
    root = rigma_home() / "engines" / man["version"] / backend
    exe = root / ("llama-server.exe" if os_name == "windows" else "llama-server")
    if exe.exists():
        return exe
    found = next(root.rglob(exe.name), None) if root.exists() else None
    if found:
        return found
    root.mkdir(parents=True, exist_ok=True)
    lock_path = rigma_home() / "engines" / "lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    assets = [man["assets"][key]] + man.get("extra_assets", {}).get(key, [])
    for asset in assets:
        archive = root / asset
        _fetch(man["url_base"] + asset, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        lock_key = f"{key}:{asset}" if asset != man["assets"][key] else key
        if lock_key in lock and lock[lock_key]["sha256"] != digest:
            archive.unlink()
            raise RuntimeError(f"checksum mismatch for {asset}: expected "
                               f"{lock[lock_key]['sha256']}, got {digest}")
        lock[lock_key] = {"asset": asset, "sha256": digest,
                          "version": man["version"]}
        _extract(archive, root)
        archive.unlink()
    lock_path.write_text(json.dumps(lock, indent=2))
    if exe.exists():
        return exe
    found = next(root.rglob(exe.name), None)  # some archives nest under build/bin/
    if not found:
        raise RuntimeError(f"{exe.name} not found in downloaded engine assets")
    return found
