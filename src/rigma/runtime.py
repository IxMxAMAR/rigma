from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
import time
import zipfile
from importlib import resources
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download

from .models import GgufFile, RunPlan


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


def ensure_model(gguf: GgufFile, polite: bool = False) -> Path:
    # Full speed by default (owner decision 2026-07-14: throttling punished
    # every download and didn't even fix gaming packet loss — the right move
    # while gaming is to not download at all, or pass --polite).
    # polite=True: classic single-stream downloader — resumes deterministically
    # and won't saturate the connection.
    if polite:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4")
    else:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "0")
        os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "16")
    local_dir = rigma_home() / "models"
    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=gguf.repo, filename=gguf.file,
                                local_dir=str(local_dir)))


class ServerProcess:
    def __init__(self, proc: subprocess.Popen, port: int, log_path: Path):
        self.proc, self.port, self.log_path = proc, port, log_path

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def is_healthy(self) -> bool:
        try:
            return httpx.get(f"{self.url}/health", timeout=3).status_code == 200
        except Exception:
            return False

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def launch_server(exe: Path, plan: RunPlan, model_path: Path, port: int = 11500,
                  timeout: float = 300.0,
                  extra_args: list[str] | None = None) -> ServerProcess:
    logs = rigma_home() / "logs"
    sessions = rigma_home() / "sessions"
    logs.mkdir(parents=True, exist_ok=True)
    sessions.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"server-{port}.log"
    argv = [str(exe), *(extra_args or []),
            *plan.server_args(str(model_path), port),
            "--slot-save-path", str(sessions)]
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(argv, stdout=log_f, stderr=subprocess.STDOUT)
    sp = ServerProcess(proc, port, log_path)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        if sp.is_healthy():
            return sp
        time.sleep(0.5)
    sp.stop()
    tail = "".join(log_path.read_text(encoding="utf-8",
                                      errors="replace").splitlines(True)[-40:])
    raise RuntimeError(f"llama-server failed to become healthy on :{port}\n{tail}")
