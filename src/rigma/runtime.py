from __future__ import annotations

import hashlib
import json
import os
import platform
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


# AUDIT F26: docs/audit-2026-09-04-full.md
# Every byte of the engine is downloaded from url_base + an asset name and then
# Popen'd with the user's environment, so url_base is the one field in the
# manifest that decides who gets code execution. A fetched engines.json
# permanently outranks the copy in the wheel, so this is checked on the way in
# AND on the way out to disk AND at the download itself — a manifest written
# before this landed is re-judged on read rather than trusted forever.
# Scope, honestly: the manifest comes from the owner's own registry branch over
# HTTPS, and ComboFlags.env is already free-form registry data merged into the
# engine's environment — this is defence in depth on an already-trusted root,
# worth having because it is three lines and because the FIRST install of any
# backend has no trust-on-first-use digest to compare against.
ENGINE_URL_ALLOWLIST = ("https://github.com/ggml-org/llama.cpp/releases/download/",)


def _manifest_ok(cand: object) -> bool:
    """version + assets present, and nothing in it can make us fetch from
    anywhere but the allowlist. Asset names are concatenated onto url_base, so
    a name carrying a slash, a scheme or a `..` would walk straight back out of
    it (and out of the extraction root, since `root / asset` uses the same
    string) — plain filenames only, which is all the registry has ever
    published."""
    if not (isinstance(cand, dict) and cand.get("version")
            and isinstance(cand.get("assets"), dict)):
        return False
    base = cand.get("url_base")
    if (not isinstance(base, str) or ".." in base
            or not base.startswith(ENGINE_URL_ALLOWLIST)):
        return False
    names = list(cand["assets"].values())
    extra = cand.get("extra_assets") or {}
    if not isinstance(extra, dict):
        return False
    for v in extra.values():
        if not isinstance(v, list):
            return False
        names += v
    return all(isinstance(n, str) and n and n == Path(n).name
               and n not in (".", "..") and ":" not in n for n in names)


def _engines_manifest() -> dict:
    """The engine pin. Resolution order:
      1. ~/.rigma/engines.json — refreshed from the registry repo (LM
         Studio's pattern: engine updates decoupled from app releases;
         llama.cpp ships weekly, pip releases don't)
      2. the copy packaged in the wheel — always present, always works

    A downloaded manifest must parse, carry version+assets and download only
    from ENGINE_URL_ALLOWLIST or it is ignored; corruption can never brick
    engine bootstrap."""
    packaged = json.loads(resources.files("rigma")
                          .joinpath("data/engines.json")
                          .read_text(encoding="utf-8"))
    local = rigma_home() / "engines.json"
    try:
        cand = json.loads(local.read_text(encoding="utf-8"))
        if _manifest_ok(cand):
            return cand
    except (FileNotFoundError, OSError, ValueError):
        pass
    return packaged


ENGINES_MANIFEST_URL = ("https://raw.githubusercontent.com/IxMxAMAR/"
                        "rigma-registry/master/engines.json")


def update_engines_manifest(url: str = ENGINES_MANIFEST_URL) -> bool:
    """Fetch a newer engine pin into ~/.rigma/engines.json. Best-effort:
    False (never an exception) when offline or the payload is unusable."""
    try:
        r = httpx.get(url, follow_redirects=True, timeout=30)
        r.raise_for_status()
        cand = r.json()
        if not _manifest_ok(cand):
            return False
        p = rigma_home() / "engines.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(cand, indent=1), encoding="utf-8")
        tmp.replace(p)
        return True
    except Exception:
        return False


def _fetch(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)


def _reject_escaping_members(t: tarfile.TarFile, dest: Path) -> None:
    """The traversal half of tarfile's "data" filter, by hand.

    Only reached on interpreters that do not have the filter at all (see
    _extract). Refusing the archive is the right call over extracting it
    unfiltered: a tar member is free to name an absolute path or climb out of
    dest with `..`, and the engine tarball lands on the Linux bootstrap path
    where that would write wherever it asked."""
    root = dest.resolve()
    for m in t.getmembers():
        target = dest / m.name
        paths = [target]
        if m.issym():                    # symlink targets are member-relative
            paths.append(target.parent / m.linkname)
        elif m.islnk():                  # hard links are archive-root-relative
            paths.append(dest / m.linkname)
        for p in paths:
            if not p.resolve().is_relative_to(root):
                raise RuntimeError(
                    f"refusing to extract {m.name!r} from the engine archive: "
                    f"it points outside {dest}")


def _extract(archive: Path, dest: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            # AUDIT F29: docs/audit-2026-09-04-full.md
            # extractall(filter=) landed in 3.12 and was backported only to
            # 3.11.4, while the package declares 3.11 — on 3.11.0-3.11.3 this
            # was a TypeError, which is not a RuntimeError, so the CLI's
            # fallback ladder did not catch it and the Linux bootstrap died
            # outright. Feature-detect on tarfile.data_filter (added by the same
            # backport) and hand-check the members when it is missing. Both
            # paths raise RuntimeError for a hostile archive so the ladder sees
            # one thing, not tarfile.FilterError.
            if hasattr(tarfile, "data_filter"):
                try:
                    t.extractall(dest, filter="data")
                except tarfile.FilterError as e:
                    raise RuntimeError(f"refusing to extract {archive.name}: "
                                       f"{e}") from e
            else:
                _reject_escaping_members(t, dest)
                t.extractall(dest)


def ensure_engine(backend: str, os_name: str) -> Path:
    man = _engines_manifest()
    key = f"{os_name}/{backend}"
    if key not in man["assets"]:
        raise RuntimeError(f"no pinned engine build for {key}")
    root = rigma_home() / "engines" / man["version"] / backend
    exe = root / ("llama-server.exe" if os_name == "windows" else "llama-server")
    ready = root / ".ready"
    # only trust an existing engine once the FULL extraction completed — a
    # crash after llama-server.exe but before its DLLs leaves a broken engine
    # that would otherwise be reused and crash on launch
    if ready.exists():
        if exe.exists():
            return exe
        found = next(root.rglob(exe.name), None) if root.exists() else None
        if found:
            return found
    # AUDIT F26: docs/audit-2026-09-04-full.md — the last gate before bytes are
    # fetched and run. It sits BELOW the .ready short-circuit on purpose: an
    # engine already extracted here was already verified, and refusing to launch
    # it because a manifest looks wrong turns a download problem into "rigma no
    # longer starts". _engines_manifest() has already discarded a stored
    # manifest that fails this, so reaching it means the packaged pin is bad —
    # deleting ~/.rigma/engines.json would not help, and saying so would send
    # the owner down the wrong path.
    if not _manifest_ok(man):
        raise RuntimeError(
            "refusing to download an engine: this build's pinned url_base is "
            f"not on the allowlist ({ENGINE_URL_ALLOWLIST[0]}). This is a "
            "packaging fault, not something on your machine — please report it.")
    root.mkdir(parents=True, exist_ok=True)
    lock_path = rigma_home() / "engines" / "lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    # AUDIT F25: docs/audit-2026-09-04-full.md — carry the pre-fix entries over
    # instead of orphaning them. They are keyed on the bare os/backend but each
    # one already records the `asset` and `version` the new key is built from,
    # so every machine provisioned before this release keeps the digest it
    # recorded on first install. Without this, re-downloading an artifact that
    # was already verified would be recorded blind — the upgrade path would be
    # fixed at the cost of silently dropping trust-on-first-use everywhere else.
    for old_key, ent in list(lock.items()):
        if ":" in old_key or not isinstance(ent, dict):
            continue                      # already migrated, or not an entry
        if ent.get("version") and ent.get("asset"):
            lock.setdefault(f"{ent['version']}:{old_key}:{ent['asset']}", ent)
    assets = [man["assets"][key]] + man.get("extra_assets", {}).get(key, [])
    for asset in assets:
        archive = root / asset
        _fetch(man["url_base"] + asset, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        # AUDIT F25: docs/audit-2026-09-04-full.md
        # the lock key MUST carry the version. Every asset filename embeds the
        # build id, so a pin bump always changes the bytes; keyed on the bare
        # os/backend the trust-on-first-use check compared the new download
        # against the previous pin's digest, deleted it, and raised "checksum
        # mismatch" on every retry forever — the engine could never be upgraded
        # and the only cure was deleting lock.json by hand. Keyed by
        # version+key+asset the entry is per-artifact, so a bump records a new
        # digest and a re-download of the SAME artifact is still verified.
        lock_key = f"{man['version']}:{key}:{asset}"
        if lock_key in lock and lock[lock_key]["sha256"] != digest:
            archive.unlink()
            raise RuntimeError(
                f"checksum mismatch for {asset}: expected "
                f"{lock[lock_key]['sha256']}, got {digest}. The bytes served "
                f"for this exact build differ from the ones recorded in "
                f"{lock_path} when it was first installed.")
        lock[lock_key] = {"asset": asset, "sha256": digest,
                          "version": man["version"]}
        _extract(archive, root)
        archive.unlink()
    lock_path.write_text(json.dumps(lock, indent=2))
    result = exe if exe.exists() else next(root.rglob(exe.name), None)
    if not result:   # some archives nest under build/bin/
        raise RuntimeError(f"{exe.name} not found in downloaded engine assets")
    ready.write_text("ok", encoding="utf-8")   # extraction fully completed
    return result


def ensure_model(gguf: GgufFile) -> Path:
    # Classic downloader, unthrottled (owner decision 2026-07-14: no artificial
    # speed caps, ever — while gaming the right move is to not download, not to
    # throttle). xet stays off: it hung mid-download on Windows twice
    # (2026-07-06, 2026-07-14); classic runs at line rate on HF's CDN and
    # resumes deterministically. Explicit HF_* env vars always win (setdefault).
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    local_dir = rigma_home() / "models"
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / gguf.file
    # on-disk short-circuit — and custom (repo="local") files have no upstream
    # at all, so a miss is an error, never an HF request for a repo named
    # "local" (Hangar review 2026-07-17)
    if dest.exists():
        return dest
    if gguf.repo == "local":
        raise RuntimeError(f"{gguf.file} is a local-only file that is missing "
                           "from Rigma's models folder — reinstall it")
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
    # on Windows, suppress the jarring console window llama-server would pop
    popen_kw = {}
    if platform.system() == "Windows":
        popen_kw["creationflags"] = 0x08000000   # CREATE_NO_WINDOW
    # per-plan engine env (e.g. GGML_VK_DISABLE_COOPMAT on the Windows
    # proprietary Vulkan driver) — merged over the inherited environment
    if plan.flags.env:
        popen_kw["env"] = {**os.environ, **plan.flags.env}
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(argv, stdout=log_f, stderr=subprocess.STDOUT,
                                **popen_kw)
    sp = ServerProcess(proc, port, log_path)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        if sp.is_healthy():
            return sp
        time.sleep(0.5)
    code = proc.poll()
    sp.stop()
    tail = "".join(log_path.read_text(encoding="utf-8",
                                      errors="replace").splitlines(True)[-40:])
    # A hard crash right after "initializing" means the ENGINE could not build a
    # context for this model — the model/build are incompatible. Saying "failed
    # to become healthy" sends people hunting for VRAM and context settings that
    # have nothing to do with it (verified 2026-07-20: one gguf crashed
    # identically at ctx 512, ctx 16384, ngl 0, ngl 99, with and without mmproj,
    # and on the CPU build, while other models loaded fine on the same engine).
    if code is not None and code != 0 and "initializing" in tail:
        raise RuntimeError(
            f"llama-server crashed while initialising this model (exit "
            f"{code & 0xFFFFFFFF:#010x}).\nThe engine loaded the weights but "
            "could not create a context, which means THIS MODEL is not "
            "supported by the current engine build — it is not a VRAM or "
            "context-size problem. Try a different quant/repo of the model, or "
            f"a newer engine build.\n{tail}")
    raise RuntimeError(f"llama-server failed to become healthy on :{port}\n{tail}")
