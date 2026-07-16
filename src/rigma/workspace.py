"""Pack a local code folder into a single prompt-ready block.

With a 262K-context model on 32GB RAM, dumping a whole small repo into the
prompt beats RAG for "refactor this module with full project awareness". We
respect .gitignore-ish rules, skip binaries, and cap total size so the fit
advisor's ctx warning stays meaningful.
"""
from __future__ import annotations

import os
from pathlib import Path

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
              "build", ".next", "target", ".idea", ".vscode", "vendor",
              ".mypy_cache", ".pytest_cache", ".ruff_cache", "coverage",
              ".gradle", "bin", "obj", ".cache"}
_SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip",
             ".gz", ".tar", ".7z", ".exe", ".dll", ".so", ".dylib", ".bin",
             ".gguf", ".safetensors", ".pt", ".onnx", ".mp4", ".mp3", ".wav",
             ".woff", ".woff2", ".ttf", ".lock", ".min.js", ".map"}
_MAX_FILE = 200_000        # per-file byte cap (skip generated blobs)
_MAX_TOTAL = 2_000_000     # ~500K tokens worst case; fit advisor warns above


class WorkspaceError(RuntimeError):
    pass


def _skip_name(name: str) -> bool:
    n = name.lower()
    return any(n.endswith(e) for e in _SKIP_EXT)


def pack_folder(folder: str, max_total: int = _MAX_TOTAL) -> dict:
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise WorkspaceError(f"not a folder: {folder}")
    files, tree, total, truncated = [], [], 0, False
    # top-down walk with in-place pruning: never enumerate skipped subtrees
    # (a monorepo/drive-root shouldn't be fully materialized before the cap)
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for fn in sorted(filenames):
            if _skip_name(fn):
                continue
            p = Path(dirpath) / fn
            rel = p.relative_to(root).as_posix()
            try:
                if p.stat().st_size > _MAX_FILE:
                    continue
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue          # binary/unreadable/permission — skip silently
            if "\x00" in text:
                continue          # sneaky binary that decoded
            if total + len(text) > max_total:
                truncated = True
                break
            total += len(text)
            tree.append(rel)
            files.append(f'<file path="{rel}">\n{text}\n</file>')
        if truncated:
            break
    if not files:
        raise WorkspaceError("no readable text files found in that folder")
    header = (f"<project root=\"{root.name}\" files=\"{len(files)}\">\n"
              + "\n".join(tree) + "\n</project>\n\n")
    return {"root": root.name, "file_count": len(files),
            "chars": total, "truncated": truncated,
            "content": header + "\n\n".join(files)}
