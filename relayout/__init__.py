"""Relayout adaptation for LIBERO.

Importing this package pins LIBERO to a repo-local config before `libero.libero` is
imported. Without a config file LIBERO asks questions on stdin (hangs in batch jobs),
and an existing ~/.libero may point to a different LIBERO checkout.
"""
import importlib.util
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _pin_libero_config():
    cfg_dir = Path(os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO / ".libero")))
    cfg = cfg_dir / "config.yaml"
    if cfg.exists():
        return
    spec = importlib.util.find_spec("libero")
    if spec is None:
        return
    root = Path(list(spec.submodule_search_locations)[0]) / "libero"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg.write_text("".join(f"{k}: {v}\n" for k, v in {
        "benchmark_root": root,
        "bddl_files": root / "bddl_files",
        "init_states": root / "init_files",
        "datasets": root.parent / "datasets",
        "assets": root / "assets",
    }.items()))


_pin_libero_config()
