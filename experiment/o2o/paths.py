"""Artifact path resolution.

The only module that knows where artifacts live. Notebooks call `set_root`
once (Drive on Colab, a local folder otherwise); every other module asks this
one for directories. Layout mirrors experiment/CLAUDE.md §5.
"""
from pathlib import Path

_ROOT: Path | None = None


def set_root(root: str | Path) -> Path:
    global _ROOT
    _ROOT = Path(root)
    _ROOT.mkdir(parents=True, exist_ok=True)
    return _ROOT


def root() -> Path:
    if _ROOT is None:
        raise RuntimeError("Artifact root not set; call o2o.paths.set_root(...) first.")
    return _ROOT


def _subdir(*parts: str) -> Path:
    path = root().joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_dir(env: str, regime: str) -> Path:
    return _subdir("datasets", env, regime)


def behaviour_dir(env: str, regime: str) -> Path:
    return _subdir("behaviour", env, regime)


def run_dir(env: str, condition: str, method: str, seed: int) -> Path:
    return _subdir("runs", env, condition, method, str(seed))


def figures_dir() -> Path:
    return _subdir("figures")
