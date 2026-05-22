import logging
import pickle
from pathlib import Path
from typing import Any

import configuration

CACHE_DIR = Path("_cache")

log = logging.getLogger(__name__)


def _ensure_dir(file_name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / file_name


def _is_disabled() -> bool:
    flag = configuration.get(configuration.USE_CACHE)

    return not flag if flag else False


def dump(obj: Any, file_name: str) -> None:
    if _is_disabled():
        return

    path = _ensure_dir(file_name)
    log.debug("Dump object at %s", path)

    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(file_name: str) -> Any:
    if _is_disabled():
        return

    path = _ensure_dir(file_name)
    log.debug("Load object from %s", path)

    with open(path, "rb") as f:
        return pickle.load(f)
