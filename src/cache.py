import logging
import pickle
from pathlib import Path
from typing import Any, TypeVar

import configuration

CACHE_DIR = Path("_cache")

log = logging.getLogger(__name__)

T = TypeVar("T")


def _cache_enabled() -> bool:
    flag = configuration.get(configuration.USE_CACHE)
    return flag is not False


def _cache_path(file_name: str) -> Path:
    return CACHE_DIR / file_name


def dump(obj: Any, file_name: str) -> bool:
    """
    Store object in cache.

    Returns:
        True if the object was cached successfully.
        False otherwise.
    """

    if not _cache_enabled():
        log.debug("Cache disabled, stopping dump")
        return False

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        path = _cache_path(file_name)

        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        log.debug("Cached object at %s", path)

        return True

    except (OSError, pickle.PickleError):
        log.exception("Failed to cache object: %s", file_name)
        return False


def load(file_name: str, default: T | None = None) -> T | None:
    """
    Load object from cache.

    Returns:
        Cached object if present and valid.
        Otherwise returns `default`.
    """

    if not _cache_enabled():
        log.debug("Cache disabled, stopping load")
        return default

    path = _cache_path(file_name)

    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)

        log.debug("Loaded cached object from %s", path)

        return obj

    except FileNotFoundError:
        log.debug("Cache miss: %s", path)

    except (OSError, pickle.PickleError):
        log.exception("Failed to load cache file: %s", path)

    return default
