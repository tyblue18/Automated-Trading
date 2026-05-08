import functools
import os

import yaml

_CONFIG_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))


@functools.lru_cache(maxsize=None)
def get_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
