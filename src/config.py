import functools
import logging
import os

import yaml

_CONFIG_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))


@functools.lru_cache(maxsize=None)
def get_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_cli_logging() -> None:
    level_str = get_config().get("logging", {}).get("level", "INFO")
    level = getattr(logging, level_str, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
