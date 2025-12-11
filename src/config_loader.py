"""
Configuration loader for 2-AMP project.
Loads settings from config.yaml and provides easy access.
"""

import yaml
import os
from typing import Dict, Any, Optional

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

_config: Optional[Dict[str, Any]] = None

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml file."""
    global _config
    if _config is None:
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
        
        with open(CONFIG_PATH, 'r') as f:
            _config = yaml.safe_load(f)
    
    return _config

def get_config(section: Optional[str] = None) -> Any:
    """
    Get configuration value(s).
    
    Args:
        section: Section name (e.g., 'data', 'sentiment'). If None, returns entire config.
    
    Returns:
        Configuration dictionary or section value
    """
    config = load_config()
    if section is None:
        return config
    return config.get(section, {})

def reload_config():
    """Force reload of configuration file."""
    global _config
    _config = None
    return load_config()

