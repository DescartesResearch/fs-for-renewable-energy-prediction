"""
Configuration utilities for Hydra configs.
"""

from typing import Any, Dict


def flatten_config(cfg: Any, parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """
    Flatten a nested config structure to a flat dictionary.
    
    Example:
        cfg = {
            "name": "exp1",
            "model": {"name": "mlp"},
            "features": {"n_features": 5}
        }
        
    Returns:
        {
            "name": "exp1",
            "model_name": "mlp",
            "features_n_features": 5
        }
    """
    items = []
    
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
    else:
        # Handle DictConfig from OmegaConf
        from omegaconf import DictConfig, OmegaConf
        
        if isinstance(cfg, DictConfig):
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            items.extend(flatten_config(cfg_dict, parent_key, sep=sep).items())
    
    return dict(items)


__all__ = ["flatten_config"]
