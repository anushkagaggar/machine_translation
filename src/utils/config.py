import yaml

def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_number(x):
    """Convert numeric-looking strings to numbers."""
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return x
    return x

def sanitize_optimizer_config(opt_cfg):
    """Clean lr, eps, betas if needed."""
    clean = {}
    for k, v in opt_cfg.items():
        if isinstance(v, list):
            clean[k] = [to_number(i) for i in v]
        else:
            clean[k] = to_number(v)
    return clean
