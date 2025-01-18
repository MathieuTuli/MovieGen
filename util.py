from torch.distributed import destroy_process_group
from pathlib import Path
from typing import Any
import yaml
import os


def convert_to_serializable(obj):
    """Convert complex objects to simple, YAML-serializable types"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, '__dict__'):  # Handle class instances
        return convert_to_serializable(obj.__dict__)
    else:
        return obj


def dump_dict_to_yaml(fout: Path | str, config: dict[str, Any]):
    fout = fout if isinstance(fout, Path) else Path(str)
    assert fout.parent.is_dir() and fout.parent.exists(), \
        f"Can't dump config to {fout}, parent doesn't exist"

    # Convert complex types to simple types
    serializable_config = convert_to_serializable(config)

    with fout.open("w") as f:
        yaml.dump(serializable_config, f,
                  default_flow_style=False, sort_keys=False)


def asspath(strarg):
    """helper to ensure arg path exists"""
    p = Path(strarg)
    if p.exists():
        return p
    else:
        raise NotADirectoryError(strarg)


def mkpath(strarg):
    """helper to mkdir arg path if it doesn't exist"""
    if not strarg:
        return ""
    p = Path(strarg)
    p.mkdir(exist_ok=True, parents=True)
    return p


def print0(*args, **kwargs):
    """modified print that only prints from the master process"""
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def cleanup():
    """Cleanup function to destroy the process group"""
    if int(os.environ.get('RANK', -1)) != -1:
        destroy_process_group()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print0("\nCtrl+C caught. Cleaning up...")
    cleanup()
    exit(0)
