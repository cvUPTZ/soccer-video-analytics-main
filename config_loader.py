import yaml
from typing import Dict, Any

CONFIG_PATH = "config.yaml"

def load_config() -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # Handle case where config file is missing, maybe load defaults or raise error
        print(f"Error: Configuration file {CONFIG_PATH} not found.")
        # For now, let's raise an error or exit, as config is crucial.
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file {CONFIG_PATH}: {e}")
        raise

# Load config once at module import if desired, or call explicitly
config = load_config()

# Example of how to access config values:
# from config_loader import config
# default_video_path = config['paths']['default_video']
