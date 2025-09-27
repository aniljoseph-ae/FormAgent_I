# config/config.py


import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from config.schema import SystemConfigSchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        try:
            load_dotenv()
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            def replace_env_vars(data):
                if isinstance(data, dict):
                    return {k: replace_env_vars(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [replace_env_vars(item) for item in data]
                elif isinstance(data, str):
                    return os.path.expandvars(data)
                return data
            
            config = replace_env_vars(config)
            if 'rag_config' not in config:
                logger.warning("rag_config missing in config.yaml. Using defaults.")
                config['rag_config'] = {
                    'chunk_size': 512,
                    'chunk_overlap': 50,
                    'max_chunks_per_doc': 100,
                    'top_k': 5,
                    'similarity_threshold': 0.7,
                    'enable_hybrid_search': True
                }
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def get_validated_config(self) -> SystemConfigSchema:
        try:
            return SystemConfigSchema(**self.config)
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise

config_manager = ConfigManager()