# config/config.py

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from .schema import SystemConfigSchema
load_dotenv()
class ConfigManager:
    """Configuration manager using Pydantic for validation"""
   
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None
        self._validated_config: Optional[SystemConfigSchema] = None
       
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
       
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
       
        # Replace environment variables
        config = self._replace_env_vars(config)
        self._config = config
        return config
   
    def _replace_env_vars(self, config: Any) -> Any:
        """Recursively replace environment variable placeholders"""
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        return config
   
    def get_validated_config(self) -> SystemConfigSchema:
        """Get validated configuration using Pydantic"""
        if self._validated_config is None:
            config_dict = self.load_config()
            self._validated_config = SystemConfigSchema(
                llm_config=config_dict['llm'],
                embedding_config=config_dict['embeddings'],
                rag_config=config_dict['rag'],
                agent_config=config_dict['agent'],
                evaluation_config=config_dict['evaluation']
            )
        return self._validated_config
   
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.get_validated_config().llm_config
   
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration"""
        return self.get_validated_config().embedding_config
   
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration"""
        return self.get_validated_config().rag_config
   
    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.get_validated_config().agent_config
   
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.get_validated_config().evaluation_config
# Global configuration instance
config_manager = ConfigManager()