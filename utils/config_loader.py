import yaml
import os
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation (e.g., 'data.batch_size')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.get('paths.data_dir'),
            self.get('paths.models_dir'),
            self.get('paths.logs_dir'),
            self.get('paths.results_dir')
        ]
        
        for dir_path in dirs:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)

# Global config instance
config = ConfigLoader()
