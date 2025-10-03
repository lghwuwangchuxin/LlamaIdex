# config_manager.py
import configparser
import os
from typing import Optional


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: str = "/Users/liuguanghu/PythonPorject/LlamaIdex/config/config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

    def get(self, section: str, key: str, fallback: Optional[str] = None) -> Optional[str]:
        """获取配置值"""
        try:
            return self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

    def get_int(self, section: str, key: str, fallback: int = 0) -> int:
        """获取整数配置值"""
        try:
            return self.config.getint(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback

    def get_float(self, section: str, key: str, fallback: float = 0.0) -> float:
        """获取浮点数配置值"""
        try:
            return self.config.getfloat(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
