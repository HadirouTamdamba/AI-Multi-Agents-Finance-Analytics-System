"""
Configuration management for AI Finance Agent Team.
Uses pydantic-settings for type-safe configuration with environment variable support.
"""

import os
from pathlib import Path
from typing import Optional, List

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    
    # Application Configuration
    app_password: Optional[str] = Field(default=None, description="App password for authentication")
    app_env: str = Field(default="development", description="Application environment")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Streamlit Configuration
    streamlit_server_port: int = Field(default=8501, description="Streamlit server port")
    streamlit_server_address: str = Field(default="0.0.0.0", description="Streamlit server address")
    streamlit_server_headless: bool = Field(default=True, description="Run Streamlit in headless mode")
    streamlit_browser_gather_usage_stats: bool = Field(default=False, description="Gather usage statistics")
    
    # Data Configuration
    data_dir: str = Field(default="./data", description="Data directory path")
    output_dir: str = Field(default="./outputs", description="Output directory path")
    log_dir: str = Field(default="./logs", description="Log directory path")
    
    # Security Configuration
    secret_key: str = Field(default="dev-secret-key-change-in-production", description="Secret key for security")
    allowed_hosts: List[str] = Field(default=["localhost", "127.0.0.1", "0.0.0.0"], description="Allowed hosts")
    
    # Observability Configuration
    enable_tracing: bool = Field(default=False, description="Enable tracing")
    trace_sampling_rate: float = Field(default=0.1, description="Trace sampling rate")
    log_format: str = Field(default="json", description="Log format (json or text)")
    
    # Feature Flags
    enable_pdf_export: bool = Field(default=True, description="Enable PDF export functionality")
    enable_advanced_charts: bool = Field(default=True, description="Enable advanced charting")
    enable_real_time_updates: bool = Field(default=False, description="Enable real-time updates")
    
    # Performance Configuration
    max_file_size_mb: int = Field(default=10, description="Maximum file size in MB")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    max_concurrent_requests: int = Field(default=5, description="Maximum concurrent requests")
    
    # Development Configuration
    debug: bool = Field(default=False, description="Debug mode")
    mock_llm_responses: bool = Field(default=True, description="Use mock LLM responses")
    sample_data_only: bool = Field(default=True, description="Use sample data only")
    
    @validator("data_dir", "output_dir", "log_dir")
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @validator("trace_sampling_rate")
    def validate_sampling_rate(cls, v):
        """Validate trace sampling rate."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Trace sampling rate must be between 0.0 and 1.0")
        return v
    
    @validator("max_file_size_mb")
    def validate_file_size(cls, v):
        """Validate maximum file size."""
        if v <= 0:
            raise ValueError("Maximum file size must be positive")
        return v
    
    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is available."""
        return bool(self.anthropic_api_key and self.anthropic_api_key != "your_anthropic_api_key_here")
    
    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is available."""
        return bool(self.openai_api_key and self.openai_api_key != "your_openai_api_key_here")
    
    @property
    def has_any_api_key(self) -> bool:
        """Check if any API key is available."""
        return self.has_anthropic_key or self.has_openai_key
    
    @property
    def use_mock_llm(self) -> bool:
        """Determine if mock LLM should be used."""
        return not self.has_any_api_key or self.mock_llm_responses
    
    @property
    def data_path(self) -> Path:
        """Get data directory as Path object."""
        return Path(self.data_dir)
    
    @property
    def output_path(self) -> Path:
        """Get output directory as Path object."""
        return Path(self.output_dir)
    
    @property
    def log_path(self) -> Path:
        """Get log directory as Path object."""
        return Path(self.log_dir)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings