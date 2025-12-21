"""
Structured logging configuration for AI Finance Agent Team.
Supports JSON and text formats with context-aware logging.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .config import get_settings


def setup_logging() -> None:
    """Setup structured logging configuration."""
    settings = get_settings()
    
    # Create logs directory
    settings.log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        add_context,
    ]
    
    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(message)s",
        stream=sys.stdout,
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=True,
                markup=True,
            )
        ],
    )
    
    # Set up file logging
    file_handler = logging.FileHandler(
        settings.log_path / "app.log",
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, settings.log_level))
    
    if settings.log_format == "json":
        file_handler.setFormatter(logging.Formatter('%(message)s'))
    else:
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    
    # Add file handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


def add_context(logger, method_name, event_dict):
    """Add common context to log entries."""
    event_dict["service"] = "ai-finance-agent-team"
    event_dict["version"] = "1.0.0"
    return event_dict


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


class LogContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, **context):
        self.context = context
        self.logger = get_logger("context")
    
    def __enter__(self):
        """Enter context and bind context to logger."""
        self.logger = self.logger.bind(**self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type:
            self.logger.error(
                "Exception in context",
                exc_type=exc_type.__name__,
                exc_value=str(exc_val),
                exc_traceback=exc_tb
            )


def log_function_call(func_name: str, **kwargs):
    """Decorator to log function calls with parameters."""
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            logger = get_logger(func.__module__)
            logger.info(
                f"Calling {func_name}",
                function=func_name,
                args_count=len(args),
                kwargs=func_kwargs,
                **kwargs
            )
            
            start_time = datetime.utcnow()
            try:
                result = func(*args, **func_kwargs)
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                logger.info(
                    f"Completed {func_name}",
                    function=func_name,
                    execution_time=execution_time,
                    success=True,
                    **kwargs
                )
                return result
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.error(
                    f"Failed {func_name}",
                    function=func_name,
                    execution_time=execution_time,
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__,
                    **kwargs
                )
                raise
        
        return wrapper
    return decorator


def log_agent_activity(agent_type: str, activity: str, **context):
    """Log agent-specific activities."""
    logger = get_logger("agent")
    logger.info(
        f"Agent {agent_type}: {activity}",
        agent_type=agent_type,
        activity=activity,
        **context
    )


def log_pipeline_step(step: str, status: str, **context):
    """Log pipeline execution steps."""
    logger = get_logger("pipeline")
    logger.info(
        f"Pipeline step: {step} - {status}",
        step=step,
        status=status,
        **context
    )


def log_data_processing(operation: str, record_count: int, **context):
    """Log data processing operations."""
    logger = get_logger("data")
    logger.info(
        f"Data processing: {operation}",
        operation=operation,
        record_count=record_count,
        **context
    )


def log_llm_interaction(provider: str, model: str, tokens: int, **context):
    """Log LLM interactions."""
    logger = get_logger("llm")
    logger.info(
        f"LLM interaction: {provider}/{model}",
        provider=provider,
        model=model,
        tokens=tokens,
        **context
    )


def log_security_event(event_type: str, severity: str, **context):
    """Log security-related events."""
    logger = get_logger("security")
    logger.warning(
        f"Security event: {event_type}",
        event_type=event_type,
        severity=severity,
        **context
    )


def log_performance_metric(metric_name: str, value: float, unit: str, **context):
    """Log performance metrics."""
    logger = get_logger("performance")
    logger.info(
        f"Performance metric: {metric_name}",
        metric_name=metric_name,
        value=value,
        unit=unit,
        **context
    )


class StructuredLogger:
    """Enhanced structured logger with common patterns."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def info(self, message: str, **context):
        """Log info message with context."""
        self.logger.info(message, **context)
    
    def warning(self, message: str, **context):
        """Log warning message with context."""
        self.logger.warning(message, **context)
    
    def error(self, message: str, **context):
        """Log error message with context."""
        self.logger.error(message, **context)
    
    def debug(self, message: str, **context):
        """Log debug message with context."""
        self.logger.debug(message, **context)
    
    def critical(self, message: str, **context):
        """Log critical message with context."""
        self.logger.critical(message, **context)
    
    def bind(self, **context):
        """Bind context to logger."""
        return self.logger.bind(**context)


# Initialize logging when module is imported
setup_logging()