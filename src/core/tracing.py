"""
Tracing and observability for AI Finance Agent Team.
Vendor-agnostic tracing interface with optional integrations.
"""

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .config import get_settings
from .logging import get_logger


logger = get_logger(__name__)


@dataclass
class TraceSpan:
    """Represents a trace span."""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_name: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    parent_span_id: Optional[str] = None
    status: str = "ok"  # ok, error, timeout
    
    def finish(self, status: str = "ok"):
        """Finish the span and calculate duration."""
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status


class TraceContext:
    """Context manager for trace spans."""
    
    def __init__(self, tracer: "Tracer", operation_name: str, **tags):
        self.tracer = tracer
        self.span = TraceSpan(operation_name=operation_name, **tags)
        self.parent_span = None
    
    def __enter__(self):
        """Enter trace context."""
        self.parent_span = self.tracer.current_span
        if self.parent_span:
            self.span.parent_span_id = self.parent_span.span_id
            self.span.trace_id = self.parent_span.trace_id
        
        self.tracer.current_span = self.span
        self.tracer.spans.append(self.span)
        
        logger.debug(
            "Starting trace span",
            span_id=self.span.span_id,
            trace_id=self.span.trace_id,
            operation=self.span.operation_name,
            parent_span_id=self.span.parent_span_id
        )
        
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit trace context."""
        if exc_type:
            self.span.finish(status="error")
            self.span.logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "level": "error",
                "message": str(exc_val),
                "error_type": exc_type.__name__
            })
        else:
            self.span.finish(status="ok")
        
        self.tracer.current_span = self.parent_span
        
        logger.debug(
            "Finished trace span",
            span_id=self.span.span_id,
            duration_ms=self.span.duration_ms,
            status=self.span.status
        )


class Tracer:
    """Simple tracer implementation."""
    
    def __init__(self):
        self.settings = get_settings()
        self.spans: List[TraceSpan] = []
        self.current_span: Optional[TraceSpan] = None
        self.enabled = self.settings.enable_tracing
    
    def start_span(self, operation_name: str, **tags) -> TraceContext:
        """Start a new trace span."""
        if not self.enabled:
            return self._noop_context()
        
        return TraceContext(self, operation_name, **tags)
    
    def add_tag(self, key: str, value: Any):
        """Add tag to current span."""
        if self.current_span and self.enabled:
            self.current_span.tags[key] = value
    
    def add_log(self, message: str, level: str = "info", **fields):
        """Add log to current span."""
        if self.current_span and self.enabled:
            self.current_span.logs.append({
                "timestamp": datetime.utcnow().isoformat(),
                "level": level,
                "message": message,
                **fields
            })
    
    def get_trace_data(self) -> Dict[str, Any]:
        """Get all trace data for export."""
        if not self.enabled:
            return {}
        
        return {
            "trace_id": self.spans[0].trace_id if self.spans else None,
            "spans": [
                {
                    "span_id": span.span_id,
                    "trace_id": span.trace_id,
                    "operation_name": span.operation_name,
                    "start_time": span.start_time.isoformat(),
                    "end_time": span.end_time.isoformat() if span.end_time else None,
                    "duration_ms": span.duration_ms,
                    "tags": span.tags,
                    "logs": span.logs,
                    "parent_span_id": span.parent_span_id,
                    "status": span.status
                }
                for span in self.spans
            ]
        }
    
    def clear_traces(self):
        """Clear all traces."""
        self.spans.clear()
        self.current_span = None
    
    @contextmanager
    def _noop_context(self):
        """No-op context manager when tracing is disabled."""
        yield None


# Global tracer instance
tracer = Tracer()


def get_tracer() -> Tracer:
    """Get global tracer instance."""
    return tracer


@contextmanager
def trace_span(operation_name: str, **tags):
    """Context manager for creating trace spans."""
    with tracer.start_span(operation_name, **tags) as span:
        yield span


def trace_function(operation_name: Optional[str] = None):
    """Decorator to trace function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with trace_span(name, function=func.__name__, module=func.__module__) as span:
                try:
                    result = func(*args, **kwargs)
                    if span:
                        span.add_tag("success", True)
                    return result
                except Exception as e:
                    if span:
                        span.add_tag("success", False)
                        span.add_tag("error", str(e))
                        span.add_tag("error_type", type(e).__name__)
                    raise
        
        return wrapper
    return decorator


def trace_agent_activity(agent_type: str, activity: str):
    """Trace agent activities."""
    return trace_span(
        f"agent.{agent_type}.{activity}",
        agent_type=agent_type,
        activity=activity
    )


def trace_llm_call(provider: str, model: str):
    """Trace LLM API calls."""
    return trace_span(
        f"llm.{provider}.{model}",
        provider=provider,
        model=model
    )


def trace_data_processing(operation: str, record_count: int):
    """Trace data processing operations."""
    return trace_span(
        f"data.{operation}",
        operation=operation,
        record_count=record_count
    )


def trace_pipeline_step(step: str):
    """Trace pipeline execution steps."""
    return trace_span(
        f"pipeline.{step}",
        step=step
    )


class MetricsCollector:
    """Simple metrics collector."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}
    
    def record_timing(self, metric_name: str, duration_ms: float):
        """Record a timing metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(duration_ms)
    
    def increment_counter(self, metric_name: str, value: int = 1):
        """Increment a counter metric."""
        if metric_name not in self.counters:
            self.counters[metric_name] = 0
        self.counters[metric_name] += value
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p95": sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[0]
                }
        
        for name, value in self.counters.items():
            summary[f"{name}_total"] = value
        
        return summary
    
    def clear_metrics(self):
        """Clear all metrics."""
        self.metrics.clear()
        self.counters.clear()


# Global metrics collector
metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    return metrics


def record_timing(metric_name: str, duration_ms: float):
    """Record a timing metric."""
    metrics.record_timing(metric_name, duration_ms)


def increment_counter(metric_name: str, value: int = 1):
    """Increment a counter metric."""
    metrics.increment_counter(metric_name, value)


@contextmanager
def time_operation(metric_name: str):
    """Context manager to time an operation."""
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        record_timing(metric_name, duration_ms)


class HealthChecker:
    """Health check utilities."""
    
    def __init__(self):
        self.checks: Dict[str, callable] = {}
        self.start_time = datetime.utcnow()
    
    def add_check(self, name: str, check_func: callable):
        """Add a health check function."""
        self.checks[name] = check_func
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {
            "status": "healthy",
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "checks": {}
        }
        
        for name, check_func in self.checks.items():
            try:
                check_result = check_func()
                results["checks"][name] = {
                    "status": "healthy" if check_result else "unhealthy",
                    "result": check_result
                }
            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                results["status"] = "unhealthy"
        
        return results


# Global health checker
health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get global health checker."""
    return health_checker


def add_health_check(name: str, check_func: callable):
    """Add a health check."""
    health_checker.add_check(name, check_func)


def run_health_checks() -> Dict[str, Any]:
    """Run all health checks."""
    return health_checker.run_checks()