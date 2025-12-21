"""
Pydantic schemas for AI Finance Agent Team.
Defines data models for type safety and validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator


class AgentType(str, Enum):
    """Agent types in the system."""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    RISK_MODELER = "risk_modeler"
    WRITER = "writer"
    ORCHESTRATOR = "orchestrator"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataSource(str, Enum):
    """Data source types."""
    CSV = "csv"
    TEXT = "text"
    NEWS = "news"
    API = "api"
    UPLOAD = "upload"


class ConfidenceLevel(str, Enum):
    """Confidence levels for analysis results."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# Base schemas
class BaseSchema(BaseModel):
    """Base schema with common fields."""
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


# Data schemas
class DataFile(BaseSchema, TimestampMixin):
    """Data file information."""
    id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    file_type: DataSource = Field(..., description="Type of data source")
    file_size: int = Field(..., description="File size in bytes")
    file_path: str = Field(..., description="Path to the file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="File metadata")
    
    @validator("file_size")
    def validate_file_size(cls, v):
        """Validate file size is positive."""
        if v <= 0:
            raise ValueError("File size must be positive")
        return v


class FinancialData(BaseSchema):
    """Financial data structure."""
    symbol: Optional[str] = Field(None, description="Financial symbol")
    date: datetime = Field(..., description="Date of the data point")
    open_price: Optional[float] = Field(None, description="Opening price")
    high_price: Optional[float] = Field(None, description="High price")
    low_price: Optional[float] = Field(None, description="Low price")
    close_price: Optional[float] = Field(None, description="Closing price")
    volume: Optional[int] = Field(None, description="Trading volume")
    additional_metrics: Dict[str, float] = Field(default_factory=dict, description="Additional metrics")


class NewsArticle(BaseSchema):
    """News article structure."""
    id: str = Field(..., description="Article identifier")
    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content")
    source: str = Field(..., description="News source")
    published_at: datetime = Field(..., description="Publication date")
    url: Optional[str] = Field(None, description="Article URL")
    sentiment: Optional[str] = Field(None, description="Sentiment analysis result")
    relevance_score: Optional[float] = Field(None, description="Relevance score")


# Agent schemas
class AgentTask(BaseSchema, TimestampMixin):
    """Agent task definition."""
    id: str = Field(..., description="Task identifier")
    agent_type: AgentType = Field(..., description="Type of agent to execute task")
    task_type: str = Field(..., description="Type of task")
    input_data: Dict[str, Any] = Field(..., description="Input data for the task")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")


class AgentResponse(BaseSchema):
    """Agent response structure."""
    agent_type: AgentType = Field(..., description="Type of agent")
    task_id: str = Field(..., description="Task identifier")
    content: str = Field(..., description="Response content")
    confidence: ConfidenceLevel = Field(..., description="Confidence level")
    citations: List[str] = Field(default_factory=list, description="Source citations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    execution_time: float = Field(..., description="Execution time in seconds")


# Analysis schemas
class FinancialMetric(BaseSchema):
    """Financial metric calculation."""
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    unit: str = Field(..., description="Metric unit")
    calculation_method: str = Field(..., description="How the metric was calculated")
    confidence: ConfidenceLevel = Field(..., description="Confidence in the calculation")


class RiskScenario(BaseSchema):
    """Risk scenario analysis."""
    scenario_name: str = Field(..., description="Name of the scenario")
    probability: float = Field(..., description="Probability of scenario (0-1)")
    impact: float = Field(..., description="Impact score")
    description: str = Field(..., description="Scenario description")
    mitigation_strategies: List[str] = Field(default_factory=list, description="Mitigation strategies")
    
    @validator("probability")
    def validate_probability(cls, v):
        """Validate probability is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Probability must be between 0 and 1")
        return v


class ChartData(BaseSchema):
    """Chart data structure."""
    chart_type: str = Field(..., description="Type of chart")
    title: str = Field(..., description="Chart title")
    x_axis: List[Union[str, float, datetime]] = Field(..., description="X-axis data")
    y_axis: List[float] = Field(..., description="Y-axis data")
    series_name: Optional[str] = Field(None, description="Series name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chart metadata")


# Pipeline schemas
class PipelineConfig(BaseSchema):
    """Pipeline configuration."""
    name: str = Field(..., description="Pipeline name")
    description: str = Field(..., description="Pipeline description")
    agents: List[AgentType] = Field(..., description="List of agents to use")
    data_sources: List[DataSource] = Field(..., description="Data sources to process")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Pipeline parameters")


class PipelineResult(BaseSchema, TimestampMixin):
    """Pipeline execution result."""
    id: str = Field(..., description="Pipeline execution ID")
    config: PipelineConfig = Field(..., description="Pipeline configuration")
    status: TaskStatus = Field(..., description="Overall pipeline status")
    agent_responses: List[AgentResponse] = Field(default_factory=list, description="Agent responses")
    final_report: Optional[str] = Field(None, description="Final report content")
    charts: List[ChartData] = Field(default_factory=list, description="Generated charts")
    metrics: List[FinancialMetric] = Field(default_factory=list, description="Calculated metrics")
    risk_scenarios: List[RiskScenario] = Field(default_factory=list, description="Risk scenarios")
    execution_time: float = Field(..., description="Total execution time")
    error_message: Optional[str] = Field(None, description="Error message if failed")


# Report schemas
class ExecutiveSummary(BaseSchema):
    """Executive summary structure."""
    overview: str = Field(..., description="High-level overview")
    key_findings: List[str] = Field(..., description="Key findings")
    recommendations: List[str] = Field(..., description="Recommendations")
    risk_assessment: str = Field(..., description="Risk assessment summary")
    confidence_level: ConfidenceLevel = Field(..., description="Overall confidence level")


class ReportSection(BaseSchema):
    """Report section structure."""
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    charts: List[ChartData] = Field(default_factory=list, description="Section charts")
    citations: List[str] = Field(default_factory=list, description="Section citations")


class FinancialReport(BaseSchema, TimestampMixin):
    """Complete financial report."""
    id: str = Field(..., description="Report identifier")
    title: str = Field(..., description="Report title")
    executive_summary: ExecutiveSummary = Field(..., description="Executive summary")
    sections: List[ReportSection] = Field(..., description="Report sections")
    methodology: str = Field(..., description="Analysis methodology")
    data_sources: List[str] = Field(..., description="Data sources used")
    limitations: List[str] = Field(default_factory=list, description="Analysis limitations")
    appendices: Dict[str, Any] = Field(default_factory=dict, description="Report appendices")


# API schemas
class APIResponse(BaseSchema):
    """Standard API response format."""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Any] = Field(None, description="Response data")
    message: str = Field(..., description="Response message")
    error: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class HealthCheck(BaseSchema):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Service uptime in seconds")
    dependencies: Dict[str, str] = Field(..., description="Dependency status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")


# Validation schemas
class DataValidationResult(BaseSchema):
    """Data validation result."""
    is_valid: bool = Field(..., description="Whether data is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    record_count: int = Field(..., description="Number of records validated")
    valid_records: int = Field(..., description="Number of valid records")


class UploadValidation(BaseSchema):
    """File upload validation."""
    filename: str = Field(..., description="Uploaded filename")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="Detected file type")
    is_valid: bool = Field(..., description="Whether upload is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    
    @validator("file_size")
    def validate_file_size(cls, v):
        """Validate file size is positive."""
        if v <= 0:
            raise ValueError("File size must be positive")
        return v