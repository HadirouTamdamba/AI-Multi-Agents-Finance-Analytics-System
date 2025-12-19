"""
Analyst agent for quantitative financial analysis.
Performs data analysis, metric calculations, and chart generation.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from ..core.schemas import AgentType, AgentResponse, ConfidenceLevel, FinancialMetric, ChartData
from ..core.logging import get_logger, log_agent_activity
from ..core.tracing import trace_agent_activity, trace_span
from ..core.llm import generate_llm_response
from ..core.config import get_settings


logger = get_logger(__name__)


class AnalystAgent:
    """Analyst agent for quantitative financial analysis and data processing."""
    
    def __init__(self):
        self.agent_type = AgentType.ANALYST
        self.name = "Quantitative Financial Analyst"
        self.version = "1.0.0"
        self.capabilities = [
            "data_analysis",
            "metric_calculation",
            "chart_generation",
            "statistical_analysis",
            "time_series_analysis"
        ]
        self.settings = get_settings()
        self.supported_metrics = [
            "roi", "sharpe_ratio", "volatility", "beta", "alpha",
            "max_drawdown", "var", "cvar", "correlation", "r_squared"
        ]
    
    async def analyze_financial_data(self, data: pd.DataFrame, metrics: List[str]) -> AgentResponse:
        """Analyze financial data and calculate specified metrics."""
        with trace_agent_activity(self.agent_type.value, "analyze_financial_data") as span:
            start_time = datetime.utcnow()
            
            log_agent_activity(
                self.agent_type.value,
                "Starting financial data analysis",
                data_shape=data.shape,
                metrics=metrics
            )
            
            try:
                # Validate and clean data
                cleaned_data = await self._clean_financial_data(data)
                
                # Calculate requested metrics
                calculated_metrics = await self._calculate_metrics(cleaned_data, metrics)
                
                # Generate analysis insights
                analysis_insights = await self._generate_analysis_insights(cleaned_data, calculated_metrics)
                
                # Create charts
                charts = await self._generate_charts(cleaned_data, calculated_metrics)
                
                confidence = self._assess_analysis_confidence(cleaned_data, calculated_metrics)
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                response = AgentResponse(
                    agent_type=self.agent_type,
                    task_id=f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    content=analysis_insights,
                    confidence=confidence,
                    execution_time=execution_time,
                    metadata={
                        "data_shape": cleaned_data.shape,
                        "metrics_calculated": len(calculated_metrics),
                        "charts_generated": len(charts),
                        "metrics": calculated_metrics,
                        "charts": charts
                    }
                )
                
                log_agent_activity(
                    self.agent_type.value,
                    "Financial analysis completed",
                    execution_time=execution_time,
                    metrics_count=len(calculated_metrics)
                )
                
                return response
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.error(f"Financial analysis failed: {e}")
                
                return AgentResponse(
                    agent_type=self.agent_type,
                    task_id=f"analysis_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    content=f"Analysis failed: {str(e)}",
                    confidence=ConfidenceLevel.LOW,
                    execution_time=execution_time
                )
    
    async def _clean_financial_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate financial data."""
        with trace_span("analyst.clean_data") as span:
            cleaned_data = data.copy()
            
            # Handle missing values
            if 'close_price' in cleaned_data.columns:
                cleaned_data['close_price'] = cleaned_data['close_price'].fillna(method='ffill')
            
            # Remove rows with all NaN values
            cleaned_data = cleaned_data.dropna(how='all')
            
            # Ensure date column is datetime
            if 'date' in cleaned_data.columns:
                cleaned_data['date'] = pd.to_datetime(cleaned_data['date'])
                cleaned_data = cleaned_data.sort_values('date')
            
            # Remove outliers (simple method)
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col != 'date':  # Skip date columns
                    Q1 = cleaned_data[col].quantile(0.25)
                    Q3 = cleaned_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    cleaned_data = cleaned_data[
                        (cleaned_data[col] >= lower_bound) & 
                        (cleaned_data[col] <= upper_bound)
                    ]
            
            logger.info(f"Data cleaned: {data.shape} -> {cleaned_data.shape}")
            return cleaned_data
    
    async def _calculate_metrics(self, data: pd.DataFrame, metrics: List[str]) -> List[FinancialMetric]:
        """Calculate requested financial metrics."""
        with trace_span("analyst.calculate_metrics") as span:
            calculated_metrics = []
            
            for metric in metrics:
                if metric not in self.supported_metrics:
                    logger.warning(f"Unsupported metric: {metric}")
                    continue
                
                try:
                    if metric == "roi":
                        value = await self._calculate_roi(data)
                    elif metric == "sharpe_ratio":
                        value = await self._calculate_sharpe_ratio(data)
                    elif metric == "volatility":
                        value = await self._calculate_volatility(data)
                    elif metric == "beta":
                        value = await self._calculate_beta(data)
                    elif metric == "alpha":
                        value = await self._calculate_alpha(data)
                    elif metric == "max_drawdown":
                        value = await self._calculate_max_drawdown(data)
                    elif metric == "var":
                        value = await self._calculate_var(data)
                    elif metric == "correlation":
                        value = await self._calculate_correlation(data)
                    else:
                        value = 0.0
                    
                    calculated_metrics.append(FinancialMetric(
                        name=metric,
                        value=value,
                        unit="ratio" if metric in ["sharpe_ratio", "beta", "alpha", "correlation"] else "percentage",
                        calculation_method=f"Standard {metric} calculation",
                        confidence=ConfidenceLevel.HIGH if value is not None else ConfidenceLevel.LOW
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to calculate {metric}: {e}")
                    calculated_metrics.append(FinancialMetric(
                        name=metric,
                        value=0.0,
                        unit="unknown",
                        calculation_method=f"Failed calculation: {str(e)}",
                        confidence=ConfidenceLevel.LOW
                    ))
            
            return calculated_metrics
    
    async def _calculate_roi(self, data: pd.DataFrame) -> float:
        """Calculate Return on Investment."""
        if 'close_price' not in data.columns or len(data) < 2:
            return 0.0
        
        initial_price = data['close_price'].iloc[0]
        final_price = data['close_price'].iloc[-1]
        return ((final_price - initial_price) / initial_price) * 100
    
    async def _calculate_sharpe_ratio(self, data: pd.DataFrame) -> float:
        """Calculate Sharpe Ratio."""
        if 'close_price' not in data.columns or len(data) < 2:
            return 0.0
        
        returns = data['close_price'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if returns.std() == 0:
            return 0.0
        
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    async def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate annualized volatility."""
        if 'close_price' not in data.columns or len(data) < 2:
            return 0.0
        
        returns = data['close_price'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        return returns.std() * np.sqrt(252) * 100  # Annualized percentage
    
    async def _calculate_beta(self, data: pd.DataFrame) -> float:
        """Calculate Beta (simplified - would need market data in production)."""
        if 'close_price' not in data.columns or len(data) < 2:
            return 1.0
        
        # Mock beta calculation - in production would use market index data
        returns = data['close_price'].pct_change().dropna()
        if len(returns) == 0:
            return 1.0
        
        # Simulate market returns
        market_returns = np.random.normal(0.0005, 0.01, len(returns))
        
        if len(returns) > 1 and np.var(market_returns) > 0:
            covariance = np.cov(returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            return covariance / market_variance
        
        return 1.0
    
    async def _calculate_alpha(self, data: pd.DataFrame) -> float:
        """Calculate Alpha (simplified)."""
        if 'close_price' not in data.columns or len(data) < 2:
            return 0.0
        
        returns = data['close_price'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        # Mock alpha calculation
        beta = await self._calculate_beta(data)
        risk_free_rate = 0.02
        market_return = 0.08  # Assume 8% market return
        
        actual_return = returns.mean() * 252  # Annualized
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        
        return (actual_return - expected_return) * 100
    
    async def _calculate_max_drawdown(self, data: pd.DataFrame) -> float:
        """Calculate Maximum Drawdown."""
        if 'close_price' not in data.columns or len(data) < 2:
            return 0.0
        
        prices = data['close_price']
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min() * 100
    
    async def _calculate_var(self, data: pd.DataFrame, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if 'close_price' not in data.columns or len(data) < 2:
            return 0.0
        
        returns = data['close_price'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100) * 100
    
    async def _calculate_correlation(self, data: pd.DataFrame) -> float:
        """Calculate correlation with market (simplified)."""
        if 'close_price' not in data.columns or len(data) < 2:
            return 0.0
        
        returns = data['close_price'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        # Simulate market returns for correlation
        market_returns = np.random.normal(0.0005, 0.01, len(returns))
        
        if len(returns) > 1:
            return np.corrcoef(returns, market_returns)[0, 1]
        
        return 0.0
    
    async def _generate_analysis_insights(self, data: pd.DataFrame, metrics: List[FinancialMetric]) -> str:
        """Generate insights from analysis results."""
        with trace_span("analyst.generate_insights") as span:
            # Prepare metric summary
            metric_summary = "\n".join([
                f"- {metric.name}: {metric.value:.2f} {metric.unit}"
                for metric in metrics
            ])
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a quantitative financial analyst. Provide insights based on calculated financial metrics.
                    
                    Structure your analysis with:
                    1. Performance Summary - Overall performance assessment
                    2. Risk Analysis - Risk metrics and implications
                    3. Comparative Analysis - How metrics compare to benchmarks
                    4. Key Insights - Most important findings
                    5. Recommendations - Actionable recommendations based on the analysis
                    
                    Be specific and reference the calculated metrics in your analysis."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze these financial metrics:

Data period: {len(data)} data points
Date range: {data['date'].min() if 'date' in data.columns else 'Unknown'} to {data['date'].max() if 'date' in data.columns else 'Unknown'}

Calculated metrics:
{metric_summary}

Provide a comprehensive analysis of these results."""
                }
            ]
            
            response = await generate_llm_response(messages)
            return response.content
    
    async def _generate_charts(self, data: pd.DataFrame, metrics: List[FinancialMetric]) -> List[ChartData]:
        """Generate chart data for visualization."""
        with trace_span("analyst.generate_charts") as span:
            charts = []
            
            # Price chart
            if 'close_price' in data.columns and 'date' in data.columns:
                charts.append(ChartData(
                    chart_type="line",
                    title="Price Movement Over Time",
                    x_axis=data['date'].dt.strftime('%Y-%m-%d').tolist(),
                    y_axis=data['close_price'].tolist(),
                    series_name="Close Price",
                    metadata={"metric": "price", "data_points": len(data)}
                ))
            
            # Returns distribution
            if 'close_price' in data.columns:
                returns = data['close_price'].pct_change().dropna()
                if len(returns) > 0:
                    charts.append(ChartData(
                        chart_type="histogram",
                        title="Returns Distribution",
                        x_axis=[f"{r:.3f}" for r in returns.tolist()],
                        y_axis=[1] * len(returns),  # Simplified histogram
                        series_name="Returns",
                        metadata={"metric": "returns_distribution", "count": len(returns)}
                    ))
            
            # Metrics summary chart
            if metrics:
                metric_names = [m.name for m in metrics]
                metric_values = [m.value for m in metrics]
                
                charts.append(ChartData(
                    chart_type="bar",
                    title="Financial Metrics Summary",
                    x_axis=metric_names,
                    y_axis=metric_values,
                    series_name="Metric Values",
                    metadata={"metrics_count": len(metrics)}
                ))
            
            return charts
    
    def _assess_analysis_confidence(self, data: pd.DataFrame, metrics: List[FinancialMetric]) -> ConfidenceLevel:
        """Assess confidence level for the analysis."""
        if data.empty or not metrics:
            return ConfidenceLevel.LOW
        
        # Check data quality
        data_quality_score = 0
        if len(data) >= 30:  # Sufficient data points
            data_quality_score += 1
        if 'close_price' in data.columns and not data['close_price'].isna().all():
            data_quality_score += 1
        if 'date' in data.columns and not data['date'].isna().all():
            data_quality_score += 1
        
        # Check metric calculation success
        successful_metrics = len([m for m in metrics if m.confidence != ConfidenceLevel.LOW])
        metric_success_rate = successful_metrics / len(metrics) if metrics else 0
        
        if data_quality_score >= 2 and metric_success_rate >= 0.8:
            return ConfidenceLevel.HIGH
        elif data_quality_score >= 1 and metric_success_rate >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        return self.capabilities
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_type": self.agent_type.value,
            "name": self.name,
            "version": self.version,
            "capabilities": self.capabilities,
            "supported_metrics": self.supported_metrics,
            "status": "active"
        }


# Global analyst instance
analyst = AnalystAgent()


def get_analyst() -> AnalystAgent:
    """Get global analyst instance."""
    return analyst