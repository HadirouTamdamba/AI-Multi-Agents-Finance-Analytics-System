"""
Risk Modeler agent for financial risk assessment and scenario analysis.
Creates risk models, stress tests, and scenario analyses.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..core.schemas import AgentType, AgentResponse, ConfidenceLevel, RiskScenario
from ..core.logging import get_logger, log_agent_activity
from ..core.tracing import trace_agent_activity, trace_span
from ..core.llm import generate_llm_response
from ..core.config import get_settings


logger = get_logger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics data structure."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    beta: float


class RiskModelerAgent:
    """Risk Modeler agent for financial risk assessment and scenario modeling."""
    
    def __init__(self):
        self.agent_type = AgentType.RISK_MODELER
        self.name = "Financial Risk Modeler"
        self.version = "1.0.0"
        self.capabilities = [
            "risk_assessment",
            "scenario_modeling",
            "stress_testing",
            "var_calculation",
            "monte_carlo_simulation"
        ]
        self.settings = get_settings()
        self.confidence_levels = [0.90, 0.95, 0.99]
    
    async def assess_risk(self, data: pd.DataFrame, scenarios: List[str] = None) -> AgentResponse:
        """Perform comprehensive risk assessment."""
        with trace_agent_activity(self.agent_type.value, "assess_risk") as span:
            start_time = datetime.utcnow()
            
            if scenarios is None:
                scenarios = ["bull", "base", "bear"]
            
            log_agent_activity(
                self.agent_type.value,
                "Starting risk assessment",
                data_shape=data.shape,
                scenarios=scenarios
            )
            
            try:
                # Calculate risk metrics
                risk_metrics = await self._calculate_risk_metrics(data)
                
                # Create scenario analyses
                scenario_analyses = await self._create_scenario_analyses(data, scenarios)
                
                # Perform stress tests
                stress_test_results = await self._perform_stress_tests(data)
                
                # Generate risk insights
                risk_insights = await self._generate_risk_insights(risk_metrics, scenario_analyses, stress_test_results)
                
                confidence = self._assess_risk_confidence(data, risk_metrics, scenario_analyses)
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                response = AgentResponse(
                    agent_type=self.agent_type,
                    task_id=f"risk_assessment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    content=risk_insights,
                    confidence=confidence,
                    execution_time=execution_time,
                    metadata={
                        "risk_metrics": risk_metrics.__dict__,
                        "scenarios": scenario_analyses,
                        "stress_tests": stress_test_results,
                        "data_points": len(data)
                    }
                )
                
                log_agent_activity(
                    self.agent_type.value,
                    "Risk assessment completed",
                    execution_time=execution_time,
                    scenarios_count=len(scenario_analyses)
                )
                
                return response
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.error(f"Risk assessment failed: {e}")
                
                return AgentResponse(
                    agent_type=self.agent_type,
                    task_id=f"risk_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    content=f"Risk assessment failed: {str(e)}",
                    confidence=ConfidenceLevel.LOW,
                    execution_time=execution_time
                )
    
    async def _calculate_risk_metrics(self, data: pd.DataFrame) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        with trace_span("risk_modeler.calculate_metrics") as span:
            if 'close_price' not in data.columns or len(data) < 2:
                return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
            
            returns = data['close_price'].pct_change().dropna()
            if len(returns) == 0:
                return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5) * 100
            var_99 = np.percentile(returns, 1) * 100
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
            cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100
            
            # Maximum Drawdown
            prices = data['close_price']
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            max_drawdown = drawdown.min() * 100
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252) * 100
            
            # Sharpe Ratio (simplified)
            risk_free_rate = 0.02
            excess_returns = returns - (risk_free_rate / 252)
            sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            # Beta (simplified - would need market data in production)
            market_returns = np.random.normal(0.0005, 0.01, len(returns))
            if len(returns) > 1 and np.var(market_returns) > 0:
                covariance = np.cov(returns, market_returns)[0, 1]
                beta = covariance / np.var(market_returns)
            else:
                beta = 1.0
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                beta=beta
            )
    
    async def _create_scenario_analyses(self, data: pd.DataFrame, scenarios: List[str]) -> List[RiskScenario]:
        """Create scenario analyses for different market conditions."""
        with trace_span("risk_modeler.create_scenarios") as span:
            scenario_analyses = []
            
            if 'close_price' not in data.columns or len(data) < 2:
                return scenario_analyses
            
            returns = data['close_price'].pct_change().dropna()
            if len(returns) == 0:
                return scenario_analyses            
            
            base_return = returns.mean()
            base_volatility = returns.std()
            
            for scenario in scenarios:
                if scenario.lower() == "bull":
                    # Bull market scenario
                    scenario_return = base_return + 0.02  # 2% higher returns
                    scenario_volatility = base_volatility * 0.8  # 20% lower volatility
                    probability = 0.25
                    impact = 0.8
                    description = "Optimistic market conditions with strong growth and low volatility"
                    mitigation = ["Diversify across growth sectors", "Consider momentum strategies"]
                
                elif scenario.lower() == "bear":
                    # Bear market scenario
                    scenario_return = base_return - 0.03  # 3% lower returns
                    scenario_volatility = base_volatility * 1.5  # 50% higher volatility
                    probability = 0.20
                    impact = 0.3
                    description = "Pessimistic market conditions with declining prices and high volatility"
                    mitigation = ["Increase defensive positions", "Consider hedging strategies", "Maintain cash reserves"]
                
                else:  # base scenario
                    # Base case scenario
                    scenario_return = base_return
                    scenario_volatility = base_volatility
                    probability = 0.55
                    impact = 0.5
                    description = "Normal market conditions with moderate growth and volatility"
                    mitigation = ["Maintain balanced portfolio", "Regular rebalancing"]
                
                scenario_analyses.append(RiskScenario(
                    scenario_name=scenario.upper(),
                    probability=probability,
                    impact=impact,
                    description=description,
                    mitigation_strategies=mitigation
                ))
            
            return scenario_analyses
    
    async def _perform_stress_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform stress tests on the portfolio."""
        with trace_span("risk_modeler.stress_tests") as span:
            stress_results = {}
            
            if 'close_price' not in data.columns or len(data) < 2:
                return stress_results
            
            returns = data['close_price'].pct_change().dropna()
            if len(returns) == 0:
                return stress_results
            
            # Historical stress test - worst 5-day period
            if len(returns) >= 5:
                rolling_5d = returns.rolling(window=5).sum()
                worst_5d = rolling_5d.min()
                stress_results["worst_5_day_period"] = {
                    "return": worst_5d * 100,
                    "description": "Worst 5-day cumulative return"
                }
            
            # Historical stress test - worst month
            if len(returns) >= 20:
                rolling_20d = returns.rolling(window=20).sum()
                worst_month = rolling_20d.min()
                stress_results["worst_month"] = {
                    "return": worst_month * 100,
                    "description": "Worst 20-day (monthly) cumulative return"
                }
            
            # Monte Carlo stress test
            monte_carlo_results = await self._monte_carlo_stress_test(returns)
            stress_results["monte_carlo"] = monte_carlo_results
            
            # Market shock scenarios
            shock_scenarios = await self._market_shock_scenarios(returns)
            stress_results["market_shocks"] = shock_scenarios
            
            return stress_results
    
    async def _monte_carlo_stress_test(self, returns: pd.Series, simulations: int = 1000) -> Dict[str, Any]:
        """Perform Monte Carlo stress test simulation."""
        with trace_span("risk_modeler.monte_carlo") as span:
            if len(returns) == 0:
                return {"error": "Insufficient data for Monte Carlo simulation"}
            
            # Fit normal distribution to returns
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random scenarios
            simulated_returns = np.random.normal(mean_return, std_return, simulations)
            
            # Calculate VaR and CVaR from simulations
            var_95_sim = np.percentile(simulated_returns, 5) * 100
            var_99_sim = np.percentile(simulated_returns, 1) * 100
            cvar_95_sim = simulated_returns[simulated_returns <= np.percentile(simulated_returns, 5)].mean() * 100
            cvar_99_sim = simulated_returns[simulated_returns <= np.percentile(simulated_returns, 1)].mean() * 100
            
            return {
                "simulations": simulations,
                "var_95": var_95_sim,
                "var_99": var_99_sim,
                "cvar_95": cvar_95_sim,
                "cvar_99": cvar_99_sim,
                "mean_simulated_return": simulated_returns.mean() * 100,
                "std_simulated_return": simulated_returns.std() * 100
            }
    
    async def _market_shock_scenarios(self, returns: pd.Series) -> Dict[str, Any]:
        """Test portfolio under various market shock scenarios."""
        with trace_span("risk_modeler.market_shocks") as span:
            if len(returns) == 0:
                return {"error": "Insufficient data for shock scenarios"}
            
            base_return = returns.mean()
            base_volatility = returns.std()
            
            shock_scenarios = {
                "mild_recession": {
                    "return_shock": -0.05,  # -5% return shock
                    "volatility_multiplier": 1.2,
                    "description": "Mild economic downturn"
                },
                "severe_recession": {
                    "return_shock": -0.15,  # -15% return shock
                    "volatility_multiplier": 2.0,
                    "description": "Severe economic downturn"
                },
                "market_crash": {
                    "return_shock": -0.25,  # -25% return shock
                    "volatility_multiplier": 3.0,
                    "description": "Market crash scenario"
                },
                "interest_rate_shock": {
                    "return_shock": -0.08,  # -8% return shock
                    "volatility_multiplier": 1.5,
                    "description": "Rapid interest rate increase"
                }
            }
            
            # Calculate impact for each scenario
            for scenario_name, scenario_data in shock_scenarios.items():
                shocked_return = base_return + scenario_data["return_shock"]
                shocked_volatility = base_volatility * scenario_data["volatility_multiplier"]
                
                # Calculate VaR under shock conditions
                shock_var_95 = np.percentile(np.random.normal(shocked_return, shocked_volatility, 1000), 5) * 100
                
                scenario_data["impact_var_95"] = shock_var_95
                scenario_data["shocked_return"] = shocked_return * 100
                scenario_data["shocked_volatility"] = shocked_volatility * 100
            
            return shock_scenarios
    
    async def _generate_risk_insights(self, risk_metrics: RiskMetrics, scenarios: List[RiskScenario], stress_tests: Dict[str, Any]) -> str:
        """Generate comprehensive risk insights and recommendations."""
        with trace_span("risk_modeler.generate_insights") as span:
            # Prepare risk summary
            risk_summary = f"""
Risk Metrics Summary:
- 95% VaR: {risk_metrics.var_95:.2f}%
- 99% VaR: {risk_metrics.var_99:.2f}%
- Maximum Drawdown: {risk_metrics.max_drawdown:.2f}%
- Volatility: {risk_metrics.volatility:.2f}%
- Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}
- Beta: {risk_metrics.beta:.2f}
"""
            
            scenario_summary = "\n".join([
                f"- {scenario.scenario_name}: {scenario.probability*100:.0f}% probability, {scenario.impact*100:.0f}% impact"
                for scenario in scenarios
            ])
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a senior risk management analyst. Provide comprehensive risk insights based on quantitative analysis.
                    
                    Structure your analysis with:
                    1. Risk Profile Summary - Overall risk characteristics
                    2. Scenario Analysis - Key scenarios and their implications
                    3. Stress Test Results - Performance under adverse conditions
                    4. Risk Factors - Major risk drivers and concerns
                    5. Recommendations - Risk management strategies and actions
                    
                    Be specific about risk levels, provide actionable recommendations, and highlight any red flags."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze the following risk assessment results:

{risk_summary}

Scenario Analysis:
{scenario_summary}

Stress Test Results:
{str(stress_tests)}

Provide a comprehensive risk analysis and recommendations."""
                }
            ]
            
            response = await generate_llm_response(messages)
            return response.content
    
    def _assess_risk_confidence(self, data: pd.DataFrame, risk_metrics: RiskMetrics, scenarios: List[RiskScenario]) -> ConfidenceLevel:
        """Assess confidence level for risk analysis."""
        if data.empty or not scenarios:
            return ConfidenceLevel.LOW
        
        # Check data quality
        data_quality_score = 0
        if len(data) >= 30:  # Sufficient data points
            data_quality_score += 1
        if 'close_price' in data.columns and not data['close_price'].isna().all():
            data_quality_score += 1
        
        # Check if risk metrics are reasonable
        metrics_quality_score = 0
        if abs(risk_metrics.var_95) < 50:  # VaR not extremely high
            metrics_quality_score += 1
        if abs(risk_metrics.max_drawdown) < 100:  # Drawdown not over 100%
            metrics_quality_score += 1
        if 0 < risk_metrics.volatility < 100:  # Volatility in reasonable range
            metrics_quality_score += 1
        
        # Check scenario completeness
        scenario_score = 1 if len(scenarios) >= 3 else 0
        
        total_score = data_quality_score + metrics_quality_score + scenario_score
        
        if total_score >= 5:
            return ConfidenceLevel.HIGH
        elif total_score >= 3:
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
            "confidence_levels": self.confidence_levels,
            "status": "active"
        }


# Global risk modeler instance
risk_modeler = RiskModelerAgent()


def get_risk_modeler() -> RiskModelerAgent:
    """Get global risk modeler instance."""
    return risk_modeler