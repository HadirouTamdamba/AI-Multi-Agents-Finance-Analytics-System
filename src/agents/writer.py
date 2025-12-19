"""
Writer agent for generating financial reports and summaries.
Creates executive reports, summaries, and documentation.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from ..core.schemas import (
    AgentType, AgentResponse, ConfidenceLevel, ExecutiveSummary, 
    ReportSection, FinancialReport
)
from ..core.logging import get_logger, log_agent_activity
from ..core.tracing import trace_agent_activity, trace_span
from ..core.llm import generate_llm_response
from ..core.config import get_settings


logger = get_logger(__name__)


class WriterAgent:
    """Writer agent for financial report generation and documentation."""
    
    def __init__(self):
        self.agent_type = AgentType.WRITER
        self.name = "Financial Report Writer"
        self.version = "1.0.0"
        self.capabilities = [
            "report_generation",
            "summary_creation",
            "document_assembly",
            "executive_communication",
            "technical_writing"
        ]
        self.settings = get_settings()
        self.report_templates = {
            "executive_summary": "executive_summary_template",
            "full_report": "full_report_template",
            "risk_report": "risk_report_template",
            "performance_report": "performance_report_template"
        }
    
    async def generate_executive_summary(self, analysis_data: Dict[str, Any], target_audience: str = "executives") -> AgentResponse:
        """Generate an executive summary from analysis data."""
        with trace_agent_activity(self.agent_type.value, "generate_executive_summary") as span:
            start_time = datetime.utcnow()
            
            log_agent_activity(
                self.agent_type.value,
                "Starting executive summary generation",
                audience=target_audience
            )
            
            try:
                # Extract key information from analysis data
                key_findings = await self._extract_key_findings(analysis_data)
                recommendations = await self._extract_recommendations(analysis_data)
                risk_assessment = await self._extract_risk_assessment(analysis_data)
                
                # Generate executive summary content
                summary_content = await self._create_executive_summary_content(
                    key_findings, recommendations, risk_assessment, target_audience
                )
                
                # Create structured executive summary
                executive_summary = ExecutiveSummary(
                    overview=summary_content.get("overview", ""),
                    key_findings=key_findings,
                    recommendations=recommendations,
                    risk_assessment=risk_assessment,
                    confidence_level=ConfidenceLevel.MEDIUM
                )
                
                confidence = self._assess_writing_confidence(analysis_data, key_findings, recommendations)
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                response = AgentResponse(
                    agent_type=self.agent_type,
                    task_id=f"executive_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    content=summary_content.get("full_summary", ""),
                    confidence=confidence,
                    execution_time=execution_time,
                    metadata={
                        "report_type": "executive_summary",
                        "target_audience": target_audience,
                        "key_findings_count": len(key_findings),
                        "recommendations_count": len(recommendations),
                        "executive_summary": executive_summary.dict()
                    }
                )
                
                log_agent_activity(
                    self.agent_type.value,
                    "Executive summary completed",
                    execution_time=execution_time,
                    findings_count=len(key_findings)
                )
                
                return response
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.error(f"Executive summary generation failed: {e}")
                
                return AgentResponse(
                    agent_type=self.agent_type,
                    task_id=f"summary_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    content=f"Executive summary generation failed: {str(e)}",
                    confidence=ConfidenceLevel.LOW,
                    execution_time=execution_time
                )
    
    async def generate_full_report(self, analysis_data: Dict[str, Any], report_type: str = "full_report") -> AgentResponse:
        """Generate a comprehensive financial report."""
        with trace_agent_activity(self.agent_type.value, "generate_full_report") as span:
            start_time = datetime.utcnow()
            
            log_agent_activity(
                self.agent_type.value,
                "Starting full report generation",
                report_type=report_type
            )
            
            try:
                # Create report sections
                sections = await self._create_report_sections(analysis_data, report_type)
                
                # Generate methodology section
                methodology = await self._generate_methodology_section(analysis_data)
                
                # Create data sources list
                data_sources = await self._extract_data_sources(analysis_data)
                
                # Generate limitations
                limitations = await self._identify_limitations(analysis_data)
                
                # Create full report
                report = FinancialReport(
                    id=f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    title=f"Financial Analysis Report - {datetime.utcnow().strftime('%B %Y')}",
                    executive_summary=await self._create_executive_summary_from_data(analysis_data),
                    sections=sections,
                    methodology=methodology,
                    data_sources=data_sources,
                    limitations=limitations
                )
                
                # Generate full report content
                full_content = await self._assemble_full_report(report)
                
                confidence = self._assess_report_confidence(analysis_data, sections)
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                response = AgentResponse(
                    agent_type=self.agent_type,
                    task_id=f"full_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    content=full_content,
                    confidence=confidence,
                    execution_time=execution_time,
                    metadata={
                        "report_type": report_type,
                        "sections_count": len(sections),
                        "data_sources_count": len(data_sources),
                        "report": report.dict()
                    }
                )
                
                return response
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.error(f"Full report generation failed: {e}")
                
                return AgentResponse(
                    agent_type=self.agent_type,
                    task_id=f"report_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    content=f"Full report generation failed: {str(e)}",
                    confidence=ConfidenceLevel.LOW,
                    execution_time=execution_time
                )
    
    async def _extract_key_findings(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis data."""
        with trace_span("writer.extract_findings") as span:
            findings = []
            
            # Extract from research insights
            research_insights = analysis_data.get("research_insights", [])
            for insight in research_insights:
                if isinstance(insight, dict) and "content" in insight:
                    # Extract key points from research content
                    key_points = await self._extract_key_points_from_text(insight["content"])
                    findings.extend(key_points)
            
            # Extract from analytical findings
            analytical_findings = analysis_data.get("analytical_findings", [])
            for finding in analytical_findings:
                if isinstance(finding, dict) and "content" in finding:
                    key_points = await self._extract_key_points_from_text(finding["content"])
                    findings.extend(key_points)
            
            # Extract from risk assessments
            risk_assessments = analysis_data.get("risk_assessments", [])
            for assessment in risk_assessments:
                if isinstance(assessment, dict) and "content" in assessment:
                    key_points = await self._extract_key_points_from_text(assessment["content"])
                    findings.extend(key_points)
            
            # Limit to top findings
            return findings[:10] if findings else ["Analysis completed with standard findings"]
    
    async def _extract_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Extract recommendations from analysis data."""
        with trace_span("writer.extract_recommendations") as span:
            recommendations = []
            
            # Extract from all analysis components
            for component_type in ["research_insights", "analytical_findings", "risk_assessments", "writing_outputs"]:
                components = analysis_data.get(component_type, [])
                for component in components:
                    if isinstance(component, dict) and "content" in component:
                        recs = await self._extract_recommendations_from_text(component["content"])
                        recommendations.extend(recs)
            
            # Remove duplicates and limit
            unique_recommendations = list(set(recommendations))
            return unique_recommendations[:8] if unique_recommendations else ["Continue monitoring market conditions"]
    
    async def _extract_risk_assessment(self, analysis_data: Dict[str, Any]) -> str:
        """Extract risk assessment summary from analysis data."""
        with trace_span("writer.extract_risk_assessment") as span:
            risk_summaries = []
            
            risk_assessments = analysis_data.get("risk_assessments", [])
            for assessment in risk_assessments:
                if isinstance(assessment, dict) and "content" in assessment:
                    risk_summaries.append(assessment["content"])
            
            if risk_summaries:
                # Combine risk assessments
                combined_risk = "\n\n".join(risk_summaries)
                return await self._summarize_risk_assessment(combined_risk)
            else:
                return "Standard market risks apply. Regular monitoring recommended."
    
    async def _extract_key_points_from_text(self, text: str) -> List[str]:
        """Extract key points from text using LLM."""
        with trace_span("writer.extract_key_points") as span:
            messages = [
                {
                    "role": "system",
                    "content": "Extract 3-5 key findings or insights from the following text. Return them as a simple list, one per line."
                },
                {
                    "role": "user",
                    "content": f"Extract key findings from:\n\n{text[:1000]}"
                }
            ]
            
            response = await generate_llm_response(messages)
            # Parse response into list
            key_points = [point.strip() for point in response.content.split('\n') if point.strip()]
            return key_points[:5]  # Limit to 5 key points
    
    async def _extract_recommendations_from_text(self, text: str) -> List[str]:
        """Extract recommendations from text using LLM."""
        with trace_span("writer.extract_recommendations") as span:
            messages = [
                {
                    "role": "system",
                    "content": "Extract actionable recommendations from the following text. Return them as a simple list, one per line."
                },
                {
                    "role": "user",
                    "content": f"Extract recommendations from:\n\n{text[:1000]}"
                }
            ]
            
            response = await generate_llm_response(messages)
            # Parse response into list
            recommendations = [rec.strip() for rec in response.content.split('\n') if rec.strip()]
            return recommendations[:5]  # Limit to 5 recommendations
    
    async def _summarize_risk_assessment(self, risk_text: str) -> str:
        """Summarize risk assessment text."""
        with trace_span("writer.summarize_risk") as span:
            messages = [
                {
                    "role": "system",
                    "content": "Summarize the following risk assessment into 2-3 concise sentences highlighting the main risks and their implications."
                },
                {
                    "role": "user",
                    "content": f"Summarize this risk assessment:\n\n{risk_text[:1500]}"
                }
            ]
            
            response = await generate_llm_response(messages)
            return response.content
    
    async def _create_executive_summary_content(self, key_findings: List[str], recommendations: List[str], risk_assessment: str, audience: str) -> Dict[str, str]:
        """Create executive summary content."""
        with trace_span("writer.create_summary_content") as span:
            messages = [
                {
                    "role": "system",
                    "content": f"""Create a professional executive summary for {audience}. Structure it with:
                    1. Overview - High-level summary of the analysis
                    2. Key Findings - Most important insights
                    3. Recommendations - Actionable next steps
                    4. Risk Assessment - Key risk factors
                    
                    Write in a clear, concise style appropriate for executive decision-making."""
                },
                {
                    "role": "user",
                    "content": f"""Create an executive summary with:

Key Findings:
{chr(10).join(f"- {finding}" for finding in key_findings)}

Recommendations:
{chr(10).join(f"- {rec}" for rec in recommendations)}

Risk Assessment:
{risk_assessment}

Target Audience: {audience}"""
                }
            ]
            
            response = await generate_llm_response(messages)
            
            # Parse the response into sections
            content = response.content
            sections = content.split('\n\n')
            
            return {
                "overview": sections[0] if len(sections) > 0 else content,
                "full_summary": content
            }
    
    async def _create_report_sections(self, analysis_data: Dict[str, Any], report_type: str) -> List[ReportSection]:
        """Create report sections based on analysis data."""
        with trace_span("writer.create_sections") as span:
            sections = []
            
            # Market Overview Section
            if analysis_data.get("research_insights"):
                sections.append(ReportSection(
                    title="Market Overview",
                    content=await self._create_market_overview_section(analysis_data),
                    citations=[f"Research Analysis {i+1}" for i in range(len(analysis_data.get("research_insights", [])))]
                ))
            
            # Quantitative Analysis Section
            if analysis_data.get("analytical_findings"):
                sections.append(ReportSection(
                    title="Quantitative Analysis",
                    content=await self._create_quantitative_section(analysis_data),
                    citations=[f"Analytical Analysis {i+1}" for i in range(len(analysis_data.get("analytical_findings", [])))]
                ))
            
            # Risk Analysis Section
            if analysis_data.get("risk_assessments"):
                sections.append(ReportSection(
                    title="Risk Analysis",
                    content=await self._create_risk_analysis_section(analysis_data),
                    citations=[f"Risk Assessment {i+1}" for i in range(len(analysis_data.get("risk_assessments", [])))]
                ))
            
            # Recommendations Section
            sections.append(ReportSection(
                title="Recommendations",
                content=await self._create_recommendations_section(analysis_data),
                citations=[]
            ))
            
            return sections
    
    async def _create_market_overview_section(self, analysis_data: Dict[str, Any]) -> str:
        """Create market overview section."""
        with trace_span("writer.market_overview") as span:
            research_insights = analysis_data.get("research_insights", [])
            content_parts = []
            
            for insight in research_insights:
                if isinstance(insight, dict) and "content" in insight:
                    content_parts.append(insight["content"])
            
            if content_parts:
                combined_content = "\n\n".join(content_parts)
                return await self._summarize_section_content(combined_content, "market overview")
            else:
                return "Market analysis indicates standard conditions with typical volatility patterns."
    
    async def _create_quantitative_section(self, analysis_data: Dict[str, Any]) -> str:
        """Create quantitative analysis section."""
        with trace_span("writer.quantitative_section") as span:
            analytical_findings = analysis_data.get("analytical_findings", [])
            content_parts = []
            
            for finding in analytical_findings:
                if isinstance(finding, dict) and "content" in finding:
                    content_parts.append(finding["content"])
            
            if content_parts:
                combined_content = "\n\n".join(content_parts)
                return await self._summarize_section_content(combined_content, "quantitative analysis")
            else:
                return "Quantitative analysis shows standard performance metrics within expected ranges."
    
    async def _create_risk_analysis_section(self, analysis_data: Dict[str, Any]) -> str:
        """Create risk analysis section."""
        with trace_span("writer.risk_analysis_section") as span:
            risk_assessments = analysis_data.get("risk_assessments", [])
            content_parts = []
            
            for assessment in risk_assessments:
                if isinstance(assessment, dict) and "content" in assessment:
                    content_parts.append(assessment["content"])
            
            if content_parts:
                combined_content = "\n\n".join(content_parts)
                return await self._summarize_section_content(combined_content, "risk analysis")
            else:
                return "Risk analysis indicates standard market risks with appropriate mitigation strategies in place."
    
    async def _create_recommendations_section(self, analysis_data: Dict[str, Any]) -> str:
        """Create recommendations section."""
        with trace_span("writer.recommendations_section") as span:
            recommendations = await self._extract_recommendations(analysis_data)
            
            if recommendations:
                return "Based on the comprehensive analysis, the following recommendations are made:\n\n" + "\n".join(f"- {rec}" for rec in recommendations)
            else:
                return "Continue current investment strategy with regular monitoring and rebalancing as needed."
    
    async def _summarize_section_content(self, content: str, section_type: str) -> str:
        """Summarize section content."""
        with trace_span("writer.summarize_section") as span:
            messages = [
                {
                    "role": "system",
                    "content": f"Summarize the following {section_type} content into 2-3 clear, professional paragraphs suitable for a financial report."
                },
                {
                    "role": "user",
                    "content": f"Summarize this {section_type}:\n\n{content[:2000]}"
                }
            ]
            
            response = await generate_llm_response(messages)
            return response.content
    
    async def _generate_methodology_section(self, analysis_data: Dict[str, Any]) -> str:
        """Generate methodology section."""
        return """This analysis employs a multi-agent approach combining qualitative research, quantitative analysis, and risk modeling. The methodology includes:

1. **Research Analysis**: Systematic review of financial news, market reports, and industry publications to identify trends and sentiment.

2. **Quantitative Analysis**: Statistical analysis of historical data including calculation of key financial metrics such as returns, volatility, Sharpe ratio, and risk-adjusted performance measures.

3. **Risk Assessment**: Comprehensive risk modeling including Value at Risk (VaR), stress testing, scenario analysis, and Monte Carlo simulations.

4. **Data Sources**: Analysis based on publicly available financial data, market indices, and news sources with appropriate validation and quality checks.

5. **Limitations**: Results are based on historical data and may not predict future performance. Market conditions can change rapidly and impact investment outcomes."""
    
    async def _extract_data_sources(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Extract data sources from analysis data."""
        sources = []
        
        # Add sources from different analysis components
        for component_type in ["research_insights", "analytical_findings", "risk_assessments"]:
            components = analysis_data.get(component_type, [])
            for component in components:
                if isinstance(component, dict) and "citations" in component:
                    sources.extend(component["citations"])
        
        # Add default sources if none found
        if not sources:
            sources = [
                "Financial market data",
                "News and research publications",
                "Historical price data",
                "Market analysis reports"
            ]
        
        return list(set(sources))  # Remove duplicates
    
    async def _identify_limitations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Identify analysis limitations."""
        return [
            "Analysis based on historical data which may not predict future performance",
            "Market conditions can change rapidly affecting investment outcomes",
            "Limited data availability may impact analysis completeness",
            "Model assumptions may not hold under extreme market conditions",
            "Results are for informational purposes and not investment advice"
        ]
    
    async def _create_executive_summary_from_data(self, analysis_data: Dict[str, Any]) -> ExecutiveSummary:
        """Create executive summary from analysis data."""
        key_findings = await self._extract_key_findings(analysis_data)
        recommendations = await self._extract_recommendations(analysis_data)
        risk_assessment = await self._extract_risk_assessment(analysis_data)
        
        return ExecutiveSummary(
            overview="Comprehensive financial analysis completed using multi-agent approach",
            key_findings=key_findings,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            confidence_level=ConfidenceLevel.MEDIUM
        )
    
    async def _assemble_full_report(self, report: FinancialReport) -> str:
        """Assemble the full report content."""
        with trace_span("writer.assemble_report") as span:
            report_content = f"""
# {report.title}

## Executive Summary

{report.executive_summary.overview}

### Key Findings
{chr(10).join(f"- {finding}" for finding in report.executive_summary.key_findings)}

### Recommendations
{chr(10).join(f"- {rec}" for rec in report.executive_summary.recommendations)}

### Risk Assessment
{report.executive_summary.risk_assessment}

## Report Sections

"""
            
            for section in report.sections:
                report_content += f"### {section.title}\n\n{section.content}\n\n"
            
            report_content += f"""
## Methodology

{report.methodology}

## Data Sources

{chr(10).join(f"- {source}" for source in report.data_sources)}

## Limitations

{chr(10).join(f"- {limitation}" for limitation in report.limitations)}

---
*Report generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            return report_content
    
    def _assess_writing_confidence(self, analysis_data: Dict[str, Any], key_findings: List[str], recommendations: List[str]) -> ConfidenceLevel:
        """Assess confidence level for writing output."""
        if not analysis_data or not key_findings:
            return ConfidenceLevel.LOW
        
        # Check data completeness
        data_components = len([k for k in analysis_data.keys() if analysis_data[k]])
        findings_quality = len(key_findings) >= 3
        recommendations_quality = len(recommendations) >= 2
        
        if data_components >= 3 and findings_quality and recommendations_quality:
            return ConfidenceLevel.HIGH
        elif data_components >= 2 and (findings_quality or recommendations_quality):
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _assess_report_confidence(self, analysis_data: Dict[str, Any], sections: List[ReportSection]) -> ConfidenceLevel:
        """Assess confidence level for full report."""
        if not analysis_data or not sections:
            return ConfidenceLevel.LOW
        
        # Check report completeness
        section_count = len(sections)
        data_components = len([k for k in analysis_data.keys() if analysis_data[k]])
        
        if section_count >= 4 and data_components >= 3:
            return ConfidenceLevel.HIGH
        elif section_count >= 3 and data_components >= 2:
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
            "report_templates": list(self.report_templates.keys()),
            "status": "active"
        }


# Global writer instance
writer = WriterAgent()


def get_writer() -> WriterAgent:
    """Get global writer instance."""
    return writer