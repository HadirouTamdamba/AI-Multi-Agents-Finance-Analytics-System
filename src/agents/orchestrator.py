"""
Orchestrator agent for coordinating the AI Finance Agent Team.
Manages workflow execution and result aggregation.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..core.schemas import (
    AgentType, AgentTask, AgentResponse, PipelineResult, 
    TaskStatus, ConfidenceLevel, ExecutiveSummary
)
from ..core.logging import get_logger, log_agent_activity
from ..core.tracing import trace_agent_activity, trace_span
from ..core.llm import generate_llm_response


logger = get_logger(__name__)


class OrchestratorAgent:
    """Orchestrator agent that coordinates the finance analysis team."""
    
    def __init__(self):
        self.agent_type = AgentType.ORCHESTRATOR
        self.name = "Finance Analysis Orchestrator"
        self.version = "1.0.0"
        self.capabilities = [
            "workflow_management",
            "task_coordination", 
            "result_aggregation",
            "report_synthesis"
        ]
    
    async def execute_pipeline(self, pipeline_config: Dict[str, Any]) -> PipelineResult:
        """Execute the complete finance analysis pipeline."""
        with trace_agent_activity(self.agent_type.value, "execute_pipeline") as span:
            pipeline_id = f"pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            log_agent_activity(
                self.agent_type.value,
                "Starting pipeline execution",
                pipeline_id=pipeline_id
            )
            
            start_time = datetime.utcnow()
            
            try:
                # Step 1: Plan workflow
                workflow_plan = await self._plan_workflow(pipeline_config)
                
                # Step 2: Execute tasks
                agent_responses = await self._execute_workflow(workflow_plan)
                
                # Step 3: Aggregate results
                aggregated_results = await self._aggregate_results(agent_responses)
                
                # Step 4: Generate final report
                final_report = await self._generate_final_report(aggregated_results)
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                result = PipelineResult(
                    id=pipeline_id,
                    config=pipeline_config,
                    status=TaskStatus.COMPLETED,
                    agent_responses=agent_responses,
                    final_report=final_report,
                    execution_time=execution_time
                )
                
                log_agent_activity(
                    self.agent_type.value,
                    "Pipeline completed successfully",
                    pipeline_id=pipeline_id,
                    execution_time=execution_time
                )
                
                return result
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.error(f"Pipeline execution failed: {e}")
                
                return PipelineResult(
                    id=pipeline_id,
                    config=pipeline_config,
                    status=TaskStatus.FAILED,
                    execution_time=execution_time,
                    error_message=str(e)
                )
    
    async def _plan_workflow(self, config: Dict[str, Any]) -> List[AgentTask]:
        """Plan the workflow based on configuration."""
        with trace_span("orchestrator.plan_workflow") as span:
            tasks = []
            task_id_counter = 1
            
            # Determine which agents to use based on data sources
            data_sources = config.get("data_sources", [])
            agents_to_use = config.get("agents", [])
            
            # Default workflow if no specific agents specified
            if not agents_to_use:
                if "news" in data_sources or "text" in data_sources:
                    agents_to_use.append(AgentType.RESEARCHER)
                if "csv" in data_sources:
                    agents_to_use.append(AgentType.ANALYST)
                agents_to_use.extend([AgentType.RISK_MODELER, AgentType.WRITER])
            
            # Create tasks for each agent
            for agent_type in agents_to_use:
                if agent_type == AgentType.RESEARCHER:
                    task = AgentTask(
                        id=f"task_{task_id_counter}",
                        agent_type=agent_type,
                        task_type="text_analysis",
                        input_data={
                            "data_sources": [ds for ds in data_sources if ds in ["news", "text"]],
                            "analysis_focus": config.get("analysis_focus", "general")
                        },
                        parameters=config.get("researcher_params", {})
                    )
                    tasks.append(task)
                    task_id_counter += 1
                
                elif agent_type == AgentType.ANALYST:
                    task = AgentTask(
                        id=f"task_{task_id_counter}",
                        agent_type=agent_type,
                        task_type="data_analysis",
                        input_data={
                            "data_sources": [ds for ds in data_sources if ds == "csv"],
                            "metrics_to_calculate": config.get("metrics", ["roi", "volatility", "sharpe_ratio"])
                        },
                        parameters=config.get("analyst_params", {})
                    )
                    tasks.append(task)
                    task_id_counter += 1
                
                elif agent_type == AgentType.RISK_MODELER:
                    task = AgentTask(
                        id=f"task_{task_id_counter}",
                        agent_type=agent_type,
                        task_type="risk_assessment",
                        input_data={
                            "scenarios": config.get("scenarios", ["bull", "base", "bear"]),
                            "confidence_level": config.get("confidence_level", 0.95)
                        },
                        parameters=config.get("risk_params", {})
                    )
                    tasks.append(task)
                    task_id_counter += 1
                
                elif agent_type == AgentType.WRITER:
                    task = AgentTask(
                        id=f"task_{task_id_counter}",
                        agent_type=agent_type,
                        task_type="report_generation",
                        input_data={
                            "report_type": config.get("report_type", "executive_summary"),
                            "target_audience": config.get("audience", "executives")
                        },
                        parameters=config.get("writer_params", {})
                    )
                    tasks.append(task)
                    task_id_counter += 1
            
            logger.info(f"Planned workflow with {len(tasks)} tasks")
            return tasks
    
    async def _execute_workflow(self, tasks: List[AgentTask]) -> List[AgentResponse]:
        """Execute the planned workflow."""
        with trace_span("orchestrator.execute_workflow") as span:
            responses = []
            
            # For now, execute tasks sequentially
            # In a production system, this would be parallel with dependency management
            for task in tasks:
                try:
                    response = await self._execute_agent_task(task)
                    responses.append(response)
                    
                    # Update task status
                    task.status = TaskStatus.COMPLETED
                    task.result = response.dict()
                    
                except Exception as e:
                    logger.error(f"Task {task.id} failed: {e}")
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    
                    # Create error response
                    error_response = AgentResponse(
                        agent_type=task.agent_type,
                        task_id=task.id,
                        content=f"Task failed: {str(e)}",
                        confidence=ConfidenceLevel.LOW,
                        execution_time=0.0
                    )
                    responses.append(error_response)
            
            return responses
    
    async def _execute_agent_task(self, task: AgentTask) -> AgentResponse:
        """Execute a single agent task."""
        with trace_span(f"orchestrator.execute_task.{task.agent_type.value}") as span:
            start_time = datetime.utcnow()
            
            # This is a simplified implementation
            # In a real system, this would delegate to the actual agent instances
            
            if task.agent_type == AgentType.RESEARCHER:
                content = await self._mock_researcher_task(task)
            elif task.agent_type == AgentType.ANALYST:
                content = await self._mock_analyst_task(task)
            elif task.agent_type == AgentType.RISK_MODELER:
                content = await self._mock_risk_modeler_task(task)
            elif task.agent_type == AgentType.WRITER:
                content = await self._mock_writer_task(task)
            else:
                content = f"Unknown agent type: {task.agent_type}"
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return AgentResponse(
                agent_type=task.agent_type,
                task_id=task.id,
                content=content,
                confidence=ConfidenceLevel.MEDIUM,
                execution_time=execution_time
            )
    
    async def _mock_researcher_task(self, task: AgentTask) -> str:
        """Mock researcher task execution."""
        messages = [
            {
                "role": "system",
                "content": "You are a financial researcher analyzing news and text data for investment insights."
            },
            {
                "role": "user", 
                "content": f"Analyze the following data sources: {task.input_data.get('data_sources', [])}. Focus on: {task.input_data.get('analysis_focus', 'general')}"
            }
        ]
        
        response = await generate_llm_response(messages)
        return response.content
    
    async def _mock_analyst_task(self, task: AgentTask) -> str:
        """Mock analyst task execution."""
        messages = [
            {
                "role": "system",
                "content": "You are a financial analyst performing quantitative analysis on market data."
            },
            {
                "role": "user",
                "content": f"Analyze CSV data and calculate these metrics: {task.input_data.get('metrics_to_calculate', [])}"
            }
        ]
        
        response = await generate_llm_response(messages)
        return response.content
    
    async def _mock_risk_modeler_task(self, task: AgentTask) -> str:
        """Mock risk modeler task execution."""
        messages = [
            {
                "role": "system",
                "content": "You are a risk modeler creating scenario analyses and risk assessments."
            },
            {
                "role": "user",
                "content": f"Create risk scenarios: {task.input_data.get('scenarios', [])} with confidence level {task.input_data.get('confidence_level', 0.95)}"
            }
        ]
        
        response = await generate_llm_response(messages)
        return response.content
    
    async def _mock_writer_task(self, task: AgentTask) -> str:
        """Mock writer task execution."""
        messages = [
            {
                "role": "system",
                "content": "You are a financial writer creating executive reports and summaries."
            },
            {
                "role": "user",
                "content": f"Create a {task.input_data.get('report_type', 'executive_summary')} for {task.input_data.get('target_audience', 'executives')}"
            }
        ]
        
        response = await generate_llm_response(messages)
        return response.content
    
    async def _aggregate_results(self, responses: List[AgentResponse]) -> Dict[str, Any]:
        """Aggregate results from all agents."""
        with trace_span("orchestrator.aggregate_results") as span:
            aggregated = {
                "research_insights": [],
                "analytical_findings": [],
                "risk_assessments": [],
                "writing_outputs": [],
                "overall_confidence": ConfidenceLevel.MEDIUM,
                "key_themes": [],
                "recommendations": []
            }
            
            for response in responses:
                if response.agent_type == AgentType.RESEARCHER:
                    aggregated["research_insights"].append({
                        "content": response.content,
                        "confidence": response.confidence,
                        "citations": response.citations
                    })
                elif response.agent_type == AgentType.ANALYST:
                    aggregated["analytical_findings"].append({
                        "content": response.content,
                        "confidence": response.confidence,
                        "metrics": response.metadata.get("metrics", [])
                    })
                elif response.agent_type == AgentType.RISK_MODELER:
                    aggregated["risk_assessments"].append({
                        "content": response.content,
                        "confidence": response.confidence,
                        "scenarios": response.metadata.get("scenarios", [])
                    })
                elif response.agent_type == AgentType.WRITER:
                    aggregated["writing_outputs"].append({
                        "content": response.content,
                        "confidence": response.confidence,
                        "report_type": response.metadata.get("report_type", "summary")
                    })
            
            # Determine overall confidence
            confidences = [r.confidence for r in responses]
            if all(c == ConfidenceLevel.HIGH for c in confidences):
                aggregated["overall_confidence"] = ConfidenceLevel.HIGH
            elif any(c == ConfidenceLevel.LOW for c in confidences):
                aggregated["overall_confidence"] = ConfidenceLevel.LOW
            
            logger.info(f"Aggregated results from {len(responses)} agent responses")
            return aggregated
    
    async def _generate_final_report(self, aggregated_results: Dict[str, Any]) -> str:
        """Generate the final executive report."""
        with trace_span("orchestrator.generate_final_report") as span:
            messages = [
                {
                    "role": "system",
                    "content": "You are a senior financial analyst creating an executive summary report. Synthesize insights from multiple analysis components into a coherent, actionable report."
                },
                {
                    "role": "user",
                    "content": f"""Create an executive summary report based on the following aggregated analysis:

Research Insights: {aggregated_results.get('research_insights', [])}
Analytical Findings: {aggregated_results.get('analytical_findings', [])}
Risk Assessments: {aggregated_results.get('risk_assessments', [])}
Writing Outputs: {aggregated_results.get('writing_outputs', [])}

Overall Confidence Level: {aggregated_results.get('overall_confidence', 'medium')}

Please provide:
1. Executive Summary
2. Key Findings
3. Risk Assessment
4. Recommendations
5. Next Steps

Format as a professional executive report."""
                }
            ]
            
            response = await generate_llm_response(messages)
            return response.content
    
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
            "status": "active"
        }


# Global orchestrator instance
orchestrator = OrchestratorAgent()


def get_orchestrator() -> OrchestratorAgent:
    """Get global orchestrator instance."""
    return orchestrator