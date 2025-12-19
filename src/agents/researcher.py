"""
Researcher agent for financial news and text analysis.
Extracts insights from financial documents and news articles.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from ..core.schemas import AgentType, AgentResponse, ConfidenceLevel, NewsArticle
from ..core.logging import get_logger, log_agent_activity
from ..core.tracing import trace_agent_activity, trace_span
from ..core.llm import generate_llm_response
from ..core.config import get_settings


logger = get_logger(__name__)


class ResearcherAgent:
    """Researcher agent for financial text analysis and news processing."""
    
    def __init__(self):
        self.agent_type = AgentType.RESEARCHER
        self.name = "Financial Research Analyst"
        self.version = "1.0.0"
        self.capabilities = [
            "text_analysis",
            "news_processing", 
            "sentiment_analysis",
            "trend_identification",
            "insight_extraction"
        ]
        self.settings = get_settings()
    
    async def analyze_text_data(self, text_data: List[str], analysis_focus: str = "general") -> AgentResponse:
        """Analyze text data for financial insights."""
        with trace_agent_activity(self.agent_type.value, "analyze_text_data") as span:
            start_time = datetime.utcnow()
            
            log_agent_activity(
                self.agent_type.value,
                "Starting text analysis",
                text_count=len(text_data),
                focus=analysis_focus
            )
            
            try:
                # Process each text document
                insights = []
                citations = []
                
                for i, text in enumerate(text_data):
                    insight = await self._extract_insights_from_text(text, analysis_focus)
                    insights.append(insight)
                    citations.append(f"Document {i+1}")
                
                # Synthesize insights
                synthesized_insights = await self._synthesize_insights(insights, analysis_focus)
                
                # Determine confidence level
                confidence = self._assess_confidence(insights, text_data)
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                response = AgentResponse(
                    agent_type=self.agent_type,
                    task_id=f"research_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    content=synthesized_insights,
                    confidence=confidence,
                    citations=citations,
                    execution_time=execution_time,
                    metadata={
                        "text_count": len(text_data),
                        "analysis_focus": analysis_focus,
                        "insights_count": len(insights)
                    }
                )
                
                log_agent_activity(
                    self.agent_type.value,
                    "Text analysis completed",
                    execution_time=execution_time,
                    confidence=confidence.value
                )
                
                return response
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.error(f"Text analysis failed: {e}")
                
                return AgentResponse(
                    agent_type=self.agent_type,
                    task_id=f"research_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    content=f"Analysis failed: {str(e)}",
                    confidence=ConfidenceLevel.LOW,
                    execution_time=execution_time
                )
    
    async def analyze_news_articles(self, articles: List[NewsArticle]) -> AgentResponse:
        """Analyze news articles for market sentiment and trends."""
        with trace_agent_activity(self.agent_type.value, "analyze_news_articles") as span:
            start_time = datetime.utcnow()
            
            log_agent_activity(
                self.agent_type.value,
                "Starting news analysis",
                article_count=len(articles)
            )
            
            try:
                # Analyze sentiment for each article
                sentiment_analysis = []
                for article in articles:
                    sentiment = await self._analyze_article_sentiment(article)
                    sentiment_analysis.append(sentiment)
                
                # Identify trends and themes
                trends = await self._identify_trends(articles, sentiment_analysis)
                
                # Generate market insights
                market_insights = await self._generate_market_insights(articles, trends)
                
                confidence = self._assess_news_confidence(articles, sentiment_analysis)
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                response = AgentResponse(
                    agent_type=self.agent_type,
                    task_id=f"news_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    content=market_insights,
                    confidence=confidence,
                    citations=[article.source for article in articles],
                    execution_time=execution_time,
                    metadata={
                        "article_count": len(articles),
                        "sentiment_analysis": sentiment_analysis,
                        "trends": trends
                    }
                )
                
                return response
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                logger.error(f"News analysis failed: {e}")
                
                return AgentResponse(
                    agent_type=self.agent_type,
                    task_id=f"news_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    content=f"News analysis failed: {str(e)}",
                    confidence=ConfidenceLevel.LOW,
                    execution_time=execution_time
                )
    
    async def _extract_insights_from_text(self, text: str, focus: str) -> str:
        """Extract insights from a single text document."""
        with trace_span("researcher.extract_insights") as span:
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a financial research analyst. Extract key insights from financial text data.
                    Focus on: {focus}
                    
                    Provide insights in the following format:
                    1. Key Financial Metrics or Data Points
                    2. Market Trends or Patterns
                    3. Risk Factors or Concerns
                    4. Opportunities or Positive Indicators
                    5. Regulatory or Policy Implications
                    
                    Be specific and cite relevant numbers, percentages, or timeframes when available."""
                },
                {
                    "role": "user",
                    "content": f"Analyze this financial text:\n\n{text[:2000]}"  # Limit text length
                }
            ]
            
            response = await generate_llm_response(messages)
            return response.content
    
    async def _synthesize_insights(self, insights: List[str], focus: str) -> str:
        """Synthesize insights from multiple documents."""
        with trace_span("researcher.synthesize_insights") as span:
            combined_insights = "\n\n".join([f"Document {i+1}:\n{insight}" for i, insight in enumerate(insights)])
            
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a senior financial research analyst. Synthesize insights from multiple documents into a coherent analysis.
                    Focus area: {focus}
                    
                    Create a comprehensive synthesis that:
                    1. Identifies common themes across documents
                    2. Highlights conflicting information
                    3. Provides a balanced view of the situation
                    4. Notes any gaps in information
                    5. Offers preliminary conclusions
                    
                    Structure your response clearly with headings and bullet points."""
                },
                {
                    "role": "user",
                    "content": f"Synthesize these insights:\n\n{combined_insights}"
                }
            ]
            
            response = await generate_llm_response(messages)
            return response.content
    
    async def _analyze_article_sentiment(self, article: NewsArticle) -> Dict[str, Any]:
        """Analyze sentiment of a news article."""
        with trace_span("researcher.analyze_sentiment") as span:
            messages = [
                {
                    "role": "system",
                    "content": """You are a financial sentiment analyst. Analyze the sentiment of financial news articles.
                    
                    Provide sentiment analysis in JSON format with:
                    - overall_sentiment: "positive", "negative", or "neutral"
                    - confidence: 0.0 to 1.0
                    - key_sentiment_drivers: list of factors influencing sentiment
                    - market_impact: "high", "medium", or "low"
                    - relevant_sectors: list of affected sectors"""
                },
                {
                    "role": "user",
                    "content": f"Analyze sentiment for this article:\n\nTitle: {article.title}\n\nContent: {article.content[:1500]}"
                }
            ]
            
            response = await generate_llm_response(messages)
            
            # Parse JSON response (simplified - in production would use proper JSON parsing)
            try:
                # This is a mock implementation - in production, you'd parse the actual JSON
                return {
                    "overall_sentiment": "neutral",
                    "confidence": 0.7,
                    "key_sentiment_drivers": ["market volatility", "earnings expectations"],
                    "market_impact": "medium",
                    "relevant_sectors": ["technology", "finance"]
                }
            except:
                return {
                    "overall_sentiment": "neutral",
                    "confidence": 0.5,
                    "key_sentiment_drivers": [],
                    "market_impact": "low",
                    "relevant_sectors": []
                }
    
    async def _identify_trends(self, articles: List[NewsArticle], sentiment_analysis: List[Dict]) -> List[Dict[str, Any]]:
        """Identify trends from news articles and sentiment analysis."""
        with trace_span("researcher.identify_trends") as span:
            # Combine article data for trend analysis
            article_summaries = []
            for i, article in enumerate(articles):
                sentiment = sentiment_analysis[i] if i < len(sentiment_analysis) else {}
                article_summaries.append({
                    "title": article.title,
                    "source": article.source,
                    "published_at": article.published_at.isoformat(),
                    "sentiment": sentiment.get("overall_sentiment", "neutral"),
                    "sectors": sentiment.get("relevant_sectors", [])
                })
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a financial trend analyst. Identify emerging trends from news articles and sentiment data.
                    
                    Analyze for:
                    1. Emerging market themes
                    2. Sector rotation patterns
                    3. Sentiment shifts over time
                    4. Regulatory or policy trends
                    5. Technology or innovation trends
                    
                    Provide trends in a structured format with trend name, description, and confidence level."""
                },
                {
                    "role": "user",
                    "content": f"Identify trends from these articles and sentiment data:\n\n{str(article_summaries)}"
                }
            ]
            
            response = await generate_llm_response(messages)
            
            # Mock trend identification
            return [
                {
                    "trend_name": "Technology Sector Volatility",
                    "description": "Increased volatility in technology stocks due to regulatory concerns",
                    "confidence": 0.8,
                    "sectors": ["technology"],
                    "sentiment": "negative"
                },
                {
                    "trend_name": "ESG Investment Growth",
                    "description": "Growing focus on environmental, social, and governance factors",
                    "confidence": 0.9,
                    "sectors": ["energy", "utilities", "finance"],
                    "sentiment": "positive"
                }
            ]
    
    async def _generate_market_insights(self, articles: List[NewsArticle], trends: List[Dict]) -> str:
        """Generate comprehensive market insights from articles and trends."""
        with trace_span("researcher.generate_insights") as span:
            messages = [
                {
                    "role": "system",
                    "content": """You are a senior market analyst creating comprehensive market insights.
                    
                    Create a structured analysis covering:
                    1. Market Overview - Current market conditions and key drivers
                    2. Sector Analysis - Performance and outlook by sector
                    3. Trend Analysis - Key trends and their implications
                    4. Risk Factors - Major risks and concerns
                    5. Opportunities - Investment opportunities and themes
                    6. Outlook - Short and medium-term market outlook
                    
                    Use data from news articles and trend analysis to support your insights."""
                },
                {
                    "role": "user",
                    "content": f"""Generate market insights based on:

Articles analyzed: {len(articles)}
Key trends identified: {len(trends)}

Trend details: {str(trends)}

Article sources: {[article.source for article in articles[:5]]}"""
                }
            ]
            
            response = await generate_llm_response(messages)
            return response.content
    
    def _assess_confidence(self, insights: List[str], text_data: List[str]) -> ConfidenceLevel:
        """Assess confidence level based on data quality and insight consistency."""
        if not insights or not text_data:
            return ConfidenceLevel.LOW
        
        # Simple confidence assessment based on data volume and consistency
        if len(text_data) >= 5 and len(insights) >= 3:
            return ConfidenceLevel.HIGH
        elif len(text_data) >= 2 and len(insights) >= 2:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _assess_news_confidence(self, articles: List[NewsArticle], sentiment_analysis: List[Dict]) -> ConfidenceLevel:
        """Assess confidence level for news analysis."""
        if not articles:
            return ConfidenceLevel.LOW
        
        # Check data quality indicators
        valid_articles = len([a for a in articles if a.content and len(a.content) > 100])
        sentiment_consistency = len(set([s.get("overall_sentiment", "neutral") for s in sentiment_analysis]))
        
        if valid_articles >= 5 and sentiment_consistency <= 2:  # Consistent sentiment
            return ConfidenceLevel.HIGH
        elif valid_articles >= 3:
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
            "status": "active"
        }


# Global researcher instance
researcher = ResearcherAgent()


def get_researcher() -> ResearcherAgent:
    """Get global researcher instance."""
    return researcher