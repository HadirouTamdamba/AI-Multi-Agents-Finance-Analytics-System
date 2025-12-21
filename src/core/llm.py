"""
LLM interface for AI Finance Agent Team.
Supports Anthropic Claude, OpenAI, and mock responses for development.
"""

import json
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import anthropic
import openai
from pydantic import BaseModel, Field

from .config import get_settings
from .logging import get_logger


logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    response_time: float = 0.0


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-sonnet-20240229"
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic Claude."""
        start_time = time.time()
        
        try:
            # Convert messages to Anthropic format
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg["content"])
            
            user_content = "\n\n".join(user_messages)
            
            response = self.client.messages.create(
                model=model or self.model,
                max_tokens=max_tokens or 4000,
                temperature=temperature,
                system=system_message,
                messages=[{"role": "user", "content": user_content}]
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.content[0].text,
                model=model or self.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                finish_reason=response.stop_reason,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4-turbo-preview"
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI GPT."""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 4000
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=model or self.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=response.choices[0].finish_reason,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class MockProvider(LLMProvider):
    """Mock LLM provider for development and testing."""
    
    def __init__(self):
        self.model = "mock-claude-3-sonnet"
        self.responses = {
            "researcher": [
                "Based on the financial data analysis, I've identified several key trends...",
                "The market indicators suggest a bullish sentiment with strong fundamentals...",
                "Recent news indicates significant volatility in the technology sector...",
                "Economic indicators point to potential growth opportunities in emerging markets..."
            ],
            "analyst": [
                "The quantitative analysis reveals strong performance metrics with ROE of 15.2%...",
                "Technical indicators show a positive trend with RSI at 65 and MACD crossing above...",
                "Financial ratios indicate healthy liquidity with current ratio of 2.1...",
                "The time series analysis suggests seasonal patterns in Q4 performance..."
            ],
            "risk_modeler": [
                "Risk assessment indicates moderate exposure with VaR of 2.3% at 95% confidence...",
                "Scenario analysis shows potential downside of 15% in bear case scenario...",
                "Stress testing reveals resilience under adverse market conditions...",
                "Portfolio diversification reduces correlation risk by 40%..."
            ],
            "writer": [
                "Executive Summary: The analysis reveals strong fundamentals with growth potential...",
                "Key Recommendations: 1) Increase exposure to technology stocks 2) Hedge against inflation...",
                "Risk Assessment: Moderate risk profile with well-diversified portfolio...",
                "Conclusion: The investment thesis remains strong with positive outlook..."
            ],
            "default": [
                "This is a mock response for development and testing purposes.",
                "The system is working correctly with mock LLM responses.",
                "In production, this would be replaced with actual LLM API calls.",
                "Mock responses ensure the system works without API keys."
            ]
        }
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate mock response."""
        start_time = time.time()
        
        # Simulate API delay
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # Determine response type based on message content
        message_content = " ".join([msg["content"] for msg in messages]).lower()
        
        if "research" in message_content or "news" in message_content:
            response_type = "researcher"
        elif "analyze" in message_content or "data" in message_content:
            response_type = "analyst"
        elif "risk" in message_content or "scenario" in message_content:
            response_type = "risk_modeler"
        elif "report" in message_content or "summary" in message_content:
            response_type = "writer"
        else:
            response_type = "default"
        
        # Select random response
        responses = self.responses[response_type]
        content = random.choice(responses)
        
        # Add some context from the input
        if messages:
            last_message = messages[-1]["content"][:100]
            content = f"Regarding: {last_message}...\n\n{content}"
        
        response_time = time.time() - start_time
        
        return LLMResponse(
            content=content,
            model=model or self.model,
            usage={
                "input_tokens": len(" ".join([msg["content"] for msg in messages])) // 4,
                "output_tokens": len(content) // 4
            },
            finish_reason="stop",
            response_time=response_time
        )


class LLMManager:
    """Manages LLM providers and routing."""
    
    def __init__(self):
        self.settings = get_settings()
        self.providers: Dict[str, LLMProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers."""
        if self.settings.has_anthropic_key:
            try:
                self.providers["anthropic"] = AnthropicProvider(self.settings.anthropic_api_key)
                logger.info("Anthropic provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic provider: {e}")
        
        if self.settings.has_openai_key:
            try:
                self.providers["openai"] = OpenAIProvider(self.settings.openai_api_key)
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI provider: {e}")
        
        # Always add mock provider as fallback
        self.providers["mock"] = MockProvider()
        logger.info("Mock provider initialized")
    
    def get_provider(self, provider_name: Optional[str] = None) -> LLMProvider:
        """Get LLM provider by name or default."""
        if provider_name and provider_name in self.providers:
            return self.providers[provider_name]
        
        # Default priority: Anthropic > OpenAI > Mock
        if "anthropic" in self.providers and not self.settings.use_mock_llm:
            return self.providers["anthropic"]
        elif "openai" in self.providers and not self.settings.use_mock_llm:
            return self.providers["openai"]
        else:
            return self.providers["mock"]
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using specified or default provider."""
        provider_instance = self.get_provider(provider)
        
        logger.info(f"Using LLM provider: {type(provider_instance).__name__}")
        
        try:
            response = await provider_instance.generate(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            logger.info(f"LLM response generated in {response.response_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback to mock provider if available
            if provider_instance != self.providers.get("mock"):
                logger.info("Falling back to mock provider")
                return await self.providers["mock"].generate(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            raise


# Global LLM manager instance
llm_manager = LLMManager()


def get_llm_manager() -> LLMManager:
    """Get global LLM manager instance."""
    return llm_manager


async def generate_llm_response(
    messages: List[Dict[str, str]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> LLMResponse:
    """Convenience function for generating LLM responses."""
    return await llm_manager.generate(
        messages=messages,
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )