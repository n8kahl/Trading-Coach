"""Utilities for working with the OpenAI Agents Python SDK.

This module instantiates a reusable trading coach agent using the
`openai-agents` package.  The agent is designed to produce two kinds
of output:

1. Conversational guidance rendered by ChatKit.
2. Optional widget payloads (JSON objects under a `widget` key) that
   the frontend can use to render trade ideas, plans, or charts.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Dict, Optional

from agents import Agent, Runner


TRADING_INSTRUCTIONS = """
You are an elite intraday options trading coach.

Primary goals:
- Provide concise, actionable guidance tailored to the user's question.
- If you recommend a trade or structured plan, emit a JSON object **on a new
  line by itself** describing the widget to render. Example shapes:

  {"widget": {"type": "trade_proposal", "symbol": "TSLA", "strategy": "...",
              "entry": "...", "stop": "...", "target": "...", "rationale": "..."}}

  {"widget": {"type": "trading_plan", "title": "TSLA Breakout Plan",
              "steps": ["Wait for premarket high reclaim", "..."]}}

  {"widget": {"type": "chart", "symbol": "TSLA", "interval": "5"}}

- Only emit JSON when you have concrete information; otherwise respond with
  plain language.
- Never wrap JSON in backticks or explanatory text—just the raw object.
"""


@lru_cache()
def get_trading_agent() -> Agent:
    """Return a cached Agent instance configured for the trading assistant."""
    return Agent(
        name="Trading Coach",
        instructions=TRADING_INSTRUCTIONS.strip(),
    )


async def run_agent_turn(
    prompt: str, conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """Execute a single turn with the trading agent.

    Args:
        prompt: The end-user message.
        conversation_id: Optional conversation identifier to maintain
            state across turns (not persisted by default).

    Returns:
        A dictionary with the model's final output plus raw metadata.
    """
    agent = get_trading_agent()
    result = await Runner.run(
        agent,
        prompt,
        conversation_id=conversation_id,
    )
    return {
        "output": result.final_output,
        "conversation_id": result.conversation_id,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        }
        if result.usage
        else None,
    }


def run_agent_turn_sync(prompt: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """Synchronous wrapper used by FastAPI dependency injection when needed."""
    return asyncio.run(run_agent_turn(prompt, conversation_id=conversation_id))
