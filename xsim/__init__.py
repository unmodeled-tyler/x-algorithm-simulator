"""
xsim — Controllable simulator of an X-style social platform.

Synthetic agents + god-mode scenario injection + fully tunable algorithm.

See `simulator/app.py` for the interactive UI (Streamlit).
"""

from .core.models import Agent, Post, SimulationConfig

__version__ = "0.1.0"

# LLM client is available but not imported at package level
# to avoid requiring ollama/openai until the user actually needs them.
# from .llm import get_llm_client, LLMConfig

__all__ = ["Agent", "Post", "SimulationConfig"]
