"""
Core data models for the xsim platform simulator.

These are intentionally simple and simulation-oriented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4


ActionType = Literal[
    "like", "repost", "reply", "quote", "dwell", "click",
    "not_interested", "block", "mute"
]


@dataclass
class Post:
    """A post in the simulated platform."""
    author_id: str
    text: str
    author_username: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    reply_to_id: str | None = None
    quote_of_id: str | None = None
    # Simple metadata for now
    topic_tags: list[str] = field(default_factory=list)

    def is_reply(self) -> bool:
        return self.reply_to_id is not None


@dataclass
class Agent:
    """
    A synthetic user / agent on the platform.

    The 'persona' field is the key: it becomes the system prompt + persistent
    character description fed to the LLM.
    """
    username: str
    persona: str  # The rich character description / system prompt
    id: str = field(default_factory=lambda: str(uuid4()))
    interests: list[str] = field(default_factory=list)
    following_ids: set[str] = field(default_factory=set)
    # Short-term memory of recent activity (post ids, engagements, etc.)
    memory: list[str] = field(default_factory=list)
    # What the agent has seen and reacted to recently
    recent_engagements: list[Engagement] = field(default_factory=list)

    def describe(self) -> str:
        return f"@{self.username} — {self.persona[:120]}..."


@dataclass
class Engagement:
    """An action an agent took on a post."""
    agent_id: str
    post_id: str
    action: ActionType
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    # Optional free text (for replies)
    text: str | None = None


@dataclass
class Scenario:
    """A 'god mode' event injected into the simulation."""
    description: str
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    # Optional structured fields later (e.g. economic impact, political leaning, etc.)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """Controllable parameters for the current run."""
    num_agents: int = 20
    model_provider: str = "ollama"          # "ollama" | "openai_compatible"
    model_name: str = "qwen2.5:7b"          # or "gpt-4o-mini", "llama3", etc.
    api_base_url: str | None = None         # For OpenAI-compatible endpoints
    api_key: str | None = None

    # Algorithm knobs (these will drive the ranker)
    in_network_weight: float = 1.0
    discovery_weight: float = 0.8
    author_diversity_penalty: float = 0.65
    reply_boost: float = 1.8
    negative_action_penalty: float = 2.5

    temperature: float = 0.7
    random_seed: int | None = 42

    # Simulation pacing
    max_posts_per_agent_per_step: int = 1


@dataclass
class FeedItem:
    """A post as presented in a personalized feed."""
    post: Post
    score: float
    reason: str | None = None   # e.g. "Strong match with your recent interests"
