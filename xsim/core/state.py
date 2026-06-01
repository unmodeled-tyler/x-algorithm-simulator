"""
Single source of truth for an in-flight xsim experiment.

`ExperimentState` replaces the loose lists previously kept in
`st.session_state`. It owns the agents, posts, scenarios, engagements,
tick history, and a snapshot of the configuration that produced it.
Downstream consumers (the Streamlit UI, the simulation engine, tests)
all read and mutate state through this object.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from xsim.core.models import (
    Agent,
    Engagement,
    Post,
    Scenario,
    SimulationConfig,
)


@dataclass
class TickRecord:
    """One step of the simulation: who acted, what was created, what was logged."""

    index: int
    description: str
    new_post_ids: list[str] = field(default_factory=list)
    new_engagement_ids: list[str] = field(default_factory=list)
    notes: dict[str, int] = field(default_factory=dict)


@dataclass
class ExperimentState:
    """A complete experiment in memory."""

    config: SimulationConfig
    agents: list[Agent] = field(default_factory=list)
    posts: list[Post] = field(default_factory=list)
    scenarios: list[Scenario] = field(default_factory=list)
    engagements: list[Engagement] = field(default_factory=list)
    ticks: list[TickRecord] = field(default_factory=list)
    current_tick: int = 0
    # Last reaction wave size, exposed for the UI's metric tile.
    last_reaction_count: int = 0

    # ---- Lookup helpers ---------------------------------------------------
    def agent_by_id(self, agent_id: str) -> Agent | None:
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def post_by_id(self, post_id: str) -> Post | None:
        for post in self.posts:
            if post.id == post_id:
                return post
        return None

    def agent_index(self) -> dict[str, Agent]:
        return {agent.id: agent for agent in self.agents}

    def post_index(self) -> dict[str, Post]:
        return {post.id: post for post in self.posts}

    # ---- Collections ------------------------------------------------------
    def iter_posts(self) -> Iterator[Post]:
        return iter(self.posts)

    def iter_engagements(self) -> Iterator[Engagement]:
        return iter(self.engagements)

    def add_agents(self, agents: Iterable[Agent]) -> None:
        self.agents.extend(agents)

    def add_posts(self, posts: Iterable[Post]) -> None:
        new_posts = list(posts)
        self.posts.extend(new_posts)
        if new_posts:
            self.last_reaction_count = len(new_posts)

    def add_scenario(self, scenario: Scenario) -> None:
        self.scenarios.append(scenario)

    def add_engagements(self, engagements: Iterable[Engagement]) -> None:
        self.engagements.extend(engagements)

    def record_tick(self, record: TickRecord) -> None:
        self.ticks.append(record)
        self.current_tick = record.index

    # ---- Reset / clear ---------------------------------------------------
    def reset_society(self) -> None:
        """Drop agents + everything derived, keep config."""
        self.agents.clear()
        self.posts.clear()
        self.engagements.clear()
        self.scenarios.clear()
        self.ticks.clear()
        self.current_tick = 0
        self.last_reaction_count = 0

    def clear_posts(self) -> None:
        self.posts.clear()
        self.engagements.clear()
        self.ticks.clear()
        self.current_tick = 0
        self.last_reaction_count = 0

    # ---- Analytics -------------------------------------------------------
    def topic_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for post in self.posts:
            counts.update(post.topic_tags)
        return counts

    def engagement_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for engagement in self.engagements:
            counts[engagement.action] += 1
        return counts

    def snapshot(self) -> dict[str, object]:
        """Lightweight JSON-friendly summary for the UI and tests."""
        return {
            "config": self.config,
            "agents": [agent.id for agent in self.agents],
            "posts": [post.id for post in self.posts],
            "scenarios": [scenario.id for scenario in self.scenarios],
            "engagements": len(self.engagements),
            "ticks": [record.index for record in self.ticks],
            "current_tick": self.current_tick,
        }
