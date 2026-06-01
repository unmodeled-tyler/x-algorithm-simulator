"""
Single source of truth for an in-flight xsim experiment.

`ExperimentState` replaces the loose lists previously kept in
`st.session_state`. It owns the agents, posts, scenarios, engagements,
tick history, and a snapshot of the configuration that produced it.
Downstream consumers (the Streamlit UI, the simulation engine, tests)
all read and mutate state through this object.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
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

    # ---- Persistence -----------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """Serialize the complete experiment to a JSON-friendly dictionary."""
        config = asdict(self.config)
        config["api_key"] = None
        return {
            "schema_version": 1,
            "config": config,
            "agents": [_agent_to_dict(agent) for agent in self.agents],
            "posts": [_post_to_dict(post) for post in self.posts],
            "scenarios": [_scenario_to_dict(scenario) for scenario in self.scenarios],
            "engagements": [
                _engagement_to_dict(engagement) for engagement in self.engagements
            ],
            "ticks": [asdict(tick) for tick in self.ticks],
            "current_tick": self.current_tick,
            "last_reaction_count": self.last_reaction_count,
        }

    def to_json(self) -> str:
        """Serialize the complete experiment to pretty JSON."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ExperimentState":
        """Rehydrate an ExperimentState from :meth:`to_dict` output."""
        config = _config_from_dict(_expect_dict(payload.get("config")))
        state = cls(config=config)
        state.agents = [_agent_from_dict(item) for item in _expect_list(payload.get("agents"))]
        state.posts = [_post_from_dict(item) for item in _expect_list(payload.get("posts"))]
        state.scenarios = [
            _scenario_from_dict(item) for item in _expect_list(payload.get("scenarios"))
        ]
        state.engagements = [
            _engagement_from_dict(item) for item in _expect_list(payload.get("engagements"))
        ]
        state.ticks = [_tick_from_dict(item) for item in _expect_list(payload.get("ticks"))]
        state.current_tick = _int_from_json(payload.get("current_tick"), 0)
        state.last_reaction_count = _int_from_json(payload.get("last_reaction_count"), 0)
        return state

    @classmethod
    def from_json(cls, raw: str) -> "ExperimentState":
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Experiment JSON must contain an object at the top level.")
        return cls.from_dict(payload)


def _config_from_dict(data: dict[str, object]) -> SimulationConfig:
    return SimulationConfig(
        num_agents=_int_from_json(data.get("num_agents"), 20),
        model_provider=str(data.get("model_provider") or "ollama"),
        model_name=str(data.get("model_name") or "qwen2.5:7b"),
        api_base_url=_optional_str(data.get("api_base_url")),
        api_key=_optional_str(data.get("api_key")),
        in_network_weight=_float_from_json(data.get("in_network_weight"), 1.0),
        discovery_weight=_float_from_json(data.get("discovery_weight"), 0.8),
        author_diversity_penalty=_float_from_json(
            data.get("author_diversity_penalty"), 0.65
        ),
        reply_boost=_float_from_json(data.get("reply_boost"), 1.8),
        negative_action_penalty=_float_from_json(data.get("negative_action_penalty"), 2.5),
        topic_match_weight=_float_from_json(data.get("topic_match_weight"), 0.85),
        recency_weight=_float_from_json(data.get("recency_weight"), 0.35),
        social_proof_weight=_float_from_json(data.get("social_proof_weight"), 0.22),
        feed_diversity_window=_int_from_json(data.get("feed_diversity_window"), 4),
        temperature=_float_from_json(data.get("temperature"), 0.7),
        random_seed=_optional_int(data.get("random_seed"), 42),
        max_posts_per_agent_per_step=_int_from_json(
            data.get("max_posts_per_agent_per_step"), 1
        ),
    )


def _agent_to_dict(agent: Agent) -> dict[str, object]:
    data = asdict(agent)
    data["following_ids"] = sorted(agent.following_ids)
    data["recent_engagements"] = [
        _engagement_to_dict(engagement) for engagement in agent.recent_engagements
    ]
    return data


def _agent_from_dict(data: object) -> Agent:
    payload = _expect_dict(data)
    agent = Agent(
        username=str(payload["username"]),
        persona=str(payload["persona"]),
        id=str(payload["id"]),
        interests=[str(item) for item in _expect_list(payload.get("interests"))],
        following_ids={str(item) for item in _expect_list(payload.get("following_ids"))},
        memory=[str(item) for item in _expect_list(payload.get("memory"))],
    )
    agent.recent_engagements = [
        _engagement_from_dict(item)
        for item in _expect_list(payload.get("recent_engagements"))
    ]
    return agent


def _post_to_dict(post: Post) -> dict[str, object]:
    data = asdict(post)
    data["timestamp"] = _datetime_to_json(post.timestamp)
    return data


def _post_from_dict(data: object) -> Post:
    payload = _expect_dict(data)
    return Post(
        author_id=str(payload["author_id"]),
        text=str(payload["text"]),
        author_username=(
            str(payload["author_username"]) if payload.get("author_username") else None
        ),
        id=str(payload["id"]),
        timestamp=_datetime_from_json(payload.get("timestamp")),
        reply_to_id=str(payload["reply_to_id"]) if payload.get("reply_to_id") else None,
        quote_of_id=str(payload["quote_of_id"]) if payload.get("quote_of_id") else None,
        topic_tags=[str(item) for item in _expect_list(payload.get("topic_tags"))],
    )


def _scenario_to_dict(scenario: Scenario) -> dict[str, object]:
    data = asdict(scenario)
    data["timestamp"] = _datetime_to_json(scenario.timestamp)
    return data


def _scenario_from_dict(data: object) -> Scenario:
    payload = _expect_dict(data)
    return Scenario(
        description=str(payload["description"]),
        id=str(payload["id"]),
        timestamp=_datetime_from_json(payload.get("timestamp")),
        metadata=dict(_expect_dict(payload.get("metadata"))),
    )


def _engagement_to_dict(engagement: Engagement) -> dict[str, object]:
    data = asdict(engagement)
    data["timestamp"] = _datetime_to_json(engagement.timestamp)
    return data


def _engagement_from_dict(data: object) -> Engagement:
    payload = _expect_dict(data)
    return Engagement(
        agent_id=str(payload["agent_id"]),
        post_id=str(payload["post_id"]),
        action=str(payload["action"]),  # type: ignore[arg-type]
        timestamp=_datetime_from_json(payload.get("timestamp")),
        text=str(payload["text"]) if payload.get("text") else None,
    )


def _tick_from_dict(data: object) -> TickRecord:
    payload = _expect_dict(data)
    return TickRecord(
        index=_int_from_json(payload.get("index"), 0),
        description=str(payload["description"]),
        new_post_ids=[str(item) for item in _expect_list(payload.get("new_post_ids"))],
        new_engagement_ids=[
            str(item) for item in _expect_list(payload.get("new_engagement_ids"))
        ],
        notes={
            str(key): _int_from_json(value, 0)
            for key, value in _expect_dict(payload.get("notes")).items()
        },
    )


def _datetime_to_json(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _datetime_from_json(value: object) -> datetime:
    if not value:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _expect_dict(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Expected object in experiment JSON.")
    return value


def _expect_list(value: object) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected list in experiment JSON.")
    return value


def _int_from_json(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value))


def _optional_int(value: object, default: int | None) -> int | None:
    if value is None:
        return default
    return _int_from_json(value, 0)


def _float_from_json(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return float(str(value))


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None
