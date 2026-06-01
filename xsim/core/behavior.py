"""
Pluggable agent behavior for xsim.

A `Behavior` decides, for one agent and the current world state, what the
agent should *do* during a simulation tick: ignore the feed, like, repost,
reply, or compose an original post. The deterministic implementation is
the local-only baseline; the LLM-backed implementation is an optional
upgrade that calls into `xsim.llm`.

`xsim.core.engine.SimulationEngine` accepts any object satisfying the
`Behavior` protocol, so swapping modes is a one-line change.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, cast

from xsim.core.models import (
    ActionType,
    Agent,
    Engagement,
    FeedItem,
    Post,
    SimulationConfig,
)
from xsim.core.state import ExperimentState


@dataclass
class Action:
    """A single decision an agent made during a tick."""

    agent_id: str
    action: ActionType
    post_id: str | None = None
    text: str | None = None
    new_post: Post | None = None
    notes: dict[str, str] = field(default_factory=dict)


class Behavior(Protocol):
    """Strategy interface: produce a list of actions for one agent per tick."""

    def reset(self, seed: int | None = None) -> None: ...

    def decide(
        self,
        agent: Agent,
        feed: list[FeedItem],
        state: ExperimentState,
        config: SimulationConfig,
    ) -> list[Action]: ...


# ---------------------------------------------------------------------------
# Deterministic baseline
# ---------------------------------------------------------------------------

DETERMINISTIC_REPLY_TEMPLATES: tuple[str, ...] = (
    "This connects to {interest}. {angle}",
    "Worth noting the {interest} angle here. {angle}",
    "Reading this through a {interest} lens: {angle}",
    "The {interest} story is the one I'd watch. {angle}",
)

DETERMINISTIC_ANGLES: tuple[str, ...] = (
    "The second-order effects will be louder than the headline.",
    "The local impact looks very different from the national take.",
    "The incentives underneath this are doing most of the work.",
    "I want to see who benefits and who pays before reacting.",
)


def _topic_for(agent: Agent, post: Post) -> str:
    overlap = sorted(set(agent.interests).intersection(post.topic_tags))
    if overlap:
        return overlap[0]
    if post.topic_tags:
        return post.topic_tags[0]
    return agent.interests[0] if agent.interests else "general"


class DeterministicBehavior:
    """Cheap, fully local, fully reproducible agent behavior."""

    def __init__(self) -> None:
        self._rng = random.Random()

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def decide(
        self,
        agent: Agent,
        feed: list[FeedItem],
        state: ExperimentState,
        config: SimulationConfig,
    ) -> list[Action]:
        actions: list[Action] = []

        if not feed:
            return actions

        # Pick the top-scoring item the agent is willing to engage with.
        for item in feed:
            overlap = len(set(agent.interests).intersection(item.post.topic_tags))
            threshold = 0.25
            if self._rng.random() < max(0.0, threshold - 0.05 * overlap):
                continue

            roll = self._rng.random()
            if roll < 0.5:
                actions.append(
                    Action(
                        agent_id=agent.id,
                        action="like",
                        post_id=item.post.id,
                        notes={"trigger": "top-item", "score": f"{item.score:.3f}"},
                    )
                )
                if overlap and self._rng.random() < 0.4:
                    actions.append(
                        Action(
                            agent_id=agent.id,
                            action="repost",
                            post_id=item.post.id,
                            notes={"trigger": "amplify", "score": f"{item.score:.3f}"},
                        )
                    )
                if overlap and self._rng.random() < 0.25:
                    actions.append(
                        self._make_reply(agent, item.post)
                    )
                break
            if roll < 0.7:
                actions.append(
                    Action(
                        agent_id=agent.id,
                        action="dwell",
                        post_id=item.post.id,
                        notes={"trigger": "browse"},
                    )
                )
                break
            # else: silently ignore and move on
            break

        # Occasionally, an agent will compose an original post on their own.
        if (
            config.max_posts_per_agent_per_step > 0
            and self._rng.random() < 0.18
            and (not actions or actions[0].action in {"like", "dwell"})
        ):
            actions.append(self._make_original(agent, state, config))

        return actions

    # ---- Helpers ----------------------------------------------------------
    def _make_reply(self, agent: Agent, post: Post) -> Action:
        interest = _topic_for(agent, post)
        text = self._rng.choice(DETERMINISTIC_REPLY_TEMPLATES).format(
            interest=interest,
            angle=self._rng.choice(DETERMINISTIC_ANGLES),
        )
        return Action(
            agent_id=agent.id,
            action="reply",
            post_id=post.id,
            text=text,
            notes={"trigger": "reply", "topic": interest},
        )

    def _make_original(
        self,
        agent: Agent,
        state: ExperimentState,
        config: SimulationConfig,
    ) -> Action:
        interest = agent.interests[0] if agent.interests else "general"
        topic_tags = [interest]
        if state.scenarios:
            inferred = _topic_for(agent, _scenario_post_stub(state, interest))
            topic_tags = sorted({interest, inferred})
        text = (
            f"Following this story for the {interest} angle. "
            f"{self._rng.choice(DETERMINISTIC_ANGLES)}"
        )
        post = Post(
            author_id=agent.id,
            author_username=agent.username,
            text=text,
            topic_tags=topic_tags,
        )
        return Action(
            agent_id=agent.id,
            action="reply",  # recorded as a self-authored post
            post_id=None,
            text=text,
            new_post=post,
            notes={"trigger": "original", "topic": interest},
        )


def _scenario_post_stub(state: ExperimentState, interest: str) -> Post:
    """Tiny synthetic Post used to recover a topic tag from the latest scenario."""
    if state.scenarios and state.scenarios[-1].description:
        from xsim.core.simulation import infer_topics  # local import: avoid cycles

        topics = infer_topics(state.scenarios[-1].description)
    else:
        topics = [interest]
    return Post(author_id="", text="", topic_tags=topics)


# ---------------------------------------------------------------------------
# Engagement / post materialization
# ---------------------------------------------------------------------------

def materialize_actions(
    agent: Agent,
    actions: list[Action],
    state: ExperimentState,
) -> tuple[list[Engagement], list[Post]]:
    """Turn behavior outputs into concrete Engagement + Post objects.

    Appends new posts to `state.posts` so subsequent ticks can see them.
    Engagement records are returned for the caller to append to state.
    """
    from datetime import UTC, datetime  # local import keeps top of file tidy

    engagements: list[Engagement] = []
    new_posts: list[Post] = []

    for action in actions:
        if action.new_post is not None:
            new_posts.append(action.new_post)
            state.add_posts([action.new_post])
            # Self-authored posts also generate a "reply"-style engagement row
            # for analytics; we tag it differently via the notes.
            engagements.append(
                Engagement(
                    agent_id=agent.id,
                    post_id=action.new_post.id,
                    action="reply" if action.action == "reply" else action.action,
                    timestamp=datetime.now(UTC),
                    text=action.text,
                )
            )
            continue

        if action.post_id is None:
            continue
        if action.action in {"like", "repost", "reply", "quote", "dwell", "click"}:
            engagement = Engagement(
                agent_id=agent.id,
                post_id=action.post_id,
                action=action.action,
                timestamp=datetime.now(UTC),
                text=action.text,
            )
            engagements.append(engagement)
            agent.recent_engagements.append(engagement)
            agent.memory.append(action.post_id)

            if action.action == "reply" and action.text:
                reply = Post(
                    author_id=agent.id,
                    author_username=agent.username,
                    text=action.text,
                    reply_to_id=action.post_id,
                    topic_tags=[],
                )
                new_posts.append(reply)
                state.add_posts([reply])

    return engagements, new_posts


# ---------------------------------------------------------------------------
# Optional LLM-backed behavior
# ---------------------------------------------------------------------------


class LLMAgentBehavior:
    """Behavior that delegates posting/engagement decisions to an LLM.

    Falls back to :class:`DeterministicBehavior` if the LLM client raises
    any exception, so a misconfigured API key never breaks a local run.
    The LLM is imported lazily so the deterministic baseline keeps working
    in environments where `ollama`/`openai` aren't installed.
    """

    def __init__(self, llm_config: object | None = None) -> None:
        self.llm_config = llm_config
        self._client: object | None = None
        self._fallback = DeterministicBehavior()
        self._enabled = False

    def _ensure_client(self) -> None:
        if self._client is not None or not self.llm_config:
            return
        try:
            from xsim.llm import LLMConfig, get_llm_client

            self._client = get_llm_client(cast(LLMConfig, self.llm_config))
            self._enabled = True
        except Exception:
            self._client = None
            self._enabled = False

    def reset(self, seed: int | None = None) -> None:
        self._fallback.reset(seed)

    def decide(
        self,
        agent: Agent,
        feed: list[FeedItem],
        state: ExperimentState,
        config: SimulationConfig,
    ) -> list[Action]:
        self._ensure_client()
        if not self._enabled or self._client is None or not feed:
            return self._fallback.decide(agent, feed, state, config)

        try:
            return self._llm_decide(agent, feed, state, config)
        except Exception:
            # Never let an LLM failure break the simulation.
            return self._fallback.decide(agent, feed, state, config)

    def _llm_decide(
        self,
        agent: Agent,
        feed: list[FeedItem],
        state: ExperimentState,
        config: SimulationConfig,
    ) -> list[Action]:
        assert self._client is not None  # for type checkers
        top = feed[0]
        system = (
            "You are role-playing an X-style social media user. "
            f"Your persona: {agent.persona} "
            f"Your interests: {', '.join(agent.interests) or 'general'}. "
            "Reply with a single short JSON object describing one action."
        )
        user = (
            "Top post in your feed:\n"
            f"author: @{top.post.author_username or top.post.author_id}\n"
            f"text: {top.post.text}\n"
            f"score: {top.score:.3f}\n\n"
            "Respond with JSON like "
            '{"action": "like|repost|reply|ignore", '
            '"text": "optional short reply"}'
        )

        raw = self._client.chat(system, user)  # type: ignore[attr-defined]
        action_name, text = _parse_llm_action(raw)

        if action_name == "ignore":
            return []
        if action_name == "reply" and text:
            return [
                Action(
                    agent_id=agent.id,
                    action="reply",
                    post_id=top.post.id,
                    text=text,
                    notes={"trigger": "llm"},
                )
            ]
        return [
            Action(
                agent_id=agent.id,
                action=action_name,  # type: ignore[arg-type]
                post_id=top.post.id,
                notes={"trigger": "llm"},
            )
        ]


def _parse_llm_action(raw: str) -> tuple[ActionType | str, str | None]:
    """Best-effort parse of an LLM JSON-ish action blob."""
    import json
    import re

    text = raw.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return "ignore", None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "ignore", None
    action = str(data.get("action", "ignore")).lower().strip()
    body = data.get("text")
    if action not in {"like", "repost", "reply", "quote", "dwell", "click", "ignore"}:
        return "ignore", None
    return action, str(body) if body is not None else None
