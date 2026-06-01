"""
Simulation engine for xsim.

`SimulationEngine` turns the static snapshot in an `ExperimentState` into
a multi-round simulation. Each tick:

  1. For every agent, build a ranked feed of recent posts.
  2. Hand the feed to a pluggable `Behavior`.
  3. Materialize the behavior's actions into Engagements + new Posts.
  4. Record a `TickRecord` summarising what happened.

The deterministic behavior keeps things reproducible. An LLM-backed
behavior can be swapped in without changing the engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from xsim.core.behavior import (
    Behavior,
    DeterministicBehavior,
    materialize_actions,
)
from xsim.core.models import Agent, Engagement, Post
from xsim.core.simulation import generate_scenario_posts, rank_feed
from xsim.core.state import ExperimentState, TickRecord


@dataclass
class TickResult:
    """Public summary of one step of the engine."""

    tick: TickRecord
    new_posts: list[Post]


class SimulationEngine:
    """Drives the experiment forward tick by tick."""

    def __init__(
        self,
        state: ExperimentState,
        behavior: Behavior | None = None,
    ) -> None:
        self.state = state
        self.behavior: Behavior = behavior or DeterministicBehavior()
        self.behavior.reset(state.config.random_seed)

    # ---- Wiring -----------------------------------------------------------
    def set_behavior(self, behavior: Behavior) -> None:
        self.behavior = behavior
        self.behavior.reset(self.state.config.random_seed)

    def attach_state(self, state: ExperimentState) -> None:
        """Swap the underlying state (e.g. after a Reset Run)."""
        self.state = state
        self.behavior.reset(state.config.random_seed)

    # ---- Public actions ---------------------------------------------------
    def inject_scenario(self, description: str) -> list[Post]:
        """Inject a 'god mode' event and generate the first-wave reactions."""
        from xsim.core.models import Scenario  # local import: keep top tidy

        if not self.state.agents:
            raise ValueError("Cannot inject scenario: no agents in state.")

        scenario = Scenario(description=description)
        self.state.add_scenario(scenario)
        posts = generate_scenario_posts(
            self.state.agents, scenario, self.state.config
        )
        self.state.add_posts(posts)
        return posts

    def step(self) -> TickResult:
        """Run a single round of read → react → reply → repost → ignore."""
        if not self.state.agents:
            raise ValueError("Cannot step: no agents in state.")

        new_engagements: list[Engagement] = []
        new_posts: list[Post] = []
        index = self.state.current_tick + 1
        config = self.state.config

        for agent in self.state.agents:
            feed = rank_feed(
                agent,
                self.state.posts,
                config,
                limit=20,
                engagements=self.state.engagements,
            )
            actions = self.behavior.decide(agent, feed, self.state, config)
            if not actions:
                continue
            engagements, posts = materialize_actions(agent, actions, self.state)
            new_engagements.extend(engagements)
            new_posts.extend(posts)

        self.state.add_engagements(new_engagements)

        record = TickRecord(
            index=index,
            description="step",
            new_post_ids=[post.id for post in new_posts],
            new_engagement_ids=[
                engagement.post_id + ":" + engagement.action
                for engagement in new_engagements
                if engagement.post_id
            ],
            notes={
                "agents": len(self.state.agents),
                "engagements": len(new_engagements),
                "new_posts": len(new_posts),
            },
        )
        self.state.record_tick(record)
        return TickResult(tick=record, new_posts=new_posts)

    def run(self, steps: int) -> list[TickResult]:
        """Run `steps` ticks in sequence and return their results."""
        results: list[TickResult] = []
        for _ in range(max(0, int(steps))):
            results.append(self.step())
        return results

    # ---- Convenience for callers / tests ---------------------------------
    def populate_society(self, count: int | None = None) -> list[Agent]:
        from xsim.core.simulation import create_default_agents

        target = count if count is not None else self.state.config.num_agents
        agents = create_default_agents(target, self.state.config.random_seed)
        self.state.add_agents(agents)
        return agents
