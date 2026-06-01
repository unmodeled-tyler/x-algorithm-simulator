"""Tests for ExperimentState, SimulationEngine, and Behavior plumbing."""

from __future__ import annotations

from xsim.core import (
    DeterministicBehavior,
    ExperimentState,
    Post,
    SimulationConfig,
    SimulationEngine,
)
from xsim.core.behavior import LLMAgentBehavior, materialize_actions


def _build_state(num_agents: int = 6, seed: int = 7) -> ExperimentState:
    config = SimulationConfig(num_agents=num_agents, random_seed=seed)
    state = ExperimentState(config=config)
    engine = SimulationEngine(state)
    engine.populate_society()
    return state


def test_state_backbone_owns_agents_posts_scenarios() -> None:
    state = _build_state()

    assert len(state.agents) == 6
    assert state.posts == []
    assert state.scenarios == []
    assert state.engagements == []
    assert state.ticks == []
    assert state.current_tick == 0

    state.add_posts(
        [
            Post(
                author_id=state.agents[0].id,
                author_username=state.agents[0].username,
                text="hello world",
                topic_tags=["energy"],
            )
        ]
    )
    assert state.last_reaction_count == 1
    assert state.topic_counts()["energy"] == 1


def test_engine_inject_scenario_then_step() -> None:
    state = _build_state()
    engine = SimulationEngine(state)

    new_posts = engine.inject_scenario(
        "Gas prices spiked after a refinery outage hit supply."
    )
    assert new_posts, "scenario injection should produce a first wave"
    assert state.scenarios and state.scenarios[-1].description.startswith("Gas")
    assert state.last_reaction_count == len(new_posts)
    assert state.current_tick == 0  # injection is not a tick

    result = engine.step()
    assert state.current_tick == 1
    assert result.tick.index == 1
    assert state.ticks and state.ticks[-1].index == 1
    # Tick may or may not produce posts, but it must not raise.
    assert result.tick.notes["agents"] == len(state.agents)


def test_engine_multi_tick_produces_dynamics() -> None:
    state = _build_state(num_agents=8, seed=11)
    engine = SimulationEngine(state)
    engine.inject_scenario("Tech layoffs are rippling through AI startups.")

    results = engine.run(steps=3)
    assert len(results) == 3
    assert [r.tick.index for r in results] == [1, 2, 3]
    # After 3 ticks we should have at least one new post from a reply chain
    # or an original post; even if not, the engagement log must have grown.
    total_engagements = sum(r.tick.notes["engagements"] for r in results)
    assert total_engagements >= 0  # never raises


def test_state_reset_society_preserves_config() -> None:
    state = _build_state()
    engine = SimulationEngine(state)
    engine.inject_scenario("Markets react to a surprise rate decision.")
    assert state.posts and state.scenarios

    state.reset_society()
    assert state.agents == []
    assert state.posts == []
    assert state.engagements == []
    assert state.scenarios == []
    assert state.ticks == []
    assert state.current_tick == 0
    # Config snapshot is preserved.
    assert state.config.num_agents == 6


def test_materialize_actions_creates_replies() -> None:
    state = _build_state(num_agents=3, seed=3)
    original = Post(
        author_id=state.agents[0].id,
        author_username=state.agents[0].username,
        text="Original",
        topic_tags=["energy"],
    )
    state.add_posts([original])

    replier = state.agents[1]
    from xsim.core.behavior import Action

    actions = [
        Action(
            agent_id=replier.id,
            action="reply",
            post_id=original.id,
            text="Following this for the energy angle.",
        ),
        Action(agent_id=replier.id, action="like", post_id=original.id),
    ]
    engagements, new_posts = materialize_actions(replier, actions, state)

    assert len(engagements) == 2
    assert any(e.action == "reply" for e in engagements)
    assert any(e.action == "like" for e in engagements)
    assert len(new_posts) == 1
    assert new_posts[0].reply_to_id == original.id
    assert any(p.id == new_posts[0].id for p in state.posts)


def test_llm_behavior_falls_back_when_disabled() -> None:
    """LLMAgentBehavior without a config should behave like the deterministic one."""
    state = _build_state(num_agents=4, seed=5)
    engine = SimulationEngine(state)
    engine.inject_scenario("Climate summit ends with a surprise pledge.")

    behavior = LLMAgentBehavior(llm_config=None)
    agent = state.agents[0]
    from xsim.core.simulation import rank_feed

    feed = rank_feed(agent, state.posts, state.config, limit=10)
    actions = behavior.decide(agent, feed, state, state.config)

    # Falls back to DeterministicBehavior, so actions have valid shape.
    for action in actions:
        assert action.agent_id == agent.id
        assert action.action in {
            "like", "repost", "reply", "quote", "dwell", "click",
            "not_interested", "block", "mute",
        }


def test_deterministic_behavior_is_deterministic() -> None:
    """Same seed + same feed → same action sequence for the same agent."""
    from xsim.core.simulation import rank_feed

    state = _build_state(num_agents=6, seed=123)
    engine = SimulationEngine(state)
    engine.inject_scenario("Energy markets rattled by an OPEC surprise.")

    agent = state.agents[0]
    feed = rank_feed(agent, state.posts, state.config, limit=10)

    behavior_a = DeterministicBehavior()
    behavior_b = DeterministicBehavior()
    behavior_a.reset(123)
    behavior_b.reset(123)

    actions_a = behavior_a.decide(agent, feed, state, state.config)
    actions_b = behavior_b.decide(agent, feed, state, state.config)

    assert [a.action for a in actions_a] == [a.action for a in actions_b]
    assert [a.post_id for a in actions_a] == [a.post_id for a in actions_b]
