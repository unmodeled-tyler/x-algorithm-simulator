from xsim.core import (
    ExperimentState,
    ModelRoleConfig,
    SimulationConfig,
    SimulationEngine,
    feed_diversity,
    reach_summary,
    tick_activity_rows,
    top_amplifiers,
    topic_author_matrix,
    topic_spread_by_tick,
)


def _run_state() -> ExperimentState:
    state = ExperimentState(config=SimulationConfig(num_agents=8, random_seed=9))
    engine = SimulationEngine(state)
    engine.populate_society()
    engine.inject_scenario("Gas prices rose after a refinery outage hit oil supply.")
    engine.run(steps=2)
    return state


def test_state_round_trips_complete_run_json() -> None:
    state = _run_state()
    state.config.model_roles["agent_decisions"] = ModelRoleConfig(
        provider="openai_compatible",
        model_name="llama-3.1-70b-versatile",
        api_base_url="https://api.groq.com/openai/v1",
        api_key="secret-key",
        temperature=0.3,
        max_tokens=256,
        enabled=True,
    )

    exported = state.to_json()
    loaded = ExperimentState.from_json(exported)

    assert "secret-key" not in exported
    assert loaded.config.num_agents == state.config.num_agents
    assert loaded.config.model_roles["agent_decisions"].enabled is True
    assert loaded.config.model_roles["agent_decisions"].api_key is None
    assert loaded.config.model_roles["agent_decisions"].model_name == "llama-3.1-70b-versatile"
    assert [agent.id for agent in loaded.agents] == [agent.id for agent in state.agents]
    assert [post.id for post in loaded.posts] == [post.id for post in state.posts]
    assert [scenario.id for scenario in loaded.scenarios] == [
        scenario.id for scenario in state.scenarios
    ]
    assert loaded.current_tick == state.current_tick
    assert loaded.last_reaction_count == state.last_reaction_count


def test_analytics_summarize_run() -> None:
    state = _run_state()

    reach = reach_summary(state)
    amplifiers = top_amplifiers(state)
    activity = tick_activity_rows(state)
    spread = topic_spread_by_tick(state)
    matrix = topic_author_matrix(state)
    diversity = feed_diversity(state, state.agents[0])

    assert reach.total_agents == 8
    assert reach.reached_agents > 0
    assert amplifiers
    assert len(activity) == 2
    assert spread
    assert matrix
    assert diversity.total_items > 0
