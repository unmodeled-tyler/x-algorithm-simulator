from dataclasses import replace

from xsim.core import (
    ExperimentState,
    SimulationConfig,
    SimulationEngine,
    clone_experiment_state,
    run_comparison,
)


def _base_state() -> ExperimentState:
    state = ExperimentState(config=SimulationConfig(num_agents=8, random_seed=13))
    engine = SimulationEngine(state)
    engine.populate_society()
    engine.inject_scenario("Gas prices rose after a refinery outage hit oil supply.")
    return state


def test_clone_experiment_state_preserves_ids_without_aliasing() -> None:
    state = _base_state()

    cloned = clone_experiment_state(state)
    cloned.posts[0].text = "changed in clone"

    assert [agent.id for agent in cloned.agents] == [agent.id for agent in state.agents]
    assert [post.id for post in cloned.posts] == [post.id for post in state.posts]
    assert cloned.posts[0].text != state.posts[0].text


def test_run_comparison_does_not_mutate_source_state() -> None:
    state = _base_state()
    original_post_ids = [post.id for post in state.posts]
    variant_config = replace(state.config, discovery_weight=2.0, in_network_weight=0.2)

    comparison = run_comparison(
        state,
        baseline_config=state.config,
        variant_config=variant_config,
        steps=3,
    )

    assert [post.id for post in state.posts] == original_post_ids
    assert state.current_tick == 0
    assert comparison.baseline.current_tick == 3
    assert comparison.variant.current_tick == 3
    assert comparison.baseline.config.discovery_weight == state.config.discovery_weight
    assert comparison.variant.config.discovery_weight == 2.0
    assert comparison.metric_rows()
