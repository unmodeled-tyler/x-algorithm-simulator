from xsim.core import Scenario, SimulationConfig, create_default_agents, generate_scenario_posts, rank_feed
from xsim.core.simulation import infer_topics


def test_infer_topics_from_scenario_text() -> None:
    topics = infer_topics("Gas prices rose after a refinery outage hit oil supply.")

    assert topics[:2] == ["energy", "economy"]


def test_scenario_generation_and_ranking_are_actionable() -> None:
    config = SimulationConfig(num_agents=12, random_seed=7)
    agents = create_default_agents(config.num_agents, config.random_seed)
    scenario = Scenario(description="Gas prices rose after a refinery outage hit oil supply.")

    posts = generate_scenario_posts(agents, scenario, config)
    ranked_feed = rank_feed(agents[0], posts, config)

    assert len(agents) == 12
    assert posts
    assert all(post.author_username for post in posts)
    assert ranked_feed == sorted(ranked_feed, key=lambda item: item.score, reverse=True)
