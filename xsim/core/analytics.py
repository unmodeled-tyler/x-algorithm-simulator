"""Analytics helpers for completed or in-flight experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from xsim.core.models import Agent, FeedItem
from xsim.core.simulation import rank_feed
from xsim.core.state import ExperimentState


@dataclass(frozen=True)
class ReachSummary:
    """How much of the society touched the current story."""

    reached_agents: int
    total_agents: int
    reach_ratio: float


@dataclass(frozen=True)
class FeedDiversity:
    """Diversity metrics for one ranked feed."""

    unique_authors: int
    total_items: int
    discovery_items: int
    in_network_items: int
    topic_count: int

    @property
    def discovery_ratio(self) -> float:
        if self.total_items == 0:
            return 0.0
        return self.discovery_items / self.total_items


def reach_summary(state: ExperimentState) -> ReachSummary:
    """Return unique agents who posted or engaged divided by total agents."""
    reached = {post.author_id for post in state.posts}
    reached.update(engagement.agent_id for engagement in state.engagements)
    total = len(state.agents)
    return ReachSummary(
        reached_agents=len(reached),
        total_agents=total,
        reach_ratio=(len(reached) / total) if total else 0.0,
    )


def top_amplifiers(state: ExperimentState, limit: int = 8) -> list[tuple[Agent, int]]:
    """Rank agents by how much they amplify: replies, reposts, quotes, and posts."""
    scores: Counter[str] = Counter()
    for post in state.posts:
        scores[post.author_id] += 1
    for engagement in state.engagements:
        if engagement.action in {"repost", "quote", "reply"}:
            scores[engagement.agent_id] += 2
        elif engagement.action in {"like", "click", "dwell"}:
            scores[engagement.agent_id] += 1

    agents = state.agent_index()
    return [
        (agents[agent_id], score)
        for agent_id, score in scores.most_common(limit)
        if agent_id in agents
    ]


def topic_spread_by_tick(state: ExperimentState) -> list[dict[str, int | str]]:
    """Count topics introduced in each simulation tick."""
    posts = state.post_index()
    rows: list[dict[str, int | str]] = []
    for tick in state.ticks:
        counts: Counter[str] = Counter()
        for post_id in tick.new_post_ids:
            post = posts.get(post_id)
            if post:
                counts.update(post.topic_tags or ["untagged"])
        if not counts:
            rows.append({"tick": tick.index, "topic": "none", "count": 0})
            continue
        for topic, count in counts.most_common():
            rows.append({"tick": tick.index, "topic": topic, "count": count})
    return rows


def feed_diversity(
    state: ExperimentState,
    viewer: Agent,
    limit: int = 20,
) -> FeedDiversity:
    """Summarize author, network, and topic diversity in a viewer's feed."""
    feed = rank_feed(viewer, state.posts, state.config, limit=limit, engagements=state.engagements)
    return feed_diversity_from_items(viewer, feed)


def feed_diversity_from_items(viewer: Agent, feed: list[FeedItem]) -> FeedDiversity:
    authors = {item.post.author_id for item in feed}
    topics = {topic for item in feed for topic in item.post.topic_tags}
    in_network = sum(1 for item in feed if item.post.author_id in viewer.following_ids)
    total = len(feed)
    return FeedDiversity(
        unique_authors=len(authors),
        total_items=total,
        discovery_items=total - in_network,
        in_network_items=in_network,
        topic_count=len(topics),
    )


def tick_activity_rows(state: ExperimentState) -> list[dict[str, int]]:
    """Compact tick history for charts and export summaries."""
    return [
        {
            "tick": tick.index,
            "new_posts": tick.notes.get("new_posts", 0),
            "engagements": tick.notes.get("engagements", 0),
            "agents": tick.notes.get("agents", 0),
        }
        for tick in state.ticks
    ]


def topic_author_matrix(state: ExperimentState) -> dict[str, dict[str, int]]:
    """Return topic -> username -> post count for cluster-ish inspection."""
    agents = state.agent_index()
    matrix: dict[str, dict[str, int]] = defaultdict(dict)
    for post in state.posts:
        agent = agents.get(post.author_id)
        username = agent.username if agent else post.author_username or post.author_id
        for topic in post.topic_tags or ["untagged"]:
            matrix[topic][username] = matrix[topic].get(username, 0) + 1
    return dict(matrix)
