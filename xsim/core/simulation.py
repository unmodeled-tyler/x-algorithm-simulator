"""
Deterministic simulation primitives for the early xsim prototype.

These functions are intentionally simple and inspectable. They create a usable
local playground before LLM-backed agents are introduced.
"""

from __future__ import annotations

import random
from collections import Counter
from datetime import UTC, datetime
from typing import cast

from xsim.core.models import Agent, Engagement, FeedItem, Post, Scenario, SimulationConfig


ARCHETYPES: tuple[dict[str, str | list[str]], ...] = (
    {
        "username": "alex_green",
        "persona": "Left-leaning environmentalist who works in renewables and posts about climate policy.",
        "interests": ["climate", "energy", "policy"],
    },
    {
        "username": "jordan_local",
        "persona": "Moderate suburban parent who cares about gas prices, schools, and local services.",
        "interests": ["economy", "schools", "energy"],
    },
    {
        "username": "sam_ai",
        "persona": "Tech worker in San Francisco who is deeply into AI, startups, and markets.",
        "interests": ["technology", "ai", "economy"],
    },
    {
        "username": "taylor_rural",
        "persona": "Rural conservative who posts about farming, fuel costs, and federal policy.",
        "interests": ["farming", "energy", "policy"],
    },
    {
        "username": "casey_housing",
        "persona": "Young progressive activist focused on housing, student debt, and social justice.",
        "interests": ["housing", "policy", "labor"],
    },
    {
        "username": "morgan_markets",
        "persona": "Finance analyst who reacts to macro news, energy prices, and company earnings.",
        "interests": ["markets", "economy", "energy"],
    },
    {
        "username": "riley_culture",
        "persona": "Culture commentator who connects news events to media narratives and public mood.",
        "interests": ["culture", "media", "politics"],
    },
    {
        "username": "jamie_builder",
        "persona": "Small business owner who talks about hiring, costs, customers, and local regulation.",
        "interests": ["business", "labor", "economy"],
    },
)

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ai": ("ai", "model", "grok", "openai", "automation", "compute"),
    "business": ("business", "customer", "startup", "company", "store"),
    "climate": ("climate", "emissions", "renewable", "carbon", "weather"),
    "culture": ("culture", "media", "viral", "narrative", "celebrity"),
    "economy": ("price", "cost", "inflation", "market", "jobs", "recession"),
    "energy": ("gas", "oil", "fuel", "refinery", "energy", "electric"),
    "farming": ("farm", "crop", "rural", "diesel", "harvest"),
    "housing": ("housing", "rent", "mortgage", "zoning", "homeless"),
    "labor": ("worker", "wage", "union", "hiring", "layoff"),
    "media": ("news", "press", "headline", "coverage", "journalist"),
    "policy": ("policy", "law", "tax", "federal", "state", "regulation"),
    "politics": ("election", "senate", "congress", "democrat", "republican"),
    "schools": ("school", "teacher", "student", "college", "education"),
    "technology": ("tech", "software", "app", "platform", "data"),
}

POST_TEMPLATES: tuple[str, ...] = (
    "{stance} This feels directly connected to {interest}. {angle}",
    "People are going to read this through their own priors, but the {interest} angle matters most.",
    "The part I keep coming back to is {interest}. {angle}",
    "This is exactly why platforms need to show more than one lens on a story. {angle}",
)

ANGLES: tuple[str, ...] = (
    "The second-order effects may be bigger than the headline.",
    "The incentives underneath this are doing a lot of hidden work.",
    "I want to see who benefits, who pays, and who gets ignored.",
    "The local impact is going to look very different from the national take.",
)


def infer_topics(text: str) -> list[str]:
    """Infer coarse scenario topics from keyword matches."""
    lower = text.lower()
    scores = {
        topic: sum(1 for keyword in keywords if keyword in lower)
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    matched = [topic for topic, score in scores.items() if score > 0]
    if matched:
        return sorted(matched, key=lambda topic: (-scores[topic], topic))[:4]
    return ["culture", "politics"]


def create_default_agents(count: int, seed: int | None = 42) -> list[Agent]:
    """Create a reproducible starter society with lightweight follow edges."""
    rng = random.Random(seed)
    agents: list[Agent] = []

    for index in range(count):
        archetype = ARCHETYPES[index % len(ARCHETYPES)]
        suffix = index // len(ARCHETYPES) + 1
        base_username = cast(str, archetype["username"])
        username = f"{base_username}_{suffix}" if suffix > 1 else base_username
        agents.append(
            Agent(
                username=username,
                persona=cast(str, archetype["persona"]),
                interests=list(cast(list[str], archetype["interests"])),
            )
        )

    for agent in agents:
        same_interest = [
            other.id
            for other in agents
            if other.id != agent.id and set(other.interests).intersection(agent.interests)
        ]
        others = [other.id for other in agents if other.id != agent.id]
        rng.shuffle(same_interest)
        rng.shuffle(others)
        agent.following_ids = set((same_interest[:3] + others[:2])[:5])

    return agents


def generate_scenario_posts(
    agents: list[Agent],
    scenario: Scenario,
    config: SimulationConfig,
) -> list[Post]:
    """Generate a deterministic first wave of reactions to a scenario."""
    rng = random.Random(f"{config.random_seed}:{scenario.id}:{scenario.description}")
    scenario_topics = infer_topics(scenario.description)
    posts: list[Post] = []

    for agent in agents:
        overlap = sorted(set(agent.interests).intersection(scenario_topics))
        interest = overlap[0] if overlap else rng.choice(agent.interests or scenario_topics)
        reaction_chance = 0.35 + (0.2 * len(overlap))
        if rng.random() > min(reaction_chance, 0.9):
            continue

        stance = rng.choice(("Worth watching.", "Not surprised.", "This is the real story.", "People will miss this."))
        text = rng.choice(POST_TEMPLATES).format(
            stance=stance,
            interest=interest,
            angle=rng.choice(ANGLES),
        )
        posts.append(
            Post(
                author_id=agent.id,
                author_username=agent.username,
                text=text,
                topic_tags=sorted(set(scenario_topics + [interest])),
            )
        )

    return posts


def rank_feed(
    viewer: Agent,
    posts: list[Post],
    config: SimulationConfig,
    limit: int = 20,
    engagements: list[Engagement] | None = None,
) -> list[FeedItem]:
    """Rank posts for one viewer with inspectable score components.

    The ranker is still intentionally lightweight, but it now has a shape
    closer to a real feed model: topical relevance, network proximity, recency,
    social proof, reply treatment, negative feedback, and author diversity.
    """
    candidates: list[FeedItem] = []
    engagement_by_post = _engagement_counts(engagements or [])

    for post in posts:
        if post.author_id == viewer.id:
            continue

        topic_overlap = len(set(viewer.interests).intersection(post.topic_tags))
        in_network = post.author_id in viewer.following_ids
        post_engagements = engagement_by_post.get(post.id, Counter())
        breakdown: dict[str, float] = {
            "base": 1.0,
            "topic": topic_overlap * config.topic_match_weight,
            "network": config.in_network_weight if in_network else config.discovery_weight,
            "recency": _recency_score(post, config),
            "social": _social_score(post_engagements, config),
            "negative": -_negative_score(post_engagements, config),
        }

        score = sum(breakdown.values())

        if topic_overlap:
            topic_reason = f"topic +{breakdown['topic']:.2f}"
        else:
            topic_reason = "no topic match"

        if post.is_reply():
            score *= config.reply_boost
            breakdown["reply_multiplier"] = config.reply_boost

        reason_parts = [
            "in-network" if in_network else "discovery",
            topic_reason,
            f"recency +{breakdown['recency']:.2f}",
        ]
        if breakdown["social"] > 0:
            reason_parts.append(f"social +{breakdown['social']:.2f}")
        if breakdown["negative"] < 0:
            reason_parts.append(f"negative {breakdown['negative']:.2f}")
        if post.is_reply():
            reason_parts.append(f"reply x{config.reply_boost:.1f}")

        candidates.append(
            FeedItem(
                post=post,
                score=round(score, 3),
                reason=", ".join(reason_parts),
                score_breakdown={key: round(value, 3) for key, value in breakdown.items()},
            )
        )

    return _select_with_author_diversity(candidates, config, limit)


def _engagement_counts(engagements: list[Engagement]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {}
    for engagement in engagements:
        counts.setdefault(engagement.post_id, Counter())[engagement.action] += 1
    return counts


def _recency_score(post: Post, config: SimulationConfig) -> float:
    now = datetime.now(UTC)
    timestamp = post.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    age_hours = max(0.0, (now - timestamp).total_seconds() / 3600)
    return config.recency_weight / (1.0 + age_hours / 24.0)


def _social_score(counts: Counter[str], config: SimulationConfig) -> float:
    weighted = (
        counts["like"] * 0.6
        + counts["repost"] * 1.0
        + counts["reply"] * 0.8
        + counts["quote"] * 0.9
        + counts["dwell"] * 0.2
        + counts["click"] * 0.3
    )
    return min(3.0, weighted * config.social_proof_weight)


def _negative_score(counts: Counter[str], config: SimulationConfig) -> float:
    weighted = counts["not_interested"] + counts["block"] * 2.0 + counts["mute"] * 1.5
    return weighted * config.negative_action_penalty


def _select_with_author_diversity(
    candidates: list[FeedItem],
    config: SimulationConfig,
    limit: int,
) -> list[FeedItem]:
    selected: list[FeedItem] = []
    recent_authors: list[str] = []

    for item in sorted(candidates, key=lambda candidate: candidate.score, reverse=True):
        repeats = recent_authors.count(item.post.author_id)
        if repeats:
            penalty = repeats * config.author_diversity_penalty
            item.score = round(item.score - penalty, 3)
            item.score_breakdown["diversity"] = round(-penalty, 3)
            item.reason = f"{item.reason}, diversity -{penalty:.2f}" if item.reason else None
        selected.append(item)
        recent_authors.append(item.post.author_id)
        recent_authors = recent_authors[-config.feed_diversity_window :]
        if len(selected) >= limit:
            break

    return sorted(selected, key=lambda candidate: candidate.score, reverse=True)
