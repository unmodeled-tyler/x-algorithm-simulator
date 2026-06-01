"""
Deterministic simulation primitives for the early xsim prototype.

These functions are intentionally simple and inspectable. They create a usable
local playground before LLM-backed agents are introduced.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import cast

from xsim.core.models import Agent, FeedItem, Post, Scenario, SimulationConfig


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
) -> list[FeedItem]:
    """Rank posts for one viewer using the knobs exposed in the Streamlit UI."""
    ranked: list[FeedItem] = []
    author_counts: Counter[str] = Counter()

    for post in posts:
        if post.author_id == viewer.id:
            continue

        topic_overlap = len(set(viewer.interests).intersection(post.topic_tags))
        in_network = post.author_id in viewer.following_ids
        score = 1.0 + (topic_overlap * 0.7)
        reason_parts: list[str] = []

        if in_network:
            score += config.in_network_weight
            reason_parts.append("in-network author")
        else:
            score += config.discovery_weight
            reason_parts.append("discovery candidate")

        if post.is_reply():
            score *= config.reply_boost
            reason_parts.append("reply boost")

        if topic_overlap:
            reason_parts.append(f"{topic_overlap} topic match{'es' if topic_overlap > 1 else ''}")

        repeated_author_penalty = author_counts[post.author_id] * config.author_diversity_penalty
        score -= repeated_author_penalty
        if repeated_author_penalty:
            reason_parts.append("author diversity penalty")

        author_counts[post.author_id] += 1
        ranked.append(
            FeedItem(
                post=post,
                score=round(score, 3),
                reason=", ".join(reason_parts) or "baseline relevance",
            )
        )

    return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]
