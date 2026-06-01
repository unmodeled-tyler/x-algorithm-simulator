"""
xsim interactive simulator.

Run with:

    streamlit run simulator/app.py
"""

from __future__ import annotations

import sys
from collections import Counter
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

try:
    from xsim.core import (
        Agent,
        FeedItem,
        Scenario,
        SimulationConfig,
        create_default_agents,
        generate_scenario_posts,
        infer_topics,
        rank_feed,
    )
except ModuleNotFoundError as e:
    st.error(
        "Could not import the `xsim` package.\n\n"
        "Run `uv sync` or `pip install -e \".[dev]\"` from the repository root.\n\n"
        f"Original error: {e}"
    )
    st.stop()


st.set_page_config(
    page_title="xsim",
    page_icon="X",
    layout="wide",
    initial_sidebar_state="collapsed",
)


APP_CSS = """
<style>
    :root {
        --xsim-bg: #f7f5ef;
        --xsim-panel: #ffffff;
        --xsim-ink: #171717;
        --xsim-muted: #62615c;
        --xsim-border: #ddd8cd;
        --xsim-accent: #166a5f;
        --xsim-accent-2: #7a3f2b;
        --xsim-soft: #edf6f3;
        --xsim-warm: #fbefe8;
    }

    .stApp {
        background: var(--xsim-bg);
        color: var(--xsim-ink);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }

    h1, h2, h3 {
        letter-spacing: 0;
    }

    div[data-testid="stMetric"] {
        background: var(--xsim-panel);
        border: 1px solid var(--xsim-border);
        border-radius: 8px;
        padding: 0.9rem 1rem;
    }

    div[data-testid="stTabs"] button {
        font-weight: 650;
    }

    .hero {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 2rem;
        border-bottom: 1px solid var(--xsim-border);
        padding-bottom: 1.2rem;
        margin-bottom: 1.2rem;
    }

    .brand {
        font-size: clamp(2.2rem, 5vw, 4.6rem);
        line-height: 0.95;
        font-weight: 800;
        letter-spacing: 0;
    }

    .tagline {
        color: var(--xsim-muted);
        font-size: 1.05rem;
        max-width: 620px;
        margin-top: 0.55rem;
    }

    .stage-rail {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 1rem 0 1.4rem;
    }

    .stage {
        background: var(--xsim-panel);
        border: 1px solid var(--xsim-border);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        min-height: 92px;
    }

    .stage.active {
        border-color: var(--xsim-accent);
        box-shadow: inset 0 0 0 1px var(--xsim-accent);
    }

    .stage-label {
        color: var(--xsim-muted);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .stage-title {
        font-weight: 750;
        font-size: 1rem;
    }

    .stage-note {
        color: var(--xsim-muted);
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 760;
        margin: 0.4rem 0 0.6rem;
    }

    .post-card, .agent-card, .event-card {
        background: var(--xsim-panel);
        border: 1px solid var(--xsim-border);
        border-radius: 8px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.75rem;
    }

    .post-topline {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.45rem;
    }

    .handle {
        font-weight: 760;
    }

    .score {
        color: var(--xsim-accent);
        font-weight: 780;
        white-space: nowrap;
    }

    .body-text {
        font-size: 1rem;
        line-height: 1.45;
    }

    .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-top: 0.65rem;
    }

    .chip {
        border: 1px solid var(--xsim-border);
        border-radius: 999px;
        padding: 0.16rem 0.5rem;
        color: var(--xsim-muted);
        font-size: 0.78rem;
        background: #faf9f5;
    }

    .chip.accent {
        border-color: #b7d6cf;
        background: var(--xsim-soft);
        color: var(--xsim-accent);
    }

    .empty-state {
        background: var(--xsim-panel);
        border: 1px dashed var(--xsim-border);
        border-radius: 8px;
        padding: 1.4rem;
        color: var(--xsim-muted);
    }

    @media (max-width: 800px) {
        .hero {
            display: block;
        }

        .stage-rail {
            grid-template-columns: 1fr;
        }

        .post-topline {
            display: block;
        }
    }
</style>
"""


def init_state() -> None:
    defaults = {
        "agents": [],
        "posts": [],
        "events": [],
        "selected_agent_id": None,
        "last_reaction_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_config() -> SimulationConfig:
    with st.sidebar:
        st.header("Experiment Setup")
        num_agents = st.slider("Society size", 5, 100, 20, step=5)
        random_seed = st.number_input("Seed", min_value=0, value=42, step=1)

        st.divider()
        st.subheader("Ranking")
        in_network_weight = st.slider("In-network boost", 0.0, 2.0, 1.0, 0.1)
        discovery_weight = st.slider("Discovery boost", 0.0, 2.0, 0.8, 0.1)
        diversity_penalty = st.slider("Repeat-author penalty", 0.0, 1.0, 0.65, 0.05)
        reply_boost = st.slider("Reply multiplier", 0.5, 4.0, 1.8, 0.1)

        st.divider()
        with st.expander("LLM backend", expanded=False):
            provider = st.selectbox("Provider", ["ollama", "openai_compatible"], index=0)
            if provider == "ollama":
                model_name = st.text_input("Ollama model", value="qwen2.5:7b")
                api_base = None
                api_key = None
            else:
                model_name = st.text_input("Model", value="llama-3.1-70b-versatile")
                api_base = st.text_input(
                    "Base URL",
                    value="https://api.groq.com/openai/v1",
                )
                api_key = st.text_input("API key", type="password")
            temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

        if st.button("Reset Run", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    return SimulationConfig(
        num_agents=num_agents,
        model_provider=provider,
        model_name=model_name,
        api_base_url=api_base,
        api_key=api_key,
        temperature=temperature,
        in_network_weight=in_network_weight,
        discovery_weight=discovery_weight,
        author_diversity_penalty=diversity_penalty,
        reply_boost=reply_boost,
        random_seed=int(random_seed),
    )


def render_stage_rail() -> None:
    agents_ready = bool(st.session_state.agents)
    event_ready = bool(st.session_state.events)
    posts_ready = bool(st.session_state.posts)
    feed_ready = agents_ready and posts_ready
    stages = [
        ("01", "Build society", "Create agents and social graph", agents_ready),
        ("02", "Inject event", "Add a shock to the world", event_ready),
        ("03", "Observe spread", "Watch first-wave reactions", posts_ready),
        ("04", "Tune feed", "Inspect ranked timelines", feed_ready),
    ]
    columns = st.columns(4)
    for column, (label, title, note, active) in zip(columns, stages, strict=True):
        with column.container(border=True):
            st.caption(f"{label} {'Ready' if active else 'Pending'}")
            st.markdown(f"**{title}**")
            st.caption(note)


def render_post(item: FeedItem | None, author: str, text: str, chips: list[str]) -> None:
    score = f"{item.score:.3f}" if item else ""
    score_html = f'<div class="score">{score}</div>' if score else ""
    reason_chips = "".join(
        f'<span class="chip accent">{escape(chip.strip())}</span>'
        for chip in chips
        if chip.strip()
    )
    st.markdown(
        f"""
        <div class="post-card">
            <div class="post-topline">
                <div class="handle">@{escape(author)}</div>
                {score_html}
            </div>
            <div class="body-text">{escape(text)}</div>
            <div class="chips">{reason_chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent(agent: Agent) -> None:
    chips = "".join(f'<span class="chip">{escape(topic)}</span>' for topic in agent.interests)
    st.markdown(
        f"""
        <div class="agent-card">
            <div class="handle">@{escape(agent.username)}</div>
            <div class="body-text">{escape(agent.persona)}</div>
            <div class="chips">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def initialize_agents(config: SimulationConfig) -> None:
    st.session_state.agents = create_default_agents(config.num_agents, config.random_seed)
    st.session_state.posts = []
    st.session_state.events = []
    st.session_state.last_reaction_count = 0
    st.session_state.selected_agent_id = st.session_state.agents[0].id


def inject_scenario(description: str, config: SimulationConfig) -> None:
    if not st.session_state.agents:
        initialize_agents(config)
    scenario = Scenario(description=description)
    new_posts = generate_scenario_posts(st.session_state.agents, scenario, config)
    st.session_state.events.append(scenario)
    st.session_state.posts.extend(new_posts)
    st.session_state.last_reaction_count = len(new_posts)


def topic_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    for post in st.session_state.posts:
        counts.update(post.topic_tags)
    return counts


def render_overview(config: SimulationConfig) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Agents", len(st.session_state.agents))
    metric_cols[1].metric("Injected events", len(st.session_state.events))
    metric_cols[2].metric("Posts generated", len(st.session_state.posts))
    metric_cols[3].metric("Last reaction wave", st.session_state.last_reaction_count)

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.markdown('<div class="section-title">Run Console</div>', unsafe_allow_html=True)
        scenario_text = st.text_area(
            "Scenario",
            placeholder=(
                "Gas prices jump after a refinery outage. Candidates immediately blame "
                "each other while commuters and small businesses react."
            ),
            height=128,
            label_visibility="collapsed",
        )
        actions = st.columns([1, 1, 1])
        if actions[0].button("Create Society", type="secondary", use_container_width=True):
            initialize_agents(config)
            st.rerun()
        if actions[1].button(
            "Inject Event",
            type="primary",
            disabled=not scenario_text.strip(),
            use_container_width=True,
        ):
            inject_scenario(scenario_text.strip(), config)
            st.rerun()
        if actions[2].button("Clear Posts", use_container_width=True):
            st.session_state.posts = []
            st.session_state.events = []
            st.session_state.last_reaction_count = 0
            st.rerun()

        if st.session_state.posts:
            st.markdown('<div class="section-title">Latest Reactions</div>', unsafe_allow_html=True)
            for post in reversed(st.session_state.posts[-5:]):
                render_post(
                    None,
                    post.author_username or post.author_id,
                    post.text,
                    post.topic_tags,
                )
        else:
            st.markdown(
                """
                <div class="empty-state">
                    <strong>Nothing has happened yet.</strong><br>
                    Create a society, write a scenario, then inject the event.
                    This panel will become the live reaction stream.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="section-title">Signal Summary</div>', unsafe_allow_html=True)
        if st.session_state.events:
            latest = st.session_state.events[-1]
            inferred = infer_topics(latest.description)
            st.markdown(
                f"""
                <div class="event-card">
                    <div class="stage-label">Latest event</div>
                    <div class="body-text">{escape(latest.description)}</div>
                    <div class="chips">
                        {''.join(f'<span class="chip accent">{escape(topic)}</span>' for topic in inferred)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="empty-state">
                    <strong>No event injected yet.</strong><br>
                    After injection, this panel summarizes the latest scenario and inferred topics.
                </div>
                """,
                unsafe_allow_html=True,
            )

        counts = topic_counts()
        if counts:
            st.bar_chart(dict(counts.most_common(8)), height=220)
        else:
            st.caption("Topic distribution appears after posts are generated.")


def render_feed_lab(config: SimulationConfig) -> None:
    if not st.session_state.agents:
        st.markdown(
            """
            <div class="empty-state">
                <strong>Feed Lab is waiting for agents.</strong><br>
                Go to Overview and create a society. Once agents exist, this tab can rank posts for each persona.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if not st.session_state.posts:
        st.markdown(
            """
            <div class="empty-state">
                <strong>No candidate posts yet.</strong><br>
                Inject a scenario from Overview. Then this tab becomes a ranked timeline inspector.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    agent_lookup = {agent.id: agent for agent in st.session_state.agents}
    selected_id = st.selectbox(
        "Inspect feed as",
        options=[agent.id for agent in st.session_state.agents],
        format_func=lambda agent_id: f"@{agent_lookup[agent_id].username}",
        key="selected_agent_id",
    )
    viewer = agent_lookup[selected_id]
    ranked_feed = rank_feed(viewer, st.session_state.posts, config)

    profile, feed = st.columns([0.8, 1.4], gap="large")
    with profile:
        st.markdown('<div class="section-title">Viewer Profile</div>', unsafe_allow_html=True)
        render_agent(viewer)
        st.metric("Following", len(viewer.following_ids))
        st.metric("Feed candidates", len(ranked_feed))

    with feed:
        st.markdown('<div class="section-title">Ranked Timeline</div>', unsafe_allow_html=True)
        for item in ranked_feed[:12]:
            render_post(
                item,
                item.post.author_username or item.post.author_id,
                item.post.text,
                (item.reason or "").split(","),
            )


def render_society() -> None:
    if not st.session_state.agents:
        st.markdown(
            """
            <div class="empty-state">
                <strong>No agents yet.</strong><br>
                Create a society from Overview to browse persona cards here.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="section-title">Agent Society</div>', unsafe_allow_html=True)
    columns = st.columns(2)
    for index, agent in enumerate(st.session_state.agents):
        with columns[index % 2]:
            render_agent(agent)


def render_tuning(config: SimulationConfig) -> None:
    st.markdown('<div class="section-title">Current Ranking Model</div>', unsafe_allow_html=True)
    st.json(
        {
            "in_network_weight": config.in_network_weight,
            "discovery_weight": config.discovery_weight,
            "author_diversity_penalty": config.author_diversity_penalty,
            "reply_boost": config.reply_boost,
            "random_seed": config.random_seed,
            "llm_backend": f"{config.model_provider} / {config.model_name}",
        }
    )
    st.caption("Change ranking controls from the sidebar, then return to Feed Lab to see the ordering shift.")


init_state()
config = build_config()

st.markdown(APP_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
        <div>
            <div class="brand">xsim</div>
            <div class="tagline">
                A local experiment bench for building a society, injecting an event,
                and watching a tunable social feed decide what each agent sees.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
render_stage_rail()

overview_tab, feed_tab, society_tab, tuning_tab = st.tabs(
    ["Overview", "Feed Lab", "Society", "Tuning"]
)

with overview_tab:
    render_overview(config)

with feed_tab:
    render_feed_lab(config)

with society_tab:
    render_society()

with tuning_tab:
    render_tuning(config)
