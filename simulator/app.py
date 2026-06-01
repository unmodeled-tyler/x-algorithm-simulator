"""
xsim interactive simulator.

Run with:

    streamlit run simulator/app.py
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

try:
    from xsim.core import (
        Agent,
        DeterministicBehavior,
        ExperimentState,
        FeedItem,
        LLMAgentBehavior,
        SimulationConfig,
        SimulationEngine,
        feed_diversity_from_items,
        infer_topics,
        rank_feed,
        reach_summary,
        tick_activity_rows,
        top_amplifiers,
        topic_author_matrix,
        topic_spread_by_tick,
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
    """Initialize a single ExperimentState object in st.session_state."""
    if "experiment" not in st.session_state:
        # Bootstrap a minimal state; the real config is filled in once the
        # sidebar renders.
        st.session_state["experiment"] = ExperimentState(
            config=SimulationConfig(num_agents=20, random_seed=42)
        )
        st.session_state["engine"] = SimulationEngine(st.session_state["experiment"])
        st.session_state["selected_agent_id"] = None
        st.session_state["use_llm"] = False


def get_state() -> ExperimentState:
    return st.session_state["experiment"]  # type: ignore[no-any-return]


def get_engine() -> SimulationEngine:
    return st.session_state["engine"]  # type: ignore[no-any-return]


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
        topic_match_weight = st.slider("Topic-match weight", 0.0, 2.0, 0.85, 0.05)
        recency_weight = st.slider("Recency weight", 0.0, 1.5, 0.35, 0.05)
        social_proof_weight = st.slider("Social-proof weight", 0.0, 1.0, 0.22, 0.02)
        negative_action_penalty = st.slider("Negative-action penalty", 0.0, 5.0, 2.5, 0.1)

        st.divider()
        with st.expander("LLM backend", expanded=False):
            use_llm = st.checkbox(
                "Enable LLM-backed agents",
                value=st.session_state.get("use_llm", False),
                help=(
                    "If disabled (default), agents use the local deterministic "
                    "behavior. Enabling this routes decisions through xsim.llm "
                    "and silently falls back to deterministic on failure."
                ),
            )
            st.session_state["use_llm"] = use_llm
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
        negative_action_penalty=negative_action_penalty,
        topic_match_weight=topic_match_weight,
        recency_weight=recency_weight,
        social_proof_weight=social_proof_weight,
        random_seed=int(random_seed),
    )


def sync_state_to_config(state: ExperimentState, config: SimulationConfig) -> None:
    """Push the sidebar config into the ExperimentState (and re-apply to engine)."""
    state.config = config


def render_stage_rail(state: ExperimentState) -> None:
    agents_ready = bool(state.agents)
    event_ready = bool(state.scenarios)
    posts_ready = bool(state.posts)
    ticks_ready = bool(state.ticks)
    stages = [
        ("01", "Build society", "Create agents and social graph", agents_ready),
        ("02", "Inject event", "Add a shock to the world", event_ready),
        ("03", "Observe spread", "Watch first-wave reactions", posts_ready),
        ("04", "Step simulation", f"Run ticks ({state.current_tick} done)", ticks_ready),
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
    state = get_state()
    state.reset_society()
    sync_state_to_config(state, config)
    engine = get_engine()
    engine.attach_state(state)
    engine.populate_society()
    if state.agents:
        st.session_state["selected_agent_id"] = state.agents[0].id


def inject_scenario(description: str, config: SimulationConfig) -> None:
    state = get_state()
    sync_state_to_config(state, config)
    engine = get_engine()
    engine.attach_state(state)
    if not state.agents:
        engine.populate_society()
    engine.inject_scenario(description)


def step_simulation(steps: int, config: SimulationConfig) -> None:
    state = get_state()
    sync_state_to_config(state, config)
    engine = get_engine()
    engine.attach_state(state)
    if not state.agents:
        engine.populate_society()
    if st.session_state.get("use_llm", False):
        try:
            from xsim.llm import LLMConfig

            llm_cfg = LLMConfig(
                provider=config.model_provider,
                model=config.model_name,
                base_url=config.api_base_url,
                api_key=config.api_key,
                temperature=config.temperature,
            )
            engine.set_behavior(LLMAgentBehavior(llm_config=llm_cfg))
        except Exception:
            engine.set_behavior(DeterministicBehavior())
    else:
        engine.set_behavior(DeterministicBehavior())
    engine.run(steps=steps)


def render_overview(config: SimulationConfig) -> None:
    state = get_state()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Agents", len(state.agents))
    metric_cols[1].metric("Injected events", len(state.scenarios))
    metric_cols[2].metric("Posts generated", len(state.posts))
    metric_cols[3].metric("Ticks completed", state.current_tick)
    reach = reach_summary(state)
    if reach.total_agents:
        st.progress(
            reach.reach_ratio,
            text=f"Scenario reach: {reach.reached_agents}/{reach.total_agents} agents touched the story",
        )

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
            key="scenario_text",
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
            state.clear_posts()
            st.rerun()

        # Step Simulation row.
        step_cols = st.columns([1, 1, 1])
        steps_to_run = step_cols[0].number_input(
            "Ticks",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            label_visibility="collapsed",
        )
        if step_cols[1].button(
            "Step Simulation",
            type="primary",
            disabled=not state.agents,
            use_container_width=True,
        ):
            step_simulation(int(steps_to_run), config)
            st.rerun()
        last_tick = state.ticks[-1] if state.ticks else None
        last_summary = (
            f"Last tick: +{last_tick.notes.get('new_posts', 0)} posts, "
            f"+{last_tick.notes.get('engagements', 0)} actions"
            if last_tick
            else "No ticks yet."
        )
        step_cols[2].markdown(
            f"<div style='text-align:center;color:var(--xsim-muted);font-size:0.85rem;padding-top:0.6rem'>{escape(last_summary)}</div>",
            unsafe_allow_html=True,
        )

        if state.posts:
            st.markdown('<div class="section-title">Latest Reactions</div>', unsafe_allow_html=True)
            for post in reversed(state.posts[-5:]):
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
        if state.scenarios:
            latest = state.scenarios[-1]
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

        counts = state.topic_counts()
        if counts:
            st.bar_chart(dict(counts.most_common(8)), height=220)
        else:
            st.caption("Topic distribution appears after posts are generated.")

        if state.engagements:
            st.markdown(
                "<div class='stage-label'>Engagement mix</div>",
                unsafe_allow_html=True,
            )
            st.bar_chart(dict(state.engagement_counts().most_common()), height=180)

        amplifiers = top_amplifiers(state, limit=5)
        if amplifiers:
            st.markdown(
                "<div class='stage-label'>Top amplifiers</div>",
                unsafe_allow_html=True,
            )
            for agent, score in amplifiers:
                st.caption(f"@{agent.username}: {score}")


def render_feed_lab(config: SimulationConfig) -> None:
    state = get_state()
    if not state.agents:
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

    if not state.posts:
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

    agent_lookup = state.agent_index()
    options = list(agent_lookup.keys())
    if st.session_state.get("selected_agent_id") not in options:
        st.session_state["selected_agent_id"] = options[0]
    selected_id = st.selectbox(
        "Inspect feed as",
        options=options,
        format_func=lambda agent_id: f"@{agent_lookup[agent_id].username}",
        key="selected_agent_id",
    )
    viewer = agent_lookup[selected_id]
    ranked_feed = rank_feed(viewer, state.posts, config, engagements=state.engagements)
    diversity = feed_diversity_from_items(viewer, ranked_feed)

    profile, feed = st.columns([0.8, 1.4], gap="large")
    with profile:
        st.markdown('<div class="section-title">Viewer Profile</div>', unsafe_allow_html=True)
        render_agent(viewer)
        st.metric("Following", len(viewer.following_ids))
        st.metric("Feed candidates", len(ranked_feed))
        st.metric("Unique authors", diversity.unique_authors)
        st.metric("Discovery mix", f"{diversity.discovery_ratio:.0%}")
        st.metric("Topics visible", diversity.topic_count)

    with feed:
        st.markdown('<div class="section-title">Ranked Timeline</div>', unsafe_allow_html=True)
        for item in ranked_feed[:12]:
            render_post(
                item,
                item.post.author_username or item.post.author_id,
                item.post.text,
                (item.reason or "").split(","),
            )
            with st.expander("Score breakdown", expanded=False):
                st.json(item.score_breakdown)


def render_society() -> None:
    state = get_state()
    if not state.agents:
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
    for index, agent in enumerate(state.agents):
        with columns[index % 2]:
            render_agent(agent)


def render_tuning(config: SimulationConfig) -> None:
    state = get_state()
    st.markdown('<div class="section-title">Current Ranking Model</div>', unsafe_allow_html=True)
    st.json(
        {
            "in_network_weight": config.in_network_weight,
            "discovery_weight": config.discovery_weight,
            "author_diversity_penalty": config.author_diversity_penalty,
            "reply_boost": config.reply_boost,
            "topic_match_weight": config.topic_match_weight,
            "recency_weight": config.recency_weight,
            "social_proof_weight": config.social_proof_weight,
            "negative_action_penalty": config.negative_action_penalty,
            "random_seed": config.random_seed,
            "llm_backend": f"{config.model_provider} / {config.model_name}",
            "behavior": "LLMAgentBehavior" if st.session_state.get("use_llm") else "DeterministicBehavior",
        }
    )
    st.caption("Change ranking controls from the sidebar, then return to Feed Lab to see the ordering shift.")
    if state.ticks:
        st.markdown('<div class="section-title">Tick History</div>', unsafe_allow_html=True)
        for record in reversed(state.ticks[-10:]):
            st.markdown(
                f"""
                <div class="event-card">
                    <div class="stage-label">Tick {record.index}</div>
                    <div class="body-text">
                        {record.notes.get('new_posts', 0)} new posts ·
                        {record.notes.get('engagements', 0)} actions ·
                        {record.notes.get('agents', 0)} agents
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Save / Replay</div>', unsafe_allow_html=True)
    st.download_button(
        "Download Run JSON",
        data=state.to_json(),
        file_name=f"xsim-run-tick-{state.current_tick}.json",
        mime="application/json",
        use_container_width=True,
    )
    uploaded = st.file_uploader("Load Run JSON", type=["json"])
    if uploaded is not None:
        try:
            loaded = ExperimentState.from_json(uploaded.getvalue().decode("utf-8"))
        except Exception as exc:
            st.error(f"Could not load run JSON: {exc}")
        else:
            if st.button("Replace Current Run", type="primary", use_container_width=True):
                st.session_state["experiment"] = loaded
                st.session_state["engine"] = SimulationEngine(loaded)
                st.session_state["selected_agent_id"] = (
                    loaded.agents[0].id if loaded.agents else None
                )
                st.rerun()


def render_analytics() -> None:
    state = get_state()
    st.markdown('<div class="section-title">Run Analytics</div>', unsafe_allow_html=True)

    reach = reach_summary(state)
    cols = st.columns(4)
    cols[0].metric("Reach", f"{reach.reach_ratio:.0%}")
    cols[1].metric("Reached agents", reach.reached_agents)
    cols[2].metric("Engagements", len(state.engagements))
    cols[3].metric("Ticks", state.current_tick)

    activity = tick_activity_rows(state)
    if activity:
        st.markdown('<div class="section-title">Tick Activity</div>', unsafe_allow_html=True)
        st.line_chart(activity, x="tick", y=["new_posts", "engagements"], height=260)

    topic_rows = topic_spread_by_tick(state)
    if topic_rows:
        st.markdown('<div class="section-title">Topic Spread By Tick</div>', unsafe_allow_html=True)
        st.dataframe(topic_rows, use_container_width=True, hide_index=True)

    amplifiers = top_amplifiers(state, limit=10)
    if amplifiers:
        st.markdown('<div class="section-title">Top Amplifiers</div>', unsafe_allow_html=True)
        st.dataframe(
            [
                {
                    "agent": f"@{agent.username}",
                    "score": score,
                    "interests": ", ".join(agent.interests),
                }
                for agent, score in amplifiers
            ],
            use_container_width=True,
            hide_index=True,
        )

    matrix = topic_author_matrix(state)
    if matrix:
        st.markdown('<div class="section-title">Topic / Author Matrix</div>', unsafe_allow_html=True)
        rows = [
            {"topic": topic, "author": author, "posts": count}
            for topic, author_counts in matrix.items()
            for author, count in author_counts.items()
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


init_state()
config = build_config()

# Keep the ExperimentState in sync with the sidebar config.
sync_state_to_config(get_state(), config)

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
render_stage_rail(get_state())

overview_tab, feed_tab, society_tab, analytics_tab, tuning_tab = st.tabs(
    ["Overview", "Feed Lab", "Society", "Analytics", "Tuning"]
)

with overview_tab:
    render_overview(config)

with feed_tab:
    render_feed_lab(config)

with society_tab:
    render_society()

with analytics_tab:
    render_analytics()

with tuning_tab:
    render_tuning(config)
