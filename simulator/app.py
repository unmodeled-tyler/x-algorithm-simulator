"""
xsim — Interactive Platform Simulator

Recommended run command:

    streamlit run simulator/app.py

Make sure you've installed the project first:

    uv sync
    # or
    pip install -e ".[dev]"

This is the main UI for controlling the simulation, injecting scenarios,
tuning the algorithm, and watching agents behave.
"""

import sys
from pathlib import Path

# Make the repository root importable so `import xsim` works
# even when the package is not installed in editable mode.
# This is the most common source of "ModuleNotFoundError: No module named 'xsim'"
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

try:
    from xsim.core import (
        Scenario,
        SimulationConfig,
        create_default_agents,
        generate_scenario_posts,
        rank_feed,
    )
except ModuleNotFoundError as e:
    st.error(
        "Could not import the `xsim` package.\n\n"
        "This usually means you haven't installed the project in editable mode.\n\n"
        "Please run one of these commands first:\n\n"
        "    uv sync\n"
        "    # or\n"
        "    pip install -e \".[dev]\"\n\n"
        f"Original error: {e}"
    )
    st.stop()

st.set_page_config(
    page_title="xsim — Platform Simulator",
    page_icon="X",
    layout="wide",
)

st.title("xsim — Controllable X-Style Platform Simulator")
st.caption("Local-first society simulation with tunable algorithm + god-mode scenario injection")

# =============================================================================
# SIDEBAR — Settings Panel (the heart of what the user asked for)
# =============================================================================
with st.sidebar:
    st.header("Simulation Settings")

    num_agents = st.slider("Number of agents (users)", 5, 100, 20, step=5)

    st.subheader("LLM Backend")
    provider = st.selectbox(
        "Provider",
        ["ollama", "openai_compatible"],
        index=0,
        help="Ollama = fully local. OpenAI-compatible lets you use Groq, Together, OpenRouter, etc."
    )

    if provider == "ollama":
        model_name = st.text_input("Ollama model", value="qwen2.5:7b")
        api_base = None
        api_key = None
    else:
        model_name = st.text_input("Model name", value="llama-3.1-70b-versatile")
        api_base = st.text_input("Base URL (OpenAI compatible)", value="https://api.groq.com/openai/v1")
        api_key = st.text_input("API Key", type="password", help="Never stored. Used only in this session.")

    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    st.subheader("Algorithm Knobs (live tunable)")
    in_network_weight = st.slider("In-network weight", 0.0, 2.0, 1.0, 0.1)
    discovery_weight = st.slider("Discovery (OON) weight", 0.0, 2.0, 0.8, 0.1)
    diversity_penalty = st.slider("Author diversity penalty", 0.0, 1.0, 0.65, 0.05)
    reply_boost = st.slider("Reply boost", 0.5, 4.0, 1.8, 0.1)

    st.divider()
    if st.button("Reset Simulation", type="secondary"):
        st.session_state.clear()
        st.rerun()

# =============================================================================
# MAIN AREA
# =============================================================================

config = SimulationConfig(
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
)

# Store config and state
if "config" not in st.session_state:
    st.session_state.config = config
if "agents" not in st.session_state:
    st.session_state.agents = []
if "posts" not in st.session_state:
    st.session_state.posts = []
if "events" not in st.session_state:
    st.session_state.events = []
if "selected_agent_id" not in st.session_state:
    st.session_state.selected_agent_id = None

st.header("God Mode — Scenario Injection")

scenario_text = st.text_area(
    "Describe an event or news item to inject into the simulation",
    placeholder="The price of gas just increased by 20 cents per gallon due to a major refinery outage in the Gulf...",
    height=100,
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Inject Scenario", type="primary", disabled=not scenario_text.strip()):
        if not st.session_state.agents:
            st.session_state.agents = create_default_agents(config.num_agents, config.random_seed)

        scenario = Scenario(description=scenario_text.strip())
        new_posts = generate_scenario_posts(st.session_state.agents, scenario, config)
        st.session_state.events.append(scenario)
        st.session_state.posts.extend(new_posts)
        st.success(f"Scenario injected. {len(new_posts)} agents posted a first reaction.")
        st.rerun()

with col2:
    st.info("Agents will decide whether to post about this, what angle to take, and how they engage with each other.")

st.divider()

# Live state views
tab1, tab2, tab3 = st.tabs(["Agents", "Global Feed", "Algorithm & Analytics"])

with tab1:
    st.subheader("Current Agents")
    if st.session_state.agents:
        for agent in st.session_state.agents:
            follows = sum(
                1
                for other in st.session_state.agents
                if other.id in agent.following_ids
            )
            st.write(
                f"**@{agent.username}** — {agent.persona}  \n"
                f"Interests: {', '.join(agent.interests)} | Following: {follows}"
            )
    else:
        st.write("No agents yet. Click 'Initialize Agents' below to create the society.")

    if st.button("Initialize / Re-roll Agents"):
        st.session_state.agents = create_default_agents(num_agents, config.random_seed)
        st.session_state.posts = []
        st.session_state.events = []
        st.session_state.selected_agent_id = st.session_state.agents[0].id
        st.success(f"Created {num_agents} agents.")
        st.rerun()

with tab2:
    st.subheader("Global Activity (all posts)")
    if st.session_state.posts:
        for post in reversed(st.session_state.posts[-25:]):
            author = post.author_username or post.author_id
            tags = ", ".join(post.topic_tags)
            st.write(f"**@{author}** • {post.text}")
            st.caption(f"Topics: {tags}")
    else:
        st.write("No posts yet. Inject a scenario or let agents generate content.")

with tab3:
    st.subheader("Current Algorithm Configuration")
    st.json({
        "in_network_weight": config.in_network_weight,
        "discovery_weight": config.discovery_weight,
        "author_diversity_penalty": config.author_diversity_penalty,
        "reply_boost": config.reply_boost,
        "model": f"{config.model_provider} / {config.model_name}",
    })
    st.caption(
        "These values are live. Changing sliders above immediately changes the ranked feed below."
    )

    st.subheader("Personalized Feed Preview")
    if st.session_state.agents and st.session_state.posts:
        agent_lookup = {agent.id: agent for agent in st.session_state.agents}
        selected_id = st.selectbox(
            "View feed as",
            options=[agent.id for agent in st.session_state.agents],
            format_func=lambda agent_id: f"@{agent_lookup[agent_id].username}",
            key="selected_agent_id",
        )
        viewer = agent_lookup[selected_id]
        ranked_feed = rank_feed(viewer, st.session_state.posts, config)

        for item in ranked_feed[:10]:
            author = item.post.author_username or item.post.author_id
            st.write(f"**{item.score:.3f}** | **@{author}**: {item.post.text}")
            st.caption(item.reason)
    elif st.session_state.agents:
        st.write("Inject a scenario to generate posts and preview a ranked feed.")
    else:
        st.write("Initialize agents first, then inject a scenario.")

st.divider()
st.caption("This is early scaffolding. Next steps: real agent posting logic, personalized feed generation per agent, and proper ranking that respects the knobs above.")
