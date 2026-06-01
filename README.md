# x-algorithm-simulator

A controllable, local-first simulator of an X-style social platform: synthetic agents, scenario injection ("god mode"), and a tunable feed algorithm you can poke at directly.

## What It Does Today

The current prototype is intentionally small, but it is runnable:

- Creates a reproducible society of synthetic agents with personas, interests, and follow edges.
- Lets you inject a scenario, such as a fuel price shock or policy announcement.
- Generates a first wave of agent reactions, then advances the world through simulation ticks.
- Ranks posts for an individual agent using live algorithm knobs and per-score breakdowns.
- Shows why each post ranked where it did: topic match, network/discovery boost, recency, social proof, reply boost, negative feedback, and author diversity.
- Tracks analytics like scenario reach, top amplifiers, topic spread, feed diversity, and tick activity.
- Exports/imports complete run JSON so experiments can be replayed.
- Compares baseline vs variant ranking configs from the same starting society and scenario.
- Provides role-based model settings for agent decisions, scenario reactions, ranking assistance, and analytics summaries.

That gives the project a working baseline before adding LLM-backed behavior.

## Product Direction

The goal is to make the platform itself the experiment surface, not just a ranking model demo. A good run should let you ask questions like:

- What happens when discovery is weighted higher than in-network content?
- Which agents amplify a scenario first, and which ignore it?
- Do diversity penalties reduce echo chambers or just flatten the feed?
- How do different network structures change virality?
- How much do LLM-generated personas change outcomes compared with deterministic agents?

## Repository Map

- `xsim/core/models.py` - simulation data types: agents, posts, scenarios, engagements, feed items.
- `xsim/core/state.py` - complete experiment state plus JSON save/load.
- `xsim/core/engine.py` - multi-tick simulation runner.
- `xsim/core/behavior.py` - deterministic and optional LLM-backed agent behavior.
- `xsim/core/simulation.py` - starter society, scenario reaction, topic inference, ranking.
- `xsim/core/analytics.py` - reach, amplifier, topic-spread, and feed-diversity analytics.
- `xsim/core/comparison.py` - baseline vs variant experiment cloning and comparison metrics.
- `xsim/llm.py` - LLM abstraction for Ollama and OpenAI-compatible providers.
- `simulator/app.py` - Streamlit experiment cockpit.
- `references/` - copied architecture reference material from X-style systems. These files are not required to run xsim and are excluded from active lint/test loops.

## Run It

```bash
# Using uv
uv sync
uv run streamlit run simulator/app.py

# Or with pip
pip install -e ".[dev]"
streamlit run simulator/app.py
```

The app opens with a collapsed sidebar for simulation and ranking controls. Start by creating a society, injecting a scenario, then stepping the simulation.

## Development

```bash
uv run pytest
uv run ruff check .
```

## LLM Strategy

The intended path is Ollama-first with easy API fallback:

- Fully local: use Ollama with a model like `qwen2.5:7b`, `qwen2.5:14b`, or `llama3.1`.
- Higher quality or faster iteration: use an OpenAI-compatible API such as Groq, OpenRouter, Together, Fireworks, vLLM, or OpenAI.

The deterministic simulation should stay available as the baseline. LLMs should improve agent realism, not become required for every experiment.

## Next Milestones

1. Add richer network visualization for follow edges, amplification paths, and topic clusters.
2. Expand behavior policies so deterministic agents can have distinct temperaments, not just personas.
3. Add optional LLM batch generation with caching so high-quality runs stay affordable.
4. Add scenario templates and seed presets for repeatable demos.
5. Add saved comparison reports so baseline/variant outcomes can be shared directly.

## Design Principles

- Local-first and cheap to run.
- Deterministic baseline before probabilistic/LLM behavior.
- Every ranking knob should have visible impact.
- Every generated outcome should be inspectable enough to debug.
- Reference the real X-style architecture, but keep the simulator small, hackable, and understandable.

## Optional Local Model Setup

```bash
ollama pull qwen2.5:7b
```

LLM-backed posting is planned, but the current simulator does not require a model.

## License

Apache 2.0 (with attribution to the original X open-source For You components this draws inspiration from).
