# x-algorithm-simulator

A controllable, local-first simulator of an X-style social platform: synthetic agents, scenario injection ("god mode"), and a tunable feed algorithm you can poke at directly.

## What It Does Today

The current prototype is intentionally small, but it is runnable:

- Creates a reproducible society of synthetic agents with personas, interests, and follow edges.
- Lets you inject a scenario, such as a fuel price shock or policy announcement.
- Generates a first wave of deterministic agent reactions.
- Ranks posts for an individual agent using live algorithm knobs.
- Shows why each post ranked where it did: topic match, in-network boost, discovery boost, reply boost, and author diversity penalty.

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
- `xsim/core/simulation.py` - deterministic starter society, scenario reaction, topic inference, ranking.
- `xsim/llm.py` - LLM abstraction for Ollama and OpenAI-compatible providers. Not wired into the simulation loop yet.
- `simulator/app.py` - Streamlit control panel and live visualization.
- `phoenix/`, `home-mixer/`, `candidate-pipeline/`, `thunder/`, `grox/` - reference material from the X-style architecture this project draws from. These are not required to run the simulator.

## Run It

```bash
# Using uv
uv sync
uv run streamlit run simulator/app.py

# Or with pip
pip install -e ".[dev]"
streamlit run simulator/app.py
```

The app opens with a sidebar for simulation and ranking controls. Start by clicking **Initialize / Re-roll Agents**, then inject a scenario.

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

1. Add an experiment state object that owns agents, posts, scenarios, and engagements instead of storing loose lists in Streamlit session state.
2. Replace template-based scenario reactions with optional LLM-generated posts behind the existing `xsim.llm` abstraction.
3. Add per-step simulation ticks: agents read ranked feeds, decide whether to engage, and optionally create replies or quotes.
4. Add network analytics: top amplifiers, topic spread, feed diversity, polarization/cluster views, and scenario reach.
5. Add run export/import so experiments are reproducible and shareable.
6. Split the reference architecture directories into documented reference assets or move them under a clear `references/` folder.

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
