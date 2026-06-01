"""Side-by-side experiment comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass

from xsim.core.analytics import feed_diversity, reach_summary
from xsim.core.engine import SimulationEngine
from xsim.core.models import SimulationConfig
from xsim.core.state import ExperimentState


@dataclass(frozen=True)
class ComparisonMetrics:
    """Compact outcome metrics for one branch of a comparison."""

    label: str
    posts: int
    engagements: int
    ticks: int
    reach_ratio: float
    reached_agents: int
    discovery_ratio: float
    unique_feed_authors: int
    visible_topics: int


@dataclass(frozen=True)
class ExperimentComparison:
    """Result of running two cloned experiments forward."""

    baseline: ExperimentState
    variant: ExperimentState
    baseline_metrics: ComparisonMetrics
    variant_metrics: ComparisonMetrics

    def metric_rows(self) -> list[dict[str, str | int | float]]:
        """Return UI-friendly rows for a side-by-side metric table."""
        return [
            _comparison_row("Posts", self.baseline_metrics.posts, self.variant_metrics.posts),
            _comparison_row(
                "Engagements",
                self.baseline_metrics.engagements,
                self.variant_metrics.engagements,
            ),
            _comparison_row("Ticks", self.baseline_metrics.ticks, self.variant_metrics.ticks),
            _comparison_row(
                "Reach",
                self.baseline_metrics.reach_ratio,
                self.variant_metrics.reach_ratio,
            ),
            _comparison_row(
                "Discovery mix",
                self.baseline_metrics.discovery_ratio,
                self.variant_metrics.discovery_ratio,
            ),
            _comparison_row(
                "Unique feed authors",
                self.baseline_metrics.unique_feed_authors,
                self.variant_metrics.unique_feed_authors,
            ),
            _comparison_row(
                "Visible feed topics",
                self.baseline_metrics.visible_topics,
                self.variant_metrics.visible_topics,
            ),
        ]


def clone_experiment_state(state: ExperimentState) -> ExperimentState:
    """Deep-clone a state through the same JSON path used for save/replay."""
    return ExperimentState.from_json(state.to_json())


def run_comparison(
    base_state: ExperimentState,
    baseline_config: SimulationConfig,
    variant_config: SimulationConfig,
    steps: int,
) -> ExperimentComparison:
    """Clone a run into baseline/variant branches and advance both."""
    baseline = clone_experiment_state(base_state)
    variant = clone_experiment_state(base_state)
    baseline.config = baseline_config
    variant.config = variant_config

    SimulationEngine(baseline).run(steps)
    SimulationEngine(variant).run(steps)

    return ExperimentComparison(
        baseline=baseline,
        variant=variant,
        baseline_metrics=summarize_comparison_branch("Baseline", baseline),
        variant_metrics=summarize_comparison_branch("Variant", variant),
    )


def summarize_comparison_branch(label: str, state: ExperimentState) -> ComparisonMetrics:
    """Compute headline metrics for one comparison branch."""
    reach = reach_summary(state)
    if state.agents:
        diversity = feed_diversity(state, state.agents[0])
        discovery_ratio = diversity.discovery_ratio
        unique_feed_authors = diversity.unique_authors
        visible_topics = diversity.topic_count
    else:
        discovery_ratio = 0.0
        unique_feed_authors = 0
        visible_topics = 0

    return ComparisonMetrics(
        label=label,
        posts=len(state.posts),
        engagements=len(state.engagements),
        ticks=state.current_tick,
        reach_ratio=reach.reach_ratio,
        reached_agents=reach.reached_agents,
        discovery_ratio=discovery_ratio,
        unique_feed_authors=unique_feed_authors,
        visible_topics=visible_topics,
    )


def _comparison_row(
    metric: str,
    baseline: int | float,
    variant: int | float,
) -> dict[str, str | int | float]:
    return {
        "metric": metric,
        "baseline": baseline,
        "variant": variant,
        "delta": variant - baseline,
    }
