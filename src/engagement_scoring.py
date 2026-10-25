"""Explainable group-level activity scoring from fused modality features."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EngagementScoringConfig:
    """Weights and decision boundaries for an activity proxy, not engagement."""

    presence_weight: float = 0.30
    movement_weight: float = 0.15
    inactivity_weight: float = 0.20
    speech_weight: float = 0.35
    movement_reference_pixels_per_second: float = 50.0
    minimum_available_weight: float = 0.50
    low_activity_threshold: float = 35.0
    high_activity_threshold: float = 65.0


@dataclass(frozen=True)
class ActivityWindowScore:
    window_index: int
    start_seconds: float
    end_seconds: float
    activity_score: float | None
    activity_band: str
    confidence: str
    available_evidence_weight: float
    visual_presence_score: float | None
    movement_score: float | None
    visual_non_inactivity_score: float | None
    potential_speech_score: float | None
    contributing_rules: str
    explanation: str


@dataclass(frozen=True)
class ActivityTrend:
    start_seconds: float
    end_seconds: float
    activity_band: str
    window_count: int
    mean_activity_score: float


@dataclass(frozen=True)
class EngagementScoringResult:
    windows: tuple[ActivityWindowScore, ...]
    trends: tuple[ActivityTrend, ...]
    mean_activity_score: float | None
    scored_window_count: int
    insufficient_evidence_window_count: int
    summary: str


@dataclass(frozen=True)
class _FusedWindow:
    index: int
    start: float
    end: float
    presence_seconds: float | None
    movement_pixels_per_second: float | None
    inactivity_seconds: float | None
    visual_coverage_seconds: float | None
    speech_ratio: float | None
    audio_coverage_seconds: float | None


def score_fused_activity(
    fused_windows_csv_path: str | Path,
    *,
    config: EngagementScoringConfig = EngagementScoringConfig(),
    scores_csv_path: str | Path | None = None,
    trends_csv_path: str | Path | None = None,
) -> EngagementScoringResult:
    """Score group-level observed activity without inferring personal states.

    The 0–100 score is a configurable activity proxy. Features with missing
    coverage are omitted and the remaining weights are normalized. If the
    available configured weight is below ``minimum_available_weight``, the
    score is left unavailable rather than treating missing values as zero.
    """
    _validate_config(config)
    source = Path(fused_windows_csv_path).resolve()
    scores_path = _resolve_output(scores_csv_path)
    trends_path = _resolve_output(trends_csv_path)
    if scores_path == source or trends_path == source:
        raise ValueError("Score outputs must not overwrite the fused input CSV")
    if scores_path is not None and scores_path == trends_path:
        raise ValueError("Score and trend outputs must be different paths")

    fused_windows = _read_fused_windows(source)
    scored = tuple(_score_window(window, config) for window in fused_windows)
    trends = _build_trends(scored)
    available_scores = [
        window.activity_score
        for window in scored
        if window.activity_score is not None
    ]
    mean_score = (
        sum(available_scores) / len(available_scores)
        if available_scores
        else None
    )
    insufficient = sum(window.activity_score is None for window in scored)
    summary = _build_summary(scored, mean_score, insufficient)
    result = EngagementScoringResult(
        windows=scored,
        trends=trends,
        mean_activity_score=mean_score,
        scored_window_count=len(available_scores),
        insufficient_evidence_window_count=insufficient,
        summary=summary,
    )

    if scores_path is not None:
        _write_records(scores_path, scored, ActivityWindowScore)
    if trends_path is not None:
        _write_records(trends_path, trends, ActivityTrend)
    return result


def _resolve_output(value: str | Path | None) -> Path | None:
    return Path(value).resolve() if value is not None else None


def _validate_config(config: EngagementScoringConfig) -> None:
    weights = (
        config.presence_weight,
        config.movement_weight,
        config.inactivity_weight,
        config.speech_weight,
    )
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Scoring weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Scoring weights must sum to 1.0")
    if (
        not math.isfinite(config.minimum_available_weight)
        or config.minimum_available_weight <= 0
        or config.minimum_available_weight > 1
    ):
        raise ValueError("minimum_available_weight must be in (0, 1]")
    if (
        not math.isfinite(config.movement_reference_pixels_per_second)
        or config.movement_reference_pixels_per_second <= 0
    ):
        raise ValueError(
            "movement_reference_pixels_per_second must be finite and greater than zero"
        )
    if (
        not math.isfinite(config.low_activity_threshold)
        or not math.isfinite(config.high_activity_threshold)
        or config.low_activity_threshold < 0
        or config.high_activity_threshold > 100
        or config.low_activity_threshold >= config.high_activity_threshold
    ):
        raise ValueError(
            "Activity thresholds must satisfy 0 <= low < high <= 100"
        )


def _read_fused_windows(path: Path) -> list[_FusedWindow]:
    required = {
        "window_index",
        "start_seconds",
        "end_seconds",
        "estimated_track_presence_seconds",
        "mean_movement_pixels_per_second",
        "visual_inactivity_seconds",
        "visual_coverage_seconds",
        "audio_coverage_seconds",
        "potential_speech_ratio",
    }
    windows = []
    seen_indices: set[int] = set()
    previous_end = 0.0
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV is missing its header row")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: CSV contains duplicate column names")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{path}: missing required columns: {', '.join(sorted(missing))}"
            )

        for row_number, row in enumerate(reader, start=2):
            index = _parse_index(row.get("window_index"), row_number, path)
            if index in seen_indices:
                raise ValueError(f"{path}: duplicate window_index {index}")
            seen_indices.add(index)
            start = _parse_number(row.get("start_seconds"), "start_seconds", row_number, path)
            end = _parse_number(row.get("end_seconds"), "end_seconds", row_number, path)
            if start < 0 or end <= start:
                raise ValueError(
                    f"{path}: invalid window time range on row {row_number}"
                )
            if windows and start < previous_end:
                raise ValueError(
                    f"{path}: overlapping or unsorted windows on row {row_number}"
                )
            previous_end = end

            presence = _parse_optional_number(
                row.get("estimated_track_presence_seconds"),
                "estimated_track_presence_seconds",
                row_number,
                path,
            )
            movement = _parse_optional_number(
                row.get("mean_movement_pixels_per_second"),
                "mean_movement_pixels_per_second",
                row_number,
                path,
            )
            inactivity = _parse_optional_number(
                row.get("visual_inactivity_seconds"),
                "visual_inactivity_seconds",
                row_number,
                path,
            )
            visual_coverage = _parse_optional_number(
                row.get("visual_coverage_seconds"),
                "visual_coverage_seconds",
                row_number,
                path,
            )
            audio_coverage = _parse_optional_number(
                row.get("audio_coverage_seconds"),
                "audio_coverage_seconds",
                row_number,
                path,
            )
            speech_ratio = _parse_optional_number(
                row.get("potential_speech_ratio"),
                "potential_speech_ratio",
                row_number,
                path,
            )
            for name, value in (
                ("presence", presence),
                ("movement", movement),
                ("inactivity", inactivity),
                ("visual_coverage", visual_coverage),
                ("audio_coverage", audio_coverage),
            ):
                if value is not None and value < 0:
                    raise ValueError(
                        f"{path}: {name} cannot be negative on row {row_number}"
                    )
            if speech_ratio is not None and not 0 <= speech_ratio <= 1:
                raise ValueError(
                    f"{path}: potential_speech_ratio must be within [0, 1] "
                    f"on row {row_number}"
                )
            if visual_coverage is None and inactivity is not None:
                raise ValueError(
                    f"{path}: visual inactivity has no visual coverage on row {row_number}"
                )
            if visual_coverage == 0 and any(
                value is not None for value in (presence, movement, inactivity)
            ):
                raise ValueError(
                    f"{path}: visual features conflict with zero visual coverage "
                    f"on row {row_number}"
                )
            if audio_coverage == 0 and speech_ratio is not None:
                raise ValueError(
                    f"{path}: speech ratio conflicts with zero audio coverage "
                    f"on row {row_number}"
                )
            windows.append(
                _FusedWindow(
                    index,
                    start,
                    end,
                    presence,
                    movement,
                    inactivity,
                    visual_coverage,
                    speech_ratio,
                    audio_coverage,
                )
            )
    return windows


def _parse_index(value: str | None, row: int, path: Path) -> int:
    text = (value or "").strip()
    try:
        index = int(text)
    except ValueError as error:
        raise ValueError(
            f"{path}: invalid window_index on row {row}: {text!r}"
        ) from error
    if index < 0:
        raise ValueError(f"{path}: window_index must be non-negative on row {row}")
    return index


def _parse_number(
    value: str | None,
    name: str,
    row: int,
    path: Path,
) -> float:
    number = _parse_optional_number(value, name, row, path)
    if number is None:
        raise ValueError(f"{path}: missing {name} on row {row}")
    return number


def _parse_optional_number(
    value: str | None,
    name: str,
    row: int,
    path: Path,
) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(f"{path}: invalid {name} on row {row}: {text!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{path}: {name} must be finite on row {row}")
    return number


def _score_window(
    window: _FusedWindow,
    config: EngagementScoringConfig,
) -> ActivityWindowScore:
    duration = window.end - window.start
    features: list[tuple[str, float, float, str]] = []
    if window.visual_coverage_seconds and window.visual_coverage_seconds > 0:
        coverage = window.visual_coverage_seconds
        if window.presence_seconds is not None:
            presence_denominator = coverage or duration
            presence_score = (
                min(1.0, window.presence_seconds / presence_denominator) * 100
            )
            features.append(
                (
                    "visual_presence",
                    presence_score,
                    config.presence_weight,
                    "visual_presence: min(track-presence seconds / window duration, 1)",
                )
            )
        if window.movement_pixels_per_second is not None:
            movement_score = (
                min(
                    window.movement_pixels_per_second
                    / config.movement_reference_pixels_per_second,
                    1.0,
                )
                * 100
            )
            features.append(
                (
                    "movement",
                    movement_score,
                    config.movement_weight,
                    "movement: min(centroid speed / configured reference speed, 1)",
                )
            )
        if window.inactivity_seconds is not None:
            inactivity_ratio = min(
                window.inactivity_seconds / coverage,
                1.0,
            )
            features.append(
                (
                    "visual_non_inactivity",
                    (1.0 - inactivity_ratio) * 100,
                    config.inactivity_weight,
                    "visual_non_inactivity: 1 - observed visual inactivity / coverage",
                )
            )
    if (
        window.audio_coverage_seconds
        and window.audio_coverage_seconds > 0
        and window.speech_ratio is not None
    ):
        features.append(
            (
                "potential_speech",
                window.speech_ratio * 100,
                config.speech_weight,
                "potential_speech: heuristic speech-flagged audio / covered audio",
            )
        )

    available_weight = sum(weight for _, _, weight, _ in features)
    if available_weight < config.minimum_available_weight:
        return ActivityWindowScore(
            window.index,
            window.start,
            window.end,
            None,
            "insufficient_evidence",
            "low",
            available_weight,
            next((score for name, score, _, _ in features if name == "visual_presence"), None),
            next((score for name, score, _, _ in features if name == "movement"), None),
            next((score for name, score, _, _ in features if name == "visual_non_inactivity"), None),
            next((score for name, score, _, _ in features if name == "potential_speech"), None),
            "; ".join(rule for _, _, _, rule in features)
            or "No covered scoring features were available.",
            (
                f"Only {available_weight:.2f} configured feature weight is "
                f"available; at least {config.minimum_available_weight:.2f} "
                "is required. Missing observations are not scored as zero."
            ),
        )

    score = sum(value * weight for _, value, weight, _ in features) / available_weight
    if score < config.low_activity_threshold:
        band = "lower_observed_activity"
    elif score >= config.high_activity_threshold:
        band = "higher_observed_activity"
    else:
        band = "moderate_observed_activity"

    modalities = set()
    if any(
        name in {"visual_presence", "movement", "visual_non_inactivity"}
        and weight > 0
        for name, _, weight, _ in features
    ):
        modalities.add("visual")
    if any(
        name == "potential_speech" and weight > 0
        for name, _, weight, _ in features
    ):
        modalities.add("audio")
    confidence = (
        "moderate"
        if modalities == {"visual", "audio"}
        and available_weight >= 0.75
        else "low"
    )
    rules = "; ".join(rule for _, _, _, rule in features)
    details = ", ".join(
        f"{name}={value:.1f}/100 (weight {weight:.2f})"
        for name, value, weight, _ in features
    )
    explanation = (
        f"Weighted observed-activity proxy is {score:.1f}/100 from {details}. "
        f"Available configured weight: {available_weight:.2f}. "
        "This is not a measure of an individual's attention, understanding, "
        "participation, or learning."
    )
    return ActivityWindowScore(
        window.index,
        window.start,
        window.end,
        score,
        band,
        confidence,
        available_weight,
        next((value for name, value, _, _ in features if name == "visual_presence"), None),
        next((value for name, value, _, _ in features if name == "movement"), None),
        next((value for name, value, _, _ in features if name == "visual_non_inactivity"), None),
        next((value for name, value, _, _ in features if name == "potential_speech"), None),
        rules,
        explanation,
    )


def _build_trends(
    windows: tuple[ActivityWindowScore, ...],
) -> tuple[ActivityTrend, ...]:
    trends: list[ActivityTrend] = []
    active_band: str | None = None
    run: list[ActivityWindowScore] = []
    for window in windows:
        band = window.activity_band
        if band != active_band:
            if active_band in {"lower_observed_activity", "higher_observed_activity"}:
                trends.append(_make_trend(active_band, run))
            run = []
            active_band = band
        if band in {"lower_observed_activity", "higher_observed_activity"}:
            run.append(window)
        else:
            run = []
    if active_band in {"lower_observed_activity", "higher_observed_activity"}:
        trends.append(_make_trend(active_band, run))
    return tuple(trends)


def _make_trend(band: str, windows: list[ActivityWindowScore]) -> ActivityTrend:
    return ActivityTrend(
        start_seconds=windows[0].start_seconds,
        end_seconds=windows[-1].end_seconds,
        activity_band=band,
        window_count=len(windows),
        mean_activity_score=sum(
            window.activity_score for window in windows
            if window.activity_score is not None
        ) / len(windows),
    )


def _build_summary(
    windows: tuple[ActivityWindowScore, ...],
    mean_score: float | None,
    insufficient_count: int,
) -> str:
    if mean_score is None:
        return (
            "No windows met the minimum evidence requirement; no activity "
            "estimate was produced."
        )
    return (
        f"Mean group-level observed-activity proxy: {mean_score:.1f}/100 across "
        f"{len(windows) - insufficient_count} scored windows; "
        f"{insufficient_count} windows had insufficient evidence. "
        "Scores are indirect activity summaries, not validated engagement "
        "or individual-level judgments."
    )


def _write_records(path: Path, records: tuple[object, ...], record_type: type) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(record_type.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    name: (
                        ""
                        if value is None
                        else str(value).lower()
                        if isinstance(value, bool)
                        else f"{value:.6f}"
                        if isinstance(value, float)
                        else value
                    )
                    for name, value in vars(record).items()
                }
            )
