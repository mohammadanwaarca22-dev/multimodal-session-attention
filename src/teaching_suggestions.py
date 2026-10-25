"""Generate explainable, non-prescriptive instructor suggestions from trends."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TeachingSuggestionConfig:
    low_activity_threshold: float = 35.0
    high_activity_threshold: float = 65.0
    minimum_low_activity_duration_seconds: float = 30.0
    sudden_drop_points: float = 20.0
    elevated_noise_dbfs: float = -20.0


@dataclass(frozen=True)
class TeachingSuggestion:
    suggestion_type: str
    priority: str
    start_seconds: float
    end_seconds: float
    observation: str
    suggestion: str
    evidence: str
    confidence: str
    human_review_recommended: bool


@dataclass(frozen=True)
class TeachingSuggestionResult:
    suggestions: tuple[TeachingSuggestion, ...]
    scored_window_count: int
    unscored_window_count: int
    summary: str


@dataclass(frozen=True)
class _ScoreWindow:
    index: int
    start: float
    end: float
    score: float | None


@dataclass(frozen=True)
class _NoiseWindow:
    index: int
    start: float
    end: float
    noise_dbfs: float | None
    coverage_seconds: float


def generate_teaching_suggestions(
    activity_scores_csv_path: str | Path,
    *,
    fused_windows_csv_path: str | Path | None = None,
    config: TeachingSuggestionConfig = TeachingSuggestionConfig(),
    suggestions_csv_path: str | Path | None = None,
) -> TeachingSuggestionResult:
    """Generate instructor-facing suggestions from temporal activity proxies.

    Suggestions refer to group-level observed activity and relative audio
    levels. They do not infer individual engagement, attention, understanding,
    or learning. Elevated audio levels are only considered when the fused
    input has positive audio coverage.
    """
    _validate_config(config)
    scores_path = Path(activity_scores_csv_path).resolve()
    fused_path = (
        Path(fused_windows_csv_path).resolve()
        if fused_windows_csv_path is not None
        else None
    )
    output_path = (
        Path(suggestions_csv_path).resolve()
        if suggestions_csv_path is not None
        else None
    )
    inputs = {scores_path}
    if fused_path is not None:
        inputs.add(fused_path)
    if output_path is not None and output_path in inputs:
        raise ValueError("Suggestions output must not overwrite an input CSV")

    score_windows = _read_score_windows(scores_path)
    noise_windows = _read_noise_windows(fused_path) if fused_path else []
    if fused_path is not None:
        _validate_aligned_windows(score_windows, noise_windows, fused_path)

    suggestions: list[TeachingSuggestion] = []
    suggestions.extend(_low_activity_suggestions(score_windows, config))
    suggestions.extend(_drop_suggestions(score_windows, config))
    suggestions.extend(_high_activity_suggestions(score_windows, config))
    suggestions.extend(_noise_suggestions(noise_windows, config))
    suggestions.sort(key=lambda item: (item.start_seconds, item.suggestion_type))

    scored_count = sum(window.score is not None for window in score_windows)
    unscored_count = len(score_windows) - scored_count
    if suggestions:
        summary = (
            f"Generated {len(suggestions)} instructor review suggestion(s) "
            f"from {scored_count} scored and {unscored_count} unscored windows. "
            "Suggestions describe indirect group-level signals and require "
            "instructor judgment."
        )
    elif not scored_count and not noise_windows:
        summary = (
            "No scored activity or covered audio observations were available; "
            "no teaching suggestions were generated."
        )
    else:
        summary = (
            f"No suggestion rules were triggered across {scored_count} scored "
            f"and {unscored_count} unscored windows. This does not establish "
            "that engagement, attention, or learning was high or low."
        )

    result = TeachingSuggestionResult(
        suggestions=tuple(suggestions),
        scored_window_count=scored_count,
        unscored_window_count=unscored_count,
        summary=summary,
    )
    if output_path is not None:
        _write_suggestions(output_path, result.suggestions)
    return result


def _validate_config(config: TeachingSuggestionConfig) -> None:
    if (
        not math.isfinite(config.low_activity_threshold)
        or not math.isfinite(config.high_activity_threshold)
        or config.low_activity_threshold < 0
        or config.high_activity_threshold > 100
        or config.low_activity_threshold >= config.high_activity_threshold
    ):
        raise ValueError("Activity thresholds must satisfy 0 <= low < high <= 100")
    if (
        not math.isfinite(config.minimum_low_activity_duration_seconds)
        or config.minimum_low_activity_duration_seconds <= 0
    ):
        raise ValueError(
            "minimum_low_activity_duration_seconds must be finite and greater than zero"
        )
    if not math.isfinite(config.sudden_drop_points) or config.sudden_drop_points <= 0:
        raise ValueError("sudden_drop_points must be finite and greater than zero")
    if not math.isfinite(config.elevated_noise_dbfs):
        raise ValueError("elevated_noise_dbfs must be finite")


def _read_score_windows(path: Path) -> list[_ScoreWindow]:
    required = {
        "window_index",
        "start_seconds",
        "end_seconds",
        "activity_score",
        "activity_band",
    }
    windows: list[_ScoreWindow] = []
    seen_indices: set[int] = set()
    previous_end = 0.0
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_headers(reader.fieldnames, required, path)
        for row_number, row in enumerate(reader, start=2):
            index = _parse_index(row.get("window_index"), row_number, path)
            if index in seen_indices:
                raise ValueError(f"{path}: duplicate window_index {index}")
            seen_indices.add(index)
            start = _parse_number(row.get("start_seconds"), "start_seconds", row_number, path)
            end = _parse_number(row.get("end_seconds"), "end_seconds", row_number, path)
            if start < 0 or end <= start:
                raise ValueError(f"{path}: invalid time range on row {row_number}")
            if windows and start < previous_end:
                raise ValueError(f"{path}: overlapping or unsorted windows")
            previous_end = end
            score = _parse_optional_number(
                row.get("activity_score"), "activity_score", row_number, path
            )
            if score is not None and not 0 <= score <= 100:
                raise ValueError(
                    f"{path}: activity_score must be within [0, 100] "
                    f"on row {row_number}"
                )
            band = (row.get("activity_band") or "").strip()
            if score is None and band != "insufficient_evidence":
                raise ValueError(
                    f"{path}: missing activity_score must use the "
                    "'insufficient_evidence' band"
                )
            if score is not None and band == "insufficient_evidence":
                raise ValueError(
                    f"{path}: scored window cannot use insufficient_evidence band"
                )
            windows.append(_ScoreWindow(index, start, end, score))
    return windows


def _read_noise_windows(path: Path) -> list[_NoiseWindow]:
    required = {
        "window_index",
        "start_seconds",
        "end_seconds",
        "audio_coverage_seconds",
        "mean_noise_level_dbfs",
    }
    windows = []
    seen_indices: set[int] = set()
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_headers(reader.fieldnames, required, path)
        for row_number, row in enumerate(reader, start=2):
            index = _parse_index(row.get("window_index"), row_number, path)
            if index in seen_indices:
                raise ValueError(f"{path}: duplicate window_index {index}")
            seen_indices.add(index)
            start = _parse_number(row.get("start_seconds"), "start_seconds", row_number, path)
            end = _parse_number(row.get("end_seconds"), "end_seconds", row_number, path)
            coverage = _parse_number(
                row.get("audio_coverage_seconds"),
                "audio_coverage_seconds",
                row_number,
                path,
            )
            noise = _parse_optional_number(
                row.get("mean_noise_level_dbfs"),
                "mean_noise_level_dbfs",
                row_number,
                path,
            )
            if start < 0 or end <= start:
                raise ValueError(f"{path}: invalid time range on row {row_number}")
            if coverage < 0 or coverage > end - start + 1e-6:
                raise ValueError(
                    f"{path}: invalid audio coverage on row {row_number}"
                )
            if coverage == 0 and noise is not None:
                raise ValueError(
                    f"{path}: noise level present without audio coverage "
                    f"on row {row_number}"
                )
            if coverage > 0 and noise is None:
                raise ValueError(
                    f"{path}: missing noise level despite audio coverage "
                    f"on row {row_number}"
                )
            windows.append(_NoiseWindow(index, start, end, noise, coverage))
    return windows


def _validate_headers(
    headers: list[str] | None,
    required: set[str],
    path: Path,
) -> None:
    if headers is None:
        raise ValueError(f"{path}: CSV is missing its header row")
    if len(headers) != len(set(headers)):
        raise ValueError(f"{path}: CSV contains duplicate column names")
    missing = required.difference(headers)
    if missing:
        raise ValueError(
            f"{path}: missing required columns: {', '.join(sorted(missing))}"
        )


def _parse_index(value: str | None, row: int, path: Path) -> int:
    text = (value or "").strip()
    try:
        index = int(text)
    except ValueError as error:
        raise ValueError(
            f"{path}: invalid window_index on row {row}: {text!r}"
        ) from error
    if index < 0:
        raise ValueError(f"{path}: window_index must be non-negative")
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


def _validate_aligned_windows(
    scores: list[_ScoreWindow],
    noise: list[_NoiseWindow],
    fused_path: Path,
) -> None:
    noise_by_index = {window.index: window for window in noise}
    for score in scores:
        fused = noise_by_index.get(score.index)
        if fused is None:
            raise ValueError(
                f"{fused_path}: missing fused window_index {score.index}"
            )
        if not (
            math.isclose(score.start, fused.start, abs_tol=1e-6)
            and math.isclose(score.end, fused.end, abs_tol=1e-6)
        ):
            raise ValueError(
                f"{fused_path}: window {score.index} timestamps do not align "
                "with activity scores"
            )


def _low_activity_suggestions(
    windows: list[_ScoreWindow],
    config: TeachingSuggestionConfig,
) -> list[TeachingSuggestion]:
    suggestions = []
    run: list[_ScoreWindow] = []
    for window in windows:
        if _is_low(window, config):
            if run and not _contiguous(run[-1], window):
                suggestions.extend(_finish_low_run(run, config))
                run = []
            run.append(window)
        else:
            suggestions.extend(_finish_low_run(run, config))
            run = []
    suggestions.extend(_finish_low_run(run, config))
    return suggestions


def _is_low(window: _ScoreWindow, config: TeachingSuggestionConfig) -> bool:
    return window.score is not None and window.score < config.low_activity_threshold


def _finish_low_run(
    run: list[_ScoreWindow],
    config: TeachingSuggestionConfig,
) -> list[TeachingSuggestion]:
    if not run or run[-1].end - run[0].start < config.minimum_low_activity_duration_seconds:
        return []
    mean_score = sum(window.score for window in run if window.score is not None) / len(run)
    duration = run[-1].end - run[0].start
    return [
        TeachingSuggestion(
            suggestion_type="sustained_lower_activity_proxy",
            priority="review",
            start_seconds=run[0].start,
            end_seconds=run[-1].end,
            observation=(
                f"The group-level activity proxy stayed below "
                f"{config.low_activity_threshold:.1f}/100 for "
                f"{duration:.1f} seconds (mean {mean_score:.1f}/100)."
            ),
            suggestion=(
                "Consider a brief change of pace, a short break, or an optional "
                "whole-class check-in; use your knowledge of the class to decide."
            ),
            evidence=(
                f"{len(run)} contiguous scored windows; configured lower-activity "
                f"threshold {config.low_activity_threshold:.1f}/100; "
                f"minimum duration {config.minimum_low_activity_duration_seconds:.1f}s."
            ),
            confidence="low",
            human_review_recommended=True,
        )
    ]


def _drop_suggestions(
    windows: list[_ScoreWindow],
    config: TeachingSuggestionConfig,
) -> list[TeachingSuggestion]:
    suggestions = []
    for previous, current in zip(windows, windows[1:]):
        if (
            previous.score is None
            or current.score is None
            or not _contiguous(previous, current)
        ):
            continue
        drop = previous.score - current.score
        if drop < config.sudden_drop_points:
            continue
        suggestions.append(
            TeachingSuggestion(
                suggestion_type="sudden_activity_proxy_drop",
                priority="review",
                start_seconds=previous.start,
                end_seconds=current.end,
                observation=(
                    f"The activity proxy decreased by {drop:.1f} points "
                    f"between adjacent windows ({previous.score:.1f} to "
                    f"{current.score:.1f})."
                ),
                suggestion=(
                    "Consider checking whether the class would benefit from a "
                    "brief recap, clarification, or change of activity."
                ),
                evidence=(
                    f"Adjacent scored windows; configured drop threshold "
                    f"{config.sudden_drop_points:.1f} points."
                ),
                confidence="low",
                human_review_recommended=True,
            )
        )
    return suggestions


def _high_activity_suggestions(
    windows: list[_ScoreWindow],
    config: TeachingSuggestionConfig,
) -> list[TeachingSuggestion]:
    suggestions = []
    run: list[_ScoreWindow] = []
    for window in windows:
        if window.score is not None and window.score >= config.high_activity_threshold:
            if run and not _contiguous(run[-1], window):
                suggestions.extend(_finish_high_run(run, config))
                run = []
            run.append(window)
        else:
            suggestions.extend(_finish_high_run(run, config))
            run = []
    suggestions.extend(_finish_high_run(run, config))
    return suggestions


def _finish_high_run(
    run: list[_ScoreWindow],
    config: TeachingSuggestionConfig,
) -> list[TeachingSuggestion]:
    if not run:
        return []
    mean_score = sum(window.score for window in run if window.score is not None) / len(run)
    return [
        TeachingSuggestion(
            suggestion_type="higher_activity_proxy_period",
            priority="information",
            start_seconds=run[0].start,
            end_seconds=run[-1].end,
            observation=(
                f"The activity proxy was at or above "
                f"{config.high_activity_threshold:.1f}/100 for this period "
                f"(mean {mean_score:.1f}/100)."
            ),
            suggestion=(
                "You may wish to note the activity format or material used "
                "during this interval when reflecting on the session."
            ),
            evidence=(
                f"{len(run)} contiguous scored windows; configured higher-activity "
                f"threshold {config.high_activity_threshold:.1f}/100."
            ),
            confidence="low",
            human_review_recommended=True,
        )
    ]


def _noise_suggestions(
    windows: list[_NoiseWindow],
    config: TeachingSuggestionConfig,
) -> list[TeachingSuggestion]:
    suggestions = []
    run: list[_NoiseWindow] = []
    for window in windows:
        elevated = (
            window.coverage_seconds > 0
            and window.noise_dbfs is not None
            and window.noise_dbfs >= config.elevated_noise_dbfs
        )
        if elevated:
            if run and not _contiguous_noise(run[-1], window):
                suggestions.extend(_finish_noise_run(run, config))
                run = []
            run.append(window)
        else:
            suggestions.extend(_finish_noise_run(run, config))
            run = []
    suggestions.extend(_finish_noise_run(run, config))
    return suggestions


def _finish_noise_run(
    run: list[_NoiseWindow],
    config: TeachingSuggestionConfig,
) -> list[TeachingSuggestion]:
    if not run:
        return []
    coverage = sum(window.coverage_seconds for window in run)
    mean_noise = (
        sum(
            window.noise_dbfs * window.coverage_seconds
            for window in run
            if window.noise_dbfs is not None
        )
        / coverage
    )
    return [
        TeachingSuggestion(
            suggestion_type="elevated_relative_audio_level",
            priority="review",
            start_seconds=run[0].start,
            end_seconds=run[-1].end,
            observation=(
                f"Relative audio level averaged {mean_noise:.1f} dBFS during "
                f"{coverage:.1f} seconds of covered audio."
            ),
            suggestion=(
                "Review the recording context before deciding whether a "
                "transition, room-volume reminder, or no action is appropriate."
            ),
            evidence=(
                f"{len(run)} contiguous covered windows; configured relative "
                f"audio threshold {config.elevated_noise_dbfs:.1f} dBFS. "
                "dBFS is not calibrated room sound pressure."
            ),
            confidence="low",
            human_review_recommended=True,
        )
    ]


def _contiguous(previous: _ScoreWindow, current: _ScoreWindow) -> bool:
    return math.isclose(previous.end, current.start, abs_tol=1e-6)


def _contiguous_noise(previous: _NoiseWindow, current: _NoiseWindow) -> bool:
    return math.isclose(previous.end, current.start, abs_tol=1e-6)


def _write_suggestions(
    path: Path,
    suggestions: tuple[TeachingSuggestion, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TeachingSuggestion.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for suggestion in suggestions:
            writer.writerow(
                {
                    name: (
                        str(value).lower()
                        if isinstance(value, bool)
                        else f"{value:.6f}"
                        if isinstance(value, float)
                        else value
                    )
                    for name, value in vars(suggestion).items()
                }
            )
