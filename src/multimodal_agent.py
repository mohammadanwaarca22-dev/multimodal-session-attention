"""Explainable, session-level alignment of visual and audio activity signals."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MultimodalWindowInsight:
    window_index: int
    start_seconds: float
    end_seconds: float
    visual_detection_count: int | None
    visible_temporary_tracks: int | None
    audio_coverage_seconds: float | None
    potential_speech_seconds: float | None
    potential_speech_ratio: float | None
    mean_noise_level_dbfs: float | None
    available_modalities: str
    activity_indicator: str
    confidence: str
    explanation: str


@dataclass(frozen=True)
class MultimodalSessionSummary:
    duration_seconds: float
    window_count: int
    visual_windows_available: int
    audio_windows_available: int
    multimodal_activity_windows: int
    visual_activity_windows: int
    audio_activity_windows: int
    insufficient_data_windows: int
    session_activity_indicator: str
    confidence: str
    explanation: str


@dataclass(frozen=True)
class MultimodalAnalysisResult:
    windows: tuple[MultimodalWindowInsight, ...]
    session_summary: MultimodalSessionSummary


@dataclass(frozen=True)
class _AudioObservation:
    start_seconds: float
    end_seconds: float
    potential_speech_activity: bool
    noise_level_dbfs: float


def analyze_multimodal_session(
    *,
    detections_csv_path: str | Path | None = None,
    audio_windows_csv_path: str | Path | None = None,
    window_duration_seconds: float = 10.0,
    windows_csv_path: str | Path | None = None,
    summary_csv_path: str | Path | None = None,
) -> MultimodalAnalysisResult:
    """Align vision detections and audio features into explainable time windows.

    Activity indicators describe observable group-level signals, not
    engagement, attention, learning, identity, or attendance. Confidence is
    always reported as low because detections and acoustic heuristics are
    indirect, incomplete evidence.
    """
    if not math.isfinite(window_duration_seconds) or window_duration_seconds <= 0:
        raise ValueError("window_duration_seconds must be finite and greater than zero")
    if detections_csv_path is None and audio_windows_csv_path is None:
        raise ValueError("Provide at least one visual or audio input CSV")

    visual_path = (
        Path(detections_csv_path).resolve()
        if detections_csv_path is not None
        else None
    )
    audio_path = (
        Path(audio_windows_csv_path).resolve()
        if audio_windows_csv_path is not None
        else None
    )
    output_path = (
        Path(windows_csv_path).resolve()
        if windows_csv_path is not None
        else None
    )
    summary_path = (
        Path(summary_csv_path).resolve()
        if summary_csv_path is not None
        else None
    )
    inputs = {path for path in (visual_path, audio_path) if path is not None}
    outputs = {path for path in (output_path, summary_path) if path is not None}
    if output_path is not None and summary_path == output_path:
        raise ValueError("Window and summary output paths must be different")
    if inputs & outputs:
        raise ValueError("Output paths must not overwrite input CSV files")

    visual_observations = (
        _read_visual_observations(visual_path) if visual_path is not None else []
    )
    audio_observations = (
        _read_audio_observations(audio_path) if audio_path is not None else []
    )
    duration = _session_duration(visual_observations, audio_observations)
    if not visual_observations and not audio_observations:
        empty_summary = MultimodalSessionSummary(
            duration_seconds=0.0,
            window_count=0,
            visual_windows_available=0,
            audio_windows_available=0,
            multimodal_activity_windows=0,
            visual_activity_windows=0,
            audio_activity_windows=0,
            insufficient_data_windows=0,
            session_activity_indicator="insufficient_data",
            confidence="low",
            explanation=(
                "No timestamped observations were available; no activity "
                "conclusion can be drawn."
            ),
        )
        result = MultimodalAnalysisResult((), empty_summary)
        if output_path is not None:
            _write_window_csv(output_path, result.windows)
        if summary_path is not None:
            _write_summary_csv(summary_path, result.session_summary)
        return result

    visual_window_count = (
        math.floor(max(timestamp for timestamp, _ in visual_observations)
                   / window_duration_seconds)
        + 1
        if visual_observations
        else 0
    )
    window_count = max(
        math.ceil(duration / window_duration_seconds),
        visual_window_count,
    )
    windows = tuple(
        _make_window_insight(
            index,
            index * window_duration_seconds,
            min(
                (index + 1) * window_duration_seconds,
                duration
                if duration > index * window_duration_seconds
                else (index + 1) * window_duration_seconds,
            ),
            index == window_count - 1,
            visual_observations,
            audio_observations,
            visual_path is not None,
            audio_path is not None,
        )
        for index in range(window_count)
    )
    summary = _summarize_session(duration, windows)
    result = MultimodalAnalysisResult(windows, summary)

    if output_path is not None:
        _write_window_csv(output_path, windows)
    if summary_path is not None:
        _write_summary_csv(summary_path, summary)
    return result


def _read_visual_observations(path: Path) -> list[tuple[float, int]]:
    observations = []
    seen: set[tuple[int, float]] = set()
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_columns(reader.fieldnames, {"timestamp_seconds", "tracking_id"}, path)
        for row_number, row in enumerate(reader, start=2):
            tracking_id = _parse_positive_int(
                row.get("tracking_id"), "tracking_id", row_number, path
            )
            timestamp = _parse_nonnegative_float(
                row.get("timestamp_seconds"), "timestamp_seconds", row_number, path
            )
            key = (tracking_id, timestamp)
            if key in seen:
                raise ValueError(
                    f"{path}: duplicate visual detection for tracking_id "
                    f"{tracking_id} at {timestamp:g} seconds"
                )
            seen.add(key)
            observations.append((timestamp, tracking_id))
    return observations


def _read_audio_observations(path: Path) -> list[_AudioObservation]:
    observations = []
    seen: set[tuple[float, float]] = set()
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_columns(
            reader.fieldnames,
            {
                "start_seconds",
                "end_seconds",
                "potential_speech_activity",
                "noise_level_dbfs",
            },
            path,
        )
        for row_number, row in enumerate(reader, start=2):
            start = _parse_nonnegative_float(
                row.get("start_seconds"), "start_seconds", row_number, path
            )
            end = _parse_nonnegative_float(
                row.get("end_seconds"), "end_seconds", row_number, path
            )
            if end <= start:
                raise ValueError(
                    f"{path}: end_seconds must be greater than start_seconds "
                    f"on row {row_number}"
                )
            key = (start, end)
            if key in seen:
                raise ValueError(
                    f"{path}: duplicate audio window {start:g}-{end:g} seconds"
                )
            seen.add(key)
            speech = _parse_bool(
                row.get("potential_speech_activity"),
                "potential_speech_activity",
                row_number,
                path,
            )
            noise = _parse_finite_float(
                row.get("noise_level_dbfs"),
                "noise_level_dbfs",
                row_number,
                path,
            )
            observations.append(_AudioObservation(start, end, speech, noise))

    observations.sort(key=lambda observation: observation.start_seconds)
    for previous, current in zip(observations, observations[1:]):
        if current.start_seconds < previous.end_seconds:
            raise ValueError(
                f"{path}: overlapping audio windows are ambiguous and cannot "
                "be aggregated safely"
            )
    return observations


def _validate_columns(
    fieldnames: list[str] | None,
    required: set[str],
    path: Path,
) -> None:
    if fieldnames is None:
        raise ValueError(f"{path}: CSV is missing its header row")
    if len(fieldnames) != len(set(fieldnames)):
        raise ValueError(f"{path}: CSV contains duplicate column names")
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(
            f"{path}: missing required columns: {', '.join(sorted(missing))}"
        )


def _parse_positive_int(
    value: str | None, name: str, row_number: int, path: Path
) -> int:
    text = (value or "").strip()
    try:
        parsed = int(text)
    except ValueError as error:
        raise ValueError(
            f"{path}: invalid {name} on row {row_number}: {text!r}"
        ) from error
    if parsed <= 0:
        raise ValueError(f"{path}: {name} must be positive on row {row_number}")
    return parsed


def _parse_finite_float(
    value: str | None, name: str, row_number: int, path: Path
) -> float:
    text = (value or "").strip()
    try:
        parsed = float(text)
    except ValueError as error:
        raise ValueError(
            f"{path}: invalid {name} on row {row_number}: {text!r}"
        ) from error
    if not math.isfinite(parsed):
        raise ValueError(f"{path}: {name} must be finite on row {row_number}")
    return parsed


def _parse_nonnegative_float(
    value: str | None, name: str, row_number: int, path: Path
) -> float:
    parsed = _parse_finite_float(value, name, row_number, path)
    if parsed < 0:
        raise ValueError(f"{path}: {name} must be non-negative on row {row_number}")
    return parsed


def _parse_bool(
    value: str | None, name: str, row_number: int, path: Path
) -> bool:
    text = (value or "").strip().lower()
    if text not in {"true", "false", "1", "0"}:
        raise ValueError(
            f"{path}: invalid {name} on row {row_number}: {text!r}"
        )
    return text in {"true", "1"}


def _session_duration(
    visual: list[tuple[float, int]],
    audio: list[_AudioObservation],
) -> float:
    visual_end = (
        max((timestamp for timestamp, _ in visual), default=-1.0)
    )
    audio_end = max(
        (observation.end_seconds for observation in audio),
        default=0.0,
    )
    if visual_end < 0 and audio_end <= 0:
        return 0.0
    return max(visual_end, audio_end)


def _make_window_insight(
    index: int,
    start: float,
    end: float,
    is_last_window: bool,
    visual: list[tuple[float, int]],
    audio: list[_AudioObservation],
    visual_available: bool,
    audio_available: bool,
) -> MultimodalWindowInsight:
    tracks = {
        tracking_id
        for timestamp, tracking_id in visual
        if start <= timestamp < end or (is_last_window and timestamp == end)
    }
    detections = sum(
        start <= timestamp < end or (is_last_window and timestamp == end)
        for timestamp, _ in visual
    )

    speech_seconds = 0.0
    audio_coverage = 0.0
    weighted_noise_sum = 0.0
    for observation in audio:
        overlap = max(
            0.0,
            min(end, observation.end_seconds)
            - max(start, observation.start_seconds),
        )
        if overlap == 0:
            continue
        audio_coverage += overlap
        weighted_noise_sum += observation.noise_level_dbfs * overlap
        if observation.potential_speech_activity:
            speech_seconds += overlap

    duration = end - start
    modalities = []
    if visual_available:
        modalities.append("visual")
    if audio_available:
        modalities.append("audio")
    if not modalities:
        modality_status = "none"
    elif visual_available and audio_available:
        modality_status = "visual+audio"
    else:
        modality_status = modalities[0]

    if not visual_available and not audio_available:
        indicator = "insufficient_data"
    elif visual_available and audio_available and detections and speech_seconds:
        indicator = "multimodal_activity_observed"
    elif visual_available and detections:
        indicator = "visual_activity_observed"
    elif audio_available and audio_coverage and speech_seconds:
        indicator = "audio_activity_observed"
    elif audio_available and not audio_coverage and not detections:
        indicator = "insufficient_data"
    else:
        indicator = "no_activity_threshold_met"

    explanation_parts = []
    if visual_available:
        explanation_parts.append(
            f"{detections} face detections across {len(tracks)} temporary "
            "tracking IDs"
        )
    else:
        explanation_parts.append("visual observations unavailable")
    if audio_available:
        if audio_coverage:
            explanation_parts.append(
                f"{speech_seconds:.2f}s of {audio_coverage:.2f}s covered audio "
                "was flagged as potential speech"
            )
        else:
            explanation_parts.append("no audio observations covered this window")
    else:
        explanation_parts.append("audio observations unavailable")
    explanation_parts.append(
        "signals describe visible/acoustic activity only; engagement is not inferred"
    )

    return MultimodalWindowInsight(
        window_index=index,
        start_seconds=start,
        end_seconds=end,
        visual_detection_count=detections if visual_available else None,
        visible_temporary_tracks=len(tracks) if visual_available else None,
        audio_coverage_seconds=audio_coverage if audio_available else None,
        potential_speech_seconds=speech_seconds if audio_available else None,
        potential_speech_ratio=(
            speech_seconds / audio_coverage if audio_available and audio_coverage else None
        ),
        mean_noise_level_dbfs=(
            weighted_noise_sum / audio_coverage
            if audio_available and audio_coverage
            else None
        ),
        available_modalities=modality_status,
        activity_indicator=indicator,
        confidence="low",
        explanation="; ".join(explanation_parts) + ".",
    )


def _summarize_session(
    duration: float,
    windows: tuple[MultimodalWindowInsight, ...],
) -> MultimodalSessionSummary:
    multimodal = sum(
        window.activity_indicator == "multimodal_activity_observed"
        for window in windows
    )
    visual_only = sum(
        window.activity_indicator == "visual_activity_observed"
        for window in windows
    )
    audio_only = sum(
        window.activity_indicator == "audio_activity_observed"
        for window in windows
    )
    insufficient = sum(
        window.activity_indicator == "insufficient_data" for window in windows
    )
    if multimodal:
        indicator = "multimodal_activity_observed"
    elif visual_only and audio_only:
        indicator = "separate_visual_and_audio_activity_observed"
    elif visual_only:
        indicator = "visual_activity_observed"
    elif audio_only:
        indicator = "audio_activity_observed"
    elif insufficient == len(windows):
        indicator = "insufficient_data"
    else:
        indicator = "no_activity_threshold_met"

    explanation = (
        f"Observed {multimodal} windows with both visual detections and "
        f"potential speech, {visual_only} with visual activity only, "
        f"{audio_only} with potential speech only, and {insufficient} with "
        "insufficient modality coverage. These are group-level activity "
        "signals, not engagement or individual participation measurements."
    )
    return MultimodalSessionSummary(
        duration_seconds=duration,
        window_count=len(windows),
        visual_windows_available=sum(
            window.visual_detection_count is not None for window in windows
        ),
        audio_windows_available=sum(
            window.audio_coverage_seconds is not None
            and window.audio_coverage_seconds > 0
            for window in windows
        ),
        multimodal_activity_windows=multimodal,
        visual_activity_windows=visual_only,
        audio_activity_windows=audio_only,
        insufficient_data_windows=insufficient,
        session_activity_indicator=indicator,
        confidence="low",
        explanation=explanation,
    )


def _write_window_csv(
    path: Path,
    windows: tuple[MultimodalWindowInsight, ...],
) -> None:
    _write_records(path, windows, MultimodalWindowInsight)


def _write_summary_csv(path: Path, summary: MultimodalSessionSummary) -> None:
    _write_records(path, (summary,), MultimodalSessionSummary)


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
                        str(value).lower()
                        if isinstance(value, bool)
                        else f"{value:.6f}"
                        if isinstance(value, float)
                        else value
                    )
                    for name, value in vars(record).items()
                }
            )
