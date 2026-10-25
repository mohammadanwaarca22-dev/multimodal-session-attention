"""Timestamp-based fusion of vision detections and acoustic activity features."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FusedActivityWindow:
    window_index: int
    start_seconds: float
    end_seconds: float
    visual_detection_count: int | None
    visible_temporary_tracks: int | None
    estimated_track_presence_seconds: float | None
    mean_movement_pixels_per_second: float | None
    visual_inactivity_seconds: float | None
    visual_coverage_seconds: float | None
    audio_coverage_seconds: float | None
    mean_noise_level_dbfs: float | None
    potential_speech_seconds: float | None
    potential_speech_ratio: float | None
    available_modalities: str
    data_quality: str


@dataclass(frozen=True)
class SignalFusionResult:
    windows: tuple[FusedActivityWindow, ...]
    session_duration_seconds: float
    visual_coverage_known: bool
    visual_detection_count: int
    audio_coverage_seconds: float
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class _VisualObservation:
    timestamp: float
    tracking_id: int
    centroid_x: float
    centroid_y: float


@dataclass(frozen=True)
class _AudioObservation:
    start: float
    end: float
    speech: bool
    noise_dbfs: float


def fuse_activity_signals(
    *,
    detections_csv_path: str | Path | None = None,
    audio_windows_csv_path: str | Path | None = None,
    window_duration_seconds: float = 10.0,
    max_visual_gap_seconds: float = 1.0,
    visual_coverage_duration_seconds: float | None = None,
    fused_windows_csv_path: str | Path | None = None,
) -> SignalFusionResult:
    """Align face detections and audio-analysis windows on a shared timeline.

    Visual presence and movement are estimates from temporary tracking IDs.
    Visual inactivity is only calculated when the caller supplies the known
    duration of continuous visual coverage. A lack of detections by itself
    never establishes inactivity. This module does not infer engagement,
    attention, identity, attendance, or learning.
    """
    _validate_positive("window_duration_seconds", window_duration_seconds)
    _validate_positive("max_visual_gap_seconds", max_visual_gap_seconds)
    if visual_coverage_duration_seconds is not None:
        _validate_positive(
            "visual_coverage_duration_seconds",
            visual_coverage_duration_seconds,
        )
    if detections_csv_path is None and audio_windows_csv_path is None:
        raise ValueError("Provide at least one visual or audio input CSV")

    visual_path = _resolve_path(detections_csv_path)
    audio_path = _resolve_path(audio_windows_csv_path)
    output_path = _resolve_path(fused_windows_csv_path)
    inputs = {path for path in (visual_path, audio_path) if path is not None}
    if output_path is not None and output_path in inputs:
        raise ValueError("Fused output must not overwrite an input CSV")

    visual = _read_visual(visual_path) if visual_path is not None else []
    audio = _read_audio(audio_path) if audio_path is not None else []

    visual_duration = visual_coverage_duration_seconds
    if visual_duration is not None and visual and max(
        observation.timestamp for observation in visual
    ) > visual_duration:
        raise ValueError(
            "A visual detection timestamp exceeds visual_coverage_duration_seconds"
        )

    audio_duration = max((observation.end for observation in audio), default=0.0)
    last_visual_timestamp = max(
        (observation.timestamp for observation in visual), default=-1.0
    )
    if visual_duration is not None:
        session_duration = max(visual_duration, audio_duration)
    elif audio_duration > 0:
        session_duration = max(audio_duration, last_visual_timestamp)
    elif last_visual_timestamp >= 0:
        session_duration = (
            math.floor(last_visual_timestamp / window_duration_seconds) + 1
        ) * window_duration_seconds
    else:
        session_duration = 0.0

    if session_duration == 0:
        windows: tuple[FusedActivityWindow, ...] = ()
    else:
        window_count = math.ceil(session_duration / window_duration_seconds)
        windows = tuple(
            _build_window(
                index,
                index * window_duration_seconds,
                min((index + 1) * window_duration_seconds, session_duration),
                visual,
                audio,
                visual_path is not None,
                audio_path is not None,
                visual_duration,
                max_visual_gap_seconds,
            )
            for index in range(window_count)
        )

    limitations = [
        "Temporary tracking IDs do not represent verified identities.",
        "Acoustic speech flags and relative dBFS are indirect, uncalibrated signals.",
        "Fused observations do not establish engagement, attention, or learning.",
    ]
    if visual_path is not None and visual_duration is None:
        limitations.append(
            "Visual coverage duration was not provided; visual inactivity is unknown."
        )

    result = SignalFusionResult(
        windows=windows,
        session_duration_seconds=session_duration,
        visual_coverage_known=visual_duration is not None,
        visual_detection_count=len(visual),
        audio_coverage_seconds=sum(
            observation.end - observation.start for observation in audio
        ),
        limitations=tuple(limitations),
    )
    if output_path is not None:
        _write_windows(output_path, windows)
    return result


def _resolve_path(value: str | Path | None) -> Path | None:
    return Path(value).resolve() if value is not None else None


def _validate_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")


def _read_rows(path: Path, required: set[str]) -> csv.DictReader:
    file = path.open(newline="", encoding="utf-8-sig")
    reader = csv.DictReader(file)
    if reader.fieldnames is None:
        file.close()
        raise ValueError(f"{path}: CSV is missing its header row")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        file.close()
        raise ValueError(f"{path}: CSV contains duplicate column names")
    missing = required.difference(reader.fieldnames)
    if missing:
        file.close()
        raise ValueError(
            f"{path}: missing required columns: {', '.join(sorted(missing))}"
        )
    reader._fusion_file = file
    return reader


def _read_visual(path: Path) -> list[_VisualObservation]:
    rows = _read_rows(
        path,
        {"timestamp_seconds", "tracking_id", "x", "y", "width", "height"},
    )
    observations = []
    seen: set[tuple[int, float]] = set()
    try:
        for row_number, row in enumerate(rows, start=2):
            tracking_id = _parse_id(row.get("tracking_id"), row_number, path)
            timestamp = _parse_time(
                row.get("timestamp_seconds"), "timestamp_seconds", row_number, path
            )
            key = (tracking_id, timestamp)
            if key in seen:
                raise ValueError(
                    f"{path}: duplicate detection for tracking_id "
                    f"{tracking_id} at {timestamp:g} seconds"
                )
            seen.add(key)
            x = _parse_finite(row.get("x"), "x", row_number, path)
            y = _parse_finite(row.get("y"), "y", row_number, path)
            width = _parse_finite(row.get("width"), "width", row_number, path)
            height = _parse_finite(row.get("height"), "height", row_number, path)
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"{path}: bounding-box dimensions must be positive on row "
                    f"{row_number}"
                )
            observations.append(
                _VisualObservation(
                    timestamp,
                    tracking_id,
                    x + width / 2,
                    y + height / 2,
                )
            )
    finally:
        rows._fusion_file.close()
    return observations


def _read_audio(path: Path) -> list[_AudioObservation]:
    rows = _read_rows(
        path,
        {
            "start_seconds",
            "end_seconds",
            "potential_speech_activity",
            "noise_level_dbfs",
        },
    )
    observations = []
    try:
        for row_number, row in enumerate(rows, start=2):
            start = _parse_time(row.get("start_seconds"), "start_seconds", row_number, path)
            end = _parse_time(row.get("end_seconds"), "end_seconds", row_number, path)
            if end <= start:
                raise ValueError(
                    f"{path}: end_seconds must be greater than start_seconds "
                    f"on row {row_number}"
                )
            speech_text = (row.get("potential_speech_activity") or "").strip().lower()
            if speech_text not in {"true", "false", "1", "0"}:
                raise ValueError(
                    f"{path}: invalid potential_speech_activity on row "
                    f"{row_number}: {speech_text!r}"
                )
            noise = _parse_finite(
                row.get("noise_level_dbfs"), "noise_level_dbfs", row_number, path
            )
            observations.append(
                _AudioObservation(
                    start,
                    end,
                    speech_text in {"true", "1"},
                    noise,
                )
            )
    finally:
        rows._fusion_file.close()

    observations.sort(key=lambda observation: observation.start)
    for previous, current in zip(observations, observations[1:]):
        if current.start < previous.end:
            raise ValueError(f"{path}: overlapping audio windows are ambiguous")
    return observations


def _parse_id(value: str | None, row_number: int, path: Path) -> int:
    text = (value or "").strip()
    try:
        tracking_id = int(text)
    except ValueError as error:
        raise ValueError(
            f"{path}: invalid tracking_id on row {row_number}: {text!r}"
        ) from error
    if tracking_id <= 0:
        raise ValueError(f"{path}: tracking_id must be positive on row {row_number}")
    return tracking_id


def _parse_time(
    value: str | None, name: str, row_number: int, path: Path
) -> float:
    timestamp = _parse_finite(value, name, row_number, path)
    if timestamp < 0:
        raise ValueError(f"{path}: {name} must be non-negative on row {row_number}")
    return timestamp


def _parse_finite(
    value: str | None, name: str, row_number: int, path: Path
) -> float:
    text = (value or "").strip()
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(
            f"{path}: invalid {name} on row {row_number}: {text!r}"
        ) from error
    if not math.isfinite(number):
        raise ValueError(f"{path}: {name} must be finite on row {row_number}")
    return number


def _build_window(
    index: int,
    start: float,
    end: float,
    visual: list[_VisualObservation],
    audio: list[_AudioObservation],
    visual_source_available: bool,
    audio_source_available: bool,
    visual_duration: float | None,
    max_visual_gap: float,
) -> FusedActivityWindow:
    duration = end - start
    visual_is_covered = visual_duration is not None and start < visual_duration
    visual_coverage = (
        max(0.0, min(end, visual_duration) - start)
        if visual_duration is not None
        else None
    )
    visual_detections = [
        observation
        for observation in visual
        if start <= observation.timestamp < end
        or (end == visual_duration and observation.timestamp == end)
    ]
    track_times: dict[int, list[float]] = {}
    track_observations: dict[int, list[_VisualObservation]] = {}
    for observation in visual:
        track_observations.setdefault(observation.tracking_id, []).append(observation)
    track_presence_intervals: list[tuple[float, float]] = []
    movements: list[float] = []
    for observations in track_observations.values():
        observations.sort(key=lambda observation: observation.timestamp)
        for previous, current in zip(observations, observations[1:]):
            gap = current.timestamp - previous.timestamp
            if gap <= 0 or gap > max_visual_gap:
                continue
            clipped_start = max(start, previous.timestamp)
            clipped_end = min(end, current.timestamp)
            if clipped_end > clipped_start:
                track_presence_intervals.append((clipped_start, clipped_end))
            if start <= current.timestamp < end or (
                end == visual_duration and current.timestamp == end
            ):
                distance = math.hypot(
                    current.centroid_x - previous.centroid_x,
                    current.centroid_y - previous.centroid_y,
                )
                movements.append(distance / gap)

    estimated_presence = sum(
        interval_end - interval_start
        for interval_start, interval_end in track_presence_intervals
    )
    observed_union_seconds = _union_duration(track_presence_intervals)
    visual_inactivity = (
        max(0.0, visual_coverage - observed_union_seconds)
        if visual_coverage is not None and visual_coverage > 0
        else None
    )
    movement = sum(movements) / len(movements) if movements else None

    audio_coverage = 0.0
    speech_seconds = 0.0
    noise_weighted = 0.0
    for observation in audio:
        overlap = max(0.0, min(end, observation.end) - max(start, observation.start))
        if overlap <= 0:
            continue
        audio_coverage += overlap
        noise_weighted += overlap * observation.noise_dbfs
        if observation.speech:
            speech_seconds += overlap

    has_visual_observation = visual_source_available and (
        visual_is_covered or bool(visual_detections)
    )
    has_audio_observation = audio_source_available and audio_coverage > 0
    available_modalities = []
    if has_visual_observation:
        available_modalities.append("visual")
    if has_audio_observation:
        available_modalities.append("audio")
    data_quality = (
        "complete"
        if has_visual_observation and has_audio_observation
        else "partial"
        if available_modalities
        else "missing"
    )

    return FusedActivityWindow(
        window_index=index,
        start_seconds=start,
        end_seconds=end,
        visual_detection_count=(
            len(visual_detections) if has_visual_observation else None
        ),
        visible_temporary_tracks=(
            len({observation.tracking_id for observation in visual_detections})
            if has_visual_observation
            else None
        ),
        estimated_track_presence_seconds=(
            estimated_presence if has_visual_observation else None
        ),
        mean_movement_pixels_per_second=movement,
        visual_inactivity_seconds=visual_inactivity,
        visual_coverage_seconds=visual_coverage,
        audio_coverage_seconds=audio_coverage if audio_source_available else None,
        mean_noise_level_dbfs=(
            noise_weighted / audio_coverage if audio_coverage > 0 else None
        ),
        potential_speech_seconds=(
            speech_seconds if audio_source_available else None
        ),
        potential_speech_ratio=(
            speech_seconds / audio_coverage if audio_coverage > 0 else None
        ),
        available_modalities="+".join(available_modalities) or "none",
        data_quality=data_quality,
    )


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _write_windows(
    path: Path,
    windows: tuple[FusedActivityWindow, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(FusedActivityWindow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for window in windows:
            writer.writerow(
                {
                    name: (
                        ""
                        if value is None
                        else f"{value:.6f}"
                        if isinstance(value, float)
                        else value
                    )
                    for name, value in vars(window).items()
                }
            )
