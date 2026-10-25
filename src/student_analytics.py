"""Privacy-conscious session summaries without attributing group scores to people."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParticipantSessionSummary:
    temporary_tracking_id: int
    observed_presence_seconds: float
    required_presence_seconds: float
    meets_presence_duration_threshold: bool
    observation_count: int
    presence_bouts: int
    individual_activity_consistency: None
    individual_activity_timeline_available: bool
    reflection_note: str


@dataclass(frozen=True)
class GroupActivityTimelineEntry:
    window_index: int
    start_seconds: float
    end_seconds: float
    activity_score: float | None
    activity_band: str
    confidence: str
    explanation: str


@dataclass(frozen=True)
class GroupActivitySummary:
    scored_window_count: int
    unscored_window_count: int
    lower_activity_window_count: int
    moderate_activity_window_count: int
    higher_activity_window_count: int
    mean_activity_score: float | None
    score_standard_deviation: float | None
    summary: str


@dataclass(frozen=True)
class StudentAnalyticsResult:
    participants: tuple[ParticipantSessionSummary, ...]
    group_activity_timeline: tuple[GroupActivityTimelineEntry, ...]
    group_activity_summary: GroupActivitySummary
    privacy_note: str


def generate_student_analytics(
    attendance_csv_path: str | Path,
    *,
    activity_scores_csv_path: str | Path | None = None,
    participants_csv_path: str | Path | None = None,
    group_timeline_csv_path: str | Path | None = None,
    group_summary_csv_path: str | Path | None = None,
) -> StudentAnalyticsResult:
    """Build temporary-track presence summaries and separate group context.

    Activity scores produced by the current scorer are session-level, not
    participant-level. They are therefore returned only as a separate group
    timeline and are never attributed to any temporary tracking ID.
    """
    attendance_path = Path(attendance_csv_path).resolve()
    scores_path = (
        Path(activity_scores_csv_path).resolve()
        if activity_scores_csv_path is not None
        else None
    )
    participants_path = _resolve_output(participants_csv_path)
    timeline_path = _resolve_output(group_timeline_csv_path)
    summary_path = _resolve_output(group_summary_csv_path)

    inputs = {attendance_path}
    if scores_path is not None:
        inputs.add(scores_path)
    outputs = {
        path
        for path in (participants_path, timeline_path, summary_path)
        if path is not None
    }
    if inputs & outputs:
        raise ValueError("Analytics output must not overwrite an input CSV")
    if len(outputs) != sum(
        path is not None
        for path in (participants_path, timeline_path, summary_path)
    ):
        raise ValueError("Analytics output paths must be different")

    participants = tuple(_read_attendance(attendance_path))
    timeline = tuple(_read_group_timeline(scores_path)) if scores_path else ()
    group_summary = _summarize_group_activity(timeline, scores_path is not None)
    result = StudentAnalyticsResult(
        participants=participants,
        group_activity_timeline=timeline,
        group_activity_summary=group_summary,
        privacy_note=(
            "Temporary tracking IDs are not verified student identities and may "
            "change or merge. Presence duration is an estimate, not proof of "
            "attendance. Group activity scores are contextual only and are not "
            "individual engagement, attention, learning, or performance measures."
        ),
    )

    if participants_path is not None:
        _write_records(
            participants_path,
            participants,
            ParticipantSessionSummary,
        )
    if timeline_path is not None:
        _write_records(
            timeline_path,
            timeline,
            GroupActivityTimelineEntry,
        )
    if summary_path is not None:
        _write_records(
            summary_path,
            (group_summary,),
            GroupActivitySummary,
        )
    return result


def _resolve_output(value: str | Path | None) -> Path | None:
    return Path(value).resolve() if value is not None else None


def _read_attendance(path: Path) -> list[ParticipantSessionSummary]:
    required = {
        "tracking_id",
        "observed_presence_seconds",
        "required_presence_seconds",
        "observation_count",
        "presence_bouts",
        "meets_threshold",
    }
    records = []
    seen_ids: set[int] = set()
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_headers(reader.fieldnames, required, path)
        for row_number, row in enumerate(reader, start=2):
            tracking_id = _parse_int(row.get("tracking_id"), "tracking_id", row_number, path)
            if tracking_id <= 0:
                raise ValueError(f"{path}: tracking_id must be positive on row {row_number}")
            if tracking_id in seen_ids:
                raise ValueError(f"{path}: duplicate tracking_id {tracking_id}")
            seen_ids.add(tracking_id)

            observed = _parse_number(
                row.get("observed_presence_seconds"),
                "observed_presence_seconds",
                row_number,
                path,
            )
            required_duration = _parse_number(
                row.get("required_presence_seconds"),
                "required_presence_seconds",
                row_number,
                path,
            )
            observations = _parse_int(
                row.get("observation_count"), "observation_count", row_number, path
            )
            bouts = _parse_int(row.get("presence_bouts"), "presence_bouts", row_number, path)
            meets_threshold = _parse_bool(
                row.get("meets_threshold"), "meets_threshold", row_number, path
            )
            if observed < 0 or required_duration <= 0:
                raise ValueError(
                    f"{path}: presence durations must be non-negative and "
                    f"required duration positive on row {row_number}"
                )
            if observations < 0 or bouts < 0 or bouts > observations:
                raise ValueError(f"{path}: invalid observation counts on row {row_number}")
            if meets_threshold != (observed >= required_duration):
                raise ValueError(
                    f"{path}: meets_threshold conflicts with durations on row {row_number}"
                )

            records.append(
                ParticipantSessionSummary(
                    temporary_tracking_id=tracking_id,
                    observed_presence_seconds=observed,
                    required_presence_seconds=required_duration,
                    meets_presence_duration_threshold=meets_threshold,
                    observation_count=observations,
                    presence_bouts=bouts,
                    individual_activity_consistency=None,
                    individual_activity_timeline_available=False,
                    reflection_note=(
                        "Review this estimated visible-presence duration if "
                        "appropriate. It is linked to a temporary track, not a "
                        "verified identity; individual activity analytics are "
                        "unavailable."
                    ),
                )
            )
    return records


def _read_group_timeline(path: Path) -> list[GroupActivityTimelineEntry]:
    required = {
        "window_index",
        "start_seconds",
        "end_seconds",
        "activity_score",
        "activity_band",
        "confidence",
        "explanation",
    }
    records = []
    seen_indices: set[int] = set()
    previous_end = 0.0
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_headers(reader.fieldnames, required, path)
        for row_number, row in enumerate(reader, start=2):
            index = _parse_int(row.get("window_index"), "window_index", row_number, path)
            if index < 0 or index in seen_indices:
                raise ValueError(f"{path}: invalid or duplicate window_index on row {row_number}")
            seen_indices.add(index)
            start = _parse_number(row.get("start_seconds"), "start_seconds", row_number, path)
            end = _parse_number(row.get("end_seconds"), "end_seconds", row_number, path)
            if start < 0 or end <= start or (records and start < previous_end):
                raise ValueError(f"{path}: invalid or overlapping windows on row {row_number}")
            previous_end = end
            score = _parse_optional_number(
                row.get("activity_score"), "activity_score", row_number, path
            )
            if score is not None and not 0 <= score <= 100:
                raise ValueError(
                    f"{path}: activity_score must be within [0, 100] on row {row_number}"
                )
            band = (row.get("activity_band") or "").strip()
            if band not in {
                "lower_observed_activity",
                "moderate_observed_activity",
                "higher_observed_activity",
                "insufficient_evidence",
            }:
                raise ValueError(f"{path}: unknown activity_band on row {row_number}")
            confidence = (row.get("confidence") or "").strip()
            if score is None and band != "insufficient_evidence":
                raise ValueError(
                    f"{path}: unscored window must use insufficient_evidence band"
                )
            if score is not None and band == "insufficient_evidence":
                raise ValueError(f"{path}: scored window has insufficient_evidence band")
            explanation = (row.get("explanation") or "").strip()
            if not explanation:
                raise ValueError(f"{path}: missing explanation on row {row_number}")
            records.append(
                GroupActivityTimelineEntry(
                    index,
                    start,
                    end,
                    score,
                    band,
                    confidence,
                    explanation,
                )
            )
    return records


def _summarize_group_activity(
    timeline: tuple[GroupActivityTimelineEntry, ...],
    input_provided: bool,
) -> GroupActivitySummary:
    scores = [
        entry.activity_score
        for entry in timeline
        if entry.activity_score is not None
    ]
    unscored = len(timeline) - len(scores)
    if not input_provided or not scores:
        return GroupActivitySummary(
            scored_window_count=0,
            unscored_window_count=len(timeline),
            lower_activity_window_count=0,
            moderate_activity_window_count=0,
            higher_activity_window_count=0,
            mean_activity_score=None,
            score_standard_deviation=None,
            summary=(
                "Group activity context is unavailable or has no scored windows; "
                "individual activity consistency cannot be determined."
            ),
        )
    lower_count = sum(
        entry.activity_band == "lower_observed_activity" for entry in timeline
    )
    moderate_count = sum(
        entry.activity_band == "moderate_observed_activity" for entry in timeline
    )
    higher_count = sum(
        entry.activity_band == "higher_observed_activity" for entry in timeline
    )
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return GroupActivitySummary(
        scored_window_count=len(scores),
        unscored_window_count=unscored,
        lower_activity_window_count=lower_count,
        moderate_activity_window_count=moderate_count,
        higher_activity_window_count=higher_count,
        mean_activity_score=mean,
        score_standard_deviation=math.sqrt(variance),
        summary=(
            f"Group-level observed-activity proxy averaged {mean:.1f}/100 "
            f"across {len(scores)} scored windows "
            f"(standard deviation {math.sqrt(variance):.1f}); "
            f"{lower_count} windows were lower, {moderate_count} moderate, "
            f"and {higher_count} higher observed-activity bands; "
            f"{unscored} windows were unscored. This context is not attributed "
            "to individuals and is not a validated engagement measure."
        ),
    )


def _validate_headers(
    headers: list[str] | None,
    required: set[str],
    path: Path,
) -> None:
    if headers is None:
        raise ValueError(f"{path}: CSV is missing its header row")
    if len(headers) != len(set(headers)):
        raise ValueError(f"{path}: duplicate CSV column names")
    missing = required.difference(headers)
    if missing:
        raise ValueError(
            f"{path}: missing required columns: {', '.join(sorted(missing))}"
        )


def _parse_int(value: str | None, name: str, row: int, path: Path) -> int:
    text = (value or "").strip()
    try:
        return int(text)
    except ValueError as error:
        raise ValueError(f"{path}: invalid {name} on row {row}: {text!r}") from error


def _parse_number(value: str | None, name: str, row: int, path: Path) -> float:
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


def _parse_bool(value: str | None, name: str, row: int, path: Path) -> bool:
    text = (value or "").strip().lower()
    if text not in {"true", "false", "1", "0"}:
        raise ValueError(f"{path}: invalid {name} on row {row}: {text!r}")
    return text in {"true", "1"}


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
