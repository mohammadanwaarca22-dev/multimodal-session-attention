"""Estimate visible presence duration from face-tracking CSV detections."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttendanceRecord:
    tracking_id: int
    observed_presence_seconds: float
    required_presence_seconds: float
    observation_count: int
    presence_bouts: int
    meets_threshold: bool

    @property
    def insufficient_observations(self) -> bool:
        return self.observation_count < 2


def calculate_attendance(
    detections_csv_path: str | Path,
    *,
    minimum_presence_seconds: float,
    gap_tolerance_seconds: float = 1.0,
    attendance_csv_path: str | Path | None = None,
) -> list[AttendanceRecord]:
    """Estimate presence from timestamp intervals between detections.

    Consecutive observations are considered part of the same presence bout when
    their interval is at most ``gap_tolerance_seconds``. Longer intervals are
    excluded from the estimate and start a new bout. A single observation has
    no measurable duration. Track IDs are temporary and do not verify identity
    or attendance.
    """
    if not math.isfinite(minimum_presence_seconds) or minimum_presence_seconds <= 0:
        raise ValueError("minimum_presence_seconds must be finite and greater than zero")
    if not math.isfinite(gap_tolerance_seconds) or gap_tolerance_seconds <= 0:
        raise ValueError("gap_tolerance_seconds must be finite and greater than zero")

    input_path = Path(detections_csv_path).resolve()
    output_path = (
        Path(attendance_csv_path).resolve()
        if attendance_csv_path is not None
        else None
    )
    if output_path == input_path:
        raise ValueError("Attendance output must not overwrite the detections CSV")

    timestamps_by_track: dict[int, list[float]] = {}
    seen_observations: set[tuple[int, float]] = set()
    with input_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"tracking_id", "timestamp_seconds"}
        if reader.fieldnames is None:
            raise ValueError("Detections CSV is missing its header row")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Detections CSV contains duplicate column names")
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Detections CSV is missing required columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            tracking_id_text = (row.get("tracking_id") or "").strip()
            if not tracking_id_text:
                raise ValueError(f"Missing tracking_id on CSV row {row_number}")
            try:
                tracking_id = int(tracking_id_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid tracking_id on CSV row {row_number}: "
                    f"{tracking_id_text!r}"
                ) from error
            if tracking_id <= 0:
                raise ValueError(
                    f"tracking_id must be positive on CSV row {row_number}"
                )

            timestamp_text = (row.get("timestamp_seconds") or "").strip()
            if not timestamp_text:
                raise ValueError(f"Missing timestamp on CSV row {row_number}")
            try:
                timestamp = float(timestamp_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid timestamp on CSV row {row_number}: "
                    f"{timestamp_text!r}"
                ) from error
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(
                    f"Timestamp must be finite and non-negative on CSV row {row_number}"
                )

            observation = (tracking_id, timestamp)
            if observation in seen_observations:
                raise ValueError(
                    f"Duplicate detection for tracking_id {tracking_id} "
                    f"at {timestamp:g} seconds"
                )
            seen_observations.add(observation)
            timestamps_by_track.setdefault(tracking_id, []).append(timestamp)

    records = [
        _summarize_track(
            tracking_id,
            timestamps,
            minimum_presence_seconds,
            gap_tolerance_seconds,
        )
        for tracking_id, timestamps in sorted(timestamps_by_track.items())
    ]

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "tracking_id",
                    "observed_presence_seconds",
                    "required_presence_seconds",
                    "observation_count",
                    "presence_bouts",
                    "meets_threshold",
                    "insufficient_observations",
                ]
            )
            for record in records:
                writer.writerow(
                    [
                        record.tracking_id,
                        f"{record.observed_presence_seconds:.6f}",
                        f"{record.required_presence_seconds:.6f}",
                        record.observation_count,
                        record.presence_bouts,
                        str(record.meets_threshold).lower(),
                        str(record.insufficient_observations).lower(),
                    ]
                )

    return records


def _summarize_track(
    tracking_id: int,
    timestamps: list[float],
    minimum_presence_seconds: float,
    gap_tolerance_seconds: float,
) -> AttendanceRecord:
    ordered_timestamps = sorted(timestamps)
    observed_seconds = 0.0
    presence_bouts = 1
    for previous, current in zip(ordered_timestamps, ordered_timestamps[1:]):
        interval = current - previous
        if interval <= gap_tolerance_seconds:
            observed_seconds += interval
            if not math.isfinite(observed_seconds):
                raise ValueError(
                    f"Presence duration overflow for tracking_id {tracking_id}"
                )
        else:
            presence_bouts += 1

    return AttendanceRecord(
        tracking_id=tracking_id,
        observed_presence_seconds=observed_seconds,
        required_presence_seconds=minimum_presence_seconds,
        observation_count=len(ordered_timestamps),
        presence_bouts=presence_bouts,
        meets_threshold=observed_seconds >= minimum_presence_seconds,
    )
