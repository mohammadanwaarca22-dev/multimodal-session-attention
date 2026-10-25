"""Face detection and temporary centroid-based tracking for video files."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class FaceDetection:
    x: int
    y: int
    width: int
    height: int
    tracking_id: int


@dataclass(frozen=True)
class FaceTrackingResult:
    frames_processed: int
    detections_written: int
    unique_tracks: int


@dataclass
class _Track:
    centroid_x: float
    centroid_y: float
    missed_frames: int = 0


class CentroidTracker:
    """Associate face boxes between frames using one-to-one centroid matching."""

    def __init__(self, max_distance: float = 80.0, max_disappeared: int = 10) -> None:
        if not math.isfinite(max_distance) or max_distance <= 0:
            raise ValueError("max_distance must be finite and greater than zero")
        if (
            not isinstance(max_disappeared, int)
            or isinstance(max_disappeared, bool)
            or max_disappeared < 0
        ):
            raise ValueError("max_disappeared must be a non-negative integer")

        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(
        self, boxes: list[tuple[int, int, int, int]]
    ) -> list[FaceDetection]:
        if not boxes:
            self._age_unmatched(set())
            return []

        centroids = [
            (x + width / 2, y + height / 2) for x, y, width, height in boxes
        ]
        if not self._tracks:
            return [
                self._register(box, centroid)
                for box, centroid in zip(boxes, centroids)
            ]

        max_distance_squared = self.max_distance**2
        candidates = sorted(
            (
                (track.centroid_x - centroid[0]) ** 2
                + (track.centroid_y - centroid[1]) ** 2,
                tracking_id,
                detection_index,
            )
            for tracking_id, track in self._tracks.items()
            for detection_index, centroid in enumerate(centroids)
            if (track.centroid_x - centroid[0]) ** 2
            + (track.centroid_y - centroid[1]) ** 2
            <= max_distance_squared
        )

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        detections_by_index: dict[int, FaceDetection] = {}
        for _, tracking_id, detection_index in candidates:
            if (
                tracking_id in matched_tracks
                or detection_index in matched_detections
            ):
                continue
            box = boxes[detection_index]
            centroid_x, centroid_y = centroids[detection_index]
            self._tracks[tracking_id] = _Track(centroid_x, centroid_y)
            matched_tracks.add(tracking_id)
            matched_detections.add(detection_index)
            detections_by_index[detection_index] = FaceDetection(*box, tracking_id)

        self._age_unmatched(matched_tracks)
        for index, (box, centroid) in enumerate(zip(boxes, centroids)):
            if index not in matched_detections:
                detections_by_index[index] = self._register(box, centroid)

        return [detections_by_index[index] for index in range(len(boxes))]

    def _register(
        self,
        box: tuple[int, int, int, int],
        centroid: tuple[float, float],
    ) -> FaceDetection:
        tracking_id = self._next_id
        self._next_id += 1
        self._tracks[tracking_id] = _Track(*centroid)
        return FaceDetection(*box, tracking_id)

    def _age_unmatched(self, matched_tracks: set[int]) -> None:
        expired = []
        for tracking_id, track in self._tracks.items():
            if tracking_id not in matched_tracks:
                track.missed_frames += 1
                if track.missed_frames > self.max_disappeared:
                    expired.append(tracking_id)
        for tracking_id in expired:
            del self._tracks[tracking_id]


def track_faces(
    video_path: str | Path,
    annotated_video_path: str | Path,
    detections_csv_path: str | Path,
    *,
    cascade_path: str | Path | None = None,
    max_centroid_distance: float = 80.0,
    max_disappeared: int = 10,
) -> FaceTrackingResult:
    """Track detections, not identities, and write video and CSV outputs."""
    input_path = Path(video_path).resolve()
    output_video = Path(annotated_video_path).resolve()
    output_csv = Path(detections_csv_path).resolve()
    if input_path in (output_video, output_csv):
        raise ValueError("Output paths must not overwrite the input video")
    if output_video == output_csv:
        raise ValueError("Video and CSV output paths must be different")
    tracker = CentroidTracker(max_centroid_distance, max_disappeared)

    classifier_file = (
        Path(cascade_path)
        if cascade_path is not None
        else Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    )
    classifier = cv2.CascadeClassifier(str(classifier_file))
    if classifier.empty():
        raise FileNotFoundError(f"Could not load Haar cascade: {classifier_file}")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {input_path}")

    writer: cv2.VideoWriter | None = None

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (
            not math.isfinite(fps)
            or fps <= 0
            or frame_width <= 0
            or frame_height <= 0
        ):
            raise ValueError(
                "Video must report a valid frame rate and frame dimensions"
            )

        output_video.parent.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise OSError(f"Could not create annotated video: {output_video}")

        frames_processed = 0
        detections_written = 0
        track_ids: set[int] = set()

        with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                [
                    "frame_number",
                    "timestamp_seconds",
                    "tracking_id",
                    "x",
                    "y",
                    "width",
                    "height",
                ]
            )

            while True:
                success, frame = capture.read()
                if not success:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes = classifier.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                )
                tracked_faces = tracker.update(
                    [tuple(int(value) for value in box) for box in boxes]
                )

                timestamp = frames_processed / fps
                for face in tracked_faces:
                    cv2.rectangle(
                        frame,
                        (face.x, face.y),
                        (face.x + face.width, face.y + face.height),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"ID {face.tracking_id}",
                        (face.x, max(face.y - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    csv_writer.writerow(
                        [
                            frames_processed,
                            f"{timestamp:.6f}",
                            face.tracking_id,
                            face.x,
                            face.y,
                            face.width,
                            face.height,
                        ]
                    )
                    track_ids.add(face.tracking_id)
                    detections_written += 1

                writer.write(frame)
                frames_processed += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return FaceTrackingResult(
        frames_processed=frames_processed,
        detections_written=detections_written,
        unique_tracks=len(track_ids),
    )
