import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.face_tracking import CentroidTracker, track_faces


class CentroidTrackerTests(unittest.TestCase):
    def test_keeps_ids_for_nearby_detections_in_new_order(self):
        tracker = CentroidTracker(max_distance=50)
        first = tracker.update([(0, 0, 10, 10), (100, 0, 10, 10)])

        next_frame = tracker.update([(104, 0, 10, 10), (4, 0, 10, 10)])

        self.assertEqual(
            [face.tracking_id for face in first],
            [1, 2],
        )
        self.assertEqual(
            [face.tracking_id for face in next_frame],
            [2, 1],
        )

    def test_keeps_track_for_configured_missing_frames(self):
        tracker = CentroidTracker(max_disappeared=1)
        first = tracker.update([(0, 0, 10, 10)])

        tracker.update([])
        after_one_missing_frame = tracker.update([(2, 0, 10, 10)])

        self.assertEqual(after_one_missing_frame[0].tracking_id, first[0].tracking_id)

    def test_assigns_new_id_after_track_expires(self):
        tracker = CentroidTracker(max_disappeared=1)
        tracker.update([(0, 0, 10, 10)])
        tracker.update([])
        tracker.update([])

        reappeared = tracker.update([(0, 0, 10, 10)])

        self.assertEqual(reappeared[0].tracking_id, 2)

    def test_rejects_invalid_tracker_settings(self):
        with self.assertRaises(ValueError):
            CentroidTracker(max_distance=0)
        with self.assertRaises(ValueError):
            CentroidTracker(max_distance=float("nan"))
        with self.assertRaises(ValueError):
            CentroidTracker(max_disappeared=-1)
        with self.assertRaises(ValueError):
            CentroidTracker(max_disappeared=1.5)

    def test_expiring_track_does_not_reuse_identifier(self):
        tracker = CentroidTracker(max_disappeared=0)
        first = tracker.update([(0, 0, 10, 10)])
        tracker.update([])
        reappeared = tracker.update([(0, 0, 10, 10)])

        self.assertEqual(first[0].tracking_id, 1)
        self.assertEqual(reappeared[0].tracking_id, 2)

    def test_video_pipeline_handles_dark_and_bright_no_face_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "lighting.mp4"
            annotated = root / "annotated.mp4"
            detections_path = root / "detections.csv"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (64, 48),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV MP4 writer is unavailable")
            for level in (0, 40, 128, 220, 255):
                writer.write(np.full((48, 64, 3), level, dtype=np.uint8))
            writer.release()

            result = track_faces(source, annotated, detections_path)
            with detections_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

            capture = cv2.VideoCapture(str(annotated))
            output_frames = 0
            while capture.read()[0]:
                output_frames += 1
            capture.release()

        self.assertEqual(result.frames_processed, 5)
        self.assertEqual(result.detections_written, 0)
        self.assertEqual(result.unique_tracks, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(output_frames, 5)


if __name__ == "__main__":
    unittest.main()
