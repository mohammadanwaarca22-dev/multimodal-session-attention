import csv
import tempfile
import unittest
from pathlib import Path

from src.signal_fusion import fuse_activity_signals


class FuseActivitySignalsTests(unittest.TestCase):
    def write_csv(self, path, headers, rows):
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            writer.writerows(rows)

    def test_fuses_presence_movement_and_audio_in_time_windows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            visual = root / "detections.csv"
            audio = root / "audio.csv"
            self.write_csv(
                visual,
                [
                    "frame_number", "timestamp_seconds", "tracking_id",
                    "x", "y", "width", "height",
                ],
                [
                    [0, 0.0, 1, 0, 0, 10, 10],
                    [1, 0.5, 1, 3, 4, 10, 10],
                    [2, 1.0, 1, 6, 8, 10, 10],
                ],
            )
            self.write_csv(
                audio,
                [
                    "start_seconds", "end_seconds",
                    "potential_speech_activity", "noise_level_dbfs",
                ],
                [[0, 1, "true", -20], [1, 2, "false", -30]],
            )

            result = fuse_activity_signals(
                detections_csv_path=visual,
                audio_windows_csv_path=audio,
                window_duration_seconds=1,
                max_visual_gap_seconds=0.6,
                visual_coverage_duration_seconds=2,
            )

        self.assertEqual(len(result.windows), 2)
        first, second = result.windows
        self.assertEqual(first.visual_detection_count, 2)
        self.assertEqual(first.visible_temporary_tracks, 1)
        self.assertAlmostEqual(first.estimated_track_presence_seconds, 1.0)
        self.assertAlmostEqual(first.mean_movement_pixels_per_second, 10.0)
        self.assertAlmostEqual(first.visual_inactivity_seconds, 0.0)
        self.assertAlmostEqual(first.potential_speech_seconds, 1.0)
        self.assertAlmostEqual(first.mean_noise_level_dbfs, -20)
        self.assertEqual(first.available_modalities, "visual+audio")
        self.assertEqual(first.data_quality, "complete")
        self.assertAlmostEqual(second.visual_inactivity_seconds, 1.0)
        self.assertEqual(result.session_duration_seconds, 2)

    def test_never_infers_visual_inactivity_without_coverage_duration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            visual = Path(temp_dir) / "detections.csv"
            self.write_csv(
                visual,
                ["timestamp_seconds", "tracking_id", "x", "y", "width", "height"],
                [[0, 1, 0, 0, 10, 10]],
            )

            result = fuse_activity_signals(
                detections_csv_path=visual,
                window_duration_seconds=1,
            )

        self.assertFalse(result.visual_coverage_known)
        self.assertIsNone(result.windows[0].visual_inactivity_seconds)
        self.assertIsNone(result.windows[0].visual_coverage_seconds)

    def test_missing_audio_coverage_is_not_a_silence_measurement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "audio.csv"
            self.write_csv(
                audio,
                [
                    "start_seconds", "end_seconds",
                    "potential_speech_activity", "noise_level_dbfs",
                ],
                [[0, 0.5, "false", -50], [2.5, 3, "true", -20]],
            )

            result = fuse_activity_signals(
                audio_windows_csv_path=audio,
                window_duration_seconds=1,
            )

        self.assertEqual(len(result.windows), 3)
        self.assertEqual(result.windows[0].audio_coverage_seconds, 0.5)
        self.assertEqual(result.windows[0].potential_speech_seconds, 0)
        self.assertEqual(result.windows[1].audio_coverage_seconds, 0)
        self.assertIsNone(result.windows[1].mean_noise_level_dbfs)
        self.assertEqual(result.windows[1].available_modalities, "none")

    def test_visual_coverage_ending_early_is_not_reported_as_inactivity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            visual = root / "visual.csv"
            audio = root / "audio.csv"
            self.write_csv(
                visual,
                ["timestamp_seconds", "tracking_id", "x", "y", "width", "height"],
                [[0.5, 1, 0, 0, 10, 10]],
            )
            self.write_csv(
                audio,
                [
                    "start_seconds", "end_seconds",
                    "potential_speech_activity", "noise_level_dbfs",
                ],
                [[0, 2, "false", -40]],
            )

            result = fuse_activity_signals(
                detections_csv_path=visual,
                audio_windows_csv_path=audio,
                visual_coverage_duration_seconds=1,
                window_duration_seconds=1,
            )

        self.assertEqual(result.windows[1].visual_coverage_seconds, 0)
        self.assertIsNone(result.windows[1].visual_inactivity_seconds)
        self.assertIsNone(result.windows[1].visual_detection_count)
        self.assertEqual(result.windows[1].available_modalities, "audio")

    def test_writes_fused_csv_and_preserves_unknown_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            visual = root / "visual.csv"
            output = root / "out" / "fused.csv"
            self.write_csv(
                visual,
                ["timestamp_seconds", "tracking_id", "x", "y", "width", "height"],
                [[0.2, 1, 0, 0, 10, 10]],
            )

            fuse_activity_signals(
                detections_csv_path=visual,
                fused_windows_csv_path=output,
                window_duration_seconds=1,
            )

            with output.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["visual_inactivity_seconds"], "")
        self.assertEqual(rows[0]["mean_movement_pixels_per_second"], "")

    def test_rejects_duplicates_invalid_boxes_overlaps_and_path_collisions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            visual = root / "visual.csv"
            self.write_csv(
                visual,
                ["timestamp_seconds", "tracking_id", "x", "y", "width", "height"],
                [[0, 1, 0, 0, 10, 10], [0, 1, 0, 0, 10, 10]],
            )
            with self.assertRaisesRegex(ValueError, "duplicate detection"):
                fuse_activity_signals(detections_csv_path=visual)

            self.write_csv(
                visual,
                ["timestamp_seconds", "tracking_id", "x", "y", "width", "height"],
                [[0, 1, 0, 0, 0, 10]],
            )
            with self.assertRaisesRegex(ValueError, "dimensions must be positive"):
                fuse_activity_signals(detections_csv_path=visual)

            audio = root / "audio.csv"
            self.write_csv(
                audio,
                [
                    "start_seconds", "end_seconds",
                    "potential_speech_activity", "noise_level_dbfs",
                ],
                [[0, 1, "true", -20], [0.5, 2, "false", -30]],
            )
            with self.assertRaisesRegex(ValueError, "overlapping audio"):
                fuse_activity_signals(audio_windows_csv_path=audio)

            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                fuse_activity_signals(
                    detections_csv_path=visual,
                    fused_windows_csv_path=visual,
                )

    def test_rejects_invalid_duration_and_visual_coverage(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            fuse_activity_signals()
        with self.assertRaisesRegex(ValueError, "window_duration_seconds"):
            fuse_activity_signals(
                detections_csv_path="visual.csv",
                window_duration_seconds=0,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            visual = Path(temp_dir) / "visual.csv"
            self.write_csv(
                visual,
                ["timestamp_seconds", "tracking_id", "x", "y", "width", "height"],
                [[2, 1, 0, 0, 10, 10]],
            )
            with self.assertRaisesRegex(ValueError, "exceeds"):
                fuse_activity_signals(
                    detections_csv_path=visual,
                    visual_coverage_duration_seconds=1,
                )


if __name__ == "__main__":
    unittest.main()
