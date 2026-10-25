import csv
import tempfile
import unittest
from pathlib import Path

from src.multimodal_agent import analyze_multimodal_session


class AnalyzeMultimodalSessionTests(unittest.TestCase):
    def write_csv(self, path, headers, rows):
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            writer.writerows(rows)

    def test_aligns_visual_and_audio_and_explains_group_activity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            visual = root / "detections.csv"
            audio = root / "audio.csv"
            self.write_csv(
                visual,
                ["frame_number", "timestamp_seconds", "tracking_id"],
                [[0, 0.25, 1], [1, 0.75, 1], [2, 1.25, 2]],
            )
            self.write_csv(
                audio,
                [
                    "start_seconds",
                    "end_seconds",
                    "potential_speech_activity",
                    "noise_level_dbfs",
                ],
                [[0, 1, "true", -20], [1, 2, "false", -30]],
            )

            result = analyze_multimodal_session(
                detections_csv_path=visual,
                audio_windows_csv_path=audio,
                window_duration_seconds=1,
            )

        self.assertEqual(len(result.windows), 2)
        first, second = result.windows
        self.assertEqual(first.activity_indicator, "multimodal_activity_observed")
        self.assertEqual(first.visual_detection_count, 2)
        self.assertEqual(first.visible_temporary_tracks, 1)
        self.assertEqual(first.potential_speech_seconds, 1)
        self.assertEqual(first.mean_noise_level_dbfs, -20)
        self.assertEqual(first.confidence, "low")
        self.assertIn("engagement is not inferred", first.explanation)
        self.assertEqual(second.activity_indicator, "visual_activity_observed")
        self.assertEqual(second.visible_temporary_tracks, 1)
        self.assertEqual(
            result.session_summary.session_activity_indicator,
            "multimodal_activity_observed",
        )

    def test_visual_only_input_is_explicitly_single_modality(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            visual = Path(temp_dir) / "detections.csv"
            self.write_csv(
                visual,
                ["frame_number", "timestamp_seconds", "tracking_id"],
                [[0, 2.5, 4]],
            )

            result = analyze_multimodal_session(
                detections_csv_path=visual,
                window_duration_seconds=2,
            )

        self.assertEqual(len(result.windows), 2)
        self.assertEqual(result.windows[1].available_modalities, "visual")
        self.assertIsNone(result.windows[1].audio_coverage_seconds)
        self.assertEqual(result.windows[1].activity_indicator, "visual_activity_observed")

    def test_visual_detection_on_bucket_boundary_is_not_dropped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            visual = Path(temp_dir) / "detections.csv"
            self.write_csv(
                visual,
                ["frame_number", "timestamp_seconds", "tracking_id"],
                [[0, 2.0, 4]],
            )

            result = analyze_multimodal_session(
                detections_csv_path=visual,
                window_duration_seconds=2,
            )

        self.assertEqual(len(result.windows), 2)
        self.assertEqual(result.windows[1].start_seconds, 2)
        self.assertEqual(result.windows[1].end_seconds, 4)
        self.assertEqual(result.windows[1].visual_detection_count, 1)

    def test_audio_only_input_and_speech_flag_are_handled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "audio.csv"
            self.write_csv(
                audio,
                [
                    "start_seconds",
                    "end_seconds",
                    "potential_speech_activity",
                    "noise_level_dbfs",
                ],
                [[0, 0.5, "false", -60], [0.5, 1, "true", -18]],
            )

            result = analyze_multimodal_session(
                audio_windows_csv_path=audio,
                window_duration_seconds=1,
            )

        self.assertEqual(len(result.windows), 1)
        self.assertEqual(result.windows[0].activity_indicator, "audio_activity_observed")
        self.assertAlmostEqual(result.windows[0].potential_speech_seconds, 0.5)
        self.assertAlmostEqual(result.windows[0].potential_speech_ratio, 0.5)

    def test_audio_gap_is_reported_as_insufficient_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "audio.csv"
            self.write_csv(
                audio,
                [
                    "start_seconds",
                    "end_seconds",
                    "potential_speech_activity",
                    "noise_level_dbfs",
                ],
                [[0, 1, "false", -60], [3, 4, "false", -60]],
            )

            result = analyze_multimodal_session(
                audio_windows_csv_path=audio,
                window_duration_seconds=1,
            )

        self.assertEqual(result.windows[1].activity_indicator, "insufficient_data")
        self.assertEqual(result.session_summary.insufficient_data_windows, 2)

    def test_writes_window_and_summary_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            visual = root / "detections.csv"
            windows = root / "reports" / "windows.csv"
            summary = root / "reports" / "summary.csv"
            self.write_csv(
                visual,
                ["frame_number", "timestamp_seconds", "tracking_id"],
                [[0, 0.1, 1]],
            )

            analyze_multimodal_session(
                detections_csv_path=visual,
                window_duration_seconds=1,
                windows_csv_path=windows,
                summary_csv_path=summary,
            )

            with windows.open(newline="", encoding="utf-8") as file:
                window_rows = list(csv.DictReader(file))
            with summary.open(newline="", encoding="utf-8") as file:
                summary_rows = list(csv.DictReader(file))

        self.assertEqual(len(window_rows), 1)
        self.assertEqual(window_rows[0]["confidence"], "low")
        self.assertEqual(len(summary_rows), 1)

    def test_rejects_missing_data_duplicate_rows_and_overlapping_audio(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            analyze_multimodal_session()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            visual = root / "detections.csv"
            audio = root / "audio.csv"
            self.write_csv(
                visual,
                ["timestamp_seconds", "tracking_id"],
                [[0, 1], [0, 1]],
            )
            with self.assertRaisesRegex(ValueError, "duplicate visual"):
                analyze_multimodal_session(detections_csv_path=visual)

            self.write_csv(
                audio,
                [
                    "start_seconds",
                    "end_seconds",
                    "potential_speech_activity",
                    "noise_level_dbfs",
                ],
                [[0, 1, "true", -10], [0.5, 2, "false", -20]],
            )
            with self.assertRaisesRegex(ValueError, "overlapping audio"):
                analyze_multimodal_session(audio_windows_csv_path=audio)

    def test_rejects_invalid_audio_rows_and_output_collision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            visual = root / "detections.csv"
            audio = root / "audio.csv"
            self.write_csv(
                visual,
                ["timestamp_seconds", "tracking_id"],
                [[0, 1]],
            )
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                analyze_multimodal_session(
                    detections_csv_path=visual,
                    windows_csv_path=visual,
                )

            self.write_csv(
                audio,
                [
                    "start_seconds",
                    "end_seconds",
                    "potential_speech_activity",
                    "noise_level_dbfs",
                ],
                [[0, 1, "maybe", -20]],
            )
            with self.assertRaisesRegex(ValueError, "invalid potential_speech_activity"):
                analyze_multimodal_session(audio_windows_csv_path=audio)

    def test_empty_inputs_return_insufficient_data_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            visual = Path(temp_dir) / "detections.csv"
            self.write_csv(
                visual,
                ["timestamp_seconds", "tracking_id"],
                [],
            )

            result = analyze_multimodal_session(detections_csv_path=visual)

        self.assertEqual(result.windows, ())
        self.assertEqual(
            result.session_summary.session_activity_indicator,
            "insufficient_data",
        )

    def test_validates_window_duration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            visual = Path(temp_dir) / "detections.csv"
            self.write_csv(visual, ["timestamp_seconds", "tracking_id"], [])

            with self.assertRaisesRegex(ValueError, "window_duration_seconds"):
                analyze_multimodal_session(
                    detections_csv_path=visual,
                    window_duration_seconds=0,
                )


if __name__ == "__main__":
    unittest.main()
