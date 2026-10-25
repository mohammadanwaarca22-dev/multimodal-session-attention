import csv
import tempfile
import unittest
from pathlib import Path

from src.engagement_scoring import (
    EngagementScoringConfig,
    score_fused_activity,
)
from src.signal_fusion import fuse_activity_signals


HEADERS = [
    "window_index",
    "start_seconds",
    "end_seconds",
    "visual_detection_count",
    "visible_temporary_tracks",
    "estimated_track_presence_seconds",
    "mean_movement_pixels_per_second",
    "visual_inactivity_seconds",
    "visual_coverage_seconds",
    "audio_coverage_seconds",
    "mean_noise_level_dbfs",
    "potential_speech_seconds",
    "potential_speech_ratio",
    "available_modalities",
    "data_quality",
]


class ScoreFusedActivityTests(unittest.TestCase):
    def write_csv(self, path, headers, rows):
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            writer.writerows(rows)

    def write_windows(self, path, rows):
        self.write_csv(path, HEADERS, rows)

    def test_scores_multimodal_features_and_explains_contributors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "fused.csv"
            scores = Path(temp_dir) / "reports" / "scores.csv"
            trends = Path(temp_dir) / "reports" / "trends.csv"
            self.write_windows(
                source,
                [
                    [0, 0, 10, 5, 2, 8, 25, 2, 10, 10, -20, 6, 0.6,
                     "visual+audio", "complete"],
                    [1, 10, 20, 0, 0, 0, 0, 10, 10, 10, -50, 0, 0,
                     "visual+audio", "complete"],
                ],
            )

            result = score_fused_activity(
                source,
                scores_csv_path=scores,
                trends_csv_path=trends,
            )

            with scores.open(newline="", encoding="utf-8") as file:
                score_rows = list(csv.DictReader(file))
            with trends.open(newline="", encoding="utf-8") as file:
                trend_rows = list(csv.DictReader(file))

        self.assertAlmostEqual(result.windows[0].activity_score, 68.5)
        self.assertEqual(result.windows[0].activity_band, "higher_observed_activity")
        self.assertEqual(result.windows[0].confidence, "moderate")
        self.assertIn("visual_presence", result.windows[0].contributing_rules)
        self.assertIn("potential_speech", result.windows[0].contributing_rules)
        self.assertIn("not a measure", result.windows[0].explanation)
        self.assertEqual(result.windows[1].activity_band, "lower_observed_activity")
        self.assertEqual(len(trend_rows), 2)
        self.assertEqual(trend_rows[0]["activity_band"], "higher_observed_activity")
        self.assertEqual(len(score_rows), 2)

    def test_renormalizes_available_weights_and_requires_minimum_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "fused.csv"
            self.write_windows(
                source,
                [
                    [0, 0, 10, 2, 1, 5, "", "", 10, "", "", "", "",
                     "visual", "partial"],
                    [1, 10, 20, "", "", "", "", "", "", 0, "", "", "",
                     "none", "missing"],
                ],
            )

            result = score_fused_activity(
                source,
                config=EngagementScoringConfig(minimum_available_weight=0.3),
            )

        self.assertAlmostEqual(result.windows[0].activity_score, 50)
        self.assertAlmostEqual(result.windows[0].available_evidence_weight, 0.3)
        self.assertEqual(result.windows[0].confidence, "low")
        self.assertIsNone(result.windows[1].activity_score)
        self.assertEqual(result.windows[1].activity_band, "insufficient_evidence")
        self.assertEqual(result.scored_window_count, 1)
        self.assertEqual(result.insufficient_evidence_window_count, 1)

    def test_does_not_treat_missing_modalities_as_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "fused.csv"
            self.write_windows(
                source,
                [
                    [0, 0, 10, "", "", "", "", "", "", 10, -20, 8, 0.8,
                     "audio", "partial"],
                ],
            )

            result = score_fused_activity(source)

        self.assertIsNone(result.windows[0].visual_presence_score)
        self.assertIsNone(result.windows[0].movement_score)
        self.assertIsNone(result.windows[0].visual_non_inactivity_score)
        self.assertIsNone(result.windows[0].activity_score)
        self.assertEqual(result.insufficient_evidence_window_count, 1)

    def test_configurable_weights_and_band_thresholds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "fused.csv"
            self.write_windows(
                source,
                [
                    [0, 0, 10, 1, 1, 10, 0, 0, 10, "", "", "", "",
                     "visual", "partial"],
                ],
            )
            config = EngagementScoringConfig(
                presence_weight=0.5,
                movement_weight=0,
                inactivity_weight=0.5,
                speech_weight=0,
                minimum_available_weight=0.5,
                high_activity_threshold=60,
            )

            result = score_fused_activity(source, config=config)

        self.assertEqual(result.windows[0].activity_score, 100)
        self.assertEqual(
            result.windows[0].activity_band,
            "higher_observed_activity",
        )

    def test_rejects_invalid_config_and_inconsistent_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "fused.csv"
            self.write_windows(
                source,
                [
                    [0, 0, 10, 1, 1, 2, "", 0, 10, "", "", "", "",
                     "visual", "partial"],
                ],
            )
            with self.assertRaisesRegex(ValueError, "sum to 1.0"):
                score_fused_activity(
                    source,
                    config=EngagementScoringConfig(
                        presence_weight=0.5,
                        movement_weight=0.1,
                        inactivity_weight=0.1,
                        speech_weight=0.1,
                    ),
                )
            with self.assertRaisesRegex(ValueError, "visual inactivity has no visual coverage"):
                self.write_windows(
                    source,
                    [
                        [0, 0, 10, 1, 1, 2, "", 0, "", "", "", "", "",
                         "visual", "partial"],
                    ],
                )
                score_fused_activity(source)

    def test_rejects_overlapping_windows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "fused.csv"
            self.write_windows(
                source,
                [
                    [0, 0, 10, "", "", "", "", "", "", 10, -20, 8, 0.8,
                     "audio", "partial"],
                    [1, 9, 19, "", "", "", "", "", "", 10, -20, 8, 0.8,
                     "audio", "partial"],
                ],
            )

            with self.assertRaisesRegex(ValueError, "overlapping or unsorted"):
                score_fused_activity(source)

    def test_scores_csv_produced_by_signal_fusion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            visual = root / "detections.csv"
            audio = root / "audio.csv"
            fused = root / "fused.csv"
            self.write_csv(
                visual,
                ["timestamp_seconds", "tracking_id", "x", "y", "width", "height"],
                [[0, 1, 0, 0, 20, 20], [0.5, 1, 5, 0, 20, 20]],
            )
            self.write_csv(
                audio,
                [
                    "start_seconds",
                    "end_seconds",
                    "potential_speech_activity",
                    "noise_level_dbfs",
                ],
                [[0, 1, "true", -20]],
            )

            fuse_activity_signals(
                detections_csv_path=visual,
                audio_windows_csv_path=audio,
                window_duration_seconds=1,
                visual_coverage_duration_seconds=1,
                fused_windows_csv_path=fused,
            )
            result = score_fused_activity(fused)

        self.assertEqual(result.scored_window_count, 1)
        self.assertEqual(
            result.windows[0].activity_band,
            "moderate_observed_activity",
        )


if __name__ == "__main__":
    unittest.main()
