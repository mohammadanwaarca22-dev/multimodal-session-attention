import csv
import tempfile
import unittest
from pathlib import Path

from src.teaching_suggestions import (
    TeachingSuggestionConfig,
    generate_teaching_suggestions,
)


SCORE_HEADERS = [
    "window_index",
    "start_seconds",
    "end_seconds",
    "activity_score",
    "activity_band",
]

FUSED_HEADERS = [
    "window_index",
    "start_seconds",
    "end_seconds",
    "audio_coverage_seconds",
    "mean_noise_level_dbfs",
]


class GenerateTeachingSuggestionsTests(unittest.TestCase):
    def write_csv(self, path, headers, rows):
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            writer.writerows(rows)

    def test_generates_sustained_low_activity_suggestion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scores = Path(temp_dir) / "scores.csv"
            self.write_csv(
                scores,
                SCORE_HEADERS,
                [
                    [0, 0, 10, 20, "lower_observed_activity"],
                    [1, 10, 20, 30, "lower_observed_activity"],
                ],
            )

            result = generate_teaching_suggestions(
                scores,
                config=TeachingSuggestionConfig(
                    minimum_low_activity_duration_seconds=20,
                ),
            )

        self.assertEqual(len(result.suggestions), 1)
        suggestion = result.suggestions[0]
        self.assertEqual(suggestion.suggestion_type, "sustained_lower_activity_proxy")
        self.assertEqual(suggestion.start_seconds, 0)
        self.assertEqual(suggestion.end_seconds, 20)
        self.assertEqual(suggestion.confidence, "low")
        self.assertTrue(suggestion.human_review_recommended)
        self.assertIn("Consider", suggestion.suggestion)
        self.assertIn("instructor judgment", result.summary)

    def test_detects_score_drop_and_high_activity_period(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scores = Path(temp_dir) / "scores.csv"
            self.write_csv(
                scores,
                SCORE_HEADERS,
                [
                    [0, 0, 10, 80, "higher_observed_activity"],
                    [1, 10, 20, 50, "moderate_observed_activity"],
                    [2, 20, 30, 70, "higher_observed_activity"],
                    [3, 30, 40, 75, "higher_observed_activity"],
                ],
            )

            result = generate_teaching_suggestions(scores)

        types = [suggestion.suggestion_type for suggestion in result.suggestions]
        self.assertEqual(types.count("sudden_activity_proxy_drop"), 1)
        high = [
            suggestion
            for suggestion in result.suggestions
            if suggestion.suggestion_type == "higher_activity_proxy_period"
        ]
        self.assertEqual(len(high), 2)
        self.assertEqual((high[1].start_seconds, high[1].end_seconds), (20, 40))

    def test_flags_elevated_noise_only_for_covered_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scores = root / "scores.csv"
            fused = root / "fused.csv"
            self.write_csv(
                scores,
                SCORE_HEADERS,
                [
                    [0, 0, 10, 50, "moderate_observed_activity"],
                    [1, 10, 20, 50, "moderate_observed_activity"],
                ],
            )
            self.write_csv(
                fused,
                FUSED_HEADERS,
                [[0, 0, 10, 10, -18], [1, 10, 20, 0, ""]],
            )

            result = generate_teaching_suggestions(
                scores,
                fused_windows_csv_path=fused,
                config=TeachingSuggestionConfig(elevated_noise_dbfs=-20),
            )

        noise = [
            suggestion
            for suggestion in result.suggestions
            if suggestion.suggestion_type == "elevated_relative_audio_level"
        ]
        self.assertEqual(len(noise), 1)
        self.assertEqual((noise[0].start_seconds, noise[0].end_seconds), (0, 10))
        self.assertIn("not calibrated", noise[0].evidence)

    def test_ignores_unscored_windows_and_breaks_sustained_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scores = Path(temp_dir) / "scores.csv"
            self.write_csv(
                scores,
                SCORE_HEADERS,
                [
                    [0, 0, 10, 10, "lower_observed_activity"],
                    [1, 10, 20, "", "insufficient_evidence"],
                    [2, 20, 30, 10, "lower_observed_activity"],
                ],
            )

            result = generate_teaching_suggestions(
                scores,
                config=TeachingSuggestionConfig(
                    minimum_low_activity_duration_seconds=11,
                ),
            )

        self.assertEqual(result.scored_window_count, 2)
        self.assertEqual(result.unscored_window_count, 1)
        self.assertFalse(
            any(
                suggestion.suggestion_type == "sustained_lower_activity_proxy"
                for suggestion in result.suggestions
            )
        )

    def test_writes_structured_suggestion_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scores = root / "scores.csv"
            output = root / "reports" / "suggestions.csv"
            self.write_csv(
                scores,
                SCORE_HEADERS,
                [[0, 0, 10, 80, "higher_observed_activity"]],
            )

            generate_teaching_suggestions(scores, suggestions_csv_path=output)

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["suggestion_type"], "higher_activity_proxy_period")
        self.assertEqual(rows[0]["human_review_recommended"], "true")

    def test_rejects_invalid_config_alignment_and_input_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scores = root / "scores.csv"
            fused = root / "fused.csv"
            self.write_csv(scores, SCORE_HEADERS, [[0, 0, 10, 50, "moderate"]])
            with self.assertRaisesRegex(ValueError, "thresholds"):
                generate_teaching_suggestions(
                    scores,
                    config=TeachingSuggestionConfig(
                        low_activity_threshold=70,
                        high_activity_threshold=60,
                    ),
                )

            self.write_csv(
                fused,
                FUSED_HEADERS,
                [[0, 0, 11, 10, -20]],
            )
            with self.assertRaisesRegex(ValueError, "do not align"):
                generate_teaching_suggestions(
                    scores,
                    fused_windows_csv_path=fused,
                )

            self.write_csv(scores, SCORE_HEADERS, [[0, 0, 10, 120, "high"]])
            with self.assertRaisesRegex(ValueError, "within"):
                generate_teaching_suggestions(scores)


if __name__ == "__main__":
    unittest.main()
