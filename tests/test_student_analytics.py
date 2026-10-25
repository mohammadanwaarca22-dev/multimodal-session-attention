import csv
import tempfile
import unittest
from pathlib import Path

from src.attendance import calculate_attendance
from src.engagement_scoring import score_fused_activity
from src.signal_fusion import fuse_activity_signals
from src.student_analytics import generate_student_analytics


ATTENDANCE_HEADERS = [
    "tracking_id",
    "observed_presence_seconds",
    "required_presence_seconds",
    "observation_count",
    "presence_bouts",
    "meets_threshold",
    "insufficient_observations",
]

SCORE_HEADERS = [
    "window_index",
    "start_seconds",
    "end_seconds",
    "activity_score",
    "activity_band",
    "confidence",
    "available_evidence_weight",
    "visual_presence_score",
    "movement_score",
    "visual_non_inactivity_score",
    "potential_speech_score",
    "contributing_rules",
    "explanation",
]


class GenerateStudentAnalyticsTests(unittest.TestCase):
    def write_csv(self, path, headers, rows):
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            writer.writerows(rows)

    def test_keeps_individual_presence_separate_from_group_activity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            attendance = root / "attendance.csv"
            scores = root / "scores.csv"
            self.write_csv(
                attendance,
                ATTENDANCE_HEADERS,
                [
                    [1, 45, 60, 20, 2, "false", "false"],
                    [2, 60, 60, 30, 1, "true", "false"],
                ],
            )
            self.write_csv(
                scores,
                SCORE_HEADERS,
                [
                    [0, 0, 30, 70, "higher_observed_activity", "low", 1,
                     70, 50, 80, 60, "rules", "group activity only"],
                    [1, 30, 60, 50, "moderate_observed_activity", "low", 1,
                     50, 40, 60, 50, "rules", "group activity only"],
                ],
            )

            result = generate_student_analytics(
                attendance,
                activity_scores_csv_path=scores,
            )

        self.assertEqual(len(result.participants), 2)
        self.assertEqual(result.participants[0].temporary_tracking_id, 1)
        self.assertEqual(result.participants[0].observed_presence_seconds, 45)
        self.assertFalse(result.participants[0].meets_presence_duration_threshold)
        self.assertIsNone(result.participants[0].individual_activity_consistency)
        self.assertFalse(result.participants[0].individual_activity_timeline_available)
        self.assertEqual(len(result.group_activity_timeline), 2)
        self.assertAlmostEqual(result.group_activity_summary.mean_activity_score, 60        )
        self.assertEqual(
            result.group_activity_summary.higher_activity_window_count,
            1,
        )
        self.assertEqual(
            result.group_activity_summary.moderate_activity_window_count,
            1,
        )
        self.assertIn("not attributed to individuals", result.group_activity_summary.summary)
        self.assertIn("not verified student identities", result.privacy_note)

    def test_attendance_only_has_unavailable_group_activity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            attendance = Path(temp_dir) / "attendance.csv"
            self.write_csv(
                attendance,
                ATTENDANCE_HEADERS,
                [[5, 10, 30, 3, 1, "false", "false"]],
            )

            result = generate_student_analytics(attendance)

        self.assertEqual(len(result.participants), 1)
        self.assertEqual(result.group_activity_timeline, ())
        self.assertIsNone(result.group_activity_summary.mean_activity_score)
        self.assertIn("unavailable", result.group_activity_summary.summary)

    def test_writes_separate_participant_timeline_and_summary_csvs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            attendance = root / "attendance.csv"
            scores = root / "scores.csv"
            participant_output = root / "out" / "participants.csv"
            timeline_output = root / "out" / "group_timeline.csv"
            summary_output = root / "out" / "group_summary.csv"
            self.write_csv(
                attendance,
                ATTENDANCE_HEADERS,
                [[7, 12.5, 20, 8, 1, "false", "false"]],
            )
            self.write_csv(
                scores,
                SCORE_HEADERS,
                [[0, 0, 10, "", "insufficient_evidence", "low", 0,
                  "", "", "", "", "No covered features", "No score."]],
            )

            generate_student_analytics(
                attendance,
                activity_scores_csv_path=scores,
                participants_csv_path=participant_output,
                group_timeline_csv_path=timeline_output,
                group_summary_csv_path=summary_output,
            )

            with participant_output.open(newline="", encoding="utf-8") as file:
                participant_rows = list(csv.DictReader(file))
            with timeline_output.open(newline="", encoding="utf-8") as file:
                timeline_rows = list(csv.DictReader(file))
            with summary_output.open(newline="", encoding="utf-8") as file:
                summary_rows = list(csv.DictReader(file))

        self.assertEqual(len(participant_rows), 1)
        self.assertEqual(participant_rows[0]["individual_activity_consistency"], "")
        self.assertEqual(participant_rows[0]["individual_activity_timeline_available"], "false")
        self.assertEqual(len(timeline_rows), 1)
        self.assertEqual(timeline_rows[0]["activity_score"], "")
        self.assertEqual(len(summary_rows), 1)

    def test_rejects_duplicate_participants_and_invalid_attendance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            attendance = Path(temp_dir) / "attendance.csv"
            self.write_csv(
                attendance,
                ATTENDANCE_HEADERS,
                [
                    [1, 10, 20, 2, 1, "false", "false"],
                    [1, 10, 20, 2, 1, "false", "false"],
                ],
            )
            with self.assertRaisesRegex(ValueError, "duplicate tracking_id"):
                generate_student_analytics(attendance)

            self.write_csv(
                attendance,
                ATTENDANCE_HEADERS,
                [[1, 25, 20, 2, 1, "false", "false"]],
            )
            with self.assertRaisesRegex(ValueError, "conflicts with durations"):
                generate_student_analytics(attendance)

    def test_rejects_invalid_score_and_output_collisions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            attendance = root / "attendance.csv"
            scores = root / "scores.csv"
            self.write_csv(
                attendance,
                ATTENDANCE_HEADERS,
                [[1, 10, 20, 2, 1, "false", "false"]],
            )
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                generate_student_analytics(
                    attendance,
                    participants_csv_path=attendance,
                )
            self.write_csv(
                scores,
                SCORE_HEADERS,
                [[0, 0, 10, 150, "higher_observed_activity", "low", 1,
                  "", "", "", "", "rules", "explanation"]],
            )
            with self.assertRaisesRegex(ValueError, "within"):
                generate_student_analytics(
                    attendance,
                    activity_scores_csv_path=scores,
                )

    def test_end_to_end_attendance_fusion_scoring_and_analytics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detections = root / "detections.csv"
            audio = root / "audio_windows.csv"
            attendance_path = root / "attendance.csv"
            fused_path = root / "fused.csv"
            scores_path = root / "scores.csv"
            self.write_csv(
                detections,
                ["frame_number", "timestamp_seconds", "tracking_id", "x", "y", "width", "height"],
                [
                    [0, 0, 1, 0, 0, 20, 20],
                    [1, 0.5, 1, 5, 0, 20, 20],
                    [2, 1.0, 1, 10, 0, 20, 20],
                ],
            )
            self.write_csv(
                audio,
                [
                    "window_index",
                    "start_seconds",
                    "end_seconds",
                    "duration_seconds",
                    "signal_energy",
                    "rms_amplitude",
                    "peak_amplitude",
                    "amplitude_variation",
                    "short_time_energy_mean",
                    "short_time_energy_max",
                    "noise_level_dbfs",
                    "is_silence",
                    "potential_speech_activity",
                ],
                [
                    [0, 0, 1, 1, 1, 0.2, 0.2, 0, 0.04, 0.04, -14, "false", "true"],
                ],
            )

            attendance_records = calculate_attendance(
                detections,
                minimum_presence_seconds=0.5,
                attendance_csv_path=attendance_path,
            )
            fuse_activity_signals(
                detections_csv_path=detections,
                audio_windows_csv_path=audio,
                window_duration_seconds=1,
                max_visual_gap_seconds=0.6,
                visual_coverage_duration_seconds=1,
                fused_windows_csv_path=fused_path,
            )
            scored = score_fused_activity(fused_path, scores_csv_path=scores_path)
            analytics = generate_student_analytics(
                attendance_path,
                activity_scores_csv_path=scores_path,
            )

        self.assertEqual(len(attendance_records), 1)
        self.assertEqual(attendance_records[0].observed_presence_seconds, 1)
        self.assertEqual(len(analytics.participants), 1)
        self.assertIsNone(analytics.participants[0].individual_activity_consistency)
        self.assertEqual(analytics.group_activity_summary.scored_window_count, 1)
        self.assertEqual(len(analytics.group_activity_timeline), 1)
        self.assertEqual(scored.scored_window_count, 1)


if __name__ == "__main__":
    unittest.main()
