import csv
import tempfile
import unittest
from pathlib import Path

from src.attendance import calculate_attendance


class CalculateAttendanceTests(unittest.TestCase):
    def write_detections(self, path, rows, header=None):
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header or ["frame_number", "timestamp_seconds", "tracking_id"])
            writer.writerows(rows)

    def test_sums_contiguous_intervals_and_applies_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            detections = Path(temp_dir) / "detections.csv"
            self.write_detections(
                detections,
                [[0, 0, 1], [1, 0.5, 1], [2, 1.0, 1], [0, 0.25, 2]],
            )

            records = calculate_attendance(
                detections,
                minimum_presence_seconds=1.0,
                gap_tolerance_seconds=0.6,
            )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].observed_presence_seconds, 1.0)
        self.assertTrue(records[0].meets_threshold)
        self.assertFalse(records[0].insufficient_observations)
        self.assertEqual(records[1].observed_presence_seconds, 0)
        self.assertFalse(records[1].meets_threshold)
        self.assertTrue(records[1].insufficient_observations)

    def test_excludes_long_gaps_and_counts_separate_bouts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            detections = Path(temp_dir) / "detections.csv"
            self.write_detections(
                detections,
                [[0, 0.0, 7], [1, 0.5, 7], [2, 4.0, 7], [3, 4.5, 7]],
            )

            record = calculate_attendance(
                detections,
                minimum_presence_seconds=0.9,
                gap_tolerance_seconds=0.6,
            )[0]

        self.assertEqual(record.observed_presence_seconds, 1.0)
        self.assertEqual(record.presence_bouts, 2)
        self.assertTrue(record.meets_threshold)

    def test_sorts_out_of_order_timestamps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            detections = Path(temp_dir) / "detections.csv"
            self.write_detections(detections, [[1, 1.0, 1], [0, 0.0, 1]])

            record = calculate_attendance(
                detections,
                minimum_presence_seconds=0.5,
                gap_tolerance_seconds=1.0,
            )[0]

        self.assertEqual(record.observed_presence_seconds, 1.0)

    def test_brief_single_detection_never_meets_positive_duration_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            detections = Path(temp_dir) / "detections.csv"
            self.write_detections(detections, [[0, 0.25, 9]])

            record = calculate_attendance(
                detections,
                minimum_presence_seconds=5,
            )[0]

        self.assertEqual(record.observed_presence_seconds, 0)
        self.assertFalse(record.meets_threshold)
        self.assertTrue(record.insufficient_observations)

    def test_writes_structured_attendance_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            detections = Path(temp_dir) / "detections.csv"
            output = Path(temp_dir) / "reports" / "attendance.csv"
            self.write_detections(detections, [[0, 0.0, 3], [1, 0.5, 3]])

            calculate_attendance(
                detections,
                minimum_presence_seconds=0.5,
                attendance_csv_path=output,
            )

            with output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tracking_id"], "3")
        self.assertEqual(rows[0]["observed_presence_seconds"], "0.500000")
        self.assertEqual(rows[0]["meets_threshold"], "true")
        self.assertEqual(rows[0]["insufficient_observations"], "false")

    def test_rejects_missing_timestamp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            detections = Path(temp_dir) / "detections.csv"
            self.write_detections(detections, [[0, "", 1]])

            with self.assertRaisesRegex(ValueError, "Missing timestamp"):
                calculate_attendance(detections, minimum_presence_seconds=1)

    def test_rejects_duplicate_track_timestamp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            detections = Path(temp_dir) / "detections.csv"
            self.write_detections(detections, [[0, 0.0, 1], [1, 0.0, 1]])

            with self.assertRaisesRegex(ValueError, "Duplicate detection"):
                calculate_attendance(detections, minimum_presence_seconds=1)

    def test_rejects_invalid_timestamp_and_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            detections = Path(temp_dir) / "detections.csv"
            self.write_detections(detections, [[0, -0.1, 1]])

            with self.assertRaisesRegex(ValueError, "Timestamp must be finite"):
                calculate_attendance(detections, minimum_presence_seconds=1)

            self.write_detections(detections, [])
            with self.assertRaisesRegex(ValueError, "minimum_presence_seconds"):
                calculate_attendance(detections, minimum_presence_seconds=0)

    def test_empty_detection_log_returns_no_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            detections = Path(temp_dir) / "detections.csv"
            output = Path(temp_dir) / "attendance.csv"
            self.write_detections(detections, [])

            records = calculate_attendance(
                detections,
                minimum_presence_seconds=1,
                attendance_csv_path=output,
            )

            self.assertEqual(records, [])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
