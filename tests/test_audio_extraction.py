import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio_extraction import extract_audio


class ExtractAudioTests(unittest.TestCase):
    def test_resamples_stereo_input_and_builds_timestamped_segments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "stereo.wav"
            source_samples = np.column_stack(
                [
                    np.ones(8_000, dtype=np.float32) * 0.5,
                    np.ones(8_000, dtype=np.float32) * -0.5,
                ]
            )
            sf.write(source, source_samples, 8_000)

            result = extract_audio(
                source,
                target_sample_rate=4_000,
                segment_duration_seconds=0.75,
            )

        self.assertEqual(result.sample_rate, 4_000)
        self.assertEqual(result.waveform.ndim, 1)
        self.assertEqual(result.waveform.size, 4_000)
        self.assertEqual(len(result.segments), 2)
        self.assertAlmostEqual(result.segments[0].start_seconds, 0)
        self.assertAlmostEqual(result.segments[0].end_seconds, 0.75)
        self.assertAlmostEqual(result.segments[1].start_seconds, 0.75)
        self.assertAlmostEqual(result.segments[1].end_seconds, 1.0)
        self.assertLess(result.segments[0].rms_amplitude, 0.01)

    def test_resampling_attenuates_frequencies_above_target_nyquist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "high_frequency.wav"
            sample_rate = 8_000
            times = np.arange(sample_rate, dtype=np.float32) / sample_rate
            samples = np.sin(2 * np.pi * 3_000 * times).astype(np.float32)
            sf.write(source, samples, sample_rate)

            result = extract_audio(
                source,
                target_sample_rate=4_000,
                segment_duration_seconds=1.0,
            )

        self.assertEqual(result.waveform.size, 4_000)
        self.assertLess(float(np.sqrt(np.mean(result.waveform**2))), 0.03)

    def test_retains_final_partial_segment_and_amplitude_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "signal.wav"
            sf.write(source, np.array([0.25, -0.5, 1.0, -1.0, 0.5]), 10)

            result = extract_audio(
                source,
                target_sample_rate=10,
                segment_duration_seconds=0.3,
            )

        self.assertEqual([segment.samples.size for segment in result.segments], [3, 2])
        self.assertAlmostEqual(result.segments[0].peak_amplitude, 1.0, places=4)
        self.assertAlmostEqual(result.segments[1].peak_amplitude, 1.0, places=4)
        self.assertAlmostEqual(result.segments[1].end_seconds, 0.5)

    def test_writes_resampled_waveform_and_feature_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            waveform_path = root / "output" / "standardized.wav"
            features_path = root / "output" / "features.csv"
            sf.write(source, np.ones(8_000, dtype=np.float32) * 0.25, 8_000)

            extract_audio(
                source,
                target_sample_rate=4_000,
                segment_duration_seconds=0.5,
                waveform_output_path=waveform_path,
                features_csv_path=features_path,
            )

            output_samples, output_rate = sf.read(waveform_path)
            with features_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(output_rate, 4_000)
        self.assertEqual(output_samples.size, 4_000)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["start_seconds"], "0.000000")
        self.assertEqual(rows[0]["end_seconds"], "0.500000")
        self.assertAlmostEqual(float(rows[0]["rms_amplitude"]), 0.25, places=2)

    def test_rejects_invalid_parameters_and_output_collision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.wav"
            sf.write(source, np.zeros(100, dtype=np.float32), 100)

            with self.assertRaisesRegex(ValueError, "target_sample_rate"):
                extract_audio(source, target_sample_rate=0)
            with self.assertRaisesRegex(ValueError, "segment_duration_seconds"):
                extract_audio(source, segment_duration_seconds=0)
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                extract_audio(source, waveform_output_path=source)

    def test_rejects_segment_shorter_than_one_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.wav"
            sf.write(source, np.zeros(100, dtype=np.float32), 100)

            with self.assertRaisesRegex(ValueError, "shorter than one"):
                extract_audio(
                    source,
                    target_sample_rate=10,
                    segment_duration_seconds=0.01,
                )

    def test_requires_ffmpeg_for_video_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "recording.mp4"
            source.touch()

            from unittest.mock import patch

            with patch("src.audio_extraction.shutil.which", return_value=None):
                with self.assertRaisesRegex(FileNotFoundError, "ffmpeg"):
                    extract_audio(source)


if __name__ == "__main__":
    unittest.main()
