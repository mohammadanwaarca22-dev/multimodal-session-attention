import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.audio_analysis import analyze_audio
from src.audio_extraction import AudioExtractionResult, extract_audio


class AnalyzeAudioTests(unittest.TestCase):
    def test_extracts_window_features_and_estimates_silence_and_speech(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "signal.wav"
            sample_rate = 100
            waveform = np.concatenate(
                [
                    np.zeros(sample_rate, dtype=np.float32),
                    np.ones(sample_rate, dtype=np.float32) * 0.2,
                    np.tile(
                        np.array([-0.2, 0.2], dtype=np.float32),
                        sample_rate // 2,
                    ),
                ]
            )
            import soundfile as sf

            sf.write(source, waveform, sample_rate)
            audio = extract_audio(
                source,
                target_sample_rate=sample_rate,
                segment_duration_seconds=0.5,
            )
            result = analyze_audio(
                audio,
                silence_rms_threshold=0.01,
                speech_rms_threshold=0.1,
                analysis_window_seconds=0.5,
                aggregation_interval_seconds=1.0,
                short_time_window_seconds=0.1,
            )

        self.assertEqual(len(result.windows), 6)
        self.assertTrue(result.windows[0].is_silence)
        self.assertFalse(result.windows[0].potential_speech_activity)
        self.assertTrue(result.windows[2].potential_speech_activity)
        self.assertAlmostEqual(result.windows[2].rms_amplitude, 0.2, places=2)
        self.assertAlmostEqual(result.windows[2].peak_amplitude, 0.2, places=2)
        self.assertAlmostEqual(result.windows[2].amplitude_variation, 0, places=2)
        self.assertAlmostEqual(result.windows[4].short_time_energy_max, 0.04, places=2)
        self.assertEqual(len(result.intervals), 3)
        self.assertAlmostEqual(result.intervals[0].silence_ratio, 1.0)
        self.assertAlmostEqual(result.intervals[1].potential_speech_ratio, 1.0)
        self.assertAlmostEqual(result.session_summary.silence_ratio, 1 / 3)
        self.assertAlmostEqual(result.session_summary.potential_speech_ratio, 2 / 3)

    def test_writes_window_interval_and_session_csv_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "signal.wav"
            import soundfile as sf

            sf.write(source, np.ones(20, dtype=np.float32) * 0.1, 10)
            audio = extract_audio(source, target_sample_rate=10)
            windows_path = root / "reports" / "windows.csv"
            intervals_path = root / "reports" / "intervals.csv"
            summary_path = root / "reports" / "summary.csv"

            analyze_audio(
                audio,
                analysis_window_seconds=0.5,
                aggregation_interval_seconds=1.0,
                windows_csv_path=windows_path,
                intervals_csv_path=intervals_path,
                summary_csv_path=summary_path,
            )

            with windows_path.open(newline="", encoding="utf-8") as file:
                window_rows = list(csv.DictReader(file))
            with intervals_path.open(newline="", encoding="utf-8") as file:
                interval_rows = list(csv.DictReader(file))
            with summary_path.open(newline="", encoding="utf-8") as file:
                summary_rows = list(csv.DictReader(file))

        self.assertEqual(len(window_rows), 4)
        self.assertEqual(len(interval_rows), 2)
        self.assertEqual(len(summary_rows), 1)
        self.assertEqual(window_rows[0]["potential_speech_activity"], "true")

    def test_rejects_invalid_thresholds_and_output_collisions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "signal.wav"
            import soundfile as sf

            sf.write(source, np.zeros(20, dtype=np.float32), 10)
            audio = extract_audio(source, target_sample_rate=10)

            with self.assertRaisesRegex(ValueError, "speech_rms_threshold"):
                analyze_audio(
                    audio,
                    silence_rms_threshold=0.1,
                    speech_rms_threshold=0.05,
                )
            with self.assertRaisesRegex(ValueError, "analysis_window_seconds"):
                analyze_audio(audio, analysis_window_seconds=0)
            with self.assertRaisesRegex(ValueError, "output paths"):
                analyze_audio(audio, windows_csv_path=root / "same.csv",
                              intervals_csv_path=root / "same.csv")

    def test_silence_quiet_audio_and_loud_noise_follow_configured_rms_rules(self):
        sample_rate = 100
        waveform = np.concatenate(
            [
                np.zeros(sample_rate, dtype=np.float32),
                np.full(sample_rate, 0.03, dtype=np.float32),
                np.tile(
                    np.array([-0.3, 0.3], dtype=np.float32),
                    sample_rate // 2,
                ),
            ]
        )
        audio = AudioExtractionResult(
            sample_rate=sample_rate,
            waveform=waveform,
            segments=(),
        )

        result = analyze_audio(
            audio,
            silence_rms_threshold=0.01,
            speech_rms_threshold=0.1,
            analysis_window_seconds=1,
            aggregation_interval_seconds=1,
            short_time_window_seconds=0.1,
        )

        self.assertEqual(len(result.windows), 3)
        self.assertTrue(result.windows[0].is_silence)
        self.assertFalse(result.windows[0].potential_speech_activity)
        self.assertFalse(result.windows[1].is_silence)
        self.assertFalse(result.windows[1].potential_speech_activity)
        self.assertTrue(result.windows[2].potential_speech_activity)
        self.assertGreater(result.windows[2].noise_level_dbfs,
                           result.windows[1].noise_level_dbfs)
        self.assertEqual(
            result.windows[2].potential_speech_activity,
            True,
            "RMS threshold flags loud noise as potential speech; it is not VAD.",
        )


if __name__ == "__main__":
    unittest.main()
