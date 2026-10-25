"""Estimate noise trends and potential speech activity from extracted audio."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.audio_extraction import AudioExtractionResult


@dataclass(frozen=True)
class AudioWindowMetrics:
    window_index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    signal_energy: float
    rms_amplitude: float
    peak_amplitude: float
    amplitude_variation: float
    short_time_energy_mean: float
    short_time_energy_max: float
    noise_level_dbfs: float
    is_silence: bool
    potential_speech_activity: bool


@dataclass(frozen=True)
class AudioIntervalSummary:
    interval_index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    window_count: int
    noise_level_dbfs: float
    silence_duration_seconds: float
    silence_ratio: float
    potential_speech_duration_seconds: float
    potential_speech_ratio: float


@dataclass(frozen=True)
class AudioSessionSummary:
    duration_seconds: float
    sample_rate: int
    window_count: int
    signal_energy: float
    rms_amplitude: float
    peak_amplitude: float
    amplitude_variation: float
    noise_level_dbfs: float
    silence_duration_seconds: float
    silence_ratio: float
    potential_speech_duration_seconds: float
    potential_speech_ratio: float


@dataclass(frozen=True)
class AudioAnalysisResult:
    windows: tuple[AudioWindowMetrics, ...]
    intervals: tuple[AudioIntervalSummary, ...]
    session_summary: AudioSessionSummary


def analyze_audio(
    audio: AudioExtractionResult,
    *,
    silence_rms_threshold: float = 0.01,
    speech_rms_threshold: float = 0.02,
    analysis_window_seconds: float = 1.0,
    aggregation_interval_seconds: float = 10.0,
    short_time_window_seconds: float = 0.025,
    windows_csv_path: str | Path | None = None,
    intervals_csv_path: str | Path | None = None,
    summary_csv_path: str | Path | None = None,
) -> AudioAnalysisResult:
    """Analyze waveform energy and estimate silence and potential speech.

    Speech activity is a simple RMS-threshold heuristic and can also flag
    music or other loud sounds. Noise level is relative dBFS, not calibrated
    acoustic sound-pressure level. No transcription or speaker identification
    is performed.
    """
    _validate_parameters(
        silence_rms_threshold,
        speech_rms_threshold,
        analysis_window_seconds,
        aggregation_interval_seconds,
        short_time_window_seconds,
    )
    waveform = np.asarray(audio.waveform, dtype=np.float32)
    if not isinstance(audio.sample_rate, int) or audio.sample_rate <= 0:
        raise ValueError("Audio sample rate must be a positive integer")
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("Audio waveform must be a non-empty mono signal")
    if not np.isfinite(waveform).all():
        raise ValueError("Audio waveform contains non-finite samples")

    output_paths = [
        Path(path).resolve()
        for path in (windows_csv_path, intervals_csv_path, summary_csv_path)
        if path is not None
    ]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("Audio analysis output paths must be different")

    sample_rate = audio.sample_rate
    window_samples = max(1, int(round(analysis_window_seconds * sample_rate)))
    interval_samples = max(
        1, int(round(aggregation_interval_seconds * sample_rate))
    )
    short_time_samples = max(
        1, int(round(short_time_window_seconds * sample_rate))
    )
    windows = tuple(
        _measure_window(
            waveform,
            sample_rate,
            start_sample,
            min(start_sample + window_samples, waveform.size),
            index,
            short_time_samples,
            silence_rms_threshold,
            speech_rms_threshold,
        )
        for index, start_sample in enumerate(range(0, waveform.size, window_samples))
    )
    intervals = _summarize_intervals(
        waveform,
        sample_rate,
        windows,
        interval_samples,
        short_time_samples,
        silence_rms_threshold,
        speech_rms_threshold,
    )
    session_summary = _summarize_session(
        waveform,
        sample_rate,
        windows,
        short_time_samples,
        silence_rms_threshold,
        speech_rms_threshold,
    )

    if windows_csv_path is not None:
        _write_windows_csv(Path(windows_csv_path), windows)
    if intervals_csv_path is not None:
        _write_intervals_csv(Path(intervals_csv_path), intervals)
    if summary_csv_path is not None:
        _write_summary_csv(Path(summary_csv_path), session_summary)

    return AudioAnalysisResult(windows, intervals, session_summary)


def _validate_parameters(
    silence_threshold: float,
    speech_threshold: float,
    analysis_window_seconds: float,
    aggregation_interval_seconds: float,
    short_time_window_seconds: float,
) -> None:
    for name, value in (
        ("silence_rms_threshold", silence_threshold),
        ("speech_rms_threshold", speech_threshold),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if speech_threshold < silence_threshold:
        raise ValueError(
            "speech_rms_threshold must be greater than or equal to "
            "silence_rms_threshold"
        )
    for name, value in (
        ("analysis_window_seconds", analysis_window_seconds),
        ("aggregation_interval_seconds", aggregation_interval_seconds),
        ("short_time_window_seconds", short_time_window_seconds),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero")


def _measure_window(
    waveform: np.ndarray,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
    index: int,
    short_time_samples: int,
    silence_threshold: float,
    speech_threshold: float,
) -> AudioWindowMetrics:
    samples = waveform[start_sample:end_sample]
    squared = np.square(samples, dtype=np.float64)
    signal_energy = float(np.sum(squared))
    rms = math.sqrt(signal_energy / samples.size)
    short_time_energies = [
        float(np.mean(np.square(samples[start : start + short_time_samples], dtype=np.float64)))
        for start in range(0, samples.size, short_time_samples)
    ]
    is_silence = rms <= silence_threshold
    return AudioWindowMetrics(
        window_index=index,
        start_seconds=start_sample / sample_rate,
        end_seconds=end_sample / sample_rate,
        duration_seconds=(end_sample - start_sample) / sample_rate,
        signal_energy=signal_energy,
        rms_amplitude=rms,
        peak_amplitude=float(np.max(np.abs(samples))),
        amplitude_variation=float(np.std(samples, dtype=np.float64)),
        short_time_energy_mean=float(np.mean(short_time_energies)),
        short_time_energy_max=max(short_time_energies),
        noise_level_dbfs=20 * math.log10(max(rms, 1e-6)),
        is_silence=is_silence,
        potential_speech_activity=(
            not is_silence and rms >= speech_threshold
        ),
    )


def _summarize_intervals(
    waveform: np.ndarray,
    sample_rate: int,
    windows: tuple[AudioWindowMetrics, ...],
    interval_samples: int,
    short_time_samples: int,
    silence_threshold: float,
    speech_threshold: float,
) -> tuple[AudioIntervalSummary, ...]:
    intervals = []
    for index, start_sample in enumerate(range(0, waveform.size, interval_samples)):
        end_sample = min(start_sample + interval_samples, waveform.size)
        start_seconds = start_sample / sample_rate
        end_seconds = end_sample / sample_rate
        covered_windows = [
            window
            for window in windows
            if window.start_seconds < end_seconds
            and window.end_seconds > start_seconds
        ]
        silence_seconds = 0.0
        speech_seconds = 0.0
        for window in covered_windows:
            overlap = max(
                0.0,
                min(window.end_seconds, end_seconds)
                - max(window.start_seconds, start_seconds),
            )
            if window.is_silence:
                silence_seconds += overlap
            if window.potential_speech_activity:
                speech_seconds += overlap

        bucket_metrics = _measure_window(
            waveform,
            sample_rate,
            start_sample,
            end_sample,
            index,
            short_time_samples,
            silence_threshold,
            speech_threshold,
        )
        duration = bucket_metrics.duration_seconds
        intervals.append(
            AudioIntervalSummary(
                interval_index=index,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                duration_seconds=duration,
                window_count=len(covered_windows),
                noise_level_dbfs=bucket_metrics.noise_level_dbfs,
                silence_duration_seconds=silence_seconds,
                silence_ratio=silence_seconds / duration,
                potential_speech_duration_seconds=speech_seconds,
                potential_speech_ratio=speech_seconds / duration,
            )
        )
    return tuple(intervals)


def _summarize_session(
    waveform: np.ndarray,
    sample_rate: int,
    windows: tuple[AudioWindowMetrics, ...],
    short_time_samples: int,
    silence_threshold: float,
    speech_threshold: float,
) -> AudioSessionSummary:
    total_duration = waveform.size / sample_rate
    silence_duration = sum(
        window.duration_seconds for window in windows if window.is_silence
    )
    speech_duration = sum(
        window.duration_seconds
        for window in windows
        if window.potential_speech_activity
    )
    metrics = _measure_window(
        waveform,
        sample_rate,
        0,
        waveform.size,
        0,
        short_time_samples,
        silence_threshold,
        speech_threshold,
    )
    return AudioSessionSummary(
        duration_seconds=total_duration,
        sample_rate=sample_rate,
        window_count=len(windows),
        signal_energy=metrics.signal_energy,
        rms_amplitude=metrics.rms_amplitude,
        peak_amplitude=metrics.peak_amplitude,
        amplitude_variation=metrics.amplitude_variation,
        noise_level_dbfs=metrics.noise_level_dbfs,
        silence_duration_seconds=silence_duration,
        silence_ratio=silence_duration / total_duration,
        potential_speech_duration_seconds=speech_duration,
        potential_speech_ratio=speech_duration / total_duration,
    )


def _write_windows_csv(path: Path, windows: tuple[AudioWindowMetrics, ...]) -> None:
    _write_dataclass_csv(path, windows)


def _write_intervals_csv(
    path: Path,
    intervals: tuple[AudioIntervalSummary, ...],
) -> None:
    _write_dataclass_csv(path, intervals)


def _write_summary_csv(path: Path, summary: AudioSessionSummary) -> None:
    _write_dataclass_csv(path, (summary,))


def _write_dataclass_csv(path: Path, records: tuple[object, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(records[0].__dataclass_fields__) if records else []
    if not field_names:
        raise ValueError("Cannot write an empty audio analysis report")
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for record in records:
            values = {
                name: (
                    str(value).lower()
                    if isinstance(value, bool)
                    else f"{value:.8f}"
                    if isinstance(value, float)
                    else value
                )
                for name, value in vars(record).items()
            }
            writer.writerow(values)
