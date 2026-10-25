"""Load, standardize, and segment audio from audio or video recordings."""

from __future__ import annotations

import csv
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


_VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}


@dataclass(frozen=True)
class AudioSegment:
    index: int
    start_seconds: float
    end_seconds: float
    rms_amplitude: float
    peak_amplitude: float
    samples: np.ndarray


@dataclass(frozen=True)
class AudioExtractionResult:
    sample_rate: int
    waveform: np.ndarray
    segments: tuple[AudioSegment, ...]


def extract_audio(
    source_path: str | Path,
    *,
    target_sample_rate: int = 16_000,
    segment_duration_seconds: float = 1.0,
    waveform_output_path: str | Path | None = None,
    features_csv_path: str | Path | None = None,
) -> AudioExtractionResult:
    """Extract mono audio, resample it, and calculate fixed-window features.

    Video inputs are decoded with the ``ffmpeg`` executable, which must be
    available on ``PATH``. Audio amplitude features describe signal level only;
    they do not identify speech or its meaning.
    """
    if (
        not isinstance(target_sample_rate, int)
        or isinstance(target_sample_rate, bool)
        or target_sample_rate <= 0
    ):
        raise ValueError("target_sample_rate must be a positive integer")
    if (
        not math.isfinite(segment_duration_seconds)
        or segment_duration_seconds <= 0
    ):
        raise ValueError("segment_duration_seconds must be finite and greater than zero")

    source = Path(source_path).resolve()
    waveform_path = (
        Path(waveform_output_path).resolve()
        if waveform_output_path is not None
        else None
    )
    features_path = (
        Path(features_csv_path).resolve()
        if features_csv_path is not None
        else None
    )
    if source in (waveform_path, features_path):
        raise ValueError("Output paths must not overwrite the source recording")
    if waveform_path is not None and waveform_path == features_path:
        raise ValueError("Waveform and feature CSV output paths must be different")

    waveform = _load_waveform(source, target_sample_rate)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("The recording contains no decodable audio samples")
    if not np.isfinite(waveform).all():
        raise ValueError("The decoded audio contains non-finite samples")

    segment_samples = int(round(segment_duration_seconds * target_sample_rate))
    if segment_samples < 1:
        raise ValueError("Segment duration is shorter than one target audio sample")

    segments = _build_segments(waveform, target_sample_rate, segment_samples)
    if waveform_path is not None:
        waveform_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(waveform_path, waveform, target_sample_rate)
    if features_path is not None:
        _write_features(features_path, segments)

    return AudioExtractionResult(
        sample_rate=target_sample_rate,
        waveform=waveform,
        segments=tuple(segments),
    )


def _load_waveform(source: Path, target_sample_rate: int) -> np.ndarray:
    if source.suffix.lower() not in _VIDEO_SUFFIXES:
        return _load_audio_file(source, target_sample_rate)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError(
            "Video audio extraction requires the ffmpeg executable on PATH"
        )

    with tempfile.TemporaryDirectory(prefix="audio-extraction-") as temp_dir:
        wav_path = Path(temp_dir) / "extracted.wav"
        process = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(target_sample_rate),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        if process.returncode != 0:
            detail = process.stderr.strip() or "ffmpeg returned no error details"
            raise RuntimeError(f"Could not extract video audio: {detail}")
        return _load_audio_file(wav_path, target_sample_rate)


def _load_audio_file(source: Path, target_sample_rate: int) -> np.ndarray:
    samples, source_sample_rate = sf.read(
        source,
        dtype="float32",
        always_2d=True,
    )
    waveform = samples.mean(axis=1, dtype=np.float32)
    if source_sample_rate == target_sample_rate:
        return waveform
    return _resample_waveform(waveform, source_sample_rate, target_sample_rate)


def _resample_waveform(
    waveform: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    output_size = int(round(waveform.size * target_sample_rate / source_sample_rate))
    if output_size == 0:
        return np.empty(0, dtype=np.float32)

    rate_ratio = target_sample_rate / source_sample_rate
    cutoff = min(rate_ratio, 1.0)
    support = int(math.ceil(32 / cutoff))
    offsets = np.arange(-support + 1, support + 1, dtype=np.int64)
    output = np.empty(output_size, dtype=np.float32)

    for block_start in range(0, output_size, 8192):
        block_end = min(block_start + 8192, output_size)
        positions = (
            np.arange(block_start, block_end, dtype=np.float64)
            * source_sample_rate
            / target_sample_rate
        )
        centers = np.floor(positions).astype(np.int64)
        indices = centers[:, None] + offsets[None, :]
        distances = positions[:, None] - indices
        scaled_distances = distances * cutoff
        window_position = np.abs(distances) / support
        window = (
            0.42
            + 0.5 * np.cos(np.pi * window_position)
            + 0.08 * np.cos(2 * np.pi * window_position)
        )
        window[window_position >= 1] = 0
        weights = cutoff * np.sinc(scaled_distances) * window
        weights /= weights.sum(axis=1, keepdims=True)

        clipped_indices = np.clip(indices, 0, waveform.size - 1)
        output[block_start:block_end] = np.sum(
            waveform[clipped_indices] * weights,
            axis=1,
            dtype=np.float64,
        )

    return output


def _build_segments(
    waveform: np.ndarray,
    sample_rate: int,
    segment_samples: int,
) -> list[AudioSegment]:
    segments = []
    for index, start_sample in enumerate(range(0, waveform.size, segment_samples)):
        end_sample = min(start_sample + segment_samples, waveform.size)
        samples = waveform[start_sample:end_sample]
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        peak = float(np.max(np.abs(samples)))
        segments.append(
            AudioSegment(
                index=index,
                start_seconds=start_sample / sample_rate,
                end_seconds=end_sample / sample_rate,
                rms_amplitude=rms,
                peak_amplitude=peak,
                samples=samples,
            )
        )
    return segments


def _write_features(path: Path, segments: list[AudioSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "segment_index",
                "start_seconds",
                "end_seconds",
                "duration_seconds",
                "rms_amplitude",
                "peak_amplitude",
            ]
        )
        for segment in segments:
            writer.writerow(
                [
                    segment.index,
                    f"{segment.start_seconds:.6f}",
                    f"{segment.end_seconds:.6f}",
                    f"{segment.end_seconds - segment.start_seconds:.6f}",
                    f"{segment.rms_amplitude:.8f}",
                    f"{segment.peak_amplitude:.8f}",
                ]
            )
