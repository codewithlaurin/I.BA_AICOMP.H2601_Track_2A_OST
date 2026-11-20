"""Live speech-to-text: the video's audio, paced like playback, through Azure Speech.

The de-CH locale takes Swiss German dialect and Standard German and writes Standard German.
Reads AZURE_SPEECH_KEY and AZURE_SPEECH_REGION from the environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Callable

SAMPLE_RATE = 16000
CHUNK_BYTES = SAMPLE_RATE * 2 // 10  # 100 ms of 16-bit mono


def spread_words(text: str, start: float, end: float) -> list[dict]:
    """Fallback word timing: the words of a segment spread evenly over its duration."""
    words = text.split()
    step = (end - start) / len(words) if words else 0.0
    return [{"t": round(start + i * step, 2), "w": word} for i, word in enumerate(words)]


def timed_words(result_json: str, text: str, start: float, end: float, stream_start: float) -> list[dict]:
    """Per-word times from Azure's detailed result; evenly spread when it has none."""
    try:
        best = json.loads(result_json)["NBest"][0]
        words = [{"t": round(stream_start + word["Offset"] / 1e7, 2), "w": word["Word"]}
                 for word in best["DisplayWords"]]
        if words:
            return words
    except Exception:
        pass
    return spread_words(text, start, end)


class AzureTranscriber:
    """Transcribes `video` from `start` seconds on and reports through `emit` (event format: see
    subtitles.Subtitles). realtime=True paces the audio like playback (live demo); False sends it as
    fast as it is read (batch). duration limits the stretch; the "end" event then carries "until"."""

    def __init__(self, video: Path, start: float, emit: Callable[[dict], None], language: str = "de-CH",
                 duration: float | None = None, realtime: bool = True):
        import azure.cognitiveservices.speech as speechsdk
        import imageio_ffmpeg

        key = os.environ.get("AZURE_SPEECH_KEY", "").strip()
        region = os.environ.get("AZURE_SPEECH_REGION", "").strip()
        if not key or not region:
            raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION fehlen (track_2a/.env)")

        self._emit, self._start, self._stopped = emit, start, False
        config = speechsdk.SpeechConfig(subscription=key, region=region)
        config.speech_recognition_language = language
        config.output_format = speechsdk.OutputFormat.Detailed
        config.request_word_level_timestamps()
        self._stream = speechsdk.audio.PushAudioInputStream(
            stream_format=speechsdk.audio.AudioStreamFormat(
                samples_per_second=SAMPLE_RATE, bits_per_sample=16, channels=1))
        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=config, audio_config=speechsdk.audio.AudioConfig(stream=self._stream))

        def times(result) -> tuple[float, float]:
            begin = start + result.offset / 1e7  # offset and duration are 100 ns ticks
            return round(begin, 2), round(begin + result.duration / 1e7, 2)

        def on_recognizing(event):
            if event.result.text.strip():
                begin, end = times(event.result)
                emit({"type": "interim", "start": begin, "end": end, "text": event.result.text})

        def on_recognized(event):
            result = event.result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech and result.text.strip():
                begin, end = times(result)
                emit({"type": "final", "start": begin, "end": end, "text": result.text,
                      "words": timed_words(result.json, result.text, begin, end, start)})

        def on_canceled(event):
            details = event.cancellation_details
            if details.reason == speechsdk.CancellationReason.Error and not self._stopped:
                emit({"type": "error", "message": f"Azure Speech: {details.error_details}"})

        def on_session_stopped(event):
            if not self._stopped:
                emit({"type": "end", **({"until": start + duration} if duration else {})})

        self._recognizer.recognizing.connect(on_recognizing)
        self._recognizer.recognized.connect(on_recognized)
        self._recognizer.canceled.connect(on_canceled)
        self._recognizer.session_stopped.connect(on_session_stopped)
        self._recognizer.start_continuous_recognition_async().get()

        # -re: ffmpeg delivers the audio at playback speed, so the service hears what the viewer hears.
        self._ffmpeg = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", *(["-re"] if realtime else []),
             "-ss", str(start), "-i", str(video), *(["-t", str(duration)] if duration else []),
             "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        try:
            while not self._stopped:
                data = self._ffmpeg.stdout.read(CHUNK_BYTES)
                if not data:
                    break
                self._stream.write(data)
        finally:
            self._stream.close()  # end of audio: Azure finishes the last utterance, then stops

    def stop(self) -> None:
        self._stopped = True
        self._ffmpeg.kill()
        self._recognizer.stop_continuous_recognition_async()
