"""Throwaway spike: how well does Azure Speech (de-CH) transcribe the Arena audio?

    uv run --with azure-cognitiveservices-speech --with imageio-ffmpeg \
        python scripts/spike_azure_stt.py data/demo/arena.mp4 --start 600 --duration 300

Cuts [start, start+duration] seconds out of the video, sends it to Azure and writes
<video>.azure.txt (readable, with video timestamps) and <video>.azure.jsonl next to the video.
Reads AZURE_SPEECH_KEY and AZURE_SPEECH_REGION from the environment or track_2a/.env.
"""

import argparse
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def clock(seconds: float) -> str:
    return f"{int(seconds // 3600):d}:{int(seconds % 3600 // 60):02d}:{seconds % 60:04.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=float, default=0, help="seconds into the video")
    parser.add_argument("--duration", type=float, default=300, help="seconds to transcribe")
    parser.add_argument("--language", default="de-CH")
    args = parser.parse_args()

    load_env(ROOT / ".env")
    key = os.environ.get("AZURE_SPEECH_KEY", "").strip()
    region = os.environ.get("AZURE_SPEECH_REGION", "").strip()
    if not key or not region:
        raise SystemExit("Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION (e.g. northeurope) in track_2a/.env")
    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    import azure.cognitiveservices.speech as speechsdk
    import imageio_ffmpeg

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "clip.wav"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-ss", str(args.start),
                        "-t", str(args.duration), "-i", str(args.video),
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)

        config = speechsdk.SpeechConfig(subscription=key, region=region)
        config.speech_recognition_language = args.language
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=config, audio_config=speechsdk.audio.AudioConfig(filename=str(wav)))

        segments, done, started = [], threading.Event(), time.perf_counter()

        def on_recognized(event):
            result = event.result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech and result.text.strip():
                begin = args.start + result.offset / 1e7           # offset/duration are 100 ns ticks
                segment = {"start": round(begin, 2), "end": round(begin + result.duration / 1e7, 2),
                           "text": result.text}
                segments.append(segment)
                print(f"[{clock(segment['start'])}] {segment['text']}", flush=True)

        def on_canceled(event):
            details = event.cancellation_details
            if details.reason == speechsdk.CancellationReason.Error:
                print(f"Azure error: {details.error_details}", flush=True)
            done.set()

        recognizer.recognized.connect(on_recognized)
        recognizer.canceled.connect(on_canceled)
        recognizer.session_stopped.connect(lambda event: done.set())
        recognizer.start_continuous_recognition()
        done.wait()
        recognizer.stop_continuous_recognition()
        del recognizer  # release the wav before the temp folder is removed (Windows)

    elapsed = time.perf_counter() - started
    base = args.video.with_suffix("")
    Path(f"{base}.azure.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in segments), encoding="utf-8")
    Path(f"{base}.azure.txt").write_text(
        "".join(f"[{clock(s['start'])}] {s['text']}\n" for s in segments), encoding="utf-8")
    print(f"\n{len(segments)} segments from {args.duration:.0f} s of audio in {elapsed:.0f} s "
          f"-> {base}.azure.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
