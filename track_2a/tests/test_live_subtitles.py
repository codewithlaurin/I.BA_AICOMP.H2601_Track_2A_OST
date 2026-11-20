"""Subtitle state of the live demo, with a fake transcriber and a fake clock (no network)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimcheck.live.subtitles import Subtitles  # noqa: E402


class FakeTranscriber:
    def __init__(self, start, emit):
        self.start, self.emit, self.stopped = start, emit, False

    def stop(self):
        self.stopped = True

    def final(self, start, end, text):
        self.emit({"type": "final", "start": start, "end": end, "text": text,
                   "words": [{"t": start, "w": word} for word in text.split()]})


class SubtitlesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "video.subtitles.jsonl"
        self.now = 100.0
        self.sessions: list[FakeTranscriber] = []

    def tearDown(self):
        self.tmp.cleanup()

    def subtitles(self, live=True) -> Subtitles:
        def start(position, emit):
            self.sessions.append(FakeTranscriber(position, emit))
            return self.sessions[-1]
        return Subtitles(self.cache, start if live else None, clock=lambda: self.now)

    def test_playing_opens_one_session_and_returns_text(self):
        subs = self.subtitles()
        self.assertFalse(subs.sync(0, False, 0)["live"])           # paused: nothing starts
        self.assertTrue(subs.sync(0, True, 0)["live"])
        self.now += 3
        subs.sync(3, True, 0)
        self.assertEqual(len(self.sessions), 1)                    # still the same session
        self.sessions[0].emit({"type": "interim", "start": 0.5, "end": 2.5, "text": "Guten"})
        self.assertEqual(subs.sync(3, True, 0)["interim"]["text"], "Guten")
        self.sessions[0].final(0.5, 4.0, "Guten Abend zusammen.")
        data = subs.sync(3, True, 0)
        self.assertIsNone(data["interim"])
        self.assertEqual([s["text"] for s in data["segments"]], ["Guten Abend zusammen."])
        self.assertEqual(subs.sync(3, True, 1)["segments"], [])    # the page already has it

    def test_seek_forward_restarts_at_the_new_position(self):
        subs = self.subtitles()
        subs.sync(0, True, 0)
        self.sessions[0].final(1, 9, "Erster Satz.")
        self.now += 10
        subs.sync(300, True, 1)
        self.assertTrue(self.sessions[0].stopped)
        self.assertEqual(self.sessions[1].start, 300)
        self.assertEqual(subs.ranges, [[0, 9]])

    def test_cached_stretch_plays_without_a_session_until_its_end_is_near(self):
        subs = self.subtitles()
        subs.sync(0, True, 0)
        self.sessions[0].final(1, 30, "Ein langer Satz.")
        subs.shutdown()

        again = self.subtitles()
        self.assertEqual([s["text"] for s in again.finals], ["Ein langer Satz."])
        self.assertEqual(again.ranges, [[0, 30]])
        self.assertFalse(again.sync(5, True, 0)["live"])           # 25 s of subtitles ahead
        self.assertEqual(len(self.sessions), 1)
        self.assertTrue(again.sync(25, True, 1)["live"])           # the cached stretch ends soon
        self.assertEqual(self.sessions[1].start, 30)

    def test_paused_player_stops_the_session(self):
        subs = self.subtitles()
        subs.sync(0, True, 0)
        self.now += 1
        subs.stop_if_idle()
        self.assertFalse(self.sessions[0].stopped)
        self.now += 2
        subs.stop_if_idle()
        self.assertTrue(self.sessions[0].stopped)

    def test_end_of_video_is_remembered(self):
        subs = self.subtitles()
        subs.sync(0, True, 0)
        self.sessions[0].final(1, 5, "Schluss.")
        self.sessions[0].emit({"type": "end"})
        self.now += 2
        self.assertFalse(subs.sync(7, True, 1)["live"])
        self.assertEqual(len(self.sessions), 1)

    def test_gaps_and_batch_transcription(self):
        subs = self.subtitles(live=False)
        subs.add_transcribed(0, 100, [{"start": 1, "end": 5, "text": "Guten Abend.", "words": []}])
        subs.add_transcribed(300, 400, [])
        self.assertEqual(subs.gaps(600), [(100, 300), (400, 600)])
        again = self.subtitles(live=False)
        self.assertEqual(again.gaps(600), [(100, 300), (400, 600)])
        self.assertEqual([s["text"] for s in again.finals], ["Guten Abend."])

    def test_stretch_end_closes_at_its_own_end(self):
        subs = self.subtitles()
        subs.sync(0, True, 0)
        self.sessions[0].final(1, 5, "Satz.")
        self.sessions[0].emit({"type": "end", "until": 60})
        self.assertEqual(subs.ranges, [[0, 60]])

    def test_offline_never_starts_a_session(self):
        subs = self.subtitles(live=False)
        data = subs.sync(0, True, 0)
        self.assertFalse(data["live"])
        self.assertFalse(data["can_transcribe"])

    def test_failed_start_is_reported_once(self):
        def broken(position, emit):
            raise RuntimeError("no key")
        subs = Subtitles(self.cache, broken, clock=lambda: self.now)
        self.assertIn("no key", subs.sync(0, True, 0)["error"])
        self.assertFalse(subs.sync(1, True, 0)["live"])


if __name__ == "__main__":
    unittest.main()
