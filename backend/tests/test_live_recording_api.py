"""Issue A: browser live recording must be accepted, decoded and scored.

MediaRecorder produces `audio/webm;codecs=opus` in Chromium/Firefox and
`audio/mp4` (AAC) in Safari. Those exact shapes are reproduced here with ffmpeg
and pushed through the real endpoint, including the awkward cases that broke
before:

* a `blob:` upload whose filename carries no extension at all
* a declared Content-Type that does not match the actual container
* empty and truncated recordings
* provisional (persist=false) scoring and the final persisted analysis

The detector is conftest.MockVoiceDetector — a labelled TEST DOUBLE. These tests
verify DECODING and PLUMBING, never detection skill.
"""
from __future__ import annotations

import subprocess

import pytest

from tests.conftest import make_wav

pytestmark = pytest.mark.usefixtures("settings")


def _ffmpeg_available() -> bool:
    try:
        return subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    except OSError:
        return False


requires_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(), reason="ffmpeg binary not available in this environment"
)


def _encode(tmp_path_factory, name: str, args: list[str]) -> bytes:
    directory = tmp_path_factory.mktemp("live")
    src = directory / "src.wav"
    src.write_bytes(make_wav(duration_s=4.0))
    out = directory / name
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), *args, str(out)],
        check=True,
    )
    return out.read_bytes()


@pytest.fixture(scope="session")
def recorder_payloads(tmp_path_factory) -> dict[str, bytes]:
    return {
        # What Chromium/Firefox MediaRecorder emits.
        "webm_opus": _encode(
            tmp_path_factory, "live.webm", ["-f", "webm", "-c:a", "libopus"]
        ),
        # What Safari's MediaRecorder emits (MP4/AAC in an .mp4 container).
        "mp4_aac": _encode(
            tmp_path_factory, "live.mp4", ["-f", "mp4", "-c:a", "aac"]
        ),
        "m4a_aac": _encode(tmp_path_factory, "live.m4a", ["-f", "ipod", "-c:a", "aac"]),
    }


@requires_ffmpeg
class TestBrowserRecordingFormats:
    def test_webm_opus_recording_is_accepted(self, client_with_detector, recorder_payloads):
        response = client_with_detector.post(
            "/api/v1/analyze",
            files={"file": ("live-recording.webm", recorder_payloads["webm_opus"], "audio/webm;codecs=opus")},
            data={"consent": "true", "persist": "false"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["input"]["duration_s"] == pytest.approx(4.0, abs=0.4)
        assert body["input"]["sample_rate_hz"] > 0

    def test_webm_opus_with_parametersless_content_type(self, client_with_detector, recorder_payloads):
        """Some browsers send a bare `audio/webm`; the bytes decide, not the header."""
        response = client_with_detector.post(
            "/api/v1/analyze",
            files={"file": ("live-recording.webm", recorder_payloads["webm_opus"], "audio/webm")},
            data={"consent": "true", "persist": "false"},
        )
        assert response.status_code == 200, response.text

    def test_mp4_aac_recording_is_accepted(self, client_with_detector, recorder_payloads):
        response = client_with_detector.post(
            "/api/v1/analyze",
            files={"file": ("live-recording.mp4", recorder_payloads["mp4_aac"], "audio/mp4")},
            data={"consent": "true", "persist": "false"},
        )
        assert response.status_code == 200, response.text

    def test_m4a_recording_is_accepted(self, client_with_detector, recorder_payloads):
        response = client_with_detector.post(
            "/api/v1/analyze",
            files={"file": ("live-recording.m4a", recorder_payloads["m4a_aac"], "audio/x-m4a")},
            data={"consent": "true", "persist": "false"},
        )
        assert response.status_code == 200, response.text

    def test_blob_upload_without_extension_is_accepted(self, client_with_detector, recorder_payloads):
        """A MediaRecorder blob can arrive as `blob` with no extension at all."""
        response = client_with_detector.post(
            "/api/v1/analyze",
            files={"file": ("blob", recorder_payloads["webm_opus"], "audio/webm;codecs=opus")},
            data={"consent": "true", "persist": "false"},
        )
        assert response.status_code == 200, response.text

    def test_mismatched_content_type_is_ignored(self, client_with_detector, recorder_payloads):
        """The declared type is a hint; the container signature is authoritative."""
        response = client_with_detector.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "live-recording.webm",
                    recorder_payloads["webm_opus"],
                    "application/octet-stream",
                )
            },
            data={"consent": "true", "persist": "false"},
        )
        assert response.status_code == 200, response.text

    def test_evidence_carries_the_window_score_slot(self, client_with_detector, recorder_payloads):
        """A recording returns the evidence block the UI charts.

        The detector here is the TEST DOUBLE, which reports no window scores, so
        the slot is empty — but it must exist and be a list, and the measured
        features must describe the real decoded recording.
        """
        response = client_with_detector.post(
            "/api/v1/analyze",
            files={"file": ("live-recording.webm", recorder_payloads["webm_opus"], "audio/webm")},
            data={"consent": "true", "persist": "false"},
        )
        assert response.status_code == 200
        evidence = response.json()["evidence"]
        assert isinstance(evidence["time_window_scores"], list)
        names = {f["name"] for f in evidence["measured_features"]}
        assert {"duration", "sample_rate", "channels"} <= names
        assert evidence["limitations"]


class TestBrokenRecordings:
    @requires_ffmpeg
    def test_truncated_recording_is_rejected_cleanly(self, client_with_detector, recorder_payloads):
        """A header with no decodable frames is a clear 422, never a fake score."""
        data = recorder_payloads["webm_opus"]
        response = client_with_detector.post(
            "/api/v1/analyze",
            files={"file": ("live-recording.webm", data[:500], "audio/webm")},
            data={"consent": "true", "persist": "false"},
        )
        assert response.status_code in (415, 422)
        payload = response.json()["error"]
        assert payload["code"] in {"invalid_audio", "unsupported_media_type"}
        assert "corrupt" in payload["message"].lower() or "decodable" in payload["message"].lower()

    def test_empty_recording_is_rejected(self, client_with_detector):
        response = client_with_detector.post(
            "/api/v1/analyze",
            files={"file": ("live-recording.webm", b"", "audio/webm")},
            data={"consent": "true", "persist": "false"},
        )
        assert response.status_code == 422
        assert "empty" in response.json()["error"]["message"].lower()

    def test_webm_that_is_not_audio_is_rejected(self, client_with_detector):
        fake = b"\x1aE\xdf\xa3" + b"definitely not audio" * 40
        response = client_with_detector.post(
            "/api/v1/analyze",
            files={"file": ("live-recording.webm", fake, "audio/webm")},
            data={"consent": "true", "persist": "false"},
        )
        assert response.status_code in (415, 422)


class TestProvisionalAndFinalScoring:
    @requires_ffmpeg
    def test_provisional_window_does_not_persist(self, client_with_detector, recorder_payloads):
        before = client_with_detector.get("/api/v1/analyses").json()
        response = client_with_detector.post(
            "/api/v1/analyze",
            files={"file": ("live-recording.webm", recorder_payloads["webm_opus"], "audio/webm")},
            data={"consent": "true", "persist": "false"},
        )
        assert response.status_code == 200
        after = client_with_detector.get("/api/v1/analyses").json()
        assert len(after["analyses"]) == len(before["analyses"])

    @requires_ffmpeg
    def test_final_recording_is_persisted(self, client_with_detector, recorder_payloads):
        before = client_with_detector.get("/api/v1/analyses").json()
        response = client_with_detector.post(
            "/api/v1/analyze",
            files={"file": ("live-recording.webm", recorder_payloads["webm_opus"], "audio/webm")},
            data={"consent": "true"},
        )
        assert response.status_code == 200
        after = client_with_detector.get("/api/v1/analyses").json()
        assert len(after["analyses"]) == len(before["analyses"]) + 1

    @requires_ffmpeg
    def test_short_recording_is_scored_with_an_honest_warning(self, client_with_detector, tmp_path):
        """Under 2 s is accepted but flagged: the UI must say results are unreliable."""
        directory = tmp_path
        src = directory / "short.wav"
        src.write_bytes(make_wav(duration_s=0.9))
        out = directory / "short.webm"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-f", "webm", "-c:a", "libopus", str(out)],
            check=True,
        )
        response = client_with_detector.post(
            "/api/v1/analyze",
            files={"file": ("short.webm", out.read_bytes(), "audio/webm")},
            data={"consent": "true", "persist": "false"},
        )
        assert response.status_code == 200
        body = response.json()
        reasons = " ".join(body["decision"].get("reasons") or []).lower()
        assert "short" in reasons
