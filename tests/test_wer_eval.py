from __future__ import annotations

import importlib.util
import io
import wave
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "wer_eval", Path(__file__).resolve().parent.parent / "eval" / "wer_eval.py"
)
wer_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wer_eval)


def _riff(fmt_tag: int, bits: int, payload: bytes, rate: int = 16000, channels: int = 1) -> bytes:
    block = channels * bits // 8
    fmt = (
        fmt_tag.to_bytes(2, "little")
        + channels.to_bytes(2, "little")
        + rate.to_bytes(4, "little")
        + (rate * block).to_bytes(4, "little")
        + block.to_bytes(2, "little")
        + bits.to_bytes(2, "little")
    )
    body = b"WAVE" + b"fmt " + len(fmt).to_bytes(4, "little") + fmt
    body += b"data" + len(payload).to_bytes(4, "little") + payload
    return b"RIFF" + len(body).to_bytes(4, "little") + body


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Ёлка, ёж!", "елка еж"),
        ("ёлка", "елка"),            # decomposed ё
        ("йод", "йод"),              # decomposed й
        ("за́мок", "замок"),          # stress mark
        ("кто-то   в  11:00", "кто то в 11 00"),
    ],
)
def test_normalize(text, expected):
    assert wer_eval.normalize(text) == expected


def test_edit_distance():
    assert wer_eval.edit_distance([], []) == 0
    assert wer_eval.edit_distance(list("abc"), []) == 3
    assert wer_eval.edit_distance([], list("ab")) == 2
    assert wer_eval.edit_distance(list("kitten"), list("sitting")) == 3


def test_float32_wav_is_converted_to_pcm16():
    samples = np.array([0.0, 0.5, -1.0, 2.0], dtype="<f4")
    wav, count = wer_eval._to_pcm16(_riff(3, 32, samples.tobytes()))
    assert count == 4
    with wave.open(io.BytesIO(wav), "rb") as wav_file:
        assert (wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth()) == (16000, 1, 2)
        pcm = np.frombuffer(wav_file.readframes(4), "<i2")
    assert pcm.tolist() == [0, 16383, -32767, 32767]


def test_pcm16_wav_passes_through():
    samples = np.array([1, -2, 3], dtype="<i2")
    _wav, count = wer_eval._to_pcm16(_riff(1, 16, samples.tobytes()))
    assert count == 3


def test_wrong_sample_rate_is_rejected():
    with pytest.raises(ValueError, match="16 kHz mono"):
        wer_eval._to_pcm16(_riff(1, 16, b"\0\0", rate=44100))


def test_clip_ok_requires_complete_wav(tmp_path):
    wav, count = wer_eval._to_pcm16(_riff(1, 16, np.zeros(10, "<i2").tobytes()))
    clip = tmp_path / "1.wav"
    clip.write_bytes(wav)
    assert wer_eval._clip_ok(clip, str(count))
    assert not wer_eval._clip_ok(clip, str(count + 1))
    clip.write_bytes(wav[:20])
    assert not wer_eval._clip_ok(clip, str(count))
    clip.write_bytes(wav[:-4])  # header intact, samples missing
    assert not wer_eval._clip_ok(clip, str(count))
    assert not wer_eval._clip_ok(tmp_path / "missing.wav", "1")
