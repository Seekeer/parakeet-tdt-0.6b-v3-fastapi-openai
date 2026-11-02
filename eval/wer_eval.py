#!/usr/bin/env python3
"""Word error rate of the running service on fixed Russian test sets.

Each set is a fixed selection of test-split recordings from a public corpus.
The selection and the reference transcripts live in eval/sets/<set>/manifest.tsv;
audio is downloaded on demand into eval/sets/<set>/audio/ (not committed) and
stored as 16 kHz mono PCM16 so the service decodes it without ffmpeg.

    python eval/wer_eval.py fetch                        # download every set's clips
    python eval/wer_eval.py fetch --set golos_crowd --create 15   # rebuild one manifest
    PARAKEET_API_KEY=<key> python eval/wer_eval.py run --url http://localhost:5092

WER and CER are computed after normalization: stress marks removed (NFC), lowercase,
"ё" -> "е", everything except letters and digits becomes a word boundary.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import tarfile
import time
import unicodedata
import urllib.request
import wave
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent
SETS_DIR = ROOT / "sets"
RESULTS_DIR = ROOT / "results"

HF = "https://huggingface.co/datasets"
# Every source is pinned to a dataset commit so refetching yields the same clips.
SETS = {
    "fleurs_ru": {
        "kind": "fleurs",
        "url": f"{HF}/google/fleurs/resolve/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/ru_ru",
        "domain": "read Wikipedia sentences",
        "license": "CC BY 4.0",
        "default_size": 40,
    },
    "golos_crowd": {
        "kind": "parquet",
        "url": f"{HF}/bond005/sberdevices_golos_10h_crowd/resolve/e634b6b810e4d30c81b4c6d8262379fe8b9f708c/data/test-00000-of-00003-cad0cfaddbc8fa71.parquet",
        "domain": "voice-assistant queries, crowd-sourced (SberDevices Golos)",
        "license": "SberDevices Golos public license (attribution)",
        "default_size": 15,
    },
    "golos_farfield": {
        "kind": "parquet",
        "url": f"{HF}/bond005/sberdevices_golos_100h_farfield/resolve/c93949f7140beef4adc404e7b54841e957f81c54/data/test-00000-of-00001-7fb58d4cda8ff9fe.parquet",
        "domain": "far-field smart-speaker commands (SberDevices Golos)",
        "license": "SberDevices Golos public license (attribution)",
        "default_size": 15,
    },
    "sova_rudevices": {
        "kind": "parquet",
        "url": f"{HF}/bond005/sova_rudevices/resolve/d9197eacfb0afff29d90a2d4e7d0d98a5dfb54bc/data/test-00000-of-00002-e3baf0fee2116c5f.parquet",
        "domain": "spontaneous speech recorded on consumer devices (SOVA RuDevices)",
        "license": "CC BY 4.0",
        "default_size": 15,
    },
    "rulibrispeech": {
        "kind": "parquet",
        "url": f"{HF}/bond005/rulibrispeech/resolve/b15916bc0ee9a35dc8cb83e6c83d242ddb9e453c/data/test-00000-of-00001-0e9859b3235b0392.parquet",
        "domain": "audiobooks (Russian LibriSpeech)",
        "license": "not stated in the dataset card",
        "default_size": 15,
    },
    "podlodka": {
        "kind": "parquet",
        "url": f"{HF}/bond005/podlodka_speech/resolve/4a9161485a70ea849ccaa54a72faa9bfc0a60627/data/test-00000-of-00001.parquet",
        "domain": "IT podcast, conversational (Podlodka)",
        "license": "not stated in the dataset card",
        "default_size": 15,
        "step": 1,  # the test split has only 20 rows
    },
}
# Parquet sets take every step-th test row (default 3) so consecutive rows from
# one speaker/book do not dominate a small selection.
PARQUET_STEP = 3
FIELDS = ["file_name", "num_samples", "reference"]
_SAFE_NAME = re.compile(r"^\d+\.wav$")
MAX_CLIP_BYTES = 50 * 1024 * 1024  # skip implausibly large archive members


# ---------------------------------------------------------------------------
# Test sets
# ---------------------------------------------------------------------------
def _to_pcm16(data: bytes) -> tuple[bytes, int]:
    """Convert a 16 kHz mono float32/PCM16 WAV to PCM16; returns (wav, samples)."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    fmt = samples = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id, size = data[offset:offset + 4], int.from_bytes(data[offset + 4:offset + 8], "little")
        body = data[offset + 8:offset + 8 + size]
        if chunk_id == b"fmt ":
            fmt = (
                int.from_bytes(body[0:2], "little"),   # format tag
                int.from_bytes(body[2:4], "little"),   # channels
                int.from_bytes(body[4:8], "little"),   # sample rate
                int.from_bytes(body[14:16], "little"), # bits per sample
            )
        elif chunk_id == b"data":
            samples = body
        offset += 8 + size + (size & 1)
    if fmt is None or samples is None:
        raise ValueError("WAV without fmt/data chunk")
    tag, channels, rate, bits = fmt
    if channels != 1 or rate != 16000:
        raise ValueError(f"expected 16 kHz mono, got {rate} Hz x{channels}")
    if tag == 3 and bits == 32:
        pcm = (np.clip(np.frombuffer(samples, "<f4"), -1.0, 1.0) * 32767).astype("<i2")
    elif tag == 1 and bits == 16:
        pcm = np.frombuffer(samples, "<i2")
    else:
        raise ValueError(f"unsupported WAV format tag={tag} bits={bits}")
    out = BytesIO()
    with wave.open(out, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(pcm.tobytes())
    return out.getvalue(), pcm.size


def _manifest_path(name: str) -> Path:
    return SETS_DIR / name / "manifest.tsv"


def _audio_dir(name: str) -> Path:
    return SETS_DIR / name / "audio"


def _read_manifest(name: str) -> list[dict]:
    with _manifest_path(name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_manifest(name: str, rows: list[dict]) -> None:
    path = _manifest_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _store(name: str, file_name: str, raw: bytes, reference: str) -> dict:
    wav, samples = _to_pcm16(raw)
    target = _audio_dir(name) / file_name
    # Write atomically so an interrupted fetch never leaves a truncated clip.
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(wav)
    os.replace(temporary, target)
    return {"file_name": file_name, "num_samples": str(samples), "reference": reference}


def _clip_ok(path: Path, num_samples: str) -> bool:
    """A clip counts as present only if it is a complete WAV of the expected length."""
    expected = int(num_samples)
    try:
        with wave.open(str(path), "rb") as wav_file:
            # The header alone is not enough: a truncated file keeps its header.
            if wav_file.getnframes() != expected:
                return False
            return len(wav_file.readframes(expected)) == expected * wav_file.getsampwidth()
    except (OSError, EOFError, wave.Error):
        return False


def _fleurs_items(spec: dict, wanted: set | None, create: int | None):
    """Yield (file_name, wav bytes, reference), one recording per sentence."""
    with urllib.request.urlopen(f"{spec['url']}/test.tsv", timeout=60) as response:
        lines = response.read().decode("utf-8").splitlines()
    rows = {}
    for line in lines:
        sentence_id, file_name, raw, *_rest = line.split("\t")
        rows[file_name] = (sentence_id, raw)
    seen = set()
    with urllib.request.urlopen(f"{spec['url']}/audio/test.tar.gz", timeout=60) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                name = os.path.basename(member.name)
                if not member.isfile() or not _SAFE_NAME.match(name) or name not in rows:
                    continue
                sentence_id, reference = rows[name]
                if create:
                    if sentence_id in seen:
                        continue
                    seen.add(sentence_id)
                elif name not in wanted:
                    continue
                if member.size > MAX_CLIP_BYTES:
                    continue
                yield name, archive.extractfile(member).read(), reference


def _parquet_items(spec: dict, wanted: set | None, create: int | None):
    """Yield (file_name, wav bytes, reference) reading the parquet file lazily."""
    import fsspec
    import pyarrow.parquet as pq

    step = spec.get("step", PARQUET_STEP)
    fs, path = fsspec.core.url_to_fs(spec["url"])
    with fs.open(path, "rb", block_size=2**20, cache_type="readahead") as handle:
        parquet = pq.ParquetFile(handle, pre_buffer=False)
        index = 0
        for batch in parquet.iter_batches(batch_size=16, columns=["audio", "transcription"]):
            for row in batch.to_pylist():
                file_name = f"{index:05d}.wav"
                index += 1
                reference = (row["transcription"] or "").strip()
                if create:
                    if (index - 1) % step or not reference:
                        continue
                elif file_name not in wanted:
                    continue
                yield file_name, row["audio"]["bytes"], reference


def fetch(name: str, create: int | None) -> None:
    spec = SETS[name]
    audio_dir = _audio_dir(name)
    audio_dir.mkdir(parents=True, exist_ok=True)
    wanted = expected = None
    if not create:
        expected = {row["file_name"]: row for row in _read_manifest(name)}
        wanted = {
            file_name
            for file_name, row in expected.items()
            if not _clip_ok(audio_dir / file_name, row["num_samples"])
        }
        if not wanted:
            print(f"[{name}] all clips already present")
            return

    items = _fleurs_items if spec["kind"] == "fleurs" else _parquet_items
    picked, mismatched = [], []
    for file_name, raw, reference in items(spec, wanted, create):
        stored = _store(name, file_name, raw, reference)
        if expected is not None:
            row = expected[file_name]
            if (stored["num_samples"], stored["reference"]) != (row["num_samples"], row["reference"]):
                # The source no longer matches the manifest; never score stale pairs.
                (audio_dir / file_name).unlink()
                mismatched.append(file_name)
        picked.append(stored)
        print(f"[{name}] {len(picked):3d} {file_name}")
        if create and len(picked) >= create:
            break
        if not create:
            wanted.discard(file_name)
            if not wanted:
                break
    if create:
        _write_manifest(name, picked)
        print(f"[{name}] manifest written ({len(picked)} clips)")
        if len(picked) < create:
            print(f"[{name}] warning: source had only {len(picked)} usable clips")
    if mismatched:
        sys.exit(f"[{name}] source differs from manifest for: {sorted(mismatched)}")
    if wanted:
        sys.exit(f"[{name}] missing from source: {sorted(wanted)}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    # Drop stress marks, then recompose so decomposed "й"/"ё" survive.
    text = unicodedata.normalize("NFD", text)
    text = text.replace("́", "").replace("̀", "")
    text = unicodedata.normalize("NFC", text)
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w]|_", " ", text)
    return " ".join(text.split())


def edit_distance(ref: list, hyp: list) -> int:
    previous = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, 1):
        current = [i]
        for j, hyp_item in enumerate(hyp, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def _score_set(client: httpx.Client, name: str, model: str) -> tuple[dict, list]:
    rows = _read_manifest(name)
    audio_dir = _audio_dir(name)
    missing = [row["file_name"] for row in rows if not _clip_ok(audio_dir / row["file_name"], row["num_samples"])]
    if missing:
        sys.exit(f"[{name}] {len(missing)} clip(s) missing; run `python eval/wer_eval.py fetch` first")

    totals = {"word_errors": 0, "words": 0, "char_errors": 0, "chars": 0, "audio_s": 0.0, "wall_s": 0.0}
    clips = []
    for row in rows:
        path = audio_dir / row["file_name"]
        started = time.perf_counter()
        with path.open("rb") as audio:
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": (path.name, audio, "audio/wav")},
                data={"model": model},
            )
        wall = time.perf_counter() - started
        response.raise_for_status()
        hypothesis = response.json()["text"]

        ref_norm, hyp_norm = normalize(row["reference"]), normalize(hypothesis)
        ref_words = ref_norm.split()
        word_errors = edit_distance(ref_words, hyp_norm.split())
        totals["word_errors"] += word_errors
        totals["words"] += len(ref_words)
        totals["char_errors"] += edit_distance(list(ref_norm), list(hyp_norm))
        totals["chars"] += len(ref_norm)
        totals["audio_s"] += int(row["num_samples"]) / 16000
        totals["wall_s"] += wall
        clips.append(
            {
                "set": name,
                "file_name": row["file_name"],
                "wer": round(word_errors / max(1, len(ref_words)), 4),
                "word_errors": word_errors,
                "words": len(ref_words),
                "reference": row["reference"],
                "hypothesis": hypothesis,
            }
        )
    summary = {
        "set": name,
        "domain": SETS[name]["domain"],
        "clips": len(rows),
        "words": totals["words"],
        "audio_seconds": round(totals["audio_s"], 1),
        "wer": round(totals["word_errors"] / max(1, totals["words"]), 4),
        "cer": round(totals["char_errors"] / max(1, totals["chars"]), 4),
        "speed_x_realtime": round(totals["audio_s"] / max(totals["wall_s"], 1e-9), 1),
        "_word_errors": totals["word_errors"],
    }
    return summary, clips


def run(url: str, model: str, api_key: str, names: list[str]) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    summaries, clips = [], []
    with httpx.Client(base_url=url, headers=headers, timeout=600) as client:
        for name in names:
            summary, set_clips = _score_set(client, name, model)
            summaries.append(summary)
            clips.extend(set_clips)
            print(
                f"{name:16} clips {summary['clips']:3d}  words {summary['words']:5d}  "
                f"WER {summary['wer']:6.1%}  CER {summary['cer']:6.1%}"
            )

    words = sum(s["words"] for s in summaries)
    errors = sum(s.pop("_word_errors") for s in summaries)
    overall = {"wer_all_words": round(errors / max(1, words), 4), "words": words, "clips": len(clips)}
    print(f"{'all':16} clips {overall['clips']:3d}  words {words:5d}  WER {overall['wer_all_words']:6.1%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{stamp}-{model.replace('/', '_')}.json"
    report = {
        "model": model,
        "url": url,
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "normalization": "NFC, stress marks removed, lowercase, ё->е, non-alphanumerics -> space",
        "sources": {name: SETS[name]["url"] for name in names},
        "overall": overall,
        "sets": summaries,
        "clips": clips,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"details: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch_parser = sub.add_parser("fetch", help="download the test clips")
    fetch_parser.add_argument("--set", choices=sorted(SETS), help="one set (default: all)")
    fetch_parser.add_argument("--create", type=int, metavar="N", help="rebuild the manifest with N clips")
    run_parser = sub.add_parser("run", help="transcribe the clips and score them")
    run_parser.add_argument("--set", choices=sorted(SETS), help="one set (default: all)")
    run_parser.add_argument("--url", default="http://localhost:5092")
    run_parser.add_argument("--model", default="parakeet-tdt-0.6b-v3")
    args = parser.parse_args()

    names = [args.set] if args.set else list(SETS)
    if args.command == "fetch":
        if args.create and not args.set:
            parser.error("--create needs --set")
        for name in names:
            fetch(name, args.create)
    else:
        run(args.url, args.model, os.environ.get("PARAKEET_API_KEY", ""), names)


if __name__ == "__main__":
    main()
