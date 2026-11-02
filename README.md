# Parakeet TDT 0.6B v3 — CPU transcription API

OpenAI-compatible speech-to-text service for NVIDIA
[Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), running on
**CPU only** through ONNX Runtime ([onnx-asr](https://pypi.org/project/onnx-asr/)).
Transcription endpoints are protected with **Bearer API keys**.

This is a fork of
[groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai](https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai)
with the GPU/TensorRT code, the legacy Flask server (`app.py`) and the web UI removed.

- 25 languages with automatic detection: English, Spanish, French, Russian, German,
  Italian, Polish, Ukrainian, Romanian, Dutch, Hungarian, Greek, Swedish, Czech,
  Bulgarian, Portuguese, Slovak, Croatian, Danish, Finnish, Lithuanian, Slovenian,
  Latvian, Estonian, Maltese
- Punctuation, capitalization, segment and word timestamps
- Long recordings are split near pauses (Silero VAD) and processed in parallel
- Output formats: `json`, `text`, `verbose_json`, `srt`, `vtt`
- No speaker diarization

## Quick start (Docker)

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # generate a key
# edit .env: PARAKEET_API_KEYS=<key1>,<key2>,...
docker compose up -d --build
```

The first start downloads the INT8 model (~700 MB) into the `parakeet-models` volume,
so the container stays `starting` for a few minutes. Watch progress with
`docker compose logs -f`; the service is ready when `GET /health` returns `200`.

Without Compose:

```bash
docker build -t parakeet-tdt:cpu .
docker run -d --name parakeet -p 5092:5092 \
  -e PARAKEET_API_KEYS=<key> \
  -v parakeet-models:/app/models \
  parakeet-tdt:cpu
```

## Authentication

Every endpoint except `/health`, `/docs`, `/redoc` and `/openapi.json` requires

```
Authorization: Bearer <API key>
```

Missing or unknown keys get `401` with `WWW-Authenticate: Bearer`. The key is checked
before the request body is read, so anonymous clients cannot push uploads.

| Variable | Meaning |
| --- | --- |
| `PARAKEET_API_KEYS` | Comma-separated keys, e.g. one per client service |
| `PARAKEET_API_KEYS_FILE` | Path to a file with one key per line (`#` comments allowed); merged with `PARAKEET_API_KEYS`. Handy for Docker/Kubernetes secrets |
| `PARAKEET_AUTH_DISABLED` | `true` turns authentication off (local experiments only) |

- The service **refuses to start** when no keys are configured and authentication
  is not explicitly disabled.
- Keys must be at least 16 printable ASCII characters. Only their SHA-256 digests are
  kept in memory and compared in constant time.
- The shipped `docker-compose.yml` reads keys from `.env`. To use
  `PARAKEET_API_KEYS_FILE` instead, replace the `PARAKEET_API_KEYS` line there with
  `PARAKEET_API_KEYS_FILE: /run/secrets/parakeet_api_keys` and mount the file.
- To rotate or revoke a key, change the variable/file and restart the container.
- There is no TLS or rate limiting inside the service: expose it through a reverse
  proxy (nginx, Traefik, …) if it is reachable outside a trusted network.

## API

### `POST /v1/audio/transcriptions`

Multipart form, compatible with the OpenAI audio API.

| Field | Default | Notes |
| --- | --- | --- |
| `file` | — | Any format ffmpeg can decode. 16 kHz mono PCM WAV skips ffmpeg entirely |
| `model` | `parakeet-tdt-0.6b-v3` | See [Models](#models). Other names (e.g. `whisper-1`) return `400` |
| `response_format` | `json` | `json`, `text`, `verbose_json`, `srt`, `vtt` |
| `timestamp_granularities[]` | — | `word` adds word timestamps to `verbose_json` |
| `language`, `prompt`, `temperature` | — | Accepted for compatibility and ignored; language is detected automatically |

```bash
curl http://localhost:5092/v1/audio/transcriptions \
  -H "Authorization: Bearer $PARAKEET_API_KEY" \
  -F file=@audio.mp3 \
  -F response_format=verbose_json
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5092/v1", api_key="<API key>")
with open("audio.mp3", "rb") as audio:
    result = client.audio.transcriptions.create(
        model="parakeet-tdt-0.6b-v3",
        file=audio,
        response_format="text",
    )
print(result)
```

### `POST /v1/audio/transcriptions/batch`

Several files in one request (`files` form field, repeated). Returns
`{"results": [{"filename", "text", "duration"}], "batch_size"}`.

### `GET /health`

Public. Returns `200` with model, runtime and CPU details when the service is ready
and `503` otherwise (for example during shutdown). While the default model is still
loading at startup the port does not accept connections yet. Used by the Docker
healthcheck.

Interactive docs: `http://localhost:5092/docs` (use **Authorize** to enter a key).

## Models

| `model` value | Precision | Notes |
| --- | --- | --- |
| `parakeet-tdt-0.6b-v3` | INT8 | Default, fastest on CPU |
| `istupakov/parakeet-tdt-0.6b-v3-onnx` | FP32 | Same accuracy in upstream tests, slower and larger |
| `grikdotnet/parakeet-tdt-0.6b-fp16` | FP16 | |

Only `PARAKEET_DEFAULT_MODEL` is loaded at startup; other variants are downloaded and
loaded on first use.

Upstream measurements on LibriSpeech test-clean (CPU, 50 clips, ~350 s of audio):
WER 2.16 %, CER 0.56 %, ~18–19× faster than real time for all three variants.
On an i7-12700KF the INT8 model ran at ~30× real time, on an i7-4790 at ~17×.

## Configuration

All variables are optional except the API keys. Invalid values stop the service at
startup.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PARAKEET_HOST` / `PARAKEET_PORT` | `0.0.0.0` / `5092` | Listen address |
| `PARAKEET_UVICORN_WORKERS` | `1` | Uvicorn processes; each loads its own model copy |
| `PARAKEET_DEFAULT_MODEL` | `parakeet-tdt-0.6b-v3` | Model loaded at startup and used when `model` is omitted |
| `PARAKEET_MODELS_DIR` | `models/` | Model cache (`/app/models` in Docker) |
| `PARAKEET_INFER_WORKERS` | `4` | Parallel inference calls |
| `PARAKEET_ORT_INTRA_THREADS` | physical cores | ONNX Runtime threads per inference call |
| `PARAKEET_ORT_INTER_THREADS` | `1` | ONNX Runtime inter-op threads |
| `PARAKEET_AUDIO_WORKERS` | `min(8, physical cores)` | Decoding/VAD threads |
| `PARAKEET_CHUNK_MIN_SEC` / `TARGET_SEC` / `MAX_SEC` | `20` / `60` / `75` | Chunk sizes for long audio |
| `PARAKEET_VAD_THRESHOLD` | `0.5` | Silero VAD speech threshold |
| `PARAKEET_VAD_MIN_SILENCE_MS` / `SPEECH_PAD_MS` | `400` / `120` | Pause detection |
| `PARAKEET_MAX_UPLOAD_BYTES` | 256 MiB | Per-file upload limit |
| `PARAKEET_MAX_AUDIO_SECONDS` | `7200` | Per-file duration limit |
| `PARAKEET_MAX_REQUEST_CHUNKS` | `512` | Chunk limit per request |
| `PARAKEET_MAX_BATCH_FILES` / `MAX_BATCH_BYTES` | `16` / 512 MiB | Batch endpoint limits |
| `PARAKEET_FFMPEG_TIMEOUT_SEC` | `180` | ffmpeg decode timeout |
| `LOG_LEVEL` | `INFO` | Logging level |

On hybrid Intel CPUs (12th gen and later) pinning the process to P-cores helps:
`./pin_pcores.sh python server.py`.

## Local development

Requires Python 3.11 and ffmpeg on `PATH` (only needed for non-WAV input).

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt   # Windows: .venv\Scripts\pip
PARAKEET_API_KEYS=dev-key-0123456789abcdef .venv/bin/python server.py
.venv/bin/pytest -q
```

### Updating dependencies

`requirements.txt` and `requirements-dev.txt` are fully pinned locks generated with
[uv](https://docs.astral.sh/uv/) from `requirements.in` / `requirements-dev.in`.
Edit the `.in` file, then regenerate both:

```bash
UVC="uv pip compile --universal --python-version 3.11 \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match --emit-index-url --no-header"
$UVC requirements.in -o requirements.txt
$UVC requirements-dev.in -o requirements-dev.txt
```

The PyTorch CPU index keeps `torch` (needed by Silero VAD) on the CPU-only wheel.
`onnxruntime` is held at 1.23.2, the version upstream validated with onnx-asr 0.12.0.

## License

This repository has no license file of its own. It derives from
[Shadowfita/parakeet-tdt-0.6b-v2-fastapi](https://github.com/Shadowfita/parakeet-tdt-0.6b-v2-fastapi),
which is licensed under GPL-3.0. The model weights are distributed by NVIDIA under
their own license (see the model card).

## Acknowledgments

- [Shadowfita](https://github.com/Shadowfita/parakeet-tdt-0.6b-v2-fastapi) — original FastAPI implementation
- [groxaxo](https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai) — ONNX optimization, multilingual support and OpenAI compatibility this fork builds on
- [NVIDIA](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — Parakeet TDT model family
- [istupakov/onnx-asr](https://github.com/istupakov/onnx-asr) — ONNX inference for Parakeet
