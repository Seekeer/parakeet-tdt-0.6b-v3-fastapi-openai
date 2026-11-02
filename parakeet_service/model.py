"""Thread-safe ONNX Runtime model loading for Parakeet TDT (CPU only)."""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple

# Import config before ONNX Runtime so thread-pool environment limits are active.
from .config import (
    DEFAULT_MODEL,
    MODEL_CONFIGS,
    ORT_INTER_THREADS,
    ORT_INTRA_THREADS,
    logger,
)

import onnx_asr
import onnxruntime as ort

_PROVIDERS = ["CPUExecutionProvider"]

_ModelKey = Tuple[str, bool]
_MODELS: Dict[_ModelKey, object] = {}
_MODEL_LOCK = threading.RLock()
_RUNTIMES: Dict[str, dict] = {}


def _build_sess_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = ORT_INTRA_THREADS
    options.inter_op_num_threads = ORT_INTER_THREADS
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.set_denormal_as_zero", "1")
    options.add_session_config_entry("session.intra_op.allow_spinning", "1")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


def _session_provider_report(model: Any) -> Dict[str, List[str]]:
    report: Dict[str, List[str]] = {}
    candidates = [("model", model), ("asr", getattr(model, "asr", None))]
    for prefix, candidate in candidates:
        if candidate is None:
            continue
        for attribute in ("_model", "_encoder", "_decoder", "_decoder_joint"):
            session = getattr(candidate, attribute, None)
            get_providers = getattr(session, "get_providers", None)
            if callable(get_providers):
                report[f"{prefix}.{attribute}"] = list(get_providers())
    return report


def load_model(name: str = DEFAULT_MODEL, *, with_timestamps: bool = True):
    normalized = (name or DEFAULT_MODEL).strip().lower()
    if normalized not in MODEL_CONFIGS:
        raise ValueError(f"unknown model {name!r}; choose one of {sorted(MODEL_CONFIGS)}")
    key = (normalized, with_timestamps)

    with _MODEL_LOCK:
        cached = _MODELS.get(key)
        if cached is not None:
            return cached

        config = MODEL_CONFIGS[normalized]
        logger.info(
            "Loading %s (quant=%s) intra=%d inter=%d",
            config["hf_id"],
            config["quantization"],
            ORT_INTRA_THREADS,
            ORT_INTER_THREADS,
        )
        model = onnx_asr.load_model(
            config["hf_id"],
            quantization=config["quantization"],
            providers=_PROVIDERS,
            sess_options=_build_sess_options(),
        )
        if with_timestamps:
            model = model.with_timestamps()
        report = _session_provider_report(model)
        if report:
            logger.info("Session providers for %s: %s", normalized, report)
        _RUNTIMES[normalized] = {
            "backend": "cpu",
            "precision": config["quantization"] or "fp32",
            "sessions": report,
        }
        _MODELS[key] = model
        logger.info("Loaded %s", normalized)
        return model


def get_model(name: str = DEFAULT_MODEL):
    return load_model(name, with_timestamps=True)


def loaded_models() -> List[str]:
    with _MODEL_LOCK:
        return sorted({name for name, _timestamps in _MODELS})


def runtime_status() -> Dict[str, dict]:
    with _MODEL_LOCK:
        return dict(_RUNTIMES)
