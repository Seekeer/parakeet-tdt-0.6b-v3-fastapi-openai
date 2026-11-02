"""Bounded asynchronous CPU inference pool."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List

from .config import INFER_WORKERS, logger

import numpy as np


def _shutdown_executor(executor: ThreadPoolExecutor) -> None:
    executor.shutdown(wait=True, cancel_futures=True)


def _infer_one(get_model_fn: Callable[[str], Any], model_name: str, wav: np.ndarray):
    return get_model_fn(model_name).recognize(wav)


class InferencePool:
    """Parallel single-item ORT calls, optimized for CPU inference."""

    def __init__(self, get_model_fn, *, workers: int = INFER_WORKERS):
        self._get_model = get_model_fn
        self._workers = max(1, workers)
        self._executor = ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix="ort"
        )
        self._closed = False
        logger.info("InferencePool started (workers=%d)", self._workers)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(_shutdown_executor, self._executor)

    async def submit(self, wav: np.ndarray, model_name: str) -> Any:
        if self._closed:
            raise RuntimeError("inference pool is stopped")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, _infer_one, self._get_model, model_name, wav
        )

    async def submit_many(self, wavs: List[np.ndarray], model_name: str) -> List[Any]:
        if not wavs:
            return []
        return list(
            await asyncio.gather(*(self.submit(wav, model_name) for wav in wavs))
        )


def build_worker(get_model_fn):
    return InferencePool(get_model_fn)
