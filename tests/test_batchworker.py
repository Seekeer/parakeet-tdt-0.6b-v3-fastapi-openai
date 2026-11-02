from __future__ import annotations

import numpy as np
import pytest

from parakeet_service.batchworker import InferencePool


class _Model:
    def recognize(self, value):
        if isinstance(value, list):
            return [float(item.sum()) for item in value]
        return float(value.sum())


@pytest.mark.asyncio
async def test_inference_pool_runs_and_rejects_after_stop():
    pool = InferencePool(lambda _name: _Model(), workers=2)
    results = await pool.submit_many(
        [np.ones(3, dtype=np.float32), np.ones(5, dtype=np.float32)], "model"
    )
    assert results == [3.0, 5.0]
    await pool.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        await pool.submit(np.ones(1, dtype=np.float32), "model")
