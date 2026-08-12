# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm_omni.diffusion.offloader.chunked_transport import (
    _DLO_METRICS_ENV,
    _set_metrics_enabled_for_tests,
)


@pytest.fixture(autouse=True)
def _enable_transport_metrics(monkeypatch):
    """Counter assertions in this suite require the metrics gate to be on.

    Production keeps the gate off (zero hot-path cost unless
    VLLM_OMNI_DLO_METRICS is set); tests exercise the counting itself.
    The env var is set as well so spawned worker processes (which
    re-import the modules fresh) inherit the same gate state.
    """
    monkeypatch.setenv(_DLO_METRICS_ENV, "1")
    _set_metrics_enabled_for_tests(True)
    yield
    _set_metrics_enabled_for_tests(None)
