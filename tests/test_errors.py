"""Error-contract tests: stable codes, hierarchy, message style, catchability.

These lock in the promises made in docs/errors.md:
- every library error derives from AxonWeaveError and carries a stable code
- ApiUsageError (AXW010) is also a ValueError for backward compatibility
- messages follow the house pattern: CODE: what failed; expected X, got Y
"""
import numpy as np
import pytest

from axonweave.core.brain import BiologicalBrain
from axonweave.errors import (
    ApiUsageError,
    AxonWeaveError,
    BackendUnavailableError,
    BiologicalAssumptionError,
    DatasetIntegrityError,
    SchemaError,
    SubstrateNotInstalledError,
    UnsupportedDeviceError,
    _require,
)
from tests.conftest import make_graph


@pytest.fixture
def brain():
    return BiologicalBrain(make_graph(8, 3))


ALL_CLASSES = [
    AxonWeaveError, SubstrateNotInstalledError, DatasetIntegrityError,
    SchemaError, UnsupportedDeviceError, BiologicalAssumptionError,
    BackendUnavailableError, ApiUsageError,
]

CODES = {
    AxonWeaveError: "AXW000",
    SubstrateNotInstalledError: "AXW001",
    DatasetIntegrityError: "AXW002",
    SchemaError: "AXW003",
    UnsupportedDeviceError: "AXW004",
    BiologicalAssumptionError: "AXW005",
    BackendUnavailableError: "AXW006",
    ApiUsageError: "AXW010",
}


def test_every_error_derives_from_base():
    for cls in ALL_CLASSES:
        assert issubclass(cls, AxonWeaveError)


def test_codes_are_stable():
    for cls, code in CODES.items():
        assert cls.code == code


def test_api_usage_error_is_valueerror():
    """Backward compatibility: except ValueError keeps catching AXW010."""
    assert issubclass(ApiUsageError, ValueError)
    with pytest.raises(ValueError):
        _require(False, "legacy handler still catches this")


def test_require_helper_prefixes_code():
    with pytest.raises(ApiUsageError) as ei:
        _require(False, "expected 4 neurons, got 8")
    assert str(ei.value).startswith("AXW010:")
    assert "expected 4 neurons, got 8" in str(ei.value)


def test_require_helper_no_double_prefix():
    with pytest.raises(ApiUsageError) as ei:
        _require(False, "AXW010: already prefixed")
    assert str(ei.value) == "AXW010: already prefixed"


def test_require_passes_silently():
    _require(True, "never raised")


def test_message_style_expected_vs_got(brain):
    """Dimension errors report expected and received values."""
    from axonweave.numpy import ConnectomeLayer

    layer = ConnectomeLayer(brain.graph)
    with pytest.raises(ApiUsageError) as ei:
        layer(np.zeros((2, 7), dtype=np.float32))
    msg = str(ei.value)
    assert "AXW010" in msg
    assert str(brain.n_neurons) in msg       # expected
    assert "7" in msg                        # got


def test_unknown_option_lists_valid_values(brain):
    """Unknown-choice errors enumerate the accepted values."""
    with pytest.raises(ApiUsageError) as ei:
        brain.agent(dynamics="hodgkin_huxley")
    msg = str(ei.value)
    assert "adaptive_lif" in msg and "lif" in msg and "rate" in msg


def test_missing_prerequisite_includes_command(brain):
    """Selection-table errors tell the user the exact reinstall command."""
    brain.graph.selection_tables = None
    with pytest.raises(ApiUsageError) as ei:
        brain.graph.neurons.by_type("kenyon_cell")
    assert "axonweave substrate install male-cns:v1.0" in str(ei.value)


def test_catchable_through_base_class(brain):
    """One except AxonWeaveError catches every library error."""
    with pytest.raises(AxonWeaveError):
        brain.graph.neurons.ids([0, 999999])


def test_agent_missing_wiring_is_actionable(brain):
    class NullEnv:
        def reset(self):
            return np.zeros(4, dtype=np.float32)

        def step(self, action):
            return {"reward": 0.0, "done": True}

    agent = brain.agent(dynamics="rate")
    with pytest.raises(AxonWeaveError) as ei:
        agent.step(np.zeros(4, dtype=np.float32), NullEnv())
    assert "sense" in str(ei.value)  # names the fixing call
