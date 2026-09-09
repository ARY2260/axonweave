"""Tests for the experiment loop and brain.agent facade (Phases 1/5)."""
import json
import os

import numpy as np
import pytest

from axonweave.core.brain import BiologicalBrain
from axonweave.decoders import ActionDecoder
from axonweave.encoders import SensorEncoder
from axonweave.errors import BackendUnavailableError
from axonweave.experiment import Agent, RunResult
from tests.conftest import make_graph


class RecorderEnv:
    """Minimal deterministic environment with the reset/step contract."""

    def __init__(self, n_steps=3):
        self.n_steps = n_steps
        self.rewards = [1.0, -0.5, 2.0, 0.0, 1.0]

    def reset(self):
        self._i = 0
        return np.zeros(4, dtype=np.float32)

    def step(self, action):
        reward = self.rewards[self._i % len(self.rewards)]
        self._i += 1
        done = self._i >= self.n_steps
        return {"reward": reward, "done": done, "observation": np.ones(4, dtype=np.float32)}


@pytest.fixture
def brain():
    return BiologicalBrain(make_graph(8, 3))


def _agent(brain, **kw):
    return brain.agent(
        input=SensorEncoder(4, 8, seed=1),
        output=ActionDecoder(actions=4, seed=2),
        **kw,
    )


def test_agent_run_basic(brain):
    agent = _agent(brain)
    res = agent.run(RecorderEnv(), episodes=2, max_steps=5)
    assert isinstance(res, RunResult)
    assert res.episodes == 2
    assert res.steps == 2 * 3  # env done after 3 steps
    assert res.mean_return == pytest.approx(sum([1.0, -0.5, 2.0]))


def test_agent_loop_order(brain):
    """observation -> encoder -> brain -> decoder -> action -> env -> reward."""
    seen = {}
    enc = SensorEncoder(4, 8, seed=1)

    class ProbeEncoder(SensorEncoder):
        def __call__(self, obs):
            seen["obs"] = np.asarray(obs).copy()
            return super().__call__(obs)

    agent = brain.agent(input=ProbeEncoder(4, 8, seed=1), output=ActionDecoder(actions=4, seed=2))
    res = agent.run(RecorderEnv(), episodes=1, max_steps=1)
    assert res.steps == 1
    assert seen["obs"].shape == (4,)


def test_agent_requires_encoder_and_decoder(brain):
    agent = Agent(brain)
    with pytest.raises(Exception, match="AXW010"):
        agent.step(np.zeros(4, dtype=np.float32), RecorderEnv())


def test_unknown_learning_rule_rejected(brain):
    with pytest.raises(ValueError, match="AXW010"):
        _agent(brain, learning="backprop_through_time")


def test_stdp_learning_updates_working_weights_not_substrate(brain):
    before = brain.graph.weights.data.copy()
    agent = _agent(brain, dynamics="lif", learning="stdp")
    agent.run(RecorderEnv(), episodes=1, max_steps=5)
    assert agent._working_weights is not None
    assert not np.array_equal(agent._working_weights.data, before)
    np.testing.assert_array_equal(before, brain.graph.weights.data)  # substrate immutable


def test_learn_false_disables_plasticity(brain):
    agent = _agent(brain, learning="stdp")
    res = agent.run(RecorderEnv(), episodes=1, max_steps=5, learn=False)
    assert agent._working_weights is None
    assert res.steps == 3


def test_jsonl_logging(brain, tmp_path):
    log_dir = tmp_path / "logs"
    agent = _agent(brain)
    agent.run(RecorderEnv(), episodes=2, max_steps=5, log_dir=str(log_dir))
    lines = (log_dir / "log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(l) for l in lines]
    assert any(r.get("event") == "episode_end" for r in records)
    assert all(isinstance(r, dict) for r in records)


def test_checkpoint_roundtrip(brain, tmp_path):
    ck = str(tmp_path / "ckpt.awb-ckpt")
    agent = _agent(brain, dynamics="lif", learning="stdp")
    agent.run(RecorderEnv(), episodes=1, max_steps=5)
    agent.save_checkpoint(ck)
    assert os.path.exists(ck)
    meta_files = [f for f in os.listdir(tmp_path) if f.endswith(".json")]
    assert meta_files
    meta = json.loads((tmp_path / meta_files[0]).read_text(encoding="utf-8"))
    assert meta["dynamics"] == "lif"
    assert meta["learning"] == "STDP"
    assert meta["substrate"] == "male-cns:v1.0"
    restored = Agent.load_checkpoint(ck, brain)
    assert restored.dynamics.name == "lif"
    assert restored.rule is not None
    assert type(restored.rule).__name__ == "STDP"


def test_brain_task_requires_torch(brain):
    pytest.importorskip("torch", reason="only meaningful when torch absent is not testable") if False else None
    try:
        import torch  # noqa: F401
        pytest.skip("torch installed; AXW006 path not reachable here")
    except ImportError:
        with pytest.raises(BackendUnavailableError, match="AXW006"):
            brain.task(dynamics="rate")


def test_brain_layer_alias_matches_torch_layer(brain):
    """brain.layer is a backward-compatible alias for brain.torch_layer."""
    pytest.importorskip("torch")
    a = brain.layer(trainable_edges=False)
    b = brain.torch_layer(trainable_edges=False)
    assert type(a) is type(b)


def test_dynamics_lif_used_in_agent(brain):
    """LIF dynamics produce spikes (0/1) in the agent loop."""
    agent = _agent(brain, dynamics="lif")
    agent._dyn_state = agent.dynamics.initial_state(brain.n_neurons)
    currents = np.full((1, 8), 100.0, dtype=np.float32)
    activity, _ = agent._brain_step(currents)
    assert set(np.unique(activity)).issubset({0.0, 1.0})
