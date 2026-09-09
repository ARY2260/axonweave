"""Experiment loop (Phase 5): observation -> encoder -> brain -> decoder ->
action -> environment step -> reward -> plasticity update.

Also provides JSON-lines logging and checkpoint identity per ARCHITECTURE.md.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..dynamics import DynamicsModel
from ..encoders import ImageEncoder, SensorEncoder, TokenEncoder
from ..decoders import ActionDecoder, TokenDecoder
from ..errors import AxonWeaveError
from ..learning import LEARNING_RULES
from ..signals import SignalPolicy


@dataclass
class StepResult:
    action: object
    reward: float | None
    done: bool
    info: dict = field(default_factory=dict)
    observation: object | None = None


@dataclass
class RunResult:
    episodes: int
    returns: list
    mean_return: float
    steps: int
    elapsed_s: float


class Agent:
    """Environment-interaction agent over a loaded substrate.

    The canonical loop per step:
      observation -> encoder -> brain dynamics -> decoder -> action
      -> environment.step -> reward -> plasticity update
    """

    def __init__(self, brain, dynamics: str | DynamicsModel = "rate",
                 learning: str | None = None, signal_policy: SignalPolicy | None = None,
                 dt: float = 1.0, seed: int | None = None):
        self.brain = brain
        self.dynamics = dynamics if isinstance(dynamics, DynamicsModel) else _resolve_dynamics(dynamics)
        self.rule = None
        if learning is not None:
            if learning not in LEARNING_RULES:
                raise ValueError(f"AXW010: unknown learning rule {learning!r}; known: {sorted(LEARNING_RULES)}")
            self.rule = LEARNING_RULES[learning]()
        self.signal_policy = signal_policy or SignalPolicy()
        self.dt = dt
        self.seed = seed
        self.encoder = None
        self.decoder = None
        self._dyn_state = None
        self._plastic_traces = None
        self._working_weights = None

    # -- interfaces ---------------------------------------------------------
    def sense(self, encoder) -> "Agent":
        self.encoder = encoder
        return self

    def act(self, decoder) -> "Agent":
        self.decoder = decoder
        return self

    # -- core loop ----------------------------------------------------------
    def _encode(self, observation) -> np.ndarray:
        if self.encoder is None:
            raise AxonWeaveError("AXW010: no encoder configured; call agent.sense(encoder)")
        return np.asarray(self.encoder(observation), dtype=np.float32)

    def _brain_step(self, currents: np.ndarray) -> np.ndarray:
        if self._dyn_state is None:
            self._dyn_state = self.dynamics.initial_state(self.brain.n_neurons, currents.shape[:-1])
        W = self._working_weights if self._working_weights is not None else self.brain.graph.weights
        activity, self._dyn_state = self.dynamics.step(self._dyn_state, currents, W)
        return np.asarray(activity, dtype=np.float32)

    def _decode(self, activity: np.ndarray):
        if self.decoder is None:
            raise AxonWeaveError("AXW010: no decoder configured; call agent.act(decoder)")
        return self.decoder(activity)

    def step(self, observation, environment) -> StepResult:
        currents = self._encode(observation)
        pre = self._brain_step(currents)
        action = self._decode(pre)
        result = environment.step(action)
        # Support attribute-style, tuple-style and dict-style env results.
        if isinstance(result, dict):
            reward = result.get("reward")
            done = result.get("done")
            info = result.get("info", {})
            observation = result.get("observation")
        elif isinstance(result, tuple):
            reward = result[1] if len(result) >= 2 else None
            done = result[2] if len(result) >= 3 else None
            info = result[3] if len(result) >= 4 else {}
            observation = None
        else:
            reward = getattr(result, "reward", None)
            done = getattr(result, "done", None)
            info = getattr(result, "info", {})
            observation = getattr(result, "observation", None)
        if self.rule is not None and self._working_weights is None:
            # Copy-on-write: plasticity never mutates the cached substrate graph.
            import scipy.sparse as sp
            self._working_weights = self.brain.graph.weights.copy()
        if self.rule is not None and self._working_weights is not None:
            if self._plastic_traces is None:
                self._plastic_traces = self.rule.initial_traces(self.brain.n_neurons)
            post = pre if pre.ndim == 1 else pre.reshape(-1, pre.shape[-1]).mean(axis=0)
            pre_trace = currents if currents.ndim == 1 else currents.reshape(-1, currents.shape[-1]).mean(axis=0)
            self.rule.update(self._working_weights, self._plastic_traces,
                             pre_trace.astype(np.float32), np.asarray(post, dtype=np.float32),
                             dt=self.dt, reward=float(reward or 0.0))
        return StepResult(action=action, reward=reward,
                          done=bool(done) if done is not None else False,
                          info=info if isinstance(info, dict) else {},
                          observation=observation)

    def reset(self):
        self._dyn_state = None
        self._plastic_traces = None

    # -- run / checkpoint ----------------------------------------------------
    def run(self, environment, episodes: int = 1, max_steps: int | None = None,
            learn: bool = True, log_dir: str | None = None) -> RunResult:
        if not learn:
            saved_rule = self.rule
            self.rule = None
        t0 = time.time()
        returns: list[float] = []
        steps = 0
        logger = _Logger(log_dir) if log_dir else None
        for ep in range(episodes):
            obs = environment.reset()
            if hasattr(obs, "observation"):
                obs = obs.observation
            ep_return = 0.0
            self.reset()
            step_i = 0
            done = False
            while not done and (max_steps is None or step_i < max_steps):
                result = self.step(obs, environment)
                ep_return += float(result.reward or 0.0)
                steps += 1
                step_i += 1
                done = result.done
                if logger:
                    logger.write({"episode": ep, "step": step_i, "reward": result.reward,
                                  "return_so_far": ep_return})
                if done:
                    break
                # Advance to the next observation from the env result.
                obs = result.observation
                if obs is None and hasattr(environment, "observe"):
                    obs = environment.observe()
                if obs is None:
                    break
            returns.append(ep_return)
            if logger:
                logger.write({"episode": ep, "event": "episode_end", "return": ep_return})
        if not learn:
            self.rule = saved_rule
        return RunResult(episodes=episodes, returns=returns,
                         mean_return=float(np.mean(returns)) if returns else 0.0,
                         steps=steps, elapsed_s=time.time() - t0)

    # -- checkpointing (AXW-CKPT contract) ------------------------------------
    def save_checkpoint(self, path: str):
        import axonweave

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        W = self._working_weights if self._working_weights is not None else self.brain.graph.weights
        np.savez_compressed(
            p, weights_data=W.data, weights_indices=W.indices, weights_indptr=W.indptr,
            weights_shape=np.asarray(W.shape), body_ids=self.brain.graph.body_ids,
        )
        meta = {
            "axonweave_version": axonweave.__version__ if hasattr(axonweave, "__version__") else "0.1.0",
            "substrate": getattr(self.brain, "substrate_id", "male-cns:v1.0"),
            "dynamics": getattr(self.dynamics, "name", str(self.dynamics)),
            "learning": type(self.rule).__name__ if self.rule else None,
            "encoder": type(self.encoder).__name__ if self.encoder else None,
            "decoder": type(self.decoder).__name__ if self.decoder else None,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "checkpoint",
        }
        p.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, path: str, brain):
        p = Path(path)
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        # save_checkpoint passes a '.awb-ckpt' path to np.savez_compressed,
        # which appends '.npz'. Resolve the actual archive.
        if p.exists():
            zpath = p
        elif p.with_suffix(p.suffix + ".npz").exists():
            zpath = p.with_suffix(p.suffix + ".npz")
        else:
            raise FileNotFoundError(f"checkpoint archive for {p} not found")
        z = np.load(zpath, allow_pickle=False)
        zpath_handle = z
        import scipy.sparse as sp
        W = sp.csr_matrix((z["weights_data"], z["weights_indices"], z["weights_indptr"]),
                          shape=tuple(z["weights_shape"]))
        agent = cls(brain, dynamics=meta.get("dynamics") or "rate",
                    learning=_rule_key(meta.get("learning")))
        agent._working_weights = W
        z.close()
        return agent


class _Logger:
    """Append-only JSON-lines logger."""

    def __init__(self, log_dir: str | None):
        self.path = None
        if log_dir:
            d = Path(log_dir)
            d.mkdir(parents=True, exist_ok=True)
            self.path = d / "log.jsonl"

    def write(self, record: dict):
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


def _rule_key(class_name: str | None) -> str | None:
    """Map a checkpointed rule class name back to the registry key."""
    if class_name is None:
        return None
    return {"STDP": "stdp", "DopamineSTDP": "dopamine_stdp"}.get(class_name, class_name)


def _resolve_dynamics(name: str) -> DynamicsModel:
    from ..dynamics import LIF, AdaptiveLIF, Rate

    table = {"lif": LIF, "adaptive_lif": AdaptiveLIF, "rate": Rate}
    if name not in table:
        raise ValueError(f"AXW010: unknown dynamics {name!r}; known: {sorted(table)}")
    return table[name]()
