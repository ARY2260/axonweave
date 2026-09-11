# Experiments

The experiment module standardizes the agent/simulation execution loop:

```text
observation → encoder → brain dynamics → decoder → action
     ↑                                              │
     └── environment step ← reward ← plasticity ←───┘
```

## Running an agent

```python
import axonweave

brain = axonweave.load("male-cns:v1.0")

agent = brain.agent(
    input=axonweave.encoders.ImageEncoder(shape=(84, 84, 3)),
    output=axonweave.decoders.ActionDecoder(actions=6),
    dynamics="lif",
    learning="dopamine_stdp",
)

result = agent.run(environment, episodes=100, log_dir="runs/exp1")
```

`agent.run` executes the canonical loop each step:

1. `observation = environment.observe()`
2. `currents = encoder(observation)`
3. `activity = brain.step(currents)` — time-stepped dynamics
4. `action = decoder(activity)`
5. `reward, next_obs = environment.step(action)`
6. `plasticity.update(reward, pre_trace, post_trace)`

## Checkpointing

Long-running experiments checkpoint substrate identity, configuration and training state:

```python
agent.save_checkpoint("runs/exp1/ckpt-0100.awb-ckpt")

# Later — refuses to load into a materially different substrate without override:
agent = axonweave.load_agent("runs/exp1/ckpt-0100.awb-ckpt")
```

A checkpoint records:

- AxonWeave version and substrate ID/version
- graph fingerprint
- dynamics, encoder, decoder and plasticity configurations
- optimizer and training metadata

## Logging

Pass `log_dir` to write episode returns, step counts, spike rates and plasticity deltas to disk. Logs are plain JSON lines — no proprietary formats.

## Evaluate without learning

```python
result = agent.run(environment, episodes=20, learn=False)
print(result.mean_return)
```

:::DOC-NOTE
`brain.simulate(...)` and `brain.experiment(...)` are aliases of the same runner for simulation-style (no-reward) and benchmark-style usage respectively.
:::
