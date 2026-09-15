"""Time-series forecasting on a connectome substrate (plan §39).

The first major end-to-end scientific ML demonstration:

    multivariate time series
        -> TimeSeriesEncoder
        -> recurrent connectome (stateful LIF / rate runtime)
        -> RegressionReadout
        -> forecast

Baselines (MLP, LSTM, GRU, standard RNN) are small torch models over the
same data. The purpose is NOT to claim AxonWeave beats conventional models —
it is to establish that the connectome can function as a reusable temporal
computational substrate, and to make every assumption explicit.

What is what (plan §37 labeling requirement):

- BIOLOGICAL:      the connectome topology and body IDs (MaleCNS v1.0), and
                   the neuron-space the dynamics run over.
- ASSUMPTION:      LIF parameters (tau, threshold, ...), the seeded random
                   encoder/readout projections, input-current scaling.
- LEARNED:         readout weights (always), edge weights and gain (when
                   ``--trainable-edges``; surrogate-gradient BPTT).
- TASK-SPECIFIC:   the synthetic forecasting dataset, the loss, the baseline
                   architectures.

Usage (requires the substrate installed and torch):

    python examples/time_series_forecasting.py                     # frozen brain
    python examples/time_series_forecasting.py --trainable-edges   # BPTT
    python examples/time_series_forecasting.py --dynamics rate     # rate model

Output: per-model final validation MSE printed as a table; results are
written to run/metrics.json so runs are comparable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic dataset: multivariate slow + fast oscillators with noise.
# Task: predict the value of target channels `horizon` steps ahead.
# ---------------------------------------------------------------------------


def make_dataset(n_samples=256, seq_len=48, n_features=8, horizon=1, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 40 * np.pi, n_samples * seq_len + horizon + seq_len)
    freqs = rng.uniform(0.2, 1.2, n_features)
    phases = rng.uniform(0, 2 * np.pi, n_features)
    base = np.stack(
        [np.sin(freqs[i] * t + phases[i]) +
         0.3 * np.sin(3.1 * freqs[i] * t) +
         0.05 * rng.standard_normal(t.shape[0])
         for i in range(n_features)], axis=1,
    ).astype(np.float32)

    X, Y = [], []
    for s in range(n_samples):
        start = s * seq_len
        window = base[start:start + seq_len]
        target = base[start + seq_len:start + seq_len + horizon,
                      0]  # forecast channel 0
        X.append(window)
        Y.append(target)
    split = int(n_samples * 0.8)
    return (
        np.stack(X[:split]), np.stack(Y[:split]),
        np.stack(X[split:]), np.stack(Y[split:]),
    )


# ---------------------------------------------------------------------------
# AxonWeave model
# ---------------------------------------------------------------------------


def build_axonweave_model(brain, args, input_dim):

    from axonweave.dynamics import LIF, Rate, SurrogateLIF
    from axonweave.encoders import TimeSeriesEncoder
    from axonweave.readout import RegressionReadout
    from axonweave.torch import BrainModel

    if args.dynamics == "rate":
        dynamics = Rate()
    elif args.dynamics == "surrogate_lif":
        dynamics = SurrogateLIF()  # differentiable spiking path (BPTT)
    else:
        dynamics = LIF()
    # Gradients through spiking dynamics require the surrogate path.
    if args.trainable_edges and args.dynamics == "lif":
        print("[note] --trainable-edges with plain LIF has no gradient path; "
              "switching to SurrogateLIF (sigmoid surrogate).")
        dynamics = SurrogateLIF()

    model = BrainModel(
        brain=brain,
        encoder=TimeSeriesEncoder(
            input_dim=input_dim,
            output_dim=min(args.hidden, brain.n_neurons),
            window=args.window,
            seed=args.seed,
        ),
        dynamics=dynamics,
        readout=RegressionReadout(
            n_source=min(args.hidden, brain.n_neurons), n_outputs=1),
    ).float()
    return model.to(args.device)


def train_axonweave(model, X, Y, args):
    import torch

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    Xt = torch.tensor(X, device=args.device)
    Yt = torch.tensor(Y, device=args.device)
    losses = []
    for _ in range(args.epochs):
        model.reset_state()
        opt.zero_grad()
        out = model.forward_sequence(Xt)        # [B, T, 1]
        pred = out[:, -args.horizon:, 0]        # last steps = forecast
        loss = (pred - Yt).pow(2).mean()
        loss.backward()
        opt.step()
        model.detach_state()                    # truncated BPTT boundary
        losses.append(float(loss))
    return losses


def eval_axonweave(model, X, Y, args):
    import torch

    model.reset_state()
    with torch.no_grad():
        out = model.forward_sequence(torch.tensor(X, device=args.device))
    pred = out[:, -args.horizon:, 0].cpu().numpy()
    return float((pred - Y).pow(2).mean())


# ---------------------------------------------------------------------------
# Baselines (task-specific engineering; not part of the library)
# ---------------------------------------------------------------------------


def _baseline_trainer(net, X, Y, args, seq_model=True):
    import torch
    import torch.nn as tnn

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    mse = tnn.MSELoss()
    Xt = torch.tensor(X, device=args.device)
    Yt = torch.tensor(Y, device=args.device)
    for _ in range(args.epochs):
        opt.zero_grad()
        out = net(Xt)
        pred = out[:, -args.horizon:, 0] if seq_model else out[:, 0]
        loss = mse(pred, Yt)
        loss.backward()
        opt.step()
    return net


def run_baselines(X, Y, vX, vY, args):
    import torch
    import torch.nn as tnn

    results = {}
    X_t = torch.tensor(vX, device=args.device)

    def eval_seq(net):
        net.eval()
        with torch.no_grad():
            out = net(X_t)
            pred = out[:, -args.horizon:, 0].cpu().numpy()
        return float((pred - vY).pow(2).mean())

    class MLP(tnn.Module):
        def __init__(self):
            super().__init__()
            self.net = tnn.Sequential(
                tnn.Flatten(), tnn.Linear(X.shape[1] * X.shape[2], 64),
                tnn.ReLU(), tnn.Linear(64, args.horizon))

        def forward(self, x):
            return self.net(x).unsqueeze(1)

    class RNN(tnn.Module):
        def __init__(self, cls):
            super().__init__()
            self.rnn = cls(input_size=X.shape[2], hidden_size=64,
                           batch_first=True)
            self.head = tnn.Linear(64, args.horizon)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.head(o)

    mlp = _baseline_trainer(MLP().to(args.device), X, Y, args, seq_model=False)
    results["mlp"] = float(
        ((mlp(X_t)[:, 0].cpu().numpy() - vY) ** 2).mean())
    for name, cls in [("rnn", tnn.RNN), ("lstm", tnn.LSTM), ("gru", tnn.GRU)]:
        net = _baseline_trainer(RNN(cls).to(args.device), X, Y, args)
        results[name] = eval_seq(net)
    return results


# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dynamics", choices=["rate", "lif", "surrogate_lif"],
                   default="rate")
    p.add_argument("--trainable-edges", action="store_true",
                   help="BPTT through the connectome (surrogate path)")
    p.add_argument("--window", type=int, default=1,
                   help="encoder delay-line length (1 = memoryless)")
    p.add_argument("--hidden", type=int, default=256,
                   help="encoder output / readout source size")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-baselines", action="store_true")
    args = p.parse_args(argv)

    import axonweave

    brain = axonweave.load("male-cns:v1.0")
    X, Y, vX, vY = make_dataset(seed=args.seed)
    print(f"substrate: {brain.n_neurons:,} neurons; "
          f"data: {X.shape} -> horizon {args.horizon}")

    results = {"config": vars(args), "n_neurons": brain.n_neurons}

    t0 = time.time()
    model = build_axonweave_model(brain, args, X.shape[2])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    losses = train_axonweave(model, X, Y, args)
    results["axonweave"] = {
        "dynamics": args.dynamics,
        "trainable_edges": args.trainable_edges,
        "trainable_params": n_params,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "val_mse": eval_axonweave(model, vX, vY, args),
        "seconds": round(time.time() - t0, 2),
    }

    if not args.skip_baselines:
        import torch
        results["baselines"] = run_baselines(X, Y, vX, vY, args)
        _ = torch

    print(f"\n{'model':<24} {'val MSE':>12} {'params':>10}")
    ax = results["axonweave"]
    label = f"axonweave ({ax['dynamics']}{' +bptt' if ax['trainable_edges'] else ''})"
    print(f"{label:<24} {ax['val_mse']:>12.5f} {ax['trainable_params']:>10,}")
    for name, mse in results.get("baselines", {}).items():
        print(f"{name:<24} {mse:>12.5f} {'-':>10}")

    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nmetrics written to {out_dir / 'metrics.json'}")
    print("\nCaveat: a single synthetic task; results establish that the "
          "connectome trains and forecasts, not superiority over baselines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
