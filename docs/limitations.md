# Scientific Limitations


:::DOC-WARN
This page is a core scientific requirement, not a disclaimer. Every limitation here should be treated as a real constraint on experimental conclusions.
:::

## The central limitation

A connectome is a wiring diagram. It specifies which neurons can influence which neurons, and with roughly what strength (synapse counts). It does **not** specify how that wiring computes. Everything dynamic — voltages, spikes, receptor effects, learning — is a model layered on top of the structure.

AxonWeave exists to make that layer explicit and configurable. It does not remove the limitation.

## Incomplete biological information

The MaleCNS v1.0 release provides:

- neuron identities (body IDs)
- directed connectivity with synapse-count-derived weights
- aggregate neurotransmitter predictions per neuron

It does not provide:

- receptor composition per synapse
- membrane properties per neuron
- synaptic release probabilities
- detailed neuromodulatory state
- complete cell-type annotations for every neuron

Where AxonWeave needs such values (dynamics parameters, sign policies), it requires you to supply them explicitly rather than guessing.

## Simplified dynamics

The bundled dynamics models are deliberately simple:

- **Rate** — instantaneous input-output; no memory, no spikes.
- **LIF** — single compartment, fixed parameters, absolute refractory period only.
- **Adaptive LIF** — adds threshold adaptation; still single compartment.

None of these reproduce Drosophila biophysics: no ion-channel diversity, no dendritic computation, no compartmental morphology, no stochastic vesicle release. They are computational reference models applied to biological structure.

## Uncertain receptor relationships

The dataset records what transmitter a neuron likely releases. It does not record what receptors the postsynaptic side expresses. Converting "releases GABA" into "inhibitory" is an assumption that holds for typical receptor compositions but is not verified per synapse. AxonWeave therefore refuses to hard-code polarity; see [Signals & Receptors](signals.md).

## Neurotransmitter ambiguity

Predictions are aggregate and probabilistic. A neuron labeled "acetylcholine" may co-release other transmitters. The substrate preserves the prediction with its provenance; it does not resolve ambiguity.

## Missing mechanisms

Not modeled at all currently:

- synaptic delays (planned interface)
- short-term plasticity (facilitation/depression)
- neuromodulation (dopamine, serotonin as global state)
- glia, gap junctions
- development and structural plasticity
- energy/metabolic constraints

## Learning-rule limitations

STDP and dopamine-modulated STDP are computational rules *inspired by* biology:

- pairwise trace-based STDP is a simplification of spike-timing effects
- the dopamine signal in `DopamineSTDP` is whatever your environment provides, not a measured neuromodulatory event
- no claim is made that Drosophila uses these rules at these synapses

Training modes 1–5 (see [Training](training.md)) are optimization strategies applied to a biological structure, not models of fly learning.

## Computational discretization

Simulation is time-discrete (`dt = 1 ms` default). Continuous biology is approximated; results depend on step size. Sparse propagation is exact with respect to the loaded weights, but weights themselves derive from a thresholded upstream release (`minconf-0.5`), so low-confidence connections are absent by construction.

## Substrate and version dependence

Results are only reproducible against the same substrate version. MaleCNS v1.0 is a fixed release; future connectome versions will have different fingerprints. Checkpoints must record substrate identity, and AxonWeave validates manifest identity at load time.

## What this means for your conclusions

- "The substrate + my dynamics produced behavior X" — a legitimate computational result.
- "The fly brain computes X" — not supported by this pipeline alone.
- Treat learned parameters as task adaptations, not biological findings, unless independently validated against experiments.

## Related

- [Scientific Reference](scientific-reference.md) — data provenance and what the source provides.
- [Signals & Receptors](signals.md) — the polarity assumption problem.
- [Training](training.md) — what each learning mode actually optimizes.
- [FAQ](faq.md) — direct answers to common overclaims.
