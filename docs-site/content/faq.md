# FAQ

Direct answers to the questions newcomers actually ask, each verified against the real implementation rather than aspiration.

## Is the MaleCNS connectome the entire model?

No. It is the **structural substrate**: wiring, neuron identities and neurotransmitter predictions. Dynamics, signaling rules, plasticity and interfaces are separate configurable models.

## Is AxonWeave a brain simulator?

No. AxonWeave provides a computational interface *to* connectome data. It explicitly does not claim that a connectome alone is a complete biophysical simulation. See [Scientific Reference](scientific-reference.md).

## Can I train the connectome?

Yes, several ways: trainable synaptic weights on existing edges (topology preserved), local plasticity (STDP, dopamine-modulated), or leave it frozen and train only interfaces. See [Training](training.md).

## Can I freeze the connectome?

Yes — that is the default (`trainable_edges=False`). Only encoders/readouts train.

## Can I change synaptic weights?

Existing edges can become trainable parameters or update via plasticity. Non-existing edges stay zero; structural plasticity (new/removes edges) is not supported.

## Can I use PyTorch?

Yes — `pip install "axonweave[torch]"`. `ConnectomeLayer` is a native `torch.nn.Module` and composes with arbitrary PyTorch layers.

## Can I use TensorFlow?

Yes — `pip install "axonweave[tensorflow]"`. `ConnectomeLayer` is a native `tf.keras.layers.Layer`.

## Can I use JAX?

The JAX adapter is planned. NumPy, PyTorch and TensorFlow are available today.

## Can I run on GPU?

Device execution is delegated to the host framework, and sparse-op support on accelerators varies. AxonWeave fails explicitly (`AXW004`) rather than silently falling back to CPU. See [Device Support](devices.md).

## Can I use my own layer?

Yes. AxonWeave blocks are ordinary framework modules — compose them with attention, convolution, custom research modules, anything the host framework supports.

## Can I use text?

Yes, as an interface experiment: `TokenEncoder` → brain → `TokenDecoder` → next-token logits. This is a computational task, not a claim about fly cognition. See [Tasks](tasks.md).

## Can I use games?

The agent/environment loop exists (`brain.agent(...)`, `agent.run(environment)`). Game-specific adapters (Doom, Gymnasium environments) are planned.

## Do I need to manually download MaleCNS?

No. `axonweave substrate install male-cns:v1.0` handles download, checksum verification, graph construction and caching. You never touch Feather files.

## Are neurotransmitter effects hard-coded?

No. Neurotransmitter predictions are stored as data. Sign/strength requires an explicit `SignalPolicy` you configure. See [Signals & Receptors](signals.md).

## Is the model biologically exact?

No. The connectome is real data; everything computed on top (dynamics, delays, plasticity) is an explicit modeling choice. AxonWeave keeps these categories separate and visible.

## Can I use another connectome later?

The substrate registry is versioned by design. Additional connectomes are a roadmap item; the manifest/fingerprint system already supports multiple substrates.
