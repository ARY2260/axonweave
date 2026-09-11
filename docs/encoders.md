# Encoders & Decoders

Encoders map external data modalities into neural stimulation patterns. Decoders map neural activity back to predictions or actions. Both are ordinary framework modules — fully composable with user code.

## Encoders

### ImageEncoder

Maps 2D/3D visual input to currents injected into a selected neuron group:

```python
from axonweave.encoders import ImageEncoder

enc = ImageEncoder(
    shape=(84, 84, 3),      # observation shape
    n_target=2048,          # number of stimulated neurons
    normalize=True,         # scale pixel intensities to [0, 1]
)
currents = enc(obs)         # (batch, n_target) tensor
```

### TokenEncoder

Maps discrete token IDs (or text embeddings) to input neuron currents:

```python
from axonweave.encoders import TokenEncoder

enc = TokenEncoder(vocab_size=50_000, n_target=1024)
currents = enc(token_ids)   # (batch, seq, n_target)
```

### SensorEncoder

Generic vector-to-current mapping for robotics/telemetry:

```python
from axonweave.encoders import SensorEncoder

enc = SensorEncoder(n_sensors=12, n_target=512)
currents = enc(sensor_vector)
```

## Decoders

### ActionDecoder

Maps readout neuron activity to discrete or continuous actions:

```python
from axonweave.decoders import ActionDecoder

dec = ActionDecoder(actions=6)   # discrete action space
action = dec(readout_activity)
```

### TokenDecoder / ClassificationHead

Linear projection from neural readout to vocabulary logits:

```python
from axonweave.decoders import TokenDecoder

dec = TokenDecoder(vocab_size=50_000)
logits = dec(selected_activity)     # (batch, seq, vocab_size)
```

## Choosing neuron groups

Encoders and decoders accept explicit neuron selection. You can select by body ID, by ROI, or by index:

```python
enc = ImageEncoder(shape=(84, 84, 3), n_target=2048, select="visual")
dec = ActionDecoder(actions=6, select="motor")
```

:::DOC-WARN
Selection semantics ("visual", "motor") require neuron annotations from the substrate. If annotations are not installed, AxonWeave raises `AXW001`/`AXW002` rather than silently using arbitrary neurons.
:::

## Design rules

- Encoders/decoders are framework modules (PyTorch `nn.Module` / Keras `Layer`), so gradients flow through them normally in Modes 1, 2, 3 and 5.
- They never modify the substrate graph.
- Custom encoders only need to produce correctly shaped tensors.
