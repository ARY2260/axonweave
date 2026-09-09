"""Framework adapters exposing the high-level connectome computing API.

Available adapters (import lazily via the framework package so optional
backends stay optional):

- ``axonweave.frameworks.torch``  — BrainModel, ConnectomeBlock (torch)
- ``axonweave.frameworks.keras``  — BrainLayer, KerasConnectomeBlock (tf.keras)
"""
