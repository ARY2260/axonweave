from __future__ import annotations

from pathlib import Path
import json
import os
from ..errors import SubstrateNotInstalledError, DatasetIntegrityError
from ..core.graph import ConnectomeGraph
from ..core.brain import BiologicalBrain

class SubstrateRegistry:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("AXONWEAVE_HOME", Path.home() / ".cache" / "axonweave"))

    def path(self, name):
        return self.root / "substrates" / name.replace(":", "-")

    def load(self, name):
        p = self.path(name)
        graph = p / "graph.npz"
        manifest = p / "manifest.json"
        if not graph.exists() or not manifest.exists():
            raise SubstrateNotInstalledError(
                f"AXW001: substrate {name!r} is not installed. Run `axonweave substrate install {name}`."
            )
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if meta.get("id") != name or meta.get("status") != "installed":
            raise DatasetIntegrityError("AXW002: substrate manifest identity/status mismatch")
        def optional(filename):
            return p / filename if (p / filename).exists() else None
        annotations_json = p / "annotations.json"
        selection_tables = None
        if annotations_json.exists():
            try:
                selection_tables = json.loads(annotations_json.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                selection_tables = None
        brain = BiologicalBrain(
            ConnectomeGraph.load(graph),
            annotations=optional("annotations.feather"),
            neurotransmitters=optional("neurotransmitters.feather"),
            receptors=optional("receptors.json"),
        )
        # Selection tables live on the graph so graph.neurons can resolve
        # by_type()/by_region() without re-reading the Feather file.
        brain.graph.selection_tables = selection_tables
        return brain
