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
        # If the manifest recorded the built graph's content fingerprint,
        # re-derive it from the stored artifact and refuse a mismatched graph
        # (AXW002) — catches corrupted or swapped graph.npz files.
        graph_meta = meta.get("graph") or {}
        recorded_fp = graph_meta.get("fingerprint")
        if recorded_fp:
            from .checksums import verify_graph_fingerprint
            from ..core.brain import substrate_fingerprint
            observed_fp = substrate_fingerprint(brain.graph)
            # Compare against the manifest's own recorded build fingerprint...
            if observed_fp != recorded_fp:
                raise DatasetIntegrityError(
                    f"AXW002: stored graph fingerprint does not match the graph "
                    f"artifact for {name!r}: manifest={recorded_fp}, observed={observed_fp}. "
                    f"Reinstall the substrate."
                )
            # ...and against the declared upstream slot when one exists.
            verify_graph_fingerprint(observed_fp, name)
        # Selection tables live on the graph so graph.neurons can resolve
        # by_type()/by_region() without re-reading the Feather file.
        brain.graph.selection_tables = selection_tables
        return brain
