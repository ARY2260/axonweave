from __future__ import annotations

import json
from pathlib import Path

from ..core.graph import ConnectomeGraph
from ..core.brain import BiologicalBrain
from ..errors import DatasetIntegrityError, ApiUsageError

AWB_FORMAT_VERSION = 1

OPTIONAL_FILES = [
    "annotations.feather",
    "neurotransmitters.feather",
    "receptors.json",
    "annotations.json",
]


def pack_substrate(brain: BiologicalBrain, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph_tmp = output_path.parent / "_pack_graph.npz"
    brain.graph.save(graph_tmp)

    manifest = {
        "id": brain.substrate_id,
        "status": "packed",
        "pack_format_version": AWB_FORMAT_VERSION,
        "graph": {
            "n_neurons": brain.graph.n_neurons,
            "n_edges": brain.graph.n_edges,
        },
        "files": {"graph.npz": graph_tmp.stat().st_size},
    }

    import zipfile

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(graph_tmp, "graph.npz")

        for key in ("annotations", "neurotransmitters"):
            path = getattr(brain, key, None)
            if path is not None and Path(path).exists():
                arcname = f"{key}.feather"
                zf.write(path, arcname)
                manifest["files"][arcname] = Path(path).stat().st_size

        path = getattr(brain, "receptors", None)
        if path is not None and Path(path).exists():
            zf.write(path, "receptors.json")
            manifest["files"]["receptors.json"] = Path(path).stat().st_size

        selection_tables = getattr(brain.graph, "selection_tables", None)
        if selection_tables:
            ann_bytes = json.dumps(selection_tables, indent=1).encode("utf-8")
            zf.writestr("annotations.json", ann_bytes)
            manifest["files"]["annotations.json"] = len(ann_bytes)

        zf.writestr("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))

    graph_tmp.unlink(missing_ok=True)
    return output_path


def unpack_substrate(
    awb_path: str | Path,
    target_dir: str | Path | None = None,
) -> BiologicalBrain:
    awb_path = Path(awb_path)
    if not awb_path.exists():
        raise ApiUsageError(f"AXW010: archive not found: {awb_path}")

    import zipfile

    if target_dir is None:
        target_dir = awb_path.parent / awb_path.stem
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(awb_path, "r") as zf:
        zf.extractall(target_dir)

    manifest_path = target_dir / "manifest.json"
    if not manifest_path.exists():
        raise DatasetIntegrityError(
            "AXW002: .awb archive missing manifest.json"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = manifest.get("status")
    if not isinstance(manifest.get("id"), str):
        raise DatasetIntegrityError("AXW002: manifest missing substrate id")
    if status not in ("packed", "installed"):
        raise DatasetIntegrityError(
            f"AXW002: manifest status is {status!r}; expected 'packed' or 'installed'"
        )

    graph_path = target_dir / "graph.npz"
    if not graph_path.exists():
        raise DatasetIntegrityError("AXW002: .awb archive missing graph.npz")

    def optional(filename):
        p = target_dir / filename
        return p if p.exists() else None

    graph = ConnectomeGraph.load(graph_path)

    selection_tables = None
    annotations_json = target_dir / "annotations.json"
    if annotations_json.exists():
        try:
            selection_tables = json.loads(annotations_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            selection_tables = None

    brain = BiologicalBrain(
        graph,
        annotations=optional("annotations.feather"),
        neurotransmitters=optional("neurotransmitters.feather"),
        receptors=optional("receptors.json"),
    )
    brain.graph.selection_tables = selection_tables
    return brain
