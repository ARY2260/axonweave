from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin
import requests

from .builder import build_graph
from .checksums import expected_checksum, md5_base64_of_file, verify_checksum
from .manifest import MALE_CNS
from .registry import SubstrateRegistry

DEFAULT_TIMEOUT = (20, 120)

def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def _download(url: str, destination: Path, session=None) -> tuple[Path, str | None]:
    """Download (resuming when possible) and return the file path plus the
    server-reported base64 MD5 content hash when the origin exposes one
    (e.g. Google Cloud Storage ``x-goog-hash: md5=...``)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    session = session or requests.Session()
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    server_md5: str | None = None
    with session.get(url, headers=headers, stream=True, timeout=DEFAULT_TIMEOUT) as response:
        if offset and response.status_code == 200:
            offset = 0
            partial.unlink(missing_ok=True)
        response.raise_for_status()
        for value in response.headers.get("x-goog-hash", "").split(","):
            match = re.search(r"md5=([\w+/=]+)", value.strip())
            if match:
                server_md5 = match.group(1)
        mode = "ab" if offset else "wb"
        with partial.open(mode) as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    partial.replace(destination)
    return destination, server_md5

def install_male_cns(root=None, include_synapses=False, include_stats=False):
    registry = SubstrateRegistry(root)
    target = registry.path(MALE_CNS["id"])
    raw = target / "source"
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    required = ["connectivity", "annotations", "neurotransmitters"]
    if include_stats:
        required.append("stats")
    if include_synapses:
        required += ["syn_points", "syn_partners", "tbar_neurotransmitters"]

    observed = {}
    for key in required:
        filename = MALE_CNS["files"][key]
        path, server_md5 = _download(
            urljoin(MALE_CNS["base_url"].rstrip("/") + "/", filename), raw / filename, session
        )
        digest = _sha256(path)
        expected = expected_checksum(MALE_CNS["id"], key)
        # Prefer the server-reported hash (no extra read pass); otherwise
        # compute the MD5 locally from the downloaded file.
        observed_md5 = server_md5 or (md5_base64_of_file(path) if expected.get("md5_base64") else None)
        verify_checksum(digest, expected, key, observed_md5_base64=observed_md5)
        observed[key] = {
            "filename": filename,
            "size": path.stat().st_size,
            "sha256": digest,
            "md5_base64": observed_md5,
        }

    graph_path = target / "graph.npz"
    graph = build_graph(raw / MALE_CNS["files"]["connectivity"], graph_path)

    import shutil
    for key in ("annotations", "neurotransmitters"):
        shutil.copy2(raw / MALE_CNS["files"][key], target / f"{key}.feather")
    if include_stats:
        shutil.copy2(raw / MALE_CNS["files"]["stats"], target / "stats.feather")

    meta = {
        "id": MALE_CNS["id"],
        "version": MALE_CNS["version"],
        "license": MALE_CNS["license"],
        "source": MALE_CNS["source"],
        "base_url": MALE_CNS["base_url"],
        "files": observed,
        "graph": {"n_neurons": graph.n_neurons, "n_edges": graph.n_edges},
        "status": "installed",
    }
    (target / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return registry.load(MALE_CNS["id"])
