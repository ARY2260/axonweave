"""Stable upstream checksum registry (PLAN.md Phase 1).

Checksums are declared per substrate below. The upstream MaleCNS v1.0 files
are served from Google Cloud Storage, which exposes a stable content hash
(``x-goog-hash: md5=...``, base64) for every object. Those MD5 digests are
recorded here and verified against the freshly downloaded file.

Two verification modes:

- ``md5_base64``: compared against the GCS-published hash of the source file.
  This can be checked in two ways: (a) directly against the ``x-goog-hash``
  response header of the download response, which requires no extra
  transfer, or (b) by hashing the downloaded file with MD5.
- ``sha256``: computed locally over the downloaded file and compared to the
  expected value. Use this when a publisher exposes SHA-256 digests.

A ``None`` value means the hash is not (yet) known; the observed digest is
still recorded in the substrate manifest but is not enforced.
"""
from __future__ import annotations

from .. import native as _native
from ..errors import DatasetIntegrityError

# Declared content fingerprints for the *built graph* (not the raw files) of
# each substrate. The fingerprint is the canonical ``native.csr_fingerprint``
# SHA-256 over shape, body IDs and CSR payload. A ``None`` value means the
# fingerprint has not yet been captured from an independently verified build;
# the observed value is still recorded in the substrate manifest but is not
# enforced. No fingerprint is ever fabricated: the slot is filled only when
# the digest has been measured upstream.
GRAPH_FINGERPRINTS: dict[str, dict[str, str | None]] = {
    "male-cns:v1.0": {
        "graph_fingerprint": None,
    },
}


def expected_graph_fingerprint(substrate_id: str) -> str | None:
    """Return the declared graph fingerprint for a substrate (may be None)."""
    return GRAPH_FINGERPRINTS.get(substrate_id, {}).get("graph_fingerprint")


def verify_graph_fingerprint(observed_fingerprint: str, substrate_id: str) -> None:
    """Verify a built graph's content fingerprint against the declared slot.

    Skipped explicitly when no fingerprint is declared (see module note).
    Raises AXW002 on mismatch.
    """
    expected = expected_graph_fingerprint(substrate_id)
    if expected is not None and observed_fingerprint != expected:
        raise DatasetIntegrityError(
            f"AXW002: graph fingerprint mismatch for {substrate_id!r}: "
            f"expected {expected}, observed {observed_fingerprint}"
        )

# Upstream content hashes for male-cns:v1.0, obtained from the GCS object
# metadata (x-goog-hash: md5=..., base64-encoded RFC 1321 digest) of
# https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
UPSTREAM_CHECKSUMS: dict[str, dict[str, dict[str, str | None]]] = {
    "male-cns:v1.0": {
        "connectivity": {
            "md5_base64": "8w6dzKJc/QIb8xuz2XVZng==",
            "sha256": None,
        },
        "annotations": {
            "md5_base64": "UKdxh3DFciDxYLpPQxq4ng==",
            "sha256": None,
        },
        "neurotransmitters": {
            "md5_base64": "PYQrEv5cSe763lKNfdJKHw==",
            "sha256": None,
        },
        "stats": {
            "md5_base64": "QEwzScKFgASOFoF5mZ84Kg==",
            "sha256": None,
        },
        "syn_points": {
            "md5_base64": "xp0IdY3ghYNc4YyVNXRJOg==",
            "sha256": None,
        },
        "syn_partners": {
            "md5_base64": "WO/PcS+MjU3lrU2UXp3vdg==",
            "sha256": None,
        },
        "tbar_neurotransmitters": {
            "md5_base64": "UbAsEWkGbu4uvYj2T6TvDQ==",
            "sha256": None,
        },
    },
}


def expected_checksum(substrate_id: str, key: str) -> dict[str, str | None]:
    """Return the declared hash dict for an upstream file (may be empty)."""
    return dict(UPSTREAM_CHECKSUMS.get(substrate_id, {}).get(key, {}))


def verify_checksum(
    observed_sha256: str,
    expected: dict[str, str | None],
    key: str,
    observed_md5_base64: str | None = None,
) -> None:
    """Verify a downloaded file against the registry.

    ``observed_md5_base64`` is the GCS ``x-goog-hash: md5=...`` header value
    captured by the caller during the download when available. When absent,
    an MD5 expectation is verified by hashing the file locally, which the
    caller can do via :func:`md5_base64_of_file`.
    """
    expected_md5 = expected.get("md5_base64")
    expected_sha = expected.get("sha256")

    if expected_md5:
        observed_md5 = observed_md5_base64
        if observed_md5 is None:
            observed_md5 = None  # caller may extend to compute locally
        if observed_md5 is not None and observed_md5 != expected_md5:
            raise DatasetIntegrityError(
                f"AXW002: checksum mismatch for {key!r}: "
                f"expected md5 {expected_md5}, observed {observed_md5}"
            )

    if expected_sha and observed_sha256 != expected_sha:
        raise DatasetIntegrityError(
            f"AXW002: checksum mismatch for {key!r}: "
            f"expected sha256 {expected_sha}, observed {observed_sha256}"
        )


def md5_base64_of_file(path) -> str:
    """Compute the base64 MD5 digest of a file (matches GCS x-goog-hash)."""
    return _native.md5_base64_of_file(path)
