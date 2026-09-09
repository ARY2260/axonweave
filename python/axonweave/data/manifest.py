from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

MALE_CNS = {
    "id": "male-cns:v1.0",
    "version": "v1.0",
    "license": "CC-BY",
    "source": "https://male-cns.janelia.org/download/",
    "base_url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome",
    "files": {
        "connectivity": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
        "stats": "body-stats-male-cns-v1.0-minconf-0.5.feather",
        "syn_points": "syn-points-male-cns-v1.0-minconf-0.5.feather",
        "syn_partners": "syn-partners-male-cns-v1.0-minconf-0.5.feather",
        "tbar_neurotransmitters": "tbar-neurotransmitters-male-cns-v1.0.feather",
    },
}

@dataclass(frozen=True)
class SubstrateManifest:
    identifier: str
    version: str
    source: str
    files: Mapping[str, str]
    license: str = ""

    @classmethod
    def male_cns(cls) -> "SubstrateManifest":
        return cls(MALE_CNS["id"], MALE_CNS["version"], MALE_CNS["source"], MALE_CNS["files"], MALE_CNS["license"])
