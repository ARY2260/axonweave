from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class NeuronMetadata:
    body_id: int
    cell_type: str | None = None
    region: str | None = None
    hemisphere: str | None = None

    def __repr__(self) -> str:
        parts = [f"body_id={self.body_id}"]
        if self.cell_type is not None:
            parts.append(f"cell_type={self.cell_type!r}")
        if self.region is not None:
            parts.append(f"region={self.region!r}")
        if self.hemisphere is not None:
            parts.append(f"hemisphere={self.hemisphere!r}")
        return f"NeuronMetadata({', '.join(parts)})"


class NeuronMetadataStore:
    def __init__(self, annotations_path: Path | None = None):
        self._annotations_path = annotations_path
        self._by_body: dict[int, NeuronMetadata] | None = None
        self._types: list[str] | None = None
        self._regions: list[str] | None = None
        self._hemispheres: list[str] | None = None
        self._has_data = False

    def _ensure_loaded(self) -> None:
        if self._by_body is not None:
            return
        self._by_body = {}
        self._types = []
        self._regions = []
        self._hemispheres = []
        if self._annotations_path is None or not self._annotations_path.exists():
            return
        raw = self._load_file(self._annotations_path)
        if raw is None:
            return
        if not isinstance(raw, dict):
            return
        self._build_from_raw(raw)
        self._has_data = bool(self._by_body)

    def _load_file(self, path: Path) -> dict | None:
        suffix = path.suffix.lower()
        if suffix == ".feather":
            return self._load_feather(path)
        if suffix == ".json":
            return self._load_json(path)
        try:
            return self._load_feather(path)
        except Exception:
            pass
        try:
            return self._load_json(path)
        except Exception:
            pass
        return None

    @staticmethod
    def _load_feather(path: Path) -> dict | None:
        try:
            import pyarrow.feather as feather
        except ImportError:
            return None
        try:
            table = feather.read_table(path)
        except Exception:
            return None
        names = [n.lower() for n in table.column_names]
        body_idx = None
        type_idx = None
        region_idx = None
        hemisphere_idx = None
        for i, n in enumerate(names):
            if body_idx is None and n in ("body_id", "bodyid", "body", "id"):
                body_idx = i
            if type_idx is None and n in ("cell_type", "type", "neuron_type", "class"):
                type_idx = i
            if region_idx is None and n in ("region", "super_class", "roi"):
                region_idx = i
            if hemisphere_idx is None and n in ("hemisphere", "side"):
                hemisphere_idx = i
        if body_idx is None:
            return None
        result: dict[str, dict[str, str | None]] = {}
        types_set: set[str] = set()
        regions_set: set[str] = set()
        hemispheres_set: set[str] = set()
        for row_idx in range(table.num_rows):
            body_val = table.column(body_idx)[row_idx].as_py()
            if body_val is None:
                continue
            body_id = int(body_val)
            entry: dict[str, str | None] = {}
            if type_idx is not None:
                v = table.column(type_idx)[row_idx].as_py()
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    s = str(v)
                    entry["cell_type"] = s
                    types_set.add(s)
                else:
                    entry["cell_type"] = None
            if region_idx is not None:
                v = table.column(region_idx)[row_idx].as_py()
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    s = str(v)
                    entry["region"] = s
                    regions_set.add(s)
                else:
                    entry["region"] = None
            if hemisphere_idx is not None:
                v = table.column(hemisphere_idx)[row_idx].as_py()
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    s = str(v)
                    entry["hemisphere"] = s
                    hemispheres_set.add(s)
                else:
                    entry["hemisphere"] = None
            result[str(body_id)] = entry
        raw: dict[str, dict[str, str | None]] = {"_meta": {"source": "feather"}}
        for body_str, fields in result.items():
            raw[body_str] = fields
        return raw

    @staticmethod
    def _load_json(path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        if "_meta" in data:
            return data
        if "body_id" in data or "bodyId" in data or "id" in data:
            return data
        keys = list(data.keys())
        if keys and keys[0].isdigit():
            return data
        return None

    def _build_from_raw(self, raw: dict) -> None:
        types_set: set[str] = set()
        regions_set: set[str] = set()
        hemispheres_set: set[str] = set()
        for key, val in raw.items():
            if key.startswith("_"):
                continue
            if not isinstance(val, dict):
                continue
            body_id = int(key)
            ct = val.get("cell_type") or val.get("type")
            rg = val.get("region") or val.get("side") or val.get("super_class")
            hm = val.get("hemisphere") or val.get("side")
            if rg and hm and rg == hm and hm in ("left", "right", "L", "R"):
                hm = rg
                rg = None
            self._by_body[body_id] = NeuronMetadata(
                body_id=body_id,
                cell_type=str(ct) if ct is not None else None,
                region=str(rg) if rg is not None else None,
                hemisphere=str(hm) if hm is not None else None,
            )
            if ct is not None:
                types_set.add(str(ct))
            if rg is not None:
                regions_set.add(str(rg))
            if hm is not None:
                hemispheres_set.add(str(hm))
        self._types = sorted(types_set)
        self._regions = sorted(regions_set)
        self._hemispheres = sorted(hemispheres_set)

    def get(self, body_id: int) -> NeuronMetadata | None:
        self._ensure_loaded()
        if self._by_body is None:
            return None
        return self._by_body.get(int(body_id))

    def get_batch(self, body_ids: np.ndarray) -> list[NeuronMetadata]:
        self._ensure_loaded()
        if self._by_body is None:
            return []
        result = []
        for bid in np.asarray(body_ids, dtype=np.int64).tolist():
            meta = self._by_body.get(int(bid))
            if meta is not None:
                result.append(meta)
        return result

    def query(
        self,
        cell_type: str | None = None,
        region: str | None = None,
        hemisphere: str | None = None,
    ) -> np.ndarray:
        self._ensure_loaded()
        if self._by_body is None or not self._by_body:
            return np.empty(0, dtype=np.int64)
        matches = []
        for body_id, meta in self._by_body.items():
            if cell_type is not None and meta.cell_type != cell_type:
                continue
            if region is not None and meta.region != region:
                continue
            if hemisphere is not None and meta.hemisphere != hemisphere:
                continue
            matches.append(body_id)
        return np.asarray(sorted(matches), dtype=np.int64)

    def types(self) -> list[str]:
        self._ensure_loaded()
        return list(self._types or [])

    def regions(self) -> list[str]:
        self._ensure_loaded()
        return list(self._regions or [])

    def hemispheres(self) -> list[str]:
        self._ensure_loaded()
        return list(self._hemispheres or [])

    @property
    def has_types(self) -> bool:
        self._ensure_loaded()
        return bool(self._types)

    @property
    def has_regions(self) -> bool:
        self._ensure_loaded()
        return bool(self._regions)

    @property
    def has_hemispheres(self) -> bool:
        self._ensure_loaded()
        return bool(self._hemispheres)

    def summary(self) -> str:
        self._ensure_loaded()
        if self._by_body is None or not self._has_data:
            return "NeuronMetadataStore: no annotations loaded"
        n = len(self._by_body)
        lines = [
            f"NeuronMetadataStore: {n:,} neurons",
            f"  cell types:      {len(self._types)}" if self._types else "  cell types:      (none)",
            f"  regions:         {len(self._regions)}" if self._regions else "  regions:         (none)",
            f"  hemispheres:     {len(self._hemispheres)}" if self._hemispheres else "  hemispheres:     (none)",
        ]
        if self._types:
            preview = ", ".join(self._types[:8])
            if len(self._types) > 8:
                preview += ", ..."
            lines.append(f"  type preview:    {preview}")
        if self._regions:
            preview = ", ".join(self._regions[:8])
            if len(self._regions) > 8:
                preview += ", ..."
            lines.append(f"  region preview:  {preview}")
        return "\n".join(lines)
