"""Unified multi-dataset loader.

This module composes the per-dataset ``DataSource`` classes from
``data_manager_isaac.py``, ``data_manager_faro.py`` and
``data_manager_pickle.py`` into a single object that exposes the same
``init_directory`` / ``get_item`` interface (mirroring
``data_manager_isaac.DataSource``) with a *flat* global index.

The aim is to let downstream training / benchmarking scripts iterate over
samples drawn from several heterogeneous datasets without caring about the
underlying storage format.

Common output schema (returned by :meth:`DataSource.get_item`)::

    {
        "index"         : int,                   # flat global index
        "source"        : "isaac"|"faro"|"pickle",
        "source_index"  : int,                   # index inside the sub-source
        "ir_left_img"   : np.ndarray | None,     # 2D, grayscale (uint8/uint16)
        "ir_right_img"  : np.ndarray | None,
        "rgb_img"       : np.ndarray | None,     # 3-channel BGR/uint8 if any
        "depth_gt_img"  : np.ndarray | None,     # GT depth, mm (float/uint16)
        "depth_rs_img"  : np.ndarray | None,     # sensor depth, mm
        "bf"            : float | None,          # baseline (mm) * fx (px)
        "intrinsics"    : dict | None,           # {fx, fy, cx, cy, [w, h]}
        "meta"          : dict,                  # raw item dict from sub-source
    }

Per-source mapping
------------------
* **isaac** (:mod:`data_manager_isaac`)
    - ``ir_left_img``  <- ``ir_left_img``
    - ``ir_right_img`` <- ``ir_right_img``
    - ``rgb_img``      <- ``rgb_left_img``
    - ``depth_gt_img`` <- ``depth_gt_img``
    - ``depth_rs_img`` <- ``depth_rs_img``
    - ``bf``           <- ``bf``
    - ``intrinsics``   <- ``intrinsics``
* **faro** (:mod:`data_manager_faro`)
    - ``ir_left_img``  <- ``left``
    - ``ir_right_img`` <- ``right``
    - ``rgb_img``      <- ``rgb``
    - ``depth_gt_img`` <- ``depth_faro``
    - ``depth_rs_img`` <- ``depth_rs``
    - ``bf``           = None (not provided by the FARO source)
    - ``intrinsics``   = None
* **pickle** (:mod:`data_manager_pickle`)
    - ``ir_left_img``  <- ``ir_left_img``
    - ``ir_right_img`` <- ``ir_right_img``
    - ``rgb_img``      <- ``rgb_img``
    - ``depth_gt_img`` = None (no per-pixel GT depth; CAD pose is available
                              in ``meta`` under ``t_camera_cad`` etc.)
    - ``depth_rs_img`` <- ``depth_img``
    - ``bf``           <- ``bf``
    - ``intrinsics``   <- ``intrinsics``  (keys ``fx, fy, ppx, ppy`` are
                                           remapped to ``fx, fy, cx, cy``)
"""

from __future__ import annotations

import logging as log
import os
import sys
import unittest
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

# Support both ``python scripts/data_manager_multiset.py`` (no ``scripts``
# package on ``sys.path``) and ``from scripts.data_manager_multiset import ...``.
try:
    from scripts import data_manager_faro, data_manager_isaac, data_manager_pickle
except ModuleNotFoundError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import data_manager_faro  # type: ignore  # noqa: E402
    import data_manager_isaac  # type: ignore  # noqa: E402
    import data_manager_pickle  # type: ignore  # noqa: E402


log.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=log.INFO)


# ---------------------------------------------------------------------------
# Per-source normalizers
# ---------------------------------------------------------------------------

def _normalize_isaac(item: dict[str, Any]) -> dict[str, Any]:
    intr = item.get("intrinsics")
    return {
        "ir_left_img"   : item.get("ir_left_img"),
        "ir_right_img"  : item.get("ir_right_img"),
        "rgb_img"       : item.get("rgb_left_img"),
        "depth_gt_img"  : item.get("depth_gt_img"),
        "depth_rs_img"  : item.get("depth_rs_img"),
        "bf"            : item.get("bf"),
        "intrinsics"    : dict(intr) if isinstance(intr, dict) else None,
    }


def _normalize_faro(item: dict[str, Any]) -> dict[str, Any]:
    def _to_arr(x: Any) -> Optional[np.ndarray]:
        return x if isinstance(x, np.ndarray) and x.size > 0 else None

    return {
        "ir_left_img"   : _to_arr(item.get("left")),
        "ir_right_img"  : _to_arr(item.get("right")),
        "rgb_img"       : _to_arr(item.get("rgb")),
        "depth_gt_img"  : _to_arr(item.get("depth_faro")),
        "depth_rs_img"  : _to_arr(item.get("depth_rs")),
        "bf"            : None,
        "intrinsics"    : None,
    }


def _normalize_pickle(item: dict[str, Any]) -> dict[str, Any]:
    intr_raw = item.get("intrinsics")
    intr: Optional[dict[str, Any]] = None
    if isinstance(intr_raw, dict):
        intr = {
            "fx"    : float(intr_raw.get("fx", 0.0)),
            "fy"    : float(intr_raw.get("fy", 0.0)),
            # Pickle intrinsics use ``ppx``/``ppy`` for principal point.
            "cx"    : float(intr_raw.get("ppx", intr_raw.get("cx", 0.0))),
            "cy"    : float(intr_raw.get("ppy", intr_raw.get("cy", 0.0))),
        }
        if "width" in intr_raw:
            intr["width"] = intr_raw["width"]
        if "height" in intr_raw:
            intr["height"] = intr_raw["height"]
    return {
        "ir_left_img"   : item.get("ir_left_img"),
        "ir_right_img"  : item.get("ir_right_img"),
        "rgb_img"       : item.get("rgb_img"),
        "depth_gt_img"  : None,
        "depth_rs_img"  : item.get("depth_img"),
        "bf"            : item.get("bf"),
        "intrinsics"    : intr,
    }


# Registry of known sub-sources.
# Each entry: name -> (module, normalizer)
_SOURCE_REGISTRY: dict[str, tuple[Any, Callable[[dict[str, Any]], dict[str, Any]]]] = {
    "isaac" : (data_manager_isaac,  _normalize_isaac),
    "faro"  : (data_manager_faro,   _normalize_faro),
    "pickle": (data_manager_pickle, _normalize_pickle),
}


# ---------------------------------------------------------------------------
# DataSource
# ---------------------------------------------------------------------------

class DataSource:
    """Flat-indexed loader composed of multiple per-dataset ``DataSource`` objects.

    Usage
    -----
    >>> ds = DataSource()
    >>> ds.init_directory()                      # add all defaults
    >>> for k in range(len(ds)):
    ...     item = ds.get_item(k)

    Or with explicit configuration::

        ds = DataSource()
        ds.add_source("isaac",  init_kwargs={"root": r"C:\\Work\\Data\\reflective_test"})
        ds.add_source("pickle", init_kwargs={"excel_path": "..."})
        ds.add_source("faro",   init_kwargs={"input_rectified": "..."})
    """

    def __init__(self, train_mode: bool = False) -> None:
        # Each entry: {"name", "ds", "normalize", "count", "offset"}
        self.sources: list[dict[str, Any]] = []
        self.train_mode = train_mode
        log.info("MultiSet DataSource is defined")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.sources[-1]["offset"] + self.sources[-1]["count"] if self.sources else 0

    @property
    def items(self) -> list[dict[str, Any]]:
        """Flat view over the underlying ``items`` lists of all sub-sources.

        Each entry is augmented with ``source`` and ``source_index`` so it can
        be inspected without a separate lookup.
        """
        out: list[dict[str, Any]] = []
        for entry in self.sources:
            name = entry["name"]
            sub_items = getattr(entry["ds"], "items", None)
            if not sub_items:
                # ``data_manager_faro.DataSource`` uses ``imgs`` (list of paths).
                sub_items = [{"path": p} for p in getattr(entry["ds"], "imgs", [])]
            for i, sub in enumerate(sub_items):
                out.append({"source": name, "source_index": i, **sub})
        return out

    # ------------------------------------------------------------------
    # Source registration
    # ------------------------------------------------------------------

    def add_source(
        self,
        name: str,
        ds_instance: Optional[Any] = None,
        init_kwargs: Optional[dict[str, Any]] = None,
    ) -> int:
        """Add one sub-source to the multiset.

        Parameters
        ----------
        name : one of ``"isaac"``, ``"faro"``, ``"pickle"``.
        ds_instance : an already-constructed ``DataSource`` instance from the
            matching ``data_manager_*`` module. If omitted, a new instance is
            built with ``train_mode=self.train_mode`` (where supported).
        init_kwargs : keyword arguments forwarded to ``ds.init_directory(...)``
            when ``ds_instance`` is freshly built or when its index is empty.

        Returns
        -------
        Number of items registered from this source (0 if init failed).
        """
        if name not in _SOURCE_REGISTRY:
            raise ValueError(
                f"Unknown source '{name}'. Known: {list(_SOURCE_REGISTRY)}"
            )
        module, normalizer = _SOURCE_REGISTRY[name]

        if ds_instance is None:
            try:
                ds_instance = module.DataSource(train_mode=self.train_mode)
            except TypeError:
                # data_manager_faro.DataSource takes no ctor args.
                ds_instance = module.DataSource()

        # Initialize on disk if not done yet.
        existing_count = self._sub_count(ds_instance)
        if existing_count == 0:
            kwargs = dict(init_kwargs or {})
            try:
                ds_instance.init_directory(**kwargs)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Failed to init source '{name}': {exc}")

        count = self._sub_count(ds_instance)
        offset = (
            self.sources[-1]["offset"] + self.sources[-1]["count"]
            if self.sources else 0
        )
        self.sources.append({
            "name"      : name,
            "ds"        : ds_instance,
            "normalize" : normalizer,
            "count"     : count,
            "offset"    : offset,
        })
        log.info(f"MultiSet: registered source '{name}' with {count} items "
                 f"(total now {len(self)})")
        return count

    @staticmethod
    def _sub_count(ds_instance: Any) -> int:
        try:
            return int(len(ds_instance))
        except TypeError:
            # data_manager_faro.DataSource doesn't implement __len__.
            return len(getattr(ds_instance, "imgs", []))

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def init_directory(
        self,
        configs: Optional[list[dict[str, Any]]] = None,
    ) -> int:
        """Initialize one or more sub-sources.

        Parameters
        ----------
        configs : list of per-source dicts of the form::

                {"name": "isaac",  "init_kwargs": {...}}
                {"name": "faro",   "init_kwargs": {...}}
                {"name": "pickle", "init_kwargs": {...}}

            If omitted, all three sources are added with their built-in
            defaults (which use hard-coded paths from each
            ``data_manager_*`` module).

        Returns
        -------
        Total number of indexed items across all sub-sources.
        """
        self.sources.clear()
        configs = configs if configs is not None else [
            {"name": "isaac"},
            {"name": "faro"},
            {"name": "pickle"},
        ]
        for cfg in configs:
            self.add_source(
                name=cfg["name"],
                ds_instance=cfg.get("ds_instance"),
                init_kwargs=cfg.get("init_kwargs"),
            )
        log.info(f"MultiSet DataSource: total {len(self)} items "
                 f"across {len(self.sources)} sources")
        return len(self)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _resolve(self, index: int) -> tuple[dict[str, Any], int]:
        """Map a flat index to ``(source_entry, local_index)``."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"MultiSet index out of range: {index}")
        # Linear scan is fine: number of sources is tiny.
        for entry in self.sources:
            local = index - entry["offset"]
            if 0 <= local < entry["count"]:
                return entry, local
        raise IndexError(f"Could not resolve MultiSet index {index}")

    def source_of(self, index: int) -> str:
        """Return the source name for a flat index."""
        entry, _ = self._resolve(index)
        return entry["name"]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def get_item(self, index: int, debug: bool = False) -> dict[str, Any]:
        """Load one sample by global flat index and return the common schema."""
        entry, local = self._resolve(index)
        sub_item     = entry["ds"].get_item(local, debug=False)
        normalized = entry["normalize"](sub_item)
        out: dict[str, Any] = {
            "index"         : index,
            "source"        : entry["name"],
            "source_index"  : local,
            **normalized,
            "meta"          : sub_item,
        }
        if debug:
            self.show_item(out)
        return out

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def show_item(self, item: dict[str, Any]) -> None:
        """Display the common fields for a normalized item."""
        img_list = [
            item.get("ir_left_img"),
            item.get("ir_right_img"),
            item.get("rgb_img"),
            item.get("depth_rs_img"),
            item.get("depth_gt_img"),
        ]
        ttl_list = [
            "IR left", "IR right", "RGB", "Depth RS [mm]", "Depth GT [mm]"
        ]
        # Filter to images that actually exist so empty axes are not drawn.
        pairs = [(im, tt) for im, tt in zip(img_list, ttl_list) if im is not None]
        if not pairs:
            log.warning(f"Item {item.get('index')}: no images to show")
            return

        img_num = len(pairs)
        col_num = min(img_num, 3)
        row_num = int(np.ceil(img_num / col_num))
        fig, axes = plt.subplots(row_num, col_num, sharey=True, sharex=True)
        axes = np.array(axes).reshape(row_num, col_num)
        for k, (img, tt) in enumerate(pairs):
            ri, ci = k // col_num, k % col_num
            if img.ndim == 3 and img.shape[2] == 3:
                # BGR -> RGB for matplotlib if dtype is uint8.
                if img.dtype == np.uint8:
                    disp = img[..., ::-1]
                else:
                    disp = img
                axes[ri, ci].imshow(disp)
            else:
                vmax = None
                if "Depth" in tt:
                    vmax = 1024
                elif "IR" in tt:
                    vmax = 200
                axes[ri, ci].imshow(img, cmap='gray', vmax=vmax)
            axes[ri, ci].set_title(tt)
        for k in range(img_num, row_num * col_num):
            axes[k // col_num, k % col_num].axis('off')
        suptitle = (
            f"[{item.get('source')}] flat={item.get('index')} "
            f"local={item.get('source_index')}"
        )
        fig.suptitle(suptitle)
        plt.show(block=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataSource(unittest.TestCase):
    """Lightweight smoke tests for the multi-set ``DataSource``."""

    def _build_one(self, name: str) -> Optional[DataSource]:
        ds = DataSource()
        count = ds.add_source(name)
        if count == 0:
            log.warning(f"No samples found for source '{name}', skipping.")
            return None
        return ds

    def test_init_isaac_only(self):
        ds = self._build_one("isaac")
        if ds is None:
            return
        self.assertGreater(len(ds), 0)
        self.assertEqual(ds.source_of(0), "isaac")

    def test_init_all_defaults(self):
        ds = DataSource()
        total = ds.init_directory()
        log.info(f"Total items across all sources: {total}")
        # We can't guarantee any individual source is reachable on every
        # machine, but at least one of them should typically exist.
        if total == 0:
            log.warning("No samples found in any source; skipping.")
            return
        # The first item should normalize without error.
        item = ds.get_item(0, debug=False)
        self.assertIn("source", item)
        self.assertIn("source_index", item)
        self.assertIn("ir_left_img", item)
        self.assertIn("meta", item)

    def test_get_item_per_source(self):
        for name in ("isaac", "faro", "pickle"):
            ds = self._build_one(name)
            if ds is None:
                continue
            item = ds.get_item(0, debug=False)
            self.assertEqual(item["source"], name)
            self.assertEqual(item["source_index"], 0)
            self.assertEqual(item["index"], 0)
            # IR-left should be present for every supported source.
            self.assertIsNotNone(
                item["ir_left_img"],
                msg=f"{name}: ir_left_img is None",
            )

    def test_index_resolution(self):
        ds = DataSource()
        ds.init_directory()
        if len(ds) == 0:
            return
        last = len(ds) - 1
        first = ds.get_item(0)
        end = ds.get_item(last)
        self.assertEqual(first["index"], 0)
        self.assertEqual(end["index"], last)
        # Negative indexing should mirror Python list semantics.
        again = ds.get_item(-1)
        self.assertEqual(again["index"], last)

    def test_show_item(self):
        ds = DataSource()
        ds.init_directory()
        if len(ds) == 0:
            return
        rng = np.random.default_rng(0)
        for k in rng.integers(0, len(ds), size=min(3, len(ds))):
            item = ds.get_item(int(k), debug=True)
            self.assertIn("ir_left_img", item)
        plt.show()


def RunTest() -> None:
    tst = TestDataSource()
    tst.test_init_all_defaults()
    # tst.test_get_item_per_source()
    tst.test_show_item()


if __name__ == "__main__":
    RunTest()
