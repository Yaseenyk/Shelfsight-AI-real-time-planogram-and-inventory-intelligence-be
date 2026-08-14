"""Cluster-based partitioning: the guarantee that held-out data is unseen.

The whole point of this module is a single property — an image and its
near-duplicates never land in different splits. A random split satisfies that
property by luck and usually fails it; these tests assert it directly, because
a regression here would silently inflate every reported accuracy rather than
raise anything.

Real images are synthesised with PIL so `dhash` is exercised on actual pixel
data, not on a stub. The augmented-sibling scenario the freshness corpus
suffered from is reproduced explicitly in `test_augmented_siblings_stay_together`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.cluster_split import (
    UnionFind,
    assign,
    cluster,
    collect,
    dhash,
    hash_all,
    run,
)


def _gradient(path: Path, seed: int, size: int = 64) -> None:
    """Write a deterministic image whose structure varies with `seed`."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data).save(path)


def _variant(source: Path, target: Path, brightness: int = 4) -> None:
    """A near-duplicate: same structure, slightly different pixels.

    This is what an augmented sibling looks like to a perceptual hash — the
    gradient relationships that dHash encodes survive a small uniform shift.
    """
    with Image.open(source) as im:
        data = np.asarray(im).astype(np.int16) + brightness
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(data, 0, 255).astype(np.uint8)).save(target)


# ------------------------------------------------------------------ dhash --
def test_dhash_is_stable_and_discriminating(tmp_path: Path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _gradient(a, seed=1)
    _gradient(b, seed=99)

    assert dhash(a) == dhash(a), "hashing must be deterministic"
    assert dhash(a) != dhash(b), "different content must hash differently"


def test_dhash_survives_re_encoding(tmp_path: Path):
    """A JPEG re-encode is the case exact-hash dedup misses entirely."""
    png = tmp_path / "x.png"
    _gradient(png, seed=7, size=128)
    jpg = tmp_path / "x.jpg"
    with Image.open(png) as im:
        im.convert("RGB").save(jpg, quality=92)

    distance = bin(dhash(png) ^ dhash(jpg)).count("1")
    assert distance <= 5, f"re-encode moved the hash by {distance} bits"


def test_dhash_returns_none_for_unreadable(tmp_path: Path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"this is not an image")
    assert dhash(broken) is None


def test_hash_all_drops_unreadable_and_reports_count(tmp_path: Path):
    good = tmp_path / "good.png"
    _gradient(good, seed=3)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")

    paths, hashes, unreadable = hash_all([good, bad])
    assert unreadable == 1
    assert paths == [good]
    assert len(hashes) == 1


# -------------------------------------------------------------- union-find --
def test_union_find_merges_transitively():
    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(4, 5)

    assert uf.find(0) == uf.find(2)
    assert uf.find(4) == uf.find(5)
    assert uf.find(0) != uf.find(4)
    assert uf.find(3) == 3, "an untouched element is its own root"


# ---------------------------------------------------------------- cluster --
def test_cluster_merges_a_transitive_chain():
    """a~b and b~c must put a and c together even when a and c exceed the radius."""
    hashes = np.array([0b0000, 0b0011, 0b1111], dtype=np.uint64)
    ids = cluster(hashes, threshold=2)
    assert ids[0] == ids[1] == ids[2]


def test_cluster_separates_distant_hashes():
    hashes = np.array([0, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
    ids = cluster(hashes, threshold=5)
    assert ids[0] != ids[1]


def test_cluster_collapses_exact_duplicates_at_zero_threshold():
    hashes = np.array([42, 42, 42, 7], dtype=np.uint64)
    ids = cluster(hashes, threshold=0)
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] != ids[0]


def test_cluster_respects_the_threshold_boundary():
    """Exactly-at-threshold links; one bit beyond does not."""
    at = np.array([0b000, 0b011], dtype=np.uint64)
    assert cluster(at, threshold=2)[0] == cluster(at, threshold=2)[1]

    beyond = np.array([0b000, 0b111], dtype=np.uint64)
    ids = cluster(beyond, threshold=2)
    assert ids[0] != ids[1]


def test_cluster_handles_more_items_than_one_chunk():
    """The chunked comparison must not lose pairs that straddle a chunk edge."""
    # 600 distinct hashes with chunk=512 forces a second block.
    hashes = np.array([i << 8 for i in range(600)], dtype=np.uint64)
    ids = cluster(hashes, threshold=0)
    assert len(set(ids.tolist())) == 600


# ----------------------------------------------------------------- assign --
def test_assign_never_splits_a_cluster():
    ids = np.array([0] * 50 + [1] * 30 + [2] * 20, dtype=np.int64)
    mapping = assign(ids, 0.15, 0.15)
    # One split per cluster id is the entire guarantee.
    assert set(mapping) == {0, 1, 2}
    for value in mapping.values():
        assert value in {"train", "val", "test"}


def test_assign_approximates_the_target_ratios():
    ids = np.array([i // 10 for i in range(1000)], dtype=np.int64)  # 100 x 10
    mapping = assign(ids, 0.15, 0.15)

    totals = {"train": 0, "val": 0, "test": 0}
    for split in mapping.values():
        totals[split] += 10

    assert abs(totals["val"] - 150) <= 20
    assert abs(totals["test"] - 150) <= 20
    assert abs(totals["train"] - 700) <= 40


def test_assign_is_deterministic():
    ids = np.array([i % 17 for i in range(200)], dtype=np.int64)
    assert assign(ids, 0.15, 0.15) == assign(ids, 0.15, 0.15)


def test_assign_places_one_dominant_cluster_without_starving_the_rest():
    """A single huge component must not swallow a split that others need."""
    ids = np.array([0] * 500 + list(range(1, 101)) * 5, dtype=np.int64)
    mapping = assign(ids, 0.15, 0.15)
    assert len(set(mapping.values())) == 3, "all three splits must be used"


# ------------------------------------------------------------------- run --
def _corpus(root: Path, labels=("fresh", "spoiled"), per_label: int = 12) -> None:
    for label_index, label in enumerate(labels):
        for i in range(per_label):
            _gradient(root / "train" / label / f"{label}_{i}.png", seed=label_index * 100 + i)


def test_run_produces_all_splits_and_a_manifest(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _corpus(src)

    manifest = run(src, out, ("fresh", "spoiled"), 0.15, 0.15, threshold=5)

    for split in ("train", "val", "test"):
        for label in ("fresh", "spoiled"):
            assert (out / split / label).is_dir()

    written = json.loads((out / "cluster_split_manifest.json").read_text(encoding="utf-8"))
    assert written["threshold_bits"] == 5
    assert written["hash"] == "dhash-64bit"
    assert sum(written["totals"].values()) == 24
    assert manifest["per_class"]["fresh"]["images"] == 12


def test_run_refuses_to_overwrite(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _corpus(src)
    out.mkdir()

    with pytest.raises(SystemExit):
        run(src, out, ("fresh", "spoiled"), 0.15, 0.15, threshold=5)


def test_run_rejects_a_missing_class(tmp_path: Path):
    src = tmp_path / "src"
    _corpus(src, labels=("fresh",))

    with pytest.raises(SystemExit):
        run(src, tmp_path / "out", ("fresh", "ripening"), 0.15, 0.15, threshold=5)


def test_augmented_siblings_stay_together(tmp_path: Path):
    """The exact failure the freshness corpus suffered from.

    One base photograph plus near-identical variants of it. Under a random
    image-level split these scatter across partitions; under cluster assignment
    they must all land in the same one.
    """
    src = tmp_path / "src"
    base_dir = src / "train" / "fresh"
    base = base_dir / "fresh_base.png"
    _gradient(base, seed=11, size=128)
    for i in range(6):
        _variant(base, base_dir / f"fresh_aug{i}.png", brightness=i + 1)
    # Unrelated content so the corpus has more than one component.
    for i in range(8):
        _gradient(src / "train" / "spoiled" / f"spoiled_{i}.png", seed=500 + i)

    out = tmp_path / "out"
    run(src, out, ("fresh", "spoiled"), 0.15, 0.15, threshold=5)

    # Every 'fresh' file descends from one photograph, so all 7 must be together.
    placements = {
        split for split in ("train", "val", "test") if any((out / split / "fresh").glob("*"))
    }
    assert len(placements) == 1, f"augmented siblings were split across {placements}"


def test_collect_pools_every_source_split(tmp_path: Path):
    """Existing boundaries are exactly what we distrust, so all are pooled."""
    src = tmp_path / "src"
    for split in ("train", "val", "test"):
        _gradient(src / split / "fresh" / f"{split}.png", seed=hash(split) % 1000)

    found = collect(src, "fresh")
    assert len(found) == 3
