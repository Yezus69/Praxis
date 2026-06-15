"""Continual-learning data streams for PMA-C."""

from __future__ import annotations

import gzip
import shutil
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MNIST_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)

MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


@dataclass
class Task:
    name: str
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    meta: dict


def _cache_path(cache_dir) -> Path:
    path = Path(cache_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_file(cache_dir: Path, filename: str) -> None:
    target = cache_dir / filename
    if target.exists():
        return
    errors = []
    for mirror in MNIST_MIRRORS:
        url = mirror + filename
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                with target.open("wb") as out:
                    shutil.copyfileobj(response, out)
            return
        except (OSError, urllib.error.URLError) as exc:
            errors.append(f"{url}: {exc}")
            if target.exists():
                target.unlink()
    raise RuntimeError(f"could not download {filename}; tried mirrors: {'; '.join(errors)}")


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise RuntimeError(f"invalid MNIST image magic {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, rows * cols).astype(np.float32) / np.float32(255.0)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise RuntimeError(f"invalid MNIST label magic {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count).astype(np.int32)


def load_mnist(cache_dir="~/.cache/pmac_mnist") -> dict:
    """Load MNIST as flattened float32 images in [0, 1], caching raw gz and npz."""
    cache = _cache_path(cache_dir)
    parsed = cache / "mnist_parsed.npz"
    if parsed.exists():
        with np.load(parsed) as data:
            return {name: data[name] for name in data.files}

    for filename in MNIST_FILES:
        _download_file(cache, filename)

    result = {
        "train_x": _read_idx_images(cache / "train-images-idx3-ubyte.gz"),
        "train_y": _read_idx_labels(cache / "train-labels-idx1-ubyte.gz"),
        "test_x": _read_idx_images(cache / "t10k-images-idx3-ubyte.gz"),
        "test_y": _read_idx_labels(cache / "t10k-labels-idx1-ubyte.gz"),
    }
    np.savez_compressed(parsed, **result)
    return result


def _task(name: str, train_x, train_y, test_x, test_y, meta: dict) -> Task:
    return Task(
        name=name,
        train_x=np.asarray(train_x, dtype=np.float32),
        train_y=np.asarray(train_y, dtype=np.int32),
        test_x=np.asarray(test_x, dtype=np.float32),
        test_y=np.asarray(test_y, dtype=np.int32),
        meta=dict(meta),
    )


def make_permuted_mnist(num_tasks=5, seed=0, cache_dir="~/.cache/pmac_mnist") -> list[Task]:
    data = load_mnist(cache_dir=cache_dir)
    rng = np.random.default_rng(seed)
    tasks = []
    n_features = data["train_x"].shape[1]
    for task_id in range(int(num_tasks)):
        if task_id == 0:
            perm = np.arange(n_features, dtype=np.int32)
        else:
            perm = rng.permutation(n_features).astype(np.int32)
        tasks.append(
            _task(
                f"permuted_mnist_{task_id}",
                data["train_x"][:, perm],
                data["train_y"],
                data["test_x"][:, perm],
                data["test_y"],
                {"task_id": task_id, "permutation": perm, "source": "mnist"},
            )
        )
    return tasks


def make_split_mnist(seed=0, cache_dir="~/.cache/pmac_mnist") -> list[Task]:
    data = load_mnist(cache_dir=cache_dir)
    rng = np.random.default_rng(seed)
    tasks = []
    for task_id, classes in enumerate(((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))):
        train_idx = np.flatnonzero(np.isin(data["train_y"], classes))
        test_idx = np.flatnonzero(np.isin(data["test_y"], classes))
        train_idx = train_idx[rng.permutation(train_idx.shape[0])]
        test_idx = test_idx[rng.permutation(test_idx.shape[0])]
        tasks.append(
            _task(
                f"split_mnist_{classes[0]}_{classes[1]}",
                data["train_x"][train_idx],
                data["train_y"][train_idx],
                data["test_x"][test_idx],
                data["test_y"][test_idx],
                {"task_id": task_id, "classes": classes, "source": "mnist"},
            )
        )
    return tasks


def make_synthetic_permuted(
    num_tasks=5,
    seed=0,
    n_features=784,
    n_classes=10,
    n_train=6000,
    n_test=1000,
) -> list[Task]:
    rng = np.random.default_rng(seed)
    n_features = int(n_features)
    n_classes = int(n_classes)
    means = rng.normal(0.0, 1.0, size=(n_classes, n_features)).astype(np.float32)

    def sample(count: int):
        y = rng.integers(0, n_classes, size=int(count), dtype=np.int32)
        x = means[y] + rng.normal(0.0, 0.35, size=(int(count), n_features)).astype(np.float32)
        return x.astype(np.float32), y

    tasks = []
    for task_id in range(int(num_tasks)):
        if task_id == 0:
            perm = np.arange(n_features, dtype=np.int32)
        else:
            perm = rng.permutation(n_features).astype(np.int32)
        train_x, train_y = sample(n_train)
        test_x, test_y = sample(n_test)
        tasks.append(
            _task(
                f"synthetic_permuted_{task_id}",
                train_x[:, perm],
                train_y,
                test_x[:, perm],
                test_y,
                {"task_id": task_id, "permutation": perm, "source": "synthetic"},
            )
        )
    return tasks


def build_stream(name="permuted_mnist", **kw) -> tuple[list[Task], str]:
    """Build a named stream, falling back to synthetic data for MNIST failures."""
    if name == "synthetic":
        return make_synthetic_permuted(**kw), "synthetic"
    if name == "permuted_mnist":
        try:
            return make_permuted_mnist(**kw), "mnist"
        except RuntimeError:
            return _synthetic_fallback(kw), "synthetic_fallback"
    if name == "split_mnist":
        try:
            return make_split_mnist(**kw), "mnist"
        except RuntimeError:
            return _synthetic_fallback(kw), "synthetic_fallback"
    raise ValueError(f"unknown stream name: {name}")


def _synthetic_fallback(kw) -> list[Task]:
    allowed = {"num_tasks", "seed", "n_features", "n_classes", "n_train", "n_test"}
    return make_synthetic_permuted(**{k: v for k, v in kw.items() if k in allowed})


def _seed_from_key(key) -> int:
    arr = np.asarray(key, dtype=np.uint32).reshape(-1)
    seed = 0
    for value in arr:
        seed = (1664525 * seed + int(value) + 1013904223) % (2**32)
    return int(seed)


def iterate_minibatches(key, x, y, batch_size, drop_last: bool = False):
    """Yield deterministic shuffled minibatches from host arrays."""
    x = np.asarray(x)
    y = np.asarray(y)
    rng = np.random.default_rng(_seed_from_key(key))
    indices = rng.permutation(x.shape[0])
    batch_size = int(batch_size)
    for start in range(0, indices.shape[0], batch_size):
        if drop_last and start + batch_size > indices.shape[0]:
            break
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


__all__ = [
    "Task",
    "load_mnist",
    "make_permuted_mnist",
    "make_split_mnist",
    "make_synthetic_permuted",
    "build_stream",
    "iterate_minibatches",
]
