import numpy as np
import pytest

from pmac.data import streams


def test_synthetic_stream_shapes_labels_and_permutations():
    tasks = streams.make_synthetic_permuted(
        num_tasks=3, seed=0, n_features=12, n_classes=4, n_train=20, n_test=10
    )

    assert len(tasks) == 3
    assert tasks[0].train_x.shape == (20, 12)
    assert tasks[0].test_x.shape == (10, 12)
    assert tasks[0].train_y.dtype == np.int32
    assert set(np.unique(tasks[0].train_y)).issubset(set(range(4)))
    assert not np.array_equal(tasks[0].meta["permutation"], tasks[1].meta["permutation"])


def test_build_stream_synthetic_returns_five_tasks():
    tasks, source = streams.build_stream(
        "synthetic", seed=1, n_features=8, n_classes=3, n_train=12, n_test=6
    )

    assert source == "synthetic"
    assert len(tasks) == 5


def test_permuted_mnist_uses_same_permutation_for_train_and_test(monkeypatch):
    train_x = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    test_x = np.arange(3 * 4, dtype=np.float32).reshape(3, 4) + 100.0
    fake = {
        "train_x": train_x,
        "train_y": np.arange(6, dtype=np.int32) % 10,
        "test_x": test_x,
        "test_y": np.arange(3, dtype=np.int32) % 10,
    }
    monkeypatch.setattr(streams, "load_mnist", lambda cache_dir: fake)

    tasks = streams.make_permuted_mnist(num_tasks=2, seed=123, cache_dir="unused")
    perm = tasks[1].meta["permutation"]

    assert np.array_equal(tasks[1].train_x, train_x[:, perm])
    assert np.array_equal(tasks[1].test_x, test_x[:, perm])


def test_load_mnist_download_or_skip_when_offline(tmp_path):
    try:
        data = streams.load_mnist(cache_dir=tmp_path)
    except RuntimeError as exc:
        pytest.skip(f"MNIST unavailable: {exc}")

    assert data["train_x"].shape == (60000, 784)
    assert data["train_y"].shape == (60000,)
    assert data["test_x"].shape == (10000, 784)
    assert data["test_y"].shape == (10000,)
