import jax.numpy as jnp
import numpy as np

from pmac.anchors import AnchorStore
from pmac.atlas import SkillNode
from pmac.router import Router
from pmac.sentinels import SentinelStore


def test_route_returns_registered_impl_and_compose_logits_weighted_sum():
    router = Router()
    router.register("ctx", "impl-a")
    assert router.route("ctx") == "impl-a"

    z0 = jnp.asarray([[1.0, 2.0]])
    z1 = jnp.asarray([[3.0, 5.0]])
    out = router.compose_logits([z0, z1], [0.25, 0.75])
    assert np.allclose(np.asarray(out), np.asarray(0.25 * z0 + 0.75 * z1))


def test_certify_route_blocks_wrong_impl():
    router = Router()
    router.register("ctx", "right")
    anchors = AnchorStore(1)
    sentinels = SentinelStore(np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=np.int32))
    node = SkillNode("s", "ctx", "protected", anchors, sentinels)

    assert router.certify_route("ctx", "right", node, adapter=None)
    assert not router.certify_route("ctx", "wrong", node, adapter=None)
