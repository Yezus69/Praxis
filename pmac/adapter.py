"""Domain adapter interface from PMA-C section 19."""

from __future__ import annotations

from abc import ABC, abstractmethod


class DomainAdapter(ABC):
    """Domain boundary for PMA-C.

    Implementations return a domain-specific behavior object from ``behavior``:
    class logits for supervised learning, policy/value objects for RL,
    embeddings for representation learning, token logits for language models,
    or another structure with a matching behavior distance.
    """

    @abstractmethod
    def behavior(self, params, batch):
        """Return behavior object: logits, policy/value, embedding, etc."""

    @abstractmethod
    def distance(self, cur, teacher, batch):
        """Return per-example behavior drift between current and teacher."""

    @abstractmethod
    def current_loss(self, params, batch):
        """Return scalar task objective for current data/environment."""

    @abstractmethod
    def evaluate_skill(self, params, skill_eval_set):
        """Return a certification/evaluation score for a protected skill."""

    @abstractmethod
    def make_challenge_batch(self, key, skill_eval_set):
        """Generate stress probes, augmentations, or adversarial cases."""

    @abstractmethod
    def grow_capacity(self, key, params, source):
        """Add a domain-appropriate no-op adapter/expert/head."""


__all__ = ["DomainAdapter"]
