"""Type stubs for the `vectorsearch_rs` extension module."""

from typing import Optional

SearchResult = tuple[float, int]

class Vectorstore:
    """An in-memory store of fixed-size vectors, each identified by an int id.

    The store holds only vectors and their ids. Any metadata associated with an
    id is the caller's concern — keep it in a plain Python ``dict[int, ...]`` on
    the Python side and look it up by the ids that :meth:`search` returns.
    """

    def __init__(
        self,
        vector_size: int,
        vectors: Optional[list[tuple[list[float], int]]] = ...,
    ) -> None: ...
    @property
    def vector_size(self) -> int:
        """The dimensionality every stored vector must have."""

    def insert(self, vector: list[float], id: int) -> None:
        """Add a vector under ``id``. Raises ``ValueError`` if its length != ``vector_size``."""

    def search(self, vector: list[float], top_k: int) -> list[SearchResult]:
        """Return the ``top_k`` most similar entries as ``(score, id)``, best first."""

    def parallel_search(
        self,
        vector: list[float],
        top_k: int,
        n_threads: Optional[int] = ...,
    ) -> list[SearchResult]:
        """Like :meth:`search`, but scored across ``n_threads`` (default: all cores)."""

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is zero)."""
