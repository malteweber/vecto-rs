import gc
import math
import random
import resource
import subprocess
import sys
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

from vectorsearch_rs import Vectorstore


# --- Pure Python implementation ---

def py_cosine_similarity(v1, v2):
    dot = sum(x * y for x, y in zip(v1, v2))
    norm1 = math.sqrt(sum(x * x for x in v1))
    norm2 = math.sqrt(sum(x * x for x in v2))
    return dot / (norm1 * norm2)


class PyVectorstore:
    def __init__(self, vector_size):
        self.vector_size = vector_size
        self.vectors = []

    def insert(self, vector, metadata=None):
        self.vectors.append((vector, metadata))

    def search(self, query, top_k):
        scored = [(py_cosine_similarity(query, v), v, m) for v, m in self.vectors]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


# --- Numpy implementation ---

class NumpyVectorstore:
    def __init__(self, vector_size):
        self.vector_size = vector_size
        self._matrix = np.empty((0, vector_size), dtype=np.float32)

    def insert(self, vector, metadata=None):
        row = np.array(vector, dtype=np.float32).reshape(1, -1)
        self._matrix = np.vstack([self._matrix, row]) if len(self._matrix) else row

    def search(self, query, top_k):
        q = np.array(query, dtype=np.float32)
        dots = self._matrix @ q
        norms = np.linalg.norm(self._matrix, axis=1) * np.linalg.norm(q)
        scores = dots / norms
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(float(scores[i]), self._matrix[i].tolist()) for i in top_idx]


# --- Scikit-learn implementation ---

class SklearnVectorstore:
    def __init__(self, _vector_size):
        self._data = []
        self._nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._fitted = False

    def insert(self, vector, metadata=None):
        self._data.append(vector)
        self._fitted = False

    def search(self, query, top_k):
        if not self._fitted:
            self._nn.fit(np.array(self._data, dtype=np.float32))
            self._fitted = True
        q = np.array(query, dtype=np.float32).reshape(1, -1)
        distances, indices = self._nn.kneighbors(q, n_neighbors=top_k)
        # sklearn cosine metric returns dissimilarity (1 - similarity)
        return [(1 - float(d), self._data[i]) for d, i in zip(distances[0], indices[0])]


# --- Benchmark helpers ---

# Maps a CLI/label key to a factory that builds an empty store for a given dim.
# Shared by the search, insertion and memory-footprint benchmarks so every
# implementation is exercised identically.
STORE_FACTORIES = {
    "rust": Vectorstore,
    "python": PyVectorstore,
    "numpy": NumpyVectorstore,
    "sklearn": SklearnVectorstore,
}


def random_vector(dim):
    return [random.uniform(-1, 1) for _ in range(dim)]


def bench(label, fn, iterations=5):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"  {label}: {avg * 1000:.1f} ms (avg over {iterations} runs)")
    return avg


def bench_insert(label, impl, data, dim, iterations=5):
    """Time building a fresh store and inserting every vector."""
    def build():
        store = STORE_FACTORIES[impl](dim)
        for i, v in enumerate(data):
            store.insert(v, i)
    return bench(label, build, iterations)


def _read_rss_kb():
    """Resident set size of the current process in KiB (Linux /proc)."""
    with open("/proc/self/statm") as f:
        resident_pages = int(f.read().split()[1])
    return resident_pages * (resource.getpagesize() // 1024)


def mem_worker(impl, dim, n_vectors, seed):
    """Build one store in isolation and print its RSS footprint in KiB.

    Run in a dedicated subprocess so the measurement only reflects a single
    implementation and includes native (e.g. Rust) allocations, not just the
    Python heap. Source vectors are generated inline and never retained
    separately, so the delta is the store's true cost of holding the data.
    """
    random.seed(seed)
    gc.collect()
    baseline = _read_rss_kb()
    store = STORE_FACTORIES[impl](dim)
    for i in range(n_vectors):
        store.insert(random_vector(dim), i)
    gc.collect()
    after = _read_rss_kb()
    print(after - baseline)


def mem_bench(label, impl, dim, n_vectors, seed):
    """Measure an implementation's memory footprint via an isolated subprocess."""
    result = subprocess.run(
        [sys.executable, __file__, "--mem-worker", impl, str(dim), str(n_vectors), str(seed)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  {label}: failed\n{result.stderr.strip()}")
        return None
    kb = int(result.stdout.strip())
    print(f"  {label}: {kb / 1024:.1f} MB")
    return kb


def run(dim, n_vectors, top_k):
    print(f"\ndim={dim}, n_vectors={n_vectors}, top_k={top_k}")
    print("-" * 50)

    data = [random_vector(dim) for _ in range(n_vectors)]
    query = random_vector(dim)

    # Insertion: time building a fresh store and inserting every vector.
    print("  [insertion]")
    bench_insert("Rust     insert", "rust", data, dim)
    #bench_insert("Python   insert", "python", data, dim)
    bench_insert("Numpy    insert", "numpy", data, dim)
    bench_insert("Sklearn  insert", "sklearn", data, dim)

    # Build stores for the search benchmark.
    rust_vs = Vectorstore(dim)
    np_vs = NumpyVectorstore(dim)
    sk_vs = SklearnVectorstore(dim)
    for i, v in enumerate(data):
        rust_vs.insert(v, i)
        np_vs.insert(v)
        sk_vs.insert(v)

    print("  [search]")
    #rust_time = bench("Rust     search", lambda: rust_vs.search(query, top_k))
    par_time  = bench("Rust par search", lambda: rust_vs.parallel_search(query, top_k))
    np_time   = bench("Numpy    search", lambda: np_vs.search(query, top_k))
    sk_time   = bench("Sklearn  search", lambda: sk_vs.search(query, top_k))

    print(f"  Speedup Rust-par vs Numpy: {np_time / par_time:.1f}x, vs Sklearn: {sk_time / par_time:.1f}x")

    # Memory footprint: each store built in an isolated subprocess so the RSS
    # delta reflects only that implementation (native allocations included).
    print("  [memory]")
    mem_bench("Rust     memory", "rust", dim, n_vectors, MEM_SEED)
    mem_bench("Python   memory", "python", dim, n_vectors, MEM_SEED)
    mem_bench("Numpy    memory", "numpy", dim, n_vectors, MEM_SEED)
    mem_bench("Sklearn  memory", "sklearn", dim, n_vectors, MEM_SEED)


MEM_SEED = 42

if __name__ == "__main__":
    # Subprocess entrypoint for the isolated memory measurement.
    if len(sys.argv) > 1 and sys.argv[1] == "--mem-worker":
        _, _, impl, dim, n_vectors, seed = sys.argv
        mem_worker(impl, int(dim), int(n_vectors), int(seed))
        sys.exit(0)

    random.seed(42)
    run(dim=128,  n_vectors=1_000,   top_k=10)
    run(dim=128,  n_vectors=10_000,  top_k=10)
    run(dim=1536, n_vectors=10_000,  top_k=10)