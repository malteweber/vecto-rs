# vectorsearch-rs

A tiny, fast **in-memory vector store** with cosine-similarity search, written
in Rust and exposed to Python via [pyo3](https://pyo3.rs). It does one thing:
keep a pile of equal-length vectors, each tagged with an integer `id`, and
return the `top_k` most similar ids to a query, with a parallel search path for
large stores.

No index building, no config, no server. Just insert and search.

## Install

```bash
pip install vectorsearch-rs
```

To build from source you need a Rust toolchain and [maturin](https://www.maturin.rs):

```bash
pip install maturin
maturin develop --release   # builds and installs into the active venv
```

## Usage

```python
from vectorsearch_rs import Vectorstore, cosine_similarity

# Every vector must have this many dimensions.
store = Vectorstore(vector_size=3)

store.insert([1.0, 2.0, 3.0], 1)
store.insert([1.0, 2.0, 4.0], 2)
store.insert([9.0, 0.0, 0.0], 3)

# Metadata lives on the Python side, keyed by the same ids.
meta = {1: {"title": "first"}, 2: {"title": "second"}, 3: {"title": "third"}}

# search returns a list of (score, id), most similar first.
for score, id in store.search([1.0, 2.0, 3.5], top_k=2):
    print(score, id, meta[id])

# For large stores, spread the scan across threads (defaults to all cores).
store.parallel_search([1.0, 2.0, 3.5], top_k=2, n_threads=8)

len(store)                                 # -> 3
cosine_similarity([1.0, 0.0], [1.0, 0.0])  # -> 1.0
```

You can also seed the store at construction time:

```python
store = Vectorstore(3, [([1.0, 2.0, 3.0], 1), ([0.0, 1.0, 0.0], 2)])
```

To run the benchmarking script, you need to install the dev dependencies (numpy and scikit-learn)

## Behaviour notes

- Inserting or querying a vector whose length differs from `vector_size`
  raises `ValueError`.
- Cosine similarity of a zero-magnitude vector is defined as `0.0` (rather than
  `NaN`), so zero vectors simply never match.
- `search` is single-threaded; `parallel_search` fans the scan out across
  threads and merges the per-thread top-k. Both return identical results.

## License

MIT — see [LICENSE](LICENSE).
