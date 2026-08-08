use std::cmp::Ordering;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

struct StoredVector {
    vector: Vec<f32>,
    id: u64,
}

pub struct Vectorstore {
    vector_size: usize,
    vectors: Vec<StoredVector>,
}

pub fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> Result<f32, String> {
    if vec1.len() != vec2.len() {
        return Err(format!(
            "vectors have different dimensions ({} vs {})",
            vec1.len(),
            vec2.len()
        ));
    }
    Ok(similarity(vec1, vec2))
}

fn similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let dot: f32 = vec1.iter().zip(vec2).map(|(x, y)| x * y).sum();
    let norm1 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot / (norm1 * norm2)
    }
}

impl StoredVector {
    fn new(vector: Vec<f32>, id: u64) -> Self {
        Self { vector, id }
    }
}

impl Vectorstore {
    pub fn new(
        vector_size: usize,
        vectors: Option<Vec<(Vec<f32>, u64)>>,
    ) -> Result<Self, String> {
        let mut stored = Vec::new();
        for (vector, id) in vectors.unwrap_or_default() {
            if vector.len() != vector_size {
                return Err(format!(
                    "vector has length {} but vectorstore expects {}",
                    vector.len(),
                    vector_size
                ));
            }
            stored.push(StoredVector::new(vector, id));
        }
        Ok(Self { vector_size, vectors: stored })
    }

    pub fn vector_size(&self) -> usize {
        self.vector_size
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn insert(&mut self, vector: Vec<f32>, id: u64) -> Result<(), String> {
        if vector.len() != self.vector_size {
            return Err(format!(
                "vector has length {} but vectorstore expects {}",
                vector.len(),
                self.vector_size
            ));
        }
        self.vectors.push(StoredVector::new(vector, id));
        Ok(())
    }

    pub fn search(&self, vector: Vec<f32>, top_k: usize) -> Result<Vec<(f32, u64)>, String> {
        if vector.len() != self.vector_size {
            return Err(format!(
                "query has length {} but vectorstore expects {}",
                vector.len(),
                self.vector_size
            ));
        }

        let mut scored: Vec<(f32, &StoredVector)> = self
            .vectors
            .iter()
            .map(|v| (similarity(&vector, &v.vector), v))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        scored.truncate(top_k);

        Ok(scored.into_iter().map(|(score, v)| (score, v.id.clone())).collect())
    }

    pub fn parallel_search(
        &self,
        query: Vec<f32>,
        top_k: usize,
        n_threads: Option<usize>,
    ) -> Result<Vec<(f32, u64)>, String> {
        if query.len() != self.vector_size {
            return Err(format!(
                "query has length {} but vectorstore expects {}",
                query.len(),
                self.vector_size
            ));
        }
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let n_threads = n_threads
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
            })
            .max(1);
        let chunk_size = self.vectors.len().div_ceil(n_threads);

        let mut all: Vec<(f32, &StoredVector)> = std::thread::scope(|s| {
            self.vectors
                .chunks(chunk_size)
                .map(|chunk| {
                    s.spawn(|| {
                        let mut local: Vec<(f32, &StoredVector)> = chunk
                            .iter()
                            .map(|v| (similarity(&query, &v.vector), v))
                            .collect();
                        local.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
                        local.truncate(top_k);
                        local
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });

        all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        all.truncate(top_k);
        Ok(all.into_iter().map(|(score, v)| (score, v.id.clone())).collect())
    }
}

#[pyclass(name = "Vectorstore")]
struct PyVectorstore {
    inner: Vectorstore,
}

#[pymethods]
impl PyVectorstore {
    #[new]
    #[pyo3(signature = (vector_size, vectors=None))]
    fn new(vector_size: usize, vectors: Option<Vec<(Vec<f32>, u64)>>) -> PyResult<Self> {
        Ok(Self {
            inner: Vectorstore::new(vector_size, vectors).map_err(PyValueError::new_err)?,
        })
    }

    fn insert(&mut self, vector: Vec<f32>, id: u64) -> PyResult<()> {
        self.inner.insert(vector, id).map_err(PyValueError::new_err)
    }

    fn search(&self, vector: Vec<f32>, top_k: usize) -> PyResult<Vec<(f32, u64)>> {
        self.inner.search(vector, top_k).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (vector, top_k, n_threads=None))]
    fn parallel_search(
        &self,
        vector: Vec<f32>,
        top_k: usize,
        n_threads: Option<usize>,
    ) -> PyResult<Vec<(f32, u64)>> {
        self.inner
            .parallel_search(vector, top_k, n_threads)
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn vector_size(&self) -> usize {
        self.inner.vector_size()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Vectorstore(vector_size={}, len={})",
            self.inner.vector_size(),
            self.inner.len()
        )
    }
}

#[pyfunction]
#[pyo3(name = "cosine_similarity")]
fn cosine_similarity_py(vec1: Vec<f32>, vec2: Vec<f32>) -> PyResult<f32> {
    cosine_similarity(&vec1, &vec2).map_err(PyValueError::new_err)
}

#[pymodule]
fn vectorstore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVectorstore>()?;
    m.add_function(wrap_pyfunction!(cosine_similarity_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_store() -> Vectorstore {
        let vectors = vec![
            (vec![1.0, 2.0, 3.0], 0),
            (vec![1.0, 2.0, 4.0], 1),
            (vec![1.0, 2.0, 5.0], 2),
            (vec![1.0, 2.0, 6.0], 3),
            (vec![1.0, 2.0, 7.5], 4),
            (vec![1.0, 2.0, 8.0], 5),
            (vec![1.0, 2.0, 9.0], 6),
            (vec![1.0, 2.0, 10.0], 7),
            (vec![1.0, 2.0, 11.0], 8),
        ];
        Vectorstore::new(3, Some(vectors)).unwrap()
    }

    #[test]
    fn test_vector_store() {
        let vs = sample_store();
        let results = vs.search(vec![1.0, 2.0, 6.0], 3).unwrap();

        let tolerance = 1e-6;
        assert!((results.first().unwrap().0 - 1.0).abs() < tolerance);
        assert_eq!(results.first().unwrap().1, 3);
    }

    #[test]
    fn test_cosine_similarity() {
        let similarity = cosine_similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch_is_error() {
        assert!(cosine_similarity(&[1.0, 2.0], &[1.0, 2.0, 3.0]).is_err());

        let mut vs = Vectorstore::new(3, None).unwrap();
        assert!(vs.insert(vec![1.0, 2.0], 2).is_err());
        assert!(vs.search(vec![1.0, 2.0], 3).is_err());
    }

    #[test]
    fn test_zero_vector_does_not_panic() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]).unwrap(), 0.0);
    }

    #[test]
    fn test_search_returns_all_when_fewer_than_top_k() {
        let vs = Vectorstore::new(2, Some(vec![
            (vec![1.0, 0.0], 0),
            (vec![-1.0, 0.0], 1),
        ]))
        .unwrap();

        let results = vs.search(vec![1.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1, 0);
        assert!((results[0].0 - 1.0).abs() < 1e-6);
        assert_eq!(results[1].1, 1);
        assert!((results[1].0 + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_serial_and_parallel_agree() {
        let vs = sample_store();
        let query = vec![1.0, 2.0, 6.0];
        let serial = vs.search(query.clone(), 4).unwrap();
        let parallel = vs.parallel_search(query, 4, Some(3)).unwrap();
        assert_eq!(serial.len(), parallel.len());
        for (a, b) in serial.iter().zip(&parallel) {
            assert!((a.0 - b.0).abs() < 1e-6);
            assert_eq!(a.1, b.1);
        }
    }
}
