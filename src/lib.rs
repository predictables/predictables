use pyo3::prelude::*;
use rayon::prelude::*;

fn is_dominated(a: &[f64], b: &[f64]) -> bool {
    let a_ge_b = a.iter().zip(b.iter()).all(|(ai, bi)| ai >= bi);
    let a_gt_b = a.iter().zip(b.iter()).any(|(ai, bi)| ai > bi);
    a_ge_b && a_gt_b
}

fn pareto_sort_rs(points: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    points
        .par_iter()
        .filter(|a| !points.par_iter().any(|b| is_dominated(a, b)))
        .cloned()
        .collect()
}

#[pyfunction]
fn pareto_sort_py(points: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    pareto_sort_rs(points)
}

#[pymodule]
fn pareto_sort(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pareto_sort_py, m)?)?;
    Ok(())
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_pareto_sort() {
        let points = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 4.0],
            vec![1.0, 2.0, 5.0],
            vec![1.0, 2.0, 6.0],
            vec![1.0, 2.0, 7.0],
            vec![1.0, 2.0, 8.0],
            vec![1.0, 2.0, 9.0],
            vec![1.0, 2.0, 10.0],
            vec![1.0, 2.0, 11.0],
            vec![1.0, 2.0, 12.0],
            vec![1.0, 2.0, 13.0],
            vec![1.0, 2.0, 14.0],
            vec![1.0, 2.0, 15.0],
            vec![1.0, 2.0, 16.0],
            vec![1.0, 2.0, 17.0],
            vec![1.0, 2.0, 18.0],
            vec![1.0, 2.0, 19.0],
            vec![1.0, 2.0, 20.0],
            vec![1.0, 2.0, 21.0],
            vec![1.0, 2.0, 22.0],
            vec![1.0, 2.0, 23.0],
            vec![1.0, 2.0, 24.0],
            vec![1.0, 2.0, 25.0],
            vec![1.0, 2.0, 26.0],
            vec![1.0, 2.0, 27.0],
            vec![1.0, 2.0, 28.0],
            vec![1.0, 2.0, 29.0],
        ];
        let result = pareto_sort_rs(points);
        assert_eq!(result.len(), 1);

        assert_eq!(result[0], vec![1.0, 2.0, 29.0]);
    }

    #[test]
    fn test_pareto_sort_py() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let points = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 4.0],
            vec![1.0, 2.0, 5.0],
            vec![1.0, 2.0, 6.0],
            vec![1.0, 2.0, 7.0],
            vec![1.0, 2.0, 8.0],
            vec![1.0, 2.0, 9.0],
            vec![1.0, 2.0, 10.0],
            vec![1.0, 2.0, 11.0],
            vec![1.0, 2.0, 12.0],
            vec![1.0, 2.0, 13.0],
            vec![1.0, 2.0, 14.0],
            vec![1.0, 2.0, 15.0],
            vec![1.0, 2.0, 16.0],
            vec![1.0, 2.0, 17.0],
            vec![1.0, 2.0, 18.0],
            vec![1.0, 2.0, 19.0],
            vec![1.0, 2.0, 20.0],
            vec![1.0, 2.0, 21.0],
            vec![1.0, 2.0, 22.0],
            vec![1.0, 2.0, 23.0],
            vec![1.0, 2.0, 24.0],
            vec![1.0, 2.0, 25.0],
            vec![1.0, 2.0, 26.0],
            vec![1.0, 2.0, 27.0],
            vec![1.0, 2.0, 28.0],
            vec![1.0, 2.0, 29.0],
        ];
        let result = pareto_sort_py(points);
        assert_eq!(result.len(), 1);

        assert_eq!(result[0], vec![1.0, 2.0, 29.0]);
    }
}
