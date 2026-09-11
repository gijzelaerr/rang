//! Small dense symmetric linear algebra for the toy's 16 pointing parameters.
//! No external numerical dependency is needed to reproduce this experiment.

pub type Matrix = Vec<Vec<f64>>;

pub fn zeros(n: usize, m: usize) -> Matrix {
    vec![vec![0.0; m]; n]
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn mv(a: &Matrix, v: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, v)).collect()
}

/// Jacobi eigensolver. Eigenvectors are columns; eigenvalues are descending.
pub fn eigen(a: &Matrix) -> (Vec<f64>, Matrix) {
    let n = a.len();
    let mut a = a.clone();
    let mut v = zeros(n, n);
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let scale = a.iter().flatten().fold(0.0_f64, |s, x| s.max(x.abs()));
    for _ in 0..100 * n * n {
        let mut largest = 0.0;
        let (mut p, mut q) = (0, 0);
        for (i, row) in a.iter().enumerate() {
            for (j, value) in row.iter().enumerate().skip(i + 1) {
                if value.abs() > largest {
                    largest = value.abs();
                    (p, q) = (i, j);
                }
            }
        }
        if largest <= 1e-13 * scale.max(1e-30) {
            break;
        }
        let angle = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let (s, c) = angle.sin_cos();
        let (app, aqq, apq) = (a[p][p], a[q][q], a[p][q]);
        for k in 0..n {
            if k != p && k != q {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - s * akq;
                a[p][k] = a[k][p];
                a[k][q] = s * akp + c * akq;
                a[q][k] = a[k][q];
            }
            let (vkp, vkq) = (v[k][p], v[k][q]);
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
        a[p][p] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
    }
    let mut indices: Vec<_> = (0..n).collect();
    indices.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let values = indices.iter().map(|&i| a[i][i]).collect();
    let vectors = v
        .iter()
        .map(|row| indices.iter().map(|&i| row[i]).collect())
        .collect();
    (values, vectors)
}

pub fn solve(a: &Matrix, b: &[f64]) -> Vec<f64> {
    let (values, vectors) = eigen(a);
    let cutoff = values.first().copied().unwrap_or(0.0).max(1e-30) * 1e-12;
    let mut x = vec![0.0; b.len()];
    for (k, &value) in values.iter().enumerate() {
        if value > cutoff {
            let projection: f64 = b.iter().enumerate().map(|(i, b)| vectors[i][k] * b).sum();
            for (i, xi) in x.iter_mut().enumerate() {
                *xi += vectors[i][k] * projection / value;
            }
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_reconstructs_symmetric_matrix() {
        let a = vec![
            vec![4.0, 1.0, -0.5],
            vec![1.0, 2.0, 0.3],
            vec![-0.5, 0.3, 1.0],
        ];
        let (d, v) = eigen(&a);
        for i in 0..3 {
            for j in 0..3 {
                let recovered: f64 = (0..3).map(|k| v[i][k] * d[k] * v[j][k]).sum();
                assert!((recovered - a[i][j]).abs() < 1e-11);
            }
        }
        let b = vec![1.0, 2.0, 3.0];
        let x = solve(&a, &b);
        assert!(mv(&a, &x).iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-10));
    }
}
