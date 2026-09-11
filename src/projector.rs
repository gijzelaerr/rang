//! Stable nuisance elimination for small calibration blocks.
//!
//! Column-pivoted Householder QR rotates the targets into nuisance-orthogonal
//! coordinates. No normal equations are formed. Returned rows are rotated
//! residual coordinates, not residuals in the original visibility order.

pub fn project_out(
    nuisance: &[f64],
    targets: &[f64],
    rows: usize,
    nuisance_cols: usize,
    target_cols: usize,
    relative_tolerance: f64,
) -> Result<(Vec<f64>, usize), &'static str> {
    if rows == 0
        || target_cols == 0
        || rows.checked_mul(nuisance_cols) != Some(nuisance.len())
        || rows.checked_mul(target_cols) != Some(targets.len())
    {
        return Err("incompatible dimensions");
    }
    if !relative_tolerance.is_finite()
        || relative_tolerance <= 0.0
        || relative_tolerance >= 1.0
        || !nuisance.iter().chain(targets).all(|x| x.is_finite())
    {
        return Err("finite matrices and tolerance in (0,1) required");
    }
    let mut a = nuisance.to_vec();
    let mut z = targets.to_vec();
    // A common rescaling preserves the column space and keeps squared norms
    // safe without calling hypot for every element of every pivot candidate.
    let normalization = a.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if normalization > 0.0 {
        for x in &mut a {
            *x /= normalization;
        }
    }
    let column_norm = |matrix: &[f64], start: usize, column: usize| {
        (start..rows)
            .map(|r| matrix[r * nuisance_cols + column].powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let scale = (0..nuisance_cols)
        .map(|c| column_norm(&a, 0, c))
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() {
        return Err("nuisance column norm overflow");
    }
    let mut rank = 0;
    for step in 0..rows.min(nuisance_cols) {
        let (pivot, norm) = (step..nuisance_cols)
            .map(|c| (c, column_norm(&a, step, c)))
            .max_by(|x, y| x.1.total_cmp(&y.1))
            .unwrap();
        if norm <= relative_tolerance * scale {
            break;
        }
        for r in 0..rows {
            a.swap(r * nuisance_cols + step, r * nuisance_cols + pivot);
        }
        let mut v: Vec<f64> = (step..rows).map(|r| a[r * nuisance_cols + step]).collect();
        v[0] += norm.copysign(v[0]);
        let vnorm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !vnorm.is_finite() || vnorm == 0.0 {
            return Err("reflector overflow");
        }
        for x in &mut v {
            *x /= vnorm;
        }
        for c in step..nuisance_cols {
            let dot: f64 = v
                .iter()
                .enumerate()
                .map(|(i, &x)| x * a[(step + i) * nuisance_cols + c])
                .sum();
            for (i, &x) in v.iter().enumerate() {
                a[(step + i) * nuisance_cols + c] -= 2.0 * x * dot;
            }
        }
        for c in 0..target_cols {
            let dot: f64 = v
                .iter()
                .enumerate()
                .map(|(i, &x)| x * z[(step + i) * target_cols + c])
                .sum();
            for (i, &x) in v.iter().enumerate() {
                z[(step + i) * target_cols + c] -= 2.0 * x * dot;
            }
        }
        rank += 1;
    }
    z[..rank * target_cols].fill(0.0);
    if !z.iter().all(|x| x.is_finite()) {
        return Err("target projection overflow");
    }
    Ok((z, rank))
}

/// Project row-major targets off nuisance columns. Returns 0 on success,
/// 1 for invalid arguments or numerical overflow, and 2 for an internal panic.
/// The output has `rows * target_cols` values; its first `rank` rows are zero.
///
/// # Safety
/// Nonempty input buffers must point to readable doubles of the stated sizes.
/// `output` must point to writable storage for `rows * target_cols` doubles,
/// and `rank_output` to a writable usize. Output buffers must not overlap any
/// input buffer or one another. The caller owns all buffers for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rang_project_out(
    rows: usize,
    nuisance_cols: usize,
    target_cols: usize,
    nuisance: *const f64,
    targets: *const f64,
    output: *mut f64,
    rank_output: *mut usize,
    relative_tolerance: f64,
) -> i32 {
    let Some(na) = rows.checked_mul(nuisance_cols) else {
        return 1;
    };
    let Some(nz) = rows.checked_mul(target_cols) else {
        return 1;
    };
    if (na > 0 && nuisance.is_null())
        || targets.is_null()
        || output.is_null()
        || rank_output.is_null()
        || na > isize::MAX as usize / size_of::<f64>()
        || nz > isize::MAX as usize / size_of::<f64>()
    {
        return 1;
    }
    let result = std::panic::catch_unwind(|| {
        let a = if na == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(nuisance, na) }
        };
        let z = unsafe { std::slice::from_raw_parts(targets, nz) };
        let (projected, rank) =
            project_out(a, z, rows, nuisance_cols, target_cols, relative_tolerance)?;
        unsafe {
            std::ptr::copy_nonoverlapping(projected.as_ptr(), output, nz);
            *rank_output = rank;
        }
        Ok::<(), &'static str>(())
    });
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(_)) => 1,
        Err(_) => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_deficient_projection_preserves_residual_gram() {
        let a = [1., 2., 2., 4., 3., 6., 4., 8.];
        let z = [1., 0., 0., 1., 0., 0., 0., 0.];
        let (p, rank) = project_out(&a, &z, 4, 2, 2, 1e-12).unwrap();
        assert_eq!(rank, 1);
        for i in 0..2 {
            for j in 0..2 {
                let gram: f64 = (0..4).map(|r| p[2 * r + i] * p[2 * r + j]).sum();
                let expected = f64::from(i == j) - ((i + 1) * (j + 1)) as f64 / 30.;
                assert!((gram - expected).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn empty_nuisance_and_complete_nuisance() {
        let z = [1., 2.];
        assert_eq!(
            project_out(&[], &z, 2, 0, 1, 1e-12).unwrap(),
            (z.to_vec(), 0)
        );
        assert_eq!(
            project_out(&[1., 0., 0., 1.], &z, 2, 2, 1, 1e-12).unwrap(),
            (vec![0., 0.], 2)
        );
        assert!(project_out(&[f64::NAN], &[1.], 1, 1, 1, 1e-12).is_err());
        assert!(project_out(&[], &z, 2, 1, 1, 1e-12).is_err());
    }
}
