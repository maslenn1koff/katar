extern crate std;

use super::fallback::ScalarFallback;
use crate::traits::{FloatSimd, MaskSimd, Simd};

const MAX_F32_LANES: usize = 8;
const MAX_F64_LANES: usize = 4;

fn f32_input<S: Simd>(base: f32, step: f32) -> [f32; MAX_F32_LANES] {
    let mut data = [0.0f32; MAX_F32_LANES];
    for (i, value) in data.iter_mut().enumerate().take(S::f32_lanes) {
        *value = base + step * i as f32;
    }
    data
}

fn f64_input<S: Simd>(base: f64, step: f64) -> [f64; MAX_F64_LANES] {
    let mut data = [0.0f64; MAX_F64_LANES];
    for (i, value) in data.iter_mut().enumerate().take(S::f64_lanes) {
        *value = base + step * i as f64;
    }
    data
}

fn store_f32<S: Simd>(simd: S, value: S::f32) -> [f32; MAX_F32_LANES] {
    let mut data = [0.0f32; MAX_F32_LANES];
    simd.store_f32(&mut data, value);
    data
}

fn store_f64<S: Simd>(simd: S, value: S::f64) -> [f64; MAX_F64_LANES] {
    let mut data = [0.0f64; MAX_F64_LANES];
    simd.store_f64(&mut data, value);
    data
}

fn check_binary_f32<S, F, G>(simd: S, a: &[f32; MAX_F32_LANES], b: &[f32; MAX_F32_LANES], op: F, scalar: G)
where
    S: Simd,
    F: Fn(S, S::f32, S::f32) -> S::f32,
    G: Fn(f32, f32) -> f32,
{
    let result = store_f32(simd, op(simd, simd.load_f32(a), simd.load_f32(b)));
    for i in 0..S::f32_lanes {
        assert_eq!(result[i], scalar(a[i], b[i]), "f32 binary op lane {i}");
    }
}

fn check_binary_f64<S, F, G>(simd: S, a: &[f64; MAX_F64_LANES], b: &[f64; MAX_F64_LANES], op: F, scalar: G)
where
    S: Simd,
    F: Fn(S, S::f64, S::f64) -> S::f64,
    G: Fn(f64, f64) -> f64,
{
    let result = store_f64(simd, op(simd, simd.load_f64(a), simd.load_f64(b)));
    for i in 0..S::f64_lanes {
        assert_eq!(result[i], scalar(a[i], b[i]), "f64 binary op lane {i}");
    }
}

fn check_bits_f32<S, F, G>(simd: S, a: &[f32; MAX_F32_LANES], b: &[f32; MAX_F32_LANES], op: F, scalar: G)
where
    S: Simd,
    F: Fn(S, S::f32, S::f32) -> S::f32,
    G: Fn(u32, u32) -> u32,
{
    let result = store_f32(simd, op(simd, simd.load_f32(a), simd.load_f32(b)));
    for i in 0..S::f32_lanes {
        assert_eq!(result[i].to_bits(), scalar(a[i].to_bits(), b[i].to_bits()), "f32 bit op lane {i}");
    }
}

fn check_bits_f64<S, F, G>(simd: S, a: &[f64; MAX_F64_LANES], b: &[f64; MAX_F64_LANES], op: F, scalar: G)
where
    S: Simd,
    F: Fn(S, S::f64, S::f64) -> S::f64,
    G: Fn(u64, u64) -> u64,
{
    let result = store_f64(simd, op(simd, simd.load_f64(a), simd.load_f64(b)));
    for i in 0..S::f64_lanes {
        assert_eq!(result[i].to_bits(), scalar(a[i].to_bits(), b[i].to_bits()), "f64 bit op lane {i}");
    }
}

fn check_ternary_f32<S, F, G>(simd: S, a: &[f32; MAX_F32_LANES], b: &[f32; MAX_F32_LANES], c: &[f32; MAX_F32_LANES], op: F, scalar: G)
where
    S: Simd,
    F: Fn(S, S::f32, S::f32, S::f32) -> S::f32,
    G: Fn(f32, f32, f32) -> f32,
{
    let result = store_f32(simd, op(simd, simd.load_f32(a), simd.load_f32(b), simd.load_f32(c)));
    for i in 0..S::f32_lanes {
        assert_eq!(result[i], scalar(a[i], b[i], c[i]), "f32 ternary op lane {i}");
    }
}

fn check_ternary_f64<S, F, G>(simd: S, a: &[f64; MAX_F64_LANES], b: &[f64; MAX_F64_LANES], c: &[f64; MAX_F64_LANES], op: F, scalar: G)
where
    S: Simd,
    F: Fn(S, S::f64, S::f64, S::f64) -> S::f64,
    G: Fn(f64, f64, f64) -> f64,
{
    let result = store_f64(simd, op(simd, simd.load_f64(a), simd.load_f64(b), simd.load_f64(c)));
    for i in 0..S::f64_lanes {
        assert_eq!(result[i], scalar(a[i], b[i], c[i]), "f64 ternary op lane {i}");
    }
}

fn check_compare_f32<S, F, G>(simd: S, a: &[f32; MAX_F32_LANES], b: &[f32; MAX_F32_LANES], op: F, scalar: G)
where
    S: Simd,
    F: Fn(S, S::f32, S::f32) -> S::m32,
    G: Fn(f32, f32) -> bool,
{
    let bits = op(simd, simd.load_f32(a), simd.load_f32(b)).to_bits();
    for i in 0..S::f32_lanes {
        assert_eq!((bits >> i) & 1, scalar(a[i], b[i]) as u64, "f32 compare lane {i}");
    }
}

fn check_compare_f64<S, F, G>(simd: S, a: &[f64; MAX_F64_LANES], b: &[f64; MAX_F64_LANES], op: F, scalar: G)
where
    S: Simd,
    F: Fn(S, S::f64, S::f64) -> S::m64,
    G: Fn(f64, f64) -> bool,
{
    let bits = op(simd, simd.load_f64(a), simd.load_f64(b)).to_bits();
    for i in 0..S::f64_lanes {
        assert_eq!((bits >> i) & 1, scalar(a[i], b[i]) as u64, "f64 compare lane {i}");
    }
}

fn check_mask_f32<S: Simd>() {
    let all = (1u64 << S::f32_lanes) - 1;
    let none = <S::m32 as MaskSimd>::from_bits(0);
    let every = <S::m32 as MaskSimd>::from_bits(all);

    assert!(!none.any());
    assert!(!none.all());
    assert!(every.any());
    assert!(every.all());

    assert_eq!(every.to_bits(), all);
    assert_eq!(none.to_bits(), 0);
    assert_eq!(<S::m32 as MaskSimd>::splat(true).to_bits(), all);
    assert_eq!(<S::m32 as MaskSimd>::splat(false).to_bits(), 0);

    assert_eq!(every.not().to_bits(), 0);
    assert_eq!(none.not().to_bits(), all);
    assert_eq!(every.and(every).to_bits(), all);
    assert_eq!(every.and(none).to_bits(), 0);
    assert_eq!(every.or(none).to_bits(), all);
    assert_eq!(none.or(none).to_bits(), 0);
}

fn check_mask_f64<S: Simd>() {
    let all = (1u64 << S::f64_lanes) - 1;
    let none = <S::m64 as MaskSimd>::from_bits(0);
    let every = <S::m64 as MaskSimd>::from_bits(all);

    assert!(!none.any());
    assert!(!none.all());
    assert!(every.any());
    assert!(every.all());

    assert_eq!(every.to_bits(), all);
    assert_eq!(none.to_bits(), 0);
    assert_eq!(<S::m64 as MaskSimd>::splat(true).to_bits(), all);
    assert_eq!(<S::m64 as MaskSimd>::splat(false).to_bits(), 0);

    assert_eq!(every.not().to_bits(), 0);
    assert_eq!(none.not().to_bits(), all);
    assert_eq!(every.and(every).to_bits(), all);
    assert_eq!(every.and(none).to_bits(), 0);
    assert_eq!(every.or(none).to_bits(), all);
    assert_eq!(none.or(none).to_bits(), 0);
}

fn exercise_f32<S: Simd>(simd: S) {
    let lanes = S::f32_lanes;

    let a = f32_input::<S>(1.0, 0.5);
    let b = f32_input::<S>(2.0, 0.25);

    let splatted = store_f32(simd, simd.splat_f32(3.5));
    for value in splatted.iter().take(lanes) {
        assert_eq!(*value, 3.5);
    }

    let roundtrip = store_f32(simd, simd.load_f32(&a));
    for i in 0..lanes {
        assert_eq!(roundtrip[i], a[i], "load/store lane {i}");
    }

    let negative = f32_input::<S>(-1.0, -0.5);
    let absolute = store_f32(simd, simd.abs_f32(simd.load_f32(&negative)));
    for i in 0..lanes {
        assert_eq!(absolute[i], negative[i].abs(), "abs lane {i}");
    }

    let negated = store_f32(simd, simd.neg_f32(simd.load_f32(&a)));
    for i in 0..lanes {
        assert_eq!(negated[i], -a[i], "neg lane {i}");
    }

    let mut squares = [0.0f32; MAX_F32_LANES];
    for (i, value) in squares.iter_mut().enumerate().take(lanes) {
        let v = (i + 1) as f32;
        *value = v * v;
    }
    let roots = store_f32(simd, simd.sqrt_f32(simd.load_f32(&squares)));
    for (i, root) in roots.iter().enumerate().take(lanes) {
        assert_eq!(*root, (i + 1) as f32, "sqrt lane {i}");
    }

    check_binary_f32(simd, &a, &b, |s, x, y| s.add_f32(x, y), |x, y| x + y);
    check_binary_f32(simd, &a, &b, |s, x, y| s.sub_f32(x, y), |x, y| x - y);
    check_binary_f32(simd, &a, &b, |s, x, y| s.mul_f32(x, y), |x, y| x * y);
    check_binary_f32(simd, &a, &b, |s, x, y| s.div_f32(x, y), |x, y| x / y);
    check_binary_f32(simd, &a, &b, |s, x, y| s.min_f32(x, y), |x, y| x.min(y));
    check_binary_f32(simd, &a, &b, |s, x, y| s.max_f32(x, y), |x, y| x.max(y));

    check_bits_f32(simd, &a, &b, |s, x, y| s.bitand_f32(x, y), |x, y| x & y);
    check_bits_f32(simd, &a, &b, |s, x, y| s.bitor_f32(x, y), |x, y| x | y);
    check_bits_f32(simd, &a, &b, |s, x, y| s.bitxor_f32(x, y), |x, y| x ^ y);
    check_bits_f32(simd, &a, &b, |s, x, y| s.bitandnot_f32(x, y), |x, y| !x & y);

    let inverted = store_f32(simd, simd.bitnot_f32(simd.load_f32(&a)));
    for i in 0..lanes {
        assert_eq!(inverted[i].to_bits(), !a[i].to_bits(), "bitnot lane {i}");
    }

    let mut ta = [0.0f32; MAX_F32_LANES];
    for (i, value) in ta.iter_mut().enumerate().take(lanes) {
        *value = (i + 1) as f32;
    }
    let tb = [2.0f32; MAX_F32_LANES];
    let tc = [3.0f32; MAX_F32_LANES];

    check_ternary_f32(simd, &ta, &tb, &tc, |s, x, y, z| s.mul_add_f32(x, y, z), |x, y, z| x * y + z);
    check_ternary_f32(simd, &ta, &tb, &tc, |s, x, y, z| s.mul_sub_f32(x, y, z), |x, y, z| x * y - z);
    check_ternary_f32(simd, &ta, &tb, &tc, |s, x, y, z| s.nmul_add_f32(x, y, z), |x, y, z| -(x * y) + z);
    check_ternary_f32(simd, &ta, &tb, &tc, |s, x, y, z| s.nmul_sub_f32(x, y, z), |x, y, z| -(x * y) - z);

    let ramp = f32_input::<S>(1.0, 1.0);
    let values = simd.load_f32(&ramp);
    assert_eq!(simd.reduce_sum_f32(values), (lanes * (lanes + 1) / 2) as f32);
    assert_eq!(simd.reduce_max_f32(values), lanes as f32);
    assert_eq!(simd.reduce_min_f32(values), 1.0);

    let cmp_a = f32_input::<S>(0.0, 1.0);
    let mut cmp_b = [0.0f32; MAX_F32_LANES];
    for value in cmp_b.iter_mut().take(lanes) {
        *value = (lanes / 2) as f32;
    }

    check_compare_f32(simd, &cmp_a, &cmp_b, |s, x, y| s.eq_f32(x, y), |x, y| x == y);
    check_compare_f32(simd, &cmp_a, &cmp_b, |s, x, y| s.ne_f32(x, y), |x, y| x != y);
    check_compare_f32(simd, &cmp_a, &cmp_b, |s, x, y| s.lt_f32(x, y), |x, y| x < y);
    check_compare_f32(simd, &cmp_a, &cmp_b, |s, x, y| s.le_f32(x, y), |x, y| x <= y);
    check_compare_f32(simd, &cmp_a, &cmp_b, |s, x, y| s.gt_f32(x, y), |x, y| x > y);
    check_compare_f32(simd, &cmp_a, &cmp_b, |s, x, y| s.ge_f32(x, y), |x, y| x >= y);

    let mask = simd.lt_f32(simd.load_f32(&cmp_a), simd.load_f32(&cmp_b));
    let selected = store_f32(simd, simd.select_f32(mask, simd.load_f32(&cmp_a), simd.load_f32(&cmp_b)));
    for i in 0..lanes {
        let expected = if cmp_a[i] < cmp_b[i] { cmp_a[i] } else { cmp_b[i] };
        assert_eq!(selected[i], expected, "select lane {i}");
    }
}

fn exercise_f64<S: Simd>(simd: S) {
    let lanes = S::f64_lanes;

    let a = f64_input::<S>(1.0, 0.5);
    let b = f64_input::<S>(2.0, 0.25);

    let splatted = store_f64(simd, simd.splat_f64(3.5));
    for value in splatted.iter().take(lanes) {
        assert_eq!(*value, 3.5);
    }

    let roundtrip = store_f64(simd, simd.load_f64(&a));
    for i in 0..lanes {
        assert_eq!(roundtrip[i], a[i], "load/store lane {i}");
    }

    let negative = f64_input::<S>(-1.0, -0.5);
    let absolute = store_f64(simd, simd.abs_f64(simd.load_f64(&negative)));
    for i in 0..lanes {
        assert_eq!(absolute[i], negative[i].abs(), "abs lane {i}");
    }

    let negated = store_f64(simd, simd.neg_f64(simd.load_f64(&a)));
    for i in 0..lanes {
        assert_eq!(negated[i], -a[i], "neg lane {i}");
    }

    let mut squares = [0.0f64; MAX_F64_LANES];
    for (i, value) in squares.iter_mut().enumerate().take(lanes) {
        let v = (i + 1) as f64;
        *value = v * v;
    }
    let roots = store_f64(simd, simd.sqrt_f64(simd.load_f64(&squares)));
    for (i, root) in roots.iter().enumerate().take(lanes) {
        assert_eq!(*root, (i + 1) as f64, "sqrt lane {i}");
    }

    check_binary_f64(simd, &a, &b, |s, x, y| s.add_f64(x, y), |x, y| x + y);
    check_binary_f64(simd, &a, &b, |s, x, y| s.sub_f64(x, y), |x, y| x - y);
    check_binary_f64(simd, &a, &b, |s, x, y| s.mul_f64(x, y), |x, y| x * y);
    check_binary_f64(simd, &a, &b, |s, x, y| s.div_f64(x, y), |x, y| x / y);
    check_binary_f64(simd, &a, &b, |s, x, y| s.min_f64(x, y), |x, y| x.min(y));
    check_binary_f64(simd, &a, &b, |s, x, y| s.max_f64(x, y), |x, y| x.max(y));

    check_bits_f64(simd, &a, &b, |s, x, y| s.bitand_f64(x, y), |x, y| x & y);
    check_bits_f64(simd, &a, &b, |s, x, y| s.bitor_f64(x, y), |x, y| x | y);
    check_bits_f64(simd, &a, &b, |s, x, y| s.bitxor_f64(x, y), |x, y| x ^ y);
    check_bits_f64(simd, &a, &b, |s, x, y| s.bitandnot_f64(x, y), |x, y| !x & y);

    let inverted = store_f64(simd, simd.bitnot_f64(simd.load_f64(&a)));
    for i in 0..lanes {
        assert_eq!(inverted[i].to_bits(), !a[i].to_bits(), "bitnot lane {i}");
    }

    let mut ta = [0.0f64; MAX_F64_LANES];
    for (i, value) in ta.iter_mut().enumerate().take(lanes) {
        *value = (i + 1) as f64;
    }
    let tb = [2.0f64; MAX_F64_LANES];
    let tc = [3.0f64; MAX_F64_LANES];

    check_ternary_f64(simd, &ta, &tb, &tc, |s, x, y, z| s.mul_add_f64(x, y, z), |x, y, z| x * y + z);
    check_ternary_f64(simd, &ta, &tb, &tc, |s, x, y, z| s.mul_sub_f64(x, y, z), |x, y, z| x * y - z);
    check_ternary_f64(simd, &ta, &tb, &tc, |s, x, y, z| s.nmul_add_f64(x, y, z), |x, y, z| -(x * y) + z);
    check_ternary_f64(simd, &ta, &tb, &tc, |s, x, y, z| s.nmul_sub_f64(x, y, z), |x, y, z| -(x * y) - z);

    let ramp = f64_input::<S>(1.0, 1.0);
    let values = simd.load_f64(&ramp);
    assert_eq!(simd.reduce_sum_f64(values), (lanes * (lanes + 1) / 2) as f64);
    assert_eq!(simd.reduce_max_f64(values), lanes as f64);
    assert_eq!(simd.reduce_min_f64(values), 1.0);

    let cmp_a = f64_input::<S>(0.0, 1.0);
    let mut cmp_b = [0.0f64; MAX_F64_LANES];
    for value in cmp_b.iter_mut().take(lanes) {
        *value = (lanes / 2) as f64;
    }

    check_compare_f64(simd, &cmp_a, &cmp_b, |s, x, y| s.eq_f64(x, y), |x, y| x == y);
    check_compare_f64(simd, &cmp_a, &cmp_b, |s, x, y| s.ne_f64(x, y), |x, y| x != y);
    check_compare_f64(simd, &cmp_a, &cmp_b, |s, x, y| s.lt_f64(x, y), |x, y| x < y);
    check_compare_f64(simd, &cmp_a, &cmp_b, |s, x, y| s.le_f64(x, y), |x, y| x <= y);
    check_compare_f64(simd, &cmp_a, &cmp_b, |s, x, y| s.gt_f64(x, y), |x, y| x > y);
    check_compare_f64(simd, &cmp_a, &cmp_b, |s, x, y| s.ge_f64(x, y), |x, y| x >= y);

    let mask = simd.lt_f64(simd.load_f64(&cmp_a), simd.load_f64(&cmp_b));
    let selected = store_f64(simd, simd.select_f64(mask, simd.load_f64(&cmp_a), simd.load_f64(&cmp_b)));
    for i in 0..lanes {
        let expected = if cmp_a[i] < cmp_b[i] { cmp_a[i] } else { cmp_b[i] };
        assert_eq!(selected[i], expected, "select lane {i}");
    }
}

#[test]
fn scalar_fallback_f32() {
    exercise_f32(ScalarFallback);
}

#[test]
fn scalar_fallback_f64() {
    exercise_f64(ScalarFallback);
}

#[test]
fn scalar_fallback_masks() {
    check_mask_f32::<ScalarFallback>();
    check_mask_f64::<ScalarFallback>();
}

#[test]
#[should_panic]
fn scalar_fallback_load_f32_panics_on_empty_slice() {
    ScalarFallback.load_f32(&[]);
}

#[test]
#[should_panic]
fn scalar_fallback_load_f64_panics_on_empty_slice() {
    ScalarFallback.load_f64(&[]);
}

#[test]
#[should_panic]
fn scalar_fallback_store_f32_panics_on_empty_slice() {
    let mut dst: [f32; 0] = [];
    ScalarFallback.store_f32(&mut dst, 1.0);
}

#[test]
#[should_panic]
fn scalar_fallback_store_f64_panics_on_empty_slice() {
    let mut dst: [f64; 0] = [];
    ScalarFallback.store_f64(&mut dst, 1.0);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn sse2_available() -> bool {
    core_detect::is_x86_feature_detected!("sse2")
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse2_f32() {
    if !sse2_available() {
        return;
    }
    exercise_f32(super::x86::sse2::Sse2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse2_f64() {
    if !sse2_available() {
        return;
    }
    exercise_f64(super::x86::sse2::Sse2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse2_masks() {
    if !sse2_available() {
        return;
    }
    check_mask_f32::<super::x86::sse2::Sse2>();
    check_mask_f64::<super::x86::sse2::Sse2>();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse2_load_f32_panics_on_short_slice() {
    if !sse2_available() {
        return;
    }

    let result = std::panic::catch_unwind(|| {
        let src = [1.0f32; 3];
        super::x86::sse2::Sse2.load_f32(&src);
    });

    assert!(result.is_err());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse2_load_f64_panics_on_short_slice() {
    if !sse2_available() {
        return;
    }

    let result = std::panic::catch_unwind(|| {
        let src = [1.0f64; 1];
        super::x86::sse2::Sse2.load_f64(&src);
    });

    assert!(result.is_err());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn avx2_fma_available() -> bool {
    core_detect::is_x86_feature_detected!("avx2") && core_detect::is_x86_feature_detected!("fma")
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx2_fma_f32() {
    if !avx2_fma_available() {
        return;
    }
    exercise_f32(super::x86::avx2::Avx2Fma);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx2_fma_f64() {
    if !avx2_fma_available() {
        return;
    }
    exercise_f64(super::x86::avx2::Avx2Fma);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx2_fma_masks() {
    if !avx2_fma_available() {
        return;
    }
    check_mask_f32::<super::x86::avx2::Avx2Fma>();
    check_mask_f64::<super::x86::avx2::Avx2Fma>();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx2_fma_load_f32_panics_on_short_slice() {
    if !avx2_fma_available() {
        return;
    }

    let result = std::panic::catch_unwind(|| {
        let src = [1.0f32; 7];
        super::x86::avx2::Avx2Fma.load_f32(&src);
    });

    assert!(result.is_err());
}
