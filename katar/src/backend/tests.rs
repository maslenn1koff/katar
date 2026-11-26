extern crate std;

use super::fallback::ScalarFallback;
use crate::traits::{FloatSimd, IntegerSimd, MaskSimd, Simd};

macro_rules! check_int_binary {
    ($simd:expr, $ty:ident, $lanes:expr, $a:expr, $b:expr, $op:ident, $scalar:expr) => {{
        paste::paste! {
            let scalar: fn($ty, $ty) -> $ty = $scalar;
            let mut out = $a;
            $simd.[<store_ $ty>](
                &mut out,
                $simd.[<$op _ $ty>]($simd.[<load_ $ty>](&$a), $simd.[<load_ $ty>](&$b)),
            );
            for i in 0..$lanes {
                assert_eq!(
                    out[i],
                    scalar($a[i], $b[i]),
                    concat!(stringify!($op), "_", stringify!($ty), " lane {}"),
                    i
                );
            }
        }
    }};
}

macro_rules! check_int_unary {
    ($simd:expr, $ty:ident, $lanes:expr, $a:expr, $op:ident, $scalar:expr) => {{
        paste::paste! {
            let scalar: fn($ty) -> $ty = $scalar;
            let mut out = $a;
            $simd.[<store_ $ty>](&mut out, $simd.[<$op _ $ty>]($simd.[<load_ $ty>](&$a)));
            for i in 0..$lanes {
                assert_eq!(
                    out[i],
                    scalar($a[i]),
                    concat!(stringify!($op), "_", stringify!($ty), " lane {}"),
                    i
                );
            }
        }
    }};
}

macro_rules! check_int_compare {
    ($simd:expr, $ty:ident, $lanes:expr, $a:expr, $b:expr, $op:ident, $scalar:expr) => {{
        paste::paste! {
            let scalar: fn($ty, $ty) -> bool = $scalar;
            let mut bits = $simd.[<$op _ $ty>]($simd.[<load_ $ty>](&$a), $simd.[<load_ $ty>](&$b)).to_bits();
            for i in 0..$lanes {
                assert_eq!(
                    bits & 1,
                    scalar($a[i], $b[i]) as u64,
                    concat!(stringify!($op), "_", stringify!($ty), " lane {}"),
                    i
                );
                bits >>= 1;
            }
        }
    }};
}

macro_rules! check_int_shift {
    ($simd:expr, $ty:ident, $lanes:expr, $a:expr, $op:ident, $n:expr, $scalar:expr) => {{
        paste::paste! {
            let scalar: fn($ty) -> $ty = $scalar;
            let mut out = $a;
            $simd.[<store_ $ty>](&mut out, $simd.[<$op _ $ty>]($simd.[<load_ $ty>](&$a), $n));
            for i in 0..$lanes {
                assert_eq!(
                    out[i],
                    scalar($a[i]),
                    concat!(stringify!($op), "_", stringify!($ty), " lane {}"),
                    i
                );
            }
        }
    }};
}

macro_rules! check_int_select {
    ($simd:expr, $ty:ident, $lanes:expr, $a:expr, $b:expr) => {{
        paste::paste! {
            let mask = $simd.[<lt_ $ty>]($simd.[<load_ $ty>](&$a), $simd.[<load_ $ty>](&$b));
            let mut out = $a;
            $simd.[<store_ $ty>](
                &mut out,
                $simd.[<select_ $ty>](mask, $simd.[<load_ $ty>](&$a), $simd.[<load_ $ty>](&$b)),
            );
            for i in 0..$lanes {
                let expected = if $a[i] < $b[i] { $a[i] } else { $b[i] };
                assert_eq!(out[i], expected, concat!("select_", stringify!($ty), " lane {}"), i);
            }
        }
    }};
}

const MAX_F32_LANES: usize = 16;
const MAX_F64_LANES: usize = 8;

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
    F: Fn(S, S::f32, S::f32) -> <S as FloatSimd>::m32,
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
    F: Fn(S, S::f64, S::f64) -> <S as FloatSimd>::m64,
    G: Fn(f64, f64) -> bool,
{
    let bits = op(simd, simd.load_f64(a), simd.load_f64(b)).to_bits();
    for i in 0..S::f64_lanes {
        assert_eq!((bits >> i) & 1, scalar(a[i], b[i]) as u64, "f64 compare lane {i}");
    }
}

fn check_mask_f32<S: Simd>() {
    let all = (1u64 << S::f32_lanes) - 1;
    let none = <<S as FloatSimd>::m32 as MaskSimd>::from_bits(0);
    let every = <<S as FloatSimd>::m32 as MaskSimd>::from_bits(all);

    assert!(!none.any());
    assert!(!none.all());
    assert!(every.any());
    assert!(every.all());

    assert_eq!(every.to_bits(), all);
    assert_eq!(none.to_bits(), 0);
    assert_eq!(<<S as FloatSimd>::m32 as MaskSimd>::splat(true).to_bits(), all);
    assert_eq!(<<S as FloatSimd>::m32 as MaskSimd>::splat(false).to_bits(), 0);

    assert_eq!(every.not().to_bits(), 0);
    assert_eq!(none.not().to_bits(), all);
    assert_eq!(every.and(every).to_bits(), all);
    assert_eq!(every.and(none).to_bits(), 0);
    assert_eq!(every.or(none).to_bits(), all);
    assert_eq!(none.or(none).to_bits(), 0);
}

fn check_mask_f64<S: Simd>() {
    let all = (1u64 << S::f64_lanes) - 1;
    let none = <<S as FloatSimd>::m64 as MaskSimd>::from_bits(0);
    let every = <<S as FloatSimd>::m64 as MaskSimd>::from_bits(all);

    assert!(!none.any());
    assert!(!none.all());
    assert!(every.any());
    assert!(every.all());

    assert_eq!(every.to_bits(), all);
    assert_eq!(none.to_bits(), 0);
    assert_eq!(<<S as FloatSimd>::m64 as MaskSimd>::splat(true).to_bits(), all);
    assert_eq!(<<S as FloatSimd>::m64 as MaskSimd>::splat(false).to_bits(), 0);

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
fn avx512f_available() -> bool {
    core_detect::is_x86_feature_detected!("avx512f")
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512f_f32() {
    if !avx512f_available() {
        return;
    }
    exercise_f32(super::x86::avx512::Avx512Software::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512f_f64() {
    if !avx512f_available() {
        return;
    }
    exercise_f64(super::x86::avx512::Avx512Software::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512f_masks() {
    if !avx512f_available() {
        return;
    }
    check_mask_f32::<super::x86::avx512::Avx512Software>();
    check_mask_f64::<super::x86::avx512::Avx512Software>();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512f_load_f32_panics_on_short_slice() {
    if !avx512f_available() {
        return;
    }

    let result = std::panic::catch_unwind(|| {
        let src = [1.0f32; 15];
        super::x86::avx512::Avx512Software::new().load_f32(&src);
    });

    assert!(result.is_err());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512f_load_f64_panics_on_short_slice() {
    if !avx512f_available() {
        return;
    }

    let result = std::panic::catch_unwind(|| {
        let src = [1.0f64; 7];
        super::x86::avx512::Avx512Software::new().load_f64(&src);
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

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_f32() {
    exercise_f32(super::aarch64::neon::Neon);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_f64() {
    exercise_f64(super::aarch64::neon::Neon);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_masks() {
    check_mask_f32::<super::aarch64::neon::Neon>();
    check_mask_f64::<super::aarch64::neon::Neon>();
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_load_f32_panics_on_short_slice() {
    let result = std::panic::catch_unwind(|| {
        let src = [1.0f32; 3];
        super::aarch64::neon::Neon.load_f32(&src);
    });

    assert!(result.is_err());
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_load_f64_panics_on_short_slice() {
    let result = std::panic::catch_unwind(|| {
        let src = [1.0f64; 1];
        super::aarch64::neon::Neon.load_f64(&src);
    });

    assert!(result.is_err());
}

const MAX_U8_LANES: usize = 64;
const MAX_U16_LANES: usize = 32;
const MAX_U32_LANES: usize = 16;
const MAX_U64_LANES: usize = 8;

fn store_u8<S: IntegerSimd>(simd: S, value: S::u8) -> [u8; MAX_U8_LANES] {
    let mut data = [0u8; MAX_U8_LANES];
    simd.store_u8(&mut data, value);
    data
}

fn store_i8<S: IntegerSimd>(simd: S, value: S::i8) -> [i8; MAX_U8_LANES] {
    let mut data = [0i8; MAX_U8_LANES];
    simd.store_i8(&mut data, value);
    data
}

fn store_u16<S: IntegerSimd>(simd: S, value: S::u16) -> [u16; MAX_U16_LANES] {
    let mut data = [0u16; MAX_U16_LANES];
    simd.store_u16(&mut data, value);
    data
}

fn store_i16<S: IntegerSimd>(simd: S, value: S::i16) -> [i16; MAX_U16_LANES] {
    let mut data = [0i16; MAX_U16_LANES];
    simd.store_i16(&mut data, value);
    data
}

fn store_u32<S: IntegerSimd>(simd: S, value: S::u32) -> [u32; MAX_U32_LANES] {
    let mut data = [0u32; MAX_U32_LANES];
    simd.store_u32(&mut data, value);
    data
}

fn store_i32<S: IntegerSimd>(simd: S, value: S::i32) -> [i32; MAX_U32_LANES] {
    let mut data = [0i32; MAX_U32_LANES];
    simd.store_i32(&mut data, value);
    data
}

fn store_u64<S: IntegerSimd>(simd: S, value: S::u64) -> [u64; MAX_U64_LANES] {
    let mut data = [0u64; MAX_U64_LANES];
    simd.store_u64(&mut data, value);
    data
}

fn store_i64<S: IntegerSimd>(simd: S, value: S::i64) -> [i64; MAX_U64_LANES] {
    let mut data = [0i64; MAX_U64_LANES];
    simd.store_i64(&mut data, value);
    data
}

fn exercise_int_u8<S: IntegerSimd>(simd: S) {
    let lanes = S::u8_lanes;
    let mut a = [0u8; MAX_U8_LANES];
    let mut b = [0u8; MAX_U8_LANES];
    for i in 0..MAX_U8_LANES {
        a[i] = (i as u8).wrapping_mul(17);
        b[i] = 255u8.wrapping_sub((i as u8).wrapping_mul(11));
    }
    a[0] = 0;
    b[0] = 255;

    let mut roundtrip = a;
    simd.store_u8(&mut roundtrip, simd.load_u8(&a));
    assert_eq!(roundtrip, a);

    let mut splat = [0u8; MAX_U8_LANES];
    simd.store_u8(&mut splat, simd.splat_u8(a[1]));
    for value in splat.iter().take(lanes) {
        assert_eq!(*value, a[1]);
    }

    check_int_binary!(simd, u8, lanes, a, b, add, |x, y| x.wrapping_add(y));
    check_int_binary!(simd, u8, lanes, a, b, sub, |x, y| x.wrapping_sub(y));
    check_int_binary!(simd, u8, lanes, a, b, mul, |x, y| x.wrapping_mul(y));
    check_int_binary!(simd, u8, lanes, a, b, min, |x, y| x.min(y));
    check_int_binary!(simd, u8, lanes, a, b, max, |x, y| x.max(y));
    check_int_binary!(simd, u8, lanes, a, b, bitand, |x, y| x & y);
    check_int_binary!(simd, u8, lanes, a, b, bitor, |x, y| x | y);
    check_int_binary!(simd, u8, lanes, a, b, bitxor, |x, y| x ^ y);
    check_int_binary!(simd, u8, lanes, a, b, bitandnot, |x, y| !x & y);
    check_int_binary!(simd, u8, lanes, a, b, saturating_add, |x, y| x.saturating_add(y));
    check_int_binary!(simd, u8, lanes, a, b, saturating_sub, |x, y| x.saturating_sub(y));
    check_int_binary!(simd, u8, lanes, a, b, avg_round, |x, y| ((x as u16 + y as u16 + 1) >> 1) as u8);

    check_int_unary!(simd, u8, lanes, a, bitnot, |x| !x);
    check_int_unary!(simd, u8, lanes, a, count_ones, |x| x.count_ones() as u8);
    check_int_unary!(simd, u8, lanes, a, leading_zeros, |x| x.leading_zeros() as u8);
    check_int_unary!(simd, u8, lanes, a, trailing_zeros, |x| x.trailing_zeros() as u8);

    check_int_shift!(simd, u8, lanes, a, shl, 3, |x| x.wrapping_shl(3));
    check_int_shift!(simd, u8, lanes, a, shr, 3, |x| x.wrapping_shr(3));
    check_int_shift!(simd, u8, lanes, a, shl, 8, |x| x.wrapping_shl(8));
    check_int_shift!(simd, u8, lanes, a, shr, 8, |x| x.wrapping_shr(8));

    check_int_compare!(simd, u8, lanes, a, b, eq, |x, y| x == y);
    check_int_compare!(simd, u8, lanes, a, b, ne, |x, y| x != y);
    check_int_compare!(simd, u8, lanes, a, b, lt, |x, y| x < y);
    check_int_compare!(simd, u8, lanes, a, b, le, |x, y| x <= y);
    check_int_compare!(simd, u8, lanes, a, b, gt, |x, y| x > y);
    check_int_compare!(simd, u8, lanes, a, b, ge, |x, y| x >= y);
    check_int_select!(simd, u8, lanes, a, b);

    if S::u8_lanes == S::u16_lanes * 2 {
        let mut wide = store_u16(simd, simd.widen_low_u8_to_u16(simd.load_u8(&a)));
        for i in 0..S::u16_lanes {
            assert_eq!(wide[i], a[i] as u16, "widen_low_u8_to_u16 lane {i}");
        }
        wide = store_u16(simd, simd.widen_high_u8_to_u16(simd.load_u8(&a)));
        for i in 0..S::u16_lanes {
            assert_eq!(wide[i], a[i + S::u16_lanes] as u16, "widen_high_u8_to_u16 lane {i}");
        }
    }
}

fn exercise_int_i8<S: IntegerSimd>(simd: S) {
    let lanes = S::i8_lanes;
    let mut a = [0i8; MAX_U8_LANES];
    let mut b = [0i8; MAX_U8_LANES];
    for i in 0..MAX_U8_LANES {
        a[i] = (i as i8).wrapping_mul(17).wrapping_sub(64);
        b[i] = (i as i8).wrapping_mul(13).wrapping_add(50);
    }
    a[0] = i8::MIN;
    b[0] = i8::MAX;
    a[1] = 0;

    let mut roundtrip = a;
    simd.store_i8(&mut roundtrip, simd.load_i8(&a));
    assert_eq!(roundtrip, a);

    let mut splat = [0i8; MAX_U8_LANES];
    simd.store_i8(&mut splat, simd.splat_i8(a[1]));
    for value in splat.iter().take(lanes) {
        assert_eq!(*value, a[1]);
    }

    check_int_binary!(simd, i8, lanes, a, b, add, |x, y| x.wrapping_add(y));
    check_int_binary!(simd, i8, lanes, a, b, sub, |x, y| x.wrapping_sub(y));
    check_int_binary!(simd, i8, lanes, a, b, mul, |x, y| x.wrapping_mul(y));
    check_int_binary!(simd, i8, lanes, a, b, min, |x, y| x.min(y));
    check_int_binary!(simd, i8, lanes, a, b, max, |x, y| x.max(y));
    check_int_binary!(simd, i8, lanes, a, b, bitand, |x, y| x & y);
    check_int_binary!(simd, i8, lanes, a, b, bitor, |x, y| x | y);
    check_int_binary!(simd, i8, lanes, a, b, bitxor, |x, y| x ^ y);
    check_int_binary!(simd, i8, lanes, a, b, bitandnot, |x, y| !x & y);
    check_int_binary!(simd, i8, lanes, a, b, saturating_add, |x, y| x.saturating_add(y));
    check_int_binary!(simd, i8, lanes, a, b, saturating_sub, |x, y| x.saturating_sub(y));
    check_int_binary!(simd, i8, lanes, a, b, avg_round, |x, y| ((x as i16 + y as i16 + 1) >> 1) as i8);

    check_int_unary!(simd, i8, lanes, a, bitnot, |x| !x);
    check_int_unary!(simd, i8, lanes, a, abs, |x| x.wrapping_abs());
    check_int_unary!(simd, i8, lanes, a, neg, |x| x.wrapping_neg());
    check_int_unary!(simd, i8, lanes, a, count_ones, |x| x.count_ones() as i8);
    check_int_unary!(simd, i8, lanes, a, leading_zeros, |x| x.leading_zeros() as i8);
    check_int_unary!(simd, i8, lanes, a, trailing_zeros, |x| x.trailing_zeros() as i8);

    check_int_shift!(simd, i8, lanes, a, shl, 3, |x| x.wrapping_shl(3));
    check_int_shift!(simd, i8, lanes, a, shr, 3, |x| x.wrapping_shr(3));
    check_int_shift!(simd, i8, lanes, a, shl, 8, |x| x.wrapping_shl(8));
    check_int_shift!(simd, i8, lanes, a, shr, 8, |x| x.wrapping_shr(8));

    check_int_compare!(simd, i8, lanes, a, b, eq, |x, y| x == y);
    check_int_compare!(simd, i8, lanes, a, b, ne, |x, y| x != y);
    check_int_compare!(simd, i8, lanes, a, b, lt, |x, y| x < y);
    check_int_compare!(simd, i8, lanes, a, b, le, |x, y| x <= y);
    check_int_compare!(simd, i8, lanes, a, b, gt, |x, y| x > y);
    check_int_compare!(simd, i8, lanes, a, b, ge, |x, y| x >= y);
    check_int_select!(simd, i8, lanes, a, b);

    if S::i8_lanes == S::i16_lanes * 2 {
        let mut wide = store_i16(simd, simd.widen_low_i8_to_i16(simd.load_i8(&a)));
        for i in 0..S::i16_lanes {
            assert_eq!(wide[i], a[i] as i16, "widen_low_i8_to_i16 lane {i}");
        }
        wide = store_i16(simd, simd.widen_high_i8_to_i16(simd.load_i8(&a)));
        for i in 0..S::i16_lanes {
            assert_eq!(wide[i], a[i + S::i16_lanes] as i16, "widen_high_i8_to_i16 lane {i}");
        }
    }
}

fn exercise_int_u16<S: IntegerSimd>(simd: S) {
    let lanes = S::u16_lanes;
    let mut a = [0u16; MAX_U16_LANES];
    let mut b = [0u16; MAX_U16_LANES];
    for i in 0..MAX_U16_LANES {
        a[i] = (i as u16).wrapping_mul(2053);
        b[i] = (i as u16).wrapping_mul(4099).wrapping_add(12345);
    }
    a[0] = 0;
    b[0] = u16::MAX;

    let mut roundtrip = a;
    simd.store_u16(&mut roundtrip, simd.load_u16(&a));
    assert_eq!(roundtrip, a);

    let mut splat = [0u16; MAX_U16_LANES];
    simd.store_u16(&mut splat, simd.splat_u16(a[1]));
    for value in splat.iter().take(lanes) {
        assert_eq!(*value, a[1]);
    }

    check_int_binary!(simd, u16, lanes, a, b, add, |x, y| x.wrapping_add(y));
    check_int_binary!(simd, u16, lanes, a, b, sub, |x, y| x.wrapping_sub(y));
    check_int_binary!(simd, u16, lanes, a, b, mul, |x, y| x.wrapping_mul(y));
    check_int_binary!(simd, u16, lanes, a, b, min, |x, y| x.min(y));
    check_int_binary!(simd, u16, lanes, a, b, max, |x, y| x.max(y));
    check_int_binary!(simd, u16, lanes, a, b, bitand, |x, y| x & y);
    check_int_binary!(simd, u16, lanes, a, b, bitor, |x, y| x | y);
    check_int_binary!(simd, u16, lanes, a, b, bitxor, |x, y| x ^ y);
    check_int_binary!(simd, u16, lanes, a, b, bitandnot, |x, y| !x & y);
    check_int_binary!(simd, u16, lanes, a, b, saturating_add, |x, y| x.saturating_add(y));
    check_int_binary!(simd, u16, lanes, a, b, saturating_sub, |x, y| x.saturating_sub(y));
    check_int_binary!(simd, u16, lanes, a, b, avg_round, |x, y| ((x as u32 + y as u32 + 1) >> 1) as u16);

    check_int_unary!(simd, u16, lanes, a, bitnot, |x| !x);
    check_int_unary!(simd, u16, lanes, a, count_ones, |x| x.count_ones() as u16);
    check_int_unary!(simd, u16, lanes, a, leading_zeros, |x| x.leading_zeros() as u16);
    check_int_unary!(simd, u16, lanes, a, trailing_zeros, |x| x.trailing_zeros() as u16);

    check_int_shift!(simd, u16, lanes, a, shl, 3, |x| x.wrapping_shl(3));
    check_int_shift!(simd, u16, lanes, a, shr, 3, |x| x.wrapping_shr(3));
    check_int_shift!(simd, u16, lanes, a, shl, 16, |x| x.wrapping_shl(16));
    check_int_shift!(simd, u16, lanes, a, shr, 16, |x| x.wrapping_shr(16));

    check_int_compare!(simd, u16, lanes, a, b, eq, |x, y| x == y);
    check_int_compare!(simd, u16, lanes, a, b, ne, |x, y| x != y);
    check_int_compare!(simd, u16, lanes, a, b, lt, |x, y| x < y);
    check_int_compare!(simd, u16, lanes, a, b, le, |x, y| x <= y);
    check_int_compare!(simd, u16, lanes, a, b, gt, |x, y| x > y);
    check_int_compare!(simd, u16, lanes, a, b, ge, |x, y| x >= y);
    check_int_select!(simd, u16, lanes, a, b);

    if S::u16_lanes == S::u32_lanes * 2 {
        let mut wide = store_u32(simd, simd.widen_low_u16_to_u32(simd.load_u16(&a)));
        for i in 0..S::u32_lanes {
            assert_eq!(wide[i], a[i] as u32, "widen_low_u16_to_u32 lane {i}");
        }
        wide = store_u32(simd, simd.widen_high_u16_to_u32(simd.load_u16(&a)));
        for i in 0..S::u32_lanes {
            assert_eq!(wide[i], a[i + S::u32_lanes] as u32, "widen_high_u16_to_u32 lane {i}");
        }
    }

    if S::u8_lanes == S::u16_lanes * 2 {
        let mut lo = [0u16; MAX_U16_LANES];
        let mut hi = [0u16; MAX_U16_LANES];
        for i in 0..S::u16_lanes {
            lo[i] = (i as u16).wrapping_mul(1000);
            hi[i] = u16::MAX.wrapping_sub((i as u16).wrapping_mul(333));
        }
        let out = store_u8(simd, simd.narrow_saturating_u16_to_u8(simd.load_u16(&lo), simd.load_u16(&hi)));
        for i in 0..S::u16_lanes {
            assert_eq!(out[i], lo[i].min(u8::MAX as u16) as u8, "narrow_u16_to_u8 lo lane {i}");
            assert_eq!(out[i + S::u16_lanes], hi[i].min(u8::MAX as u16) as u8, "narrow_u16_to_u8 hi lane {i}");
        }
    }
}

fn exercise_int_i16<S: IntegerSimd>(simd: S) {
    let lanes = S::i16_lanes;
    let mut a = [0i16; MAX_U16_LANES];
    let mut b = [0i16; MAX_U16_LANES];
    for i in 0..MAX_U16_LANES {
        a[i] = (i as i16).wrapping_mul(2053).wrapping_sub(16000);
        b[i] = (i as i16).wrapping_mul(4099).wrapping_add(7000);
    }
    a[0] = i16::MIN;
    b[0] = i16::MAX;
    a[1] = 0;

    let mut roundtrip = a;
    simd.store_i16(&mut roundtrip, simd.load_i16(&a));
    assert_eq!(roundtrip, a);

    let mut splat = [0i16; MAX_U16_LANES];
    simd.store_i16(&mut splat, simd.splat_i16(a[1]));
    for value in splat.iter().take(lanes) {
        assert_eq!(*value, a[1]);
    }

    check_int_binary!(simd, i16, lanes, a, b, add, |x, y| x.wrapping_add(y));
    check_int_binary!(simd, i16, lanes, a, b, sub, |x, y| x.wrapping_sub(y));
    check_int_binary!(simd, i16, lanes, a, b, mul, |x, y| x.wrapping_mul(y));
    check_int_binary!(simd, i16, lanes, a, b, min, |x, y| x.min(y));
    check_int_binary!(simd, i16, lanes, a, b, max, |x, y| x.max(y));
    check_int_binary!(simd, i16, lanes, a, b, bitand, |x, y| x & y);
    check_int_binary!(simd, i16, lanes, a, b, bitor, |x, y| x | y);
    check_int_binary!(simd, i16, lanes, a, b, bitxor, |x, y| x ^ y);
    check_int_binary!(simd, i16, lanes, a, b, bitandnot, |x, y| !x & y);
    check_int_binary!(simd, i16, lanes, a, b, saturating_add, |x, y| x.saturating_add(y));
    check_int_binary!(simd, i16, lanes, a, b, saturating_sub, |x, y| x.saturating_sub(y));
    check_int_binary!(simd, i16, lanes, a, b, avg_round, |x, y| ((x as i32 + y as i32 + 1) >> 1) as i16);

    check_int_unary!(simd, i16, lanes, a, bitnot, |x| !x);
    check_int_unary!(simd, i16, lanes, a, abs, |x| x.wrapping_abs());
    check_int_unary!(simd, i16, lanes, a, neg, |x| x.wrapping_neg());
    check_int_unary!(simd, i16, lanes, a, count_ones, |x| x.count_ones() as i16);
    check_int_unary!(simd, i16, lanes, a, leading_zeros, |x| x.leading_zeros() as i16);
    check_int_unary!(simd, i16, lanes, a, trailing_zeros, |x| x.trailing_zeros() as i16);

    check_int_shift!(simd, i16, lanes, a, shl, 3, |x| x.wrapping_shl(3));
    check_int_shift!(simd, i16, lanes, a, shr, 3, |x| x.wrapping_shr(3));
    check_int_shift!(simd, i16, lanes, a, shl, 16, |x| x.wrapping_shl(16));
    check_int_shift!(simd, i16, lanes, a, shr, 16, |x| x.wrapping_shr(16));

    check_int_compare!(simd, i16, lanes, a, b, eq, |x, y| x == y);
    check_int_compare!(simd, i16, lanes, a, b, ne, |x, y| x != y);
    check_int_compare!(simd, i16, lanes, a, b, lt, |x, y| x < y);
    check_int_compare!(simd, i16, lanes, a, b, le, |x, y| x <= y);
    check_int_compare!(simd, i16, lanes, a, b, gt, |x, y| x > y);
    check_int_compare!(simd, i16, lanes, a, b, ge, |x, y| x >= y);
    check_int_select!(simd, i16, lanes, a, b);

    if S::i16_lanes == S::i32_lanes * 2 {
        let mut wide = store_i32(simd, simd.widen_low_i16_to_i32(simd.load_i16(&a)));
        for i in 0..S::i32_lanes {
            assert_eq!(wide[i], a[i] as i32, "widen_low_i16_to_i32 lane {i}");
        }
        wide = store_i32(simd, simd.widen_high_i16_to_i32(simd.load_i16(&a)));
        for i in 0..S::i32_lanes {
            assert_eq!(wide[i], a[i + S::i32_lanes] as i32, "widen_high_i16_to_i32 lane {i}");
        }
    }

    if S::i8_lanes == S::i16_lanes * 2 {
        let mut lo = [0i16; MAX_U16_LANES];
        let mut hi = [0i16; MAX_U16_LANES];
        for i in 0..S::i16_lanes {
            lo[i] = (i as i16).wrapping_mul(3000).wrapping_sub(16000);
            hi[i] = 16000i16.wrapping_sub((i as i16).wrapping_mul(2500));
        }
        let out = store_i8(simd, simd.narrow_saturating_i16_to_i8(simd.load_i16(&lo), simd.load_i16(&hi)));
        for i in 0..S::i16_lanes {
            assert_eq!(out[i], lo[i].clamp(i8::MIN as i16, i8::MAX as i16) as i8, "narrow_i16_to_i8 lo lane {i}");
            assert_eq!(out[i + S::i16_lanes], hi[i].clamp(i8::MIN as i16, i8::MAX as i16) as i8, "narrow_i16_to_i8 hi lane {i}");
        }
    }
}

fn exercise_int_u32<S: IntegerSimd>(simd: S) {
    let lanes = S::u32_lanes;
    let mut a = [0u32; MAX_U32_LANES];
    let mut b = [0u32; MAX_U32_LANES];
    for i in 0..MAX_U32_LANES {
        a[i] = (i as u32).wrapping_mul(0x9E37_79B9);
        b[i] = (i as u32).wrapping_mul(0x85EB_CA6B).wrapping_add(0x1234_5678);
    }
    a[0] = 0;
    b[0] = u32::MAX;

    let mut roundtrip = a;
    simd.store_u32(&mut roundtrip, simd.load_u32(&a));
    assert_eq!(roundtrip, a);

    let mut splat = [0u32; MAX_U32_LANES];
    simd.store_u32(&mut splat, simd.splat_u32(a[1]));
    for value in splat.iter().take(lanes) {
        assert_eq!(*value, a[1]);
    }

    check_int_binary!(simd, u32, lanes, a, b, add, |x, y| x.wrapping_add(y));
    check_int_binary!(simd, u32, lanes, a, b, sub, |x, y| x.wrapping_sub(y));
    check_int_binary!(simd, u32, lanes, a, b, mul, |x, y| x.wrapping_mul(y));
    check_int_binary!(simd, u32, lanes, a, b, min, |x, y| x.min(y));
    check_int_binary!(simd, u32, lanes, a, b, max, |x, y| x.max(y));
    check_int_binary!(simd, u32, lanes, a, b, bitand, |x, y| x & y);
    check_int_binary!(simd, u32, lanes, a, b, bitor, |x, y| x | y);
    check_int_binary!(simd, u32, lanes, a, b, bitxor, |x, y| x ^ y);
    check_int_binary!(simd, u32, lanes, a, b, bitandnot, |x, y| !x & y);
    check_int_binary!(simd, u32, lanes, a, b, saturating_add, |x, y| x.saturating_add(y));
    check_int_binary!(simd, u32, lanes, a, b, saturating_sub, |x, y| x.saturating_sub(y));
    check_int_binary!(simd, u32, lanes, a, b, avg_round, |x, y| ((x as u64 + y as u64 + 1) >> 1) as u32);

    check_int_unary!(simd, u32, lanes, a, bitnot, |x| !x);
    check_int_unary!(simd, u32, lanes, a, count_ones, |x| x.count_ones());
    check_int_unary!(simd, u32, lanes, a, leading_zeros, |x| x.leading_zeros());
    check_int_unary!(simd, u32, lanes, a, trailing_zeros, |x| x.trailing_zeros());

    check_int_shift!(simd, u32, lanes, a, shl, 3, |x| x.wrapping_shl(3));
    check_int_shift!(simd, u32, lanes, a, shr, 3, |x| x.wrapping_shr(3));
    check_int_shift!(simd, u32, lanes, a, shl, 32, |x| x.wrapping_shl(32));
    check_int_shift!(simd, u32, lanes, a, shr, 32, |x| x.wrapping_shr(32));

    check_int_compare!(simd, u32, lanes, a, b, eq, |x, y| x == y);
    check_int_compare!(simd, u32, lanes, a, b, ne, |x, y| x != y);
    check_int_compare!(simd, u32, lanes, a, b, lt, |x, y| x < y);
    check_int_compare!(simd, u32, lanes, a, b, le, |x, y| x <= y);
    check_int_compare!(simd, u32, lanes, a, b, gt, |x, y| x > y);
    check_int_compare!(simd, u32, lanes, a, b, ge, |x, y| x >= y);
    check_int_select!(simd, u32, lanes, a, b);

    if S::u32_lanes == S::u64_lanes * 2 {
        let mut wide = store_u64(simd, simd.widen_low_u32_to_u64(simd.load_u32(&a)));
        for i in 0..S::u64_lanes {
            assert_eq!(wide[i], a[i] as u64, "widen_low_u32_to_u64 lane {i}");
        }
        wide = store_u64(simd, simd.widen_high_u32_to_u64(simd.load_u32(&a)));
        for i in 0..S::u64_lanes {
            assert_eq!(wide[i], a[i + S::u64_lanes] as u64, "widen_high_u32_to_u64 lane {i}");
        }

        let mut lo = [0u64; MAX_U64_LANES];
        let mut hi = [0u64; MAX_U64_LANES];
        for i in 0..S::u64_lanes {
            lo[i] = (i as u64).wrapping_mul(0x1234_5678_9ABC_DEF0);
            hi[i] = u64::MAX.wrapping_sub(i as u64);
        }
        let out = store_u32(simd, simd.narrow_saturating_u64_to_u32(simd.load_u64(&lo), simd.load_u64(&hi)));
        for i in 0..S::u64_lanes {
            assert_eq!(out[i], lo[i].min(u32::MAX as u64) as u32, "narrow_u64_to_u32 lo lane {i}");
            assert_eq!(out[i + S::u64_lanes], hi[i].min(u32::MAX as u64) as u32, "narrow_u64_to_u32 hi lane {i}");
        }
    }
}

fn exercise_int_i32<S: IntegerSimd>(simd: S) {
    let lanes = S::i32_lanes;
    let mut a = [0i32; MAX_U32_LANES];
    let mut b = [0i32; MAX_U32_LANES];
    for i in 0..MAX_U32_LANES {
        a[i] = (i as i32).wrapping_mul(0x9E37_79B9u32 as i32).wrapping_sub(0x4000_0000);
        b[i] = (i as i32).wrapping_mul(0x85EB_CA6Bu32 as i32).wrapping_add(0x1234_5678);
    }
    a[0] = i32::MIN;
    b[0] = i32::MAX;
    a[1] = 0;

    let mut roundtrip = a;
    simd.store_i32(&mut roundtrip, simd.load_i32(&a));
    assert_eq!(roundtrip, a);

    let mut splat = [0i32; MAX_U32_LANES];
    simd.store_i32(&mut splat, simd.splat_i32(a[1]));
    for value in splat.iter().take(lanes) {
        assert_eq!(*value, a[1]);
    }

    check_int_binary!(simd, i32, lanes, a, b, add, |x, y| x.wrapping_add(y));
    check_int_binary!(simd, i32, lanes, a, b, sub, |x, y| x.wrapping_sub(y));
    check_int_binary!(simd, i32, lanes, a, b, mul, |x, y| x.wrapping_mul(y));
    check_int_binary!(simd, i32, lanes, a, b, min, |x, y| x.min(y));
    check_int_binary!(simd, i32, lanes, a, b, max, |x, y| x.max(y));
    check_int_binary!(simd, i32, lanes, a, b, bitand, |x, y| x & y);
    check_int_binary!(simd, i32, lanes, a, b, bitor, |x, y| x | y);
    check_int_binary!(simd, i32, lanes, a, b, bitxor, |x, y| x ^ y);
    check_int_binary!(simd, i32, lanes, a, b, bitandnot, |x, y| !x & y);
    check_int_binary!(simd, i32, lanes, a, b, saturating_add, |x, y| x.saturating_add(y));
    check_int_binary!(simd, i32, lanes, a, b, saturating_sub, |x, y| x.saturating_sub(y));
    check_int_binary!(simd, i32, lanes, a, b, avg_round, |x, y| ((x as i64 + y as i64 + 1) >> 1) as i32);

    check_int_unary!(simd, i32, lanes, a, bitnot, |x| !x);
    check_int_unary!(simd, i32, lanes, a, abs, |x| x.wrapping_abs());
    check_int_unary!(simd, i32, lanes, a, neg, |x| x.wrapping_neg());
    check_int_unary!(simd, i32, lanes, a, count_ones, |x| x.count_ones() as i32);
    check_int_unary!(simd, i32, lanes, a, leading_zeros, |x| x.leading_zeros() as i32);
    check_int_unary!(simd, i32, lanes, a, trailing_zeros, |x| x.trailing_zeros() as i32);

    check_int_shift!(simd, i32, lanes, a, shl, 3, |x| x.wrapping_shl(3));
    check_int_shift!(simd, i32, lanes, a, shr, 3, |x| x.wrapping_shr(3));
    check_int_shift!(simd, i32, lanes, a, shl, 32, |x| x.wrapping_shl(32));
    check_int_shift!(simd, i32, lanes, a, shr, 32, |x| x.wrapping_shr(32));

    check_int_compare!(simd, i32, lanes, a, b, eq, |x, y| x == y);
    check_int_compare!(simd, i32, lanes, a, b, ne, |x, y| x != y);
    check_int_compare!(simd, i32, lanes, a, b, lt, |x, y| x < y);
    check_int_compare!(simd, i32, lanes, a, b, le, |x, y| x <= y);
    check_int_compare!(simd, i32, lanes, a, b, gt, |x, y| x > y);
    check_int_compare!(simd, i32, lanes, a, b, ge, |x, y| x >= y);
    check_int_select!(simd, i32, lanes, a, b);

    if S::i32_lanes == S::i64_lanes * 2 {
        let mut wide = store_i64(simd, simd.widen_low_i32_to_i64(simd.load_i32(&a)));
        for i in 0..S::i64_lanes {
            assert_eq!(wide[i], a[i] as i64, "widen_low_i32_to_i64 lane {i}");
        }
        wide = store_i64(simd, simd.widen_high_i32_to_i64(simd.load_i32(&a)));
        for i in 0..S::i64_lanes {
            assert_eq!(wide[i], a[i + S::i64_lanes] as i64, "widen_high_i32_to_i64 lane {i}");
        }

        let mut lo = [0i64; MAX_U64_LANES];
        let mut hi = [0i64; MAX_U64_LANES];
        for i in 0..S::i64_lanes {
            lo[i] = (i as i64).wrapping_mul(0x1234_5678_9ABC_DEF0).wrapping_sub(0x4000_0000_0000_0000);
            hi[i] = 0x4000_0000_0000_0000i64.wrapping_sub(i as i64);
        }
        let out = store_i32(simd, simd.narrow_saturating_i64_to_i32(simd.load_i64(&lo), simd.load_i64(&hi)));
        for i in 0..S::i64_lanes {
            assert_eq!(out[i], lo[i].clamp(i32::MIN as i64, i32::MAX as i64) as i32, "narrow_i64_to_i32 lo lane {i}");
            assert_eq!(out[i + S::i64_lanes], hi[i].clamp(i32::MIN as i64, i32::MAX as i64) as i32, "narrow_i64_to_i32 hi lane {i}");
        }
    }
}

fn exercise_int_u64<S: IntegerSimd>(simd: S) {
    let lanes = S::u64_lanes;
    let mut a = [0u64; MAX_U64_LANES];
    let mut b = [0u64; MAX_U64_LANES];
    for i in 0..MAX_U64_LANES {
        a[i] = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        b[i] = (i as u64).wrapping_mul(0x85EB_CA6B_1234_5678).wrapping_add(0x1234_5678);
    }
    a[0] = 0;
    b[0] = u64::MAX;

    let mut roundtrip = a;
    simd.store_u64(&mut roundtrip, simd.load_u64(&a));
    assert_eq!(roundtrip, a);

    let mut splat = [0u64; MAX_U64_LANES];
    simd.store_u64(&mut splat, simd.splat_u64(a[1]));
    for value in splat.iter().take(lanes) {
        assert_eq!(*value, a[1]);
    }

    check_int_binary!(simd, u64, lanes, a, b, add, |x, y| x.wrapping_add(y));
    check_int_binary!(simd, u64, lanes, a, b, sub, |x, y| x.wrapping_sub(y));
    check_int_binary!(simd, u64, lanes, a, b, mul, |x, y| x.wrapping_mul(y));
    check_int_binary!(simd, u64, lanes, a, b, min, |x, y| x.min(y));
    check_int_binary!(simd, u64, lanes, a, b, max, |x, y| x.max(y));
    check_int_binary!(simd, u64, lanes, a, b, bitand, |x, y| x & y);
    check_int_binary!(simd, u64, lanes, a, b, bitor, |x, y| x | y);
    check_int_binary!(simd, u64, lanes, a, b, bitxor, |x, y| x ^ y);
    check_int_binary!(simd, u64, lanes, a, b, bitandnot, |x, y| !x & y);
    check_int_binary!(simd, u64, lanes, a, b, saturating_add, |x, y| x.saturating_add(y));
    check_int_binary!(simd, u64, lanes, a, b, saturating_sub, |x, y| x.saturating_sub(y));
    check_int_binary!(simd, u64, lanes, a, b, avg_round, |x, y| ((x as u128 + y as u128 + 1) >> 1) as u64);

    check_int_unary!(simd, u64, lanes, a, bitnot, |x| !x);
    check_int_unary!(simd, u64, lanes, a, count_ones, |x| x.count_ones() as u64);
    check_int_unary!(simd, u64, lanes, a, leading_zeros, |x| x.leading_zeros() as u64);
    check_int_unary!(simd, u64, lanes, a, trailing_zeros, |x| x.trailing_zeros() as u64);

    check_int_shift!(simd, u64, lanes, a, shl, 3, |x| x.wrapping_shl(3));
    check_int_shift!(simd, u64, lanes, a, shr, 3, |x| x.wrapping_shr(3));
    check_int_shift!(simd, u64, lanes, a, shl, 64, |x| x.wrapping_shl(64));
    check_int_shift!(simd, u64, lanes, a, shr, 64, |x| x.wrapping_shr(64));

    check_int_compare!(simd, u64, lanes, a, b, eq, |x, y| x == y);
    check_int_compare!(simd, u64, lanes, a, b, ne, |x, y| x != y);
    check_int_compare!(simd, u64, lanes, a, b, lt, |x, y| x < y);
    check_int_compare!(simd, u64, lanes, a, b, le, |x, y| x <= y);
    check_int_compare!(simd, u64, lanes, a, b, gt, |x, y| x > y);
    check_int_compare!(simd, u64, lanes, a, b, ge, |x, y| x >= y);
    check_int_select!(simd, u64, lanes, a, b);
}

fn exercise_int_i64<S: IntegerSimd>(simd: S) {
    let lanes = S::i64_lanes;
    let mut a = [0i64; MAX_U64_LANES];
    let mut b = [0i64; MAX_U64_LANES];
    for i in 0..MAX_U64_LANES {
        a[i] = (i as i64).wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64).wrapping_sub(0x4000_0000_0000_0000);
        b[i] = (i as i64).wrapping_mul(0x85EB_CA6B_1234_5678u64 as i64).wrapping_add(0x1234_5678);
    }
    a[0] = i64::MIN;
    b[0] = i64::MAX;
    a[1] = 0;

    let mut roundtrip = a;
    simd.store_i64(&mut roundtrip, simd.load_i64(&a));
    assert_eq!(roundtrip, a);

    let mut splat = [0i64; MAX_U64_LANES];
    simd.store_i64(&mut splat, simd.splat_i64(a[1]));
    for value in splat.iter().take(lanes) {
        assert_eq!(*value, a[1]);
    }

    check_int_binary!(simd, i64, lanes, a, b, add, |x, y| x.wrapping_add(y));
    check_int_binary!(simd, i64, lanes, a, b, sub, |x, y| x.wrapping_sub(y));
    check_int_binary!(simd, i64, lanes, a, b, mul, |x, y| x.wrapping_mul(y));
    check_int_binary!(simd, i64, lanes, a, b, min, |x, y| x.min(y));
    check_int_binary!(simd, i64, lanes, a, b, max, |x, y| x.max(y));
    check_int_binary!(simd, i64, lanes, a, b, bitand, |x, y| x & y);
    check_int_binary!(simd, i64, lanes, a, b, bitor, |x, y| x | y);
    check_int_binary!(simd, i64, lanes, a, b, bitxor, |x, y| x ^ y);
    check_int_binary!(simd, i64, lanes, a, b, bitandnot, |x, y| !x & y);
    check_int_binary!(simd, i64, lanes, a, b, saturating_add, |x, y| x.saturating_add(y));
    check_int_binary!(simd, i64, lanes, a, b, saturating_sub, |x, y| x.saturating_sub(y));
    check_int_binary!(simd, i64, lanes, a, b, avg_round, |x, y| ((x as i128 + y as i128 + 1) >> 1) as i64);

    check_int_unary!(simd, i64, lanes, a, bitnot, |x| !x);
    check_int_unary!(simd, i64, lanes, a, abs, |x| x.wrapping_abs());
    check_int_unary!(simd, i64, lanes, a, neg, |x| x.wrapping_neg());
    check_int_unary!(simd, i64, lanes, a, count_ones, |x| x.count_ones() as i64);
    check_int_unary!(simd, i64, lanes, a, leading_zeros, |x| x.leading_zeros() as i64);
    check_int_unary!(simd, i64, lanes, a, trailing_zeros, |x| x.trailing_zeros() as i64);

    check_int_shift!(simd, i64, lanes, a, shl, 3, |x| x.wrapping_shl(3));
    check_int_shift!(simd, i64, lanes, a, shr, 3, |x| x.wrapping_shr(3));
    check_int_shift!(simd, i64, lanes, a, shl, 64, |x| x.wrapping_shl(64));
    check_int_shift!(simd, i64, lanes, a, shr, 64, |x| x.wrapping_shr(64));

    check_int_compare!(simd, i64, lanes, a, b, eq, |x, y| x == y);
    check_int_compare!(simd, i64, lanes, a, b, ne, |x, y| x != y);
    check_int_compare!(simd, i64, lanes, a, b, lt, |x, y| x < y);
    check_int_compare!(simd, i64, lanes, a, b, le, |x, y| x <= y);
    check_int_compare!(simd, i64, lanes, a, b, gt, |x, y| x > y);
    check_int_compare!(simd, i64, lanes, a, b, ge, |x, y| x >= y);
    check_int_select!(simd, i64, lanes, a, b);
}

#[test]
fn scalar_fallback_integer_u8() {
    exercise_int_u8(ScalarFallback);
}

#[test]
fn scalar_fallback_integer_i8() {
    exercise_int_i8(ScalarFallback);
}

#[test]
fn scalar_fallback_integer_u16() {
    exercise_int_u16(ScalarFallback);
}

#[test]
fn scalar_fallback_integer_i16() {
    exercise_int_i16(ScalarFallback);
}

#[test]
fn scalar_fallback_integer_u32() {
    exercise_int_u32(ScalarFallback);
}

#[test]
fn scalar_fallback_integer_i32() {
    exercise_int_i32(ScalarFallback);
}

#[test]
fn scalar_fallback_integer_u64() {
    exercise_int_u64(ScalarFallback);
}

#[test]
fn scalar_fallback_integer_i64() {
    exercise_int_i64(ScalarFallback);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse2_integer_u8() {
    if !sse2_available() {
        return;
    }
    exercise_int_u8(super::x86::sse2::Sse2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse2_integer_i8() {
    if !sse2_available() {
        return;
    }
    exercise_int_i8(super::x86::sse2::Sse2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse2_integer_u16() {
    if !sse2_available() {
        return;
    }
    exercise_int_u16(super::x86::sse2::Sse2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse2_integer_i16() {
    if !sse2_available() {
        return;
    }
    exercise_int_i16(super::x86::sse2::Sse2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse2_integer_u32() {
    if !sse2_available() {
        return;
    }
    exercise_int_u32(super::x86::sse2::Sse2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse2_integer_i32() {
    if !sse2_available() {
        return;
    }
    exercise_int_i32(super::x86::sse2::Sse2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse2_integer_u64() {
    if !sse2_available() {
        return;
    }
    exercise_int_u64(super::x86::sse2::Sse2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn sse2_integer_i64() {
    if !sse2_available() {
        return;
    }
    exercise_int_i64(super::x86::sse2::Sse2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx2_integer_u8() {
    if !avx2_fma_available() {
        return;
    }
    exercise_int_u8(super::x86::avx2::Avx2Fma);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx2_integer_i8() {
    if !avx2_fma_available() {
        return;
    }
    exercise_int_i8(super::x86::avx2::Avx2Fma);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx2_integer_u16() {
    if !avx2_fma_available() {
        return;
    }
    exercise_int_u16(super::x86::avx2::Avx2Fma);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx2_integer_i16() {
    if !avx2_fma_available() {
        return;
    }
    exercise_int_i16(super::x86::avx2::Avx2Fma);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx2_integer_u32() {
    if !avx2_fma_available() {
        return;
    }
    exercise_int_u32(super::x86::avx2::Avx2Fma);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx2_integer_i32() {
    if !avx2_fma_available() {
        return;
    }
    exercise_int_i32(super::x86::avx2::Avx2Fma);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx2_integer_u64() {
    if !avx2_fma_available() {
        return;
    }
    exercise_int_u64(super::x86::avx2::Avx2Fma);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx2_integer_i64() {
    if !avx2_fma_available() {
        return;
    }
    exercise_int_i64(super::x86::avx2::Avx2Fma);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn avx512_int_software_available() -> bool {
    core_detect::is_x86_feature_detected!("avx512f") && core_detect::is_x86_feature_detected!("avx512bw") && core_detect::is_x86_feature_detected!("avx512dq")
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn avx512_int_native_available() -> bool {
    avx512_int_software_available()
        && core_detect::is_x86_feature_detected!("avx512cd")
        && core_detect::is_x86_feature_detected!("avx512bitalg")
        && core_detect::is_x86_feature_detected!("avx512vpopcntdq")
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_software_u8() {
    if !avx512_int_software_available() {
        return;
    }
    exercise_int_u8(super::x86::avx512::Avx512Software::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_software_i8() {
    if !avx512_int_software_available() {
        return;
    }
    exercise_int_i8(super::x86::avx512::Avx512Software::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_software_u16() {
    if !avx512_int_software_available() {
        return;
    }
    exercise_int_u16(super::x86::avx512::Avx512Software::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_software_i16() {
    if !avx512_int_software_available() {
        return;
    }
    exercise_int_i16(super::x86::avx512::Avx512Software::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_software_u32() {
    if !avx512_int_software_available() {
        return;
    }
    exercise_int_u32(super::x86::avx512::Avx512Software::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_software_i32() {
    if !avx512_int_software_available() {
        return;
    }
    exercise_int_i32(super::x86::avx512::Avx512Software::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_software_u64() {
    if !avx512_int_software_available() {
        return;
    }
    exercise_int_u64(super::x86::avx512::Avx512Software::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_software_i64() {
    if !avx512_int_software_available() {
        return;
    }
    exercise_int_i64(super::x86::avx512::Avx512Software::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_native_u8() {
    if !avx512_int_native_available() {
        return;
    }
    exercise_int_u8(super::x86::avx512::Avx512Native::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_native_i8() {
    if !avx512_int_native_available() {
        return;
    }
    exercise_int_i8(super::x86::avx512::Avx512Native::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_native_u16() {
    if !avx512_int_native_available() {
        return;
    }
    exercise_int_u16(super::x86::avx512::Avx512Native::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_native_i16() {
    if !avx512_int_native_available() {
        return;
    }
    exercise_int_i16(super::x86::avx512::Avx512Native::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_native_u32() {
    if !avx512_int_native_available() {
        return;
    }
    exercise_int_u32(super::x86::avx512::Avx512Native::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_native_i32() {
    if !avx512_int_native_available() {
        return;
    }
    exercise_int_i32(super::x86::avx512::Avx512Native::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_native_u64() {
    if !avx512_int_native_available() {
        return;
    }
    exercise_int_u64(super::x86::avx512::Avx512Native::new());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_integer_native_i64() {
    if !avx512_int_native_available() {
        return;
    }
    exercise_int_i64(super::x86::avx512::Avx512Native::new());
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_integer_u8() {
    exercise_int_u8(super::aarch64::neon::Neon);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_integer_i8() {
    exercise_int_i8(super::aarch64::neon::Neon);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_integer_u16() {
    exercise_int_u16(super::aarch64::neon::Neon);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_integer_i16() {
    exercise_int_i16(super::aarch64::neon::Neon);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_integer_u32() {
    exercise_int_u32(super::aarch64::neon::Neon);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_integer_i32() {
    exercise_int_i32(super::aarch64::neon::Neon);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_integer_u64() {
    exercise_int_u64(super::aarch64::neon::Neon);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_integer_i64() {
    exercise_int_i64(super::aarch64::neon::Neon);
}
