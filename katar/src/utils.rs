use crate::traits::Simd;
use core::iter::ExactSizeIterator;

macro_rules! define_as_simd {
    ($ty:ident) => {
        paste::paste! {
            /// Splits a slice of `$ty` values into an iterator of arrays of `N` SIMD vectors, plus a remaining slice of `$ty` values.
            pub fn [<as_simd_ $ty>]<'a, const N: usize, S: Simd>(simd: &'a S, src: &'a [$ty]) -> (impl ExactSizeIterator<Item = [S::$ty; N]> + 'a, &'a [$ty]) {
                assert!(N > 0);
                assert!(S::[<$ty _lanes>] > 0);

                let block = S::[<$ty _lanes>] * N;
                let count = src.len() / block;
                let (head, tail) = src.split_at(count * block);
                let vectors = head.chunks_exact(block).map(move |chunk| {
                    unsafe {
                        core::array::from_fn(|i| simd.[<load_unchecked_ $ty>](chunk.get_unchecked(i * S::[<$ty _lanes>]..)))
                    }
                });

                (vectors, tail)
            }
        }
    };
}

define_as_simd!(f32);
define_as_simd!(f64);
define_as_simd!(u8);
define_as_simd!(i8);
define_as_simd!(u16);
define_as_simd!(i16);
define_as_simd!(u32);
define_as_simd!(i32);
define_as_simd!(u64);
define_as_simd!(i64);

#[cfg(test)]
mod tests {
    extern crate std;

    use crate::backend::fallback::ScalarFallback;
    use crate::traits::Simd;

    macro_rules! define_as_simd_check {
        ($check:ident, $ty:ty, $lanes:ident, $as_simd:ident, $store:ident) => {
            fn $check<S: Simd, const N: usize>(simd: S) {
                let lanes = S::$lanes;
                let block = lanes * N;

                let (vectors, tail) = super::$as_simd::<N, _>(&simd, &[]);
                assert_eq!(vectors.len(), 0);
                assert!(tail.is_empty());

                let short = std::vec![<$ty>::default(); block.saturating_sub(1)];
                let (vectors, tail) = super::$as_simd::<N, _>(&simd, &short);
                assert_eq!(vectors.len(), 0);
                assert_eq!(tail, short.as_slice());

                let exact: std::vec::Vec<$ty> = (0..2 * block).map(|i| i as $ty).collect();
                let (vectors, tail) = super::$as_simd::<N, _>(&simd, &exact);
                assert_eq!(vectors.len(), 2);
                assert!(tail.is_empty());

                let tail_len = block.saturating_sub(1);
                let total = 3 * block + tail_len;
                let src: std::vec::Vec<$ty> = (0..total).map(|i| i as $ty).collect();
                let (vectors, tail) = super::$as_simd::<N, _>(&simd, &src);
                assert_eq!(vectors.len(), 3);
                assert_eq!(tail.len(), tail_len);
                assert_eq!(tail, &src[3 * block..]);

                let mut buf = std::vec![<$ty>::default(); lanes];
                for (block_index, block_vectors) in vectors.enumerate() {
                    for (vector_index, vector) in block_vectors.iter().enumerate() {
                        paste::paste! { simd.[<$store>](&mut buf, *vector) };
                        let start = (block_index * N + vector_index) * lanes;
                        assert_eq!(buf.as_slice(), &src[start..start + lanes]);
                    }
                }
            }
        };
    }

    define_as_simd_check!(check_as_simd_f32, f32, f32_lanes, as_simd_f32, store_f32);
    define_as_simd_check!(check_as_simd_f64, f64, f64_lanes, as_simd_f64, store_f64);
    define_as_simd_check!(check_as_simd_u8, u8, u8_lanes, as_simd_u8, store_u8);
    define_as_simd_check!(check_as_simd_i8, i8, i8_lanes, as_simd_i8, store_i8);
    define_as_simd_check!(check_as_simd_u16, u16, u16_lanes, as_simd_u16, store_u16);
    define_as_simd_check!(check_as_simd_i16, i16, i16_lanes, as_simd_i16, store_i16);
    define_as_simd_check!(check_as_simd_u32, u32, u32_lanes, as_simd_u32, store_u32);
    define_as_simd_check!(check_as_simd_i32, i32, i32_lanes, as_simd_i32, store_i32);
    define_as_simd_check!(check_as_simd_u64, u64, u64_lanes, as_simd_u64, store_u64);
    define_as_simd_check!(check_as_simd_i64, i64, i64_lanes, as_simd_i64, store_i64);

    macro_rules! check_all_as_simd {
        ($simd:expr, $n:literal) => {{
            let simd = $simd;
            check_as_simd_f32::<_, $n>(simd);
            check_as_simd_f64::<_, $n>(simd);
            check_as_simd_u8::<_, $n>(simd);
            check_as_simd_i8::<_, $n>(simd);
            check_as_simd_u16::<_, $n>(simd);
            check_as_simd_i16::<_, $n>(simd);
            check_as_simd_u32::<_, $n>(simd);
            check_as_simd_i32::<_, $n>(simd);
            check_as_simd_u64::<_, $n>(simd);
            check_as_simd_i64::<_, $n>(simd);
        }};
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn sse2_available() -> bool {
        core_detect::is_x86_feature_detected!("sse2")
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx2_fma_available() -> bool {
        core_detect::is_x86_feature_detected!("avx2") && core_detect::is_x86_feature_detected!("fma")
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx512f_available() -> bool {
        core_detect::is_x86_feature_detected!("avx512f") && core_detect::is_x86_feature_detected!("avx512bw") && core_detect::is_x86_feature_detected!("avx512dq")
    }

    #[test]
    fn as_simd_scalar_fallback() {
        check_all_as_simd!(ScalarFallback, 1);
        check_all_as_simd!(ScalarFallback, 2);
        check_all_as_simd!(ScalarFallback, 3);
    }

    #[test]
    #[should_panic]
    fn as_simd_zero_vectors_panics() {
        let simd = ScalarFallback;
        let src = [1.0f32; 4];
        let _ = super::as_simd_f32::<0, _>(&simd, &src);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn as_simd_sse2() {
        if !sse2_available() {
            return;
        }
        check_all_as_simd!(crate::backend::x86::sse2::Sse2, 1);
        check_all_as_simd!(crate::backend::x86::sse2::Sse2, 2);
        check_all_as_simd!(crate::backend::x86::sse2::Sse2, 3);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn as_simd_avx2() {
        if !avx2_fma_available() {
            return;
        }
        check_all_as_simd!(crate::backend::x86::avx2::Avx2Fma, 1);
        check_all_as_simd!(crate::backend::x86::avx2::Avx2Fma, 2);
        check_all_as_simd!(crate::backend::x86::avx2::Avx2Fma, 3);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn as_simd_avx512() {
        if !avx512f_available() {
            return;
        }
        check_all_as_simd!(crate::backend::x86::avx512::Avx512Software::new(), 1);
        check_all_as_simd!(crate::backend::x86::avx512::Avx512Software::new(), 2);
        check_all_as_simd!(crate::backend::x86::avx512::Avx512Software::new(), 3);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn as_simd_neon() {
        check_all_as_simd!(crate::backend::aarch64::neon::Neon, 1);
        check_all_as_simd!(crate::backend::aarch64::neon::Neon, 2);
        check_all_as_simd!(crate::backend::aarch64::neon::Neon, 3);
    }
}
