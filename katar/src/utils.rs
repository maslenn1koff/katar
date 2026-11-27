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
