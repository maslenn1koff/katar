use crate::traits::{FloatSimd, MaskSimd, Simd};
use core::ops::{Add, Div, Mul, Neg, Sub};

macro_rules! define_io_ops {
    ($ty:ident) => {
        paste::paste! {
            #[inline(always)]
            fn [<splat_ $ty>](self, value: $ty) -> Self::[<$ty>] {
                value
            }

            #[inline(always)]
            unsafe fn [<load_unchecked_ $ty>](self, src: &[$ty]) -> Self::[<$ty>] {
                src[0]
            }

            #[inline(always)]
            unsafe fn [<store_unchecked_ $ty>](self, dst: &mut [$ty], value: Self::[<$ty>]) {
                dst[0] = value;
            }
        }
    };
}

macro_rules! define_unary_ops {
    ($ty:ident, $($name:ident),*) => {
        paste::paste! {
            $(
                #[inline(always)]
                fn [<$name _ $ty>](self, a: Self::[<$ty>]) -> Self::[<$ty>] {
                    a.[<$name>]()
                }
            )*
        }
    };
}

macro_rules! define_binary_ops {
    ($ty:ident, $($name:ident),*) => {
        paste::paste! {
            $(
                #[inline(always)]
                fn [<$name _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>]) -> Self::[<$ty>] {
                    a.[<$name>](b)
                }
            )*
        }
    };
}

macro_rules! define_ternary_ops {
    ($ty:ident) => {
        paste::paste! {
            #[inline(always)]
            fn [<mul_add _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>], c: Self::[<$ty>]) -> Self::[<$ty>] {
                a * b + c
            }

            #[inline(always)]
            fn [<mul_sub _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>], c: Self::[<$ty>]) -> Self::[<$ty>] {
                a * b - c
            }

            #[inline(always)]
            fn [<nmul_add _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>], c: Self::[<$ty>]) -> Self::[<$ty>] {
                -(a * b) + c
            }

            #[inline(always)]
            fn [<nmul_sub _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>], c: Self::[<$ty>]) -> Self::[<$ty>] {
                -(a * b) - c
            }
        }
    };
}

macro_rules! define_compare_ops {
    ($ty:ident, $mask:ident, $($name:ident),*) => {
        paste::paste! {
            $(
                #[inline(always)]
                fn [<$name _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>]) -> Self::[<$mask>] {
                    FallbackMask(a.[<$name>](&b))
                }
            )*
        }
    };
}

macro_rules! define_bit_ops {
    ($ty:ident) => {
        paste::paste! {
            #[inline(always)]
            fn [<bitand_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(a.to_bits() & b.to_bits())
            }

            #[inline(always)]
            fn [<bitandnot_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(!a.to_bits() & b.to_bits())
            }

            #[inline(always)]
            fn [<bitor_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(a.to_bits() | b.to_bits())
            }

            #[inline(always)]
            fn [<bitxor_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(a.to_bits() ^ b.to_bits())
            }

            #[inline(always)]
            fn [<bitnot_ $ty>](self, a: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(!a.to_bits())
            }
        }
    };
}

macro_rules! define_reduce_ops {
    ($ty:ident, $($name:ident),*) => {
        paste::paste! {
            $(
                #[inline(always)]
                fn [<reduce_ $name _ $ty>](self, a: $ty) -> $ty {
                    a
                }
            )*
        }
    };
}

#[derive(Clone, Copy)]
pub struct FallbackMask(bool);

impl MaskSimd for FallbackMask {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        FallbackMask(v)
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        FallbackMask(v & 1 != 0)
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        self.0 as u64
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        FallbackMask(self.0 && other.0)
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        FallbackMask(self.0 || other.0)
    }

    #[inline(always)]
    fn not(self) -> Self {
        FallbackMask(!self.0)
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.0
    }
}

#[derive(Clone, Copy)]
pub struct ScalarFallback;

impl FloatSimd for ScalarFallback {
    type f32 = f32;
    type f64 = f64;
    type m32 = FallbackMask;
    type m64 = FallbackMask;

    const f32_lanes: usize = 1;
    const f64_lanes: usize = 1;

    define_io_ops!(f32);
    define_io_ops!(f64);

    define_unary_ops!(f32, abs, neg);
    define_unary_ops!(f64, abs, neg);

    #[inline(always)]
    fn sqrt_f32(self, a: Self::f32) -> Self::f32 {
        libm::sqrtf(a)
    }

    #[inline(always)]
    fn sqrt_f64(self, a: Self::f64) -> Self::f64 {
        libm::sqrt(a)
    }

    define_binary_ops!(f32, add, sub, mul, div, min, max);
    define_binary_ops!(f64, add, sub, mul, div, min, max);

    define_ternary_ops!(f32);
    define_ternary_ops!(f64);

    define_compare_ops!(f32, m32, eq, ne, lt, le, gt, ge);
    define_compare_ops!(f64, m64, eq, ne, lt, le, gt, ge);

    define_bit_ops!(f32);
    define_bit_ops!(f64);

    define_reduce_ops!(f32, sum, max, min);
    define_reduce_ops!(f64, sum, max, min);

    #[inline(always)]
    fn select_f32(self, mask: Self::m32, a: Self::f32, b: Self::f32) -> Self::f32 {
        if mask.0 { a } else { b }
    }

    #[inline(always)]
    fn select_f64(self, mask: Self::m64, a: Self::f64, b: Self::f64) -> Self::f64 {
        if mask.0 { a } else { b }
    }
}

impl Simd for ScalarFallback {}
