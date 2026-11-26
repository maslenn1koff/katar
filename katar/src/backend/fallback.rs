use crate::traits::{FloatSimd, IntegerSimd, MaskSimd, Simd};
use core::ops::{Add, Div, Mul, Neg, Sub};

macro_rules! define_io_ops {
    ($ty:ident) => {
        paste::paste! {
            #[inline(always)] fn [<splat_ $ty>](self, value: $ty) -> Self::[<$ty>] { value }
            #[inline(always)] unsafe fn [<load_unchecked_ $ty>](self, src: &[$ty]) -> Self::[<$ty>] { src[0] }
            #[inline(always)] unsafe fn [<store_unchecked_ $ty>](self, dst: &mut [$ty], value: Self::[<$ty>]) { dst[0] = value }
        }
    };
}

macro_rules! define_unary_ops {
    ($ty:ident, $($name:ident),*) => {
        paste::paste! {
            $(
                #[inline(always)]
                fn [<$name _ $ty>](self, a: Self::[<$ty>]) -> Self::[<$ty>] {
                    a.[<$name>]() as _
                }
            )*
        }
    };
}

macro_rules! define_unary_wrapping_ops {
    ($ty:ident, $($name:ident),*) => {
        paste::paste! {
            $(
                #[inline(always)]
                fn [<$name _ $ty>](self, a: Self::[<$ty>]) -> Self::[<$ty>] {
                    a.[<wrapping_ $name>]()
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
                    a.$name(b)
                }
            )*
        }
    };
}

macro_rules! define_binary_wrapping_ops {
    ($ty:ident, $($name:ident),*) => {
        paste::paste! {
            $(
                #[inline(always)]
                fn [<$name _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>]) -> Self::[<$ty>] {
                    a.[<wrapping_ $name>](b)
                }
            )*
        }
    };
}

macro_rules! define_ternary_ops {
    ($ty:ident) => {
        paste::paste! {
            #[inline(always)] fn [<mul_add _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>], c: Self::[<$ty>]) -> Self::[<$ty>] { a * b + c }
            #[inline(always)] fn [<mul_sub _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>], c: Self::[<$ty>]) -> Self::[<$ty>] { a * b - c }
            #[inline(always)] fn [<nmul_add _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>], c: Self::[<$ty>]) -> Self::[<$ty>] { -(a * b) + c }
            #[inline(always)] fn [<nmul_sub _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>], c: Self::[<$ty>]) -> Self::[<$ty>] { -(a * b) - c }
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
            #[inline(always)] fn [<bitand_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty { <$ty>::from_bits(a.to_bits() & b.to_bits()) }
            #[inline(always)] fn [<bitandnot_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty { <$ty>::from_bits(!a.to_bits() & b.to_bits()) }
            #[inline(always)] fn [<bitor_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty { <$ty>::from_bits(a.to_bits() | b.to_bits()) }
            #[inline(always)] fn [<bitxor_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty { <$ty>::from_bits(a.to_bits() ^ b.to_bits()) }
            #[inline(always)] fn [<bitnot_ $ty>](self, a: Self::$ty) -> Self::$ty { <$ty>::from_bits(!a.to_bits()) }
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

macro_rules! define_int_bit_ops {
    ($ty:ident) => {
        paste::paste! {
            #[inline(always)] fn [<bitand_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty { a & b }
            #[inline(always)] fn [<bitor_  $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty { a | b }
            #[inline(always)] fn [<bitxor_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty { a ^ b }
            #[inline(always)] fn [<bitandnot_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty { !a & b }
            #[inline(always)] fn [<bitnot_ $ty>](self, a: Self::$ty) -> Self::$ty { !a }
        }
    };
}

macro_rules! define_shift_ops {
    ($ty:ident) => {
        paste::paste! {
            #[inline(always)] fn [<shl_ $ty>](self, a: Self::$ty, n: u32) -> Self::$ty { a.wrapping_shl(n) }
            #[inline(always)] fn [<shr_ $ty>](self, a: Self::$ty, n: u32) -> Self::$ty { a.wrapping_shr(n) }
        }
    };
}

macro_rules! define_select_ops {
    ($ty:ident, $mask:ident) => {
        paste::paste! {
            #[inline(always)]
            fn [<select_ $ty>](self, mask: Self::$mask, a: Self::$ty, b: Self::$ty) -> Self::$ty {
                if mask.0 { a } else { b }
            }
        }
    };
}

macro_rules! define_widen_ops {
    ($from:ident, $to:ident) => {
        paste::paste! {
            #[inline(always)] fn [<widen_low_  $from _to_ $to>](self, x: Self::$from) -> Self::$to { x as _ }
            #[inline(always)] fn [<widen_high_ $from _to_ $to>](self, x: Self::$from) -> Self::$to { x as _ }
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
    type mf32 = FallbackMask;
    type mf64 = FallbackMask;

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

    define_compare_ops!(f32, mf32, eq, ne, lt, le, gt, ge);
    define_compare_ops!(f64, mf64, eq, ne, lt, le, gt, ge);

    define_bit_ops!(f32);
    define_bit_ops!(f64);

    define_reduce_ops!(f32, sum, max, min);
    define_reduce_ops!(f64, sum, max, min);

    define_select_ops!(f32, mf32);
    define_select_ops!(f64, mf64);
}

impl IntegerSimd for ScalarFallback {
    type u8 = u8;
    type i8 = i8;
    type u16 = u16;
    type i16 = i16;
    type u32 = u32;
    type i32 = i32;
    type u64 = u64;
    type i64 = i64;

    type m8 = FallbackMask;
    type m16 = FallbackMask;
    type m32 = FallbackMask;
    type m64 = FallbackMask;

    const u8_lanes: usize = 1;
    const i8_lanes: usize = 1;
    const u16_lanes: usize = 1;
    const i16_lanes: usize = 1;
    const u32_lanes: usize = 1;
    const i32_lanes: usize = 1;
    const u64_lanes: usize = 1;
    const i64_lanes: usize = 1;

    define_io_ops!(u8);
    define_io_ops!(i8);
    define_io_ops!(u16);
    define_io_ops!(i16);
    define_io_ops!(u32);
    define_io_ops!(i32);
    define_io_ops!(u64);
    define_io_ops!(i64);

    define_unary_wrapping_ops!(i8, abs, neg);
    define_unary_wrapping_ops!(i16, abs, neg);
    define_unary_wrapping_ops!(i32, abs, neg);
    define_unary_wrapping_ops!(i64, abs, neg);

    define_unary_ops!(u8, count_ones, leading_zeros, trailing_zeros);
    define_unary_ops!(i8, count_ones, leading_zeros, trailing_zeros);
    define_unary_ops!(u16, count_ones, leading_zeros, trailing_zeros);
    define_unary_ops!(i16, count_ones, leading_zeros, trailing_zeros);
    define_unary_ops!(u32, count_ones, leading_zeros, trailing_zeros);
    define_unary_ops!(i32, count_ones, leading_zeros, trailing_zeros);
    define_unary_ops!(u64, count_ones, leading_zeros, trailing_zeros);
    define_unary_ops!(i64, count_ones, leading_zeros, trailing_zeros);

    define_binary_wrapping_ops!(u8, add, sub, mul);
    define_binary_wrapping_ops!(i8, add, sub, mul);
    define_binary_wrapping_ops!(u16, add, sub, mul);
    define_binary_wrapping_ops!(i16, add, sub, mul);
    define_binary_wrapping_ops!(u32, add, sub, mul);
    define_binary_wrapping_ops!(i32, add, sub, mul);
    define_binary_wrapping_ops!(u64, add, sub, mul);
    define_binary_wrapping_ops!(i64, add, sub, mul);

    define_binary_ops!(u8, max, min, saturating_add, saturating_sub);
    define_binary_ops!(i8, max, min, saturating_add, saturating_sub);
    define_binary_ops!(u16, max, min, saturating_add, saturating_sub);
    define_binary_ops!(i16, max, min, saturating_add, saturating_sub);
    define_binary_ops!(u32, max, min, saturating_add, saturating_sub);
    define_binary_ops!(i32, max, min, saturating_add, saturating_sub);
    define_binary_ops!(u64, max, min, saturating_add, saturating_sub);
    define_binary_ops!(i64, max, min, saturating_add, saturating_sub);

    define_int_bit_ops!(u8);
    define_int_bit_ops!(i8);
    define_int_bit_ops!(u16);
    define_int_bit_ops!(i16);
    define_int_bit_ops!(u32);
    define_int_bit_ops!(i32);
    define_int_bit_ops!(u64);
    define_int_bit_ops!(i64);

    define_shift_ops!(u8);
    define_shift_ops!(i8);
    define_shift_ops!(u16);
    define_shift_ops!(i16);
    define_shift_ops!(u32);
    define_shift_ops!(i32);
    define_shift_ops!(u64);
    define_shift_ops!(i64);

    define_compare_ops!(u8, m8, eq, ne, lt, le, gt, ge);
    define_compare_ops!(i8, m8, eq, ne, lt, le, gt, ge);
    define_compare_ops!(u16, m16, eq, ne, lt, le, gt, ge);
    define_compare_ops!(i16, m16, eq, ne, lt, le, gt, ge);
    define_compare_ops!(u32, m32, eq, ne, lt, le, gt, ge);
    define_compare_ops!(i32, m32, eq, ne, lt, le, gt, ge);
    define_compare_ops!(u64, m64, eq, ne, lt, le, gt, ge);
    define_compare_ops!(i64, m64, eq, ne, lt, le, gt, ge);

    define_select_ops!(u8, m8);
    define_select_ops!(i8, m8);
    define_select_ops!(u16, m16);
    define_select_ops!(i16, m16);
    define_select_ops!(u32, m32);
    define_select_ops!(i32, m32);
    define_select_ops!(u64, m64);
    define_select_ops!(i64, m64);

    #[rustfmt::skip] #[inline(always)]    fn avg_round_u8 (self, a: Self::u8,  b: Self::u8)  -> Self::u8  { ((a as u16  + b as u16  + 1) >> 1) as u8  }
    #[rustfmt::skip] #[inline(always)]    fn avg_round_i8 (self, a: Self::i8,  b: Self::i8)  -> Self::i8  { ((a as i16  + b as i16  + 1) >> 1) as i8  }
    #[rustfmt::skip] #[inline(always)]    fn avg_round_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 { ((a as u32  + b as u32  + 1) >> 1) as u16 }
    #[rustfmt::skip] #[inline(always)]    fn avg_round_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 { ((a as i32  + b as i32  + 1) >> 1) as i16 }
    #[rustfmt::skip] #[inline(always)]    fn avg_round_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 { ((a as u64  + b as u64  + 1) >> 1) as u32 }
    #[rustfmt::skip] #[inline(always)]    fn avg_round_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 { ((a as i64  + b as i64  + 1) >> 1) as i32 }
    #[rustfmt::skip] #[inline(always)]    fn avg_round_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 { ((a as u128 + b as u128 + 1) >> 1) as u64 }
    #[rustfmt::skip] #[inline(always)]    fn avg_round_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 { ((a as i128 + b as i128 + 1) >> 1) as i64 }

    define_widen_ops!(u8, u16);
    define_widen_ops!(i8, i16);
    define_widen_ops!(u16, u32);
    define_widen_ops!(i16, i32);
    define_widen_ops!(u32, u64);
    define_widen_ops!(i32, i64);

    #[rustfmt::skip] #[inline(always)]    fn narrow_saturating_u16_to_u8 (self, lo: Self::u16, _hi: Self::u16) -> Self::u8  { lo.clamp(0, u8::MAX  as u16) as u8  }
    #[rustfmt::skip] #[inline(always)]    fn narrow_saturating_i16_to_i8 (self, lo: Self::i16, _hi: Self::i16) -> Self::i8  { lo.clamp(i8::MIN  as i16, i8::MAX  as i16) as i8  }
    #[rustfmt::skip] #[inline(always)]    fn narrow_saturating_u32_to_u16(self, lo: Self::u32, _hi: Self::u32) -> Self::u16 { lo.clamp(0, u16::MAX as u32) as u16 }
    #[rustfmt::skip] #[inline(always)]    fn narrow_saturating_i32_to_i16(self, lo: Self::i32, _hi: Self::i32) -> Self::i16 { lo.clamp(i16::MIN as i32, i16::MAX as i32) as i16 }
    #[rustfmt::skip] #[inline(always)]    fn narrow_saturating_u64_to_u32(self, lo: Self::u64, _hi: Self::u64) -> Self::u32 { lo.clamp(0, u32::MAX as u64) as u32 }
    #[rustfmt::skip] #[inline(always)]    fn narrow_saturating_i64_to_i32(self, lo: Self::i64, _hi: Self::i64) -> Self::i32 { lo.clamp(i32::MIN as i64, i32::MAX as i64) as i32 }
}

impl Simd for ScalarFallback {}
