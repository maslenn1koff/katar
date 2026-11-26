macro_rules! define_io_ops {
    ($ty:ident) => {
        paste::paste! {
            #[inline(always)]
            /// Loads the first `LANES` elements of `src`.
            ///
            /// # Panics
            ///
            /// Panics if `src.len()` is shorter than the lane count.
            fn [<load_ $ty>](self, value: &[$ty]) -> Self::$ty {
                assert!(value.len() >= Self::[<$ty _lanes>]);
                unsafe { self.[<load_unchecked_ $ty>](value) }
            }

            #[inline(always)]
            /// Stores the vector into the first `LANES` elements of `dst`.
            ///
            /// # Panics
            ///
            /// Panics if `dst.len()` is shorter than the lane count.
            fn [<store_ $ty>](self, dst: &mut [$ty], value: Self::$ty) {
                assert!(dst.len() >= Self::[<$ty _lanes>]);
                unsafe { self.[<store_unchecked_ $ty>](dst, value) }
            }

            /// Returns a vector with every lane set to `value`.
            fn [<splat_ $ty>](self, value: $ty) -> Self::$ty;

            /// Loads the first `LANES` elements of `src`.
            ///
            /// # Safety
            ///
            /// UB if `src.len()` is shorter than the lane count.
            unsafe fn [<load_unchecked_ $ty>](self, value: &[$ty]) -> Self::$ty;

            /// Stores the vector into the first `LANES` elements of `dst`.
            ///
            /// # Safety
            ///
            /// UB if `dst.len()` is shorter than the lane count.
            unsafe fn [<store_unchecked_ $ty>](self, dst: &mut [$ty], value: Self::$ty);
        }
    };
}

macro_rules! define_simd_unary_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise unary operation.
                fn [<$name _ $ty>](self, a: Self::$ty) -> Self::$ty;
            }
        )*
    };
}

macro_rules! define_simd_binary_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise binary operation.
                fn [<$name _ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty;
            }
        )*
    };
}

macro_rules! define_simd_ternary_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise ternary operation.
                fn [<$name _ $ty>](self, a: Self::$ty, b: Self::$ty, c: Self::$ty) -> Self::$ty;
            }
        )*
    };
}

macro_rules! define_simd_reduce_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Horizontal reduction over all lanes.
                ///
                /// The reduction order is backend-dependent.
                /// Floating-point results may differ from sequential scalar accumulation
                /// due to floating-point non-associativity.
                fn [<reduce_ $name _ $ty>](self, a: Self::$ty) -> $ty;
            }
        )*
    };
}

macro_rules! define_simd_compare_ops {
    ($ty:ident, $mask:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise comparison producing a lane mask.
                fn [<$name _ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$mask;
            }
        )*
    };
}

macro_rules! define_simd_shift_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                fn [<$name _ $ty>](self, a: Self::$ty, n: u32) -> Self::$ty;
            }
        )*
    };
}

pub trait MaskSimd: Copy + Send + Sync + 'static {
    fn splat(v: bool) -> Self;
    fn from_bits(v: u64) -> Self;
    fn to_bits(self) -> u64;

    fn and(self, other: Self) -> Self;
    fn or(self, other: Self) -> Self;
    fn not(self) -> Self;
    fn any(self) -> bool;
    fn all(self) -> bool;
}

pub trait Simd: FloatSimd {}

pub trait FloatSimd: Copy + Sync + Send + 'static {
    type f32: Copy + Sync + Send + 'static;
    type f64: Copy + Sync + Send + 'static;
    type m32: MaskSimd;
    type m64: MaskSimd;

    const f32_lanes: usize;
    const f64_lanes: usize;
    const m32_lanes: usize = Self::f32_lanes;
    const m64_lanes: usize = Self::f64_lanes;

    define_io_ops!(f32);
    define_io_ops!(f64);

    define_simd_unary_ops!(f32, abs, neg, sqrt, bitnot);
    define_simd_unary_ops!(f64, abs, neg, sqrt, bitnot);

    define_simd_binary_ops!(f32, add, sub, mul, div, min, max, bitand, bitor, bitxor, bitandnot);
    define_simd_binary_ops!(f64, add, sub, mul, div, min, max, bitand, bitor, bitxor, bitandnot);

    define_simd_ternary_ops!(f32, mul_add, mul_sub, nmul_add, nmul_sub);
    define_simd_ternary_ops!(f64, mul_add, mul_sub, nmul_add, nmul_sub);

    define_simd_reduce_ops!(f32, sum, max, min);
    define_simd_reduce_ops!(f64, sum, max, min);

    define_simd_compare_ops!(f32, m32, eq, ne, lt, le, gt, ge);
    define_simd_compare_ops!(f64, m64, eq, ne, lt, le, gt, ge);

    /// Per lane: `mask ? a : b`.
    fn select_f32(self, mask: Self::m32, a: Self::f32, b: Self::f32) -> Self::f32;
    /// Per lane: `mask ? a : b`.
    fn select_f64(self, mask: Self::m64, a: Self::f64, b: Self::f64) -> Self::f64;
}

pub trait IntegerSimd: Copy + Sync + Send + 'static {
    type u8: Copy + Sync + Send + 'static;
    type i8: Copy + Sync + Send + 'static;
    const u8_lanes: usize;
    const i8_lanes: usize;

    type u16: Copy + Sync + Send + 'static;
    type i16: Copy + Sync + Send + 'static;
    const u16_lanes: usize;
    const i16_lanes: usize;

    type u32: Copy + Sync + Send + 'static;
    type i32: Copy + Sync + Send + 'static;
    const u32_lanes: usize;
    const i32_lanes: usize;

    type u64: Copy + Sync + Send + 'static;
    type i64: Copy + Sync + Send + 'static;
    const u64_lanes: usize;
    const i64_lanes: usize;

    type m8: MaskSimd;
    type m16: MaskSimd;
    type m32: MaskSimd;
    type m64: MaskSimd;

    const m8_lanes: usize = Self::u8_lanes;
    const m16_lanes: usize = Self::u16_lanes;
    const m32_lanes: usize = Self::u32_lanes;
    const m64_lanes: usize = Self::u64_lanes;

    define_io_ops!(u8);
    define_io_ops!(i8);
    define_io_ops!(u16);
    define_io_ops!(i16);
    define_io_ops!(u32);
    define_io_ops!(i32);
    define_io_ops!(u64);
    define_io_ops!(i64);

    define_simd_unary_ops!(u8, bitnot, count_ones, leading_zeros, trailing_zeros);
    define_simd_unary_ops!(i8, bitnot, count_ones, leading_zeros, trailing_zeros, abs, neg);
    define_simd_unary_ops!(u16, bitnot, count_ones, leading_zeros, trailing_zeros);
    define_simd_unary_ops!(i16, bitnot, count_ones, leading_zeros, trailing_zeros, abs, neg);
    define_simd_unary_ops!(u32, bitnot, count_ones, leading_zeros, trailing_zeros);
    define_simd_unary_ops!(i32, bitnot, count_ones, leading_zeros, trailing_zeros, abs, neg);
    define_simd_unary_ops!(u64, bitnot, count_ones, leading_zeros, trailing_zeros);
    define_simd_unary_ops!(i64, bitnot, count_ones, leading_zeros, trailing_zeros);

    define_simd_binary_ops!(u8, add, sub, mul, min, max, bitand, bitor, bitxor, bitandnot, saturating_add, saturating_sub, avg_round);
    define_simd_binary_ops!(i8, add, sub, mul, min, max, bitand, bitor, bitxor, bitandnot, saturating_add, saturating_sub, avg_round);
    define_simd_binary_ops!(u16, add, sub, mul, min, max, bitand, bitor, bitxor, bitandnot, saturating_add, saturating_sub, avg_round);
    define_simd_binary_ops!(i16, add, sub, mul, min, max, bitand, bitor, bitxor, bitandnot, saturating_add, saturating_sub, avg_round);
    define_simd_binary_ops!(u32, add, sub, mul, min, max, bitand, bitor, bitxor, bitandnot, saturating_add, saturating_sub, avg_round);
    define_simd_binary_ops!(i32, add, sub, mul, min, max, bitand, bitor, bitxor, bitandnot, saturating_add, saturating_sub, avg_round);
    define_simd_binary_ops!(u64, add, sub, mul, min, max, bitand, bitor, bitxor, bitandnot, saturating_add, saturating_sub, avg_round);
    define_simd_binary_ops!(i64, add, sub, mul, min, max, bitand, bitor, bitxor, bitandnot, saturating_add, saturating_sub, avg_round);

    define_simd_compare_ops!(u8, m8, eq, ne, lt, le, gt, ge);
    define_simd_compare_ops!(i8, m8, eq, ne, lt, le, gt, ge);
    define_simd_compare_ops!(u16, m16, eq, ne, lt, le, gt, ge);
    define_simd_compare_ops!(i16, m16, eq, ne, lt, le, gt, ge);
    define_simd_compare_ops!(u32, m32, eq, ne, lt, le, gt, ge);
    define_simd_compare_ops!(i32, m32, eq, ne, lt, le, gt, ge);
    define_simd_compare_ops!(u64, m64, eq, ne, lt, le, gt, ge);
    define_simd_compare_ops!(i64, m64, eq, ne, lt, le, gt, ge);

    define_simd_shift_ops!(u8, shl, shr);
    define_simd_shift_ops!(i8, shl, shr);
    define_simd_shift_ops!(u16, shl, shr);
    define_simd_shift_ops!(i16, shl, shr);
    define_simd_shift_ops!(u32, shl, shr);
    define_simd_shift_ops!(i32, shl, shr);
    define_simd_shift_ops!(u64, shl, shr);
    define_simd_shift_ops!(i64, shl, shr);

    fn select_u8(self, mask: Self::m8, a: Self::u8, b: Self::u8) -> Self::u8;
    fn select_i8(self, mask: Self::m8, a: Self::i8, b: Self::i8) -> Self::i8;
    fn select_u16(self, mask: Self::m16, a: Self::u16, b: Self::u16) -> Self::u16;
    fn select_i16(self, mask: Self::m16, a: Self::i16, b: Self::i16) -> Self::i16;
    fn select_u32(self, mask: Self::m32, a: Self::u32, b: Self::u32) -> Self::u32;
    fn select_i32(self, mask: Self::m32, a: Self::i32, b: Self::i32) -> Self::i32;
    fn select_u64(self, mask: Self::m64, a: Self::u64, b: Self::u64) -> Self::u64;
    fn select_i64(self, mask: Self::m64, a: Self::i64, b: Self::i64) -> Self::i64;

    fn widen_low_u8_to_u16(self, x: Self::u8) -> Self::u16;
    fn widen_high_u8_to_u16(self, x: Self::u8) -> Self::u16;
    fn widen_low_i8_to_i16(self, x: Self::i8) -> Self::i16;
    fn widen_high_i8_to_i16(self, x: Self::i8) -> Self::i16;

    fn widen_low_u16_to_u32(self, x: Self::u16) -> Self::u32;
    fn widen_high_u16_to_u32(self, x: Self::u16) -> Self::u32;
    fn widen_low_i16_to_i32(self, x: Self::i16) -> Self::i32;
    fn widen_high_i16_to_i32(self, x: Self::i16) -> Self::i32;

    fn widen_low_u32_to_u64(self, x: Self::u32) -> Self::u64;
    fn widen_high_u32_to_u64(self, x: Self::u32) -> Self::u64;
    fn widen_low_i32_to_i64(self, x: Self::i32) -> Self::i64;
    fn widen_high_i32_to_i64(self, x: Self::i32) -> Self::i64;

    fn narrow_saturating_u16_to_u8(self, lo: Self::u16, hi: Self::u16) -> Self::u8;
    fn narrow_saturating_i16_to_i8(self, lo: Self::i16, hi: Self::i16) -> Self::i8;
    fn narrow_saturating_u32_to_u16(self, lo: Self::u32, hi: Self::u32) -> Self::u16;
    fn narrow_saturating_i32_to_i16(self, lo: Self::i32, hi: Self::i32) -> Self::i16;
    fn narrow_saturating_u64_to_u32(self, lo: Self::u64, hi: Self::u64) -> Self::u32;
    fn narrow_saturating_i64_to_i32(self, lo: Self::i64, hi: Self::i64) -> Self::i32;
}
