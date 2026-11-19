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
                fn [<$name _ $ty>](self, a: Self::[<$ty>]) -> Self::[<$ty>];
            }
        )*
    };
}

macro_rules! define_simd_binary_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise binary operation.
                fn [<$name _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>]) -> Self::[<$ty>];
            }
        )*
    };
}

macro_rules! define_simd_ternary_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise ternary operation.
                fn [<$name _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>], c: Self::[<$ty>]) -> Self::[<$ty>];
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
                fn [<reduce_ $name _ $ty>](self, a: Self::[<$ty>]) -> $ty;
            }
        )*
    };
}

macro_rules! define_simd_compare_ops {
    ($ty:ident, $mask:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise comparison producing a lane mask.
                fn [<$name _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>]) -> Self::$mask;
            }
        )*
    };
}

pub trait SimdMask: Copy + Send + Sync + 'static {
    type bits: Copy + Sync + Send + 'static;

    fn splat(v: bool) -> Self;
    fn from_bits(v: Self::bits) -> Self;
    fn to_bits(self) -> Self::bits;

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
    type m32: SimdMask;
    type m64: SimdMask;

    const f32_lanes: usize;
    const f64_lanes: usize;
    const m32_lanes: usize = Self::f32_lanes;
    const m64_lanes: usize = Self::f32_lanes;

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
