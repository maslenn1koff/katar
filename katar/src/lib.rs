//! Portable SIMD with runtime backend selection.
//!
//! Write vector code once as a [`SimdFn`] generic over [`traits::Simd`], then run it with
//! [`Arch::dispatch`]. The best available backend — AVX-512, AVX2+FMA, SSE2, NEON or a scalar
//! fallback — is selected at runtime from the CPU features. The crate is `no_std` and
//! allocation-free.
//!
//! # Example
//!
//! ```
//! use katar::traits::Simd;
//! use katar::{Arch, SimdFn};
//!
//! fn sum(data: &[f32]) -> f32 {
//!     struct Sum<'a>(&'a [f32]);
//!
//!     impl SimdFn for Sum<'_> {
//!         type Result = f32;
//!
//!         #[inline(always)]
//!         fn call<S: Simd>(self, simd: S) -> f32 {
//!             let (vectors, tail) = katar::utils::as_simd_f32::<4, _>(&simd, self.0);
//!
//!             let mut acc = simd.splat_f32(0.0);
//!             for block in vectors {
//!                 for v in block {
//!                     acc = simd.add_f32(acc, v);
//!                 }
//!             }
//!
//!             let mut total = simd.reduce_sum_f32(acc);
//!             for x in tail {
//!                 total += x;
//!             }
//!             total
//!         }
//!     }
//!
//!     Arch::new().dispatch(Sum(data))
//! }
//!
//! assert_eq!(sum(&[1.0, 2.0, 3.0]), 6.0);
//! ```
//!
//! # Writing portable code
//!
//! Lane counts are backend-dependent: use `S::*_lanes` and the helpers in [`utils`] instead of
//! assuming a vector width. Put loops inside [`SimdFn::call`] so dispatch is paid once per
//! operation, and reuse one [`Arch`] instead of constructing it in a loop.
//!
//! For hot paths, mark `call` and its small helper functions `#[inline(always)]` so the
//! backend-specific loop body is inlined into the dispatched target-feature function rather than
//! remaining an opaque call. Without this the backend intrinsics cannot be inlined either, and the
//! operation can become tens of times slower.
//!
//! Dispatch is safe: hardware backends are entered only after [`Arch::new`] has detected their
//! features. Only the `load_unchecked_*` and `store_unchecked_*` operations are `unsafe`.
#![no_std]
#![allow(non_upper_case_globals, non_camel_case_types, clippy::new_without_default)]
use crate::{backend::fallback::ScalarFallback, traits::Simd};

mod backend;
pub mod traits;
pub mod utils;

/// A portable SIMD operation.
///
/// Implement `call` using only [`traits::Simd`] operations and pass the value to
/// [`Arch::dispatch`]. `call` is instantiated once per available backend.
pub trait SimdFn {
    /// Operation result.
    type Result;
    /// Runs the operation with the given backend.
    ///
    /// Mark the implementation `#[inline(always)]`. Dispatch calls this method from a function
    /// compiled with the backend's target features; if it is not inlined, the backend intrinsics
    /// cannot be inlined into that function either and every SIMD operation turns into an
    /// out-of-line call, which can be tens of times slower.
    fn call<S: Simd>(self, simd: S) -> Self::Result;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    #[allow(dead_code)]
    ScalarFallback,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512F,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512Ex,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2Fma,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// Runtime CPU feature detector and dispatcher.
///
/// Create once with [`Arch::new`] and reuse it for repeated [`Arch::dispatch`] calls.
#[derive(Clone, Copy)]
pub struct Arch {
    backend: Backend,
}

impl Arch {
    /// Detects the best available backend, falling back to scalar.
    pub fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Arch { backend: Backend::Neon }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if core_detect::is_x86_feature_detected!("avx512f") && core_detect::is_x86_feature_detected!("avx512bw") && core_detect::is_x86_feature_detected!("avx512dq") {
                if core_detect::is_x86_feature_detected!("avx512cd")
                    && core_detect::is_x86_feature_detected!("avx512bitalg")
                    && core_detect::is_x86_feature_detected!("avx512vpopcntdq")
                {
                    return Arch { backend: Backend::Avx512Ex };
                }
                return Arch { backend: Backend::Avx512F };
            }

            if core_detect::is_x86_feature_detected!("avx2") && core_detect::is_x86_feature_detected!("fma") {
                return Arch { backend: Backend::Avx2Fma };
            }

            if core_detect::is_x86_feature_detected!("sse2") {
                return Arch { backend: Backend::Sse2 };
            }

            Arch { backend: Backend::ScalarFallback }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        Arch { backend: Backend::ScalarFallback }
    }

    /// Runs `op` on the selected backend.
    ///
    /// Put loops inside the [`SimdFn`] implementation so dispatch happens once per operation.
    #[inline(always)]
    pub fn dispatch<Op: SimdFn>(&self, op: Op) -> Op::Result {
        match self.backend {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512Ex => unsafe { backend::x86::avx512::vectorize_ex(op) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512F => unsafe { backend::x86::avx512::vectorize(op) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx2Fma => unsafe { backend::x86::avx2::vectorize(op) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => unsafe { backend::x86::sse2::vectorize(op) },
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => unsafe { backend::aarch64::neon::vectorize(op) },
            Backend::ScalarFallback => op.call(ScalarFallback),
        }
    }
}
