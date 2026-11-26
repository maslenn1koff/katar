#![no_std]
#![allow(non_upper_case_globals, non_camel_case_types, clippy::new_without_default)]
use crate::{backend::fallback::ScalarFallback, traits::Simd};

mod backend;
pub mod traits;
pub mod utils;

pub trait SimdFn {
    type Result;
    fn call<S: Simd>(self, simd: S) -> Self::Result;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    #[allow(dead_code)]
    ScalarFallback,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512F,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2Fma,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

#[derive(Clone, Copy)]
pub struct Arch {
    backend: Backend,
}

impl Arch {
    pub fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Arch { backend: Backend::Neon }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if core_detect::is_x86_feature_detected!("avx512f") && core_detect::is_x86_feature_detected!("avx512bw") && core_detect::is_x86_feature_detected!("avx512dq") {
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

    pub fn dispatch<Op: SimdFn>(&self, op: Op) -> Op::Result {
        match self.backend {
            Backend::ScalarFallback => op.call(ScalarFallback),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512F => unsafe { backend::x86::avx512::vectorize(op) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx2Fma => unsafe { backend::x86::avx2::vectorize(op) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => unsafe { backend::x86::sse2::vectorize(op) },
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => unsafe { backend::aarch64::neon::vectorize(op) },
        }
    }
}
