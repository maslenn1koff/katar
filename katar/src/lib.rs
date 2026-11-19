#![no_std]
#![allow(non_upper_case_globals, non_camel_case_types)]
use crate::{backend::fallback::ScalarFallback, traits::Simd};

mod backend;
pub mod traits;

pub trait SimdFn {
    type Result;
    fn call<S: Simd>(self, simd: S) -> Self::Result;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    ScalarFallback,
    Avx2Fma,
}

#[derive(Clone, Copy)]
pub struct Arch {
    backend: Backend,
}

impl Default for Arch {
    fn default() -> Self {
        Arch { backend: Backend::ScalarFallback }
    }
}

impl Arch {
    pub fn new() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if core_detect::is_x86_feature_detected!("avx2") && core_detect::is_x86_feature_detected!("fma") {
                return Arch { backend: Backend::Avx2Fma };
            }
        }

        Arch { backend: Backend::ScalarFallback }
    }

    pub fn dispatch<Op: SimdFn>(&self, op: Op) -> Op::Result {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if self.backend == Backend::Avx2Fma {
                return unsafe { backend::x86::avx2::vectorize(op) };
            }
        }

        op.call(ScalarFallback)
    }
}
