# katar

[![CI](https://img.shields.io/github/actions/workflow/status/maslenn1koff/katar/rust.yml?style=flat&label=CI&labelColor=white&color=white)](https://github.com/maslenn1koff/katar/actions/workflows/rust.yml)
[![License](https://img.shields.io/badge/license-MIT-white?style=flat&labelColor=white&color=white)](LICENSE.md)
![MSRV](https://img.shields.io/badge/MSRV-1.89-white?style=flat&labelColor=white&color=white)

**Portable SIMD abstraction for Rust.**

`katar` provides architecture-independent SIMD operations with runtime CPU dispatch. It supports floating-point and integer types through a common trait interface, with implementations for x86, x86_64, and aarch64.

The library is `no_std` and does not require dynamic allocation.

## Usage

### Manual Vectorization

SIMD operations are implemented using `SimdFn`, which defines an operation generic over `traits::Simd`.

The following example implements a vectorized sum:

```rust
use katar::traits::Simd;
use katar::{Arch, SimdFn};

struct Sum<'a>(&'a [f32]);

impl SimdFn for Sum<'_> {
    type Result = f32;

    #[inline(always)]
    fn call<S: Simd>(self, simd: S) -> Self::Result {
        let (vectors, tail) =
            katar::utils::as_simd_f32::<4, _>(&simd, self.0);

        let mut acc = simd.splat_f32(0.0);

        for block in vectors {
            for vector in block {
                acc = simd.add_f32(acc, vector);
            }
        }

        let mut sum = simd.reduce_sum_f32(acc);

        for &x in tail {
            sum += x;
        }

        sum
    }
}

fn main() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let sum = Arch::new().dispatch(Sum(&data));

    assert_eq!(sum, 10.0);
}
```

`SimdFn::call` is monomorphized for each available backend. The `Simd` parameter provides access to the operations supported by that backend.

### Runtime Dispatch

`Arch::new().dispatch()` selects a backend using runtime CPU feature detection.

On x86 and x86_64, detection uses CPUID and XCR0 to verify both processor capabilities and operating system support. On aarch64, NEON is used as the baseline backend.
