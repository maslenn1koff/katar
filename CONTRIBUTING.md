# Contributing to katar

Thanks for your interest in contributing to `katar`! Contributions of bug
reports, fixes, tests, documentation, and performance improvements are welcome.

Before participating, please read the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting started

`katar` is a Rust crate located in the `katar/` directory. It requires Rust
1.89 or newer (edition 2024) with the `rustfmt` and `clippy` components. Run
development commands from that directory:

```sh
cd katar
cargo test
```

The crate is `no_std` and does not require dynamic allocation. Keep it that way:
`std` may only be used in tests, benchmarks, and examples.

## Before opening a pull request

Run the same checks used by CI:

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features
cargo test --all --all-features
cargo test --release --all --all-features
```

CI also builds the documentation with warnings denied:

```sh
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
```

CI treats compiler warnings as errors. To check locally with the same setting,
run the commands with `RUSTFLAGS="-D warnings"`.

The MSRV (1.89.0) is checked separately, so avoid using newer language or
library features. The no_std build is verified for the following targets:

```sh
cargo build --release --target thumbv7em-none-eabihf --lib --all-features
cargo build --release --target wasm32-unknown-unknown --lib --all-features
```

The test matrix covers x86_64 and i686 Linux, aarch64 and x86_64 macOS, aarch64
Linux, and x86_64 Windows. If you have the relevant targets installed, check
them explicitly, for example:

```sh
cargo test --target aarch64-unknown-linux-gnu --all --all-features
```

The default `cargo test` only exercises the backends supported by the host CPU.
Changes touching an architecture-specific backend should be validated on that
architecture, or at least checked with the corresponding target.

## Code and tests

- Keep changes focused and follow the existing Rust style. `rustfmt.toml`
  configures formatting (`max_width = 180`).
- Public API changes should include clear rustdoc and, where useful, an
  example. Keep the crate-level documentation in `src/lib.rs` accurate.
- Implement operations through the generic `traits::Simd` interface (and its
  `FloatSimd`, `IntegerSimd`, and `MaskSimd` supertraits) so the same operation
  runs on every backend: AVX-512, AVX2+FMA, SSE2, NEON, and the scalar fallback.
- Use the helpers in `utils` such as `utils::as_simd_f32` or
  `utils::as_simd_f64` to split slices into complete vectors and a scalar tail.
  Do not assume a fixed vector width: lane counts are backend-dependent and
  available through the `S::*_lanes` associated constants.
- Mark `SimdFn::call` implementations and their small helper functions with
  `#[inline(always)]`. Without inlining, backend intrinsics cannot propagate
  into the target-feature function and the operation can become tens of times
  slower.
- Put loops inside `SimdFn::call` so dispatch happens once per operation, and
  reuse an `Arch` instead of constructing one inside a loop.
- Add or update tests for behavioral changes. Backend tests live in
  `src/backend/tests.rs`, and backend-selection/utility tests live alongside
  their modules. Account for backend-dependent floating-point rounding in
  reductions and fused multiply-add operations; compare floating-point results
  with an appropriate tolerance rather than assuming bit-for-bit equality.
  Note that `mul_add` may map to a fused or non-fused operation depending on
  the backend.
- Keep architecture-specific operations behind the existing backend structure
  in `src/backend/` and ensure runtime feature detection in `Arch::new` remains
  consistent with the target features used by each implementation.
- Keep dispatch safe: only the `load_unchecked_*` and `store_unchecked_*`
  operations should be `unsafe`. Document any additional safety requirements.

## Examples and benchmarks

Examples live in `examples/` and should keep working:

```sh
cargo run --example dot
cargo run --example ascii
```

Benchmarks use Criterion and compare `katar` against `pulp` and
`fearless_simd`:

```sh
cargo bench
```

When reporting performance changes, include the CPU model, the backend that was
selected, and before/after numbers. Benchmark results are collected in
`assets/` for the README.

## Pull requests

Please open a pull request against `main` with a concise description of the
change, its motivation, and the checks you ran. Include relevant test results
and call out any public API, `no_std`, or floating-point behavior changes.
