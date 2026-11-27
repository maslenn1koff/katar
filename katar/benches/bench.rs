use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fearless_simd::{Level, Select, Simd as FearlessSimd, SimdBase, SimdFloat};
use katar::traits::Simd as KatarSimd;
use pulp::{Arch as PulpArch, Simd as PulpSimd, WithSimd};
use std::hint::black_box;
use std::time::Duration;

const DOT_SIZES: &[usize] = &[512, 1_024, 16_384, 65_536, 1_048_576];
const ASCII_SIZES: &[usize] = &[512, 1_024, 16_384, 65_536, 1_048_576];

fn make_f32_data(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let unit = ((bits >> 40) as u32) as f32 / 16_777_215.0;
            unit.mul_add(2.0, -1.0)
        })
        .collect()
}

fn make_ascii_data(len: usize) -> Vec<u8> {
    (0..len)
        .map(|index| match index % 8 {
            0 | 1 => b'A' + (index % 26) as u8,
            2 | 3 => b'a' + (index % 26) as u8,
            4 | 5 => b'0' + (index % 10) as u8,
            6 => b'!',
            _ => 0x80 + (index % 0x80) as u8,
        })
        .collect()
}

#[inline(always)]
fn dot_katar<S: KatarSimd>(simd: S, x: &[f32], y: &[f32]) -> f32 {
    let (x_head, x_tail) = katar::utils::as_simd_f32::<8, _>(&simd, x);
    let (y_head, y_tail) = katar::utils::as_simd_f32::<8, _>(&simd, y);
    let zero = simd.splat_f32(0.0);
    let mut acc = [zero; 8];
    for (a, b) in x_head.zip(y_head) {
        for i in 0..8 {
            acc[i] = simd.mul_add_f32(a[i], b[i], acc[i]);
        }
    }
    acc[0] = simd.add_f32(acc[0], acc[1]);
    acc[2] = simd.add_f32(acc[2], acc[3]);
    acc[4] = simd.add_f32(acc[4], acc[5]);
    acc[6] = simd.add_f32(acc[6], acc[7]);
    acc[0] = simd.add_f32(acc[0], acc[2]);
    acc[4] = simd.add_f32(acc[4], acc[6]);
    acc[0] = simd.add_f32(acc[0], acc[4]);
    let mut sum = simd.reduce_sum_f32(acc[0]);
    for (&a, &b) in x_tail.iter().zip(y_tail) {
        sum += a * b;
    }
    sum
}

struct KatarDot<'a>(&'a [f32], &'a [f32]);
impl katar::SimdFn for KatarDot<'_> {
    type Result = f32;
    #[inline(always)]
    fn call<S: KatarSimd>(self, simd: S) -> f32 {
        dot_katar(simd, self.0, self.1)
    }
}

#[inline(always)]
fn dot_pulp<S: PulpSimd>(simd: S, x: &[f32], y: &[f32]) -> f32 {
    let (xs, x_tail) = S::as_simd_f32s(x);
    let (ys, y_tail) = S::as_simd_f32s(y);
    let zero = simd.splat_f32s(0.0);
    let mut acc = [zero; 8];
    let (xs, x_remainder) = xs.as_chunks::<8>();
    let (ys, y_remainder) = ys.as_chunks::<8>();
    for (a, b) in xs.iter().zip(ys) {
        for i in 0..8 {
            acc[i] = simd.mul_add_f32s(a[i], b[i], acc[i]);
        }
    }
    for (index, (a, b)) in x_remainder.iter().zip(y_remainder).enumerate() {
        acc[index] = simd.mul_add_f32s(*a, *b, acc[index]);
    }
    acc[0] = simd.add_f32s(acc[0], acc[1]);
    acc[2] = simd.add_f32s(acc[2], acc[3]);
    acc[4] = simd.add_f32s(acc[4], acc[5]);
    acc[6] = simd.add_f32s(acc[6], acc[7]);
    acc[0] = simd.add_f32s(acc[0], acc[2]);
    acc[4] = simd.add_f32s(acc[4], acc[6]);
    acc[0] = simd.add_f32s(acc[0], acc[4]);
    let mut sum = simd.reduce_sum_f32s(acc[0]);
    for (&a, &b) in x_tail.iter().zip(y_tail) {
        sum += a * b;
    }
    sum
}

struct PulpDot<'a>(&'a [f32], &'a [f32]);
impl WithSimd for PulpDot<'_> {
    type Output = f32;
    #[inline(always)]
    fn with_simd<S: PulpSimd>(self, simd: S) -> f32 {
        dot_pulp(simd, self.0, self.1)
    }
}

#[allow(clippy::needless_range_loop)]
#[inline(always)]
fn dot_fearless<S: FearlessSimd>(simd: S, x: &[f32], y: &[f32]) -> f32 {
    let zero = S::f32s::splat(simd, 0.0);
    let mut acc = [zero; 8];
    let lanes = S::f32s::LEN;
    let block_len = 8 * lanes;
    let mut x_blocks = x.chunks_exact(block_len);
    let mut y_blocks = y.chunks_exact(block_len);
    for (a, b) in x_blocks.by_ref().zip(y_blocks.by_ref()) {
        for i in 0..8 {
            let lo = i * lanes;
            let hi = lo + lanes;
            acc[i] = S::f32s::from_slice(simd, &a[lo..hi]).mul_add(S::f32s::from_slice(simd, &b[lo..hi]), acc[i]);
        }
    }
    let mut x_vectors = x_blocks.remainder().chunks_exact(lanes);
    let mut y_vectors = y_blocks.remainder().chunks_exact(lanes);
    for (index, (a, b)) in x_vectors.by_ref().zip(y_vectors.by_ref()).enumerate() {
        acc[index] = S::f32s::from_slice(simd, a).mul_add(S::f32s::from_slice(simd, b), acc[index]);
    }
    acc[0] += acc[1];
    acc[2] += acc[3];
    acc[4] += acc[5];
    acc[6] += acc[7];
    acc[0] += acc[2];
    acc[4] += acc[6];
    acc[0] += acc[4];
    let mut sum = acc[0].reduce_sum();
    for (&a, &b) in x_vectors.remainder().iter().zip(y_vectors.remainder()) {
        sum += a * b;
    }
    sum
}

#[fearless_simd_macros::simd]
#[inline(always)]
fn fearless_dot<S: FearlessSimd>(simd: S, x: &[f32], y: &[f32]) -> f32 {
    dot_fearless(simd, x, y)
}

#[inline(always)]
fn lowercase_scalar(data: &mut [u8]) {
    for byte in data {
        if byte.is_ascii_uppercase() {
            *byte |= 0x20;
        }
    }
}

#[inline(always)]
fn lowercase_katar<S: KatarSimd>(simd: S, data: &mut [u8]) {
    let lanes = S::u8_lanes;
    let case_bit = simd.splat_u8(0x20);
    let lower_bound = simd.splat_u8(b'a');
    let range_len = simd.splat_u8(26);

    let mut blocks = data.chunks_exact_mut(4 * lanes);
    for block in &mut blocks {
        for chunk in block.chunks_exact_mut(lanes) {
            let value = unsafe { simd.load_unchecked_u8(chunk) };
            let lowered = simd.bitor_u8(value, case_bit);
            let in_ascii_alpha = simd.lt_u8(simd.sub_u8(lowered, lower_bound), range_len);
            let result = simd.select_u8(in_ascii_alpha, lowered, value);
            unsafe { simd.store_unchecked_u8(chunk, result) };
        }
    }
    lowercase_scalar(blocks.into_remainder());
}

struct KatarLowercase<'a>(&'a mut [u8]);
impl katar::SimdFn for KatarLowercase<'_> {
    type Result = ();
    #[inline(always)]
    fn call<S: KatarSimd>(self, simd: S) {
        lowercase_katar(simd, self.0);
    }
}

#[inline(always)]
fn lowercase_pulp<S: PulpSimd>(simd: S, data: &mut [u8]) {
    let (vectors, tail) = S::as_mut_simd_u8s(data);
    let case_bit = simd.splat_u8s(0x20);
    let upper_bound = simd.splat_u8s(b'A');
    let range_len = simd.splat_u8s(26);
    for vector in vectors {
        let value = *vector;
        let lowered = simd.or_u8s(value, case_bit);
        let is_uppercase = simd.less_than_u8s(simd.sub_u8s(value, upper_bound), range_len);
        let mask = simd.transmute_u8s_m8s(is_uppercase);
        let result = simd.or_u8s(simd.and_u8s(lowered, mask), simd.and_u8s(value, simd.not_u8s(mask)));
        *vector = result;
    }
    lowercase_scalar(tail);
}

struct PulpLowercase<'a>(&'a mut [u8]);
impl WithSimd for PulpLowercase<'_> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: PulpSimd>(self, simd: S) {
        lowercase_pulp(simd, self.0);
    }
}

#[inline(always)]
fn lowercase_fearless<S: FearlessSimd>(simd: S, data: &mut [u8]) {
    let mut chunks = data.chunks_exact_mut(S::u8s::LEN);
    for chunk in &mut chunks {
        let value = S::u8s::from_slice(simd, chunk);
        let lowered = value | 0x20;
        let in_ascii_alpha = (lowered - b'a').simd_lt(26);
        in_ascii_alpha.select(lowered, value).store_slice(chunk);
    }
    lowercase_scalar(chunks.into_remainder());
}

#[fearless_simd_macros::simd]
#[inline(always)]
fn fearless_lowercase<S: FearlessSimd>(simd: S, data: &mut [u8]) {
    lowercase_fearless(simd, data)
}

fn bench_dot(c: &mut Criterion) {
    let katar_arch = katar::Arch::new();
    let pulp_arch = PulpArch::new();
    let fearless_level = Level::new();

    let mut group = c.benchmark_group("dot_f32");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(50);

    for &size in DOT_SIZES {
        let x = make_f32_data(size, 0x1234_5678_9ABC_DEF0);
        let y = make_f32_data(size, 0x0FED_CBA9_8765_4321);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("katar", size), &size, |b, _| {
            b.iter(|| black_box(katar_arch.dispatch(KatarDot(black_box(x.as_slice()), black_box(y.as_slice())))))
        });
        group.bench_with_input(BenchmarkId::new("pulp", size), &size, |b, _| {
            b.iter(|| black_box(pulp_arch.dispatch(PulpDot(black_box(x.as_slice()), black_box(y.as_slice())))))
        });
        group.bench_with_input(BenchmarkId::new("fearless_simd", size), &size, |b, _| {
            b.iter(|| {
                black_box(fearless_simd::dispatch!(
                    fearless_level,
                    simd => fearless_dot(simd, black_box(x.as_slice()), black_box(y.as_slice()))
                ))
            })
        });
    }
    group.finish();
}

fn bench_ascii_lowercase(c: &mut Criterion) {
    let katar_arch = katar::Arch::new();
    let pulp_arch = PulpArch::new();
    let fearless_level = Level::new();

    let mut group = c.benchmark_group("ascii_lowercase");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(50);

    for &size in ASCII_SIZES {
        let input = make_ascii_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("katar", size), &input, |b, input| {
            b.iter_batched_ref(
                || input.clone(),
                |data| {
                    katar_arch.dispatch(KatarLowercase(black_box(data.as_mut_slice())));
                    black_box(data.as_slice());
                },
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("pulp", size), &input, |b, input| {
            b.iter_batched_ref(
                || input.clone(),
                |data| {
                    pulp_arch.dispatch(PulpLowercase(black_box(data.as_mut_slice())));
                    black_box(data.as_slice());
                },
                BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("fearless_simd", size), &input, |b, input| {
            b.iter_batched_ref(
                || input.clone(),
                |data| {
                    fearless_simd::dispatch!(
                        fearless_level,
                        simd => fearless_lowercase(simd, black_box(data.as_mut_slice()))
                    );
                    black_box(data.as_slice());
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_dot, bench_ascii_lowercase
}
criterion_main!(benches);
