fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());

    struct Impl<'a>(&'a [f32], &'a [f32]);

    impl<'a> katar::SimdFn for Impl<'a> {
        type Result = f32;

        #[inline(always)]
        fn call<S: katar::traits::Simd>(self, simd: S) -> Self::Result {
            let (x_head, x_tail) = katar::utils::as_simd_f32::<8, _>(&simd, self.0);
            let (y_head, y_tail) = katar::utils::as_simd_f32::<8, _>(&simd, self.1);

            let mut acc = [simd.splat_f32(0.0); 8];

            for (x, y) in x_head.zip(y_head) {
                acc[0] = simd.mul_add_f32(x[0], y[0], acc[0]);
                acc[1] = simd.mul_add_f32(x[1], y[1], acc[1]);
                acc[2] = simd.mul_add_f32(x[2], y[2], acc[2]);
                acc[3] = simd.mul_add_f32(x[3], y[3], acc[3]);
                acc[4] = simd.mul_add_f32(x[4], y[4], acc[4]);
                acc[5] = simd.mul_add_f32(x[5], y[5], acc[5]);
                acc[6] = simd.mul_add_f32(x[6], y[6], acc[6]);
                acc[7] = simd.mul_add_f32(x[7], y[7], acc[7]);
            }

            acc[0] = simd.add_f32(acc[0], acc[1]);
            acc[2] = simd.add_f32(acc[2], acc[3]);
            acc[4] = simd.add_f32(acc[4], acc[5]);
            acc[6] = simd.add_f32(acc[6], acc[7]);
            acc[0] = simd.add_f32(acc[0], acc[2]);
            acc[4] = simd.add_f32(acc[4], acc[6]);
            acc[0] = simd.add_f32(acc[0], acc[4]);

            let mut product = simd.reduce_sum_f32(acc[0]);
            for (x, y) in x_tail.iter().zip(y_tail.iter()) {
                product += x * y;
            }

            product
        }
    }

    let arch = katar::Arch::new();
    arch.dispatch(Impl(x, y))
}

fn main() {
    let x = vec![1.0; 4096];
    let y = vec![2.0; 4096];

    let product = dot_f32(&x, &y);
    println!("dot product: {product}");
}
