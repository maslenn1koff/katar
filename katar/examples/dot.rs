fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    struct Impl<'a>(&'a [f32], &'a [f32]);

    impl<'a> katar::SimdFn for Impl<'a> {
        type Result = f32;

        #[inline(always)]
        fn call<S: katar::traits::Simd>(self, simd: S) -> Self::Result {
            assert_eq!(self.0.len(), self.1.len());

            let (x_head, x_tail) = katar::utils::as_simd_f32(&simd, self.0);
            let (y_head, y_tail) = katar::utils::as_simd_f32(&simd, self.1);

            let mut acc = [simd.splat_f32(0.0); 4];

            let mut vectors = x_head.zip(y_head);
            while vectors.len() >= 4 {
                let (x0, y0) = vectors.next().unwrap();
                let (x1, y1) = vectors.next().unwrap();
                let (x2, y2) = vectors.next().unwrap();
                let (x3, y3) = vectors.next().unwrap();

                acc[0] = simd.mul_add_f32(x0, y0, acc[0]);
                acc[1] = simd.mul_add_f32(x1, y1, acc[1]);
                acc[2] = simd.mul_add_f32(x2, y2, acc[2]);
                acc[3] = simd.mul_add_f32(x3, y3, acc[3]);
            }

            for (x, y) in vectors {
                acc[0] = simd.mul_add_f32(x, y, acc[0]);
            }

            acc[0] = simd.add_f32(acc[0], acc[1]);
            acc[1] = simd.add_f32(acc[2], acc[3]);
            acc[0] = simd.add_f32(acc[0], acc[1]);
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
