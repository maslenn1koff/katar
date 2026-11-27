fn ascii_lowercase(data: &mut [u8]) {
    struct Impl<'a>(&'a mut [u8]);

    impl<'a> katar::SimdFn for Impl<'a> {
        type Result = ();

        fn call<S: katar::traits::Simd>(self, simd: S) -> Self::Result {
            let case_bit = simd.splat_u8(0x20);

            let mut i = 0;
            while i + S::u8_lanes <= self.0.len() {
                let x = unsafe { simd.load_unchecked_u8(&self.0[i..]) };
                let x_conv = simd.bitor_u8(x, case_bit);

                let mask = simd.lt_u8(simd.sub_u8(x_conv, simd.splat_u8(b'a')), simd.splat_u8(26));
                let result = simd.select_u8(mask, x_conv, x);
                unsafe { simd.store_unchecked_u8(&mut self.0[i..], result) };

                i += S::u8_lanes;
            }

            for byte in &mut self.0[i..] {
                if byte.is_ascii_uppercase() {
                    *byte |= 0x20;
                }
            }
        }
    }

    let arch = katar::Arch::new();
    arch.dispatch(Impl(data));
}

fn main() {
    let mut raw_str = String::from("HELLo WoRlD!!! :3");
    println!("raw: {raw_str}");

    ascii_lowercase(unsafe { raw_str.as_bytes_mut() });

    println!("result: {raw_str}");
}
