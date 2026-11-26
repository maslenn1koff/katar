use crate::{
    SimdFn,
    traits::{FloatSimd, IntegerSimd, MaskSimd, Simd},
};
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[target_feature(enable = "sse2")]
pub fn vectorize<Op: SimdFn>(op: Op) -> Op::Result {
    op.call(Sse2)
}

#[derive(Copy, Clone, Debug)]
pub struct Sse2MaskF32(__m128);

impl MaskSimd for Sse2MaskF32 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        unsafe { Self(_mm_castsi128_ps(_mm_set1_epi32(if v { -1 } else { 0 }))) }
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        unsafe {
            Self(_mm_castsi128_ps(_mm_set_epi32(
                if v & 8 != 0 { -1 } else { 0 },
                if v & 4 != 0 { -1 } else { 0 },
                if v & 2 != 0 { -1 } else { 0 },
                if v & 1 != 0 { -1 } else { 0 },
            )))
        }
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        unsafe { _mm_movemask_ps(self.0) as u64 }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(_mm_and_ps(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(_mm_or_ps(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(_mm_xor_ps(self.0, _mm_castsi128_ps(_mm_set1_epi32(-1)))) }
    }

    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm_movemask_ps(self.0) != 0 }
    }

    #[inline(always)]
    fn all(self) -> bool {
        unsafe { _mm_movemask_ps(self.0) == 0b1111 }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Sse2MaskF64(__m128d);

impl MaskSimd for Sse2MaskF64 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        unsafe { Self(_mm_castsi128_pd(_mm_set1_epi32(if v { -1 } else { 0 }))) }
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        unsafe { Self(_mm_castsi128_pd(_mm_set_epi64x(if v & 2 != 0 { -1 } else { 0 }, if v & 1 != 0 { -1 } else { 0 }))) }
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        unsafe { _mm_movemask_pd(self.0) as u64 }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(_mm_and_pd(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(_mm_or_pd(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(_mm_xor_pd(self.0, _mm_castsi128_pd(_mm_set1_epi32(-1)))) }
    }

    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm_movemask_pd(self.0) != 0 }
    }

    #[inline(always)]
    fn all(self) -> bool {
        unsafe { _mm_movemask_pd(self.0) == 0b11 }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Sse2MaskI8(__m128i);

impl MaskSimd for Sse2MaskI8 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        unsafe { Self(_mm_set1_epi8(if v { -1 } else { 0 })) }
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        unsafe {
            Self(_mm_set_epi8(
                if v & 0x8000 != 0 { -1 } else { 0 },
                if v & 0x4000 != 0 { -1 } else { 0 },
                if v & 0x2000 != 0 { -1 } else { 0 },
                if v & 0x1000 != 0 { -1 } else { 0 },
                if v & 0x0800 != 0 { -1 } else { 0 },
                if v & 0x0400 != 0 { -1 } else { 0 },
                if v & 0x0200 != 0 { -1 } else { 0 },
                if v & 0x0100 != 0 { -1 } else { 0 },
                if v & 0x0080 != 0 { -1 } else { 0 },
                if v & 0x0040 != 0 { -1 } else { 0 },
                if v & 0x0020 != 0 { -1 } else { 0 },
                if v & 0x0010 != 0 { -1 } else { 0 },
                if v & 0x0008 != 0 { -1 } else { 0 },
                if v & 0x0004 != 0 { -1 } else { 0 },
                if v & 0x0002 != 0 { -1 } else { 0 },
                if v & 0x0001 != 0 { -1 } else { 0 },
            ))
        }
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        unsafe { _mm_movemask_epi8(self.0) as u32 as u64 }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(_mm_and_si128(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(_mm_or_si128(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(_mm_xor_si128(self.0, _mm_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.to_bits() != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.to_bits() == 0xFFFF
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Sse2MaskI16(__m128i);

impl MaskSimd for Sse2MaskI16 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        unsafe { Self(_mm_set1_epi16(if v { -1 } else { 0 })) }
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        unsafe {
            Self(_mm_set_epi16(
                if v & 0x80 != 0 { -1 } else { 0 },
                if v & 0x40 != 0 { -1 } else { 0 },
                if v & 0x20 != 0 { -1 } else { 0 },
                if v & 0x10 != 0 { -1 } else { 0 },
                if v & 0x08 != 0 { -1 } else { 0 },
                if v & 0x04 != 0 { -1 } else { 0 },
                if v & 0x02 != 0 { -1 } else { 0 },
                if v & 0x01 != 0 { -1 } else { 0 },
            ))
        }
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        unsafe { (_mm_movemask_epi8(_mm_packs_epi16(self.0, self.0)) as u32 & 0xFF) as u64 }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(_mm_and_si128(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(_mm_or_si128(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(_mm_xor_si128(self.0, _mm_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.to_bits() != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.to_bits() == 0xFF
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Sse2MaskI32(__m128i);

impl MaskSimd for Sse2MaskI32 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        unsafe { Self(_mm_set1_epi32(if v { -1 } else { 0 })) }
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        unsafe {
            Self(_mm_set_epi32(
                if v & 8 != 0 { -1 } else { 0 },
                if v & 4 != 0 { -1 } else { 0 },
                if v & 2 != 0 { -1 } else { 0 },
                if v & 1 != 0 { -1 } else { 0 },
            ))
        }
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(self.0)) as u32 as u64 }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(_mm_and_si128(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(_mm_or_si128(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(_mm_xor_si128(self.0, _mm_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.to_bits() != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.to_bits() == 0xF
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Sse2MaskI64(__m128i);

impl MaskSimd for Sse2MaskI64 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        unsafe { Self(_mm_set1_epi64x(if v { -1 } else { 0 })) }
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        unsafe { Self(_mm_set_epi64x(if v & 2 != 0 { -1 } else { 0 }, if v & 1 != 0 { -1 } else { 0 })) }
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        unsafe { _mm_movemask_pd(_mm_castsi128_pd(self.0)) as u32 as u64 }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(_mm_and_si128(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(_mm_or_si128(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(_mm_xor_si128(self.0, _mm_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.to_bits() != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.to_bits() == 0x3
    }
}

#[derive(Clone, Copy)]
pub struct Sse2;

impl FloatSimd for Sse2 {
    type f32 = __m128;
    type f64 = __m128d;
    type m32 = Sse2MaskF32;
    type m64 = Sse2MaskF64;

    const f32_lanes: usize = 4;
    const f64_lanes: usize = 2;

    #[inline(always)]
    fn splat_f32(self, value: f32) -> Self::f32 {
        unsafe { _mm_set1_ps(value) }
    }

    #[inline(always)]
    fn splat_f64(self, value: f64) -> Self::f64 {
        unsafe { _mm_set1_pd(value) }
    }

    #[inline(always)]
    fn load_f32(self, src: &[f32]) -> Self::f32 {
        assert!(src.len() >= Self::f32_lanes);
        unsafe { _mm_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    fn load_f64(self, src: &[f64]) -> Self::f64 {
        assert!(src.len() >= Self::f64_lanes);
        unsafe { _mm_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    fn store_f32(self, dst: &mut [f32], value: Self::f32) {
        assert!(dst.len() >= Self::f32_lanes);
        unsafe { _mm_storeu_ps(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    fn store_f64(self, dst: &mut [f64], value: Self::f64) {
        assert!(dst.len() >= Self::f64_lanes);
        unsafe { _mm_storeu_pd(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    unsafe fn load_unchecked_f32(self, src: &[f32]) -> Self::f32 {
        unsafe { _mm_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_f64(self, src: &[f64]) -> Self::f64 {
        unsafe { _mm_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f32(self, dst: &mut [f32], value: Self::f32) {
        unsafe { _mm_storeu_ps(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f64(self, dst: &mut [f64], value: Self::f64) {
        unsafe { _mm_storeu_pd(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    fn abs_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm_andnot_ps(_mm_set1_ps(-0.0), a) }
    }

    #[inline(always)]
    fn abs_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm_andnot_pd(_mm_set1_pd(-0.0), a) }
    }

    #[inline(always)]
    fn add_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_add_ps(a, b) }
    }

    #[inline(always)]
    fn add_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_add_pd(a, b) }
    }

    #[inline(always)]
    fn sub_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_sub_ps(a, b) }
    }

    #[inline(always)]
    fn sub_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_sub_pd(a, b) }
    }

    #[inline(always)]
    fn mul_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_mul_ps(a, b) }
    }

    #[inline(always)]
    fn mul_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_mul_pd(a, b) }
    }

    #[inline(always)]
    fn div_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_div_ps(a, b) }
    }

    #[inline(always)]
    fn div_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_div_pd(a, b) }
    }

    #[inline(always)]
    fn max_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_max_ps(a, b) }
    }

    #[inline(always)]
    fn max_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_max_pd(a, b) }
    }

    #[inline(always)]
    fn min_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_min_ps(a, b) }
    }

    #[inline(always)]
    fn min_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_min_pd(a, b) }
    }

    #[inline(always)]
    fn neg_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm_xor_ps(a, _mm_set1_ps(-0.0)) }
    }

    #[inline(always)]
    fn neg_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm_xor_pd(a, _mm_set1_pd(-0.0)) }
    }

    #[inline(always)]
    fn sqrt_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm_sqrt_ps(a) }
    }

    #[inline(always)]
    fn sqrt_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm_sqrt_pd(a) }
    }

    #[inline(always)]
    fn mul_add_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm_add_ps(_mm_mul_ps(a, b), c) }
    }

    #[inline(always)]
    fn mul_add_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm_add_pd(_mm_mul_pd(a, b), c) }
    }

    #[inline(always)]
    fn mul_sub_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm_sub_ps(_mm_mul_ps(a, b), c) }
    }

    #[inline(always)]
    fn mul_sub_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm_sub_pd(_mm_mul_pd(a, b), c) }
    }

    #[inline(always)]
    fn nmul_add_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm_sub_ps(c, _mm_mul_ps(a, b)) }
    }

    #[inline(always)]
    fn nmul_add_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm_sub_pd(c, _mm_mul_pd(a, b)) }
    }

    #[inline(always)]
    fn nmul_sub_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm_sub_ps(_mm_xor_ps(_mm_mul_ps(a, b), _mm_set1_ps(-0.0)), c) }
    }

    #[inline(always)]
    fn nmul_sub_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm_sub_pd(_mm_xor_pd(_mm_mul_pd(a, b), _mm_set1_pd(-0.0)), c) }
    }

    #[inline(always)]
    fn eq_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Sse2MaskF32(_mm_cmpeq_ps(a, b)) }
    }

    #[inline(always)]
    fn eq_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Sse2MaskF64(_mm_cmpeq_pd(a, b)) }
    }

    #[inline(always)]
    fn ge_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Sse2MaskF32(_mm_cmpge_ps(a, b)) }
    }

    #[inline(always)]
    fn ge_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Sse2MaskF64(_mm_cmpge_pd(a, b)) }
    }

    #[inline(always)]
    fn gt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Sse2MaskF32(_mm_cmpgt_ps(a, b)) }
    }

    #[inline(always)]
    fn gt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Sse2MaskF64(_mm_cmpgt_pd(a, b)) }
    }

    #[inline(always)]
    fn le_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Sse2MaskF32(_mm_cmple_ps(a, b)) }
    }

    #[inline(always)]
    fn le_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Sse2MaskF64(_mm_cmple_pd(a, b)) }
    }

    #[inline(always)]
    fn lt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Sse2MaskF32(_mm_cmplt_ps(a, b)) }
    }

    #[inline(always)]
    fn lt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Sse2MaskF64(_mm_cmplt_pd(a, b)) }
    }

    #[inline(always)]
    fn ne_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Sse2MaskF32(_mm_cmpneq_ps(a, b)) }
    }

    #[inline(always)]
    fn ne_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Sse2MaskF64(_mm_cmpneq_pd(a, b)) }
    }

    #[inline(always)]
    fn select_f32(self, mask: Self::m32, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_or_ps(_mm_and_ps(mask.0, a), _mm_andnot_ps(mask.0, b)) }
    }

    #[inline(always)]
    fn select_f64(self, mask: Self::m64, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_or_pd(_mm_and_pd(mask.0, a), _mm_andnot_pd(mask.0, b)) }
    }

    #[inline(always)]
    fn bitand_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_and_ps(a, b) }
    }

    #[inline(always)]
    fn bitor_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_or_ps(a, b) }
    }

    #[inline(always)]
    fn bitxor_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_xor_ps(a, b) }
    }

    #[inline(always)]
    fn bitand_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_and_pd(a, b) }
    }

    #[inline(always)]
    fn bitor_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_or_pd(a, b) }
    }

    #[inline(always)]
    fn bitxor_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_xor_pd(a, b) }
    }

    #[inline(always)]
    fn bitnot_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn bitnot_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm_xor_pd(a, _mm_castsi128_pd(_mm_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn bitandnot_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_andnot_ps(a, b) }
    }

    #[inline(always)]
    fn bitandnot_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_andnot_pd(a, b) }
    }

    #[inline(always)]
    fn reduce_max_f32(self, a: Self::f32) -> f32 {
        unsafe {
            let s = _mm_max_ps(a, _mm_movehl_ps(a, a));
            let s = _mm_max_ss(s, _mm_shuffle_ps::<0x55>(s, s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_max_f64(self, a: Self::f64) -> f64 {
        unsafe {
            let s = _mm_max_sd(a, _mm_unpackhi_pd(a, a));
            _mm_cvtsd_f64(s)
        }
    }

    #[inline(always)]
    fn reduce_min_f32(self, a: Self::f32) -> f32 {
        unsafe {
            let s = _mm_min_ps(a, _mm_movehl_ps(a, a));
            let s = _mm_min_ss(s, _mm_shuffle_ps::<0x55>(s, s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_min_f64(self, a: Self::f64) -> f64 {
        unsafe {
            let s = _mm_min_sd(a, _mm_unpackhi_pd(a, a));
            _mm_cvtsd_f64(s)
        }
    }

    #[inline(always)]
    fn reduce_sum_f32(self, a: Self::f32) -> f32 {
        unsafe {
            let s = _mm_add_ps(a, _mm_movehl_ps(a, a));
            let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_sum_f64(self, a: Self::f64) -> f64 {
        unsafe {
            let s = _mm_add_sd(a, _mm_unpackhi_pd(a, a));
            _mm_cvtsd_f64(s)
        }
    }
}

impl IntegerSimd for Sse2 {
    type u8 = __m128i;
    type i8 = __m128i;
    type u16 = __m128i;
    type i16 = __m128i;
    type u32 = __m128i;
    type i32 = __m128i;
    type u64 = __m128i;
    type i64 = __m128i;

    type m8 = Sse2MaskI8;
    type m16 = Sse2MaskI16;
    type m32 = Sse2MaskI32;
    type m64 = Sse2MaskI64;

    const u8_lanes: usize = 16;
    const i8_lanes: usize = 16;
    const u16_lanes: usize = 8;
    const i16_lanes: usize = 8;
    const u32_lanes: usize = 4;
    const i32_lanes: usize = 4;
    const u64_lanes: usize = 2;
    const i64_lanes: usize = 2;

    #[inline(always)]
    fn splat_u8(self, value: u8) -> Self::u8 {
        unsafe { _mm_set1_epi8(value as i8) }
    }

    #[inline(always)]
    fn splat_i8(self, value: i8) -> Self::i8 {
        unsafe { _mm_set1_epi8(value) }
    }

    #[inline(always)]
    fn splat_u16(self, value: u16) -> Self::u16 {
        unsafe { _mm_set1_epi16(value as i16) }
    }

    #[inline(always)]
    fn splat_i16(self, value: i16) -> Self::i16 {
        unsafe { _mm_set1_epi16(value) }
    }

    #[inline(always)]
    fn splat_u32(self, value: u32) -> Self::u32 {
        unsafe { _mm_set1_epi32(value as i32) }
    }

    #[inline(always)]
    fn splat_i32(self, value: i32) -> Self::i32 {
        unsafe { _mm_set1_epi32(value) }
    }

    #[inline(always)]
    fn splat_u64(self, value: u64) -> Self::u64 {
        unsafe { _mm_set1_epi64x(value as i64) }
    }

    #[inline(always)]
    fn splat_i64(self, value: i64) -> Self::i64 {
        unsafe { _mm_set1_epi64x(value) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_u8(self, src: &[u8]) -> Self::u8 {
        unsafe { _mm_loadu_si128(src.as_ptr() as *const __m128i) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_i8(self, src: &[i8]) -> Self::i8 {
        unsafe { _mm_loadu_si128(src.as_ptr() as *const __m128i) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_u16(self, src: &[u16]) -> Self::u16 {
        unsafe { _mm_loadu_si128(src.as_ptr() as *const __m128i) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_i16(self, src: &[i16]) -> Self::i16 {
        unsafe { _mm_loadu_si128(src.as_ptr() as *const __m128i) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_u32(self, src: &[u32]) -> Self::u32 {
        unsafe { _mm_loadu_si128(src.as_ptr() as *const __m128i) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_i32(self, src: &[i32]) -> Self::i32 {
        unsafe { _mm_loadu_si128(src.as_ptr() as *const __m128i) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_u64(self, src: &[u64]) -> Self::u64 {
        unsafe { _mm_loadu_si128(src.as_ptr() as *const __m128i) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_i64(self, src: &[i64]) -> Self::i64 {
        unsafe { _mm_loadu_si128(src.as_ptr() as *const __m128i) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_u8(self, dst: &mut [u8], value: Self::u8) {
        unsafe { _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_i8(self, dst: &mut [i8], value: Self::i8) {
        unsafe { _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_u16(self, dst: &mut [u16], value: Self::u16) {
        unsafe { _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_i16(self, dst: &mut [i16], value: Self::i16) {
        unsafe { _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_u32(self, dst: &mut [u32], value: Self::u32) {
        unsafe { _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_i32(self, dst: &mut [i32], value: Self::i32) {
        unsafe { _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_u64(self, dst: &mut [u64], value: Self::u64) {
        unsafe { _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_i64(self, dst: &mut [i64], value: Self::i64) {
        unsafe { _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, value) }
    }

    #[inline(always)]
    fn bitnot_u8(self, a: Self::u8) -> Self::u8 {
        unsafe { _mm_xor_si128(a, _mm_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn bitnot_i8(self, a: Self::i8) -> Self::i8 {
        unsafe { _mm_xor_si128(a, _mm_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn abs_i8(self, a: Self::i8) -> Self::i8 {
        unsafe {
            let sign = _mm_cmpgt_epi8(_mm_setzero_si128(), a);
            _mm_sub_epi8(_mm_xor_si128(a, sign), sign)
        }
    }

    #[inline(always)]
    fn neg_i8(self, a: Self::i8) -> Self::i8 {
        unsafe { _mm_sub_epi8(_mm_setzero_si128(), a) }
    }

    #[inline(always)]
    fn bitnot_u16(self, a: Self::u16) -> Self::u16 {
        unsafe { _mm_xor_si128(a, _mm_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn bitnot_i16(self, a: Self::i16) -> Self::i16 {
        unsafe { _mm_xor_si128(a, _mm_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn abs_i16(self, a: Self::i16) -> Self::i16 {
        unsafe {
            let sign = _mm_srai_epi16(a, 15);
            _mm_sub_epi16(_mm_xor_si128(a, sign), sign)
        }
    }

    #[inline(always)]
    fn neg_i16(self, a: Self::i16) -> Self::i16 {
        unsafe { _mm_sub_epi16(_mm_setzero_si128(), a) }
    }

    #[inline(always)]
    fn bitnot_u32(self, a: Self::u32) -> Self::u32 {
        unsafe { _mm_xor_si128(a, _mm_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn bitnot_i32(self, a: Self::i32) -> Self::i32 {
        unsafe { _mm_xor_si128(a, _mm_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn abs_i32(self, a: Self::i32) -> Self::i32 {
        unsafe {
            let sign = _mm_srai_epi32(a, 31);
            _mm_sub_epi32(_mm_xor_si128(a, sign), sign)
        }
    }

    #[inline(always)]
    fn neg_i32(self, a: Self::i32) -> Self::i32 {
        unsafe { _mm_sub_epi32(_mm_setzero_si128(), a) }
    }

    #[inline(always)]
    fn bitnot_u64(self, a: Self::u64) -> Self::u64 {
        unsafe { _mm_xor_si128(a, _mm_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn bitnot_i64(self, a: Self::i64) -> Self::i64 {
        unsafe { _mm_xor_si128(a, _mm_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn abs_i64(self, a: Self::i64) -> Self::i64 {
        let mut data = [0i64; 2];
        self.store_i64(&mut data, a);
        for value in data.iter_mut() {
            *value = value.wrapping_abs();
        }
        self.load_i64(&data)
    }

    #[inline(always)]
    fn neg_i64(self, a: Self::i64) -> Self::i64 {
        unsafe { _mm_sub_epi64(_mm_setzero_si128(), a) }
    }

    #[inline(always)]
    fn count_ones_u8(self, a: Self::u8) -> Self::u8 {
        let mut input = [0u8; 16];
        self.store_u8(&mut input, a);
        let mut output = [0u8; 16];
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = src.count_ones() as u8;
        }
        self.load_u8(&output)
    }

    #[inline(always)]
    fn count_ones_i8(self, a: Self::i8) -> Self::i8 {
        self.count_ones_u8(a)
    }

    #[inline(always)]
    fn count_ones_u16(self, a: Self::u16) -> Self::u16 {
        let mut input = [0u16; 8];
        self.store_u16(&mut input, a);
        let mut output = [0u16; 8];
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = src.count_ones() as u16;
        }
        self.load_u16(&output)
    }

    #[inline(always)]
    fn count_ones_i16(self, a: Self::i16) -> Self::i16 {
        self.count_ones_u16(a)
    }

    #[inline(always)]
    fn count_ones_u32(self, a: Self::u32) -> Self::u32 {
        let mut input = [0u32; 4];
        self.store_u32(&mut input, a);
        let mut output = [0u32; 4];
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = src.count_ones();
        }
        self.load_u32(&output)
    }

    #[inline(always)]
    fn count_ones_i32(self, a: Self::i32) -> Self::i32 {
        self.count_ones_u32(a)
    }

    #[inline(always)]
    fn count_ones_u64(self, a: Self::u64) -> Self::u64 {
        let mut input = [0u64; 2];
        self.store_u64(&mut input, a);
        let mut output = [0u64; 2];
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = src.count_ones() as u64;
        }
        self.load_u64(&output)
    }

    #[inline(always)]
    fn count_ones_i64(self, a: Self::i64) -> Self::i64 {
        self.count_ones_u64(a)
    }

    #[inline(always)]
    fn leading_zeros_u8(self, a: Self::u8) -> Self::u8 {
        let mut input = [0u8; 16];
        self.store_u8(&mut input, a);
        let mut output = [0u8; 16];
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = src.leading_zeros() as u8;
        }
        self.load_u8(&output)
    }

    #[inline(always)]
    fn leading_zeros_i8(self, a: Self::i8) -> Self::i8 {
        self.leading_zeros_u8(a)
    }

    #[inline(always)]
    fn leading_zeros_u16(self, a: Self::u16) -> Self::u16 {
        let mut input = [0u16; 8];
        self.store_u16(&mut input, a);
        let mut output = [0u16; 8];
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = src.leading_zeros() as u16;
        }
        self.load_u16(&output)
    }

    #[inline(always)]
    fn leading_zeros_i16(self, a: Self::i16) -> Self::i16 {
        self.leading_zeros_u16(a)
    }

    #[inline(always)]
    fn leading_zeros_u32(self, a: Self::u32) -> Self::u32 {
        let mut input = [0u32; 4];
        self.store_u32(&mut input, a);
        let mut output = [0u32; 4];
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = src.leading_zeros();
        }
        self.load_u32(&output)
    }

    #[inline(always)]
    fn leading_zeros_i32(self, a: Self::i32) -> Self::i32 {
        self.leading_zeros_u32(a)
    }

    #[inline(always)]
    fn leading_zeros_u64(self, a: Self::u64) -> Self::u64 {
        let mut input = [0u64; 2];
        self.store_u64(&mut input, a);
        let mut output = [0u64; 2];
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = src.leading_zeros() as u64;
        }
        self.load_u64(&output)
    }

    #[inline(always)]
    fn leading_zeros_i64(self, a: Self::i64) -> Self::i64 {
        self.leading_zeros_u64(a)
    }

    #[inline(always)]
    fn trailing_zeros_u8(self, a: Self::u8) -> Self::u8 {
        let mut input = [0u8; 16];
        self.store_u8(&mut input, a);
        let mut output = [0u8; 16];
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = src.trailing_zeros() as u8;
        }
        self.load_u8(&output)
    }

    #[inline(always)]
    fn trailing_zeros_i8(self, a: Self::i8) -> Self::i8 {
        self.trailing_zeros_u8(a)
    }

    #[inline(always)]
    fn trailing_zeros_u16(self, a: Self::u16) -> Self::u16 {
        let mut input = [0u16; 8];
        self.store_u16(&mut input, a);
        let mut output = [0u16; 8];
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = src.trailing_zeros() as u16;
        }
        self.load_u16(&output)
    }

    #[inline(always)]
    fn trailing_zeros_i16(self, a: Self::i16) -> Self::i16 {
        self.trailing_zeros_u16(a)
    }

    #[inline(always)]
    fn trailing_zeros_u32(self, a: Self::u32) -> Self::u32 {
        let mut input = [0u32; 4];
        self.store_u32(&mut input, a);
        let mut output = [0u32; 4];
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = src.trailing_zeros();
        }
        self.load_u32(&output)
    }

    #[inline(always)]
    fn trailing_zeros_i32(self, a: Self::i32) -> Self::i32 {
        self.trailing_zeros_u32(a)
    }

    #[inline(always)]
    fn trailing_zeros_u64(self, a: Self::u64) -> Self::u64 {
        let mut input = [0u64; 2];
        self.store_u64(&mut input, a);
        let mut output = [0u64; 2];
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = src.trailing_zeros() as u64;
        }
        self.load_u64(&output)
    }

    #[inline(always)]
    fn trailing_zeros_i64(self, a: Self::i64) -> Self::i64 {
        self.trailing_zeros_u64(a)
    }

    #[inline(always)]
    fn add_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm_add_epi8(a, b) }
    }

    #[inline(always)]
    fn sub_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm_sub_epi8(a, b) }
    }

    #[inline(always)]
    fn mul_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        let mut x = [0u8; 16];
        let mut y = [0u8; 16];
        self.store_u8(&mut x, a);
        self.store_u8(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.wrapping_mul(*src);
        }
        self.load_u8(&x)
    }

    #[inline(always)]
    fn min_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm_min_epu8(a, b) }
    }

    #[inline(always)]
    fn max_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm_max_epu8(a, b) }
    }

    #[inline(always)]
    fn bitand_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm_and_si128(a, b) }
    }

    #[inline(always)]
    fn bitor_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm_or_si128(a, b) }
    }

    #[inline(always)]
    fn bitxor_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm_xor_si128(a, b) }
    }

    #[inline(always)]
    fn bitandnot_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm_andnot_si128(a, b) }
    }

    #[inline(always)]
    fn saturating_add_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm_adds_epu8(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm_subs_epu8(a, b) }
    }

    #[inline(always)]
    fn avg_round_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm_avg_epu8(a, b) }
    }

    #[inline(always)]
    fn add_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm_add_epi8(a, b) }
    }

    #[inline(always)]
    fn sub_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm_sub_epi8(a, b) }
    }

    #[inline(always)]
    fn mul_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        let mut x = [0i8; 16];
        let mut y = [0i8; 16];
        self.store_i8(&mut x, a);
        self.store_i8(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.wrapping_mul(*src);
        }
        self.load_i8(&x)
    }

    #[inline(always)]
    fn min_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe {
            let sign = _mm_set1_epi8(i8::MIN);
            _mm_xor_si128(_mm_min_epu8(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign)), sign)
        }
    }

    #[inline(always)]
    fn max_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe {
            let sign = _mm_set1_epi8(i8::MIN);
            _mm_xor_si128(_mm_max_epu8(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign)), sign)
        }
    }

    #[inline(always)]
    fn bitand_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm_and_si128(a, b) }
    }

    #[inline(always)]
    fn bitor_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm_or_si128(a, b) }
    }

    #[inline(always)]
    fn bitxor_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm_xor_si128(a, b) }
    }

    #[inline(always)]
    fn bitandnot_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm_andnot_si128(a, b) }
    }

    #[inline(always)]
    fn saturating_add_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm_adds_epi8(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm_subs_epi8(a, b) }
    }

    #[inline(always)]
    fn avg_round_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe {
            let sign = _mm_set1_epi8(i8::MIN);
            _mm_xor_si128(_mm_avg_epu8(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign)), sign)
        }
    }

    #[inline(always)]
    fn add_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm_add_epi16(a, b) }
    }

    #[inline(always)]
    fn sub_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm_sub_epi16(a, b) }
    }

    #[inline(always)]
    fn mul_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm_mullo_epi16(a, b) }
    }

    #[inline(always)]
    fn min_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe {
            let sign = _mm_set1_epi16(i16::MIN);
            _mm_xor_si128(_mm_min_epi16(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign)), sign)
        }
    }

    #[inline(always)]
    fn max_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe {
            let sign = _mm_set1_epi16(i16::MIN);
            _mm_xor_si128(_mm_max_epi16(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign)), sign)
        }
    }

    #[inline(always)]
    fn bitand_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm_and_si128(a, b) }
    }

    #[inline(always)]
    fn bitor_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm_or_si128(a, b) }
    }

    #[inline(always)]
    fn bitxor_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm_xor_si128(a, b) }
    }

    #[inline(always)]
    fn bitandnot_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm_andnot_si128(a, b) }
    }

    #[inline(always)]
    fn saturating_add_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm_adds_epu16(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm_subs_epu16(a, b) }
    }

    #[inline(always)]
    fn avg_round_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm_avg_epu16(a, b) }
    }

    #[inline(always)]
    fn add_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm_add_epi16(a, b) }
    }

    #[inline(always)]
    fn sub_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm_sub_epi16(a, b) }
    }

    #[inline(always)]
    fn mul_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm_mullo_epi16(a, b) }
    }

    #[inline(always)]
    fn min_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm_min_epi16(a, b) }
    }

    #[inline(always)]
    fn max_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm_max_epi16(a, b) }
    }

    #[inline(always)]
    fn bitand_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm_and_si128(a, b) }
    }

    #[inline(always)]
    fn bitor_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm_or_si128(a, b) }
    }

    #[inline(always)]
    fn bitxor_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm_xor_si128(a, b) }
    }

    #[inline(always)]
    fn bitandnot_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm_andnot_si128(a, b) }
    }

    #[inline(always)]
    fn saturating_add_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm_adds_epi16(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm_subs_epi16(a, b) }
    }

    #[inline(always)]
    fn avg_round_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe {
            let sign = _mm_set1_epi16(i16::MIN);
            _mm_xor_si128(_mm_avg_epu16(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign)), sign)
        }
    }

    #[inline(always)]
    fn add_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm_add_epi32(a, b) }
    }

    #[inline(always)]
    fn sub_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm_sub_epi32(a, b) }
    }

    #[inline(always)]
    fn mul_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        let mut x = [0u32; 4];
        let mut y = [0u32; 4];
        self.store_u32(&mut x, a);
        self.store_u32(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.wrapping_mul(*src);
        }
        self.load_u32(&x)
    }

    #[inline(always)]
    fn min_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe {
            let sign = _mm_set1_epi32(i32::MIN);
            let mask = _mm_cmpgt_epi32(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign));
            _mm_or_si128(_mm_and_si128(mask, b), _mm_andnot_si128(mask, a))
        }
    }

    #[inline(always)]
    fn max_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe {
            let sign = _mm_set1_epi32(i32::MIN);
            let mask = _mm_cmpgt_epi32(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign));
            _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b))
        }
    }

    #[inline(always)]
    fn bitand_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm_and_si128(a, b) }
    }

    #[inline(always)]
    fn bitor_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm_or_si128(a, b) }
    }

    #[inline(always)]
    fn bitxor_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm_xor_si128(a, b) }
    }

    #[inline(always)]
    fn bitandnot_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm_andnot_si128(a, b) }
    }

    #[inline(always)]
    fn saturating_add_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        let mut x = [0u32; 4];
        let mut y = [0u32; 4];
        self.store_u32(&mut x, a);
        self.store_u32(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.saturating_add(*src);
        }
        self.load_u32(&x)
    }

    #[inline(always)]
    fn saturating_sub_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        let mut x = [0u32; 4];
        let mut y = [0u32; 4];
        self.store_u32(&mut x, a);
        self.store_u32(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.saturating_sub(*src);
        }
        self.load_u32(&x)
    }

    #[inline(always)]
    fn avg_round_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe {
            let and = _mm_and_si128(a, b);
            let xor = _mm_xor_si128(a, b);
            let half = _mm_srli_epi32(xor, 1);
            let odd = _mm_and_si128(xor, _mm_set1_epi32(1));
            _mm_add_epi32(and, _mm_add_epi32(half, odd))
        }
    }

    #[inline(always)]
    fn add_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm_add_epi32(a, b) }
    }

    #[inline(always)]
    fn sub_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm_sub_epi32(a, b) }
    }

    #[inline(always)]
    fn mul_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        let mut x = [0i32; 4];
        let mut y = [0i32; 4];
        self.store_i32(&mut x, a);
        self.store_i32(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.wrapping_mul(*src);
        }
        self.load_i32(&x)
    }

    #[inline(always)]
    fn min_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe {
            let mask = _mm_cmpgt_epi32(a, b);
            _mm_or_si128(_mm_and_si128(mask, b), _mm_andnot_si128(mask, a))
        }
    }

    #[inline(always)]
    fn max_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe {
            let mask = _mm_cmpgt_epi32(a, b);
            _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b))
        }
    }

    #[inline(always)]
    fn bitand_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm_and_si128(a, b) }
    }

    #[inline(always)]
    fn bitor_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm_or_si128(a, b) }
    }

    #[inline(always)]
    fn bitxor_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm_xor_si128(a, b) }
    }

    #[inline(always)]
    fn bitandnot_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm_andnot_si128(a, b) }
    }

    #[inline(always)]
    fn saturating_add_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        let mut x = [0i32; 4];
        let mut y = [0i32; 4];
        self.store_i32(&mut x, a);
        self.store_i32(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.saturating_add(*src);
        }
        self.load_i32(&x)
    }

    #[inline(always)]
    fn saturating_sub_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        let mut x = [0i32; 4];
        let mut y = [0i32; 4];
        self.store_i32(&mut x, a);
        self.store_i32(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.saturating_sub(*src);
        }
        self.load_i32(&x)
    }

    #[inline(always)]
    fn avg_round_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe {
            let and = _mm_and_si128(a, b);
            let xor = _mm_xor_si128(a, b);
            let half = _mm_srai_epi32(xor, 1);
            let odd = _mm_and_si128(xor, _mm_set1_epi32(1));
            _mm_add_epi32(and, _mm_add_epi32(half, odd))
        }
    }

    #[inline(always)]
    fn add_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm_add_epi64(a, b) }
    }

    #[inline(always)]
    fn sub_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm_sub_epi64(a, b) }
    }

    #[inline(always)]
    fn mul_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        let mut x = [0u64; 2];
        let mut y = [0u64; 2];
        self.store_u64(&mut x, a);
        self.store_u64(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.wrapping_mul(*src);
        }
        self.load_u64(&x)
    }

    #[inline(always)]
    fn min_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        let mut x = [0u64; 2];
        let mut y = [0u64; 2];
        self.store_u64(&mut x, a);
        self.store_u64(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = (*dst).min(*src);
        }
        self.load_u64(&x)
    }

    #[inline(always)]
    fn max_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        let mut x = [0u64; 2];
        let mut y = [0u64; 2];
        self.store_u64(&mut x, a);
        self.store_u64(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = (*dst).max(*src);
        }
        self.load_u64(&x)
    }

    #[inline(always)]
    fn bitand_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm_and_si128(a, b) }
    }

    #[inline(always)]
    fn bitor_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm_or_si128(a, b) }
    }

    #[inline(always)]
    fn bitxor_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm_xor_si128(a, b) }
    }

    #[inline(always)]
    fn bitandnot_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm_andnot_si128(a, b) }
    }

    #[inline(always)]
    fn saturating_add_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        let mut x = [0u64; 2];
        let mut y = [0u64; 2];
        self.store_u64(&mut x, a);
        self.store_u64(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.saturating_add(*src);
        }
        self.load_u64(&x)
    }

    #[inline(always)]
    fn saturating_sub_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        let mut x = [0u64; 2];
        let mut y = [0u64; 2];
        self.store_u64(&mut x, a);
        self.store_u64(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.saturating_sub(*src);
        }
        self.load_u64(&x)
    }

    #[inline(always)]
    fn avg_round_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe {
            let and = _mm_and_si128(a, b);
            let xor = _mm_xor_si128(a, b);
            let half = _mm_srli_epi64(xor, 1);
            let odd = _mm_and_si128(xor, _mm_set1_epi64x(1));
            _mm_add_epi64(and, _mm_add_epi64(half, odd))
        }
    }

    #[inline(always)]
    fn add_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm_add_epi64(a, b) }
    }

    #[inline(always)]
    fn sub_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm_sub_epi64(a, b) }
    }

    #[inline(always)]
    fn mul_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        let mut x = [0i64; 2];
        let mut y = [0i64; 2];
        self.store_i64(&mut x, a);
        self.store_i64(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.wrapping_mul(*src);
        }
        self.load_i64(&x)
    }

    #[inline(always)]
    fn min_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        let mut x = [0i64; 2];
        let mut y = [0i64; 2];
        self.store_i64(&mut x, a);
        self.store_i64(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = (*dst).min(*src);
        }
        self.load_i64(&x)
    }

    #[inline(always)]
    fn max_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        let mut x = [0i64; 2];
        let mut y = [0i64; 2];
        self.store_i64(&mut x, a);
        self.store_i64(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = (*dst).max(*src);
        }
        self.load_i64(&x)
    }

    #[inline(always)]
    fn bitand_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm_and_si128(a, b) }
    }

    #[inline(always)]
    fn bitor_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm_or_si128(a, b) }
    }

    #[inline(always)]
    fn bitxor_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm_xor_si128(a, b) }
    }

    #[inline(always)]
    fn bitandnot_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm_andnot_si128(a, b) }
    }

    #[inline(always)]
    fn saturating_add_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        let mut x = [0i64; 2];
        let mut y = [0i64; 2];
        self.store_i64(&mut x, a);
        self.store_i64(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.saturating_add(*src);
        }
        self.load_i64(&x)
    }

    #[inline(always)]
    fn saturating_sub_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        let mut x = [0i64; 2];
        let mut y = [0i64; 2];
        self.store_i64(&mut x, a);
        self.store_i64(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.saturating_sub(*src);
        }
        self.load_i64(&x)
    }

    #[inline(always)]
    fn avg_round_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe {
            let sign = _mm_set1_epi64x(i64::MIN);
            _mm_xor_si128(self.avg_round_u64(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign)), sign)
        }
    }

    #[inline(always)]
    fn eq_u8(self, a: Self::u8, b: Self::u8) -> Self::m8 {
        unsafe { Sse2MaskI8(_mm_cmpeq_epi8(a, b)) }
    }

    #[inline(always)]
    fn ne_u8(self, a: Self::u8, b: Self::u8) -> Self::m8 {
        self.eq_u8(a, b).not()
    }

    #[inline(always)]
    fn lt_u8(self, a: Self::u8, b: Self::u8) -> Self::m8 {
        self.gt_u8(b, a)
    }

    #[inline(always)]
    fn le_u8(self, a: Self::u8, b: Self::u8) -> Self::m8 {
        self.gt_u8(a, b).not()
    }

    #[inline(always)]
    fn gt_u8(self, a: Self::u8, b: Self::u8) -> Self::m8 {
        unsafe {
            let sign = _mm_set1_epi8(i8::MIN);
            Sse2MaskI8(_mm_cmpgt_epi8(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign)))
        }
    }

    #[inline(always)]
    fn ge_u8(self, a: Self::u8, b: Self::u8) -> Self::m8 {
        self.lt_u8(a, b).not()
    }

    #[inline(always)]
    fn eq_i8(self, a: Self::i8, b: Self::i8) -> Self::m8 {
        unsafe { Sse2MaskI8(_mm_cmpeq_epi8(a, b)) }
    }

    #[inline(always)]
    fn ne_i8(self, a: Self::i8, b: Self::i8) -> Self::m8 {
        self.eq_i8(a, b).not()
    }

    #[inline(always)]
    fn lt_i8(self, a: Self::i8, b: Self::i8) -> Self::m8 {
        self.gt_i8(b, a)
    }

    #[inline(always)]
    fn le_i8(self, a: Self::i8, b: Self::i8) -> Self::m8 {
        self.gt_i8(a, b).not()
    }

    #[inline(always)]
    fn gt_i8(self, a: Self::i8, b: Self::i8) -> Self::m8 {
        unsafe { Sse2MaskI8(_mm_cmpgt_epi8(a, b)) }
    }

    #[inline(always)]
    fn ge_i8(self, a: Self::i8, b: Self::i8) -> Self::m8 {
        self.lt_i8(a, b).not()
    }

    #[inline(always)]
    fn eq_u16(self, a: Self::u16, b: Self::u16) -> Self::m16 {
        unsafe { Sse2MaskI16(_mm_cmpeq_epi16(a, b)) }
    }

    #[inline(always)]
    fn ne_u16(self, a: Self::u16, b: Self::u16) -> Self::m16 {
        self.eq_u16(a, b).not()
    }

    #[inline(always)]
    fn lt_u16(self, a: Self::u16, b: Self::u16) -> Self::m16 {
        self.gt_u16(b, a)
    }

    #[inline(always)]
    fn le_u16(self, a: Self::u16, b: Self::u16) -> Self::m16 {
        self.gt_u16(a, b).not()
    }

    #[inline(always)]
    fn gt_u16(self, a: Self::u16, b: Self::u16) -> Self::m16 {
        unsafe {
            let sign = _mm_set1_epi16(i16::MIN);
            Sse2MaskI16(_mm_cmpgt_epi16(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign)))
        }
    }

    #[inline(always)]
    fn ge_u16(self, a: Self::u16, b: Self::u16) -> Self::m16 {
        self.lt_u16(a, b).not()
    }

    #[inline(always)]
    fn eq_i16(self, a: Self::i16, b: Self::i16) -> Self::m16 {
        unsafe { Sse2MaskI16(_mm_cmpeq_epi16(a, b)) }
    }

    #[inline(always)]
    fn ne_i16(self, a: Self::i16, b: Self::i16) -> Self::m16 {
        self.eq_i16(a, b).not()
    }

    #[inline(always)]
    fn lt_i16(self, a: Self::i16, b: Self::i16) -> Self::m16 {
        self.gt_i16(b, a)
    }

    #[inline(always)]
    fn le_i16(self, a: Self::i16, b: Self::i16) -> Self::m16 {
        self.gt_i16(a, b).not()
    }

    #[inline(always)]
    fn gt_i16(self, a: Self::i16, b: Self::i16) -> Self::m16 {
        unsafe { Sse2MaskI16(_mm_cmpgt_epi16(a, b)) }
    }

    #[inline(always)]
    fn ge_i16(self, a: Self::i16, b: Self::i16) -> Self::m16 {
        self.lt_i16(a, b).not()
    }

    #[inline(always)]
    fn eq_u32(self, a: Self::u32, b: Self::u32) -> Self::m32 {
        unsafe { Sse2MaskI32(_mm_cmpeq_epi32(a, b)) }
    }

    #[inline(always)]
    fn ne_u32(self, a: Self::u32, b: Self::u32) -> Self::m32 {
        self.eq_u32(a, b).not()
    }

    #[inline(always)]
    fn lt_u32(self, a: Self::u32, b: Self::u32) -> Self::m32 {
        self.gt_u32(b, a)
    }

    #[inline(always)]
    fn le_u32(self, a: Self::u32, b: Self::u32) -> Self::m32 {
        self.gt_u32(a, b).not()
    }

    #[inline(always)]
    fn gt_u32(self, a: Self::u32, b: Self::u32) -> Self::m32 {
        unsafe {
            let sign = _mm_set1_epi32(i32::MIN);
            Sse2MaskI32(_mm_cmpgt_epi32(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign)))
        }
    }

    #[inline(always)]
    fn ge_u32(self, a: Self::u32, b: Self::u32) -> Self::m32 {
        self.lt_u32(a, b).not()
    }

    #[inline(always)]
    fn eq_i32(self, a: Self::i32, b: Self::i32) -> Self::m32 {
        unsafe { Sse2MaskI32(_mm_cmpeq_epi32(a, b)) }
    }

    #[inline(always)]
    fn ne_i32(self, a: Self::i32, b: Self::i32) -> Self::m32 {
        self.eq_i32(a, b).not()
    }

    #[inline(always)]
    fn lt_i32(self, a: Self::i32, b: Self::i32) -> Self::m32 {
        self.gt_i32(b, a)
    }

    #[inline(always)]
    fn le_i32(self, a: Self::i32, b: Self::i32) -> Self::m32 {
        self.gt_i32(a, b).not()
    }

    #[inline(always)]
    fn gt_i32(self, a: Self::i32, b: Self::i32) -> Self::m32 {
        unsafe { Sse2MaskI32(_mm_cmpgt_epi32(a, b)) }
    }

    #[inline(always)]
    fn ge_i32(self, a: Self::i32, b: Self::i32) -> Self::m32 {
        self.lt_i32(a, b).not()
    }

    #[inline(always)]
    fn eq_u64(self, a: Self::u64, b: Self::u64) -> Self::m64 {
        let mut x = [0u64; 2];
        let mut y = [0u64; 2];
        self.store_u64(&mut x, a);
        self.store_u64(&mut y, b);
        unsafe { Sse2MaskI64(_mm_set_epi64x(if x[1] == y[1] { -1 } else { 0 }, if x[0] == y[0] { -1 } else { 0 })) }
    }

    #[inline(always)]
    fn ne_u64(self, a: Self::u64, b: Self::u64) -> Self::m64 {
        self.eq_u64(a, b).not()
    }

    #[inline(always)]
    fn lt_u64(self, a: Self::u64, b: Self::u64) -> Self::m64 {
        self.gt_u64(b, a)
    }

    #[inline(always)]
    fn le_u64(self, a: Self::u64, b: Self::u64) -> Self::m64 {
        self.gt_u64(a, b).not()
    }

    #[inline(always)]
    fn gt_u64(self, a: Self::u64, b: Self::u64) -> Self::m64 {
        let mut x = [0u64; 2];
        let mut y = [0u64; 2];
        self.store_u64(&mut x, a);
        self.store_u64(&mut y, b);
        unsafe { Sse2MaskI64(_mm_set_epi64x(if x[1] > y[1] { -1 } else { 0 }, if x[0] > y[0] { -1 } else { 0 })) }
    }

    #[inline(always)]
    fn ge_u64(self, a: Self::u64, b: Self::u64) -> Self::m64 {
        self.lt_u64(a, b).not()
    }

    #[inline(always)]
    fn eq_i64(self, a: Self::i64, b: Self::i64) -> Self::m64 {
        let mut x = [0i64; 2];
        let mut y = [0i64; 2];
        self.store_i64(&mut x, a);
        self.store_i64(&mut y, b);
        unsafe { Sse2MaskI64(_mm_set_epi64x(if x[1] == y[1] { -1 } else { 0 }, if x[0] == y[0] { -1 } else { 0 })) }
    }

    #[inline(always)]
    fn ne_i64(self, a: Self::i64, b: Self::i64) -> Self::m64 {
        self.eq_i64(a, b).not()
    }

    #[inline(always)]
    fn lt_i64(self, a: Self::i64, b: Self::i64) -> Self::m64 {
        self.gt_i64(b, a)
    }

    #[inline(always)]
    fn le_i64(self, a: Self::i64, b: Self::i64) -> Self::m64 {
        self.gt_i64(a, b).not()
    }

    #[inline(always)]
    fn gt_i64(self, a: Self::i64, b: Self::i64) -> Self::m64 {
        let mut x = [0i64; 2];
        let mut y = [0i64; 2];
        self.store_i64(&mut x, a);
        self.store_i64(&mut y, b);
        unsafe { Sse2MaskI64(_mm_set_epi64x(if x[1] > y[1] { -1 } else { 0 }, if x[0] > y[0] { -1 } else { 0 })) }
    }

    #[inline(always)]
    fn ge_i64(self, a: Self::i64, b: Self::i64) -> Self::m64 {
        self.lt_i64(a, b).not()
    }

    #[inline(always)]
    fn shl_u8(self, a: Self::u8, n: u32) -> Self::u8 {
        let mut data = [0u8; 16];
        self.store_u8(&mut data, a);
        for value in data.iter_mut() {
            *value = value.wrapping_shl(n);
        }
        self.load_u8(&data)
    }

    #[inline(always)]
    fn shr_u8(self, a: Self::u8, n: u32) -> Self::u8 {
        let mut data = [0u8; 16];
        self.store_u8(&mut data, a);
        for value in data.iter_mut() {
            *value = value.wrapping_shr(n);
        }
        self.load_u8(&data)
    }

    #[inline(always)]
    fn shl_i8(self, a: Self::i8, n: u32) -> Self::i8 {
        let mut data = [0i8; 16];
        self.store_i8(&mut data, a);
        for value in data.iter_mut() {
            *value = value.wrapping_shl(n);
        }
        self.load_i8(&data)
    }

    #[inline(always)]
    fn shr_i8(self, a: Self::i8, n: u32) -> Self::i8 {
        let mut data = [0i8; 16];
        self.store_i8(&mut data, a);
        for value in data.iter_mut() {
            *value = value.wrapping_shr(n);
        }
        self.load_i8(&data)
    }

    #[inline(always)]
    fn shl_u16(self, a: Self::u16, n: u32) -> Self::u16 {
        unsafe { _mm_sll_epi16(a, _mm_cvtsi32_si128((n & 15) as i32)) }
    }

    #[inline(always)]
    fn shr_u16(self, a: Self::u16, n: u32) -> Self::u16 {
        unsafe { _mm_srl_epi16(a, _mm_cvtsi32_si128((n & 15) as i32)) }
    }

    #[inline(always)]
    fn shl_i16(self, a: Self::i16, n: u32) -> Self::i16 {
        unsafe { _mm_sll_epi16(a, _mm_cvtsi32_si128((n & 15) as i32)) }
    }

    #[inline(always)]
    fn shr_i16(self, a: Self::i16, n: u32) -> Self::i16 {
        unsafe { _mm_sra_epi16(a, _mm_cvtsi32_si128((n & 15) as i32)) }
    }

    #[inline(always)]
    fn shl_u32(self, a: Self::u32, n: u32) -> Self::u32 {
        unsafe { _mm_sll_epi32(a, _mm_cvtsi32_si128((n & 31) as i32)) }
    }

    #[inline(always)]
    fn shr_u32(self, a: Self::u32, n: u32) -> Self::u32 {
        unsafe { _mm_srl_epi32(a, _mm_cvtsi32_si128((n & 31) as i32)) }
    }

    #[inline(always)]
    fn shl_i32(self, a: Self::i32, n: u32) -> Self::i32 {
        unsafe { _mm_sll_epi32(a, _mm_cvtsi32_si128((n & 31) as i32)) }
    }

    #[inline(always)]
    fn shr_i32(self, a: Self::i32, n: u32) -> Self::i32 {
        unsafe { _mm_sra_epi32(a, _mm_cvtsi32_si128((n & 31) as i32)) }
    }

    #[inline(always)]
    fn shl_u64(self, a: Self::u64, n: u32) -> Self::u64 {
        unsafe { _mm_sll_epi64(a, _mm_cvtsi32_si128((n & 63) as i32)) }
    }

    #[inline(always)]
    fn shr_u64(self, a: Self::u64, n: u32) -> Self::u64 {
        unsafe { _mm_srl_epi64(a, _mm_cvtsi32_si128((n & 63) as i32)) }
    }

    #[inline(always)]
    fn shl_i64(self, a: Self::i64, n: u32) -> Self::i64 {
        unsafe { _mm_sll_epi64(a, _mm_cvtsi32_si128((n & 63) as i32)) }
    }

    #[inline(always)]
    fn shr_i64(self, a: Self::i64, n: u32) -> Self::i64 {
        let mut data = [0i64; 2];
        self.store_i64(&mut data, a);
        for value in data.iter_mut() {
            *value = value.wrapping_shr(n);
        }
        self.load_i64(&data)
    }

    #[inline(always)]
    fn select_u8(self, mask: Self::m8, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm_or_si128(_mm_and_si128(mask.0, a), _mm_andnot_si128(mask.0, b)) }
    }

    #[inline(always)]
    fn select_i8(self, mask: Self::m8, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm_or_si128(_mm_and_si128(mask.0, a), _mm_andnot_si128(mask.0, b)) }
    }

    #[inline(always)]
    fn select_u16(self, mask: Self::m16, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm_or_si128(_mm_and_si128(mask.0, a), _mm_andnot_si128(mask.0, b)) }
    }

    #[inline(always)]
    fn select_i16(self, mask: Self::m16, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm_or_si128(_mm_and_si128(mask.0, a), _mm_andnot_si128(mask.0, b)) }
    }

    #[inline(always)]
    fn select_u32(self, mask: Self::m32, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm_or_si128(_mm_and_si128(mask.0, a), _mm_andnot_si128(mask.0, b)) }
    }

    #[inline(always)]
    fn select_i32(self, mask: Self::m32, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm_or_si128(_mm_and_si128(mask.0, a), _mm_andnot_si128(mask.0, b)) }
    }

    #[inline(always)]
    fn select_u64(self, mask: Self::m64, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm_or_si128(_mm_and_si128(mask.0, a), _mm_andnot_si128(mask.0, b)) }
    }

    #[inline(always)]
    fn select_i64(self, mask: Self::m64, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm_or_si128(_mm_and_si128(mask.0, a), _mm_andnot_si128(mask.0, b)) }
    }

    #[inline(always)]
    fn widen_low_u8_to_u16(self, x: Self::u8) -> Self::u16 {
        unsafe { _mm_unpacklo_epi8(x, _mm_setzero_si128()) }
    }

    #[inline(always)]
    fn widen_high_u8_to_u16(self, x: Self::u8) -> Self::u16 {
        unsafe { _mm_unpackhi_epi8(x, _mm_setzero_si128()) }
    }

    #[inline(always)]
    fn widen_low_i8_to_i16(self, x: Self::i8) -> Self::i16 {
        unsafe {
            let sign = _mm_cmpgt_epi8(_mm_setzero_si128(), x);
            _mm_unpacklo_epi8(x, sign)
        }
    }

    #[inline(always)]
    fn widen_high_i8_to_i16(self, x: Self::i8) -> Self::i16 {
        unsafe {
            let sign = _mm_cmpgt_epi8(_mm_setzero_si128(), x);
            _mm_unpackhi_epi8(x, sign)
        }
    }

    #[inline(always)]
    fn widen_low_u16_to_u32(self, x: Self::u16) -> Self::u32 {
        unsafe { _mm_unpacklo_epi16(x, _mm_setzero_si128()) }
    }

    #[inline(always)]
    fn widen_high_u16_to_u32(self, x: Self::u16) -> Self::u32 {
        unsafe { _mm_unpackhi_epi16(x, _mm_setzero_si128()) }
    }

    #[inline(always)]
    fn widen_low_i16_to_i32(self, x: Self::i16) -> Self::i32 {
        unsafe { _mm_unpacklo_epi16(x, _mm_srai_epi16(x, 15)) }
    }

    #[inline(always)]
    fn widen_high_i16_to_i32(self, x: Self::i16) -> Self::i32 {
        unsafe { _mm_unpackhi_epi16(x, _mm_srai_epi16(x, 15)) }
    }

    #[inline(always)]
    fn widen_low_u32_to_u64(self, x: Self::u32) -> Self::u64 {
        unsafe { _mm_unpacklo_epi32(x, _mm_setzero_si128()) }
    }

    #[inline(always)]
    fn widen_high_u32_to_u64(self, x: Self::u32) -> Self::u64 {
        unsafe { _mm_unpackhi_epi32(x, _mm_setzero_si128()) }
    }

    #[inline(always)]
    fn widen_low_i32_to_i64(self, x: Self::i32) -> Self::i64 {
        unsafe { _mm_unpacklo_epi32(x, _mm_srai_epi32(x, 31)) }
    }

    #[inline(always)]
    fn widen_high_i32_to_i64(self, x: Self::i32) -> Self::i64 {
        unsafe { _mm_unpackhi_epi32(x, _mm_srai_epi32(x, 31)) }
    }

    #[inline(always)]
    fn narrow_saturating_u16_to_u8(self, lo: Self::u16, hi: Self::u16) -> Self::u8 {
        unsafe {
            let max = _mm_set1_epi16(u8::MAX as i16);
            let lo = _mm_sub_epi16(lo, _mm_subs_epu16(lo, max));
            let hi = _mm_sub_epi16(hi, _mm_subs_epu16(hi, max));
            _mm_packus_epi16(lo, hi)
        }
    }

    #[inline(always)]
    fn narrow_saturating_i16_to_i8(self, lo: Self::i16, hi: Self::i16) -> Self::i8 {
        unsafe { _mm_packs_epi16(lo, hi) }
    }

    #[inline(always)]
    fn narrow_saturating_u32_to_u16(self, lo: Self::u32, hi: Self::u32) -> Self::u16 {
        let mut x = [0u32; 4];
        let mut y = [0u32; 4];
        self.store_u32(&mut x, lo);
        self.store_u32(&mut y, hi);
        let mut output = [0u16; 8];
        for (dst, src) in output.iter_mut().zip(x.iter()) {
            *dst = (*src).min(u16::MAX as u32) as u16;
        }
        for (dst, src) in output.iter_mut().skip(4).zip(y.iter()) {
            *dst = (*src).min(u16::MAX as u32) as u16;
        }
        self.load_u16(&output)
    }

    #[inline(always)]
    fn narrow_saturating_i32_to_i16(self, lo: Self::i32, hi: Self::i32) -> Self::i16 {
        unsafe { _mm_packs_epi32(lo, hi) }
    }

    #[inline(always)]
    fn narrow_saturating_u64_to_u32(self, lo: Self::u64, hi: Self::u64) -> Self::u32 {
        let mut x = [0u64; 2];
        let mut y = [0u64; 2];
        self.store_u64(&mut x, lo);
        self.store_u64(&mut y, hi);
        let mut output = [0u32; 4];
        for (dst, src) in output.iter_mut().zip(x.iter()) {
            *dst = (*src).min(u32::MAX as u64) as u32;
        }
        for (dst, src) in output.iter_mut().skip(2).zip(y.iter()) {
            *dst = (*src).min(u32::MAX as u64) as u32;
        }
        self.load_u32(&output)
    }

    #[inline(always)]
    fn narrow_saturating_i64_to_i32(self, lo: Self::i64, hi: Self::i64) -> Self::i32 {
        let mut x = [0i64; 2];
        let mut y = [0i64; 2];
        self.store_i64(&mut x, lo);
        self.store_i64(&mut y, hi);
        let mut output = [0i32; 4];
        for (dst, src) in output.iter_mut().zip(x.iter()) {
            *dst = (*src).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
        for (dst, src) in output.iter_mut().skip(2).zip(y.iter()) {
            *dst = (*src).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
        self.load_i32(&output)
    }
}

impl Simd for Sse2 {}
