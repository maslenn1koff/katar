use crate::{
    SimdFn,
    traits::{FloatSimd, IntegerSimd, MaskSimd, Simd},
};
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::marker::PhantomData;

#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512dq")]
pub fn vectorize<Op: SimdFn>(op: Op) -> Op::Result {
    op.call(Avx512Software::new())
}

#[derive(Copy, Clone)]
pub struct Avx512MaskF32(__mmask16);

impl MaskSimd for Avx512MaskF32 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        Self(if v { !0 } else { 0 })
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        Self(v as __mmask16)
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        self.0 as u64
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.0 == !0
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Avx512MaskF64(__mmask8);

impl MaskSimd for Avx512MaskF64 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        Self(if v { !0 } else { 0 })
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        Self(v as __mmask8)
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        self.0 as u64
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.0 == !0
    }
}

#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub struct Avx512MaskI8(__mmask64);

impl MaskSimd for Avx512MaskI8 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        Self(if v { u64::MAX } else { 0 })
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        Self(v)
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.0 == u64::MAX
    }
}

#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub struct Avx512MaskI16(__mmask32);

impl MaskSimd for Avx512MaskI16 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        Self(if v { u32::MAX } else { 0 })
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        Self(v as __mmask32)
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        self.0 as u64
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.0 == u32::MAX
    }
}

#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub struct Avx512MaskI32(__mmask16);

impl MaskSimd for Avx512MaskI32 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        Self(if v { u16::MAX } else { 0 })
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        Self(v as __mmask16)
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        self.0 as u64
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.0 == u16::MAX
    }
}

#[derive(Copy, Clone, Debug)]
#[allow(dead_code)]
pub struct Avx512MaskI64(__mmask8);

impl MaskSimd for Avx512MaskI64 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        Self(if v { u8::MAX } else { 0 })
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        Self(v as __mmask8)
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        self.0 as u64
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.0 == u8::MAX
    }
}

#[allow(dead_code)]
pub trait Avx512IntegerExt: Copy + Send + Sync + 'static {
    #[inline(always)]
    unsafe fn count_ones_u8(a: __m512i) -> __m512i {
        unsafe {
            let mut data = [0u8; 64];
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, a);
            for value in data.iter_mut() {
                *value = (*value).count_ones() as u8;
            }
            _mm512_loadu_si512(data.as_ptr() as *const __m512i)
        }
    }

    #[inline(always)]
    unsafe fn count_ones_u16(a: __m512i) -> __m512i {
        unsafe {
            let mut data = [0u16; 32];
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, a);
            for value in data.iter_mut() {
                *value = (*value).count_ones() as u16;
            }
            _mm512_loadu_si512(data.as_ptr() as *const __m512i)
        }
    }

    #[inline(always)]
    unsafe fn count_ones_u32(a: __m512i) -> __m512i {
        unsafe {
            let mut data = [0u32; 16];
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, a);
            for value in data.iter_mut() {
                *value = (*value).count_ones();
            }
            _mm512_loadu_si512(data.as_ptr() as *const __m512i)
        }
    }

    #[inline(always)]
    unsafe fn count_ones_u64(a: __m512i) -> __m512i {
        unsafe {
            let mut data = [0u64; 8];
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, a);
            for value in data.iter_mut() {
                *value = (*value).count_ones() as u64;
            }
            _mm512_loadu_si512(data.as_ptr() as *const __m512i)
        }
    }

    #[inline(always)]
    unsafe fn leading_zeros_u8(a: __m512i) -> __m512i {
        unsafe {
            let mut data = [0u8; 64];
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, a);
            for value in data.iter_mut() {
                *value = (*value).leading_zeros() as u8;
            }
            _mm512_loadu_si512(data.as_ptr() as *const __m512i)
        }
    }

    #[inline(always)]
    unsafe fn leading_zeros_u16(a: __m512i) -> __m512i {
        unsafe {
            let mut data = [0u16; 32];
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, a);
            for value in data.iter_mut() {
                *value = (*value).leading_zeros() as u16;
            }
            _mm512_loadu_si512(data.as_ptr() as *const __m512i)
        }
    }

    #[inline(always)]
    unsafe fn leading_zeros_u32(a: __m512i) -> __m512i {
        unsafe {
            let mut data = [0u32; 16];
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, a);
            for value in data.iter_mut() {
                *value = (*value).leading_zeros();
            }
            _mm512_loadu_si512(data.as_ptr() as *const __m512i)
        }
    }

    #[inline(always)]
    unsafe fn leading_zeros_u64(a: __m512i) -> __m512i {
        unsafe {
            let mut data = [0u64; 8];
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, a);
            for value in data.iter_mut() {
                *value = (*value).leading_zeros() as u64;
            }
            _mm512_loadu_si512(data.as_ptr() as *const __m512i)
        }
    }

    #[inline(always)]
    unsafe fn trailing_zeros_u8(a: __m512i) -> __m512i {
        unsafe {
            let mut data = [0u8; 64];
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, a);
            for value in data.iter_mut() {
                *value = (*value).trailing_zeros() as u8;
            }
            _mm512_loadu_si512(data.as_ptr() as *const __m512i)
        }
    }

    #[inline(always)]
    unsafe fn trailing_zeros_u16(a: __m512i) -> __m512i {
        unsafe {
            let mut data = [0u16; 32];
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, a);
            for value in data.iter_mut() {
                *value = (*value).trailing_zeros() as u16;
            }
            _mm512_loadu_si512(data.as_ptr() as *const __m512i)
        }
    }

    #[inline(always)]
    unsafe fn trailing_zeros_u32(a: __m512i) -> __m512i {
        unsafe {
            let mut data = [0u32; 16];
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, a);
            for value in data.iter_mut() {
                *value = (*value).trailing_zeros();
            }
            _mm512_loadu_si512(data.as_ptr() as *const __m512i)
        }
    }

    #[inline(always)]
    unsafe fn trailing_zeros_u64(a: __m512i) -> __m512i {
        unsafe {
            let mut data = [0u64; 8];
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, a);
            for value in data.iter_mut() {
                *value = (*value).trailing_zeros() as u64;
            }
            _mm512_loadu_si512(data.as_ptr() as *const __m512i)
        }
    }
}

#[derive(Copy, Clone)]
#[allow(dead_code)]
pub struct Avx512SoftwareExt;

impl Avx512IntegerExt for Avx512SoftwareExt {}

#[derive(Copy, Clone)]
#[allow(dead_code)]
pub struct Avx512NativeExt;

impl Avx512IntegerExt for Avx512NativeExt {
    #[inline(always)]
    unsafe fn count_ones_u8(a: __m512i) -> __m512i {
        unsafe { _mm512_popcnt_epi8(a) }
    }

    #[inline(always)]
    unsafe fn count_ones_u16(a: __m512i) -> __m512i {
        unsafe { _mm512_popcnt_epi16(a) }
    }

    #[inline(always)]
    unsafe fn count_ones_u32(a: __m512i) -> __m512i {
        unsafe { _mm512_popcnt_epi32(a) }
    }

    #[inline(always)]
    unsafe fn count_ones_u64(a: __m512i) -> __m512i {
        unsafe { _mm512_popcnt_epi64(a) }
    }

    #[inline(always)]
    unsafe fn leading_zeros_u32(a: __m512i) -> __m512i {
        unsafe { _mm512_lzcnt_epi32(a) }
    }

    #[inline(always)]
    unsafe fn leading_zeros_u64(a: __m512i) -> __m512i {
        unsafe { _mm512_lzcnt_epi64(a) }
    }

    #[inline(always)]
    unsafe fn trailing_zeros_u8(a: __m512i) -> __m512i {
        unsafe {
            let low = _mm512_and_si512(a, _mm512_sub_epi8(_mm512_setzero_si512(), a));
            _mm512_popcnt_epi8(_mm512_sub_epi8(low, _mm512_set1_epi8(1)))
        }
    }

    #[inline(always)]
    unsafe fn trailing_zeros_u16(a: __m512i) -> __m512i {
        unsafe {
            let low = _mm512_and_si512(a, _mm512_sub_epi16(_mm512_setzero_si512(), a));
            _mm512_popcnt_epi16(_mm512_sub_epi16(low, _mm512_set1_epi16(1)))
        }
    }

    #[inline(always)]
    unsafe fn trailing_zeros_u32(a: __m512i) -> __m512i {
        unsafe {
            let low = _mm512_and_si512(a, _mm512_sub_epi32(_mm512_setzero_si512(), a));
            _mm512_popcnt_epi32(_mm512_sub_epi32(low, _mm512_set1_epi32(1)))
        }
    }

    #[inline(always)]
    unsafe fn trailing_zeros_u64(a: __m512i) -> __m512i {
        unsafe {
            let low = _mm512_and_si512(a, _mm512_sub_epi64(_mm512_setzero_si512(), a));
            _mm512_popcnt_epi64(_mm512_sub_epi64(low, _mm512_set1_epi64(1)))
        }
    }
}

#[derive(Clone, Copy)]
pub struct Avx512F<E: Avx512IntegerExt = Avx512SoftwareExt> {
    _ext: PhantomData<E>,
}

impl<E: Avx512IntegerExt> Avx512F<E> {
    #[inline(always)]
    pub fn new() -> Self {
        Self { _ext: PhantomData }
    }
}

pub type Avx512Software = Avx512F<Avx512SoftwareExt>;
#[allow(dead_code)]
pub type Avx512Native = Avx512F<Avx512NativeExt>;

impl<E: Avx512IntegerExt> FloatSimd for Avx512F<E> {
    type f32 = __m512;
    type f64 = __m512d;

    type mf32 = Avx512MaskF32;
    type mf64 = Avx512MaskF64;

    const f32_lanes: usize = 16;
    const f64_lanes: usize = 8;

    #[inline(always)]
    fn splat_f32(self, value: f32) -> Self::f32 {
        unsafe { _mm512_set1_ps(value) }
    }

    #[inline(always)]
    fn splat_f64(self, value: f64) -> Self::f64 {
        unsafe { _mm512_set1_pd(value) }
    }

    #[inline(always)]
    fn load_f32(self, src: &[f32]) -> Self::f32 {
        assert!(src.len() >= Self::f32_lanes);
        unsafe { _mm512_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    fn load_f64(self, src: &[f64]) -> Self::f64 {
        assert!(src.len() >= Self::f64_lanes);
        unsafe { _mm512_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    fn store_f32(self, dst: &mut [f32], value: Self::f32) {
        assert!(dst.len() >= Self::f32_lanes);
        unsafe { _mm512_storeu_ps(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    fn store_f64(self, dst: &mut [f64], value: Self::f64) {
        assert!(dst.len() >= Self::f64_lanes);
        unsafe { _mm512_storeu_pd(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    unsafe fn load_unchecked_f32(self, src: &[f32]) -> Self::f32 {
        unsafe { _mm512_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_f64(self, src: &[f64]) -> Self::f64 {
        unsafe { _mm512_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f32(self, dst: &mut [f32], value: Self::f32) {
        unsafe { _mm512_storeu_ps(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f64(self, dst: &mut [f64], value: Self::f64) {
        unsafe { _mm512_storeu_pd(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    fn abs_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm512_andnot_ps(_mm512_set1_ps(-0.0), a) }
    }

    #[inline(always)]
    fn abs_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm512_andnot_pd(_mm512_set1_pd(-0.0), a) }
    }

    #[inline(always)]
    fn add_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_add_ps(a, b) }
    }

    #[inline(always)]
    fn add_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_add_pd(a, b) }
    }

    #[inline(always)]
    fn sub_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_sub_ps(a, b) }
    }

    #[inline(always)]
    fn sub_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_sub_pd(a, b) }
    }

    #[inline(always)]
    fn mul_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_mul_ps(a, b) }
    }

    #[inline(always)]
    fn mul_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_mul_pd(a, b) }
    }

    #[inline(always)]
    fn div_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_div_ps(a, b) }
    }

    #[inline(always)]
    fn div_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_div_pd(a, b) }
    }

    #[inline(always)]
    fn max_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_max_ps(a, b) }
    }

    #[inline(always)]
    fn max_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_max_pd(a, b) }
    }

    #[inline(always)]
    fn min_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_min_ps(a, b) }
    }

    #[inline(always)]
    fn min_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_min_pd(a, b) }
    }

    #[inline(always)]
    fn neg_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm512_xor_ps(a, _mm512_set1_ps(-0.0)) }
    }

    #[inline(always)]
    fn neg_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm512_xor_pd(a, _mm512_set1_pd(-0.0)) }
    }

    #[inline(always)]
    fn sqrt_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm512_sqrt_ps(a) }
    }

    #[inline(always)]
    fn sqrt_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm512_sqrt_pd(a) }
    }

    #[inline(always)]
    fn mul_add_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm512_fmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn mul_add_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm512_fmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm512_fmsub_ps(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm512_fmsub_pd(a, b, c) }
    }

    #[inline(always)]
    fn nmul_add_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm512_fnmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn nmul_add_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm512_fnmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn nmul_sub_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm512_fnmsub_ps(a, b, c) }
    }

    #[inline(always)]
    fn nmul_sub_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm512_fnmsub_pd(a, b, c) }
    }

    #[inline(always)]
    fn eq_f32(self, a: Self::f32, b: Self::f32) -> Self::mf32 {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_EQ_OQ>(a, b)) }
    }

    #[inline(always)]
    fn eq_f64(self, a: Self::f64, b: Self::f64) -> Self::mf64 {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_EQ_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ge_f32(self, a: Self::f32, b: Self::f32) -> Self::mf32 {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ge_f64(self, a: Self::f64, b: Self::f64) -> Self::mf64 {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_GE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn gt_f32(self, a: Self::f32, b: Self::f32) -> Self::mf32 {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_GT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn gt_f64(self, a: Self::f64, b: Self::f64) -> Self::mf64 {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_GT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn le_f32(self, a: Self::f32, b: Self::f32) -> Self::mf32 {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_LE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn le_f64(self, a: Self::f64, b: Self::f64) -> Self::mf64 {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_LE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn lt_f32(self, a: Self::f32, b: Self::f32) -> Self::mf32 {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_LT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn lt_f64(self, a: Self::f64, b: Self::f64) -> Self::mf64 {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_LT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ne_f32(self, a: Self::f32, b: Self::f32) -> Self::mf32 {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(a, b)) }
    }

    #[inline(always)]
    fn ne_f64(self, a: Self::f64, b: Self::f64) -> Self::mf64 {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(a, b)) }
    }

    #[inline(always)]
    fn select_f32(self, mask: Self::mf32, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_mask_blend_ps(mask.0, b, a) }
    }

    #[inline(always)]
    fn select_f64(self, mask: Self::mf64, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_mask_blend_pd(mask.0, b, a) }
    }

    #[inline(always)]
    fn bitand_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_and_ps(a, b) }
    }

    #[inline(always)]
    fn bitor_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_or_ps(a, b) }
    }

    #[inline(always)]
    fn bitxor_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_xor_ps(a, b) }
    }

    #[inline(always)]
    fn bitand_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_and_pd(a, b) }
    }

    #[inline(always)]
    fn bitor_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_or_pd(a, b) }
    }

    #[inline(always)]
    fn bitxor_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_xor_pd(a, b) }
    }

    #[inline(always)]
    fn bitnot_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm512_xor_ps(a, _mm512_castsi512_ps(_mm512_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn bitnot_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm512_xor_pd(a, _mm512_castsi512_pd(_mm512_set1_epi64(-1))) }
    }

    #[inline(always)]
    fn bitandnot_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_andnot_ps(a, b) }
    }

    #[inline(always)]
    fn bitandnot_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_andnot_pd(a, b) }
    }

    #[inline(always)]
    fn reduce_max_f32(self, a: Self::f32) -> f32 {
        unsafe { _mm512_reduce_max_ps(a) }
    }

    #[inline(always)]
    fn reduce_max_f64(self, a: Self::f64) -> f64 {
        unsafe { _mm512_reduce_max_pd(a) }
    }

    #[inline(always)]
    fn reduce_min_f32(self, a: Self::f32) -> f32 {
        unsafe { _mm512_reduce_min_ps(a) }
    }

    #[inline(always)]
    fn reduce_min_f64(self, a: Self::f64) -> f64 {
        unsafe { _mm512_reduce_min_pd(a) }
    }

    #[inline(always)]
    fn reduce_sum_f32(self, a: Self::f32) -> f32 {
        unsafe { _mm512_reduce_add_ps(a) }
    }

    #[inline(always)]
    fn reduce_sum_f64(self, a: Self::f64) -> f64 {
        unsafe { _mm512_reduce_add_pd(a) }
    }
}

impl<E: Avx512IntegerExt> Simd for Avx512F<E> {}

impl<E: Avx512IntegerExt> IntegerSimd for Avx512F<E> {
    type u8 = __m512i;
    type i8 = __m512i;
    type u16 = __m512i;
    type i16 = __m512i;
    type u32 = __m512i;
    type i32 = __m512i;
    type u64 = __m512i;
    type i64 = __m512i;

    type m8 = Avx512MaskI8;
    type m16 = Avx512MaskI16;
    type m32 = Avx512MaskI32;
    type m64 = Avx512MaskI64;

    const u8_lanes: usize = 64;
    const i8_lanes: usize = 64;
    const u16_lanes: usize = 32;
    const i16_lanes: usize = 32;
    const u32_lanes: usize = 16;
    const i32_lanes: usize = 16;
    const u64_lanes: usize = 8;
    const i64_lanes: usize = 8;

    #[inline(always)]
    fn splat_u8(self, value: u8) -> Self::u8 {
        unsafe { _mm512_set1_epi8(value as i8) }
    }

    #[inline(always)]
    fn splat_i8(self, value: i8) -> Self::i8 {
        unsafe { _mm512_set1_epi8(value) }
    }

    #[inline(always)]
    fn splat_u16(self, value: u16) -> Self::u16 {
        unsafe { _mm512_set1_epi16(value as i16) }
    }

    #[inline(always)]
    fn splat_i16(self, value: i16) -> Self::i16 {
        unsafe { _mm512_set1_epi16(value) }
    }

    #[inline(always)]
    fn splat_u32(self, value: u32) -> Self::u32 {
        unsafe { _mm512_set1_epi32(value as i32) }
    }

    #[inline(always)]
    fn splat_i32(self, value: i32) -> Self::i32 {
        unsafe { _mm512_set1_epi32(value) }
    }

    #[inline(always)]
    fn splat_u64(self, value: u64) -> Self::u64 {
        unsafe { _mm512_set1_epi64(value as i64) }
    }

    #[inline(always)]
    fn splat_i64(self, value: i64) -> Self::i64 {
        unsafe { _mm512_set1_epi64(value) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_u8(self, src: &[u8]) -> Self::u8 {
        unsafe { _mm512_loadu_si512(src.as_ptr() as *const __m512i) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_i8(self, src: &[i8]) -> Self::i8 {
        unsafe { _mm512_loadu_si512(src.as_ptr() as *const __m512i) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_u16(self, src: &[u16]) -> Self::u16 {
        unsafe { _mm512_loadu_si512(src.as_ptr() as *const __m512i) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_i16(self, src: &[i16]) -> Self::i16 {
        unsafe { _mm512_loadu_si512(src.as_ptr() as *const __m512i) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_u32(self, src: &[u32]) -> Self::u32 {
        unsafe { _mm512_loadu_si512(src.as_ptr() as *const __m512i) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_i32(self, src: &[i32]) -> Self::i32 {
        unsafe { _mm512_loadu_si512(src.as_ptr() as *const __m512i) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_u64(self, src: &[u64]) -> Self::u64 {
        unsafe { _mm512_loadu_si512(src.as_ptr() as *const __m512i) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_i64(self, src: &[i64]) -> Self::i64 {
        unsafe { _mm512_loadu_si512(src.as_ptr() as *const __m512i) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_u8(self, dst: &mut [u8], value: Self::u8) {
        unsafe { _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_i8(self, dst: &mut [i8], value: Self::i8) {
        unsafe { _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_u16(self, dst: &mut [u16], value: Self::u16) {
        unsafe { _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_i16(self, dst: &mut [i16], value: Self::i16) {
        unsafe { _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_u32(self, dst: &mut [u32], value: Self::u32) {
        unsafe { _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_i32(self, dst: &mut [i32], value: Self::i32) {
        unsafe { _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_u64(self, dst: &mut [u64], value: Self::u64) {
        unsafe { _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_i64(self, dst: &mut [i64], value: Self::i64) {
        unsafe { _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, value) }
    }

    #[inline(always)]
    fn bitnot_u8(self, a: Self::u8) -> Self::u8 {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn bitnot_i8(self, a: Self::i8) -> Self::i8 {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn abs_i8(self, a: Self::i8) -> Self::i8 {
        unsafe { _mm512_abs_epi8(a) }
    }

    #[inline(always)]
    fn neg_i8(self, a: Self::i8) -> Self::i8 {
        unsafe { _mm512_sub_epi8(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn bitnot_u16(self, a: Self::u16) -> Self::u16 {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn bitnot_i16(self, a: Self::i16) -> Self::i16 {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn abs_i16(self, a: Self::i16) -> Self::i16 {
        unsafe { _mm512_abs_epi16(a) }
    }

    #[inline(always)]
    fn neg_i16(self, a: Self::i16) -> Self::i16 {
        unsafe { _mm512_sub_epi16(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn bitnot_u32(self, a: Self::u32) -> Self::u32 {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn bitnot_i32(self, a: Self::i32) -> Self::i32 {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn abs_i32(self, a: Self::i32) -> Self::i32 {
        unsafe { _mm512_abs_epi32(a) }
    }

    #[inline(always)]
    fn neg_i32(self, a: Self::i32) -> Self::i32 {
        unsafe { _mm512_sub_epi32(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn bitnot_u64(self, a: Self::u64) -> Self::u64 {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn bitnot_i64(self, a: Self::i64) -> Self::i64 {
        unsafe { _mm512_xor_si512(a, _mm512_set1_epi32(-1)) }
    }

    #[inline(always)]
    fn abs_i64(self, a: Self::i64) -> Self::i64 {
        let mut data = [0i64; 8];
        self.store_i64(&mut data, a);
        for value in data.iter_mut() {
            *value = value.wrapping_abs();
        }
        self.load_i64(&data)
    }

    #[inline(always)]
    fn neg_i64(self, a: Self::i64) -> Self::i64 {
        unsafe { _mm512_sub_epi64(_mm512_setzero_si512(), a) }
    }

    #[inline(always)]
    fn count_ones_u8(self, a: Self::u8) -> Self::u8 {
        unsafe { E::count_ones_u8(a) }
    }

    #[inline(always)]
    fn count_ones_i8(self, a: Self::i8) -> Self::i8 {
        unsafe { E::count_ones_u8(a) }
    }

    #[inline(always)]
    fn count_ones_u16(self, a: Self::u16) -> Self::u16 {
        unsafe { E::count_ones_u16(a) }
    }

    #[inline(always)]
    fn count_ones_i16(self, a: Self::i16) -> Self::i16 {
        unsafe { E::count_ones_u16(a) }
    }

    #[inline(always)]
    fn count_ones_u32(self, a: Self::u32) -> Self::u32 {
        unsafe { E::count_ones_u32(a) }
    }

    #[inline(always)]
    fn count_ones_i32(self, a: Self::i32) -> Self::i32 {
        unsafe { E::count_ones_u32(a) }
    }

    #[inline(always)]
    fn count_ones_u64(self, a: Self::u64) -> Self::u64 {
        unsafe { E::count_ones_u64(a) }
    }

    #[inline(always)]
    fn count_ones_i64(self, a: Self::i64) -> Self::i64 {
        unsafe { E::count_ones_u64(a) }
    }

    #[inline(always)]
    fn leading_zeros_u8(self, a: Self::u8) -> Self::u8 {
        unsafe { E::leading_zeros_u8(a) }
    }

    #[inline(always)]
    fn leading_zeros_i8(self, a: Self::i8) -> Self::i8 {
        unsafe { E::leading_zeros_u8(a) }
    }

    #[inline(always)]
    fn leading_zeros_u16(self, a: Self::u16) -> Self::u16 {
        unsafe { E::leading_zeros_u16(a) }
    }

    #[inline(always)]
    fn leading_zeros_i16(self, a: Self::i16) -> Self::i16 {
        unsafe { E::leading_zeros_u16(a) }
    }

    #[inline(always)]
    fn leading_zeros_u32(self, a: Self::u32) -> Self::u32 {
        unsafe { E::leading_zeros_u32(a) }
    }

    #[inline(always)]
    fn leading_zeros_i32(self, a: Self::i32) -> Self::i32 {
        unsafe { E::leading_zeros_u32(a) }
    }

    #[inline(always)]
    fn leading_zeros_u64(self, a: Self::u64) -> Self::u64 {
        unsafe { E::leading_zeros_u64(a) }
    }

    #[inline(always)]
    fn leading_zeros_i64(self, a: Self::i64) -> Self::i64 {
        unsafe { E::leading_zeros_u64(a) }
    }

    #[inline(always)]
    fn trailing_zeros_u8(self, a: Self::u8) -> Self::u8 {
        unsafe { E::trailing_zeros_u8(a) }
    }

    #[inline(always)]
    fn trailing_zeros_i8(self, a: Self::i8) -> Self::i8 {
        unsafe { E::trailing_zeros_u8(a) }
    }

    #[inline(always)]
    fn trailing_zeros_u16(self, a: Self::u16) -> Self::u16 {
        unsafe { E::trailing_zeros_u16(a) }
    }

    #[inline(always)]
    fn trailing_zeros_i16(self, a: Self::i16) -> Self::i16 {
        unsafe { E::trailing_zeros_u16(a) }
    }

    #[inline(always)]
    fn trailing_zeros_u32(self, a: Self::u32) -> Self::u32 {
        unsafe { E::trailing_zeros_u32(a) }
    }

    #[inline(always)]
    fn trailing_zeros_i32(self, a: Self::i32) -> Self::i32 {
        unsafe { E::trailing_zeros_u32(a) }
    }

    #[inline(always)]
    fn trailing_zeros_u64(self, a: Self::u64) -> Self::u64 {
        unsafe { E::trailing_zeros_u64(a) }
    }

    #[inline(always)]
    fn trailing_zeros_i64(self, a: Self::i64) -> Self::i64 {
        unsafe { E::trailing_zeros_u64(a) }
    }

    #[inline(always)]
    fn add_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm512_add_epi8(a, b) }
    }

    #[inline(always)]
    fn sub_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm512_sub_epi8(a, b) }
    }

    #[inline(always)]
    fn mul_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        let mut x = [0u8; 64];
        let mut y = [0u8; 64];
        self.store_u8(&mut x, a);
        self.store_u8(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.wrapping_mul(*src);
        }
        self.load_u8(&x)
    }

    #[inline(always)]
    fn min_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm512_min_epu8(a, b) }
    }

    #[inline(always)]
    fn max_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm512_max_epu8(a, b) }
    }

    #[inline(always)]
    fn bitand_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn bitandnot_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm512_andnot_si512(a, b) }
    }

    #[inline(always)]
    fn saturating_add_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm512_adds_epu8(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm512_subs_epu8(a, b) }
    }

    #[inline(always)]
    fn avg_round_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm512_avg_epu8(a, b) }
    }

    #[inline(always)]
    fn add_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm512_add_epi8(a, b) }
    }

    #[inline(always)]
    fn sub_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm512_sub_epi8(a, b) }
    }

    #[inline(always)]
    fn mul_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        let mut x = [0i8; 64];
        let mut y = [0i8; 64];
        self.store_i8(&mut x, a);
        self.store_i8(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.wrapping_mul(*src);
        }
        self.load_i8(&x)
    }

    #[inline(always)]
    fn min_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm512_min_epi8(a, b) }
    }

    #[inline(always)]
    fn max_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm512_max_epi8(a, b) }
    }

    #[inline(always)]
    fn bitand_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn bitandnot_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm512_andnot_si512(a, b) }
    }

    #[inline(always)]
    fn saturating_add_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm512_adds_epi8(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm512_subs_epi8(a, b) }
    }

    #[inline(always)]
    fn avg_round_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe {
            let sign = _mm512_set1_epi8(i8::MIN);
            _mm512_xor_si512(_mm512_avg_epu8(_mm512_xor_si512(a, sign), _mm512_xor_si512(b, sign)), sign)
        }
    }

    #[inline(always)]
    fn add_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm512_add_epi16(a, b) }
    }

    #[inline(always)]
    fn sub_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm512_sub_epi16(a, b) }
    }

    #[inline(always)]
    fn mul_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm512_mullo_epi16(a, b) }
    }

    #[inline(always)]
    fn min_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm512_min_epu16(a, b) }
    }

    #[inline(always)]
    fn max_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm512_max_epu16(a, b) }
    }

    #[inline(always)]
    fn bitand_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn bitandnot_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm512_andnot_si512(a, b) }
    }

    #[inline(always)]
    fn saturating_add_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm512_adds_epu16(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm512_subs_epu16(a, b) }
    }

    #[inline(always)]
    fn avg_round_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm512_avg_epu16(a, b) }
    }

    #[inline(always)]
    fn add_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm512_add_epi16(a, b) }
    }

    #[inline(always)]
    fn sub_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm512_sub_epi16(a, b) }
    }

    #[inline(always)]
    fn mul_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm512_mullo_epi16(a, b) }
    }

    #[inline(always)]
    fn min_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm512_min_epi16(a, b) }
    }

    #[inline(always)]
    fn max_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm512_max_epi16(a, b) }
    }

    #[inline(always)]
    fn bitand_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn bitandnot_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm512_andnot_si512(a, b) }
    }

    #[inline(always)]
    fn saturating_add_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm512_adds_epi16(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm512_subs_epi16(a, b) }
    }

    #[inline(always)]
    fn avg_round_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe {
            let sign = _mm512_set1_epi16(i16::MIN);
            _mm512_xor_si512(_mm512_avg_epu16(_mm512_xor_si512(a, sign), _mm512_xor_si512(b, sign)), sign)
        }
    }

    #[inline(always)]
    fn add_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm512_add_epi32(a, b) }
    }

    #[inline(always)]
    fn sub_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm512_sub_epi32(a, b) }
    }

    #[inline(always)]
    fn mul_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm512_mullo_epi32(a, b) }
    }

    #[inline(always)]
    fn min_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm512_min_epu32(a, b) }
    }

    #[inline(always)]
    fn max_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm512_max_epu32(a, b) }
    }

    #[inline(always)]
    fn bitand_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn bitandnot_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm512_andnot_si512(a, b) }
    }

    #[inline(always)]
    fn saturating_add_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        let mut x = [0u32; 16];
        let mut y = [0u32; 16];
        self.store_u32(&mut x, a);
        self.store_u32(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.saturating_add(*src);
        }
        self.load_u32(&x)
    }

    #[inline(always)]
    fn saturating_sub_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        let mut x = [0u32; 16];
        let mut y = [0u32; 16];
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
            let and = _mm512_and_si512(a, b);
            let xor = _mm512_xor_si512(a, b);
            let half = _mm512_srli_epi32(xor, 1);
            let odd = _mm512_and_si512(xor, _mm512_set1_epi32(1));
            _mm512_add_epi32(and, _mm512_add_epi32(half, odd))
        }
    }

    #[inline(always)]
    fn add_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm512_add_epi32(a, b) }
    }

    #[inline(always)]
    fn sub_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm512_sub_epi32(a, b) }
    }

    #[inline(always)]
    fn mul_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm512_mullo_epi32(a, b) }
    }

    #[inline(always)]
    fn min_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm512_min_epi32(a, b) }
    }

    #[inline(always)]
    fn max_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm512_max_epi32(a, b) }
    }

    #[inline(always)]
    fn bitand_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn bitandnot_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm512_andnot_si512(a, b) }
    }

    #[inline(always)]
    fn saturating_add_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        let mut x = [0i32; 16];
        let mut y = [0i32; 16];
        self.store_i32(&mut x, a);
        self.store_i32(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.saturating_add(*src);
        }
        self.load_i32(&x)
    }

    #[inline(always)]
    fn saturating_sub_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        let mut x = [0i32; 16];
        let mut y = [0i32; 16];
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
            let and = _mm512_and_si512(a, b);
            let xor = _mm512_xor_si512(a, b);
            let half = _mm512_srai_epi32(xor, 1);
            let odd = _mm512_and_si512(xor, _mm512_set1_epi32(1));
            _mm512_add_epi32(and, _mm512_add_epi32(half, odd))
        }
    }

    #[inline(always)]
    fn add_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm512_add_epi64(a, b) }
    }

    #[inline(always)]
    fn sub_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm512_sub_epi64(a, b) }
    }

    #[inline(always)]
    fn mul_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm512_mullo_epi64(a, b) }
    }

    #[inline(always)]
    fn min_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm512_min_epu64(a, b) }
    }

    #[inline(always)]
    fn max_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm512_max_epu64(a, b) }
    }

    #[inline(always)]
    fn bitand_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn bitandnot_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm512_andnot_si512(a, b) }
    }

    #[inline(always)]
    fn saturating_add_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        let mut x = [0u64; 8];
        let mut y = [0u64; 8];
        self.store_u64(&mut x, a);
        self.store_u64(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.saturating_add(*src);
        }
        self.load_u64(&x)
    }

    #[inline(always)]
    fn saturating_sub_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        let mut x = [0u64; 8];
        let mut y = [0u64; 8];
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
            let and = _mm512_and_si512(a, b);
            let xor = _mm512_xor_si512(a, b);
            let half = _mm512_srli_epi64(xor, 1);
            let odd = _mm512_and_si512(xor, _mm512_set1_epi64(1));
            _mm512_add_epi64(and, _mm512_add_epi64(half, odd))
        }
    }

    #[inline(always)]
    fn add_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm512_add_epi64(a, b) }
    }

    #[inline(always)]
    fn sub_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm512_sub_epi64(a, b) }
    }

    #[inline(always)]
    fn mul_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm512_mullo_epi64(a, b) }
    }

    #[inline(always)]
    fn min_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm512_min_epi64(a, b) }
    }

    #[inline(always)]
    fn max_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm512_max_epi64(a, b) }
    }

    #[inline(always)]
    fn bitand_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm512_and_si512(a, b) }
    }

    #[inline(always)]
    fn bitor_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm512_or_si512(a, b) }
    }

    #[inline(always)]
    fn bitxor_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm512_xor_si512(a, b) }
    }

    #[inline(always)]
    fn bitandnot_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm512_andnot_si512(a, b) }
    }

    #[inline(always)]
    fn saturating_add_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        let mut x = [0i64; 8];
        let mut y = [0i64; 8];
        self.store_i64(&mut x, a);
        self.store_i64(&mut y, b);
        for (dst, src) in x.iter_mut().zip(y.iter()) {
            *dst = dst.saturating_add(*src);
        }
        self.load_i64(&x)
    }

    #[inline(always)]
    fn saturating_sub_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        let mut x = [0i64; 8];
        let mut y = [0i64; 8];
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
            let sign = _mm512_set1_epi64(i64::MIN);
            _mm512_xor_si512(self.avg_round_u64(_mm512_xor_si512(a, sign), _mm512_xor_si512(b, sign)), sign)
        }
    }

    #[inline(always)]
    fn eq_u8(self, a: Self::u8, b: Self::u8) -> Self::m8 {
        unsafe { Avx512MaskI8(_mm512_cmpeq_epi8_mask(a, b)) }
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
        unsafe { Avx512MaskI8(_mm512_cmpgt_epu8_mask(a, b)) }
    }

    #[inline(always)]
    fn ge_u8(self, a: Self::u8, b: Self::u8) -> Self::m8 {
        self.lt_u8(a, b).not()
    }

    #[inline(always)]
    fn eq_i8(self, a: Self::i8, b: Self::i8) -> Self::m8 {
        unsafe { Avx512MaskI8(_mm512_cmpeq_epi8_mask(a, b)) }
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
        unsafe { Avx512MaskI8(_mm512_cmpgt_epi8_mask(a, b)) }
    }

    #[inline(always)]
    fn ge_i8(self, a: Self::i8, b: Self::i8) -> Self::m8 {
        self.lt_i8(a, b).not()
    }

    #[inline(always)]
    fn eq_u16(self, a: Self::u16, b: Self::u16) -> Self::m16 {
        unsafe { Avx512MaskI16(_mm512_cmpeq_epi16_mask(a, b)) }
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
        unsafe { Avx512MaskI16(_mm512_cmpgt_epu16_mask(a, b)) }
    }

    #[inline(always)]
    fn ge_u16(self, a: Self::u16, b: Self::u16) -> Self::m16 {
        self.lt_u16(a, b).not()
    }

    #[inline(always)]
    fn eq_i16(self, a: Self::i16, b: Self::i16) -> Self::m16 {
        unsafe { Avx512MaskI16(_mm512_cmpeq_epi16_mask(a, b)) }
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
        unsafe { Avx512MaskI16(_mm512_cmpgt_epi16_mask(a, b)) }
    }

    #[inline(always)]
    fn ge_i16(self, a: Self::i16, b: Self::i16) -> Self::m16 {
        self.lt_i16(a, b).not()
    }

    #[inline(always)]
    fn eq_u32(self, a: Self::u32, b: Self::u32) -> Self::m32 {
        unsafe { Avx512MaskI32(_mm512_cmpeq_epi32_mask(a, b)) }
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
        unsafe { Avx512MaskI32(_mm512_cmpgt_epu32_mask(a, b)) }
    }

    #[inline(always)]
    fn ge_u32(self, a: Self::u32, b: Self::u32) -> Self::m32 {
        self.lt_u32(a, b).not()
    }

    #[inline(always)]
    fn eq_i32(self, a: Self::i32, b: Self::i32) -> Self::m32 {
        unsafe { Avx512MaskI32(_mm512_cmpeq_epi32_mask(a, b)) }
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
        unsafe { Avx512MaskI32(_mm512_cmpgt_epi32_mask(a, b)) }
    }

    #[inline(always)]
    fn ge_i32(self, a: Self::i32, b: Self::i32) -> Self::m32 {
        self.lt_i32(a, b).not()
    }

    #[inline(always)]
    fn eq_u64(self, a: Self::u64, b: Self::u64) -> Self::m64 {
        unsafe { Avx512MaskI64(_mm512_cmpeq_epi64_mask(a, b)) }
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
        unsafe { Avx512MaskI64(_mm512_cmpgt_epu64_mask(a, b)) }
    }

    #[inline(always)]
    fn ge_u64(self, a: Self::u64, b: Self::u64) -> Self::m64 {
        self.lt_u64(a, b).not()
    }

    #[inline(always)]
    fn eq_i64(self, a: Self::i64, b: Self::i64) -> Self::m64 {
        unsafe { Avx512MaskI64(_mm512_cmpeq_epi64_mask(a, b)) }
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
        unsafe { Avx512MaskI64(_mm512_cmpgt_epi64_mask(a, b)) }
    }

    #[inline(always)]
    fn ge_i64(self, a: Self::i64, b: Self::i64) -> Self::m64 {
        self.lt_i64(a, b).not()
    }

    #[inline(always)]
    fn shl_u8(self, a: Self::u8, n: u32) -> Self::u8 {
        let mut data = [0u8; 64];
        self.store_u8(&mut data, a);
        for value in data.iter_mut() {
            *value = value.wrapping_shl(n);
        }
        self.load_u8(&data)
    }

    #[inline(always)]
    fn shr_u8(self, a: Self::u8, n: u32) -> Self::u8 {
        let mut data = [0u8; 64];
        self.store_u8(&mut data, a);
        for value in data.iter_mut() {
            *value = value.wrapping_shr(n);
        }
        self.load_u8(&data)
    }

    #[inline(always)]
    fn shl_i8(self, a: Self::i8, n: u32) -> Self::i8 {
        let mut data = [0i8; 64];
        self.store_i8(&mut data, a);
        for value in data.iter_mut() {
            *value = value.wrapping_shl(n);
        }
        self.load_i8(&data)
    }

    #[inline(always)]
    fn shr_i8(self, a: Self::i8, n: u32) -> Self::i8 {
        let mut data = [0i8; 64];
        self.store_i8(&mut data, a);
        for value in data.iter_mut() {
            *value = value.wrapping_shr(n);
        }
        self.load_i8(&data)
    }

    #[inline(always)]
    fn shl_u16(self, a: Self::u16, n: u32) -> Self::u16 {
        unsafe { _mm512_sll_epi16(a, _mm_cvtsi32_si128((n & 15) as i32)) }
    }

    #[inline(always)]
    fn shr_u16(self, a: Self::u16, n: u32) -> Self::u16 {
        unsafe { _mm512_srl_epi16(a, _mm_cvtsi32_si128((n & 15) as i32)) }
    }

    #[inline(always)]
    fn shl_i16(self, a: Self::i16, n: u32) -> Self::i16 {
        unsafe { _mm512_sll_epi16(a, _mm_cvtsi32_si128((n & 15) as i32)) }
    }

    #[inline(always)]
    fn shr_i16(self, a: Self::i16, n: u32) -> Self::i16 {
        unsafe { _mm512_sra_epi16(a, _mm_cvtsi32_si128((n & 15) as i32)) }
    }

    #[inline(always)]
    fn shl_u32(self, a: Self::u32, n: u32) -> Self::u32 {
        unsafe { _mm512_sll_epi32(a, _mm_cvtsi32_si128((n & 31) as i32)) }
    }

    #[inline(always)]
    fn shr_u32(self, a: Self::u32, n: u32) -> Self::u32 {
        unsafe { _mm512_srl_epi32(a, _mm_cvtsi32_si128((n & 31) as i32)) }
    }

    #[inline(always)]
    fn shl_i32(self, a: Self::i32, n: u32) -> Self::i32 {
        unsafe { _mm512_sll_epi32(a, _mm_cvtsi32_si128((n & 31) as i32)) }
    }

    #[inline(always)]
    fn shr_i32(self, a: Self::i32, n: u32) -> Self::i32 {
        unsafe { _mm512_sra_epi32(a, _mm_cvtsi32_si128((n & 31) as i32)) }
    }

    #[inline(always)]
    fn shl_u64(self, a: Self::u64, n: u32) -> Self::u64 {
        unsafe { _mm512_sll_epi64(a, _mm_cvtsi32_si128((n & 63) as i32)) }
    }

    #[inline(always)]
    fn shr_u64(self, a: Self::u64, n: u32) -> Self::u64 {
        unsafe { _mm512_srl_epi64(a, _mm_cvtsi32_si128((n & 63) as i32)) }
    }

    #[inline(always)]
    fn shl_i64(self, a: Self::i64, n: u32) -> Self::i64 {
        unsafe { _mm512_sll_epi64(a, _mm_cvtsi32_si128((n & 63) as i32)) }
    }

    #[inline(always)]
    fn shr_i64(self, a: Self::i64, n: u32) -> Self::i64 {
        let mut data = [0i64; 8];
        self.store_i64(&mut data, a);
        for value in data.iter_mut() {
            *value = value.wrapping_shr(n);
        }
        self.load_i64(&data)
    }

    #[inline(always)]
    fn select_u8(self, mask: Self::m8, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { _mm512_mask_blend_epi8(mask.0, b, a) }
    }

    #[inline(always)]
    fn select_i8(self, mask: Self::m8, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { _mm512_mask_blend_epi8(mask.0, b, a) }
    }

    #[inline(always)]
    fn select_u16(self, mask: Self::m16, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { _mm512_mask_blend_epi16(mask.0, b, a) }
    }

    #[inline(always)]
    fn select_i16(self, mask: Self::m16, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { _mm512_mask_blend_epi16(mask.0, b, a) }
    }

    #[inline(always)]
    fn select_u32(self, mask: Self::m32, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { _mm512_mask_blend_epi32(mask.0, b, a) }
    }

    #[inline(always)]
    fn select_i32(self, mask: Self::m32, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { _mm512_mask_blend_epi32(mask.0, b, a) }
    }

    #[inline(always)]
    fn select_u64(self, mask: Self::m64, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { _mm512_mask_blend_epi64(mask.0, b, a) }
    }

    #[inline(always)]
    fn select_i64(self, mask: Self::m64, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { _mm512_mask_blend_epi64(mask.0, b, a) }
    }

    #[inline(always)]
    fn widen_low_u8_to_u16(self, x: Self::u8) -> Self::u16 {
        unsafe { _mm512_cvtepu8_epi16(_mm512_castsi512_si256(x)) }
    }

    #[inline(always)]
    fn widen_high_u8_to_u16(self, x: Self::u8) -> Self::u16 {
        unsafe { _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(x)) }
    }

    #[inline(always)]
    fn widen_low_i8_to_i16(self, x: Self::i8) -> Self::i16 {
        unsafe { _mm512_cvtepi8_epi16(_mm512_castsi512_si256(x)) }
    }

    #[inline(always)]
    fn widen_high_i8_to_i16(self, x: Self::i8) -> Self::i16 {
        unsafe { _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(x)) }
    }

    #[inline(always)]
    fn widen_low_u16_to_u32(self, x: Self::u16) -> Self::u32 {
        unsafe { _mm512_cvtepu16_epi32(_mm512_castsi512_si256(x)) }
    }

    #[inline(always)]
    fn widen_high_u16_to_u32(self, x: Self::u16) -> Self::u32 {
        unsafe { _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(x)) }
    }

    #[inline(always)]
    fn widen_low_i16_to_i32(self, x: Self::i16) -> Self::i32 {
        unsafe { _mm512_cvtepi16_epi32(_mm512_castsi512_si256(x)) }
    }

    #[inline(always)]
    fn widen_high_i16_to_i32(self, x: Self::i16) -> Self::i32 {
        unsafe { _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(x)) }
    }

    #[inline(always)]
    fn widen_low_u32_to_u64(self, x: Self::u32) -> Self::u64 {
        unsafe { _mm512_cvtepu32_epi64(_mm512_castsi512_si256(x)) }
    }

    #[inline(always)]
    fn widen_high_u32_to_u64(self, x: Self::u32) -> Self::u64 {
        unsafe { _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<1>(x)) }
    }

    #[inline(always)]
    fn widen_low_i32_to_i64(self, x: Self::i32) -> Self::i64 {
        unsafe { _mm512_cvtepi32_epi64(_mm512_castsi512_si256(x)) }
    }

    #[inline(always)]
    fn widen_high_i32_to_i64(self, x: Self::i32) -> Self::i64 {
        unsafe { _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(x)) }
    }

    #[inline(always)]
    fn narrow_saturating_u16_to_u8(self, lo: Self::u16, hi: Self::u16) -> Self::u8 {
        unsafe {
            let max = _mm512_set1_epi16(u8::MAX as i16);
            let lo = _mm512_min_epu16(lo, max);
            let hi = _mm512_min_epu16(hi, max);
            let lo = _mm512_cvtusepi16_epi8(lo);
            let hi = _mm512_cvtusepi16_epi8(hi);
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
        }
    }

    #[inline(always)]
    fn narrow_saturating_i16_to_i8(self, lo: Self::i16, hi: Self::i16) -> Self::i8 {
        unsafe {
            let lo = _mm512_cvtsepi16_epi8(lo);
            let hi = _mm512_cvtsepi16_epi8(hi);
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
        }
    }

    #[inline(always)]
    fn narrow_saturating_u32_to_u16(self, lo: Self::u32, hi: Self::u32) -> Self::u16 {
        unsafe {
            let max = _mm512_set1_epi32(u16::MAX as i32);
            let lo = _mm512_min_epu32(lo, max);
            let hi = _mm512_min_epu32(hi, max);
            let lo = _mm512_cvtusepi32_epi16(lo);
            let hi = _mm512_cvtusepi32_epi16(hi);
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
        }
    }

    #[inline(always)]
    fn narrow_saturating_i32_to_i16(self, lo: Self::i32, hi: Self::i32) -> Self::i16 {
        unsafe {
            let lo = _mm512_cvtsepi32_epi16(lo);
            let hi = _mm512_cvtsepi32_epi16(hi);
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
        }
    }

    #[inline(always)]
    fn narrow_saturating_u64_to_u32(self, lo: Self::u64, hi: Self::u64) -> Self::u32 {
        unsafe {
            let max = _mm512_set1_epi64(u32::MAX as i64);
            let lo = _mm512_min_epu64(lo, max);
            let hi = _mm512_min_epu64(hi, max);
            let lo = _mm512_cvtusepi64_epi32(lo);
            let hi = _mm512_cvtusepi64_epi32(hi);
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
        }
    }

    #[inline(always)]
    fn narrow_saturating_i64_to_i32(self, lo: Self::i64, hi: Self::i64) -> Self::i32 {
        unsafe {
            let lo = _mm512_cvtsepi64_epi32(lo);
            let hi = _mm512_cvtsepi64_epi32(hi);
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
        }
    }
}
