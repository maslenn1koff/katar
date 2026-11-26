use crate::{
    SimdFn,
    traits::{FloatSimd, IntegerSimd, MaskSimd, Simd},
};
use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub fn vectorize<Op: SimdFn>(op: Op) -> Op::Result {
    op.call(Neon)
}

#[derive(Copy, Clone, Debug)]
pub struct NeonMaskF32(uint32x4_t);

impl MaskSimd for NeonMaskF32 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        unsafe { Self(vdupq_n_u32(if v { u32::MAX } else { 0 })) }
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        unsafe {
            Self(vld1q_u32(
                [
                    if v & 1 != 0 { u32::MAX } else { 0 },
                    if v & 2 != 0 { u32::MAX } else { 0 },
                    if v & 4 != 0 { u32::MAX } else { 0 },
                    if v & 8 != 0 { u32::MAX } else { 0 },
                ]
                .as_ptr(),
            ))
        }
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        unsafe {
            let mut lanes = [0u32; 4];
            vst1q_u32(lanes.as_mut_ptr(), self.0);
            let mut bits = 0u64;
            for (i, lane) in lanes.iter().enumerate() {
                if *lane & 0x8000_0000 != 0 {
                    bits |= 1u64 << i;
                }
            }
            bits
        }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(vandq_u32(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(vorrq_u32(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(vmvnq_u32(self.0)) }
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.to_bits() != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.to_bits() == 0b1111
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NeonMaskF64(uint64x2_t);

impl MaskSimd for NeonMaskF64 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        unsafe { Self(vdupq_n_u64(if v { u64::MAX } else { 0 })) }
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        unsafe { Self(vld1q_u64([if v & 1 != 0 { u64::MAX } else { 0 }, if v & 2 != 0 { u64::MAX } else { 0 }].as_ptr())) }
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        unsafe {
            let mut lanes = [0u64; 2];
            vst1q_u64(lanes.as_mut_ptr(), self.0);
            let mut bits = 0u64;
            for (i, lane) in lanes.iter().enumerate() {
                if *lane & 0x8000_0000_0000_0000 != 0 {
                    bits |= 1u64 << i;
                }
            }
            bits
        }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(vandq_u64(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(vorrq_u64(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(veorq_u64(self.0, vdupq_n_u64(u64::MAX))) }
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.to_bits() != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.to_bits() == 0b11
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NeonMaskI8(uint8x16_t);

impl MaskSimd for NeonMaskI8 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        unsafe { Self(vdupq_n_u8(if v { u8::MAX } else { 0 })) }
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        unsafe {
            let mut lanes = [0u8; 16];
            for (i, lane) in lanes.iter_mut().enumerate() {
                *lane = if v & (1u64 << i) != 0 { u8::MAX } else { 0 };
            }
            Self(vld1q_u8(lanes.as_ptr()))
        }
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        unsafe {
            let mut lanes = [0u8; 16];
            vst1q_u8(lanes.as_mut_ptr(), self.0);
            let mut bits = 0u64;
            for (i, lane) in lanes.iter().enumerate() {
                if *lane & 0x80 != 0 {
                    bits |= 1u64 << i;
                }
            }
            bits
        }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(vandq_u8(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(vorrq_u8(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(veorq_u8(self.0, vdupq_n_u8(u8::MAX))) }
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
pub struct NeonMaskI16(uint16x8_t);

impl MaskSimd for NeonMaskI16 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        unsafe { Self(vdupq_n_u16(if v { u16::MAX } else { 0 })) }
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        unsafe {
            let mut lanes = [0u16; 8];
            for (i, lane) in lanes.iter_mut().enumerate() {
                *lane = if v & (1u64 << i) != 0 { u16::MAX } else { 0 };
            }
            Self(vld1q_u16(lanes.as_ptr()))
        }
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        unsafe {
            let mut lanes = [0u16; 8];
            vst1q_u16(lanes.as_mut_ptr(), self.0);
            let mut bits = 0u64;
            for (i, lane) in lanes.iter().enumerate() {
                if *lane & 0x8000 != 0 {
                    bits |= 1u64 << i;
                }
            }
            bits
        }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(vandq_u16(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(vorrq_u16(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(veorq_u16(self.0, vdupq_n_u16(u16::MAX))) }
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
pub struct NeonMaskI32(uint32x4_t);

impl MaskSimd for NeonMaskI32 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        unsafe { Self(vdupq_n_u32(if v { u32::MAX } else { 0 })) }
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        unsafe {
            let mut lanes = [0u32; 4];
            for (i, lane) in lanes.iter_mut().enumerate() {
                *lane = if v & (1u64 << i) != 0 { u32::MAX } else { 0 };
            }
            Self(vld1q_u32(lanes.as_ptr()))
        }
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        unsafe {
            let mut lanes = [0u32; 4];
            vst1q_u32(lanes.as_mut_ptr(), self.0);
            let mut bits = 0u64;
            for (i, lane) in lanes.iter().enumerate() {
                if *lane & 0x8000_0000 != 0 {
                    bits |= 1u64 << i;
                }
            }
            bits
        }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(vandq_u32(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(vorrq_u32(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(veorq_u32(self.0, vdupq_n_u32(u32::MAX))) }
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
pub struct NeonMaskI64(uint64x2_t);

impl MaskSimd for NeonMaskI64 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        unsafe { Self(vdupq_n_u64(if v { u64::MAX } else { 0 })) }
    }

    #[inline(always)]
    fn from_bits(v: u64) -> Self {
        unsafe {
            let mut lanes = [0u64; 2];
            for (i, lane) in lanes.iter_mut().enumerate() {
                *lane = if v & (1u64 << i) != 0 { u64::MAX } else { 0 };
            }
            Self(vld1q_u64(lanes.as_ptr()))
        }
    }

    #[inline(always)]
    fn to_bits(self) -> u64 {
        unsafe {
            let mut lanes = [0u64; 2];
            vst1q_u64(lanes.as_mut_ptr(), self.0);
            let mut bits = 0u64;
            for (i, lane) in lanes.iter().enumerate() {
                if *lane & 0x8000_0000_0000_0000 != 0 {
                    bits |= 1u64 << i;
                }
            }
            bits
        }
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(vandq_u64(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(vorrq_u64(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(veorq_u64(self.0, vdupq_n_u64(u64::MAX))) }
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
pub struct Neon;

impl FloatSimd for Neon {
    type f32 = float32x4_t;
    type f64 = float64x2_t;

    type m32 = NeonMaskF32;
    type m64 = NeonMaskF64;

    const f32_lanes: usize = 4;
    const f64_lanes: usize = 2;

    #[inline(always)]
    fn splat_f32(self, value: f32) -> Self::f32 {
        unsafe { vdupq_n_f32(value) }
    }

    #[inline(always)]
    fn splat_f64(self, value: f64) -> Self::f64 {
        unsafe { vdupq_n_f64(value) }
    }

    #[inline(always)]
    fn load_f32(self, src: &[f32]) -> Self::f32 {
        assert!(src.len() >= Self::f32_lanes);
        unsafe { vld1q_f32(src.as_ptr()) }
    }

    #[inline(always)]
    fn load_f64(self, src: &[f64]) -> Self::f64 {
        assert!(src.len() >= Self::f64_lanes);
        unsafe { vld1q_f64(src.as_ptr()) }
    }

    #[inline(always)]
    fn store_f32(self, dst: &mut [f32], value: Self::f32) {
        assert!(dst.len() >= Self::f32_lanes);
        unsafe { vst1q_f32(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    fn store_f64(self, dst: &mut [f64], value: Self::f64) {
        assert!(dst.len() >= Self::f64_lanes);
        unsafe { vst1q_f64(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    unsafe fn load_unchecked_f32(self, src: &[f32]) -> Self::f32 {
        unsafe { vld1q_f32(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_f64(self, src: &[f64]) -> Self::f64 {
        unsafe { vld1q_f64(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f32(self, dst: &mut [f32], value: Self::f32) {
        unsafe { vst1q_f32(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f64(self, dst: &mut [f64], value: Self::f64) {
        unsafe { vst1q_f64(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    fn abs_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { vabsq_f32(a) }
    }

    #[inline(always)]
    fn abs_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { vabsq_f64(a) }
    }

    #[inline(always)]
    fn add_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vaddq_f32(a, b) }
    }

    #[inline(always)]
    fn add_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vaddq_f64(a, b) }
    }

    #[inline(always)]
    fn sub_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vsubq_f32(a, b) }
    }

    #[inline(always)]
    fn sub_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vsubq_f64(a, b) }
    }

    #[inline(always)]
    fn mul_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vmulq_f32(a, b) }
    }

    #[inline(always)]
    fn mul_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vmulq_f64(a, b) }
    }

    #[inline(always)]
    fn div_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vdivq_f32(a, b) }
    }

    #[inline(always)]
    fn div_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vdivq_f64(a, b) }
    }

    #[inline(always)]
    fn max_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vmaxq_f32(a, b) }
    }

    #[inline(always)]
    fn max_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vmaxq_f64(a, b) }
    }

    #[inline(always)]
    fn min_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vminq_f32(a, b) }
    }

    #[inline(always)]
    fn min_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vminq_f64(a, b) }
    }

    #[inline(always)]
    fn neg_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { vnegq_f32(a) }
    }

    #[inline(always)]
    fn neg_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { vnegq_f64(a) }
    }

    #[inline(always)]
    fn sqrt_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { vsqrtq_f32(a) }
    }

    #[inline(always)]
    fn sqrt_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { vsqrtq_f64(a) }
    }

    #[inline(always)]
    fn mul_add_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { vfmaq_f32(c, a, b) }
    }

    #[inline(always)]
    fn mul_add_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { vfmaq_f64(c, a, b) }
    }

    #[inline(always)]
    fn mul_sub_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { vfmaq_f32(vnegq_f32(c), a, b) }
    }

    #[inline(always)]
    fn mul_sub_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { vfmaq_f64(vnegq_f64(c), a, b) }
    }

    #[inline(always)]
    fn nmul_add_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { vfmsq_f32(c, a, b) }
    }

    #[inline(always)]
    fn nmul_add_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { vfmsq_f64(c, a, b) }
    }

    #[inline(always)]
    fn nmul_sub_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { vfmsq_f32(vnegq_f32(c), a, b) }
    }

    #[inline(always)]
    fn nmul_sub_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { vfmsq_f64(vnegq_f64(c), a, b) }
    }

    #[inline(always)]
    fn eq_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { NeonMaskF32(vceqq_f32(a, b)) }
    }

    #[inline(always)]
    fn eq_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { NeonMaskF64(vceqq_f64(a, b)) }
    }

    #[inline(always)]
    fn ge_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { NeonMaskF32(vcgeq_f32(a, b)) }
    }

    #[inline(always)]
    fn ge_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { NeonMaskF64(vcgeq_f64(a, b)) }
    }

    #[inline(always)]
    fn gt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { NeonMaskF32(vcgtq_f32(a, b)) }
    }

    #[inline(always)]
    fn gt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { NeonMaskF64(vcgtq_f64(a, b)) }
    }

    #[inline(always)]
    fn le_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { NeonMaskF32(vcleq_f32(a, b)) }
    }

    #[inline(always)]
    fn le_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { NeonMaskF64(vcleq_f64(a, b)) }
    }

    #[inline(always)]
    fn lt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { NeonMaskF32(vcltq_f32(a, b)) }
    }

    #[inline(always)]
    fn lt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { NeonMaskF64(vcltq_f64(a, b)) }
    }

    #[inline(always)]
    fn ne_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { NeonMaskF32(vmvnq_u32(vceqq_f32(a, b))) }
    }

    #[inline(always)]
    fn ne_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { NeonMaskF64(veorq_u64(vceqq_f64(a, b), vdupq_n_u64(u64::MAX))) }
    }

    #[inline(always)]
    fn select_f32(self, mask: Self::m32, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vbslq_f32(mask.0, a, b) }
    }

    #[inline(always)]
    fn select_f64(self, mask: Self::m64, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vbslq_f64(mask.0, a, b) }
    }

    #[inline(always)]
    fn bitand_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))) }
    }

    #[inline(always)]
    fn bitor_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))) }
    }

    #[inline(always)]
    fn bitxor_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))) }
    }

    #[inline(always)]
    fn bitand_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b))) }
    }

    #[inline(always)]
    fn bitor_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b))) }
    }

    #[inline(always)]
    fn bitxor_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b))) }
    }

    #[inline(always)]
    fn bitnot_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a))) }
    }

    #[inline(always)]
    fn bitnot_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a), vdupq_n_u64(u64::MAX))) }
    }

    #[inline(always)]
    fn bitandnot_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(b), vreinterpretq_u32_f32(a))) }
    }

    #[inline(always)]
    fn bitandnot_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(b), vreinterpretq_u64_f64(a))) }
    }

    #[inline(always)]
    fn reduce_max_f32(self, a: Self::f32) -> f32 {
        unsafe { vmaxvq_f32(a) }
    }

    #[inline(always)]
    fn reduce_max_f64(self, a: Self::f64) -> f64 {
        unsafe { vmaxvq_f64(a) }
    }

    #[inline(always)]
    fn reduce_min_f32(self, a: Self::f32) -> f32 {
        unsafe { vminvq_f32(a) }
    }

    #[inline(always)]
    fn reduce_min_f64(self, a: Self::f64) -> f64 {
        unsafe { vminvq_f64(a) }
    }

    #[inline(always)]
    fn reduce_sum_f32(self, a: Self::f32) -> f32 {
        unsafe { vaddvq_f32(a) }
    }

    #[inline(always)]
    fn reduce_sum_f64(self, a: Self::f64) -> f64 {
        unsafe { vaddvq_f64(a) }
    }
}

impl IntegerSimd for Neon {
    type u8 = uint8x16_t;
    type i8 = int8x16_t;
    type u16 = uint16x8_t;
    type i16 = int16x8_t;
    type u32 = uint32x4_t;
    type i32 = int32x4_t;
    type u64 = uint64x2_t;
    type i64 = int64x2_t;

    type m8 = NeonMaskI8;
    type m16 = NeonMaskI16;
    type m32 = NeonMaskI32;
    type m64 = NeonMaskI64;

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
        unsafe { vdupq_n_u8(value) }
    }

    #[inline(always)]
    fn splat_i8(self, value: i8) -> Self::i8 {
        unsafe { vdupq_n_s8(value) }
    }

    #[inline(always)]
    fn splat_u16(self, value: u16) -> Self::u16 {
        unsafe { vdupq_n_u16(value) }
    }

    #[inline(always)]
    fn splat_i16(self, value: i16) -> Self::i16 {
        unsafe { vdupq_n_s16(value) }
    }

    #[inline(always)]
    fn splat_u32(self, value: u32) -> Self::u32 {
        unsafe { vdupq_n_u32(value) }
    }

    #[inline(always)]
    fn splat_i32(self, value: i32) -> Self::i32 {
        unsafe { vdupq_n_s32(value) }
    }

    #[inline(always)]
    fn splat_u64(self, value: u64) -> Self::u64 {
        unsafe { vdupq_n_u64(value) }
    }

    #[inline(always)]
    fn splat_i64(self, value: i64) -> Self::i64 {
        unsafe { vdupq_n_s64(value) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_u8(self, src: &[u8]) -> Self::u8 {
        unsafe { vld1q_u8(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_i8(self, src: &[i8]) -> Self::i8 {
        unsafe { vld1q_s8(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_u16(self, src: &[u16]) -> Self::u16 {
        unsafe { vld1q_u16(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_i16(self, src: &[i16]) -> Self::i16 {
        unsafe { vld1q_s16(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_u32(self, src: &[u32]) -> Self::u32 {
        unsafe { vld1q_u32(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_i32(self, src: &[i32]) -> Self::i32 {
        unsafe { vld1q_s32(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_u64(self, src: &[u64]) -> Self::u64 {
        unsafe { vld1q_u64(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_i64(self, src: &[i64]) -> Self::i64 {
        unsafe { vld1q_s64(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_u8(self, dst: &mut [u8], value: Self::u8) {
        unsafe { vst1q_u8(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_i8(self, dst: &mut [i8], value: Self::i8) {
        unsafe { vst1q_s8(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_u16(self, dst: &mut [u16], value: Self::u16) {
        unsafe { vst1q_u16(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_i16(self, dst: &mut [i16], value: Self::i16) {
        unsafe { vst1q_s16(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_u32(self, dst: &mut [u32], value: Self::u32) {
        unsafe { vst1q_u32(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_i32(self, dst: &mut [i32], value: Self::i32) {
        unsafe { vst1q_s32(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_u64(self, dst: &mut [u64], value: Self::u64) {
        unsafe { vst1q_u64(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_i64(self, dst: &mut [i64], value: Self::i64) {
        unsafe { vst1q_s64(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    fn bitnot_u8(self, a: Self::u8) -> Self::u8 {
        unsafe { veorq_u8(a, vdupq_n_u8(u8::MAX)) }
    }

    #[inline(always)]
    fn bitnot_i8(self, a: Self::i8) -> Self::i8 {
        unsafe { vreinterpretq_s8_u8(veorq_u8(vreinterpretq_u8_s8(a), vdupq_n_u8(u8::MAX))) }
    }

    #[inline(always)]
    fn abs_i8(self, a: Self::i8) -> Self::i8 {
        unsafe {
            let sign = vcgtq_s8(vdupq_n_s8(0), a);
            let value = veorq_u8(vreinterpretq_u8_s8(a), sign);
            vreinterpretq_s8_u8(vsubq_u8(value, sign))
        }
    }

    #[inline(always)]
    fn neg_i8(self, a: Self::i8) -> Self::i8 {
        unsafe { vsubq_s8(vdupq_n_s8(0), a) }
    }

    #[inline(always)]
    fn bitnot_u16(self, a: Self::u16) -> Self::u16 {
        unsafe { veorq_u16(a, vdupq_n_u16(u16::MAX)) }
    }

    #[inline(always)]
    fn bitnot_i16(self, a: Self::i16) -> Self::i16 {
        unsafe { vreinterpretq_s16_u16(veorq_u16(vreinterpretq_u16_s16(a), vdupq_n_u16(u16::MAX))) }
    }

    #[inline(always)]
    fn abs_i16(self, a: Self::i16) -> Self::i16 {
        unsafe {
            let sign = vcgtq_s16(vdupq_n_s16(0), a);
            let value = veorq_u16(vreinterpretq_u16_s16(a), sign);
            vreinterpretq_s16_u16(vsubq_u16(value, sign))
        }
    }

    #[inline(always)]
    fn neg_i16(self, a: Self::i16) -> Self::i16 {
        unsafe { vsubq_s16(vdupq_n_s16(0), a) }
    }

    #[inline(always)]
    fn bitnot_u32(self, a: Self::u32) -> Self::u32 {
        unsafe { veorq_u32(a, vdupq_n_u32(u32::MAX)) }
    }

    #[inline(always)]
    fn bitnot_i32(self, a: Self::i32) -> Self::i32 {
        unsafe { vreinterpretq_s32_u32(veorq_u32(vreinterpretq_u32_s32(a), vdupq_n_u32(u32::MAX))) }
    }

    #[inline(always)]
    fn abs_i32(self, a: Self::i32) -> Self::i32 {
        unsafe {
            let sign = vcgtq_s32(vdupq_n_s32(0), a);
            let value = veorq_u32(vreinterpretq_u32_s32(a), sign);
            vreinterpretq_s32_u32(vsubq_u32(value, sign))
        }
    }

    #[inline(always)]
    fn neg_i32(self, a: Self::i32) -> Self::i32 {
        unsafe { vsubq_s32(vdupq_n_s32(0), a) }
    }

    #[inline(always)]
    fn bitnot_u64(self, a: Self::u64) -> Self::u64 {
        unsafe { veorq_u64(a, vdupq_n_u64(u64::MAX)) }
    }

    #[inline(always)]
    fn bitnot_i64(self, a: Self::i64) -> Self::i64 {
        unsafe { vreinterpretq_s64_u64(veorq_u64(vreinterpretq_u64_s64(a), vdupq_n_u64(u64::MAX))) }
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
        unsafe { vsubq_s64(vdupq_n_s64(0), a) }
    }

    #[inline(always)]
    fn count_ones_u8(self, a: Self::u8) -> Self::u8 {
        unsafe { vcntq_u8(a) }
    }

    #[inline(always)]
    fn count_ones_i8(self, a: Self::i8) -> Self::i8 {
        unsafe { vreinterpretq_s8_u8(self.count_ones_u8(vreinterpretq_u8_s8(a))) }
    }

    #[inline(always)]
    fn count_ones_u16(self, a: Self::u16) -> Self::u16 {
        unsafe { vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u16(a))) }
    }

    #[inline(always)]
    fn count_ones_i16(self, a: Self::i16) -> Self::i16 {
        unsafe { vreinterpretq_s16_u16(self.count_ones_u16(vreinterpretq_u16_s16(a))) }
    }

    #[inline(always)]
    fn count_ones_u32(self, a: Self::u32) -> Self::u32 {
        unsafe { vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(a)))) }
    }

    #[inline(always)]
    fn count_ones_i32(self, a: Self::i32) -> Self::i32 {
        unsafe { vreinterpretq_s32_u32(self.count_ones_u32(vreinterpretq_u32_s32(a))) }
    }

    #[inline(always)]
    fn count_ones_u64(self, a: Self::u64) -> Self::u64 {
        unsafe { vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(a))))) }
    }

    #[inline(always)]
    fn count_ones_i64(self, a: Self::i64) -> Self::i64 {
        unsafe { vreinterpretq_s64_u64(self.count_ones_u64(vreinterpretq_u64_s64(a))) }
    }

    #[inline(always)]
    fn leading_zeros_u8(self, a: Self::u8) -> Self::u8 {
        unsafe { vclzq_u8(a) }
    }

    #[inline(always)]
    fn leading_zeros_i8(self, a: Self::i8) -> Self::i8 {
        unsafe { vreinterpretq_s8_u8(self.leading_zeros_u8(vreinterpretq_u8_s8(a))) }
    }

    #[inline(always)]
    fn leading_zeros_u16(self, a: Self::u16) -> Self::u16 {
        unsafe { vclzq_u16(a) }
    }

    #[inline(always)]
    fn leading_zeros_i16(self, a: Self::i16) -> Self::i16 {
        unsafe { vreinterpretq_s16_u16(self.leading_zeros_u16(vreinterpretq_u16_s16(a))) }
    }

    #[inline(always)]
    fn leading_zeros_u32(self, a: Self::u32) -> Self::u32 {
        unsafe { vclzq_u32(a) }
    }

    #[inline(always)]
    fn leading_zeros_i32(self, a: Self::i32) -> Self::i32 {
        unsafe { vreinterpretq_s32_u32(self.leading_zeros_u32(vreinterpretq_u32_s32(a))) }
    }

    #[inline(always)]
    fn leading_zeros_u64(self, a: Self::u64) -> Self::u64 {
        let mut data = [0u64; 2];
        self.store_u64(&mut data, a);
        for value in data.iter_mut() {
            *value = value.leading_zeros() as u64;
        }
        self.load_u64(&data)
    }

    #[inline(always)]
    fn leading_zeros_i64(self, a: Self::i64) -> Self::i64 {
        unsafe { vreinterpretq_s64_u64(self.leading_zeros_u64(vreinterpretq_u64_s64(a))) }
    }

    #[inline(always)]
    fn trailing_zeros_u8(self, a: Self::u8) -> Self::u8 {
        unsafe {
            let low = vandq_u8(a, vsubq_u8(vdupq_n_u8(0), a));
            vcntq_u8(vsubq_u8(low, vdupq_n_u8(1)))
        }
    }

    #[inline(always)]
    fn trailing_zeros_i8(self, a: Self::i8) -> Self::i8 {
        unsafe { vreinterpretq_s8_u8(self.trailing_zeros_u8(vreinterpretq_u8_s8(a))) }
    }

    #[inline(always)]
    fn trailing_zeros_u16(self, a: Self::u16) -> Self::u16 {
        unsafe {
            let low = vandq_u16(a, vsubq_u16(vdupq_n_u16(0), a));
            let mask = vsubq_u16(low, vdupq_n_u16(1));
            vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u16(mask)))
        }
    }

    #[inline(always)]
    fn trailing_zeros_i16(self, a: Self::i16) -> Self::i16 {
        unsafe { vreinterpretq_s16_u16(self.trailing_zeros_u16(vreinterpretq_u16_s16(a))) }
    }

    #[inline(always)]
    fn trailing_zeros_u32(self, a: Self::u32) -> Self::u32 {
        unsafe {
            let low = vandq_u32(a, vsubq_u32(vdupq_n_u32(0), a));
            let mask = vsubq_u32(low, vdupq_n_u32(1));
            vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(mask))))
        }
    }

    #[inline(always)]
    fn trailing_zeros_i32(self, a: Self::i32) -> Self::i32 {
        unsafe { vreinterpretq_s32_u32(self.trailing_zeros_u32(vreinterpretq_u32_s32(a))) }
    }

    #[inline(always)]
    fn trailing_zeros_u64(self, a: Self::u64) -> Self::u64 {
        unsafe {
            let low = vandq_u64(a, vsubq_u64(vdupq_n_u64(0), a));
            let mask = vsubq_u64(low, vdupq_n_u64(1));
            vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(mask)))))
        }
    }

    #[inline(always)]
    fn trailing_zeros_i64(self, a: Self::i64) -> Self::i64 {
        unsafe { vreinterpretq_s64_u64(self.trailing_zeros_u64(vreinterpretq_u64_s64(a))) }
    }

    #[inline(always)]
    fn add_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { vaddq_u8(a, b) }
    }

    #[inline(always)]
    fn sub_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { vsubq_u8(a, b) }
    }

    #[inline(always)]
    fn mul_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { vmulq_u8(a, b) }
    }

    #[inline(always)]
    fn min_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { vminq_u8(a, b) }
    }

    #[inline(always)]
    fn max_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { vmaxq_u8(a, b) }
    }

    #[inline(always)]
    fn bitand_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { vandq_u8(a, b) }
    }

    #[inline(always)]
    fn bitor_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { vorrq_u8(a, b) }
    }

    #[inline(always)]
    fn bitxor_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { veorq_u8(a, b) }
    }

    #[inline(always)]
    fn bitandnot_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { vandq_u8(b, veorq_u8(a, vdupq_n_u8(u8::MAX))) }
    }

    #[inline(always)]
    fn saturating_add_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { vqaddq_u8(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { vqsubq_u8(a, b) }
    }

    #[inline(always)]
    fn avg_round_u8(self, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { vrhaddq_u8(a, b) }
    }

    #[inline(always)]
    fn add_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { vaddq_s8(a, b) }
    }

    #[inline(always)]
    fn sub_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { vsubq_s8(a, b) }
    }

    #[inline(always)]
    fn mul_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { vmulq_s8(a, b) }
    }

    #[inline(always)]
    fn min_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { vminq_s8(a, b) }
    }

    #[inline(always)]
    fn max_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { vmaxq_s8(a, b) }
    }

    #[inline(always)]
    fn bitand_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(a), vreinterpretq_u8_s8(b))) }
    }

    #[inline(always)]
    fn bitor_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(a), vreinterpretq_u8_s8(b))) }
    }

    #[inline(always)]
    fn bitxor_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { vreinterpretq_s8_u8(veorq_u8(vreinterpretq_u8_s8(a), vreinterpretq_u8_s8(b))) }
    }

    #[inline(always)]
    fn bitandnot_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(b), veorq_u8(vreinterpretq_u8_s8(a), vdupq_n_u8(u8::MAX)))) }
    }

    #[inline(always)]
    fn saturating_add_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { vqaddq_s8(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { vqsubq_s8(a, b) }
    }

    #[inline(always)]
    fn avg_round_i8(self, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { vrhaddq_s8(a, b) }
    }

    #[inline(always)]
    fn add_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { vaddq_u16(a, b) }
    }

    #[inline(always)]
    fn sub_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { vsubq_u16(a, b) }
    }

    #[inline(always)]
    fn mul_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { vmulq_u16(a, b) }
    }

    #[inline(always)]
    fn min_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { vminq_u16(a, b) }
    }

    #[inline(always)]
    fn max_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { vmaxq_u16(a, b) }
    }

    #[inline(always)]
    fn bitand_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { vandq_u16(a, b) }
    }

    #[inline(always)]
    fn bitor_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { vorrq_u16(a, b) }
    }

    #[inline(always)]
    fn bitxor_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { veorq_u16(a, b) }
    }

    #[inline(always)]
    fn bitandnot_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { vandq_u16(b, veorq_u16(a, vdupq_n_u16(u16::MAX))) }
    }

    #[inline(always)]
    fn saturating_add_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { vqaddq_u16(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { vqsubq_u16(a, b) }
    }

    #[inline(always)]
    fn avg_round_u16(self, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { vrhaddq_u16(a, b) }
    }

    #[inline(always)]
    fn add_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { vaddq_s16(a, b) }
    }

    #[inline(always)]
    fn sub_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { vsubq_s16(a, b) }
    }

    #[inline(always)]
    fn mul_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { vmulq_s16(a, b) }
    }

    #[inline(always)]
    fn min_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { vminq_s16(a, b) }
    }

    #[inline(always)]
    fn max_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { vmaxq_s16(a, b) }
    }

    #[inline(always)]
    fn bitand_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(a), vreinterpretq_u16_s16(b))) }
    }

    #[inline(always)]
    fn bitor_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { vreinterpretq_s16_u16(vorrq_u16(vreinterpretq_u16_s16(a), vreinterpretq_u16_s16(b))) }
    }

    #[inline(always)]
    fn bitxor_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { vreinterpretq_s16_u16(veorq_u16(vreinterpretq_u16_s16(a), vreinterpretq_u16_s16(b))) }
    }

    #[inline(always)]
    fn bitandnot_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(b), veorq_u16(vreinterpretq_u16_s16(a), vdupq_n_u16(u16::MAX)))) }
    }

    #[inline(always)]
    fn saturating_add_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { vqaddq_s16(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { vqsubq_s16(a, b) }
    }

    #[inline(always)]
    fn avg_round_i16(self, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { vrhaddq_s16(a, b) }
    }

    #[inline(always)]
    fn add_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { vaddq_u32(a, b) }
    }

    #[inline(always)]
    fn sub_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { vsubq_u32(a, b) }
    }

    #[inline(always)]
    fn mul_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { vmulq_u32(a, b) }
    }

    #[inline(always)]
    fn min_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { vminq_u32(a, b) }
    }

    #[inline(always)]
    fn max_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { vmaxq_u32(a, b) }
    }

    #[inline(always)]
    fn bitand_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { vandq_u32(a, b) }
    }

    #[inline(always)]
    fn bitor_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { vorrq_u32(a, b) }
    }

    #[inline(always)]
    fn bitxor_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { veorq_u32(a, b) }
    }

    #[inline(always)]
    fn bitandnot_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { vandq_u32(b, veorq_u32(a, vdupq_n_u32(u32::MAX))) }
    }

    #[inline(always)]
    fn saturating_add_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { vqaddq_u32(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { vqsubq_u32(a, b) }
    }

    #[inline(always)]
    fn avg_round_u32(self, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe {
            let and = vandq_u32(a, b);
            let xor = veorq_u32(a, b);
            let half = vshrq_n_u32::<1>(xor);
            let odd = vandq_u32(xor, vdupq_n_u32(1));
            vaddq_u32(and, vaddq_u32(half, odd))
        }
    }

    #[inline(always)]
    fn add_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { vaddq_s32(a, b) }
    }

    #[inline(always)]
    fn sub_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { vsubq_s32(a, b) }
    }

    #[inline(always)]
    fn mul_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { vmulq_s32(a, b) }
    }

    #[inline(always)]
    fn min_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { vminq_s32(a, b) }
    }

    #[inline(always)]
    fn max_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { vmaxq_s32(a, b) }
    }

    #[inline(always)]
    fn bitand_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_s32(a), vreinterpretq_u32_s32(b))) }
    }

    #[inline(always)]
    fn bitor_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { vreinterpretq_s32_u32(vorrq_u32(vreinterpretq_u32_s32(a), vreinterpretq_u32_s32(b))) }
    }

    #[inline(always)]
    fn bitxor_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { vreinterpretq_s32_u32(veorq_u32(vreinterpretq_u32_s32(a), vreinterpretq_u32_s32(b))) }
    }

    #[inline(always)]
    fn bitandnot_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_s32(b), veorq_u32(vreinterpretq_u32_s32(a), vdupq_n_u32(u32::MAX)))) }
    }

    #[inline(always)]
    fn saturating_add_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { vqaddq_s32(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { vqsubq_s32(a, b) }
    }

    #[inline(always)]
    fn avg_round_i32(self, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe {
            let and = vandq_s32(a, b);
            let xor = veorq_s32(a, b);
            let half = vshrq_n_s32::<1>(xor);
            let odd = vandq_s32(xor, vdupq_n_s32(1));
            vaddq_s32(and, vaddq_s32(half, odd))
        }
    }

    #[inline(always)]
    fn add_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { vaddq_u64(a, b) }
    }

    #[inline(always)]
    fn sub_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { vsubq_u64(a, b) }
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
        unsafe { vandq_u64(a, b) }
    }

    #[inline(always)]
    fn bitor_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { vorrq_u64(a, b) }
    }

    #[inline(always)]
    fn bitxor_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { veorq_u64(a, b) }
    }

    #[inline(always)]
    fn bitandnot_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { vandq_u64(b, veorq_u64(a, vdupq_n_u64(u64::MAX))) }
    }

    #[inline(always)]
    fn saturating_add_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { vqaddq_u64(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { vqsubq_u64(a, b) }
    }

    #[inline(always)]
    fn avg_round_u64(self, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe {
            let and = vandq_u64(a, b);
            let xor = veorq_u64(a, b);
            let half = vshrq_n_u64::<1>(xor);
            let odd = vandq_u64(xor, vdupq_n_u64(1));
            vaddq_u64(and, vaddq_u64(half, odd))
        }
    }

    #[inline(always)]
    fn add_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { vaddq_s64(a, b) }
    }

    #[inline(always)]
    fn sub_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { vsubq_s64(a, b) }
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
        unsafe { vreinterpretq_s64_u64(vandq_u64(vreinterpretq_u64_s64(a), vreinterpretq_u64_s64(b))) }
    }

    #[inline(always)]
    fn bitor_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { vreinterpretq_s64_u64(vorrq_u64(vreinterpretq_u64_s64(a), vreinterpretq_u64_s64(b))) }
    }

    #[inline(always)]
    fn bitxor_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { vreinterpretq_s64_u64(veorq_u64(vreinterpretq_u64_s64(a), vreinterpretq_u64_s64(b))) }
    }

    #[inline(always)]
    fn bitandnot_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { vreinterpretq_s64_u64(vandq_u64(vreinterpretq_u64_s64(b), veorq_u64(vreinterpretq_u64_s64(a), vdupq_n_u64(u64::MAX)))) }
    }

    #[inline(always)]
    fn saturating_add_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { vqaddq_s64(a, b) }
    }

    #[inline(always)]
    fn saturating_sub_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { vqsubq_s64(a, b) }
    }

    #[inline(always)]
    fn avg_round_i64(self, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe {
            let and = vandq_s64(a, b);
            let xor = veorq_s64(a, b);
            let half = vshrq_n_s64::<1>(xor);
            let odd = vandq_s64(xor, vdupq_n_s64(1));
            vaddq_s64(and, vaddq_s64(half, odd))
        }
    }

    #[inline(always)]
    fn eq_u8(self, a: Self::u8, b: Self::u8) -> Self::m8 {
        unsafe { NeonMaskI8(vceqq_u8(a, b)) }
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
        unsafe { NeonMaskI8(vcgtq_u8(a, b)) }
    }

    #[inline(always)]
    fn ge_u8(self, a: Self::u8, b: Self::u8) -> Self::m8 {
        self.lt_u8(a, b).not()
    }

    #[inline(always)]
    fn eq_i8(self, a: Self::i8, b: Self::i8) -> Self::m8 {
        unsafe { NeonMaskI8(vceqq_s8(a, b)) }
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
        unsafe { NeonMaskI8(vcgtq_s8(a, b)) }
    }

    #[inline(always)]
    fn ge_i8(self, a: Self::i8, b: Self::i8) -> Self::m8 {
        self.lt_i8(a, b).not()
    }

    #[inline(always)]
    fn eq_u16(self, a: Self::u16, b: Self::u16) -> Self::m16 {
        unsafe { NeonMaskI16(vceqq_u16(a, b)) }
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
        unsafe { NeonMaskI16(vcgtq_u16(a, b)) }
    }

    #[inline(always)]
    fn ge_u16(self, a: Self::u16, b: Self::u16) -> Self::m16 {
        self.lt_u16(a, b).not()
    }

    #[inline(always)]
    fn eq_i16(self, a: Self::i16, b: Self::i16) -> Self::m16 {
        unsafe { NeonMaskI16(vceqq_s16(a, b)) }
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
        unsafe { NeonMaskI16(vcgtq_s16(a, b)) }
    }

    #[inline(always)]
    fn ge_i16(self, a: Self::i16, b: Self::i16) -> Self::m16 {
        self.lt_i16(a, b).not()
    }

    #[inline(always)]
    fn eq_u32(self, a: Self::u32, b: Self::u32) -> Self::m32 {
        unsafe { NeonMaskI32(vceqq_u32(a, b)) }
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
        unsafe { NeonMaskI32(vcgtq_u32(a, b)) }
    }

    #[inline(always)]
    fn ge_u32(self, a: Self::u32, b: Self::u32) -> Self::m32 {
        self.lt_u32(a, b).not()
    }

    #[inline(always)]
    fn eq_i32(self, a: Self::i32, b: Self::i32) -> Self::m32 {
        unsafe { NeonMaskI32(vceqq_s32(a, b)) }
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
        unsafe { NeonMaskI32(vcgtq_s32(a, b)) }
    }

    #[inline(always)]
    fn ge_i32(self, a: Self::i32, b: Self::i32) -> Self::m32 {
        self.lt_i32(a, b).not()
    }

    #[inline(always)]
    fn eq_u64(self, a: Self::u64, b: Self::u64) -> Self::m64 {
        unsafe { NeonMaskI64(vceqq_u64(a, b)) }
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
        unsafe {
            let sign = vdupq_n_u64(1u64 << 63);
            NeonMaskI64(vcgtq_s64(vreinterpretq_s64_u64(veorq_u64(a, sign)), vreinterpretq_s64_u64(veorq_u64(b, sign))))
        }
    }

    #[inline(always)]
    fn ge_u64(self, a: Self::u64, b: Self::u64) -> Self::m64 {
        self.lt_u64(a, b).not()
    }

    #[inline(always)]
    fn eq_i64(self, a: Self::i64, b: Self::i64) -> Self::m64 {
        unsafe { NeonMaskI64(vceqq_s64(a, b)) }
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
        unsafe { NeonMaskI64(vcgtq_s64(a, b)) }
    }

    #[inline(always)]
    fn ge_i64(self, a: Self::i64, b: Self::i64) -> Self::m64 {
        self.lt_i64(a, b).not()
    }

    #[inline(always)]
    fn shl_u8(self, a: Self::u8, n: u32) -> Self::u8 {
        unsafe { vshlq_u8(a, vdupq_n_s8((n & 7) as i8)) }
    }

    #[inline(always)]
    fn shr_u8(self, a: Self::u8, n: u32) -> Self::u8 {
        unsafe { vshlq_u8(a, vdupq_n_s8(-((n & 7) as i8))) }
    }

    #[inline(always)]
    fn shl_i8(self, a: Self::i8, n: u32) -> Self::i8 {
        unsafe { vshlq_s8(a, vdupq_n_s8((n & 7) as i8)) }
    }

    #[inline(always)]
    fn shr_i8(self, a: Self::i8, n: u32) -> Self::i8 {
        unsafe { vshlq_s8(a, vdupq_n_s8(-((n & 7) as i8))) }
    }

    #[inline(always)]
    fn shl_u16(self, a: Self::u16, n: u32) -> Self::u16 {
        unsafe { vshlq_u16(a, vdupq_n_s16((n & 15) as i16)) }
    }

    #[inline(always)]
    fn shr_u16(self, a: Self::u16, n: u32) -> Self::u16 {
        unsafe { vshlq_u16(a, vdupq_n_s16(-((n & 15) as i16))) }
    }

    #[inline(always)]
    fn shl_i16(self, a: Self::i16, n: u32) -> Self::i16 {
        unsafe { vshlq_s16(a, vdupq_n_s16((n & 15) as i16)) }
    }

    #[inline(always)]
    fn shr_i16(self, a: Self::i16, n: u32) -> Self::i16 {
        unsafe { vshlq_s16(a, vdupq_n_s16(-((n & 15) as i16))) }
    }

    #[inline(always)]
    fn shl_u32(self, a: Self::u32, n: u32) -> Self::u32 {
        unsafe { vshlq_u32(a, vdupq_n_s32((n & 31) as i32)) }
    }

    #[inline(always)]
    fn shr_u32(self, a: Self::u32, n: u32) -> Self::u32 {
        unsafe { vshlq_u32(a, vdupq_n_s32(-((n & 31) as i32))) }
    }

    #[inline(always)]
    fn shl_i32(self, a: Self::i32, n: u32) -> Self::i32 {
        unsafe { vshlq_s32(a, vdupq_n_s32((n & 31) as i32)) }
    }

    #[inline(always)]
    fn shr_i32(self, a: Self::i32, n: u32) -> Self::i32 {
        unsafe { vshlq_s32(a, vdupq_n_s32(-((n & 31) as i32))) }
    }

    #[inline(always)]
    fn shl_u64(self, a: Self::u64, n: u32) -> Self::u64 {
        unsafe { vshlq_u64(a, vdupq_n_s64((n & 63) as i64)) }
    }

    #[inline(always)]
    fn shr_u64(self, a: Self::u64, n: u32) -> Self::u64 {
        unsafe { vshlq_u64(a, vdupq_n_s64(-((n & 63) as i64))) }
    }

    #[inline(always)]
    fn shl_i64(self, a: Self::i64, n: u32) -> Self::i64 {
        unsafe { vshlq_s64(a, vdupq_n_s64((n & 63) as i64)) }
    }

    #[inline(always)]
    fn shr_i64(self, a: Self::i64, n: u32) -> Self::i64 {
        unsafe { vshlq_s64(a, vdupq_n_s64(-((n & 63) as i64))) }
    }

    #[inline(always)]
    fn select_u8(self, mask: Self::m8, a: Self::u8, b: Self::u8) -> Self::u8 {
        unsafe { vbslq_u8(mask.0, a, b) }
    }

    #[inline(always)]
    fn select_i8(self, mask: Self::m8, a: Self::i8, b: Self::i8) -> Self::i8 {
        unsafe { vreinterpretq_s8_u8(vbslq_u8(mask.0, vreinterpretq_u8_s8(a), vreinterpretq_u8_s8(b))) }
    }

    #[inline(always)]
    fn select_u16(self, mask: Self::m16, a: Self::u16, b: Self::u16) -> Self::u16 {
        unsafe { vbslq_u16(mask.0, a, b) }
    }

    #[inline(always)]
    fn select_i16(self, mask: Self::m16, a: Self::i16, b: Self::i16) -> Self::i16 {
        unsafe { vreinterpretq_s16_u16(vbslq_u16(mask.0, vreinterpretq_u16_s16(a), vreinterpretq_u16_s16(b))) }
    }

    #[inline(always)]
    fn select_u32(self, mask: Self::m32, a: Self::u32, b: Self::u32) -> Self::u32 {
        unsafe { vbslq_u32(mask.0, a, b) }
    }

    #[inline(always)]
    fn select_i32(self, mask: Self::m32, a: Self::i32, b: Self::i32) -> Self::i32 {
        unsafe { vreinterpretq_s32_u32(vbslq_u32(mask.0, vreinterpretq_u32_s32(a), vreinterpretq_u32_s32(b))) }
    }

    #[inline(always)]
    fn select_u64(self, mask: Self::m64, a: Self::u64, b: Self::u64) -> Self::u64 {
        unsafe { vbslq_u64(mask.0, a, b) }
    }

    #[inline(always)]
    fn select_i64(self, mask: Self::m64, a: Self::i64, b: Self::i64) -> Self::i64 {
        unsafe { vreinterpretq_s64_u64(vbslq_u64(mask.0, vreinterpretq_u64_s64(a), vreinterpretq_u64_s64(b))) }
    }

    #[inline(always)]
    fn widen_low_u8_to_u16(self, x: Self::u8) -> Self::u16 {
        unsafe { vmovl_u8(vget_low_u8(x)) }
    }

    #[inline(always)]
    fn widen_high_u8_to_u16(self, x: Self::u8) -> Self::u16 {
        unsafe { vmovl_high_u8(x) }
    }

    #[inline(always)]
    fn widen_low_i8_to_i16(self, x: Self::i8) -> Self::i16 {
        unsafe { vmovl_s8(vget_low_s8(x)) }
    }

    #[inline(always)]
    fn widen_high_i8_to_i16(self, x: Self::i8) -> Self::i16 {
        unsafe { vmovl_high_s8(x) }
    }

    #[inline(always)]
    fn widen_low_u16_to_u32(self, x: Self::u16) -> Self::u32 {
        unsafe { vmovl_u16(vget_low_u16(x)) }
    }

    #[inline(always)]
    fn widen_high_u16_to_u32(self, x: Self::u16) -> Self::u32 {
        unsafe { vmovl_high_u16(x) }
    }

    #[inline(always)]
    fn widen_low_i16_to_i32(self, x: Self::i16) -> Self::i32 {
        unsafe { vmovl_s16(vget_low_s16(x)) }
    }

    #[inline(always)]
    fn widen_high_i16_to_i32(self, x: Self::i16) -> Self::i32 {
        unsafe { vmovl_high_s16(x) }
    }

    #[inline(always)]
    fn widen_low_u32_to_u64(self, x: Self::u32) -> Self::u64 {
        unsafe { vmovl_u32(vget_low_u32(x)) }
    }

    #[inline(always)]
    fn widen_high_u32_to_u64(self, x: Self::u32) -> Self::u64 {
        unsafe { vmovl_high_u32(x) }
    }

    #[inline(always)]
    fn widen_low_i32_to_i64(self, x: Self::i32) -> Self::i64 {
        unsafe { vmovl_s32(vget_low_s32(x)) }
    }

    #[inline(always)]
    fn widen_high_i32_to_i64(self, x: Self::i32) -> Self::i64 {
        unsafe { vmovl_high_s32(x) }
    }

    #[inline(always)]
    fn narrow_saturating_u16_to_u8(self, lo: Self::u16, hi: Self::u16) -> Self::u8 {
        unsafe { vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi)) }
    }

    #[inline(always)]
    fn narrow_saturating_i16_to_i8(self, lo: Self::i16, hi: Self::i16) -> Self::i8 {
        unsafe { vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi)) }
    }

    #[inline(always)]
    fn narrow_saturating_u32_to_u16(self, lo: Self::u32, hi: Self::u32) -> Self::u16 {
        unsafe { vcombine_u16(vqmovn_u32(lo), vqmovn_u32(hi)) }
    }

    #[inline(always)]
    fn narrow_saturating_i32_to_i16(self, lo: Self::i32, hi: Self::i32) -> Self::i16 {
        unsafe { vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)) }
    }

    #[inline(always)]
    fn narrow_saturating_u64_to_u32(self, lo: Self::u64, hi: Self::u64) -> Self::u32 {
        unsafe { vcombine_u32(vqmovn_u64(lo), vqmovn_u64(hi)) }
    }

    #[inline(always)]
    fn narrow_saturating_i64_to_i32(self, lo: Self::i64, hi: Self::i64) -> Self::i32 {
        unsafe { vcombine_s32(vqmovn_s64(lo), vqmovn_s64(hi)) }
    }
}

impl Simd for Neon {}
