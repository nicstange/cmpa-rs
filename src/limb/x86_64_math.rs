#![cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]

use core::arch::asm;
use super::{LimbType, DoubleLimb};

#[cfg(feature = "zeroize")]
use zeroize::Zeroize;

pub fn ct_is_nonzero_l(v: LimbType) -> LimbType {
    let result: LimbType;
    unsafe {
        asm!("xor {result:r}, {result:r};\
              test {v:r}, {v:r};\
              setnz {result:l};\
              ",
             v = in(reg) v,
             result = out(reg) result
        );
    }
    result
}

pub fn ct_is_zero_l(v: LimbType) -> LimbType {
    let result: LimbType;
    unsafe {
        asm!("xor {result:r}, {result:r};\
              test {v:r}, {v:r};\
              setz {result:l};\
              ",
             v = in(reg) v,
             result = out(reg) result
        );
    }
    result
}

pub fn ct_add_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    let result: LimbType;
    let carry: LimbType;
    unsafe {
        asm!("xor {carry:r}, {carry:r};\
              add {v0:r}, {v1:r};\
              setc {carry:l};\
              ",
             v0 = inout(reg) v0 => result,
             v1 = in(reg) v1,
             carry = out(reg) carry,
             options(pure, nomem, nostack),
        );
    }
    (carry, result)
}

pub fn ct_sub_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    let result: LimbType;
    let borrow: LimbType;
    unsafe {
        asm!("xor {borrow:r}, {borrow:r};\
              sub {v0:r}, {v1:r};\
              setc {borrow:l};\
              ",
             v0 = inout(reg) v0 => result,
             v1 = in(reg) v1,
             borrow = out(reg) borrow,
             options(pure, nomem, nostack),
        );
    }
    (borrow, result)
}

pub fn ct_mul_l_l(v0: LimbType, v1: LimbType) -> DoubleLimb {
    let mut l: LimbType = v0;
    let mut h: LimbType = 0;
    unsafe {
        asm!("mul {v1:r};",
             inout("ax") l,
             out("dx") h,
             v1 = in(reg) v1,
             options(pure, nomem, nostack),
        );
    }
    DoubleLimb::new(h, l)
}

#[cfg_attr(feature = "zeroize", derive(Zeroize))]
pub struct CtDivDlLNormalizedDivisor {
    v: LimbType,
}

impl CtDivDlLNormalizedDivisor {
    pub fn new(v: LimbType) -> Self {
        Self { v }
    }
}

pub fn ct_div_dl_l(u: &DoubleLimb, v: &CtDivDlLNormalizedDivisor) -> (DoubleLimb, LimbType) {
    let q_high;
    let q_low;
    let r;

    unsafe {
        asm!(
            "xor rdx, rdx\n\
             div {v:r};\n\
             xchg {u_low_in_q_high_out}, rax;\n\
             div {v};\n\
             ",
            u_low_in_q_high_out = inout(reg) u.low() => q_high,
            v = in(reg) v.v,
            inout("ax") u.high() => q_low,
            out("dx") r
        );
    }

    (DoubleLimb::new(q_high, q_low), r)
}
