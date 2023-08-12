#![cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]

use super::{DoubleLimb, LimbType};
use core::arch::asm;

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
    let l: LimbType;
    let h: LimbType;
    unsafe {
        asm!("mul {v1:r};",
             inout("ax") v0 => l,
             out("dx") h,
             v1 = in(reg) v1,
             options(pure, nomem, nostack),
        );
    }
    DoubleLimb::new(h, l)
}

pub struct NonCtLDivisor {
    v: LimbType,
}

impl NonCtLDivisor {
    pub fn new(v: LimbType) -> Result<Self, super::NonCtLDivisorError> {
        if v == 0 {
            return Err(super::NonCtLDivisorError::DivisorIsZero);
        }
        Ok(Self { v })
    }
}

impl super::LDivisorPrivate for NonCtLDivisor {
    fn do_div(&self, u: &DoubleLimb) -> (LimbType, LimbType) {
        let q: LimbType;
        let r: LimbType;
        unsafe {
            asm!(
                "div {v}\n",
                inout("ax") u.low() => q,
                inout("dx") u.high() => r,
                v = in(reg) self.v,
                options(pure, nomem, nostack)
            );
            (q, r)
        }
    }

    fn get_v(&self) -> LimbType {
        self.v
    }
}
