// SPDX-License-Identifier: Apache-2.0
// Copyright 2023 SUSE LLC
// Author: Nicolai Stange <nstange@suse.de>

//! Basic arithmetic primitive implementations optimized for the x86_64
//! architecture.
//!
//! Provide optimized implementation alternatives for a couple of basic
//! arithmetic primitives taking advantage of certain x86_64 instruction set
//! features.
//!
//! Invocation of the x86_64 instruction set requires the use of inline
//! `asm!()`, which is `unsafe {}`. So all functionality is gated by the
//! `enable_arch_math_asm` Cargo feature and must be opted in for explicitly, if
//! desired. However, none of the inline `asm!()` snippets accesses memory, all
//! are fairly trivial and straight forward to review and there's only a very
//! few of those to begin with. That being said, they provide a huge performance
//! gain over the generic implementations and thus, it is highly recommended to
//! enable the `enable_arch_math_asm` Cargo feature.

#![cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]

use super::{DoubleLimb, LimbType};
use core::arch::asm;

/// x86_64 specific implementation alternative of
/// [`ct_is_nonzero_l()`](super::ct_is_nonzero_l()).
///
/// This alternative implementation takes advantage of the x86_64 `test`
/// instruction, which can directly examine a value for zeroness.
pub fn ct_is_nonzero_l(v: LimbType) -> LimbType {
    let result: LimbType;
    unsafe {
        asm!("xor {result:r}, {result:r};\
              test {v:r}, {v:r};\
              setnz {result:l};\
              ",
             v = in(reg) v,
             result = out(reg) result,
             options(pure, nomem, nostack)
        );
    }
    result
}

/// x86_64 specific implementation alternative of
/// [`ct_is_zero_l()`](super::ct_is_zero_l()).
///
/// This alternative implementation takes advantage of the x86_64 `test`
/// instruction, which can directly examine a value for zeroness.
pub fn ct_is_zero_l(v: LimbType) -> LimbType {
    let result: LimbType;
    unsafe {
        asm!("xor {result:r}, {result:r};\
              test {v:r}, {v:r};\
              setz {result:l};\
              ",
             v = in(reg) v,
             result = out(reg) result,
             options(pure, nomem, nostack)
        );
    }
    result
}

/// x86_64 specific implementation alternative of
/// [`ct_add_l_l()`](super::ct_add_l_l()).
///
/// This alternative implementation takes advantage of the fact that x86_64's
/// `add` instructions sets the carry flag, which can be read out directly.
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

/// x86_64 specific implementation alternative of
/// [`ct_sub_l_l()`](super::ct_sub_l_l()).
///
/// This alternative implementation takes advantage of the fact that x86_64's
/// `sub` instructions sets the carry flag on borrow, which can be read out
/// directly.
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

/// x86_64 specific implementation alternative of
/// [`ct_mul_l_l()`](super::ct_mul_l_l()).
///
/// This alternative implementation takes advantage of the fact that x86_64's
/// `mul` instruction computes the full double word multiplication result.
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

/// x86_64 specific implementation alternative of
/// [`NonCtLDivisor`](super::NonCtLDivisor).
///
/// This alternative implementation takes advantage of the fact that x86_64's
/// `div` instruction
/// - makes both, the quotient and remainder available and
/// - supports divison of a double [`LimbType`] by a [`LimbType`], as long as
///   the resulting quotient fits a [`LimbType`] again.
pub struct NonCtLDivisor {
    v: LimbType,
}

impl NonCtLDivisor {
    /// **Non**-constant-time instantiation of a [`NonCtLDivisor`] from a
    /// [`LimbType`] divisor value.
    ///
    /// Not intended to be used directly, but only through the
    /// [`NonCtLDivisor`](super::NonCtLDivisor) configuration abstraction,
    /// please refer to [`NonCtLDivisor::new()`](super::NonCtLDivisor::new).
    ///
    /// # Arguments:
    ///
    /// * `v` - The divisor value.
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
