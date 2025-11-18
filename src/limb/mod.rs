// SPDX-License-Identifier: Apache-2.0
// Copyright 2023 SUSE LLC
// Author: Nicolai Stange <nstange@suse.de>

//! Definitions and arithmetic primitives related to [LimbType], the basic unit
//! of multiprecision integer arithmetic.
use core::arch::asm;
use core::convert;
use core::mem;
use core::ops;

/// The basic unit used by the multiprecision integer arithmetic implementation.
///
/// The [`LimbType`] is mostly an internal implementation detail and users must
/// make no assumption about it representation, the only guarantee is that
/// it's a type alias to one of Rust's native unsigned integer types.
///
/// # Notes on constant-time
///
/// The following arithmetic on a [`LimbType`] is assumed to be constant-time:
/// - Binary operations: `not`, `or`, `and`, `xor`.
/// - Wrapping addition and subtraction of two [`LimbType`] words.
/// - Multiplication of two [`LimbType`] words where the result also fits a
///   [`LimbType`].
///
/// Wider [`LimbType`] type defines would improve multiprecision arithmetic
/// performance, but it must be made sure that all of the operations from above
/// map to (constant-time) CPU instructions and are not implemented by e.g. some
/// architecture support runtime library. In the generic case, the
/// (supposedly) smallest common denominator of `u32` is chosen, but it might be
/// overriden for specific architectures.
pub type LimbType = CfgLimbType;

#[cfg(not(target_arch = "x86_64"))]
#[doc(hidden)]
type CfgLimbType = u32;
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
type CfgLimbType = u64;

/// The bit width of a [`LimbType`].
pub const LIMB_BITS: u32 = LimbType::BITS;
/// The size of a [`LimbType`] in bytes.
pub const LIMB_BYTES: usize = mem::size_of::<LimbType>();

/// The bit width of half a [`LimbType`], i.e. a "halfword".
const HALF_LIMB_BITS: u32 = LIMB_BITS / 2;
/// The size of a half a [`LimbType`], i.e. a "halfword", in bytes.
const HALF_LIMB_MASK: LimbType = ct_lsb_mask_l(HALF_LIMB_BITS);

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
mod x86_64_math;

/// Prevent the compiler from making any assumptions on a [`LimbType`]'s value.
///
/// Optimizing compilers are quite smart in deducing a value's possible set of
/// values at certain points in the code and subsequently take advantage of that
/// knowledge to apply optimizing code transformations. The problem with that is
/// that the resulting code may look surprisingly different from the original
/// and may even contain conditional branches, even though the original
/// input had been authored as "branchless". This would invalidate any
/// assumptions crucial for constant-time execution, of course.
/// `black_box_l()` is an identity function on a [`LimbType`] that forces
/// the compiler to forget any knowledge about the value it has accumulated at
/// the point of invocation.
///
///
/// # Relation to Rust's built-in [`core::hint::black_box()`]
///
/// First, the documentation about Rust's [`core::hint::black_box()`] is fairly
/// conservative about the guarantees it provides: it only works on a
/// "best-effort" basis, making it unsuitable for any "cryptographic or security
/// purposes". This alone would render it useless for the purposes here,
/// but lacking any more robust way to prevent the compiler from applying
/// certain optimizations, "best-effort" is the best that can hoped for anyway.
///
/// That being said, given the limited guarantees Rust's
/// [`core::hint::black_box()`] provides, it's relatively expensive:
/// - As of writing, it moves the value through the stack, probably because it
///   is generic on the blackboxed type and must support arbitrary type sizes.
/// - It prevents optimizing unused values away entirely. Aiming primarily at
///   benchmarking usecases, this is excatly what's expected from it, but
///   inhibiting this kind of dead-code elimination is not needed at all for
///   maintaining constant-time guarantees.
///
/// On the other hand, the Rust Reference about inline assembly requires that
/// (quote)
/// > The compiler cannot assume that the instructions in the asm are the ones
/// > that will actually end up executed.
/// > - This effectively means that the compiler must treat the `asm!` as a
/// > black box and only take the interface specification into account, not the
/// > instructions themselves.
/// > - Runtime code patching is allowed, via target-specific mechanisms.
/// This should be sufficient to implement an almost zero-cost, robust black box
/// for [`LimbType`] values fitting the architecture's register width. The only
/// potential extra cost is an additional register allocation for the blackboxed
/// value:
/// - If the blackboxed value is a constant, it might have get emitted as an
///   immediate value operand to some CPU instruction otherwise.
/// - If the CPU architecture supports memory operands for some instructions,
///   blackboxed values which could have been used as such one now need to get
///   loaded into a register first.
#[inline(always)]
pub fn black_box_l(v: LimbType) -> LimbType {
    let result: LimbType;
    unsafe {
        asm!("/* {v} */", v = inout(reg) v => result, options(pure, nomem, nostack));
    }
    result
}

/// Condition value representation supporting branchless programming patterns.
///
/// This is very similar -- and in fact heavily inspired by -- the `subtle`
/// crates's `subtle::Choice`, but less generic in that it applies to
/// [`LimbType`] values only: A [`LimbChoice`] is constructed only from a
/// [`LimbType`] condition value equal to either `0` or `1` and can subsequently
/// only be used to conditionally select between two [`LimbType`] value choices.
///
/// Compare this to `subtle::Choice`, which can be constructed only from `u8`
/// and select between any type of values implementing the
/// `subtle::ConditionallySelectable` trait.
///
/// By limiting the [`LimbChoice`] to the case of [`LimbType`] relevant here
/// only, certain usage code sites can be written in a slightly less verbose
/// manner and also enables tuning of the implementation to this specific type.
#[derive(Clone, Copy, Debug)]
pub struct LimbChoice {
    mask: LimbType,
}

impl LimbChoice {
    /// Wrap a condition [`LimbType`] integer value in a [`LimbChoice`].
    ///
    /// # Arguments:
    ///
    /// * `cond` - The condition calue to represent, must be `0` or `1`.
    pub const fn new(cond: LimbType) -> Self {
        debug_assert!(cond == 0 || cond == 1);
        Self {
            mask: (0 as LimbType).wrapping_sub(cond),
        }
    }

    /// Unwrap the [`LimbChoice`]'s associated condition value.
    ///
    /// Returns the condition value of the [`LimbChoice`] instance
    /// as a [`LimbType`] value, either `0` or `1`.
    pub fn unwrap(&self) -> LimbType {
        black_box_l(self.mask & 1)
    }

    /// Branchless selection between two [`LimbType`] values based on condition.
    ///
    /// Selects between one of two given [`LimbType`] value alternatives, `v0`
    /// and `v1`, based on the condition value represented by the
    /// [`LimbChoice`] instance and returns it.
    ///
    /// # Arguments:
    ///
    /// * `v0` - The value to be returned if the [`LimbChoice`] represents a
    ///   `false` condition.
    /// * `v1` - The value to be returned if the [`LimbChoice`] represents a
    ///   `true` condition.
    pub const fn select(&self, v0: LimbType, v1: LimbType) -> LimbType {
        v0 ^ (self.mask & (v0 ^ v1))
    }

    /// Branchless selection between two `usize` values based on condition.
    ///
    /// Selects between one of two given `usize` value alternatives, `v0`
    /// and `v1`, based on the condition value represented by the
    /// [`LimbChoice`] instance and returns it.
    ///
    /// # Arguments:
    ///
    /// * `v0` - The value to be returned if the [`LimbChoice`] represents a
    ///   `false` condition.
    /// * `v1` - The value to be returned if the [`LimbChoice`] represents a
    ///   `true` condition.
    pub fn select_usize(&self, v0: usize, v1: usize) -> usize {
        let cond = self.unwrap() as usize;
        let mask = (0_usize).wrapping_sub(cond);
        v0 ^ (mask & (v0 ^ v1))
    }
}

/// Convert from a [`LimbType`] condition value representation to
/// [`LimbChoice`].
///
/// The [`LimbType`] condition value begin converted must be equal to either of
/// `0` and `1`.
impl convert::From<LimbType> for LimbChoice {
    fn from(value: LimbType) -> Self {
        Self::new(value)
    }
}

/// Logical negation of a condition represented by a [`LimbChoice`].
impl ops::Not for LimbChoice {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self { mask: !self.mask }
    }
}

/// Logical `&&` of two conditions represented by a [`LimbChoice`] each.
impl ops::BitAnd for LimbChoice {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            mask: self.mask & rhs.mask,
        }
    }
}

/// Logical `&&` with assignment of two conditions represented by a
/// [`LimbChoice`] each.
impl ops::BitAndAssign for LimbChoice {
    fn bitand_assign(&mut self, rhs: Self) {
        self.mask &= rhs.mask
    }
}

/// Logical `||` of two conditions represented by a [`LimbChoice`] each.
impl ops::BitOr for LimbChoice {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            mask: self.mask | rhs.mask,
        }
    }
}

/// Logical `||` with assignment of two conditions represented by a
/// [`LimbChoice`] each.
impl ops::BitOrAssign for LimbChoice {
    fn bitor_assign(&mut self, rhs: Self) {
        self.mask |= rhs.mask
    }
}

/// [`zeroize::Zeroize`] implementations so that external types containing it
/// can easily be made to implement this trait as well.
#[cfg(feature = "zeroize")]
impl zeroize::Zeroize for LimbChoice {
    fn zeroize(&mut self) {
        self.mask.zeroize()
    }
}

/// Portable implementation of [`ct_is_nonzero_l()`].
///
/// Generic, CPU architecture independent implementation of
/// [`ct_is_nonzero_l()`]. Tuned alternative implementations might be provided
/// for specific architectures, if supported and enabled.
#[allow(unused)]
fn generic_ct_is_nonzero_l(v: LimbType) -> LimbType {
    // This trick is from subtle::*::ct_eq():
    // if v is non-zero, then v or -v or both have the high bit set.
    black_box_l((v | v.wrapping_neg()) >> (LIMB_BITS - 1))
}

#[cfg(not(all(feature = "enable_arch_math_asm", target_arch = "x86_64")))]
#[doc(hidden)]
use self::generic_ct_is_nonzero_l as arch_ct_is_nonzero_l;

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
#[doc(hidden)]
use x86_64_math::ct_is_nonzero_l as arch_ct_is_nonzero_l;

/// Constant-time test if a [`LimbType`] value is non-zero.
///
/// Test whether the given `v` is non-zero and return either a `0` or a `1`
/// accordingly. Note that the returned [`LimbType`] value can get readily
/// converted into a [`LimbChoice`] representation for further use.
///
/// Runs in constant time, independent of the argument value.
///
/// # Arguments:
///
/// * `v` - The value to test.
pub fn ct_is_nonzero_l(v: LimbType) -> LimbType {
    arch_ct_is_nonzero_l(v)
}

/// Portable implementation of [`ct_is_zero_l()`].
///
/// Generic, CPU architecture independent implementation of [`ct_is_zero_l()`].
/// Tuned alternative implementations might be provided for specific
/// architectures, if supported and enabled.
#[allow(unused)]
fn generic_ct_is_zero_l(v: LimbType) -> LimbType {
    (1 as LimbType) ^ ct_is_nonzero_l(v)
}

#[cfg(not(all(feature = "enable_arch_math_asm", target_arch = "x86_64")))]
#[doc(hidden)]
use self::generic_ct_is_zero_l as arch_ct_is_zero_l;

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
#[doc(hidden)]
use x86_64_math::ct_is_zero_l as arch_ct_is_zero_l;

/// Constant-time test if a [`LimbType`] value is zero.
///
/// Test whether the given `v` is zero and return either a `0` or a `1`
/// accordingly. Note that the returned [`LimbType`] value can get readily
/// converted into a [`LimbChoice`] representation for further use.
///
/// Runs in constant time, independent of the argument value.
///
/// # Arguments:
///
/// * `v` - The value to test.
pub fn ct_is_zero_l(v: LimbType) -> LimbType {
    arch_ct_is_zero_l(v)
}

/// Constant-time comparison of two [`LimbType`] values for `==`.
///
/// Compare `v0` and `v1` for `==` and return a [`LimbChoice`] representing the
/// outcome.
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `v0` - The first value to compare.
/// * `v1` - The second value to compare.
pub fn ct_eq_l_l(v0: LimbType, v1: LimbType) -> LimbChoice {
    LimbChoice::from(ct_is_zero_l(v0 ^ v1))
}

/// Constant-time comparison of two [`LimbType`] values for `!=`.
///
/// Compare `v0` and `v1` for `!=` and return a [`LimbChoice`] representing the
/// outcome.
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `v0` - The first value to compare.
/// * `v1` - The second value to compare.
pub fn ct_neq_l_l(v0: LimbType, v1: LimbType) -> LimbChoice {
    !ct_eq_l_l(v0, v1)
}

/// Simultaneously compare two [`LimbType`] values for `<` and `==` at once in
/// constant time.
///
/// Compare `v0` and `v1` for both, `<` and `==` at once, but return the
/// respective results separately as a pair of [`LimbType`]s:
/// - The returned pair's first value will be `1` if `v0` < `v1`, zero
///   otherwise.
/// - The returned pair's second value will be `1` if `v0` == `v1`, zero
///   otherwise.
/// Note that both of the pair's values can get readily converted into a
/// [`LimbChoice`] each.  This "fused" primitive is primarily intended for to
/// support the implementation of constant-time multiprecision integer
/// comparisons and helps to avoid running two independent comparisons for `<`
/// and `==` separately each.
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `v0` - The first value to compare.
/// * `v1` - The second value to compare.
pub fn ct_lt_or_eq_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    let (borrow, diff) = ct_sub_l_l(v0, v1);
    debug_assert!(diff != 0 || borrow == 0);
    (borrow, ct_is_zero_l(diff))
}

/// Constant-time comparison of two [`LimbType`] values for `<`.
///
/// Compare `v0` and `v1` for `<` and return a [`LimbChoice`] representing the
/// outcome.
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `v0` - The first value to compare.
/// * `v1` - The second value to compare.
pub fn ct_lt_l_l(v0: LimbType, v1: LimbType) -> LimbChoice {
    let (borrow, _) = ct_sub_l_l(v0, v1);
    LimbChoice::from(borrow)
}

/// Constant-time comparison of two [`LimbType`] values for `<=`.
///
/// Compare `v0` and `v1` for `<=` and return a [`LimbChoice`] representing the
/// outcome.
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `v0` - The first value to compare.
/// * `v1` - The second value to compare.
pub fn ct_leq_l_l(v0: LimbType, v1: LimbType) -> LimbChoice {
    !ct_lt_l_l(v1, v0)
}

/// Constant-time comparison of two [`LimbType`] values for `>`.
///
/// Compare `v0` and `v1` for `>` and return a [`LimbChoice`] representing the
/// outcome.
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `v0` - The first value to compare.
/// * `v1` - The second value to compare.
pub fn ct_gt_l_l(v0: LimbType, v1: LimbType) -> LimbChoice {
    ct_lt_l_l(v1, v0)
}

/// Constant-time comparison of two [`LimbType`] values for `>=`.
///
/// Compare `v0` and `v1` for `>=` and return a [`LimbChoice`] representing the
/// outcome.
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `v0` - The first value to compare.
/// * `v1` - The second value to compare.
pub fn ct_geq_l_l(v0: LimbType, v1: LimbType) -> LimbChoice {
    ct_leq_l_l(v1, v0)
}

/// Construct a [`LimbType`] with specified number of least significant bits set
/// in constant time.
///
/// Construct a [`LimbType`] value with the `nbits` lower bits set and the rest
/// clear.
///
/// Runs in constant time, independent of the argument value.
///
/// # Arguments:
///
/// * `nbits` - The number of least significant bits to set in the result. Must
///   be in the range from `0` to the [`LimbType`] width (inclusive).
pub const fn ct_lsb_mask_l(nbits: u32) -> LimbType {
    debug_assert!(nbits <= LIMB_BITS);
    // The standard way for generating a mask with nbits of the lower bits set is
    // (1 << nbits) - 1. However, for nbits == LIMB_BITS, the right shift would be
    // undefined behaviour. Split nbits into the nbits_lo < LIMB_BITS and
    // nbits_hi == (nbits == LIMB_BITS) components and generate masks for each
    // individually.
    let nbits_lo = nbits % LIMB_BITS;
    let nbits_hi = nbits / LIMB_BITS;
    debug_assert!(nbits_hi <= 1);
    debug_assert!(nbits_hi == 0 || nbits_lo == 0);

    let mask_for_lo = (1 << nbits_lo) - 1;
    let mask_for_hi = (0 as LimbType).wrapping_sub(nbits_hi as LimbType);
    mask_for_lo | mask_for_hi
}

#[test]
fn test_ct_lsb_mask_l() {
    for i in 0..LIMB_BITS {
        let mask = ct_lsb_mask_l(i);
        assert_eq!(mask, (1 << i) - 1);
    }
    assert_eq!(ct_lsb_mask_l(LIMB_BITS), !0);
}

/// Split a limb into upper and lower half limbs.
///
/// Returns a pair of upper and lower half limbs, in this order.
///
/// Runs in constant time.
///
/// # Arguments
///
/// * `v` - the limb to split.
fn ct_l_to_hls(v: LimbType) -> (LimbType, LimbType) {
    (
        black_box_l(v >> HALF_LIMB_BITS),
        black_box_l(v & HALF_LIMB_MASK),
    )
}

/// Portable implementation of [`ct_add_l_l()`].
///
/// Generic, CPU architecture independent implementation of [`ct_add_l_l()`].
/// Tuned alternative implementations might be provided for specific
/// architectures, if supported and enabled.
#[allow(unused)]
fn generic_ct_add_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    // Don't rely on overflowing_add() for determining the carry -- that would
    // almost certainly branch and not be constant-time.
    let v0 = black_box_l(v0);
    let v1 = black_box_l(v1);
    let r = v0.wrapping_add(v1);
    let carry = black_box_l((((v0 | v1) & !r) | (v0 & v1)) >> (LIMB_BITS - 1));
    (carry, r)
}

#[cfg(not(all(feature = "enable_arch_math_asm", target_arch = "x86_64")))]
#[doc(hidden)]
use self::generic_ct_add_l_l as arch_ct_add_l_l;

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
#[doc(hidden)]
use x86_64_math::ct_add_l_l as arch_ct_add_l_l;

/// Constant-time addition of two [`LimbType`] values.
///
/// Computes the sum of `v0 + v1` and returns it as a a pair of carry, either
/// `0` or `1`, and the resulting [`LimbType`] value, in this order.
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `v0` - The first operand.
/// * `v1` - The second operand
pub fn ct_add_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    arch_ct_add_l_l(v0, v1)
}

#[test]
fn test_ct_add_l_l() {
    assert_eq!(ct_add_l_l(0, 0), (0, 0));
    assert_eq!(ct_add_l_l(1, 0), (0, 1));
    assert_eq!(ct_add_l_l(!0 - 1, 1), (0, !0));
    assert_eq!(ct_add_l_l(!0, 1), (1, 0));
    assert_eq!(
        ct_add_l_l(1 << (LIMB_BITS - 1), 1 << (LIMB_BITS - 1)),
        (1, 0)
    );
    assert_eq!(
        ct_add_l_l(!0, 1 << (LIMB_BITS - 1)),
        (1, ct_lsb_mask_l(LIMB_BITS - 1))
    );
    assert_eq!(ct_add_l_l(!0, !0), (1, !0 - 1));
}

/// Constant-time addition of two [`LimbType`] values with carry propagation.
///
/// Computes the sum of `v0 + v1 + carry` and returns it as a a pair of carry,
/// either `0` or `1`, and the resulting [`LimbType`] value, in this order.
/// The intended use is addition with carry propagation in a multiprecision
/// integer.
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `v0` - The first operand.
/// * `v1` - The second operand
/// * `carry` - The carry from a preceeding limb addition at the next lower
///   position in a multiprecision integer. Must be equal to either of `0` and
///   `1`.
pub fn ct_add_l_l_c(v0: LimbType, v1: LimbType, carry: LimbType) -> (LimbType, LimbType) {
    debug_assert!(carry <= 1);
    let (carry0, r) = ct_add_l_l(v0, carry);
    let (carry1, r) = ct_add_l_l(r, v1);
    let carry = carry0 + carry1;
    debug_assert!(carry <= 1);
    (carry, r)
}

/// Portable implementation of [`ct_sub_l_l()`].
///
/// Generic, CPU architecture independent implementation of [`ct_sub_l_l()`].
/// Tuned alternative implementations might be provided for specific
/// architectures, if supported and enabled.
#[allow(unused)]
fn generic_ct_sub_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    // Don't rely on overflowing_sub() for determining the borrow -- that would
    // almost certainly branch and not be constant-time.
    let v0 = black_box_l(v0);
    let v1 = black_box_l(v1);
    let r = v0.wrapping_sub(v1);
    let borrow = black_box_l((((r | v1) & !v0) | (v1 & r)) >> (LIMB_BITS - 1));
    (borrow, r)
}

#[cfg(not(all(feature = "enable_arch_math_asm", target_arch = "x86_64")))]
#[doc(hidden)]
use self::generic_ct_sub_l_l as arch_ct_sub_l_l;

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
#[doc(hidden)]
use x86_64_math::ct_sub_l_l as arch_ct_sub_l_l;

/// Constant-time subtraction of two [`LimbType`] values.
///
/// Computes the difference of `v0 - v1` and returns it as a a pair of borrow,
/// either `0` or `1`, and the resulting [`LimbType`] value, in this order.
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `v0` - The first, "minuend" operand.
/// * `v1` - The second, "subtrahend" operand
pub fn ct_sub_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    arch_ct_sub_l_l(v0, v1)
}

#[test]
fn test_ct_sub_l_l() {
    assert_eq!(ct_sub_l_l(0, 0), (0, 0));
    assert_eq!(ct_sub_l_l(1, 0), (0, 1));
    assert_eq!(ct_sub_l_l(0, 1), (1, !0));
    assert_eq!(
        ct_sub_l_l(1 << (LIMB_BITS - 1), 1 << (LIMB_BITS - 1)),
        (0, 0)
    );
    assert_eq!(
        ct_sub_l_l(0, 1 << (LIMB_BITS - 1)),
        (1, 1 << (LIMB_BITS - 1))
    );
    assert_eq!(
        ct_sub_l_l(1 << (LIMB_BITS - 1), (1 << (LIMB_BITS - 1)) + 1),
        (1, !0)
    );
}

/// Constant-time subtaction of two [`LimbType`] values with borrow propagation.
///
/// Computes the difference of `v0 - v1 - borrow` and returns it as a a pair of
/// borrow, either `0` or `1`, and the resulting [`LimbType`] value, in this
/// order.  The intended use is subtraction with borrow propagation in a
/// multiprecision integer.
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `v0` - The first, "minuend" operand.
/// * `v1` - The second, "subtrahend" operand
/// * `borrow` - The carry from a preceeding limb subtraction at the next lower
///   position in a multiprecision integer. Must be equal to either of `0` and
///   `1`.
pub fn ct_sub_l_l_b(v0: LimbType, v1: LimbType, borrow: LimbType) -> (LimbType, LimbType) {
    debug_assert!(borrow <= 1);
    let (borrow0, r) = ct_sub_l_l(v0, borrow);
    let (borrow1, r) = ct_sub_l_l(r, v1);
    let borrow = borrow0 + borrow1;
    debug_assert!(borrow <= 1);
    (borrow, r)
}

/// A pair of [`LimbType`] values interpreted as a double precision integer.
///
/// This is being used internally in the context of [`LimbType`] multiplications
/// and double precision divisions, both of which are needed for the respective
/// multiprecision implementations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DoubleLimb {
    /// Double precision integer in "native endian" layout: the less significant
    /// [`LimbType`] is at index zero.
    v: [LimbType; 2],
}

impl DoubleLimb {
    /// Construct a [`DoubleLimb`] integer from the high and low [`LimbType`]
    /// halves.
    ///
    /// # Arguments:
    ///
    /// * `h` - The most sigificant [`LimbType`] half of the resulting
    ///   [`DoubleLimb`].
    /// * `l` - The least sigificant [`LimbType`] half of the resulting
    ///   [`DoubleLimb`].
    pub fn new(h: LimbType, l: LimbType) -> Self {
        Self { v: [l, h] }
    }

    /// Read a [`DoubleLimb`]'s most sigificant [`LimbType`] half.
    pub fn high(&self) -> LimbType {
        self.v[1]
    }

    /// Read a [`DoubleLimb`]'s least sigificant [`LimbType`] half.
    pub fn low(&self) -> LimbType {
        self.v[0]
    }
}

/// Portable implementation of [`ct_mul_l_l()`].
///
/// Generic, CPU architecture independent implementation of [`ct_mul_l_l()`].
/// Tuned alternative implementations might be provided for specific
/// architectures, if supported and enabled.
#[allow(unused)]
fn generic_ct_mul_l_l(v0: LimbType, v1: LimbType) -> DoubleLimb {
    let (v0h, v0l) = ct_l_to_hls(v0);
    let (v1h, v1l) = ct_l_to_hls(v1);

    let prod_v0l_v1l = v0l * v1l;
    let prod_v0l_v1h = v0l * v1h;
    let prod_v0h_v1l = v0h * v1l;
    let prod_v0h_v1h = v0h * v1h;

    let mut result_low: LimbType = prod_v0l_v1l;
    let mut result_high: LimbType = prod_v0h_v1h;

    let (prod_v0l_v1h_h, prod_v0l_v1h_l) = ct_l_to_hls(prod_v0l_v1h);
    let (prod_v0h_v1l_h, prod_v0h_v1l_l) = ct_l_to_hls(prod_v0h_v1l);

    let (result_low_carry, result_low_sum) =
        ct_add_l_l(result_low, prod_v0l_v1h_l << HALF_LIMB_BITS);
    result_low = result_low_sum;
    result_high += result_low_carry;
    result_high += prod_v0l_v1h_h;

    let (result_low_carry, result_low_sum) =
        ct_add_l_l(result_low, prod_v0h_v1l_l << HALF_LIMB_BITS);
    result_low = result_low_sum;
    result_high += result_low_carry;
    result_high += prod_v0h_v1l_h;

    DoubleLimb::new(result_high, result_low)
}

#[cfg(not(all(feature = "enable_arch_math_asm", target_arch = "x86_64")))]
#[doc(hidden)]
use self::generic_ct_mul_l_l as arch_ct_mul_l_l;

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
#[doc(hidden)]
use x86_64_math::ct_mul_l_l as arch_ct_mul_l_l;

/// Constant-time multiplication of two [`LimbType`] values.
///
/// Computes the product of `v0 * v1` and returns the resulting
/// double precision integer as a [`DoubleLimb`].
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `v0` - The first operand.
/// * `v1` - The second operand
pub fn ct_mul_l_l(v0: LimbType, v1: LimbType) -> DoubleLimb {
    arch_ct_mul_l_l(v0, v1)
}

#[test]
fn test_ct_mul_l_l() {
    let p = ct_mul_l_l(0, 0);
    assert_eq!(p.low(), 0);
    assert_eq!(p.high(), 0);

    let p = ct_mul_l_l(2, 2);
    assert_eq!(p.low(), 4);
    assert_eq!(p.high(), 0);

    let p = ct_mul_l_l(1 << (LIMB_BITS - 1), 2);
    assert_eq!(p.low(), 0);
    assert_eq!(p.high(), 1);

    let p = ct_mul_l_l(2, 1 << (LIMB_BITS - 1));
    assert_eq!(p.low(), 0);
    assert_eq!(p.high(), 1);

    let p = ct_mul_l_l(1 << (LIMB_BITS - 1), 1 << (LIMB_BITS - 1));
    assert_eq!(p.low(), 0);
    assert_eq!(p.high(), 1 << (LIMB_BITS - 2));

    let p = ct_mul_l_l(!0, !0);
    assert_eq!(p.low(), 1);
    assert_eq!(p.high(), !1);
}

/// Constant-time multiply-add for [`LimbType`] operands with carry propagation.
///
/// Computes `op0 + op01 * op10 + carry` and returns the double precision
/// integer as a pair of [`LimbType`] values, with the most significant, i.e.
/// the new carry value, first. Note that unlike it is the case for simple
/// additions, the carries can not be just either `0` or `1`, but any possible
/// value in the [`LimbType`] range.
///
/// The intended use is the implementation of multiplication related
/// multiprecision integer arithmetic, where the final result is typically
/// accumulated (`op0`) from several [`LimbType`]-by-multiprecision-integer
/// (`op01` for the [`LimbType`] operand, `op10` and the `carry` for iterating
/// over the multiprecision integer operand) multiplications.
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `op0` - The value to accumulate to, it will get added to the result.
/// * `op01` - The first operand to the multiplication.
/// * `op10` - The second operand to the multiplication.
/// * `carry` - The carry from a preceeding limb multiplication, i.e. from
///   another [`ct_mul_add_l_l_l_c()`] invocation, at the next lower position in
///   a multiprecision integer. It will be added to the final result and can be
///   any possible value in the [`LimbType`] range.
pub fn ct_mul_add_l_l_l_c(
    op0: LimbType,
    op10: LimbType,
    op11: LimbType,
    carry: LimbType,
) -> (LimbType, LimbType) {
    let prod = ct_mul_l_l(op10, op11);
    // Basic property of the multiplication.
    debug_assert!(prod.high() < !1 || prod.high() == !1 && prod.low() == 1);
    let (carry0, result) = ct_add_l_l(op0, carry);
    let (carry1, result) = ct_add_l_l(result, prod.low());
    // The new carry does not overflow: if carry0 != 0,
    // then the result after after the first addition is
    // <= !1, because that addition did wrap around.
    // If in addition prod.high() == !1, then prod.low() <= 1
    // and the second addition would not overflow.
    debug_assert!(prod.high() < !1 || carry0 + carry1 <= 1);
    let carry = prod.high() + carry0 + carry1;
    (carry, result)
}

/// Constant-time multiply-subtract for [`LimbType`] operands with borrow
/// propagation.
///
/// Computes `op0 - op01 * op10 - carry` and returns the double precision
/// integer as a pair of [`LimbType`] values, with the most significant, i.e.
/// the new borrow value, first. Note that unlike it is the case for simple
/// subtractions, the borrows can not be just either `0` or `1`, but any
/// possible value in the [`LimbType`] range.
///
/// The intended use is the implementation of multiprecision division, where
/// the multiprecision integer divisor needs to get scaled by a [`LimbType`]
/// value and subtracted from the intermediate remainder upon over-estimation of
/// the quotient.
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `op0` - The value to subtract from.
/// * `op01` - The first operand to the multiplication.
/// * `op10` - The second operand to the multiplication.
/// * `borrow` - The borrow from a preceeding limb multiplication, i.e. from
///   another [`ct_mul_sub_l_l_l_b()`] invocation, at the next lower position in
///   a multiprecision integer. It will be subtracted from the final result and
///   can be any possible value in the [`LimbType`] range.
pub fn ct_mul_sub_l_l_l_b(
    op0: LimbType,
    op10: LimbType,
    op11: LimbType,
    borrow: LimbType,
) -> (LimbType, LimbType) {
    let prod = ct_mul_l_l(op10, op11);
    // Basic property of the multiplication.
    debug_assert!(prod.high() < !1 || prod.high() == !1 && prod.low() == 1);
    let (borrow0, result) = ct_sub_l_l(op0, borrow);
    let (borrow1, result) = ct_sub_l_l(result, prod.low());
    // The new borrow does not overflow: if borrow0 != 0,
    // then the result after after the first addition is
    // >= 1, because that subtraction did wrap around.
    // If in addition prod.high() == !1, then prod.low() <= 1
    // and the second addition would not overflow.
    debug_assert!(prod.high() < !1 || borrow0 + borrow1 <= 1);
    let borrow = prod.high() + borrow0 + borrow1;
    (borrow, result)
}

/// Constant-time [`LimbType`] divisor.
///
/// [`CtLDivisor`] provides a constant-time implementation of the basic
/// [`DoubleLimb`] by [`LimbType`] division primitive relied upon by the higher
/// level division related algorithms such
/// as [`ct_div_mp_mp()`](super::div_impl::ct_div_mp_mp),
/// [`ct_div_mp_l()`](super::div_impl::ct_div_mp_l) and many more.
///
/// Intended to serve as a basic primitive, only cases where the quotient is
/// known a priori to fit a [`LimbType`] are supported. To prevent accidental
/// misuse by external users, the actual division functionality it exposed only
/// through a private trait, [`LDivisorPrivate`]. That is, external users are
/// only supposed to instantiate [`CtLDivisor`], but not to use it directly.
///
/// In order to not having to invoke the CPU's division instructions,
/// which are _not_ constant-time in general, it follows the approach from
/// > [GRAN_MONT94](https://doi.org/10.1145/773473.178249)
/// > "Division by Invariant Integers using Multiplication",
/// > Torbjörn Granlund, Peter L. Montgomery,
/// > ACM SIGPLAN Notices,  Volume 29, Issue 6, June 1994, pp 61–72, section 8.
///
/// This method does involve a certain amount of divisor value dependent
/// one-time computations at instantiation, namely the division of a certain
/// double limb value by the divisor to determine the associated
/// runtime-constant multiplier value. There are two constructors provided:
/// - A constant-time constructor, [`CtLDivisor::new()`], which implements said
///   division by bitwise long division taking [`LimbType::BITS`] iterations,
///   and involving only subtractions and some binary logic.
/// - A less expensive, but **non**-constant-time alternative,
///   [`CtLDivisor::nonct_new()`], for cases where the divisor value is not
///   sensitive.
/// Note that for multiprecision divisions, the difference probably won't matter
/// much most of the time, relative to the cost of the multiprecision integer
/// division operation itself, which has a quadratic runtime. However,
/// whenever the divisor value is not sensitive it is certainly favorable
/// to avoid the unnecessary cost and use the [`CtLDivisor::nonct_new()`]
/// constructor variant.
pub struct CtLDivisor {
    /// The multiplier associated with the divisor `v`.
    m: LimbType,
    /// The divisor value.
    v: LimbType,
    /// The number of least significant bits set in `v`.
    v_width: u32,
}

/// Error type returned by [`CtLDivisor::new()`].
#[derive(Debug)]
pub enum CtLDivisorError {
    /// Attempt to instantiate a [`CtLDivisor`] from a zero divisor value.
    DivisorIsZero,
}

impl CtLDivisor {
    /// Constant-time instantiation of a [`CtLDivisor`] from a [`LimbType`]
    /// divisor value.
    ///
    /// Instantiation of a [`CtLDivisor`] has a non-trivial cost, because it
    /// involves some precomputational division, whose constant-time
    /// implementation takes [`LimbType::BITS`] iterations. Consider reusing
    /// instances whenever feasible and also, if the divisor value is
    /// not sensitive, [`CtLDivisor::nonct_new()`] might be a good alternative.
    /// C.f. the rationale at the `struct`-level [`CtLDivisor`] documentation.
    ///
    /// Runs in constant time, independent of the argument value.
    ///
    /// # Arguments:
    ///
    /// * `v` - The divisor value.
    pub fn new(v: LimbType) -> Result<Self, CtLDivisorError> {
        if ct_is_zero_l(v) != 0 {
            return Err(CtLDivisorError::DivisorIsZero);
        }
        debug_assert_ne!(v, 0);
        let v_width = ct_find_last_set_bit_l(v) as u32;
        let m = Self::ct_compute_m(v, v_width);
        Ok(Self { m, v, v_width })
    }

    /// Compute the multiplier associated with a given divisor in constant time.
    ///
    /// Compute
    /// *(2<sup>[`LIMB_BITS`] + `v_width`</sup> - 1) / `v` -
    /// 2<sup>[`LIMB_BITS`]</sup>*
    ///
    /// Runs in constant time, independent of the argument value.
    ///
    /// # Arguments:
    ///
    /// * `v` - The divisor value.
    /// * `v_width` - The number of least significant bits set in `v`.
    fn ct_compute_m(v: LimbType, v_width: u32) -> LimbType {
        debug_assert!(v_width > 0);
        // Align the nominator and denominator all the way to the left. Then run a
        // bit-wise long division for the sake of constant-time.  The quotient's
        // high bit at position 2^LIMB_BITS is known to come out as one, but is
        // to be subtracted at the end anyway. So just shift it out in the last
        // iteration.
        let v = v << (LIMB_BITS - v_width);
        let mut r_msb = LimbChoice::from(0);
        let mut r_h = !0;
        let mut r_l = !ct_lsb_mask_l(LIMB_BITS - v_width);
        let mut q = 0;
        for _ in 0..LIMB_BITS + 1 {
            let q_new_lsb = r_msb | ct_geq_l_l(r_h, v);
            q = (q << 1) | q_new_lsb.unwrap();
            r_h = r_h.wrapping_sub(q_new_lsb.select(0, v));
            r_msb = LimbChoice::from(r_h >> (LIMB_BITS - 1));
            r_h = (r_h << 1) | (r_l >> (LIMB_BITS - 1));
            r_l <<= 1;
        }
        q
    }

    /// **Non**-constant-time instantiation of a [`CtLDivisor`] from a
    /// [`LimbType`] divisor value.
    ///
    /// Does **not** run in constant time, it invokes the CPU's division
    /// instruction. Must **not** be used if the divisor value `v` is sensitive.
    /// Refer to [`CtLDivisor::new()`] in this case.
    ///
    /// # Arguments
    ///
    /// * `v` - The divisor value.
    pub fn nonct_new(v: LimbType) -> Result<Self, CtLDivisorError> {
        if v == 0 {
            return Err(CtLDivisorError::DivisorIsZero);
        }
        let v_width = ct_find_last_set_bit_l(v) as u32;
        let m = Self::nonct_compute_m(v, v_width);
        Ok(Self { m, v, v_width })
    }

    /// Compute the multiplier for a given divisor in **non**-constant time.
    ///
    /// Compute
    /// *(2<sup>[`LIMB_BITS`] + `v_width`</sup> - 1) / `v` -
    ///
    /// Does **not** run in constant time, it invokes the CPU's division
    /// instruction. Must **not** be used if the divisor value `v` is sensitive.
    ///
    /// # Arguments:
    ///
    /// * `v` - The divisor value.
    /// * `v_width` - The number of least significant bits set in `v`.
    fn nonct_compute_m(v: LimbType, v_width: u32) -> LimbType {
        NonCtLDivisor::new(v)
            .unwrap()
            .do_div(&DoubleLimb::new(ct_lsb_mask_l(v_width) - v, !0))
            .0
    }
}

/// Private interface of [`DoubleLimb`] by [`LimbType`] divisor implementations.
///
/// The basic [`DoubleLimb`] by [`LimbType`] division primitive needed by higher
/// level divison related algorithms is only required to support cases where the
/// quotient is known a priori to fit a [`LimbType`]. The respective divisor
/// implementations, [`CtLDivisor`] and [`NonCtLDivisor`], restrict themselves
/// to this special case accordingly in order to save some superfluous work in
/// performance critical code paths.
///
/// In order to prevent accidental misuse by external users, the actual division
/// functionality is begin exposed only through the private [`LDivisorPrivate`]
/// interface.
pub trait LDivisorPrivate {
    /// Divide a [`DoubleLimb`] value by the represented [`LimbType`] divisor.
    ///
    /// The dividend value **must** be limited to a range such that the
    /// resulting quotient will always fit a [`LimbType`]. Otherwise,
    /// **behaviour is undefined**!
    ///
    /// # Arguments:
    ///
    /// * `u` - The dividend value.
    fn do_div(&self, u: &DoubleLimb) -> (LimbType, LimbType);

    /// Accessor to the represented [`LimbType`] divisor value.
    fn get_v(&self) -> LimbType;
}

impl LDivisorPrivate for CtLDivisor {
    /// Constant-time implementation of [`LDivisorPrivate::do_div()`]
    fn do_div(&self, u: &DoubleLimb) -> (LimbType, LimbType) {
        let l = self.v_width;
        debug_assert!(l > 0 && l <= LIMB_BITS);
        // The dividend u is divided in three parts:
        // - u0: The l - 1 least significant bits.
        // - u1: The single bit at position l - 1.
        // - u2: The remaining most significant bits. Because the quotient fits a limb
        //   by assumption, this fits a limb as well. In fact, it is smaller than
        //   LimbType::MAX.
        //
        // Shift in two steps to avoid undefined behaviour.
        let u2 = (u.low() >> (l - 1)) >> 1;
        let u2 = u2 | u.high() << (LIMB_BITS - l);
        // u10 is (u1, u0) shifted all the way to the left limb boundary.
        // That is, it equals u1 * 2^(LIMB_BITS - 1) + u0 * (LIMB_BITS - l).
        let u10 = u.low() << (LIMB_BITS - l);

        let u1 = u10 >> (LIMB_BITS - 1);
        // Normalized divisor, shifted all the way to the left such that the MSB
        // is set.
        let v_norm = self.v << (LIMB_BITS - l);
        debug_assert_eq!(v_norm >> (LIMB_BITS - 1), 1);
        // u1 is equal to the high bit in u10. The high bit in v_norm is always set.
        // So, if u1 is set, adding the two will (virtually) overflow into
        // 2^(LIMB_BITS). Dismissing this carry is equivalent to a subtracting
        // 2^(LIMB_BITS). So, n_adj will equal
        // u10 + u1 * v_norm
        // = u1 * 2^(LIMB_BITS - 1) + u0 * 2^(LIMB_BITS - l) + u1 * v_norm
        // = u1 * (2^(LIMB_BITS - 1) + v_norm) + u0 * ...
        // = u1 * (2^LIMB_BITS + v_norm - 2^(LIMB_BITS - 1)) + u0 * ...
        // and, after dismissing the carry
        // = u1 * (v_norm - 2^(LIMB_BITS - 1)) + u0 * ...
        let n_adj = u10.wrapping_add(LimbChoice::from(u1).select(0, v_norm));

        // u2 + u1 does not overflow. Otherwise the quotient would not fit a limb.
        debug_assert!(u2 < !0 || u1 == 0);
        let p = ct_mul_l_l(self.m, u2 + u1);
        let (carry, _) = ct_add_l_l(p.low(), n_adj);
        let (carry, q1) = ct_add_l_l_c(u2, p.high(), carry);
        debug_assert_eq!(carry, 0);

        // Compute the remainder u - (q1  + 1) * v.
        // u - q1 * v is known to be in the range [0, 2 * v), subtracting
        // an extra v brings it into the range [-v, v).

        // -(q1 + 1) with a (virtual) borrow of 2^LIMB_BITS.
        // The +1 guarantees that negation will indeed use a (virtual) borrow, so it
        // can be uncondiitionally subtracted from the high part below.
        let neg_q_plus_one = ct_negate_l(q1.wrapping_add(1));
        // (2^LIMB_BITS - (q1 + 1)) * v
        let p = ct_mul_l_l(neg_q_plus_one, self.v);
        let (carry, r_l) = ct_add_l_l(u.low(), p.low());
        let (carry, r_h) = ct_add_l_l_c(u.high(), p.high(), carry);
        // Subtract the 2^LIMB_BITS * v borrowed above from the
        // high part.
        let (borrow, r_h) = ct_sub_l_l(r_h, self.v);
        debug_assert!(carry == 0 || borrow != 0);
        let borrow = borrow ^ carry;
        let is_negative = LimbChoice::from(borrow);
        let q = q1 + (!is_negative).select(0, 1);
        let (carry, r_l) = ct_add_l_l(r_l, is_negative.select(0, self.v));
        debug_assert_eq!(r_h.wrapping_add(carry), 0);
        debug_assert!(r_l < self.v);
        (q, r_l)
    }

    fn get_v(&self) -> LimbType {
        self.v
    }
}

#[test]
fn test_ct_l_divisor() {
    fn div_and_check(u: DoubleLimb, v: LimbType) {
        let nonct_v = CtLDivisor::nonct_new(v).unwrap();
        let v = CtLDivisor::new(v).unwrap();
        assert_eq!(v.m, nonct_v.m);
        assert_eq!(v.v_width, nonct_v.v_width);
        assert_eq!(v.v, nonct_v.v);

        let (q, r) = v.do_div(&u);

        // Multiply q by v again and add the remainder back, the result should match the
        // initial u.
        let prod = ct_mul_l_l(q, v.v);

        let (carry, result_l) = ct_add_l_l(prod.low(), r);
        let (carry, result_h) = ct_add_l_l(prod.high(), carry);
        assert_eq!(carry, 0);
        assert_eq!(result_l, u.low());
        assert_eq!(result_h, u.high());
    }

    div_and_check(DoubleLimb::new(!1, !0), !0);

    div_and_check(DoubleLimb::new(!1, !1), !0);

    div_and_check(DoubleLimb::new(0, 0), !0);

    div_and_check(DoubleLimb::new(0, !1), !0);

    div_and_check(DoubleLimb::new(0, !0), 1);

    div_and_check(DoubleLimb::new(1, !0), 2);

    for i in 0..LIMB_BITS {
        for j1 in i..LIMB_BITS {
            for j2 in 0..j1 + 1 {
                let u_h = if i != 0 { !0 >> (LIMB_BITS - i) } else { 0 };
                let u = DoubleLimb::new(u_h, !0);
                let v = 1 << j1 | 1 << j2;
                div_and_check(u, v);
            }
        }
    }
}

/// Error type returned by [`NonCtLDivisor::new()`].
#[derive(Debug)]
pub enum NonCtLDivisorError {
    DivisorIsZero,
}

/// Portable implementation of [`NonCtLDivisor`].
///
/// Generic, CPU architecture independent implementation of [`NonCtLDivisor`].
/// Tuned alternative implementations might be provided for specific
/// architectures, if supported and enabled.
#[allow(unused)]
struct GenericNonCtLDivisor {
    v: LimbType,
    scaling_shift: u32,
    scaling_low_src_rshift: u32,
    scaling_low_src_mask: LimbType,
}

impl GenericNonCtLDivisor {
    /// **Non**-constant-time instantiation of a [`GenericNonCtLDivisor`] from a
    /// [`LimbType`] divisor value.
    ///
    /// Not intended to be used directly, but only through the [`NonCtLDivisor`]
    /// configuration abstraction, please refer to [`NonCtLDivisor::new()`].
    ///
    /// # Arguments:
    ///
    /// * `v` - The divisor value.
    #[allow(unused)]
    fn new(v: LimbType) -> Result<Self, NonCtLDivisorError> {
        if v == 0 {
            return Err(NonCtLDivisorError::DivisorIsZero);
        }

        // Shift distance to normalize v such that its MSB is set.
        let scaling_shift = v.leading_zeros();
        let scaling_low_src_rshift = (LIMB_BITS - scaling_shift) % LIMB_BITS;
        let scaling_low_src_mask = ct_lsb_mask_l(scaling_shift);

        Ok(Self {
            v,
            scaling_shift,
            scaling_low_src_rshift,
            scaling_low_src_mask,
        })
    }
}

impl LDivisorPrivate for GenericNonCtLDivisor {
    fn do_div(&self, u: &DoubleLimb) -> (LimbType, LimbType) {
        // Division algorithm according to D. E. Knuth, "The Art of Computer
        // Programming", vol 2 for the special case of a double limb dividend
        // interpreted as four half limbs.

        let v = self.v << self.scaling_shift;
        let v_hl_h = v >> HALF_LIMB_BITS;
        let v_hl_l = v & HALF_LIMB_MASK;

        // Scale u. The quotient is known to fit a limb, so
        // no non-zero bits will be shifted out.
        let mut u_h = (u.high() << self.scaling_shift)
            | ((u.low() >> self.scaling_low_src_rshift) & self.scaling_low_src_mask);
        debug_assert_eq!(u_h >> self.scaling_shift, u.high());
        let mut u_l = u.low() << self.scaling_shift;

        let mut q: LimbType = 0;
        for _j in [1, 0] {
            // Loop invariant, with n == 2 denoting the number of half limbs in v.
            // - u[n + j] is in the high half limb of u_h
            // - u[n + j - 1] is in the low half limb of u_h
            // - u[n + j - 2] is in the high half limb of u_l
            // - u[n + j - 3], if any,  is in the low half limb of u_l

            // Estimate q.
            let mut cur_q_hl = {
                // q = u[n + j:n + j - 1] / v[n - 1].
                let q = u_h / v_hl_h;

                // Check whether q fits a half limb and cap it otherwise.
                let q = q.min(HALF_LIMB_MASK);
                let r = u_h - q * v_hl_h;

                // As long as r does not overflow b, i.e. half a LimbType, check whether
                // q * v[n - 2] > b * r + u[j + n - 2]. If so, decrement q and adjust r
                // accordingly by adding v[n-1] back.
                // Note that because v[n-1] is normalized to have its MSB set, r would overflow
                // in the second iteration at latest.  The second iteration is
                // not necessary for correctness, but only serves optimization
                // purposes: it would help to avoid the "add-back" step below in
                // the majority of cases. However, for the small v fitting a limb, the add-back
                // step is not really expensive, so the second iteration of the
                // "over-estimated" check here would be quite pointless. Skip
                // it.
                let r_ov = r >> HALF_LIMB_BITS != 0;
                if !r_ov && q * v_hl_l > (r << HALF_LIMB_BITS) | (u_l >> HALF_LIMB_BITS) {
                    q - 1
                } else {
                    q
                }
            };

            // Subtract q_val * v from u[n + j:j] = u[n + j:n + j - 2].
            // To simplify the computations, shift u[] by half a limb to the left first
            // (which is also needed at some point to uphold the loop invariant anyway).
            let u2 = u_h >> HALF_LIMB_BITS;
            let u1 = (u_h << HALF_LIMB_BITS) | u_l >> HALF_LIMB_BITS;
            let u0 = u_l << HALF_LIMB_BITS;
            let (borrow, mut u1) = ct_mul_sub_l_l_l_b(u1, cur_q_hl, v, 0);
            let (borrow, mut u2) = ct_sub_l_l(u2, borrow);
            // If borrow is set, q had been overestimated, decrement it and
            // add one v back to the remainder in this case.
            if borrow != 0 {
                cur_q_hl -= 1;
                let carry;
                (carry, u1) = ct_add_l_l(u1, v);
                (_, u2) = ct_add_l_l(u2, carry);
            }
            debug_assert_eq!(u2, 0);

            u_h = u1;
            u_l = u0;

            // Finally update q.
            q = (q << HALF_LIMB_BITS) | cur_q_hl
        }

        // The scaled remainder is in u_h now. Undo the scaling.
        debug_assert_eq!(u_l, 0);
        (q, u_h >> self.scaling_shift)
    }

    fn get_v(&self) -> LimbType {
        self.v
    }
}

#[cfg(not(all(feature = "enable_arch_math_asm", target_arch = "x86_64")))]
#[doc(hidden)]
use self::GenericNonCtLDivisor as ArchNonCtLDivisor;

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
#[doc(hidden)]
use x86_64_math::NonCtLDivisor as ArchNonCtLDivisor;

/// **Non**-constant-time [`LimbType`] divisor.
///
/// [`NonCtLDivisor`] provides a **non**-constant-time implementation of the
/// basic [`DoubleLimb`] by [`LimbType`] division primitive. The main purpose is
/// to enable more efficient, but non-constant-time construction of
/// [`CtLDivisor`] instances where the divisor value is not sensitive. It must
/// **not** get used if either the divisor value or any of the dividends
/// subsequently to be divided is sensitive.
///
/// Just as it's the case for [`CtLDivisor`] only the very basic [`DoubleLimb`]
/// by [`LimbType`] division primitive, where the quotient is known a priori to
/// fit a [`LimbType`], is supported. To prevent accidental misuse by external
/// users, the actual division functionality it similarly exposed only through a
/// private trait, [`LDivisorPrivate`].
pub struct NonCtLDivisor {
    arch: ArchNonCtLDivisor,
}

impl NonCtLDivisor {
    /// **Non**-constant-time instantiation of a [`NonCtLDivisor`] from a
    /// [`LimbType`] divisor value.
    ///
    /// Does **not** run in constant time, it invokes the CPU's division
    /// instruction. Must **not** be used if the divisor value `v` is
    /// sensitive.
    ///
    /// # Arguments
    ///
    /// * `v` - The divisor value.
    pub fn new(v: LimbType) -> Result<Self, NonCtLDivisorError> {
        Ok(Self {
            arch: ArchNonCtLDivisor::new(v)?,
        })
    }
}

impl LDivisorPrivate for NonCtLDivisor {
    /// **Non**-constant-time implementation of [`LDivisorPrivate::do_div()`]
    fn do_div(&self, u: &DoubleLimb) -> (LimbType, LimbType) {
        self.arch.do_div(u)
    }

    fn get_v(&self) -> LimbType {
        self.arch.get_v()
    }
}

/// **Non**-constant-time division of a [`DoubleLimb`] by a [`LimbType`].
///
/// Double precsision integer division implementation. The result will be
/// returned as a pair of quotient and remainder.
///
/// Does **not** run in constant time, it invokes the CPU's division
/// instruction. Must **not** be used if either of the operands is sensitive.
///
/// # Arguments
///
/// * `u` - The [`DoubleLimb`] dividend.
/// * `v` - The [`NonCtLDivisor`] divisor.
#[cfg(test)]
fn nonct_div_dl_l(u: &DoubleLimb, v: &NonCtLDivisor) -> (DoubleLimb, LimbType) {
    let (q_h, r_h) = v.do_div(&DoubleLimb::new(0, u.high()));
    let (q_l, r_l) = v.do_div(&DoubleLimb::new(r_h, u.low()));
    (DoubleLimb::new(q_h, q_l), r_l)
}

#[test]
fn test_nonct_div_dl_l() {
    fn div_and_check(u: DoubleLimb, v: LimbType) {
        let normalized_v = NonCtLDivisor::new(v).unwrap();

        let (q, r) = nonct_div_dl_l(&u.clone(), &normalized_v);

        // Multiply q by v again and add the remainder back, the result should match the
        // initial u.
        let prod_l = ct_mul_l_l(q.low(), v);
        let prod_h = ct_mul_l_l(q.high(), v);
        assert_eq!(prod_h.high(), 0);

        let (carry, result_l) = ct_add_l_l(prod_l.low(), r);
        let (carry, result_h) = ct_add_l_l_c(prod_l.high(), prod_h.low(), carry);
        assert_eq!(carry, 0);
        assert_eq!(result_l, u.low());
        assert_eq!(result_h, u.high());
    }

    div_and_check(DoubleLimb::new(!1, !0), !0);

    div_and_check(DoubleLimb::new(!1, !1), !0);

    div_and_check(DoubleLimb::new(0, 0), !0);

    div_and_check(DoubleLimb::new(0, !1), !0);

    div_and_check(DoubleLimb::new(!0, !0), 1);

    div_and_check(DoubleLimb::new(!0, !1), 2);

    for i in 0..LIMB_BITS {
        for j1 in 0..LIMB_BITS {
            for j2 in 0..j1 + 1 {
                let u_h = if i != 0 { !0 >> (LIMB_BITS - i) } else { 0 };
                let u = DoubleLimb::new(u_h, !0);
                let v = 1 << j1 | 1 << j2;
                div_and_check(u, v);
            }
        }
    }
}

/// Constant-time inversion of a [`LimbType`] modulo two to the power of
/// [`LimbType::BITS`].
///
/// Use Hensel lifting to compute
/// *`v`<sup>-1</sup> mod 2<sup>[`LimbType::BITS`]</sup>*,
/// i.e. such that
/// *`v` * `v`<sup>-1</sup> mod 2<sup>[`LimbType::BITS`]</sup> = 1*.
///
/// This is intended as a supporting routine for the word-by-word Montgomery
/// reductions. It is a fairly cheap operation, as it's running time is
/// *~log<sub>2</sub>([`LimbType::BITS`])*, that is 5 for a 32 bit [`LimbType`]
/// width, 6 for a 64 bit one.
///
/// Runs in constant time, independent of the argument values.
///
/// # Arguments:
///
/// * `v` - The [`LimbType`] integer value to invert.
pub fn ct_inv_mod_l(v: LimbType) -> LimbType {
    // Apply Hensel's lifting lemma for v * x - 1 to lift the trivial root
    // (i.e. inverse of v) mod 2^1 to a root mod 2^LIMB_BITS. Successive steps
    // double the bits, i.e. if r is a root mod 2^k, one step makes it a root mod
    // 2^2*k.
    debug_assert_eq!(v & 1, 1);
    let mut k = 1;
    let mut r: LimbType = 1;
    while k < LIMB_BITS {
        r = (r << 1).wrapping_sub(v.wrapping_mul(r).wrapping_mul(r));
        k *= 2;
    }

    r
}

#[test]
fn test_ct_inv_mod_l() {
    for j in 0..LIMB_BITS {
        let v = ((1 as LimbType) << j) | 1;
        assert_eq!(v.wrapping_mul(ct_inv_mod_l(v)), 1);
    }

    for j in 1..LIMB_BITS {
        let v = ((1 as LimbType) << j).wrapping_sub(1);
        assert_eq!(v.wrapping_mul(ct_inv_mod_l(v)), 1);
    }

    let v: LimbType = !0;
    assert_eq!(v.wrapping_mul(ct_inv_mod_l(v)), 1);
}

/// Find a [`LimbType`]'s most significant set bit in constant time.
///
/// Return the position past the most significant set bit in `v`, if any,
/// zero otherwise. That is, return the effective width of the value in `v`.
///
/// Runs in constant time, independent of the argument value.
///
/// The running time is
/// *~log<sub>2</sub>([`LimbType::BITS`])*, that is 5 for a 32 bit [`LimbType`]
/// width, 6 for a 64 bit one.
///
/// # Arguments:
///
/// * `v` - The [`LimbType`] integer value whose most significant bit to find.
pub fn ct_find_last_set_bit_l(mut v: LimbType) -> usize {
    let mut bits = LIMB_BITS as LimbType;
    assert!(bits != 0);
    assert!(bits & (bits - 1) == 0); // Is a power of two.
    let mut count: usize = 0;
    let mut lsb_mask = !0;
    while bits > 1 {
        bits /= 2;
        lsb_mask >>= bits;
        let v_l = v & lsb_mask;
        let v_h = v >> bits;
        let upper = ct_neq_l_l(v_h, 0);
        count += upper.select(0, bits) as usize;
        v = upper.select(v_l, v_h);
    }
    debug_assert!(v <= 1);
    count += v as usize;
    count
}

#[test]
fn test_ct_find_last_set_bit_l() {
    assert_eq!(ct_find_last_set_bit_l(0), 0);

    for i in 0..LIMB_BITS as usize {
        let v = 1 << i;
        assert_eq!(ct_find_last_set_bit_l(v), i + 1);
        assert_eq!(ct_find_last_set_bit_l(v - 1), i);
        assert_eq!(ct_find_last_set_bit_l(!(v - 1)), LIMB_BITS as usize);
    }
}

/// Find a [`LimbType`]'s most significant non-zero byte in constant time.
///
/// Return the position past the most significant non-zero byte in `v` in units
/// of bytes, if any, zero otherwise.
///
/// Runs in constant time, independent of the argument value.
///
/// # Arguments:
///
/// * `v` - The [`LimbType`] integer value whose most significant non-zero byte
///   to find.
pub fn ct_find_last_set_byte_l(v: LimbType) -> usize {
    (ct_find_last_set_bit_l(v) + 8 - 1) / 8
}

/// Find a [`LimbType`]'s least significant set bit in constant time.
///
/// Return the position of the least significant set bit in `v`, if any,
/// [`LimbType::BITS`] otherwise.
///
/// Runs in constant time, independent of the argument value.
///
/// The running time is
/// *~log<sub>2</sub>([`LimbType::BITS`])*, that is 5 for a 32 bit [`LimbType`]
/// width, 6 for a 64 bit one.
///
/// # Arguments:
///
/// * `v` - The [`LimbType`] integer value whose leas significant bit to find.
pub fn ct_find_first_set_bit_l(mut v: LimbType) -> usize {
    let mut bits = LIMB_BITS as LimbType;
    assert!(bits != 0);
    assert!(bits & (bits - 1) == 0); // Is a power of two.
    let mut count: usize = LIMB_BITS as usize;
    let mut lsb_mask = !0;
    while bits > 1 {
        bits /= 2;
        lsb_mask >>= bits;
        let v_l = v & lsb_mask;
        let v_h = v >> bits;
        let lower = ct_neq_l_l(v_l, 0);
        count -= lower.select(0, bits) as usize;
        v = lower.select(v_h, v_l);
    }
    debug_assert!(v <= 1);
    count -= v as usize;
    count
}

#[test]
fn test_ct_find_first_set_bit_l() {
    assert_eq!(ct_find_first_set_bit_l(0), LIMB_BITS as usize);

    for i in 0..LIMB_BITS as usize {
        let v = 1 << i;
        assert_eq!(ct_find_first_set_bit_l(v), i);
        if i != 0 {
            assert_eq!(ct_find_first_set_bit_l(v - 1), 0);
        }
        assert_eq!(ct_find_first_set_bit_l(!(v - 1)), i);
    }
}

/// Constant-time arithmetic right shift of a [`LimbType`].
///
/// Right shift `v` by `rshift` bits arithmetically, i.e. fill vacant bit
/// positions with the sign bit (interpreting the unsigned [`LimbType`] as a
/// signed integer in two's complement).
///
/// # Arguments:
///
/// * `v` - The [`LimbType`] value to shift.
/// * `rshift` - The shift distance. Must be between zero and [`LimbType::BITS`]
///   (inclusive).
pub fn ct_arithmetic_rshift_l(v: LimbType, rshift: LimbType) -> LimbType {
    let rshift_nz_mask = LimbChoice::from(ct_is_nonzero_l(rshift)).select(0, !0);
    let sign_extend = (0 as LimbType).wrapping_sub(v >> (LIMB_BITS - 1));
    let sign_extend = sign_extend << ((LIMB_BITS as LimbType - rshift) & rshift_nz_mask);
    let sign_extend = sign_extend & rshift_nz_mask;
    let rshift_lt_max_mask = ct_eq_l_l(rshift, LIMB_BITS as LimbType).select(!0, 0);
    sign_extend | ((v >> (rshift & rshift_lt_max_mask)) & rshift_lt_max_mask)
}

#[test]
fn test_ct_arithmetic_shift_l() {
    assert_eq!(
        ct_arithmetic_rshift_l(ct_lsb_mask_l(LIMB_BITS - 1), 0),
        ct_lsb_mask_l(LIMB_BITS - 1)
    );

    assert_eq!(
        ct_arithmetic_rshift_l(ct_lsb_mask_l(LIMB_BITS - 1), (LIMB_BITS - 2) as LimbType),
        1
    );

    assert_eq!(
        ct_arithmetic_rshift_l(ct_lsb_mask_l(LIMB_BITS - 1), (LIMB_BITS - 1) as LimbType),
        0
    );

    assert_eq!(
        ct_arithmetic_rshift_l(ct_lsb_mask_l(LIMB_BITS - 1), LIMB_BITS as LimbType),
        0
    );

    assert_eq!(ct_arithmetic_rshift_l(!0 ^ 1, 0), !0 ^ 1);

    assert_eq!(
        ct_arithmetic_rshift_l(!0 ^ 1, (LIMB_BITS - 2) as LimbType),
        !0
    );

    assert_eq!(
        ct_arithmetic_rshift_l(!0 ^ 1, (LIMB_BITS - 1) as LimbType),
        !0
    );

    assert_eq!(ct_arithmetic_rshift_l(!0 ^ 1, LIMB_BITS as LimbType), !0);
}

/// Constant-time negate a [`LimbType`] in two's complement representation.
///
///
/// Negate the unsigned [`LimbType`] in two's complement representation, i.e.
/// modulo *2<sup>[`LimbType::BITS`]</sup>*.
///
/// # Arguments:
///
/// * `v` - The [`LimbType`] value to negate.
pub fn ct_negate_l(v: LimbType) -> LimbType {
    (!v).wrapping_add(1)
}
