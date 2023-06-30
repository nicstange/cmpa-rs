//! Definitions and arithmetic primitives related to [LimbType], the basic unit of multiprecision
//! integer arithmetic.
use core;
use core::ops::Deref as _;
use core::mem;
use std::ops::{Deref, DerefMut};
use subtle::{self, ConditionallySelectable as _, ConstantTimeEq as _, ConstantTimeGreater as _};
#[cfg(feature = "zeroize")]
use zeroize::Zeroize;
use super::cond_helpers::{cond_choice_to_mask, cond_select_with_mask};
use super::zeroize::ZeroizeableSubtleChoice;

/// The basic unit used by the multiprecision integer arithmetic implementation.
///
/// # Notes
///
/// The following arithmetic on a [`LimbType`] is assumed to be constant-time:
/// - Binary operations: `not`, `or`, `and`, `xor`.
/// - Wrapping addition and subtraction of two [`LimbType`] words.
/// - Multiplication of two [`LimbType`] words where the result also fits
///   a [`LimbType`].
/// - Division of one [`LimbType`] by another.
///
/// Wider [`LimbType`] type defines would improve multiprecision arithmetic performance, but it must
/// be made sure that all of the operations from above map to (constant-time) CPU instructions and
/// are not implemented by e.g. some architecture support runtime library. For now, the smallest
/// common denominator of `u32` is chosen.
///

#[cfg(not(target_arch = "x86_64"))]
pub type LimbType = u32;
#[cfg(target_arch = "x86_64")]
pub type LimbType = u64;

/// The bit width of a [`LimbType`].
pub const LIMB_BITS: u32 = LimbType::BITS;
/// The size of a [`LimbType`] in bytes.
pub const LIMB_BYTES: usize = mem::size_of::<LimbType>();

/// The bit width of half a [`LimbType`], i.e. a "halfword".
const HALF_LIMB_BITS: u32 = LIMB_BITS / 2;
/// The size of a half a [`LimbType`], i.e. a "halfword", in bytes.
const HALF_LIMB_MASK: LimbType = (1 << HALF_LIMB_BITS) - 1;

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
mod x86_64_math;

pub fn ct_eq_l_l(v0: LimbType, v1: LimbType) -> subtle::Choice {
    v0.ct_eq(&v1)
}

pub fn ct_neq_l_l(v0: LimbType, v1: LimbType) -> subtle::Choice {
    !ct_eq_l_l(v0, v1)
}

pub fn ct_lt_l_l(v0: LimbType, v1: LimbType) -> subtle::Choice {
    v1.ct_gt(&v0)
}

pub fn ct_le_l_l(v0: LimbType, v1: LimbType) -> subtle::Choice {
    !v0.ct_gt(&v1)
}

pub fn ct_gt_l_l(v0: LimbType, v1: LimbType) -> subtle::Choice {
    ct_lt_l_l(v1, v0)
}

pub fn ct_ge_l_l(v0: LimbType, v1: LimbType) -> subtle::Choice {
    ct_le_l_l(v1, v0)
}

pub fn ct_l_to_subtle_choice(v: LimbType) -> subtle::Choice {
    debug_assert!(v <= 1);
    subtle::Choice::from(core::hint::black_box(v) as u8)
}

/// Split a limb into upper and lower half limbs.
///
/// Returns a pair of upper and lower half limb, in this order.
///
/// Runs in constant time.
///
/// # Arguments
///
/// * `v` - the limb to split.
///
fn ct_l_to_hls(v: LimbType) -> (LimbType, LimbType) {
    (core::hint::black_box(v >> HALF_LIMB_BITS), core::hint::black_box(v & HALF_LIMB_MASK))
}

/// Construct a limb from upper and lower half limbs in constant time.
///
/// Runs in constant time.
///
/// # Arguments
///
/// * `vh` - the upper half limb.
/// * `vl` - the upper half limb.
///
fn ct_hls_to_l(vh: LimbType, vl: LimbType) -> LimbType {
    vh << HALF_LIMB_BITS | vl
}

/// Add two limbs.
///
/// Returns a pair of carry and the [`LimbType::BITS`] lower bits of the sum.
///
/// Runs in constant time.
///
/// # Arguments:
///
/// * `v0` - first operand
/// * `v1` - second operand
///
pub fn ct_add_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    // Don't rely on overflowing_add() for determining the carry -- that would almost certainly
    // branch and not be constant-time.
    let v0 = core::hint::black_box(v0);
    let v1 = core::hint::black_box(v1);
    let r = v0.wrapping_add(v1);
    let carry = (((v0 | v1) & !r) | (v0 & v1)) >> core::hint::black_box(LIMB_BITS - 1);
    (carry, r)
}

#[test]
fn test_ct_add_l_l() {
    assert_eq!(ct_add_l_l(0, 0), (0, 0));
    assert_eq!(ct_add_l_l(1, 0), (0, 1));
    assert_eq!(ct_add_l_l(!0 - 1, 1), (0, !0));
    assert_eq!(ct_add_l_l(!0, 1), (1, 0));
    assert_eq!(ct_add_l_l(1 << (LIMB_BITS - 1), 1 << (LIMB_BITS - 1)), (1, 0));
    assert_eq!(ct_add_l_l(!0, 1 << (LIMB_BITS - 1)), (1, (1 << (LIMB_BITS - 1)) - 1));
    assert_eq!(ct_add_l_l(!0, !0), (1, !0 - 1));
}

pub fn ct_add_l_l_c(v0: LimbType, v1: LimbType, carry: LimbType) -> (LimbType, LimbType) {
    debug_assert!(carry <= 1);
    let (carry0, r) = ct_add_l_l(v0, carry);
    let (carry1, r) = ct_add_l_l(r, v1);
    let carry = carry0 + carry1;
    debug_assert!(carry <= 1);
    (carry, r)
}

/// Subtract two limbs.
///
/// Returns a pair of borrow and the [`LimbType::BITS`] lower bits of the difference.
///
/// Runs in constant time.
///
/// # Arguments:
///
/// * `v0` - first operand
/// * `v1` - second operand
///
pub fn ct_sub_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    // Don't rely on overflowing_sub() for determining the borrow -- that would almost certainly
    // branch and not be constant-time.
    let v0 = core::hint::black_box(v0);
    let v1 = core::hint::black_box(v1);
    let r = v0.wrapping_sub(v1);
    let borrow = (((r | v1) & !v0) | (v1 & r)) >> core::hint::black_box(LIMB_BITS - 1);
    (borrow, r)
}

#[test]
fn test_ct_sub_l_l() {
    assert_eq!(ct_sub_l_l(0, 0), (0, 0));
    assert_eq!(ct_sub_l_l(1, 0), (0, 1));
    assert_eq!(ct_sub_l_l(0, 1), (1, !0));
    assert_eq!(ct_sub_l_l(1 << (LIMB_BITS - 1), 1 << (LIMB_BITS - 1)), (0, 0));
    assert_eq!(ct_sub_l_l(0, 1 << (LIMB_BITS - 1)), (1, 1 << (LIMB_BITS - 1)));
    assert_eq!(ct_sub_l_l(1 << (LIMB_BITS - 1), (1 << (LIMB_BITS - 1)) + 1), (1, !0));
}

pub fn ct_sub_l_l_b(v0: LimbType, v1: LimbType, borrow: LimbType) -> (LimbType, LimbType) {
    debug_assert!(borrow <= 1);
    let (borrow0, r) = ct_sub_l_l(v0, borrow);
    let (borrow1, r) = ct_sub_l_l(r, v1);
    let borrow = borrow0 + borrow1;
    debug_assert!(borrow <= 1);
    (borrow, r)
}

/// A pair of [`LimbType`]s interpreted as a double precision integer.
///
/// Used internally for the result of [`LimbType`] multiplications and also for the implementation
/// of multiprecision integer division.
///
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "zeroize", derive(Zeroize))]
pub struct DoubleLimb {
    v: [LimbType; 2],
}

impl DoubleLimb {
    pub fn new(h: LimbType, l: LimbType) -> Self {
        Self { v: [l, h] }
    }

    pub fn high(&self) -> LimbType {
        self.v[1]
    }

    pub fn low(&self) -> LimbType {
        self.v[0]
    }

    /// Extract a half limb from a double limb.
    ///
    /// Interpret the [`DoubleLimb`] as an integer composed of four half limbs and extract the one
    /// at at a given position.
    ///
    /// # Arguments
    ///
    /// * `i` - index of the half limb to extract, the half limbs are ordered by increasing
    ///         significance.
    fn get_half_limb(&self, i: usize) -> LimbType {
        if i & 1 != 0 {
            core::hint::black_box(self.v[i / 2] >> HALF_LIMB_BITS)
        } else {
            core::hint::black_box(self.v[i / 2] & HALF_LIMB_MASK)
        }
    }
}

impl subtle::ConditionallySelectable for DoubleLimb {
    fn conditional_select(a: &Self, b: &Self, choice: subtle::Choice) -> Self {
        let choice_mask = cond_choice_to_mask(choice);
        Self::new(cond_select_with_mask(a.high(), b.high(), choice_mask),
                  cond_select_with_mask(a.low(), b.low(), choice_mask))
    }
}

/// Mutiply two limbs in constant time.
///
/// Returns the result a double precision [`DoubleLimb`].
///
/// Runs in constant time.
///
/// # Arguments:
///
/// * `v0` - first operand
/// * `v1` - second operand
///
#[allow(unused)]
pub fn generic_ct_mul_l_l(v0: LimbType, v1: LimbType) -> DoubleLimb {
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

    let (result_low_carry, result_low_sum) = ct_add_l_l(result_low, prod_v0l_v1h_l << HALF_LIMB_BITS);
    result_low = result_low_sum;
    result_high += result_low_carry;
    result_high += prod_v0l_v1h_h;

    let (result_low_carry, result_low_sum) = ct_add_l_l(result_low, prod_v0h_v1l_l << HALF_LIMB_BITS);
    result_low = result_low_sum;
    result_high += result_low_carry;
    result_high += prod_v0h_v1l_h;

    DoubleLimb::new(result_high, result_low)
}

#[cfg(not(all(feature = "enable_arch_math_asm", target_arch = "x86_64")))]
pub use self::generic_ct_mul_l_l as ct_mul_l_l;

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
pub use x86_64_math::ct_mul_l_l as ct_mul_l_l;

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

pub fn ct_mul_add_l_l_l_c(op0: LimbType, op10: LimbType, op11: LimbType, carry: LimbType) -> (LimbType, LimbType) {
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

pub fn ct_mul_sub_b(op0: LimbType, op10: LimbType, op11: LimbType, borrow: LimbType) -> (LimbType, LimbType) {
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

/// A normalized divisor for input to [`generic_ct_div_dl_l()`].
///
/// In order to avoid scaling the same divisor all over again in loop invoking
/// [`generic_ct_div_dl_l()`], the latter takes an already scaled divisor as input. This allows for
/// reusing the result of the scaling operation.
///
#[cfg_attr(feature = "zeroize", derive(Zeroize))]
#[allow(unused)]
pub struct GenericCtDivDlLNormalizedDivisor {
    scaling: LimbType,
    normalized_v: LimbType,
    shifted_v: ZeroizeableSubtleChoice,
}

impl GenericCtDivDlLNormalizedDivisor {
    /// Normalize a divisor for subsequent use with [`ct_div_dl_l()`].
    ///
    /// Runs in constant time.
    ///
    /// # Arguments
    ///
    /// * `v` - the unscaled, but non-zero divisor to normalize.
    ///
    pub fn new(v: LimbType) -> Self {
        debug_assert!(v != 0);
        // If the upper half is zero, shift v by half limb. This is accounted for by shifting u
        // accordingly in mp_ct_div() and allows for constant-time division computation, independent
        // of the value of v.
        let v_h = v >> HALF_LIMB_BITS;
        let shifted_v = ZeroizeableSubtleChoice(ct_eq_l_l(v_h, 0));
        let v = LimbType::conditional_select(&v, &(v << HALF_LIMB_BITS), *shifted_v.deref());
        let v_h = core::hint::black_box(v >> HALF_LIMB_BITS);
        let scaling = core::hint::black_box(1 << HALF_LIMB_BITS) / (v_h + 1);
        let normalized_v = scaling * v;
        Self { scaling, normalized_v, shifted_v }
    }
}

/// Divide a double limb by a limb.
///
/// This routine is intended as a supporting primitive for the multiprecision integer division
/// implementation. The result will be returned as a pair of quotient and remainder.
///
/// Runs in constant time.
///
/// # Arguments
///
/// * `u` - The [`DoubleLimb`] dividend.
/// * `v` - The [`GenericCtDivDlLNormalizedDivisor`] divisor.
///
/// # Note
///
/// This software implementation of double limb division relies only on single precision
/// [`LimbType`] division operations mapping to constant-time CPU instructions and will be
/// _much_ slower than the native double word division instructions available on some
/// architectures like e.g. the `div` instruction on x86.
///
#[allow(unused)]
pub fn generic_ct_div_dl_l(u: &DoubleLimb, v: &GenericCtDivDlLNormalizedDivisor) -> (DoubleLimb, LimbType) {
    // Division algorithm according to D. E. Knuth, "The Art of Computer Programming", vol 2 for the
    // special case of a double limb dividend interpreted as four half limbs.
    //
    // The local u and temporary q's are stored as an array of LimbType words, in little endian order,
    // interpreted as a multiprecision integer of half limbs. Define some helpers for
    // accessiong the individual half limbs, indexed from least to most significant as usual.
    fn le_limbs_half_limb_index(i: usize) -> (usize, u32) {
        let half_limb_shift = if i % 2 != 0 {
            HALF_LIMB_BITS
        } else {
            0
        };
        (i / 2, half_limb_shift)
    }

    fn _le_limbs_load_half_limb(limbs: &[LimbType], (limb_index, half_limb_shift): (usize, u32)) -> LimbType {
        let half_limb_shift = core::hint::black_box(half_limb_shift);
        core::hint::black_box((limbs[limb_index] >> half_limb_shift) & HALF_LIMB_MASK)
    }

    fn le_limbs_load_half_limb(limbs: &[LimbType], half_limb_index: usize) -> LimbType {
        _le_limbs_load_half_limb(limbs, le_limbs_half_limb_index(half_limb_index))
    }

    fn _le_limbs_store_half_limb(limbs: &mut [LimbType], (limb_index, half_limb_shift): (usize, u32), value: LimbType) {
        let value = core::hint::black_box(value);
        let half_limb_shift = core::hint::black_box(half_limb_shift);
        limbs[limb_index] &= !(HALF_LIMB_MASK << half_limb_shift);
        limbs[limb_index] |= value << half_limb_shift;
    }

    fn le_limbs_store_half_limb(limbs: &mut [LimbType], half_limb_index: usize, value: LimbType) {
        _le_limbs_store_half_limb(limbs, le_limbs_half_limb_index(half_limb_index), value)
    }

    // u as a sequence of half words in little endian order. Allocate one extra half word
    // to accomodate for the scaling and another one to shift in case v had been shifted.
    let mut u: [LimbType; 3] = [u.low(), u.high(), 0];
    // Conditionally shift u by one half limb in order to align with the shifting of the normalized
    // v, if any.
    let shifted_v_mask = cond_choice_to_mask(*v.shifted_v);
    for i in [2, 1] {
        let shifted_u_val = u[i] << HALF_LIMB_BITS | u[i - 1] >> HALF_LIMB_BITS;
        u[i] = cond_select_with_mask(u[i], shifted_u_val, shifted_v_mask);
    }
    u[0] = cond_select_with_mask(u[0], u[0] << HALF_LIMB_BITS, shifted_v_mask);
    // Scale u by the divisor normalization scaling.
    let mut carry = 0;
    for i in 0..6 {
        let (limb_index, half_limb_shift) = le_limbs_half_limb_index(i);
        let u_val = _le_limbs_load_half_limb(u.as_slice(), (limb_index, half_limb_shift));
        let u_val = v.scaling * u_val;
        let u_val = carry + u_val;
        carry = core::hint::black_box(u_val >> HALF_LIMB_BITS);
        let u_val = u_val & HALF_LIMB_MASK;
        _le_limbs_store_half_limb(u.as_mut_slice(), (limb_index, half_limb_shift), u_val)
    }
    debug_assert_eq!(carry, 0);

    let (v_h, v_l) = ct_l_to_hls(v.normalized_v);
    let mut qs: [LimbType; 2] = [0; 2];
    for j in [3, 2, 1, 0] {
        let q = {
            // Retrieve the current most significant double half limb, i.e. a limb in width.
            let u_cur_dhl_h = le_limbs_load_half_limb(u.as_slice(), 2 + j);
            let u_cur_dhl_l = le_limbs_load_half_limb(u.as_slice(), 2 + j - 1);
            let u_cur_dhl = u_cur_dhl_h << HALF_LIMB_BITS | u_cur_dhl_l;

            // Estimate q.
            let q = u_cur_dhl / v_h;

            // Check whether q fits a half limb and cap it otherwise.
            let ov = ct_l_to_subtle_choice(q >> HALF_LIMB_BITS);
            let q = LimbType::conditional_select(&q, &HALF_LIMB_MASK, ov);
            let r = u_cur_dhl - q * v_h;
            let r_carry = ct_l_to_subtle_choice(r >> HALF_LIMB_BITS);

            // As long as r does not overflow b, i.e. a LimbType,
            // check whether q * v[n - 2] > b * r + u[j + n - 2].
            // If so, decrement q and adjust r accordingly by adding v[n-1] back.
            // Note that because v[n-1] is normalized to have its MSB set,
            // r would overflow in the second iteration at latest.
            // The second iteration is not necessary for correctness, but only serves
            // optimization purposes: it would help to avoid the "add-back" step
            // below in the majority of cases. However, for constant-time execution,
            // the add-back must get executed anyways and thus, the second iteration
            // of the "over-estimated" check here would be quite pointless. Skip it.
            let u_tail_high = le_limbs_load_half_limb(u.as_slice(), 2 + j - 2);
            let qv_tail_high = q * v_l;
            let over_estimated = !r_carry &
                ct_gt_l_l(qv_tail_high, r << HALF_LIMB_BITS | u_tail_high);
            q - LimbType::conditional_select(&0, &1, over_estimated)
        };

        // Subtract q * v at from u at position j.
        let qv = ct_mul_l_l(q, v.normalized_v);
        debug_assert!(qv.high() < (1 << HALF_LIMB_BITS) - 1);
        let mut borrow = 0;
        for k in 0..3 {
            let qv_val = qv.get_half_limb(k);
            let (u_limb_index, u_half_limb_shift) = le_limbs_half_limb_index(j + k);
            let u_val = _le_limbs_load_half_limb(u.as_slice(), (u_limb_index, u_half_limb_shift));
            let u_val = u_val.wrapping_sub(borrow);
            let u_val = u_val.wrapping_sub(qv_val);
            borrow = core::hint::black_box(u_val >> LIMB_BITS - 1);
            let u_val = u_val & HALF_LIMB_MASK;
            _le_limbs_store_half_limb(u.as_mut_slice(), (u_limb_index, u_half_limb_shift), u_val);
        }
        for k in 3..6 - j {
            let (u_limb_index, u_half_limb_shift) = le_limbs_half_limb_index(j + k);
            let u_val = _le_limbs_load_half_limb(u.as_slice(), (u_limb_index, u_half_limb_shift));
            let u_val = u_val.wrapping_sub(borrow);
            borrow = core::hint::black_box(u_val >> LIMB_BITS - 1);
            let u_val = u_val & HALF_LIMB_MASK;
            _le_limbs_store_half_limb(u.as_mut_slice(), (u_limb_index, u_half_limb_shift), u_val);
        }

        // If q had been overestimated, decrement q and add one v back to u.
        let over_estimated = ct_l_to_subtle_choice(borrow);
        let over_estimated_mask = cond_choice_to_mask(over_estimated);
        let mut carry = 0;
        let q = q - cond_select_with_mask(0, 1, over_estimated_mask);
        le_limbs_store_half_limb(qs.as_mut_slice(), j, q);
        for (k, v_val) in [(0, v_l), (1, v_h)] {
            let v_val = cond_select_with_mask(0, v_val, over_estimated_mask);
            let (u_limb_index, u_half_limb_shift) = le_limbs_half_limb_index(j + k);
            let u_val = _le_limbs_load_half_limb(u.as_slice(), (u_limb_index, u_half_limb_shift));
            let u_val = u_val.wrapping_add(carry);
            let u_val = u_val.wrapping_add(v_val);
            carry = core::hint::black_box(u_val >> HALF_LIMB_BITS);
            let u_val = u_val & HALF_LIMB_MASK;
            _le_limbs_store_half_limb(u.as_mut_slice(), (u_limb_index, u_half_limb_shift), u_val);
        }
        for k in 2..6 - j {
            let (u_limb_index, u_half_limb_shift) = le_limbs_half_limb_index(j + k);
            let u_val = _le_limbs_load_half_limb(u.as_slice(), (u_limb_index, u_half_limb_shift));
            let u_val = u_val.wrapping_add(carry);
            carry = core::hint::black_box(u_val >> HALF_LIMB_BITS);
            let u_val = u_val & HALF_LIMB_MASK;
            _le_limbs_store_half_limb(u.as_mut_slice(), (u_limb_index, u_half_limb_shift), u_val);
        }
    }

    // Conditionally shift u back by one half limb in order to undo the shifting to align with v.
    for i in [0, 1] {
        let shifted_u_val = u[i] >> HALF_LIMB_BITS | (u[i + 1] & HALF_LIMB_MASK) << HALF_LIMB_BITS;
        u[i] = cond_select_with_mask(u[i], shifted_u_val, shifted_v_mask);
    }
    // Divide u by the scaling.
    debug_assert_eq!(u[2], 0);
    debug_assert_eq!(u[1], 0);
    debug_assert_eq!(u[0] % v.scaling, 0);
    u[0] /= v.scaling;

    (DoubleLimb::new(qs[1], qs[0]), u[0])
}

#[cfg(not(all(feature = "enable_arch_math_asm", target_arch = "x86_64")))]
pub use self::GenericCtDivDlLNormalizedDivisor as CtDivDlLNormalizedDivisor;
#[cfg(not(all(feature = "enable_arch_math_asm", target_arch = "x86_64")))]
pub use self::generic_ct_div_dl_l as ct_div_dl_l;

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
pub use x86_64_math::CtDivDlLNormalizedDivisor as CtDivDlLNormalizedDivisor;
#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
pub use x86_64_math::ct_div_dl_l as ct_div_dl_l;

#[test]
fn test_ct_div_dl_l() {
    fn div_and_check(u: DoubleLimb, v: LimbType) {
        let normalized_v = CtDivDlLNormalizedDivisor::new(v);

        let (q, r) = ct_div_dl_l(&u.clone(), &normalized_v);

        // Multiply q by v again and add the remainder back, the result should match the initial u.
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
                let u_h = if i != 0 {
                    !0 >> (LIMB_BITS - i)
                } else {
                    0
                };
                let u = DoubleLimb::new(u_h, !0);
                let v = 1 << j1 | 1 << j2;
                div_and_check(u, v);
            }
        }
    }
}

// Position of MSB + 1, if any, zero otherwise.
pub fn ct_find_last_set_bit_l(mut v: LimbType) -> u32 {
    let mut bits = LIMB_BITS;
    assert!(bits != 0);
    assert!(bits & bits - 1 == 0); // Is a power of two.
    let mut count: u32 = 0;
    while bits > 1 {
        bits /= 2;
        let v_l = v & (1 << bits) - 1;
        let v_h = v >> bits;
        let upper = ct_neq_l_l(v_h, 0);
        count += u32::conditional_select(&0, &bits, upper);
        v = LimbType::conditional_select(&v_l, &v_h, upper);
    }
    debug_assert!(v <= 1);
    count += v as u32;
    count
}

#[test]
fn test_ct_find_last_set_bit_l() {
    assert_eq!(ct_find_last_set_bit_l(0), 0);

    for i in 0..LIMB_BITS {
        let v = 1 << i;
        assert_eq!(ct_find_last_set_bit_l(v), i + 1);
        assert_eq!(ct_find_last_set_bit_l(v - 1), i);
    }
}

pub fn ct_find_last_set_byte_l(v: LimbType) -> usize {
    ((ct_find_last_set_bit_l(v) + 8 - 1) / 8) as usize
}
