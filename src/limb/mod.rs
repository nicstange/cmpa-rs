//! Definitions and arithmetic primitives related to [LimbType], the basic unit
//! of multiprecision integer arithmetic.
use core::arch::asm;
use core::convert;
use core::mem;
use core::ops;
#[cfg(feature = "zeroize")]
use zeroize::Zeroize;

/// The basic unit used by the multiprecision integer arithmetic implementation.
///
/// # Notes
///
/// The following arithmetic on a [`LimbType`] is assumed to be constant-time:
/// - Binary operations: `not`, `or`, `and`, `xor`.
/// - Wrapping addition and subtraction of two [`LimbType`] words.
/// - Multiplication of two [`LimbType`] words where the result also fits a
///   [`LimbType`].
/// - Division of one [`LimbType`] by another.
///
/// Wider [`LimbType`] type defines would improve multiprecision arithmetic
/// performance, but it must be made sure that all of the operations from above
/// map to (constant-time) CPU instructions and are not implemented by e.g. some
/// architecture support runtime library. For now, the smallest
/// common denominator of `u32` is chosen.

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
const HALF_LIMB_MASK: LimbType = ct_lsb_mask_l(HALF_LIMB_BITS);

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
mod x86_64_math;

// core::hint::black_box() is inefficient: it writes and reads from memory.
#[inline(always)]
pub fn black_box_l(v: LimbType) -> LimbType {
    let result: LimbType;
    unsafe {
        asm!("/* {v} */", v = inout(reg) v => result, options(pure, nomem, nostack));
    }
    result
}

#[derive(Clone, Copy, Debug)]
pub struct LimbChoice {
    mask: LimbType,
}

impl LimbChoice {
    pub const fn new(cond: LimbType) -> Self {
        debug_assert!(cond == 0 || cond == 1);
        Self {
            mask: (0 as LimbType).wrapping_sub(cond),
        }
    }

    pub fn unwrap(&self) -> LimbType {
        black_box_l(self.mask & 1)
    }

    pub const fn select(&self, v0: LimbType, v1: LimbType) -> LimbType {
        v0 ^ (self.mask & (v0 ^ v1))
    }

    pub fn select_usize(&self, v0: usize, v1: usize) -> usize {
        let cond = self.unwrap() as usize;
        let mask = (0_usize).wrapping_sub(cond);
        v0 ^ (mask & (v0 ^ v1))
    }
}

impl convert::From<LimbType> for LimbChoice {
    fn from(value: LimbType) -> Self {
        Self::new(value)
    }
}

impl ops::Not for LimbChoice {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self { mask: !self.mask }
    }
}

impl ops::BitAnd for LimbChoice {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            mask: self.mask & rhs.mask,
        }
    }
}

impl ops::BitAndAssign for LimbChoice {
    fn bitand_assign(&mut self, rhs: Self) {
        self.mask &= rhs.mask
    }
}

impl ops::BitOr for LimbChoice {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            mask: self.mask | rhs.mask,
        }
    }
}

impl ops::BitOrAssign for LimbChoice {
    fn bitor_assign(&mut self, rhs: Self) {
        self.mask |= rhs.mask
    }
}

/// Prerequisite trait for the [`zeroize::DefaultIsZeroes`] marker trait.
#[cfg(feature = "zeroize")]
impl Default for LimbChoice {
    fn default() -> Self {
        Self::from(0)
    }
}

/// Marker trait enabling a generic [`zeroize::Zeroize`] trait implementation.
#[cfg(feature = "zeroize")]
impl zeroize::DefaultIsZeroes for LimbChoice {}

#[allow(unused)]
pub fn generic_ct_is_nonzero_l(v: LimbType) -> LimbType {
    // This trick is from subtle::*::ct_eq():
    // if v is non-zero, then v or -v or both have the high bit set.
    black_box_l((v | v.wrapping_neg()) >> (LIMB_BITS - 1))
}

#[cfg(not(all(feature = "enable_arch_math_asm", target_arch = "x86_64")))]
pub use self::generic_ct_is_nonzero_l as ct_is_nonzero_l;

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
pub use x86_64_math::ct_is_nonzero_l;

#[allow(unused)]
pub fn generic_ct_is_zero_l(v: LimbType) -> LimbType {
    (1 as LimbType) ^ ct_is_nonzero_l(v)
}

#[cfg(not(all(feature = "enable_arch_math_asm", target_arch = "x86_64")))]
pub use self::generic_ct_is_zero_l as ct_is_zero_l;

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
pub use x86_64_math::ct_is_zero_l;

pub fn ct_eq_l_l(v0: LimbType, v1: LimbType) -> LimbChoice {
    LimbChoice::from(ct_is_zero_l(v0 ^ v1))
}

pub fn ct_neq_l_l(v0: LimbType, v1: LimbType) -> LimbChoice {
    !ct_eq_l_l(v0, v1)
}

pub fn ct_lt_or_eq_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    let (borrow, diff) = ct_sub_l_l(v0, v1);
    debug_assert!(diff != 0 || borrow == 0);
    (borrow, ct_is_zero_l(diff))
}

pub fn ct_lt_l_l(v0: LimbType, v1: LimbType) -> LimbChoice {
    let (borrow, _) = ct_sub_l_l(v0, v1);
    LimbChoice::from(borrow)
}

pub fn ct_leq_l_l(v0: LimbType, v1: LimbType) -> LimbChoice {
    !ct_lt_l_l(v1, v0)
}

pub fn ct_gt_or_eq_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    ct_lt_or_eq_l_l(v1, v0)
}

pub fn ct_gt_l_l(v0: LimbType, v1: LimbType) -> LimbChoice {
    ct_lt_l_l(v1, v0)
}

pub fn ct_geq_l_l(v0: LimbType, v1: LimbType) -> LimbChoice {
    ct_leq_l_l(v1, v0)
}

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
/// Returns a pair of upper and lower half limb, in this order.
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

/// Construct a limb from upper and lower half limbs in constant time.
///
/// Runs in constant time.
///
/// # Arguments
///
/// * `vh` - the upper half limb.
/// * `vl` - the upper half limb.
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
#[allow(unused)]
pub fn generic_ct_add_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    // Don't rely on overflowing_add() for determining the carry -- that would
    // almost certainly branch and not be constant-time.
    let v0 = black_box_l(v0);
    let v1 = black_box_l(v1);
    let r = v0.wrapping_add(v1);
    let carry = black_box_l((((v0 | v1) & !r) | (v0 & v1)) >> (LIMB_BITS - 1));
    (carry, r)
}

#[cfg(not(all(feature = "enable_arch_math_asm", target_arch = "x86_64")))]
pub use self::generic_ct_add_l_l as ct_add_l_l;

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
pub use x86_64_math::ct_add_l_l;

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
/// Returns a pair of borrow and the [`LimbType::BITS`] lower bits of the
/// difference.
///
/// Runs in constant time.
///
/// # Arguments:
///
/// * `v0` - first operand
/// * `v1` - second operand
#[allow(unused)]
pub fn generic_ct_sub_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    // Don't rely on overflowing_sub() for determining the borrow -- that would
    // almost certainly branch and not be constant-time.
    let v0 = black_box_l(v0);
    let v1 = black_box_l(v1);
    let r = v0.wrapping_sub(v1);
    let borrow = black_box_l((((r | v1) & !v0) | (v1 & r)) >> (LIMB_BITS - 1));
    (borrow, r)
}

#[cfg(not(all(feature = "enable_arch_math_asm", target_arch = "x86_64")))]
pub use self::generic_ct_sub_l_l as ct_sub_l_l;

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
pub use x86_64_math::ct_sub_l_l;

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
/// Used internally for the result of [`LimbType`] multiplications and also for
/// the implementation of multiprecision integer division.
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
    /// Interpret the [`DoubleLimb`] as an integer composed of four half limbs
    /// and extract the one at at a given position.
    ///
    /// # Arguments
    ///
    /// * `i` - index of the half limb to extract, the half limbs are ordered by
    ///   increasing significance.
    fn get_half_limb(&self, i: usize) -> LimbType {
        if i & 1 != 0 {
            black_box_l(self.v[i / 2] >> HALF_LIMB_BITS)
        } else {
            black_box_l(self.v[i / 2] & HALF_LIMB_MASK)
        }
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
pub use self::generic_ct_mul_l_l as ct_mul_l_l;

#[cfg(all(feature = "enable_arch_math_asm", target_arch = "x86_64"))]
pub use x86_64_math::ct_mul_l_l;

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

pub struct CtLDivisor {
    m: LimbType,
    v: LimbType,
    v_width: u32,
}

#[derive(Debug)]
pub enum CtLDivisorError {
    DivisorIsZero,
}

impl CtLDivisor {
    pub fn new(v: LimbType) -> Result<Self, CtLDivisorError> {
        if ct_is_zero_l(v) != 0 {
            return Err(CtLDivisorError::DivisorIsZero);
        }
        debug_assert_ne!(v, 0);
        let v_width = ct_find_last_set_bit_l(v) as u32;
        let m = Self::ct_compute_m(v, v_width);
        Ok(Self { m, v, v_width })
    }

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

    pub fn nonct_new(v: LimbType) -> Result<Self, CtLDivisorError> {
        if v == 0 {
            return Err(CtLDivisorError::DivisorIsZero);
        }
        let v_width = ct_find_last_set_bit_l(v) as u32;
        let m = Self::nonct_compute_m(v, v_width);
        Ok(Self { m, v, v_width })
    }

    /// Compute the multiplier for a given divisor in _non-constant time_.
    ///
    /// Compute
    /// *(2<sup>[`LIMB_BITS`] + `v_width`</sup> - 1) / `v` -
    /// 2<sup>[`LIMB_BITS`]</sup>*
    fn nonct_compute_m(v: LimbType, v_width: u32) -> LimbType {
        NonCtLDivisor::new(v).unwrap().do_div(
            &DoubleLimb::new(ct_lsb_mask_l(v_width) - v, !0)
        ).0
    }
}

pub trait LDivisorPrivate {
    fn do_div(&self, u: &DoubleLimb) -> (LimbType, LimbType);
    fn get_v(&self) -> LimbType;
}

impl LDivisorPrivate for CtLDivisor {
    fn do_div(&self, u: &DoubleLimb) -> (LimbType, LimbType) {
        let l = self.v_width;
        debug_assert!(l > 0 && l <= LIMB_BITS);
        // The dividend u is divided in three parts:
        // - u0: The l - 1 least significant bits.
        // - u1: The single bit at position l - 1.
        // - u2: The remaining most significant bits. Because the quotient fits a limb
        //       by assumption, this fits a limb as well. In fact, it is smaller
        //       than LimbType::MAX.
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
        // So, if u1 is set, adding the two will (virtually) overflow into 2^(LIMB_BITS).
        // Dismissing this carry is equivalent to a subtracting 2^(LIMB_BITS).
        // So, n_adj will equal
        // u10 + u1 * v_norm
        // = u1 * 2^(LIMB_BITS - 1) + u0 * (LIMB_BITS - l) + u1 * v_norm
        // = u1 * (2^(LIMB_BITS - 1) + v_norm) + u0 * ...
        // = u1 * (2^LIMB_BITS + v_norm - 2^(LIMB_BITS - 1)) + u0 * ...
        // and, after dismisiing the carry
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

#[derive(Debug)]
pub enum NonCtLDivisorError {
    DivisorIsZero
}

#[allow(unused)]
struct GenericNonCtLDivisor {
    v: LimbType,
    scaling_shift: u32,
    scaling_low_src_rshift: u32,
    scaling_low_src_mask: LimbType,
}

impl GenericNonCtLDivisor {
    /// Normalize a divisor.
    ///
    /// _Does not run in constant-time_, it invokes the CPU's division
    /// instruction.
    ///
    /// # Arguments
    ///
    /// * `v` - The unscaled, but non-zero divisor to normalize.
    fn new(v: LimbType) -> Result<Self, NonCtLDivisorError> {
        if v == 0 {
            return Err(NonCtLDivisorError::DivisorIsZero);
        }

        // Shift distance to normalize v such that its MSB is set.
        let scaling_shift = v.leading_zeros();
        let scaling_low_src_rshift = (LIMB_BITS - scaling_shift) % LIMB_BITS;
        let scaling_low_src_mask = ct_lsb_mask_l(scaling_shift);

        Ok(Self { v, scaling_shift, scaling_low_src_rshift, scaling_low_src_mask })
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

/// A normalized divisor.
///
pub struct NonCtLDivisor {
    arch: ArchNonCtLDivisor,
}

impl NonCtLDivisor {
    /// Normalize a divisor.
    ///
    /// _Does not run in constant-time_, it invokes the CPU's division
    /// instruction.
    ///
    /// # Arguments
    ///
    /// * `v` - The unscaled, but non-zero divisor to normalize.
    pub fn new(v: LimbType) -> Result<Self, NonCtLDivisorError> {
        Ok(Self {
            arch: ArchNonCtLDivisor::new(v)?,
        })
    }
}

impl LDivisorPrivate for NonCtLDivisor {
    fn do_div(&self, u: &DoubleLimb) -> (LimbType, LimbType) {
        self.arch.do_div(u)
    }

    fn get_v(&self) -> LimbType {
        self.arch.get_v()
    }
}

/// Division of a [`DoubleLimb`] by a [`LimbType`].
///
///
/// # Arguments
///
/// * `u` - The [`DoubleLimb`] dividend.
/// * `v` - The [`NonCtLDivisor`] divisor.
pub fn nonct_div_dl_l(u: &DoubleLimb, v: &NonCtLDivisor) -> (DoubleLimb, LimbType) {
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

// Position of MSB + 1, if any, zero otherwise.
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

pub fn ct_find_last_set_byte_l(v: LimbType) -> usize {
    (ct_find_last_set_bit_l(v) + 8 - 1) / 8
}

// Position of LSB, if any, LIMB_BITS otherwise.
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
