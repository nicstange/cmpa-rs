//! Implementation of multiprecision integer multiplication primitives.

use core::ops::Deref as _;
use subtle::{self, ConditionallySelectable as _};
use crate::limb::LIMB_BITS;

use super::limb::{LimbType, LIMB_BYTES, DoubleLimb, ct_mul_l_l, ct_add_l_l};
use super::limbs_buffer::{mp_ct_nlimbs, MPEndianess, MPBigEndianOrder, MPLittleEndianOrder};
use super::zeroize::Zeroizing;

/// Conditionally multiply two multiprecision integers of specified endianess.
///
/// Conditionally multiply two multiprecision integers as stored in byte slices of endianess as
/// specified by the `E0` and `E1` generic parameters each.
///
/// If the `cond` argument is unset, this function is effectively a nop, but execution time is
/// independent of the value of`cond`.
///
/// Otherwise, if `cond` is set, the first operand's contents will be replaced by the computed
/// product. If the product width exceeds the available space, its most significant head part will
/// get truncated to make the result fit. That is, at most `op0.len()` of the product's least
/// significant bytes will be placed in `op0`. If truncation is to be avoided, `op0.len() >=
/// op0_in_len + op1.len()` should hold.
///
/// Runs in constant time for a given configuration of input operand widths, i.e. execution time
/// depends only on the integers' widths, but not their values and neither on `cond`.
///
/// # Arguments
///
/// * `op0` - The first input factor. Only the `op0_in_len` least significant
///           tail bytes are considered non-zero.
///           `op0` will get overwritten with the resulting, possibly
///           truncated product if `cond` is set.
///           Its endianess is specified by the `E0` generic parameter.
/// * `op0_in_len` - Number of bytes in the first input factor. Must be `<= op0.len()`.
///                  As `op0` also receives the resulting product, it must usually be allocated
///                  much larger than what would be required to only accomodate for the first input
///                  factor. Thus, the first operand's is not implicit from `op0.len()` and it must
///                  get specified separately as `op0_in_len`.
/// * `op1` - The second input factor.
///           Its endianess is specified by the `E0` generic parameter.
/// * `cond` - Whether or not to replace `op0` by the product. Intended to facilitate constant time
///            implementations of algorithms relying on conditional executiopns of multiprecision
///            integer multiplication, like e.g. binary exponentation.
///
pub fn mp_ct_mul_trunc_cond_mp_mp<E0: MPEndianess, E1: MPEndianess>(
    op0: &mut [u8], op0_in_len: usize, op1: &[u8], cond: subtle::Choice
) {
    debug_assert!(op0_in_len <= op0.len());
    let op1_nlimbs = mp_ct_nlimbs(op1.len());

    let op0_len = op0.len();
    let result_high_npartial = op0_len % LIMB_BYTES;
    let result_high_mask = if result_high_npartial == 0 {
        !0
    } else {
        (1 << 8 * result_high_npartial) - 1
    };
    let op0_nlimbs = mp_ct_nlimbs(op0_len);
    E0::zeroize_bytes_above(op0, op0_len);
    let op0_in_nlimbs = mp_ct_nlimbs(op0_in_len);

    let mut j = op0_in_nlimbs;
    while j > 0 {
        j -= 1;

        let op0_val = E0::load_l(op0, j);
        E0::store_l(op0, j, 0);

        let mut carry = 0;
        let result_nlimbs = op0_nlimbs - j;
        for k in 0..op1_nlimbs.min(result_nlimbs) {
            let unit = if k == 0 {
                1
            } else {
                0
            };
            let op1_val = LimbType::conditional_select(&unit, &E1::load_l(op1, k), cond);
            let prod: Zeroizing<DoubleLimb> = ct_mul_l_l(op0_val, op1_val).into();

            let mut result_val = E0::load_l(op0, j + k);
            let carry0;
            (carry0, result_val) = ct_add_l_l(result_val, carry);
            // prod.high() is always <= LimbType::MAX - 1, that is adding a single carry won't
            // overflow. Either prod.high() is even <= LimbType::MAX - 2, i.e. adding two carries
            // won't overflow it either, or prod.low() <= 1 and only a single carry out of the two
            // would be set.
            debug_assert!(carry0 == 0 || result_val != !0);
            debug_assert!(prod.high() < !1 || prod.low() <= 1);
            let carry1;
            (carry1, result_val) = ct_add_l_l(result_val, prod.low());
            carry = prod.high() + carry0 + carry1;

            if k != result_nlimbs - 1 {
                E0::store_l_full(op0, j + k, result_val);
            } else {
                E0::store_l(op0, j + k, result_val & result_high_mask);
            }
        }
        // Propagate the carry all the way up.
        for k in op1_nlimbs..result_nlimbs {
            let mut result_val = E0::load_l(op0, j + k);
            (carry, result_val) = ct_add_l_l(result_val, carry);
            if k != result_nlimbs - 1 {
                E0::store_l_full(op0, j + k, result_val);
            } else {
                E0::store_l(op0, j + k, result_val & result_high_mask);
            }
        }
    }
}

/// Conditionally multiply two big-endian multiprecision integers.
///
/// This is a convenience specialization of the generic [`mp_ct_mul_trunc_cond_mp_mp()`] for the
/// big-endian case.
///
pub fn mp_ct_mul_trunc_cond_be_be(
    op0: &mut [u8], op0_in_len: usize, op1: &[u8], cond: subtle::Choice
) {
    mp_ct_mul_trunc_cond_mp_mp::<MPBigEndianOrder, MPBigEndianOrder>(op0, op0_in_len, op1, cond)
}

/// Conditionally multiply two little-endian multiprecision integers.
///
/// This is a convenience specialization of the generic [`mp_ct_mul_trunc_cond_mp_mp()`] for the
/// little-endian case.
///
pub fn mp_ct_mul_trunc_cond_le_le(
    op0: &mut [u8], op0_in_len: usize, op1: &[u8], cond: subtle::Choice
) {
    mp_ct_mul_trunc_cond_mp_mp::<MPLittleEndianOrder, MPLittleEndianOrder>(op0, op0_in_len, op1, cond)
}

#[cfg(test)]
fn test_mp_ct_mul_trunc_cond_mp_mp<E0: MPEndianess, E1: MPEndianess>() {
    let mut op0: [u8; 5 * LIMB_BYTES] = [0; 5 * LIMB_BYTES];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !0);
    E0::store_l(op0, 1, !0);
    E1::store_l(op1, 0, !0);
    E1::store_l(op1, 1, !0);
    mp_ct_mul_trunc_cond_mp_mp::<E0, E1>(op0, 2 * LIMB_BYTES, op1, subtle::Choice::from(0u8));
    assert_eq!(E0::load_l(op0, 0), !0);
    assert_eq!(E0::load_l(op0, 1), !0);
    assert_eq!(E0::load_l(op0, 2), 0);
    assert_eq!(E0::load_l(op0, 3), 0);
    assert_eq!(E0::load_l(op0, 4), 0);
    mp_ct_mul_trunc_cond_mp_mp::<E0, E1>(op0, 2 * LIMB_BYTES, op1, subtle::Choice::from(1u8));
    assert_eq!(E0::load_l(op0, 0), 1);
    assert_eq!(E0::load_l(op0, 1), 0);
    assert_eq!(E0::load_l(op0, 2), !1);
    assert_eq!(E0::load_l(op0, 3), !0);
    assert_eq!(E0::load_l(op0, 4), 0);

    let mut op0: [u8; 4 * LIMB_BYTES - 1] = [0; 4 * LIMB_BYTES - 1];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !0);
    E0::store_l(op0, 1, !0);
    E1::store_l(op1, 0, !0);
    E1::store_l(op1, 1, !0);
    mp_ct_mul_trunc_cond_mp_mp::<E0, E1>(op0, 2 * LIMB_BYTES, op1, subtle::Choice::from(0u8));
    assert_eq!(E0::load_l(op0, 0), !0);
    assert_eq!(E0::load_l(op0, 1), !0);
    assert_eq!(E0::load_l(op0, 2), 0);
    assert_eq!(E0::load_l(op0, 3), 0);
    mp_ct_mul_trunc_cond_mp_mp::<E0, E1>(op0, 2 * LIMB_BYTES, op1, subtle::Choice::from(1u8));
    assert_eq!(E0::load_l(op0, 0), 1);
    assert_eq!(E0::load_l(op0, 1), 0);
    assert_eq!(E0::load_l(op0, 2), !1);
    assert_eq!(E0::load_l(op0, 3), !0 >> 8);

    let mut op0: [u8; 3 * LIMB_BYTES - 1] = [0; 3 * LIMB_BYTES - 1];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !0);
    E0::store_l(op0, 1, !0);
    E1::store_l(op1, 0, !0);
    E1::store_l(op1, 1, !0);
    mp_ct_mul_trunc_cond_mp_mp::<E0, E1>(op0, 2 * LIMB_BYTES, op1, subtle::Choice::from(0u8));
    assert_eq!(E0::load_l(op0, 0), !0);
    assert_eq!(E0::load_l(op0, 1), !0);
    assert_eq!(E0::load_l(op0, 2), 0);
    mp_ct_mul_trunc_cond_mp_mp::<E0, E1>(op0, 2 * LIMB_BYTES, op1, subtle::Choice::from(1u8));
    assert_eq!(E0::load_l(op0, 0), 1);
    assert_eq!(E0::load_l(op0, 1), 0);
    assert_eq!(E0::load_l(op0, 2), (!0 >> 8) ^ 1);

    let mut op0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !0);
    E0::store_l(op0, 1, !0 >> 2 * 8);
    E1::store_l(op1, 0, !0);
    E1::store_l(op1, 1, !0 >> 2 * 8);
    mp_ct_mul_trunc_cond_mp_mp::<E0, E1>(op0, 2 * LIMB_BYTES - 1, op1, subtle::Choice::from(0u8));
    assert_eq!(E0::load_l(op0, 0), !0);
    assert_eq!(E0::load_l(op0, 1), !0 >> 2 * 8);
    mp_ct_mul_trunc_cond_mp_mp::<E0, E1>(op0, 2 * LIMB_BYTES - 1, op1, subtle::Choice::from(1u8));
    assert_eq!(E0::load_l(op0, 0), 1);
    assert_eq!(E0::load_l(op0, 1), 0xfe << 8 * (LIMB_BYTES - 2));
}

#[test]
fn test_mp_ct_mul_trunc_cond_be_be() {
    use super::limbs_buffer::MPBigEndianOrder;
    test_mp_ct_mul_trunc_cond_mp_mp::<MPBigEndianOrder, MPBigEndianOrder>()
}

#[test]
fn test_mp_ct_mul_trunc_cond_le_le() {
    use super::limbs_buffer::MPLittleEndianOrder;
    test_mp_ct_mul_trunc_cond_mp_mp::<MPLittleEndianOrder, MPLittleEndianOrder>()
}

/// Square a multiprecision integer of specified endianess.
///
/// Square a multiprecision integer as stored in a byte slice of endianess as specified by the `E0`
/// generic parameter.
///
/// The operand's contents will be replaced by the computed square.
/// If the square's width exceeds the available space, its most significant head part will
/// get truncated to make the result fit. That is, at most `op0.len()` of the square's least
/// significant bytes will be placed in `op0`. If truncation is to be avoided, `op0.len() >=
/// 2 * op0_in_len` should hold.
///
/// Runs in constant time for a given input operand width, i.e. execution time depends only on the
/// integer's width, but not its value.
///
/// # Arguments
///
/// * `op0` - The input operand to square. Only the `op0_in_len` least significant
///           tail bytes are considered non-zero.
///           `op0` will get overwritten with the resulting, possibly
///           truncated square.
///           Its endianess is specified by the `E0` generic parameter.
/// * `op0_in_len` - Number of bytes in the input operand. Must be `<= op0.len()`.
///                  As `op0` also receives the resulting square, it must usually be allocated
///                  much larger than what would be required to only accomodate for the input
///                  operand. Thus, the operand's is not implicit from `op0.len()` and it must
///                  get specified separately as `op0_in_len`.
///
pub fn mp_ct_square_trunc_mp<E0: MPEndianess>(op0: &mut [u8], op0_in_len: usize) {
    debug_assert!(op0_in_len <= op0.len());
    let op0_len = op0.len();
    let result_high_npartial = op0_len % LIMB_BYTES;
    let result_high_mask = if result_high_npartial == 0 {
        !0
    } else {
        (1 << 8 * result_high_npartial) - 1
    };
    let op0_nlimbs = mp_ct_nlimbs(op0_len);
    E0::zeroize_bytes_above(op0, op0_len);
    let op0_in_nlimbs = mp_ct_nlimbs(op0_in_len);

    let mut j = op0_in_nlimbs;
    while j > 0 {
        j -= 1;

        let op0_val = E0::load_l(op0, j);
        E0::store_l(op0, j, 0);

        let mut last_prod_high: LimbType = 0;
        let mut carry = 0;
        let result_nlimbs = op0_nlimbs - j;
        // (u[j] + u[j-1:0])^2 = u[j]^2 + 2 * u[j] * u[j-1:0]) + u[j-1:0]^2.
        // Account for the middle term first.
        for k in 0..j.min(result_nlimbs) {
            // Being the upper half of a multiplication result,
            // last_prod_high <= !1 always holds trivially.
            // As will be shown below, even the invariant
            // carry <= 2 || (2 < carry <= 4 && last_prod_high <= !2)
            // holds at loop entry.
            let op1_val = E0::load_l(op0, k);
            let prod: Zeroizing<DoubleLimb> = ct_mul_l_l(op0_val, op1_val).into();
            let mut result_val = E0::load_l(op0, j + k);

            // Multiply last_prod_high, the upper half of the last iteration's multiplication, by
            // two.
            let carry0 = last_prod_high >> core::hint::black_box(LIMB_BITS - 1);
            last_prod_high = last_prod_high.wrapping_mul(2);
            // From the loop invariant, it follows that
            // - if carry <= 2, then last_prod_high <= !3 and
            // - if 2 < carry <= 4, then last_prod_high <= !5.
            // In either case, addition of carry to last_prod_high
            // does not overflow and the sum is <= !1.
            debug_assert!(last_prod_high <= !3);
            debug_assert!(carry <= 2 || last_prod_high <= !5);
            last_prod_high += carry;
            // If the sum below wraps around, then by virtue of the fact that
            // last_prod_high <= !1, the result will be <= !2. That is
            // if carry1 != 0, then result_val <= !2
            let carry1;
            (carry1, result_val) = ct_add_l_l(result_val, last_prod_high);

            // Done with last_prod_high, store away the half part to account for in the next
            // iteration. Note that as a basic property of the multiplication,
            // prod.high() is always <= LimbType::MAX - 1. Moreover, either
            // prod.high() is even <= LimbType::MAX - 2 or prod.low() <= 1.
            last_prod_high = prod.high();

            // Multiply the lower part of the current iteration's multiplication by two.
            // If prod.high() == !1, i.e. at the maximum possible value, then
            // the scaled lower half <= 2 * 1 == 2, c.f. the remark above. That is,
            // prod.high() <= !2 || (carry2 == 0 && prod_low <= 2).
            let carry2 = prod.low() >> core::hint::black_box(LIMB_BITS - 1);
            let prod_low = prod.low().wrapping_mul(2);

            // Add the scaled prod_low to the result.
            // To prove that the loop invariant does indeed hold, assume
            // that last_prod_high (== prod.high()) > !2. It needs to be
            // shown that the carry calculated below as
            // carry = carry0 + carry1 + carry2 + carry3 is <= 2.
            //
            // From the remark right above the preceeding step, the
            // assumption prod.high() > !2 implies that
            // carry2 == 0 && prod_low <= 2.
            //
            // Consider the following two cases:
            // a.) carry1 != 0:
            //     If carry1 != 0, then by the remark right before
            //     the computation of (carry1, result_val) above,
            //     result_val <= !2 at this point. As prod_low <= 2,
            //     the sum of result_val and prod_low below will not
            //     overflow, from which it follows that the associated
            //     carry3 == 0. Thus, carry2 == 0 && carry3 == 0 and
            //     the sum carry = carry0 + carry1 + carry2 + carry3 <= 2.
            // b.) carry1 == 0:
            //     Both carry1 and carry2 are zero, from which it follows
            //     trivially that
            //     carry = carry0 + carry1 + carry2 + carry3 <= 2.
            let carry3;
            (carry3, result_val) = ct_add_l_l(result_val, prod_low);
            carry = carry0 + carry1 + carry2 + carry3;
            // Confirm the loop invariant.
            debug_assert!(last_prod_high <= !1);
            debug_assert!(carry <= 2 || last_prod_high <= !2);

            if k != result_nlimbs - 1 {
                E0::store_l_full(op0, j + k, result_val);
            } else {
                E0::store_l(op0, j + k, result_val & result_high_mask);
            }
        }

        // Now handle the u[j]^2 part of the quadratic expansion.
        if j >= result_nlimbs {
            continue;
        }
        let prod: Zeroizing<DoubleLimb> = ct_mul_l_l(op0_val, op0_val).into();
         let mut result_val = E0::load_l(op0, 2 * j);
        // Multiply last_prod_high from the previous loop's last iteration by two.
        let carry0 = last_prod_high >> core::hint::black_box(LIMB_BITS - 1);
        last_prod_high = last_prod_high.wrapping_mul(2);
        // From the previous loop's invariant, it again follows that the addition of carry to
        // last_prod_high does not overflow the sum.
        debug_assert!(last_prod_high <= !3);
        debug_assert!(carry <= 2 || last_prod_high <= !5);
        last_prod_high += carry;
        let carry1;
        (carry1, result_val) = ct_add_l_l(result_val, last_prod_high);
        last_prod_high = prod.high();
        let carry2;
        (carry2, result_val) = ct_add_l_l(result_val, prod.low());
        carry = carry0 + carry1 + carry2;
        if j != result_nlimbs - 1 {
            E0::store_l_full(op0, 2 * j, result_val);
        } else {
            E0::store_l(op0, 2 * j, result_val & result_high_mask);
        }

        // Propagate the carry all the way up. The first iteration will also account for the
        // previous multiplications upper limb.
        for k in j + 1..result_nlimbs {
            let mut result_val = E0::load_l(op0, j + k);
            let carry0;
            (carry0, result_val) = ct_add_l_l(result_val, last_prod_high);
            last_prod_high = 0;
            let carry1;
            (carry1, result_val) = ct_add_l_l(result_val, carry);
            carry = carry0 + carry1;
            if k != result_nlimbs - 1 {
                E0::store_l_full(op0, j + k, result_val);
            } else {
                E0::store_l(op0, j + k, result_val & result_high_mask);
            }
        }
    }
}

#[cfg(test)]
fn test_mp_ct_square_trunc_mp<E0: MPEndianess>() {
    fn square_by_mul<E0: MPEndianess, const N: usize>(op0: &[u8; N], op0_in_len: usize) -> [u8; N] {
        let mut result = op0.clone();
        let (_, op0) = E0::split_at(op0, op0_in_len);
        mp_ct_mul_trunc_cond_mp_mp::<E0, E0>(
            result.as_mut_slice(), op0_in_len,
            op0,
            subtle::Choice::from(1u8)
        );
        result
    }

    let mut op0: [u8; 5 * LIMB_BYTES] = [0; 5 * LIMB_BYTES];
    E0::store_l(op0.as_mut_slice(), 0, !0);
    E0::store_l(op0.as_mut_slice(), 1, !0);
    let expected = square_by_mul::<E0, {5 * LIMB_BYTES}>(&op0, 2 * LIMB_BYTES);
    mp_ct_square_trunc_mp::<E0>(op0.as_mut_slice(), 2 * LIMB_BYTES);
    assert_eq!(op0, expected);

    let mut op0: [u8; 4 * LIMB_BYTES - 1] = [0; 4 * LIMB_BYTES - 1];
    E0::store_l(op0.as_mut_slice(), 0, !0);
    E0::store_l(op0.as_mut_slice(), 1, !0);
    let expected = square_by_mul::<E0, {4 * LIMB_BYTES - 1}>(&op0, 2 * LIMB_BYTES);
    mp_ct_square_trunc_mp::<E0>(op0.as_mut_slice(), 2 * LIMB_BYTES);
    assert_eq!(op0, expected);

    let mut op0: [u8; 3 * LIMB_BYTES - 1] = [0; 3 * LIMB_BYTES - 1];
    E0::store_l(op0.as_mut_slice(), 0, !0);
    E0::store_l(op0.as_mut_slice(), 1, !0);
    let expected = square_by_mul::<E0, {3 * LIMB_BYTES - 1}>(&op0, 2 * LIMB_BYTES);
    mp_ct_square_trunc_mp::<E0>(op0.as_mut_slice(), 2 * LIMB_BYTES);
    assert_eq!(op0, expected);


    let mut op0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    E0::store_l(op0.as_mut_slice(), 0, !0);
    E0::store_l(op0.as_mut_slice(), 1, !0 >> 2 * 8);
    let expected = square_by_mul::<E0, {2 * LIMB_BYTES - 1}>(&op0, 2 * LIMB_BYTES - 1);
    mp_ct_square_trunc_mp::<E0>(op0.as_mut_slice(), 2 * LIMB_BYTES - 1);
    assert_eq!(op0, expected);
}

#[test]
fn test_mp_ct_square_trunc_be() {
    test_mp_ct_square_trunc_mp::<MPBigEndianOrder>()
}

#[test]
fn test_mp_ct_square_trunc_le() {
    test_mp_ct_square_trunc_mp::<MPLittleEndianOrder>()
}

// Multiply multiprecision integer by a limb.
pub fn mp_ct_mul_trunc_mp_l<E0: MPEndianess>(op0: &mut [u8], op0_in_len: usize, op1: LimbType) {
    debug_assert!(op0_in_len <= op0.len());
    let op0_len = op0.len();
    let result_high_npartial = op0_len % LIMB_BYTES;
    let result_high_mask = if result_high_npartial == 0 {
        !0
    } else {
        (1 << 8 * result_high_npartial) - 1
    };
    let op0_nlimbs = mp_ct_nlimbs(op0_len);
    E0::zeroize_bytes_above(op0, op0_in_len);
    let op0_in_nlimbs = mp_ct_nlimbs(op0_in_len);

    if op0_in_len == 0 {
        return;
    }

    let mut carry = 0;
    for j in 0..op0_in_nlimbs {
        let op0_val = E0::load_l(op0, j);
        let prod: Zeroizing<DoubleLimb> = ct_mul_l_l(op0_val, op1).into();

        let (carry0, result_val) = ct_add_l_l(prod.low(), carry);
        debug_assert!(prod.high() <= !1);
        carry = prod.high() + carry0; // Does not overflow.

        if j != op0_nlimbs - 1 {
            E0::store_l_full(op0, j, result_val);
        } else {
            E0::store_l(op0, j, result_val & result_high_mask);
        }
    }

    if op0_in_nlimbs != op0_nlimbs {
        if op0_in_nlimbs != op0_nlimbs - 1 {
            E0::store_l_full(op0, op0_in_nlimbs, carry);
        } else {
            E0::store_l(op0, op0_in_nlimbs, carry & result_high_mask);
        }
    }
}

#[cfg(test)]
fn test_mp_ct_mul_trunc_mp_l<E0: MPEndianess>() {
    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op0 = op0.as_mut_slice();
    E0::store_l(op0, 0, !0);
    E0::store_l(op0, 1, !0);
    let op1 = 0;
    mp_ct_mul_trunc_mp_l::<E0>(op0, 2 * LIMB_BYTES, op1);
    assert_eq!(E0::load_l(op0, 0), 0);
    assert_eq!(E0::load_l(op0, 1), 0);

    let mut op0: [u8; 2 * LIMB_BYTES + 1] = [0; 2 * LIMB_BYTES + 1];
    let op0 = op0.as_mut_slice();
    E0::store_l(op0, 0, !0);
    E0::store_l(op0, 1, !0);
    let op1 = 2;
    mp_ct_mul_trunc_mp_l::<E0>(op0, 2 * LIMB_BYTES, op1);
    assert_eq!(E0::load_l(op0, 0), !1);
    assert_eq!(E0::load_l(op0, 1), !0);
    assert_eq!(E0::load_l(op0, 2), 1);

    let mut op0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op0 = op0.as_mut_slice();
    E0::store_l(op0, 0, !0);
    E0::store_l(op0, 1, !0 >> 8);
    let op1 = 2;
    mp_ct_mul_trunc_mp_l::<E0>(op0, 2 * LIMB_BYTES - 1, op1);
    assert_eq!(E0::load_l(op0, 0), !1);
    assert_eq!(E0::load_l(op0, 1), !0 >> 8);
}

#[test]
fn test_mp_ct_mul_trunc_be_l() {
    test_mp_ct_mul_trunc_mp_l::<MPBigEndianOrder>()
}

#[test]
fn test_mp_ct_mul_trunc_le_l() {
    test_mp_ct_mul_trunc_mp_l::<MPLittleEndianOrder>()
}
