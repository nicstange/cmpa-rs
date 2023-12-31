// SPDX-License-Identifier: Apache-2.0
// Copyright 2023 SUSE LLC
// Author: Nicolai Stange <nstange@suse.de>

//! Implementation of multiprecision integer multiplication primitives.

use super::limb::{
    black_box_l, ct_add_l_l, ct_mul_add_l_l_l_c, ct_mul_l_l, LimbChoice, LimbType, LIMB_BITS,
};
#[cfg(test)]
use super::limbs_buffer::MpMutUIntSlice;
use super::limbs_buffer::{ct_mp_nlimbs, MpMutUInt, MpUIntCommon};

/// Conditionally multiply two multiprecision integers of specified endianess.
///
/// If the `cond` argument is unset, this function is effectively a nop, but
/// execution time is independent of the value of`cond`.
///
/// Otherwise, if `cond` is set, the first operand's contents will be replaced
/// by the computed product. If the product width exceeds the available space,
/// its most significant head part will get truncated to make the result fit.
/// That is, at most `op0.len()` of the product's least significant bytes will
/// be placed in `op0`. If truncation is to be avoided, `op0.len() >= op0_in_len
/// + op1.len()` should hold.
///
/// Runs in constant time for a given configuration of input operand widths,
/// i.e. execution time depends only on the integers' widths, but not their
/// values and neither on `cond`.
///
/// # Arguments
///
/// * `op0` - The first input factor. Only the `op0_in_len` least significant
///   tail bytes are considered non-zero. `op0` will get overwritten with the
///   resulting, possibly truncated product if `cond` is set.
/// * `op0_in_len` - Number of bytes in the first input factor. Must be `<=
///   op0.len()`. As `op0` also receives the resulting product, it must usually
///   be allocated much larger than what would be required to only accomodate
///   for the first input factor. Thus, the first operand's is not implicit from
///   `op0.len()` and it must get specified separately as `op0_in_len`.
/// * `op1` - The second input factor.
/// * `cond` - Whether or not to replace `op0` by the product. Intended to
///   facilitate constant time implementations of algorithms relying on
///   conditional executiopns of multiprecision integer multiplication, like
///   e.g. binary exponentation.
pub fn ct_mul_trunc_cond_mp_mp<T0: MpMutUInt, T1: MpUIntCommon>(
    op0: &mut T0,
    op0_in_len: usize,
    op1: &T1,
    cond: LimbChoice,
) {
    debug_assert!(op0_in_len <= op0.len());
    // If op1's length is zero, interpret that as a zero.
    if op1.is_empty() {
        let cond_mask = cond.select(0, !0);
        for j in 0..op0.nlimbs() {
            op0.store_l(j, op0.load_l(j) & !cond_mask);
        }
        return;
    }
    let op1_nlimbs = op1.nlimbs();

    let result_high_mask = op0.partial_high_mask();
    let op0_nlimbs = op0.nlimbs();
    op0.clear_bytes_above(op0_in_len);
    let op0_in_nlimbs = ct_mp_nlimbs(op0_in_len);

    let mut j = op0_in_nlimbs;
    while j > 0 {
        j -= 1;

        let op0_val = op0.load_l(j);
        op0.store_l(j, 0);

        let mut carry = 0;
        let result_nlimbs = op0_nlimbs - j;
        let mut cond_unit = cond.select(1, 0);
        for k in 0..op1_nlimbs.min(result_nlimbs) {
            let op1_val = cond.select(0, op1.load_l(k)) | cond_unit;
            cond_unit = 0;

            let mut result_val = op0.load_l(j + k);
            (carry, result_val) = ct_mul_add_l_l_l_c(result_val, op0_val, op1_val, carry);

            if k != result_nlimbs - 1 || !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
                op0.store_l_full(j + k, result_val);
            } else {
                op0.store_l(j + k, result_val & result_high_mask);
            }
        }
        // Propagate the carry all the way up.
        for k in op1_nlimbs..result_nlimbs {
            let mut result_val = op0.load_l(j + k);
            (carry, result_val) = ct_add_l_l(result_val, carry);
            if k != result_nlimbs - 1 || !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
                op0.store_l_full(j + k, result_val);
            } else {
                op0.store_l(j + k, result_val & result_high_mask);
            }
        }
    }
}

#[cfg(test)]
fn test_ct_mul_trunc_cond_mp_mp<T0: MpMutUIntSlice, T1: MpMutUIntSlice>() {
    use super::limb::LIMB_BYTES;

    let mut op0 = tst_mk_mp_backing_vec!(T0, 5 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    op1.store_l(0, !0);
    op1.store_l(1, !0);
    ct_mul_trunc_cond_mp_mp(&mut op0, 2 * LIMB_BYTES, &op1, LimbChoice::from(0));
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), !0);
    assert_eq!(op0.load_l(2), 0);
    assert_eq!(op0.load_l(3), 0);
    assert_eq!(op0.load_l(4), 0);
    ct_mul_trunc_cond_mp_mp(&mut op0, 2 * LIMB_BYTES, &op1, LimbChoice::from(1));
    assert_eq!(op0.load_l(0), 1);
    assert_eq!(op0.load_l(1), 0);
    assert_eq!(op0.load_l(2), !1);
    assert_eq!(op0.load_l(3), !0);
    assert_eq!(op0.load_l(4), 0);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 3 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    op1.store_l(0, !0);
    op1.store_l(1, !0);
    ct_mul_trunc_cond_mp_mp(&mut op0, 2 * LIMB_BYTES, &op1, LimbChoice::from(0));
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), !0);
    assert_eq!(op0.load_l(2), 0);
    ct_mul_trunc_cond_mp_mp(&mut op0, 2 * LIMB_BYTES, &op1, LimbChoice::from(1));
    assert_eq!(op0.load_l(0), 1);
    assert_eq!(op0.load_l(1), 0);
    assert_eq!(op0.load_l(2), !1);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    op1.store_l(0, !0);
    op1.store_l(1, !0);
    ct_mul_trunc_cond_mp_mp(&mut op0, 2 * LIMB_BYTES, &op1, LimbChoice::from(0));
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), !0);
    ct_mul_trunc_cond_mp_mp(&mut op0, 2 * LIMB_BYTES, &op1, LimbChoice::from(1));
    assert_eq!(op0.load_l(0), 1);
    assert_eq!(op0.load_l(1), 0);

    if !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
        return;
    }

    let mut op0 = tst_mk_mp_backing_vec!(T0, 4 * LIMB_BYTES - 1);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    op1.store_l(0, !0);
    op1.store_l(1, !0);
    ct_mul_trunc_cond_mp_mp(&mut op0, 2 * LIMB_BYTES, &op1, LimbChoice::from(0));
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), !0);
    assert_eq!(op0.load_l(2), 0);
    assert_eq!(op0.load_l(3), 0);
    ct_mul_trunc_cond_mp_mp(&mut op0, 2 * LIMB_BYTES, &op1, LimbChoice::from(1));
    assert_eq!(op0.load_l(0), 1);
    assert_eq!(op0.load_l(1), 0);
    assert_eq!(op0.load_l(2), !1);
    assert_eq!(op0.load_l(3), !0 >> 8);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 3 * LIMB_BYTES - 1);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    op1.store_l(0, !0);
    op1.store_l(1, !0);
    ct_mul_trunc_cond_mp_mp(&mut op0, 2 * LIMB_BYTES, &op1, LimbChoice::from(0));
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), !0);
    assert_eq!(op0.load_l(2), 0);
    ct_mul_trunc_cond_mp_mp(&mut op0, 2 * LIMB_BYTES, &op1, LimbChoice::from(1));
    assert_eq!(op0.load_l(0), 1);
    assert_eq!(op0.load_l(1), 0);
    assert_eq!(op0.load_l(2), (!0 >> 8) ^ 1);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES - 1);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0 >> 2 * 8);
    op1.store_l(0, !0);
    op1.store_l(1, !0 >> 2 * 8);
    ct_mul_trunc_cond_mp_mp(&mut op0, 2 * LIMB_BYTES - 1, &op1, LimbChoice::from(0));
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), !0 >> 2 * 8);
    ct_mul_trunc_cond_mp_mp(&mut op0, 2 * LIMB_BYTES - 1, &op1, LimbChoice::from(1));
    assert_eq!(op0.load_l(0), 1);
    assert_eq!(op0.load_l(1), 0xfe << 8 * (LIMB_BYTES - 2));
}

#[test]
fn test_ct_mul_trunc_cond_be_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_mul_trunc_cond_mp_mp::<MpMutBigEndianUIntByteSlice, MpMutBigEndianUIntByteSlice>()
}

#[test]
fn test_ct_mul_trunc_cond_le_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_mul_trunc_cond_mp_mp::<MpMutLittleEndianUIntByteSlice, MpMutLittleEndianUIntByteSlice>()
}

#[test]
fn test_ct_mul_trunc_cond_ne_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_mul_trunc_cond_mp_mp::<MpMutNativeEndianUIntLimbsSlice, MpMutNativeEndianUIntLimbsSlice>(
    )
}

pub fn ct_mul_trunc_mp_mp<T0: MpMutUInt, T1: MpUIntCommon>(
    op0: &mut T0,
    op0_in_len: usize,
    op1: &T1,
) {
    ct_mul_trunc_cond_mp_mp(op0, op0_in_len, op1, LimbChoice::from(1))
}

/// Square a multiprecision integer of specified endianess.
///
/// The operand's contents will be replaced by the computed square.
/// If the square's width exceeds the available space, its most significant head
/// part will get truncated to make the result fit. That is, at most `op0.len()`
/// of the square's least significant bytes will be placed in `op0`. If
/// truncation is to be avoided, `op0.len() >= 2 * op0_in_len` should hold.
///
/// Runs in constant time for a given input operand width, i.e. execution time
/// depends only on the integer's width, but not its value.
///
/// # Arguments
///
/// * `op0` - The input operand to square. Only the `op0_in_len` least
///   significant tail bytes are considered non-zero. `op0` will get overwritten
///   with the resulting, possibly truncated square.
/// * `op0_in_len` - Number of bytes in the input operand. Must be `<=
///   op0.len()`. As `op0` also receives the resulting square, it must usually
///   be allocated much larger than what would be required to only accomodate
///   for the input operand. Thus, the operand's is not implicit from
///   `op0.len()` and it must get specified separately as `op0_in_len`.
pub fn ct_square_trunc_mp<T0: MpMutUInt>(op0: &mut T0, op0_in_len: usize) {
    debug_assert!(op0_in_len <= op0.len());
    let result_high_mask = op0.partial_high_mask();
    let op0_nlimbs = op0.nlimbs();
    op0.clear_bytes_above(op0_in_len);
    let op0_in_nlimbs = ct_mp_nlimbs(op0_in_len);

    let mut j = op0_in_nlimbs;
    while j > 0 {
        j -= 1;

        let op0_val = op0.load_l(j);
        op0.store_l(j, 0);

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
            let op1_val = op0.load_l(k);
            let prod = ct_mul_l_l(op0_val, op1_val);
            let mut result_val = op0.load_l(j + k);

            // Multiply last_prod_high, the upper half of the last iteration's
            // multiplication, by two.
            let carry0 = black_box_l(last_prod_high >> (LIMB_BITS - 1));
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
            let carry2 = black_box_l(prod.low() >> (LIMB_BITS - 1));
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

            if k != result_nlimbs - 1 || !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
                op0.store_l_full(j + k, result_val);
            } else {
                op0.store_l(j + k, result_val & result_high_mask);
            }
        }

        // Now handle the u[j]^2 part of the quadratic expansion.
        if j >= result_nlimbs {
            continue;
        }
        let prod = ct_mul_l_l(op0_val, op0_val);
        let mut result_val = op0.load_l(2 * j);
        // Multiply last_prod_high from the previous loop's last iteration by two.
        let carry0 = black_box_l(last_prod_high >> (LIMB_BITS - 1));
        last_prod_high = last_prod_high.wrapping_mul(2);
        // From the previous loop's invariant, it again follows that the addition of
        // carry to last_prod_high does not overflow the sum.
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
            op0.store_l_full(2 * j, result_val);
        } else {
            op0.store_l(2 * j, result_val & result_high_mask);
        }

        // Propagate the carry all the way up. The first iteration will also account for
        // the previous multiplications upper limb.
        for k in j + 1..result_nlimbs {
            let mut result_val = op0.load_l(j + k);
            let carry0;
            (carry0, result_val) = ct_add_l_l(result_val, last_prod_high);
            last_prod_high = 0;
            let carry1;
            (carry1, result_val) = ct_add_l_l(result_val, carry);
            carry = carry0 + carry1;
            if k != result_nlimbs - 1 || !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
                op0.store_l_full(j + k, result_val);
            } else {
                op0.store_l(j + k, result_val & result_high_mask);
            }
        }
    }
}

#[cfg(test)]
fn test_ct_square_trunc_mp<T0: MpMutUIntSlice>() {
    extern crate alloc;
    use super::limb::LIMB_BYTES;
    use alloc::vec::Vec;

    fn square_by_mul<T0: MpMutUIntSlice>(
        op0: &[T0::BackingSliceElementType],
        op0_in_len: usize,
    ) -> Vec<T0::BackingSliceElementType> {
        let mut _result = Vec::from(op0);
        let mut result = T0::from_slice(&mut _result).unwrap();
        let mut op0 = Vec::from(op0);
        let mut op0 = T0::from_slice(&mut op0).unwrap();
        let op0 = op0.shrink_to(op0_in_len);
        ct_mul_trunc_mp_mp(&mut result, op0_in_len, &op0);
        drop(result);
        _result
    }

    let mut _op0 = tst_mk_mp_backing_vec!(T0, 5 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut _op0).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    drop(op0);
    let expected = square_by_mul::<T0>(&_op0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut _op0).unwrap();
    ct_square_trunc_mp(&mut op0, 2 * LIMB_BYTES);
    drop(op0);
    assert_eq!(_op0, expected);

    let mut _op0 = tst_mk_mp_backing_vec!(T0, 3 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut _op0).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    drop(op0);
    let expected = square_by_mul::<T0>(&_op0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut _op0).unwrap();
    ct_square_trunc_mp(&mut op0, 2 * LIMB_BYTES);
    drop(op0);
    assert_eq!(_op0, expected);

    let mut _op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut _op0).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    drop(op0);
    let expected = square_by_mul::<T0>(&_op0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut _op0).unwrap();
    ct_square_trunc_mp(&mut op0, 2 * LIMB_BYTES);
    drop(op0);
    assert_eq!(_op0, expected);

    if !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
        return;
    }

    let mut _op0 = tst_mk_mp_backing_vec!(T0, 4 * LIMB_BYTES - 1);
    let mut op0 = T0::from_slice(&mut _op0).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    drop(op0);
    let expected = square_by_mul::<T0>(&_op0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut _op0).unwrap();
    ct_square_trunc_mp(&mut op0, 2 * LIMB_BYTES);
    drop(op0);
    assert_eq!(_op0, expected);

    let mut _op0 = tst_mk_mp_backing_vec!(T0, 3 * LIMB_BYTES - 1);
    let mut op0 = T0::from_slice(&mut _op0).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    drop(op0);
    let expected = square_by_mul::<T0>(&_op0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut _op0).unwrap();
    ct_square_trunc_mp(&mut op0, 2 * LIMB_BYTES);
    drop(op0);
    assert_eq!(_op0, expected);

    let mut _op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES - 1);
    let mut op0 = T0::from_slice(&mut _op0).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0 >> 2 * 8);
    drop(op0);
    let expected = square_by_mul::<T0>(&_op0, 2 * LIMB_BYTES - 1);
    let mut op0 = T0::from_slice(&mut _op0).unwrap();
    ct_square_trunc_mp(&mut op0, 2 * LIMB_BYTES - 1);
    drop(op0);
    assert_eq!(_op0, expected);
}

#[test]
fn test_ct_square_trunc_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_square_trunc_mp::<MpMutBigEndianUIntByteSlice>()
}

#[test]
fn test_ct_square_trunc_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_square_trunc_mp::<MpMutLittleEndianUIntByteSlice>()
}

#[test]
fn test_ct_square_trunc_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_square_trunc_mp::<MpMutNativeEndianUIntLimbsSlice>()
}

// Multiply multiprecision integer by a limb.
pub fn ct_mul_trunc_mp_l<T0: MpMutUInt>(op0: &mut T0, op0_in_len: usize, op1: LimbType) {
    debug_assert!(op0_in_len <= op0.len());
    let result_high_mask = op0.partial_high_mask();
    let op0_nlimbs = op0.nlimbs();
    op0.clear_bytes_above(op0_in_len);
    let op0_in_nlimbs = ct_mp_nlimbs(op0_in_len);

    if op0_in_len == 0 {
        return;
    }

    let mut carry = 0;
    for j in 0..op0_in_nlimbs {
        let op0_val = op0.load_l(j);
        let result_val;
        (carry, result_val) = ct_mul_add_l_l_l_c(0, op0_val, op1, carry);

        if j != op0_nlimbs - 1 || !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
            op0.store_l_full(j, result_val);
        } else {
            op0.store_l(j, result_val & result_high_mask);
        }
    }

    if op0_in_nlimbs != op0_nlimbs {
        if op0_in_nlimbs != op0_nlimbs - 1 || !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
            op0.store_l_full(op0_in_nlimbs, carry);
        } else {
            op0.store_l(op0_in_nlimbs, carry & result_high_mask);
        }
    }
}

#[cfg(test)]
fn test_ct_mul_trunc_mp_l<T0: MpMutUIntSlice>() {
    use super::limb::LIMB_BYTES;

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    let op1 = 0;
    ct_mul_trunc_mp_l(&mut op0, 2 * LIMB_BYTES, op1);
    assert_eq!(op0.load_l(0), 0);
    assert_eq!(op0.load_l(1), 0);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 3 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    let op1 = 2;
    ct_mul_trunc_mp_l(&mut op0, 2 * LIMB_BYTES, op1);
    assert_eq!(op0.load_l(0), !1);
    assert_eq!(op0.load_l(1), !0);
    assert_eq!(op0.load_l(2), 1);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    let op1 = 2;
    ct_mul_trunc_mp_l(&mut op0, 2 * LIMB_BYTES, op1);
    assert_eq!(op0.load_l(0), !1);
    assert_eq!(op0.load_l(1), !0);

    if !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
        return;
    }

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES + 1);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    let op1 = 2;
    ct_mul_trunc_mp_l(&mut op0, 2 * LIMB_BYTES, op1);
    assert_eq!(op0.load_l(0), !1);
    assert_eq!(op0.load_l(1), !0);
    assert_eq!(op0.load_l(2), 1);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES - 1);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0 >> 8);
    let op1 = 2;
    ct_mul_trunc_mp_l(&mut op0, 2 * LIMB_BYTES - 1, op1);
    assert_eq!(op0.load_l(0), !1);
    assert_eq!(op0.load_l(1), !0 >> 8);
}

#[test]
fn test_ct_mul_trunc_be_l() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_mul_trunc_mp_l::<MpMutBigEndianUIntByteSlice>()
}

#[test]
fn test_ct_mul_trunc_le_l() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_mul_trunc_mp_l::<MpMutLittleEndianUIntByteSlice>()
}

#[test]
fn test_ct_mul_trunc_ne_l() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_mul_trunc_mp_l::<MpMutNativeEndianUIntLimbsSlice>()
}
