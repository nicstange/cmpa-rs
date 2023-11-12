// SPDX-License-Identifier: Apache-2.0
// Copyright 2023 SUSE LLC
// Author: Nicolai Stange <nstange@suse.de>

//! Implementation of multiprecision integer addition related primitives.

use super::cmp_impl::{ct_is_zero_mp, ct_lt_mp_mp, CtGeqMpMpKernel};
use super::limb::{
    ct_add_l_l, ct_add_l_l_c, ct_find_last_set_byte_l, ct_sub_l_l, ct_sub_l_l_b, LimbChoice,
    LimbType, LIMB_BITS,
};
#[cfg(test)]
use super::limbs_buffer::MpMutUIntSlice;
use super::limbs_buffer::{MpMutUInt, MpUIntCommon};

/// Conditionally add two multiprecision integers of specified endianess.
///
/// The first operand's contents will be replaced by the resulting sum and the
/// carry, if any, returned from the function.
///
/// Runs in constant time for a given configuration of input operand widths,
/// i.e. execution time depends only on the integers' widths, but not their
/// values.
///
/// # Arguments:
///
/// * `op0` - The first input addend. It will be overwritten by the resulting
///   sum. The slice length must greater or equal than the length of the second
///   addend.
/// * `op1` - The second input addend. Its length must not exceed the length of
///   `op0`.
pub fn ct_add_cond_mp_mp<T0: MpMutUInt, T1: MpUIntCommon>(
    op0: &mut T0,
    op1: &T1,
    cond: LimbChoice,
) -> LimbType {
    let op0_nlimbs = op0.nlimbs();
    let op1_nlimbs = op1.nlimbs();
    debug_assert!(op1_nlimbs <= op0_nlimbs);
    if op1_nlimbs == 0 {
        return 0;
    }

    let mut carry = 0;
    for i in 0..op1_nlimbs - 1 {
        let mut op0_val = op0.load_l_full(i);
        let op1_val = cond.select(0, op1.load_l_full(i));
        (carry, op0_val) = ct_add_l_l_c(op0_val, op1_val, carry);
        op0.store_l_full(i, op0_val);
    }

    // Propagate the carry upwards. The first iteration will also account
    // for op1's high limb.
    let mut op1_val = cond.select(0, op1.load_l(op1_nlimbs - 1));
    for i in op1_nlimbs - 1..op0_nlimbs {
        let mut op0_val = op0.load_l(i);
        (carry, op0_val) = ct_add_l_l_c(op0_val, op1_val, carry);
        op1_val = 0;
        if i != op0_nlimbs - 1 || !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
            op0.store_l_full(i, op0_val);
        } else {
            let op0_high_mask = op0.partial_high_mask();
            let op0_high_shift = op0.partial_high_shift();
            op0.store_l(i, op0_val & op0_high_mask);
            let carry_in_op0_high = (op0_val & !op0_high_mask) >> op0_high_shift;
            debug_assert!(carry == 0 || carry_in_op0_high == 0);
            carry |= carry_in_op0_high;
        }
    }
    carry
}

#[cfg(test)]
fn test_ct_add_cond_mp_mp<T0: MpMutUIntSlice, T1: MpMutUIntSlice>() {
    use super::limb::LIMB_BYTES;

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !0);
    op1.store_l(0, !0);
    let carry = ct_add_cond_mp_mp(&mut op0, &op1, LimbChoice::from(0));
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), 0);
    assert_eq!(carry, 0);
    let carry = ct_add_cond_mp_mp(&mut op0, &op1, LimbChoice::from(1));
    assert_eq!(carry, 0);
    assert_eq!(op0.load_l(0), !1);
    assert_eq!(op0.load_l(1), 1);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    op1.store_l(0, !0);
    op1.store_l(1, !0);
    let carry = ct_add_cond_mp_mp(&mut op0, &op1, LimbChoice::from(0));
    assert_eq!(carry, 0);
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), !0);
    let carry = ct_add_cond_mp_mp(&mut op0, &op1, LimbChoice::from(1));
    assert_eq!(carry, 1);
    assert_eq!(op0.load_l(0), !1);
    assert_eq!(op0.load_l(1), !0);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 1 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0);
    op1.store_l(0, !0);
    let carry = ct_add_cond_mp_mp(&mut op0, &op1, LimbChoice::from(0));
    assert_eq!(carry, 0);
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), !0);
    let carry = ct_add_cond_mp_mp(&mut op0, &op1, LimbChoice::from(1));
    assert_eq!(carry, 1);
    assert_eq!(op0.load_l(0), !1);
    assert_eq!(op0.load_l(1), 0);

    if !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS || !T1::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
        return;
    }

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES - 1);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES - 1);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !0);
    op1.store_l(0, !0);
    let carry = ct_add_cond_mp_mp(&mut op0, &op1, LimbChoice::from(0));
    assert_eq!(carry, 0);
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), 0);
    let carry = ct_add_cond_mp_mp(&mut op0, &op1, LimbChoice::from(1));
    assert_eq!(carry, 0);
    assert_eq!(op0.load_l(0), !1);
    assert_eq!(op0.load_l(1), 1);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES - 1);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES - 1);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !0);
    op0.store_l(1, !0 >> 8);
    op1.store_l(0, !0);
    op1.store_l(1, !0 >> 8);
    let carry = ct_add_cond_mp_mp(&mut op0, &op1, LimbChoice::from(0));
    assert_eq!(carry, 0);
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), !0 >> 8);
    let carry = ct_add_cond_mp_mp(&mut op0, &op1, LimbChoice::from(1));
    assert_eq!(carry, 1);
    assert_eq!(op0.load_l(0), !1);
    assert_eq!(op0.load_l(1), !0 >> 8);

    let mut op0 = tst_mk_mp_backing_vec!(T0, LIMB_BYTES - 1);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, LIMB_BYTES - 1);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !0 >> 8);
    op1.store_l(0, !0 >> 8);
    let carry = ct_add_cond_mp_mp(&mut op0, &op1, LimbChoice::from(0));
    assert_eq!(carry, 0);
    assert_eq!(op0.load_l(0), !0 >> 8);
    let carry = ct_add_cond_mp_mp(&mut op0, &op1, LimbChoice::from(1));
    assert_eq!(carry, 1);
    assert_eq!(op0.load_l(0), (!0 >> 8) ^ 1);
}

#[test]
fn test_ct_add_cond_be_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_add_cond_mp_mp::<MpMutBigEndianUIntByteSlice, MpMutBigEndianUIntByteSlice>()
}

#[test]
fn test_ct_add_cond_le_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_add_cond_mp_mp::<MpMutLittleEndianUIntByteSlice, MpMutLittleEndianUIntByteSlice>()
}

#[test]
fn test_ct_add_cond_ne_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_add_cond_mp_mp::<MpMutNativeEndianUIntLimbsSlice, MpMutNativeEndianUIntLimbsSlice>()
}

pub fn ct_add_mp_mp<T0: MpMutUInt, T1: MpUIntCommon>(op0: &mut T0, op1: &T1) -> LimbType {
    ct_add_cond_mp_mp(op0, op1, LimbChoice::from(1))
}

#[derive(Debug)]
pub enum CtAddModMpMpError {
    InconsistentOperandLengths,
}

pub fn ct_add_mod_mp_mp<T0: MpMutUInt, T1: MpUIntCommon, NT: MpUIntCommon>(
    op0: &mut T0,
    op1: &T1,
    n: &NT,
) -> Result<(), CtAddModMpMpError> {
    if !n.len_is_compatible_with(op0.len()) {
        return Err(CtAddModMpMpError::InconsistentOperandLengths);
    }
    debug_assert!(op0.nlimbs() >= n.nlimbs());

    debug_assert_ne!(ct_lt_mp_mp(op0, n).unwrap(), 0);
    debug_assert_ne!(ct_lt_mp_mp(op1, n).unwrap(), 0);
    debug_assert!(!n.is_empty());

    let mut result_geq_n_kernel = CtGeqMpMpKernel::new();
    let mut carry = 0;
    let op1_n_common_nlimbs = op1.nlimbs().min(n.nlimbs());
    for i in 0..op1_n_common_nlimbs {
        let result_val;
        (carry, result_val) = ct_add_l_l_c(op0.load_l(i), op1.load_l(i), carry);
        result_geq_n_kernel.update(result_val, n.load_l(i));
        if !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS || i != op0.nlimbs() - 1 {
            op0.store_l_full(i, result_val);
        } else {
            op0.store_l(i, result_val & op0.partial_high_mask());
        }
    }
    // Propagate the carry upwards.
    for i in op1_n_common_nlimbs..n.nlimbs() {
        let result_val;
        (carry, result_val) = ct_add_l_l(op0.load_l(i), carry);
        result_geq_n_kernel.update(result_val, n.load_l(i));
        if !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS || i != op0.nlimbs() - 1 {
            op0.store_l_full(i, result_val);
        } else {
            op0.store_l(i, result_val & op0.partial_high_mask());
        }
    }

    // Subtract an n if the value is >= n to bring it back into range.
    let ov = result_geq_n_kernel.finish() | LimbChoice::from(carry);
    ct_sub_cond_mp_mp(op0, n, ov);

    Ok(())
}

#[cfg(test)]
fn test_ct_add_mod_mp_mp<T0: MpMutUIntSlice, T1: MpMutUIntSlice, NT: MpMutUIntSlice>() {
    use crate::limbs_buffer::MpUIntCommonPriv;

    use super::cmp_impl::ct_eq_mp_mp;
    use super::div_impl::{ct_mod_mp_mp, CtMpDivisor};
    use super::limb::LIMB_BYTES;

    let mut n = tst_mk_mp_backing_vec!(NT, LIMB_BYTES + 1);
    let mut n = NT::from_slice(&mut n).unwrap();
    for i in 0..n.nlimbs() {
        if i != n.nlimbs() - 1 {
            n.store_l_full(i, !0);
        } else {
            n.store_l(i, n.partial_high_mask());
        }
    }
    let op0_max_len = T0::n_backing_elements_for_len(LIMB_BYTES + 1) * T0::BACKING_ELEMENT_SIZE;
    n.clear_bytes_above(op0_max_len);
    let n_divisor = CtMpDivisor::new(&n, None).unwrap();

    for op0_len in 1..n.len() + 1 {
        let mut op0 = tst_mk_mp_backing_vec!(T0, n.len());
        let mut op0 = T0::from_slice(&mut op0).unwrap();
        for op1_len in 1..op0_len + 1 {
            let mut op1 = tst_mk_mp_backing_vec!(T1, op1_len);
            let mut op1 = T1::from_slice(&mut op1).unwrap();
            for i in 0..op0.nlimbs() {
                if i != op0.nlimbs() - 1 {
                    op0.store_l_full(i, !0);
                } else {
                    op0.store_l(i, op0.partial_high_mask());
                }
            }
            op0.clear_bytes_above(op0_len);
            ct_sub_mp_l(&mut op0, 1);
            debug_assert_ne!(ct_lt_mp_mp(&op0, &n).unwrap(), 0);

            for i in 0..op1.nlimbs() {
                if i != op1.nlimbs() - 1 {
                    op1.store_l_full(i, !0);
                } else {
                    op1.store_l(i, op1.partial_high_mask());
                }
            }
            op1.clear_bytes_above(op1_len);
            ct_sub_mp_l(&mut op1, 1);
            debug_assert_ne!(ct_lt_mp_mp(&op1, &n).unwrap(), 0);

            let mut expected = tst_mk_mp_backing_vec!(T0, op0_len + 1);
            let mut expected = T0::from_slice(&mut expected).unwrap();
            expected.copy_from(&op0);
            ct_add_mp_mp(&mut expected, &op1);
            ct_mod_mp_mp(None, &mut expected, &n_divisor);

            ct_add_mod_mp_mp(&mut op0, &op1, &n).unwrap();
            debug_assert_ne!(ct_lt_mp_mp(&op0, &n).unwrap(), 0);

            assert_ne!(ct_eq_mp_mp(&expected, &op0).unwrap(), 0);
        }
    }
}

#[test]
fn test_ct_add_mod_be_be_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_add_mod_mp_mp::<
        MpMutBigEndianUIntByteSlice,
        MpMutBigEndianUIntByteSlice,
        MpMutBigEndianUIntByteSlice,
    >();
}

#[test]
fn test_ct_add_mod_le_le_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_add_mod_mp_mp::<
        MpMutLittleEndianUIntByteSlice,
        MpMutLittleEndianUIntByteSlice,
        MpMutLittleEndianUIntByteSlice,
    >();
}

#[test]
fn test_ct_add_mod_ne_ne_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_add_mod_mp_mp::<
        MpMutNativeEndianUIntLimbsSlice,
        MpMutNativeEndianUIntLimbsSlice,
        MpMutNativeEndianUIntLimbsSlice,
    >();
}

// Add a limb to a multiprecision integer.
pub fn ct_add_mp_l<T0: MpMutUInt>(op0: &mut T0, op1: LimbType) -> LimbType {
    debug_assert!(ct_find_last_set_byte_l(op1) <= op0.len());
    let op0_nlimbs = op0.nlimbs();
    if op0_nlimbs == 0 {
        return 0;
    }

    // Set carry to op1 and propagate the carry upwards.
    let mut carry = op1;
    for i in 0..op0_nlimbs - 1 {
        let mut op0_val = op0.load_l_full(i);
        (carry, op0_val) = ct_add_l_l(op0_val, carry);
        op0.store_l_full(i, op0_val);
    }

    let mut op0_val = op0.load_l(op0_nlimbs - 1);
    (carry, op0_val) = ct_add_l_l(op0_val, carry);
    let op0_high_mask = op0.partial_high_mask();
    let op0_high_shift = op0.partial_high_shift();
    op0.store_l(op0_nlimbs - 1, op0_val & op0_high_mask);
    let carry_in_op0_high = (op0_val & !op0_high_mask) >> op0_high_shift;
    debug_assert!(carry == 0 || carry_in_op0_high == 0);
    carry |= carry_in_op0_high;

    carry
}

/// Conditionally subtract two multiprecision integers of specified endianess.
///
/// The first operand's contents will be replaced by the resulting difference if
/// `cond` is set and the borrow, if any, returned from the function.
///
/// Runs in constant time for a given configuration of input operand widths,
/// i.e. execution time depends only on the integers' widths, but not their
/// values and neither on `cond`.
///
/// # Arguments:
///
/// * `op0` - The minuend. It will be overwritten by the resulting difference if
///   `cond` is set. The slice length must greater or equal than the length of
///   the second operand.
/// * `op1` - The subtrahend. Its length must not exceed the length of `op0`.
/// * `cond` - Whether or not to replace `ob0` by the difference.
pub fn ct_sub_cond_mp_mp<T0: MpMutUInt, T1: MpUIntCommon>(
    op0: &mut T0,
    op1: &T1,
    cond: LimbChoice,
) -> LimbType {
    let op0_nlimbs = op0.nlimbs();
    let op1_nlimbs = op1.nlimbs();
    debug_assert!(op1_nlimbs <= op0_nlimbs);
    if op1_nlimbs == 0 {
        return 0;
    }

    let mut borrow = 0;
    for i in 0..op1_nlimbs - 1 {
        let mut op0_val = op0.load_l_full(i);
        let op1_val = cond.select(0, op1.load_l_full(i));
        (borrow, op0_val) = ct_sub_l_l_b(op0_val, op1_val, borrow);
        op0.store_l_full(i, op0_val);
    }

    // Propagate the borrow upwards. The first iteration will also account
    // for op1's high limb.
    let mut op1_val = cond.select(0, op1.load_l(op1_nlimbs - 1));
    for i in op1_nlimbs - 1..op0_nlimbs {
        let mut op0_val = op0.load_l(i);
        (borrow, op0_val) = ct_sub_l_l_b(op0_val, op1_val, borrow);
        op1_val = 0;
        if i != op0_nlimbs - 1 || !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
            op0.store_l_full(i, op0_val);
        } else {
            let op0_high_mask = op0.partial_high_mask();
            op0.store_l(i, op0_val & op0_high_mask);
            debug_assert!(op0.partial_high_shift() == 0 || borrow == op0_val >> (LIMB_BITS - 1));
        }
    }
    borrow
}

#[cfg(test)]
fn test_ct_sub_cond_mp_mp<T0: MpMutUIntSlice, T1: MpMutUIntSlice>() {
    use super::limb::LIMB_BYTES;

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !0);
    op1.store_l(0, !0);
    let borrow = ct_sub_cond_mp_mp(&mut op0, &op1, LimbChoice::from(0));
    assert_eq!(borrow, 0);
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), 0);
    let borrow = ct_sub_cond_mp_mp(&mut op0, &op1, LimbChoice::from(1));
    assert_eq!(borrow, 0);
    assert_eq!(op0.load_l(0), 0);
    assert_eq!(op0.load_l(1), 0);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !1);
    op0.store_l(1, 1);
    op1.store_l(0, !0);
    let borrow = ct_sub_cond_mp_mp(&mut op0, &op1, LimbChoice::from(0));
    assert_eq!(borrow, 0);
    assert_eq!(op0.load_l(0), !1);
    assert_eq!(op0.load_l(1), 1);
    let borrow = ct_sub_cond_mp_mp(&mut op0, &op1, LimbChoice::from(1));
    assert_eq!(borrow, 0);
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), 0);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 1 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !1);
    op0.store_l(1, 1);
    op1.store_l(0, !0);
    let borrow = ct_sub_cond_mp_mp(&mut op0, &op1, LimbChoice::from(0));
    assert_eq!(borrow, 0);
    assert_eq!(op0.load_l(0), !1);
    assert_eq!(op0.load_l(1), 1);
    let borrow = ct_sub_cond_mp_mp(&mut op0, &op1, LimbChoice::from(1));
    assert_eq!(borrow, 0);
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), 0);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !1);
    op0.store_l(1, 0);
    op1.store_l(0, !0);
    op1.store_l(1, 1);
    let borrow = ct_sub_cond_mp_mp(&mut op0, &op1, LimbChoice::from(0));
    assert_eq!(borrow, 0);
    assert_eq!(op0.load_l(0), !1);
    assert_eq!(op0.load_l(1), 0);
    let borrow = ct_sub_cond_mp_mp(&mut op0, &op1, LimbChoice::from(1));
    assert_eq!(borrow, 1);
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), !1);

    if !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS || !T1::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
        return;
    }

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES - 1);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES - 1);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !0);
    op1.store_l(0, !0);
    let borrow = ct_sub_cond_mp_mp(&mut op0, &op1, LimbChoice::from(0));
    assert_eq!(borrow, 0);
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), 0);
    let borrow = ct_sub_cond_mp_mp(&mut op0, &op1, LimbChoice::from(1));
    assert_eq!(borrow, 0);
    assert_eq!(op0.load_l(0), 0);
    assert_eq!(op0.load_l(1), 0);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES - 1);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES - 1);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !1);
    op0.store_l(1, 1);
    op1.store_l(0, !0);
    let borrow = ct_sub_cond_mp_mp(&mut op0, &op1, LimbChoice::from(0));
    assert_eq!(borrow, 0);
    assert_eq!(op0.load_l(0), !1);
    assert_eq!(op0.load_l(1), 1);
    let borrow = ct_sub_cond_mp_mp(&mut op0, &op1, LimbChoice::from(1));
    assert_eq!(borrow, 0);
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), 0);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES - 1);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES - 1);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, !1);
    op0.store_l(1, 0);
    op1.store_l(0, !0);
    op1.store_l(1, 1);
    let borrow = ct_sub_cond_mp_mp(&mut op0, &op1, LimbChoice::from(0));
    assert_eq!(borrow, 0);
    assert_eq!(op0.load_l(0), !1);
    assert_eq!(op0.load_l(1), 0);
    let borrow = ct_sub_cond_mp_mp(&mut op0, &op1, LimbChoice::from(1));
    assert_eq!(borrow, 1);
    assert_eq!(op0.load_l(0), !0);
    assert_eq!(op0.load_l(1), !0 >> 8 & !1);
}

#[test]
fn test_ct_sub_cond_be_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_sub_cond_mp_mp::<MpMutBigEndianUIntByteSlice, MpMutBigEndianUIntByteSlice>()
}

#[test]
fn test_ct_sub_cond_le_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_sub_cond_mp_mp::<MpMutLittleEndianUIntByteSlice, MpMutLittleEndianUIntByteSlice>()
}

#[test]
fn test_ct_sub_cond_ne_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_sub_cond_mp_mp::<MpMutNativeEndianUIntLimbsSlice, MpMutNativeEndianUIntLimbsSlice>()
}

pub fn ct_sub_mp_mp<T0: MpMutUInt, T1: MpUIntCommon>(op0: &mut T0, op1: &T1) -> LimbType {
    ct_sub_cond_mp_mp(op0, op1, LimbChoice::from(1))
}

pub type CtSubModMpMpError = CtAddModMpMpError;

pub fn ct_sub_mod_mp_mp<T0: MpMutUInt, T1: MpUIntCommon, NT: MpUIntCommon>(
    op0: &mut T0,
    op1: &T1,
    n: &NT,
) -> Result<(), CtSubModMpMpError> {
    if !n.len_is_compatible_with(op0.len()) {
        return Err(CtSubModMpMpError::InconsistentOperandLengths);
    }
    debug_assert!(op0.nlimbs() >= n.nlimbs());

    debug_assert_ne!(ct_lt_mp_mp(op0, n).unwrap(), 0);
    debug_assert_ne!(ct_lt_mp_mp(op1, n).unwrap(), 0);
    debug_assert!(!n.is_empty());

    let mut borrow = 0;
    let op1_n_common_nlimbs = op1.nlimbs().min(n.nlimbs());
    for i in 0..op1_n_common_nlimbs {
        let result_val;
        (borrow, result_val) = ct_sub_l_l_b(op0.load_l(i), op1.load_l(i), borrow);
        if !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS || i != op0.nlimbs() - 1 {
            op0.store_l_full(i, result_val);
        } else {
            op0.store_l(i, result_val & op0.partial_high_mask());
        }
    }
    // Propagate the borrow upwards.
    for i in op1_n_common_nlimbs..n.nlimbs() {
        let result_val;
        (borrow, result_val) = ct_sub_l_l(op0.load_l(i), borrow);
        if !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS || i != op0.nlimbs() - 1 {
            op0.store_l_full(i, result_val);
        } else {
            op0.store_l(i, result_val & op0.partial_high_mask());
        }
    }

    // Add an n back if negative to bring the result back into range.
    ct_add_cond_mp_mp(op0, n, LimbChoice::from(borrow));

    Ok(())
}

#[cfg(test)]
fn test_ct_sub_mod_mp_mp<T0: MpMutUIntSlice, T1: MpMutUIntSlice, NT: MpMutUIntSlice>() {
    use super::cmp_impl::ct_eq_mp_mp;
    use super::limb::LIMB_BYTES;
    use crate::limbs_buffer::MpUIntCommonPriv;

    let mut n = tst_mk_mp_backing_vec!(NT, LIMB_BYTES + 1);
    let mut n = NT::from_slice(&mut n).unwrap();
    for i in 0..n.nlimbs() {
        if i != n.nlimbs() - 1 {
            n.store_l_full(i, !0);
        } else {
            n.store_l(i, n.partial_high_mask());
        }
    }
    let op0_max_len = T0::n_backing_elements_for_len(LIMB_BYTES + 1) * T0::BACKING_ELEMENT_SIZE;
    n.clear_bytes_above(op0_max_len);

    for op0_len in 1..n.len() + 1 {
        let mut op0 = tst_mk_mp_backing_vec!(T0, n.len());
        let mut op0 = T0::from_slice(&mut op0).unwrap();
        for op1_len in 1..(op0_len + 1).min(n.len()) + 1 {
            let mut op1 = tst_mk_mp_backing_vec!(T1, op1_len);
            let mut op1 = T1::from_slice(&mut op1).unwrap();
            for i in 0..op0.nlimbs() {
                if i != op0.nlimbs() - 1 {
                    op0.store_l_full(i, !0);
                } else {
                    op0.store_l(i, op0.partial_high_mask());
                }
            }
            op0.clear_bytes_above(op0_len);
            ct_sub_mp_l(&mut op0, 1);
            debug_assert_ne!(ct_lt_mp_mp(&op0, &n).unwrap(), 0);

            for i in 0..op1.nlimbs() {
                if i != op1.nlimbs() - 1 {
                    op1.store_l_full(i, !0);
                } else {
                    op1.store_l(i, op1.partial_high_mask());
                }
            }
            op1.clear_bytes_above(op1_len);
            ct_sub_mp_l(&mut op1, 1);
            debug_assert_ne!(ct_lt_mp_mp(&op1, &n).unwrap(), 0);

            let mut expected = tst_mk_mp_backing_vec!(T0, n.len() + 1);
            let mut expected = T0::from_slice(&mut expected).unwrap();
            expected.copy_from(&op0);
            let borrow = ct_sub_mp_mp(&mut expected, &op1);
            if borrow != 0 {
                ct_add_mp_mp(&mut expected, &n);
            }

            ct_sub_mod_mp_mp(&mut op0, &op1, &n).unwrap();
            debug_assert_ne!(ct_lt_mp_mp(&op0, &n).unwrap(), 0);

            assert_ne!(ct_eq_mp_mp(&expected, &op0).unwrap(), 0);
        }
    }
}

#[test]
fn test_ct_sub_mod_be_be_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_sub_mod_mp_mp::<
        MpMutBigEndianUIntByteSlice,
        MpMutBigEndianUIntByteSlice,
        MpMutBigEndianUIntByteSlice,
    >();
}

#[test]
fn test_ct_sub_mod_le_le_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_sub_mod_mp_mp::<
        MpMutLittleEndianUIntByteSlice,
        MpMutLittleEndianUIntByteSlice,
        MpMutLittleEndianUIntByteSlice,
    >();
}

#[test]
fn test_ct_sub_mod_ne_ne_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_sub_mod_mp_mp::<
        MpMutNativeEndianUIntLimbsSlice,
        MpMutNativeEndianUIntLimbsSlice,
        MpMutNativeEndianUIntLimbsSlice,
    >();
}

// Subtract a limb from a multiprecision integer.
pub fn ct_sub_mp_l<T0: MpMutUInt>(op0: &mut T0, op1: LimbType) -> LimbType {
    debug_assert!(ct_find_last_set_byte_l(op1) <= op0.len());
    let op0_nlimbs = op0.nlimbs();
    if op0_nlimbs == 0 {
        return 0;
    }

    // Set borrow to op1 and propagate the borrow upwards.
    let mut borrow = op1;
    for i in 0..op0_nlimbs - 1 {
        let mut op0_val = op0.load_l_full(i);
        (borrow, op0_val) = ct_sub_l_l(op0_val, borrow);
        op0.store_l_full(i, op0_val);
    }

    let mut op0_val = op0.load_l(op0_nlimbs - 1);
    (borrow, op0_val) = ct_sub_l_l(op0_val, borrow);
    let op0_high_mask = op0.partial_high_mask();
    op0.store_l(op0_nlimbs - 1, op0_val & op0_high_mask);
    debug_assert!(op0.partial_high_shift() == 0 || borrow == op0_val >> (LIMB_BITS - 1));

    borrow
}

pub fn ct_negate_cond_mp<T0: MpMutUInt>(op0: &mut T0, cond: LimbChoice) {
    if op0.is_empty() {
        return;
    }

    let nlimbs = op0.nlimbs();
    let negate_mask = cond.select(0, !0);
    let mut negate_carry = negate_mask & 1;
    for i in 0..nlimbs - 1 {
        let mut op0_val = op0.load_l(i);
        op0_val ^= negate_mask;
        (negate_carry, op0_val) = ct_add_l_l(op0_val, negate_carry);
        op0.store_l(i, op0_val);
    }

    let mut op0_val = op0.load_l(nlimbs - 1);
    op0_val ^= negate_mask;
    (_, op0_val) = ct_add_l_l(op0_val, negate_carry);
    op0.store_l(nlimbs - 1, op0_val & op0.partial_high_mask());
}

#[cfg(test)]
fn test_ct_negate_cond_mp<T0: MpMutUIntSlice>() {
    use super::cmp_impl::ct_is_one_mp;
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::MpUIntCommonPriv as _;

    let op0_len = 2 * LIMB_BYTES - 1;

    let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    ct_negate_cond_mp(&mut op0, LimbChoice::from(0));
    assert_ne!(ct_is_zero_mp(&op0).unwrap(), 0);
    ct_negate_cond_mp(&mut op0, LimbChoice::from(1));
    assert_ne!(ct_is_zero_mp(&op0).unwrap(), 0);

    let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    // A one.
    op0.store_l(0, 1);
    ct_negate_cond_mp(&mut op0, LimbChoice::from(0));
    assert_ne!(ct_is_one_mp(&op0).unwrap(), 0);

    // Make it a minus one.
    ct_negate_cond_mp(&mut op0, LimbChoice::from(1));
    for i in 0..op0.nlimbs() - 1 {
        assert_eq!(op0.load_l(i), !0);
    }
    assert_eq!(op0.load_l(op0.nlimbs() - 1), op0.partial_high_mask());

    ct_negate_cond_mp(&mut op0, LimbChoice::from(0));
    for i in 0..op0.nlimbs() - 1 {
        assert_eq!(op0.load_l(i), !0);
    }
    assert_eq!(op0.load_l(op0.nlimbs() - 1), op0.partial_high_mask());

    // And negate again to obtain positive one.
    ct_negate_cond_mp(&mut op0, LimbChoice::from(1));
    assert_ne!(ct_is_one_mp(&op0).unwrap(), 0);
}

#[test]
fn test_ct_negate_cond_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_negate_cond_mp::<MpMutBigEndianUIntByteSlice>()
}

#[test]
fn test_ct_negate_cond_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_negate_cond_mp::<MpMutLittleEndianUIntByteSlice>()
}

#[test]
fn test_ct_negate_cond_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_negate_cond_mp::<MpMutNativeEndianUIntLimbsSlice>()
}

pub fn ct_negate_mp<T0: MpMutUInt>(op0: &mut T0) {
    ct_negate_cond_mp(op0, LimbChoice::from(1))
}

pub type CtNegateModMpError = CtAddModMpMpError;

pub fn ct_negate_mod_mp<T0: MpMutUInt, NT: MpUIntCommon>(
    op0: &mut T0,
    n: &NT,
) -> Result<(), CtNegateModMpError> {
    if !n.len_is_compatible_with(op0.len()) {
        return Err(CtNegateModMpError::InconsistentOperandLengths);
    }
    debug_assert_ne!(ct_lt_mp_mp(op0, n).unwrap(), 0);

    let op0_is_nonzero = !ct_is_zero_mp(op0);
    ct_negate_cond_mp(op0, op0_is_nonzero);
    ct_add_cond_mp_mp(op0, n, op0_is_nonzero);
    Ok(())
}

#[cfg(test)]
fn test_ct_negate_mod_mp<T0: MpMutUIntSlice, NT: MpMutUIntSlice>() {
    use super::limb::LIMB_BYTES;
    use crate::limbs_buffer::MpUIntCommonPriv;
    use crate::shift_impl::ct_rshift_mp;

    let n_max_len = T0::n_backing_elements_for_len(LIMB_BYTES + 1) * NT::BACKING_ELEMENT_SIZE;
    for n_len in 1..n_max_len {
        let mut n = tst_mk_mp_backing_vec!(NT, n_len);
        let mut n = NT::from_slice(&mut n).unwrap();
        let mut n_limb_val = 0;
        for _ in 0..LIMB_BYTES {
            n_limb_val <<= 8;
            n_limb_val |= 0xcc;
        }
        for i in 0..n.nlimbs() {
            if i != n.nlimbs() - 1 {
                n.store_l_full(i, n_limb_val);
            } else {
                n.store_l(i, n_limb_val & n.partial_high_mask());
            }
        }

        let mut op0 = tst_mk_mp_backing_vec!(T0, n.len());
        let mut op0 = T0::from_slice(&mut op0).unwrap();
        let mut result = tst_mk_mp_backing_vec!(T0, n.len());
        let mut result = T0::from_slice(&mut result).unwrap();

        // Negate a zero.
        ct_negate_mod_mp(&mut result, &n).unwrap();
        assert_ne!(ct_is_zero_mp(&result).unwrap(), 0);

        // Negate a one.
        op0.set_to_u8(1);
        result.copy_from(&op0);
        ct_negate_mod_mp(&mut result, &n).unwrap();
        // Adding the original value should result in a sum of zero.
        ct_add_mod_mp_mp(&mut result, &op0, &n).unwrap();
        assert_ne!(ct_is_zero_mp(&result).unwrap(), 0);

        // Negate n - 1.
        op0.copy_from(&n);
        ct_sub_mp_l(&mut op0, 1);
        result.copy_from(&op0);
        ct_negate_mod_mp(&mut result, &n).unwrap();
        // Adding the original value should result in a sum of zero.
        ct_add_mod_mp_mp(&mut result, &op0, &n).unwrap();
        assert_ne!(ct_is_zero_mp(&result).unwrap(), 0);

        // Negate n / 2.
        op0.copy_from(&n);
        ct_rshift_mp(&mut op0, 1);
        result.copy_from(&op0);
        ct_negate_mod_mp(&mut result, &n).unwrap();
        // Adding the original value should result in a sum of zero.
        ct_add_mod_mp_mp(&mut result, &op0, &n).unwrap();
        assert_ne!(ct_is_zero_mp(&result).unwrap(), 0);
    }
}

#[test]
fn test_ct_negate_mod_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_negate_mod_mp::<MpMutBigEndianUIntByteSlice, MpMutBigEndianUIntByteSlice>()
}

#[test]
fn test_ct_negate_mod_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_negate_mod_mp::<MpMutLittleEndianUIntByteSlice, MpMutLittleEndianUIntByteSlice>()
}

#[test]
fn test_ct_negate_mod_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_negate_mod_mp::<MpMutNativeEndianUIntLimbsSlice, MpMutNativeEndianUIntLimbsSlice>()
}
