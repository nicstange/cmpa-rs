//! Implementation of multiprecision integer addition related primitives.

use super::limb::{
    ct_add_l_l, ct_add_l_l_c, ct_find_last_set_byte_l, ct_sub_l_l, ct_sub_l_l_b, LimbChoice,
    LimbType, LIMB_BITS,
};
use super::limbs_buffer::{MpIntMutSlice, MpIntSliceCommon};

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
pub fn ct_add_cond_mp_mp<T0: MpIntMutSlice, T1: MpIntSliceCommon>(
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
fn test_ct_add_cond_mp_mp<T0: MpIntMutSlice, T1: MpIntMutSlice>() {
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
    use super::limbs_buffer::MpBigEndianMutByteSlice;
    test_ct_add_cond_mp_mp::<MpBigEndianMutByteSlice, MpBigEndianMutByteSlice>()
}

#[test]
fn test_ct_add_cond_le_le() {
    use super::limbs_buffer::MpLittleEndianMutByteSlice;
    test_ct_add_cond_mp_mp::<MpLittleEndianMutByteSlice, MpLittleEndianMutByteSlice>()
}

#[test]
fn test_ct_add_cond_ne_ne() {
    use super::limbs_buffer::MpNativeEndianMutByteSlice;
    test_ct_add_cond_mp_mp::<MpNativeEndianMutByteSlice, MpNativeEndianMutByteSlice>()
}

pub fn ct_add_mp_mp<T0: MpIntMutSlice, T1: MpIntSliceCommon>(op0: &mut T0, op1: &T1) -> LimbType {
    ct_add_cond_mp_mp(op0, op1, LimbChoice::from(1))
}

// Add a limb to a multiprecision integer.
pub fn ct_add_mp_l<T0: MpIntMutSlice>(op0: &mut T0, op1: LimbType) -> LimbType {
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
pub fn ct_sub_cond_mp_mp<T0: MpIntMutSlice, T1: MpIntSliceCommon>(
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
fn test_ct_sub_cond_mp_mp<T0: MpIntMutSlice, T1: MpIntMutSlice>() {
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
    use super::limbs_buffer::MpBigEndianMutByteSlice;
    test_ct_sub_cond_mp_mp::<MpBigEndianMutByteSlice, MpBigEndianMutByteSlice>()
}

#[test]
fn test_ct_sub_cond_le_le() {
    use super::limbs_buffer::MpLittleEndianMutByteSlice;
    test_ct_sub_cond_mp_mp::<MpLittleEndianMutByteSlice, MpLittleEndianMutByteSlice>()
}

#[test]
fn test_ct_sub_cond_ne_ne() {
    use super::limbs_buffer::MpNativeEndianMutByteSlice;
    test_ct_sub_cond_mp_mp::<MpNativeEndianMutByteSlice, MpNativeEndianMutByteSlice>()
}

pub fn ct_sub_mp_mp<T0: MpIntMutSlice, T1: MpIntSliceCommon>(op0: &mut T0, op1: &T1) -> LimbType {
    ct_sub_cond_mp_mp(op0, op1, LimbChoice::from(1))
}

// Subtract a limb from a multiprecision integer.
pub fn ct_sub_mp_l<T0: MpIntMutSlice>(op0: &mut T0, op1: LimbType) -> LimbType {
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

pub fn ct_negate_cond_mp<T0: MpIntMutSlice>(op0: &mut T0, cond: LimbChoice) {
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
fn test_ct_negate_cond_mp<T0: MpIntMutSlice>() {
    use super::cmp_impl::{ct_is_one_mp, ct_is_zero_mp};
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::MpIntSliceCommonPriv as _;

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
    use super::limbs_buffer::MpBigEndianMutByteSlice;
    test_ct_negate_cond_mp::<MpBigEndianMutByteSlice>()
}

#[test]
fn test_ct_negate_cond_le() {
    use super::limbs_buffer::MpLittleEndianMutByteSlice;
    test_ct_negate_cond_mp::<MpLittleEndianMutByteSlice>()
}

#[test]
fn test_ct_negate_cond_ne() {
    use super::limbs_buffer::MpNativeEndianMutByteSlice;
    test_ct_negate_cond_mp::<MpNativeEndianMutByteSlice>()
}
