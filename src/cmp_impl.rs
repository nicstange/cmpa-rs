//! Implementation of multiprecision integer comparison primitives.

use clap::builder::BoolValueParser;

use super::limb::{LimbChoice, ct_eq_l_l, ct_neq_l_l, ct_lt_l_l, ct_is_zero_l, ct_is_nonzero_l, ct_sub_l_l, black_box_l, ct_sub_l_l_b, LimbType};
use super::limbs_buffer::MPIntByteSliceCommon;
#[cfg(test)]
use super::limbs_buffer::{MPIntMutByteSlice, MPIntMutByteSlicePriv as _};

/// Compare two multiprecision integers of specified endianess for `==`.
///
/// Evaluates to `true` iff `op0` is equal to `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
///
pub fn mp_ct_eq_mp_mp<T0: MPIntByteSliceCommon, T1: MPIntByteSliceCommon>(op0: &T0, op1: &T1) -> LimbChoice {
    let op0_nlimbs = op0.nlimbs();
    let op1_nlimbs = op1.nlimbs();
    let common_nlimbs = op0_nlimbs.min(op1_nlimbs);

    let mut is_neq = 0;
    for i in 0..common_nlimbs {
        let op0_val = op0.load_l(i);
        let op1_val = op1.load_l(i);
        is_neq |= op0_val ^ op1_val;
    }

    for i in common_nlimbs..op0_nlimbs {
        let op0_val = op0.load_l(i);
        is_neq |= op0_val;
    }

    for i in common_nlimbs..op1_nlimbs {
        let op1_val = op1.load_l(i);
        is_neq |= op1_val;
    }

    LimbChoice::from(ct_is_zero_l(is_neq))
}

#[cfg(test)]
fn test_mp_ct_eq_mp_mp<T0: MPIntMutByteSlice, T1: MPIntMutByteSlice>() {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::MPIntMutByteSlice as _;

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = T0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = T1::from_bytes(&mut op1).unwrap();
    op0.store_l(0, 1);
    op1.store_l(0, 1);
    assert_eq!(mp_ct_eq_mp_mp(&op0, &op1).unwrap(), 1);
    assert_eq!(mp_ct_eq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 1);
    assert_eq!(mp_ct_eq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 1);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = T0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = T1::from_bytes(&mut op1).unwrap();
    op0.store_l(0, 1);
    op0.store_l(1, 2);
    op1.store_l(0, 1);
    op1.store_l(1, 2);
    assert_eq!(mp_ct_eq_mp_mp(&op0, &op1).unwrap(), 1);
    assert_eq!(mp_ct_eq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 0);
    assert_eq!(mp_ct_eq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = T0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = T1::from_bytes(&mut op1).unwrap();
    op0.store_l(0, 1);
    op1.store_l(0, 2);
    assert_eq!(mp_ct_eq_mp_mp(&op0, &op1).unwrap(), 0);
    assert_eq!(mp_ct_eq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 0);
    assert_eq!(mp_ct_eq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 0);
}

#[test]
fn test_mp_ct_eq_be_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_eq_mp_mp::<MPBigEndianMutByteSlice, MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_eq_le_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_eq_mp_mp::<MPLittleEndianMutByteSlice, MPLittleEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_eq_ne_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_ct_eq_mp_mp::<MPNativeEndianMutByteSlice, MPNativeEndianMutByteSlice>()
}

/// Compare two multiprecision integers of specified endianess for `!=`.
///
/// Evaluates to `true` iff `op0` is not equal to `op1` in value .
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
///
pub fn mp_ct_neq_mp_mp<T0: MPIntByteSliceCommon, T1: MPIntByteSliceCommon>(op0: &T0, op1: &T1) -> LimbChoice {
    !mp_ct_eq_mp_mp(op0, op1)
}

/// Compare two multiprecision integers of specified endianess for `<=`.
///
/// Evaluates to `true` iff `op0` is less or equal than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
///
pub fn mp_ct_leq_mp_mp<T0: MPIntByteSliceCommon, T1: MPIntByteSliceCommon>(op0: &T0, op1: &T1) -> LimbChoice {
    let op0_nlimbs = op0.nlimbs();
    let op1_nlimbs = op1.nlimbs();
    let common_nlimbs = op0_nlimbs.min(op1_nlimbs);

    let mut is_eq = black_box_l(1);
    for i in common_nlimbs..op0_nlimbs {
        let op0_val = op0.load_l(i);
        is_eq &= ct_is_zero_l(op0_val);
    }

    let mut is_lt = 0 ;
    for i in common_nlimbs..op1_nlimbs {
        let op1_val = op1.load_l(i);
        is_lt |= ct_is_nonzero_l(op1_val);
    }

    let mut i = common_nlimbs;
    while i > 0 {
        i -= 1;
        let op0_val = op0.load_l(i);
        let op1_val = op1.load_l(i);
        let (borrow, diff) = ct_sub_l_l(op0_val, op1_val);

        is_lt |= is_eq & borrow;
        is_eq &= ct_is_zero_l(diff | borrow);
    }

    LimbChoice::from(is_lt | is_eq)
}

#[cfg(test)]
fn test_mp_ct_leq_mp_mp<T0: MPIntMutByteSlice, T1: MPIntMutByteSlice>() {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::MPIntMutByteSlice as _;

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = T0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = T1::from_bytes(&mut op1).unwrap();
    // [0 1]
    op0.store_l(0, 1);
    // [0 1]
    op1.store_l(0, 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1).unwrap(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 1);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = T0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = T1::from_bytes(&mut op1).unwrap();
    // [0 1]
    op0.store_l(0, 1);
    // [0 2]
    op1.store_l(0, 2);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1).unwrap(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 1);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = T0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = T1::from_bytes(&mut op1).unwrap();
    // [0 1]
    op0.store_l(0, 1);
    // [1 0]
    op1.store_l(1, 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1).unwrap(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = T0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = T1::from_bytes(&mut op1).unwrap();
    // [1 0]
    op0.store_l(1, 1);
    // [0 1]
    op1.store_l(0, 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1).unwrap(), 0);
    assert_eq!(mp_ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = T0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = T1::from_bytes(&mut op1).unwrap();
    // [1 1]
    op0.store_l(0, 1);
    op0.store_l(1, 1);
    // [0 1]
    op1.store_l(0, 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1).unwrap(), 0);
    assert_eq!(mp_ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 0);
}

#[test]
fn test_mp_ct_leq_be_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_leq_mp_mp::<MPBigEndianMutByteSlice, MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_leq_le_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_leq_mp_mp::<MPLittleEndianMutByteSlice, MPLittleEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_leq_ne_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_ct_leq_mp_mp::<MPNativeEndianMutByteSlice, MPNativeEndianMutByteSlice>()
}

/// Compare two multiprecision integers of specified endianess for `<`.
///
/// Evaluates to `true` iff `op0` is less than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
///
pub fn mp_ct_lt_mp_mp<T0: MPIntByteSliceCommon, T1: MPIntByteSliceCommon>(op0: &T0, op1: &T1) -> LimbChoice {
    !mp_ct_geq_mp_mp(op0, op1)
}

/// Compare two multiprecision integers of specified_endianess for `>=`.
///
/// Evaluates to `true` iff `op0` is greater or equal than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
///
pub fn mp_ct_geq_mp_mp<T0: MPIntByteSliceCommon, T1: MPIntByteSliceCommon>(op0: &T0, op1: &T1) -> LimbChoice {
    mp_ct_leq_mp_mp(op1, op0)
}

/// Compare two multiprecision integers of specified endianess for `>`.
///
/// Evaluates to `true` iff `op0` is greater than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
///
pub fn mp_ct_gt_mp_mp<T0: MPIntByteSliceCommon, T1: MPIntByteSliceCommon>(op0: &T0, op1: &T1) -> LimbChoice {
    !mp_ct_leq_mp_mp(op0, op1)
}

pub fn mp_ct_is_zero_mp<T0: MPIntByteSliceCommon>(op0: &T0) -> LimbChoice {
    let mut is_nz: LimbType = 0;
    for i in 0..op0.nlimbs() {
        let op0_val = op0.load_l(i);
        is_nz |= op0_val;
    }
    LimbChoice::from(ct_is_zero_l(is_nz))
}

pub fn mp_ct_is_one_mp<T0: MPIntByteSliceCommon>(op0: &T0) -> LimbChoice {
    if op0.is_empty() {
        return LimbChoice::from(0);
    }

    let tail_is_not_one = op0.load_l(0) ^ 1;

    let mut head_is_nz: LimbType = 0;
    for i in 1..op0.nlimbs() {
        let op0_val = op0.load_l(i);
        head_is_nz |= op0_val;
    }

    LimbChoice::from(ct_is_zero_l(tail_is_not_one | head_is_nz))
}


