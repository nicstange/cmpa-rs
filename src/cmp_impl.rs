//! Implementation of multiprecision integer comparison primitives.

use super::limb::{ct_eq_l_l, ct_neq_l_l, ct_lt_l_l};
use super::limbs_buffer::{mp_ct_nlimbs, MPIntByteSlice};
#[cfg(test)]
use super::limbs_buffer::MPIntMutByteSliceFactory;
use subtle;

/// Compare two multiprecision integers of specified endianess for `==`.
///
/// Evaluates to `true` iff `op0` is equal to `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
///
pub fn mp_ct_eq_mp_mp<'a, 'b, T0: MPIntByteSlice<'a>, T1: MPIntByteSlice<'b>>(op0: &T0, op1: &T1) -> subtle::Choice {
    let op0_nlimbs = mp_ct_nlimbs(op0.len());
    let op1_nlimbs = mp_ct_nlimbs(op1.len());
    let common_nlimbs = op0_nlimbs.min(op1_nlimbs);

    let mut is_eq = subtle::Choice::from(1);
    for i in 0..common_nlimbs {
        let op0_val = op0.load_l(i);
        let op1_val = op1.load_l(i);
        is_eq &= ct_eq_l_l(op0_val, op1_val);
    }

    for i in common_nlimbs..op0_nlimbs {
        let op0_val = op0.load_l(i);
        is_eq &= ct_eq_l_l(op0_val, 0);
    }

    for i in common_nlimbs..op1_nlimbs {
        let op1_val = op1.load_l(i);
        is_eq &= ct_eq_l_l(op1_val, 0);
    }

    is_eq
}

#[cfg(test)]
fn test_mp_ct_eq_mp_mp<F0: MPIntMutByteSliceFactory, F1: MPIntMutByteSliceFactory>() {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::MPIntMutByteSlice as _;

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = F0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = F1::from_bytes(&mut op1).unwrap();
    op0.store_l(0, 1);
    op1.store_l(0, 1);
    assert_eq!(mp_ct_eq_mp_mp(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_eq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_eq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap_u8(), 1);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = F0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = F1::from_bytes(&mut op1).unwrap();
    op0.store_l(0, 1);
    op0.store_l(1, 2);
    op1.store_l(0, 1);
    op1.store_l(1, 2);
    assert_eq!(mp_ct_eq_mp_mp(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_eq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_eq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap_u8(), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = F0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = F1::from_bytes(&mut op1).unwrap();
    op0.store_l(0, 1);
    op1.store_l(0, 2);
    assert_eq!(mp_ct_eq_mp_mp(&op0, &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_eq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_eq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap_u8(), 0);
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

/// Compare two multiprecision integers of specified endianess for `!=`.
///
/// Evaluates to `true` iff `op0` is not equal to `op1` in value .
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
///
pub fn mp_ct_neq_mp_mp<'a, 'b, T0: MPIntByteSlice<'a>, T1: MPIntByteSlice<'b>>(op0: &T0, op1: &T1) -> subtle::Choice {
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
pub fn mp_ct_leq_mp_mp<'a, 'b, T0: MPIntByteSlice<'a>, T1: MPIntByteSlice<'b>>(op0: &T0, op1: &T1) -> subtle::Choice {
    let op0_nlimbs = mp_ct_nlimbs(op0.len());
    let op1_nlimbs = mp_ct_nlimbs(op1.len());
    let common_nlimbs = op0_nlimbs.min(op1_nlimbs);

    let mut is_eq = subtle::Choice::from(1);
    for i in common_nlimbs..op0_nlimbs {
        let op0_val = op0.load_l(i);
        is_eq &= ct_eq_l_l(op0_val, 0);
    }

    let mut is_lt = subtle::Choice::from(0);
    for i in common_nlimbs..op1_nlimbs {
        let op1_val = op1.load_l(i);
        is_lt |= ct_neq_l_l(op1_val, 0);
    }

    let mut i = common_nlimbs;
    while i > 0 {
        i -= 1;
        let op0_val = op0.load_l(i);
        let op1_val = op1.load_l(i);
        is_lt |= is_eq & ct_lt_l_l(op0_val, op1_val);
        is_eq &= ct_eq_l_l(op0_val, op1_val);
    }

    is_lt | is_eq
}

#[cfg(test)]
fn test_mp_ct_leq_mp_mp<F0: MPIntMutByteSliceFactory, F1: MPIntMutByteSliceFactory>() {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::MPIntMutByteSlice as _;

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = F0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = F1::from_bytes(&mut op1).unwrap();
    // [0 1]
    op0.store_l(0, 1);
    // [0 1]
    op1.store_l(0, 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap_u8(), 1);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = F0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = F1::from_bytes(&mut op1).unwrap();
    // [0 1]
    op0.store_l(0, 1);
    // [0 2]
    op1.store_l(0, 2);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap_u8(), 1);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = F0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = F1::from_bytes(&mut op1).unwrap();
    // [0 1]
    op0.store_l(0, 1);
    // [1 0]
    op1.store_l(1, 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap_u8(), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = F0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = F1::from_bytes(&mut op1).unwrap();
    // [1 0]
    op0.store_l(1, 1);
    // [0 1]
    op1.store_l(0, 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap_u8(), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = F0::from_bytes(&mut op0).unwrap();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1 = F1::from_bytes(&mut op1).unwrap();
    // [1 1]
    op0.store_l(0, 1);
    op0.store_l(1, 1);
    // [0 1]
    op1.store_l(0, 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap_u8(), 0);
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

/// Compare two multiprecision integers of specified endianess for `<`.
///
/// Evaluates to `true` iff `op0` is less than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
///
pub fn mp_ct_lt_mp_mp<'a, 'b, T0: MPIntByteSlice<'a>, T1: MPIntByteSlice<'b>>(op0: &T0, op1: &T1) -> subtle::Choice {
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
pub fn mp_ct_geq_mp_mp<'a, 'b, T0: MPIntByteSlice<'a>, T1: MPIntByteSlice<'b>>(op0: &T0, op1: &T1) -> subtle::Choice {
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
pub fn mp_ct_gt_mp_mp<'a, 'b, T0: MPIntByteSlice<'a>, T1: MPIntByteSlice<'b>>(op0: &T0, op1: &T1) -> subtle::Choice {
    !mp_ct_leq_mp_mp(op0, op1)
}
