//! Implementation of multiprecision integer comparison primitives.

use super::limb::{ct_eq_l_l, ct_neq_l_l, ct_lt_l_l};
use super::limbs_buffer::{mp_ct_nlimbs, MPEndianess, MPBigEndianOrder, MPLittleEndianOrder};
use subtle;

/// Compare two multiprecision integers of specified endianess for `==`.
///
/// Evaluates to `true` iff `op0` is equal to `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
///           Its endianess is specified by the `E0` generic parameter.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
///           Its endianess is specified by the `E1` generic parameter.
///
pub fn mp_ct_eq_mp_mp<E0: MPEndianess, E1: MPEndianess>(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    let op0_nlimbs = mp_ct_nlimbs(op0.len());
    let op1_nlimbs = mp_ct_nlimbs(op1.len());
    let common_nlimbs = op0_nlimbs.min(op1_nlimbs);

    let mut is_eq = subtle::Choice::from(1);
    for i in 0..common_nlimbs {
        let op0_val = E0::load_l(&op0, i);
        let op1_val = E1::load_l(&op1, i);
        is_eq &= ct_eq_l_l(op0_val, op1_val);
    }

    for i in common_nlimbs..op0_nlimbs {
        let op0_val = E0::load_l(&op0, i);
        is_eq &= ct_eq_l_l(op0_val, 0);
    }

    for i in common_nlimbs..op1_nlimbs {
        let op1_val = E1::load_l(&op1, i);
        is_eq &= ct_eq_l_l(op1_val, 0);
    }

    is_eq
}

/// Compare two big-endian multprecision integers for `==`.
///
/// This is a convenience specialization of the generic [`mp_ct_eq_mp_mp()`] for the big-endian
/// case.
///
pub fn mp_ct_eq_be_be(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    mp_ct_eq_mp_mp::<MPBigEndianOrder, MPBigEndianOrder>(op0, op1)
}

/// Compare two little-endian multprecision integers for `==`.
///
/// This is a convenience specialization of the generic [`mp_ct_eq_mp_mp()`] for the little-endian
/// case.
///
pub fn mp_ct_eq_le_le(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    mp_ct_eq_mp_mp::<MPLittleEndianOrder, MPLittleEndianOrder>(op0, op1)
}

#[cfg(test)]
fn test_mp_ct_eq_mp_mp<E0: MPEndianess, E1: MPEndianess>() {
    use super::limb::LIMB_BYTES;

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    E0::store_l(&mut op0, 0, 1);
    E1::store_l(&mut op1, 0, 1);
    assert_eq!(mp_ct_eq_mp_mp::<E0, E1>(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_eq_mp_mp::<E0, E1>(E0::split_at(&op0, LIMB_BYTES).1, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_eq_mp_mp::<E0, E1>(&op0, E1::split_at(&op1, LIMB_BYTES).1).unwrap_u8(), 1);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    E0::store_l(&mut op0, 0, 1);
    E0::store_l(&mut op0, 1, 2);
    E1::store_l(&mut op1, 0, 1);
    E1::store_l(&mut op1, 1, 2);
    assert_eq!(mp_ct_eq_mp_mp::<E0, E1>(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_eq_mp_mp::<E0, E1>(E0::split_at(&op0, LIMB_BYTES).1, &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_eq_mp_mp::<E0, E1>(&op0, E1::split_at(&op1, LIMB_BYTES).1).unwrap_u8(), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    E0::store_l(&mut op0, 0, 1);
    E1::store_l(&mut op1, 0, 2);
    assert_eq!(mp_ct_eq_mp_mp::<E0, E1>(&op0, &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_eq_mp_mp::<E0, E1>(E0::split_at(&op0, LIMB_BYTES).1, &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_eq_mp_mp::<E0, E1>(&op0, E1::split_at(&op1, LIMB_BYTES).1).unwrap_u8(), 0);
}

#[test]
fn test_mp_ct_eq_be_be() {
    test_mp_ct_eq_mp_mp::<MPBigEndianOrder, MPBigEndianOrder>()
}

#[test]
fn test_mp_ct_eq_le_le() {
    test_mp_ct_eq_mp_mp::<MPLittleEndianOrder, MPLittleEndianOrder>()
}

/// Compare two multiprecision integers of specified endianess for `!=`.
///
/// Evaluates to `true` iff `op0` is not equal to `op1` in value .
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer big-endian byte buffer.
///           Its endianess is specified by the `E0` generic parameter.
/// * `op1` - The second operand as a multiprecision integer big-endian byte buffer.
///           Its endianess is specified by the `E1` generic parameter.
///
pub fn mp_ct_neq_mp_mp<E0: MPEndianess, E1: MPEndianess>(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    !mp_ct_eq_mp_mp::<E0, E1>(op0, op1)
}

/// Compare two big-endian multprecision integers for `!=`.
///
/// This is a convenience specialization of the generic [`mp_ct_neq_mp_mp()`] for the big-endian
/// case.
///
pub fn mp_ct_neq_be_be(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    mp_ct_neq_mp_mp::<MPBigEndianOrder, MPBigEndianOrder>(op0, op1)
}

/// Compare two little-endian multprecision integers for `!=`.
///
/// This is a convenience specialization of the generic [`mp_ct_neq_mp_mp()`] for the little-endian
/// case.
///
pub fn mp_ct_neq_le_le(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    mp_ct_neq_mp_mp::<MPLittleEndianOrder, MPLittleEndianOrder>(op0, op1)
}

/// Compare two multiprecision integers of specified endianess for `<=`.
///
/// Evaluates to `true` iff `op0` is less or equal than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer big-endian byte buffer.
///           Its endianess is specified by the `E0` generic parameter.
/// * `op1` - The second operand as a multiprecision integer big-endian byte buffer.
///           Its endianess is specified by the `E1` generic parameter.
///
pub fn mp_ct_leq_mp_mp<E0: MPEndianess, E1: MPEndianess>(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    let op0_nlimbs = mp_ct_nlimbs(op0.len());
    let op1_nlimbs = mp_ct_nlimbs(op1.len());
    let common_nlimbs = op0_nlimbs.min(op1_nlimbs);

    let mut is_eq = subtle::Choice::from(1);
    for i in common_nlimbs..op0_nlimbs {
        let op0_val = E0::load_l(&op0, i);
        is_eq &= ct_eq_l_l(op0_val, 0);
    }

    let mut is_lt = subtle::Choice::from(0);
    for i in common_nlimbs..op1_nlimbs {
        let op1_val = E1::load_l(&op1, i);
        is_lt |= ct_neq_l_l(op1_val, 0);
    }

    let mut i = common_nlimbs;
    while i > 0 {
        i -= 1;
        let op0_val = E0::load_l(&op0, i);
        let op1_val = E1::load_l(&op1, i);
        is_lt |= is_eq & ct_lt_l_l(op0_val, op1_val);
        is_eq &= ct_eq_l_l(op0_val, op1_val);
    }

    is_lt | is_eq
}

/// Compare two big-endian multprecision integers for `<=`.
///
/// This is a convenience specialization of the generic [`mp_ct_leq_mp_mp()`] for the big-endian
/// case.
///
pub fn mp_ct_leq_be_be(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    mp_ct_leq_mp_mp::<MPBigEndianOrder, MPBigEndianOrder>(op0, op1)
}

/// Compare two little-endian multprecision integers for `<=`.
///
/// This is a convenience specialization of the generic [`mp_ct_leq_mp_mp()`] for the little-endian
/// case.
///
pub fn mp_ct_leq_le_le(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    mp_ct_leq_mp_mp::<MPLittleEndianOrder, MPLittleEndianOrder>(op0, op1)
}

#[cfg(test)]
fn test_mp_ct_leq_mp_mp<E0: MPEndianess, E1: MPEndianess>() {
    use super::limb::LIMB_BYTES;

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    // [0 1]
    E0::store_l(&mut op0, 0, 1);
    // [0 1]
    E1::store_l(&mut op1, 0, 1);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(E0::split_at(&op0, LIMB_BYTES).1, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(&op0, E1::split_at(&op1, LIMB_BYTES).1).unwrap_u8(), 1);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    // [0 1]
    E0::store_l(&mut op0, 0, 1);
    // [0 2]
    E1::store_l(&mut op1, 0, 2);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(E0::split_at(&op0, LIMB_BYTES).1, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(&op0, E1::split_at(&op1, LIMB_BYTES).1).unwrap_u8(), 1);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    // [0 1]
    E0::store_l(&mut op0, 0, 1);
    // [1 0]
    E1::store_l(&mut op1, 1, 1);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(E0::split_at(&op0, LIMB_BYTES).1, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(&op0, E1::split_at(&op1, LIMB_BYTES).1).unwrap_u8(), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    // [1 0]
    E0::store_l(&mut op0, 1, 1);
    // [0 1]
    E1::store_l(&mut op1, 0, 1);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(&op0, &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(E0::split_at(&op0, LIMB_BYTES).1, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(&op0, E1::split_at(&op1, LIMB_BYTES).1).unwrap_u8(), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    // [1 1]
    E0::store_l(&mut op0, 0, 1);
    E0::store_l(&mut op0, 1, 1);
    // [0 1]
    E1::store_l(&mut op1, 0, 1);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(&op0, &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(E0::split_at(&op0, LIMB_BYTES).1, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_leq_mp_mp::<E0, E1>(&op0, E1::split_at(&op1, LIMB_BYTES).1).unwrap_u8(), 0);
}

#[test]
fn test_mp_ct_leq_be_be() {
    test_mp_ct_leq_mp_mp::<MPBigEndianOrder, MPBigEndianOrder>()
}

#[test]
fn test_mp_ct_leq_le_le() {
    test_mp_ct_leq_mp_mp::<MPLittleEndianOrder, MPLittleEndianOrder>()
}

/// Compare two multiprecision integers of specified endianess for `<`.
///
/// Evaluates to `true` iff `op0` is less than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer big-endian byte buffer.
///           Its endianess is specified by the `E0` generic parameter.
/// * `op1` - The second operand as a multiprecision integer big-endian byte buffer.
///           Its endianess is specified by the `E1` generic parameter.
///
pub fn mp_ct_lt_mp_mp<E0: MPEndianess, E1: MPEndianess>(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    !mp_ct_geq_mp_mp::<E0, E1>(op0, op1)
}

/// Compare two big-endian multprecision integers for `<`.
///
/// This is a convenience specialization of the generic [`mp_ct_lt_mp_mp()`] for the big-endian
/// case.
///
pub fn mp_ct_lt_be_be(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    mp_ct_lt_mp_mp::<MPBigEndianOrder, MPBigEndianOrder>(op0, op1)
}

/// Compare two little-endian multprecision integers for `<`.
///
/// This is a convenience specialization of the generic [`mp_ct_lt_mp_mp()`] for the little-endian
/// case.
///
pub fn mp_ct_lt_le_le(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    mp_ct_lt_mp_mp::<MPLittleEndianOrder, MPLittleEndianOrder>(op0, op1)
}


/// Compare two multiprecision integers of specified_endianess for `>=`.
///
/// Evaluates to `true` iff `op0` is greater or equal than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer big-endian byte buffer.
///           Its endianess is specified by the `E0` generic parameter.
/// * `op1` - The second operand as a multiprecision integer big-endian byte buffer.
///           Its endianess is specified by the `E1` generic parameter.
///
pub fn mp_ct_geq_mp_mp<E0: MPEndianess, E1: MPEndianess>(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    mp_ct_leq_mp_mp::<E1, E0>(op1, op0)
}

/// Compare two big-endian multprecision integers for `>=`.
///
/// This is a convenience specialization of the generic [`mp_ct_geq_mp_mp()`] for the big-endian
/// case.
///
pub fn mp_ct_geq_be_be(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    mp_ct_geq_mp_mp::<MPBigEndianOrder, MPBigEndianOrder>(op0, op1)
}

/// Compare two little-endian multprecision integers for `>=`.
///
/// This is a convenience specialization of the generic [`mp_ct_geq_mp_mp()`] for the little-endian
/// case.
///
pub fn mp_ct_geq_le_le(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    mp_ct_geq_mp_mp::<MPLittleEndianOrder, MPLittleEndianOrder>(op0, op1)
}


/// Compare two multiprecision integers of specified endianess for `>`.
///
/// Evaluates to `true` iff `op0` is greater than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer big-endian byte buffer.
///           Its endianess is specified by the `E0` generic parameter.
/// * `op1` - The second operand as a multiprecision integer big-endian byte buffer.
///           Its endianess is specified by the `E1` generic parameter.
///
pub fn mp_ct_gt_mp_mp<E0: MPEndianess, E1: MPEndianess>(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    !mp_ct_leq_mp_mp::<E0, E1>(op0, op1)
}

/// Compare two big-endian multprecision integers for `>`.
///
/// This is a convenience specialization of the generic [`mp_ct_gt_mp_mp()`] for the big-endian
/// case.
///
pub fn mp_ct_gt_be_be(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    mp_ct_gt_mp_mp::<MPBigEndianOrder, MPBigEndianOrder>(op0, op1)
}

/// Compare two little-endian multprecision integers for `>`.
///
/// This is a convenience specialization of the generic [`mp_ct_gt_mp_mp()`] for the little-endian
/// case.
///
pub fn mp_ct_gt_le_le(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    mp_ct_gt_mp_mp::<MPLittleEndianOrder, MPLittleEndianOrder>(op0, op1)
}
