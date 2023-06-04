//! Implementation of multiprecision integer comparison primitives.

use super::limb::{ct_eq_l_l, ct_neq_l_l, ct_lt_l_l};
use super::limbs_buffer::{mp_be_load_l, mp_ct_nlimbs};
use subtle;

/// Compare two multiprecision integers for `==`.
///
/// Evaluates to `true` iff `op0` is equal to `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer big-endian byte buffer.
/// * `op1` - The second operand as a multiprecision integer big-endian byte buffer.
///
fn mp_ct_eq(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    let op0_nlimbs = mp_ct_nlimbs(op0.len());
    let op1_nlimbs = mp_ct_nlimbs(op1.len());
    let common_nlimbs = op0_nlimbs.min(op1_nlimbs);

    let mut is_eq = subtle::Choice::from(1);
    for i in 0..common_nlimbs {
        let op0_val = mp_be_load_l(&op0, i);
        let op1_val = mp_be_load_l(&op1, i);
        is_eq &= ct_eq_l_l(op0_val, op1_val);
    }

    for i in common_nlimbs..op0_nlimbs {
        let op0_val = mp_be_load_l(&op0, i);
        is_eq &= ct_eq_l_l(op0_val, 0);
    }

    for i in common_nlimbs..op1_nlimbs {
        let op1_val = mp_be_load_l(&op1, i);
        is_eq &= ct_eq_l_l(op1_val, 0);
    }

    is_eq
}

#[test]
fn test_mp_ct_eq() {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::mp_be_store_l;

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_be_store_l(&mut op0, 0, 1);
    mp_be_store_l(&mut op1, 0, 1);
    assert_eq!(mp_ct_eq(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_eq(&op0[LIMB_BYTES..], &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_eq(&op0, &op1[LIMB_BYTES..]).unwrap_u8(), 1);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_be_store_l(&mut op0, 0, 1);
    mp_be_store_l(&mut op0, 1, 2);
    mp_be_store_l(&mut op1, 0, 1);
    mp_be_store_l(&mut op1, 1, 2);
    assert_eq!(mp_ct_eq(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_eq(&op0[LIMB_BYTES..], &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_eq(&op0, &op1[LIMB_BYTES..]).unwrap_u8(), 0);


    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_be_store_l(&mut op0, 0, 1);
    mp_be_store_l(&mut op1, 0, 2);
    assert_eq!(mp_ct_eq(&op0, &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_eq(&op0[LIMB_BYTES..], &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_eq(&op0, &op1[LIMB_BYTES..]).unwrap_u8(), 0);
}

/// Compare two multiprecision integers for `!=`.
///
/// Evaluates to `true` iff `op0` is not equal to `op1` in value .
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer big-endian byte buffer.
/// * `op1` - The second operand as a multiprecision integer big-endian byte buffer.
///
fn mp_ct_neq(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    !mp_ct_eq(op0, op1)
}

/// Compare two multiprecision integers for `<=`.
///
/// Evaluates to `true` iff `op0` is less or equal than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer big-endian byte buffer.
/// * `op1` - The second operand as a multiprecision integer big-endian byte buffer.
///
pub fn mp_ct_le(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    let op0_nlimbs = mp_ct_nlimbs(op0.len());
    let op1_nlimbs = mp_ct_nlimbs(op1.len());
    let common_nlimbs = op0_nlimbs.min(op1_nlimbs);

    let mut is_eq = subtle::Choice::from(1);
    for i in common_nlimbs..op0_nlimbs {
        let op0_val = mp_be_load_l(&op0, i);
        is_eq &= ct_eq_l_l(op0_val, 0);
    }

    let mut is_lt = subtle::Choice::from(0);
    for i in common_nlimbs..op1_nlimbs {
        let op1_val = mp_be_load_l(&op1, i);
        is_lt |= ct_neq_l_l(op1_val, 0);
    }

    let mut i = common_nlimbs;
    while i > 0 {
        i -= 1;
        let op0_val = mp_be_load_l(&op0, i);
        let op1_val = mp_be_load_l(&op1, i);
        is_lt |= is_eq & ct_lt_l_l(op0_val, op1_val);
        is_eq &= ct_eq_l_l(op0_val, op1_val);
    }

    is_lt | is_eq
}

#[test]
fn test_mp_ct_le() {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::mp_be_store_l;

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    // [0 1]
    mp_be_store_l(&mut op0, 0, 1);
    // [0 1]
    mp_be_store_l(&mut op1, 0, 1);
    assert_eq!(mp_ct_le(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_le(&op0[LIMB_BYTES..], &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_le(&op0, &op1[LIMB_BYTES..]).unwrap_u8(), 1);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    // [0 1]
    mp_be_store_l(&mut op0, 0, 1);
    // [0 2]
    mp_be_store_l(&mut op1, 0, 2);
    assert_eq!(mp_ct_le(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_le(&op0[LIMB_BYTES..], &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_le(&op0, &op1[LIMB_BYTES..]).unwrap_u8(), 1);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    // [0 1]
    mp_be_store_l(&mut op0, 0, 1);
    // [1 0]
    mp_be_store_l(&mut op1, 1, 1);
    assert_eq!(mp_ct_le(&op0, &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_le(&op0[LIMB_BYTES..], &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_le(&op0, &op1[LIMB_BYTES..]).unwrap_u8(), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    // [1 0]
    mp_be_store_l(&mut op0, 1, 1);
    // [0 1]
    mp_be_store_l(&mut op1, 0, 1);
    assert_eq!(mp_ct_le(&op0, &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_le(&op0[LIMB_BYTES..], &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_le(&op0, &op1[LIMB_BYTES..]).unwrap_u8(), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    // [1 1]
    mp_be_store_l(&mut op0, 0, 1);
    mp_be_store_l(&mut op0, 1, 1);
    // [0 1]
    mp_be_store_l(&mut op1, 0, 1);
    assert_eq!(mp_ct_le(&op0, &op1).unwrap_u8(), 0);
    assert_eq!(mp_ct_le(&op0[LIMB_BYTES..], &op1).unwrap_u8(), 1);
    assert_eq!(mp_ct_le(&op0, &op1[LIMB_BYTES..]).unwrap_u8(), 0);
}

/// Compare two multiprecision integers for `<`.
///
/// Evaluates to `true` iff `op0` is less than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer big-endian byte buffer.
/// * `op1` - The second operand as a multiprecision integer big-endian byte buffer.
///
pub fn mp_ct_lt(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    !mp_ct_ge(op0, op1)
}

/// Compare two multiprecision integers for `>=`.
///
/// Evaluates to `true` iff `op0` is greater or equal than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer big-endian byte buffer.
/// * `op1` - The second operand as a multiprecision integer big-endian byte buffer.
///
fn mp_ct_ge(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    mp_ct_le(op1, op0)
}

/// Compare two multiprecision integers for `>`.
///
/// Evaluates to `true` iff `op0` is greater than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer big-endian byte buffer.
/// * `op1` - The second operand as a multiprecision integer big-endian byte buffer.
///
fn mp_ct_gt(op0: &[u8], op1: &[u8]) -> subtle::Choice {
    !mp_ct_le(op0, op1)
}
