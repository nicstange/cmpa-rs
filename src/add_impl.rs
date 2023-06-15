//! Implementation of multiprecision integer addition related primitives.

use crate::limb::ct_find_last_set_byte_l;

use super::limb::{LimbType, LIMB_BYTES, LIMB_BITS, ct_add_l_l, ct_sub_l_l};
use super::limbs_buffer::{mp_ct_nlimbs, MPEndianess, MPBigEndianOrder, MPLittleEndianOrder};

use subtle::{self, ConditionallySelectable as _};

/// Add two multiprecision integers of specified endianess.
///
/// Add two multiprecision integers as stored in byte slices of endianess as specified by the `E0`
/// and `E1` generic parameters each. The first operand's contents will be replaced by the resulting
/// sum and the carry, if any, returned from the function.
///
/// Runs in constant time for a given configuration of input operand widths, i.e. execution time
/// depends only on the integers' widths, but not their values.
///
/// # Arguments:
///
/// * `op0` - The first input addend. It will be overwritten by the resulting sum. The slice
///           length must greater or equal than the length of the second addend.
///           Its endianess is specified by the `E0` generic parameter.
/// * `op1` - The second input addend. Its length must not exceed the length of `op0`.
///           Its endianess is specified by the `E0` generic parameter.
///
pub fn mp_ct_add_mp_mp<E0: MPEndianess, E1: MPEndianess>(op0: &mut [u8], op1: &[u8]) -> LimbType {
    debug_assert!(op1.len() <= op0.len());
    let op0_nlimbs = mp_ct_nlimbs(op0.len());
    let op1_nlimbs = mp_ct_nlimbs(op1.len());
    if op1_nlimbs == 0 {
        return 0;
    }

    let mut carry = 0;
    for i in 0..op1_nlimbs - 1 {
        let mut op0_val = E0::load_l_full(op0, i);
        let op1_val = E1::load_l_full(op1, i);
        let carry0;
        (carry0, op0_val) = ct_add_l_l(op0_val, carry);
        let carry1;
        (carry1, op0_val) = ct_add_l_l(op0_val, op1_val);
        carry = carry0 + carry1;
        E0::store_l_full(op0, i, op0_val);
    }

    // Propagate the carry upwards. The first iteration will also account
    // for op1's high limb.
    let mut op1_val = E1::load_l(op1, op1_nlimbs - 1);
    for i in op1_nlimbs - 1..op0_nlimbs {
        let mut op0_val = E0::load_l(op0, i);
        let carry0;
        (carry0, op0_val) = ct_add_l_l(op0_val, carry);
        let carry1;
        (carry1, op0_val) = ct_add_l_l(op0_val, op1_val);
        op1_val = 0;
        carry = carry0 + carry1;
        if i != op0_nlimbs - 1 {
            E0::store_l_full(op0, i, op0_val);
        } else {
            let op0_high_npartial = op0.len() % LIMB_BYTES;
            let op0_high_mask = if op0_high_npartial == 0 {
                !0
            } else {
                (1 << 8 * op0_high_npartial) - 1
            };
            E0::store_l(op0, i, op0_val & op0_high_mask);
            let carry_in_op0_high = (op0_val & !op0_high_mask) >> 8 * op0_high_npartial;
            debug_assert!(carry == 0 || carry_in_op0_high == 0);
            carry |= carry_in_op0_high;
        }
    }
    carry
}

/// Add two big-endian multiprecision integers.
///
/// This is a convenience specialization of the generic [`mp_ct_add_mp_mp()`] for the big-endian
/// case.
///
pub fn mp_ct_add_be_be(op0: &mut [u8], op1: &[u8]) -> LimbType {
    mp_ct_add_mp_mp::<MPBigEndianOrder, MPBigEndianOrder>(op0, op1)
}

/// Add two little-endian multiprecision integers.
///
/// This is a convenience specialization of the generic [`mp_ct_add_mp_mp()`] for the little-endian
/// case.
///
pub fn mp_ct_add_le_le(op0: &mut [u8], op1: &[u8]) -> LimbType {
    mp_ct_add_mp_mp::<MPLittleEndianOrder, MPLittleEndianOrder>(op0, op1)
}

#[cfg(test)]
fn test_mp_ct_add_mp_mp<E0: MPEndianess, E1: MPEndianess>() {
    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !0);
    E1::store_l(op1, 0, !0);
    let carry = mp_ct_add_mp_mp::<E0, E1>(op0, op1);
    assert_eq!(carry, 0);
    assert_eq!(E0::load_l(op0, 0), !1);
    assert_eq!(E0::load_l(op0, 1), 1);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !0);
    E0::store_l(op0, 1, !0);
    E1::store_l(op1, 0, !0);
    E1::store_l(op1, 1, !0);
    let carry = mp_ct_add_mp_mp::<E0, E1>(op0, op1);
    assert_eq!(carry, 1);
    assert_eq!(E0::load_l(op0, 0), !1);
    assert_eq!(E0::load_l(op0, 1), !0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 1 * LIMB_BYTES] = [0; 1 * LIMB_BYTES];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !0);
    E0::store_l(op0, 1, !0);
    E1::store_l(op1, 0, !0);
    let carry = mp_ct_add_mp_mp::<E0, E1>(op0, op1);
    assert_eq!(carry, 1);
    assert_eq!(E0::load_l(op0, 0), !1);
    assert_eq!(E0::load_l(op0, 1), 0);

    let mut op0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !0);
    E1::store_l(op1, 0, !0);
    let carry = mp_ct_add_mp_mp::<E0, E1>(op0, op1);
    assert_eq!(carry, 0);
    assert_eq!(E0::load_l(op0, 0), !1);
    assert_eq!(E0::load_l(op0, 1), 1);

    let mut op0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !0);
    E0::store_l(op0, 1, !0 >> 8);
    E1::store_l(op1, 0, !0);
    E1::store_l(op1, 1, !0 >> 8);
    let carry = mp_ct_add_mp_mp::<E0, E1>(op0, op1);
    assert_eq!(carry, 1);
    assert_eq!(E0::load_l(op0, 0), !1);
    assert_eq!(E0::load_l(op0, 1), !0 >> 8);

    let mut op0: [u8; LIMB_BYTES - 1] = [0; LIMB_BYTES - 1];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; LIMB_BYTES - 1] = [0; LIMB_BYTES - 1];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !0 >> 8);
    E1::store_l(op1, 0, !0 >> 8);
    let carry = mp_ct_add_mp_mp::<E0, E1>(op0, op1);
    assert_eq!(carry, 1);
    assert_eq!(E0::load_l(op0, 0), (!0 >> 8) ^ 1);
}


#[test]
fn test_mp_ct_add_be_be() {
    test_mp_ct_add_mp_mp::<MPBigEndianOrder, MPBigEndianOrder>()
}

#[test]
fn test_mp_ct_add_le_le() {
    test_mp_ct_add_mp_mp::<MPLittleEndianOrder, MPLittleEndianOrder>()
}

// Add a limb to a multiprecision integer.
pub fn mp_ct_add_mp_l<E0: MPEndianess>(op0: &mut [u8], op1: LimbType) -> LimbType {
    debug_assert!(ct_find_last_set_byte_l(op1) <= op0.len());
    let op0_nlimbs = mp_ct_nlimbs(op0.len());
    if op0_nlimbs == 0 {
        return 0;
    }

    // Set carry to op1 and propagate the carry upwards.
    let mut carry = op1;
    for i in 0..op0_nlimbs - 1 {
        let mut op0_val = E0::load_l_full(op0, i);
        (carry, op0_val) = ct_add_l_l(op0_val, carry);
        E0::store_l_full(op0, i, op0_val);
    }

    let mut op0_val = E0::load_l(op0, op0_nlimbs - 1);
    (carry, op0_val) = ct_add_l_l(op0_val, carry);
    let op0_high_npartial = op0.len() % LIMB_BYTES;
    let op0_high_mask = if op0_high_npartial == 0 {
        !0
    } else {
        (1 << 8 * op0_high_npartial) - 1
    };
    E0::store_l(op0, op0_nlimbs - 1, op0_val & op0_high_mask);
    let carry_in_op0_high = (op0_val & !op0_high_mask) >> 8 * op0_high_npartial;
    debug_assert!(carry == 0 || carry_in_op0_high == 0);
    carry |= carry_in_op0_high;

    carry
}

/// Conditionally subtract two multiprecision integers of specified endianess.
///
/// Conditionally subtract two multiprecision integers as stored in byte slices of endianess as
/// specified by the `E0` and `E1` generic parameters each. The first operand's contents will be
/// replaced by the resulting difference if `cond` is set and the borrow, if any, returned from the
/// function.
///
/// Runs in constant time for a given configuration of input operand widths, i.e. execution time
/// depends only on the integers' widths, but not their values and neither on `cond`.
///
/// # Arguments:
///
/// * `op0` - The minuend. It will be overwritten by the resulting difference if `cond` is set. The slice
///           length must greater or equal than the length of the second operand.
///           Its endianess is specified by the `E0` generic parameter.
/// * `op1` - The subtrahend. Its length must not exceed the length of `op0`.
///           Its endianess is specified by the `E0` generic parameter.
/// * `cond` - Whether or not to replace `ob0` by the difference.
///
pub fn mp_ct_sub_cond_mp_mp<E0: MPEndianess, E1: MPEndianess>(
    op0: &mut [u8], op1: &[u8], cond: subtle::Choice
) -> LimbType {
    debug_assert!(op1.len() <= op0.len());
    let op0_nlimbs = mp_ct_nlimbs(op0.len());
    let op1_nlimbs = mp_ct_nlimbs(op1.len());
    if op1_nlimbs == 0 {
        return 0;
    }

    let mut borrow = 0;
    for i in 0..op1_nlimbs - 1 {
        let mut op0_val = E0::load_l_full(op0, i);
        let op1_val = E1::load_l_full(op1, i);
        let op1_val = LimbType::conditional_select(&0, &op1_val, cond);
        let borrow0;
        (borrow0, op0_val) = ct_sub_l_l(op0_val, borrow);
        let borrow1;
        (borrow1, op0_val) = ct_sub_l_l(op0_val, op1_val);
        borrow = borrow0 + borrow1;
        E0::store_l_full(op0, i, op0_val);
    }

    // Propagate the borrow upwards. The first iteration will also account
    // for op1's high limb.
    let op1_val = E1::load_l(op1, op1_nlimbs - 1);
    let mut op1_val = LimbType::conditional_select(&0, &op1_val, cond);
    for i in op1_nlimbs - 1..op0_nlimbs {
        let mut op0_val = E0::load_l(op0, i);
        let borrow0;
        (borrow0, op0_val) = ct_sub_l_l(op0_val, borrow);
        let borrow1;
        (borrow1, op0_val) = ct_sub_l_l(op0_val, op1_val);
        op1_val = 0;
        borrow = borrow0 + borrow1;
        if i != op0_nlimbs - 1 {
            E0::store_l_full(op0, i, op0_val);
        } else {
            let op0_high_npartial = op0.len() % LIMB_BYTES;
            let op0_high_mask = if op0_high_npartial == 0 {
                !0
            } else {
                (1 << 8 * op0_high_npartial) - 1
            };
            E0::store_l(op0, i, op0_val & op0_high_mask);
            debug_assert!(op0_high_npartial == 0 || borrow == op0_val >> (LIMB_BITS - 1));
        }
    }
    borrow
}

/// Conditionally subtract two big-endian multiprecision integers.
///
/// This is a convenience specialization of the generic [`mp_ct_sub_mp_mp()`] for the big-endian
/// case.
///
pub fn mp_ct_sub_cond_be_be(op0: &mut [u8], op1: &[u8], cond: subtle::Choice) -> LimbType {
    mp_ct_sub_cond_mp_mp::<MPBigEndianOrder, MPBigEndianOrder> (op0, op1, cond)
}

/// Conditionally subtract two little-endian multiprecision integers.
///
/// This is a convenience specialization of the generic [`mp_ct_sub_mp_mp()`] for the little-endian
/// case.
///
pub fn mp_ct_sub_cond_le_le(op0: &mut [u8], op1: &[u8], cond: subtle::Choice) -> LimbType {
    mp_ct_sub_cond_mp_mp::<MPLittleEndianOrder, MPLittleEndianOrder> (op0, op1, cond)
}

#[cfg(test)]
fn test_mp_ct_sub_cond_mp_mp<E0: MPEndianess, E1: MPEndianess>() {
    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !0);
    E1::store_l(op1, 0, !0);
    let borrow = mp_ct_sub_cond_mp_mp::<E0, E1>(op0, op1, subtle::Choice::from(0u8));
    assert_eq!(borrow, 0);
    assert_eq!(E0::load_l(op0, 0), !0);
    assert_eq!(E0::load_l(op0, 1), 0);
    let borrow = mp_ct_sub_cond_mp_mp::<E0, E1>(op0, op1, subtle::Choice::from(1u8));
    assert_eq!(borrow, 0);
    assert_eq!(E0::load_l(op0, 0), 0);
    assert_eq!(E0::load_l(op0, 1), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !1);
    E0::store_l(op0, 1, 1);
    E1::store_l(op1, 0, !0);
    let borrow = mp_ct_sub_cond_mp_mp::<E0, E1>(op0, op1, subtle::Choice::from(0u8));
    assert_eq!(borrow, 0);
    assert_eq!(E0::load_l(op0, 0), !1);
    assert_eq!(E0::load_l(op0, 1), 1);
    let borrow = mp_ct_sub_cond_mp_mp::<E0, E1>(op0, op1, subtle::Choice::from(1u8));
    assert_eq!(borrow, 0);
    assert_eq!(E0::load_l(op0, 0), !0);
    assert_eq!(E0::load_l(op0, 1), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 1 * LIMB_BYTES] = [0; 1 * LIMB_BYTES];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !1);
    E0::store_l(op0, 1, 1);
    E1::store_l(op1, 0, !0);
    let borrow = mp_ct_sub_cond_mp_mp::<E0, E1>(op0, op1, subtle::Choice::from(0u8));
    assert_eq!(borrow, 0);
    assert_eq!(E0::load_l(op0, 0), !1);
    assert_eq!(E0::load_l(op0, 1), 1);
    let borrow = mp_ct_sub_cond_mp_mp::<E0, E1>(op0, op1, subtle::Choice::from(1u8));
    assert_eq!(borrow, 0);
    assert_eq!(E0::load_l(op0, 0), !0);
    assert_eq!(E0::load_l(op0, 1), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !1);
    E0::store_l(op0, 1, 0);
    E1::store_l(op1, 0, !0);
    E1::store_l(op1, 1, 1);
    let borrow = mp_ct_sub_cond_mp_mp::<E0, E1>(op0, op1, subtle::Choice::from(0u8));
    assert_eq!(borrow, 0);
    assert_eq!(E0::load_l(op0, 0), !1);
    assert_eq!(E0::load_l(op0, 1), 0);
    let borrow = mp_ct_sub_cond_mp_mp::<E0, E1>(op0, op1, subtle::Choice::from(1u8));
    assert_eq!(borrow, 1);
    assert_eq!(E0::load_l(op0, 0), !0);
    assert_eq!(E0::load_l(op0, 1), !1);

    let mut op0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !0);
    E1::store_l(op1, 0, !0);
    let borrow = mp_ct_sub_cond_mp_mp::<E0, E1>(op0, op1, subtle::Choice::from(0u8));
    assert_eq!(borrow, 0);
    assert_eq!(E0::load_l(op0, 0), !0);
    assert_eq!(E0::load_l(op0, 1), 0);
    let borrow = mp_ct_sub_cond_mp_mp::<E0, E1>(op0, op1, subtle::Choice::from(1u8));
    assert_eq!(borrow, 0);
    assert_eq!(E0::load_l(op0, 0), 0);
    assert_eq!(E0::load_l(op0, 1), 0);

    let mut op0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !1);
    E0::store_l(op0, 1, 1);
    E1::store_l(op1, 0, !0);
    let borrow = mp_ct_sub_cond_mp_mp::<E0, E1>(op0, op1, subtle::Choice::from(0u8));
    assert_eq!(borrow, 0);
    assert_eq!(E0::load_l(op0, 0), !1);
    assert_eq!(E0::load_l(op0, 1), 1);
    let borrow = mp_ct_sub_cond_mp_mp::<E0, E1>(op0, op1, subtle::Choice::from(1u8));
    assert_eq!(borrow, 0);
    assert_eq!(E0::load_l(op0, 0), !0);
    assert_eq!(E0::load_l(op0, 1), 0);

    let mut op0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op1 = op1.as_mut_slice();
    E0::store_l(op0, 0, !1);
    E0::store_l(op0, 1, 0);
    E1::store_l(op1, 0, !0);
    E1::store_l(op1, 1, 1);
    let borrow = mp_ct_sub_cond_mp_mp::<E0, E1>(op0, op1, subtle::Choice::from(0u8));
    assert_eq!(borrow, 0);
    assert_eq!(E0::load_l(op0, 0), !1);
    assert_eq!(E0::load_l(op0, 1), 0);
    let borrow = mp_ct_sub_cond_mp_mp::<E0, E1>(op0, op1, subtle::Choice::from(1u8));
    assert_eq!(borrow, 1);
    assert_eq!(E0::load_l(op0, 0), !0);
    assert_eq!(E0::load_l(op0, 1), !0 >> 8 & !1);
}

#[test]
fn test_mp_ct_sub_cond_be_be() {
    test_mp_ct_add_mp_mp::<MPBigEndianOrder, MPBigEndianOrder>()
}

#[test]
fn test_mp_ct_sub_cond_le_le() {
    test_mp_ct_add_mp_mp::<MPLittleEndianOrder, MPLittleEndianOrder>()
}

// Subtract a limb from a multiprecision integer.
pub fn mp_ct_sub_mp_l<E0: MPEndianess>(op0: &mut [u8], op1: LimbType) -> LimbType {
    debug_assert!(ct_find_last_set_byte_l(op1) <= op0.len());
    let op0_nlimbs = mp_ct_nlimbs(op0.len());
    if op0_nlimbs == 0 {
        return 0;
    }

    // Set borrow to op1 and propagate the borrow upwards.
    let mut borrow = op1;
    for i in 0..op0_nlimbs - 1 {
        let mut op0_val = E0::load_l_full(op0, i);
        (borrow, op0_val) = ct_sub_l_l(op0_val, borrow);
        E0::store_l_full(op0, i, op0_val);
    }

    let mut op0_val = E0::load_l(op0, op0_nlimbs - 1);
    (borrow, op0_val) = ct_sub_l_l(op0_val, borrow);
    let op0_high_npartial = op0.len() % LIMB_BYTES;
    let op0_high_mask = if op0_high_npartial == 0 {
        !0
    } else {
        (1 << 8 * op0_high_npartial) - 1
    };
    E0::store_l(op0, op0_nlimbs, op0_val & op0_high_mask);
    debug_assert!(op0_high_npartial == 0 || borrow == op0_val >> (LIMB_BITS - 1));

    borrow
}
