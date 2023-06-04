//! Implementation of multiprecision integer addition related primitives.

use super::limb::{LimbType, LIMB_BYTES, ct_add_l_l};
use super::limbs_buffer::{mp_be_load_l, mp_be_load_l_full, mp_be_store_l, mp_be_store_l_full, mp_ct_nlimbs};

/// Add two multiprecision integers.
///
/// Add two multiprecision integers as stored in big-endian byte slices. The first operand's
/// contents will be replaced by the resulting sum and the carry, if any, returned from the
/// function.
///
/// Runs in constant time for a given configuration of input operand widths, i.e. execution time
/// depends only on the integers' widths, but not their values.
///
/// # Arguments:
///
/// * `op0` - The first input addend. It will be overwritten by the resulting sum. The slice
///           length must greater or equal than the length of the second addend.
/// * `op1` - The second input addend. Its length must not exceed the length of `op0`.
///
pub fn mp_ct_add(op0: &mut [u8], op1: &[u8]) -> LimbType {
    debug_assert!(op1.len() <= op0.len());
    let op0_nlimbs = mp_ct_nlimbs(op0.len());
    let op1_nlimbs = mp_ct_nlimbs(op1.len());
    if op1_nlimbs == 0 {
        return 0;
    }

    let mut carry = 0;
    for i in 0..op1_nlimbs - 1 {
        let mut op0_val = mp_be_load_l_full(op0, i);
        let op1_val = mp_be_load_l_full(op1, i);
        let carry0;
        (carry0, op0_val) = ct_add_l_l(op0_val, carry);
        let carry1;
        (carry1, op0_val) = ct_add_l_l(op0_val, op1_val);
        carry = carry0 + carry1;
        mp_be_store_l_full(op0, i, op0_val);
    }

    // Propagate the carry upwards. The first iteration will also account
    // for op1's high limb.
    let mut op1_val = mp_be_load_l(op1, op1_nlimbs - 1);
    for i in op1_nlimbs - 1..op0_nlimbs {
        let mut op0_val = mp_be_load_l(op0, i);
        let carry0;
        (carry0, op0_val) = ct_add_l_l(op0_val, carry);
        let carry1;
        (carry1, op0_val) = ct_add_l_l(op0_val, op1_val);
        op1_val = 0;
        carry = carry0 + carry1;
        if i != op0_nlimbs - 1 {
            mp_be_store_l_full(op0, i, op0_val);
        } else {
            let op0_high_npartial = op0.len() % LIMB_BYTES;
            let op0_high_mask = if op0_high_npartial == 0 {
                !0
            } else {
                (1 << 8 * op0_high_npartial) - 1
            };
            mp_be_store_l(op0, i, op0_val & op0_high_mask);
            let carry_in_op0_high = (op0_val & !op0_high_mask) >> 8 * op0_high_npartial;
            debug_assert!(carry == 0 || carry_in_op0_high == 0);
            carry |= carry_in_op0_high;
        }
    }
    carry
}

#[test]
fn test_mp_ct_add() {
    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op1 = op1.as_mut_slice();
    mp_be_store_l(op0, 0, !0);
    mp_be_store_l(op1, 0, !0);
    let carry = mp_ct_add(op0, op1);
    assert_eq!(carry, 0);
    assert_eq!(mp_be_load_l(op0, 0), !1);
    assert_eq!(mp_be_load_l(op0, 1), 1);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op1 = op1.as_mut_slice();
    mp_be_store_l(op0, 0, !0);
    mp_be_store_l(op0, 1, !0);
    mp_be_store_l(op1, 0, !0);
    mp_be_store_l(op1, 1, !0);
    let carry = mp_ct_add(op0, op1);
    assert_eq!(carry, 1);
    assert_eq!(mp_be_load_l(op0, 0), !1);
    assert_eq!(mp_be_load_l(op0, 1), !0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 1 * LIMB_BYTES] = [0; 1 * LIMB_BYTES];
    let op1 = op1.as_mut_slice();
    mp_be_store_l(op0, 0, !0);
    mp_be_store_l(op0, 1, !0);
    mp_be_store_l(op1, 0, !0);
    let carry = mp_ct_add(op0, op1);
    assert_eq!(carry, 1);
    assert_eq!(mp_be_load_l(op0, 0), !1);
    assert_eq!(mp_be_load_l(op0, 1), 0);

    let mut op0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op1 = op1.as_mut_slice();
    mp_be_store_l(op0, 0, !0);
    mp_be_store_l(op1, 0, !0);
    let carry = mp_ct_add(op0, op1);
    assert_eq!(carry, 0);
    assert_eq!(mp_be_load_l(op0, 0), !1);
    assert_eq!(mp_be_load_l(op0, 1), 1);

    let mut op0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let op1 = op1.as_mut_slice();
    mp_be_store_l(op0, 0, !0);
    mp_be_store_l(op0, 1, !0 >> 8);
    mp_be_store_l(op1, 0, !0);
    mp_be_store_l(op1, 1, !0 >> 8);
    let carry = mp_ct_add(op0, op1);
    assert_eq!(carry, 1);
    assert_eq!(mp_be_load_l(op0, 0), !1);
    assert_eq!(mp_be_load_l(op0, 1), !0 >> 8);

    let mut op0: [u8; LIMB_BYTES - 1] = [0; LIMB_BYTES - 1];
    let op0 = op0.as_mut_slice();
    let mut op1: [u8; LIMB_BYTES - 1] = [0; LIMB_BYTES - 1];
    let op1 = op1.as_mut_slice();
    mp_be_store_l(op0, 0, !0 >> 8);
    mp_be_store_l(op1, 0, !0 >> 8);
    let carry = mp_ct_add(op0, op1);
    assert_eq!(carry, 1);
    assert_eq!(mp_be_load_l(op0, 0), (!0 >> 8) ^ 1);
}
