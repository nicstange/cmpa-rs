use crate::div_impl::mp_ct_div_lshifted_mp_mp;

use super::limb::{LimbType, LIMB_BITS, ct_add_l_l, ct_mul_add_l_l_l_c, LIMB_BYTES, LimbChoice, ct_inv_mod_l};
use super::limbs_buffer::{MPIntMutByteSlice, MPIntMutByteSlicePriv as _, MPIntByteSliceCommon, mp_ct_limbs_align_len, mp_ct_nlimbs};
use super::cmp_impl::mp_ct_geq_mp_mp;
use super::add_impl::mp_ct_sub_cond_mp_mp;
use super::div_impl::{mp_ct_div_mp_mp, mp_ct_div_pow2_mp, MpCtDivisionError};

pub fn mp_ct_montgomery_radix_shift_len(n_len: usize) -> usize {
    mp_ct_limbs_align_len(n_len)
}

pub fn mp_ct_montgomery_radix_shift_nlimbs(n_len: usize) -> usize {
    mp_ct_nlimbs(n_len)
}

pub fn mp_ct_montgomery_n0_inv_mod_l<'a, NT: MPIntByteSliceCommon>(n: &NT) -> LimbType {
    debug_assert!(!n.is_empty());
    let n0 = n.load_l(0);
    ct_inv_mod_l(n0)
}

#[test]
fn test_mp_ct_montgomery_n0_inv_mod_l() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;

    for n0 in 0 as LimbType..128 {
        let n0 = 2 * n0 + 1;
        for j in 0..2048 {
            const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
            let v = MERSENNE_PRIME_13.wrapping_mul((511 as LimbType).wrapping_mul(j));
            let v = v << 8;
            let n0 = n0.wrapping_add(v);

            let mut n0_buf: [u8; LIMB_BYTES] = [0; LIMB_BYTES];
            let mut n = MPBigEndianMutByteSlice::from_bytes(n0_buf.as_mut_slice()).unwrap();
            n.store_l(0, n0);
            let n0_inv = mp_ct_montgomery_n0_inv_mod_l(&n);
            assert_eq!(n0.wrapping_mul(n0_inv), 1);
        }
    }
}

// ATTENTION: does not read or update the most significant limb in t, it's getting returned
// separately as t_high_shadow.
fn mp_ct_montgomery_redc_one_cond_mul<TT: MPIntMutByteSlice, NT: MPIntByteSliceCommon>(
    t_carry: LimbType, t_high_shadow: LimbType,
    t: &mut TT, n: &NT, neg_n0_inv_mod_l: LimbType,
    cond_mul: LimbChoice) -> (LimbType, LimbType
) {
    debug_assert!(cond_mul.unwrap() == 0 || t_carry <= 1); // The REDC loop invariant.
    debug_assert!(!n.is_empty());
    debug_assert!(t.len() >= n.len());
    let n_nlimbs = n.nlimbs();
    let t_nlimbs = t.nlimbs();

    let (m, mut carry) = {
        let t_val = t.load_l(0);
        let m = cond_mul.select(0, t_val.wrapping_mul(neg_n0_inv_mod_l));
        let n_val = n.load_l(0);
        let (carry, t_val) = ct_mul_add_l_l_l_c(t_val, m, n_val, 0);
        debug_assert!(t_val == 0);
        (m, carry)
    };

    for j in 0..n_nlimbs - 1 {
        // Do not read the potentially partial, stale high limb directly from t, use the
        // t_high_shadow shadow instead.
        let mut t_val = if j + 1 != t_nlimbs - 1 {
            t.load_l_full(j + 1)
        } else {
            t_high_shadow
        };
        let n_val = n.load_l(j + 1);
        (carry, t_val) = ct_mul_add_l_l_l_c(t_val, m, n_val, carry);
        t.store_l_full(j, t_val);
    }

    for j in n_nlimbs - 1..t_nlimbs - 1 {
        // Do not read the potentially partial, stale high limb directly from t, use the t_high_shadow
        // shadow instead.
        let mut t_val = if j + 1 != t_nlimbs - 1 {
            t.load_l_full(j + 1)
        } else {
            t_high_shadow
        };
        (carry, t_val) = ct_add_l_l(t_val, carry);
        t.store_l_full(j, t_val);
    }

    // Replicated function entry invariant.
    debug_assert!(cond_mul.unwrap() == 0 || t_carry <= 1);
    // Do not update t's potentially partial high limb with a value that could overflow in the
    // course of the reduction. Return it separately in a t_high_shadow shadow instead.
    let (t_carry, t_high_shadow) = ct_add_l_l(carry, t_carry);
    (t_carry, t_high_shadow)
}


pub fn mp_ct_montgomery_redc<TT: MPIntMutByteSlice, NT: MPIntByteSliceCommon>(t: &mut TT, n: &NT, n0_inv_mod_l: LimbType) {
    debug_assert!(!n.is_empty());
    debug_assert!(t.len() >= n.len());
    debug_assert!(t.len() <= 2 * n.len());
    let t_nlimbs = t.nlimbs();
    let n0_val = n.load_l(0);
    debug_assert!(n0_val.wrapping_mul(n0_inv_mod_l) == 1);
    let neg_n0_inv_mod_l = !n0_inv_mod_l + 1;

    let mut reduced_t_carry = 0;
    // t's high limb might be a partial one, do not update directly in the course of reducing in
    // order to avoid overflowing it. Use a shadow instead.
    let mut t_high_shadow = t.load_l(t_nlimbs - 1);
    for _i in 0..mp_ct_montgomery_radix_shift_nlimbs(n.len()) {
        (reduced_t_carry, t_high_shadow) =
            mp_ct_montgomery_redc_one_cond_mul(
                reduced_t_carry, t_high_shadow, t, n, neg_n0_inv_mod_l, LimbChoice::from(1)
            );
    }

    // Now apply the high limb shadow back.
    let t_high_shadow_mask = t.partial_high_mask();
    let t_high_shadow_shift = t.partial_high_shift();
    assert!(t_high_shadow_shift == 0 || reduced_t_carry == 0);
    reduced_t_carry |= (t_high_shadow & !t_high_shadow_mask) >> t_high_shadow_shift;
    t_high_shadow &= t_high_shadow_mask;
    t.store_l(t_nlimbs - 1, t_high_shadow);

    mp_ct_sub_cond_mp_mp(t, n, LimbChoice::from(reduced_t_carry) | mp_ct_geq_mp_mp(t, n));
    debug_assert!(mp_ct_geq_mp_mp(t, n).unwrap() == 0);
}

#[cfg(test)]
fn test_mp_ct_montgomery_redc<TT: MPIntMutByteSlice, NT: MPIntMutByteSlice>() {
    for i in 0..64 {
        const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
        let n_high = MERSENNE_PRIME_13.wrapping_mul((16385 as LimbType).wrapping_mul(i));
        for j in 0..64 {
            const MERSENNE_PRIME_17: LimbType = 131071 as LimbType;
            let n_low = MERSENNE_PRIME_17.wrapping_mul((1023 as LimbType).wrapping_mul(j));
            // Force n_low odd.
            let n_low = n_low | 1;
            let mut n: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
            let mut n = NT::from_bytes(n.as_mut_slice()).unwrap();
            n.store_l(0, n_low);
            n.store_l(1, n_high);
            let n0_inv = mp_ct_montgomery_n0_inv_mod_l(&n);

            for k in 0..8 {
                let t_high = MERSENNE_PRIME_17.wrapping_mul((8191 as LimbType).wrapping_mul(k));
                for l in 0..8 {
                    let t_low = MERSENNE_PRIME_13.wrapping_mul((131087 as LimbType).wrapping_mul(l));

                    let mut t: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                    let mut t = TT::from_bytes(t.as_mut_slice()).unwrap();
                    t.store_l(0, t_low);
                    t.store_l(1, t_high);

                    // All montgomery operations are defined mod n, compute t mod n
                    mp_ct_div_mp_mp::<_, _, TT::SelfT<'_>>(None, &mut t, &n, None).unwrap();
                    let t_low = t.load_l(0);
                    let t_high = t.load_l(1);

                    // To Montgomery form: t * R mod n
                    let mut scratch: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                    mp_ct_to_montgomery_form_direct(&mut t, &n).unwrap();

                    // And back to normal: (t * R mod n) / R mod n
                    mp_ct_montgomery_redc(&mut t, &n, n0_inv);
                    assert_eq!(t.load_l(0), t_low);
                    assert_eq!(t.load_l(1), t_high);
                }
            }
        }
    }
}

#[test]
fn test_mp_ct_montgomery_redc_be_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_montgomery_redc::<MPBigEndianMutByteSlice, MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_montgomery_redc_le_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_montgomery_redc::<MPLittleEndianMutByteSlice, MPLittleEndianMutByteSlice>()
}

#[test]
 fn test_mp_ct_montgomery_redc_ne_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_ct_montgomery_redc::<MPNativeEndianMutByteSlice, MPNativeEndianMutByteSlice>()
}

pub fn mp_ct_montgomery_mul_mod_cond<RT: MPIntMutByteSlice, T0: MPIntByteSliceCommon,
                                 T1: MPIntByteSliceCommon, NT: MPIntByteSliceCommon> (
    result: &mut RT, op0: &T0, op1: &T1, n: &NT, n0_inv_mod_l: LimbType,
    cond: LimbChoice
) {
    // This is an implementation of the "Finely Integrated Operand Scanning (FIOS) Method"
    // approach to fused multiplication and Montgomery reduction, as described in "Analyzing
    // and Comparing Montgomery Multiplication Algorithm", IEEE Micro, 16(3):26-33, June 1996.
    debug_assert!(!n.is_empty());
    debug_assert_eq!(result.len(), n.len());
    debug_assert!(op0.len() <= n.len());
    debug_assert!(op1.len() <= n.len());

    let op0_nlimbs = op0.nlimbs();
    let op1_nlimbs = op1.nlimbs();
    if op1_nlimbs == 0 {
        // The product is zero, but if cond == 0, result is supposed to receive op0 instead of the
        // product. And it must be done in constant-time.
        for i in 0..op0_nlimbs {
            let result_val = cond.select(op0.load_l(i), 0);
            result.store_l(i, result_val);
        }
        result.zeroize_bytes_above(op0.len());
        return;
    }

    let n_nlimbs = n.nlimbs();
    let n0_val = n.load_l(0);
    debug_assert!(n0_val.wrapping_mul(n0_inv_mod_l) == 1);
    let neg_n0_inv_mod_l = !n0_inv_mod_l + 1;

    result.zeroize_bytes_above(0);
    let mut result_carry = 0;
    // result's high limb might be a partial one, do not update directly in the course of reducing in
    // order to avoid overflowing it. Use a shadow instead.
    let mut result_high_shadow = 0;
    for i in 0..op0_nlimbs {
        debug_assert!(result_carry <= 1); // Loop invariant.
        let op0_val = op0.load_l(i);
        result_carry = cond.select(op0_val, result_carry);
        let (m, mut carry_high, mut carry_low) = {
            // Do not read the potentially partial, stale high limb directly from result, use the
            // result_high_shadow shadow instead.
            let result_val = if n_nlimbs != 1 {
                result.load_l_full(0)
            } else {
                result_high_shadow
            };

            let op1_val = cond.select(0, op1.load_l(0));
            let (op0_op1_add_carry, result_val) = ct_mul_add_l_l_l_c(result_val, op0_val, op1_val, 0);

            let m = cond.select(0, result_val.wrapping_mul(neg_n0_inv_mod_l));
            let n_val = n.load_l(0);
            let (m_n_add_carry, result_val) = ct_mul_add_l_l_l_c(result_val, m, n_val, 0);

            debug_assert_eq!(result_val, 0);
            let (carry_high, carry_low) = ct_add_l_l(op0_op1_add_carry, m_n_add_carry);
            (m, carry_high, carry_low)
        };

        for j in 0..(op1_nlimbs - 1) {
            debug_assert!(carry_high <= 1); // Loop invariant LI0.
            debug_assert!(carry_high == 0 || carry_low <= !1); // Loop invariant LI1.
            let op1_val = cond.select(0, op1.load_l(j + 1));
            let n_val = n.load_l(j + 1);

            // Do not read the potentially partial, stale high limb directly from result, use the
            // result_high_shadow shadow instead.
            let result_val = if j + 1 != n_nlimbs - 1 {
                result.load_l_full(j + 1)
            } else {
                result_high_shadow
            };

            // Assume carry_high == 1 and op0_op1_add_carry == !0 below.
            // If carry_high == 1, then carry_low <= !1 per the loop invariant LI1.
            // If op0_op1_add_carry == !0, then
            // - either the op0_val * op1_val product's high_limb <= !2 and both the
            //   additions in ct_mul_add_l_l_l_c() wrapped around,
            // - or the high limb was == !1, the low limb <= 1 (as per a basic property of
            //   multiplications) and only one of the two additions wrapped.
            // In the first case, result_val <= !1 as per a basic property of
            // wrapping additions.
            // For the second case, note that adding carry_low <= !1 to the product's low limb <= 1
            // does not wrap and equals !0, at most. Adding this to the input result_val with
            // wraparound likewise yields a value <= !1.
            let (op0_op1_add_carry, result_val) = ct_mul_add_l_l_l_c(result_val, op0_val, op1_val, carry_low);
            debug_assert!(carry_high == 0 || op0_op1_add_carry <= !1 || result_val <= !1);

            // Assume carry_high == 1 and op0_op1_add_carry == !0, from which it follows
            // that result_val <= !1 at this point.
            // If m_n_add_carry == !0 below, then
            // - the m * n_val product's high limb was == !1 and the low limb <= 1
            // - and the addition of the low limb to result_val did overflow.
            // But that would contradic result_val <= !1. It follows that
            // m_n_add_carry <= !1.
            let (m_n_add_carry, result_val) = ct_mul_add_l_l_l_c(result_val, m, n_val, 0);
            debug_assert!(carry_high == 0 || op0_op1_add_carry <= !1 || m_n_add_carry <= !1);

            let carry0;
            (carry0, carry_low) = ct_add_l_l(op0_op1_add_carry, carry_high);
            // Basic property of adding a one with wraparound:
            debug_assert!(carry0 == 0 || carry_low == 0);
            debug_assert!(carry0 == 0 || m_n_add_carry <= !1);
            let carry1;
            (carry1, carry_low) = ct_add_l_l(carry_low, m_n_add_carry);
            debug_assert!(carry0 == 0 || carry1 == 0); // Loop invariant LI0 still holds.
            // Loop invariant LI1 still holds as well, because from the above it follows that ...
            debug_assert!(carry0 == 0 || carry_low <= !1);
            // ... and moreover, from a basic property of addition with workaround:
            debug_assert!(carry1 == 0 || carry_low <= !1);
            carry_high = carry0 + carry1;

            result.store_l_full(j, result_val);
        }

        // If op1_nlimbs < n_nlimbs, handle the rest by only adding the tail of m * n to it.
        for j in (op1_nlimbs - 1)..(n_nlimbs - 1) {
            debug_assert!(carry_high <= 1); // Loop invariant LI0.
            debug_assert!(carry_high == 0 || carry_low <= !1); // Loop invariant LI1.
            let n_val = n.load_l(j + 1);

            // Do not read the potentially partial, stale high limb directly from result, use the
            // result_high_shadow shadow instead.
            let mut result_val = if j + 1 != n_nlimbs - 1 {
                result.load_l_full(j + 1)
            } else {
                result_high_shadow
            };

            (carry_low, result_val) = ct_mul_add_l_l_l_c(result_val, m, n_val, carry_low);
            (carry_high, carry_low) = ct_add_l_l(carry_low, carry_high);
            result.store_l_full(j, result_val);

            // Loop invariants LI0  and LI1 are maintained trivially as per properties
            // of wrapping addition.
            debug_assert!(carry_high <= 1);
            debug_assert!(carry_high == 0 || carry_low <= !1);
        }

        // Finally, handle the most significant limb, it's effectively the sum
        // of the previously computed result_carry and this loop iteration's final
        // (carry_high, carry_low).
        debug_assert!(cond.unwrap() == 0 || result_carry <= 1); // Outer loop's invariant replicated.
        // Inner loops' invariants LI0 and LI1 replicated:
        debug_assert!(carry_high <= 1);
        debug_assert!(carry_high == 0 || carry_low <= !1);
        (result_carry, carry_low) = ct_add_l_l(carry_low, result_carry);
        // Do not update result's potentially partial high limb with a value that could overflow in
        // the course of the reduction. Maintain it separately in a result_high_shadow shadow
        // instead.
        result_high_shadow = carry_low;
        // Adding a value of result_carry <= 1 to carry_low <= !1 would not wrap.
        debug_assert!(carry_high == 0 || result_carry == 0);
        result_carry += carry_high;
        debug_assert!(result_carry <= 1); // Outer loop invariant still maintained.
    }

    // If op0_nlimbs < the montgomery radix shift distance, handle the rest by REDCing it.
    for _i in op0_nlimbs..mp_ct_montgomery_radix_shift_nlimbs(n.len()) {
        (result_carry, result_high_shadow) = mp_ct_montgomery_redc_one_cond_mul(
            result_carry, result_high_shadow, result,
            n, neg_n0_inv_mod_l, cond
        );
    }

    // Now apply the high limb shadow back.
    debug_assert!(result.len() == n.len());
    let result_high_shadow_mask = n.partial_high_mask();
    let result_high_shadow_shift = n.partial_high_shift();
    assert!(result_high_shadow_shift == 0 || result_carry == 0);
    result_carry |= (result_high_shadow & !result_high_shadow_mask) >> result_high_shadow_shift;
    result_high_shadow &= result_high_shadow_mask;
    result.store_l(n_nlimbs - 1, result_high_shadow);

    mp_ct_sub_cond_mp_mp(
        result, n, LimbChoice::from(result_carry) | mp_ct_geq_mp_mp(result, n)
    );
    debug_assert!(mp_ct_geq_mp_mp(result, n).unwrap() == 0);
}

#[cfg(test)]
fn test_mp_ct_montgomery_mul_mod_cond<RT: MPIntMutByteSlice,
                                      T0: MPIntMutByteSlice, T1: MPIntMutByteSlice,
                                      NT: MPIntMutByteSlice>() {
    use crate::limbs_buffer::MPIntByteSliceCommonPriv;

    use super::mul_impl::mp_ct_mul_trunc_cond_mp_mp;

    for i in 0..16 {
        const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
        let n_high = MERSENNE_PRIME_13.wrapping_mul((65543 as LimbType).wrapping_mul(i));
        for j in 0..16 {
            const MERSENNE_PRIME_17: LimbType = 131071 as LimbType;
            let n_low = MERSENNE_PRIME_17.wrapping_mul((4095 as LimbType).wrapping_mul(j));
            // Force n_low odd.
            let n_low = n_low | 1;
            let mut n: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
            let mut n = NT::from_bytes(n.as_mut_slice()).unwrap();
            n.store_l(0, n_low);
            n.store_l(1, n_high);
            let n_lengths = if !RT::SUPPORTS_UNALIGNED_BUFFER_LENGTHS ||
                !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS ||
                !T1::SUPPORTS_UNALIGNED_BUFFER_LENGTHS ||
                !NT::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
                    [LIMB_BYTES, 2 * LIMB_BYTES]
                } else {
                    [2 * LIMB_BYTES - 1, 2 * LIMB_BYTES]
                };
            for n_len in n_lengths {
                let (_, n) = n.split_at(n_len);
                let n0_inv = mp_ct_montgomery_n0_inv_mod_l(&n);

                // r_mod_n = 2^(2 * LIMB_BITS) % n.
                let mut r_mod_n: [u8; 3 * LIMB_BYTES] = [0; 3 * LIMB_BYTES];
                let mut r_mod_n = RT::from_bytes(r_mod_n.as_mut_slice()).unwrap();
                r_mod_n.store_l_full(mp_ct_montgomery_radix_shift_nlimbs(n_len), 1);
                mp_ct_div_mp_mp::<_, _, RT::SelfT<'_>>(None, &mut r_mod_n, &n, None).unwrap();
                let (_, r_mod_n) = r_mod_n.split_at(n.len());

                for k in 0..4 {
                    let a_high = MERSENNE_PRIME_17.wrapping_mul((16383 as LimbType).wrapping_mul(k));
                    for l in 0..4 {
                        let a_low = MERSENNE_PRIME_13.wrapping_mul((262175 as LimbType).wrapping_mul(l));
                        let mut a: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                        let mut a = T0::from_bytes(a.as_mut_slice()).unwrap();
                        a.store_l(0, a_low);
                        a.store_l(1, a_high);
                        // All montgomery operations are defined mod n, compute a mod n
                        mp_ct_div_mp_mp::<_, _, T0::SelfT<'_>>(None, &mut a, &n, None).unwrap();
                        for s in 0..4 {
                            let b_high = MERSENNE_PRIME_13.wrapping_mul((262175 as LimbType).wrapping_mul(s));
                            for t in 0..4 {
                                const MERSENNE_PRIME_19: LimbType = 524287 as LimbType;
                                let b_low = MERSENNE_PRIME_19.wrapping_mul((4095 as LimbType).wrapping_mul(t));
                                let mut b: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                                let mut b = T1::from_bytes(b.as_mut_slice()).unwrap();
                                b.store_l(0, b_low);
                                b.store_l(1, b_high);
                                // All montgomery operations are defined mod n, compute b mod n
                                mp_ct_div_mp_mp::<_, _, T1::SelfT<'_>>(None, &mut b, &n, None).unwrap();

                                for op_len in [0, 1 * LIMB_BYTES, n_len] {
                                    let (_, a) = a.split_at(op_len);
                                    let (_, b) = b.split_at(op_len);

                                    let mut _result: [u8; 4 * LIMB_BYTES] = [0; 4 * LIMB_BYTES];
                                    let mut result = RT::from_bytes(_result.as_mut_slice()).unwrap();
                                    let (_, mut mg_mul_result) = result.split_at(n_len);
                                    mp_ct_montgomery_mul_mod_cond(
                                        &mut mg_mul_result, &a, &b, &n, n0_inv,
                                        LimbChoice::from(0)
                                    );
                                    let a_nlimbs = a.nlimbs();
                                    for i in 0..a_nlimbs {
                                        assert_eq!(mg_mul_result.load_l(i), a.load_l(i));
                                    }
                                    for i in a_nlimbs..mg_mul_result.nlimbs() {
                                        assert_eq!(mg_mul_result.load_l(i), 0);
                                    }

                                    mp_ct_montgomery_mul_mod_cond(
                                        &mut mg_mul_result, &a, &b, &n, n0_inv,
                                        LimbChoice::from(1)
                                    );
                                    drop(mg_mul_result);

                                    // For testing against the expected result computed using the
                                    // "conventional" methods only, multiply by r_mod_n -- this avoids
                                    // having to multiply the conventional product by r^-1 mod n, which is
                                    // not known without implementing Euklid's algorithm.
                                    mp_ct_mul_trunc_cond_mp_mp(
                                        &mut result, n.len(), &r_mod_n, LimbChoice::from(1)
                                    );
                                    mp_ct_div_mp_mp::<_, _, RT::SelfT<'_>>(None, &mut result, &n, None).unwrap();
                                    drop(result);

                                    let mut _expected: [u8; 4 * LIMB_BYTES] = [0; 4 * LIMB_BYTES];
                                    let mut expected = RT::from_bytes(_expected.as_mut_slice()).unwrap();
                                    expected.copy_from(&a);
                                    mp_ct_mul_trunc_cond_mp_mp(
                                        &mut expected, op_len, &b, LimbChoice::from(1)
                                    );
                                    mp_ct_div_mp_mp::<_, _, RT::SelfT<'_>>(None, &mut expected, &n, None).unwrap();
                                    drop(expected);

                                    assert_eq!(_result, _expected);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_mp_ct_montgomery_mul_mod_cond_be_be_be_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_montgomery_mul_mod_cond::<MPBigEndianMutByteSlice,
                                         MPBigEndianMutByteSlice,
                                         MPBigEndianMutByteSlice,
                                         MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_montgomery_mul_mod_cond_le_le_le_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_montgomery_mul_mod_cond::<MPLittleEndianMutByteSlice,
                                         MPLittleEndianMutByteSlice,
                                         MPLittleEndianMutByteSlice,
                                         MPLittleEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_montgomery_mul_mod_cond_ne_ne_ne_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_ct_montgomery_mul_mod_cond::<MPNativeEndianMutByteSlice,
                                         MPNativeEndianMutByteSlice,
                                         MPNativeEndianMutByteSlice,
                                         MPNativeEndianMutByteSlice>()
}

pub fn mp_ct_montgomery_mul_mod<RT: MPIntMutByteSlice, T0: MPIntByteSliceCommon,
                                T1: MPIntByteSliceCommon, NT: MPIntByteSliceCommon> (
    result: &mut RT, op0: &T0, op1: &T1, n: &NT, n0_inv_mod_l: LimbType
) {
    mp_ct_montgomery_mul_mod_cond(result, op0, op1, n, n0_inv_mod_l, LimbChoice::from(1))
}

pub fn mp_ct_to_montgomery_form_direct<TT: MPIntMutByteSlice, NT: MPIntByteSliceCommon>(
    t: &mut TT, n: &NT
) -> Result<(), MpCtDivisionError> {
    debug_assert!(t.nlimbs() >= n.nlimbs());
    let radix_shift_len = mp_ct_montgomery_radix_shift_len(n.len());
    mp_ct_div_lshifted_mp_mp::<_, _, TT>(t, t.len(), radix_shift_len, n, None)
}

pub fn mp_ct_montgomery_radix2_mod_n<RX2T: MPIntMutByteSlice, NT: MPIntByteSliceCommon>(
    radix2_mod_n_out: &mut RX2T, n: &NT
) -> Result<(), MpCtDivisionError> {
    debug_assert!(mp_ct_nlimbs(radix2_mod_n_out.len()) >= mp_ct_nlimbs(n.len()));
    let radix_shift_len = mp_ct_montgomery_radix_shift_len(n.len());
    mp_ct_div_pow2_mp::<_, _, RX2T>(2 * 8 * radix_shift_len, radix2_mod_n_out, n, None)
}

pub fn mp_ct_to_montgomery_form<MGT: MPIntMutByteSlice, TT: MPIntByteSliceCommon, NT: MPIntByteSliceCommon, RX2T: MPIntByteSliceCommon> (
    mg_t_out: &mut MGT, t: &TT, n: &NT, n0_inv_mod_l: LimbType, radix2_mod_n: &RX2T
) {
    mp_ct_montgomery_mul_mod(mg_t_out, t, radix2_mod_n, n, n0_inv_mod_l);
}


#[cfg(test)]
fn test_mp_ct_to_montgomery_form<TT: MPIntMutByteSlice, NT: MPIntMutByteSlice, RX2T: MPIntMutByteSlice>() {
    use super::limbs_buffer::MPIntMutByteSlicePriv as _;
    use super::cmp_impl::mp_ct_eq_mp_mp;

    for i in 0..16 {
        const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
        let n_high = MERSENNE_PRIME_13.wrapping_mul((65543 as LimbType).wrapping_mul(i));
        for j in 0..16 {
            const MERSENNE_PRIME_17: LimbType = 131071 as LimbType;
            let n_low = MERSENNE_PRIME_17.wrapping_mul((4095 as LimbType).wrapping_mul(j));
            // Force n_low odd.
            let n_low = n_low | 1;
            let mut n: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
            let mut n = NT::from_bytes(n.as_mut_slice()).unwrap();
            n.store_l(0, n_low);
            n.store_l(1, n_high);
            let n_lengths = if !NT::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
                    [LIMB_BYTES, 2 * LIMB_BYTES]
                } else {
                    [2 * LIMB_BYTES - 1, 2 * LIMB_BYTES]
                };

            for n_len in n_lengths {
                let (_, n) = n.split_at(n_len);
                let n0_inv = mp_ct_montgomery_n0_inv_mod_l(&n);

                let mut radix2_mod_n: [u8; 2 * LIMB_BYTES] = [0xffu8; 2 * LIMB_BYTES];
                let mut radix2_mod_n = RX2T::from_bytes(radix2_mod_n.as_mut_slice()).unwrap();
                let (_, mut radix2_mod_n) = radix2_mod_n.split_at(RX2T::limbs_align_len(n_len));
                mp_ct_montgomery_radix2_mod_n(&mut radix2_mod_n, &n).unwrap();

                for k in 0..4 {
                    let a_high = MERSENNE_PRIME_17.wrapping_mul((16383 as LimbType).wrapping_mul(k));
                    for l in 0..4 {
                        let a_low = MERSENNE_PRIME_13.wrapping_mul((262175 as LimbType).wrapping_mul(l));
                        let mut a: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                        let mut a = TT::from_bytes(a.as_mut_slice()).unwrap();
                        a.store_l(0, a_low);
                        a.store_l(1, a_high);
                        // All montgomery operations are defined mod n, compute a mod n
                        mp_ct_div_mp_mp::<_, _, TT::SelfT<'_>>(None, &mut a, &n, None).unwrap();
                        let (_, mut a) = a.split_at(TT::limbs_align_len(n_len));

                        let mut result: [u8; 2 * LIMB_BYTES] = [0xff; 2 * LIMB_BYTES];
                        let mut result = TT::from_bytes(result.as_mut_slice()).unwrap();
                        let (_, mut result) = result.split_at(TT::limbs_align_len(n_len));
                        let (_, mut result) = result.split_at(TT::limbs_align_len(n_len));
                        mp_ct_to_montgomery_form(&mut result, &a, &n, n0_inv, &radix2_mod_n);

                        mp_ct_to_montgomery_form_direct(&mut a, &n).unwrap();
                        assert_eq!(mp_ct_eq_mp_mp(&result, &a).unwrap(), 1);
                    }
                }
            }
        }
    }
}

#[test]
fn test_mp_ct_to_montgomery_form_be_be_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_to_montgomery_form::<MPBigEndianMutByteSlice,
                                         MPBigEndianMutByteSlice,
                                         MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_to_montgomery_form_le_le_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_to_montgomery_form::<MPLittleEndianMutByteSlice,
                                         MPLittleEndianMutByteSlice,
                                         MPLittleEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_to_montgomery_form_ne_ne_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_ct_to_montgomery_form::<MPNativeEndianMutByteSlice,
                                         MPNativeEndianMutByteSlice,
                                         MPNativeEndianMutByteSlice>()
}

