use crate::limb::ct_neq_l_l;

use super::limb::{LimbType, LIMB_BITS, ct_add_l_l, ct_mul_l_l, LIMB_BYTES, ct_l_to_subtle_choice};
use super::limbs_buffer::{MPEndianess, MPBigEndianOrder, MPLittleEndianOrder, mp_ct_nlimbs};
use super::cmp_impl::mp_ct_geq_mp_mp;
use super::add_impl::mp_ct_sub_cond_mp_mp;
use super::div_impl::mp_ct_div_mp_mp;
use subtle::{self, ConditionallySelectable as _};
use zeroize::Zeroize;

pub fn mp_ct_to_montgomery_form<TE: MPEndianess, NE: MPEndianess>(t: &mut [u8], scratch: &mut [u8], n: &[u8]) {
    debug_assert!(t.len() >= n.len());
    let n_limbs = mp_ct_nlimbs(n.len());
    let shift_len = n_limbs * LIMB_BYTES;
    debug_assert!(scratch.len() >= shift_len);
    let (_, t_low) = TE::split_at_mut(scratch, shift_len);
    TE::zeroize_bytes_above(t_low, 0);
    mp_ct_div_mp_mp::<TE, NE, TE>(Some(t), t_low, n, None).unwrap();

    debug_assert!(t.iter().all(|v| *v == 0));
    let (_, t_low) = TE::split_at(&t_low, n.len());
    TE::copy_from::<TE>(t, t_low);
}

pub fn mp_ct_montgomery_n0_inv_mod_l<NE: MPEndianess>(n: &[u8]) -> LimbType {
    debug_assert!(!n.is_empty());
    let n0 = NE::load_l(&n, 0);
    debug_assert!(n0 % 2 != 0);
    // The odd numbers mod 2^LIMB_BITS form a multiplicative group of order 2^(LIMB_BITS - 1).
    // It follows that n0^(2^(LIMB_BITS - 1) - 1) is n0's inverse mod 2^LIMB_BITS.
    // The exponent (2^(LIMB_BITS - 1) - 1) has exactly the lower (LIMB_BITS - 1) bits
    // set. Compute the exponentation of n0 by binary exponentation hard-coded for the
    // constant exponent.
    let mut n0_inv = n0;
    for _i in 0..LIMB_BITS - 2 {
        n0_inv = n0_inv.wrapping_mul(n0_inv);
        n0_inv = n0_inv.wrapping_mul(n0);
    }
    n0_inv
}

#[test]
fn test_mp_ct_montgomery_n0_inv_mod_l() {
    for n0 in 0 as LimbType..128 {
        let n0 = 2 * n0 + 1;
        for j in 0..2048 {
            const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
            let v = MERSENNE_PRIME_13.wrapping_mul((511 as LimbType).wrapping_mul(j));
            let v = v << 8;
            let n0 = n0.wrapping_add(v);

            let mut n0_buf: [u8; LIMB_BYTES] = [0; LIMB_BYTES];
            MPBigEndianOrder::store_l(&mut n0_buf, 0, n0);
            let n0_inv = mp_ct_montgomery_n0_inv_mod_l::<MPBigEndianOrder>(&n0_buf);
            assert_eq!(n0.wrapping_mul(n0_inv), 1);
        }
    }
}

// ATTENTION: does not read or update the most significant limb in t, it's getting returned
// separately as t_high_shadow.
fn mp_ct_montgomery_redc_one_cond_mul<TE: MPEndianess, NE: MPEndianess>(
    t_carry: LimbType, t_high_shadow: LimbType,
    t: &mut [u8], n: &[u8], neg_n0_inv_mod_l: LimbType,
    cond_mul: Option<subtle::Choice>) -> (LimbType, LimbType
) {
    debug_assert!(cond_mul.map(|c| c.unwrap_u8()).unwrap_or(1) == 0 || t_carry <= 1); // The REDC loop invariant.
    debug_assert!(!n.is_empty());
    debug_assert!(t.len() >= n.len());
    let n_nlimbs = mp_ct_nlimbs(n.len());
    let t_nlimbs = mp_ct_nlimbs(t.len());

    let m;
    let mut carry_low;
    (m, carry_low) = {
        let t_val = TE::load_l(&t, 0);
        let m = t_val.wrapping_mul(neg_n0_inv_mod_l);
        let m = cond_mul.map(|c| LimbType::conditional_select(&0, &m, c)).unwrap_or(m);
        let n_val = NE::load_l(n, 0);
        let m_n_val = ct_mul_l_l(m, n_val);
        let (carry_low, t_val) = ct_add_l_l(t_val, m_n_val.low());
        debug_assert!(t_val == 0);
        debug_assert!(m_n_val.high() <= !1); // Property of product.
        (
            m,
            carry_low + m_n_val.high() // Does not wrap.
        )
    };

    let mut carry_high = 0;
    for j in 0..n_nlimbs - 1 {
        let n_val = NE::load_l(n, j + 1);
        let m_n_val = ct_mul_l_l(m, n_val);
        // Do not read the potentially partial, stale high limb directly from t, use the
        // t_high_shadow shadow instead.
        let mut t_val = if j + 1 != t_nlimbs - 1 {
            TE::load_l_full(t, j + 1)
        } else {
            t_high_shadow
        };
        let carry0;
        (carry0, t_val) = ct_add_l_l(t_val, carry_low);
        let carry1;
        (carry1, t_val) = ct_add_l_l(t_val, m_n_val.low());
        debug_assert!(m_n_val.high() <= !1); // Property of product.
        carry_low = carry_high + m_n_val.high(); // Does not wrap.
        (carry_high, carry_low) = ct_add_l_l(carry_low, carry0 + carry1);
        debug_assert!(carry_high == 0 || carry_low <= 1); // Property of addition.
        TE::store_l_full(t, j, t_val);
    }

    for j in n_nlimbs - 1..t_nlimbs - 1 {
        // Do not read the potentially partial, stale high limb directly from t, use the t_high_shadow
        // shadow instead.
        let mut t_val = if j + 1 != t_nlimbs - 1 {
            TE::load_l_full(t, j + 1)
        } else {
            t_high_shadow
        };
        (carry_low, t_val) = ct_add_l_l(t_val, carry_low);
        carry_low += carry_high;
        carry_high = 0;
        TE::store_l_full(t, j, t_val);
    }

    debug_assert!(carry_high == 0 || carry_low <= 1); // Replicated from above.
    debug_assert!(
        cond_mul.map(|c| c.unwrap_u8()).unwrap_or(1) == 0
            || t_carry <= 1 // Replicated function entry invariant.
    );
    let (mut t_carry, carry_low) = ct_add_l_l(carry_low, t_carry);

    // Do not update t's potentially partial high limb with a value that could overflow in the
    // course of the reduction. Return it separately in a t_high_shadow shadow instead.
    let t_high_shadow = carry_low;
    debug_assert!(t_carry == 0 || carry_high == 0); // Property of addition.
    t_carry += carry_high;
    debug_assert!(t_carry <= 1); // Loop invariant still maintained.

    (t_carry, t_high_shadow)
}


pub fn mp_ct_montgomery_redc<TE: MPEndianess, NE: MPEndianess>(t: &mut [u8], n: &[u8], n0_inv_mod_l: LimbType) {
    debug_assert!(!n.is_empty());
    debug_assert!(t.len() >= n.len());
    debug_assert!(t.len() <= 2 * n.len());
    let t_nlimbs = mp_ct_nlimbs(t.len());
    let n_nlimbs = mp_ct_nlimbs(n.len());
    let n0_val = NE::load_l(&n, 0);
    debug_assert!(n0_val.wrapping_mul(n0_inv_mod_l) == 1);
    let neg_n0_inv_mod_l = !n0_inv_mod_l + 1;

    let mut reduced_t_carry = 0;
    // t's high limb might be a partial one, do not update directly in the course of reducing in
    // order to avoid overflowing it. Use a shadow instead.
    let mut t_high_shadow = TE::load_l(t, t_nlimbs - 1);
    for _i in 0..n_nlimbs {
        (reduced_t_carry, t_high_shadow) =
            mp_ct_montgomery_redc_one_cond_mul::<TE, NE>(reduced_t_carry, t_high_shadow, t, n, neg_n0_inv_mod_l, None);
    }

    // Now apply the high limb shadow back.
    let t_high_npartial = t.len() % LIMB_BYTES;
    if t_high_npartial == 0 {
        TE::store_l_full(t, t_nlimbs - 1, t_high_shadow);
    } else {
        assert_eq!(reduced_t_carry, 0);
        reduced_t_carry = t_high_shadow >> 8 * t_high_npartial;
        t_high_shadow &= (1 << 8 * t_high_npartial) - 1;
        TE::store_l(t, t_nlimbs - 1, t_high_shadow);
    }

    mp_ct_sub_cond_mp_mp::<TE, NE>(t, n, ct_l_to_subtle_choice(reduced_t_carry) | mp_ct_geq_mp_mp::<TE, NE>(t, n));
    debug_assert!(mp_ct_geq_mp_mp::<TE, NE>(t, n).unwrap_u8() == 0);
}

#[cfg(test)]
fn test_mp_ct_montgomery_redc<TE: MPEndianess, NE: MPEndianess>() {
    use super::limbs_buffer::mp_find_last_set_byte_mp;

    for i in 0..64 {
        const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
        let n_high = MERSENNE_PRIME_13.wrapping_mul((16385 as LimbType).wrapping_mul(i));
        for j in 0..64 {
            const MERSENNE_PRIME_17: LimbType = 131071 as LimbType;
            let n_low = MERSENNE_PRIME_17.wrapping_mul((1023 as LimbType).wrapping_mul(j));
            // Force n_low odd.
            let n_low = n_low | 1;
            let mut n: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
            NE::store_l(&mut n, 0, n_low);
            NE::store_l(&mut n, 1, n_high);
            //let (_, n) = NE::split_at(&n, mp_find_last_set_byte_mp::<NE>(&n));
            let n = n.as_slice();
            let n0_inv = mp_ct_montgomery_n0_inv_mod_l::<NE>(n);

            for k in 0..8 {
                let t_high = MERSENNE_PRIME_17.wrapping_mul((8191 as LimbType).wrapping_mul(k));
                for l in 0..8 {
                    let t_low = MERSENNE_PRIME_13.wrapping_mul((131087 as LimbType).wrapping_mul(l));

                    let mut t: [u8; 2 * LIMB_BYTES + 1] = [0; 2 * LIMB_BYTES + 1];
                    TE::store_l(&mut t, 0, t_low);
                    TE::store_l(&mut t, 1, t_high);

                    // All montgomery operations are defined mod n, compute t mod n
                    mp_ct_div_mp_mp::<TE, NE, TE>(None, &mut t, n, None).unwrap();
                    let t_low = TE::load_l(&t, 0);
                    let t_high = TE::load_l(&t, 1);

                    // To Montgomery form: t * R mod n
                    let mut scratch: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                    mp_ct_to_montgomery_form::<TE, NE>(&mut t, &mut scratch, n);

                    // And back to normal: (t * R mod n) / R mod n
                    mp_ct_montgomery_redc::<TE, NE>(&mut t, n, n0_inv);
                    assert_eq!(TE::load_l(&t, 0), t_low);
                    assert_eq!(TE::load_l(&t, 1), t_high);
                }
            }
        }
    }
}

#[test]
fn test_mp_ct_montgomery_redc_be_be() {
    test_mp_ct_montgomery_redc::<MPBigEndianOrder, MPBigEndianOrder>()
}

#[test]
fn test_mp_ct_montgomery_redc_le_le() {
    test_mp_ct_montgomery_redc::<MPLittleEndianOrder, MPLittleEndianOrder>()
}

fn mp_ct_montgomery_mul_mod_cond<RE: MPEndianess, E0: MPEndianess, E1: MPEndianess, NE: MPEndianess>
                                 (result: &mut [u8], op0: &[u8], op1: &[u8], n: &[u8], n0_inv_mod_l: LimbType,
                                  cond: subtle::Choice)
{
    // This is an implementation of the "Finely Integrated Operand Scanning (FIOS) Method"
    // approach to fused multiplication and Montgomery reduction, as described in "Analyzing
    // and Comparing Montgomery Multiplication Algorithm", IEEE Micro, 16(3):26-33, June 1996.
    debug_assert!(!n.is_empty());
    debug_assert_eq!(result.len(), n.len());
    debug_assert!(op0.len() <= n.len());
    debug_assert!(op1.len() <= n.len());

    let op0_nlimbs = mp_ct_nlimbs(op0.len());
    let op1_nlimbs = mp_ct_nlimbs(op1.len());
    if op1_nlimbs == 0 {
        // The product is zero, but if cond == 0, result is supposed to receive op0 instead of the
        // product. And it must be done in constant-time.
        for i in 0..op0_nlimbs {
            let op0_val = E0::load_l(op0, i);
            let result_val = LimbType::conditional_select(&op0_val, &0, cond);
            RE::store_l(result, i, result_val);
        }
        RE::zeroize_bytes_above(result, op0.len());
        return;
    }

    let n_nlimbs = mp_ct_nlimbs(n.len());
    let n0_val = NE::load_l(&n, 0);
                                     debug_assert!(n0_val.wrapping_mul(n0_inv_mod_l) == 1);
    let neg_n0_inv_mod_l = !n0_inv_mod_l + 1;

    RE::zeroize_bytes_above(result, 0);
    let mut result_carry = 0;
    // result's high limb might be a partial one, do not update directly in the course of reducing in
    // order to avoid overflowing it. Use a shadow instead.
    let mut result_high_shadow = 0;
    for i in 0..op0_nlimbs {
        debug_assert!(result_carry <= 1); // Loop invariant.
        let op0_val = E0::load_l(op0, i);
        result_carry = LimbType::conditional_select(&op0_val, &result_carry, cond);
        let m;
        let mut carry_low;
        let mut carry_high;
        (m, carry_high, carry_low) = {
            // Do not read the potentially partial, stale high limb directly from result, use the
            // result_high_shadow shadow instead.
            let result_val = if n_nlimbs != 1 {
                RE::load_l_full(result, 0)
            } else {
                result_high_shadow
            };

            let op1_val = E1::load_l(op1, 0);
            let op1_val = LimbType::conditional_select(&0, &op1_val, cond);
            let prod = ct_mul_l_l(op1_val, op0_val);
            let (carry0, result_val) = ct_add_l_l(result_val, prod.low());

            let m = result_val.wrapping_mul(neg_n0_inv_mod_l);
            let m = LimbType::conditional_select(&0, &m, cond);

            let n_val = NE::load_l(n, 0);
            let m_n_val = ct_mul_l_l(m, n_val);
            let (carry1, result_val) = ct_add_l_l(result_val, m_n_val.low());

            debug_assert_eq!(result_val, 0);
            debug_assert!(prod.high() <= !1);
            let carry_low = prod.high() + carry0; // Does not wrap.
            let (carry_high, carry_low) = ct_add_l_l(
                carry_low,
                m_n_val.high() + carry1  // Does not wrap.
            );
            (m, carry_high, carry_low)
        };

        for j in 0..(op1_nlimbs - 1) {
            debug_assert!(carry_high <= 1); // Loop invariant LI0.
            debug_assert!(carry_high == 0 || carry_low <= !1); // Loop invariant LI1.
            let op1_val = E1::load_l(op1, j + 1);
            let op1_val = LimbType::conditional_select(&0, &op1_val, cond);
            let prod = ct_mul_l_l(op1_val, op0_val);
            let n_val = NE::load_l(n, j + 1);
            let m_n_val = ct_mul_l_l(m, n_val);

            // Do not read the potentially partial, stale high limb directly from result, use the
            // result_high_shadow shadow instead.
            let mut result_val = if j + 1 != n_nlimbs - 1 {
                RE::load_l_full(result, j + 1)
            } else {
                result_high_shadow
            };
            (carry_low, result_val) = ct_add_l_l(result_val, carry_low);
            carry_low += carry_high;
            debug_assert!(carry_low <= 1 || result_val <= !2); // LI1.a: Property of addition and from LI1.
            debug_assert!(carry_low <= 2); // Trivial, for documentation only.

            let carry0;
            (carry0, result_val) = ct_add_l_l(result_val, prod.low());
            debug_assert!(prod.high() <= !2 || prod.low() <= 1); // Property of product.
            // LI1.b: Adding prod.low() <= 1 to result_val <= !2 in LI1.a would not wrap and yield a
            // value <= !1.
            debug_assert!(prod.high() <= !2 || carry_low <= 1 || (carry0 == 0 && result_val <= !1));
            // C0: In particular, if carry_low == 2, then carry0 == 0.
            debug_assert!(prod.high() <= !2 || carry_low + carry0 <= 2);
            debug_assert!(carry0 == 0 || result_val <= !1); // Property of addition.

            let carry1;
            (carry1, result_val) = ct_add_l_l(result_val, m_n_val.low());
            debug_assert!(m_n_val.high() <= !2 || m_n_val.low() <= 1); // Property of product.
            // Adding m_n_val.low() <= 1 to result_val <= !1 would not wrap.
            debug_assert!(m_n_val.high() <= !2 || carry0 == 0 || carry1 == 0);
            // C1: Or written differently:
            debug_assert!(m_n_val.high() <= !2 || carry0 + carry1 <= 1);
            // C2: Adding m_n_val.low() <= 1 to result_val <= !1 in LI1.b from above would not wrap.
            debug_assert!(
                m_n_val.high() <= !2 ||
                    (prod.high() <= !2 || carry_low <= 1 || (carry0 == 0 && carry1 == 0))
            );

            debug_assert!(j < n_nlimbs - 1);
            RE::store_l_full(result, j, result_val);

            // Finally compute the carries as
            // prod.high() + m_n_val.high() + carry_low + carry0 + carry1.
            // Remember that prod.high(), m_n_val.high() <= !1 always as a basic property of the
            // products and moreover, from the conclusions drawn above:
            // - from C0:
            debug_assert!(prod.high() <= !2 || carry_low + carry0 + carry1 <= 3);
            // - from C1:
            debug_assert!(m_n_val.high() <= !2 || carry_low + carry0 + carry1 <= 3);
            // - from C2 (and C1):
            debug_assert!(m_n_val.high() <= !2 || prod.high() <= !2 || carry_low + carry0 + carry1 <= 2);
            // That is, the resulting sum computed below will be <= (1, !1) in either of
            // the possible combinations of prod.high() and m_n_val.high() being <= !2 or not.
            let carry2;
            (carry2, carry_low) = ct_add_l_l(
                carry_low,
                prod.high() + carry0 // Does not wrap.
            );
            let carry3;
            (carry3, carry_low) = ct_add_l_l(
                carry_low,
                m_n_val.high() + carry1 // Does not wrap.
            );
            carry_high = carry2 + carry3;
            debug_assert!(carry_high <= 1); // Loop invariant LI0 is maintained.
            debug_assert!(carry_high == 0 || carry_low <= !1); // Loop invariant LI1 is maintained.
        }

        // If op1_nlimbs < n_nlimbs, handle the rest by only adding the tail of m * n to it.
        for j in (op1_nlimbs - 1)..(n_nlimbs - 1) {
            debug_assert!(carry_high <= 1); // Loop invariant LI0.
            debug_assert!(carry_high == 0 || carry_low <= !1); // Loop invariant LI1.
            let n_val = NE::load_l(n, j + 1);
            let m_n_val = ct_mul_l_l(m, n_val);

            // Do not read the potentially partial, stale high limb directly from result, use the
            // result_high_shadow shadow instead.
            let mut result_val = if j + 1 != n_nlimbs - 1 {
                RE::load_l_full(result, j + 1)
            } else {
                result_high_shadow
            };
            (carry_low, result_val) = ct_add_l_l(result_val, carry_low);
            carry_low += carry_high;
            debug_assert!(carry_low <= 2);

            let carry0;
            (carry0, result_val) = ct_add_l_l(result_val, m_n_val.low());
            RE::store_l_full(result, j, result_val);
            debug_assert!(m_n_val.high() <= !1); // Property of product.

            (carry_high, carry_low) = ct_add_l_l(
                carry_low,
                m_n_val.high() + carry0 // Does not wrap.
            );

            // Loop invariants LI0  and LI1 are maintained trivially as per properties
            // of wrapping addition.
            debug_assert!(carry_high <= 1);
            debug_assert!(carry_high == 0 || carry_low <= !1);
        }

        // Finally, handle the most significant limb, it's effectively the sum
        // of the previously computed result_carry and this loop iteration's final
        // (carry_high, carry_low).
        debug_assert!(cond.unwrap_u8() == 0 || result_carry <= 1); // Outer loop's invariant replicated.
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

    // If op0_nlimbs < n_nlimbs, handle the rest by REDCing it.
    for _i in op0_nlimbs..n_nlimbs {
        (result_carry, result_high_shadow) = mp_ct_montgomery_redc_one_cond_mul::<RE, NE>(
            result_carry, result_high_shadow, result,
            n, neg_n0_inv_mod_l, Some(cond)
        );
    }

    // Now apply the high limb shadow back.
    debug_assert!(result.len() == n.len());
    let result_high_npartial = n.len() % LIMB_BYTES;
    if result_high_npartial == 0 {
        RE::store_l_full(result, n_nlimbs - 1, result_high_shadow);
    } else {
        assert_eq!(result_carry, 0);
        result_carry = result_high_shadow >> 8 * result_high_npartial;
        result_high_shadow &= (1 << 8 * result_high_npartial) - 1;
        RE::store_l(result, n_nlimbs - 1, result_high_shadow);
    }

    mp_ct_sub_cond_mp_mp::<RE, NE>(
        result, n, ct_l_to_subtle_choice(result_carry) | mp_ct_geq_mp_mp::<RE, NE>(result, n)
    );
    debug_assert!(mp_ct_geq_mp_mp::<RE, NE>(result, n).unwrap_u8() == 0);
}

#[cfg(test)]
fn test_mp_ct_montgomery_mul_mod_cond<RE: MPEndianess, E0: MPEndianess, E1: MPEndianess, NE: MPEndianess>() {
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
            NE::store_l(&mut n, 0, n_low);
            NE::store_l(&mut n, 1, n_high);
            for n_len in [2 * LIMB_BYTES - 1, 2 * LIMB_BYTES] {
                let (_, n) = NE::split_at(&n, n_len);
                let n0_inv = mp_ct_montgomery_n0_inv_mod_l::<NE>(n);

                // r_mod_n = 2^(2 * LIMB_BITS) % n.
                let mut r_mod_n: [u8; 3 * LIMB_BYTES] = [0; 3 * LIMB_BYTES];
                RE::store_l_full(&mut r_mod_n, 2, 1);
                mp_ct_div_mp_mp::<RE, NE, RE>(None, &mut r_mod_n, n, None).unwrap();
                let (_, r_mod_n) = RE::split_at(&r_mod_n, n.len());

                for k in 0..4 {
                    let a_high = MERSENNE_PRIME_17.wrapping_mul((16383 as LimbType).wrapping_mul(k));
                    for l in 0..4 {
                        let a_low = MERSENNE_PRIME_13.wrapping_mul((262175 as LimbType).wrapping_mul(l));
                        let mut a: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                        E0::store_l(&mut a, 0, a_low);
                        E0::store_l(&mut a, 1, a_high);
                        // All montgomery operations are defined mod n, compute a mod n
                        mp_ct_div_mp_mp::<E0, NE, E0>(None, &mut a, n, None).unwrap();
                        for s in 0..4 {
                            let b_high = MERSENNE_PRIME_13.wrapping_mul((262175 as LimbType).wrapping_mul(s));
                            for t in 0..4 {
                                const MERSENNE_PRIME_19: LimbType = 524287 as LimbType;
                                let b_low = MERSENNE_PRIME_19.wrapping_mul((4095 as LimbType).wrapping_mul(t));
                                let mut b: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                                E1::store_l(&mut b, 0, b_low);
                                E1::store_l(&mut b, 1, b_high);
                                // All montgomery operations are defined mod n, compute b mod n
                                mp_ct_div_mp_mp::<E1, NE, E1>(None, &mut b, n, None).unwrap();

                                for op_len in [0, 1 * LIMB_BYTES, n_len] {
                                    let (_, a) = E0::split_at(&a, op_len);
                                    let (_, b) = E1::split_at(&b, op_len);

                                    let mut result: [u8; 4 * LIMB_BYTES] = [0; 4 * LIMB_BYTES];
                                    let (_, mg_mul_result) = RE::split_at_mut(&mut result, n_len);
                                    mp_ct_montgomery_mul_mod_cond::<RE, E0, E1, NE>(
                                        mg_mul_result, &a, &b, n, n0_inv,
                                        subtle::Choice::from(0u8)
                                    );
                                    assert_eq!(RE::split_at(mg_mul_result, op_len).1, a);

                                    mp_ct_montgomery_mul_mod_cond::<RE, E0, E1, NE>(
                                        mg_mul_result, a, b, n, n0_inv,
                                        subtle::Choice::from(1u8)
                                    );

                                    // For testing against the expected result computed using the
                                    // "conventional" methods only, multiply by r_mod_n -- this avoids
                                    // having to multiply the conventional product by r^-1 mod n, which is
                                    // not known without implementing Euklid's algorithm.
                                    mp_ct_mul_trunc_cond_mp_mp::<RE, RE>(
                                        &mut result, n.len(), r_mod_n, subtle::Choice::from(1)
                                    );
                                    mp_ct_div_mp_mp::<RE, NE, RE>(None, &mut result, n, None).unwrap();

                                    let mut expected: [u8; 4 * LIMB_BYTES] = [0; 4 * LIMB_BYTES];
                                    RE::copy_from::<E0>(&mut expected, a);
                                    mp_ct_mul_trunc_cond_mp_mp::<RE, E1>(
                                        &mut expected, op_len, b, subtle::Choice::from(1)
                                    );
                                    mp_ct_div_mp_mp::<RE, NE, RE>(None, &mut expected, n, None).unwrap();

                                    assert_eq!(result, expected);
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
    test_mp_ct_montgomery_mul_mod_cond::<MPBigEndianOrder, MPBigEndianOrder, MPBigEndianOrder, MPBigEndianOrder>()
}

#[test]
fn test_mp_ct_montgomery_mul_mod_cond_le_le_le_le() {
    test_mp_ct_montgomery_mul_mod_cond::<MPLittleEndianOrder, MPLittleEndianOrder, MPLittleEndianOrder, MPLittleEndianOrder>()
}
