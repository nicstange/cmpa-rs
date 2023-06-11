use core::ops::Deref as _;
use subtle::{self, ConditionallySelectable as _};
use crate::limb::ct_l_to_subtle_choice;
use crate::limbs_buffer::mp_be_store_l;

use super::limb::{LimbType, LIMB_BYTES, DoubleLimb,
                  ct_eq_l_l, ct_gt_l_l, ct_add_l_l, ct_sub_l_l, ct_mul_l_l, ct_div_dl_l, CtDivDlLNormalizedDivisor};
use super::limbs_buffer::{CompositeLimbsMutBuffer, mp_be_load_l, mp_be_load_l_full, mp_ct_nlimbs};
use super::zeroize::Zeroizing;

#[derive(Debug)]
pub enum CtMpDivisionError {
    DivisionByZero,
    InsufficientQuotientSpace,
}

pub fn mp_ct_div(u_h: Option<&mut [u8]>, u_l: &mut [u8], v: &[u8], mut q_out: Option<&mut [u8]>) -> Result<(), CtMpDivisionError> {
    // Division algorithm according to D. E. Knuth, "The Art of Computer Programming", vol 2.
    //
    // Find the index of the highest set limb in v. For divisors, constant time evaluation doesn't
    // really matter, probably. Also, the long division algorithm's runtime depends highly on the
    // divisor's length anyway.
    let mut v_nlimbs = mp_ct_nlimbs(v.len());
    while v_nlimbs > 0 {
        let limb = mp_be_load_l(v, v_nlimbs - 1);
        if limb != 0 {
            break;
        }
        v_nlimbs -= 1;
    }
    let v_nlimbs = v_nlimbs;
    if v_nlimbs == 0 {
        return Err(CtMpDivisionError::DivisionByZero);
    }
    let v_high = mp_be_load_l(v, v_nlimbs - 1);
    // Determine the effective length of v in terms of bytes.
    let mut v_len = 1;
    for i in 1..LIMB_BYTES {
        if v_high >> 8 * i != 0 {
            v_len += 1;
        }
    }
    let v_len = v_len + (v_nlimbs - 1) * LIMB_BYTES;

    // If u_h is None, set it to an empty slice for code uniformity.
    let mut _u_h: [u8; 0] = [0; 0];
    let u_h = u_h.unwrap_or(&mut _u_h);
    let u_len = u_l.len() + u_h.len();
    if u_len < v_len {
        if let Some(q_out) = q_out {
            q_out.fill(0);
        }
        return Ok(())
    }
    let u_nlimbs = mp_ct_nlimbs(u_len);
    let q_out_len = u_len - v_len + 1;
    let q_out_nlimbs = mp_ct_nlimbs(q_out_len);
    if let Some(q_out) = &mut q_out {
        // Check that q_out has enough space for storing the maximum possible quotient.
        if q_out.len() < q_out_len {
            return Err(CtMpDivisionError::InsufficientQuotientSpace);
        }
        q_out.fill(0);
    };

    // Create a padding buffer extending u at its more significant end:
    // - ensure that the resulting length aligns to LIMB_BYTES and
    // - allocate an extra limb to provide sufficient space for the
    //   scaling below.
    let u_pad_len = if u_len % LIMB_BYTES == 0 {
        0
    } else {
        LIMB_BYTES - u_len % LIMB_BYTES
    };
    let mut _u_pad: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let u_pad = &mut _u_pad[0..LIMB_BYTES + u_pad_len];
    let mut u_parts =  CompositeLimbsMutBuffer::new([u_l, u_h, u_pad]);

    // Normalize divisor's high limb. Calculate 2^LIMB_BITS / (v_high + 1)
    let scaling = {
        // Be careful to avoid overflow in calculating v_high + 1. The subsequent code below
        // still returns the correct result if the increment is skipped in this case.
        let den = v_high + LimbType::conditional_select(&1, &0, ct_eq_l_l(v_high, !0));

        // First calculate (2^LIMB_BITS - 1) / (v_high + 1).
        let q = !0 / den;
        let rem = !0 - den * q;
        // And possibly round up to get 2^LimbType::BITS / (v_high + 1).
        // Note that the test below is equivalent to rem + 1 == v_high + 1.
        q + LimbType::conditional_select(&0, &1, ct_eq_l_l(rem, v_high))
    };

    // Read-only v won't get scaled in-place, but on the fly as needed.
    // For now, multiply v by scaling only to calculate the scaled v_high.
    // Note that multiplying v by scaling will not overflow the width of v:
    // b = scaling * (v_high + 1) + rest,
    // b >= scaling * (v_high + 1)
    // The claim follows by interpreting the tail of v as a fractional number < 1.
    let mut carry = 0;
    let mut scaled_v_tail_high = 0; // scaled v[n - 2]
    for i in 0..v_nlimbs - 1 {
        let scaled: Zeroizing<DoubleLimb> = ct_mul_l_l(mp_be_load_l_full(v, i), scaling).into();
        let (carry0, scaled_v_low) = ct_add_l_l(scaled.low(), carry);
        scaled_v_tail_high = scaled_v_low;
        let (carry1, carry0) = ct_add_l_l(scaled.high(), carry0);
        carry = carry0;
        debug_assert_eq!(carry1, 0);
    }
    let scaled_v_high = v_high * scaling;
    let (carry, scaled_v_high) = ct_add_l_l(scaled_v_high, carry);
    debug_assert_eq!(carry, 0);
    drop(v_high);

    // Scale u.
    let mut carry = 0;
    for i in 0..u_nlimbs + 1 {
        let scaled: Zeroizing<DoubleLimb> = ct_mul_l_l(u_parts.load(i), scaling).into();
        let (carry0, scaled_low) = ct_add_l_l(scaled.low(), carry);
        u_parts.store(i, scaled_low);
        debug_assert!(scaled.high() < !0);
        carry = carry0 + scaled.high();
        debug_assert!(carry >= carry0); // Overflow cannot happen.
    }
    // The extra high limb in u_h_pad, initialized to zero, would have absorbed the last carry.
    debug_assert_eq!(carry, 0);

    let u_sub_scaled_qv_at = |u_parts: &mut  CompositeLimbsMutBuffer<'_, 3>, j: usize, q: LimbType| -> LimbType {
        let mut scaled_v_carry = 0;
        let mut qv_carry = 0;
        let mut u_borrow = 0;
        for i in 0..v_nlimbs {
            let scaled_v: Zeroizing<DoubleLimb> = ct_mul_l_l(mp_be_load_l(v, i), scaling).into();
            let (carry0, scaled_v_low) = ct_add_l_l(scaled_v.low(), scaled_v_carry);
            let (carry1, carry0) = ct_add_l_l(scaled_v.high(), carry0);
            scaled_v_carry = carry0;
            debug_assert_eq!(carry1, 0);

            let qv: Zeroizing<DoubleLimb> = ct_mul_l_l(scaled_v_low, q).into();
            let (carry0, qv_low) = ct_add_l_l(qv.low(), qv_carry);
            let (carry1, carry0) = ct_add_l_l(qv.high(), carry0);
            qv_carry = carry0;
            debug_assert_eq!(carry1, 0);

            let u = u_parts.load(j + i);
            let (borrow0, u) = ct_sub_l_l(u, u_borrow);
            let (borrow1, u) = ct_sub_l_l(u, qv_low);
            u_borrow = borrow0 + borrow1;
            debug_assert!(u_borrow <= 1);
            u_parts.store(j + i, u);
        }
        debug_assert_eq!(scaled_v_carry, 0);

        let u = u_parts.load(j + v_nlimbs);
        let (borrow0, u) = ct_sub_l_l(u, u_borrow);
        let (borrow1, u) = ct_sub_l_l(u, qv_carry);
        u_borrow = borrow0 + borrow1;
        debug_assert!(u_borrow <= 1);
        u_parts.store(j + v_nlimbs, u);
        u_borrow
    };

    let u_cond_add_scaled_v_at = |u_parts: &mut  CompositeLimbsMutBuffer<'_, 3>, j: usize, cond: subtle::Choice| -> LimbType {
        let mut scaled_v_carry = 0;
        let mut u_carry = 0;
        for i in 0..v_nlimbs {
            let scaled_v: Zeroizing<DoubleLimb> = ct_mul_l_l(mp_be_load_l(v, i), scaling).into();
            let (carry0, scaled_v_low) = ct_add_l_l(scaled_v.low(), scaled_v_carry);
            let (carry1, carry0) = ct_add_l_l(scaled_v.high(), carry0);
            scaled_v_carry = carry0;
            debug_assert_eq!(carry1, 0);

            let scaled_v_low = LimbType::conditional_select(&0, &scaled_v_low, cond);

            let u = u_parts.load(j + i);
            let (carry0, u) = ct_add_l_l(u, u_carry);
            let (carry1, u) = ct_add_l_l(u, scaled_v_low);
            u_carry = carry0 + carry1;
            debug_assert!(u_carry <= 1);
            u_parts.store(j + i, u);
        }
        debug_assert_eq!(scaled_v_carry, 0);

        let u = u_parts.load(j + v_nlimbs);
        let (u_carry, u) = ct_add_l_l(u, u_carry);
        u_parts.store(j + v_nlimbs, u);
        u_carry
    };

    let normalized_scaled_v_high: Zeroizing<CtDivDlLNormalizedDivisor> =
        CtDivDlLNormalizedDivisor::new(scaled_v_high).into();
    let mut j = q_out_nlimbs;
    while j > 0 {
        j -= 1;
        let q = {
            let u_h = u_parts.load(v_nlimbs + j);
            let u_l = u_parts.load(v_nlimbs + j - 1);

            let (q, r) = |(q, r): (DoubleLimb, LimbType)| -> (Zeroizing<DoubleLimb> , LimbType) {
                (q.into(), r)
            }(ct_div_dl_l(&DoubleLimb::new(u_h, u_l), &normalized_scaled_v_high));
            debug_assert!(q.high() <= 1); // As per the normalization of v_high.
            // If q.high() is set, q needs to get capped to fit single limb
            // and r adjusted accordingly.
            //
            // For determining the adjusted r, note that if q.high() is set,
            // then then u_h == scaled_v_high.
            // To see this, observe that u_h >= scaled_v_high holds trivially.
            //
            // OTOH, the invariant throughout the loop over j is that u[j+n:j] / v[n-1:0] < b,
            // from which it follows that u_h <= scaled_v_high.
            // Assume not, i.e. u_h >= v_h + 1. We have v < (v_h + 1) * b^(n - 1).
            // It would follow that
            // u[j+n:j] >= (u_h * b + u_l) * b^(n - 1)
            //          >= u_h * b * b^(n - 1) >= (v_h + 1) * b * b^(n - 1)
            //          >  v * b,
            // a contradiction to the loop invariant.
            //
            // Thus, in summary, if q.high() is set, then u_h == scaled_v_high.
            //
            // It follows that the adjusted r for capping q to q == b - 1 equals
            // u_h * b + u_l - (b - 1) * v_h
            // = v_h * b + u_l - (b - 1) * v_h = v_h + u_l.
            debug_assert!(q.high() == 0 || u_h == scaled_v_high);
            debug_assert_eq!(q.high() & !1, 0); // At most LSB is set
            let ov = ct_l_to_subtle_choice(q.high());
            let q = LimbType::conditional_select(&q.low(), &!0, ov);
            let (r_carry_on_ov, r_on_ov) = ct_add_l_l(u_l, scaled_v_high);
            let r = LimbType::conditional_select(&r, &r_on_ov, ov);
            debug_assert_eq!(r_carry_on_ov & !1, 0); // At most LSB is set
            let r_carry = ov & ct_l_to_subtle_choice(r_carry_on_ov);

            // Now, as long as r does not overflow b, i.e. a LimbType,
            // check whether q * v[n - 2] > b * r + u[j + n - 2].
            // If so, decrement q and adjust r accordingly by adding v[n-1] back.
            // Note that because v[n-1] is normalized to have its MSB set,
            // r would overflow in the second iteration at latest.
            // The second iteration is not necessary for correctness, but only serves
            // optimization purposes: it would help to avoid the "add-back" step
            // below in the majority of cases. However, for constant-time execution,
            // the add-back must get executed anyways and thus, the second iteration
            // of the "over-estimated" check here would be quite pointless. Skip it.
            //
            // Load u[u[j + n - 2] first. If v_nlimbs < 2 and j == 0, it might not be defined. But
            // in this case scaled_v_tail_high is zero anyway and the comparison test will always
            // come out negative, so load an arbitrary value then.
            let u_tail_high = if v_nlimbs + j >= 2 {
                u_parts.load(v_nlimbs + j - 2)
            } else {
                0
            };
            // First iteration of the test.
            let qv_tail_high: Zeroizing<DoubleLimb> = ct_mul_l_l(scaled_v_tail_high, q).into();
            let over_estimated = !r_carry &
                (ct_gt_l_l(qv_tail_high.high(), r) |
                 (ct_eq_l_l(qv_tail_high.high(), r) & ct_gt_l_l(qv_tail_high.low(), u_tail_high)));
            LimbType::conditional_select(&q, &q.wrapping_sub(1), over_estimated)
        };

        let borrow = u_sub_scaled_qv_at(&mut u_parts, j, q);
        debug_assert_eq!(borrow & !1, 0); // At most LSB is set
        let over_estimated = ct_l_to_subtle_choice(borrow);
        u_cond_add_scaled_v_at(&mut u_parts, j, over_estimated);
        if let Some(q_out) = &mut q_out {
            let q = LimbType::conditional_select(&q, &q.wrapping_sub(1), over_estimated);
            mp_be_store_l(q_out, j, q);
        }
    }

    // Finally, divide the resulting remainder in u by the scaling again.
    let mut u_h = 0;
    let mut j = u_nlimbs + 1;
    let scaling: Zeroizing<CtDivDlLNormalizedDivisor> =
        CtDivDlLNormalizedDivisor::new(scaling).into();
    while j > 0 {
        j -= 1;
        let u_l = u_parts.load(j);
        let (u, r) = |(u, r): (DoubleLimb, LimbType)| -> (Zeroizing<DoubleLimb> , LimbType) {
            (u.into(), r)
        }(ct_div_dl_l(&DoubleLimb::new(u_h, u_l), &scaling));
        debug_assert_eq!(u.high(), 0);
        u_parts.store(j, u.low());
        u_h = r;
    }

    Ok(())
}

#[test]
fn test_mp_ct_div() {
    fn div_and_check(u: &[u8], v: &[u8], split_u: bool) {
        use super::add_impl::mp_ct_add;
        use super::mul_impl::mp_ct_mul_trunc_cond;

        let v_begin = v.iter().enumerate().find(|(_, v)| **v != 0).map(|(i, _)| i).unwrap();
        let v = &v[v_begin..];
        let q_len = if u.len() >= v.len() {
            u.len() - v.len() + 1
        } else {
            0
        };
        let mut q = vec![0xffu8; q_len];
        let mut rem = u.to_vec();
        let (rem_h, rem_l) = if split_u {
            let (rem_h, rem_l) = rem.split_at_mut((LIMB_BYTES + 1).min(u.len()));
            (Some(rem_h), rem_l)
        } else {
            (None, rem.as_mut_slice())
        };
        mp_ct_div(rem_h, rem_l, v, Some(&mut q)).unwrap();

        // Multiply q by v again and add the remainder back, the result should match the initial u.
        let mut result = vec![0u8; u.len()];
        result[u.len() - q_len..].copy_from_slice(&q);
        mp_ct_mul_trunc_cond(&mut result, q_len, v, subtle::Choice::from(1));
        let carry = mp_ct_add(&mut result, &rem);
        assert_eq!(carry, 0);
        assert_eq!(u, &result);
    }

    let u = vec![1, 0];
    let v = vec![1];
    div_and_check(&u, &v, false);
    div_and_check(&u, &v, true);

    let u = vec![1, 0];
    let v = vec![3];
    div_and_check(&u, &v, false);
    div_and_check(&u, &v, true);


    let u = vec![!0, !0, !1, !0, !0, !0];
    let v = vec![!0, !0, !0];
    div_and_check(&u, &v, false);
    div_and_check(&u, &v, true);

    let u = vec![!0, !0, !1, !0, !0, !1];
    let v = vec![!0, !0, !0];
    div_and_check(&u, &v, false);
    div_and_check(&u, &v, true);

    let u = vec![0, 0, 0, 0, 0, 0];
    let v = vec![!0, !0, !0];
    div_and_check(&u, &v, false);
    div_and_check(&u, &v, true);

    let u = vec![0, 0, 0, 0, 0, !1];
    let v = vec![!0, !0, !0];
    div_and_check(&u, &v, false);
    div_and_check(&u, &v, true);

    let u = vec![!0, !0, !0, !0, !0, 0];
    let v = vec![0, 1, 0];
    div_and_check(&u, &v, false);
    div_and_check(&u, &v, true);

    let u = vec![!0, !0, !0, !0, !1, 0];
    let v = vec![0, 2, 0];
    div_and_check(&u, &v, false);
    div_and_check(&u, &v, true);

    use super::limb::LIMB_BITS;
    const N_MAX_LIMBS: u32 = 3;
    for i in 0..N_MAX_LIMBS * LIMB_BITS + 1 {
        let u_len = (i as usize + 8 - 1) / 8;
        let mut u = vec![0u8; u_len];
        if i != 0 {
            let u_nlimbs = mp_ct_nlimbs(u_len);
            for k in 0..u_nlimbs - 1 {
                mp_be_store_l(u.as_mut_slice(), k, !0);
            }
            if i % LIMB_BITS != 0 {
                let i = i % LIMB_BITS;
                mp_be_store_l(u.as_mut_slice(), u_nlimbs - 1, !0 >> (LIMB_BITS - i));
            } else  {
                mp_be_store_l(u.as_mut_slice(), u_nlimbs - 1, !0);
            }
        }

        for j1 in 0..i + 1 {
            for j2 in 0..j1 + 1 {
                let v_len = ((j1 + 1) as usize + 8 - 1) / 8;
                let mut v = vec![0u8; v_len];
                mp_be_store_l(&mut v, (j1 / LIMB_BITS) as usize, 1 << (j1 % LIMB_BITS));
                mp_be_store_l(&mut v, (j2 / LIMB_BITS) as usize, 1 << (j2 % LIMB_BITS));
                div_and_check(&u, &v, false);
                div_and_check(&u, &v, true);
            }
        }
    }
}
