use core::ops::Deref as _;
use subtle::{self, ConditionallySelectable as _};
use crate::limb::ct_l_to_subtle_choice;

use super::limb::{LimbType, LIMB_BYTES, DoubleLimb,
                  ct_eq_l_l, ct_gt_l_l, ct_add_l_l, ct_add_l_l_c, ct_sub_l_l, ct_sub_l_l_b, ct_mul_l_l, ct_div_dl_l, CtDivDlLNormalizedDivisor, ct_find_last_set_byte_l, LIMB_BITS};
use super::limbs_buffer::{CompositeLimbsBuffer, mp_ct_nlimbs,
                          mp_find_last_set_byte_mp, MPIntMutByteSlice, MPIntByteSliceCommon};
use super::shift_impl::mp_lshift_mp;
use super::zeroize::Zeroizing;

#[derive(Debug)]
pub enum MpCtDivisionError {
    DivisionByZero,
    InsufficientQuotientSpace,
    InsufficientRemainderSpace,
}

fn v_scaling(v_high: LimbType) -> LimbType {
    // Normalize divisor's high limb. Calculate 2^LIMB_BITS / (v_high + 1)
    // Be careful to avoid overflow in calculating v_high + 1. The subsequent code below
    // still returns the correct result if the increment is skipped in this case.
    let den = v_high + LimbType::conditional_select(&1, &0, ct_eq_l_l(v_high, !0));

    // First calculate (2^LIMB_BITS - 1) / (v_high + 1).
    let q = !0 / den;
    let rem = !0 - den * q;
    // And possibly round up to get 2^LimbType::BITS / (v_high + 1).
    // Note that the test below is equivalent to rem + 1 == v_high + 1.
    q + LimbType::conditional_select(&0, &1, ct_eq_l_l(rem, v_high))
}

fn scaled_v_val<VT: MPIntByteSliceCommon>(
    i: usize, scaling: LimbType, v: &VT, scaled_v_carry: LimbType) -> (LimbType, LimbType) {
    let scaled_v: Zeroizing<DoubleLimb> = ct_mul_l_l(v.load_l(i), scaling).into();
    let (carry0, scaled_v_low) = ct_add_l_l(scaled_v.low(), scaled_v_carry);
    let scaled_v_carry;
    let carry1;
    (carry1, scaled_v_carry) = ct_add_l_l(scaled_v.high(), carry0);
    debug_assert_eq!(carry1, 0);

    (scaled_v_low, scaled_v_carry)
}

fn scaled_qv_val<VT: MPIntByteSliceCommon>(
    i: usize, q: LimbType, scaling: LimbType, v: &VT, scaled_v_carry: LimbType, qv_carry: LimbType
) -> (LimbType, LimbType, LimbType) {
    let (v_val, scaled_v_carry) = scaled_v_val(i, scaling, v, scaled_v_carry);
    let qv: Zeroizing<DoubleLimb> = ct_mul_l_l(v_val, q).into();
    let (carry0, qv_low) = ct_add_l_l(qv.low(), qv_carry);
    let qv_carry;
    let carry1;
    (carry1, qv_carry) = ct_add_l_l(qv.high(), carry0);
    debug_assert_eq!(carry1, 0);

    (qv_low, scaled_v_carry, qv_carry)
}

fn v_head_scaled<VT: MPIntByteSliceCommon>(scaling: LimbType, v: &VT, v_nlimbs: usize)
                                           -> (LimbType, LimbType) {
    // Read-only v won't get scaled in-place, but on the fly as needed. For now, multiply v by
    // scaling only to calculate the two scaled head limbs of v, as are needed for the q estimates.
    // Note that multiplying v by scaling will not overflow the width of v:
    // b = scaling * (v_high + 1) + rest,
    // b >= scaling * (v_high + 1)
    // The claim follows by interpreting the tail of v as a fractional number < 1.
    let mut scaled_v_carry = 0;
    // scaled v[n - 2], if v_nlimbs == 1, it will remain zero on purpose.
    let mut scaled_v_tail_high = 0;
    for i in 0..v_nlimbs - 1 {
        (scaled_v_tail_high, scaled_v_carry) = scaled_v_val(i, scaling, v, scaled_v_carry);
    }

    let scaled_v_high;
    (scaled_v_high, scaled_v_carry) = scaled_v_val(v_nlimbs - 1, scaling, v, scaled_v_carry);
    debug_assert_eq!(scaled_v_carry, 0);
    (scaled_v_high, scaled_v_tail_high)
}

fn q_estimate(
    u_head: &[LimbType; 3],
    scaled_v_head: &[LimbType; 2],
    normalized_scaled_v_high: &CtDivDlLNormalizedDivisor) -> LimbType {
    let (q, r) = |(q, r): (DoubleLimb, LimbType)| -> (Zeroizing<DoubleLimb> , LimbType) {
        (q.into(), r)
    }(ct_div_dl_l(&DoubleLimb::new(u_head[0], u_head[1]), normalized_scaled_v_high));
    debug_assert!(q.high() <= 1); // As per the normalization of v_high.
    // If q.high() is set, q needs to get capped to fit single limb
    // and r adjusted accordingly.
    //
    // For determining the adjusted r, note that if q.high() is set,
    // then then u_head[0] == v_head[0].
    // To see this, observe that u_head[0] >= v_head[0] holds trivially.
    //
    // OTOH, the invariant throughout the caller's loop over j is that u[j+n:j] / v[n-1:0] < b, from
    // which it follows that u_head[0] <= v_head[0].
    // Assume not, i.e. u_head[0] >= v_head[0] + 1. We have v < (v_head[0] + 1) * b^(n - 1).
    // It would follow that
    // u[j+n:j] >= (u_head[0] * b + u_head[1]) * b^(n - 1)
    //          >= u_head[0] * b * b^(n - 1) >= (v_head[0] + 1) * b * b^(n - 1)
    //          >  v * b,
    // a contradiction to the loop invariant.
    //
    // Thus, in summary, if q.high() is set, then u_head[0] == v_head[0].
    //
    // It follows that the adjusted r for capping q to q == b - 1 equals
    // u_head[0] * b + u_head[1] - (b - 1) * v_head[0]
    // = v_head[0] * b + u_head[1] - (b - 1) * v_head[0] = v_head[0] + u_head[1].
    debug_assert!(q.high() == 0 || u_head[0] == scaled_v_head[0]);
    debug_assert_eq!(q.high() & !1, 0); // At most LSB is set
    let ov = ct_l_to_subtle_choice(q.high());
    let q = LimbType::conditional_select(&q.low(), &!0, ov);
    let (r_carry_on_ov, r_on_ov) = ct_add_l_l(u_head[1], scaled_v_head[0]);
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
    // If v_nlimbs < 2 and j == 0, u[j + n - 2] might not be defined. But in this case v[n-2] (found
    // in scaled_v_head[1]) is zero anyway and the comparison test will always come out negative, so the
    // caller may load arbitrary value into its corresponding location at q_head[2].
    let qv_head_low: Zeroizing<DoubleLimb> = ct_mul_l_l(scaled_v_head[1], q).into();
    let over_estimated = !r_carry &
        (ct_gt_l_l(qv_head_low.high(), r) |
         (ct_eq_l_l(qv_head_low.high(), r) & ct_gt_l_l(qv_head_low.low(), u_head[2])));
    LimbType::conditional_select(&q, &q.wrapping_sub(1), over_estimated)
}

fn u_sub_scaled_qv_at<'a, UT: MPIntMutByteSlice, VT: MPIntByteSliceCommon>(
    u_parts: &mut CompositeLimbsBuffer<'a, UT, 3>, j: usize, q: LimbType,
    v: &VT, v_nlimbs: usize, scaling: LimbType) -> LimbType {
    let mut scaled_v_carry = 0;
    let mut qv_carry = 0;
    let mut u_borrow = 0;
    for i in 0..v_nlimbs {
        let qv_val;
        (qv_val, scaled_v_carry, qv_carry) = scaled_qv_val(i, q, scaling, v, scaled_v_carry, qv_carry);
        let mut u = u_parts.load(j + i);
        (u_borrow, u) = ct_sub_l_l_b(u, qv_val, u_borrow);
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
}

fn u_cond_add_scaled_v_at<'a, UT: MPIntMutByteSlice, VT: MPIntByteSliceCommon>(
    u_parts: &mut CompositeLimbsBuffer<'a, UT, 3>, j: usize,
    v: &VT, v_nlimbs: usize, scaling: LimbType, cond: subtle::Choice
) -> LimbType {
    let mut scaled_v_carry = 0;
    let mut u_carry = 0;
    for i in 0..v_nlimbs {
        let v_val: LimbType;
        (v_val, scaled_v_carry) = scaled_v_val(i, scaling, v, scaled_v_carry);
        let v_val = LimbType::conditional_select(&0, &v_val, cond);
        let mut u = u_parts.load(j + i);
        (u_carry, u) = ct_add_l_l_c(u, v_val, u_carry);
        u_parts.store(j + i, u);
    }
    debug_assert_eq!(scaled_v_carry, 0);

    let u = u_parts.load(j + v_nlimbs);
    let (u_carry, u) = ct_add_l_l(u, u_carry);
    u_parts.store(j + v_nlimbs, u);
    u_carry
}

pub fn mp_ct_div_mp_mp<UT: MPIntMutByteSlice, VT: MPIntByteSliceCommon, QT: MPIntMutByteSlice>(
    u_h: Option<&mut UT>, u_l: &mut UT, v: &VT, mut q_out: Option<&mut QT>
) -> Result<(), MpCtDivisionError> {
    // Division algorithm according to D. E. Knuth, "The Art of Computer Programming", vol 2.
    //
    // Find the index of the highest set limb in v. For divisors, constant time evaluation doesn't
    // really matter, probably as far as the number of zero high bytes is concerned. Also, the long
    // division algorithm's runtime depends highly on the divisor's length anyway.
    let v_len = mp_find_last_set_byte_mp(v);
    if v_len == 0 {
        return Err(MpCtDivisionError::DivisionByZero);
    }
    let v_nlimbs = mp_ct_nlimbs(v_len);
    let v_high = v.load_l(v_nlimbs - 1);

    // If u_h is None, set it to an empty slice for code uniformity.
    let mut __u_h: [u8; 0] = [0; 0];
    let u_h: UT::SelfT::<'_> = match u_h {
        Some(u_h) => u_h.coerce_lifetime(),
        None => UT::from_bytes(__u_h.as_mut()).unwrap(),
    };

    let u_len = u_l.len() + u_h.len();
    let u_nlimbs = mp_ct_nlimbs(u_len);

    if let Some(q_out) = &mut q_out {
        // Check that q_out has enough space for storing the maximum possible quotient.
        if q_out.len() + v_len < u_len + 1 {
            return Err(MpCtDivisionError::InsufficientQuotientSpace);
        }
    };

    if u_len < v_len {
        if let Some(q_out) = q_out {
            q_out.zeroize_bytes_above(0);
        }
        return Ok(());
    }

    let q_out_len = u_len + 1 - v_len;
    let q_out_nlimbs = mp_ct_nlimbs(q_out_len);
    if let Some(q_out) = &mut q_out {
        q_out.zeroize_bytes_above(q_out_len);
    }

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
    let u_pad = UT::from_bytes(&mut _u_pad[0..LIMB_BYTES + u_pad_len]).unwrap();
    let u_l = u_l.coerce_lifetime();
    let mut u_parts = CompositeLimbsBuffer::new([u_l, u_h, u_pad]);

    // Normalize divisor's high limb.
    let scaling = v_scaling(v_high);

    // Read-only v won't get scaled in-place, but on the fly as needed. Calculate only its scaled
    // two high limbs, which will be needed for making the q estimates.
    let (scaled_v_high, scaled_v_tail_high) = v_head_scaled(scaling, v, v_nlimbs);
    let scaled_v_head: Zeroizing<[LimbType; 2]> = Zeroizing::from([scaled_v_high, scaled_v_tail_high]);
    let normalized_scaled_v_high: Zeroizing<CtDivDlLNormalizedDivisor> =
        CtDivDlLNormalizedDivisor::new(scaled_v_high).into();

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

    let mut j = q_out_nlimbs;
    while j > 0 {
        j -= 1;
        let q = {
            let u_h = u_parts.load(v_nlimbs + j);
            let u_l = u_parts.load(v_nlimbs + j - 1);
            // Load u[j + n - 2]. If v_nlimbs < 2 and j == 0, it might not be defined -- make it
            // zero in this case, c.f. the corresponding comment in q_estimate().
            let u_tail_high = if v_nlimbs + j >= 2 {
                u_parts.load(v_nlimbs + j - 2)
            } else {
                0
            };
            let cur_u_head: Zeroizing<[LimbType; 3]> = Zeroizing::from([u_h, u_l, u_tail_high]);

            q_estimate(&cur_u_head, &scaled_v_head, normalized_scaled_v_high.deref())
        };

        let borrow = u_sub_scaled_qv_at(&mut u_parts, j, q, v, v_nlimbs, scaling);
        debug_assert_eq!(borrow & !1, 0); // At most LSB is set
        let over_estimated = ct_l_to_subtle_choice(borrow);
        u_cond_add_scaled_v_at(&mut u_parts, j, v, v_nlimbs, scaling, over_estimated);
        if let Some(q_out) = &mut q_out {
            let q = LimbType::conditional_select(&q, &q.wrapping_sub(1), over_estimated);
            q_out.store_l(j, q);
        }
    }

    // Finally, divide the resulting remainder in u by the scaling again.
    let mut u_h = 0;
    for j in v_nlimbs..u_nlimbs + 1 {
        debug_assert_eq!(u_parts.load(j), 0);
    }
    let mut j = v_nlimbs;
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
    debug_assert_eq!(u_h, 0);

    Ok(())
}

#[cfg(test)]
fn test_limbs_from_be_bytes<DT: MPIntMutByteSlice, const N: usize>(bytes: [u8; N]) -> Vec<u8> {
    use super::limbs_buffer::{MPBigEndianByteSlice, MPIntByteSlice};
    let mut limbs = vec![0u8; DT::limbs_align_len(N)];
    let mut dst = DT::from_bytes(limbs.as_mut_slice()).unwrap();
    dst.copy_from(&MPBigEndianByteSlice::from_bytes(bytes.as_slice()).unwrap());
    drop(dst);
    limbs
}

#[cfg(test)]
fn test_mp_ct_div_mp_mp<UT: MPIntMutByteSlice, VT: MPIntMutByteSlice, QT: MPIntMutByteSlice>() {
    use super::limbs_buffer::MPIntMutByteSlicePriv as _;
    use super::cmp_impl::mp_ct_eq_mp_mp;

    fn div_and_check<UT: MPIntMutByteSlice, VT: MPIntMutByteSlice, QT: MPIntMutByteSlice>(
        u: &UT::SelfT<'_>, v: &VT::SelfT<'_>, split_u: bool
    ) {
        use super::add_impl::mp_ct_add_mp_mp;
        use super::mul_impl::mp_ct_mul_trunc_cond_mp_mp;

        let v_len = mp_find_last_set_byte_mp(v);
        let q_len = if u.len() >= v_len {
            u.len() - v_len + 1
        } else {
            0
        };
        let mut q = vec![0xffu8; QT::limbs_align_len(q_len)];
        let mut q = QT::from_bytes(&mut q).unwrap();
        let mut rem = vec![0u8; u.len()];
        let mut rem = UT::from_bytes(&mut rem).unwrap();
        rem.copy_from(u);
        let (mut rem_h, mut rem_l) = if split_u {
            let split_point = if UT::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
                LIMB_BYTES + 1
            } else {
                LIMB_BYTES
            };
            let (rem_h, rem_l) = rem.split_at(split_point.min(u.len()));
            (Some(rem_h), rem_l)
        } else {
            (None, rem.coerce_lifetime())
        };
        mp_ct_div_mp_mp(rem_h.as_mut(), &mut rem_l, v, Some(&mut q)).unwrap();
        drop(rem_h);
        drop(rem_l);

        // Multiply q by v again and add the remainder back, the result should match the initial u.
        // Reserve one extra limb, which is expected to come to zero.
        let mut result = vec![0u8; u.len() + LIMB_BYTES];
        let mut result = UT::from_bytes(&mut result).unwrap();
        result.copy_from(&q);
        mp_ct_mul_trunc_cond_mp_mp(&mut result, q_len, v, subtle::Choice::from(1));
        let carry = mp_ct_add_mp_mp(&mut result, &rem);
        assert_eq!(carry, 0);
        assert_eq!(mp_ct_eq_mp_mp(u, &result).unwrap_u8(), 1);
    }

    let mut u = test_limbs_from_be_bytes::<UT, 2>([1, 0]);
    let u = UT::from_bytes(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 1>([1]);
    let v = VT::from_bytes(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, false);
    div_and_check::<UT, VT, QT>(&u, &v, true);

    let mut u = test_limbs_from_be_bytes::<UT, 2>([1, 0]);
    let u = UT::from_bytes(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 1>([3]);
    let v = VT::from_bytes(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, false);
    div_and_check::<UT, VT, QT>(&u, &v, true);

    let mut u = test_limbs_from_be_bytes::<UT, 6>([!0, !0, !1, !0, !0, !0]);
    let u = UT::from_bytes(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 3>([!0, !0, !0]);
    let v = VT::from_bytes(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, false);
    div_and_check::<UT, VT, QT>(&u, &v, true);

    let mut u = test_limbs_from_be_bytes::<UT, 6>([!0, !0, !1, !0, !0, !1]);
    let u = UT::from_bytes(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 3>([!0, !0, !0]);
    let v = VT::from_bytes(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, false);
    div_and_check::<UT, VT, QT>(&u, &v, true);

    let mut u = test_limbs_from_be_bytes::<UT, 6>([0, 0, 0, 0, 0, 0]);
    let u = UT::from_bytes(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 3>([!0, !0, !0]);
    let v = VT::from_bytes(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, false);
    div_and_check::<UT, VT, QT>(&u, &v, true);

    let mut u = test_limbs_from_be_bytes::<UT, 6>([0, 0, 0, 0, 0, !1]);
    let u = UT::from_bytes(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 3>([!0, !0, !0]);
    let v = VT::from_bytes(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, false);
    div_and_check::<UT, VT, QT>(&u, &v, true);

    let mut u = test_limbs_from_be_bytes::<UT, 6>([!0, !0, !0, !0, !0, 0]);
    let u = UT::from_bytes(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 3>([0, 1, 0]);
    let v = VT::from_bytes(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, false);
    div_and_check::<UT, VT, QT>(&u, &v, true);

    let mut u = test_limbs_from_be_bytes::<UT, 6>([!0, !0, !0, !0, !1, 0]);
    let u = UT::from_bytes(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 3>([0, 2, 0]);
    let v = VT::from_bytes(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, false);
    div_and_check::<UT, VT, QT>(&u, &v, true);

    const N_MAX_LIMBS: u32 = 3;
    for i in 0..N_MAX_LIMBS * LIMB_BITS + 1 {
        let u_len = (i as usize + 8 - 1) / 8;
        let mut u = vec![0u8; UT::limbs_align_len(u_len)];
        let mut u = UT::from_bytes(&mut u).unwrap();
        if i != 0 {
            let u_nlimbs = mp_ct_nlimbs(u_len);
            for k in 0..u_nlimbs - 1 {
                u.store_l(k, !0);
            }
            if i % LIMB_BITS != 0 {
                let i = i % LIMB_BITS;
                u.store_l(u_nlimbs - 1, !0 >> (LIMB_BITS - i));
            } else  {
                u.store_l(u_nlimbs - 1, !0);
            }
        }

        for j1 in 0..i + 1 {
            for j2 in 0..j1 + 1 {
                let v_len = ((j1 + 1) as usize + 8 - 1) / 8;
                let mut v = vec![0u8; VT::limbs_align_len(v_len)];
                let mut v = VT::from_bytes(&mut v).unwrap();
                v.store_l((j1 / LIMB_BITS) as usize, 1 << (j1 % LIMB_BITS));
                v.store_l((j2 / LIMB_BITS) as usize, 1 << (j2 % LIMB_BITS));
                div_and_check::<UT, VT, QT>(&u, &v, false);
                div_and_check::<UT, VT, QT>(&u, &v, true);
            }
        }
    }
}

#[test]
fn test_mp_ct_div_be_be_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_div_mp_mp::<MPBigEndianMutByteSlice, MPBigEndianMutByteSlice, MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_div_le_le_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_div_mp_mp::<MPLittleEndianMutByteSlice, MPLittleEndianMutByteSlice, MPLittleEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_div_ne_ne_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_ct_div_mp_mp::<MPNativeEndianMutByteSlice, MPNativeEndianMutByteSlice, MPNativeEndianMutByteSlice>()
}

pub fn mp_ct_div_pow2_mp<RT: MPIntMutByteSlice, VT: MPIntByteSliceCommon, QT: MPIntMutByteSlice>(
    u_pow2_exp: usize, r_out: &mut RT, v: &VT,  mut q_out: Option<&mut QT>
) -> Result<(), MpCtDivisionError> {
    // Division algorithm according to D. E. Knuth, "The Art of Computer Programming", vol 2 for the
    // special case of the dividend being a power of two.
    //
    // Find the index of the highest set limb in v. For divisors, constant time evaluation doesn't
    // really matter, probably as far as the number of zero high bytes is concerned. Also, the long
    // division algorithm's runtime depends highly on the divisor's length anyway.
    let v_len = mp_find_last_set_byte_mp(v);
    if v_len == 0 {
        return Err(MpCtDivisionError::DivisionByZero);
    }
    let v_nlimbs = mp_ct_nlimbs(v_len);
    let v_high = v.load_l(v_nlimbs - 1);

    // The virtual length of the base 2 power in bytes.
    let virtual_u_len = ((u_pow2_exp + 1) + 8 - 1) / 8;
    let virtual_u_nlimbs = mp_ct_nlimbs(virtual_u_len);

    if let Some(q_out) = &mut q_out {
        // Check that q_out has enough space for storing the maximum possible quotient.
        if q_out.len() + v_len < virtual_u_len + 1 {
            return Err(MpCtDivisionError::InsufficientQuotientSpace);
        }
    };

    if r_out.len() < v_len.min(virtual_u_len) {
        return Err(MpCtDivisionError::InsufficientRemainderSpace);
    }
    r_out.zeroize_bytes_above(0);

    // The virtual high limb of the base 2 power.
    let u_high_shift = (u_pow2_exp % LIMB_BITS as usize) as u32;
    let u_high = (1 as LimbType) << u_high_shift;
    if virtual_u_len < v_len {
        r_out.store_l(virtual_u_nlimbs - 1, u_high);
        if let Some(q_out) = q_out {
            q_out.zeroize_bytes_above(0);
        }
        return Ok(());
    }

    let q_out_len = virtual_u_len + 1 - v_len;
    let q_out_nlimbs = mp_ct_nlimbs(q_out_len);
    if let Some(q_out) = &mut q_out {
        q_out.zeroize_bytes_above(q_out_len);
    };

    // Normalize divisor's high limb.
    let scaling = v_scaling(v_high);

    // Read-only v won't get scaled in-place, but on the fly as needed. Calculate only its scaled
    // two high limbs, which will be needed for making the q estimates.
    let (scaled_v_high, scaled_v_tail_high) = v_head_scaled(scaling, v, v_nlimbs);
    let scaled_v_head: Zeroizing<[LimbType; 2]> = Zeroizing::from([scaled_v_high, scaled_v_tail_high]);
    let normalized_scaled_v_high: Zeroizing<CtDivDlLNormalizedDivisor> =
        CtDivDlLNormalizedDivisor::new(scaled_v_high).into();

    // Scale u. Note that as being a power of two, only its (current) most significant high limb is non-zero.
    let scaled_u_high: Zeroizing<DoubleLimb> = ct_mul_l_l(u_high, scaling).into();
    let mut r_out_head_shadow: Zeroizing<[LimbType; 2]> = Zeroizing::from([scaled_u_high.low(), scaled_u_high.high()]);

    // Note that q_out_nlimbs as calculate above doesn't necessarily equal u_nlimbs - v_nlimbs + 1,
    // but might come out to be one less. It can be shown that q_out_nlimbs = u_nlimbs - v_nlimbs
    // implies that the scaling of u above would not have overflown into the scaled_u_high.high()
    // now found in the r_out_head_shadow[] high limb at index 1.
    //
    // [ To see this, write
    //    u.len() - 1 = q_ul * LIMB_BYTES + r_ul, with r_ul < LIMB_BYTES
    //   and
    //    v.len() - 1 = q_vl * LIMB_BYTES + r_vl, with r_vl < LIMB_BYTES.
    //
    //   With that,
    //     u_nlimbs = q_ul + 1
    //   and
    //     v_nlimbs = q_vl + 1.
    //   By definition (all subsequent divisions are meant to be integer divisions),
    //     q_out_nlimbs
    //          = (u.len() - v.len() + 1 + LIMB_BYTES - 1) / LIMB_BYTES
    //          = (u.len() - v.len()) / LIMB_BYTES + 1
    //          = (u.len() - 1 - (v.len() - 1)) / LIMB_BYTES + 1
    //          = (q_ul * LIMB_BYTES + r_ul - (q_vl * LIMB_BYTES + r_vl)) / LIMB_BYTES + 1
    //          = q_ul - q_vl - (r_vl - r_ul + LIMB_BYTES - 1) / LIMB_BYTES + 1
    //   The latter equals either
    //          = q_ul - q_vl + 1 = u_nlimbs - v_nlimbs + 1
    //   or
    //          = q_ul - q_vl     = u_nlimbs - v_nlimbs,
    //   depending on whether rv <= r_ul or not.
    //
    //   To see how this relates to the scaled u overflowing into the next higher limb or not, note
    //   that the high limb of (unscaled) u has exactly r_ul + 1 of its least signigicant bytes
    //   non-zero and similarly does the high limb of v have exactly r_vl + 1 of its least
    //   significant bytes non-zero. The latter determines the value of scaling, which won't have
    //   more than LIMB_BYTES - (r_vl + 1) + 1 = LIMB_BYTES - r_vl of the least significant bytes set:
    //   remember that the scaling is chosen such that the high limb of v multiplied by the scaling
    //   makes the high bit in the limb set, but does not overflow it.
    //   Multiplying u by the scaling extends its length by the length of the scaling at most, i.e.
    //   by LIMB_BYTES - r_vl. Before the scaling operation, u has LIMB_BYTES - (r_ul + 1) of
    //   its most signifcant bytes zero, and thus, as long as
    //   LIMB_BYTES - (r_ul + 1) >= LIMB_BYTES - r_vl,
    //   the scaling is guaranteed not to overflow into the next higher limb. Observe
    //   how this latter condition is equivalent to r_vl > r_ul, which, as shown above,
    //   is in turn equivalent to q_out_nlimbs taking the smaller of the two possible values:
    //   q_out_nlimbs = u_nlimbs - v_nlimbs. ]
    //
    // This matters insofar, as the current setup of r_out_head_shadow[] is such that the sliding
    // window approach to the division from below would start at the point in the virtual, scaled u
    // valid only for the case that the scaling did overflow, i.e. at the (virtual) limb position
    // just above the one which should have been taken for the non-overflowing case. The quotient
    // limb obtained for this initial position would come out to zero, but this superfluous
    // computation would consume one iteration from the total count of q_out_nlimbs ones actually
    // needed. Simply incrementing q_out_nlimbs to account for that would not work, as there would
    // be no slot for storing the extra zero high limb of q available in the output q_out[]
    // argument. Instead, if q_out_nlimbs is the smaller of the two possible values, i.e. equals
    // u_nlimbs - v_nlimbs, tweak r_out_head_shadow[] to the state it would have had after one (otherwise
    // superfluous) initial iteration of the division loop.
    debug_assert!(
        v_nlimbs + q_out_nlimbs == virtual_u_nlimbs + 1 ||
            v_nlimbs + q_out_nlimbs == virtual_u_nlimbs
    );
    if v_nlimbs + q_out_nlimbs == virtual_u_nlimbs {
        debug_assert_eq!(r_out_head_shadow[1], 0);
        r_out_head_shadow[1] = r_out_head_shadow[0];
        r_out_head_shadow[0] = 0;
    }

    // For the division loop, (u_h, u_l, r_out[v_nlimbs - 3:0]) acts as a sliding window over the
    // v_nlimbs most significant limbs of the dividend, which is known to have its remaining tail equal
    // all zeroes. As the loop progresses, the sliding window gets extended on the right by
    // a virtual zero to construct the v_nlimbs + 1 dividend for a single division step.
    // After the division step, the the most significant limb is known to haven been made
    // zero as per the basic long division algorithm's underlying principle. That is, it can
    // be removed from the left, thereby effectively moving the sliding window one limb to
    // the right.
    let mut j = q_out_nlimbs;
    while j > 0 {
        j -= 1;
        let q = {
            // Load u[j + n - 2] for the q estimate.
            // - If v_nlimbs <= 2, it's the zero shifted in from the right.
            // - If v_nlimbs < 2 and j == 0, it's actually undefined, but as per the comment
            //   in q_estimate(), its value can be set to an arbitrary value in this case,
            //   including to zero.
            let u_tail_high = if v_nlimbs > 2 {
                r_out.load_l_full(v_nlimbs - 2 - 1)
            } else {
                0
            };
            let cur_u_head: Zeroizing<[LimbType; 3]> = Zeroizing::from([r_out_head_shadow[1], r_out_head_shadow[0], u_tail_high]);

            q_estimate(&cur_u_head, &scaled_v_head, normalized_scaled_v_high.deref())
        };

        // Virtually extend the v_nlimbs-limb sliding window by a zero on the right,
        // subtract q * v from it and remove the most significant limb, which is known
        // to eventually turn out zero anyway.
        // In case v_nlimbs < 2, this initialization of u_val reflects the extension by
        // a zero limb on the right, which will land in r_out_head_shadow[0] below.
        let mut u_borrow = 0;
        let mut scaled_v_carry = 0;
        let mut qv_carry = 0;
        let mut i = 0;
        let mut u_val = if v_nlimbs >= 2 {
            // Subtract q*v from the virtually shifted tail maintained in r_out[v_nlimbs - 3:0], if any,
            // return the value shifted out on the left.
            let mut next_u_val = 0; // The zero shifted in from the right.
            while i + 2 < v_nlimbs {
                let qv_val;
                (qv_val, scaled_v_carry, qv_carry) = scaled_qv_val(i, q, scaling, v, scaled_v_carry, qv_carry);
                let mut u_val = next_u_val;
                next_u_val = r_out.load_l_full(i);
                (u_borrow, u_val) = ct_sub_l_l_b(u_val, qv_val, u_borrow);
                r_out.store_l_full(i, u_val);
                i += 1;
            }

            // Calculate the value that got shifted out on the left and goes into the next higher
            // limb, r_out_head_shadow[0].
            {
                let qv_val;
                (qv_val, scaled_v_carry, qv_carry) = scaled_qv_val(i, q, scaling, v, scaled_v_carry, qv_carry);
                let mut u_val = next_u_val;
                (u_borrow, u_val) = ct_sub_l_l_b(u_val, qv_val, u_borrow);
                i += 1;

                u_val
            }

        } else {
            // For the case that v_nlimbs == 1, only store the shifted in zero in
            // r_out_head_shadow[0] below. It will serve as input to the the next long division
            // iteration, if any.
            0
        };

        // The remaining two head limbs in r_out_head_shadow[]. Note that for the most significant
        // limb in r_out_head_shadow[1], there's only the qv_carry left to add.
        debug_assert_eq!(i + 1, v_nlimbs);
        let mut qv_val;
        (qv_val, scaled_v_carry, qv_carry) = scaled_qv_val(i, q, scaling, v, scaled_v_carry, qv_carry);
        for k in [0, 1] {
            let cur_u_val = r_out_head_shadow[k];
            r_out_head_shadow[k] = u_val;
            (u_borrow, u_val) = ct_sub_l_l_b(cur_u_val, qv_val, u_borrow);
            qv_val = qv_carry;
        }
        debug_assert!(u_borrow != 0 || u_val == 0);

        // If u_borrow != 0, then the estimate for q had been one too large. Decrement it
        // and add one v back to the remainder accordingly.
        let over_estimated = ct_l_to_subtle_choice(u_borrow);
        if let Some(q_out) = &mut q_out {
            let q = LimbType::conditional_select(&q, &q.wrapping_sub(1), over_estimated);
            q_out.store_l(j, q);
        }
        let mut u_carry = 0;
        let mut scaled_v_carry = 0;
        let mut i = 0;
        // Update the tail maintained in r_out[v_nlimbs - 3:0], if any:
        while i + 2 < v_nlimbs {
            let v_val;
            (v_val, scaled_v_carry) = scaled_v_val(i, scaling, v, scaled_v_carry);
            let v_val = LimbType::conditional_select(&0, &v_val, over_estimated);
            let mut u_val = r_out.load_l_full(i);
            (u_carry, u_val) = ct_add_l_l_c(u_val, v_val, u_carry);
            r_out.store_l_full(i, u_val);
            i += 1;
        }
        // Take care of the two high limbs in r_out_head_shadow[].
        // Note that if v_nlimbs == 1, then r_out_head_shadow[0] does not correspond to an actual result
        // limb of the preceeding q * v subtraction, but already holds the zero to virtually append
        // to the sliding window in the loop's next iteration, if any. In this case, it must not
        // be considered for the addition of v here.
        let r_out_head_shadow_cur_sliding_window_overlap = v_nlimbs.min(2);
        for k in 0..r_out_head_shadow_cur_sliding_window_overlap {
            let v_val;
            (v_val, scaled_v_carry) = scaled_v_val(i, scaling, v, scaled_v_carry);
            let v_val = LimbType::conditional_select(&0, &v_val, over_estimated);
            let k = 2 - r_out_head_shadow_cur_sliding_window_overlap + k;
            let mut u_val = r_out_head_shadow[k];
            (u_carry, u_val) = ct_add_l_l_c(u_val, v_val, u_carry);
            r_out_head_shadow[k] = u_val;
            i += 1;
        }
        debug_assert_eq!(i, v_nlimbs);
    }

    // Finally, divide the resulting remainder in r_out by the scaling again.
    let scaling: Zeroizing<CtDivDlLNormalizedDivisor> =
        CtDivDlLNormalizedDivisor::new(scaling).into();
    let mut u_h = 0;
    // The two high limbs in r_out_head_shadow come first. Descale them and store them into their
    // corresponding locations in the returned r_out[]. Note that if v_nlimbs == 1, then the less
    // significant one in r_out_head_shadow[0] bears no significance.
    debug_assert!(v_nlimbs > 1 || r_out_head_shadow[0] == 0);
    for k in 0..v_nlimbs.min(2) {
        let u_l = r_out_head_shadow[2 - 1 - k];
        let (u, r) = |(u, r): (DoubleLimb, LimbType)| -> (Zeroizing<DoubleLimb> , LimbType) {
            (u.into(), r)
        }(ct_div_dl_l(&DoubleLimb::new(u_h, u_l), &scaling));
        debug_assert_eq!(u.high(), 0);
        r_out.store_l(v_nlimbs - 1 - k, u.low());
        u_h = r;
    }

    // Now do the remaining part in r_out[v_nlimbs - 3:0].
    let mut j = v_nlimbs;
    while j > 2 {
        j -= 1;
        let u_l = r_out.load_l_full(j - 2);
        let (u, r) = |(u, r): (DoubleLimb, LimbType)| -> (Zeroizing<DoubleLimb> , LimbType) {
            (u.into(), r)
        }(ct_div_dl_l(&DoubleLimb::new(u_h, u_l), &scaling));
        debug_assert_eq!(u.high(), 0);
        r_out.store_l_full(j - 2, u.low());
        u_h = r;
    };
    debug_assert_eq!(u_h, 0);

    Ok(())
}

#[cfg(test)]
fn test_mp_ct_div_pow2_mp<RT: MPIntMutByteSlice, VT: MPIntMutByteSlice, QT: MPIntMutByteSlice>() {
    fn div_and_check<RT: MPIntMutByteSlice, VT: MPIntMutByteSlice, QT: MPIntMutByteSlice>(
        u_pow2_exp: usize, v: &VT::SelfT<'_>
    ) {
        use super::add_impl::mp_ct_add_mp_mp;
        use super::mul_impl::mp_ct_mul_trunc_cond_mp_mp;

        let u_len = (u_pow2_exp + 1 + 8 - 1) / 8;
        let v_len = mp_find_last_set_byte_mp(v);
        let q_len = if u_len >= v_len {
            u_len - v_len + 1
        } else {
            0
        };

        let mut q = vec![0xffu8; QT::limbs_align_len(q_len)];
        let mut q = QT::from_bytes(&mut q).unwrap();
        let mut rem = vec![0xffu8; RT::limbs_align_len(v_len)];
        let mut rem = RT::from_bytes(&mut rem).unwrap();
        mp_ct_div_pow2_mp(u_pow2_exp as usize, &mut rem, v, Some(&mut q)).unwrap();

        // Multiply q by v again and add the remainder back, the result should match the initial u.
        // Reserve one extra limb, which is expected to come to zero.
        let mut result = vec![0xffu8; QT::limbs_align_len(u_len + LIMB_BYTES)];
        let mut result = QT::from_bytes(&mut result).unwrap();
        result.copy_from(&q);
        mp_ct_mul_trunc_cond_mp_mp(&mut result, q_len, v, subtle::Choice::from(1));
        let carry = mp_ct_add_mp_mp(&mut result, &rem);
        assert_eq!(carry, 0);
        let u_nlimbs = mp_ct_nlimbs(u_len);
        for i in 0..u_nlimbs - 1 {
            assert_eq!(result.load_l_full(i), 0);
        }
        let expected_high = (1 as LimbType) << (u_pow2_exp % (LIMB_BITS as usize));
        assert_eq!(result.load_l_full(u_nlimbs - 1), expected_high);
        assert_eq!(result.load_l(u_nlimbs), 0);
    }

    let mut v = vec![0u8; LIMB_BYTES];
    for v0 in [1 as LimbType, 7, 13, 17, 251] {
        for k in 0..LIMB_BYTES {
            let v0 = v0 << 8 * k;
            let mut v = VT::from_bytes(v.as_mut_slice()).unwrap();
            v.store_l(0, v0);
            for i in 0..5 * LIMB_BITS as usize {
                div_and_check::<RT, VT, QT>(i, &v);
            }
        }
    }

    let mut v = vec![0u8; 2 * LIMB_BYTES];
    for v_h in [0 as LimbType, 1, 7, 13, 17, 251] {
        for v_l in [0 as LimbType, 1, 7, 13, 17, 251] {
            if v_h == 0 && v_l == 0 {
                continue;
            }

            for k in 0..LIMB_BYTES {
                let v_h = v_h << 8 * k;
                let mut v = VT::from_bytes(v.as_mut_slice()).unwrap();
                v.store_l(0, v_l);
                v.store_l(1, v_h);
                for i in 0..6 * LIMB_BITS as usize {
                    div_and_check::<RT, VT, QT>(i, &v);
                }
            }
        }
    }
}

#[test]
fn test_mp_ct_div_pow2_be_be_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_div_pow2_mp::<MPBigEndianMutByteSlice, MPBigEndianMutByteSlice, MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_div_pow2_le_le_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_div_pow2_mp::<MPLittleEndianMutByteSlice, MPLittleEndianMutByteSlice, MPLittleEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_div_pow2_ne_ne_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_ct_div_pow2_mp::<MPNativeEndianMutByteSlice, MPNativeEndianMutByteSlice, MPNativeEndianMutByteSlice>()
}

pub fn mp_ct_div_lshifted_mp_mp<UT: MPIntMutByteSlice, VT: MPIntByteSliceCommon, QT: MPIntMutByteSlice>(
    u: &mut UT, u_in_len: usize, u_lshift_len: usize, v: &VT, mut q_out: Option<&mut QT>
) -> Result<(), MpCtDivisionError> {
    // Division algorithm according to D. E. Knuth, "The Art of Computer Programming", vol 2 adapted
    // to the case of a dividend extended on the right by u_lshift_len zero bytes.
    //
    // The division proceeds in two parts:
    // - first a regular long division is run on the head passed in in u[],
    // - and subsequently the long division is continued on the zero bytes shifted in on the
    //   right. This second part takes advantage of the fact that each step of the long division
    //   algorithm makes the dividend's (or, more precisely, the current remainder's) high limb zero
    //   by design. This allows to employ a memory-efficient "sliding window" approach where u[] is
    //   successively getting moved to the right in the virtually shifted dividend for each zero
    //   extension on the right and high limb correspondingly eliminated.

    // Find the index of the highest set limb in v. For divisors, constant time evaluation doesn't
    // really matter, probably as far as the number of zero high bytes is concerned. Also, the long
    // division algorithm's runtime depends highly on the divisor's length anyway.
    let v_len = mp_find_last_set_byte_mp(v);
    if v_len == 0 {
        return Err(MpCtDivisionError::DivisionByZero);
    }
    let v_nlimbs = mp_ct_nlimbs(v_len);
    let v_high = v.load_l(v_nlimbs - 1);

    debug_assert!(u_in_len <= u.len());
    let virtual_u_in_len = u_in_len + u_lshift_len;

    if let Some(q_out) = &mut q_out {
        // Check that q_out has enough space for storing the maximum possible quotient.
        if q_out.len() + v_len < virtual_u_in_len + 1 {
            return Err(MpCtDivisionError::InsufficientQuotientSpace);
        }
    }

    if u.len() < v_len.min(virtual_u_in_len) {
        return Err(MpCtDivisionError::InsufficientRemainderSpace);
    }

    if virtual_u_in_len < v_len {
        mp_lshift_mp(u, u_lshift_len);
        if let Some(q_out) = q_out {
            q_out.zeroize_bytes_above(0);
        }
        return Ok(());
    }

    let q_out_len = virtual_u_in_len + 1 - v_len;
    let q_out_nlimbs = mp_ct_nlimbs(q_out_len);
    if let Some(q_out) = &mut q_out {
        q_out.zeroize_bytes_above(q_out_len);
    }

    // The unaligned part of u_lshift_len will be taken into account by shifting the dividend by
    // that (small) amount to the left before running the regular long division on the (now shifted)
    // dividend in a first step. The remaining, aligned tail of the shift distance will subsequently
    // be handled by the memory-efficient "sliding window" approach in a second step below.
    let u_lshift_head_len = u_lshift_len % LIMB_BYTES;
    let u_lshift_tail_len = u_lshift_len - u_lshift_head_len;
    // If there's enough room left in u[] plus the high limb receiving the couple of bytes shifted
    // out from u on the left below, shovel an integral number of limbs from the tail to the
    // head. This will save the corresponding number of iterations in the subsequent "sliding
    // window" step.
    let u_lshift_head_len =
        ((
            mp_ct_nlimbs(
                (
                    u.len() - u_in_len +
                        LIMB_BYTES - u_lshift_head_len
                ) + 1 // For rounding downwards.
            ) - 1     // Ditto.
        ) * LIMB_BYTES).min(u_lshift_tail_len)
        + u_lshift_head_len;
    debug_assert_eq!(u_lshift_head_len % LIMB_BYTES, u_lshift_len % LIMB_BYTES);
    let u_lshift_tail_len = u_lshift_len - u_lshift_head_len;

    // Maintain a shadow of the shifted + scaled dividend's/current remainder's three most significant limbs
    // throughout the regular long division run in the first part:
    // - The least significant limb in u_head_high_shadow[0] will be needed to account for
    //   fact that u.len() might not be aligned to a limb boundary and thus, a partial high limb
    //   could potentially overflow during the computations.
    // - The next, more significant shadow limb in in u_head_high_shadow[1] will receive the bits
    //   shifted out from u[] on the left.
    // - The most significant shadow limb in u_head_high_shadow[2] will store the overflow, if any,
    //   the scaling.
    let mut u_head_high_shadow: Zeroizing<[LimbType; 3]> = Zeroizing::from([0; 3]);
    u_head_high_shadow[0] = mp_lshift_mp(u, u_lshift_head_len);
    // The (original) u length might not be aligned to the limb size. Move the high limb into the
    // u_head_high_shadow[0] shadow for the duration of the computation. Make sure the limb shifted
    // out on the left from u just above moves to the left in u_head_high_shadow[] accordingly.
    let u_head_high_partial_len = u.len() % LIMB_BYTES;
    let u_nlimbs = mp_ct_nlimbs(u.len());
    if u_head_high_partial_len != 0 {
        u_head_high_shadow[1] = u_head_high_shadow[0] >> 8 * (LIMB_BYTES - (u_head_high_partial_len));
        u_head_high_shadow[0] <<= 8 * u_head_high_partial_len;
        u_head_high_shadow[0] |= u.load_l(u_nlimbs - 1);
    } else {
        u_head_high_shadow[1] = u_head_high_shadow[0];
        u_head_high_shadow[0] = u.load_l(u_nlimbs - 1);
    }
    // At this point,
    // - u_head_high_shadow[1] contains (some of) the bits shifted out from u[] on the left and
    // - u_head_high_shadow[0] acts as a shadow for the potentially partial high limb
    //   of the shifted u[].

    // Normalize divisor's high limb.
    let scaling = v_scaling(v_high);

    // Read-only v won't get scaled in-place, but on the fly as needed. Calculate only its scaled
    // two high limbs, which will be needed for making the q estimates.
    let (scaled_v_high, scaled_v_tail_high) = v_head_scaled(scaling, v, v_nlimbs);
    let scaled_v_head: Zeroizing<[LimbType; 2]> = Zeroizing::from([scaled_v_high, scaled_v_tail_high]);
    let normalized_scaled_v_high: Zeroizing<CtDivDlLNormalizedDivisor> =
        CtDivDlLNormalizedDivisor::new(scaled_v_high).into();

    // Scale u.
    let mut carry = 0;
    for i in 0..u_nlimbs - 1 {
        let scaled: Zeroizing<DoubleLimb> = ct_mul_l_l(u.load_l(i), scaling).into();
        let (carry0, scaled_low) = ct_add_l_l(scaled.low(), carry);
        u.store_l(i, scaled_low);
        debug_assert!(scaled.high() < !0);
        carry = carry0 + scaled.high();
        debug_assert!(carry >= carry0); // Overflow cannot happen.
    }
    for i in 0..3 - 1 {
        let scaled: Zeroizing<DoubleLimb> = ct_mul_l_l(u_head_high_shadow[i], scaling).into();
        let (carry0, scaled_low) = ct_add_l_l(scaled.low(), carry);
        u_head_high_shadow[i] = scaled_low;
        debug_assert!(scaled.high() < !0);
        carry = carry0 + scaled.high();
        debug_assert!(carry >= carry0); // Overflow cannot happen.
    }
    u_head_high_shadow[2] = carry;

    // Now, in a first step, run the regular long division on the head part of the shifted u,
    // i.e. on the original u shifted left by u_lshift_len_head.
    let u_lshift_tail_nlimbs = u_lshift_tail_len / LIMB_BYTES; // The len is aligned.
    let q_out_head_nlimbs = q_out_nlimbs - u_lshift_tail_nlimbs;
    let mut j = q_out_head_nlimbs;
    while j > 0 {
        j -= 1;
        let q  = {
            // Estimate q. Load the first three u-limbs needed for the q estimation, i.e. the ones
            // at indices v_nlimbs + j, v_nlimbs + j - 1 and v_nlimbs + j - 2. Depending on where we
            // are currently, they need to be read either from the u_head_shadow[] or from u[]
            // itself. In case v_nlimbs == 1 and j == 0, the least signigicant of the three u-limbs
            // would be undefined. As per the comment in q_estimate(), it can be set to an arbitrary
            // value in this case, so just leave it zero.
            let mut cur_u_head: Zeroizing<[LimbType; 3]> = Zeroizing::from([0; 3]);
            let mut i = 3;
            while i > 0 && v_nlimbs + j + i >= u_nlimbs - 1 + 3 {
                i -= 1;
                cur_u_head[3 - i - 1] = u_head_high_shadow[(v_nlimbs + j + i - 2) - (u_nlimbs - 1)];
            }
            while i > 0 && v_nlimbs + j + i >= 3 {
                i -= 1;
                cur_u_head[3 - i - 1] = u.load_l_full(v_nlimbs + j + i - 2);
            }
            q_estimate(&cur_u_head, &scaled_v_head, normalized_scaled_v_high.deref())
        };

        // Subtract q * v at limb position j upwards in u[].
        let mut u_borrow = 0;
        let mut scaled_v_carry = 0;
        let mut qv_carry = 0;
        let mut i = 0;
        while i < v_nlimbs && j + i < u_nlimbs - 1 {
            let qv_val;
            (qv_val, scaled_v_carry, qv_carry) = scaled_qv_val(i, q, scaling, v, scaled_v_carry, qv_carry);
            let mut u_val = u.load_l_full(j + i);
            (u_borrow, u_val) = ct_sub_l_l_b(u_val, qv_val, u_borrow);
            u.store_l_full(j + i, u_val);
            i += 1;
        }
        while i < v_nlimbs {
            let qv_val;
            (qv_val, scaled_v_carry, qv_carry) = scaled_qv_val(i, q, scaling, v, scaled_v_carry, qv_carry);
            let mut u_val = u_head_high_shadow[j + i - (u_nlimbs - 1)];
            (u_borrow, u_val) = ct_sub_l_l_b(u_val, qv_val, u_borrow);
            u_head_high_shadow[j + i - (u_nlimbs - 1)] = u_val;
            i += 1;
        }
        // Take the final qv_carry into account.
        assert_eq!(i, v_nlimbs);
        if j + i < u_nlimbs - 1 {
            let qv_val = qv_carry;
            let mut u_val = u.load_l_full(j + i);
            (u_borrow, u_val) = ct_sub_l_l_b(u_val, qv_val, u_borrow);
            u.store_l_full(j + i, u_val);
        } else {
            let qv_val = qv_carry;
            let mut u_val = u_head_high_shadow[j + i - (u_nlimbs - 1)];
            (u_borrow, u_val) = ct_sub_l_l_b(u_val, qv_val, u_borrow);
            u_head_high_shadow[j + i - (u_nlimbs - 1)] = u_val;
        }

        // If u_borrow != 0, then the estimate for q had been one too large. Decrement it
        // and add one v back to the remainder accordingly.
        let over_estimated = ct_l_to_subtle_choice(u_borrow);
        if let Some(q_out) = &mut q_out {
            let q = LimbType::conditional_select(&q, &q.wrapping_sub(1), over_estimated);
            q_out.store_l( u_lshift_tail_nlimbs + j, q);
        }
        let mut u_carry = 0;
        let mut scaled_v_carry = 0;
        let mut i = 0;
        while i < v_nlimbs && j + i < u_nlimbs - 1 {
            let v_val;
            (v_val, scaled_v_carry) = scaled_v_val(i, scaling, v, scaled_v_carry);
            let v_val = LimbType::conditional_select(&0, &v_val, over_estimated);
            let mut u_val = u.load_l_full(j + i);
            (u_carry, u_val) = ct_add_l_l_c(u_val, v_val, u_carry);
            u.store_l_full(j + i, u_val);
            i += 1;
        }
        while i < v_nlimbs {
            let v_val;
            (v_val, scaled_v_carry) = scaled_v_val(i, scaling, v, scaled_v_carry);
            let v_val = LimbType::conditional_select(&0, &v_val, over_estimated);
            let mut u_val = u_head_high_shadow[j + i - (u_nlimbs - 1)];
            (u_carry, u_val) = ct_add_l_l_c(u_val, v_val, u_carry);
            u_head_high_shadow[j + i - (u_nlimbs - 1)] = u_val;
            i += 1;
        }
        // Take the final u_carry into account.
        assert_eq!(i, v_nlimbs);
        if j + i < u_nlimbs - 1 {
            let v_val = u_carry;
            let mut u_val = u.load_l_full(j + i);
            (_, u_val) = ct_add_l_l(u_val, v_val);
            u.store_l_full(j + i, u_val);
        } else {
            let v_val = u_carry;
            let mut u_val = u_head_high_shadow[j + i - (u_nlimbs - 1)];
            (_, u_val) = ct_add_l_l(u_val, v_val);
            u_head_high_shadow[j + i - (u_nlimbs - 1)] = u_val;
        }
    }
    debug_assert_eq!(u_head_high_shadow[2], 0);
    debug_assert_eq!(u_head_high_shadow[1], 0);
    if u_nlimbs - 1 >= v_nlimbs {
        debug_assert_eq!(u_head_high_shadow[0], 0);
        u.store_l(u_nlimbs - 1, 0);
    }
    for i in v_nlimbs..u_nlimbs - 1 {
        debug_assert_eq!(u.load_l_full(i), 0);
    }

    // Second step: divide the current remainder in u[], extended virtually by u_lshift_tail_len
    // more zeroes at the right. Again, because u.len() might not be aligned, maintain
    // a shadow limb for the case that u[]'s high limb is partial and u_nlimbs == v_nlimbs.
    let mut u_high_shadow = if v_nlimbs - 1 == u_nlimbs - 1 {
        u_head_high_shadow[0]
    } else {
        u.load_l_full(v_nlimbs - 1)
    };
    drop(u_head_high_shadow);

    let mut j = u_lshift_tail_nlimbs;
    while j > 0 {
        j -= 1;

        // Estimate q. Load the first three u-limbs needed for the q estimation. Depending on where
        // we are currently, they need to be read either from the u_head_shadow[] or from u[]
        // itself. In case v_nlimbs == 1 and j == 0, the least signigicant of the three u-limbs
        // would be undefined.  As per the comment in q_estimate(), it can be set to an arbitrary
        // value in this case, so leave it zero.
        let q = {
            let mut cur_u_head: Zeroizing<[LimbType; 3]> = Zeroizing::from([0; 3]);
            cur_u_head[0] = u_high_shadow;
            cur_u_head[1] = if v_nlimbs >= 2 {
                u.load_l_full(v_nlimbs - 2)
            } else {
                0 // Virtual zero shifted in on the right.
            };
            cur_u_head[2] = if v_nlimbs >= 3 {
                u.load_l_full(v_nlimbs - 3)
            } else {
                0 // Virtual zero shifted in on the right or, if v_nlimbs == 1 and j == 0, undefined.
            };

            q_estimate(&cur_u_head, &scaled_v_head, normalized_scaled_v_high.deref())
        };

        // Virtually shift u one limb to the left, add q * v and drop the (now zero) high limb.
        // This effectively moves the sliding window one limb to the right.
        let mut next_u_val = 0; // The zero shifted in on the right.
        let mut u_borrow = 0;
        let mut scaled_v_carry = 0;
        let mut qv_carry = 0;
        for i in 0..v_nlimbs - 1 {
            let qv_val;
            (qv_val, scaled_v_carry, qv_carry) = scaled_qv_val(i, q, scaling, v, scaled_v_carry, qv_carry);
            let mut u_val = next_u_val;
            next_u_val = u.load_l_full(i);
            (u_borrow, u_val) = ct_sub_l_l_b(u_val, qv_val, u_borrow);
            u.store_l_full(i, u_val);
        }
        // u[v_nlimbs - 1] is maintained in the u_high_shadow shadow, handle it separately.
        {
            let i = v_nlimbs - 1;
            let qv_val;
            (qv_val, scaled_v_carry, qv_carry) = scaled_qv_val(i, q, scaling, v, scaled_v_carry, qv_carry);
            debug_assert_eq!(scaled_v_carry, 0);
            let mut u_val = next_u_val;
            next_u_val = u_high_shadow;
            (u_borrow, u_val) = ct_sub_l_l_b(u_val, qv_val, u_borrow);
            u_high_shadow = u_val;
        }
        (u_borrow, _) = ct_sub_l_l_b(next_u_val, qv_carry, u_borrow);

        // If u_borrow != 0, then the estimate for q had been one too large. Decrement it
        // and add one v back to the remainder accordingly.
        let over_estimated = ct_l_to_subtle_choice(u_borrow);
        if let Some(q_out) = &mut q_out {
            let q = LimbType::conditional_select(&q, &q.wrapping_sub(1), over_estimated);
            q_out.store_l(j, q);
        }

        let mut u_carry = 0;
        let mut scaled_v_carry = 0;
        for i in 0..v_nlimbs - 1 {
            let v_val;
            (v_val, scaled_v_carry) = scaled_v_val(i, scaling, v, scaled_v_carry);
            let v_val = LimbType::conditional_select(&0, &v_val, over_estimated);
            let mut u_val = u.load_l_full(i);
            (u_carry, u_val) = ct_add_l_l_c(u_val, v_val, u_carry);
            u.store_l_full(i, u_val);
        }
        // u[v_nlimbs - 1] is maintained in the u_high_shadow shadow, handle it separately.
        {
            let i = v_nlimbs - 1;
            let v_val;
            (v_val, scaled_v_carry) = scaled_v_val(i, scaling, v, scaled_v_carry);
            let v_val = LimbType::conditional_select(&0, &v_val, over_estimated);
            let mut u_val = u_high_shadow;
            (_, u_val) = ct_add_l_l_c(u_val, v_val, u_carry);
            u_high_shadow = u_val;
        }
    }

    // Finally, divide the resulting remainder in u by the scaling again.
    let scaling: Zeroizing<CtDivDlLNormalizedDivisor> =
        CtDivDlLNormalizedDivisor::new(scaling).into();
    let mut u_h = 0;
    // The high limb maintained at u_high_shadow comes first. Descale and store in its final
    // location.
    {
        let u_l = u_high_shadow;
        let (u_val, r) = |(u_val, r): (DoubleLimb, LimbType)| -> (Zeroizing<DoubleLimb> , LimbType) {
            (u_val.into(), r)
        }(ct_div_dl_l(&DoubleLimb::new(u_h, u_l), &scaling));
        debug_assert_eq!(u_val.high(), 0);
        u.store_l(v_nlimbs - 1, u_val.low());
        u_h = r;
    }
    // Now do the remaining limbs in u[v_nlimbs - 2:0].
    let mut j = v_nlimbs - 1;
    while j > 0 {
        j -= 1;
        let u_l = u.load_l_full(j);
        let (u_val, r) = |(u_val, r): (DoubleLimb, LimbType)| -> (Zeroizing<DoubleLimb> , LimbType) {
            (u_val.into(), r)
        }(ct_div_dl_l(&DoubleLimb::new(u_h, u_l), &scaling));
        debug_assert_eq!(u_val.high(), 0);
        u.store_l(j, u_val.low());
        u_h = r;
    }
    debug_assert_eq!(u_h, 0);

    Ok(())
}

#[cfg(test)]
fn test_mp_ct_div_lshifted_mp_mp<UT: MPIntMutByteSlice, VT: MPIntMutByteSlice, QT: MPIntMutByteSlice>() {
    fn div_and_check<UT: MPIntMutByteSlice, VT: MPIntMutByteSlice, QT: MPIntMutByteSlice>(
        u: &UT::SelfT<'_>, u_in_len: usize , u_lshift_len: usize, v: &VT::SelfT<'_>
    ) {
        use super::add_impl::mp_ct_add_mp_mp;
        use super::cmp_impl::mp_ct_eq_mp_mp;
        use super::mul_impl::mp_ct_mul_trunc_cond_mp_mp;
        use super::shift_impl::mp_rshift_mp;

        let v_len = mp_find_last_set_byte_mp(v);
        let virtual_u_len = u_in_len + u_lshift_len;
        let q_len = virtual_u_len + 1  - v_len;
        let mut q = vec![0xffu8; QT::limbs_align_len(q_len)];
        let mut q = QT::from_bytes(q.as_mut_slice()).unwrap();
        let mut rem = vec![0u8; u.len()];
        let mut rem = UT::from_bytes(&mut rem).unwrap();
        rem.copy_from(u);
        mp_ct_div_lshifted_mp_mp(&mut rem, u_in_len, u_lshift_len, v, Some(&mut q)).unwrap();

        // Multiply q by v again and add the remainder back, the result should match the initial u.
        // Reserve one extra limb, which is expected to come to zero.
        let mut result = vec![0xffu8; UT::limbs_align_len(virtual_u_len + LIMB_BYTES)];
        let mut result = UT::from_bytes(&mut result).unwrap();
        result.copy_from(&q);
        mp_ct_mul_trunc_cond_mp_mp(&mut result, q_len, v, subtle::Choice::from(1));
        let carry = mp_ct_add_mp_mp(&mut result, &rem);
        assert_eq!(carry, 0);
        for i in 0..mp_ct_nlimbs(u_lshift_len + 1) - 1 {
            assert_eq!(result.load_l(i), 0);
        }
        if u_lshift_len % LIMB_BYTES != 0 {
            let u_val = result.load_l(mp_ct_nlimbs(u_lshift_len + 1) - 1);
            assert_eq!(u_val & (((1 as LimbType) << (8 * (u_lshift_len % LIMB_BYTES))) - 1), 0);
        }
        assert_eq!(result.load_l(mp_ct_nlimbs(virtual_u_len)), 0);
        mp_rshift_mp(&mut result, u_lshift_len);
        assert_eq!(mp_ct_eq_mp_mp(u, &result).unwrap_u8(), 1);
    }

    const N_MAX_LIMBS: u32 = 3;
    for i in 0..N_MAX_LIMBS * LIMB_BITS + 1 {
        let u_len = (i as usize + 8 - 1) / 8;
        for j1 in 0..i + 1 {
            for j2 in 0..j1 + 1 {
                let v_len = ((j1 + 1) as usize + 8 - 1) / 8;
                for u_lshift_len in 0..2 * LIMB_BYTES {
                    dbg!(u_len, v_len);
                    let mut u = vec![0u8; UT::limbs_align_len(u_len.max(v_len))];
                    let mut u = UT::from_bytes(&mut u).unwrap();
                    if i != 0 {
                        let u_nlimbs = mp_ct_nlimbs(u_len);
                        for k in 0..u_nlimbs - 1 {
                            u.store_l(k, !0);
                        }
                        if i % LIMB_BITS != 0 {
                            let i = i % LIMB_BITS;
                            u.store_l(u_nlimbs - 1, !0 >> (LIMB_BITS - i));
                        } else  {
                            u.store_l(u_nlimbs - 1, !0);
                        }
                    }

                    let mut v = vec![0u8; VT::limbs_align_len(v_len)];
                    let mut v = VT::from_bytes(&mut v).unwrap();
                    v.store_l((j1 / LIMB_BITS) as usize, 1 << (j1 % LIMB_BITS));
                    v.store_l((j2 / LIMB_BITS) as usize, 1 << (j2 % LIMB_BITS));
                    div_and_check::<UT, VT, QT>(&u, u_len, u_lshift_len, &v);
                }
            }
        }
    }
}

#[test]
fn test_mp_ct_div_lshifted_be_be_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_div_lshifted_mp_mp::<MPBigEndianMutByteSlice, MPBigEndianMutByteSlice, MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_div_lshifted_le_le_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_div_lshifted_mp_mp::<MPLittleEndianMutByteSlice, MPLittleEndianMutByteSlice, MPLittleEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_div_lshifted_ne_ne_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_ct_div_lshifted_mp_mp::<MPNativeEndianMutByteSlice, MPNativeEndianMutByteSlice, MPNativeEndianMutByteSlice>()
}


// Compute the modulo of a multiprecision integer modulo a [`LimbType`] divisisor.
pub fn mp_ct_div_mp_l<UT: MPIntByteSliceCommon, QT: MPIntMutByteSlice>(
    u: &UT, v: LimbType, mut q_out: Option<&mut QT>
) -> Result<LimbType, MpCtDivisionError> {
    if v == 0 {
        return Err(MpCtDivisionError::DivisionByZero);
    }

    let u_nlimbs = mp_ct_nlimbs(u.len());
    if u_nlimbs == 0 {
        return Ok(0);
    }

    if let Some(q_out) = &mut q_out {
        let v_len = ct_find_last_set_byte_l(v);
        if q_out.len() + v_len < u.len() + 1 {
            return Err(MpCtDivisionError::InsufficientQuotientSpace);
        }
        if u.len() < v_len {
            q_out.zeroize_bytes_above(0);
            return Ok(u.load_l(0));
        }
        let q_out_len = u.len() - v_len + 1;
        q_out.zeroize_bytes_above(q_out_len);
    }

    let normalized_v: Zeroizing<CtDivDlLNormalizedDivisor> =
        CtDivDlLNormalizedDivisor::new(v).into();

    let mut u_h = 0;
    let mut j = u_nlimbs;
    while j > 0 {
        j -= 1;
        let u_l = u.load_l(j);
        let (q_val, r) = |(q_val, r): (DoubleLimb, LimbType)| -> (Zeroizing<DoubleLimb> , LimbType) {
            (q_val.into(), r)
        }(ct_div_dl_l(&DoubleLimb::new(u_h, u_l), normalized_v.deref()));
        debug_assert_eq!(q_val.high(), 0);

        if let Some(q_out) = &mut q_out {
            q_out.store_l(j, q_val.low())
        }
        u_h = r;
    }

    Ok(u_h)
}
