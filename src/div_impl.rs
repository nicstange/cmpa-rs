// SPDX-License-Identifier: Apache-2.0
// Copyright 2023 SUSE LLC
// Author: Nicolai Stange <nstange@suse.de>

#[cfg(test)]
extern crate alloc;

use super::limb::{
    ct_add_l_l, ct_eq_l_l, ct_find_last_set_bit_l, ct_find_last_set_byte_l, ct_gt_l_l,
    ct_lsb_mask_l, ct_mul_l_l, ct_mul_sub_l_l_l_b, ct_sub_l_l, CtLDivisor, DoubleLimb,
    LDivisorPrivate, LimbChoice, LimbType, LIMB_BITS, LIMB_BYTES,
};
use super::limbs_buffer::{
    ct_mp_nlimbs, find_last_set_byte_mp, CompositeLimbsBuffer, MpMutNativeEndianUIntLimbsSlice,
    MpMutUInt, MpMutUIntSlice, MpUIntCommon,
};
use super::shift_impl::ct_lshift_mp;
use super::usize_ct_cmp::ct_is_zero_usize;
#[cfg(test)]
use alloc::vec;

pub struct CtMpDivisor<'a, VT: MpUIntCommon> {
    v: &'a VT,
    v_decrement: Option<LimbType>,
    v_len: usize,
    scaling_shift: u32,
    scaling_low_src_rshift: u32,
    scaling_low_src_mask: LimbType,
    scaled_v_head_divisor: CtLDivisor,
    scaled_v_head: LimbType,
    scaled_v_tail_head: LimbType,
}

#[derive(Debug)]
pub enum CtMpDivisorError {
    DivisorIsZero,
}

impl<'a, VT: MpUIntCommon> CtMpDivisor<'a, VT> {
    pub fn new(v: &'a VT, v_decrement: Option<LimbType>) -> Result<Self, CtMpDivisorError> {
        // Find the index of the highest set limb in v, reduced by v_decrement. For
        // divisors, constant time evaluation doesn't really matter, probably,
        // as far as the number of zero high bytes is concerned. Also, the long
        // division algorithm's runtime depends highly on the divisor's length
        // anyway.
        if v.is_empty() {
            return Err(CtMpDivisorError::DivisorIsZero);
        }
        let (v_len, v_head) = if let Some(v_decrement) = v_decrement {
            let mut v_decrement_borrow = v_decrement;
            let mut v_len = 0;
            let mut v_head = 0;
            for i in 0..v.nlimbs() {
                let v_val;
                (v_decrement_borrow, v_val) = ct_sub_l_l(v.load_l(i), v_decrement_borrow);
                let v_val_is_zero = ct_eq_l_l(v_val, 0);
                v_len = v_val_is_zero
                    .select_usize(i * LIMB_BYTES + ct_find_last_set_byte_l(v_val), v_len);
                v_head = v_val_is_zero.select(v_val, v_head);
            }
            if v_decrement_borrow != 0 {
                // Not exactly zero, but in a "saturating" subtraction sense.
                return Err(CtMpDivisorError::DivisorIsZero);
            }
            (v_len, v_head)
        } else {
            let v_len = find_last_set_byte_mp(v);
            let v_head = v.load_l(ct_mp_nlimbs(v_len) - 1);
            (v_len, v_head)
        };
        if ct_is_zero_usize(v_len) != 0 {
            return Err(CtMpDivisorError::DivisorIsZero);
        }
        let v_nlimbs = ct_mp_nlimbs(v_len);

        let v_head_width = ct_find_last_set_bit_l(v_head);
        debug_assert_ne!(v_head_width, 0);

        // Normalize by shift such that the scaled v high limb's MSB is set.
        let scaling_shift = LIMB_BITS - v_head_width as u32;
        let scaling_low_src_rshift = (LIMB_BITS - scaling_shift) % LIMB_BITS;
        let scaling_low_src_mask = ct_lsb_mask_l(scaling_shift);

        // Read-only v won't get scaled in-place, but on the fly as needed. For now,
        // scale v only to calculate the two scaled head limbs of v,
        // as are needed for the q estimates. For scaling by shifting (as opposed to
        // by multiplication of a scaling factor), only the highest three limbs
        // contribute to the highest two scaled limbs. if v_decrement is set,
        // v_decrement_borrow still needs to get propagated all the way up
        // though.
        let mut v_decrement_borrow = if let Some(v_decrement) = v_decrement {
            let mut v_decrement_borrow = v_decrement;
            for i in 0..v_nlimbs - 3.min(v_nlimbs) {
                (v_decrement_borrow, _) = ct_sub_l_l(v.load_l_full(i), v_decrement_borrow);
            }
            v_decrement_borrow
        } else {
            0
        };
        let mut scaled_v_carry = 0;
        // If v_nlimbs == 1, it will remain zero on purpose.
        let mut scaled_v_tail_head = 0; // Silence the compiler.
        let mut scaled_v_head = 0;
        for i in v_nlimbs - 3.min(v_nlimbs)..v_nlimbs {
            scaled_v_tail_head = scaled_v_head;
            let v_val;
            (v_decrement_borrow, v_val) = ct_sub_l_l(v.load_l(i), v_decrement_borrow);
            (scaled_v_carry, scaled_v_head) = Self::_scale_val(
                v_val,
                scaling_shift,
                scaling_low_src_rshift,
                scaling_low_src_mask,
                scaled_v_carry,
            );
        }
        debug_assert_eq!(scaled_v_carry, 0);

        let scaled_v_head_divisor = CtLDivisor::new(scaled_v_head).unwrap();

        Ok(Self {
            v,
            v_decrement,
            v_len,
            scaling_shift,
            scaling_low_src_rshift,
            scaling_low_src_mask,
            scaled_v_head_divisor,
            scaled_v_head,
            scaled_v_tail_head,
        })
    }

    fn make_q_estimate(&self, u_head: &[LimbType; 3]) -> LimbType {
        // The double limb by limb division supports only dividends for
        // which the quotient fits a limb. However, the quotient here
        // might exceed a limb by one bit, but not more, because of
        // the normalization of v_high.
        // Before invoking the the CtDivDlLLDivisor, handle this
        // case explicitly.
        let (q_high_is_zero, u_high) = ct_sub_l_l(u_head[0], self.scaled_v_head);
        let q_high_is_zero = LimbChoice::from(q_high_is_zero);
        let u_high = q_high_is_zero.select(u_high, u_head[0]);
        debug_assert!(u_high < self.scaled_v_head);
        let (q_low, r) = self
            .scaled_v_head_divisor
            .do_div(&DoubleLimb::new(u_high, u_head[1]));

        // If !q_high_is_zero, q needs to get capped to fit single limb and r adjusted
        // accordingly.
        //
        // For determining the adjusted r, note that if !q_high_is_zero, then then
        // u_head[0] == scaled_v_head. To see this, observe that
        // u_head[0] >= scaled_v_head holds trivially.
        //
        // OTOH, the invariant throughout the caller's loop over j is that
        // u[j+n:j] / v[n-1:0] < b (u and v both scaled), from which it follows that
        // u_head[0] <= scaled_v_head. Assume not, i.e.
        // u_head[0] >= scaled_v_head + 1. We have v < (scaled_v_head + 1) * b^(n - 1).
        // It would follow that
        // u[j+n:j] >= (u_head[0] * b + u_head[1]) * b^(n - 1)
        //          >= u_head[0] * b * b^(n - 1) >= (scaled_v_head + 1) * b * b^(n - 1)
        //          > v * b,
        // a contradiction to the loop invariant.
        //
        // Thus, in summary, if !q_high_is_zero, then u_head[0] == scaled_v_head.
        //
        // It follows that the adjusted r for capping q to q == b - 1 equals
        // u_head[0] * b + u_head[1] - (b - 1) * scaled_v_head
        // = scaled_v_head * b + u_head[1] - (b - 1) * scaled_v_head
        // = scaled_v_head + u_head[1].
        debug_assert!(q_high_is_zero.unwrap() != 0 || u_head[0] == self.scaled_v_head);
        let ov = !q_high_is_zero;
        let q = ov.select(q_low, !0);
        let (r_carry_on_ov, r_on_ov) = ct_add_l_l(u_head[1], self.scaled_v_head);
        let r = ov.select(r, r_on_ov);
        debug_assert_eq!(r_carry_on_ov & !1, 0); // At most LSB is set
        let r_carry = ov & LimbChoice::from(r_carry_on_ov);

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
        // If v_nlimbs < 2 and j == 0, u[j + n - 2] might not be defined. But in this
        // case v[n-2] (found in scaled_v_head[1]) is zero anyway and the comparison
        // test will always come out negative, so the caller may load arbitrary
        // value into its corresponding location at q_head[2].
        let qv_head_low: DoubleLimb = ct_mul_l_l(self.scaled_v_tail_head, q);
        let over_estimated = !r_carry
            & (ct_gt_l_l(qv_head_low.high(), r)
                | (ct_eq_l_l(qv_head_low.high(), r) & ct_gt_l_l(qv_head_low.low(), u_head[2])));
        q - over_estimated.select(0, 1)
    }

    // Scale a multiprecision integer limb with carry application.
    fn _scale_val(
        val: LimbType,
        scaling_shift: u32,
        scaling_low_src_rshift: u32,
        scaling_low_src_mask: LimbType,
        carry: LimbType,
    ) -> (LimbType, LimbType) {
        debug_assert!(carry <= (1 << scaling_shift));
        let (carry, scaled_val) = ct_add_l_l(val << scaling_shift, carry);
        // If the addition wrapped around, it wrapped to zero: in this case
        // the original carry had been == 1 << scaling_shift.
        debug_assert!(carry == 0 || scaled_val == 0);
        // Does no wrap: either scaling_low_src_rshift > 0 or the mask is zero.
        let carry = ((val >> scaling_low_src_rshift) & scaling_low_src_mask) + carry;
        debug_assert!(carry <= (1 << scaling_shift));
        (carry, scaled_val)
    }

    fn scale_val(&self, val: LimbType, carry: LimbType) -> (LimbType, LimbType) {
        Self::_scale_val(
            val,
            self.scaling_shift,
            self.scaling_low_src_rshift,
            self.scaling_low_src_mask,
            carry,
        )
    }

    fn scaled_v_val(
        &self,
        i: usize,
        v_decrement_borrow: LimbType,
        carry: LimbType,
    ) -> (LimbType, LimbType, LimbType) {
        let (v_decrement_borrow, v_val) = ct_sub_l_l(self.v.load_l(i), v_decrement_borrow);
        let (carry, scaled_v_val) = self.scale_val(v_val, carry);
        (v_decrement_borrow, carry, scaled_v_val)
    }

    fn unscale_val(&self, val: LimbType, last_higher_val: LimbType) -> LimbType {
        let dst_high_lshift = self.scaling_low_src_rshift;
        let dst_high_mask = self.scaling_low_src_mask;

        let val_low = val >> self.scaling_shift;
        let val_high = (last_higher_val & dst_high_mask) << dst_high_lshift;
        val_high | val_low
    }

    fn add_scaled_v_val_cond(
        &self,
        op0: LimbType,
        i: usize,
        v_decrement_borrow: LimbType,
        carry: LimbType,
        cond: LimbChoice,
    ) -> (LimbType, LimbType, LimbType) {
        let (v_decrement_borrow, carry0, scaled_v_val) =
            self.scaled_v_val(i, v_decrement_borrow, carry);
        let carry0 = cond.select(0, carry0);
        let scaled_v_val = cond.select(0, scaled_v_val);
        let (carry1, result) = ct_add_l_l(op0, scaled_v_val);
        let carry = carry0 + carry1;
        (v_decrement_borrow, carry, result)
    }

    fn sub_scaled_qv_val(
        &self,
        op0: LimbType,
        i: usize,
        q: LimbType,
        v_decrement_borrow: LimbType,
        scaled_v_carry: LimbType,
        borrow: LimbType,
    ) -> (LimbType, LimbType, LimbType, LimbType) {
        let (v_decrement_borrow, scaled_v_carry, scaled_v_val) =
            self.scaled_v_val(i, v_decrement_borrow, scaled_v_carry);
        let (borrow, result) = ct_mul_sub_l_l_l_b(op0, q, scaled_v_val, borrow);
        (v_decrement_borrow, scaled_v_carry, borrow, result)
    }
}

#[derive(Debug)]
pub enum CtDivMpMpError {
    InsufficientQuotientSpace,
}

pub fn ct_div_mp_mp<UT: MpMutUIntSlice, VT: MpUIntCommon, QT: MpMutUInt>(
    u_h: Option<&mut UT>,
    u_l: &mut UT,
    v: &CtMpDivisor<VT>,
    mut q_out: Option<&mut QT>,
) -> Result<(), CtDivMpMpError> {
    // Division algorithm according to D. E. Knuth, "The Art of Computer
    // Programming", vol 2.

    // If u_h is None, set it to an empty slice for code uniformity.
    let mut __u_h = [UT::BackingSliceElementType::from(0u8); 0];
    let u_h: UT::SelfT<'_> = match u_h {
        Some(u_h) => u_h.coerce_lifetime(),
        None => UT::from_slice(__u_h.as_mut()).unwrap(),
    };

    let u_len = u_l.len() + u_h.len();
    let u_nlimbs = ct_mp_nlimbs(u_len);

    let v_len = v.v_len;
    let v_nlimbs = ct_mp_nlimbs(v_len);

    if let Some(q_out) = &mut q_out {
        // Check that q_out has enough space for storing the maximum possible quotient.
        // In general, at most one more bit than the bit width difference
        // between u and v is needed. The byte granularity is too coarse to
        // catch that, so only check the absolute lower bound here.
        // The code storing the quotient below will verify that the head limb is zero in
        // case it's not been provided storage for.
        if q_out.len() + v_len < u_len {
            return Err(CtDivMpMpError::InsufficientQuotientSpace);
        }
    };

    if u_len < v_len {
        if let Some(q_out) = q_out {
            q_out.clear_bytes_above(0);
        }
        return Ok(());
    }

    let q_out_max_len = u_len + 1 - v_len;
    let q_out_max_nlimbs = ct_mp_nlimbs(q_out_max_len);
    if let Some(q_out) = &mut q_out {
        q_out.clear_bytes_above(q_out_max_len);
    }

    // Create a padding buffer extending u at its more significant end:
    // - ensure that the resulting length aligns to LIMB_BYTES and
    // - allocate an extra limb to provide sufficient space for the scaling below.
    let u_pad_len = if u_len % LIMB_BYTES == 0 {
        0
    } else {
        LIMB_BYTES - u_len % LIMB_BYTES
    };

    // This is a horrid way to work around Rust's limitation of not yet supporting
    // constant expressions involving generics in array length specifiers. What
    // is really wanted is something along the lines of
    // [UT::BackingSliceElementType;
    //  UT::n_backing_elements_for_len(u_pad_len + LIMB_BYTES)]
    // So, to avoid a memory allocation (which would be the only one in the whole
    // library), create an array of maximum element type among all MpIntMutSlice
    // implementations and unsafely cast the slice to the desired type. Once
    // Rust supports the needed constant expression for array length specifiers,
    // this atrocity is bound for removal.
    let mut _u_pad = [0 as LimbType; 2];
    let _u_pad = {
        use core::{mem, slice};
        let _u_pad_size = mem::size_of_val(&_u_pad);
        let _u_pad = _u_pad.as_mut_ptr() as *mut UT::BackingSliceElementType;
        let _u_pad = unsafe {
            slice::from_raw_parts_mut(
                _u_pad,
                _u_pad_size / mem::size_of::<UT::BackingSliceElementType>(),
            )
        };
        &mut _u_pad[..UT::n_backing_elements_for_len(LIMB_BYTES + u_pad_len)]
    };

    let u_pad = UT::from_slice(_u_pad).unwrap();
    let u_l = u_l.coerce_lifetime();
    let mut u_parts = CompositeLimbsBuffer::new([u_l, u_h, u_pad]);

    // Scale u.
    let mut carry = 0;
    for i in 0..u_nlimbs + 1 {
        let mut u_val = u_parts.load(i);
        (carry, u_val) = v.scale_val(u_val, carry);
        u_parts.store(i, u_val);
    }
    // The extra high limb in u_h_pad, initialized to zero, would have absorbed the
    // last carry.
    debug_assert_eq!(carry, 0);

    let mut j = q_out_max_nlimbs;
    while j > 0 {
        j -= 1;
        let q = {
            let u_h = u_parts.load(v_nlimbs + j);
            let u_l = u_parts.load(v_nlimbs + j - 1);
            // Load u[j + n - 2]. If v_nlimbs < 2 and j == 0, it might not be defined --
            // make it zero in this case, c.f. the corresponding comment in
            // q_estimate().
            let u_tail_high = if v_nlimbs + j >= 2 {
                u_parts.load(v_nlimbs + j - 2)
            } else {
                0
            };
            let cur_u_head: [LimbType; 3] = [u_h, u_l, u_tail_high];

            v.make_q_estimate(&cur_u_head)
        };

        // Subtract q * v from u at position j.
        let mut v_decrement_borrow = v.v_decrement.unwrap_or(0);
        let mut scaled_v_carry = 0;
        let mut borrow = 0;
        for i in 0..v_nlimbs {
            let mut u_val = u_parts.load(j + i);
            (v_decrement_borrow, scaled_v_carry, borrow, u_val) =
                v.sub_scaled_qv_val(u_val, i, q, v_decrement_borrow, scaled_v_carry, borrow);
            u_parts.store(j + i, u_val);
        }
        debug_assert_eq!(scaled_v_carry, 0);
        let u_val = u_parts.load(j + v_nlimbs);
        let (borrow, u_val) = ct_sub_l_l(u_val, borrow);
        u_parts.store(j + v_nlimbs, u_val);

        // If borrow != 0, then the estimate for q had been one too large. Decrement it
        // and add one v back to the remainder accordingly.
        let over_estimated = LimbChoice::from(borrow);
        if let Some(q_out) = &mut q_out {
            let q = q - over_estimated.select(0, 1);
            if j != q_out.nlimbs() {
                debug_assert!(j < q_out.nlimbs());
                q_out.store_l(j, q);
            } else if q != 0 {
                return Err(CtDivMpMpError::InsufficientQuotientSpace);
            }
        }
        let mut v_decrement_borrow = v.v_decrement.unwrap_or(0);
        let mut carry = 0;
        for i in 0..v_nlimbs {
            let mut u_val = u_parts.load(j + i);
            (v_decrement_borrow, carry, u_val) =
                v.add_scaled_v_val_cond(u_val, i, v_decrement_borrow, carry, over_estimated);
            u_parts.store(j + i, u_val);
        }
        let u_val = u_parts.load(j + v_nlimbs);
        let (_, u_val) = ct_add_l_l(u_val, carry);
        u_parts.store(j + v_nlimbs, u_val);
    }

    // Finally, divide the resulting remainder in u by the scaling again.
    let mut u_h = 0;
    for j in v_nlimbs..u_nlimbs + 1 {
        debug_assert_eq!(u_parts.load(j), 0);
    }
    let mut j = v_nlimbs;
    while j > 0 {
        j -= 1;
        let u_l = u_parts.load(j);
        let u_val = v.unscale_val(u_l, u_h);
        u_h = u_l;
        u_parts.store(j, u_val);
    }

    Ok(())
}

#[cfg(test)]
fn test_limbs_from_be_bytes<DT: MpMutUIntSlice, const N: usize>(
    bytes: [u8; N],
) -> vec::Vec<DT::BackingSliceElementType> {
    use super::limbs_buffer::{MpBigEndianUIntByteSlice, MpUIntSlicePriv as _};
    let mut limbs = tst_mk_mp_backing_vec!(DT, N);
    let mut dst = DT::from_slice(limbs.as_mut_slice()).unwrap();
    dst.copy_from(&MpBigEndianUIntByteSlice::from_slice(bytes.as_slice()).unwrap());
    drop(dst);
    limbs
}

#[cfg(test)]
fn test_ct_div_mp_mp<UT: MpMutUIntSlice, VT: MpMutUIntSlice, QT: MpMutUIntSlice>() {
    use super::cmp_impl::ct_eq_mp_mp;

    fn div_and_check<UT: MpMutUIntSlice, VT: MpMutUIntSlice, QT: MpMutUIntSlice>(
        u: &UT::SelfT<'_>,
        v: &VT::SelfT<'_>,
        v_decrement: Option<LimbType>,
        split_u: bool,
    ) {
        use super::add_impl::{ct_add_mp_mp, ct_sub_mp_l};
        use super::mul_impl::ct_mul_trunc_mp_mp;

        let mut decremented_v = tst_mk_mp_backing_vec!(VT, v.len());
        let mut decremented_v = VT::from_slice(&mut decremented_v).unwrap();
        decremented_v.copy_from(v);
        if let Some(v_decrement) = v_decrement {
            ct_sub_mp_l(&mut decremented_v, v_decrement);
        }

        let v_len = find_last_set_byte_mp(&decremented_v);
        let q_len = if u.len() >= v_len {
            u.len() - v_len + 1
        } else {
            0
        };
        let mut q = tst_mk_mp_backing_vec!(QT, q_len);
        q.fill(0xffu8.into());
        let mut q = QT::from_slice(&mut q).unwrap();
        let divisor = CtMpDivisor::new(v, v_decrement).unwrap();
        let mut rem_buf = if split_u {
            let split_point = if UT::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
                LIMB_BYTES + 1
            } else {
                LIMB_BYTES
            };
            let split_point = split_point.min(u.len());
            let mut rem_l_buf = tst_mk_mp_backing_vec!(UT, split_point);
            let mut rem_h_buf = tst_mk_mp_backing_vec!(UT, u.len() - split_point);
            let mut rem_composite = CompositeLimbsBuffer::new([
                UT::from_slice(&mut rem_l_buf).unwrap(),
                UT::from_slice(&mut rem_h_buf).unwrap(),
            ]);
            for i in 0..u.nlimbs() {
                rem_composite.store(i, u.load_l(i));
            }
            drop(rem_composite);
            ct_div_mp_mp(
                Some(&mut UT::from_slice(&mut rem_h_buf).unwrap()),
                &mut UT::from_slice(&mut rem_l_buf).unwrap(),
                &divisor,
                Some(&mut q),
            )
            .unwrap();
            let rem_composite = CompositeLimbsBuffer::new([
                UT::from_slice(&mut rem_l_buf).unwrap(),
                UT::from_slice(&mut rem_h_buf).unwrap(),
            ]);
            let mut rem_buf = tst_mk_mp_backing_vec!(UT, u.len());
            let mut rem = UT::from_slice(&mut rem_buf).unwrap();
            for i in 0..u.nlimbs() {
                rem.store_l(i, rem_composite.load(i));
            }
            drop(rem);
            rem_buf
        } else {
            let mut rem_buf = tst_mk_mp_backing_vec!(UT, u.len());
            let mut rem = UT::from_slice(&mut rem_buf).unwrap();
            rem.copy_from(u);
            ct_div_mp_mp(None, &mut rem, &divisor, Some(&mut q)).unwrap();
            drop(rem);
            rem_buf
        };
        let rem = UT::from_slice(&mut rem_buf).unwrap();

        // Multiply q by v again and add the remainder back, the result should match the
        // initial u. Reserve one extra limb, which is expected to come to zero.
        let mut result = tst_mk_mp_backing_vec!(UT, u.len() + LIMB_BYTES);
        let mut result = UT::from_slice(&mut result).unwrap();
        result.copy_from(&q);
        ct_mul_trunc_mp_mp(&mut result, q_len, &decremented_v);
        let carry = ct_add_mp_mp(&mut result, &rem);
        assert_eq!(carry, 0);
        assert_eq!(ct_eq_mp_mp(u, &result).unwrap(), 1);
    }

    let mut u = test_limbs_from_be_bytes::<UT, 2>([1, 0]);
    let u = UT::from_slice(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 1>([1]);
    let v = VT::from_slice(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, None, false);
    div_and_check::<UT, VT, QT>(&u, &v, None, true);

    let mut u = test_limbs_from_be_bytes::<UT, 2>([1, 0]);
    let u = UT::from_slice(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 1>([3]);
    let v = VT::from_slice(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, None, false);
    div_and_check::<UT, VT, QT>(&u, &v, None, true);
    div_and_check::<UT, VT, QT>(&u, &v, Some(1), false);

    let mut u = test_limbs_from_be_bytes::<UT, 6>([!0, !0, !1, !0, !0, !0]);
    let u = UT::from_slice(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 3>([!0, !0, !0]);
    let v = VT::from_slice(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, None, false);
    div_and_check::<UT, VT, QT>(&u, &v, None, true);
    div_and_check::<UT, VT, QT>(&u, &v, Some(1), false);

    let mut u = test_limbs_from_be_bytes::<UT, 6>([!0, !0, !1, !0, !0, !1]);
    let u = UT::from_slice(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 3>([!0, !0, !0]);
    let v = VT::from_slice(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, None, false);
    div_and_check::<UT, VT, QT>(&u, &v, None, true);
    div_and_check::<UT, VT, QT>(&u, &v, Some(1), false);

    let mut u = test_limbs_from_be_bytes::<UT, 6>([0, 0, 0, 0, 0, 0]);
    let u = UT::from_slice(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 3>([!0, !0, !0]);
    let v = VT::from_slice(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, None, false);
    div_and_check::<UT, VT, QT>(&u, &v, None, true);
    div_and_check::<UT, VT, QT>(&u, &v, Some(1), false);

    let mut u = test_limbs_from_be_bytes::<UT, 6>([0, 0, 0, 0, 0, !1]);
    let u = UT::from_slice(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 3>([!0, !0, !0]);
    let v = VT::from_slice(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, None, false);
    div_and_check::<UT, VT, QT>(&u, &v, None, true);
    div_and_check::<UT, VT, QT>(&u, &v, Some(1), false);

    let mut u = test_limbs_from_be_bytes::<UT, 6>([!0, !0, !0, !0, !0, 0]);
    let u = UT::from_slice(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 3>([0, 1, 0]);
    let v = VT::from_slice(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, None, false);
    div_and_check::<UT, VT, QT>(&u, &v, None, true);
    div_and_check::<UT, VT, QT>(&u, &v, Some(1), false);

    let mut u = test_limbs_from_be_bytes::<UT, 6>([!0, !0, !0, !0, !1, 0]);
    let u = UT::from_slice(u.as_mut_slice()).unwrap();
    let mut v = test_limbs_from_be_bytes::<VT, 3>([0, 2, 0]);
    let v = VT::from_slice(v.as_mut_slice()).unwrap();
    div_and_check::<UT, VT, QT>(&u, &v, None, false);
    div_and_check::<UT, VT, QT>(&u, &v, None, true);
    div_and_check::<UT, VT, QT>(&u, &v, Some(1), false);

    const N_MAX_LIMBS: u32 = 3;
    for i in 0..N_MAX_LIMBS * LIMB_BITS + 1 {
        let u_len = (i as usize + 8 - 1) / 8;
        let mut u = tst_mk_mp_backing_vec!(UT, u_len);
        let mut u = UT::from_slice(&mut u).unwrap();
        if i != 0 {
            let u_nlimbs = ct_mp_nlimbs(u_len);
            for k in 0..u_nlimbs - 1 {
                u.store_l(k, !0);
            }
            if i % LIMB_BITS != 0 {
                let i = i % LIMB_BITS;
                u.store_l(u_nlimbs - 1, !0 >> (LIMB_BITS - i));
            } else {
                u.store_l(u_nlimbs - 1, !0);
            }
        }

        for j1 in 0..i + 1 {
            for j2 in 0..j1 + 1 {
                let v_len = ((j1 + 1) as usize + 8 - 1) / 8;
                let mut v = tst_mk_mp_backing_vec!(VT, v_len);
                let mut v = VT::from_slice(&mut v).unwrap();
                v.set_bit_to(j1 as usize, true);
                v.set_bit_to(j2 as usize, true);
                div_and_check::<UT, VT, QT>(&u, &v, None, false);
                div_and_check::<UT, VT, QT>(&u, &v, None, true);
                if j1 != 0 {
                    div_and_check::<UT, VT, QT>(&u, &v, Some(1), false);
                }
            }
        }
    }
}

#[test]
fn test_ct_div_be_be_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_div_mp_mp::<
        MpMutBigEndianUIntByteSlice,
        MpMutBigEndianUIntByteSlice,
        MpMutBigEndianUIntByteSlice,
    >()
}

#[test]
fn test_ct_div_le_le_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_div_mp_mp::<
        MpMutLittleEndianUIntByteSlice,
        MpMutLittleEndianUIntByteSlice,
        MpMutLittleEndianUIntByteSlice,
    >()
}

#[test]
fn test_ct_div_ne_ne_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_div_mp_mp::<
        MpMutNativeEndianUIntLimbsSlice,
        MpMutNativeEndianUIntLimbsSlice,
        MpMutNativeEndianUIntLimbsSlice,
    >()
}

pub fn ct_mod_mp_mp<UT: MpMutUIntSlice, VT: MpUIntCommon>(
    u_h: Option<&mut UT>,
    u_l: &mut UT,
    v: &CtMpDivisor<VT>,
) {
    // Specify an arbitrary MPIntMutByteSlice type for the non-existant q-argument.
    // The division will not return an InsufficientQuotientSpace error, because
    // there is none.
    ct_div_mp_mp::<_, _, MpMutNativeEndianUIntLimbsSlice>(u_h, u_l, v, None).unwrap()
}

#[derive(Debug)]
pub enum CtDivPow2MpError {
    InsufficientQuotientSpace,
    InsufficientRemainderSpace,
}

pub fn ct_div_pow2_mp<RT: MpMutUInt, VT: MpUIntCommon, QT: MpMutUInt>(
    u_pow2_exp: usize,
    r_out: &mut RT,
    v: &CtMpDivisor<VT>,
    mut q_out: Option<&mut QT>,
) -> Result<(), CtDivPow2MpError> {
    // Division algorithm according to D. E. Knuth, "The Art of Computer
    // Programming", vol 2 for the special case of the dividend being a power of
    // two.

    // The virtual length of the base 2 power in bytes.
    let virtual_u_len = ((u_pow2_exp + 1) + 8 - 1) / 8;
    let virtual_u_nlimbs = ct_mp_nlimbs(virtual_u_len);

    let v_len = v.v_len;
    let v_nlimbs = ct_mp_nlimbs(v_len);

    if let Some(q_out) = &mut q_out {
        // Check that q_out has enough space for storing the maximum possible quotient.
        // In general, at most one more bit than the bit width difference
        // between u and v is needed. The byte granularity is too coarse to
        // catch that, so only check the absolute lower bound here.
        // The code storing the quotient below will verify that the head limb is zero in
        // case it's not been provided storage for.
        if q_out.len() + v_len < virtual_u_len {
            return Err(CtDivPow2MpError::InsufficientQuotientSpace);
        }
    };

    if r_out.len() < v_len.min(virtual_u_len) {
        return Err(CtDivPow2MpError::InsufficientRemainderSpace);
    }
    r_out.clear_bytes_above(0);

    // The virtual high limb of the base 2 power.
    let u_high_shift = (u_pow2_exp % LIMB_BITS as usize) as u32;
    let u_high = (1 as LimbType) << u_high_shift;
    if virtual_u_len < v_len {
        r_out.store_l(virtual_u_nlimbs - 1, u_high);
        if let Some(q_out) = q_out {
            q_out.clear_bytes_above(0);
        }
        return Ok(());
    }

    let q_out_max_len = virtual_u_len + 1 - v_len;
    let q_out_max_nlimbs = ct_mp_nlimbs(q_out_max_len);
    if let Some(q_out) = &mut q_out {
        q_out.clear_bytes_above(q_out_max_len);
    };

    // Scale u. Note that as being a power of two, only its (current) most
    // significant high limb is non-zero.
    let (scaled_u_head_carry, scaled_u_head_low) = v.scale_val(u_high, 0);
    let (scaled_u_head_carry, scaled_u_head_high) = v.scale_val(0, scaled_u_head_carry);
    debug_assert_eq!(scaled_u_head_carry, 0);
    let mut r_out_head_shadow: [LimbType; 2] = [scaled_u_head_low, scaled_u_head_high];

    // Note that q_out_max_nlimbs as calculate above doesn't necessarily equal
    // u_nlimbs - v_nlimbs + 1, but might come out to be one less. It can be
    // shown that q_out_max_nlimbs = u_nlimbs - v_nlimbs implies that the
    // scaling of u above would not have overflown into the scaled_u_head_high
    // now found in the r_out_head_shadow[] high limb at index 1.
    //
    // [To see this, write
    //   u.len() - 1 = q_ul * LIMB_BYTES + r_ul, with r_ul < LIMB_BYTES
    //  and
    //   v.len() - 1 = q_vl * LIMB_BYTES + r_vl, with r_vl < LIMB_BYTES.
    //
    //  With that,
    //   u_nlimbs = q_ul + 1
    //  and
    //   v_nlimbs = q_vl + 1.
    //  By definition (all subsequent divisions are meant to be integer divisions),
    //   q_out_max_nlimbs
    //    = (u.len() - v.len() + 1 + LIMB_BYTES - 1) / LIMB_BYTES
    //    = (u.len() - v.len()) / LIMB_BYTES + 1
    //    = (u.len() - 1 - (v.len() - 1)) / LIMB_BYTES + 1
    //    = (q_ul * LIMB_BYTES + r_ul - (q_vl * LIMB_BYTES + r_vl)) / LIMB_BYTES + 1
    //    = q_ul - q_vl - (r_vl - r_ul + LIMB_BYTES - 1) / LIMB_BYTES + 1
    //  The latter equals either
    //    = q_ul - q_vl + 1 = u_nlimbs - v_nlimbs + 1
    //  or
    //    = q_ul - q_vl     = u_nlimbs - v_nlimbs,
    //  depending on whether rv <= r_ul or not.
    //
    //  To see how this relates to the scaled u overflowing into the next higher
    //  limb or not, note that the high limb of (unscaled) u has exactly r_ul + 1
    //  of its least signigicant bytes non-zero and similarly does the high
    //  limb of v have exactly r_vl + 1 of its least significant bytes
    //  non-zero. The latter determines the value of scaling, which won't have
    //  more than LIMB_BYTES - (r_vl + 1) + 1 = LIMB_BYTES - r_vl of the least
    //  significant bytes set: remember that the scaling is chosen such that
    //  the high limb of v multiplied by the scaling makes the high bit in the
    //  limb set, but does not overflow it. Multiplying u by the scaling
    //  extends its length by the length of the scaling at most, i.e.
    //  by LIMB_BYTES - r_vl. Before the scaling operation, u has
    //  LIMB_BYTES - (r_ul + 1) of its most signifcant bytes zero, and thus, as
    //  long as LIMB_BYTES - (r_ul + 1) >= LIMB_BYTES - r_vl, the scaling is
    //  guaranteed not to overflow into the next higher limb. Observe how this
    //  latter condition is equivalent to r_vl > r_ul, which, as shown above, is
    //  in turn equivalent to q_out_max_nlimbs taking the smaller of the two
    //  possible values: q_out_max_nlimbs = u_nlimbs - v_nlimbs.]
    //
    // This matters insofar, as the current setup of r_out_head_shadow[] is such
    // that the sliding window approach to the division from below would start
    // at the point in the virtual, scaled u valid only for the case that the
    // scaling did overflow, i.e. at the (virtual) limb position just above the
    // one which should have been taken for the non-overflowing case. The quotient
    // limb obtained for this initial position would come out to zero, but this
    // superfluous computation would consume one iteration from the total count
    // of q_out_max_nlimbs ones actually needed. Simply incrementing
    // q_out_max_nlimbs to account for that would not work, as there would be no
    // slot for storing the extra zero high limb of q available in the output
    // q_out[] argument. Instead, if q_out_max_nlimbs is the smaller of the two
    // possible values, i.e. equals u_nlimbs - v_nlimbs, tweak
    // r_out_head_shadow[] to the state it would have had after one (otherwise
    // superfluous) initial iteration of the division loop.
    debug_assert!(
        v_nlimbs + q_out_max_nlimbs == virtual_u_nlimbs + 1
            || v_nlimbs + q_out_max_nlimbs == virtual_u_nlimbs
    );
    if v_nlimbs + q_out_max_nlimbs == virtual_u_nlimbs {
        debug_assert_eq!(r_out_head_shadow[1], 0);
        r_out_head_shadow[1] = r_out_head_shadow[0];
        r_out_head_shadow[0] = 0;
    }

    // For the division loop, (u_h, u_l, r_out[v_nlimbs - 3:0]) acts as a sliding
    // window over the v_nlimbs most significant limbs of the dividend, which is
    // known to have its remaining tail equal all zeroes. As the loop
    // progresses, the sliding window gets extended on the right by
    // a virtual zero to construct the v_nlimbs + 1 dividend for a single division
    // step. After the division step, the the most significant limb is known to
    // haven been made zero as per the basic long division algorithm's
    // underlying principle. That is, it can be removed from the left, thereby
    // effectively moving the sliding window one limb to the right.
    let mut j = q_out_max_nlimbs;
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
            let cur_u_head: [LimbType; 3] =
                [r_out_head_shadow[1], r_out_head_shadow[0], u_tail_high];

            v.make_q_estimate(&cur_u_head)
        };

        // Virtually extend the v_nlimbs-limb sliding window by a zero on the right,
        // subtract q * v from it and remove the most significant limb, which is known
        // to eventually turn out zero anyway.
        // In case v_nlimbs < 2, this initialization of u_val reflects the extension by
        // a zero limb on the right, which will land in r_out_head_shadow[0] below.
        let mut v_decrement_borrow = v.v_decrement.unwrap_or(0);
        let mut scaled_v_carry = 0;
        let mut borrow = 0;
        let mut i = 0;
        let mut u_val = if v_nlimbs >= 2 {
            // Subtract q*v from the virtually shifted tail maintained in
            // r_out[v_nlimbs - 3:0], if any, return the value shifted out
            // on the left.
            let mut next_u_val = 0; // The zero shifted in from the right.
            while i + 2 < v_nlimbs {
                let mut u_val = next_u_val;
                next_u_val = r_out.load_l_full(i);
                (v_decrement_borrow, scaled_v_carry, borrow, u_val) =
                    v.sub_scaled_qv_val(u_val, i, q, v_decrement_borrow, scaled_v_carry, borrow);
                r_out.store_l_full(i, u_val);
                i += 1;
            }

            // Calculate the value that got shifted out on the left and goes into the next
            // higher limb, r_out_head_shadow[0].
            {
                let mut u_val = next_u_val;
                (v_decrement_borrow, scaled_v_carry, borrow, u_val) =
                    v.sub_scaled_qv_val(u_val, i, q, v_decrement_borrow, scaled_v_carry, borrow);
                i += 1;
                u_val
            }
        } else {
            // For the case that v_nlimbs == 1, only store the shifted in zero in
            // r_out_head_shadow[0] below. It will serve as input to the the next long
            // division iteration, if any.
            0
        };

        // The remaining two head limbs in r_out_head_shadow[]. Note that for the most
        // significant limb in r_out_head_shadow[1], there's only the borrow
        // left to subtract.
        debug_assert_eq!(i + 1, v_nlimbs);
        {
            let cur_u_val = r_out_head_shadow[0];
            r_out_head_shadow[0] = u_val;
            (_, scaled_v_carry, borrow, u_val) =
                v.sub_scaled_qv_val(cur_u_val, i, q, v_decrement_borrow, scaled_v_carry, borrow);
            debug_assert_eq!(scaled_v_carry, 0);
            let cur_u_val = r_out_head_shadow[1];
            r_out_head_shadow[1] = u_val;
            (borrow, u_val) = ct_sub_l_l(cur_u_val, borrow);
            debug_assert!(borrow != 0 || u_val == 0);
        }

        // If borrow != 0, then the estimate for q had been one too large. Decrement it
        // and add one v back to the remainder accordingly.
        let over_estimated = LimbChoice::from(borrow);
        if let Some(q_out) = &mut q_out {
            let q = q - over_estimated.select(0, 1);
            if j != q_out.nlimbs() {
                debug_assert!(j < q_out.nlimbs());
                q_out.store_l(j, q);
            } else if q != 0 {
                return Err(CtDivPow2MpError::InsufficientQuotientSpace);
            }
        }
        let mut v_decrement_borrow = v.v_decrement.unwrap_or(0);
        let mut carry = 0;
        let mut i = 0;
        // Update the tail maintained in r_out[v_nlimbs - 3:0], if any:
        while i + 2 < v_nlimbs {
            let mut u_val = r_out.load_l_full(i);
            (v_decrement_borrow, carry, u_val) =
                v.add_scaled_v_val_cond(u_val, i, v_decrement_borrow, carry, over_estimated);
            r_out.store_l_full(i, u_val);
            i += 1;
        }
        // Take care of the two high limbs in r_out_head_shadow[].
        // Note that if v_nlimbs == 1, then r_out_head_shadow[0] does not correspond to
        // an actual result limb of the preceeding q * v subtraction, but
        // already holds the zero to virtually append to the sliding window in
        // the loop's next iteration, if any. In this case, it must not
        // be considered for the addition of v here.
        let r_out_head_shadow_cur_sliding_window_overlap = v_nlimbs.min(2);
        for k in 0..r_out_head_shadow_cur_sliding_window_overlap {
            let k = 2 - r_out_head_shadow_cur_sliding_window_overlap + k;
            let mut u_val = r_out_head_shadow[k];
            (v_decrement_borrow, carry, u_val) =
                v.add_scaled_v_val_cond(u_val, i, v_decrement_borrow, carry, over_estimated);
            r_out_head_shadow[k] = u_val;
            i += 1;
        }
        debug_assert_eq!(i, v_nlimbs);
    }

    // Finally, divide the resulting remainder in r_out by the scaling again.
    let mut u_h = 0;
    // The two high limbs in r_out_head_shadow come first. Descale them and store
    // them into their corresponding locations in the returned r_out[]. Note
    // that if v_nlimbs == 1, then the less significant one in
    // r_out_head_shadow[0] bears no significance.
    debug_assert!(v_nlimbs > 1 || r_out_head_shadow[0] == 0);
    for k in 0..v_nlimbs.min(2) {
        let u_l = r_out_head_shadow[2 - 1 - k];
        let u_val = v.unscale_val(u_l, u_h);
        u_h = u_l;
        r_out.store_l(v_nlimbs - 1 - k, u_val);
    }

    // Now do the remaining part in r_out[v_nlimbs - 3:0].
    let mut j = v_nlimbs;
    while j > 2 {
        j -= 1;
        let u_l = r_out.load_l_full(j - 2);
        let u_val = v.unscale_val(u_l, u_h);
        u_h = u_l;
        r_out.store_l_full(j - 2, u_val);
    }

    Ok(())
}

#[cfg(test)]
fn test_ct_div_pow2_mp<RT: MpMutUIntSlice, VT: MpMutUIntSlice, QT: MpMutUIntSlice>() {
    fn div_and_check<RT: MpMutUIntSlice, VT: MpMutUIntSlice, QT: MpMutUIntSlice>(
        u_pow2_exp: usize,
        v: &VT::SelfT<'_>,
        v_decrement: Option<LimbType>,
    ) {
        use super::add_impl::{ct_add_mp_mp, ct_sub_mp_l};
        use super::mul_impl::ct_mul_trunc_mp_mp;

        let mut decremented_v = tst_mk_mp_backing_vec!(VT, v.len());
        let mut decremented_v = VT::from_slice(&mut decremented_v).unwrap();
        decremented_v.copy_from(v);
        if let Some(v_decrement) = v_decrement {
            ct_sub_mp_l(&mut decremented_v, v_decrement);
        }

        let u_len = (u_pow2_exp + 1 + 8 - 1) / 8;
        let v_len = find_last_set_byte_mp(&decremented_v);
        let q_len = if u_len >= v_len { u_len - v_len + 1 } else { 0 };

        let mut q = tst_mk_mp_backing_vec!(QT, q_len);
        q.fill(0xffu8.into());
        let mut q = QT::from_slice(&mut q).unwrap();
        let mut rem = tst_mk_mp_backing_vec!(RT, v_len);
        rem.fill(0xffu8.into());
        let mut rem = RT::from_slice(&mut rem).unwrap();
        ct_div_pow2_mp(
            u_pow2_exp as usize,
            &mut rem,
            &CtMpDivisor::new(v, v_decrement).unwrap(),
            Some(&mut q),
        )
        .unwrap();

        // Multiply q by v again and add the remainder back, the result should match the
        // initial u. Reserve one extra limb, which is expected to come to zero.
        let mut result = tst_mk_mp_backing_vec!(QT, u_len + LIMB_BYTES);
        result.fill(0xffu8.into());
        let mut result = QT::from_slice(&mut result).unwrap();
        result.copy_from(&q);
        ct_mul_trunc_mp_mp(&mut result, q_len, &decremented_v);
        let carry = ct_add_mp_mp(&mut result, &rem);
        assert_eq!(carry, 0);
        let u_nlimbs = ct_mp_nlimbs(u_len);
        for i in 0..u_nlimbs - 1 {
            assert_eq!(result.load_l_full(i), 0);
        }
        let expected_high = (1 as LimbType) << (u_pow2_exp % (LIMB_BITS as usize));
        assert_eq!(result.load_l_full(u_nlimbs - 1), expected_high);
        assert_eq!(result.load_l(u_nlimbs), 0);
    }

    let mut v = tst_mk_mp_backing_vec!(VT, LIMB_BYTES);
    for v0 in [1 as LimbType, 7, 13, 17, 251] {
        for k in 0..LIMB_BYTES {
            let v0 = v0 << 8 * k;
            let mut v = VT::from_slice(v.as_mut_slice()).unwrap();
            v.store_l(0, v0);
            for i in 0..5 * LIMB_BITS as usize {
                div_and_check::<RT, VT, QT>(i, &v, None);
                if v0 != 1 {
                    div_and_check::<RT, VT, QT>(i, &v, Some(1));
                }
            }
        }
    }

    let mut v = tst_mk_mp_backing_vec!(VT, 2 * LIMB_BYTES);
    for v_h in [0 as LimbType, 1, 7, 13, 17, 251] {
        for v_l in [0 as LimbType, 1, 7, 13, 17, 251] {
            if v_h == 0 && v_l == 0 {
                continue;
            }

            for k in 0..LIMB_BYTES {
                let v_h = v_h << 8 * k;
                let mut v = VT::from_slice(v.as_mut_slice()).unwrap();
                v.store_l(0, v_l);
                v.store_l(1, v_h);
                for i in 0..6 * LIMB_BITS as usize {
                    div_and_check::<RT, VT, QT>(i, &v, None);
                    if v_l != 0 && v_h != 0 {
                        div_and_check::<RT, VT, QT>(i, &v, Some(v_l + 1));
                    }
                }
            }
        }
    }
}

#[test]
fn test_ct_div_pow2_be_be_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_div_pow2_mp::<
        MpMutBigEndianUIntByteSlice,
        MpMutBigEndianUIntByteSlice,
        MpMutBigEndianUIntByteSlice,
    >()
}

#[test]
fn test_ct_div_pow2_le_le_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_div_pow2_mp::<
        MpMutLittleEndianUIntByteSlice,
        MpMutLittleEndianUIntByteSlice,
        MpMutLittleEndianUIntByteSlice,
    >()
}

#[test]
fn test_ct_div_pow2_ne_ne_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_div_pow2_mp::<
        MpMutNativeEndianUIntLimbsSlice,
        MpMutNativeEndianUIntLimbsSlice,
        MpMutNativeEndianUIntLimbsSlice,
    >()
}

#[derive(Debug)]
pub enum CtModPow2MpError {
    InsufficientRemainderSpace,
}

pub fn ct_mod_pow2_mp<RT: MpMutUInt, VT: MpUIntCommon>(
    u_pow2_exp: usize,
    r_out: &mut RT,
    v: &CtMpDivisor<VT>,
) -> Result<(), CtModPow2MpError> {
    // Specify an arbitrary MPIntMutByteSlice type for the non-existant q-argument.
    ct_div_pow2_mp::<_, _, MpMutNativeEndianUIntLimbsSlice>(u_pow2_exp, r_out, v, None).map_err(
        |e| {
            match e {
                CtDivPow2MpError::InsufficientRemainderSpace => {
                    CtModPow2MpError::InsufficientRemainderSpace
                }
                CtDivPow2MpError::InsufficientQuotientSpace => {
                    // No quotient, no insufficient quotient space.
                    unreachable!()
                }
            }
        },
    )
}

pub type CtDivLshiftedMpMpError = CtDivPow2MpError;

pub fn ct_div_lshifted_mp_mp<UT: MpMutUInt, VT: MpUIntCommon, QT: MpMutUInt>(
    u: &mut UT,
    u_in_len: usize,
    u_lshift_len: usize,
    v: &CtMpDivisor<VT>,
    mut q_out: Option<&mut QT>,
) -> Result<(), CtDivLshiftedMpMpError> {
    // Division algorithm according to D. E. Knuth, "The Art of Computer
    // Programming", vol 2 adapted to the case of a dividend extended on the
    // right by u_lshift_len zero bytes.
    //
    // The division proceeds in two parts:
    // - first a regular long division is run on the head passed in in u[],
    // - and subsequently the long division is continued on the zero bytes shifted
    //   in on the right. This second part takes advantage of the fact that each
    //   step of the long division algorithm makes the dividend's (or, more
    //   precisely, the current remainder's) high limb zero by design. This allows
    //   to employ a memory-efficient "sliding window" approach where u[] is
    //   successively getting moved to the right in the virtually shifted dividend
    //   for each zero extension on the right and high limb correspondingly
    //   eliminated.

    debug_assert!(u_in_len <= u.len());
    let virtual_u_in_len = u_in_len + u_lshift_len;

    let v_len = v.v_len;
    let v_nlimbs = ct_mp_nlimbs(v_len);

    if let Some(q_out) = &mut q_out {
        // Check that q_out has enough space for storing the maximum possible quotient.
        // In general, at most one more bit than the bit width difference
        // between u and v is needed. The byte granularity is too coarse to
        // catch that, so only check the absolute lower bound here.
        // The code storing the quotient below will verify that the head limb is zero in
        // case it's not been provided storage for.
        if q_out.len() + v_len < virtual_u_in_len {
            return Err(CtDivLshiftedMpMpError::InsufficientQuotientSpace);
        }
    }

    if u.len() < v_len.min(virtual_u_in_len) {
        return Err(CtDivLshiftedMpMpError::InsufficientRemainderSpace);
    }

    if virtual_u_in_len < v_len {
        ct_lshift_mp(u, 8 * u_lshift_len);
        if let Some(q_out) = q_out {
            q_out.clear_bytes_above(0);
        }
        return Ok(());
    }

    let q_out_max_len = virtual_u_in_len + 1 - v_len;
    let q_out_max_nlimbs = ct_mp_nlimbs(q_out_max_len);
    if let Some(q_out) = &mut q_out {
        q_out.clear_bytes_above(q_out_max_len);
    }

    // The unaligned part of u_lshift_len will be taken into account by shifting the
    // dividend by that (small) amount to the left before running the regular
    // long division on the (now shifted) dividend in a first step. The
    // remaining, aligned tail of the shift distance will subsequently
    // be handled by the memory-efficient "sliding window" approach in a second step
    // below.
    let u_lshift_head_len = u_lshift_len % LIMB_BYTES;
    let u_lshift_tail_len = u_lshift_len - u_lshift_head_len;
    // If there's enough room left in u[] plus the high limb receiving the couple of
    // bytes shifted out from u on the left below, shovel an integral number of
    // limbs from the tail to the head. This will save the corresponding number
    // of iterations in the subsequent "sliding window" step.
    let u_lshift_head_len = ((
        ct_mp_nlimbs(
            (u.len() - u_in_len + LIMB_BYTES - u_lshift_head_len) + 1, // For rounding downwards.
        ) - 1
        // Ditto.
    ) * LIMB_BYTES)
        .min(u_lshift_tail_len)
        + u_lshift_head_len;
    debug_assert_eq!(u_lshift_head_len % LIMB_BYTES, u_lshift_len % LIMB_BYTES);
    let u_lshift_tail_len = u_lshift_len - u_lshift_head_len;

    // Maintain a shadow of the shifted + scaled dividend's/current remainder's
    // three most significant limbs throughout the regular long division run in
    // the first part:
    // - The least significant limb in u_head_high_shadow[0] will be needed to
    //   account for fact that u.len() might not be aligned to a limb boundary and
    //   thus, a partial high limb could potentially overflow during the
    //   computations.
    // - The next, more significant shadow limb in in u_head_high_shadow[1] will
    //   receive the bits shifted out from u[] on the left.
    // - The most significant shadow limb in u_head_high_shadow[2] will store the
    //   overflow, if any, of the scaling.
    let mut u_head_high_shadow: [LimbType; 3] = [0; 3];
    u_head_high_shadow[0] = ct_lshift_mp(u, 8 * u_lshift_head_len);
    // The (original) u length might not be aligned to the limb size. Move the high
    // limb into the u_head_high_shadow[0] shadow for the duration of the
    // computation. Make sure the limb shifted out on the left from u just above
    // moves to the left in u_head_high_shadow[] accordingly.
    let u_head_high_partial_len = u.len() % LIMB_BYTES;
    let u_nlimbs = ct_mp_nlimbs(u.len());
    if u_head_high_partial_len != 0 {
        u_head_high_shadow[1] =
            u_head_high_shadow[0] >> (8 * (LIMB_BYTES - (u_head_high_partial_len)));
        u_head_high_shadow[0] <<= 8 * u_head_high_partial_len;
        u_head_high_shadow[0] |= u.load_l(u_nlimbs - 1);
    } else {
        u_head_high_shadow[1] = u_head_high_shadow[0];
        u_head_high_shadow[0] = u.load_l(u_nlimbs - 1);
    }
    // At this point,
    // - u_head_high_shadow[1] contains (some of) the bits shifted out from u[] on
    //   the left and
    // - u_head_high_shadow[0] acts as a shadow for the potentially partial high
    //   limb of the shifted u[].
    // Scale u.
    let mut carry = 0;
    for i in 0..u_nlimbs - 1 {
        let mut u_val = u.load_l_full(i);
        (carry, u_val) = v.scale_val(u_val, carry);
        u.store_l_full(i, u_val);
    }
    for u_val in u_head_high_shadow.iter_mut().take(3 - 1) {
        (carry, *u_val) = v.scale_val(*u_val, carry);
    }
    u_head_high_shadow[2] = carry;

    // Now, in a first step, run the regular long division on the head part of the
    // shifted u, i.e. on the original u shifted left by u_lshift_len_head.
    let u_lshift_tail_nlimbs = u_lshift_tail_len / LIMB_BYTES; // The len is aligned.
    let q_out_head_nlimbs = q_out_max_nlimbs - u_lshift_tail_nlimbs;
    let mut j = q_out_head_nlimbs;
    while j > 0 {
        j -= 1;
        let q = {
            // Estimate q. Load the first three u-limbs needed for the q estimation, i.e.
            // the ones at indices v_nlimbs + j, v_nlimbs + j - 1 and
            // v_nlimbs + j - 2. Depending on where we are currently, they need to be read
            // either from the u_head_shadow[] or from u[] itself. In case
            // v_nlimbs == 1 and j == 0, the least signigicant of the three u-limbs
            // would be undefined. As per the comment in q_estimate(), it can be set to an
            // arbitrary value in this case, so just leave it zero.
            let mut cur_u_head: [LimbType; 3] = [0; 3];
            let mut i = 3;
            while i > 0 && v_nlimbs + j + i >= u_nlimbs - 1 + 3 {
                i -= 1;
                cur_u_head[3 - i - 1] = u_head_high_shadow[(v_nlimbs + j + i - 2) - (u_nlimbs - 1)];
            }
            while i > 0 && v_nlimbs + j + i >= 3 {
                i -= 1;
                cur_u_head[3 - i - 1] = u.load_l_full(v_nlimbs + j + i - 2);
            }
            v.make_q_estimate(&cur_u_head)
        };

        // Subtract q * v at limb position j upwards in u[].
        let mut v_decrement_borrow = v.v_decrement.unwrap_or(0);
        let mut scaled_v_carry = 0;
        let mut borrow = 0;
        let mut i = 0;
        while i < v_nlimbs && j + i < u_nlimbs - 1 {
            let mut u_val = u.load_l_full(j + i);
            (v_decrement_borrow, scaled_v_carry, borrow, u_val) =
                v.sub_scaled_qv_val(u_val, i, q, v_decrement_borrow, scaled_v_carry, borrow);
            u.store_l_full(j + i, u_val);
            i += 1;
        }
        while i < v_nlimbs {
            let mut u_val = u_head_high_shadow[j + i - (u_nlimbs - 1)];
            (v_decrement_borrow, scaled_v_carry, borrow, u_val) =
                v.sub_scaled_qv_val(u_val, i, q, v_decrement_borrow, scaled_v_carry, borrow);
            u_head_high_shadow[j + i - (u_nlimbs - 1)] = u_val;
            i += 1;
        }
        debug_assert_eq!(scaled_v_carry, 0);
        // Take the final borrow into account.
        assert_eq!(i, v_nlimbs);
        if j + i < u_nlimbs - 1 {
            let mut u_val = u.load_l_full(j + i);
            (borrow, u_val) = ct_sub_l_l(u_val, borrow);
            u.store_l_full(j + i, u_val);
        } else {
            let mut u_val = u_head_high_shadow[j + i - (u_nlimbs - 1)];
            (borrow, u_val) = ct_sub_l_l(u_val, borrow);
            u_head_high_shadow[j + i - (u_nlimbs - 1)] = u_val;
        }

        // If borrow != 0, then the estimate for q had been one too large. Decrement it
        // and add one v back to the remainder accordingly.
        let over_estimated = LimbChoice::from(borrow);
        if let Some(q_out) = &mut q_out {
            let q = q - over_estimated.select(0, 1);
            if u_lshift_tail_nlimbs + j != q_out.nlimbs() {
                debug_assert!(u_lshift_tail_nlimbs + j < q_out.nlimbs());
                q_out.store_l(u_lshift_tail_nlimbs + j, q);
            } else if q != 0 {
                return Err(CtDivLshiftedMpMpError::InsufficientQuotientSpace);
            }
        }
        let mut v_decrement_borrow = v.v_decrement.unwrap_or(0);
        let mut carry = 0;
        let mut i = 0;
        while i < v_nlimbs && j + i < u_nlimbs - 1 {
            let mut u_val = u.load_l_full(j + i);
            (v_decrement_borrow, carry, u_val) =
                v.add_scaled_v_val_cond(u_val, i, v_decrement_borrow, carry, over_estimated);
            u.store_l_full(j + i, u_val);
            i += 1;
        }
        while i < v_nlimbs {
            let mut u_val = u_head_high_shadow[j + i - (u_nlimbs - 1)];
            (v_decrement_borrow, carry, u_val) =
                v.add_scaled_v_val_cond(u_val, i, v_decrement_borrow, carry, over_estimated);
            u_head_high_shadow[j + i - (u_nlimbs - 1)] = u_val;
            i += 1;
        }
        // Take the final carry into account.
        assert_eq!(i, v_nlimbs);
        if j + i < u_nlimbs - 1 {
            let mut u_val = u.load_l_full(j + i);
            (_, u_val) = ct_add_l_l(u_val, carry);
            u.store_l_full(j + i, u_val);
        } else {
            let mut u_val = u_head_high_shadow[j + i - (u_nlimbs - 1)];
            (_, u_val) = ct_add_l_l(u_val, carry);
            u_head_high_shadow[j + i - (u_nlimbs - 1)] = u_val;
        }
    }
    debug_assert_eq!(u_head_high_shadow[2], 0);
    debug_assert_eq!(u_head_high_shadow[1], 0);
    if u_nlimbs > v_nlimbs {
        debug_assert_eq!(u_head_high_shadow[0], 0);
        u.store_l(u_nlimbs - 1, 0);
    }
    for i in v_nlimbs..u_nlimbs - 1 {
        debug_assert_eq!(u.load_l_full(i), 0);
    }

    // Second step: divide the current remainder in u[], extended virtually by
    // u_lshift_tail_len more zeroes at the right. Again, because u.len() might
    // not be aligned, maintain a shadow limb for the case that u[]'s high limb
    // is partial and u_nlimbs == v_nlimbs.
    let mut u_high_shadow = if v_nlimbs - 1 == u_nlimbs - 1 {
        u_head_high_shadow[0]
    } else {
        u.load_l_full(v_nlimbs - 1)
    };

    let mut j = u_lshift_tail_nlimbs;
    while j > 0 {
        j -= 1;

        // Estimate q. Load the first three u-limbs needed for the q estimation.
        // Depending on where we are currently, they need to be read either from
        // the u_head_shadow[] or from u[] itself. In case v_nlimbs == 1 and
        // j == 0, the least signigicant of the three u-limbs would be undefined.
        // As per the comment in q_estimate(), it can be set to an arbitrary
        // value in this case, so leave it zero.
        let q = {
            let mut cur_u_head: [LimbType; 3] = [0; 3];
            cur_u_head[0] = u_high_shadow;
            cur_u_head[1] = if v_nlimbs >= 2 {
                u.load_l_full(v_nlimbs - 2)
            } else {
                0 // Virtual zero shifted in on the right.
            };
            cur_u_head[2] = if v_nlimbs >= 3 {
                u.load_l_full(v_nlimbs - 3)
            } else {
                0 // Virtual zero shifted in on the right or, if v_nlimbs == 1
                  // and j == 0, undefined.
            };

            v.make_q_estimate(&cur_u_head)
        };

        // Virtually shift u one limb to the left, add q * v and drop the (now zero)
        // high limb. This effectively moves the sliding window one limb to the
        // right.
        let mut next_u_val = 0; // The zero shifted in on the right.
        let mut v_decrement_borrow = v.v_decrement.unwrap_or(0);
        let mut scaled_v_carry = 0;
        let mut borrow = 0;
        for i in 0..v_nlimbs - 1 {
            let mut u_val = next_u_val;
            next_u_val = u.load_l_full(i);
            (v_decrement_borrow, scaled_v_carry, borrow, u_val) =
                v.sub_scaled_qv_val(u_val, i, q, v_decrement_borrow, scaled_v_carry, borrow);
            u.store_l_full(i, u_val);
        }
        // u[v_nlimbs - 1] is maintained in the u_high_shadow shadow, handle it
        // separately.
        {
            let i = v_nlimbs - 1;
            let mut u_val = next_u_val;
            next_u_val = u_high_shadow;
            (_, scaled_v_carry, borrow, u_val) =
                v.sub_scaled_qv_val(u_val, i, q, v_decrement_borrow, scaled_v_carry, borrow);
            debug_assert_eq!(scaled_v_carry, 0);
            u_high_shadow = u_val;
        }
        (borrow, _) = ct_sub_l_l(next_u_val, borrow);

        // If borrow != 0, then the estimate for q had been one too large. Decrement it
        // and add one v back to the remainder accordingly.
        let over_estimated = LimbChoice::from(borrow);
        if let Some(q_out) = &mut q_out {
            let q = q - over_estimated.select(0, 1);
            if j != q_out.nlimbs() {
                debug_assert!(j < q_out.nlimbs());
                q_out.store_l(j, q);
            } else if q != 0 {
                return Err(CtDivLshiftedMpMpError::InsufficientQuotientSpace);
            }
        }

        let mut v_decrement_borrow = v.v_decrement.unwrap_or(0);
        let mut carry = 0;
        for i in 0..v_nlimbs - 1 {
            let mut u_val = u.load_l_full(i);
            (v_decrement_borrow, carry, u_val) =
                v.add_scaled_v_val_cond(u_val, i, v_decrement_borrow, carry, over_estimated);
            u.store_l_full(i, u_val);
        }
        // u[v_nlimbs - 1] is maintained in the u_high_shadow shadow, handle it
        // separately.
        {
            let i = v_nlimbs - 1;
            let mut u_val = u_high_shadow;
            (_, _, u_val) =
                v.add_scaled_v_val_cond(u_val, i, v_decrement_borrow, carry, over_estimated);
            u_high_shadow = u_val;
        }
    }

    // Finally, divide the resulting remainder in u by the scaling again.
    let mut u_h = 0;
    // The high limb maintained at u_high_shadow comes first. Descale and store in
    // its final location.
    {
        let u_l = u_high_shadow;
        let u_val = v.unscale_val(u_l, u_h);
        u_h = u_l;
        u.store_l(v_nlimbs - 1, u_val);
    }
    // Now do the remaining limbs in u[v_nlimbs - 2:0].
    let mut j = v_nlimbs - 1;
    while j > 0 {
        j -= 1;
        let u_l = u.load_l_full(j);
        let u_val = v.unscale_val(u_l, u_h);
        u_h = u_l;
        u.store_l(j, u_val);
    }

    Ok(())
}

#[cfg(test)]
fn test_ct_div_lshifted_mp_mp<UT: MpMutUIntSlice, VT: MpMutUIntSlice, QT: MpMutUIntSlice>() {
    fn div_and_check<UT: MpMutUIntSlice, VT: MpMutUIntSlice, QT: MpMutUIntSlice>(
        u: &UT::SelfT<'_>,
        u_in_len: usize,
        u_lshift_len: usize,
        v: &VT::SelfT<'_>,
        v_decrement: Option<LimbType>,
    ) {
        use super::add_impl::{ct_add_mp_mp, ct_sub_mp_l};
        use super::cmp_impl::ct_eq_mp_mp;
        use super::mul_impl::ct_mul_trunc_mp_mp;
        use super::shift_impl::ct_rshift_mp;

        let mut decremented_v = tst_mk_mp_backing_vec!(VT, v.len());
        let mut decremented_v = VT::from_slice(&mut decremented_v).unwrap();
        decremented_v.copy_from(v);
        if let Some(v_decrement) = v_decrement {
            ct_sub_mp_l(&mut decremented_v, v_decrement);
        }

        let v_len = find_last_set_byte_mp(&decremented_v);
        let virtual_u_len = u_in_len + u_lshift_len;
        let q_len = virtual_u_len + 1 - v_len;
        let mut q = tst_mk_mp_backing_vec!(QT, q_len);
        q.fill(0xffu8.into());
        let mut q = QT::from_slice(q.as_mut_slice()).unwrap();
        let mut rem = tst_mk_mp_backing_vec!(UT, u.len());
        let mut rem = UT::from_slice(&mut rem).unwrap();
        rem.copy_from(u);
        ct_div_lshifted_mp_mp(
            &mut rem,
            u_in_len,
            u_lshift_len,
            &CtMpDivisor::new(v, v_decrement).unwrap(),
            Some(&mut q),
        )
        .unwrap();

        // Multiply q by v again and add the remainder back, the result should match the
        // initial u. Reserve one extra limb, which is expected to come to zero.
        let mut result = tst_mk_mp_backing_vec!(UT, virtual_u_len + LIMB_BYTES);
        result.fill(0xffu8.into());
        let mut result = UT::from_slice(&mut result).unwrap();
        result.copy_from(&q);
        ct_mul_trunc_mp_mp(&mut result, q_len, &decremented_v);
        let carry = ct_add_mp_mp(&mut result, &rem);
        assert_eq!(carry, 0);
        for i in 0..ct_mp_nlimbs(u_lshift_len + 1) - 1 {
            assert_eq!(result.load_l(i), 0);
        }
        if u_lshift_len % LIMB_BYTES != 0 {
            let u_val = result.load_l(ct_mp_nlimbs(u_lshift_len + 1) - 1);
            assert_eq!(
                u_val & ct_lsb_mask_l(8 * (u_lshift_len % LIMB_BYTES) as u32),
                0
            );
        }
        assert_eq!(result.load_l(ct_mp_nlimbs(virtual_u_len)), 0);
        ct_rshift_mp(&mut result, 8 * u_lshift_len);
        assert_eq!(ct_eq_mp_mp(u, &result).unwrap(), 1);
    }

    const N_MAX_LIMBS: u32 = 3;
    for i in 0..N_MAX_LIMBS * LIMB_BITS + 1 {
        let u_len = (i as usize + 8 - 1) / 8;
        for j1 in 0..i + 1 {
            for j2 in 0..j1 + 1 {
                let v_len = ((j1 + 1) as usize + 8 - 1) / 8;
                for u_lshift_len in [
                    0,
                    LIMB_BYTES - 1,
                    LIMB_BYTES,
                    LIMB_BYTES + 1,
                    2 * LIMB_BYTES - 1,
                    2 * LIMB_BYTES,
                ] {
                    let mut u = tst_mk_mp_backing_vec!(UT, u_len.max(v_len));
                    let mut u = UT::from_slice(&mut u).unwrap();
                    if i != 0 {
                        let u_nlimbs = ct_mp_nlimbs(u_len);
                        for k in 0..u_nlimbs - 1 {
                            u.store_l(k, !0);
                        }
                        if i % LIMB_BITS != 0 {
                            let i = i % LIMB_BITS;
                            u.store_l(u_nlimbs - 1, !0 >> (LIMB_BITS - i));
                        } else {
                            u.store_l(u_nlimbs - 1, !0);
                        }
                    }

                    let mut v = tst_mk_mp_backing_vec!(VT, v_len);
                    let mut v = VT::from_slice(&mut v).unwrap();
                    v.store_l((j1 / LIMB_BITS) as usize, 1 << (j1 % LIMB_BITS));
                    v.store_l(
                        (j2 / LIMB_BITS) as usize,
                        v.load_l((j2 / LIMB_BITS) as usize) | 1 << (j2 % LIMB_BITS),
                    );
                    div_and_check::<UT, VT, QT>(&u, u_len, u_lshift_len, &v, None);
                    if j1 != 0 {
                        div_and_check::<UT, VT, QT>(&u, u_len, u_lshift_len, &v, Some(1));
                    }
                }
            }
        }
    }
}

#[test]
fn test_ct_div_lshifted_be_be_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_div_lshifted_mp_mp::<
        MpMutBigEndianUIntByteSlice,
        MpMutBigEndianUIntByteSlice,
        MpMutBigEndianUIntByteSlice,
    >()
}

#[test]
fn test_ct_div_lshifted_le_le_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_div_lshifted_mp_mp::<
        MpMutLittleEndianUIntByteSlice,
        MpMutLittleEndianUIntByteSlice,
        MpMutLittleEndianUIntByteSlice,
    >()
}

#[test]
fn test_ct_div_lshifted_ne_ne_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_div_lshifted_mp_mp::<
        MpMutNativeEndianUIntLimbsSlice,
        MpMutNativeEndianUIntLimbsSlice,
        MpMutNativeEndianUIntLimbsSlice,
    >()
}

pub type CtModLshiftedMpMpError = CtModPow2MpError;

pub fn ct_mod_lshifted_mp_mp<UT: MpMutUInt, VT: MpUIntCommon>(
    u: &mut UT,
    u_in_len: usize,
    u_lshift_len: usize,
    v: &CtMpDivisor<VT>,
) -> Result<(), CtModLshiftedMpMpError> {
    // Specify an arbitrary MPIntMutByteSlice type for the non-existant q-argument.
    ct_div_lshifted_mp_mp::<_, _, MpMutNativeEndianUIntLimbsSlice>(
        u,
        u_in_len,
        u_lshift_len,
        v,
        None,
    )
    .map_err(|e| match e {
        CtDivLshiftedMpMpError::InsufficientRemainderSpace => {
            CtModLshiftedMpMpError::InsufficientRemainderSpace
        }
        CtDivLshiftedMpMpError::InsufficientQuotientSpace => {
            // No quotient, no insufficient quotient space.
            unreachable!()
        }
    })
}

pub type CtDivMpLError = CtDivMpMpError;

// Compute the modulo of a multiprecision integer modulo a [`LimbType`]
// divisisor.
pub fn ct_div_mp_l<UT: MpUIntCommon, QT: MpMutUInt>(
    u: &UT,
    v: &CtLDivisor,
    mut q_out: Option<&mut QT>,
) -> Result<LimbType, CtDivMpLError> {
    let u_nlimbs = ct_mp_nlimbs(u.len());
    if u_nlimbs == 0 {
        return Ok(0);
    }

    if let Some(q_out) = &mut q_out {
        let v_len = ct_find_last_set_byte_l(v.get_v());
        if q_out.len() + v_len < u.len() + 1 {
            return Err(CtDivMpLError::InsufficientQuotientSpace);
        }
        if u.len() < v_len {
            q_out.clear_bytes_above(0);
            return Ok(u.load_l(0));
        }
        let q_out_len = u.len() - v_len + 1;
        q_out.clear_bytes_above(q_out_len);
    }

    let mut u_h = 0;
    let mut j = u_nlimbs;
    while j > 0 {
        j -= 1;
        let u_l = u.load_l(j);
        let (q_val, r) = v.do_div(&DoubleLimb::new(u_h, u_l));

        if let Some(q_out) = &mut q_out {
            q_out.store_l(j, q_val)
        }
        u_h = r;
    }

    Ok(u_h)
}

pub fn ct_mod_mp_l<UT: MpUIntCommon>(u: &UT, v: &CtLDivisor) -> LimbType {
    // Specify an arbitrary MPIntMutByteSlice type for the non-existant q-argument.
    ct_div_mp_l::<_, MpMutNativeEndianUIntLimbsSlice>(u, v, None)
        .map_err(|e| match e {
            CtDivMpLError::InsufficientQuotientSpace => {
                // No quotient, no insufficient quotient space.
                unreachable!()
            }
        })
        .unwrap()
}
