use super::cmp_impl::ct_is_zero_mp;
use super::div_impl::{ct_div_mp_mp, CtMpDivisor};
use super::euclid_impl::ct_gcd_mp_mp;
use super::limb::{ct_lsb_mask_l, LIMB_BITS, LIMB_BYTES};
use super::limbs_buffer::{
    ct_find_last_set_bit_mp, ct_mp_nlimbs, MpMutUInt, MpMutUIntSlice, MpUIntCommon as _,
    MpUIntSliceCommonPriv as _,
};
use super::mul_impl::ct_mul_trunc_mp_mp;
use super::shift_impl::{ct_lshift_mp, ct_rshift_mp};

#[derive(Debug)]
pub enum CtLcmMpMpError {
    InsufficientResultSpace,
    InsufficientScratchSpace,
    InconsistentInputOperandLengths,
}

pub fn ct_lcm_mp_mp<RT: MpMutUInt, T0: MpMutUInt, T1: MpMutUIntSlice>(
    result: &mut RT,
    op0: &mut T0,
    op0_len: usize,
    op1: &mut T1,
    op1_len: usize,
    scratch: &mut [T1::BackingSliceElementType],
) -> Result<(), CtLcmMpMpError> {
    // The Least Common Multiple (LCM) is the product divided by the GCD of the
    // operands. Be careful to maintain constant-time: the division's runtime
    // depends on the divisor's length. Always left shift the the dividend and
    // the divisor so that the latter attains its maximum possible width before
    // doing the division.

    let op0_len = op0_len.min(op0.len());
    let op1_len = op1_len.min(op1.len());
    if op0_len == 0 || op1_len == 0 {
        result.clear_bytes_above(0);
        return Ok(());
    }

    // op0 and op1 must be equal in length (within alignment excess), i.e. callers
    // are required to allocate the larger of the two operand's length for both.
    // This way, they can be reused as scratch buffers later on.
    if !op0.len_is_compatible_with(op1.len()) || !op1.len_is_compatible_with(op0.len()) {
        return Err(CtLcmMpMpError::InconsistentInputOperandLengths);
    }

    debug_assert_eq!(op0.len(), op1.len());

    if result.len() < op0_len + op1_len {
        return Err(CtLcmMpMpError::InsufficientResultSpace);
    }

    let prod_len = op0_len + op1_len;
    if scratch.len() < T1::n_backing_elements_for_len(prod_len) {
        return Err(CtLcmMpMpError::InsufficientScratchSpace);
    }
    let prod_scratch = T1::from_slice(scratch).unwrap();
    let mut prod_scratch = prod_scratch.take(prod_len).1;

    // Compute the product before messing around with op0's and op1's values below.
    prod_scratch.copy_from(op0);
    ct_mul_trunc_mp_mp(&mut prod_scratch, op0_len, &op1.shrink_to(op1_len));

    // Compute the GCD, result will be in op0.
    ct_gcd_mp_mp(op0, op1).unwrap();
    let gcd: &mut T0 = op0;
    debug_assert_eq!(ct_is_zero_mp(gcd).unwrap(), 0);

    // And divide the product by the GCD to arrive at the LCM. As initially said, be
    // careful to scale the GCD to the maximum possible value so that the
    // division's runtime is independent of its actual value's width.
    let (_, gcd_width) = ct_find_last_set_bit_mp(gcd);
    // Maximum GCD length is the larger of the two operands' lengths: in the most
    // common case of non-zero operands, it's <= the smaller one actually, but
    // if either of the operands' values equals zero, then it would come out as
    // the other one.
    let gcd_max_len = op0_len.max(op1_len);
    let gcd_max_nlimbs = ct_mp_nlimbs(gcd_max_len);
    let scaling_shift = 8 * gcd_max_len - gcd_width;
    ct_lshift_mp(gcd, scaling_shift);
    // Scale, the dividend, i.e. the product of op0 and op1 as well. This scaling
    // will overflow the prod_scratch[], so reuse the currently unused op1[]
    // buffer to receive the high parts shifted out on the left. For
    // constant-time, copy the maximum possible high part over to op1[] and
    // shift that to the right as appropriate.
    let scaled_prod_high: &mut T1 = op1;
    debug_assert!(scaled_prod_high.nlimbs() >= gcd_max_nlimbs);
    let src_begin = prod_scratch.len() - gcd_max_len;
    let src_low_rshift = 8 * (src_begin % LIMB_BYTES) as u32;
    let dst_high_lshift = (LIMB_BITS - src_low_rshift) % LIMB_BITS;
    let src_high_mask = ct_lsb_mask_l(src_low_rshift);
    let src_begin_limb_index = src_begin / LIMB_BYTES; // Rounding down is on purpose.
    debug_assert!(src_begin_limb_index + gcd_max_nlimbs <= prod_scratch.nlimbs());
    let mut last_src_val = prod_scratch.load_l(src_begin_limb_index);
    let mut i = 0;
    while i < gcd_max_nlimbs {
        let src_limb_index = src_begin_limb_index + i + 1;
        let next_src_val = if src_limb_index < prod_scratch.nlimbs() {
            prod_scratch.load_l(src_limb_index)
        } else {
            0
        };
        let dst_low = last_src_val >> src_low_rshift;
        let dst_high = (next_src_val & src_high_mask) << dst_high_lshift;
        scaled_prod_high.store_l(i, dst_high | dst_low);
        last_src_val = next_src_val;
        i += 1;
    }
    debug_assert_eq!(last_src_val >> src_low_rshift, 0);
    scaled_prod_high.clear_bytes_above(gcd_max_len);
    ct_rshift_mp(scaled_prod_high, 8 * gcd_max_len - scaling_shift);
    let mut scaled_prod_high = scaled_prod_high.shrink_to(gcd_max_len);
    let mut scaled_prod_low = prod_scratch;
    ct_lshift_mp(&mut scaled_prod_low, scaling_shift);

    // Finally, do the division and be done.
    let gcd = CtMpDivisor::new(gcd).unwrap();
    ct_div_mp_mp(
        Some(&mut scaled_prod_high),
        &mut scaled_prod_low,
        &gcd,
        Some(result),
    )
    .unwrap();
    // The remainder should be zero.
    debug_assert_ne!(ct_is_zero_mp(&scaled_prod_low).unwrap(), 0);
    debug_assert_ne!(ct_is_zero_mp(&scaled_prod_high).unwrap(), 0);
    Ok(())
}

#[cfg(test)]
fn test_ct_lcm_mp_mp<RT: MpMutUIntSlice, OT: MpMutUIntSlice>() {
    use super::limbs_buffer::MpUIntCommon;
    use super::mul_impl::ct_mul_trunc_mp_l;

    fn test_one<
        RT: MpMutUIntSlice,
        OT: MpMutUIntSlice,
        T0: MpUIntCommon,
        T1: MpUIntCommon,
        GT: MpUIntCommon,
    >(
        op0: &T0,
        op1: &T1,
        gcd: &GT,
    ) {
        use super::cmp_impl::ct_eq_mp_mp;

        let gcd_len = gcd.len();
        let op0_len = op0.len() + gcd_len;
        let op1_len = op1.len() + gcd_len;

        let op_max_len = op0_len.max(op1_len);
        let mut op0_lcm_work = tst_mk_mp_backing_vec!(OT, op_max_len);
        let mut op0_lcm_work = OT::from_slice(&mut op0_lcm_work).unwrap();
        op0_lcm_work.copy_from(op0);
        ct_mul_trunc_mp_mp(&mut op0_lcm_work, op0.len(), gcd);
        let mut op1_lcm_work = tst_mk_mp_backing_vec!(OT, op_max_len);
        let mut op1_lcm_work = OT::from_slice(&mut op1_lcm_work).unwrap();
        op1_lcm_work.copy_from(op1);
        ct_mul_trunc_mp_mp(&mut op1_lcm_work, op1.len(), gcd);

        let lcm_len = op0_len + op1_len;
        let mut lcm = tst_mk_mp_backing_vec!(RT, lcm_len);
        let mut lcm = RT::from_slice(&mut lcm).unwrap();
        let mut scratch = tst_mk_mp_backing_vec!(OT, lcm_len);
        ct_lcm_mp_mp(
            &mut lcm,
            &mut op0_lcm_work,
            op0_len,
            &mut op1_lcm_work,
            op1_len,
            &mut scratch,
        )
        .unwrap();

        let expected_len = op0.len() + op1.len() + gcd_len;
        let mut expected = tst_mk_mp_backing_vec!(RT, expected_len);
        let mut expected = RT::from_slice(&mut expected).unwrap();
        expected.copy_from(op0);
        ct_mul_trunc_mp_mp(&mut expected, op0.len(), op1);
        ct_mul_trunc_mp_mp(&mut expected, op0.len() + op1.len(), gcd);
        assert_ne!(ct_eq_mp_mp(&lcm, &expected).unwrap(), 0);
    }

    for i in [
        0,
        LIMB_BYTES - 1,
        LIMB_BYTES,
        2 * LIMB_BYTES,
        2 * LIMB_BYTES + 1,
    ] {
        for j in [
            0,
            LIMB_BYTES - 1,
            LIMB_BYTES,
            2 * LIMB_BYTES,
            2 * LIMB_BYTES + 1,
        ] {
            for k in [
                0,
                LIMB_BYTES - 1,
                LIMB_BYTES,
                2 * LIMB_BYTES,
                2 * LIMB_BYTES + 1,
            ] {
                for total_shift in [
                    0,
                    LIMB_BITS - 1,
                    LIMB_BITS,
                    2 * LIMB_BITS,
                    2 * LIMB_BITS + 1,
                ] {
                    for l in 0..total_shift as usize {
                        let gcd_shift = l.min(total_shift as usize - l);
                        let op0_shift = l - gcd_shift;
                        let op1_shift = total_shift as usize - l - gcd_shift;

                        let op0_len = i + (op0_shift + 7) / 8;
                        let mut op0 = tst_mk_mp_backing_vec!(OT, op0_len);
                        let mut op0 = OT::from_slice(&mut op0).unwrap();
                        if !op0.is_empty() {
                            op0.store_l(0, 1);
                        }
                        for _ in 0..i {
                            ct_mul_trunc_mp_l(&mut op0, op0_len, 251);
                        }
                        ct_lshift_mp(&mut op0, op0_shift);

                        let op1_len = j + (op1_shift + 7) / 8;
                        let mut op1 = tst_mk_mp_backing_vec!(OT, op1_len);
                        let mut op1 = OT::from_slice(&mut op1).unwrap();
                        if !op1.is_empty() {
                            op1.store_l(0, 1);
                        }
                        for _ in 0..j {
                            ct_mul_trunc_mp_l(&mut op1, op1_len, 241);
                        }
                        ct_lshift_mp(&mut op1, op1_shift);

                        let gcd_len = k + (gcd_shift + 7) / 8;
                        let mut gcd = tst_mk_mp_backing_vec!(OT, gcd_len);
                        let mut gcd = OT::from_slice(&mut gcd).unwrap();
                        if !gcd.is_empty() {
                            gcd.store_l(0, 1);
                        }
                        for _ in 0..k {
                            ct_mul_trunc_mp_l(&mut gcd, gcd_len, 239);
                        }
                        ct_lshift_mp(&mut gcd, gcd_shift);

                        test_one::<RT, OT, _, _, _>(&op0, &op1, &gcd);
                    }
                }
            }
        }
    }
}

#[test]
fn test_ct_lcm_be_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_lcm_mp_mp::<MpMutBigEndianUIntByteSlice, MpMutBigEndianUIntByteSlice>();
}

#[test]
fn test_ct_lcm_le_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_lcm_mp_mp::<MpMutLittleEndianUIntByteSlice, MpMutLittleEndianUIntByteSlice>();
}

#[test]
fn test_ct_lcm_ne_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_lcm_mp_mp::<MpMutNativeEndianUIntLimbsSlice, MpMutNativeEndianUIntLimbsSlice>();
}
