use super::cmp_impl::mp_ct_is_zero_mp;
use super::div_impl::mp_ct_div_mp_mp;
use super::euclid::mp_ct_gcd;
use super::limb::{LIMB_BITS, LIMB_BYTES, ct_lsb_mask_l};
use super::limbs_buffer::{MPIntMutByteSlice, MPIntByteSliceCommon, MPIntByteSliceCommonPriv as _, mp_ct_find_first_set_bit_mp, mp_ct_find_last_set_bit_mp, mp_ct_nlimbs,  mp_ct_limbs_align_len, mp_ct_swap_cond};
use super::mul_impl::mp_ct_mul_trunc_mp_mp;
use super::shift_impl::{mp_ct_lshift_mp, mp_ct_rshift_mp};
use super::usize_ct_cmp::ct_lt_usize_usize;

pub fn mp_ct_lcm<RT: MPIntMutByteSlice, T0: MPIntMutByteSlice, T1: MPIntMutByteSlice>(
    result: &mut RT,
    op0: &mut T0, op0_len: usize,
    op1: &mut T1, op1_len: usize,
    scratch: &mut [u8]
) {
    // The Least Common Multiple (LCM) is the product divided by the GCD of the operands. Be careful
    // to maintain constant-time: the division's runtime depends on the divisor's length. Always left shift
    // the the dividend and the divisor so that the latter attains its maximum possible width before
    // doing the division.

    // op0 and op1 must be equal in length, i.e. callers are required to allocate the larger of the
    // two operand's length for both. This way, they can be reused as scratch buffers later on.
    debug_assert!(op0_len <= op0.len());
    debug_assert!(op1_len <= op1.len());
    debug_assert_eq!(op0.len(), op1.len());
    debug_assert!(result.len() >= op0_len + op1_len);
    if op0_len == 0 || op1_len == 0 {
        result.zeroize_bytes_above(0);
        return;
    }

    let prod_len = op0_len + op1_len;
    let prod_aligned_len = T1::limbs_align_len(prod_len);
    debug_assert!(scratch.len() >= prod_aligned_len);
    let (scratch, _) = scratch.split_at_mut(prod_aligned_len);
    let mut prod_scratch = T1::from_bytes(scratch).unwrap();

    // Compute the product before messing around with op0's and op1's values below.
    prod_scratch.copy_from(op0);
    let op1_aligned_len = mp_ct_limbs_align_len(op1_len);
    if op1_aligned_len < op1.len() {
        let (_, trimmed_op1) = op1.split_at(op1_aligned_len);
        mp_ct_mul_trunc_mp_mp(&mut prod_scratch, op0_len, &trimmed_op1);
    } else {
        mp_ct_mul_trunc_mp_mp(&mut prod_scratch, op0_len, op1);
    };

    // The GCD implementation requires the first operand to be odd. Factor out
    // common powers of two.
    let (op0_is_nonzero, op0_powers_of_two) = mp_ct_find_first_set_bit_mp(op0);
    let (op1_is_nonzero, op1_powers_of_two) = mp_ct_find_first_set_bit_mp(op1);
    debug_assert!(op0_is_nonzero.unwrap() != 0 || op0_powers_of_two == 0);
    let op1_has_fewer_powers_of_two =
        !op0_is_nonzero // If op0 == 0, consider op1 only.
        | op1_is_nonzero & ct_lt_usize_usize(op1_powers_of_two, op0_powers_of_two);
    let min_powers_of_two = op1_has_fewer_powers_of_two.select_usize(op0_powers_of_two, op1_powers_of_two);
    mp_ct_rshift_mp(op0, min_powers_of_two);
    mp_ct_rshift_mp(op1, min_powers_of_two);
    mp_ct_swap_cond(op0, op1, op1_has_fewer_powers_of_two);
    let op0_and_op1_zero = !op0_is_nonzero & !op1_is_nonzero;
    debug_assert!(op0_and_op1_zero.unwrap() != 0 ||
                  op0.load_l(0) & 1 == 1);
    // If both inputs are zero, force op0 to 1, the GCD needs that.
    op0.store_l(0, op0_and_op1_zero.select(op0.load_l(0), 1));
    mp_ct_gcd(op0, op1);
    // Now the GCD (odd factors only) is in op0.
    let gcd_odd: &mut T0 = op0;
    debug_assert_eq!(mp_ct_is_zero_mp(gcd_odd).unwrap(), 0);

    // Reduce the product by the actual GCD's powers of two.
    mp_ct_rshift_mp(&mut prod_scratch, min_powers_of_two);

    // And divide the product by the remaining, odd factors of the GCD to
    // arrive at the LCM. As initially said, be careful to scale the GCD
    // to the maximum possible value so that the division's runtime is
    // independent of its actual value's width.
    let (_, gcd_odd_width) = mp_ct_find_last_set_bit_mp(gcd_odd);
    // Maximum GCD length is the larger of the two operands' lengths: in the most common case of
    // non-zero operands, it's <= the smaller one actually, but if either of the operands' values
    // equals zero, then it would come out as the other one.
    let gcd_max_len = op0_len.max(op1_len);
    let gcd_max_nlimbs = mp_ct_nlimbs(gcd_max_len);
    let scaling_shift = 8 * gcd_max_len - gcd_odd_width;
    mp_ct_lshift_mp(gcd_odd, scaling_shift);
    // Scale, the dividend, i.e. the product of op0 and op1 as well. This scaling will overflow the
    // prod_scratch[], so reuse the currently unused op1[] buffer to receive the high parts shifted
    // out on the left. For constant-time, copy the maximum possible high part over to op1[] and
    // shift that to the right as appropriate.
    let scaled_prod_high: &mut T1 = op1;
    debug_assert!(scaled_prod_high.nlimbs() >= gcd_max_nlimbs);
    let src_begin = prod_scratch.len() - gcd_max_len;
    let src_low_rshift = 8 * (src_begin % LIMB_BYTES as usize) as u32;
    let dst_high_lshift = (LIMB_BITS - src_low_rshift) % LIMB_BITS;
    let src_high_mask = ct_lsb_mask_l(src_low_rshift);
    let src_begin_limb_index = src_begin / LIMB_BYTES as usize; // Rounding down is on purpose.
    debug_assert!(src_begin_limb_index + gcd_max_nlimbs <= prod_scratch.nlimbs());
    let mut last_src_val = prod_scratch.load_l(src_begin_limb_index);
    let mut i = 0;
    while i < gcd_max_nlimbs {
        let src_limb_index = src_begin_limb_index + i + 1;
        let next_src_val  = if src_limb_index < prod_scratch.nlimbs() {
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
    scaled_prod_high.zeroize_bytes_above(gcd_max_len);
    mp_ct_rshift_mp(scaled_prod_high, 8 * gcd_max_len - scaling_shift);
    let (_, mut scaled_prod_high) = scaled_prod_high.split_at(T1::limbs_align_len(gcd_max_len));
    let mut scaled_prod_low = prod_scratch;
    mp_ct_lshift_mp(&mut scaled_prod_low, scaling_shift);

    // Finally, do the division and be done.
    mp_ct_div_mp_mp(Some(&mut scaled_prod_high), &mut scaled_prod_low, gcd_odd, Some(result)).unwrap();
    // The remainder should be zero.
    debug_assert_ne!(mp_ct_is_zero_mp(&scaled_prod_low).unwrap(), 0);
    debug_assert_ne!(mp_ct_is_zero_mp(&scaled_prod_high).unwrap(), 0);
}

#[cfg(test)]
fn test_mp_ct_lcm_common<RT: MPIntMutByteSlice, OT: MPIntMutByteSlice>() {
    use super::mul_impl::mp_ct_mul_trunc_mp_l;

    fn test_one<RT: MPIntMutByteSlice, OT: MPIntMutByteSlice,
                T0: MPIntByteSliceCommon, T1: MPIntByteSliceCommon,
                GT: MPIntByteSliceCommon>(
        op0: &T0, op1: &T1, gcd: &GT
    ) {
        use super::cmp_impl::mp_ct_eq_mp_mp;

        let gcd_len = gcd.len();
        let op0_len = op0.len() + gcd_len;
        let op1_len = op1.len() + gcd_len;

        let op_max_len = op0_len.max(op1_len);
        let op_max_aligned_len = OT::limbs_align_len(op_max_len);
        let mut op0_lcm_work = vec![0u8; op_max_aligned_len];
        let mut op0_lcm_work = OT::from_bytes(&mut op0_lcm_work).unwrap();
        op0_lcm_work.copy_from(op0);
        mp_ct_mul_trunc_mp_mp(&mut op0_lcm_work, op0.len(), gcd);
        let mut op1_lcm_work = vec![0u8; op_max_aligned_len];
        let mut op1_lcm_work = OT::from_bytes(&mut op1_lcm_work).unwrap();
        op1_lcm_work.copy_from(op1);
        mp_ct_mul_trunc_mp_mp(&mut op1_lcm_work, op1.len(), gcd);

        let lcm_len = op0_len + op1_len;
        let mut lcm = vec![0u8; RT::limbs_align_len(lcm_len)];
        let mut lcm = RT::from_bytes(&mut lcm).unwrap();
        let mut scratch = vec![0u8; OT::limbs_align_len(lcm_len)];
        mp_ct_lcm(&mut lcm, &mut op0_lcm_work, op0_len, &mut op1_lcm_work, op1_len, &mut scratch);

        let expected_len = op0.len() + op1.len() + gcd_len;
        let mut expected = vec![0u8; RT::limbs_align_len(expected_len)];
        let mut expected = RT::from_bytes(&mut expected).unwrap();
        expected.copy_from(op0);
        mp_ct_mul_trunc_mp_mp(&mut expected, op0.len(), op1);
        mp_ct_mul_trunc_mp_mp(&mut expected, op0.len() + op1.len(), gcd);
        assert_ne!(mp_ct_eq_mp_mp(&lcm, &expected).unwrap(), 0);
    }

    for i in [0, LIMB_BYTES - 1, LIMB_BYTES, 2 * LIMB_BYTES, 2 * LIMB_BYTES + 1] {
        for j in [0, LIMB_BYTES - 1, LIMB_BYTES, 2 * LIMB_BYTES, 2 * LIMB_BYTES + 1] {
            for k in [0, LIMB_BYTES - 1, LIMB_BYTES, 2 * LIMB_BYTES, 2 * LIMB_BYTES + 1] {
                for total_shift in [0, LIMB_BITS - 1, LIMB_BITS, 2 * LIMB_BITS, 2 * LIMB_BITS + 1] {
                    for l in 0..total_shift as usize {
                        let gcd_shift = l.min(total_shift as usize - l);
                        let op0_shift = l - gcd_shift;
                        let op1_shift = total_shift as usize- l - gcd_shift;

                        let op0_len = i + (op0_shift + 7) / 8;
                        let mut op0 = vec![0u8; OT::limbs_align_len(op0_len)];
                        let mut op0 = OT::from_bytes(&mut op0).unwrap();
                        if !op0.is_empty() {
                            op0.store_l(0, 1);
                        }
                        for _ in 0..i {
                            mp_ct_mul_trunc_mp_l(&mut op0, op0_len, 251);
                        }
                        mp_ct_lshift_mp(&mut op0, op0_shift);

                        let op1_len = j + (op1_shift + 7) / 8;
                        let mut op1 = vec![0u8; OT::limbs_align_len(op1_len)];
                        let mut op1 = OT::from_bytes(&mut op1).unwrap();
                        if !op1.is_empty() {
                            op1.store_l(0, 1);
                        }
                        for _ in 0..j {
                            mp_ct_mul_trunc_mp_l(&mut op1, op1_len, 241);
                        }
                        mp_ct_lshift_mp(&mut op1, op1_shift);

                        let gcd_len = k + (gcd_shift + 7) / 8;
                        let mut gcd = vec![0u8; OT::limbs_align_len(gcd_len)];
                        let mut gcd = OT::from_bytes(&mut gcd).unwrap();
                        if !gcd.is_empty() {
                            gcd.store_l(0, 1);
                        }
                        for _ in 0..k {
                            mp_ct_mul_trunc_mp_l(&mut gcd, gcd_len, 239);
                        }
                        mp_ct_lshift_mp(&mut gcd, gcd_shift);

                        test_one::<RT, OT, _, _, _>(&op0, &op1, &gcd);
                    }
                }
            }
        }
    }
}

#[test]
fn test_mp_ct_lcm_be_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_lcm_common::<MPBigEndianMutByteSlice, MPBigEndianMutByteSlice>();
}

#[test]
fn test_mp_ct_lcm_le_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_lcm_common::<MPLittleEndianMutByteSlice, MPLittleEndianMutByteSlice>();
}

#[test]
fn test_mp_ct_lcm_ne_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_ct_lcm_common::<MPNativeEndianMutByteSlice, MPNativeEndianMutByteSlice>();
}
