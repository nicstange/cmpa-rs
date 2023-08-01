use super::cmp_impl::{ct_is_zero_mp, ct_lt_mp_mp};
use super::limbs_buffer::{MpIntMutByteSlice, MpIntByteSliceCommon, MpNativeEndianMutByteSlice, MpIntByteSliceCommonPriv as _, zeroize_bits_above_mp, ct_zeroize_bits_above_mp, ct_find_first_set_bit_mp, ct_mp_nlimbs};
use super::add_impl::{ct_add_mp_mp,ct_sub_mp_mp};
use super::shift_impl::{ct_rshift_mp, ct_lshift_mp};
use super::limb::{ct_sub_l_l_b, ct_is_nonzero_l, LIMB_BITS};
use super::mul_impl::{ct_mul_trunc_mp_mp, ct_square_trunc_mp};
use super::euclid::{ct_inv_mod_odd_mp_mp, CtInvModOddMpMpError};

// max_pow2_exp: non-sensitive upper bound (inclusive) on pow2_exp, for CT.
// If pow2_exp == 0, the result is all zero.
fn ct_inv_mod_pow2_mp<RT: MpIntMutByteSlice, T0: MpIntByteSliceCommon>(
    result: &mut RT, op0: &T0, pow2_exp: usize, max_pow2_exp: usize, scratch: &mut [u8]
) {
    debug_assert!(!op0.is_empty());
    debug_assert!(pow2_exp == 0 || op0.load_l(0) & 1 == 1);

    debug_assert!(max_pow2_exp != 0);
    debug_assert!(pow2_exp <= max_pow2_exp);

    let max_pow2_exp_len = (max_pow2_exp + 7) / 8;
    debug_assert!(result.len() >= RT::limbs_align_len(max_pow2_exp_len));
    debug_assert!(scratch.len() >= MpNativeEndianMutByteSlice::limbs_align_len(max_pow2_exp_len));

    result.zeroize_bytes_above(max_pow2_exp_len);
    let (_, mut result) = result.split_at(RT::limbs_align_len(max_pow2_exp_len));
    let mut scratch = &mut scratch[..MpNativeEndianMutByteSlice::limbs_align_len(max_pow2_exp_len)];
    let mut tmp = MpNativeEndianMutByteSlice::from_bytes(&mut scratch).unwrap();

    // Use Hensel's lemma to lift the root x == 1 of op0 * x - 1 mod 2 to a root mod
    // 2^{2^pow2_exp_bits}.  Note that an inverse modulo 2^pow2_exp_bits is also an inverse for any
    // smaller power of two (more formally, a representative of the inverse's residue class modulo
    // that smaller power of two), including pow2_exp. Since only the latter is of interest, all
    // calculations can be done in precision of pow2_exp bits.
    // Quadratic lifting via op0'_{i + 1} = 2 * op0'_i - op0 * op0'_i^2 allows for a runtime of
    // log_2(pow2_exp), i.e. pow2_exp_bits (of the loop).
    result.zeroize_bytes_below(max_pow2_exp_len);
    result.store_l(0, 1);
    let mut cur_pow2_exp = 1;
    while cur_pow2_exp < max_pow2_exp {
        // Lift the inverse from mod 2^2^i to an inverse mod 2^2^{i + 1}
        tmp.copy_from(&result);
        // cur_result_len is an optimization to limit the multiplications to the significant
        // parts. It could just as well be set to the upper bound of max_pow2_exp_len, but this way
        // a total of ~1/2 of multiplications is avoided.
        let cur_result_len = (cur_pow2_exp + 7) / 8;
        // tmp = n'_i^2.
        ct_square_trunc_mp(&mut tmp, cur_result_len);
        let cur_result_len = (2 * cur_result_len).min(max_pow2_exp_len);
        // tmp *= n.
        ct_mul_trunc_mp_mp(&mut tmp, cur_result_len, op0);

        // result = 2 * result - tmp
        let mut carry = 0;
        let mut borrow = 0;
        for j in 0..ct_mp_nlimbs(cur_result_len) {
            let result_val = result.load_l(j);
            let tmp_val = tmp.load_l(j);
            let result_msb = result_val >> (LIMB_BITS - 1);
            let mut result_val = (result_val << 1) + carry;
            carry = result_msb;
            (borrow, result_val) = ct_sub_l_l_b(result_val, tmp_val, borrow);
            if j + 1 != result.nlimbs() {
                result.store_l_full(j, result_val);
            } else {
                result.store_l(j, result_val & result.partial_high_mask());
            }
        }
        // The result is valid only within the truncated precision used for the multiplications
        // above. Truncate the upper bits again so that they won't be considered in the next
        // iteration.
        cur_pow2_exp *= 2;
        zeroize_bits_above_mp(&mut result, cur_pow2_exp);
    }

    // Finally, wipe anything beyond the real pow2_exp, that is, take the result mod 2^{pow2_exp}.
    ct_zeroize_bits_above_mp(&mut result, pow2_exp);
    debug_assert!(pow2_exp != 0 || ct_is_zero_mp(&result).unwrap() == 1);
}

#[cfg(test)]
fn test_ct_inv_mod_pow2_mp_common<RT: MpIntMutByteSlice, T0: MpIntMutByteSlice>(op0_len: usize) {
    use super::cmp_impl::ct_is_one_mp;
    use super::limb::LimbType;

    for i in 0..8 {
        const MERSENNE_PRIME_17: LimbType = 131071 as LimbType;
        let op0_high = MERSENNE_PRIME_17.wrapping_mul((8191 as LimbType).wrapping_mul(i));
        for j in 0..8 {
            const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
            let op0_low = MERSENNE_PRIME_13.wrapping_mul((131087 as LimbType).wrapping_mul(j)) | 1;
            let mut op0 = vec![0u8; T0::limbs_align_len(op0_len)];
            let mut op0 = T0::from_bytes(&mut op0).unwrap();
            let op0_vals = [op0_low, op0_high];
            for k in 0..op0.nlimbs() {
                let op0_val = op0_vals[k.min(1)];
                if k + 1 != op0.nlimbs() {
                    op0.store_l_full(k, op0_val);
                } else {
                    op0.store_l(k, op0_val & op0.partial_high_mask());
                }
            }
            op0.zeroize_bytes_above(op0_len);

            for pow2_exp in [0, 7, 8 * op0_len - 1, 8 * op0_len, 2 * 8 * op0_len] {
                for max_pow2_exp in [pow2_exp, pow2_exp + 1, 2 * pow2_exp] {
                    let max_pow2_exp = max_pow2_exp.max(1);
                    let max_pow2_exp_len = max_pow2_exp + 7 / 8;
                    let mut inv_mod_pow2 = vec![0u8; RT::limbs_align_len(max_pow2_exp_len)];
                    let mut inv_mod_pow2 = RT::from_bytes(&mut inv_mod_pow2).unwrap();
                    let mut scratch = vec![0u8; MpNativeEndianMutByteSlice::limbs_align_len(max_pow2_exp_len)];
                    ct_inv_mod_pow2_mp(&mut inv_mod_pow2, &op0, pow2_exp, max_pow2_exp, &mut scratch);
                    if pow2_exp != 0 {
                        // Multiply the inverse by n and verify the result comes out as 1 mod 2^pow2_exp.
                        let inv_mod_pow2_len = inv_mod_pow2.len();
                        ct_mul_trunc_mp_mp(&mut inv_mod_pow2, inv_mod_pow2_len, &op0);
                        zeroize_bits_above_mp(&mut inv_mod_pow2, pow2_exp);
                        assert_eq!(ct_is_one_mp(&inv_mod_pow2).unwrap(), 1);
                    } else {
                        // For pow2_exp == 0, the inverse is undefined, the result
                        // shall be set to zero.
                        assert_eq!(ct_is_zero_mp(&inv_mod_pow2).unwrap(), 1);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
fn test_ct_inv_mod_pow2_mp_with_aligned_lengths<RT: MpIntMutByteSlice, NT: MpIntMutByteSlice>() {
    use super::limb::LIMB_BYTES;

    for n_len in [LIMB_BYTES, 2 * LIMB_BYTES] {
        test_ct_inv_mod_pow2_mp_common::<RT, NT>(n_len);
    }
}

#[cfg(test)]
fn test_ct_inv_mod_pow2_mp_with_unaligned_lengths<RT: MpIntMutByteSlice, NT: MpIntMutByteSlice>() {
    use super::limb::LIMB_BYTES;

    for n_len in [LIMB_BYTES - 1, LIMB_BYTES + 1, 2 * LIMB_BYTES - 1] {
        test_ct_inv_mod_pow2_mp_common::<RT, NT>(n_len);
    }
}

#[test]
fn test_ct_inv_mod_pow2_be_be() {
    use super::limbs_buffer::MpBigEndianMutByteSlice;

    test_ct_inv_mod_pow2_mp_with_aligned_lengths::<MpBigEndianMutByteSlice,
                                                     MpBigEndianMutByteSlice>();
    test_ct_inv_mod_pow2_mp_with_unaligned_lengths::<MpBigEndianMutByteSlice,
                                                       MpBigEndianMutByteSlice>();
}

#[test]
fn test_ct_inv_mod_pow2_le_le() {
    use super::limbs_buffer::MpLittleEndianMutByteSlice;

    test_ct_inv_mod_pow2_mp_with_aligned_lengths::<MpLittleEndianMutByteSlice,
                                                     MpLittleEndianMutByteSlice>();
    test_ct_inv_mod_pow2_mp_with_unaligned_lengths::<MpLittleEndianMutByteSlice,
                                                       MpLittleEndianMutByteSlice>();
}

#[test]
fn test_ct_inv_mod_pow2_ne_ne() {
    test_ct_inv_mod_pow2_mp_with_aligned_lengths::<MpNativeEndianMutByteSlice,
                                                     MpNativeEndianMutByteSlice>();
}

#[derive(Debug)]
pub enum CtInvModMpMpError {
    OperandsNotCoprime,
}

// If n == 1, the multiplicative group does not exist and the
// result is set to zero.
pub fn ct_inv_mod_mp_mp<T0: MpIntMutByteSlice, NT: MpIntMutByteSlice>(
    op0: &mut T0, n: &mut NT, scratch: [&mut [u8]; 4]
) -> Result<(), CtInvModMpMpError> {
    assert!(!n.is_empty());
    assert_eq!(op0.nlimbs(), n.nlimbs());

    // Verify if both, op0 and n, have factors of two in common and
    // return an error if so.
    if ct_is_nonzero_l(!op0.load_l(0) & !n.load_l(0) & 1) != 0 {
        return Err(CtInvModMpMpError::OperandsNotCoprime);
    }

    let [scratch0, scratch1, scratch2, scratch3] = scratch;
    let scratch_len = MpNativeEndianMutByteSlice::limbs_align_len(n.len());
    assert!(scratch0.len() >= scratch_len);
    let (scratch0, _) = &mut scratch0.split_at_mut(scratch_len);
    let mut scratch0 = MpNativeEndianMutByteSlice::from_bytes(scratch0).unwrap();
    assert!(scratch1.len() >= scratch_len);
    let (scratch1, _) = &mut scratch1.split_at_mut(scratch_len);
    let mut scratch1 = MpNativeEndianMutByteSlice::from_bytes(scratch1).unwrap();
    assert!(scratch2.len() >= scratch_len);
    let (scratch2, _) = &mut scratch2.split_at_mut(scratch_len);
    assert!(scratch3.len() >= scratch_len);
    let (scratch3, _) = &mut scratch3.split_at_mut(scratch_len);

    // The Extended Euclidean Algorithm (as it is implemented) only works for odd n.
    // 1.) Factor out powers of two to obtain n = n_{*} * 2^e.
    // 2.) Compute the inverse of op0 mod n_{*} using Extended Euclid first.
    // 3.) Use Garner's algorithm to obtain the the inverse of op0 mod n_{*} * 2^e
    //     as follows:
    // op0^{-1} mod n_{*} * 2^e
    //  = (op0^{-1} mod n_{*})
    //    + ((((op0^{-1} mod 2^e) - (op0^{-1} mod n_{*})) * (n_{*}^{-1} mod 2^e)) mod 2^e)
    //      * n_{*}
    // Note that the inverse modulo the power 2^e of two can be computed efficiently,
    // in time logarithmic in (the upper bound on) e, via Hensel lifting.
    let (n_is_nonzero, n_pow2_exp) = ct_find_first_set_bit_mp(n);
    assert!(n_is_nonzero.unwrap() != 0);
    ct_rshift_mp(n, n_pow2_exp);
    let max_n_pow2_exp = 8 * n.len() - 1;

    // Compute scratch0 = op0^{-1} mod 2^e first, the subsequently executed Extended Euclidean Algorithm
    // will destroy the value of op0.
    ct_inv_mod_pow2_mp(&mut scratch0, op0, n_pow2_exp, max_n_pow2_exp, scratch2);

    // Compute scratch1 = op0^{-1} mod n_{*}. Will destroy the value in op0.
    match ct_inv_mod_odd_mp_mp(&mut scratch1, op0, n, [scratch2, scratch3]) {
        Ok(_) => (),
        Err(CtInvModOddMpMpError::OperandsNotCoprime) => {
            // Undo the rshift on n for good measure.
            ct_lshift_mp(n, n_pow2_exp);
            return Err(CtInvModMpMpError::OperandsNotCoprime);
        }
    };

    // Duplicate scratch1 = op0^{-1} mod n_{*} into op0.
    op0.copy_from(&scratch1);

    // Compute scratch0 = (op0^{-1} mod 2^e) - (op0^{-1} mod n_{*}). Dismiss the borrow,
    // the result will eventually be taken modulo 2^e anyway.
    ct_sub_mp_mp(&mut scratch0, &scratch1);

    // Compute scratch1 = n_{*}^{-1} mod 2^e.
    ct_inv_mod_pow2_mp(&mut scratch1, n, n_pow2_exp, max_n_pow2_exp, scratch2);

    // Compute scratch0 = (op0^{-1} mod 2^e) - (op0^{-1} mod n_{*}) * (n_{*}^{-1} mod 2^e).
    ct_mul_trunc_mp_mp(&mut scratch0, scratch_len, &scratch1);
    // And take the product modulo 2^e
    ct_zeroize_bits_above_mp(&mut scratch0, n_pow2_exp);

    // Multiply by n_{*} and add to op0 to obtain the final result.
    ct_mul_trunc_mp_mp(&mut scratch0, scratch_len, n);
    let carry = ct_add_mp_mp(op0, &scratch0);
    assert_eq!(carry, 0);

    // Undo the rshift on n for the debug_assert!() comparison.
    ct_lshift_mp(n, n_pow2_exp);
    assert!(ct_lt_mp_mp(op0, n).unwrap() != 0);

    Ok(())
}

#[cfg(test)]
fn test_ct_inv_mod_mp_mp<T0: MpIntMutByteSlice, NT: MpIntMutByteSlice>() {
    use super::limb::LIMB_BYTES;
    use super::mul_impl::ct_mul_trunc_mp_l;

    fn test_one<T0: MpIntMutByteSlice, NT: MpIntMutByteSlice>(
        op0: &T0, n: &mut NT
    ) {
        use super::div_impl::ct_div_mp_mp;
        use super::cmp_impl::ct_is_one_mp;

        // Reserve an extra byte for the NotCoprime checks below.
        let mut scratch0 = vec![0u8; MpNativeEndianMutByteSlice::limbs_align_len(n.len() + 1)];
        let mut scratch1 = vec![0u8; MpNativeEndianMutByteSlice::limbs_align_len(n.len() + 1)];
        let mut scratch2 = vec![0u8; MpNativeEndianMutByteSlice::limbs_align_len(n.len() + 1)];
        let mut scratch3 = vec![0u8; MpNativeEndianMutByteSlice::limbs_align_len(n.len() + 1)];

        let mut op0_inv_mod_n = vec![0u8; T0::limbs_align_len(n.len())];
        let mut op0_inv_mod_n = T0::from_bytes(&mut op0_inv_mod_n).unwrap();
        op0_inv_mod_n.copy_from(op0);

        let scratch = [
            scratch0.as_mut_slice(), scratch1.as_mut_slice(),
            scratch2.as_mut_slice(), scratch3.as_mut_slice()
        ];
        ct_inv_mod_mp_mp(&mut op0_inv_mod_n, n, scratch).unwrap();

        // If n == 1, the multiplicative group does not exist and the result is fixed to zero.
        if ct_is_one_mp(n).unwrap() != 0 {
            assert_eq!(ct_is_zero_mp(&op0_inv_mod_n).unwrap(), 1);
            return;
        }

        // Multiply op0_inv_mod_n by op0 modulo n and verify the result comes out as 1.
        let mut product_buf = vec![0u8; MpNativeEndianMutByteSlice::limbs_align_len(2 * n.len())];
        let mut product = MpNativeEndianMutByteSlice::from_bytes(&mut product_buf).unwrap();
        product.copy_from(&op0_inv_mod_n);
        ct_mul_trunc_mp_mp(&mut product, n.len(), op0);
        ct_div_mp_mp::<_, _, MpNativeEndianMutByteSlice>(None, &mut product, n, None).unwrap();
        assert_eq!(ct_is_one_mp(&product).unwrap(), 1);

        // Verify that ct_inv_mod_mp_mp() correctly detects common factors and would error out.
        // Scaled by 2.
        let mut scaled_n = vec![0u8; NT::limbs_align_len(n.len() + 1)];
        let mut scaled_n = NT::from_bytes(&mut scaled_n).unwrap();
        scaled_n.copy_from(n);
        ct_lshift_mp(&mut scaled_n, 1);
        let mut scaled_op0 = vec![0u8; T0::limbs_align_len(n.len() + 1)];
        let mut scaled_op0 = NT::from_bytes(&mut scaled_op0).unwrap();
        scaled_op0.copy_from(n);
        ct_lshift_mp(&mut scaled_op0, 1);
        let scratch = [
            scratch0.as_mut_slice(), scratch1.as_mut_slice(),
            scratch2.as_mut_slice(), scratch3.as_mut_slice()
        ];
        assert!(matches!(
            ct_inv_mod_mp_mp(&mut scaled_op0, &mut scaled_n, scratch),
            Err(CtInvModMpMpError::OperandsNotCoprime)
        ));
        // Scaled by 3.
        let mut scaled_n = vec![0u8; NT::limbs_align_len(n.len() + 1)];
        let mut scaled_n = NT::from_bytes(&mut scaled_n).unwrap();
        scaled_n.copy_from(n);
        ct_mul_trunc_mp_l(&mut scaled_n, n.len(), 3);
        ct_lshift_mp(&mut scaled_n, 1);
        let mut scaled_op0 = vec![0u8; T0::limbs_align_len(n.len() + 1)];
        let mut scaled_op0 = NT::from_bytes(&mut scaled_op0).unwrap();
        scaled_op0.copy_from(n);
        ct_mul_trunc_mp_l(&mut scaled_op0, op0.len(), 3);
        let scratch = [
            scratch0.as_mut_slice(), scratch1.as_mut_slice(),
            scratch2.as_mut_slice(), scratch3.as_mut_slice()
        ];
        assert!(matches!(
            ct_inv_mod_mp_mp(&mut scaled_op0, &mut scaled_n, scratch),
            Err(CtInvModMpMpError::OperandsNotCoprime)
        ));
    }

    for l in [LIMB_BYTES - 1, 2 * LIMB_BYTES - 1, 3 * LIMB_BYTES - 1, 4 * LIMB_BYTES - 1] {
        let n_len = NT::limbs_align_len(l);
        let op0_len = T0::limbs_align_len(l);
        let n_op0_min_len = n_len.min(op0_len);
        let n_op0_high_min_len = (n_op0_min_len - 1) % LIMB_BYTES + 1;

        let mut n_buf = vec![0u8; n_len];
        let mut op0_buf = vec![0u8; op0_len];
        let mut n = NT::from_bytes(n_buf.as_mut_slice()).unwrap();
        let mut op0 = T0::from_bytes(op0_buf.as_mut_slice()).unwrap();
        n.store_l(0, 3);
        op0.store_l(0, 2);
        test_one(&op0, &mut n);

        for i in 0..8 * LIMB_BYTES.min(n_op0_min_len) - 1 {
            for j in 1..8 * LIMB_BYTES.min(n_op0_min_len) - 1 {
                let mut n_buf = vec![0u8; n_len];
                let mut op0_buf = vec![0u8; op0_len];
                let mut n = NT::from_bytes(n_buf.as_mut_slice()).unwrap();
                let mut op0 = T0::from_bytes(op0_buf.as_mut_slice()).unwrap();
                let limb_index = i / (8 * LIMB_BYTES);
                let i = i % (8 * LIMB_BYTES);
                n.store_l(limb_index, 1 << i);
                n.store_l(0, n.load_l(0) | 1);
                let limb_index = j / (8 * LIMB_BYTES);
                let j = j % (8 * LIMB_BYTES);
                op0.store_l(limb_index, 1 << j);
                test_one(&op0, &mut n);
            }
        }

        for i in 0..8 * LIMB_BYTES.min(n_op0_min_len) - 1 {
            for j in 1..8 * LIMB_BYTES.min(n_op0_min_len) - 1 {
                let mut n_buf = vec![0u8; 2 * n_len];
                let mut op0_buf = vec![0u8; 2 * op0_len];
                let mut n = NT::from_bytes(n_buf.as_mut_slice()).unwrap();
                let mut op0 = T0::from_bytes(op0_buf.as_mut_slice()).unwrap();
                let limb_index = i / (8 * LIMB_BYTES);
                let i = i % (8 * LIMB_BYTES);
                n.store_l(limb_index, 1 << i);
                ct_lshift_mp(&mut n, 8 * n_len);
                let limb_index = j / (8 * LIMB_BYTES);
                let j = j % (8 * LIMB_BYTES);
                op0.store_l(limb_index, 1 << j);
                op0.store_l(0, op0.load_l(0) | 1);
                test_one(&op0, &mut n);
            }
        }

        let mut n_buf = vec![0u8; n_len];
        let mut op0_buf = vec![0u8; op0_len];
        let mut n = NT::from_bytes(n_buf.as_mut_slice()).unwrap();
        let mut op0 = T0::from_bytes(op0_buf.as_mut_slice()).unwrap();
        n.store_l(0, 1);
        while n.load_l(n.nlimbs() - 1) >> 8 * (n_op0_high_min_len - 1) == 0 {
            ct_mul_trunc_mp_l(&mut n, n_len, 251);
        }
        op0.store_l(0, 1);
        while op0.load_l(op0.nlimbs() - 1) >> 8 * (n_op0_high_min_len - 1) == 0 {
            ct_mul_trunc_mp_l(&mut op0, op0_len, 241);
        }
        while op0.load_l(op0.nlimbs() - 1) >> 8 * (n_op0_high_min_len) - 1 == 0 {
            ct_mul_trunc_mp_l(&mut op0, op0_len, 2);
        }
        test_one(&op0, &mut n);

        let mut n_buf = vec![0u8; 2 * n_len];
        let mut op0_buf = vec![0u8; 2 * op0_len];
        let mut n = NT::from_bytes(n_buf.as_mut_slice()).unwrap();
        let mut op0 = T0::from_bytes(op0_buf.as_mut_slice()).unwrap();
        n.store_l(0, 1);
        while n.load_l(ct_mp_nlimbs(n_len) - 1) >> 8 * (n_op0_high_min_len - 1) == 0 {
            ct_mul_trunc_mp_l(&mut n, n_len, 251);
        }
        ct_lshift_mp(&mut n, 8 * n_len);
        op0.store_l(0, 1);
        let n_op0_high_min_len = (2 * n_op0_min_len - 1) % LIMB_BYTES + 1;
        while op0.load_l(op0.nlimbs() - 1) >> 8 * (n_op0_high_min_len - 1) == 0 {
            ct_mul_trunc_mp_l(&mut op0, 2 * op0_len, 241);
        }
        test_one(&op0, &mut n);
    }
}

#[test]
fn test_ct_inv_mod_be_be() {
    use super::limbs_buffer::MpBigEndianMutByteSlice;
    test_ct_inv_mod_mp_mp::<MpBigEndianMutByteSlice, MpBigEndianMutByteSlice>();
}

#[test]
fn test_ct_inv_mod_le_le() {
    use super::limbs_buffer::MpLittleEndianMutByteSlice;
    test_ct_inv_mod_mp_mp::<MpLittleEndianMutByteSlice, MpLittleEndianMutByteSlice>();
}

#[test]
fn test_ct_inv_mod_ne_ne() {
    test_ct_inv_mod_mp_mp::<MpNativeEndianMutByteSlice, MpNativeEndianMutByteSlice>();
}
