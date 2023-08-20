use super::limb::{ct_lsb_mask_l, LimbType, LIMB_BITS};
#[cfg(test)]
use super::limbs_buffer::MpMutUIntSlice;
use super::limbs_buffer::{
    ct_clear_bits_above_mp, ct_clear_bits_below_mp, ct_mp_nlimbs, MpMutUInt,
};
use super::usize_ct_cmp::{
    ct_eq_usize_usize, ct_geq_usize_usize, ct_gt_usize_usize, ct_leq_usize_usize, ct_lt_usize_usize,
};

pub fn ct_lshift_mp<T0: MpMutUInt>(op0: &mut T0, distance: usize) -> LimbType {
    if op0.is_empty() {
        return 0;
    }

    // First determine what would get shifted out into the (virtual) next higher
    // LIMB_BITS beyond op0.len().
    let op0_nlimbs = op0.nlimbs();
    let op0_nbits = 8 * op0.len();
    let shifted_out_low = {
        // The bits from [op0_nbits, op0_nbits + LIMB_BITS) - distance are getting
        // shifted into the (virtual) next higher limb. The begin or the whole range
        // might be out of bounds, i.e. the begin might be < 0, or even the end <= 0.
        // Cap them to zero as appropriate. Also, cap from above.
        let src_bits_begin_lt_0 = ct_lt_usize_usize(op0_nbits, distance);
        let src_bits_begin = op0_nbits.wrapping_sub(distance);
        let src_bits_begin = src_bits_begin_lt_0.select_usize(src_bits_begin, 0);
        let src_bits_begin_ge_op0_nbits = ct_geq_usize_usize(src_bits_begin, op0_nbits);
        let src_bits_begin = src_bits_begin_ge_op0_nbits.select_usize(src_bits_begin, op0_nbits);

        let src_bits_end_le_0 = ct_leq_usize_usize(op0_nbits + LIMB_BITS as usize, distance);
        let src_bits_end = (op0_nbits + LIMB_BITS as usize).wrapping_sub(distance);
        let src_bits_end = src_bits_end_le_0.select_usize(src_bits_end, 0);
        // Don't use op0_nbits, but the bits rounded up to the next limb boundary for
        // the upper bound on the end, that's important for correctly masking of
        // the high_src_mask below.
        let src_bits_end_gt_op0_nbits =
            ct_gt_usize_usize(src_bits_end, op0_nlimbs * LIMB_BITS as usize);
        let src_bits_end = src_bits_end_gt_op0_nbits.select_usize(src_bits_end, op0_nbits);

        let low_src_limb_index = src_bits_begin_ge_op0_nbits.select_usize(
            src_bits_begin / LIMB_BITS as usize,
            op0_nlimbs - 1, // Random valid index, the result will get masked.
        );
        let low_src_rshift = src_bits_begin % LIMB_BITS as usize;

        let high_src_limb_index = src_bits_end_le_0.select_usize(
            src_bits_end.wrapping_sub(1) / LIMB_BITS as usize,
            0, // Random valid index, the result will get masked.
        );
        let dst_high_lshift = (LIMB_BITS as usize - low_src_rshift) % LIMB_BITS as usize;
        // If src_bits_end had been capped from above, the limb will be handled by
        // src_bits_begin already. Be careful not to use the high source limb
        // part (with a wrong mask + shift) in this case.
        let high_src_mask =
            src_bits_end_gt_op0_nbits.select(ct_lsb_mask_l(low_src_rshift as u32), 0);

        let src_low = op0.load_l(low_src_limb_index) >> low_src_rshift;
        let src_high = op0.load_l(high_src_limb_index) & high_src_mask;
        // If all source bits are out of range, all calculations are void.
        let shifted_out_low = (src_bits_end_le_0 | src_bits_begin_ge_op0_nbits)
            .select((src_high << dst_high_lshift) | src_low, 0);
        let result_lshift = src_bits_begin_lt_0.select_usize(0, distance.wrapping_sub(op0_nbits));
        let result_lshift = src_bits_end_le_0.select_usize(result_lshift, 0);
        shifted_out_low << result_lshift
    };

    // For the actual in-place operand shifting, cap distance to the maximum
    // possible value.
    let distance = ct_gt_usize_usize(distance, op0_nbits).select_usize(distance, op0_nbits);

    // Only the limbs >= the shift distance actually receive contents from the lower
    // part. To obliterate timing traces dependent on the shift distance,
    // continue into the lower parts and let them receive some (garbage)
    // contents from the upper ones -- they'll eventually get cleared out again
    // before returning. Note that this makes the number of operations a constant,
    // but the stride length might still be obvservable as cache-induced timing
    // variations.
    let distance_nlimbs = ct_mp_nlimbs((distance + 7) / 8);
    let dst_high_lshift = distance % LIMB_BITS as usize;
    let distance_is_aligned = ct_eq_usize_usize(dst_high_lshift, 0);
    let low_src_rshift = distance_is_aligned.select_usize(LIMB_BITS as usize - dst_high_lshift, 0);
    let high_src_mask = distance_is_aligned.select(!0, 0);

    let mut dst_limb_index = op0_nlimbs;
    let mut src_limb_index_offset = 0;
    // The distance had been capped to op0_nbits.
    debug_assert!(distance_nlimbs <= dst_limb_index);
    // Initialize last_src_low as if the (non-existant) limb at index
    // op0_nlimbs had just been processed. If distance_nlimbs == 0,
    // avoid an out-of-bound access by decrementing, the loaded limb won't
    // get used anyway as high_src_mask == 0 in this case.
    let mut last_src_low = op0.load_l(
        dst_limb_index - ct_eq_usize_usize(distance_nlimbs, 0).select_usize(distance_nlimbs, 1),
    );
    while dst_limb_index > 0 {
        src_limb_index_offset = ct_eq_usize_usize(distance_nlimbs, dst_limb_index)
            .select_usize(src_limb_index_offset, op0_nlimbs);
        dst_limb_index -= 1;
        let src_high = last_src_low;
        let low_src_limb_index = dst_limb_index + src_limb_index_offset - distance_nlimbs;
        let src_low = op0.load_l(low_src_limb_index);
        last_src_low = src_low;
        let dst_low = src_low >> low_src_rshift;
        let dst_high = (src_high & high_src_mask) << dst_high_lshift;
        let dst_val = dst_high | dst_low;
        if dst_limb_index + 1 != op0_nlimbs {
            op0.store_l_full(dst_limb_index, dst_val);
        } else {
            op0.store_l(dst_limb_index, dst_val & op0.partial_high_mask());
        }
    }

    // The circular-buffer approach to shifting for the sake of constant time
    // filled the bits below distance with garbage. Clear them out.
    ct_clear_bits_below_mp(op0, distance);

    shifted_out_low
}

#[cfg(test)]
fn test_fill_limb_with_seq(first: u8) -> LimbType {
    use super::limb::LIMB_BYTES;

    let mut l: LimbType = 0;
    for i in 0..LIMB_BYTES as u8 {
        l <<= 8;
        l |= (LIMB_BYTES as u8 - i - 1 + first) as LimbType;
    }
    l
}

#[cfg(test)]
fn test_ct_lshift_mp_common<T0: MpMutUIntSlice>(op0_len: usize) {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::{MpUIntCommon as _, MpUIntCommonPriv as _};

    fn test_fill_limb_with_seq_lshifted(limb_index: usize, lshift_distance: usize) -> LimbType {
        let lshift_len = (lshift_distance + 7) / 8;
        let lshift_nlimbs = ct_mp_nlimbs(lshift_len);
        let dst_high_lshift = (lshift_distance % LIMB_BITS as usize) as u32;
        let low = if lshift_nlimbs <= limb_index {
            let src_low =
                test_fill_limb_with_seq(((limb_index - lshift_nlimbs) * LIMB_BYTES) as u8 + 1);
            if dst_high_lshift != 0 {
                src_low >> LIMB_BITS - dst_high_lshift
            } else {
                src_low
            }
        } else {
            0
        };

        let high = if dst_high_lshift != 0 && limb_index + 1 >= lshift_nlimbs {
            let src_high =
                test_fill_limb_with_seq(((limb_index + 1 - lshift_nlimbs) * LIMB_BYTES) as u8 + 1);
            let src_high = src_high & ct_lsb_mask_l(LIMB_BITS - dst_high_lshift);
            src_high << dst_high_lshift
        } else {
            0
        };

        high | low
    }

    let op0_nlimbs = ct_mp_nlimbs(op0_len);
    for i in 0..op0_nlimbs + 3 {
        for j in 0..LIMB_BITS as usize {
            let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
            let mut op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
            for k in 0..op0.nlimbs() {
                let val = test_fill_limb_with_seq(1 + (k * LIMB_BYTES) as u8);
                if k != op0.nlimbs() - 1 {
                    op0.store_l_full(k, val);
                } else {
                    op0.store_l(k, val & op0.partial_high_mask());
                }
            }
            let shift_distance = i * LIMB_BITS as usize + j;
            let shifted_out = ct_lshift_mp(&mut op0, shift_distance);

            for k in 0..op0.nlimbs() {
                let expected_val = test_fill_limb_with_seq_lshifted(k, shift_distance);
                let expected_val = if k + 1 != op0.nlimbs() {
                    expected_val
                } else {
                    expected_val & op0.partial_high_mask()
                };
                assert_eq!(op0.load_l(k), expected_val);
            }

            let op0_high_nbytes = op0.len() % LIMB_BYTES;
            let expected_shifted_out = if op0_high_nbytes != 0 {
                let src_low = if !op0.is_empty() {
                    test_fill_limb_with_seq_lshifted(op0.nlimbs() - 1, shift_distance)
                } else {
                    0
                };
                let src_high = test_fill_limb_with_seq_lshifted(op0.nlimbs(), shift_distance);

                let src_low_rshift = 8 * op0_high_nbytes;
                let low = src_low >> src_low_rshift;
                let dst_high_lshift = LIMB_BITS as usize - src_low_rshift;
                let high = src_high << dst_high_lshift;
                high | low
            } else {
                test_fill_limb_with_seq_lshifted(op0.nlimbs(), shift_distance)
            };
            let expected_shifted_out = if shift_distance < LIMB_BITS as usize {
                expected_shifted_out & ct_lsb_mask_l(shift_distance as u32)
            } else {
                expected_shifted_out
            };
            assert_eq!(shifted_out, expected_shifted_out)
        }
    }
}

#[cfg(test)]
fn test_ct_lshift_mp_with_unaligned_lengths<T0: MpMutUIntSlice>() {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::MpUIntCommon as _;

    let mut op0 = [0xffu8.into(); 1];
    let mut op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
    let shifted_out = ct_lshift_mp(&mut op0, 0);
    assert_eq!(op0.load_l(0), 0xff);
    assert_eq!(shifted_out, 0);

    for i in 1..LIMB_BYTES + 1 {
        let mut op0 = [0xffu8.into(); 1];
        let mut op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
        let shifted_out = ct_lshift_mp(&mut op0, 8 * i);
        assert_eq!(op0.load_l(0), 0);
        assert_eq!(shifted_out, 0xff << 8 * (i - 1));
    }

    let mut op0 = [0xffu8.into(); 1];
    let mut op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
    let shifted_out = ct_lshift_mp(&mut op0, 8 * (LIMB_BYTES + 1));
    assert_eq!(op0.load_l(0), 0);
    assert_eq!(shifted_out, 0);

    test_ct_lshift_mp_common::<T0>(3 * LIMB_BYTES - 1);
}

#[cfg(test)]
fn test_ct_lshift_mp_with_aligned_lengths<T0: MpMutUIntSlice>() {
    use super::limb::LIMB_BYTES;

    for i in 0..LIMB_BYTES + 3 {
        let mut op0 = [0u8.into(); 0];
        let mut op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
        let shifted_out = ct_lshift_mp(&mut op0, 8 * i);
        assert_eq!(shifted_out, 0);
    }

    test_ct_lshift_mp_common::<T0>(4 * LIMB_BYTES);
}

#[test]
fn test_ct_lshift_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_lshift_mp_with_unaligned_lengths::<MpMutBigEndianUIntByteSlice>();
    test_ct_lshift_mp_with_aligned_lengths::<MpMutBigEndianUIntByteSlice>();
}

#[test]
fn test_ct_lshift_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_lshift_mp_with_unaligned_lengths::<MpMutLittleEndianUIntByteSlice>();
    test_ct_lshift_mp_with_aligned_lengths::<MpMutLittleEndianUIntByteSlice>();
}

#[test]
fn test_ct_lshift_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_lshift_mp_with_aligned_lengths::<MpMutNativeEndianUIntLimbsSlice>();
}

pub fn ct_rshift_mp<T0: MpMutUInt>(op0: &mut T0, distance: usize) -> LimbType {
    if op0.is_empty() {
        return 0;
    }

    // First determine what would get shifted out into the (virtual) next lower
    // LIMB_BITS beyond the zeroth limb.
    let op0_nlimbs = op0.nlimbs();
    let op0_nbits = 8 * op0.len();
    let shifted_out_high = {
        let src_bits_begin_lt_0 = ct_lt_usize_usize(distance, LIMB_BITS as usize);
        let src_bits_begin = distance.wrapping_sub(LIMB_BITS as usize);
        let src_bits_begin = src_bits_begin_lt_0.select_usize(src_bits_begin, 0);
        let src_bits_begin_ge_op0_nbits = ct_geq_usize_usize(src_bits_begin, op0_nbits);
        let src_bits_begin = src_bits_begin_ge_op0_nbits.select_usize(src_bits_begin, op0_nbits);

        let src_bits_end = distance;
        // Don't use op0_nbits, but the bits rounded up to the next limb boundary for
        // the upper bound on the end, that's important for correctly masking of
        // the high_src_mask below.
        let src_bits_end_gt_op0_nbits =
            ct_gt_usize_usize(src_bits_end, op0_nlimbs * LIMB_BITS as usize);
        let src_bits_end = src_bits_end_gt_op0_nbits.select_usize(src_bits_end, op0_nbits);
        let src_bits_end_eq_0 = ct_eq_usize_usize(0, distance);

        let low_src_limb_index = src_bits_begin_ge_op0_nbits.select_usize(
            src_bits_begin / LIMB_BITS as usize,
            op0_nlimbs - 1, // Random valid index, the result will get masked.
        );
        let low_src_rshift = src_bits_begin % LIMB_BITS as usize;

        let high_src_limb_index = src_bits_end_eq_0.select_usize(
            src_bits_end.wrapping_sub(1) / LIMB_BITS as usize,
            0, // Random valid index, the result will get masked.
        );
        let dst_high_lshift = (LIMB_BITS as usize - low_src_rshift) % LIMB_BITS as usize;
        // If src_bits_end had been capped from above, the limb will be handled by
        // src_bits_begin already. Be careful not to use the high source limb
        // part (with a wrong mask + shift) in this case.
        let high_src_mask =
            src_bits_end_gt_op0_nbits.select(ct_lsb_mask_l(low_src_rshift as u32), 0);

        let src_low = op0.load_l(low_src_limb_index) >> low_src_rshift;
        let src_high = op0.load_l(high_src_limb_index) & high_src_mask;
        // If all source bits are out of range, all calculations are void.
        let shifted_out_high = (src_bits_end_eq_0 | src_bits_begin_ge_op0_nbits)
            .select((src_high << dst_high_lshift) | src_low, 0);
        let result_lshift =
            src_bits_begin_lt_0.select_usize(0, (LIMB_BITS as usize).wrapping_sub(distance));
        let result_lshift = src_bits_end_eq_0.select_usize(result_lshift, 0);
        shifted_out_high << result_lshift
    };

    // For the actual in-place operand shifting, cap distance to the maximum
    // possible value.
    let distance = ct_gt_usize_usize(distance, op0_nbits).select_usize(distance, op0_nbits);

    // Only the limbs <= op0_nbits - distance actually receive contents from the
    // upper part. To obliterate timing traces dependent on the shift distance,
    // continue into the upper parts and let them receive some (garbage)
    // contents from the lower ones -- they'll eventually get cleared out again
    // before returning. Note that this makes the number of operations a constant,
    // but the stride length might still be obvservable as cache-induced timing
    // variations.
    let distance_nlimbs = distance / LIMB_BITS as usize; // Not rounding up is on purpose.
    let low_src_rshift = distance % LIMB_BITS as usize;
    let distance_is_aligned = ct_eq_usize_usize(low_src_rshift, 0);
    let dst_high_lshift = distance_is_aligned.select_usize(LIMB_BITS as usize - low_src_rshift, 0);
    let high_src_mask = distance_is_aligned.select(!0, 0);

    let mut dst_limb_index = 0;
    let mut src_limb_index_offset =
        ct_geq_usize_usize(distance_nlimbs, op0_nlimbs).select_usize(0, op0_nlimbs);
    // Initialize last_src_high as if the (non-existant) limb at index
    // -1 had just been processed.
    let mut last_src_high = op0.load_l(distance_nlimbs - src_limb_index_offset);
    while dst_limb_index < op0_nlimbs {
        let src_low = last_src_high;
        let high_src_limb_index = dst_limb_index + distance_nlimbs + 1;
        src_limb_index_offset =
            ct_eq_usize_usize(high_src_limb_index - src_limb_index_offset, op0_nlimbs)
                .select_usize(src_limb_index_offset, src_limb_index_offset + op0_nlimbs);
        let high_src_limb_index = high_src_limb_index - src_limb_index_offset;
        let src_high = op0.load_l(high_src_limb_index);
        last_src_high = src_high;
        let dst_low = src_low >> low_src_rshift;
        let dst_high = (src_high & high_src_mask) << dst_high_lshift;
        let dst_val = dst_high | dst_low;
        if dst_limb_index + 1 != op0_nlimbs {
            op0.store_l_full(dst_limb_index, dst_val);
        } else {
            op0.store_l(dst_limb_index, dst_val & op0.partial_high_mask());
        }
        dst_limb_index += 1;
    }

    // The circular-buffer approach to shifting for the sake of constant time
    // filled the bits above op0_nbits - distance with garbage. Clear them out.
    ct_clear_bits_above_mp(op0, op0_nbits - distance);

    shifted_out_high
}

#[cfg(test)]
fn test_ct_rshift_mp_common<T0: MpMutUIntSlice>(op0_len: usize) {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::{MpUIntCommon as _, MpUIntCommonPriv as _};

    // limb_index is offset by one: an index of zero is used
    // for specifiying the virtual limb rshifted into.
    fn test_fill_limb_with_seq_rshifted(
        limb_index: usize,
        rshift_distance: usize,
        op_len: usize,
    ) -> LimbType {
        let op_nlimbs = ct_mp_nlimbs(op_len);
        let op_partial_high_mask = if op_len % LIMB_BYTES != 0 {
            ct_lsb_mask_l(8 * (op_len % LIMB_BYTES) as u32)
        } else {
            !0
        };

        let rshift_len = rshift_distance / 8;
        let rshift_nlimbs = rshift_len / LIMB_BYTES;
        let src_low_rshift = rshift_distance % LIMB_BITS as usize;
        let low = if rshift_nlimbs == 0 && limb_index == 0 {
            0
        } else if limb_index + rshift_nlimbs <= op_nlimbs {
            let src_low =
                test_fill_limb_with_seq(((limb_index + rshift_nlimbs - 1) * LIMB_BYTES) as u8 + 1);
            let src_low = if limb_index + rshift_nlimbs == op_nlimbs {
                src_low & op_partial_high_mask
            } else {
                src_low
            };
            src_low >> src_low_rshift
        } else {
            0
        };

        let high = if src_low_rshift != 0 && limb_index + rshift_nlimbs + 1 <= op_nlimbs {
            let src_high =
                test_fill_limb_with_seq(((limb_index + rshift_nlimbs) * LIMB_BYTES) as u8 + 1);
            let src_high = if limb_index + rshift_nlimbs + 1 == op_nlimbs {
                src_high & op_partial_high_mask
            } else {
                src_high
            };
            src_high << (LIMB_BITS as usize - src_low_rshift)
        } else {
            0
        };

        high | low
    }

    let op0_nlimbs = ct_mp_nlimbs(op0_len);
    for i in 0..op0_nlimbs + 3 {
        for j in 0..LIMB_BITS as usize {
            let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
            let mut op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
            for k in 0..op0.nlimbs() {
                let val = test_fill_limb_with_seq(1 + (k * LIMB_BYTES) as u8);
                if k != op0.nlimbs() - 1 {
                    op0.store_l_full(k, val);
                } else {
                    op0.store_l(k, val & op0.partial_high_mask());
                }
            }
            let shift_distance = i * LIMB_BITS as usize + j;
            let shifted_out = ct_rshift_mp(&mut op0, shift_distance);

            for k in 0..op0.nlimbs() {
                let expected_val = test_fill_limb_with_seq_rshifted(k + 1, shift_distance, op0_len);
                assert_eq!(op0.load_l(k), expected_val);
            }

            assert_eq!(
                shifted_out,
                test_fill_limb_with_seq_rshifted(0, shift_distance, op0_len)
            );
        }
    }
}

#[cfg(test)]
fn test_ct_rshift_mp_with_unaligned_lengths<T0: MpMutUIntSlice>() {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::MpUIntCommon as _;

    let mut op0 = [0xffu8.into(); 1];
    let mut op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
    let shifted_out = ct_rshift_mp(&mut op0, 0);
    assert_eq!(op0.load_l(0), 0xff);
    assert_eq!(shifted_out, 0);

    for i in 1..LIMB_BYTES + 1 {
        let mut op0 = [0xffu8.into(); 1];
        let mut op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
        let shifted_out = ct_rshift_mp(&mut op0, 8 * i);
        assert_eq!(op0.load_l(0), 0);
        assert_eq!(shifted_out, 0xff << 8 * (LIMB_BYTES - i));
    }

    let mut op0 = [0xffu8.into(); 1];
    let mut op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
    let shifted_out = ct_rshift_mp(&mut op0, 8 * (LIMB_BYTES + 1));
    assert_eq!(op0.load_l(0), 0);
    assert_eq!(shifted_out, 0);

    test_ct_rshift_mp_common::<T0>(3 * LIMB_BYTES - 1);
}

#[cfg(test)]
fn test_ct_rshift_mp_with_aligned_lengths<T0: MpMutUIntSlice>() {
    use super::limb::LIMB_BYTES;

    for i in 0..LIMB_BYTES + 3 {
        let mut op0 = [0u8.into(); 0];
        let mut op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
        let shifted_out = ct_rshift_mp(&mut op0, 8 * i);
        assert_eq!(shifted_out, 0);
    }

    test_ct_rshift_mp_common::<T0>(4 * LIMB_BYTES);
}

#[test]
fn test_ct_rshift_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_rshift_mp_with_unaligned_lengths::<MpMutBigEndianUIntByteSlice>();
    test_ct_rshift_mp_with_aligned_lengths::<MpMutBigEndianUIntByteSlice>();
}

#[test]
fn test_ct_rshift_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_rshift_mp_with_unaligned_lengths::<MpMutLittleEndianUIntByteSlice>();
    test_ct_rshift_mp_with_aligned_lengths::<MpMutLittleEndianUIntByteSlice>();
}

#[test]
fn test_ct_rshift_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_rshift_mp_with_aligned_lengths::<MpMutNativeEndianUIntLimbsSlice>();
}
