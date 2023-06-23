use super::limbs_buffer::{MPIntMutByteSlice, mp_ct_nlimbs, MPIntMutByteSlicePriv as _};
use super::limb::{LIMB_BYTES, LimbType};

pub fn mp_lshift_mp<T0: MPIntMutByteSlice>(op0: &mut T0, distance_len: usize) -> LimbType{
    if op0.is_empty() || distance_len == 0 {
        return 0;
    }

    let op0_nlimbs = op0.nlimbs();

    // First determine what would get shifted out into the (virtual) next higher LIMB_BITS beyond
    // op0.len(). If the shift distance is past these virtual next higher LIMB_BITS, that would be
    // zero.
    let shifted_out_low = if distance_len < op0.len() + LIMB_BYTES {
        // If the shift distance exceeds the op0.len() (by at most LIMB_BYTES - 1, as per the
        // condition right above), the part of the shifted out bits would stem from the original
        // least significant limb and part from the zero bits shifted in from the right.
        if op0.len() >= distance_len {
            let low_src_byte_index = op0.len() - distance_len;
            let low_src_limb_index = low_src_byte_index / LIMB_BYTES;
            let low_src_rshift = low_src_byte_index % LIMB_BYTES;
            let (dst_high_lshift, high_src_mask) = if low_src_rshift != 0 {
                let dst_high_lshift = LIMB_BYTES - low_src_rshift;
                let high_src_mask = ((1 as LimbType) << 8 * low_src_rshift) - 1;
                (dst_high_lshift, high_src_mask)
            } else {
                (0, 0)
            };

            let low = op0.load_l(low_src_limb_index) >> 8 * low_src_rshift;
            let high = if low_src_limb_index + 1 < op0_nlimbs {
                op0.load_l(low_src_limb_index + 1) & high_src_mask
            } else {
                0
            };
            (high << 8 * dst_high_lshift) | low
        } else {
            op0.load_l(0) << 8 * (distance_len - op0.len())
        }
    } else {
        0
    };

    // Update the high limbs above the shift distance, i.e. those
    // which will receive non-trivial contents from the lower part.
    let distance_nlimbs = mp_ct_nlimbs(distance_len);
    let dst_high_lshift = distance_len % LIMB_BYTES;
    let (low_src_rshift, high_src_mask) = if dst_high_lshift != 0 {
        let low_src_rshift = LIMB_BYTES - dst_high_lshift;
        let high_src_mask = ((1 as LimbType) << 8 * low_src_rshift) - 1;
        (low_src_rshift, high_src_mask)
    } else {
        (0, 0)
    };

    let mut dst_limb_index = op0_nlimbs;
    let mut last_src_low = if dst_limb_index > distance_nlimbs {
        dst_limb_index -= 1;
        let src_high = op0.load_l(dst_limb_index - distance_nlimbs + 1);
        let src_low = op0.load_l_full(dst_limb_index - distance_nlimbs);
        // distance_nlimbs >= 1, so the load of src_low is not
        // from the high limb, i.e. it cannot be partial.
        let dst_low =  src_low >> 8 * low_src_rshift;
        let dst_high = (src_high & high_src_mask) << 8 * dst_high_lshift;
        op0.store_l(dst_limb_index, (dst_high | dst_low) & op0.partial_high_mask());
        src_low
    } else {
        // This will be used for the single limb containing the shift distance boundary, i.e. op0's
        // high limb in this case, below.
        op0.load_l(0)
    };
    while dst_limb_index > distance_nlimbs {
        dst_limb_index -= 1;
        let low_src_limb_index = dst_limb_index - distance_nlimbs;
        let src_high = last_src_low;
        let src_low = op0.load_l_full(low_src_limb_index);
        last_src_low = src_low;
        let dst_low =  src_low >> 8 * low_src_rshift;
        let dst_high = (src_high & high_src_mask) << 8 * dst_high_lshift;
        op0.store_l_full(dst_limb_index, dst_high | dst_low);
    }
    // The limb containing the shift distance boundary will receive bits from both, parts of the
    // least signigicant original limb and from the zero bits shifted in from the right.  Note that
    // if there is no such limb containing the shift distance boundary, either because the shift
    // distance is aligned or because it's out of bounds, the subsequent zeroize_bytes_below() will
    // account for it.
    dst_limb_index -= 1;
    // src_low is zero for the least significant limb which might still contain parts of
    // the shifted contents.
    let src_high = last_src_low;
    let dst_high = (src_high & high_src_mask) << 8 * dst_high_lshift;
    if dst_limb_index != op0_nlimbs - 1 {
        op0.store_l_full(dst_limb_index, dst_high);
    } else {
        debug_assert!(op0_nlimbs <= distance_nlimbs);
        op0.store_l(dst_limb_index, dst_high & op0.partial_high_mask());
    }

    // All the other lower limbs below the shift distance are to be set to zero.
    op0.zeroize_bytes_below(op0.len().min(distance_len));

    shifted_out_low
}

#[cfg(test)]
fn test_fill_limb_with_seq(first: u8, offset_len: usize) -> LimbType {
    if offset_len >= LIMB_BYTES {
        return 0;
    }
    let mut l: LimbType = 0;
    for i in 0..(LIMB_BYTES - offset_len) as u8 {
        l <<= 8;
        l |= ((LIMB_BYTES - offset_len) as u8 - i - 1 + first) as LimbType;
    }
    l <<= 8 * offset_len;
    l
}

#[cfg(test)]
fn test_mp_lshift_mp_common<T0: MPIntMutByteSlice>(op0_len: usize) {
    use super::limbs_buffer::{MPIntByteSliceCommon, MPIntByteSliceCommonPriv};

    let op0_nlimbs = mp_ct_nlimbs(op0_len);
    for i in 0..op0_nlimbs + 3 {
        for j in 0..LIMB_BYTES {
            let mut op0 = vec![0u8; op0_len];
            let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
            for k in 0..op0.nlimbs() {
                let val = test_fill_limb_with_seq(1 + (k * LIMB_BYTES) as u8, 0);
                if k != op0.nlimbs() - 1 {
                    op0.store_l_full(k, val);
                } else {
                    op0.store_l(k, val & op0.partial_high_mask());
                }
            }
            let shift_len = i * LIMB_BYTES + j;
            let shifted_out = mp_lshift_mp(&mut op0, shift_len);

            for k in 0..i.min(op0.nlimbs()) {
                assert_eq!(op0.load_l(k), 0);
            }

            if i < op0.nlimbs() {
                let expected_val = test_fill_limb_with_seq(1, j);
                if i != op0.nlimbs() - 1 {
                    assert_eq!(op0.load_l_full(i), expected_val);
                } else {
                    assert_eq!(op0.load_l(i), expected_val & op0.partial_high_mask());
                }
            }

            for k in i + 1..op0.nlimbs() {
                let expected_val = test_fill_limb_with_seq(
                    (k * LIMB_BYTES - shift_len + 1) as u8,
                    0
                );

                if k != op0.nlimbs() - 1 {
                    assert_eq!(op0.load_l_full(k), expected_val);
                } else {
                    assert_eq!(op0.load_l(k), expected_val & op0.partial_high_mask());
                }
            }

            let (shifted_out_mask, shifted_out_first, shifted_out_offset) = if shift_len < LIMB_BYTES {
                (
                    ((1 as LimbType) << 8 * shift_len) - 1,
                    (op0.len() - shift_len + 1) as u8,
                    0
                )
            } else if shift_len <= op0.len() {
                (
                    !0,
                    (op0.len() - shift_len + 1) as u8,
                    0
                )
            } else {
                (
                    !0,
                    1,
                    shift_len - op0.len(),
                )
            };
            assert_eq!(
                shifted_out,
                test_fill_limb_with_seq(
                    shifted_out_first,
                    shifted_out_offset
                ) & shifted_out_mask
            );
        }
    }
}


#[cfg(test)]
fn test_mp_lshift_mp_with_unaligned_lengths<T0: MPIntMutByteSlice>() {
    use super::limbs_buffer::MPIntByteSliceCommon;

    let mut op0: [u8; 1] = [0xff; 1];
    let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
    let shifted_out = mp_lshift_mp(&mut op0, 0);
    assert_eq!(op0.load_l(0), 0xff);
    assert_eq!(shifted_out, 0);

    for i in 1..LIMB_BYTES + 1 {
        let mut op0: [u8; 1] = [0xff; 1];
        let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
        let shifted_out = mp_lshift_mp(&mut op0, i);
        assert_eq!(op0.load_l(0), 0);
        assert_eq!(shifted_out, 0xff << 8 * (i - 1));
    }

    let mut op0: [u8; 1] = [0xff; 1];
    let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
    let shifted_out = mp_lshift_mp(&mut op0, LIMB_BYTES + 1);
    assert_eq!(op0.load_l(0), 0);
    assert_eq!(shifted_out, 0);

    test_mp_lshift_mp_common::<T0>(3 * LIMB_BYTES - 1);
}

#[cfg(test)]
fn test_mp_lshift_mp_with_aligned_lengths<T0: MPIntMutByteSlice>() {
    for i in 0..LIMB_BYTES + 3 {
        let mut op0: [u8; 0] = [0; 0];
        let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
        let shifted_out = mp_lshift_mp(&mut op0, i);
        assert_eq!(shifted_out, 0);
    }

    test_mp_lshift_mp_common::<T0>(4 * LIMB_BYTES);
}

#[test]
fn test_mp_lshift_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_lshift_mp_with_unaligned_lengths::<MPBigEndianMutByteSlice>();
    test_mp_lshift_mp_with_aligned_lengths::<MPBigEndianMutByteSlice>();
}

#[test]
fn test_mp_lshift_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_lshift_mp_with_unaligned_lengths::<MPLittleEndianMutByteSlice>();
    test_mp_lshift_mp_with_aligned_lengths::<MPLittleEndianMutByteSlice>();
}

#[test]
fn test_mp_lshift_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_lshift_mp_with_aligned_lengths::<MPNativeEndianMutByteSlice>();
}

pub fn mp_rshift_mp<T0: MPIntMutByteSlice>(op0: &mut T0, distance_len: usize) -> LimbType{
    if op0.is_empty() || distance_len == 0 {
        return 0;
    }

    let op0_nlimbs = op0.nlimbs();

    // First determine what would get shifted out into the (virtual) next lower LIMB_BITS beyond
    // the zeroth limb. If the shift distance is past these virtual next lower LIMB_BITS, that would be
    // zero.
    let shifted_out_high = if distance_len < op0.len() + LIMB_BYTES {
        // If the shift distance is smaller than limb, only the distance_len least significant limbs
        // of the low limb are shifted out into the virtual limb on the right.
        if distance_len >= LIMB_BYTES {
            let low_src_byte_index = distance_len - LIMB_BYTES;
            let low_src_limb_index = low_src_byte_index / LIMB_BYTES;
            let low_src_rshift = low_src_byte_index % LIMB_BYTES;
            let (dst_high_lshift, high_src_mask) = if low_src_rshift != 0 {
                let dst_high_lshift = LIMB_BYTES - low_src_rshift;
                let high_src_mask = ((1 as LimbType) << 8 * low_src_rshift) - 1;
                (dst_high_lshift, high_src_mask)
            } else {
                (0, 0)
            };

            let low = op0.load_l(low_src_limb_index) >> 8 * low_src_rshift;
            let high = if low_src_limb_index + 1 < op0_nlimbs {
                op0.load_l(low_src_limb_index + 1) & high_src_mask
            } else {
                0
            };
            (high << 8 * dst_high_lshift) | low
        } else {
            op0.load_l(0) << 8 * (LIMB_BYTES - distance_len)
        }
    } else {
        0
    };

    // Update the low limbs below the shift distance, i.e. those
    // which will receive non-trivial contents from the hight part.
    let low_src_distance_nlimbs = distance_len / LIMB_BYTES; // Not rounding up is on purpose.
    let low_src_rshift = distance_len % LIMB_BYTES;
    let (dst_high_lshift, high_src_mask) = if low_src_rshift != 0 {
        let dst_high_lshift = LIMB_BYTES - low_src_rshift;
        let high_src_mask = ((1 as LimbType) << 8 * low_src_rshift) - 1;
        (dst_high_lshift, high_src_mask)
    } else {
        (0, 0)
    };

    let mut dst_limb_index = 0;
    let mut last_src_high = if dst_limb_index + low_src_distance_nlimbs < op0_nlimbs {
        op0.load_l(dst_limb_index + low_src_distance_nlimbs)
    } else {
        // This will be used for the single limb containing the shift distance boundary, i.e. op0's
        // low limb in this case, below.
        0
    };
    while dst_limb_index + low_src_distance_nlimbs + 1 < op0_nlimbs {
        let src_low = last_src_high;
        let src_high = op0.load_l(dst_limb_index + low_src_distance_nlimbs + 1);
        last_src_high = src_high;
        let dst_low =  src_low >> 8 * low_src_rshift;
        let dst_high = (src_high & high_src_mask) << 8 * dst_high_lshift;
        op0.store_l_full(dst_limb_index, dst_high | dst_low);
        dst_limb_index += 1;
    }

    // The limb containing the shift distance boundary, if any, will receive bits from both, parts of the
    // most signigicant original limb and from the zero bits shifted in from the left.
    let src_low = last_src_high;
    // src_high is zero for the most significant limb which might still contain parts of
    // the shifted contents.
    let dst_low =  src_low >> 8 * low_src_rshift;
    debug_assert!(
        dst_limb_index + 1 < op0_nlimbs ||
            dst_low & !op0.partial_high_mask() == 0
    );
    op0.store_l(dst_limb_index, dst_low);
    dst_limb_index += 1;

    // All the other higher bits above the shift distance are to be set to zero.
    op0.zeroize_bytes_above(op0.len().min(dst_limb_index * LIMB_BYTES));

    shifted_out_high
}

#[cfg(test)]
fn test_mp_rshift_mp_common<T0: MPIntMutByteSlice>(op0_len: usize) {
    use super::limbs_buffer::{MPIntByteSliceCommon, MPIntByteSliceCommonPriv};


    let op0_nlimbs = mp_ct_nlimbs(op0_len);
    for i in 0..op0_nlimbs + 3 {
        for j in 0..LIMB_BYTES {
            let mut op0 = vec![0u8; op0_len];
            let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
            for k in 0..op0.nlimbs() {
                let val = test_fill_limb_with_seq(1 + (k * LIMB_BYTES) as u8, 0);
                if k != op0.nlimbs() - 1 {
                    op0.store_l_full(k, val);
                } else {
                    op0.store_l(k, val & op0.partial_high_mask());
                }
            }
            let shift_len = i * LIMB_BYTES + j;
            let shifted_out = mp_rshift_mp(&mut op0, shift_len);

            let shifted_zeroes_begin = op0.len().max(shift_len) - shift_len;
            for k in mp_ct_nlimbs(shifted_zeroes_begin)..op0.nlimbs() {
                assert_eq!(op0.load_l(k), 0);
            }

            if shifted_zeroes_begin % LIMB_BYTES != 0  {
                let k = shifted_zeroes_begin / LIMB_BYTES;
                let expected_val = test_fill_limb_with_seq(
                    (k * LIMB_BYTES + shift_len + 1) as u8,
                    0
                );
                let high_zeroes = LIMB_BYTES - shifted_zeroes_begin % LIMB_BYTES;
                let expected_val = expected_val << 8 * high_zeroes >> 8 * high_zeroes;
                assert_eq!(op0.load_l(k), expected_val);
            }

            for k in 0..shifted_zeroes_begin / LIMB_BYTES {
                let expected_val = test_fill_limb_with_seq(
                    (k * LIMB_BYTES + shift_len + 1) as u8,
                    0
                );
                assert_eq!(op0.load_l(k), expected_val);
            }

            let (shifted_out_mask, shifted_out_first, shifted_out_offset) = if shift_len < LIMB_BYTES {
                (
                    !0,
                    1,
                    LIMB_BYTES - shift_len
                )
            } else if shift_len <= op0.len() {
                (
                    !0,
                    (shift_len - LIMB_BYTES + 1) as u8,
                    0
                )
            } else if shift_len < op0.len() + LIMB_BYTES {
                (
                    ((1 as LimbType) << 8 * (op0.len() + LIMB_BYTES - shift_len)) - 1,
                    (shift_len - LIMB_BYTES + 1) as u8,
                    0
                )
            } else {
                (0, 0, 0)
            };

            assert_eq!(
                shifted_out,
                test_fill_limb_with_seq(
                    shifted_out_first,
                    shifted_out_offset
                ) & shifted_out_mask
            );
        }
    }
}

#[cfg(test)]
fn test_mp_rshift_mp_with_unaligned_lengths<T0: MPIntMutByteSlice>() {
    use super::limbs_buffer::MPIntByteSliceCommon;

    let mut op0: [u8; 1] = [0xff; 1];
    let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
    let shifted_out = mp_rshift_mp(&mut op0, 0);
    assert_eq!(op0.load_l(0), 0xff);
    assert_eq!(shifted_out, 0);

    for i in 1..LIMB_BYTES + 1 {
        let mut op0: [u8; 1] = [0xff; 1];
        let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
        let shifted_out = mp_rshift_mp(&mut op0, i);
        assert_eq!(op0.load_l(0), 0);
        assert_eq!(shifted_out, 0xff << 8 * (LIMB_BYTES - i));
    }

    let mut op0: [u8; 1] = [0xff; 1];
    let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
    let shifted_out = mp_rshift_mp(&mut op0, LIMB_BYTES + 1);
    assert_eq!(op0.load_l(0), 0);
    assert_eq!(shifted_out, 0);

    test_mp_rshift_mp_common::<T0>(3 * LIMB_BYTES - 1);
}

#[cfg(test)]
fn test_mp_rshift_mp_with_aligned_lengths<T0: MPIntMutByteSlice>() {
    for i in 0..LIMB_BYTES + 3 {
        let mut op0: [u8; 0] = [0; 0];
        let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
        let shifted_out = mp_rshift_mp(&mut op0, i);
        assert_eq!(shifted_out, 0);
    }

    test_mp_rshift_mp_common::<T0>(4 * LIMB_BYTES);
}

#[test]
fn test_mp_rshift_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_rshift_mp_with_unaligned_lengths::<MPBigEndianMutByteSlice>();
    test_mp_rshift_mp_with_aligned_lengths::<MPBigEndianMutByteSlice>();
}

#[test]
fn test_mp_rshift_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_rshift_mp_with_unaligned_lengths::<MPLittleEndianMutByteSlice>();
    test_mp_rshift_mp_with_aligned_lengths::<MPLittleEndianMutByteSlice>();
}

#[test]
fn test_mp_rshift_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_rshift_mp_with_aligned_lengths::<MPNativeEndianMutByteSlice>();
}
