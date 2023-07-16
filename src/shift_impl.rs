use super::limbs_buffer::{MPIntMutByteSlice, mp_ct_nlimbs, MPIntMutByteSlicePriv as _};
use super::limb::{LIMB_BYTES, LIMB_BITS, LimbType};

pub fn mp_lshift_mp<T0: MPIntMutByteSlice>(op0: &mut T0, distance: usize) -> LimbType{
    if op0.is_empty() || distance == 0 {
        return 0;
    }

    let op0_nlimbs = op0.nlimbs();

    // First determine what would get shifted out into the (virtual) next higher LIMB_BITS beyond
    // op0.len(). If the shift distance is past these virtual next higher LIMB_BITS, that would be
    // zero.
    let shifted_out_low = if distance < 8 * op0.len() + LIMB_BITS as usize {
        // If the shift distance exceeds the op0.len() (by at most LIMB_BYTES - 1, as per the
        // condition right above), the part of the shifted out bits would stem from the original
        // least significant limb and part from the zero bits shifted in from the right.
        if 8 * op0.len() >= distance {
            let low_src_bit_index = 8 * op0.len() - distance;
            let low_src_limb_index = low_src_bit_index / LIMB_BITS as usize;
            let low_src_rshift = low_src_bit_index % LIMB_BITS as usize;
            let (dst_high_lshift, high_src_mask) = if low_src_rshift != 0 {
                let dst_high_lshift = LIMB_BITS as usize - low_src_rshift;
                let high_src_mask = ((1 as LimbType) << low_src_rshift) - 1;
                (dst_high_lshift, high_src_mask)
            } else {
                (0, 0)
            };

            let low = op0.load_l(low_src_limb_index) >> low_src_rshift;
            let high = if low_src_limb_index + 1 < op0_nlimbs {
                op0.load_l(low_src_limb_index + 1) & high_src_mask
            } else {
                0
            };
            (high << dst_high_lshift) | low
        } else {
            op0.load_l(0) << (distance - 8 * op0.len())
        }
    } else {
        0
    };

    // Update the high limbs above the shift distance, i.e. those
    // which will receive non-trivial contents from the lower part.
    let distance_nlimbs = mp_ct_nlimbs((distance + 7) / 8);
    let dst_high_lshift = distance % LIMB_BITS as usize;
    let (low_src_rshift, high_src_mask) = if dst_high_lshift != 0 {
        let low_src_rshift = LIMB_BITS as usize - dst_high_lshift;
        let high_src_mask = ((1 as LimbType) << low_src_rshift) - 1;
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
        let dst_low =  src_low >> low_src_rshift;
        let dst_high = (src_high & high_src_mask) << dst_high_lshift;
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
        let dst_low =  src_low >> low_src_rshift;
        let dst_high = (src_high & high_src_mask) << dst_high_lshift;
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
    let dst_high = (src_high & high_src_mask) << dst_high_lshift;
    if dst_limb_index != op0_nlimbs - 1 {
        op0.store_l_full(dst_limb_index, dst_high);
    } else {
        debug_assert!(op0_nlimbs <= distance_nlimbs);
        op0.store_l(dst_limb_index, dst_high & op0.partial_high_mask());
    }

    // All the other lower limbs below the shift distance are to be set to zero.
    op0.zeroize_bytes_below(op0.len().min(distance / 8));

    shifted_out_low
}

#[cfg(test)]
fn test_fill_limb_with_seq(first: u8) -> LimbType {
    let mut l: LimbType = 0;
    for i in 0..LIMB_BYTES as u8 {
        l <<= 8;
        l |= (LIMB_BYTES as u8 - i - 1 + first) as LimbType;
    }
    l
}

#[cfg(test)]
fn test_mp_lshift_mp_common<T0: MPIntMutByteSlice>(op0_len: usize) {
    use super::limbs_buffer::{MPIntByteSliceCommon, MPIntByteSliceCommonPriv};

    fn test_fill_limb_with_seq_lshifted(limb_index: usize, lshift_distance: usize) -> LimbType {
        let lshift_len = (lshift_distance + 7) / 8;
        let lshift_nlimbs = mp_ct_nlimbs(lshift_len);
        let dst_high_lshift = lshift_distance % LIMB_BITS as usize;
        let low = if lshift_nlimbs <= limb_index {
            let src_low = test_fill_limb_with_seq(((limb_index - lshift_nlimbs) * LIMB_BYTES) as u8 + 1);
            if dst_high_lshift != 0 {
                src_low >> (LIMB_BITS as usize - dst_high_lshift)
            } else {
                src_low
            }
        } else {
            0
        };


        let high = if dst_high_lshift != 0 && limb_index + 1 >= lshift_nlimbs {
            let src_high = test_fill_limb_with_seq(((limb_index + 1 - lshift_nlimbs) * LIMB_BYTES) as u8 + 1);
            let src_high = src_high & ((1 << (LIMB_BITS as usize - dst_high_lshift)) - 1);
            src_high << dst_high_lshift
        } else {
            0
        };

        high | low
    }

    let op0_nlimbs = mp_ct_nlimbs(op0_len);
    for i in 0..op0_nlimbs + 3 {
        for j in 0..LIMB_BITS as usize {
            let mut op0 = vec![0u8; op0_len];
            let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
            for k in 0..op0.nlimbs() {
                let val = test_fill_limb_with_seq(1 + (k * LIMB_BYTES) as u8);
                if k != op0.nlimbs() - 1 {
                    op0.store_l_full(k, val);
                } else {
                    op0.store_l(k, val & op0.partial_high_mask());
                }
            }
            let shift_distance = i * LIMB_BITS as usize + j;
            let shifted_out = mp_lshift_mp(&mut op0, shift_distance);

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
            let expected_shifted_out = if  op0_high_nbytes != 0 {
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
                expected_shifted_out & ((1 << shift_distance) - 1)
            } else {
                expected_shifted_out
            };
            assert_eq!(shifted_out, expected_shifted_out)
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
        let shifted_out = mp_lshift_mp(&mut op0, 8 * i);
        assert_eq!(op0.load_l(0), 0);
        assert_eq!(shifted_out, 0xff << 8 * (i - 1));
    }

    let mut op0: [u8; 1] = [0xff; 1];
    let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
    let shifted_out = mp_lshift_mp(&mut op0, 8 * (LIMB_BYTES + 1));
    assert_eq!(op0.load_l(0), 0);
    assert_eq!(shifted_out, 0);

    test_mp_lshift_mp_common::<T0>(3 * LIMB_BYTES - 1);
}

#[cfg(test)]
fn test_mp_lshift_mp_with_aligned_lengths<T0: MPIntMutByteSlice>() {
    for i in 0..LIMB_BYTES + 3 {
        let mut op0: [u8; 0] = [0; 0];
        let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
        let shifted_out = mp_lshift_mp(&mut op0, 8 * i);
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

pub fn mp_rshift_mp<T0: MPIntMutByteSlice>(op0: &mut T0, distance: usize) -> LimbType{
    if op0.is_empty() || distance == 0 {
        return 0;
    }

    let op0_nlimbs = op0.nlimbs();

    // First determine what would get shifted out into the (virtual) next lower LIMB_BITS beyond
    // the zeroth limb. If the shift distance is past these virtual next lower LIMB_BITS, that would be
    // zero.
    let shifted_out_high = if distance < 8 * op0.len() + LIMB_BITS as usize {
        // If the shift distance is smaller than a limb, only the distance least significant limbs
        // of the low limb are shifted out into the virtual limb on the right.
        if distance >= LIMB_BITS as usize {
            let low_src_bit_index = distance - LIMB_BITS as usize;
            let low_src_limb_index = low_src_bit_index / LIMB_BITS as usize;
            let low_src_rshift = low_src_bit_index % LIMB_BITS as usize;
            let (dst_high_lshift, high_src_mask) = if low_src_rshift != 0 {
                let dst_high_lshift = LIMB_BITS as usize - low_src_rshift;
                let high_src_mask = ((1 as LimbType) << low_src_rshift) - 1;
                (dst_high_lshift, high_src_mask)
            } else {
                (0, 0)
            };

            let low = op0.load_l(low_src_limb_index) >> low_src_rshift;
            let high = if low_src_limb_index + 1 < op0_nlimbs {
                op0.load_l(low_src_limb_index + 1) & high_src_mask
            } else {
                0
            };
            (high << dst_high_lshift) | low
        } else {
            op0.load_l(0) << (LIMB_BITS as usize - distance)
        }
    } else {
        0
    };

    // Update the low limbs below the shift distance, i.e. those
    // which will receive non-trivial contents from the hight part.
    let low_src_distance_nlimbs = distance / LIMB_BITS as usize; // Not rounding up is on purpose.
    let low_src_rshift = distance % LIMB_BITS as usize;
    let (dst_high_lshift, high_src_mask) = if low_src_rshift != 0 {
        let dst_high_lshift = LIMB_BITS as usize - low_src_rshift;
        let high_src_mask = ((1 as LimbType) << low_src_rshift) - 1;
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
        let dst_low =  src_low >> low_src_rshift;
        let dst_high = (src_high & high_src_mask) << dst_high_lshift;
        op0.store_l_full(dst_limb_index, dst_high | dst_low);
        dst_limb_index += 1;
    }

    // The limb containing the shift distance boundary, if any, will receive bits from both, parts of the
    // most signigicant original limb and from the zero bits shifted in from the left.
    let src_low = last_src_high;
    // src_high is zero for the most significant limb which might still contain parts of
    // the shifted contents.
    let dst_low =  src_low >> low_src_rshift;
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

    // limb_index is offset by one: an index of zero is used
    // for specifiying the virtual limb rshifted into.
    fn test_fill_limb_with_seq_rshifted(limb_index: usize, rshift_distance: usize, op_len: usize) -> LimbType {
        let op_nlimbs = mp_ct_nlimbs(op_len);
        let op_partial_high_mask = if op_len % LIMB_BYTES != 0 {
            (1 << 8 * (op_len % LIMB_BYTES)) - 1
        } else {
            !0
        };

        let rshift_len = rshift_distance / 8;
        let rshift_nlimbs = rshift_len / LIMB_BYTES;
        let src_low_rshift = rshift_distance % LIMB_BITS as usize;
        let low = if rshift_nlimbs == 0 && limb_index == 0 {
            0
        } else if limb_index + rshift_nlimbs <= op_nlimbs {
            let src_low = test_fill_limb_with_seq(((limb_index + rshift_nlimbs - 1) * LIMB_BYTES) as u8 + 1);
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
            let src_high = test_fill_limb_with_seq(((limb_index + rshift_nlimbs) * LIMB_BYTES) as u8 + 1);
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

    let op0_nlimbs = mp_ct_nlimbs(op0_len);
    for i in 0..op0_nlimbs + 3 {
        for j in 0..LIMB_BITS as usize {
            let mut op0 = vec![0u8; op0_len];
            let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
            for k in 0..op0.nlimbs() {
                let val = test_fill_limb_with_seq(1 + (k * LIMB_BYTES) as u8);
                if k != op0.nlimbs() - 1 {
                    op0.store_l_full(k, val);
                } else {
                    op0.store_l(k, val & op0.partial_high_mask());
                }
            }
            let shift_distance = i * LIMB_BITS as usize + j;
            let shifted_out = mp_rshift_mp(&mut op0, shift_distance);

            for k in 0..op0.nlimbs() {
                let expected_val = test_fill_limb_with_seq_rshifted(k + 1, shift_distance, op0_len);
                assert_eq!(op0.load_l(k), expected_val);
            }

            assert_eq!(shifted_out, test_fill_limb_with_seq_rshifted(0, shift_distance, op0_len));
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
        let shifted_out = mp_rshift_mp(&mut op0, 8 * i);
        assert_eq!(op0.load_l(0), 0);
        assert_eq!(shifted_out, 0xff << 8 * (LIMB_BYTES - i));
    }

    let mut op0: [u8; 1] = [0xff; 1];
    let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
    let shifted_out = mp_rshift_mp(&mut op0, 8 * (LIMB_BYTES + 1));
    assert_eq!(op0.load_l(0), 0);
    assert_eq!(shifted_out, 0);

    test_mp_rshift_mp_common::<T0>(3 * LIMB_BYTES - 1);
}

#[cfg(test)]
fn test_mp_rshift_mp_with_aligned_lengths<T0: MPIntMutByteSlice>() {
    for i in 0..LIMB_BYTES + 3 {
        let mut op0: [u8; 0] = [0; 0];
        let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
        let shifted_out = mp_rshift_mp(&mut op0, 8 * i);
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
