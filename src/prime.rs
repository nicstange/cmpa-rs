use super::limb::{LIMB_BITS, LimbChoice, ct_is_zero_l, ct_sub_l_l_b};
use super::limbs_buffer::{MPIntByteSliceCommon, MPNativeEndianMutByteSlice, MPIntMutByteSlice, mp_ct_find_first_set_bit_mp};
use super::cmp_impl::mp_ct_lt_mp_mp;
use super::montgomery::{ct_montgomery_neg_n0_inv_mod_l, mp_ct_montgomery_mul_mod, mp_ct_montgomery_mul_mod_cond};
use super::usize_ct_cmp::{ct_eq_usize_usize, ct_lt_usize_usize};

pub fn mp_ct_prime_test_miller_rabin<BT: MPIntByteSliceCommon, PT: MPIntByteSliceCommon, RXT: MPIntByteSliceCommon>(
    mg_base: &BT, p: &PT, mg_radix_mod_p: &RXT, scratch: [&mut [u8]; 2]
) -> LimbChoice {
    debug_assert!(!p.is_empty());
    debug_assert_eq!(p.load_l(0) & 1, 1);
    debug_assert!(mg_base.nlimbs() >= p.nlimbs());
    debug_assert_ne!(mp_ct_lt_mp_mp(mg_base, p).unwrap(), 0);
    debug_assert_ne!(mp_ct_lt_mp_mp(mg_radix_mod_p, p).unwrap(), 0);

    let [scratch0, scratch1] = scratch;
    let scratch_len = MPNativeEndianMutByteSlice::limbs_align_len(p.len());
    debug_assert!(scratch0.len() >= scratch_len);
    let (scratch0, _) = scratch0.split_at_mut(scratch_len);
    debug_assert!(scratch1.len() >= scratch_len);
    let (scratch1, _) = scratch1.split_at_mut(scratch_len);

    let mut p_minus_one = MPNativeEndianMutByteSlice::from_bytes(scratch0).unwrap();
    p_minus_one.copy_from(p);
    p_minus_one.store_l(0, p_minus_one.load_l(0) ^ 1); // Minus one for odd p.
    let (p_minus_one_is_nonzero, p_minus_one_first_set_bit_pos) = mp_ct_find_first_set_bit_mp(&p_minus_one);
    debug_assert!(p_minus_one_is_nonzero.unwrap() != 0);
    debug_assert!(p_minus_one_first_set_bit_pos > 0);

    let mut base_pow = MPNativeEndianMutByteSlice::from_bytes(scratch0).unwrap();
    let mut pow_scratch = MPNativeEndianMutByteSlice::from_bytes(scratch1).unwrap();
    // Initialize with a 1 in Montgomery form.
    base_pow.copy_from(mg_radix_mod_p);

    let neg_p0_inv_mod_l = ct_montgomery_neg_n0_inv_mod_l(p);

    let mut is_probable_prime = 0;
    let mut exp_bit_pos = 8 * p.len();
    while exp_bit_pos > 1 {
        exp_bit_pos -= 1;

        mp_ct_montgomery_mul_mod(&mut pow_scratch, &base_pow, &base_pow, p, neg_p0_inv_mod_l);

        // The exponent is p - 1, p is odd, so the lsb of p - 1 is zero and there's no borrow
        // into the next bit position. exp_bit_pos != 0 by the loop's condition, so
        // simply load from the oiginal p itself.
        let exp_bit_limb_index = exp_bit_pos / LIMB_BITS as usize;
        let exp_bit_pos_in_limb = exp_bit_pos % LIMB_BITS as usize;
        let exp_bit = (p.load_l(exp_bit_limb_index) >> exp_bit_pos_in_limb) & 1;
        mp_ct_montgomery_mul_mod_cond(
            &mut base_pow,
            &pow_scratch, mg_base,
            p, neg_p0_inv_mod_l,
            LimbChoice::from(exp_bit)
        );

        // Compare the current power of the base against 1 and -1 (in Montgomery form).
        let mut is_one = 1;
        let mut is_minus_one = 1;
        let mut neg_one_borrow = 0;
        for i in 0..p.nlimbs() {
            let one_val = mg_radix_mod_p.load_l(i);
            let base_pow_val = base_pow.load_l(i);
            is_one &= ct_is_zero_l(base_pow_val ^ one_val);
            let p_val = p.load_l(i);
            let minus_one_val;
            (neg_one_borrow, minus_one_val) = ct_sub_l_l_b(p_val, one_val, neg_one_borrow);
            is_minus_one &= ct_is_zero_l(base_pow_val ^ minus_one_val);
        }

        // Evaluate the Miller-Rabin conditions for the
        // exp_bit_pos == p_minus_one_first_set_bit_pos case.
        let bit_pos_eq_first_set = ct_eq_usize_usize(exp_bit_pos, p_minus_one_first_set_bit_pos);
        is_probable_prime |= bit_pos_eq_first_set.select(0, is_one | is_minus_one);

        // Evaluate the Miller-Rabin conditions for the case that
        // 0 < exp_bit_pos < p_minus_one_first_set_bit_pos.
        //
        // Implementations found in literature often break out early with a result of "composite" if
        // the intermediate power of the base, which is a square by the range of exp_bit_pos, equals
        // one at this point. This is an optimization though, because once the value is one,
        // squaring cannot ever yield -1 (in fact, nothing but 1) again. This is a basic property of
        // (pretty much by definition) a unit in a ring, independent of whether p is a prime or
        // not. This optimization is effective only if p - 1 has a significant number of trailing
        // zeroes, which, for random prime candidates is highly unlikely. So don't bother and keep
        // the algorithm constant-time even for non-primes.
        let bit_pos_below_first_set = ct_lt_usize_usize(exp_bit_pos, p_minus_one_first_set_bit_pos);
        is_probable_prime |= bit_pos_below_first_set.select(0, is_minus_one);
    }

    LimbChoice::from(is_probable_prime)
}

#[cfg(test)]
fn test_mp_ct_prime_test_miller_rabin_common<BT: MPIntMutByteSlice, PT: MPIntMutByteSlice, RXT: MPIntMutByteSlice>() {
    use super::limbs_buffer::mp_ct_nlimbs;
    use super::add_impl::mp_ct_sub_mp_l;
    use super::mul_impl::mp_ct_mul_trunc_mp_l;

    fn is_probable_prime<BT: MPIntMutByteSlice, PT: MPIntByteSliceCommon, RXT: MPIntMutByteSlice>(
        base: &BT, p: &PT
    ) -> bool {
        use super::montgomery::{
            mp_ct_montgomery_radix2_mod_n,
            mp_ct_to_montgomery_form,
            mp_ct_montgomery_redc
        };
        use super::div_impl::mp_ct_div_mp_mp;

        let mut base_mod_p = vec![0u8; base.len()];
        let mut base_mod_p = BT::from_bytes(&mut base_mod_p).unwrap();
        base_mod_p.copy_from(base);
        mp_ct_div_mp_mp::<_, _, BT>(None, &mut base_mod_p, p, None).unwrap();

        let mut mg_radix2_mod_p = vec![0u8; RXT::limbs_align_len(p.len())];
        let mut mg_radix2_mod_p = RXT::from_bytes(&mut mg_radix2_mod_p).unwrap();
        mp_ct_montgomery_radix2_mod_n(&mut mg_radix2_mod_p, p).unwrap();

        let mut mg_base_mod_p = vec![0u8; RXT::limbs_align_len(p.len())];
        let mut mg_base_mod_p = RXT::from_bytes(&mut mg_base_mod_p).unwrap();
        let neg_p0_inv_mod_l = ct_montgomery_neg_n0_inv_mod_l(p);
        mp_ct_to_montgomery_form(&mut mg_base_mod_p, &base_mod_p, p, neg_p0_inv_mod_l, &mg_radix2_mod_p);

        let mut mg_radix_mod_p = mg_radix2_mod_p;
        mp_ct_montgomery_redc(&mut mg_radix_mod_p, p, neg_p0_inv_mod_l);

        let mut scratch0 = vec![0u8; MPNativeEndianMutByteSlice::limbs_align_len(p.len())];
        let mut scratch1 = vec![0u8; MPNativeEndianMutByteSlice::limbs_align_len(p.len())];
        mp_ct_prime_test_miller_rabin(
            &mg_base_mod_p, p, &mg_radix_mod_p,
            [scratch0.as_mut_slice(), scratch1.as_mut_slice()]
        ).unwrap() != 0
    }

    // p = 2^255 - 19.
    let mut p = vec![0u8; PT::limbs_align_len(256 / 8)];
    let mut p = PT::from_bytes(&mut p).unwrap();
    p.zeroize_bytes_above(0);
    let bit255_limb_index = (255 / LIMB_BITS) as usize;
    let bit255_pos_in_limb = 255 % LIMB_BITS;
    p.store_l(bit255_limb_index, 1 << bit255_pos_in_limb);
    mp_ct_sub_mp_l(&mut p, 19);
    for i in 1..256 {
        let mut base = vec![0u8; BT::limbs_align_len(1)];
        let mut base = BT::from_bytes(&mut base).unwrap();
        base.store_l(0, i);
        assert_eq!(is_probable_prime::<_, _, RXT>(&base, &p), true);
    }
    for i in 1..256 {
        let mut base = vec![0u8; BT::limbs_align_len(p.len())];
        let mut base = BT::from_bytes(&mut base).unwrap();
        for j in 0..mp_ct_nlimbs(base.len()) {
            base.store_l(j, i);
        }
        base.store_l(0, i);
        assert_eq!(is_probable_prime::<_, _, RXT>(&base, &p), true);
    }

    // c = 251 * (2^255 - 19)
    let mut c = vec![0u8; PT::limbs_align_len((256 + 8) / 8)];
    let mut c = PT::from_bytes(&mut c).unwrap();
    c.copy_from(&p);
    mp_ct_mul_trunc_mp_l(&mut c, (256 + 8) / 8, 251);
    for i in 2..256 {
        let mut base = vec![0u8; BT::limbs_align_len(1)];
        let mut base = BT::from_bytes(&mut base).unwrap();
        base.store_l(0, i);
        assert_eq!(is_probable_prime::<_, _, RXT>(&base, &c), false);
    }
    for i in 1..256 {
        let mut base = vec![0u8; BT::limbs_align_len(p.len())];
        let mut base = BT::from_bytes(&mut base).unwrap();
        for j in 0..mp_ct_nlimbs(base.len()) {
            base.store_l(j, i);
        }
        base.store_l(0, i);
        assert_eq!(is_probable_prime::<_, _, RXT>(&base, &c), false);
    }
}

#[test]
fn test_mp_ct_prime_test_miller_rabin_be_be_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_prime_test_miller_rabin_common::<MPBigEndianMutByteSlice,
                                                MPBigEndianMutByteSlice,
                                                MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_prime_test_miller_rabin_le_le_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_prime_test_miller_rabin_common::<MPLittleEndianMutByteSlice,
                                                MPLittleEndianMutByteSlice,
                                                MPLittleEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_prime_test_miller_rabin_ne_ne_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_ct_prime_test_miller_rabin_common::<MPNativeEndianMutByteSlice,
                                                MPNativeEndianMutByteSlice,
                                                MPNativeEndianMutByteSlice>()
}
