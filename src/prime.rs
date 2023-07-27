use crate::euclid::mp_ct_gcd;
use crate::mp_ct_is_one_mp;

use super::limb::{LIMB_BITS, LimbChoice, ct_is_zero_l, ct_sub_l_l_b};
use super::limbs_buffer::{MPIntByteSliceCommon, MPBigEndianByteSlice, MPNativeEndianMutByteSlice, MPIntMutByteSlice, MPIntByteSlice as _, mp_ct_find_first_set_bit_mp};
use super::cmp_impl::mp_ct_lt_mp_mp;
use super::montgomery::{ct_montgomery_neg_n0_inv_mod_l, mp_ct_montgomery_mul_mod, mp_ct_montgomery_mul_mod_cond};
use super::usize_ct_cmp::{ct_eq_usize_usize, ct_lt_usize_usize};
use super::hexstr;

// Product of first 3 primes > 2, i.e. from 3 to 7 (inclusive).
// Filters ~54% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_8: [u8; 1] = hexstr::bytes_from_hexstr_cnst::<1>(
    "69"
);

// Product of first 5 primes > 2, i.e. from 3 to 13 (inclusive).
// Filters ~62% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_16: [u8; 2] = hexstr::bytes_from_hexstr_cnst::<2>(
    "3aa7"
);

// Product of first 9 primes > 2, i.e. from 3 to 29 (inclusive).
// Filters ~68% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_32: [u8; 4] = hexstr::bytes_from_hexstr_cnst::<4>(
    "c0cfd797"
);

// Product of first 15 primes > 2, i.e. from 3 to 53 (inclusive).
// Filters ~73% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_64: [u8; 8] = hexstr::bytes_from_hexstr_cnst::<8>(
    "e221f97c30e94e1d"
);

// Product of first 25 primes > 2, i.e. from 3 to 101 (inclusive).
// Filters ~76% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_128: [u8; 16] = hexstr::bytes_from_hexstr_cnst::<16>(
    "5797d47c51681549d734e4fc4c3eaf7f"
);

// Product of first 43 primes > 2, i.e. from 3 to 193 (inclusive).
// Filters ~79% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_256: [u8; 32] = hexstr::bytes_from_hexstr_cnst::<32>(
    "dbf05b6f5654b3c0f524355143958688\
     9f155887819aed2ac05b93352be98677"
);

// Product of first 74 primes > 2, i.e. from 3 to 379 (inclusive).
// Filters ~81% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_512: [u8; 64] = hexstr::bytes_from_hexstr_cnst::<64>(
    "106aa9fb7646fa6eb0813c28c5d5f09f\
     077ec3ba238bfb99c1b631a203e81187\
     233db117cbc384056ef04659a4a11de4\
     9f7ecb29bada8f980decece92e30c48f"
);

// Product of first 130 primes > 2, i.e. from 3 to 739 (inclusive).
// Filters ~83% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_1024: [u8; 128] = hexstr::bytes_from_hexstr_cnst::<128>(
    "02c85ff870f24be80f62b1ba6c20bd72\
     b837efdf121206d87db56b7d69fa4c02\
     1c107c3ca206fe8fa7080ef576effc82\
     f9b10f5750656b7794b16afd70996e91\
     aef6e0ad15e91b071ac9b24d98b233ad\
     86ee055518e58e56638ef18bac5c74cb\
     35bbb6e5dae2783dd1c0ce7dec4fc70e\
     5186d411df36368f061aa36011f30179"
);

// Product of first 232 primes > 2 i.e. from 3 to 1471 (inclusive).
// Filters ~85% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_2048: [u8; 256] = hexstr::bytes_from_hexstr_cnst::<256>(
    "2465a7bd85011e1c9e0527929fff268c\
     82ef7efa416863baa5acdb0971dba0cc\
     ac3ee4999345029f2cf810b99e406aac\
     5fce5dd69d1c717daea5d18ab913f456\
     505679bc91c57d46d9888857862b36e2\
     ede2e473c1f0ab359da25271affe15ff\
     240e299d0b04f4cd0e4d7c0e47b1a7ba\
     007de89aae848fd5bdcd7f9815564eb0\
     60ae14f19cb50c291f0bbd8ed1c4c7f8\
     fc5fba51662001939b532d92dac844a8\
     431d400c832d039f5f900b278a75219c\
     2986140c79045d7759540854c31504dc\
     56f1df5eebe7bee447658b917bf696d6\
     927f2e2428fbeb340e515cb9835d6387\
     1be8bbe09cf13445799f2e6778815157\
     1a93b4c1eee55d1b9072e0b2f5c4607f"
);

const SMALL_ODD_PRIME_PRODUCTS: [&[u8]; 9] = [
    &SMALL_ODD_PRIME_PRODUCT_8,
    &SMALL_ODD_PRIME_PRODUCT_16,
    &SMALL_ODD_PRIME_PRODUCT_32,
    &SMALL_ODD_PRIME_PRODUCT_64,
    &SMALL_ODD_PRIME_PRODUCT_128,
    &SMALL_ODD_PRIME_PRODUCT_256,
    &SMALL_ODD_PRIME_PRODUCT_512,
    &SMALL_ODD_PRIME_PRODUCT_1024,
    &SMALL_ODD_PRIME_PRODUCT_2048,
];

pub fn mp_ct_composite_test_small_prime_gcd<PT: MPIntByteSliceCommon>(p: &PT, scratch: [&mut [u8]; 2]) -> LimbChoice {
    debug_assert!(!p.is_empty());

    // The GCD runtime depends on the maximum of both operands' bit length.
    // Select the largest small prime product with length <= p.len().
    let p_len_aligned = MPNativeEndianMutByteSlice::limbs_align_len(p.len());
    let mut i = SMALL_ODD_PRIME_PRODUCTS.len();
    while i > 0 {
        i -= 1;
        if SMALL_ODD_PRIME_PRODUCTS[i].len() <= p_len_aligned {
            break;
        }
    }
    let small_prime_product = SMALL_ODD_PRIME_PRODUCTS[i];

    let [scratch0, scratch1] = scratch;
    debug_assert!(scratch0.len() >= p_len_aligned);
    let (scratch0, _) = scratch0.split_at_mut(p_len_aligned);
    debug_assert!(scratch1.len() >= p_len_aligned);
    let (scratch1, _) = scratch1.split_at_mut(p_len_aligned);

    let mut gcd = MPNativeEndianMutByteSlice::from_bytes(scratch0).unwrap();
    gcd.copy_from(&MPBigEndianByteSlice::from_bytes(small_prime_product).unwrap());
    let mut p_work_scratch = MPNativeEndianMutByteSlice::from_bytes(scratch1).unwrap();
    p_work_scratch.copy_from(p);

    mp_ct_gcd(&mut gcd, &mut p_work_scratch);
    let gcd_is_one = mp_ct_is_one_mp(&gcd);

    // The small prime products don't include a factor of two.
    // Test for it separately.
    let p_is_odd = LimbChoice::from(p.load_l(0) & 1);

    !gcd_is_one | !p_is_odd
}

#[cfg(test)]
fn test_mp_ct_composite_test_small_prime_gcd_common<PT: MPIntMutByteSlice>() {
    use super::add_impl::mp_ct_sub_mp_l;

    for i in 0..8 {
        let l = SMALL_ODD_PRIME_PRODUCTS[i].len();
        let p_len = if l != 1 {
            l + 1
        } else {
            l
        };
        let p_len = PT::limbs_align_len(p_len);

        let mut p = vec![0u8; p_len];
        let mut p = PT::from_bytes(&mut p).unwrap();
        let scratch_len = MPNativeEndianMutByteSlice::limbs_align_len(p_len);
        let mut scratch0 = vec![0u8; scratch_len];
        let mut scratch1 = vec![0u8; scratch_len];

        p.zeroize_bytes_above(0);
        let scratch = [scratch0.as_mut_slice(), scratch1.as_mut_slice()];
        assert!(mp_ct_composite_test_small_prime_gcd(&p, scratch).unwrap() != 0);

        p.store_l(0, 2);
        let scratch = [scratch0.as_mut_slice(), scratch1.as_mut_slice()];
        assert!(mp_ct_composite_test_small_prime_gcd(&p, scratch).unwrap() != 0);

        p.store_l(0, 3);
        let scratch = [scratch0.as_mut_slice(), scratch1.as_mut_slice()];
        assert!(mp_ct_composite_test_small_prime_gcd(&p, scratch).unwrap() != 0);

        let j = if i > 0 {
            i - 1
        } else {
            0
        };
        p.copy_from(&MPBigEndianByteSlice::from_bytes(SMALL_ODD_PRIME_PRODUCTS[j]).unwrap());
        let scratch = [scratch0.as_mut_slice(), scratch1.as_mut_slice()];
        assert!(mp_ct_composite_test_small_prime_gcd(&p, scratch).unwrap() != 0);
    }

    // p = 2^255 - 19.
    let p_len = 256 / 8;
    let mut p = vec![0u8; PT::limbs_align_len(p_len)];
    let mut p = PT::from_bytes(&mut p).unwrap();
    p.zeroize_bytes_above(0);
    let bit255_limb_index = (255 / LIMB_BITS) as usize;
    let bit255_pos_in_limb = 255 % LIMB_BITS;
    p.store_l(bit255_limb_index, 1 << bit255_pos_in_limb);
    mp_ct_sub_mp_l(&mut p, 19);
    let scratch_len = MPNativeEndianMutByteSlice::limbs_align_len(p_len);
    let mut scratch0 = vec![0u8; scratch_len];
    let mut scratch1 = vec![0u8; scratch_len];
    let scratch = [scratch0.as_mut_slice(), scratch1.as_mut_slice()];
    assert_eq!(mp_ct_composite_test_small_prime_gcd(&p, scratch).unwrap(), 0);
}

#[test]
fn test_mp_ct_composite_test_small_prime_gcd_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_composite_test_small_prime_gcd_common::<MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_composite_test_small_prime_gcd_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_composite_test_small_prime_gcd_common::<MPLittleEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_composite_test_small_prime_gcd_ne() {
    test_mp_ct_composite_test_small_prime_gcd_common::<MPNativeEndianMutByteSlice>()
}

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
