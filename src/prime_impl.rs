// SPDX-License-Identifier: Apache-2.0
// Copyright 2023 SUSE LLC
// Author: Nicolai Stange <nstange@suse.de>

use super::cmp_impl::{ct_is_one_mp, ct_leq_mp_l, ct_lt_mp_mp};
use super::div_impl::ct_mod_mp_l;
use super::euclid_impl::ct_gcd_odd_mp_mp;
use super::hexstr;
use super::limb::{
    ct_eq_l_l, ct_geq_l_l, ct_is_zero_l, ct_lt_l_l, ct_sub_l_l_b, CtLDivisor, LimbChoice, LimbType,
    LIMB_BYTES,
};
use super::limbs_buffer::{
    ct_find_first_set_bit_mp, MpBigEndianUIntByteSlice, MpMutNativeEndianUIntLimbsSlice, MpMutUInt,
    MpMutUIntSlice, MpUIntCommon, MpUIntSlicePriv as _,
};
use super::montgomery_impl::{
    ct_montgomery_mul_mod_mp_mp, ct_montgomery_neg_n0_inv_mod_l_mp, CtMontgomeryNegN0InvModLMpError,
};
use super::usize_ct_cmp::{ct_eq_usize_usize, ct_lt_usize_usize};

// Product of first 3 primes > 2, i.e. from 3 to 7 (inclusive).
// Filters ~54% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_8: [u8; 1] = hexstr::bytes_from_hexstr_cnst::<1>("69");

// Product of first 5 primes > 2, i.e. from 3 to 13 (inclusive).
// Filters ~62% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_16: [u8; 2] = hexstr::bytes_from_hexstr_cnst::<2>("3aa7");

// Product of first 9 primes > 2, i.e. from 3 to 29 (inclusive).
// Filters ~68% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_32: [u8; 4] = hexstr::bytes_from_hexstr_cnst::<4>("c0cfd797");

// Product of first 15 primes > 2, i.e. from 3 to 53 (inclusive).
// Filters ~73% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_64: [u8; 8] = hexstr::bytes_from_hexstr_cnst::<8>("e221f97c30e94e1d");

// Product of first 25 primes > 2, i.e. from 3 to 101 (inclusive).
// Filters ~76% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_128: [u8; 16] =
    hexstr::bytes_from_hexstr_cnst::<16>("5797d47c51681549d734e4fc4c3eaf7f");

// Product of first 43 primes > 2, i.e. from 3 to 193 (inclusive).
// Filters ~79% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_256: [u8; 32] = hexstr::bytes_from_hexstr_cnst::<32>(
    "dbf05b6f5654b3c0f524355143958688\
     9f155887819aed2ac05b93352be98677",
);

// Product of first 74 primes > 2, i.e. from 3 to 379 (inclusive).
// Filters ~81% of odd prime candidates.
const SMALL_ODD_PRIME_PRODUCT_512: [u8; 64] = hexstr::bytes_from_hexstr_cnst::<64>(
    "106aa9fb7646fa6eb0813c28c5d5f09f\
     077ec3ba238bfb99c1b631a203e81187\
     233db117cbc384056ef04659a4a11de4\
     9f7ecb29bada8f980decece92e30c48f",
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
     5186d411df36368f061aa36011f30179",
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
     1a93b4c1eee55d1b9072e0b2f5c4607f",
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

#[derive(Debug)]
pub enum CtCompositeTestSmallPrimeGcdMpError {
    InsufficientScratchSpace,
}

pub fn ct_composite_test_small_prime_gcd_mp<PT: MpUIntCommon>(
    p: &PT,
    scratch: [&mut [LimbType]; 2],
) -> Result<LimbChoice, CtCompositeTestSmallPrimeGcdMpError> {
    if p.is_empty() {
        return Ok(LimbChoice::from(1));
    }

    let p_nlimbs = MpMutNativeEndianUIntLimbsSlice::nlimbs_for_len(p.len());
    for s in scratch.iter() {
        if s.len() < p_nlimbs {
            return Err(CtCompositeTestSmallPrimeGcdMpError::InsufficientScratchSpace);
        }
    }
    let p_len_aligned = p_nlimbs * LIMB_BYTES;

    // The GCD runtime depends on the maximum of both operands' bit length.
    // Select the largest small prime product with length <= p.len().
    let mut i = SMALL_ODD_PRIME_PRODUCTS.len();
    while i > 0 {
        i -= 1;
        if SMALL_ODD_PRIME_PRODUCTS[i].len() <= p_len_aligned {
            break;
        }
    }
    let small_prime_product = SMALL_ODD_PRIME_PRODUCTS[i];

    let [scratch0, scratch1] = scratch;

    let mut gcd = MpMutNativeEndianUIntLimbsSlice::from_limbs(scratch0);
    gcd.clear_bytes_above(p_len_aligned);
    let mut gcd = gcd.shrink_to(p_len_aligned);
    gcd.copy_from(&MpBigEndianUIntByteSlice::from_slice(small_prime_product).unwrap());
    let mut p_work_scratch = MpMutNativeEndianUIntLimbsSlice::from_limbs(scratch1);
    p_work_scratch.copy_from(p);
    let mut p_work_scratch = p_work_scratch.shrink_to(p_len_aligned);

    ct_gcd_odd_mp_mp(&mut gcd, &mut p_work_scratch).unwrap();
    let gcd_is_one = ct_is_one_mp(&gcd);

    // The small prime products don't include a factor of two.
    // Test for it separately.
    let p_is_odd = p.test_bit(0);

    Ok(!gcd_is_one | !p_is_odd)
}

#[cfg(test)]
fn test_ct_composite_test_small_prime_gcd_mp<PT: MpMutUIntSlice>() {
    extern crate alloc;
    use super::add_impl::ct_sub_mp_l;
    use alloc::vec;

    for i in 0..8 {
        let l = SMALL_ODD_PRIME_PRODUCTS[i].len();
        let p_len = if l != 1 { l + 1 } else { l };
        let mut p = tst_mk_mp_backing_vec!(PT, p_len);
        let mut p = PT::from_slice(&mut p).unwrap();
        let scratch_nlimbs = MpMutNativeEndianUIntLimbsSlice::nlimbs_for_len(p_len);
        let mut scratch0 = vec![0 as LimbType; scratch_nlimbs];
        let mut scratch1 = vec![0 as LimbType; scratch_nlimbs];

        p.clear_bytes_above(0);
        let scratch = [scratch0.as_mut_slice(), scratch1.as_mut_slice()];
        assert!(
            ct_composite_test_small_prime_gcd_mp(&p, scratch)
                .unwrap()
                .unwrap()
                != 0
        );

        p.store_l(0, 2);
        let scratch = [scratch0.as_mut_slice(), scratch1.as_mut_slice()];
        assert!(
            ct_composite_test_small_prime_gcd_mp(&p, scratch)
                .unwrap()
                .unwrap()
                != 0
        );

        p.store_l(0, 3);
        let scratch = [scratch0.as_mut_slice(), scratch1.as_mut_slice()];
        assert!(
            ct_composite_test_small_prime_gcd_mp(&p, scratch)
                .unwrap()
                .unwrap()
                != 0
        );

        let j = if i > 0 { i - 1 } else { 0 };
        p.copy_from(&MpBigEndianUIntByteSlice::from_slice(SMALL_ODD_PRIME_PRODUCTS[j]).unwrap());
        let scratch = [scratch0.as_mut_slice(), scratch1.as_mut_slice()];
        assert!(
            ct_composite_test_small_prime_gcd_mp(&p, scratch)
                .unwrap()
                .unwrap()
                != 0
        );
    }

    // p = 2^255 - 19.
    let p_len = 256 / 8;
    let mut p = tst_mk_mp_backing_vec!(PT, p_len);
    let mut p = PT::from_slice(&mut p).unwrap();
    p.set_bit_to(255, true);
    ct_sub_mp_l(&mut p, 19);
    let scratch_nlimbs = MpMutNativeEndianUIntLimbsSlice::nlimbs_for_len(p_len);
    let mut scratch0 = vec![0 as LimbType; scratch_nlimbs];
    let mut scratch1 = vec![0 as LimbType; scratch_nlimbs];
    let scratch = [scratch0.as_mut_slice(), scratch1.as_mut_slice()];
    assert_eq!(
        ct_composite_test_small_prime_gcd_mp(&p, scratch)
            .unwrap()
            .unwrap(),
        0
    );
}

#[test]
fn test_ct_composite_test_small_prime_gcd_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_composite_test_small_prime_gcd_mp::<MpMutBigEndianUIntByteSlice>()
}

#[test]
fn test_ct_composite_test_small_prime_gcd_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_composite_test_small_prime_gcd_mp::<MpMutLittleEndianUIntByteSlice>()
}

#[test]
fn test_ct_composite_test_small_prime_gcd_ne() {
    test_ct_composite_test_small_prime_gcd_mp::<MpMutNativeEndianUIntLimbsSlice>()
}

#[derive(Debug)]
pub enum CtPrimeTestMillerRabinMpError {
    InvalidCandidate,
    InsufficientScratchSpace,
    InconsistentBaseLength,
    InconsistentMgRadixModPLength,
}

pub fn ct_prime_test_miller_rabin_mp<BT: MpUIntCommon, PT: MpUIntCommon, RXT: MpUIntCommon>(
    mg_base: &BT,
    p: &PT,
    mg_radix_mod_p: &RXT,
    scratch: [&mut [LimbType]; 2],
) -> Result<LimbChoice, CtPrimeTestMillerRabinMpError> {
    if p.test_bit(0).unwrap() == 0 {
        return Err(CtPrimeTestMillerRabinMpError::InvalidCandidate);
    }
    if !mg_base.len_is_compatible_with(p.len()) {
        return Err(CtPrimeTestMillerRabinMpError::InconsistentBaseLength);
    }
    debug_assert_ne!(ct_lt_mp_mp(mg_base, p).unwrap(), 0);
    if !mg_radix_mod_p.len_is_compatible_with(p.len()) {
        return Err(CtPrimeTestMillerRabinMpError::InconsistentMgRadixModPLength);
    }
    debug_assert_ne!(ct_lt_mp_mp(mg_radix_mod_p, p).unwrap(), 0);

    let p_nlimbs = MpMutNativeEndianUIntLimbsSlice::nlimbs_for_len(p.len());
    for s in scratch.iter() {
        if s.len() < p_nlimbs {
            return Err(CtPrimeTestMillerRabinMpError::InsufficientScratchSpace);
        }
    }
    let [scratch0, scratch1] = scratch;

    let mut p_minus_one = MpMutNativeEndianUIntLimbsSlice::from_limbs(scratch0);
    p_minus_one.copy_from(p);
    let mut p_minus_one = p_minus_one.shrink_to(p.len());
    p_minus_one.store_l(0, p_minus_one.load_l(0) ^ 1); // Minus one for odd p.
    let (p_minus_one_is_nonzero, p_minus_one_first_set_bit_pos) =
        ct_find_first_set_bit_mp(&p_minus_one);
    debug_assert!(p_minus_one_is_nonzero.unwrap() != 0);
    debug_assert!(p_minus_one_first_set_bit_pos > 0);

    let mut base_pow = MpMutNativeEndianUIntLimbsSlice::from_limbs(scratch0);
    // Initialize with a 1 in Montgomery form.
    base_pow.copy_from(mg_radix_mod_p);
    let mut base_pow = base_pow.shrink_to(p.len());
    let mut pow_scratch = MpMutNativeEndianUIntLimbsSlice::from_limbs(scratch1);
    pow_scratch.clear_bytes_above(p.len());
    let mut pow_scratch = pow_scratch.shrink_to(p.len());

    let neg_p0_inv_mod_l = ct_montgomery_neg_n0_inv_mod_l_mp(p).map_err(|e| match e {
        CtMontgomeryNegN0InvModLMpError::InvalidModulus => {
            // Should not be possible, p had been validated for being an odd number at the
            // beginning, but play safe.
            debug_assert!(false);
            CtPrimeTestMillerRabinMpError::InvalidCandidate
        }
    })?;

    let mut is_probable_prime = 0;
    let mut exp_bit_pos = 8 * p.len();
    while exp_bit_pos > 1 {
        exp_bit_pos -= 1;
        ct_montgomery_mul_mod_mp_mp(&mut pow_scratch, &base_pow, &base_pow, p, neg_p0_inv_mod_l)
            .unwrap();
        ct_montgomery_mul_mod_mp_mp(&mut base_pow, &pow_scratch, mg_base, p, neg_p0_inv_mod_l)
            .unwrap();
        // If the current exponent bit is zero, "undo" the latter multiplication.
        // The exponent is p - 1, p is odd, so the lsb of p - 1 is zero and there's no
        // borrow into the next bit position. exp_bit_pos != 0 by the loop's
        // condition, so simply load from the oiginal p itself.
        base_pow.copy_from_cond(&pow_scratch, !p.test_bit(exp_bit_pos));

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
        // Implementations found in literature often break out early with a result of
        // "composite" if the intermediate power of the base, which is a square
        // by the range of exp_bit_pos, equals one at this point. This is an
        // optimization though, because once the value is one, squaring cannot
        // ever yield -1 (in fact, nothing but 1) again. This is a basic property of
        // (pretty much by definition) a unit in a ring, independent of whether p is a
        // prime or not. This optimization is effective only if p - 1 has a
        // significant number of trailing zeroes, which, for random prime
        // candidates is highly unlikely. So don't bother and keep the algorithm
        // constant-time even for non-primes.
        let bit_pos_below_first_set = ct_lt_usize_usize(exp_bit_pos, p_minus_one_first_set_bit_pos);
        is_probable_prime |= bit_pos_below_first_set.select(0, is_minus_one);
    }

    Ok(LimbChoice::from(is_probable_prime))
}

#[cfg(test)]
fn test_ct_prime_test_miller_rabin_mp<
    BT: MpMutUIntSlice,
    PT: MpMutUIntSlice,
    RXT: MpMutUIntSlice,
>() {
    extern crate alloc;
    use super::add_impl::ct_sub_mp_l;
    use super::limbs_buffer::ct_mp_nlimbs;
    use super::mul_impl::ct_mul_trunc_mp_l;
    use alloc::vec;

    fn is_probable_prime<BT: MpMutUIntSlice, PT: MpUIntCommon, RXT: MpMutUIntSlice>(
        base: &BT,
        p: &PT,
    ) -> bool {
        use super::div_impl::{ct_mod_mp_mp, CtMpDivisor};
        use super::montgomery_impl::{
            ct_montgomery_radix2_mod_n_mp, ct_montgomery_redc_mp, ct_to_montgomery_form_mp,
        };

        let mut base_mod_p = tst_mk_mp_backing_vec!(BT, base.len());
        let mut base_mod_p = BT::from_slice(&mut base_mod_p).unwrap();
        base_mod_p.copy_from(base);
        ct_mod_mp_mp(None, &mut base_mod_p, &CtMpDivisor::new(p, None).unwrap());

        let mut mg_radix2_mod_p = tst_mk_mp_backing_vec!(RXT, p.len());
        let mut mg_radix2_mod_p = RXT::from_slice(&mut mg_radix2_mod_p).unwrap();
        ct_montgomery_radix2_mod_n_mp(&mut mg_radix2_mod_p, p).unwrap();

        let mut mg_base_mod_p = tst_mk_mp_backing_vec!(RXT, p.len());
        let mut mg_base_mod_p = RXT::from_slice(&mut mg_base_mod_p).unwrap();
        let neg_p0_inv_mod_l = ct_montgomery_neg_n0_inv_mod_l_mp(p).unwrap();
        ct_to_montgomery_form_mp(
            &mut mg_base_mod_p,
            &base_mod_p,
            p,
            neg_p0_inv_mod_l,
            &mg_radix2_mod_p,
        )
        .unwrap();

        let mut mg_radix_mod_p = mg_radix2_mod_p;
        ct_montgomery_redc_mp(&mut mg_radix_mod_p, p, neg_p0_inv_mod_l).unwrap();

        let mut scratch0 =
            vec![0 as LimbType; MpMutNativeEndianUIntLimbsSlice::nlimbs_for_len(p.len())];
        let mut scratch1 =
            vec![0 as LimbType; MpMutNativeEndianUIntLimbsSlice::nlimbs_for_len(p.len())];
        ct_prime_test_miller_rabin_mp(
            &mg_base_mod_p,
            p,
            &mg_radix_mod_p,
            [scratch0.as_mut_slice(), scratch1.as_mut_slice()],
        )
        .unwrap()
        .unwrap()
            != 0
    }

    // p = 2^255 - 19.
    let mut p = tst_mk_mp_backing_vec!(PT, 256 / 8);
    let mut p = PT::from_slice(&mut p).unwrap();
    p.set_bit_to(255, true);
    ct_sub_mp_l(&mut p, 19);
    for i in 1..256 {
        let mut base = tst_mk_mp_backing_vec!(BT, 1);
        let mut base = BT::from_slice(&mut base).unwrap();
        base.store_l(0, i);
        assert_eq!(is_probable_prime::<_, _, RXT>(&base, &p), true);
    }
    for i in 1..256 {
        let mut base = tst_mk_mp_backing_vec!(BT, p.len());
        let mut base = BT::from_slice(&mut base).unwrap();
        for j in 0..ct_mp_nlimbs(base.len()) {
            base.store_l(j, i);
        }
        base.store_l(0, i);
        assert_eq!(is_probable_prime::<_, _, RXT>(&base, &p), true);
    }

    // c = 251 * (2^255 - 19)
    let mut c = tst_mk_mp_backing_vec!(PT, (256 + 8) / 8);
    let mut c = PT::from_slice(&mut c).unwrap();
    c.copy_from(&p);
    ct_mul_trunc_mp_l(&mut c, (256 + 8) / 8, 251);
    for i in 2..256 {
        let mut base = tst_mk_mp_backing_vec!(BT, 1);
        let mut base = BT::from_slice(&mut base).unwrap();
        base.store_l(0, i);
        assert_eq!(is_probable_prime::<_, _, RXT>(&base, &c), false);
    }
    for i in 1..256 {
        let mut base = tst_mk_mp_backing_vec!(BT, p.len());
        let mut base = BT::from_slice(&mut base).unwrap();
        for j in 0..ct_mp_nlimbs(base.len()) {
            base.store_l(j, i);
        }
        base.store_l(0, i);
        assert_eq!(is_probable_prime::<_, _, RXT>(&base, &c), false);
    }
}

#[test]
fn test_ct_prime_test_miller_rabin_be_be_be() {
    use super::limbs_buffer::MpMutBigEndianUIntByteSlice;
    test_ct_prime_test_miller_rabin_mp::<
        MpMutBigEndianUIntByteSlice,
        MpMutBigEndianUIntByteSlice,
        MpMutBigEndianUIntByteSlice,
    >()
}

#[test]
fn test_ct_prime_test_miller_rabin_le_le_le() {
    use super::limbs_buffer::MpMutLittleEndianUIntByteSlice;
    test_ct_prime_test_miller_rabin_mp::<
        MpMutLittleEndianUIntByteSlice,
        MpMutLittleEndianUIntByteSlice,
        MpMutLittleEndianUIntByteSlice,
    >()
}

#[test]
fn test_ct_prime_test_miller_rabin_ne_ne_ne() {
    use super::limbs_buffer::MpMutNativeEndianUIntLimbsSlice;
    test_ct_prime_test_miller_rabin_mp::<
        MpMutNativeEndianUIntLimbsSlice,
        MpMutNativeEndianUIntLimbsSlice,
        MpMutNativeEndianUIntLimbsSlice,
    >()
}

// Prime candidate generation. The general method follows the "wheel sieve"
// approach: start out from a multiple of a primorial (a product of the first
// few N primes) as a base and add offsets not divisible by any of the factors
// to it. The larger the primorial, the better in terms of preselection
// effectiveness, because all potential candidates divisible by any of its
// factors are filtered. On the other hand, the list of offsets (not divisible
// by any of the primorial's factors) grows ~linearly with the primorial's
// magnitude and for practical reasons, we cannot store a number of precomputed
// offsets proportional to, say 2^64. Choose a two-level approach instead. At
// level 0, there will be a generator using a very small primorial,
// 2 * 3 * 5 == 30, whose sole purpose is to generate offset candidates for the
// "real", next level1 primorial, which will be chosen such that it still fits a
// limb.

const FIRST_PRIMES: [u8; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];

fn ct_find_first_ge_than(v: &[u8], val: u8) -> usize {
    debug_assert!(*v.iter().last().unwrap() >= val);

    // Binary search. Be careful to always keep the left and right
    // halves equal in length for constant-time.
    let mut lb = 0;
    let mut ub = v.len();
    let mut r = ub - lb;
    while r > 1 {
        // m points past the mid element if r is odd, inbetween the two
        // mid elements otherwise.
        let m = lb + (r + 1) / 2;
        let pivot_is_lt = ct_lt_l_l(v[m - 1] as LimbType, val as LimbType);
        // Keep the halves balanced for constant-time. If r is odd and
        // the right half is taken, extend it at its front by the pivot
        // element, even though it's not needed for the correctness of
        // the search.
        let r_odd = r & 1;
        debug_assert_eq!(ub - m + r_odd, m - lb);
        lb = pivot_is_lt.select_usize(lb, m - r_odd);
        ub = pivot_is_lt.select_usize(m, ub);
        r = ub - lb;
    }

    lb
}

#[test]
fn test_ct_find_first_ge_than() {
    for val in FIRST_PRIMES.iter() {
        let i = ct_find_first_ge_than(&FIRST_PRIMES, *val);
        assert_eq!(FIRST_PRIMES[i], *val);
    }

    for i in 0..FIRST_PRIMES.len() - 1 {
        let val = FIRST_PRIMES[i] + 1;
        let j = ct_find_first_ge_than(&FIRST_PRIMES, val);
        assert_eq!(i + 1, j);
    }
}

// Non-constant-time but constant-context variant of the ct_find_first_ge_than()
// binary search.
const fn cnst_find_first_ge_than(v: &[u8], val: u8) -> usize {
    debug_assert!(!v.is_empty());
    debug_assert!(v[v.len() - 1] >= val);

    // Binary search.
    let mut lb = 0;
    let mut ub = v.len();
    let mut r = ub - lb;
    while r > 1 {
        // m points past the mid element if r is odd, inbetween the two
        // mid elements otherwise.
        let m = lb + (r + 1) / 2;
        if v[m - 1] < val {
            lb = m;
        } else {
            ub = m;
        }
        r = ub - lb;
    }

    lb
}

#[test]
fn test_cnst_find_first_ge_than() {
    for val in FIRST_PRIMES.iter() {
        let i = cnst_find_first_ge_than(&FIRST_PRIMES, *val);
        assert_eq!(FIRST_PRIMES[i], *val);
    }

    for i in 0..FIRST_PRIMES.len() - 1 {
        let val = FIRST_PRIMES[i] + 1;
        let j = cnst_find_first_ge_than(&FIRST_PRIMES, val);
        assert_eq!(i + 1, j);
    }
}

// Maximum number of primorial factors such that the result is <= max.
const fn first_primes_primorial_nfactors_for_max(max: u64) -> usize {
    let mut prod: u64 = 1;
    let mut n = 0;
    while n < FIRST_PRIMES.len() {
        let next_p = FIRST_PRIMES[n] as u64;
        if max / next_p < prod {
            break;
        }
        prod *= next_p;
        n += 1;
    }

    n
}

const fn first_primes_primorial(n: usize) -> LimbType {
    debug_assert!(n <= first_primes_primorial_nfactors_for_max(u64::MAX));
    let mut prod: LimbType = 1;
    let mut i = 0;
    while i < n {
        prod *= FIRST_PRIMES[i] as LimbType;
        i += 1;
    }
    prod
}

struct PrimeWheelSieveLvl0 {
    next_index: usize,
    last_offset: u8,
    next_offset_delta: u8,
}

impl PrimeWheelSieveLvl0 {
    // The primorial of the first 4 primes would still fit an u8. However, by
    // sticking to the first three factors only, the list of wheel offsets is
    // _much_ smaller.
    const PRIMORIAL_NFACTORS: usize = 3;
    const PRIMORIAL: u8 = first_primes_primorial(Self::PRIMORIAL_NFACTORS) as u8;

    const fn assert_all_offsets_are_first_primes() {
        // The next prime not a factor of the level 0 primorial squared is >= the
        // primorial. It follows that all numbers less than the primorial have a
        // square root < that next prime. It follows in turn that any composite
        // number < the primorial has factors only < that next prime, i.e.
        // exactly the factors of the primorial. For the wheel sieve in general, the set
        // of offsets is defined to be the set of numbers < the primorial, that have no
        // factor in common with it. By the preceeding, these cannot be
        // composite, i.e. must be prime.
        assert!(
            (FIRST_PRIMES[Self::PRIMORIAL_NFACTORS] as u64
                * FIRST_PRIMES[Self::PRIMORIAL_NFACTORS] as u64)
                > Self::PRIMORIAL as u64
        );
    }

    const NOFFSETS: usize = {
        Self::assert_all_offsets_are_first_primes();
        cnst_find_first_ge_than(&FIRST_PRIMES, Self::PRIMORIAL) - Self::PRIMORIAL_NFACTORS + 1
    };

    const OFFSETS: [u8; Self::NOFFSETS] = [1, 7, 11, 13, 17, 19, 23, 29];
    const MAX_OFFSET: u8 = Self::OFFSETS[Self::NOFFSETS - 1];

    const fn offset_delta_at_wrap() -> u8 {
        Self::PRIMORIAL - Self::MAX_OFFSET
    }

    fn start_geq_than(lower_bound: LimbType) -> Self {
        // Take the lower bound modulo the primorial to get a lower bound
        // on the first offset.
        let offset_lower_bound = (lower_bound % Self::PRIMORIAL as LimbType) as u8;

        // Lookup the first wheel offset >= the lower bound.
        // The wheel's last offset is always equal to the primorial minus 1,
        // so the search is well-defined.
        debug_assert_eq!(Self::OFFSETS[Self::NOFFSETS - 1], Self::PRIMORIAL - 1);
        let offset_index = ct_find_first_ge_than(&Self::OFFSETS, offset_lower_bound);
        // By initializing last_offset to offset_lower_bound, the first produced
        // delta will advance the input lower_bound to the first offset.
        Self {
            next_index: offset_index,
            last_offset: offset_lower_bound,
            next_offset_delta: 0,
        }
    }

    fn produce_next_delta(&mut self) -> u8 {
        let next_offset = Self::OFFSETS[self.next_index];
        let delta = next_offset + self.next_offset_delta - self.last_offset;
        self.advance();
        delta
    }

    fn advance(&mut self) {
        let last_index = self.next_index;
        self.next_index += 1;
        let wrapped = ct_eq_usize_usize(self.next_index, Self::NOFFSETS);
        self.next_index = wrapped.select_usize(self.next_index, 0);
        self.last_offset = wrapped.select(Self::OFFSETS[last_index] as LimbType, 0) as u8;
        self.next_offset_delta = wrapped.select(0, Self::offset_delta_at_wrap() as LimbType) as u8;
    }
}

#[test]
fn test_prime_wheel_sieve_lvl0() {
    fn advance_candidate(candidate: &mut LimbType, wheel: &mut PrimeWheelSieveLvl0) {
        let candidate_is_zero = *candidate == 0;
        let delta = wheel.produce_next_delta();
        for j in 0..delta {
            *candidate += 1;
            if candidate_is_zero && j == 0 {
                continue;
            }
            let mut is_not_coprime = false;
            for k in 0..PrimeWheelSieveLvl0::PRIMORIAL_NFACTORS {
                let factor = FIRST_PRIMES[k] as LimbType;
                let rem = *candidate % factor;
                if rem == 0 {
                    is_not_coprime = true;
                    break;
                }
            }
            if j + 1 != delta {
                assert_eq!(is_not_coprime, true);
            } else {
                assert_eq!(is_not_coprime, false);
            }
        }
    }

    let mut candidate = 0;
    let mut wheel = PrimeWheelSieveLvl0::start_geq_than(candidate);
    advance_candidate(&mut candidate, &mut wheel);
    assert_eq!(candidate, 1);
    for _i in 0..1024 {
        advance_candidate(&mut candidate, &mut wheel);
    }

    let mut candidate = 1;
    let mut wheel = PrimeWheelSieveLvl0::start_geq_than(candidate);
    advance_candidate(&mut candidate, &mut wheel);
    assert_eq!(candidate, 1);
    for _i in 0..1024 {
        advance_candidate(&mut candidate, &mut wheel);
    }

    let mut candidate = 2;
    let mut wheel = PrimeWheelSieveLvl0::start_geq_than(candidate);
    advance_candidate(&mut candidate, &mut wheel);
    assert_eq!(candidate, 7);
    for _i in 0..1024 {
        advance_candidate(&mut candidate, &mut wheel);
    }

    let mut candidate = PrimeWheelSieveLvl0::PRIMORIAL as LimbType - 2;
    let mut wheel = PrimeWheelSieveLvl0::start_geq_than(candidate);
    advance_candidate(&mut candidate, &mut wheel);
    assert_eq!(candidate, PrimeWheelSieveLvl0::PRIMORIAL as LimbType - 1);
    for _i in 0..1024 {
        advance_candidate(&mut candidate, &mut wheel);
    }

    let mut candidate = PrimeWheelSieveLvl0::PRIMORIAL as LimbType - 1;
    let mut wheel = PrimeWheelSieveLvl0::start_geq_than(candidate);
    advance_candidate(&mut candidate, &mut wheel);
    assert_eq!(candidate, PrimeWheelSieveLvl0::PRIMORIAL as LimbType - 1);
    for _i in 0..1024 {
        advance_candidate(&mut candidate, &mut wheel);
    }

    let mut candidate = PrimeWheelSieveLvl0::PRIMORIAL as LimbType;
    let mut wheel = PrimeWheelSieveLvl0::start_geq_than(candidate);
    advance_candidate(&mut candidate, &mut wheel);
    assert_eq!(candidate, PrimeWheelSieveLvl0::PRIMORIAL as LimbType + 1);
    for _i in 0..1024 {
        advance_candidate(&mut candidate, &mut wheel);
    }
}

pub struct PrimeWheelSieveLvl1 {
    lvl0_wheel: PrimeWheelSieveLvl0,
    last_offset: LimbType,
    last_offset_delta: u8,
}

impl PrimeWheelSieveLvl1 {
    const PRIMORIAL_NFACTORS: usize = first_primes_primorial_nfactors_for_max(!(0 as LimbType));
    const PRIMORIAL: LimbType = first_primes_primorial(Self::PRIMORIAL_NFACTORS);

    pub fn start_geq_than<LBT: MpUIntCommon>(lower_bound: &LBT) -> Self {
        let offset_lower_bound = ct_mod_mp_l(
            lower_bound,
            &CtLDivisor::nonct_new(Self::PRIMORIAL).unwrap(),
        );
        // If the lower bound is <= 1, skip the 1 generated by the level 0.
        // It only got the potential for causing confusion or subtle corner cases.
        // It's important to account for that internal increment when producing
        // the first offset delta to the outside, so remember it in last_offset_delta.
        let lower_bound_leq_one = ct_leq_mp_l(lower_bound, 1);
        let last_offset_delta =
            lower_bound_leq_one.select(0, (2 as LimbType).wrapping_sub(offset_lower_bound));
        let offset_lower_bound = offset_lower_bound + last_offset_delta;
        // Remember that increment and adjust for it when emitting the first produced
        // offset delta.
        Self {
            lvl0_wheel: PrimeWheelSieveLvl0::start_geq_than(offset_lower_bound),
            last_offset: offset_lower_bound,
            last_offset_delta: last_offset_delta as u8,
        }
    }

    pub fn produce_next_delta(&mut self) -> LimbType {
        let mut next_offset = self.last_offset;
        let mut next_offset_delta;
        let mut last_offset = self.last_offset;
        loop {
            let lvl0_delta = self.lvl0_wheel.produce_next_delta();
            // The addition does not wrap: next_offset < Self::PRIMORIAL and Self::PRIMORIAL
            // has been chosen such that there's enough room until LimbType::MAX
            // to accomodate the largest delta value the level 0 wheel could
            // ever produce.
            next_offset += lvl0_delta as LimbType;
            let wrapped = ct_geq_l_l(next_offset, Self::PRIMORIAL);
            next_offset = wrapped.select(next_offset, next_offset.wrapping_sub(Self::PRIMORIAL));
            last_offset = wrapped.select(last_offset, 0);
            next_offset_delta = wrapped.select(0, Self::PRIMORIAL - self.last_offset);

            let mut is_not_coprime = LimbChoice::from(0);
            for factor in FIRST_PRIMES
                .iter()
                .take(Self::PRIMORIAL_NFACTORS)
                .skip(PrimeWheelSieveLvl0::PRIMORIAL_NFACTORS)
            {
                let rem = next_offset % *factor as LimbType;
                is_not_coprime |= ct_eq_l_l(rem, 0);
            }
            if is_not_coprime.unwrap() != 0 {
                continue;
            }
            break;
        }

        let last_offset_delta = self.last_offset_delta;
        self.last_offset_delta = 0;
        self.last_offset = next_offset;
        next_offset + next_offset_delta - (last_offset - last_offset_delta as LimbType)
    }
}

#[test]
fn test_prime_wheel_sieve_lvl1() {
    use super::add_impl::ct_add_mp_l;
    use super::cmp_impl::ct_is_zero_mp;

    fn advance_candidate(
        candidate: &mut MpMutNativeEndianUIntLimbsSlice,
        wheel: &mut PrimeWheelSieveLvl1,
    ) {
        let candidate_is_zero = ct_is_zero_mp(candidate).unwrap() != 0;
        let delta = wheel.produce_next_delta();
        for j in 0..delta {
            ct_add_mp_l(candidate, 1);
            if candidate_is_zero && j == 0 {
                continue;
            }
            let mut is_not_coprime = false;
            for k in 0..PrimeWheelSieveLvl1::PRIMORIAL_NFACTORS {
                let factor = FIRST_PRIMES[k] as LimbType;
                let factor = CtLDivisor::nonct_new(factor).unwrap();
                let rem = ct_mod_mp_l(candidate, &factor);
                if rem == 0 {
                    is_not_coprime = true;
                    break;
                }
            }
            if j + 1 != delta {
                assert_eq!(is_not_coprime, true);
            } else {
                assert_eq!(is_not_coprime, false);
            }
        }
    }

    let mut candidate = [0 as LimbType; 2];
    let mut candidate = MpMutNativeEndianUIntLimbsSlice::from_limbs(&mut candidate);
    let mut wheel = PrimeWheelSieveLvl1::start_geq_than(&candidate);
    for _i in 0..1024 {
        advance_candidate(&mut candidate, &mut wheel);
    }

    let mut candidate = [0 as LimbType; 2];
    let mut candidate = MpMutNativeEndianUIntLimbsSlice::from_limbs(&mut candidate);
    candidate.store_l(0, 1);
    let mut wheel = PrimeWheelSieveLvl1::start_geq_than(&candidate);
    for _i in 0..1024 {
        advance_candidate(&mut candidate, &mut wheel);
    }

    let mut candidate = [0 as LimbType; 2];
    let mut candidate = MpMutNativeEndianUIntLimbsSlice::from_limbs(&mut candidate);
    candidate.store_l(0, 2);
    let mut wheel = PrimeWheelSieveLvl1::start_geq_than(&candidate);
    for _i in 0..1024 {
        advance_candidate(&mut candidate, &mut wheel);
    }

    let mut candidate = [0 as LimbType; 2];
    let mut candidate = MpMutNativeEndianUIntLimbsSlice::from_limbs(&mut candidate);
    candidate.store_l(0, PrimeWheelSieveLvl1::PRIMORIAL - 2);
    let mut wheel = PrimeWheelSieveLvl1::start_geq_than(&candidate);
    advance_candidate(&mut candidate, &mut wheel);
    assert_eq!(candidate.load_l(0), PrimeWheelSieveLvl1::PRIMORIAL - 1);
    for _i in 0..1024 {
        advance_candidate(&mut candidate, &mut wheel);
    }
}

pub use PrimeWheelSieveLvl1 as PrimeWheelSieve;
