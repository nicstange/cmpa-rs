use crate::div_impl::mp_ct_div_lshifted_mp_mp;

use super::limb::{LimbType, LIMB_BITS, ct_add_l_l, ct_mul_add_l_l_l_c, LIMB_BYTES, LimbChoice, ct_inv_mod_l, ct_lsb_mask_l};
use super::limbs_buffer::{MPIntMutByteSlice, MPIntMutByteSlicePriv as _, MPIntByteSliceCommon, mp_ct_limbs_align_len, mp_ct_nlimbs};
use super::cmp_impl::mp_ct_geq_mp_mp;
use super::add_impl::mp_ct_sub_cond_mp_mp;
use super::div_impl::{mp_ct_div_mp_mp, mp_ct_div_pow2_mp, MpCtDivisionError};

pub fn mp_ct_montgomery_radix_shift_len(n_len: usize) -> usize {
    mp_ct_limbs_align_len(n_len)
}

pub fn mp_ct_montgomery_radix_shift_nlimbs(n_len: usize) -> usize {
    mp_ct_nlimbs(n_len)
}

pub fn ct_montgomery_neg_n0_inv_mod_l<'a, NT: MPIntByteSliceCommon>(n: &NT) -> LimbType {
    debug_assert!(!n.is_empty());
    let n0 = n.load_l(0);
    let n0_inv_mod_l = ct_inv_mod_l(n0);
    (!n0_inv_mod_l).wrapping_add(1)
}

#[test]
fn test_mp_ct_montgomery_neg_n0_inv_mod_l() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;

    for n0 in 0 as LimbType..128 {
        let n0 = 2 * n0 + 1;
        for j in 0..2048 {
            const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
            let v = MERSENNE_PRIME_13.wrapping_mul((511 as LimbType).wrapping_mul(j));
            let v = v << 8;
            let n0 = n0.wrapping_add(v);

            let mut n0_buf: [u8; LIMB_BYTES] = [0; LIMB_BYTES];
            let mut n = MPBigEndianMutByteSlice::from_bytes(n0_buf.as_mut_slice()).unwrap();
            n.store_l(0, n0);
            let neg_n0_inv = ct_montgomery_neg_n0_inv_mod_l(&n);
            assert_eq!(n0.wrapping_mul(neg_n0_inv), !0);
        }
    }
}

pub struct MpCtMontgomeryRedcKernel {
    redc_pow2_rshift: u32,
    redc_pow2_mask: LimbType,
    m: LimbType,
    last_redced_val: LimbType,
    carry: LimbType,
}

impl MpCtMontgomeryRedcKernel {
    const fn redc_rshift_lo(redc_pow2_rshift: u32, redc_pow2_mask: LimbType, val: LimbType) -> LimbType {
        // There are two possible cases where redc_pow2_rshift == 0:
        // a.) redc_pow2_exp == 0. In this case !redc_pow2_mask == !0.
        // b.) redc_pow2_exp == LIMB_BITS. In this case !redc_pow2_mask == !!0 == 0.
        (val & !redc_pow2_mask) >> redc_pow2_rshift
    }

    const fn redc_lshift_hi(redc_pow2_rshift: u32, redc_pow2_mask: LimbType, val: LimbType) -> LimbType {
        // There are two possible cases where redc_pow2_rshift == 0:
        // a.) redc_pow2_exp == 0. In this case redc_pow2_mask == 0 as well.
        // b.) redc_pow2_exp == LIMB_BITS. In this case redc_pow2_mask == !0.
        (val & redc_pow2_mask) << (LIMB_BITS - redc_pow2_rshift) % LIMB_BITS
    }

    pub fn start(redc_pow2_exp: u32, t0_val: LimbType, n0_val: LimbType,
                 neg_n0_inv_mod_l: LimbType) -> Self {
        debug_assert!(redc_pow2_exp <= LIMB_BITS);

        // Calculate the shift distance for right shifting a redced limb into its final position.
        // Be careful to keep it < LIMB_BITS for not running into undefined behaviour with the
        // shift.  See the comments in redc_rshift_lo()/redc_rshift_hi() for the mask<->rshift
        // interaction.
        let redc_pow2_rshift = redc_pow2_exp % LIMB_BITS;
        let redc_pow2_mask = ct_lsb_mask_l(redc_pow2_exp as u32);

        // For any i >= j, if n' == -n^{-1} mod 2^i, then n' mod 2^j == -n^{-1} mod 2^j.
        let neg_n0_inv_mod_l = neg_n0_inv_mod_l & redc_pow2_mask;
        debug_assert_eq!(neg_n0_inv_mod_l.wrapping_mul(n0_val) & redc_pow2_mask, redc_pow2_mask);

        let m = t0_val.wrapping_mul(neg_n0_inv_mod_l) & redc_pow2_mask;

        let (carry, redced_t0_val) = ct_mul_add_l_l_l_c(t0_val, m, n0_val, 0);
        debug_assert_eq!(redced_t0_val & redc_pow2_mask, 0);
        // If redc_pow2_exp < LIMB_BITS, the upper bits of the reduced zeroth limb
        // will become the lower bits of the resulting zeroth limb.
        let last_redced_val = Self::redc_rshift_lo(redc_pow2_rshift, redc_pow2_mask, redced_t0_val);

        Self { redc_pow2_rshift, redc_pow2_mask, m, last_redced_val, carry }
    }

    pub fn update(&mut self, t_val: LimbType, n_val: LimbType) -> LimbType {
        let redced_t_val;
        (self.carry, redced_t_val) = ct_mul_add_l_l_l_c(t_val, self.m, n_val, self.carry);

        // If redc_pow2_exp < LIMB_BITS, the lower bits of the reduced current limb correspond to
        // the upper bits of the returned result limb.
        let result_val = self.last_redced_val
            | Self::redc_lshift_hi(
                self.redc_pow2_rshift,
                self.redc_pow2_mask,
                redced_t_val
            );
        // If redc_pow2_exp < LIMB_BITS, the upper bits of the reduced current limb
        // will become the lower bits of the subsequently returned result limb.
        self.last_redced_val = Self::redc_rshift_lo(
            self.redc_pow2_rshift,
            self.redc_pow2_mask,
            redced_t_val
        );

        result_val
    }

    pub fn finish(self, t_val: LimbType) -> (LimbType, LimbType) {
        debug_assert_eq!(t_val & !self.redc_pow2_mask, 0);
        let (carry, redced_t_val) = ct_add_l_l(t_val, self.carry);
        (
            carry,
            self.last_redced_val
                | Self::redc_lshift_hi(
                    self.redc_pow2_rshift,
                    self.redc_pow2_mask,
                    redced_t_val
                )
         )
    }

    pub fn finish_in_twos_complement(self, t_val: LimbType) -> (LimbType, LimbType) {
        let t_val_sign = t_val >> LIMB_BITS - 1;
        let (carry, redced_t_val) = ct_add_l_l(t_val, self.carry);

        // In two's complement representation, the addition overflows iff the sign
        // bit (indicating a virtual borrow) is getting neutralized.
        debug_assert!(carry == 0 || t_val_sign == 1);
        debug_assert!(carry == 0 || redced_t_val >> LIMB_BITS - 1 == 0);
        debug_assert!(carry == 1 || redced_t_val >> LIMB_BITS - 1 == t_val_sign);
        let redced_t_val_sign = carry ^ t_val_sign;
        debug_assert!(redced_t_val >> LIMB_BITS - 1 == redced_t_val_sign);
        let redced_t_val_extended_sign = (0 as LimbType).wrapping_sub(redced_t_val_sign) & !self.redc_pow2_mask;
        (
            redced_t_val_sign,
            self.last_redced_val
                | Self::redc_lshift_hi(
                    self.redc_pow2_rshift,
                    self.redc_pow2_mask,
                redced_t_val
                )
                | redced_t_val_extended_sign
        )
    }
}

pub fn mp_ct_montgomery_redc<TT: MPIntMutByteSlice, NT: MPIntByteSliceCommon>(t: &mut TT, n: &NT, neg_n0_inv_mod_l: LimbType) {
    debug_assert!(!n.is_empty());
    debug_assert!(t.len() >= n.len());
    debug_assert!(t.len() <= 2 * n.len());
    let t_nlimbs = t.nlimbs();
    let n_nlimbs = n.nlimbs();
    let n0_val = n.load_l(0);
    debug_assert!(n0_val.wrapping_mul(neg_n0_inv_mod_l) == !0);

    let mut reduced_t_carry = 0;
    // t's high limb might be a partial one, do not update directly in the course of reducing in
    // order to avoid overflowing it. Use a shadow instead.
    let mut t_high_shadow = t.load_l(t_nlimbs - 1);
    for _i in 0..mp_ct_montgomery_radix_shift_nlimbs(n.len()) {
        let mut redc_kernel = MpCtMontgomeryRedcKernel::start(LIMB_BITS, t.load_l(0), n0_val, neg_n0_inv_mod_l);
        let mut j = 0;
        while j + 2 < n_nlimbs {
            debug_assert!(j < t_nlimbs - 1);
            t.store_l_full(j, redc_kernel.update(t.load_l_full(j + 1), n.load_l_full(j + 1)));
            j += 1;
        }

        debug_assert_eq!(j + 2, n_nlimbs);
        // Do not read the potentially partial, stale high limb directly from t, use the
        // t_high_shadow shadow instead.
        let t_val = if j + 2 != t_nlimbs {
            t.load_l_full(j + 1)
        } else {
            t_high_shadow
        };
        t.store_l_full(j, redc_kernel.update(t_val, n.load_l(j + 1)));
        j += 1;

        while j + 2 < t_nlimbs {
            t.store_l_full(j, redc_kernel.update(t.load_l_full(j + 1), 0));
            j += 1;
        }
        if j + 1 == t_nlimbs - 1 {
            t.store_l_full(j, redc_kernel.update(t_high_shadow, 0));
            j += 1;
        }
        debug_assert_eq!(j, t_nlimbs - 1);

        (reduced_t_carry, t_high_shadow) = redc_kernel.finish(reduced_t_carry);
    }

    // Now apply the high limb shadow back.
    let t_high_shadow_mask = t.partial_high_mask();
    let t_high_shadow_shift = t.partial_high_shift();
    assert!(t_high_shadow_shift == 0 || reduced_t_carry == 0);
    reduced_t_carry |= (t_high_shadow & !t_high_shadow_mask) >> t_high_shadow_shift;
    t_high_shadow &= t_high_shadow_mask;
    t.store_l(t_nlimbs - 1, t_high_shadow);

    mp_ct_sub_cond_mp_mp(t, n, LimbChoice::from(reduced_t_carry) | mp_ct_geq_mp_mp(t, n));
    debug_assert!(mp_ct_geq_mp_mp(t, n).unwrap() == 0);
}

#[cfg(test)]
fn test_mp_ct_montgomery_redc<TT: MPIntMutByteSlice, NT: MPIntMutByteSlice>() {
    for i in 0..64 {
        const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
        let n_high = MERSENNE_PRIME_13.wrapping_mul((16385 as LimbType).wrapping_mul(i));
        for j in 0..64 {
            const MERSENNE_PRIME_17: LimbType = 131071 as LimbType;
            let n_low = MERSENNE_PRIME_17.wrapping_mul((1023 as LimbType).wrapping_mul(j));
            // Force n_low odd.
            let n_low = n_low | 1;
            let mut n: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
            let mut n = NT::from_bytes(n.as_mut_slice()).unwrap();
            n.store_l(0, n_low);
            n.store_l(1, n_high);
            let neg_n0_inv = ct_montgomery_neg_n0_inv_mod_l(&n);

            for k in 0..8 {
                let t_high = MERSENNE_PRIME_17.wrapping_mul((8191 as LimbType).wrapping_mul(k));
                for l in 0..8 {
                    let t_low = MERSENNE_PRIME_13.wrapping_mul((131087 as LimbType).wrapping_mul(l));

                    let mut t: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                    let mut t = TT::from_bytes(t.as_mut_slice()).unwrap();
                    t.store_l(0, t_low);
                    t.store_l(1, t_high);

                    // All montgomery operations are defined mod n, compute t mod n
                    mp_ct_div_mp_mp::<_, _, TT::SelfT<'_>>(None, &mut t, &n, None).unwrap();
                    let t_low = t.load_l(0);
                    let t_high = t.load_l(1);

                    // To Montgomery form: t * R mod n
                    let mut scratch: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                    mp_ct_to_montgomery_form_direct(&mut t, &n).unwrap();

                    // And back to normal: (t * R mod n) / R mod n
                    mp_ct_montgomery_redc(&mut t, &n, neg_n0_inv);
                    assert_eq!(t.load_l(0), t_low);
                    assert_eq!(t.load_l(1), t_high);
                }
            }
        }
    }
}

#[test]
fn test_mp_ct_montgomery_redc_be_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_montgomery_redc::<MPBigEndianMutByteSlice, MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_montgomery_redc_le_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_montgomery_redc::<MPLittleEndianMutByteSlice, MPLittleEndianMutByteSlice>()
}

#[test]
 fn test_mp_ct_montgomery_redc_ne_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_ct_montgomery_redc::<MPNativeEndianMutByteSlice, MPNativeEndianMutByteSlice>()
}

pub fn mp_ct_montgomery_mul_mod_cond<RT: MPIntMutByteSlice, T0: MPIntByteSliceCommon,
                                     T1: MPIntByteSliceCommon, NT: MPIntByteSliceCommon> (
    result: &mut RT, op0: &T0, op1: &T1, n: &NT, neg_n0_inv_mod_l: LimbType,
    cond: LimbChoice
) {
    // This is an implementation of the "Finely Integrated Operand Scanning (FIOS) Method"
    // approach to fused multiplication and Montgomery reduction, as described in "Analyzing
    // and Comparing Montgomery Multiplication Algorithm", IEEE Micro, 16(3):26-33, June 1996.
    debug_assert!(!n.is_empty());
    debug_assert_eq!(result.len(), n.len());
    debug_assert!(op0.len() <= n.len());
    debug_assert!(op1.len() <= n.len());

    let op0_nlimbs = op0.nlimbs();
    let op1_nlimbs = op1.nlimbs();
    let n_nlimbs = n.nlimbs();
    let n0_val = n.load_l(0);
    debug_assert!(n0_val.wrapping_mul(neg_n0_inv_mod_l) == !0);

    result.zeroize_bytes_above(0);
    let mut result_carry = 0;
    // result's high limb might be a partial one, do not update directly in the course of reducing in
    // order to avoid overflowing it. Use a shadow instead.
    let mut result_high_shadow = 0;
    for i in 0..op0_nlimbs {
        debug_assert!(result_carry <= 1); // Loop invariant.
        let op0_val = op0.load_l(i);

        // If cond == false, then the Montgomery kernel's multiplication factor,
        // MpCtMontgomeryRedcKernel will be set to zero below and repeated application
        // of the kernel would effectively shift the result by one word to the right,
        // pulling in result_carry from the left.
        result_carry = cond.select(op0_val, result_carry);

        // Do not read the potentially partial, stale high limb directly from result, use the
        // result_high_shadow shadow instead.
        let result_val = if n_nlimbs != 1 {
            result.load_l_full(0)
        } else {
            result_high_shadow
        };
        let op1_val = cond.select(0, op1.load_l(0));
        let (mut op0_op1_add_carry, result_val) = ct_mul_add_l_l_l_c(result_val, op0_val, op1_val, 0);

        let mut redc_kernel = MpCtMontgomeryRedcKernel::start(LIMB_BITS, result_val, n0_val, neg_n0_inv_mod_l);
        redc_kernel.m = cond.select(0, redc_kernel.m);

        let mut j = 0;
        while j + 1 < op1_nlimbs {
            let op1_val = cond.select(0, op1.load_l(j + 1));

            // Do not read the potentially partial, stale high limb directly from result, use the
            // result_high_shadow shadow instead.
            let mut result_val = if j + 1 != n_nlimbs - 1 {
                result.load_l_full(j + 1)
            } else {
                result_high_shadow
            };

            (op0_op1_add_carry, result_val) = ct_mul_add_l_l_l_c(result_val, op0_val, op1_val, op0_op1_add_carry);

            let n_val = n.load_l(j + 1);
            let result_val = redc_kernel.update(result_val, n_val);
            result.store_l_full(j, result_val);
            j += 1;
        }
        debug_assert_eq!(j + 1, op1_nlimbs);

        // If op1_nlimbs < n_nlimbs, handle the rest by propagating the multiplication carry and
        // continue redcing.
        while j + 1 < n_nlimbs {
            // Do not read the potentially partial, stale high limb directly from result, use the
            // result_high_shadow shadow instead.
            let mut result_val = if j + 1 != n_nlimbs - 1 {
                result.load_l_full(j + 1)
            } else {
                result_high_shadow
            };

            (op0_op1_add_carry, result_val) = ct_add_l_l(result_val, op0_op1_add_carry);

            let n_val = n.load_l(j + 1);
            let result_val = redc_kernel.update(result_val, n_val);
            result.store_l_full(j, result_val);
            j += 1;
        }
        debug_assert_eq!(j + 1, n_nlimbs);

        let mut result_val;
        debug_assert!(cond.unwrap() == 0 || result_carry <= 1);
        debug_assert!(cond.unwrap() == 1 || op0_op1_add_carry == 0);
        (result_carry, result_val) = ct_add_l_l(result_carry, op0_op1_add_carry);
        debug_assert!(result_carry <= 1);
        debug_assert!(result_carry == 0 || result_val == 0);

        (result_carry, result_val) = redc_kernel.finish(result_val);
        debug_assert!(result_carry <= 1);
        debug_assert!(cond.unwrap() == 1 || result_carry == 0);
        result_high_shadow = result_val;
    }

    // If op0_nlimbs < the montgomery radix shift distance, handle the rest by REDCing it.
    for _i in op0_nlimbs..mp_ct_montgomery_radix_shift_nlimbs(n.len()) {
        // Do not read the potentially partial, stale high limb directly from result, use the
        // result_high_shadow shadow instead.
        let result_val = if n_nlimbs != 1 {
            result.load_l_full(0)
        } else {
            result_high_shadow
        };

        let mut redc_kernel = MpCtMontgomeryRedcKernel::start(LIMB_BITS, result_val, n0_val, neg_n0_inv_mod_l);
        redc_kernel.m = cond.select(0, redc_kernel.m);

        let mut j = 0;
        while j + 1 < n_nlimbs {
            // Do not read the potentially partial, stale high limb directly from result, use the
            // result_high_shadow shadow instead.
            let result_val = if j + 1 != n_nlimbs - 1 {
                result.load_l_full(j + 1)
            } else {
                result_high_shadow
            };

            let n_val = n.load_l(j + 1);
            let result_val = redc_kernel.update(result_val, n_val);
            result.store_l_full(j, result_val);
            j += 1;
        }
        debug_assert_eq!(j + 1, n_nlimbs);

        (result_carry, result_high_shadow) = redc_kernel.finish(result_carry);
        debug_assert!(result_carry <= 1);
    }

    // Now apply the high limb shadow back.
    debug_assert!(result.len() == n.len());
    let result_high_shadow_mask = n.partial_high_mask();
    let result_high_shadow_shift = n.partial_high_shift();
    assert!(result_high_shadow_shift == 0 || result_carry == 0);
    result_carry |= (result_high_shadow & !result_high_shadow_mask) >> result_high_shadow_shift;
    result_high_shadow &= result_high_shadow_mask;
    result.store_l(n_nlimbs - 1, result_high_shadow);

    mp_ct_sub_cond_mp_mp(
        result, n, LimbChoice::from(result_carry) | mp_ct_geq_mp_mp(result, n)
    );
    debug_assert!(mp_ct_geq_mp_mp(result, n).unwrap() == 0);
}

#[cfg(test)]
fn test_mp_ct_montgomery_mul_mod_cond<RT: MPIntMutByteSlice,
                                      T0: MPIntMutByteSlice, T1: MPIntMutByteSlice,
                                      NT: MPIntMutByteSlice>() {
    use crate::limbs_buffer::MPIntByteSliceCommonPriv;

    use super::mul_impl::mp_ct_mul_trunc_cond_mp_mp;

    for i in 0..16 {
        const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
        let n_high = MERSENNE_PRIME_13.wrapping_mul((65543 as LimbType).wrapping_mul(i));
        for j in 0..16 {
            const MERSENNE_PRIME_17: LimbType = 131071 as LimbType;
            let n_low = MERSENNE_PRIME_17.wrapping_mul((4095 as LimbType).wrapping_mul(j));
            // Force n_low odd.
            let n_low = n_low | 1;
            let mut n: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
            let mut n = NT::from_bytes(n.as_mut_slice()).unwrap();
            n.store_l(0, n_low);
            n.store_l(1, n_high);
            let n_lengths = if !RT::SUPPORTS_UNALIGNED_BUFFER_LENGTHS ||
                !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS ||
                !T1::SUPPORTS_UNALIGNED_BUFFER_LENGTHS ||
                !NT::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
                    [LIMB_BYTES, 2 * LIMB_BYTES]
                } else {
                    [2 * LIMB_BYTES - 1, 2 * LIMB_BYTES]
                };
            for n_len in n_lengths {
                let (_, n) = n.split_at(n_len);
                let neg_n0_inv = ct_montgomery_neg_n0_inv_mod_l(&n);

                // r_mod_n = 2^(2 * LIMB_BITS) % n.
                let mut r_mod_n: [u8; 3 * LIMB_BYTES] = [0; 3 * LIMB_BYTES];
                let mut r_mod_n = RT::from_bytes(r_mod_n.as_mut_slice()).unwrap();
                r_mod_n.store_l_full(mp_ct_montgomery_radix_shift_nlimbs(n_len), 1);
                mp_ct_div_mp_mp::<_, _, RT::SelfT<'_>>(None, &mut r_mod_n, &n, None).unwrap();
                let (_, r_mod_n) = r_mod_n.split_at(n.len());

                for k in 0..4 {
                    let a_high = MERSENNE_PRIME_17.wrapping_mul((16383 as LimbType).wrapping_mul(k));
                    for l in 0..4 {
                        let a_low = MERSENNE_PRIME_13.wrapping_mul((262175 as LimbType).wrapping_mul(l));
                        let mut a: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                        let mut a = T0::from_bytes(a.as_mut_slice()).unwrap();
                        a.store_l(0, a_low);
                        a.store_l(1, a_high);
                        // All montgomery operations are defined mod n, compute a mod n
                        mp_ct_div_mp_mp::<_, _, T0::SelfT<'_>>(None, &mut a, &n, None).unwrap();
                        for s in 0..4 {
                            let b_high = MERSENNE_PRIME_13.wrapping_mul((262175 as LimbType).wrapping_mul(s));
                            for t in 0..4 {
                                const MERSENNE_PRIME_19: LimbType = 524287 as LimbType;
                                let b_low = MERSENNE_PRIME_19.wrapping_mul((4095 as LimbType).wrapping_mul(t));
                                let mut b: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                                let mut b = T1::from_bytes(b.as_mut_slice()).unwrap();
                                b.store_l(0, b_low);
                                b.store_l(1, b_high);
                                // All montgomery operations are defined mod n, compute b mod n
                                mp_ct_div_mp_mp::<_, _, T1::SelfT<'_>>(None, &mut b, &n, None).unwrap();

                                for op_len in [0, 1 * LIMB_BYTES, n_len] {
                                    let (_, a) = a.split_at(op_len);
                                    let (_, b) = b.split_at(op_len);

                                    let mut _result: [u8; 4 * LIMB_BYTES] = [0; 4 * LIMB_BYTES];
                                    let mut result = RT::from_bytes(_result.as_mut_slice()).unwrap();
                                    let (_, mut mg_mul_result) = result.split_at(n_len);
                                    mp_ct_montgomery_mul_mod_cond(
                                        &mut mg_mul_result, &a, &b, &n, neg_n0_inv,
                                        LimbChoice::from(0)
                                    );
                                    let a_nlimbs = a.nlimbs();
                                    for i in 0..a_nlimbs {
                                        assert_eq!(mg_mul_result.load_l(i), a.load_l(i));
                                    }
                                    for i in a_nlimbs..mg_mul_result.nlimbs() {
                                        assert_eq!(mg_mul_result.load_l(i), 0);
                                    }

                                    mp_ct_montgomery_mul_mod_cond(
                                        &mut mg_mul_result, &a, &b, &n, neg_n0_inv,
                                        LimbChoice::from(1)
                                    );
                                    drop(mg_mul_result);

                                    // For testing against the expected result computed using the
                                    // "conventional" methods only, multiply by r_mod_n -- this avoids
                                    // having to multiply the conventional product by r^-1 mod n, which is
                                    // not known without implementing Euklid's algorithm.
                                    mp_ct_mul_trunc_cond_mp_mp(
                                        &mut result, n.len(), &r_mod_n, LimbChoice::from(1)
                                    );
                                    mp_ct_div_mp_mp::<_, _, RT::SelfT<'_>>(None, &mut result, &n, None).unwrap();
                                    drop(result);

                                    let mut _expected: [u8; 4 * LIMB_BYTES] = [0; 4 * LIMB_BYTES];
                                    let mut expected = RT::from_bytes(_expected.as_mut_slice()).unwrap();
                                    expected.copy_from(&a);
                                    mp_ct_mul_trunc_cond_mp_mp(
                                        &mut expected, op_len, &b, LimbChoice::from(1)
                                    );
                                    mp_ct_div_mp_mp::<_, _, RT::SelfT<'_>>(None, &mut expected, &n, None).unwrap();
                                    drop(expected);

                                    assert_eq!(_result, _expected);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_mp_ct_montgomery_mul_mod_cond_be_be_be_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_montgomery_mul_mod_cond::<MPBigEndianMutByteSlice,
                                         MPBigEndianMutByteSlice,
                                         MPBigEndianMutByteSlice,
                                         MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_montgomery_mul_mod_cond_le_le_le_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_montgomery_mul_mod_cond::<MPLittleEndianMutByteSlice,
                                         MPLittleEndianMutByteSlice,
                                         MPLittleEndianMutByteSlice,
                                         MPLittleEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_montgomery_mul_mod_cond_ne_ne_ne_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_ct_montgomery_mul_mod_cond::<MPNativeEndianMutByteSlice,
                                         MPNativeEndianMutByteSlice,
                                         MPNativeEndianMutByteSlice,
                                         MPNativeEndianMutByteSlice>()
}

pub fn mp_ct_montgomery_mul_mod<RT: MPIntMutByteSlice, T0: MPIntByteSliceCommon,
                                T1: MPIntByteSliceCommon, NT: MPIntByteSliceCommon> (
    result: &mut RT, op0: &T0, op1: &T1, n: &NT, neg_n0_inv_mod_l: LimbType
) {
    mp_ct_montgomery_mul_mod_cond(result, op0, op1, n, neg_n0_inv_mod_l, LimbChoice::from(1))
}

pub fn mp_ct_to_montgomery_form_direct<TT: MPIntMutByteSlice, NT: MPIntByteSliceCommon>(
    t: &mut TT, n: &NT
) -> Result<(), MpCtDivisionError> {
    debug_assert!(t.nlimbs() >= n.nlimbs());
    let radix_shift_len = mp_ct_montgomery_radix_shift_len(n.len());
    mp_ct_div_lshifted_mp_mp::<_, _, TT>(t, t.len(), radix_shift_len, n, None)
}

pub fn mp_ct_montgomery_radix2_mod_n<RX2T: MPIntMutByteSlice, NT: MPIntByteSliceCommon>(
    radix2_mod_n_out: &mut RX2T, n: &NT
) -> Result<(), MpCtDivisionError> {
    debug_assert!(mp_ct_nlimbs(radix2_mod_n_out.len()) >= mp_ct_nlimbs(n.len()));
    let radix_shift_len = mp_ct_montgomery_radix_shift_len(n.len());
    mp_ct_div_pow2_mp::<_, _, RX2T>(2 * 8 * radix_shift_len, radix2_mod_n_out, n, None)
}

pub fn mp_ct_to_montgomery_form<MGT: MPIntMutByteSlice, TT: MPIntByteSliceCommon, NT: MPIntByteSliceCommon, RX2T: MPIntByteSliceCommon> (
    mg_t_out: &mut MGT, t: &TT, n: &NT, neg_n0_inv_mod_l: LimbType, radix2_mod_n: &RX2T
) {
    mp_ct_montgomery_mul_mod(mg_t_out, t, radix2_mod_n, n, neg_n0_inv_mod_l);
}


#[cfg(test)]
fn test_mp_ct_to_montgomery_form<TT: MPIntMutByteSlice, NT: MPIntMutByteSlice, RX2T: MPIntMutByteSlice>() {
    use super::limbs_buffer::MPIntMutByteSlicePriv as _;
    use super::cmp_impl::mp_ct_eq_mp_mp;

    for i in 0..16 {
        const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
        let n_high = MERSENNE_PRIME_13.wrapping_mul((65543 as LimbType).wrapping_mul(i));
        for j in 0..16 {
            const MERSENNE_PRIME_17: LimbType = 131071 as LimbType;
            let n_low = MERSENNE_PRIME_17.wrapping_mul((4095 as LimbType).wrapping_mul(j));
            // Force n_low odd.
            let n_low = n_low | 1;
            let mut n: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
            let mut n = NT::from_bytes(n.as_mut_slice()).unwrap();
            n.store_l(0, n_low);
            n.store_l(1, n_high);
            let n_lengths = if !NT::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
                    [LIMB_BYTES, 2 * LIMB_BYTES]
                } else {
                    [2 * LIMB_BYTES - 1, 2 * LIMB_BYTES]
                };

            for n_len in n_lengths {
                let (_, n) = n.split_at(n_len);
                let neg_n0_inv = ct_montgomery_neg_n0_inv_mod_l(&n);

                let mut radix2_mod_n: [u8; 2 * LIMB_BYTES] = [0xffu8; 2 * LIMB_BYTES];
                let mut radix2_mod_n = RX2T::from_bytes(radix2_mod_n.as_mut_slice()).unwrap();
                let (_, mut radix2_mod_n) = radix2_mod_n.split_at(RX2T::limbs_align_len(n_len));
                mp_ct_montgomery_radix2_mod_n(&mut radix2_mod_n, &n).unwrap();

                for k in 0..4 {
                    let a_high = MERSENNE_PRIME_17.wrapping_mul((16383 as LimbType).wrapping_mul(k));
                    for l in 0..4 {
                        let a_low = MERSENNE_PRIME_13.wrapping_mul((262175 as LimbType).wrapping_mul(l));
                        let mut a: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                        let mut a = TT::from_bytes(a.as_mut_slice()).unwrap();
                        a.store_l(0, a_low);
                        a.store_l(1, a_high);
                        // All montgomery operations are defined mod n, compute a mod n
                        mp_ct_div_mp_mp::<_, _, TT::SelfT<'_>>(None, &mut a, &n, None).unwrap();
                        let (_, mut a) = a.split_at(TT::limbs_align_len(n_len));

                        let mut result: [u8; 2 * LIMB_BYTES] = [0xff; 2 * LIMB_BYTES];
                        let mut result = TT::from_bytes(result.as_mut_slice()).unwrap();
                        let (_, mut result) = result.split_at(TT::limbs_align_len(n_len));
                        let (_, mut result) = result.split_at(TT::limbs_align_len(n_len));
                        mp_ct_to_montgomery_form(&mut result, &a, &n, neg_n0_inv, &radix2_mod_n);

                        mp_ct_to_montgomery_form_direct(&mut a, &n).unwrap();
                        assert_eq!(mp_ct_eq_mp_mp(&result, &a).unwrap(), 1);
                    }
                }
            }
        }
    }
}

#[test]
fn test_mp_ct_to_montgomery_form_be_be_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_to_montgomery_form::<MPBigEndianMutByteSlice,
                                         MPBigEndianMutByteSlice,
                                         MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_to_montgomery_form_le_le_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_to_montgomery_form::<MPLittleEndianMutByteSlice,
                                         MPLittleEndianMutByteSlice,
                                         MPLittleEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_to_montgomery_form_ne_ne_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_ct_to_montgomery_form::<MPNativeEndianMutByteSlice,
                                         MPNativeEndianMutByteSlice,
                                         MPNativeEndianMutByteSlice>()
}

