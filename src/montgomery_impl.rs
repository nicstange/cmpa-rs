use super::add_impl::ct_sub_cond_mp_mp;
use super::cmp_impl::{ct_geq_mp_mp, ct_lt_mp_mp};
use super::div_impl::{
    ct_mod_lshifted_mp_mp, ct_mod_pow2_mp, CtModLshiftedMpMpError, CtModPow2MpError, CtMpDivisor,
    CtMpDivisorError,
};
use super::limb::{
    ct_add_l_l, ct_inv_mod_l, ct_lsb_mask_l, ct_mul_add_l_l_l_c, LimbChoice, LimbType, LIMB_BITS,
};
use super::limbs_buffer::{
    ct_mp_limbs_align_len, ct_mp_nlimbs, MpIntByteSliceCommon, MpIntMutByteSlice,
    MpNativeEndianMutByteSlice,
};

fn ct_montgomery_radix_shift_len(n_len: usize) -> usize {
    ct_mp_limbs_align_len(n_len)
}

fn ct_montgomery_radix_shift_mp_nlimbs(n_len: usize) -> usize {
    ct_mp_nlimbs(n_len)
}

pub fn ct_montgomery_neg_n0_inv_mod_l_mp<NT: MpIntByteSliceCommon>(n: &NT) -> LimbType {
    debug_assert!(!n.is_empty());
    let n0 = n.load_l(0);
    let n0_inv_mod_l = ct_inv_mod_l(n0);
    (!n0_inv_mod_l).wrapping_add(1)
}

#[test]
fn test_ct_montgomery_neg_n0_inv_mod_l_mp() {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::MpBigEndianMutByteSlice;

    for n0 in 0 as LimbType..128 {
        let n0 = 2 * n0 + 1;
        for j in 0..2048 {
            const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
            let v = MERSENNE_PRIME_13.wrapping_mul((511 as LimbType).wrapping_mul(j));
            let v = v << 8;
            let n0 = n0.wrapping_add(v);

            let mut n0_buf: [u8; LIMB_BYTES] = [0; LIMB_BYTES];
            let mut n = MpBigEndianMutByteSlice::from_bytes(n0_buf.as_mut_slice()).unwrap();
            n.store_l(0, n0);
            let neg_n0_inv = ct_montgomery_neg_n0_inv_mod_l_mp(&n);
            assert_eq!(n0.wrapping_mul(neg_n0_inv), !0);
        }
    }
}

pub struct CtMontgomeryRedcKernel {
    redc_pow2_rshift: u32,
    redc_pow2_mask: LimbType,
    m: LimbType,
    last_redced_val: LimbType,
    carry: LimbType,
}

impl CtMontgomeryRedcKernel {
    const fn redc_rshift_lo(
        redc_pow2_rshift: u32,
        redc_pow2_mask: LimbType,
        val: LimbType,
    ) -> LimbType {
        // There are two possible cases where redc_pow2_rshift == 0:
        // a.) redc_pow2_exp == 0. In this case !redc_pow2_mask == !0.
        // b.) redc_pow2_exp == LIMB_BITS. In this case !redc_pow2_mask == !!0 == 0.
        (val & !redc_pow2_mask) >> redc_pow2_rshift
    }

    const fn redc_lshift_hi(
        redc_pow2_rshift: u32,
        redc_pow2_mask: LimbType,
        val: LimbType,
    ) -> LimbType {
        // There are two possible cases where redc_pow2_rshift == 0:
        // a.) redc_pow2_exp == 0. In this case redc_pow2_mask == 0 as well.
        // b.) redc_pow2_exp == LIMB_BITS. In this case redc_pow2_mask == !0.
        (val & redc_pow2_mask) << ((LIMB_BITS - redc_pow2_rshift) % LIMB_BITS)
    }

    pub fn start(
        redc_pow2_exp: u32,
        t0_val: LimbType,
        n0_val: LimbType,
        neg_n0_inv_mod_l: LimbType,
    ) -> Self {
        debug_assert!(redc_pow2_exp <= LIMB_BITS);

        // Calculate the shift distance for right shifting a redced limb into its final
        // position. Be careful to keep it < LIMB_BITS for not running into
        // undefined behaviour with the shift.  See the comments in
        // redc_rshift_lo()/redc_rshift_hi() for the mask<->rshift interaction.
        let redc_pow2_rshift = redc_pow2_exp % LIMB_BITS;
        let redc_pow2_mask = ct_lsb_mask_l(redc_pow2_exp);

        // For any i >= j, if n' == -n^{-1} mod 2^i, then n' mod 2^j == -n^{-1} mod 2^j.
        let neg_n0_inv_mod_l = neg_n0_inv_mod_l & redc_pow2_mask;
        debug_assert_eq!(
            neg_n0_inv_mod_l.wrapping_mul(n0_val) & redc_pow2_mask,
            redc_pow2_mask
        );

        let m = t0_val.wrapping_mul(neg_n0_inv_mod_l) & redc_pow2_mask;

        let (carry, redced_t0_val) = ct_mul_add_l_l_l_c(t0_val, m, n0_val, 0);
        debug_assert_eq!(redced_t0_val & redc_pow2_mask, 0);
        // If redc_pow2_exp < LIMB_BITS, the upper bits of the reduced zeroth limb
        // will become the lower bits of the resulting zeroth limb.
        let last_redced_val = Self::redc_rshift_lo(redc_pow2_rshift, redc_pow2_mask, redced_t0_val);

        Self {
            redc_pow2_rshift,
            redc_pow2_mask,
            m,
            last_redced_val,
            carry,
        }
    }

    pub fn update(&mut self, t_val: LimbType, n_val: LimbType) -> LimbType {
        let redced_t_val;
        (self.carry, redced_t_val) = ct_mul_add_l_l_l_c(t_val, self.m, n_val, self.carry);

        // If redc_pow2_exp < LIMB_BITS, the lower bits of the reduced current limb
        // correspond to the upper bits of the returned result limb.
        let result_val = self.last_redced_val
            | Self::redc_lshift_hi(self.redc_pow2_rshift, self.redc_pow2_mask, redced_t_val);
        // If redc_pow2_exp < LIMB_BITS, the upper bits of the reduced current limb
        // will become the lower bits of the subsequently returned result limb.
        self.last_redced_val =
            Self::redc_rshift_lo(self.redc_pow2_rshift, self.redc_pow2_mask, redced_t_val);

        result_val
    }

    pub fn finish(self, t_val: LimbType) -> (LimbType, LimbType) {
        debug_assert_eq!(t_val & !self.redc_pow2_mask, 0);
        let (carry, redced_t_val) = ct_add_l_l(t_val, self.carry);
        (
            carry,
            self.last_redced_val
                | Self::redc_lshift_hi(self.redc_pow2_rshift, self.redc_pow2_mask, redced_t_val),
        )
    }

    pub fn finish_in_twos_complement(self, t_val: LimbType) -> (LimbType, LimbType) {
        let t_val_sign = t_val >> (LIMB_BITS - 1);
        let (carry, redced_t_val) = ct_add_l_l(t_val, self.carry);

        // In two's complement representation, the addition overflows iff the sign
        // bit (indicating a virtual borrow) is getting neutralized.
        debug_assert!(carry == 0 || t_val_sign == 1);
        debug_assert!(carry == 0 || redced_t_val >> (LIMB_BITS - 1) == 0);
        debug_assert!(carry == 1 || redced_t_val >> (LIMB_BITS - 1) == t_val_sign);
        let redced_t_val_sign = carry ^ t_val_sign;
        debug_assert!(redced_t_val >> (LIMB_BITS - 1) == redced_t_val_sign);
        let redced_t_val_extended_sign =
            (0 as LimbType).wrapping_sub(redced_t_val_sign) & !self.redc_pow2_mask;
        (
            redced_t_val_sign,
            self.last_redced_val
                | Self::redc_lshift_hi(self.redc_pow2_rshift, self.redc_pow2_mask, redced_t_val)
                | redced_t_val_extended_sign,
        )
    }
}

pub fn ct_montgomery_redc_mp<TT: MpIntMutByteSlice, NT: MpIntByteSliceCommon>(
    t: &mut TT,
    n: &NT,
    neg_n0_inv_mod_l: LimbType,
) {
    debug_assert!(!n.is_empty());
    debug_assert!(t.len() >= n.len());
    debug_assert!(t.len() <= 2 * n.len());
    let t_nlimbs = t.nlimbs();
    let n_nlimbs = n.nlimbs();
    let n0_val = n.load_l(0);
    debug_assert!(n0_val.wrapping_mul(neg_n0_inv_mod_l) == !0);

    let mut reduced_t_carry = 0;
    // t's high limb might be a partial one, do not update directly in the course of
    // reducing in order to avoid overflowing it. Use a shadow instead.
    let mut t_high_shadow = t.load_l(t_nlimbs - 1);
    for _i in 0..ct_montgomery_radix_shift_mp_nlimbs(n.len()) {
        let mut redc_kernel =
            CtMontgomeryRedcKernel::start(LIMB_BITS, t.load_l(0), n0_val, neg_n0_inv_mod_l);
        let mut j = 0;
        while j + 2 < n_nlimbs {
            debug_assert!(j < t_nlimbs - 1);
            t.store_l_full(
                j,
                redc_kernel.update(t.load_l_full(j + 1), n.load_l_full(j + 1)),
            );
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

    ct_sub_cond_mp_mp(t, n, LimbChoice::from(reduced_t_carry) | ct_geq_mp_mp(t, n));
    debug_assert!(ct_geq_mp_mp(t, n).unwrap() == 0);
}

#[cfg(test)]
fn test_ct_montgomery_redc_mp<TT: MpIntMutByteSlice, NT: MpIntMutByteSlice>() {
    use super::div_impl::ct_mod_mp_mp;
    use super::limb::LIMB_BYTES;

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
            let neg_n0_inv = ct_montgomery_neg_n0_inv_mod_l_mp(&n);

            for k in 0..8 {
                let t_high = MERSENNE_PRIME_17.wrapping_mul((8191 as LimbType).wrapping_mul(k));
                for l in 0..8 {
                    let t_low =
                        MERSENNE_PRIME_13.wrapping_mul((131087 as LimbType).wrapping_mul(l));

                    let mut t: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                    let mut t = TT::from_bytes(t.as_mut_slice()).unwrap();
                    t.store_l(0, t_low);
                    t.store_l(1, t_high);

                    // All montgomery operations are defined mod n, compute t mod n
                    ct_mod_mp_mp(None, &mut t, &CtMpDivisor::new(&n).unwrap());
                    let t_low = t.load_l(0);
                    let t_high = t.load_l(1);

                    // To Montgomery form: t * R mod n
                    ct_to_montgomery_form_direct_mp(&mut t, &n).unwrap();

                    // And back to normal: (t * R mod n) / R mod n
                    ct_montgomery_redc_mp(&mut t, &n, neg_n0_inv);
                    assert_eq!(t.load_l(0), t_low);
                    assert_eq!(t.load_l(1), t_high);
                }
            }
        }
    }
}

#[test]
fn test_ct_montgomery_redc_be_be() {
    use super::limbs_buffer::MpBigEndianMutByteSlice;
    test_ct_montgomery_redc_mp::<MpBigEndianMutByteSlice, MpBigEndianMutByteSlice>()
}

#[test]
fn test_ct_montgomery_redc_le_le() {
    use super::limbs_buffer::MpLittleEndianMutByteSlice;
    test_ct_montgomery_redc_mp::<MpLittleEndianMutByteSlice, MpLittleEndianMutByteSlice>()
}

#[test]
fn test_ct_montgomery_redc_ne_ne() {
    use super::limbs_buffer::MpNativeEndianMutByteSlice;
    test_ct_montgomery_redc_mp::<MpNativeEndianMutByteSlice, MpNativeEndianMutByteSlice>()
}

pub fn ct_montgomery_mul_mod_cond_mp_mp<
    RT: MpIntMutByteSlice,
    T0: MpIntByteSliceCommon,
    T1: MpIntByteSliceCommon,
    NT: MpIntByteSliceCommon,
>(
    result: &mut RT,
    op0: &T0,
    op1: &T1,
    n: &NT,
    neg_n0_inv_mod_l: LimbType,
    cond: LimbChoice,
) {
    // This is an implementation of the "Finely Integrated Operand Scanning (FIOS)
    // Method" approach to fused multiplication and Montgomery reduction, as
    // described in "Analyzing and Comparing Montgomery Multiplication
    // Algorithm", IEEE Micro, 16(3):26-33, June 1996.
    debug_assert!(!n.is_empty());
    debug_assert_eq!(result.nlimbs(), n.nlimbs());
    debug_assert!(op0.nlimbs() <= n.nlimbs());
    debug_assert!(op1.nlimbs() <= n.nlimbs());

    let op0_nlimbs = op0.nlimbs();
    let op1_nlimbs = op1.nlimbs();
    let n_nlimbs = n.nlimbs();
    let n0_val = n.load_l(0);
    debug_assert!(n0_val.wrapping_mul(neg_n0_inv_mod_l) == !0);

    result.clear_bytes_above(0);
    let mut result_carry = 0;
    // result's high limb might be a partial one, do not update directly in the
    // course of reducing in order to avoid overflowing it. Use a shadow
    // instead.
    let mut result_high_shadow = 0;
    for i in 0..op0_nlimbs {
        debug_assert!(result_carry <= 1); // Loop invariant.
        let op0_val = op0.load_l(i);

        // If cond == false, then the Montgomery kernel's multiplication factor,
        // MpCtMontgomeryRedcKernel will be set to zero below and repeated application
        // of the kernel would effectively shift the result by one word to the right,
        // pulling in result_carry from the left.
        result_carry = cond.select(op0_val, result_carry);

        // Do not read the potentially partial, stale high limb directly from result,
        // use the result_high_shadow shadow instead.
        let result_val = if n_nlimbs != 1 {
            result.load_l_full(0)
        } else {
            result_high_shadow
        };
        let op1_val = cond.select(0, op1.load_l(0));
        let (mut op0_op1_add_carry, result_val) =
            ct_mul_add_l_l_l_c(result_val, op0_val, op1_val, 0);

        let mut redc_kernel =
            CtMontgomeryRedcKernel::start(LIMB_BITS, result_val, n0_val, neg_n0_inv_mod_l);
        redc_kernel.m = cond.select(0, redc_kernel.m);

        let mut j = 0;
        while j + 1 < op1_nlimbs {
            let op1_val = cond.select(0, op1.load_l(j + 1));

            // Do not read the potentially partial, stale high limb directly from result,
            // use the result_high_shadow shadow instead.
            let mut result_val = if j + 1 != n_nlimbs - 1 {
                result.load_l_full(j + 1)
            } else {
                result_high_shadow
            };

            (op0_op1_add_carry, result_val) =
                ct_mul_add_l_l_l_c(result_val, op0_val, op1_val, op0_op1_add_carry);

            let n_val = n.load_l(j + 1);
            let result_val = redc_kernel.update(result_val, n_val);
            result.store_l_full(j, result_val);
            j += 1;
        }
        debug_assert_eq!(j + 1, op1_nlimbs);

        // If op1_nlimbs < n_nlimbs, handle the rest by propagating the multiplication
        // carry and continue redcing.
        while j + 1 < n_nlimbs {
            // Do not read the potentially partial, stale high limb directly from result,
            // use the result_high_shadow shadow instead.
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

    // If op0_nlimbs < the montgomery radix shift distance, handle the rest by
    // REDCing it.
    for _i in op0_nlimbs..ct_montgomery_radix_shift_mp_nlimbs(n.len()) {
        // Do not read the potentially partial, stale high limb directly from result,
        // use the result_high_shadow shadow instead.
        let result_val = if n_nlimbs != 1 {
            result.load_l_full(0)
        } else {
            result_high_shadow
        };

        let mut redc_kernel =
            CtMontgomeryRedcKernel::start(LIMB_BITS, result_val, n0_val, neg_n0_inv_mod_l);
        redc_kernel.m = cond.select(0, redc_kernel.m);

        let mut j = 0;
        while j + 1 < n_nlimbs {
            // Do not read the potentially partial, stale high limb directly from result,
            // use the result_high_shadow shadow instead.
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
    debug_assert!(result.nlimbs() == n.nlimbs());
    let result_high_shadow_mask = n.partial_high_mask();
    let result_high_shadow_shift = n.partial_high_shift();
    assert!(result_high_shadow_shift == 0 || result_carry == 0);
    result_carry |= (result_high_shadow & !result_high_shadow_mask) >> result_high_shadow_shift;
    result_high_shadow &= result_high_shadow_mask;
    result.store_l(n_nlimbs - 1, result_high_shadow);

    ct_sub_cond_mp_mp(
        result,
        n,
        LimbChoice::from(result_carry) | ct_geq_mp_mp(result, n),
    );
    debug_assert!(ct_geq_mp_mp(result, n).unwrap() == 0);
}

#[cfg(test)]
fn test_ct_montgomery_mul_mod_cond_mp_mp<
    RT: MpIntMutByteSlice,
    T0: MpIntMutByteSlice,
    T1: MpIntMutByteSlice,
    NT: MpIntMutByteSlice,
>() {
    use super::div_impl::ct_mod_mp_mp;
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::{MpIntByteSliceCommonPriv as _, MpIntMutByteSlicePriv as _};
    use super::mul_impl::ct_mul_trunc_mp_mp;

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
            let n_lengths = if !RT::SUPPORTS_UNALIGNED_BUFFER_LENGTHS
                || !T0::SUPPORTS_UNALIGNED_BUFFER_LENGTHS
                || !T1::SUPPORTS_UNALIGNED_BUFFER_LENGTHS
                || !NT::SUPPORTS_UNALIGNED_BUFFER_LENGTHS
            {
                [LIMB_BYTES, 2 * LIMB_BYTES]
            } else {
                [2 * LIMB_BYTES - 1, 2 * LIMB_BYTES]
            };
            for n_len in n_lengths {
                let (_, n) = n.split_at(n_len);
                let neg_n0_inv = ct_montgomery_neg_n0_inv_mod_l_mp(&n);

                // r_mod_n = 2^(2 * LIMB_BITS) % n.
                let mut r_mod_n: [u8; 3 * LIMB_BYTES] = [0; 3 * LIMB_BYTES];
                let mut r_mod_n = RT::from_bytes(r_mod_n.as_mut_slice()).unwrap();
                r_mod_n.store_l_full(ct_montgomery_radix_shift_mp_nlimbs(n_len), 1);
                ct_mod_mp_mp(None, &mut r_mod_n, &CtMpDivisor::new(&n).unwrap());
                let (_, r_mod_n) = r_mod_n.split_at(n.len());

                for k in 0..4 {
                    let a_high =
                        MERSENNE_PRIME_17.wrapping_mul((16383 as LimbType).wrapping_mul(k));
                    for l in 0..4 {
                        let a_low =
                            MERSENNE_PRIME_13.wrapping_mul((262175 as LimbType).wrapping_mul(l));
                        let mut a: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                        let mut a = T0::from_bytes(a.as_mut_slice()).unwrap();
                        a.store_l(0, a_low);
                        a.store_l(1, a_high);
                        // All montgomery operations are defined mod n, compute a mod n
                        ct_mod_mp_mp(None, &mut a, &CtMpDivisor::new(&n).unwrap());
                        for s in 0..4 {
                            let b_high = MERSENNE_PRIME_13
                                .wrapping_mul((262175 as LimbType).wrapping_mul(s));
                            for t in 0..4 {
                                const MERSENNE_PRIME_19: LimbType = 524287 as LimbType;
                                let b_low = MERSENNE_PRIME_19
                                    .wrapping_mul((4095 as LimbType).wrapping_mul(t));
                                let mut b: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                                let mut b = T1::from_bytes(b.as_mut_slice()).unwrap();
                                b.store_l(0, b_low);
                                b.store_l(1, b_high);
                                // All montgomery operations are defined mod n, compute b mod n
                                ct_mod_mp_mp(None, &mut b, &CtMpDivisor::new(&n).unwrap());

                                for op_len in [0, 1 * LIMB_BYTES, n_len] {
                                    let (_, a) = a.split_at(op_len);
                                    let (_, b) = b.split_at(op_len);

                                    let mut _result: [u8; 4 * LIMB_BYTES] = [0; 4 * LIMB_BYTES];
                                    let mut result =
                                        RT::from_bytes(_result.as_mut_slice()).unwrap();
                                    let (_, mut mg_mul_result) = result.split_at(n_len);
                                    ct_montgomery_mul_mod_cond_mp_mp(
                                        &mut mg_mul_result,
                                        &a,
                                        &b,
                                        &n,
                                        neg_n0_inv,
                                        LimbChoice::from(0),
                                    );
                                    let a_nlimbs = a.nlimbs();
                                    for i in 0..a_nlimbs {
                                        assert_eq!(mg_mul_result.load_l(i), a.load_l(i));
                                    }
                                    for i in a_nlimbs..mg_mul_result.nlimbs() {
                                        assert_eq!(mg_mul_result.load_l(i), 0);
                                    }

                                    ct_montgomery_mul_mod_cond_mp_mp(
                                        &mut mg_mul_result,
                                        &a,
                                        &b,
                                        &n,
                                        neg_n0_inv,
                                        LimbChoice::from(1),
                                    );
                                    drop(mg_mul_result);

                                    // For testing against the expected result computed using the
                                    // "conventional" methods only, multiply by r_mod_n -- this
                                    // avoids having to multiply
                                    // the conventional product by r^-1 mod n, which is
                                    // not known without implementing Euklid's algorithm.
                                    ct_mul_trunc_mp_mp(&mut result, n.len(), &r_mod_n);
                                    ct_mod_mp_mp(None, &mut result, &CtMpDivisor::new(&n).unwrap());
                                    drop(result);

                                    let mut _expected: [u8; 4 * LIMB_BYTES] = [0; 4 * LIMB_BYTES];
                                    let mut expected =
                                        RT::from_bytes(_expected.as_mut_slice()).unwrap();
                                    expected.copy_from(&a);
                                    ct_mul_trunc_mp_mp(&mut expected, op_len, &b);
                                    ct_mod_mp_mp(
                                        None,
                                        &mut expected,
                                        &CtMpDivisor::new(&n).unwrap(),
                                    );
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
fn test_ct_montgomery_mul_mod_cond_be_be_be_be() {
    use super::limbs_buffer::MpBigEndianMutByteSlice;
    test_ct_montgomery_mul_mod_cond_mp_mp::<
        MpBigEndianMutByteSlice,
        MpBigEndianMutByteSlice,
        MpBigEndianMutByteSlice,
        MpBigEndianMutByteSlice,
    >()
}

#[test]
fn test_ct_montgomery_mul_mod_cond_le_le_le_le() {
    use super::limbs_buffer::MpLittleEndianMutByteSlice;
    test_ct_montgomery_mul_mod_cond_mp_mp::<
        MpLittleEndianMutByteSlice,
        MpLittleEndianMutByteSlice,
        MpLittleEndianMutByteSlice,
        MpLittleEndianMutByteSlice,
    >()
}

#[test]
fn test_ct_montgomery_mul_mod_cond_ne_ne_ne_ne() {
    use super::limbs_buffer::MpNativeEndianMutByteSlice;
    test_ct_montgomery_mul_mod_cond_mp_mp::<
        MpNativeEndianMutByteSlice,
        MpNativeEndianMutByteSlice,
        MpNativeEndianMutByteSlice,
        MpNativeEndianMutByteSlice,
    >()
}

pub fn ct_montgomery_mul_mod_mp_mp<
    RT: MpIntMutByteSlice,
    T0: MpIntByteSliceCommon,
    T1: MpIntByteSliceCommon,
    NT: MpIntByteSliceCommon,
>(
    result: &mut RT,
    op0: &T0,
    op1: &T1,
    n: &NT,
    neg_n0_inv_mod_l: LimbType,
) {
    ct_montgomery_mul_mod_cond_mp_mp(result, op0, op1, n, neg_n0_inv_mod_l, LimbChoice::from(1))
}

#[derive(Debug)]
pub enum CtMontgomeryTransformationError {
    InvalidModulus,
    InsufficientDestinationSpace,
}

pub fn ct_to_montgomery_form_direct_mp<TT: MpIntMutByteSlice, NT: MpIntByteSliceCommon>(
    t: &mut TT,
    n: &NT,
) -> Result<(), CtMontgomeryTransformationError> {
    debug_assert!(t.nlimbs() >= n.nlimbs());
    let radix_shift_len = ct_montgomery_radix_shift_len(n.len());
    let n = CtMpDivisor::new(n).map_err(|e| match e {
        CtMpDivisorError::DivisorIsZero => CtMontgomeryTransformationError::InvalidModulus,
    })?;
    ct_mod_lshifted_mp_mp(t, t.len(), radix_shift_len, &n).map_err(|e| match e {
        CtModLshiftedMpMpError::InsufficientRemainderSpace => {
            CtMontgomeryTransformationError::InsufficientDestinationSpace
        }
    })
}

pub fn ct_montgomery_radix2_mod_n_mp<RX2T: MpIntMutByteSlice, NT: MpIntByteSliceCommon>(
    radix2_mod_n_out: &mut RX2T,
    n: &NT,
) -> Result<(), CtMontgomeryTransformationError> {
    debug_assert!(ct_mp_nlimbs(radix2_mod_n_out.len()) >= ct_mp_nlimbs(n.len()));
    let radix_shift_len = ct_montgomery_radix_shift_len(n.len());
    let n = CtMpDivisor::new(n).map_err(|e| match e {
        CtMpDivisorError::DivisorIsZero => CtMontgomeryTransformationError::InvalidModulus,
    })?;
    ct_mod_pow2_mp::<_, _>(2 * 8 * radix_shift_len, radix2_mod_n_out, &n).map_err(|e| match e {
        CtModPow2MpError::InsufficientRemainderSpace => {
            CtMontgomeryTransformationError::InsufficientDestinationSpace
        }
    })
}

pub fn ct_to_montgomery_form_mp<
    MGT: MpIntMutByteSlice,
    TT: MpIntByteSliceCommon,
    NT: MpIntByteSliceCommon,
    RX2T: MpIntByteSliceCommon,
>(
    mg_t_out: &mut MGT,
    t: &TT,
    n: &NT,
    neg_n0_inv_mod_l: LimbType,
    radix2_mod_n: &RX2T,
) {
    ct_montgomery_mul_mod_mp_mp(mg_t_out, t, radix2_mod_n, n, neg_n0_inv_mod_l);
}

#[cfg(test)]
fn test_ct_to_montgomery_form_mp<
    TT: MpIntMutByteSlice,
    NT: MpIntMutByteSlice,
    RX2T: MpIntMutByteSlice,
>() {
    use super::cmp_impl::ct_eq_mp_mp;
    use super::div_impl::ct_mod_mp_mp;
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::MpIntMutByteSlicePriv as _;

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
                let neg_n0_inv = ct_montgomery_neg_n0_inv_mod_l_mp(&n);

                let mut radix2_mod_n: [u8; 2 * LIMB_BYTES] = [0xffu8; 2 * LIMB_BYTES];
                let mut radix2_mod_n = RX2T::from_bytes(radix2_mod_n.as_mut_slice()).unwrap();
                let (_, mut radix2_mod_n) = radix2_mod_n.split_at(RX2T::limbs_align_len(n_len));
                ct_montgomery_radix2_mod_n_mp(&mut radix2_mod_n, &n).unwrap();

                for k in 0..4 {
                    let a_high =
                        MERSENNE_PRIME_17.wrapping_mul((16383 as LimbType).wrapping_mul(k));
                    for l in 0..4 {
                        let a_low =
                            MERSENNE_PRIME_13.wrapping_mul((262175 as LimbType).wrapping_mul(l));
                        let mut a: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
                        let mut a = TT::from_bytes(a.as_mut_slice()).unwrap();
                        a.store_l(0, a_low);
                        a.store_l(1, a_high);
                        // All montgomery operations are defined mod n, compute a mod n
                        ct_mod_mp_mp(None, &mut a, &CtMpDivisor::new(&n).unwrap());
                        let (_, mut a) = a.split_at(TT::limbs_align_len(n_len));

                        let mut result: [u8; 2 * LIMB_BYTES] = [0xff; 2 * LIMB_BYTES];
                        let mut result = TT::from_bytes(result.as_mut_slice()).unwrap();
                        let (_, mut result) = result.split_at(TT::limbs_align_len(n_len));
                        let (_, mut result) = result.split_at(TT::limbs_align_len(n_len));
                        ct_to_montgomery_form_mp(&mut result, &a, &n, neg_n0_inv, &radix2_mod_n);

                        ct_to_montgomery_form_direct_mp(&mut a, &n).unwrap();
                        assert_eq!(ct_eq_mp_mp(&result, &a).unwrap(), 1);
                    }
                }
            }
        }
    }
}

#[test]
fn test_ct_to_montgomery_form_be_be_be() {
    use super::limbs_buffer::MpBigEndianMutByteSlice;
    test_ct_to_montgomery_form_mp::<
        MpBigEndianMutByteSlice,
        MpBigEndianMutByteSlice,
        MpBigEndianMutByteSlice,
    >()
}

#[test]
fn test_ct_to_montgomery_form_le_le_le() {
    use super::limbs_buffer::MpLittleEndianMutByteSlice;
    test_ct_to_montgomery_form_mp::<
        MpLittleEndianMutByteSlice,
        MpLittleEndianMutByteSlice,
        MpLittleEndianMutByteSlice,
    >()
}

#[test]
fn test_ct_to_montgomery_form_ne_ne_ne() {
    use super::limbs_buffer::MpNativeEndianMutByteSlice;
    test_ct_to_montgomery_form_mp::<
        MpNativeEndianMutByteSlice,
        MpNativeEndianMutByteSlice,
        MpNativeEndianMutByteSlice,
    >()
}

// result must have been initialized with a one in Montgomery form before the
// call.
fn _ct_montogmery_exp_mod_mp_mp<
    RT: MpIntMutByteSlice,
    T0: MpIntByteSliceCommon,
    NT: MpIntByteSliceCommon,
    ET: MpIntByteSliceCommon,
>(
    result: &mut RT,
    op0: &T0,
    n: &NT,
    neg_n0_inv_mod_l: LimbType,
    exponent: &ET,
    exponent_nbits: usize,
    scratch: &mut [u8],
) {
    debug_assert!(result.nlimbs() >= n.nlimbs());

    let scratch_len = MpNativeEndianMutByteSlice::limbs_align_len(n.len());
    debug_assert!(scratch.len() >= scratch_len);
    let (scratch, _) = scratch.split_at_mut(scratch_len);
    let mut scratch = MpNativeEndianMutByteSlice::from_bytes(scratch).unwrap();

    let exponent_nbits = exponent_nbits.min(8 * exponent.len());
    for i in 0..exponent_nbits {
        ct_montgomery_mul_mod_mp_mp(&mut scratch, result, result, n, neg_n0_inv_mod_l);
        ct_montgomery_mul_mod_cond_mp_mp(
            result,
            &scratch,
            op0,
            n,
            neg_n0_inv_mod_l,
            exponent.test_bit(exponent_nbits - i - 1),
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ct_montogmery_exp_mod_odd_mp_mp<
    RT: MpIntMutByteSlice,
    T0: MpIntByteSliceCommon,
    NT: MpIntByteSliceCommon,
    RXT: MpIntByteSliceCommon,
    ET: MpIntByteSliceCommon,
>(
    result: &mut RT,
    op0: &T0,
    n: &NT,
    neg_n0_inv_mod_l: LimbType,
    radix_mod_n: &RXT,
    exponent: &ET,
    exponent_nbits: usize,
    scratch: &mut [u8],
) {
    // Initialize the result with a one in Montgomery form.
    result.copy_from(radix_mod_n);

    _ct_montogmery_exp_mod_mp_mp(
        result,
        op0,
        n,
        neg_n0_inv_mod_l,
        exponent,
        exponent_nbits,
        scratch,
    );
}

pub fn ct_exp_mod_odd_mp_mp<
    RT: MpIntMutByteSlice,
    T0: MpIntMutByteSlice,
    NT: MpIntByteSliceCommon,
    ET: MpIntByteSliceCommon,
>(
    result: &mut RT,
    op0: &mut T0,
    n: &NT,
    exponent: &ET,
    exponent_nbits: usize,
    scratch: &mut [u8],
) {
    debug_assert!(result.nlimbs() >= n.nlimbs());
    debug_assert_ne!(ct_lt_mp_mp(op0, n).unwrap(), 0);

    let scratch_len = MpNativeEndianMutByteSlice::limbs_align_len(n.len());
    debug_assert!(scratch.len() >= scratch_len);

    let neg_n0_inv_mod_l = ct_montgomery_neg_n0_inv_mod_l_mp(n);
    // The radix squared mod n gets into result[], it will be reduced
    // later on to a one in Montgomery form.
    ct_montgomery_radix2_mod_n_mp(result, n).unwrap();

    // Transform op0 into Montgomery form, the function argument will get
    // overwritten to save an extra scratch buffer.
    let mut mg_op0 = MpNativeEndianMutByteSlice::from_bytes(scratch).unwrap();
    ct_to_montgomery_form_mp(&mut mg_op0, op0, n, neg_n0_inv_mod_l, result);
    op0.copy_from(&mg_op0);

    // Reduce the radix squared mod n in result[] to the radix mod n,
    // i.e. to a one in Montgomery form.
    ct_montgomery_redc_mp(result, n, neg_n0_inv_mod_l);

    // Do the Montgomery exponentiation.
    _ct_montogmery_exp_mod_mp_mp(
        result,
        op0,
        n,
        neg_n0_inv_mod_l,
        exponent,
        exponent_nbits,
        scratch,
    );

    // And transform the result back from Montgomery form.
    ct_montgomery_redc_mp(result, n, neg_n0_inv_mod_l);
}

#[cfg(test)]
fn test_ct_exp_mod_odd_mp_mp<
    RT: MpIntMutByteSlice,
    T0: MpIntMutByteSlice,
    NT: MpIntMutByteSlice,
    ET: MpIntMutByteSlice,
>() {
    use super::limb::LIMB_BYTES;
    use super::mul_impl::ct_mul_trunc_mp_l;
    use super::shift_impl::ct_lshift_mp;

    fn test_one<
        'a,
        RT: MpIntMutByteSlice,
        T0: MpIntMutByteSlice,
        NT: MpIntByteSliceCommon,
        ET: MpIntByteSliceCommon,
    >(
        op0: &T0,
        n: &NT,
        exponent: &'a ET,
    ) {
        use super::cmp_impl::ct_eq_mp_mp;
        use super::div_impl::ct_mod_mp_mp;
        use super::mul_impl::{ct_mul_trunc_mp_mp, ct_square_trunc_mp};

        let n_len = n.len();

        let op0_mod_n_aligned_len = T0::limbs_align_len(n_len.max(op0.len()));
        let mut op0_mod_n = vec![0u8; op0_mod_n_aligned_len];
        let mut op0_mod_n = T0::from_bytes(&mut op0_mod_n).unwrap();
        op0_mod_n.copy_from(op0);
        ct_mod_mp_mp(None, &mut op0_mod_n, &CtMpDivisor::new(n).unwrap());

        let mut op0_scratch = vec![0u8; T0::limbs_align_len(n_len)];
        let mut op0_scratch = T0::from_bytes(&mut op0_scratch).unwrap();
        op0_scratch.copy_from(&op0_mod_n);
        let mut result = vec![0u8; RT::limbs_align_len(n_len)];
        let mut result = RT::from_bytes(&mut result).unwrap();
        let mut scratch = vec![0u8; MpNativeEndianMutByteSlice::limbs_align_len(n_len)];
        ct_exp_mod_odd_mp_mp(
            &mut result,
            &mut op0_scratch,
            n,
            exponent,
            8 * exponent.len(),
            &mut scratch,
        );

        // Compute the expected value using repeated multiplications/squarings and
        // modular reductions.
        let mut expected = vec![0u8; RT::limbs_align_len(2 * n_len)];
        let mut expected = RT::from_bytes(&mut expected).unwrap();
        expected.clear_bytes_above(0);
        expected.store_l(0, 1);
        for i in 0..8 * exponent.len() {
            ct_square_trunc_mp(&mut expected, n_len);
            ct_mod_mp_mp(None, &mut expected, &CtMpDivisor::new(n).unwrap());
            if exponent.test_bit(8 * exponent.len() - i - 1).unwrap() != 0 {
                ct_mul_trunc_mp_mp(&mut expected, n_len, &op0_mod_n);
                ct_mod_mp_mp(None, &mut expected, &CtMpDivisor::new(n).unwrap());
            }
        }
        assert_ne!(ct_eq_mp_mp(&result, &expected).unwrap(), 0);
    }

    let exponent_len = ET::limbs_align_len(LIMB_BYTES + 1);
    let mut e0_buf = vec![0u8; exponent_len];
    let mut e1_buf = vec![0u8; exponent_len];
    let mut e1 = ET::from_bytes(&mut e1_buf).unwrap();
    e1.store_l(0, 1);
    drop(e1);
    let mut e2_buf = vec![0u8; exponent_len];
    let mut e2 = ET::from_bytes(&mut e2_buf).unwrap();
    e2.store_l(0, 2);
    drop(e2);
    let mut ef_buf = vec![!0u8; exponent_len];

    for n_len in [1, LIMB_BYTES + 1, 2 * LIMB_BYTES - 1, 3 * LIMB_BYTES] {
        let mut n = vec![0u8; NT::limbs_align_len(n_len)];
        let mut n = NT::from_bytes(&mut n).unwrap();
        n.store_l(0, 1);
        while n.load_l((n_len - 1) / LIMB_BYTES) >> (8 * ((n_len - 1) % LIMB_BYTES)) == 0 {
            ct_mul_trunc_mp_l(&mut n, n_len, 251);
        }

        for op0_len in 1..n_len {
            let mut op0 = vec![0u8; T0::limbs_align_len(n_len)];
            let mut op0 = T0::from_bytes(&mut op0).unwrap();
            op0.store_l(0, 1);
            for _ in 0..op0_len {
                ct_mul_trunc_mp_l(&mut op0, n_len, 241);
            }
            ct_lshift_mp(&mut op0, 8 * (n_len - op0_len));
            for e_buf in [&mut e0_buf, &mut e1_buf, &mut e2_buf, &mut ef_buf] {
                let e = ET::from_bytes(e_buf).unwrap();
                test_one::<RT, _, _, _>(&op0, &n, &e);
            }
        }
    }
}

#[test]
fn test_ct_exp_mod_odd_be_be_be_be() {
    use super::limbs_buffer::MpBigEndianMutByteSlice;
    test_ct_exp_mod_odd_mp_mp::<
        MpBigEndianMutByteSlice,
        MpBigEndianMutByteSlice,
        MpBigEndianMutByteSlice,
        MpBigEndianMutByteSlice,
    >()
}

#[test]
fn test_ct_exp_mod_odd_le_le_le_le() {
    use super::limbs_buffer::MpLittleEndianMutByteSlice;
    test_ct_exp_mod_odd_mp_mp::<
        MpLittleEndianMutByteSlice,
        MpLittleEndianMutByteSlice,
        MpLittleEndianMutByteSlice,
        MpLittleEndianMutByteSlice,
    >()
}

#[test]
fn test_ct_exp_mod_odd_ne_ne_ne_ne() {
    use super::limbs_buffer::MpNativeEndianMutByteSlice;
    test_ct_exp_mod_odd_mp_mp::<
        MpNativeEndianMutByteSlice,
        MpNativeEndianMutByteSlice,
        MpNativeEndianMutByteSlice,
        MpNativeEndianMutByteSlice,
    >()
}
