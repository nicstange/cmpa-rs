use super::add_impl::ct_sub_cond_mp_mp;
use super::cmp_impl::{ct_geq_mp_mp, ct_lt_mp_mp};
use super::div_impl::{
    ct_mod_lshifted_mp_mp, ct_mod_pow2_mp, CtModLshiftedMpMpError, CtModPow2MpError, CtMpDivisor,
    CtMpDivisorError,
};
use super::limb::{
    ct_add_l_l, ct_eq_l_l, ct_inv_mod_l, ct_lsb_mask_l, ct_mul_add_l_l_l_c, LimbChoice, LimbType,
    LIMB_BITS,
};
use super::limbs_buffer::{
    ct_mp_limbs_align_len, ct_mp_nlimbs, MpIntMutSlice, MpIntMutSlicePriv as _, MpIntSliceCommon,
    MpNativeEndianMutByteSlice,
};

fn ct_montgomery_radix_shift_len(n_len: usize) -> usize {
    ct_mp_limbs_align_len(n_len)
}

fn ct_montgomery_radix_shift_mp_nlimbs(n_len: usize) -> usize {
    ct_mp_nlimbs(n_len)
}

#[derive(Debug)]
pub enum CtMontgomeryNegN0InvModLMpError {
    InvalidModulus,
}

pub fn ct_montgomery_neg_n0_inv_mod_l_mp<NT: MpIntSliceCommon>(
    n: &NT,
) -> Result<LimbType, CtMontgomeryNegN0InvModLMpError> {
    if n.is_empty() {
        return Err(CtMontgomeryNegN0InvModLMpError::InvalidModulus);
    }
    let n0 = n.load_l(0);
    if ct_eq_l_l(n0 & 1, 0).unwrap() != 0 {
        return Err(CtMontgomeryNegN0InvModLMpError::InvalidModulus);
    }
    let n0_inv_mod_l = ct_inv_mod_l(n0);
    Ok((!n0_inv_mod_l).wrapping_add(1))
}

#[test]
fn test_ct_montgomery_neg_n0_inv_mod_l_mp() {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::{
        MpBigEndianMutByteSlice, MpIntMutSlice as _, MpIntMutSlicePriv as _,
    };

    for n0 in 0 as LimbType..128 {
        let n0 = 2 * n0 + 1;
        for j in 0..2048 {
            const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
            let v = MERSENNE_PRIME_13.wrapping_mul((511 as LimbType).wrapping_mul(j));
            let v = v << 8;
            let n0 = n0.wrapping_add(v);

            let mut n0_buf: [u8; LIMB_BYTES] = [0; LIMB_BYTES];
            let mut n = MpBigEndianMutByteSlice::from_slice(n0_buf.as_mut_slice()).unwrap();
            n.store_l(0, n0);
            let neg_n0_inv = ct_montgomery_neg_n0_inv_mod_l_mp(&n).unwrap();
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

#[derive(Debug)]
pub enum CtMontgomeryRedcMpError {
    InvalidModulus,
    InsufficientResultSpace,
    InputValueOutOfRange,
}

pub fn ct_montgomery_redc_mp<TT: MpIntMutSlice, NT: MpIntSliceCommon>(
    t: &mut TT,
    n: &NT,
    neg_n0_inv_mod_l: LimbType,
) -> Result<(), CtMontgomeryRedcMpError> {
    if n.is_empty() {
        return Err(CtMontgomeryRedcMpError::InvalidModulus);
    }
    let n0_val = n.load_l(0);
    if ct_eq_l_l(n0_val & 1, 0).unwrap() != 0 {
        return Err(CtMontgomeryRedcMpError::InvalidModulus);
    }
    if !n.len_is_compatible_with(t.len()) {
        return Err(CtMontgomeryRedcMpError::InsufficientResultSpace);
    }
    if !t.len_is_compatible_with(2 * n.len()) {
        return Err(CtMontgomeryRedcMpError::InputValueOutOfRange);
    }
    let t_nlimbs = t.nlimbs();
    let n_nlimbs = n.nlimbs();
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
    Ok(())
}

#[cfg(test)]
fn test_ct_montgomery_redc_mp<TT: MpIntMutSlice, NT: MpIntMutSlice>() {
    use super::div_impl::ct_mod_mp_mp;
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::MpIntMutSlice as _;

    for i in 0..64 {
        const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
        let n_high = MERSENNE_PRIME_13.wrapping_mul((16385 as LimbType).wrapping_mul(i));
        for j in 0..64 {
            const MERSENNE_PRIME_17: LimbType = 131071 as LimbType;
            let n_low = MERSENNE_PRIME_17.wrapping_mul((1023 as LimbType).wrapping_mul(j));
            // Force n_low odd.
            let n_low = n_low | 1;
            let mut n = tst_mk_mp_backing_vec!(NT, 2 * LIMB_BYTES);
            let mut n = NT::from_slice(n.as_mut_slice()).unwrap();
            n.store_l(0, n_low);
            n.store_l(1, n_high);
            let neg_n0_inv = ct_montgomery_neg_n0_inv_mod_l_mp(&n).unwrap();

            for k in 0..8 {
                let t_high = MERSENNE_PRIME_17.wrapping_mul((8191 as LimbType).wrapping_mul(k));
                for l in 0..8 {
                    let t_low =
                        MERSENNE_PRIME_13.wrapping_mul((131087 as LimbType).wrapping_mul(l));

                    let mut t = tst_mk_mp_backing_vec!(TT, 2 * LIMB_BYTES);
                    let mut t = TT::from_slice(t.as_mut_slice()).unwrap();
                    t.store_l(0, t_low);
                    t.store_l(1, t_high);

                    // All montgomery operations are defined mod n, compute t mod n
                    ct_mod_mp_mp(None, &mut t, &CtMpDivisor::new(&n).unwrap());
                    let t_low = t.load_l(0);
                    let t_high = t.load_l(1);

                    // To Montgomery form: t * R mod n
                    ct_to_montgomery_form_direct_mp(&mut t, &n).unwrap();

                    // And back to normal: (t * R mod n) / R mod n
                    ct_montgomery_redc_mp(&mut t, &n, neg_n0_inv).unwrap();
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

#[derive(Debug)]
pub enum CtMontgomeryMulModCondMpMpError {
    InvalidModulus,
    InsufficientResultSpace,
    InconsistentInputOperandLength,
}

pub fn ct_montgomery_mul_mod_cond_mp_mp<
    RT: MpIntMutSlice,
    T0: MpIntSliceCommon,
    T1: MpIntSliceCommon,
    NT: MpIntSliceCommon,
>(
    result: &mut RT,
    op0: &T0,
    op1: &T1,
    n: &NT,
    neg_n0_inv_mod_l: LimbType,
    cond: LimbChoice,
) -> Result<(), CtMontgomeryMulModCondMpMpError> {
    // This is an implementation of the "Finely Integrated Operand Scanning (FIOS)
    // Method" approach to fused multiplication and Montgomery reduction, as
    // described in "Analyzing and Comparing Montgomery Multiplication
    // Algorithm", IEEE Micro, 16(3):26-33, June 1996.
    if n.is_empty() {
        return Err(CtMontgomeryMulModCondMpMpError::InvalidModulus);
    }
    let n0_val = n.load_l(0);
    if ct_eq_l_l(n0_val & 1, 0).unwrap() != 0 {
        return Err(CtMontgomeryMulModCondMpMpError::InvalidModulus);
    }
    if !n.len_is_compatible_with(result.len()) {
        return Err(CtMontgomeryMulModCondMpMpError::InsufficientResultSpace);
    }
    debug_assert!(n.nlimbs() <= result.nlimbs());
    if !op0.len_is_compatible_with(n.len()) || !op1.len_is_compatible_with(n.len()) {
        return Err(CtMontgomeryMulModCondMpMpError::InconsistentInputOperandLength);
    }
    debug_assert!(op0.nlimbs() <= n.nlimbs());
    debug_assert!(ct_lt_mp_mp(op0, n).unwrap() != 0);
    debug_assert!(op1.nlimbs() <= n.nlimbs());
    debug_assert!(ct_lt_mp_mp(op1, n).unwrap() != 0);

    let op0_nlimbs = op0.nlimbs();
    let op1_nlimbs = op1.nlimbs();
    let n_nlimbs = n.nlimbs();
    debug_assert!(n0_val.wrapping_mul(neg_n0_inv_mod_l) == !0);

    result.clear_bytes_above(0);
    let mut result = result.shrink_to(n.len());
    debug_assert_eq!(result.nlimbs(), n.nlimbs());
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

    let result_geq_n = LimbChoice::from(result_carry) | ct_geq_mp_mp(&result, n);
    ct_sub_cond_mp_mp(&mut result, n, result_geq_n);
    debug_assert!(ct_geq_mp_mp(&result, n).unwrap() == 0);
    Ok(())
}

#[cfg(test)]
fn test_ct_montgomery_mul_mod_cond_mp_mp<
    RT: MpIntMutSlice,
    T0: MpIntMutSlice,
    T1: MpIntMutSlice,
    NT: MpIntMutSlice,
>() {
    use super::div_impl::ct_mod_mp_mp;
    use super::limb::LIMB_BYTES;
    use super::mul_impl::ct_mul_trunc_mp_mp;

    for i in 0..16 {
        const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
        let n_high = MERSENNE_PRIME_13.wrapping_mul((65543 as LimbType).wrapping_mul(i));
        for j in 0..16 {
            const MERSENNE_PRIME_17: LimbType = 131071 as LimbType;
            let n_low = MERSENNE_PRIME_17.wrapping_mul((4095 as LimbType).wrapping_mul(j));
            // Force n_low odd.
            let n_low = n_low | 1;
            let mut n_buf = tst_mk_mp_backing_vec!(NT, 2 * LIMB_BYTES);
            let mut n = NT::from_slice(n_buf.as_mut_slice()).unwrap();
            n.store_l(0, n_low);
            n.store_l(1, n_high);
            drop(n);
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
                let mut n_buf = n_buf.clone();
                let mut n = NT::from_slice(n_buf.as_mut_slice()).unwrap();
                n.clear_bytes_above(n_len);
                let n = n.shrink_to(n_len);
                let neg_n0_inv = ct_montgomery_neg_n0_inv_mod_l_mp(&n).unwrap();

                // r_mod_n = 2^(2 * LIMB_BITS) % n.
                let mut r_mod_n = tst_mk_mp_backing_vec!(RT, 3 * LIMB_BYTES);
                let mut r_mod_n = RT::from_slice(r_mod_n.as_mut_slice()).unwrap();
                r_mod_n.store_l_full(ct_montgomery_radix_shift_mp_nlimbs(n_len), 1);
                ct_mod_mp_mp(None, &mut r_mod_n, &CtMpDivisor::new(&n).unwrap());
                let r_mod_n = r_mod_n.shrink_to(n.len());

                for k in 0..4 {
                    let a_high =
                        MERSENNE_PRIME_17.wrapping_mul((16383 as LimbType).wrapping_mul(k));
                    for l in 0..4 {
                        let a_low =
                            MERSENNE_PRIME_13.wrapping_mul((262175 as LimbType).wrapping_mul(l));
                        let mut a_buf = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
                        let mut a = T0::from_slice(a_buf.as_mut_slice()).unwrap();
                        a.store_l(0, a_low);
                        a.store_l(1, a_high);
                        // All montgomery operations are defined mod n, compute a mod n
                        ct_mod_mp_mp(None, &mut a, &CtMpDivisor::new(&n).unwrap());
                        drop(a);
                        for s in 0..4 {
                            let b_high = MERSENNE_PRIME_13
                                .wrapping_mul((262175 as LimbType).wrapping_mul(s));
                            for t in 0..4 {
                                const MERSENNE_PRIME_19: LimbType = 524287 as LimbType;
                                let b_low = MERSENNE_PRIME_19
                                    .wrapping_mul((4095 as LimbType).wrapping_mul(t));
                                let mut b_buf = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
                                let mut b = T1::from_slice(b_buf.as_mut_slice()).unwrap();
                                b.store_l(0, b_low);
                                b.store_l(1, b_high);
                                // All montgomery operations are defined mod n, compute b mod n
                                ct_mod_mp_mp(None, &mut b, &CtMpDivisor::new(&n).unwrap());
                                drop(b);

                                for op_len in [0, 1 * LIMB_BYTES, n_len] {
                                    let mut a_buf = a_buf.clone();
                                    let mut a = T0::from_slice(a_buf.as_mut_slice()).unwrap();
                                    a.clear_bytes_above(op_len);
                                    let a = a.shrink_to(op_len);
                                    let mut b_buf = b_buf.clone();
                                    let mut b = T1::from_slice(b_buf.as_mut_slice()).unwrap();
                                    b.clear_bytes_above(op_len);
                                    let b = b.shrink_to(op_len);

                                    let mut _result = tst_mk_mp_backing_vec!(RT, 4 * LIMB_BYTES);
                                    let mut result =
                                        RT::from_slice(_result.as_mut_slice()).unwrap();
                                    let mut mg_mul_result = result.shrink_to(n_len);
                                    ct_montgomery_mul_mod_cond_mp_mp(
                                        &mut mg_mul_result,
                                        &a,
                                        &b,
                                        &n,
                                        neg_n0_inv,
                                        LimbChoice::from(0),
                                    )
                                    .unwrap();
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
                                    )
                                    .unwrap();
                                    drop(mg_mul_result);

                                    // For testing against the expected result computed using the
                                    // "conventional" methods only, multiply by r_mod_n -- this
                                    // avoids having to multiply
                                    // the conventional product by r^-1 mod n, which is
                                    // not known without implementing Euklid's algorithm.
                                    ct_mul_trunc_mp_mp(&mut result, n.len(), &r_mod_n);
                                    ct_mod_mp_mp(None, &mut result, &CtMpDivisor::new(&n).unwrap());
                                    drop(result);

                                    let mut _expected = tst_mk_mp_backing_vec!(RT, 4 * LIMB_BYTES);
                                    let mut expected =
                                        RT::from_slice(_expected.as_mut_slice()).unwrap();
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

pub type CtMontgomeryMulModMpMpError = CtMontgomeryMulModCondMpMpError;

pub fn ct_montgomery_mul_mod_mp_mp<
    RT: MpIntMutSlice,
    T0: MpIntSliceCommon,
    T1: MpIntSliceCommon,
    NT: MpIntSliceCommon,
>(
    result: &mut RT,
    op0: &T0,
    op1: &T1,
    n: &NT,
    neg_n0_inv_mod_l: LimbType,
) -> Result<(), CtMontgomeryMulModMpMpError> {
    ct_montgomery_mul_mod_cond_mp_mp(result, op0, op1, n, neg_n0_inv_mod_l, LimbChoice::from(1))
}

#[derive(Debug)]
pub enum CtMontgomeryTransformationError {
    InvalidModulus,
    InsufficientResultSpace,
}

pub fn ct_to_montgomery_form_direct_mp<TT: MpIntMutSlice, NT: MpIntSliceCommon>(
    t: &mut TT,
    n: &NT,
) -> Result<(), CtMontgomeryTransformationError> {
    if n.test_bit(0).unwrap() == 0 {
        return Err(CtMontgomeryTransformationError::InvalidModulus);
    }
    if !n.len_is_compatible_with(t.len()) {
        return Err(CtMontgomeryTransformationError::InsufficientResultSpace);
    }
    debug_assert!(t.nlimbs() >= n.nlimbs());
    let radix_shift_len = ct_montgomery_radix_shift_len(n.len());
    let n = CtMpDivisor::new(n).map_err(|e| match e {
        CtMpDivisorError::DivisorIsZero => {
            // n had been checked for being odd above, so should be unreachable, but play
            // safe.
            debug_assert!(false);
            CtMontgomeryTransformationError::InvalidModulus
        }
    })?;
    ct_mod_lshifted_mp_mp(t, t.len(), radix_shift_len, &n).map_err(|e| match e {
        CtModLshiftedMpMpError::InsufficientRemainderSpace => {
            // The result space had been checked at function entry already, but play safe.
            debug_assert!(false);
            CtMontgomeryTransformationError::InsufficientResultSpace
        }
    })?;
    Ok(())
}

pub fn ct_montgomery_radix2_mod_n_mp<RX2T: MpIntMutSlice, NT: MpIntSliceCommon>(
    radix2_mod_n_out: &mut RX2T,
    n: &NT,
) -> Result<(), CtMontgomeryTransformationError> {
    if n.test_bit(0).unwrap() == 0 {
        return Err(CtMontgomeryTransformationError::InvalidModulus);
    }
    if !n.len_is_compatible_with(radix2_mod_n_out.len()) {
        return Err(CtMontgomeryTransformationError::InsufficientResultSpace);
    }

    radix2_mod_n_out.clear_bytes_above(n.len());
    let mut radix2_mod_n_out = radix2_mod_n_out.shrink_to(n.len());
    debug_assert_eq!(radix2_mod_n_out.nlimbs(), n.nlimbs());

    let radix_shift_len = ct_montgomery_radix_shift_len(n.len());
    let n = CtMpDivisor::new(n).map_err(|e| match e {
        CtMpDivisorError::DivisorIsZero => {
            // n had been checked for being odd above, so should be unreachable, but play
            // safe.
            debug_assert!(false);
            CtMontgomeryTransformationError::InvalidModulus
        }
    })?;
    ct_mod_pow2_mp::<_, _>(2 * 8 * radix_shift_len, &mut radix2_mod_n_out, &n).map_err(
        |e| match e {
            CtModPow2MpError::InsufficientRemainderSpace => {
                // The result space had been checked at function entry already, but play safe.
                debug_assert!(false);
                CtMontgomeryTransformationError::InsufficientResultSpace
            }
        },
    )?;
    Ok(())
}

#[derive(Debug)]
pub enum CtToMontgomeryFormMpError {
    InvalidModulus,
    InsufficientResultSpace,
    InconsistentInputOperandLength,
    InconsistentRadix2ModNLenth,
}

pub fn ct_to_montgomery_form_mp<
    RT: MpIntMutSlice,
    TT: MpIntSliceCommon,
    NT: MpIntSliceCommon,
    RX2T: MpIntSliceCommon,
>(
    result: &mut RT,
    t: &TT,
    n: &NT,
    neg_n0_inv_mod_l: LimbType,
    radix2_mod_n: &RX2T,
) -> Result<(), CtToMontgomeryFormMpError> {
    // The Montgomery multiplication will do all error checking needed. However, to
    // disambiguate which of the two factors has a length inconsistent with n,
    // if any, check that here.
    if !t.len_is_compatible_with(n.len()) {
        return Err(CtToMontgomeryFormMpError::InconsistentInputOperandLength);
    }
    if !radix2_mod_n.len_is_compatible_with(n.len()) {
        return Err(CtToMontgomeryFormMpError::InconsistentRadix2ModNLenth);
    }
    debug_assert!(ct_lt_mp_mp(t, n).unwrap() != 0);
    debug_assert!(ct_lt_mp_mp(radix2_mod_n, n).unwrap() != 0);

    // All input arguments have been validated above, just unwrap().
    ct_montgomery_mul_mod_mp_mp(result, t, radix2_mod_n, n, neg_n0_inv_mod_l).map_err(
        |e| match e {
            CtMontgomeryMulModCondMpMpError::InsufficientResultSpace => {
                CtToMontgomeryFormMpError::InsufficientResultSpace
            }
            CtMontgomeryMulModCondMpMpError::InvalidModulus => {
                CtToMontgomeryFormMpError::InvalidModulus
            }
            CtMontgomeryMulModCondMpMpError::InconsistentInputOperandLength => {
                // The multiplication's factors have been validated above, but play safe.
                CtToMontgomeryFormMpError::InconsistentInputOperandLength
            }
        },
    )?;

    Ok(())
}

#[cfg(test)]
fn test_ct_to_montgomery_form_mp<TT: MpIntMutSlice, NT: MpIntMutSlice, RX2T: MpIntMutSlice>() {
    use super::cmp_impl::ct_eq_mp_mp;
    use super::div_impl::ct_mod_mp_mp;
    use super::limb::LIMB_BYTES;

    for i in 0..16 {
        const MERSENNE_PRIME_13: LimbType = 8191 as LimbType;
        let n_high = MERSENNE_PRIME_13.wrapping_mul((65543 as LimbType).wrapping_mul(i));
        for j in 0..16 {
            const MERSENNE_PRIME_17: LimbType = 131071 as LimbType;
            let n_low = MERSENNE_PRIME_17.wrapping_mul((4095 as LimbType).wrapping_mul(j));
            // Force n_low odd.
            let n_low = n_low | 1;
            let mut n_buf = tst_mk_mp_backing_vec!(NT, 2 * LIMB_BYTES);
            let mut n = NT::from_slice(n_buf.as_mut_slice()).unwrap();
            n.store_l(0, n_low);
            n.store_l(1, n_high);
            drop(n);
            let n_lengths = if !NT::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
                [LIMB_BYTES, 2 * LIMB_BYTES]
            } else {
                [2 * LIMB_BYTES - 1, 2 * LIMB_BYTES]
            };

            for n_len in n_lengths {
                let mut n_buf = n_buf.clone();
                let mut n = NT::from_slice(n_buf.as_mut_slice()).unwrap();
                n.clear_bytes_above(n_len);
                let n = n.shrink_to(n_len);
                let neg_n0_inv = ct_montgomery_neg_n0_inv_mod_l_mp(&n).unwrap();

                let mut radix2_mod_n = tst_mk_mp_backing_vec!(RX2T, n_len);
                let mut radix2_mod_n = RX2T::from_slice(radix2_mod_n.as_mut_slice()).unwrap();
                ct_montgomery_radix2_mod_n_mp(&mut radix2_mod_n, &n).unwrap();

                for k in 0..4 {
                    let a_high =
                        MERSENNE_PRIME_17.wrapping_mul((16383 as LimbType).wrapping_mul(k));
                    for l in 0..4 {
                        let a_low =
                            MERSENNE_PRIME_13.wrapping_mul((262175 as LimbType).wrapping_mul(l));
                        let mut a = tst_mk_mp_backing_vec!(TT, 2 * LIMB_BYTES);
                        let mut a = TT::from_slice(a.as_mut_slice()).unwrap();
                        a.store_l(0, a_low);
                        a.store_l(1, a_high);
                        // All montgomery operations are defined mod n, compute a mod n
                        ct_mod_mp_mp(None, &mut a, &CtMpDivisor::new(&n).unwrap());
                        let mut a = a.shrink_to(n_len);

                        let mut result = tst_mk_mp_backing_vec!(TT, n_len);
                        let mut result = TT::from_slice(result.as_mut_slice()).unwrap();
                        ct_to_montgomery_form_mp(&mut result, &a, &n, neg_n0_inv, &radix2_mod_n)
                            .unwrap();

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
    RT: MpIntMutSlice,
    T0: MpIntSliceCommon,
    NT: MpIntSliceCommon,
    ET: MpIntSliceCommon,
>(
    result: &mut RT,
    op0: &T0,
    n: &NT,
    neg_n0_inv_mod_l: LimbType,
    exponent: &ET,
    exponent_nbits: usize,
    scratch: &mut [u8],
) {
    debug_assert_eq!(result.nlimbs(), n.nlimbs());

    let scratch_len = MpNativeEndianMutByteSlice::limbs_align_len(n.len());
    debug_assert!(scratch.len() >= scratch_len);
    let mut scratch = MpNativeEndianMutByteSlice::from_slice(scratch).unwrap();
    scratch.clear_bytes_above(scratch_len);
    let mut scratch = scratch.shrink_to(scratch_len);

    let exponent_nbits = exponent_nbits.min(8 * exponent.len());
    for i in 0..exponent_nbits {
        // Input arguments have been validated/setup by callers, just unwrap() the
        // result.
        ct_montgomery_mul_mod_mp_mp(&mut scratch, result, result, n, neg_n0_inv_mod_l).unwrap();
        ct_montgomery_mul_mod_cond_mp_mp(
            result,
            &scratch,
            op0,
            n,
            neg_n0_inv_mod_l,
            exponent.test_bit(exponent_nbits - i - 1),
        )
        .unwrap();
    }
}

#[derive(Debug)]
pub enum CtMontgomeryExpModOddMpMpError {
    InvalidModulus,
    InsufficientResultSpace,
    InsufficientScratchSpace,
    InconsistentInputOperandLength,
    InconsistendRadixModNLengh,
}

#[allow(clippy::too_many_arguments)]
pub fn ct_montogmery_exp_mod_odd_mp_mp<
    RT: MpIntMutSlice,
    T0: MpIntSliceCommon,
    NT: MpIntSliceCommon,
    RXT: MpIntSliceCommon,
    ET: MpIntSliceCommon,
>(
    result: &mut RT,
    op0: &T0,
    n: &NT,
    neg_n0_inv_mod_l: LimbType,
    radix_mod_n: &RXT,
    exponent: &ET,
    exponent_nbits: usize,
    scratch: &mut [u8],
) -> Result<(), CtMontgomeryExpModOddMpMpError> {
    if n.test_bit(0).unwrap() == 0 {
        return Err(CtMontgomeryExpModOddMpMpError::InvalidModulus);
    }
    if !n.len_is_compatible_with(result.len()) {
        return Err(CtMontgomeryExpModOddMpMpError::InsufficientResultSpace);
    }
    if scratch.len() < MpNativeEndianMutByteSlice::limbs_align_len(n.len()) {
        return Err(CtMontgomeryExpModOddMpMpError::InsufficientScratchSpace);
    }
    if !op0.len_is_compatible_with(n.len()) {
        return Err(CtMontgomeryExpModOddMpMpError::InconsistentInputOperandLength);
    }
    debug_assert!(ct_lt_mp_mp(op0, n).unwrap() != 0);
    if !radix_mod_n.len_is_compatible_with(n.len()) {
        return Err(CtMontgomeryExpModOddMpMpError::InconsistendRadixModNLengh);
    }
    debug_assert!(ct_lt_mp_mp(radix_mod_n, n).unwrap() != 0);

    // Initialize the result with a one in Montgomery form.
    result.clear_bytes_above(n.len());
    let mut result = result.shrink_to(n.len());
    debug_assert_eq!(result.nlimbs(), n.nlimbs());
    result.copy_from(radix_mod_n);

    _ct_montogmery_exp_mod_mp_mp(
        &mut result,
        op0,
        n,
        neg_n0_inv_mod_l,
        exponent,
        exponent_nbits,
        scratch,
    );
    Ok(())
}

#[derive(Debug)]
pub enum CtExpModOddMpMpError {
    InvalidModulus,
    InsufficientResultSpace,
    InsufficientScratchSpace,
    InconsistentInputOperandLength,
}

pub fn ct_exp_mod_odd_mp_mp<
    RT: MpIntMutSlice,
    T0: MpIntMutSlice,
    NT: MpIntSliceCommon,
    ET: MpIntSliceCommon,
>(
    result: &mut RT,
    op0: &mut T0,
    n: &NT,
    exponent: &ET,
    exponent_nbits: usize,
    scratch: &mut [u8],
) -> Result<(), CtExpModOddMpMpError> {
    if !n.len_is_compatible_with(result.len()) {
        return Err(CtExpModOddMpMpError::InsufficientResultSpace);
    }
    if !op0.len_is_compatible_with(n.len()) {
        return Err(CtExpModOddMpMpError::InconsistentInputOperandLength);
    }
    debug_assert!(ct_lt_mp_mp(op0, n).unwrap() != 0);
    if !n.len_is_compatible_with(op0.len()) {
        // op0 will get transformed in-place into Montgomery form. So the
        // backing byte slice must be large enough.
        return Err(CtExpModOddMpMpError::InconsistentInputOperandLength);
    }
    debug_assert_eq!(op0.nlimbs(), n.nlimbs());
    let scratch_len = MpNativeEndianMutByteSlice::limbs_align_len(n.len());
    if scratch.len() < scratch_len {
        return Err(CtExpModOddMpMpError::InsufficientScratchSpace);
    }

    // This checks the modulus for validity as a side-effect.
    let neg_n0_inv_mod_l = ct_montgomery_neg_n0_inv_mod_l_mp(n).map_err(|e| match e {
        CtMontgomeryNegN0InvModLMpError::InvalidModulus => CtExpModOddMpMpError::InvalidModulus,
    })?;

    // Shrink result[] to the length of n. It will be used to temporarily hold the
    // radix^2 mod n and ct_to_montgomery_form_mp() below would complain if its
    // length is unexpectedly large.
    result.clear_bytes_above(n.len());
    let mut result = result.shrink_to(n.len());
    debug_assert_eq!(result.nlimbs(), n.nlimbs());

    // The radix squared mod n gets into result[], it will be reduced
    // later on to a one in Montgomery form.
    ct_montgomery_radix2_mod_n_mp(&mut result, n).unwrap();

    // Transform op0 into Montgomery form, the function argument will get
    // overwritten to save an extra scratch buffer.
    let mut mg_op0 = MpNativeEndianMutByteSlice::from_slice(scratch).unwrap();
    mg_op0.clear_bytes_above(scratch_len);
    let mut mg_op0 = mg_op0.shrink_to(scratch_len);
    ct_to_montgomery_form_mp(&mut mg_op0, op0, n, neg_n0_inv_mod_l, &result).unwrap();
    op0.copy_from(&mg_op0);

    // Reduce the radix squared mod n in result[] to the radix mod n,
    // i.e. to a one in Montgomery form.
    ct_montgomery_redc_mp(&mut result, n, neg_n0_inv_mod_l).unwrap();

    // Do the Montgomery exponentiation.
    _ct_montogmery_exp_mod_mp_mp(
        &mut result,
        op0,
        n,
        neg_n0_inv_mod_l,
        exponent,
        exponent_nbits,
        scratch,
    );

    // And transform the result back from Montgomery form.
    ct_montgomery_redc_mp(&mut result, n, neg_n0_inv_mod_l).unwrap();
    Ok(())
}

#[cfg(test)]
fn test_ct_exp_mod_odd_mp_mp<
    RT: MpIntMutSlice,
    T0: MpIntMutSlice,
    NT: MpIntMutSlice,
    ET: MpIntMutSlice,
>() {
    extern crate alloc;
    use super::limb::LIMB_BYTES;
    use super::mul_impl::ct_mul_trunc_mp_l;
    use super::shift_impl::ct_lshift_mp;
    use alloc::vec;

    fn test_one<
        'a,
        RT: MpIntMutSlice,
        T0: MpIntMutSlice,
        NT: MpIntSliceCommon,
        ET: MpIntSliceCommon,
    >(
        op0: &T0,
        n: &NT,
        exponent: &'a ET,
    ) {
        use super::cmp_impl::ct_eq_mp_mp;
        use super::div_impl::ct_mod_mp_mp;
        use super::mul_impl::{ct_mul_trunc_mp_mp, ct_square_trunc_mp};

        let n_len = n.len();

        let mut op0_mod_n = tst_mk_mp_backing_vec!(T0, n_len.max(op0.len()));
        let mut op0_mod_n = T0::from_slice(&mut op0_mod_n).unwrap();
        op0_mod_n.copy_from(op0);
        ct_mod_mp_mp(None, &mut op0_mod_n, &CtMpDivisor::new(n).unwrap());

        let mut op0_scratch = tst_mk_mp_backing_vec!(T0, n_len);
        let mut op0_scratch = T0::from_slice(&mut op0_scratch).unwrap();
        op0_scratch.copy_from(&op0_mod_n);
        let mut result = tst_mk_mp_backing_vec!(RT, n_len);
        let mut result = RT::from_slice(&mut result).unwrap();
        let mut scratch = vec![0u8; MpNativeEndianMutByteSlice::limbs_align_len(n_len)];
        ct_exp_mod_odd_mp_mp(
            &mut result,
            &mut op0_scratch,
            n,
            exponent,
            8 * exponent.len(),
            &mut scratch,
        )
        .unwrap();

        // Compute the expected value using repeated multiplications/squarings and
        // modular reductions.
        let mut expected = tst_mk_mp_backing_vec!(RT, 2 * n_len);
        let mut expected = RT::from_slice(&mut expected).unwrap();
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

    let exponent_len = LIMB_BYTES + 1;
    let mut e0_buf = tst_mk_mp_backing_vec!(ET, exponent_len);
    let mut e1_buf = tst_mk_mp_backing_vec!(ET, exponent_len);
    let mut e1 = ET::from_slice(&mut e1_buf).unwrap();
    e1.store_l(0, 1);
    drop(e1);
    let mut e2_buf = tst_mk_mp_backing_vec!(ET, exponent_len);
    let mut e2 = ET::from_slice(&mut e2_buf).unwrap();
    e2.store_l(0, 2);
    drop(e2);
    let mut ef_buf = tst_mk_mp_backing_vec!(ET, exponent_len);
    ef_buf.fill(0xefu8.into());

    for n_len in [1, LIMB_BYTES + 1, 2 * LIMB_BYTES - 1, 3 * LIMB_BYTES] {
        let mut n = tst_mk_mp_backing_vec!(NT, n_len);
        let mut n = NT::from_slice(&mut n).unwrap();
        n.store_l(0, 1);
        while n.load_l((n_len - 1) / LIMB_BYTES) >> (8 * ((n_len - 1) % LIMB_BYTES)) == 0 {
            ct_mul_trunc_mp_l(&mut n, n_len, 251);
        }

        for op0_len in 1..n_len {
            let mut op0 = tst_mk_mp_backing_vec!(T0, n_len);
            let mut op0 = T0::from_slice(&mut op0).unwrap();
            op0.store_l(0, 1);
            for _ in 0..op0_len {
                ct_mul_trunc_mp_l(&mut op0, n_len, 241);
            }
            ct_lshift_mp(&mut op0, 8 * (n_len - op0_len));
            for e_buf in [&mut e0_buf, &mut e1_buf, &mut e2_buf, &mut ef_buf] {
                let e = ET::from_slice(e_buf).unwrap();
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
