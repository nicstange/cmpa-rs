use core::array;

use super::cmp_impl::{ct_gt_mp_mp, ct_is_one_mp, ct_is_zero_mp, ct_lt_mp_mp, CtGeqMpMpKernel};
use super::limb::{
    ct_add_l_l, ct_add_l_l_c, ct_arithmetic_rshift_l, ct_is_nonzero_l, ct_mul_add_l_l_l_c,
    ct_mul_l_l, ct_sub_l_l, LimbChoice, LimbType, LIMB_BITS,
};
use super::limbs_buffer::{
    ct_find_first_set_bit_mp, ct_swap_cond_mp, MpIntByteSliceCommon, MpIntByteSliceCommonPriv as _,
    MpIntMutByteSlice, MpNativeEndianMutByteSlice,
};
use super::montgomery_impl::{
    ct_montgomery_neg_n0_inv_mod_l_mp, CtMontgomeryNegN0InvModLMpError, CtMontgomeryRedcKernel,
};
use super::shift_impl::{ct_lshift_mp, ct_rshift_mp};
use super::usize_ct_cmp::ct_lt_usize_usize;

// Implementation of Euclid's algorithm after
// [BERN_YANG19] "Fast constant-time gcd computation and modular inversion",
//                Daniel J. Bernstein and Bo-Yin Yang, IACR Transactions on
//                Cryptographic Hardware and Embedded Systems ISSN 2569-2925,
//                Vol. 2019, No. 3, pp. 340-398
//
// The main advantages over other common binary gcd algorithms for the purposes
// here are
// - only the least significant bit determines the operations to be carried out
//   in each step,
// - steps can be batched into transition matrices
// - and, last but not least, it's particularly constant-time implementation
//   friendly overall.
//
// [BERN_YANG19] define transition matrices describing their divstep(\delta, f,
// g) step: /  \                     / \
// |f'|                     |f|
// |  | = T(\delta, f, g) * | |
// |g'|                     |g|
// \  /                     \ /
// with
//                   /        \   /
//                   | 1    0 |   | [[0, 1], [-1, 1]]    if \delta > 0 and g odd
// T(\delta, f, g) = |        | * |
//                   | 0  1/2 |   | [[1, 0], [g % 2, 1]] otherwise.
//                   \        /   \
// c.f. p. 361.
//
// The matrices in the second operand of the product on the right hand side
// above both have inf-norm == 2, so multiplying any against some other matrix
// (like a product of these) at most doubles the maximum of the entries' values.
// Representing negative values in two's complement as stored in in (unsigned)
// LimbTypes and tracking the shifting factor from the first operand on the
// RHS separately allows for batching up to LIMB_BITS - 2 divstep() invocations
// and accumulate them in such a matrix (without overflow into the sign bit)
// before applying them to MP integers.

struct TransitionMatrix {
    t: [[LimbType; 2]; 2],
    row_shift: [LimbType; 2],
}

impl TransitionMatrix {
    fn identity() -> Self {
        Self {
            t: [[1, 0], [0, 1]],
            row_shift: [0, 0],
        }
    }

    // Multiply a single step transition matrix from the left against self.
    fn push_transition(&mut self, case0: LimbChoice, g0: LimbChoice) {
        // The original first row is either to be added to or subtracted from
        // the second row. Save it away for later before updating it.
        let t0: [LimbType; 2] = self.t[0];
        let row_shift_diff = self.row_shift[1] - self.row_shift[0];

        // The new first row is either the original second row (case0 == true)
        // or the original first one (case0 == false).
        for j in [0, 1] {
            self.t[0][j] = case0.select(self.t[0][j], self.t[1][j])
        }
        self.row_shift[0] = case0.select(self.row_shift[0], self.row_shift[1]);

        // Compute the new second row.
        let case0_mask = case0.select(0, !0);
        // If not case0, then the original first row is multiplied by g & 1 (== g0)
        // before getting added to the second one.
        let t0_mask = case0.select(g0.select(0, !0), !0);
        for j in [0, 1] {
            let mut t0j = t0[j];
            // If case0 == true, then negate the original first row to be added to the
            // second row.
            t0j ^= case0_mask; // Conditional bitwise negation.
            t0j = t0j.wrapping_add(case0.unwrap()); // And add one to finally obtain -t0j.

            // If case0 == false, clear out the original first row in case g & 1 == 0.
            t0j &= t0_mask;

            self.t[1][j] = self.t[1][j].wrapping_add(t0j << row_shift_diff);
        }
        self.row_shift[1] += 1;
    }

    // Apply the transition matrix to the column vector (f, g), for which its is
    // guaranteed by construction that the matrix' row shifts (or rather the
    // powers of two thereof) evenly divide the intermediate results after plain
    // multiplication. {f,g}_shadow_head[0] shadows the potentially partial high
    // limbs of f and g respectively, {f,g}_shadow_head[1] is used to store
    // temporary excess (before right-shifting) and two's complement sign bits.
    fn apply_to_f_g<FT: MpIntMutByteSlice, GT: MpIntMutByteSlice>(
        &self,
        t_is_neg_mask: &[[LimbType; 2]; 2],
        f_shadow_head: &mut [LimbType; 2],
        f: &mut FT,
        g_shadow_head: &mut [LimbType; 2],
        g: &mut GT,
    ) {
        let nlimbs = f.nlimbs();
        assert_eq!(nlimbs, g.nlimbs());
        assert!(nlimbs != 0);

        let row_shift_mask: [LimbType; 2] = [
            LimbChoice::from(ct_is_nonzero_l(self.row_shift[0])).select(0, !0),
            LimbChoice::from(ct_is_nonzero_l(self.row_shift[1])).select(0, !0),
        ];

        let mut carry: [[LimbType; 2]; 2] = [[0; 2]; 2];
        let mut borrow: [LimbType; 2] = [0; 2];
        let mut last_orig_val: [LimbType; 2] = [0; 2];
        let mut last_new_val: [LimbType; 2] = [0; 2];

        // Compute one limb of either the new f or g.
        // All less significant limbs are assumed to have been computed,
        // already with a resulting and carry and borrow as passed here.
        // cur_orig[] contains the original limbs of both, f and g, at
        // the current position, last_orig[] the ones from the previous,
        // less significant position.
        fn mat_mul_row_compute_limb(
            carry: &mut [LimbType; 2],
            borrow: &mut LimbType,
            cur_orig_val: &[LimbType; 2],
            last_orig_val: &[LimbType; 2],
            t_row: &[LimbType; 2],
            t_row_is_neg_mask: &[LimbType; 2],
        ) -> LimbType {
            // t_{i, f} * f
            let t_if_f = ct_mul_l_l(t_row[0], cur_orig_val[0]);
            let (f_carry, result) = (t_if_f.high(), t_if_f.low());
            // + t_{i, g} * g + carry[0]
            let (g_carry, result) = ct_mul_add_l_l_l_c(result, t_row[1], cur_orig_val[1], carry[0]);
            // Update the carry for the next round.
            (carry[1], carry[0]) = ct_add_l_l_c(f_carry, g_carry, carry[1]);
            // If t_{i, f} is negative, don't sign extend it all the way up to
            // nlimbs, but simply account for it by subtracting f shifted by one
            // limb position to the left: write the (virtually) sign extended t_{i, f} as
            // 2^LIMB_BITS * (-1) + t_{i, f} and multiply by f.
            let (f_borrow, result) = ct_sub_l_l(result, last_orig_val[0] & t_row_is_neg_mask[0]);
            debug_assert!(f_borrow == 0 || result >= 1);
            // Similar for negative t_{i, g}.
            let (g_borrow, result) = ct_sub_l_l(result, last_orig_val[1] & t_row_is_neg_mask[1]);
            debug_assert!(g_borrow == 0 || result >= 1);
            debug_assert!((f_borrow + g_borrow) < 2 || result >= 2);
            debug_assert!(*borrow <= 2); // Loop invariant.
            let (borrow0, result) = ct_sub_l_l(result, *borrow);
            *borrow = borrow0 + f_borrow + g_borrow;
            debug_assert!(*borrow <= 2); // Loop invariant still true.
            result
        }

        let mut k = 0;
        while k + 1 < nlimbs {
            let cur_orig_val: [LimbType; 2] = [f.load_l_full(k), g.load_l_full(k)];
            let mut new_val: [LimbType; 2] = [0; 2];
            for i in 0..2 {
                new_val[i] = mat_mul_row_compute_limb(
                    &mut carry[i],
                    &mut borrow[i],
                    &cur_orig_val,
                    &last_orig_val,
                    &self.t[i],
                    &t_is_neg_mask[i],
                );
            }

            if k > 0 {
                // Apply shift and store the result limbs back to f and g respectively.
                for i in 0..2 {
                    let s = self.row_shift[i];
                    let sm = row_shift_mask[i];
                    last_new_val[i] >>= s;
                    last_new_val[i] |= (new_val[i] << ((LIMB_BITS as LimbType - s) & sm)) & sm;
                }
                f.store_l_full(k - 1, last_new_val[0]);
                g.store_l_full(k - 1, last_new_val[1]);
            }

            last_orig_val = cur_orig_val;
            last_new_val = new_val;
            k += 1;
        }

        // Process {f,g}_shadow_head[].
        for k in 0..2 {
            let cur_orig_val: [LimbType; 2] = [f_shadow_head[k], g_shadow_head[k]];
            let mut new_val: [LimbType; 2] = [0; 2];
            for i in 0..2 {
                new_val[i] = mat_mul_row_compute_limb(
                    &mut carry[i],
                    &mut borrow[i],
                    &cur_orig_val,
                    &last_orig_val,
                    &self.t[i],
                    &t_is_neg_mask[i],
                );
            }

            // Apply shift and store the previous result limbs back to either f/g or to
            // {f,g}_shadow_head[0] respectively.
            for i in 0..2 {
                let s = self.row_shift[i];
                let sm = row_shift_mask[i];
                last_new_val[i] >>= s;
                last_new_val[i] |= (new_val[i] << ((LIMB_BITS as LimbType - s) & sm)) & sm;
            }
            if k == 0 && nlimbs > 1 {
                f.store_l_full(nlimbs - 2, last_new_val[0]);
                g.store_l_full(nlimbs - 2, last_new_val[1]);
            } else if k == 1 {
                f_shadow_head[0] = last_new_val[0];
                g_shadow_head[0] = last_new_val[1];
            }

            last_new_val = new_val;
            last_orig_val = cur_orig_val;
        }

        // Apply shift and store the high result limbs back to to {f,g}_shadow_head[1]
        // respectively.
        for (i, last_new_val) in last_new_val.iter_mut().enumerate() {
            let s = self.row_shift[i];
            *last_new_val = ct_arithmetic_rshift_l(*last_new_val, s);
            debug_assert!(*last_new_val == 0 || *last_new_val == !0);
        }
        f_shadow_head[1] = last_new_val[0];
        g_shadow_head[1] = last_new_val[1];
    }

    // Multiplicate the matrix times a column vector and reduce the result modulo n.
    // n needs to be odd as a prerequisite of the fused Montgomery reductions.
    fn apply_to_mod_odd_n<
        UFT: MpIntMutByteSlice,
        UGT: MpIntMutByteSlice,
        NT: MpIntByteSliceCommon,
    >(
        &self,
        t_is_neg_mask: &[[LimbType; 2]; 2],
        u_f: &mut UFT,
        u_g: &mut UGT,
        n: &NT,
        neg_n0_inv_mod_l: LimbType,
    ) {
        // The matrix' rows are to be scaled by their associated 1/2^row_shift[i] each.
        // Unlike the primary f and g, which are getting reduced iteratively to
        // their final GCD, the new column vector's entry are not evenly
        // divisible by this power of two in general. Apply a Montgomery
        // reduction to divide by 2^row_shift[i] mod n. In fact, there's one individual
        // Montgomery reduction running for each of the two rows respectively, of
        // course. In order to take advantage of cache locality, the Montgomery
        // reductions are fused with the row * column multiplication loop over
        // the result entries' limbs.

        fn mat_mul_row_scale_mod_odd_n_compute_start(
            u_first_orig: &[LimbType; 2],
            t_row: &[LimbType; 2],
            row_shift: u32,
            n0_val: LimbType,
            neg_n0_inv_mod_l: LimbType,
        ) -> ([LimbType; 2], CtMontgomeryRedcKernel) {
            let mut carry: [LimbType; 2] = [0; 2];
            // t_{i, f} * u_f
            let u_i_f = ct_mul_l_l(t_row[0], u_first_orig[0]);
            let (carry_i_f, u_i_f) = (u_i_f.high(), u_i_f.low());
            // t_{i, f} * u_f + t_{i, g} * u_g
            let (carry_i_g, u_i) = ct_mul_add_l_l_l_c(u_i_f, t_row[1], u_first_orig[1], 0);
            (carry[1], carry[0]) = ct_add_l_l(carry_i_f, carry_i_g);
            debug_assert!(carry[1] <= 1);

            let redc_kernel =
                CtMontgomeryRedcKernel::start(row_shift, u_i, n0_val, neg_n0_inv_mod_l);
            (carry, redc_kernel)
        }

        #[allow(clippy::too_many_arguments)]
        fn mat_mul_row_scale_mod_odd_n_compute_limb(
            carry: &mut [LimbType; 2],
            borrow: &mut LimbType,
            redc_kernel: &mut CtMontgomeryRedcKernel,
            n_val: LimbType,
            u_cur_orig_val: &[LimbType; 2],
            u_last_orig_val: &[LimbType; 2],
            t_row: &[LimbType; 2],
            t_row_is_neg_mask: &[LimbType; 2],
        ) -> LimbType {
            // t_{i, f} * u_f
            let u_i_f = ct_mul_l_l(t_row[0], u_cur_orig_val[0]);
            let (carry_i_f, u_i_f) = (u_i_f.high(), u_i_f.low());
            // t_{i, f} * u_f + t_{i, g} * u_g + carry[0]
            let (carry_i_g, u_i) = ct_mul_add_l_l_l_c(u_i_f, t_row[1], u_cur_orig_val[1], carry[0]);

            // Update the carry for the next round.
            (carry[1], carry[0]) = ct_add_l_l_c(carry_i_f, carry_i_g, carry[1]);

            // If t_{i, f} is negative, don't sign extend it all the way up to
            // nlimbs, but simply account for it by subtracting u_f shifted by one
            // limb position to the left: write the (virtually) sign extended t_{i, f} as
            // 2^LIMB_BITS * (-1) + t_{i, f} and multiply by u_f.
            let (borrow_i_f, u_i) = ct_sub_l_l(u_i, u_last_orig_val[0] & t_row_is_neg_mask[0]);
            debug_assert!(borrow_i_f == 0 || u_i >= 1);
            // Analoguous for negative t_{i, g}.
            let (borrow_i_g, u_i) = ct_sub_l_l(u_i, u_last_orig_val[1] & t_row_is_neg_mask[1]);
            debug_assert!(borrow_i_f + borrow_i_g < 2 || u_i >= 2);

            // Apply the borrow from last iteration and update it for the next round
            debug_assert!(*borrow <= 2); // Loop invariant.
            let (borrow0, u_i) = ct_sub_l_l(u_i, *borrow);
            *borrow = borrow0 + borrow_i_f + borrow_i_g;
            debug_assert!(*borrow <= 2); // Loop invariant still true.

            redc_kernel.update(u_i, n_val)
        }

        debug_assert!(!n.is_empty());
        let n_nlimbs = n.nlimbs();
        debug_assert!(u_f.nlimbs() == n_nlimbs);
        debug_assert_eq!(ct_lt_mp_mp(u_f, n).unwrap(), 1);
        debug_assert!(u_g.nlimbs() == n_nlimbs);
        debug_assert_eq!(ct_lt_mp_mp(u_g, n).unwrap(), 1);

        let n0_val = n.load_l(0);
        let mut u_last_orig_val: [LimbType; 2] = [u_f.load_l(0), u_g.load_l(0)];
        let mut carry: [[LimbType; 2]; 2] = [[0; 2]; 2];
        let mut redc_kernel: [CtMontgomeryRedcKernel; 2] = array::from_fn(|i: usize| {
            let redc_kernel;
            (carry[i], redc_kernel) = mat_mul_row_scale_mod_odd_n_compute_start(
                &u_last_orig_val,
                &self.t[i],
                self.row_shift[i] as u32,
                n0_val,
                neg_n0_inv_mod_l,
            );
            redc_kernel
        });

        // In order to improve cache locality, compare the resulting (redced) column
        // vector entries against n[] on the fly each. (For positive values, the
        // Montgomery reduction yiels a value in the range [0, 2*n[] and a n
        // needs to get conditionally subtracted to bring it into range).
        let mut u_geq_n_kernel: [CtGeqMpMpKernel; 2] = array::from_fn(|_| CtGeqMpMpKernel::new());
        let mut last_n_val = n0_val;

        let mut borrow: [LimbType; 2] = [0; 2];
        let mut j = 0;
        while j + 1 < n_nlimbs {
            let n_val = n.load_l(j + 1);
            let u_cur_orig_val: [LimbType; 2] = [u_f.load_l(j + 1), u_g.load_l(j + 1)];
            let mut u_new_val: [LimbType; 2] = [0; 2];
            for i in 0..2 {
                let u_i_new_val = mat_mul_row_scale_mod_odd_n_compute_limb(
                    &mut carry[i],
                    &mut borrow[i],
                    &mut redc_kernel[i],
                    n_val,
                    &u_cur_orig_val,
                    &u_last_orig_val,
                    &self.t[i],
                    &t_is_neg_mask[i],
                );
                u_new_val[i] = u_i_new_val;
                u_geq_n_kernel[i].update(u_i_new_val, last_n_val);
            }
            u_f.store_l_full(j, u_new_val[0]);
            u_g.store_l_full(j, u_new_val[1]);
            u_last_orig_val = u_cur_orig_val;
            last_n_val = n_val;
            j += 1;
        }
        debug_assert_eq!(j + 1, n_nlimbs);

        let mut u_head_new_val: [LimbType; 2] = [0; 2];
        for i in 0..2 {
            let u_i = carry[i][0];
            // If t_i_f/t_i_g is negative, subtract the corresponding high limb of the
            // original u_f/u_g from the value, c.f. the comment in
            // mat_mul_row_scale_mod_n_compute_limb().
            let (borrow_i_f, u_i_new_val) =
                ct_sub_l_l(u_i, u_last_orig_val[0] & t_is_neg_mask[i][0]);
            debug_assert!(borrow_i_f == 0 || u_i_new_val >= 1);
            // Analoguous for negative t_{i, g}.
            let (borrow_i_g, u_i_new_val) =
                ct_sub_l_l(u_i_new_val, u_last_orig_val[1] & t_is_neg_mask[i][1]);
            debug_assert!(borrow_i_f + borrow_i_g < 2 || u_i_new_val >= 2);

            // Apply the borrow from last iteration and update it for the next round
            debug_assert!(borrow[i] <= 2); // Loop (on j from above) invariant.
            let (borrow0, u_i_new_val) = ct_sub_l_l(u_i_new_val, borrow[i]);
            borrow[i] = borrow0 + borrow_i_f + borrow_i_g;
            debug_assert!(borrow[i] <= 2); // Loop invariant still true.

            // The scaled transition matrices always have (induced and sub-multiplicative)
            // maximum norm <= 1, meaning that their application to a row
            // vector, like (u_f, u_g) does not increase the maximum over the
            // absolute values of the two resulting entries. Upon entry to this
            // function, the invariant 0 <= u_f, u_g < n had been true by assumption.
            // From that, it follows that the unscaled resulting column vectors have
            // absolute values < 2^row_shift[i] * |n|,
            // with row_shift[i] <= STEPS_PER_BATCH == LIMB_BITS - 2.
            // u_i currently holds the (unscaled) top limb with such an column entry vector,
            // carry[1] and borrow any carry or borrow on its left.
            // With the upper bounds from above, it follows that the unscaled result in
            // u_i does not exceed its width; in particular, the effective borrow equals
            // its high (sign) bit.
            debug_assert!(carry[i][1] == 0 || borrow[i] != 0);
            debug_assert!(borrow[i] - carry[i][1] <= 1);
            debug_assert_eq!(borrow[i] - carry[i][1], u_i_new_val >> (LIMB_BITS - 1));
            u_head_new_val[i] = u_i_new_val;
        }
        // Final reduction outside of the loop over i in [0, 1] above, because the
        // operation consumes the kernels and this requires destructuring of the
        // redc_kernel[] array.
        let [redc_kernel_u_f, redc_kernel_u_g] = redc_kernel;
        let [mut u_f_geq_n_kernel, mut u_g_geq_n_kernel] = u_geq_n_kernel;
        let (u_f_sign, u_f_new_val) = redc_kernel_u_f.finish_in_twos_complement(u_head_new_val[0]);
        u_f_geq_n_kernel.update(u_f_new_val, last_n_val);
        u_f.store_l(j, u_f_new_val & u_f.partial_high_mask());
        let (u_g_sign, u_g_new_val) = redc_kernel_u_g.finish_in_twos_complement(u_head_new_val[1]);
        u_g_geq_n_kernel.update(u_g_new_val, last_n_val);
        u_g.store_l(j, u_g_new_val & u_g.partial_high_mask());

        // The resulting, redced entries are int the range (-n[], 2 * n): the not yet
        // reduced values are in the range 2^row_shift[i] * (-n[], n) and the
        // Montgomery reduction adds a (positive) value in the range
        // (0, 2^row_shift[i] * n[]) to that before dividing by 2^row_shift[i].
        let u_f_geq_n = u_f_geq_n_kernel.finish().select(0, u_f_sign ^ 1);
        let u_g_geq_n = u_g_geq_n_kernel.finish().select(0, u_g_sign ^ 1);

        // Either add or subtract one n (or don't do anything at all), depending on
        // u_i_sign and u_i_geq_n.
        fn u_i_into_range<UT: MpIntMutByteSlice, NT: MpIntByteSliceCommon>(
            u_i: &mut UT,
            u_i_sign: LimbType,
            u_i_geq_n: LimbType,
            n: &NT,
        ) {
            // If u_i_sign, n[] needs to get added to u_i[].
            // If u_i_geq_n, n[] needs to get subtracted from u_i[]. Negate it by standard
            // two's complement negation, !n[] + 1, before adding it to u_i[].
            let neg_n = LimbChoice::from(u_i_geq_n);
            let neg_n_mask = neg_n.select(0, !0);
            let mut neg_n_carry = neg_n.select(0, 1);
            let cond_mask = LimbChoice::from(u_i_sign | u_i_geq_n).select(0, !0);
            let mut carry = 0;
            for i in 0..n.nlimbs() {
                let n_val;
                (neg_n_carry, n_val) = ct_add_l_l(n.load_l(i) ^ neg_n_mask, neg_n_carry);
                let new_u_i_val;
                (carry, new_u_i_val) = ct_add_l_l_c(u_i.load_l(i), n_val & cond_mask, carry);
                if i + 1 != u_i.nlimbs() {
                    u_i.store_l_full(i, new_u_i_val);
                } else {
                    u_i.store_l(i, new_u_i_val & u_i.partial_high_mask());
                }
            }
            debug_assert_eq!(ct_gt_mp_mp(n, u_i).unwrap(), 1);
        }

        u_i_into_range(u_f, u_f_sign, u_f_geq_n, n);
        u_i_into_range(u_g, u_g_sign, u_g_geq_n, n);
    }

    fn t_is_neg_mask(&self) -> [[LimbType; 2]; 2] {
        let t_f_f_is_neg = LimbChoice::from(self.t[0][0] >> (LIMB_BITS - 1));
        let t_f_g_is_neg = LimbChoice::from(self.t[0][1] >> (LIMB_BITS - 1));
        let t_g_f_is_neg = LimbChoice::from(self.t[1][0] >> (LIMB_BITS - 1));
        let t_g_g_is_neg = LimbChoice::from(self.t[1][1] >> (LIMB_BITS - 1));
        [
            [t_f_f_is_neg.select(0, !0), t_f_g_is_neg.select(0, !0)],
            [t_g_f_is_neg.select(0, !0), t_g_g_is_neg.select(0, !0)],
        ]
    }
}

const STEPS_PER_BATCH: usize = LIMB_BITS as usize - 2;

// Given the previous delta, and the least significant limbs of f and g
// respectively, determine STEPS_PER_BATCH steps and accumulate them in a single
// transition matrix.
fn batch_divsteps(
    mut delta: usize,
    mut f_low: LimbType,
    mut g_low: LimbType,
) -> (usize, TransitionMatrix) {
    debug_assert_eq!(f_low & 1, 1);
    let mut m = TransitionMatrix::identity();
    for _ in 0..STEPS_PER_BATCH {
        let delta_is_pos =
            LimbChoice::from((delta.wrapping_neg() >> (usize::BITS - 1)) as LimbType);
        debug_assert!(delta == 0 || delta > isize::MAX as usize || delta_is_pos.unwrap() == 1);
        debug_assert!(!(delta == 0 || delta > isize::MAX as usize) || delta_is_pos.unwrap() == 0);
        let g0 = LimbChoice::from(g_low & 1);
        let case0 = delta_is_pos & g0;
        m.push_transition(case0, g0);

        delta = case0.select_usize((1_usize).wrapping_add(delta), (1_usize).wrapping_sub(delta));
        let new_f_low = case0.select(f_low, g_low);
        // Calculate the new g.
        let case0_mask = case0.select(0, !0);
        let f_low_mask = case0.select(g0.select(0, !0), !0);
        // If case0 == true, then negate the original f_low to be added to g_val..
        f_low ^= case0_mask; // Conditional bitwise negation.
        f_low = f_low.wrapping_add(case0.unwrap()); // And add one to finally obtain -f_low.

        // If case0 == false, clear out the original f_low in case g_low & 1 == 0.
        f_low &= f_low_mask;

        let new_g_low = ct_arithmetic_rshift_l(g_low.wrapping_add(f_low), 1);
        f_low = new_f_low;
        g_low = new_g_low;
    }

    (delta, m)
}

fn nbatches(len: usize) -> usize {
    // For the minimum number of steps, c.f. [BERN_YANG19], Theorem 11.2.
    // Assuming len = f.len().min()(g.len()), then bits == len * 8, but the second
    // operand, g, is assumed to be even initially and to have been halved
    // before getting input to the GCD, hence the +1 below.
    let nbits = len * 8 + 1;
    let d = nbits + 1; // The +1 to account for taking the log2 of the Euclidean norm of (f, g).
    let nsteps = if d < 46 {
        (49 * d + 80) / 17
    } else {
        (49 * d + 57) / 17
    };
    (nsteps + STEPS_PER_BATCH - 1) / STEPS_PER_BATCH
}

fn ct_gcd_ext_odd_mp_mp<
    UES: FnMut(&TransitionMatrix, &[[LimbType; 2]; 2]),
    FT: MpIntMutByteSlice,
    GT: MpIntMutByteSlice,
>(
    mut update_ext_state: UES,
    f: &mut FT,
    g: &mut GT,
) -> LimbChoice {
    let nlimbs = f.nlimbs();
    debug_assert_eq!(nlimbs, g.nlimbs());
    debug_assert!(nlimbs != 0);
    debug_assert_eq!(f.load_l(0) & 1, 1);
    let nbatches = nbatches(f.len().max(g.len()));

    // One shadow limb is for shadowing potentially partial high limbs of f or g
    // respectively and another one for excess precision needed for temporary
    // intermediate values due to the batching and also, for two's complement
    // sign bits.
    let mut f_shadow_head: [LimbType; 2] = [f.load_l(nlimbs - 1), 0];
    let mut g_shadow_head: [LimbType; 2] = [g.load_l(nlimbs - 1), 0];
    let mut delta: usize = 1;
    for _ in 0..nbatches {
        let (f_low, g_low) = if nlimbs > 1 {
            (f.load_l_full(0), g.load_l_full(0))
        } else {
            (f_shadow_head[0], g_shadow_head[0])
        };
        debug_assert_eq!(f_low & 1, 1);
        let transition_matrix;
        (delta, transition_matrix) = batch_divsteps(delta, f_low, g_low);
        let t_is_neg_mask = transition_matrix.t_is_neg_mask();
        transition_matrix.apply_to_f_g(
            &t_is_neg_mask,
            &mut f_shadow_head,
            f,
            &mut g_shadow_head,
            g,
        );
        update_ext_state(&transition_matrix, &t_is_neg_mask);
    }

    debug_assert_eq!(g_shadow_head[0], 0);
    g.store_l(nlimbs - 1, 0);
    debug_assert_eq!(g_shadow_head[1], 0);
    for k in 0..nlimbs {
        debug_assert_eq!(g.load_l(k), 0);
    }

    debug_assert!(f_shadow_head[1] == 0 || f_shadow_head[1] == !0);
    let f_is_neg = LimbChoice::from(f_shadow_head[1] >> (LIMB_BITS - 1));
    f.store_l(nlimbs - 1, f_shadow_head[0] & f.partial_high_mask());

    f_is_neg
}

#[derive(Debug)]
pub enum CtGcdOddMpMpError {
    InconsistentInputOperandLengths,
    InvalidInputOperandValue,
}

pub fn ct_gcd_odd_mp_mp<FT: MpIntMutByteSlice, GT: MpIntMutByteSlice>(
    f: &mut FT,
    g: &mut GT,
) -> Result<(), CtGcdOddMpMpError> {
    if f.test_bit(0).unwrap() == 0 {
        return Err(CtGcdOddMpMpError::InvalidInputOperandValue);
    }
    if !f.len_is_compatible_with(g.len()) || !g.len_is_compatible_with(f.len()) {
        return Err(CtGcdOddMpMpError::InconsistentInputOperandLengths);
    }

    let gcd_is_neg = ct_gcd_ext_odd_mp_mp(|_, _| {}, f, g);

    // If the GCD came out as negative, negate before returning.
    let neg_mask = gcd_is_neg.select(0, !0);
    let mut carry = gcd_is_neg.select(0, 1);
    let nlimbs = f.nlimbs();
    let mut k = 0;
    while k + 1 < nlimbs {
        let mut f_val = f.load_l_full(k);
        f_val ^= neg_mask;
        (carry, f_val) = ct_add_l_l(f_val, carry);
        f.store_l_full(k, f_val);
        k += 1;
    }
    let mut f_val = f.load_l(nlimbs - 1);
    f_val ^= neg_mask;
    (_, f_val) = ct_add_l_l(f_val, carry);
    f.store_l(nlimbs - 1, f_val & f.partial_high_mask());
    Ok(())
}

#[cfg(test)]
fn test_ct_gcd_odd_mp_mp<FT: MpIntMutByteSlice, GT: MpIntMutByteSlice>() {
    extern crate alloc;
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::{MpIntByteSliceCommon as _, MpIntByteSliceCommonPriv as _};
    use super::mul_impl::ct_mul_trunc_mp_l;
    use alloc::vec;

    fn assert_mp_is_equal<T: MpIntMutByteSlice>(v: &T, expected: LimbType) {
        assert_eq!(v.load_l(0), expected);
        for i in 1..v.nlimbs() {
            assert_eq!(v.load_l(i), 0);
        }
    }

    for l in [LIMB_BYTES - 1, 2 * LIMB_BYTES - 1] {
        let f_len = FT::limbs_align_len(l);
        let g_len = GT::limbs_align_len(l);
        let f_g_min_len = f_len.min(g_len);
        let f_g_high_min_len = (f_g_min_len - 1) % LIMB_BYTES + 1;

        let mut f_buf = vec![0u8; f_len];
        let mut g_buf = vec![0u8; g_len];
        let mut f = FT::from_bytes(f_buf.as_mut_slice()).unwrap();
        let mut g = GT::from_bytes(g_buf.as_mut_slice()).unwrap();
        f.store_l(0, 1);
        ct_gcd_odd_mp_mp(&mut f, &mut g).unwrap();
        assert_mp_is_equal(&f, 1);
        assert_mp_is_equal(&g, 0);

        for i in 1..8 * LIMB_BYTES.min(f_g_min_len) - 1 {
            for j in 1..8 * LIMB_BYTES.min(f_g_min_len) - 1 {
                let mut f_buf = vec![0u8; f_len];
                let mut g_buf = vec![0u8; g_len];
                let mut f = FT::from_bytes(f_buf.as_mut_slice()).unwrap();
                let mut g = GT::from_bytes(g_buf.as_mut_slice()).unwrap();
                f.set_bit_to(i, true);
                f.set_bit_to(0, true);
                g.set_bit_to(j, true);
                ct_gcd_odd_mp_mp(&mut f, &mut g).unwrap();
                assert_mp_is_equal(&f, 1);
                assert_mp_is_equal(&g, 0);
            }
        }

        let mut f_buf = vec![0u8; f_len];
        let mut g_buf = vec![0u8; g_len];
        let mut f = FT::from_bytes(f_buf.as_mut_slice()).unwrap();
        let mut g = GT::from_bytes(g_buf.as_mut_slice()).unwrap();
        f.store_l(0, 3 * 3 * 3 * 7);
        g.store_l(0, 3 * 3 * 5);
        ct_gcd_odd_mp_mp(&mut f, &mut g).unwrap();
        assert_mp_is_equal(&f, 3 * 3);
        assert_mp_is_equal(&g, 0);

        let mut f_buf = vec![0u8; f_len];
        let mut g_buf = vec![0u8; g_len];
        let mut f = FT::from_bytes(f_buf.as_mut_slice()).unwrap();
        let mut g = GT::from_bytes(g_buf.as_mut_slice()).unwrap();
        f.store_l(0, 3 * 3 * 5);
        g.store_l(0, 3 * 3 * 3 * 7);
        ct_gcd_odd_mp_mp(&mut f, &mut g).unwrap();
        assert_mp_is_equal(&f, 3 * 3);
        assert_mp_is_equal(&g, 0);

        let mut f_buf = vec![0u8; f_len];
        let mut g_buf = vec![0u8; g_len];
        let mut f = FT::from_bytes(f_buf.as_mut_slice()).unwrap();
        let mut g = GT::from_bytes(g_buf.as_mut_slice()).unwrap();
        f.store_l(0, 1);
        while f.load_l(f.nlimbs() - 1) >> 8 * (f_g_high_min_len - 1) == 0 {
            ct_mul_trunc_mp_l(&mut f, f_len, 251);
        }
        g.store_l(0, 1);
        while g.load_l(g.nlimbs() - 1) >> 8 * (f_g_high_min_len - 1) == 0 {
            ct_mul_trunc_mp_l(&mut g, g_len, 241);
        }
        while g.load_l(g.nlimbs() - 1) >> 8 * (f_g_high_min_len) - 1 == 0 {
            ct_mul_trunc_mp_l(&mut g, g_len, 2);
        }
        ct_gcd_odd_mp_mp(&mut f, &mut g).unwrap();
        assert_mp_is_equal(&f, 1);
        assert_mp_is_equal(&g, 0);

        let mut f_buf = vec![0u8; f_len];
        let mut g_buf = vec![0u8; g_len];
        let mut f = FT::from_bytes(f_buf.as_mut_slice()).unwrap();
        let mut g = GT::from_bytes(g_buf.as_mut_slice()).unwrap();
        f.store_l(0, 1);
        while f.load_l(f.nlimbs() - 1) >> 8 * (f_g_high_min_len - 2) == 0 {
            ct_mul_trunc_mp_l(&mut f, f_len, 251);
        }
        ct_mul_trunc_mp_l(&mut f, f_len, 241);
        g.store_l(0, 1);
        while g.load_l(g.nlimbs() - 1) >> 8 * (f_g_high_min_len - 1) == 0 {
            ct_mul_trunc_mp_l(&mut g, g_len, 241);
        }
        while g.load_l(g.nlimbs() - 1) >> 8 * (f_g_high_min_len) - 1 == 0 {
            ct_mul_trunc_mp_l(&mut g, g_len, 2);
        }
        ct_gcd_odd_mp_mp(&mut f, &mut g).unwrap();
        assert_mp_is_equal(&f, 241);
        assert_mp_is_equal(&g, 0);

        let mut f_buf = vec![0u8; f_len];
        let mut g_buf = vec![0u8; g_len];
        let mut f = FT::from_bytes(f_buf.as_mut_slice()).unwrap();
        let mut g = GT::from_bytes(g_buf.as_mut_slice()).unwrap();
        f.store_l(0, 1);
        while f.load_l(f.nlimbs() - 1) >> 8 * (f_g_high_min_len - 1) == 0 {
            ct_mul_trunc_mp_l(&mut f, f_len, 251);
        }
        g.store_l(0, 1);
        while g.load_l(g.nlimbs() - 1) >> 8 * (f_g_high_min_len - 2) == 0 {
            ct_mul_trunc_mp_l(&mut g, g_len, 241);
        }
        ct_mul_trunc_mp_l(&mut g, g_len, 251);
        while g.load_l(g.nlimbs() - 1) >> 8 * (f_g_high_min_len) - 1 == 0 {
            ct_mul_trunc_mp_l(&mut g, g_len, 2);
        }
        ct_gcd_odd_mp_mp(&mut f, &mut g).unwrap();
        assert_mp_is_equal(&f, 251);
        assert_mp_is_equal(&g, 0);
    }
}

#[test]
fn test_ct_gcd_odd_be_be() {
    use super::limbs_buffer::MpBigEndianMutByteSlice;
    test_ct_gcd_odd_mp_mp::<MpBigEndianMutByteSlice, MpBigEndianMutByteSlice>()
}

#[test]
fn test_ct_gcd_odd_le_le() {
    use super::limbs_buffer::MpLittleEndianMutByteSlice;
    test_ct_gcd_odd_mp_mp::<MpLittleEndianMutByteSlice, MpLittleEndianMutByteSlice>()
}

#[test]
fn test_ct_gcd_odd_ne_ne() {
    use super::limbs_buffer::MpNativeEndianMutByteSlice;
    test_ct_gcd_odd_mp_mp::<MpNativeEndianMutByteSlice, MpNativeEndianMutByteSlice>()
}

#[derive(Debug)]
pub enum CtGcdMpMpError {
    InsufficientResultSpace,
    InconsistentInputOperandLengths,
}

pub fn ct_gcd_mp_mp<T0: MpIntMutByteSlice, T1: MpIntMutByteSlice>(
    op0: &mut T0,
    op1: &mut T1,
) -> Result<(), CtGcdMpMpError> {
    // In case both op0 and op1 are zero, op0 is forced to one below.
    if op0.is_empty() {
        return Err(CtGcdMpMpError::InsufficientResultSpace);
    }
    if !op0.len_is_compatible_with(op1.len()) || !op1.len_is_compatible_with(op0.len()) {
        return Err(CtGcdMpMpError::InconsistentInputOperandLengths);
    }

    // The GCD implementation requires the first operand to be odd. Factor out
    // common powers of two.
    let (op0_is_nonzero, op0_powers_of_two) = ct_find_first_set_bit_mp(op0);
    let (op1_is_nonzero, op1_powers_of_two) = ct_find_first_set_bit_mp(op1);
    debug_assert!(op0_is_nonzero.unwrap() != 0 || op0_powers_of_two == 0);
    let op1_has_fewer_powers_of_two = !op0_is_nonzero // If op0 == 0, consider op1 only.
        | op1_is_nonzero & ct_lt_usize_usize(op1_powers_of_two, op0_powers_of_two);
    let min_powers_of_two =
        op1_has_fewer_powers_of_two.select_usize(op0_powers_of_two, op1_powers_of_two);
    ct_rshift_mp(op0, min_powers_of_two);
    ct_rshift_mp(op1, min_powers_of_two);
    ct_swap_cond_mp(op0, op1, op1_has_fewer_powers_of_two);
    let op0_and_op1_zero = !op0_is_nonzero & !op1_is_nonzero;
    debug_assert!(op0_and_op1_zero.unwrap() != 0 || op0.load_l(0) & 1 == 1);
    // If both inputs are zero, force op0 to 1, the GCD needs that.
    op0.store_l(0, op0_and_op1_zero.select(op0.load_l(0), 1));
    ct_gcd_odd_mp_mp(op0, op1).unwrap();
    // Now the GCD (odd factors only) is in op0.
    debug_assert_eq!(ct_is_zero_mp(op0).unwrap(), 0);
    debug_assert!(op0_and_op1_zero.unwrap() == 0 || ct_is_one_mp(op0).unwrap() != 0);
    debug_assert!(op0_and_op1_zero.unwrap() == 0 || min_powers_of_two == 0);
    // Scale the GCD by the common powers of two to obtain the final result.
    ct_lshift_mp(op0, min_powers_of_two);
    Ok(())
}

#[cfg(test)]
fn test_ct_gcd_mp_mp<T0: MpIntMutByteSlice, T1: MpIntMutByteSlice>() {
    extern crate alloc;
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::MpIntByteSliceCommon as _;
    use super::mul_impl::ct_mul_trunc_mp_l;
    use alloc::vec;

    fn test_one<T0: MpIntMutByteSlice, T1: MpIntMutByteSlice, GT: MpIntMutByteSlice>(
        op0: &T0,
        op1: &T1,
        gcd: &GT,
    ) {
        use super::cmp_impl::ct_eq_mp_mp;
        use super::mul_impl::ct_mul_trunc_mp_mp;

        let gcd_len = gcd.len();
        let op0_len = op0.len() + gcd_len;
        let op1_len = op1.len() + gcd_len;

        let op_max_len = op0_len.max(op1_len);
        let op_max_aligned_len = T0::limbs_align_len(op_max_len);
        let op_max_aligned_len = T1::limbs_align_len(op_max_aligned_len);
        let mut op0_gcd_work = vec![0u8; op_max_aligned_len];
        let mut op0_gcd_work = T0::from_bytes(&mut op0_gcd_work).unwrap();
        op0_gcd_work.copy_from(op0);
        ct_mul_trunc_mp_mp(&mut op0_gcd_work, op0.len(), gcd);
        let mut op1_gcd_work = vec![0u8; op_max_aligned_len];
        let mut op1_gcd_work = T1::from_bytes(&mut op1_gcd_work).unwrap();
        op1_gcd_work.copy_from(op1);
        ct_mul_trunc_mp_mp(&mut op1_gcd_work, op1.len(), gcd);

        ct_gcd_mp_mp(&mut op0_gcd_work, &mut op1_gcd_work).unwrap();

        let op0_is_zero = ct_is_zero_mp(op0).unwrap() != 0;
        let op1_is_zero = ct_is_zero_mp(op1).unwrap() != 0;
        if ct_is_zero_mp(gcd).unwrap() != 0 || (op0_is_zero && op1_is_zero) {
            // If both operands are zero, the result is fixed to 1 for definiteness.
            assert_ne!(ct_is_one_mp(&op0_gcd_work).unwrap(), 0);
        } else if op0_is_zero {
            // If op0 == 0, but op1 is not, the result is expected to equal the latter.
            let mut expected = vec![0u8; T1::limbs_align_len(op1.len() + gcd.len())];
            let mut expected = T1::from_bytes(&mut expected).unwrap();
            expected.copy_from(op1);
            ct_mul_trunc_mp_mp(&mut expected, op1.len(), gcd);
            assert_ne!(ct_eq_mp_mp(&mut op0_gcd_work, &expected).unwrap(), 0);
        } else if op1_is_zero {
            // If op1 == 0, but op0 is not, the result is expected to equal the latter.
            let mut expected = vec![0u8; T1::limbs_align_len(op0.len() + gcd.len())];
            let mut expected = T1::from_bytes(&mut expected).unwrap();
            expected.copy_from(op0);
            ct_mul_trunc_mp_mp(&mut expected, op0.len(), gcd);
            assert_ne!(ct_eq_mp_mp(&mut op0_gcd_work, &expected).unwrap(), 0);
        } else {
            assert_ne!(ct_eq_mp_mp(&mut op0_gcd_work, gcd).unwrap(), 0);
        }
    }

    for i in [
        0,
        LIMB_BYTES - 1,
        LIMB_BYTES,
        2 * LIMB_BYTES,
        2 * LIMB_BYTES + 1,
    ] {
        for j in [
            0,
            LIMB_BYTES - 1,
            LIMB_BYTES,
            2 * LIMB_BYTES,
            2 * LIMB_BYTES + 1,
        ] {
            for k in [
                0,
                LIMB_BYTES - 1,
                LIMB_BYTES,
                2 * LIMB_BYTES,
                2 * LIMB_BYTES + 1,
            ] {
                for total_shift in [
                    0,
                    LIMB_BITS - 1,
                    LIMB_BITS,
                    2 * LIMB_BITS,
                    2 * LIMB_BITS + 1,
                ] {
                    for l in 0..total_shift as usize {
                        let gcd_shift = l.min(total_shift as usize - l);
                        let op0_shift = l - gcd_shift;
                        let op1_shift = total_shift as usize - l - gcd_shift;

                        let op0_len = i + (op0_shift + 7) / 8;
                        let mut op0 = vec![0u8; T0::limbs_align_len(op0_len)];
                        let mut op0 = T0::from_bytes(&mut op0).unwrap();
                        if !op0.is_empty() {
                            op0.store_l(0, 1);
                        }
                        for _ in 0..i {
                            ct_mul_trunc_mp_l(&mut op0, op0_len, 251);
                        }
                        ct_lshift_mp(&mut op0, op0_shift);

                        let op1_len = j + (op1_shift + 7) / 8;
                        let mut op1 = vec![0u8; T1::limbs_align_len(op1_len)];
                        let mut op1 = T1::from_bytes(&mut op1).unwrap();
                        if !op1.is_empty() {
                            op1.store_l(0, 1);
                        }
                        for _ in 0..j {
                            ct_mul_trunc_mp_l(&mut op1, op1_len, 241);
                        }
                        ct_lshift_mp(&mut op1, op1_shift);

                        let gcd_len = k + (gcd_shift + 7) / 8;
                        let mut gcd = vec![0u8; T0::limbs_align_len(gcd_len)];
                        let mut gcd = T0::from_bytes(&mut gcd).unwrap();
                        if !gcd.is_empty() {
                            gcd.store_l(0, 1);
                        }
                        for _ in 0..k {
                            ct_mul_trunc_mp_l(&mut gcd, gcd_len, 239);
                        }
                        ct_lshift_mp(&mut gcd, gcd_shift);

                        test_one(&op0, &op1, &gcd);
                    }
                }
            }
        }
    }
}

#[test]
fn test_ct_gcd_be_be() {
    use super::limbs_buffer::MpBigEndianMutByteSlice;
    test_ct_gcd_mp_mp::<MpBigEndianMutByteSlice, MpBigEndianMutByteSlice>()
}

#[test]
fn test_ct_gcd_le_le() {
    use super::limbs_buffer::MpLittleEndianMutByteSlice;
    test_ct_gcd_mp_mp::<MpLittleEndianMutByteSlice, MpLittleEndianMutByteSlice>()
}

#[test]
fn test_ct_gcd_ne_ne() {
    use super::limbs_buffer::MpNativeEndianMutByteSlice;
    test_ct_gcd_mp_mp::<MpNativeEndianMutByteSlice, MpNativeEndianMutByteSlice>()
}

#[derive(Debug)]
pub enum CtInvModOddMpMpError {
    InvalidModulus,
    InsufficientResultSpace,
    InsufficientScratchSpace,
    InconsistentInputOperandLength,
    OperandsNotCoprime,
}

pub fn ct_inv_mod_odd_mp_mp<
    RT: MpIntMutByteSlice,
    T0: MpIntMutByteSlice,
    NT: MpIntByteSliceCommon,
>(
    result: &mut RT,
    op0: &mut T0,
    n: &NT,
    scratch: [&mut [u8]; 2],
) -> Result<(), CtInvModOddMpMpError> {
    if !n.len_is_compatible_with(result.len()) {
        return Err(CtInvModOddMpMpError::InsufficientResultSpace);
    }
    if !op0.len_is_compatible_with(n.len()) || !n.len_is_compatible_with(op0.len()) {
        return Err(CtInvModOddMpMpError::InconsistentInputOperandLength);
    }

    let n_aligned_len = MpNativeEndianMutByteSlice::limbs_align_len(n.len());
    for s in scratch.iter() {
        if s.len() < n_aligned_len {
            return Err(CtInvModOddMpMpError::InsufficientScratchSpace);
        }
    }

    let neg_n0_inv_mod_l = ct_montgomery_neg_n0_inv_mod_l_mp(n).map_err(|e| match e {
        CtMontgomeryNegN0InvModLMpError::InvalidModulus => CtInvModOddMpMpError::InvalidModulus,
    })?;

    // The column vector (ext_u0, ext_u1) tracking the coefficents of the extended
    // Euclidean algorithm for the modular inversion case will be stored in
    // (result, ext_u1_scratch).
    let [f_work_scratch, ext_u1_scratch] = scratch;
    let (f_work_scratch, _) = f_work_scratch.split_at_mut(n_aligned_len);
    let mut f_work_scratch = MpNativeEndianMutByteSlice::from_bytes(f_work_scratch).unwrap();
    f_work_scratch.copy_from(n);
    let (ext_u1_scratch, _) = ext_u1_scratch.split_at_mut(n_aligned_len);
    let mut ext_u1_scratch = MpNativeEndianMutByteSlice::from_bytes(ext_u1_scratch).unwrap();

    // ext_u0 starts out as 0.
    result.clear_bytes_above(0);
    let mut result = result.shrink_to(n.len());

    // ext_u1 starts out as 1. If the modulus is == 1, the result is undefined (the
    // multiplicative group does not exist). Force the result to zero in this
    // case as is needed by the application of Garner's method in the generic
    // ct_inv_mod_n_mp_mp().
    let n_is_one = ct_is_one_mp(n);
    ext_u1_scratch.clear_bytes_above(0);
    ext_u1_scratch.store_l_full(0, 1 ^ n_is_one.unwrap());

    let gcd_is_neg = ct_gcd_ext_odd_mp_mp(
        |t, t_is_neg_mask| {
            t.apply_to_mod_odd_n(
                t_is_neg_mask,
                &mut result,
                &mut ext_u1_scratch,
                n,
                neg_n0_inv_mod_l,
            );
        },
        &mut f_work_scratch,
        op0,
    );

    // Check whether the GCD is one (or -1) and return an error otherwise.
    let neg_mask = gcd_is_neg.select(0, !0);
    let mut expected_val = gcd_is_neg.select(1, !0);
    let nlimbs = n.nlimbs();
    for i in 0..nlimbs {
        if f_work_scratch.load_l_full(i) != expected_val {
            return Err(CtInvModOddMpMpError::OperandsNotCoprime);
        }
        expected_val = neg_mask;
    }

    // If the computed GCD came out as a negative 1, the inverse needs
    // to get negated mod f. Note that -result mod f == f - result.
    debug_assert!(ct_is_zero_mp(&result).unwrap() == 0 || n_is_one.unwrap() != 0);
    let neg_mask = n_is_one.select(neg_mask, 0);
    let mut neg_carry = (gcd_is_neg & !n_is_one).select(0, 1);
    let mut carry = 0;
    for i in 0..nlimbs {
        let result_val = result.load_l(i);
        let cond_neg_result_val;
        (neg_carry, cond_neg_result_val) = ct_add_l_l(result_val ^ neg_mask, neg_carry);
        let cond_f_val = n.load_l(i) & neg_mask;
        let result_val;
        (carry, result_val) = ct_add_l_l_c(cond_f_val, cond_neg_result_val, carry);
        if i + 1 != nlimbs {
            result.store_l_full(i, result_val);
        } else {
            result.store_l(i, result_val & result.partial_high_mask());
        }
    }

    Ok(())
}

#[cfg(test)]
fn test_ct_inv_mod_odd_mp_mp<
    RT: MpIntMutByteSlice,
    T0: MpIntMutByteSlice,
    NT: MpIntMutByteSlice,
>() {
    extern crate alloc;
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::{MpIntByteSliceCommon as _, MpIntByteSliceCommonPriv as _};
    use super::mul_impl::ct_mul_trunc_mp_l;
    use alloc::vec;

    fn test_one<RT: MpIntMutByteSlice, T0: MpIntMutByteSlice, NT: MpIntByteSliceCommon>(
        op0: &T0,
        n: &NT,
    ) {
        use super::div_impl::{ct_mod_mp_mp, CtMpDivisor};
        use super::mul_impl::ct_mul_trunc_mp_mp;

        let mut op0_work_scratch = vec![0u8; op0.len()];
        let mut op0_work_scratch = T0::from_bytes(&mut op0_work_scratch).unwrap();
        op0_work_scratch.copy_from(op0);
        ct_mod_mp_mp(None, &mut op0_work_scratch, &CtMpDivisor::new(n).unwrap());

        let mut op0_inv_mod_n = vec![0u8; RT::limbs_align_len(n.len())];
        let mut op0_inv_mod_n = RT::from_bytes(&mut op0_inv_mod_n).unwrap();

        let mut scratch0 = vec![0u8; MpNativeEndianMutByteSlice::limbs_align_len(n.len())];
        let mut scratch1 = vec![0u8; MpNativeEndianMutByteSlice::limbs_align_len(n.len())];

        ct_inv_mod_odd_mp_mp(
            &mut op0_inv_mod_n,
            &mut op0_work_scratch,
            n,
            [&mut scratch0, &mut scratch1],
        )
        .unwrap();

        // If n == 1, the multiplicative group does not exist and the result is fixed to
        // zero.
        if ct_is_one_mp(n).unwrap() != 0 {
            assert_eq!(ct_is_zero_mp(&op0_inv_mod_n).unwrap(), 1);
            return;
        }

        // Multiply op0_inv_mod_n by op0 modulo n and verify the result comes out as 1.
        let mut product_buf = vec![0u8; MpNativeEndianMutByteSlice::limbs_align_len(2 * n.len())];
        let mut product = MpNativeEndianMutByteSlice::from_bytes(&mut product_buf).unwrap();
        product.copy_from(&op0_inv_mod_n);
        ct_mul_trunc_mp_mp(&mut product, n.len(), op0);
        ct_mod_mp_mp(None, &mut product, &CtMpDivisor::new(n).unwrap());
        assert_eq!(ct_is_one_mp(&product).unwrap(), 1);
    }

    for l in [
        LIMB_BYTES - 1,
        2 * LIMB_BYTES - 1,
        3 * LIMB_BYTES - 1,
        4 * LIMB_BYTES - 1,
    ] {
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
        test_one::<RT, _, _>(&op0, &n);

        for i in 0..8 * LIMB_BYTES.min(n_op0_min_len) - 1 {
            for j in 1..8 * LIMB_BYTES.min(n_op0_min_len) - 1 {
                let mut n_buf = vec![0u8; n_len];
                let mut op0_buf = vec![0u8; op0_len];
                let mut n = NT::from_bytes(n_buf.as_mut_slice()).unwrap();
                let mut op0 = T0::from_bytes(op0_buf.as_mut_slice()).unwrap();
                n.set_bit_to(i, true);
                n.set_bit_to(0, true);
                op0.set_bit_to(j, true);
                test_one::<RT, _, _>(&op0, &n);
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
        test_one::<RT, _, _>(&op0, &n);
    }
}

#[test]
fn test_ct_inv_mod_odd_be_be_be() {
    use super::limbs_buffer::MpBigEndianMutByteSlice;
    test_ct_inv_mod_odd_mp_mp::<
        MpBigEndianMutByteSlice,
        MpBigEndianMutByteSlice,
        MpBigEndianMutByteSlice,
    >()
}

#[test]
fn test_ct_inv_mod_odd_le_le_le() {
    use super::limbs_buffer::MpLittleEndianMutByteSlice;
    test_ct_inv_mod_odd_mp_mp::<
        MpLittleEndianMutByteSlice,
        MpLittleEndianMutByteSlice,
        MpLittleEndianMutByteSlice,
    >()
}

#[test]
fn test_ct_inv_mod_odd_ne_ne_ne() {
    use super::limbs_buffer::MpNativeEndianMutByteSlice;
    test_ct_inv_mod_odd_mp_mp::<
        MpNativeEndianMutByteSlice,
        MpNativeEndianMutByteSlice,
        MpNativeEndianMutByteSlice,
    >()
}
