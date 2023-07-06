use crate::limb::{ct_sub_l_l, ct_add_l_l_c, ct_is_zero_l, ct_is_nonzero_l, ct_arithmetic_rshift_l};

use super::limb::{LimbType, LimbChoice, LIMB_BITS, ct_mul_l_l, ct_mul_add_l_l_l_c, ct_add_l_l};
use super::limbs_buffer::MPIntMutByteSlice;

// Implementation of Euclid's algorithm after
// [BERN_YANG19] "Fast constant-time gcd computation and modular inversion", Daniel J. Bernstein and
//               Bo-Yin Yang, IACR Transactions on Cryptographic Hardware and Embedded Systems ISSN
//               2569-2925, Vol. 2019, No. 3, pp. 340-398
//
// The main advantages over other common binary gcd algorithms for the purposes here are
// - only the least significant bit determines the operations to be carried out
//   in each step,
// - steps can be batched into transition matrices
// - and, last but not least, it's particularly constant-time implementation friendly overall.
//
// [BERN_YANG19] define transition matrices describing their divstep(\delta, f, g) step:
// /  \                     / \
// |f'|                     |f|
// |  | = T(\delta, f, g) * | |
// |g'|                     |g|
// \  /                     \ /
// with
//                                        /
//                                        | [[0, 1], [-1, 1]]     if \delta > 0 and g odd
// T(\delta, f, g) = [[1, 0], [0, 1/2]] * |
//                                        | [[1, 0], [g % 2, 1]]  otherwise.
//                                        \
// c.f. p. 361.
//
// The matrices in the second operand of the product on the right hand side above both have
// inf-norm == 2, so multiplying any against some other matrix (like a product of these) at most
// doubles the maximum of the entries' values. Representing negative values in two's complement as
// stored in in (unsigned) LimbTypes and tracking the shifting factor from the first operand on the
// RHS separately allows for batching up to LIMB_BITS - 2 divstep() invocations and accumulate them
// in such a matrix (without overflow into the sign bit) before applying them to MP integers.

struct TransitionMatrix {
    t: [[LimbType; 2]; 2],
    row_shift: [LimbType; 2],
}

impl TransitionMatrix {
    fn identity() -> Self {
        Self { t: [[1, 0], [0, 1]], row_shift: [0, 0] }
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
        // If not case0, then the original first row is multiplied by g & 1 (== g0) before
        // getting added to the second one.
        let t0_mask = case0.select(g0.select(0, !0), !0);
        for j in [0, 1] {
            let mut t0j = t0[j];
            // If case0 == true, then negate the original first row to be added to the second row.
            t0j ^= case0_mask; // Conditional bitwise negation.
            t0j = t0j.wrapping_add(case0.unwrap()); // And add one to finally obtain -t0j.
            // If case0 == false, clear out the original first row in case g & 1 == 0.
            t0j &= t0_mask;
            self.t[1][j] = self.t[1][j].wrapping_add(t0j << row_shift_diff);
        }
        self.row_shift[1] += 1;
    }

    // Apply the transition matrix to the column vector (f, g).  {f,g}_shadow_head[0] shadows the
    // potentially partial high limbs of f and g respectively, {f,g}_shadow_head[1] is used to store
    // temporary excess (before right-shifting) and two's complement sign bits.
    fn apply_to<FT: MPIntMutByteSlice, GT: MPIntMutByteSlice>(
        &self,
        f_shadow_head: &mut [LimbType; 2], f: &mut FT,
        g_shadow_head: &mut [LimbType; 2], g: &mut GT
    ) {
        let nlimbs = f.nlimbs();
        assert_eq!(nlimbs, g.nlimbs());
        assert!(nlimbs != 0);

        let t_f_f_is_neg = LimbChoice::from(self.t[0][0] >> LIMB_BITS - 1);
        let t_f_g_is_neg = LimbChoice::from(self.t[0][1] >> LIMB_BITS - 1);
        let t_g_f_is_neg = LimbChoice::from(self.t[1][0] >> LIMB_BITS - 1);
        let t_g_g_is_neg = LimbChoice::from(self.t[1][1] >> LIMB_BITS - 1);
        let t_is_neg_mask: [[LimbType; 2]; 2] = [
            [t_f_f_is_neg.select(0, !0), t_f_g_is_neg.select(0, !0)],
            [t_g_f_is_neg.select(0, !0), t_g_g_is_neg.select(0, !0)],
        ];

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
        fn mat_mul_compute_row(carry: &mut [LimbType; 2],
                               borrow: &mut LimbType,
                               cur_orig: &[LimbType; 2],
                               last_orig: &[LimbType; 2],
                               t_row: &[LimbType; 2],
                               t_row_is_neg_mask: &[LimbType; 2]) -> LimbType {
            // t_{i, f} * f
            let t_if_f = ct_mul_l_l(t_row[0], cur_orig[0]);
            let (f_carry, result) = (t_if_f.high(), t_if_f.low());
            // + t_{i, g} * g + carry[0]
            let (g_carry, result) = ct_mul_add_l_l_l_c(result, t_row[1], cur_orig[1], carry[0]);
            // Update the carry for the next round.
            (carry[1], carry[0]) = ct_add_l_l_c(f_carry, g_carry, carry[1]);
            // If t_{i, f} is negative, don't sign extend it all the way up to
            // nlimbs, but simply account for it by subtracting f shifted by one
            // limb position to the left: write the (virtually) sign extended t_{i, f} as
            // 2^LIMB_BITS * (-1) + t_{i, f} and multiply by f.
            let (f_borrow, result) = ct_sub_l_l(result, last_orig[0] & t_row_is_neg_mask[0]);
            debug_assert!(f_borrow == 0 || result >= 1);
            // Similar for negative t_{i, g}.
            let (g_borrow, result) = ct_sub_l_l(result, last_orig[1] & t_row_is_neg_mask[1]);
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
                new_val[i] = mat_mul_compute_row(
                    &mut carry[i], &mut borrow[i],
                    &cur_orig_val, &last_orig_val,
                    &self.t[i], &t_is_neg_mask[i]
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
                new_val[i] = mat_mul_compute_row(
                    &mut carry[i], &mut borrow[i],
                    &cur_orig_val, &last_orig_val,
                    &self.t[i], &t_is_neg_mask[i]
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

        // Apply shift and store the high result limbs back to to {f,g}_shadow_head[1] respectively.
        for i in 0..2 {
            let s = self.row_shift[i];
            last_new_val[i] = ct_arithmetic_rshift_l(last_new_val[i], s);
            debug_assert!(last_new_val[i] == 0 || last_new_val[i] == !0);
        }
        f_shadow_head[1] = last_new_val[0];
        g_shadow_head[1] = last_new_val[1];
    }
}

const STEPS_PER_BATCH: usize = LIMB_BITS as usize - 2;

// Given the previous delta, and the least significant limbs of f and g respectively, determine
// STEPS_PER_BATCH steps and accumulate them in a single transition matrix.
fn batch_divsteps(
    mut delta: usize, mut f_low: LimbType, mut g_low: LimbType
) -> (usize, TransitionMatrix) {
    debug_assert_eq!(f_low & 1, 1);
    let mut m = TransitionMatrix::identity();
    for _ in 0..STEPS_PER_BATCH {
        let delta_is_pos = LimbChoice::from((delta.wrapping_neg() >> usize::BITS - 1) as LimbType);
        debug_assert!(delta == 0 || delta > isize::MAX as usize || delta_is_pos.unwrap() == 1);
        debug_assert!(!(delta == 0 || delta > isize::MAX as usize) || delta_is_pos.unwrap() == 0);
        let g0 = LimbChoice::from(g_low & 1);
        let case0 = delta_is_pos & g0;
        m.push_transition(case0, g0);

        delta = case0.select_usize((1 as usize).wrapping_add(delta), (1 as usize).wrapping_sub(delta));
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
    // For the minimum number of steps, c.f. [BERN_YANG19], Theorem G.6.
    // Assuming len = f.len().min()(g.len()), then bits == len * 8, but the second operand, g, is
    // assumed to be even initially and to have been halved before getting input to the GCD, hence the +1 below.
    let nbits = len * 8 + 1;
    let b = nbits + 1; // The +1 to account for taking the log2 of the Euclidean norm of (f, g).
    let nsteps = if b <= 21 {
        19 * b / 7
    } else if b <= 46 {
        (49 * b + 23) / 17
    } else {
        49 * b / 17
    };
    (nsteps + STEPS_PER_BATCH - 1) / STEPS_PER_BATCH
}

pub fn mp_ct_gcd<FT: MPIntMutByteSlice, GT: MPIntMutByteSlice>(f: &mut FT, g: &mut GT) {
    let nlimbs = f.nlimbs();
    assert_eq!(nlimbs, g.nlimbs());
    assert!(nlimbs != 0);
    assert_eq!(f.load_l(0) & 1, 1);
    let nbatches = nbatches(f.len().max(g.len()));

    // One shadow limb is for shadowing potentially partial high limbs of f or g respectively and
    // another one for excess precision needed for temporary intermediate values due to the batching
    // and also, for two's complement sign bits.
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
        transition_matrix.apply_to(&mut f_shadow_head, f, &mut g_shadow_head, g);
    }

    debug_assert_eq!(g_shadow_head[0], 0);
    g.store_l(nlimbs - 1, 0);
    debug_assert_eq!(g_shadow_head[1], 0);
    for k in 0..nlimbs {
        debug_assert_eq!(g.load_l(k), 0);
    }

    debug_assert!(f_shadow_head[1] == 0 || f_shadow_head[1] == !0);
    let f_is_neg = LimbChoice::from(f_shadow_head[1] >> LIMB_BITS - 1);
    let neg_mask = f_is_neg.select(0, !0);
    let mut carry = f_is_neg.select(0, 1);
    let mut k = 0;
    while k + 1 < nlimbs {
        let mut f_val = f.load_l_full(k);
        f_val ^= neg_mask;
        (carry, f_val) = ct_add_l_l(f_val, carry);
        f.store_l_full(k, f_val);
        k += 1;
    }
    let mut f_val = f_shadow_head[0];
    f_val ^= neg_mask;
    (_, f_val) = ct_add_l_l(f_val, carry);
    f.store_l(nlimbs - 1, f_val);
}

#[cfg(test)]
fn test_mp_ct_gcd<FT: MPIntMutByteSlice, GT: MPIntMutByteSlice>() {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::{MPIntByteSliceCommon as _, MPIntByteSliceCommonPriv as _};
    use super::mul_impl::mp_ct_mul_trunc_mp_l;

    fn assert_mp_is_equal<T: MPIntMutByteSlice>(v: &T, expected: LimbType) {
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
        mp_ct_gcd(&mut f, &mut g);
        assert_mp_is_equal(&f, 1);
        assert_mp_is_equal(&g, 0);

        for i in 1..8 * LIMB_BYTES.min(f_g_min_len) - 1 {
            for j in 1..8 * LIMB_BYTES.min(f_g_min_len) - 1 {
                let mut f_buf = vec![0u8; f_len];
                let mut g_buf = vec![0u8; g_len];
                let mut f = FT::from_bytes(f_buf.as_mut_slice()).unwrap();
                let mut g = GT::from_bytes(g_buf.as_mut_slice()).unwrap();
                let limb_index = i / (8 * LIMB_BYTES);
                let i = i % (8 * LIMB_BYTES);
                f.store_l(limb_index, 1 << i);
                f.store_l(0, f.load_l(0) | 1);
                let limb_index = j / (8 * LIMB_BYTES);
                let j = j % (8 * LIMB_BYTES);
                g.store_l(limb_index, 1 << j);
                mp_ct_gcd(&mut f, &mut g);
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
        mp_ct_gcd(&mut f, &mut g);
        assert_mp_is_equal(&f, 3 * 3);
        assert_mp_is_equal(&g, 0);

        let mut f_buf = vec![0u8; f_len];
        let mut g_buf = vec![0u8; g_len];
        let mut f = FT::from_bytes(f_buf.as_mut_slice()).unwrap();
        let mut g = GT::from_bytes(g_buf.as_mut_slice()).unwrap();
        f.store_l(0, 3 * 3 * 5);
        g.store_l(0, 3 * 3 * 3 * 7);
        mp_ct_gcd(&mut f, &mut g);
        assert_mp_is_equal(&f, 3 * 3);
        assert_mp_is_equal(&g, 0);

        let mut f_buf = vec![0u8; f_len];
        let mut g_buf = vec![0u8; g_len];
        let mut f = FT::from_bytes(f_buf.as_mut_slice()).unwrap();
        let mut g = GT::from_bytes(g_buf.as_mut_slice()).unwrap();
        f.store_l(0, 1);
        while f.load_l(f.nlimbs() - 1) >> 8 * (f_g_high_min_len - 1) == 0 {
            mp_ct_mul_trunc_mp_l(&mut f, f_len, 251);
        }
        g.store_l(0, 1);
        while g.load_l(g.nlimbs() - 1) >> 8 * (f_g_high_min_len - 1) == 0 {
            mp_ct_mul_trunc_mp_l(&mut g, g_len, 241);
        }
        while g.load_l(g.nlimbs() - 1) >> 8 * (f_g_high_min_len) - 1 == 0 {
            mp_ct_mul_trunc_mp_l(&mut g, g_len, 2);
        }
        mp_ct_gcd(&mut f, &mut g);
        assert_mp_is_equal(&f, 1);
        assert_mp_is_equal(&g, 0);

        let mut f_buf = vec![0u8; f_len];
        let mut g_buf = vec![0u8; g_len];
        let mut f = FT::from_bytes(f_buf.as_mut_slice()).unwrap();
        let mut g = GT::from_bytes(g_buf.as_mut_slice()).unwrap();
        f.store_l(0, 1);
        while f.load_l(f.nlimbs() - 1) >> 8 * (f_g_high_min_len - 2) == 0 {
            mp_ct_mul_trunc_mp_l(&mut f, f_len, 251);
        }
        mp_ct_mul_trunc_mp_l(&mut f, f_len, 241);
        g.store_l(0, 1);
        while g.load_l(g.nlimbs() - 1) >> 8 * (f_g_high_min_len - 1) == 0 {
            mp_ct_mul_trunc_mp_l(&mut g, g_len, 241);
        }
        while g.load_l(g.nlimbs() - 1) >> 8 * (f_g_high_min_len) - 1 == 0 {
            mp_ct_mul_trunc_mp_l(&mut g, g_len, 2);
        }
        mp_ct_gcd(&mut f, &mut g);
        assert_mp_is_equal(&f, 241);
        assert_mp_is_equal(&g, 0);

        let mut f_buf = vec![0u8; f_len];
        let mut g_buf = vec![0u8; g_len];
        let mut f = FT::from_bytes(f_buf.as_mut_slice()).unwrap();
        let mut g = GT::from_bytes(g_buf.as_mut_slice()).unwrap();
        f.store_l(0, 1);
        while f.load_l(f.nlimbs() - 1) >> 8 * (f_g_high_min_len - 1) == 0 {
            mp_ct_mul_trunc_mp_l(&mut f, f_len, 251);
        }
        g.store_l(0, 1);
        while g.load_l(g.nlimbs() - 1) >> 8 * (f_g_high_min_len - 2) == 0 {
            mp_ct_mul_trunc_mp_l(&mut g, g_len, 241);
        }
        mp_ct_mul_trunc_mp_l(&mut g, g_len, 251);
        while g.load_l(g.nlimbs() - 1) >> 8 * (f_g_high_min_len) - 1 == 0 {
            mp_ct_mul_trunc_mp_l(&mut g, g_len, 2);
        }
        mp_ct_gcd(&mut f, &mut g);
        assert_mp_is_equal(&f, 251);
        assert_mp_is_equal(&g, 0);
    }
}

#[test]
fn test_mp_ct_gcd_be_be() {
    use super::limbs_buffer::MPBigEndianMutByteSlice;
    test_mp_ct_gcd::<MPBigEndianMutByteSlice, MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_gcd_le_le() {
    use super::limbs_buffer::MPLittleEndianMutByteSlice;
    test_mp_ct_gcd::<MPLittleEndianMutByteSlice, MPLittleEndianMutByteSlice>()
}

#[test]
fn test_mp_ct_gcd_ne_ne() {
    use super::limbs_buffer::MPNativeEndianMutByteSlice;
    test_mp_ct_gcd::<MPNativeEndianMutByteSlice, MPNativeEndianMutByteSlice>()
}
