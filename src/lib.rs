extern crate alloc;

use core::mem;
use std::io::LineWriter;
use alloc::borrow;
use subtle::{self, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater};
use zeroize;

// Arithmetic on u32 is assumed to be constant-time.
type LimbType = u64;
const LIMB_BITS: u32 = LimbType::BITS;
const HALF_LIMB_BITS: u32 = LIMB_BITS / 2;
const HALF_LIMB_MASK: LimbType = (1 << HALF_LIMB_BITS) - 1;

fn ct_eq(v0: LimbType, v1: LimbType) -> subtle::Choice {
    v0.ct_eq(&v1)
}

fn ct_neq(v0: LimbType, v1: LimbType) -> subtle::Choice {
    !ct_eq(v0, v1)
}

fn ct_lt(v0: LimbType, v1: LimbType) -> subtle::Choice {
    v1.ct_gt(&v0)
}

fn ct_le(v0: LimbType, v1: LimbType) -> subtle::Choice {
    !v0.ct_gt(&v1)
}

fn ct_gt(v0: LimbType, v1: LimbType) -> subtle::Choice {
    ct_lt(v1, v0)
}

fn ct_ge(v0: LimbType, v1: LimbType) -> subtle::Choice {
    ct_le(v1, v0)
}

fn ct_limb_to_halves(v: LimbType) -> (LimbType, LimbType) {
    (v >> HALF_LIMB_BITS, v & HALF_LIMB_MASK)
}

fn ct_limb_from_halves(vh: LimbType, vl: LimbType) -> LimbType {
    vh << HALF_LIMB_BITS | vl
}

fn ct_add_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    // Don't rely on overflowing_add() for determining the carry -- that would almost certainly
    // branch and not be constant-time.
    let r = v0.wrapping_add(v1);
    let carry = (((v0 | v1) & !r) | (v0 & v1)) >> (LIMB_BITS - 1);
    (carry, r)
}

#[test]
fn test_ct_add_l_l() {
    assert_eq!(ct_add_l_l(0, 0), (0, 0));
    assert_eq!(ct_add_l_l(1, 0), (0, 1));
    assert_eq!(ct_add_l_l(!0 - 1, 1), (0, !0));
    assert_eq!(ct_add_l_l(!0, 1), (1, 0));
    assert_eq!(ct_add_l_l(1 << (LIMB_BITS - 1), 1 << (LIMB_BITS - 1)), (1, 0));
    assert_eq!(ct_add_l_l(!0, 1 << (LIMB_BITS - 1)), (1, (1 << (LIMB_BITS - 1)) - 1));
    assert_eq!(ct_add_l_l(!0, !0), (1, !0 - 1));
}

fn ct_sub_l_l(v0: LimbType, v1: LimbType) -> (LimbType, LimbType) {
    // Don't rely on overflowing_sub() for determining the borrow -- that would almost certainly
    // branch and not be constant-time.
    let r = v0.wrapping_sub(v1);
    let borrow = (((r | v1) & !v0) | (v1 & r)) >> (LIMB_BITS - 1);
    (borrow, r)
}

#[test]
fn test_ct_sub_l_l() {
    assert_eq!(ct_sub_l_l(0, 0), (0, 0));
    assert_eq!(ct_sub_l_l(1, 0), (0, 1));
    assert_eq!(ct_sub_l_l(0, 1), (1, !0));
    assert_eq!(ct_sub_l_l(1 << (LIMB_BITS - 1), 1 << (LIMB_BITS - 1)), (0, 0));
    assert_eq!(ct_sub_l_l(0, 1 << (LIMB_BITS - 1)), (1, 1 << (LIMB_BITS - 1)));
    assert_eq!(ct_sub_l_l(1 << (LIMB_BITS - 1), ((1 << (LIMB_BITS - 1)) + 1)), (1, !0));
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct DoubleLimb {
    v: [LimbType; 2],
}

impl DoubleLimb {
    fn new(h: LimbType, l: LimbType) -> Self {
        Self { v: [h, l] }
    }

    fn high(&self) -> LimbType {
        self.v[0]
    }

    fn low(&self) -> LimbType {
        self.v[1]
    }

    fn set_high(&mut self, val: LimbType) {
        self.v[0] = val;
    }

    fn set_low(&mut self, val: LimbType) {
        self.v[1] = val;
    }

    fn get_half_limb(&self, i: usize) -> LimbType {
        let i = 4 - i - 1;
        if i & 1 == 0 {
            self.v[i / 2] >> HALF_LIMB_BITS
        } else {
            self.v[i / 2] & HALF_LIMB_MASK
        }
    }
}

impl subtle::ConditionallySelectable for DoubleLimb {
    fn conditional_select(a: &Self, b: &Self, choice: subtle::Choice) -> Self {
        Self::new(LimbType::conditional_select(&a.high(), &b.high(), choice),
                  LimbType::conditional_select(&a.low(), &b.low(), choice))
    }
}

fn ct_mul_l_l(v0: LimbType, v1: LimbType) -> DoubleLimb {
    let (v0h, v0l) = ct_limb_to_halves(v0);
    let (v1h, v1l) = ct_limb_to_halves(v1);

    let prod_v0l_v1l = v0l * v1l;
    let prod_v0l_v1h = v0l * v1h;
    let prod_v0h_v1l = v0h * v1l;
    let prod_v0h_v1h = v0h * v1h;

    let mut result_low: LimbType = prod_v0l_v1l;
    let mut result_high: LimbType = prod_v0h_v1h;

    let (result_low_carry, result_low_sum) = ct_add_l_l(result_low, (prod_v0l_v1h & HALF_LIMB_MASK) << HALF_LIMB_BITS);
    result_low = result_low_sum;
    result_high += result_low_carry;
    result_high += prod_v0l_v1h >> HALF_LIMB_BITS;

    let (result_low_carry, result_low_sum) = ct_add_l_l(result_low, (prod_v0h_v1l & HALF_LIMB_MASK) << HALF_LIMB_BITS);
    result_low = result_low_sum;
    result_high += result_low_carry;
    result_high += prod_v0h_v1l >> HALF_LIMB_BITS;

    DoubleLimb::new(result_high, result_low)
}

fn ct_div_dl_l(mut u: DoubleLimb, v: LimbType) -> (DoubleLimb, LimbType) {
    debug_assert!(v >> (LIMB_BITS - 1) != 0);

    fn u_add_sub_at<F>(f: F, u: DoubleLimb, j: usize, val: DoubleLimb) -> (LimbType, DoubleLimb)
    where F: Fn(LimbType, LimbType) -> (LimbType, LimbType) {
        debug_assert!(matches!(j, 0 | 1 | 2));
        let (val_h, val_l) = if j & 1 != 0 {
            let (val_hh, val_hl) = ct_limb_to_halves(val.high());
            debug_assert_eq!(val_hh, 0);
            let (val_lh, val_ll) = ct_limb_to_halves(val.low());
            (ct_limb_from_halves(val_hl, val_lh), ct_limb_from_halves(val_ll, 0))
        } else {
            debug_assert_eq!(ct_limb_to_halves(val.high()).0, 0);
            (val.high(), val.low())
        };

        let j = j & !1;
        if j == 0 {
            let (carry_l, res_l) = f(u.low(), val_l);
            let (carry_h0, res_h) = f(u.high(), carry_l);
            let (carry_h1, res_h) = f(res_h, val_h);
            (carry_h0 + carry_h1, DoubleLimb::new(res_h, res_l))
        } else {
            let (carry, res_h) = f(u.high(), val_l);
            (carry + val_h, DoubleLimb::new(res_h, u.low()))
        }
    }

    fn u_add_at(u: DoubleLimb, j: usize, val: DoubleLimb) -> (LimbType, DoubleLimb) {
        u_add_sub_at(ct_add_l_l, u, j, val)
    }

    fn u_sub_at(u: DoubleLimb, j: usize, val: DoubleLimb) -> (LimbType, DoubleLimb) {
        u_add_sub_at(ct_sub_l_l, u, j, val)
    }

    let (v_h, v_l) = ct_limb_to_halves(v);

    // In the first iteration, the dividend's upper half limb is zero.
    // j = 2 = m
    let q = {
        let u_cur_dhl = u.get_half_limb(3);
        let q = u_cur_dhl / v_h;
        let r = u_cur_dhl - q * v_h;
        debug_assert_eq!(q >> HALF_LIMB_BITS, 0);
        debug_assert_eq!(r >> HALF_LIMB_BITS, 0);

        // q * v[n - 2] > b * r + u[j + n - 2]
        let over_estimated = ct_gt(q * v_l, (r << HALF_LIMB_BITS) | u.get_half_limb(2));
        let q = LimbType::conditional_select(&q, &q.wrapping_sub(1), over_estimated);
        let r = LimbType::conditional_select(&r, &(r + v_h), over_estimated);
        let over_estimated = ct_eq(r >> HALF_LIMB_BITS, 0) &
            ct_gt(q * v_l, (r << HALF_LIMB_BITS) | u.get_half_limb(2));
        debug_assert!(!<subtle::Choice as Into<bool>>::into(over_estimated) ||
                      (r + v_h) >> HALF_LIMB_BITS != 0);
        LimbType::conditional_select(&q, &q.wrapping_sub(1), over_estimated)
    };

    let qv = ct_mul_l_l(q, v);
    let (borrow, r) = u_sub_at(u, 2, qv);
    let borrow = ct_neq(borrow, 0);
    let (_, added_back) = u_add_at(r, 2, DoubleLimb::new(0, v));
    u = DoubleLimb::conditional_select(&r, &added_back, borrow);
    let q = LimbType::conditional_select(&q, &q.wrapping_sub(1), borrow);

    let mut qs: [LimbType; 3] = [0, 0, q];

    // j = 1, 0
    for j in [1, 0].iter().map(|j| *j) {
        let q = {
            let u_cur_dhl = ct_limb_from_halves(u.get_half_limb(2 + j), u.get_half_limb(1 + j)) ;
            let q = u_cur_dhl / v_h;
            let q = LimbType::conditional_select(&q, &HALF_LIMB_MASK, ct_neq(q >> HALF_LIMB_BITS, 0));
            let r = u_cur_dhl - q * v_h;
            debug_assert_eq!(q >> HALF_LIMB_BITS, 0);

            // q * v[n - 2] > b * r + u[j + n - 2] ?
            let over_estimated = ct_eq(r >> HALF_LIMB_BITS, 0) &
                ct_gt(q * v_l, (r << HALF_LIMB_BITS) | u.get_half_limb(j));
            let q = LimbType::conditional_select(&q, &q.wrapping_sub(1), over_estimated);
            let r = LimbType::conditional_select(&r, &(r + v_h), over_estimated);
            let over_estimated = ct_eq(r >> HALF_LIMB_BITS, 0) &
                ct_gt(q * v_l, (r << HALF_LIMB_BITS) | u.get_half_limb(j));
            debug_assert!(!<subtle::Choice as Into<bool>>::into(over_estimated) ||
                          (r + v_h) >> HALF_LIMB_BITS != 0);
            LimbType::conditional_select(&q, &q.wrapping_sub(1), over_estimated)
        };

        let qv = ct_mul_l_l(q, v);
        let (borrow, r) = u_sub_at(u, j, qv);
        let borrow = ct_neq(borrow, 0);
        let (_, added_back) = u_add_at(r, j, DoubleLimb::new(0, v));
        u = DoubleLimb::conditional_select(&r, &added_back, borrow);
        let q = LimbType::conditional_select(&q, &q.wrapping_sub(1), borrow);
        qs[j] = q;
    }

    let q  = DoubleLimb::new(qs[2], ct_limb_from_halves(qs[1], qs[0]));
    debug_assert_eq!(u.high(), 0);
    (q, u.low())
}

#[test]
fn test_div_dl_l() {
    fn mul_div_cmp(q0: DoubleLimb, r0: LimbType, v: LimbType) {
        let u_l = ct_mul_l_l(q0.low(), v);
        let u_h = ct_mul_l_l(q0.high(), v);
        assert_eq!(u_h.high(), 0);
        let (carry, u_h) = ct_add_l_l(u_h.low(), u_l.high());
        assert_eq!(carry, 0);
        let u_l = u_l.low();

        let (carry, u_l) = ct_add_l_l(u_l, r0);
        let (carry, u_h) = ct_add_l_l(u_h, carry);
        assert_eq!(carry, 0);

        let (q1, r1) = ct_div_dl_l(DoubleLimb::new(u_h, u_l), v);
        assert_eq!(q0, q1);
        assert_eq!(r0, r1);
    }

    mul_div_cmp(
        DoubleLimb::new(0, 0),
        0,
        !0
    );

    mul_div_cmp(
        DoubleLimb::new(0, 0),
        !0 - 1,
        !0,
    );


    mul_div_cmp(
        DoubleLimb::new(0x1, !0),
        0,
        1 << (LIMB_BITS - 1),
    );

    mul_div_cmp(
        DoubleLimb::new(0x1, !0),
        (1 << (LIMB_BITS - 1)) - 1,
        1 << (LIMB_BITS - 1),
    );


    mul_div_cmp(
        DoubleLimb::new(0x1, 0x1),
        0,
        !0
    );


    mul_div_cmp(
        DoubleLimb::new(0, 0x1),
        0,
        !0 - 1
    );

    mul_div_cmp(
        DoubleLimb::new(0, 0x1),
        1,
        !0 - 1
    );


    for i in 0..LIMB_BITS {
        for j in 0..LIMB_BITS {
            let u_h0 = if i == 0 {
                0
            } else {
                !0 >> (LIMB_BITS - i)
            };
            let u_l0 = !0;
            let u0 = DoubleLimb::new(u_h0, u_l0);

            let v = (1 << j) | (1 << (LIMB_BITS - 1));

            let (q, r) = ct_div_dl_l(u0, v);
            assert_eq!(q.high(), 0);

            let p = ct_mul_l_l(v, q.low());
            let (carry, u_l1) = ct_add_l_l(p.low(), r);
            let (carry, u_h1) = ct_add_l_l(p.high(), carry);
            assert_eq!(carry, 0);
            assert_eq!(u_l0, u_l1);
            assert_eq!(u_h0, u_h1);
        }
    }
}

fn load_limb(limbs: &[u8], i: usize) -> LimbType {
    let i = limbs.len() - (i + 1) * mem::size_of::<LimbType>();
    let bytes = &limbs[i..i + mem::size_of::<LimbType>()];
    let bytes = <[u8; mem::size_of::<LimbType>()] as TryFrom<&[u8]>>::try_from(bytes).unwrap();
    LimbType::from_be_bytes(bytes)
}

fn store_limb(limbs: &mut [u8], i: usize, value: LimbType) {
    let i = limbs.len() - (i + 1) * mem::size_of::<LimbType>();
    let bytes = &mut limbs[i..i + mem::size_of::<LimbType>()];
    let bytes = <&mut [u8; mem::size_of::<LimbType>()] as TryFrom<&mut [u8]>>::try_from(bytes).unwrap();
    *bytes = value.to_be_bytes();
}

pub enum CtMpDivisionError {
    UnalignedLength,
    LengthOverflow,
    DivisionByZero,
}

fn ct_mp_rem(dividend: &mut [u8], divisor: &[u8]) -> Result<(), CtMpDivisionError> {
    let dividend_nlimbs = dividend.len() / mem::size_of::<LimbType>();
    if dividend.len() != dividend_nlimbs * mem::size_of::<LimbType>() {
        return Err(CtMpDivisionError::UnalignedLength);
    }

    let divisor_nlimbs = divisor.len() / mem::size_of::<LimbType>();
    if divisor.len() != divisor_nlimbs * mem::size_of::<LimbType>() {
        return Err(CtMpDivisionError::UnalignedLength);
    }
    if divisor_nlimbs == 0 {
        return Err(CtMpDivisionError::DivisionByZero);
    }
    // Find the index of the highest set limb. For divisors, constant time evaluation doesn't really
    // matter, probably, but do it anyway. Note that the subtle crate doesn't support conditional
    // selection of usizes, so cast back and forth between u32 and usize.
    let divisor_set_nlimbs = {
        let divisor_nlimbs = u32::try_from(dividend_nlimbs).map_err(|_| CtMpDivisionError::LengthOverflow)?;
        let mut divisor_set_nlimbs: u32 = 0;
        for i in 0..divisor_nlimbs - 1 {
            let limb = load_limb(divisor, usize::try_from(i).map_err(|_| CtMpDivisionError::LengthOverflow)?);
            divisor_set_nlimbs.conditional_assign(&(i + 1), ct_neq(limb, 0));
        }
        usize::try_from(divisor_set_nlimbs).map_err(|_| CtMpDivisionError::LengthOverflow)?
    };
    if divisor_set_nlimbs == 0 {
        return Err(CtMpDivisionError::DivisionByZero);
    }

    if dividend_nlimbs < divisor_set_nlimbs {
        return Ok(())
    }

    // Normalize divisor's high limb. Calculate 2^LimbType::BITS / (divisor_high + 1)
    let divisor_high = load_limb(divisor, divisor_set_nlimbs - 1);
    let scaling = {
        // Be careful to avoid overflow in calculating divisor_high + 1. The subsequent code below
        // still returns the correct result if the increment is skipped in this case.
        let den = divisor_high + LimbType::conditional_select(&1, &0, ct_eq(divisor_high, !0));

        // First calculate (2^LimbType::BITS - 1) / (divisor_high + 1).
        let q = !0 / den;
        let rem = !0 - den * q;
        // And possibly round up to get 2^LimbType::BITS / (divisor_high + 1).
        // Note that the test below is equivalent to rem + 1 == divisor_high + 1.
        q + LimbType::conditional_select(&0, &1, ct_eq(rem, divisor_high))
    };

    let scaled_divisor_high = divisor_high * scaling;
    drop(divisor_high);

    

    Ok(())
}
