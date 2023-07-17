use core::arch::asm;
use super::limb::{LimbChoice, LimbType};

// core::hint::black_box() is inefficient: it writes and reads from memory.
#[inline(always)]
pub fn black_box_usize(v: usize) -> usize {
    let result: usize;
    unsafe { asm!("/* {v} */", v = inout(reg) v => result, options(pure, nomem, nostack)); }
    result
}

pub fn ct_is_nonzero_usize(v: usize) -> LimbType {
    // This trick is from subtle::*::ct_eq():
    // if v is non-zero, then v or -v or both have the high bit set.
    black_box_usize((v | v.wrapping_neg()) >> usize::BITS - 1) as LimbType
}

pub fn ct_is_zero_usize(v: usize) -> LimbType {
    (1 as LimbType) ^ ct_is_nonzero_usize(v)
}

pub fn ct_eq_usize_usize(v0: usize, v1: usize) -> LimbChoice {
    LimbChoice::from(ct_is_zero_usize(v0 ^ v1))
}

pub fn ct_neq_usize_usize(v0: usize, v1: usize) -> LimbChoice {
    !ct_eq_usize_usize(v0, v1)
}

fn ct_sub_usize_usize(v0: usize, v1: usize) -> (usize, usize) {
    // Don't rely on overflowing_sub() for determining the borrow -- that would almost certainly
    // branch and not be constant-time.
    let v0 = black_box_usize(v0);
    let v1 = black_box_usize(v1);
    let r = v0.wrapping_sub(v1);
    let borrow = black_box_usize((((r | v1) & !v0) | (v1 & r)) >> usize::BITS - 1);
    (borrow, r)
}

#[test]
fn test_ct_sub_usize_usize() {
    assert_eq!(ct_sub_usize_usize(0, 0), (0, 0));
    assert_eq!(ct_sub_usize_usize(1, 0), (0, 1));
    assert_eq!(ct_sub_usize_usize(0, 1), (1, !0));
    assert_eq!(ct_sub_usize_usize(1 << (usize::BITS - 1), 1 << (usize::BITS - 1)), (0, 0));
    assert_eq!(ct_sub_usize_usize(0, 1 << (usize::BITS - 1)), (1, 1 << (usize::BITS - 1)));
    assert_eq!(ct_sub_usize_usize(1 << (usize::BITS - 1), (1 << (usize::BITS - 1)) + 1), (1, !0));
}

pub fn ct_lt_usize_usize(v0: usize, v1: usize) -> LimbChoice {
    let (borrow, _) = ct_sub_usize_usize(v0, v1);
    LimbChoice::from(borrow as LimbType)
}

pub fn ct_le_usize_usize(v0: usize, v1: usize) -> LimbChoice {
    !ct_lt_usize_usize(v1, v0)
}

pub fn ct_gt_usize_usize(v0: usize, v1: usize) -> LimbChoice {
    ct_lt_usize_usize(v1, v0)
}

pub fn ct_ge_usize_usize(v0: usize, v1: usize) -> LimbChoice {
    ct_le_usize_usize(v1, v0)
}
