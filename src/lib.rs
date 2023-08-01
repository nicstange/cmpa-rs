//! Multiprecision integer arithmetic primitives.

mod add_impl;
mod cmp_impl;
mod div_impl;
mod euclid_impl;
pub mod hexstr;
mod invmod_impl;
mod lcm_impl;
mod limb;
mod limbs_buffer;
mod montgomery_impl;
mod mul_impl;
mod prime_impl;
mod shift_impl;
mod usize_ct_cmp;

pub use limb::{
    ct_eq_l_l, ct_geq_l_l, ct_gt_l_l, ct_is_nonzero_l, ct_is_zero_l, ct_leq_l_l, ct_lt_l_l,
    ct_neq_l_l, LimbType,
};

pub use limbs_buffer::{
    clear_bits_above_mp, clear_bits_below_mp, ct_clear_bits_above_mp, ct_clear_bits_below_mp,
    ct_find_first_set_bit_mp, ct_find_last_set_bit_mp, ct_swap_cond_mp, MpBigEndianByteSlice,
    MpBigEndianMutByteSlice, MpIntByteSlice, MpIntByteSliceCommon, MpIntMutByteSlice,
    MpLittleEndianByteSlice, MpLittleEndianMutByteSlice, MpNativeEndianByteSlice,
    MpNativeEndianMutByteSlice, UnalignedMpByteSliceLenError,
};

pub use add_impl::{ct_add_mp_l, ct_add_mp_mp, ct_sub_mp_l, ct_sub_mp_mp};

pub use cmp_impl::{
    ct_eq_mp_mp, ct_geq_mp_mp, ct_gt_mp_mp, ct_is_one_mp, ct_is_zero_mp, ct_leq_mp_l, ct_leq_mp_mp,
    ct_lt_mp_mp, ct_neq_mp_mp,
};

pub use div_impl::{ct_div_mp_l, ct_div_mp_mp, CtDivMpError};

pub use euclid_impl::{ct_gcd_mp_mp, ct_gcd_odd_mp_mp, ct_inv_mod_odd_mp_mp, CtInvModOddMpMpError};

pub use invmod_impl::{ct_inv_mod_mp_mp, CtInvModMpMpError};

pub use lcm_impl::ct_lcm_mp_mp;

pub use montgomery_impl::{
    ct_montgomery_mul_mod_mp_mp, ct_montgomery_neg_n0_inv_mod_l_mp, ct_montgomery_radix2_mod_n_mp,
    ct_montgomery_redc_mp, ct_to_montgomery_form_direct_mp, ct_to_montgomery_form_mp,
};

pub use mul_impl::{ct_mul_trunc_mp_l, ct_mul_trunc_mp_mp, ct_square_trunc_mp};

pub use prime_impl::{ct_composite_test_small_prime_gcd_mp, ct_prime_test_miller_rabin_mp};

pub use shift_impl::{ct_lshift_mp, ct_rshift_mp};
