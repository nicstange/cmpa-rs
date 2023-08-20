//! Implementation of multiprecision integer comparison primitives.

use super::limb::{ct_is_zero_l, ct_leq_l_l, ct_lt_or_eq_l_l, LimbChoice, LimbType};
#[cfg(test)]
use super::limbs_buffer::MpIntMutSlice;
use super::limbs_buffer::MpIntSliceCommon;

/// Compare two multiprecision integers of specified endianess for `==`.
///
/// Evaluates to `true` iff `op0` is equal to `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
pub fn ct_eq_mp_mp<T0: MpIntSliceCommon, T1: MpIntSliceCommon>(op0: &T0, op1: &T1) -> LimbChoice {
    let op0_nlimbs = op0.nlimbs();
    let op1_nlimbs = op1.nlimbs();
    let common_nlimbs = op0_nlimbs.min(op1_nlimbs);

    let mut is_neq = 0;
    for i in 0..common_nlimbs {
        let op0_val = op0.load_l(i);
        let op1_val = op1.load_l(i);
        is_neq |= op0_val ^ op1_val;
    }

    for i in common_nlimbs..op0_nlimbs {
        let op0_val = op0.load_l(i);
        is_neq |= op0_val;
    }

    for i in common_nlimbs..op1_nlimbs {
        let op1_val = op1.load_l(i);
        is_neq |= op1_val;
    }

    LimbChoice::from(ct_is_zero_l(is_neq))
}

#[cfg(test)]
fn test_ct_eq_mp_mp<T0: MpIntMutSlice, T1: MpIntMutSlice>() {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::MpIntMutSlicePriv as _;

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, 1);
    op1.store_l(0, 1);
    assert_eq!(ct_eq_mp_mp(&op0, &op1).unwrap(), 1);
    assert_eq!(ct_eq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 1);
    assert_eq!(ct_eq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 1);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, 1);
    op0.store_l(1, 2);
    op1.store_l(0, 1);
    op1.store_l(1, 2);
    assert_eq!(ct_eq_mp_mp(&op0, &op1).unwrap(), 1);
    assert_eq!(ct_eq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 0);
    assert_eq!(ct_eq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 0);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    op0.store_l(0, 1);
    op1.store_l(0, 2);
    assert_eq!(ct_eq_mp_mp(&op0, &op1).unwrap(), 0);
    assert_eq!(ct_eq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 0);
    assert_eq!(ct_eq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 0);
}

#[test]
fn test_ct_eq_be_be() {
    use super::limbs_buffer::MpBigEndianMutByteSlice;
    test_ct_eq_mp_mp::<MpBigEndianMutByteSlice, MpBigEndianMutByteSlice>()
}

#[test]
fn test_ct_eq_le_le() {
    use super::limbs_buffer::MpLittleEndianMutByteSlice;
    test_ct_eq_mp_mp::<MpLittleEndianMutByteSlice, MpLittleEndianMutByteSlice>()
}

#[test]
fn test_ct_eq_ne_ne() {
    use super::limbs_buffer::MpNativeEndianMutByteSlice;
    test_ct_eq_mp_mp::<MpNativeEndianMutByteSlice, MpNativeEndianMutByteSlice>()
}

/// Compare two multiprecision integers of specified endianess for `!=`.
///
/// Evaluates to `true` iff `op0` is not equal to `op1` in value .
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
pub fn ct_neq_mp_mp<T0: MpIntSliceCommon, T1: MpIntSliceCommon>(op0: &T0, op1: &T1) -> LimbChoice {
    !ct_eq_mp_mp(op0, op1)
}

pub struct CtLeqMpMpKernel {
    tail_is_leq: LimbType,
}

impl CtLeqMpMpKernel {
    pub fn new() -> Self {
        Self { tail_is_leq: 1 }
    }

    pub fn update(&mut self, v0_val: LimbType, v1_val: LimbType) {
        let (is_lt, is_eq) = ct_lt_or_eq_l_l(v0_val, v1_val);
        // Consider the result from the tail comparisons only if the currently
        // inspected limbs are equal.
        self.tail_is_leq &= is_eq;
        // If the current v0_val is < v1_val, the tail does not matter.
        self.tail_is_leq |= is_lt;
    }

    pub fn finish(self) -> LimbChoice {
        LimbChoice::from(self.tail_is_leq)
    }
}

pub struct CtGeqMpMpKernel {
    leq_kernel: CtLeqMpMpKernel,
}

impl CtGeqMpMpKernel {
    pub fn new() -> Self {
        Self {
            leq_kernel: CtLeqMpMpKernel::new(),
        }
    }

    pub fn update(&mut self, v0_val: LimbType, v1_val: LimbType) {
        self.leq_kernel.update(v1_val, v0_val)
    }

    pub fn finish(self) -> LimbChoice {
        self.leq_kernel.finish()
    }
}

/// Compare two multiprecision integers of specified endianess for `<=`.
///
/// Evaluates to `true` iff `op0` is less or equal than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
pub fn ct_leq_mp_mp<T0: MpIntSliceCommon, T1: MpIntSliceCommon>(op0: &T0, op1: &T1) -> LimbChoice {
    let op0_nlimbs = op0.nlimbs();
    let op1_nlimbs = op1.nlimbs();
    let common_nlimbs = op0_nlimbs.min(op1_nlimbs);

    let mut leq_kernel = CtLeqMpMpKernel::new();
    for i in 0..common_nlimbs {
        leq_kernel.update(op0.load_l(i), op1.load_l(i));
    }
    for i in common_nlimbs..op0_nlimbs {
        leq_kernel.update(op0.load_l(i), 0);
    }
    for i in common_nlimbs..op1_nlimbs {
        leq_kernel.update(0, op1.load_l(i));
    }

    leq_kernel.finish()
}

#[cfg(test)]
fn test_ct_leq_mp_mp<T0: MpIntMutSlice, T1: MpIntMutSlice>() {
    use super::limb::LIMB_BYTES;
    use super::limbs_buffer::MpIntMutSlicePriv as _;

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    // [0 1]
    op0.store_l(0, 1);
    // [0 1]
    op1.store_l(0, 1);
    assert_eq!(ct_leq_mp_mp(&op0, &op1).unwrap(), 1);
    assert_eq!(ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 1);
    assert_eq!(ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 1);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    // [0 1]
    op0.store_l(0, 1);
    // [0 2]
    op1.store_l(0, 2);
    assert_eq!(ct_leq_mp_mp(&op0, &op1).unwrap(), 1);
    assert_eq!(ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 1);
    assert_eq!(ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 1);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    // [0 1]
    op0.store_l(0, 1);
    // [1 0]
    op1.store_l(1, 1);
    assert_eq!(ct_leq_mp_mp(&op0, &op1).unwrap(), 1);
    assert_eq!(ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 1);
    assert_eq!(ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 0);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    // [1 0]
    op0.store_l(1, 1);
    // [0 1]
    op1.store_l(0, 1);
    assert_eq!(ct_leq_mp_mp(&op0, &op1).unwrap(), 0);
    assert_eq!(ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 1);
    assert_eq!(ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 0);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    let mut op1 = tst_mk_mp_backing_vec!(T1, 2 * LIMB_BYTES);
    let mut op1 = T1::from_slice(&mut op1).unwrap();
    // [1 1]
    op0.store_l(0, 1);
    op0.store_l(1, 1);
    // [0 1]
    op1.store_l(0, 1);
    assert_eq!(ct_leq_mp_mp(&op0, &op1).unwrap(), 0);
    assert_eq!(ct_leq_mp_mp(&op0.split_at(LIMB_BYTES).1, &op1).unwrap(), 1);
    assert_eq!(ct_leq_mp_mp(&op0, &op1.split_at(LIMB_BYTES).1).unwrap(), 0);
}

#[test]
fn test_ct_leq_be_be() {
    use super::limbs_buffer::MpBigEndianMutByteSlice;
    test_ct_leq_mp_mp::<MpBigEndianMutByteSlice, MpBigEndianMutByteSlice>()
}

#[test]
fn test_ct_leq_le_le() {
    use super::limbs_buffer::MpLittleEndianMutByteSlice;
    test_ct_leq_mp_mp::<MpLittleEndianMutByteSlice, MpLittleEndianMutByteSlice>()
}

#[test]
fn test_ct_leq_ne_ne() {
    use super::limbs_buffer::MpNativeEndianMutByteSlice;
    test_ct_leq_mp_mp::<MpNativeEndianMutByteSlice, MpNativeEndianMutByteSlice>()
}

/// Compare two multiprecision integers of specified endianess for `<`.
///
/// Evaluates to `true` iff `op0` is less than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
pub fn ct_lt_mp_mp<T0: MpIntSliceCommon, T1: MpIntSliceCommon>(op0: &T0, op1: &T1) -> LimbChoice {
    !ct_geq_mp_mp(op0, op1)
}

/// Compare two multiprecision integers of specified_endianess for `>=`.
///
/// Evaluates to `true` iff `op0` is greater or equal than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
pub fn ct_geq_mp_mp<T0: MpIntSliceCommon, T1: MpIntSliceCommon>(op0: &T0, op1: &T1) -> LimbChoice {
    ct_leq_mp_mp(op1, op0)
}

/// Compare two multiprecision integers of specified endianess for `>`.
///
/// Evaluates to `true` iff `op0` is greater than `op1` in value.
///
/// # Arguments
///
/// * `op0` - The first operand as a multiprecision integer byte buffer.
/// * `op1` - The second operand as a multiprecision integer byte buffer.
pub fn ct_gt_mp_mp<T0: MpIntSliceCommon, T1: MpIntSliceCommon>(op0: &T0, op1: &T1) -> LimbChoice {
    !ct_leq_mp_mp(op0, op1)
}

pub fn ct_is_zero_mp<T0: MpIntSliceCommon>(op0: &T0) -> LimbChoice {
    let mut is_nz: LimbType = 0;
    for i in 0..op0.nlimbs() {
        let op0_val = op0.load_l(i);
        is_nz |= op0_val;
    }
    LimbChoice::from(ct_is_zero_l(is_nz))
}

pub fn ct_is_one_mp<T0: MpIntSliceCommon>(op0: &T0) -> LimbChoice {
    if op0.is_empty() {
        return LimbChoice::from(0);
    }

    let tail_is_not_one = op0.load_l(0) ^ 1;

    let mut head_is_nz: LimbType = 0;
    for i in 1..op0.nlimbs() {
        let op0_val = op0.load_l(i);
        head_is_nz |= op0_val;
    }

    LimbChoice::from(ct_is_zero_l(tail_is_not_one | head_is_nz))
}

pub fn ct_leq_mp_l<T0: MpIntSliceCommon>(op0: &T0, op1: LimbType) -> LimbChoice {
    if op0.is_empty() {
        return LimbChoice::from(1);
    }

    let l0_is_leq = ct_leq_l_l(op0.load_l(0), op1);

    let mut head_is_nz: LimbType = 0;
    for i in 1..op0.nlimbs() {
        let op0_val = op0.load_l(i);
        head_is_nz |= op0_val;
    }

    l0_is_leq & LimbChoice::from(ct_is_zero_l(head_is_nz))
}
