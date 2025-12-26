// SPDX-License-Identifier: Apache-2.0
// Copyright 2023-2025 SUSE LLC
// Author: Nicolai Stange <nstange@suse.de>

//! Accessors for multiprecision integers as stored in byte buffers of certain
//! endian layouts.
//!
//! Multiprecision integers are stored in bytes buffers in big-, little- or
//! internal, "native"-endian order. The arithmetic primitives all operate on
//! those in units of [`LimbType`] for efficiency reasons.  This helper module
//! provides a couple of utilities for accessing these byte buffers in units of
//! [`LimbType`].

use core::{self, cmp, convert, fmt, marker, mem, slice};

use super::limb::{
    LIMB_BITS, LIMB_BYTES, LimbChoice, LimbType, ct_find_first_set_bit_l, ct_find_last_set_bit_l,
    ct_find_last_set_byte_l, ct_is_nonzero_l, ct_is_zero_l, ct_lsb_mask_l,
};
use super::usize_ct_cmp::ct_eq_usize_usize;

/// Determine the number of [`LimbType`] limbs stored in a multiprecision
/// integer big-endian byte buffer.
///
/// # Arguments
///
/// * `len` - The multiprecision integer's underlying big-endian byte buffer's
///   length in bytes.
pub const fn ct_mp_nlimbs(len: usize) -> usize {
    (len + LIMB_BYTES - 1) / LIMB_BYTES
}

pub const fn ct_mp_limbs_align_len(len: usize) -> usize {
    ct_mp_nlimbs(len) * LIMB_BYTES
}

#[test]
fn test_ct_mp_nlimbs() {
    assert_eq!(ct_mp_nlimbs(0), 0);
    assert_eq!(ct_mp_nlimbs(1), 1);
    assert_eq!(ct_mp_nlimbs(LIMB_BYTES - 1), 1);
    assert_eq!(ct_mp_nlimbs(LIMB_BYTES), 1);
    assert_eq!(ct_mp_nlimbs(LIMB_BYTES + 1), 2);
    assert_eq!(ct_mp_nlimbs(2 * LIMB_BYTES - 1), 2);
}

/// Internal helper to load some fully contained limb from a multiprecision
/// integer big-endian byte buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. This
/// internal helper is used whenever the specified limb is not such a partially
/// covered high limb.
///
/// Runs in constant time as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   big-endian order.
/// * `src_end` - The position past the end of the limb to be loaded from
///   `limbs`. Must be `>= core::mem::size_of::<LimbType>()`.
fn _be_mp_load_l_full(limbs: &[u8], src_end: usize) -> LimbType {
    let src_begin = src_end - LIMB_BYTES;
    let src = &limbs[src_begin..src_end];
    let src = <[u8; LIMB_BYTES] as TryFrom<&[u8]>>::try_from(src).unwrap();
    LimbType::from_be_bytes(src)
}

/// Internal helper to load some partially contained limb from a multiprecision
/// integer big-endian byte buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. This
/// internal helper is used whenever the specified limb is such a partially
/// covered high limb.
///
/// Execution time depends only on the `src_end` argument and is otherwise
/// constant as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   big-endian order.
/// * `src_end` - The position past the end of the limb to be loaded from
///   `limbs`. Must be `< core::mem::size_of::<LimbType>()`.
fn _be_mp_load_l_high_partial(limbs: &[u8], src_end: usize) -> LimbType {
    let mut src: [u8; LIMB_BYTES] = [0; LIMB_BYTES];
    src[LIMB_BYTES - src_end..LIMB_BYTES].copy_from_slice(&limbs[0..src_end]);
    LimbType::from_be_bytes(src)
}

/// Load some fully contained limb from a multiprecision integer big-endian byte
/// buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. The
/// generic [`be_mp_load_l()`] implements some extra logic for handling this
/// special case of a partially stored high limb, which can be avoided if it is
/// known that the limb to be accessed is completely covered by the `limbs`
/// buffer. This function here provides such a less generic, but more performant
/// variant.
///
/// Runs in constant time as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   big-endian order.
/// * `i` - The index of the limb to load, counted from least to most
///   significant.
fn be_mp_load_l_full(limbs: &[u8], i: usize) -> LimbType {
    debug_assert!(i * LIMB_BYTES < limbs.len());
    let src_end = limbs.len() - i * LIMB_BYTES;
    _be_mp_load_l_full(limbs, src_end)
}

/// Load some limb from a multiprecision integer big-endian byte buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. If it
/// is known that the limb to be accessed at position `i` is fully covered by
/// the `limbs` buffer, consider using [`be_mp_load_l_full()`] for improved
/// performance instead.
///
/// Execution time depends only on the multiprecision integer's length and the
/// limb index argument `i`, and is otherwise constant as far as branching is
/// concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   big-endian order.
/// * `i` - The index of the limb to load, counted from least to most
///   significant.
fn be_mp_load_l(limbs: &[u8], i: usize) -> LimbType {
    debug_assert!(i * LIMB_BYTES <= limbs.len());
    let src_end = limbs.len() - i * LIMB_BYTES;
    if src_end >= LIMB_BYTES {
        _be_mp_load_l_full(limbs, src_end)
    } else {
        _be_mp_load_l_high_partial(limbs, src_end)
    }
}

#[test]
fn test_be_mp_load_l() {
    let limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    assert_eq!(be_mp_load_l(&limbs, 0), 0);
    assert_eq!(be_mp_load_l(&limbs, 1), 0);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    limbs[LIMB_BYTES] = 0x80;
    limbs[LIMB_BYTES - 1] = 1;
    assert_eq!(be_mp_load_l(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(be_mp_load_l_full(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(be_mp_load_l(&limbs, 1), 1);
    assert_eq!(be_mp_load_l_full(&limbs, 1), 1);

    let limbs: [u8; 1] = [0; 1];
    assert_eq!(be_mp_load_l(&limbs, 0), 0);

    let limbs: [u8; LIMB_BYTES + 1] = [0; LIMB_BYTES + 1];
    assert_eq!(be_mp_load_l(&limbs, 0), 0);
    assert_eq!(be_mp_load_l_full(&limbs, 0), 0);
    assert_eq!(be_mp_load_l(&limbs, 1), 0);

    let limbs: [u8; 2] = [0, 1];
    assert_eq!(be_mp_load_l(&limbs, 0), 1);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    limbs[1] = 1;
    assert_eq!(be_mp_load_l(&limbs, 0), 0);
    assert_eq!(be_mp_load_l_full(&limbs, 0), 0);
    assert_eq!(be_mp_load_l(&limbs, 1), 1);

    let limbs: [u8; 2] = [1, 0];
    assert_eq!(be_mp_load_l(&limbs, 0), 0x0100);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    limbs[0] = 1;
    assert_eq!(be_mp_load_l(&limbs, 0), 0);
    assert_eq!(be_mp_load_l_full(&limbs, 0), 0);
    assert_eq!(be_mp_load_l(&limbs, 1), 0x0100);
}

/// Internal helper to update some fully contained limb in a multiprecision
/// integer big-endian byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. This
/// internal helper is used whenever the specified limb is not such a partially
/// covered high limb.
///
/// Runs in constant time as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   big-endian order.
/// * `dst_end` - The position past the end of the limb to be updated in
///   `limbs`. Must be `>= core::mem::size_of::<LimbType>()`.
/// * `value` - The value to store in `limbs` at the specified position.
fn _be_mp_store_l_full(limbs: &mut [u8], dst_end: usize, value: LimbType) {
    let dst_begin = dst_end - LIMB_BYTES;
    let dst = &mut limbs[dst_begin..dst_end];
    let dst = <&mut [u8; LIMB_BYTES] as TryFrom<&mut [u8]>>::try_from(dst).unwrap();
    *dst = value.to_be_bytes();
}

/// Internal helper to update some partially contained limb in a multiprecision
/// integer big-endian byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. This
/// internal helper is used whenever the specified limb is not such a partially
/// covered high limb.
///
/// Execution time depends only on the `dst_end` argument and is otherwise
/// constant as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   big-endian order.
/// * `dst_end` - The position past the end of the limb to be updated in
///   `limbs`. Must be `< core::mem::size_of::<LimbType>()`.
/// * `value` - The value to store in `limbs` at the specified position. The
///   high bits corresponding to any of the excess limb bytes not covered by
///   `limbs` can have arbitrary values, but are ignored.
fn _be_mp_store_l_high_partial(limbs: &mut [u8], dst_end: usize, value: LimbType) {
    let dst = &mut limbs[0..dst_end];
    let src: [u8; LIMB_BYTES] = value.to_be_bytes();
    dst.copy_from_slice(&src[LIMB_BYTES - dst_end..LIMB_BYTES]);
}

/// Update some fully contained limb in a multiprecision integer big-endian byte
/// buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. The
/// generic [`be_mp_store_l()`] implements some extra logic for handling this
/// special case of a partially stored high limb, which can be avoided if it is
/// known that the limb to be accessed is completely covered by the `limbs`
/// buffer. This function here provides such a less generic, but more performant
/// variant.
///
/// Runs in constant time.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   big-endian order.
/// * `i` - The index of the limb to update, counted from least to most
///   significant.
/// * `value` - The value to store at limb index `i` in `limbs`.
fn be_mp_store_l_full(limbs: &mut [u8], i: usize, value: LimbType) {
    debug_assert!(i * LIMB_BYTES < limbs.len());
    let dst_end = limbs.len() - i * LIMB_BYTES;
    _be_mp_store_l_full(limbs, dst_end, value);
}

/// Update some limb in a multiprecision integer big-endian byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. If the
/// most significant limb is to be stored, all bytes in `value` corresponding to
/// these excess bytes must be zero accordingly.

/// If it is known that the limb to be stored at position `i` is fully covered
/// by the `limbs` buffer, consider using [`be_mp_store_l_full()`] for improved
/// performance instead.
///
/// Execution time depends only on the multiprecision integer's length and the
/// limb index argument `i`, and is otherwise constant as far as branching is
/// concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   big-endian order.
/// * `i` - The index of the limb to update, counted from least to most
///   significant.
/// * `value` - The value to store at limb index `i` in `limbs`.
fn be_mp_store_l(limbs: &mut [u8], i: usize, value: LimbType) {
    debug_assert!(i * LIMB_BYTES <= limbs.len());
    let dst_end = limbs.len() - i * LIMB_BYTES;
    if dst_end >= LIMB_BYTES {
        _be_mp_store_l_full(limbs, dst_end, value);
    } else {
        debug_assert_eq!(value >> (8 * dst_end), 0);
        _be_mp_store_l_high_partial(limbs, dst_end, value);
    }
}

#[test]
fn test_be_mp_store_l() {
    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    be_mp_store_l(&mut limbs, 0, 1 << (LIMB_BITS - 1));
    be_mp_store_l(&mut limbs, 1, 1);
    assert_eq!(be_mp_load_l(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(be_mp_load_l(&limbs, 1), 1);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    be_mp_store_l_full(&mut limbs, 0, 1 << (LIMB_BITS - 1));
    be_mp_store_l_full(&mut limbs, 1, 1);
    assert_eq!(be_mp_load_l(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(be_mp_load_l(&limbs, 1), 1);

    let mut limbs: [u8; 1] = [0; 1];
    be_mp_store_l(&mut limbs, 0, 1);
    assert_eq!(be_mp_load_l(&limbs, 0), 1);

    let mut limbs: [u8; LIMB_BYTES - 1] = [0; LIMB_BYTES - 1];
    be_mp_store_l(&mut limbs, 0, 1);
    assert_eq!(be_mp_load_l(&limbs, 0), 1);

    let mut limbs: [u8; LIMB_BYTES - 1] = [0; LIMB_BYTES - 1];
    be_mp_store_l(&mut limbs, 0, 1 << LIMB_BITS - 8 - 1);
    assert_eq!(be_mp_load_l(&limbs, 0), 1 << LIMB_BITS - 8 - 1);

    let mut limbs: [u8; 2] = [0; 2];
    be_mp_store_l(&mut limbs, 0, 1);
    assert_eq!(be_mp_load_l(&limbs, 0), 1);

    let mut limbs: [u8; 2] = [0; 2];
    be_mp_store_l(&mut limbs, 0, 0x0100);
    assert_eq!(be_mp_load_l(&limbs, 0), 0x0100);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    be_mp_store_l(&mut limbs, 1, 1);
    assert_eq!(be_mp_load_l(&limbs, 0), 0);
    assert_eq!(be_mp_load_l(&limbs, 1), 1);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    be_mp_store_l(&mut limbs, 1, 0x0100);
    assert_eq!(be_mp_load_l(&limbs, 0), 0);
    assert_eq!(be_mp_load_l(&limbs, 1), 0x0100);
}

fn be_mp_clear_bytes_above(limbs: &mut [u8], begin: usize) {
    let limbs_len = limbs.len();
    if limbs_len <= begin {
        return;
    }
    limbs[..limbs_len - begin].fill(0);
}

#[test]
fn test_be_mp_clear_bytes_above() {
    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    be_mp_store_l(&mut limbs, 0, !0);
    be_mp_store_l(&mut limbs, 1, !0 >> 8);
    be_mp_clear_bytes_above(&mut limbs, LIMB_BYTES + 1);
    assert_eq!(be_mp_load_l(&mut limbs, 0), !0);
    assert_eq!(be_mp_load_l(&mut limbs, 1), !0 & 0xff);

    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    be_mp_store_l(&mut limbs, 0, !0);
    be_mp_store_l(&mut limbs, 1, !0 >> 8);
    be_mp_clear_bytes_above(&mut limbs, LIMB_BYTES - 1);
    assert_eq!(be_mp_load_l(&mut limbs, 0), !0 >> 8);
    assert_eq!(be_mp_load_l(&mut limbs, 1), 0);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    be_mp_store_l(&mut limbs, 0, !0);
    be_mp_store_l(&mut limbs, 1, !0);
    be_mp_clear_bytes_above(&mut limbs, 2 * LIMB_BYTES);
    assert_eq!(be_mp_load_l(&mut limbs, 0), !0);
    assert_eq!(be_mp_load_l(&mut limbs, 1), !0);
}

fn be_mp_clear_bytes_below(limbs: &mut [u8], end: usize) {
    let limbs_len = limbs.len();
    let end = end.min(limbs_len);
    limbs[limbs_len - end..].fill(0);
}

#[test]
fn test_be_mp_clear_bytes_below() {
    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    be_mp_store_l(&mut limbs, 0, !0);
    be_mp_store_l(&mut limbs, 1, !0 >> 8);
    be_mp_clear_bytes_below(&mut limbs, LIMB_BYTES + 1);
    assert_eq!(be_mp_load_l(&mut limbs, 0), 0);
    assert_eq!(be_mp_load_l(&mut limbs, 1), (!0 >> 8) & !0xff);

    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    be_mp_store_l(&mut limbs, 0, !0);
    be_mp_store_l(&mut limbs, 1, !0 >> 8);
    be_mp_clear_bytes_below(&mut limbs, LIMB_BYTES - 1);
    assert_eq!(be_mp_load_l(&mut limbs, 0), 0xff << 8 * (LIMB_BYTES - 1));
    assert_eq!(be_mp_load_l(&mut limbs, 1), !0 >> 8);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    be_mp_store_l(&mut limbs, 0, !0);
    be_mp_store_l(&mut limbs, 1, !0);
    be_mp_clear_bytes_below(&mut limbs, 0);
    assert_eq!(be_mp_load_l(&mut limbs, 0), !0);
    assert_eq!(be_mp_load_l(&mut limbs, 1), !0);
}

/// Internal helper to load some fully contained limb from a multiprecision
/// integer little-endian byte buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. This
/// internal helper is used whenever the specified limb is not such a partially
/// covered high limb.
///
/// Runs in constant time as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   little-endian order.
/// * `src_begin` - The position of the limb to be loaded from `limbs`.
fn _le_mp_load_l_full(limbs: &[u8], src_begin: usize) -> LimbType {
    let src_end = src_begin + LIMB_BYTES;
    let src = &limbs[src_begin..src_end];
    let src = <[u8; LIMB_BYTES] as TryFrom<&[u8]>>::try_from(src).unwrap();
    LimbType::from_le_bytes(src)
}

/// Internal helper to load some partially contained limb from a multiprecision
/// integer little-endian byte buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. This
/// internal helper is used whenever the specified limb is such a partially
/// covered high limb.
///
/// Execution time depends only on the `src_end` argument and is otherwise
/// constant as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   little-endian order.
/// * `src_begin` - The position of the limb to be loaded from `limbs`.
fn _le_mp_load_l_high_partial(limbs: &[u8], src_begin: usize) -> LimbType {
    let mut src: [u8; LIMB_BYTES] = [0; LIMB_BYTES];
    src[..limbs.len() - src_begin].copy_from_slice(&limbs[src_begin..]);
    LimbType::from_le_bytes(src)
}

/// Load some fully contained limb from a multiprecision integer little-endian
/// byte buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. The
/// generic [`le_mp_load_l()`] implements some extra logic for handling this
/// special case of a partially stored high limb, which can be avoided if it is
/// known that the limb to be accessed is completely covered by the `limbs`
/// buffer. This function here provides such a less generic, but more performant
/// variant.
///
/// Runs in constant time as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   little-endian order.
/// * `i` - The index of the limb to load, counted from least to most
///   significant.
fn le_mp_load_l_full(limbs: &[u8], i: usize) -> LimbType {
    let src_begin = i * LIMB_BYTES;
    debug_assert!(src_begin < limbs.len());
    _le_mp_load_l_full(limbs, src_begin)
}

/// Load some limb from a multiprecision integer little-endian byte buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. If it
/// is known that the limb to be accessed at position `i` is fully covered by
/// the `limbs` buffer, consider using [`le_mp_load_l_full()`] for improved
/// performance instead.
///
/// Execution time depends only on the multiprecision integer's length and the
/// limb index argument `i`, and is otherwise constant as far as branching is
/// concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   little-endian order.
/// * `i` - The index of the limb to load, counted from least to most
///   significant.
fn le_mp_load_l(limbs: &[u8], i: usize) -> LimbType {
    let src_begin = i * LIMB_BYTES;
    debug_assert!(src_begin < limbs.len());
    if src_begin + LIMB_BYTES <= limbs.len() {
        _le_mp_load_l_full(limbs, src_begin)
    } else {
        _le_mp_load_l_high_partial(limbs, src_begin)
    }
}

#[test]
fn test_le_mp_load_l() {
    let limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    assert_eq!(le_mp_load_l(&limbs, 0), 0);
    assert_eq!(le_mp_load_l(&limbs, 1), 0);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    limbs[LIMB_BYTES - 1] = 0x80;
    limbs[LIMB_BYTES] = 1;
    assert_eq!(le_mp_load_l(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(le_mp_load_l_full(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(le_mp_load_l(&limbs, 1), 1);
    assert_eq!(le_mp_load_l_full(&limbs, 1), 1);

    let limbs: [u8; 1] = [0; 1];
    assert_eq!(le_mp_load_l(&limbs, 0), 0);

    let limbs: [u8; LIMB_BYTES + 1] = [0; LIMB_BYTES + 1];
    assert_eq!(le_mp_load_l(&limbs, 0), 0);
    assert_eq!(le_mp_load_l_full(&limbs, 0), 0);
    assert_eq!(le_mp_load_l(&limbs, 1), 0);

    let limbs: [u8; 2] = [1, 0];
    assert_eq!(le_mp_load_l(&limbs, 0), 1);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    limbs[LIMB_BYTES] = 1;
    assert_eq!(le_mp_load_l(&limbs, 0), 0);
    assert_eq!(le_mp_load_l_full(&limbs, 0), 0);
    assert_eq!(le_mp_load_l(&limbs, 1), 1);

    let limbs: [u8; 2] = [0, 1];
    assert_eq!(le_mp_load_l(&limbs, 0), 0x0100);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    limbs[LIMB_BYTES + 1] = 1;
    assert_eq!(le_mp_load_l(&limbs, 0), 0);
    assert_eq!(le_mp_load_l_full(&limbs, 0), 0);
    assert_eq!(le_mp_load_l(&limbs, 1), 0x0100);
}

/// Internal helper to update some fully contained limb in a multiprecision
/// integer little-endian byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. This
/// internal helper is used whenever the specified limb is not such a partially
/// covered high limb.
///
/// Runs in constant time as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   little-endian order.
/// * `dst_begin` - The position of the limb to be updated in `limbs`.
/// * `value` - The value to store in `limbs` at the specified position.
fn _le_mp_store_l_full(limbs: &mut [u8], dst_begin: usize, value: LimbType) {
    let dst_end = dst_begin + LIMB_BYTES;
    let dst = &mut limbs[dst_begin..dst_end];
    let dst = <&mut [u8; LIMB_BYTES] as TryFrom<&mut [u8]>>::try_from(dst).unwrap();
    *dst = value.to_le_bytes();
}

/// Internal helper to update some partially contained limb in a multiprecision
/// integer little-endian byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. This
/// internal helper is used whenever the specified limb is not such a partially
/// covered high limb.
///
/// Execution time depends only on the `dst_end` argument and is otherwise
/// constant as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   little-endian order.
/// * `dst_begin` - The position of the limb to be updated in `limbs`.
/// * `value` - The value to store in `limbs` at the specified position. The
///   high bits corresponding to any of the excess limb bytes not covered by
///   `limbs` can have arbitrary values, but are ignored.
fn _le_mp_store_l_high_partial(limbs: &mut [u8], dst_begin: usize, value: LimbType) {
    let dst_end = limbs.len();
    let dst = &mut limbs[dst_begin..];
    let src: [u8; LIMB_BYTES] = value.to_le_bytes();
    dst.copy_from_slice(&src[0..dst_end - dst_begin]);
}

/// Update some fully contained limb in a multiprecision integer little-endian
/// byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. The
/// generic [`le_mp_store_l()`] implements some extra logic for handling this
/// special case of a partially stored high limb, which can be avoided if it is
/// known that the limb to be accessed is completely covered by the `limbs`
/// buffer. This function here provides such a less generic, but more performant
/// variant.
///
/// Runs in constant time.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   little-endian order.
/// * `i` - The index of the limb to update, counted from least to most
///   significant.
/// * `value` - The value to store at limb index `i` in `limbs`.
fn le_mp_store_l_full(limbs: &mut [u8], i: usize, value: LimbType) {
    let dst_begin = i * LIMB_BYTES;
    debug_assert!(dst_begin < limbs.len());
    _le_mp_store_l_full(limbs, dst_begin, value);
}

/// Update some limb in a multiprecision integer little-endian byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a
/// [`LimbType`], in which case the most significant limb's might be stored only
/// partially, with its virtual excess high bytes defined to equal zero. If the
/// most significant limb is to be stored, all bytes in `value` corresponding to
/// these excess bytes must be zero accordingly.

/// If it is known that the limb to be stored at position `i` is fully covered
/// by the `limbs` buffer, consider using [`le_mp_store_l_full()`] for improved
/// performance instead.
///
/// Execution time depends only on the multiprecision integer's length and the
/// limb index argument `i`, and is otherwise constant as far as branching is
/// concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in
///   little-endian order.
/// * `i` - The index of the limb to update, counted from least to most
///   significant.
/// * `value` - The value to store at limb index `i` in `limbs`.
fn le_mp_store_l(limbs: &mut [u8], i: usize, value: LimbType) {
    let dst_begin = i * LIMB_BYTES;
    debug_assert!(dst_begin < limbs.len());
    if dst_begin + LIMB_BYTES <= limbs.len() {
        _le_mp_store_l_full(limbs, dst_begin, value);
    } else {
        debug_assert_eq!(value >> (8 * (limbs.len() - dst_begin)), 0);
        _le_mp_store_l_high_partial(limbs, dst_begin, value);
    }
}

#[test]
fn test_le_mp_store_l() {
    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    le_mp_store_l(&mut limbs, 0, 1 << (LIMB_BITS - 1));
    le_mp_store_l(&mut limbs, 1, 1);
    assert_eq!(le_mp_load_l(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(le_mp_load_l(&limbs, 1), 1);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    le_mp_store_l_full(&mut limbs, 0, 1 << (LIMB_BITS - 1));
    le_mp_store_l_full(&mut limbs, 1, 1);
    assert_eq!(le_mp_load_l(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(le_mp_load_l(&limbs, 1), 1);

    let mut limbs: [u8; 1] = [0; 1];
    le_mp_store_l(&mut limbs, 0, 1);
    assert_eq!(le_mp_load_l(&limbs, 0), 1);

    let mut limbs: [u8; LIMB_BYTES - 1] = [0; LIMB_BYTES - 1];
    le_mp_store_l(&mut limbs, 0, 1);
    assert_eq!(le_mp_load_l(&limbs, 0), 1);

    let mut limbs: [u8; LIMB_BYTES - 1] = [0; LIMB_BYTES - 1];
    le_mp_store_l(&mut limbs, 0, 1 << LIMB_BITS - 8 - 1);
    assert_eq!(le_mp_load_l(&limbs, 0), 1 << LIMB_BITS - 8 - 1);

    let mut limbs: [u8; 2] = [0; 2];
    le_mp_store_l(&mut limbs, 0, 1);
    assert_eq!(le_mp_load_l(&limbs, 0), 1);

    let mut limbs: [u8; 2] = [0; 2];
    le_mp_store_l(&mut limbs, 0, 0x0100);
    assert_eq!(le_mp_load_l(&limbs, 0), 0x0100);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    le_mp_store_l(&mut limbs, 1, 1);
    assert_eq!(le_mp_load_l(&limbs, 0), 0);
    assert_eq!(le_mp_load_l(&limbs, 1), 1);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    le_mp_store_l(&mut limbs, 1, 0x0100);
    assert_eq!(le_mp_load_l(&limbs, 0), 0);
    assert_eq!(le_mp_load_l(&limbs, 1), 0x0100);
}

fn le_mp_clear_bytes_above(limbs: &mut [u8], begin: usize) {
    if limbs.len() <= begin {
        return;
    }
    limbs[begin..].fill(0);
}

#[test]
fn test_le_mp_clear_bytes_above() {
    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    le_mp_store_l(&mut limbs, 0, !0);
    le_mp_store_l(&mut limbs, 1, !0 >> 8);
    le_mp_clear_bytes_above(&mut limbs, LIMB_BYTES + 1);
    assert_eq!(le_mp_load_l(&mut limbs, 0), !0);
    assert_eq!(le_mp_load_l(&mut limbs, 1), !0 & 0xff);

    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    le_mp_store_l(&mut limbs, 0, !0);
    le_mp_store_l(&mut limbs, 1, !0 >> 8);
    le_mp_clear_bytes_above(&mut limbs, LIMB_BYTES - 1);
    assert_eq!(le_mp_load_l(&mut limbs, 0), !0 >> 8);
    assert_eq!(le_mp_load_l(&mut limbs, 1), 0);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    le_mp_store_l(&mut limbs, 0, !0);
    le_mp_store_l(&mut limbs, 1, !0);
    le_mp_clear_bytes_above(&mut limbs, 2 * LIMB_BYTES);
    assert_eq!(le_mp_load_l(&mut limbs, 0), !0);
    assert_eq!(le_mp_load_l(&mut limbs, 1), !0);
}

fn le_mp_clear_bytes_below(limbs: &mut [u8], end: usize) {
    let end = end.min(limbs.len());
    limbs[..end].fill(0);
}

#[test]
fn test_le_mp_clear_bytes_below() {
    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    le_mp_store_l(&mut limbs, 0, !0);
    le_mp_store_l(&mut limbs, 1, !0 >> 8);
    le_mp_clear_bytes_below(&mut limbs, LIMB_BYTES + 1);
    assert_eq!(le_mp_load_l(&mut limbs, 0), 0);
    assert_eq!(le_mp_load_l(&mut limbs, 1), (!0 >> 8) & !0xff);

    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    le_mp_store_l(&mut limbs, 0, !0);
    le_mp_store_l(&mut limbs, 1, !0 >> 8);
    le_mp_clear_bytes_below(&mut limbs, LIMB_BYTES - 1);
    assert_eq!(le_mp_load_l(&mut limbs, 0), 0xff << 8 * (LIMB_BYTES - 1));
    assert_eq!(le_mp_load_l(&mut limbs, 1), !0 >> 8);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    le_mp_store_l(&mut limbs, 0, !0);
    le_mp_store_l(&mut limbs, 1, !0);
    le_mp_clear_bytes_below(&mut limbs, 0);
    assert_eq!(le_mp_load_l(&mut limbs, 0), !0);
    assert_eq!(le_mp_load_l(&mut limbs, 1), !0);
}

pub trait MpUIntCommonPriv: Sized {
    const SUPPORTS_UNALIGNED_BUFFER_LENGTHS: bool;

    fn _len(&self) -> usize;

    fn partial_high_mask(&self) -> LimbType {
        if Self::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
            let high_npartial = if self._len() != 0 {
                ((self._len()) - 1) % LIMB_BYTES + 1
            } else {
                0
            };
            ct_lsb_mask_l(8 * high_npartial as u32)
        } else {
            !0
        }
    }

    fn partial_high_shift(&self) -> u32 {
        if Self::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
            let high_npartial = self._len() % LIMB_BYTES;
            if high_npartial == 0 {
                0
            } else {
                8 * high_npartial as u32
            }
        } else {
            0
        }
    }
}

#[derive(Debug)]
pub struct MpUIntCommonTryIntoNativeError {}

macro_rules! _mpu_try_into_native_u {
    ($nt:ty, $name:ident) => {
        fn $name(&self) -> Result<$nt, MpUIntCommonTryIntoNativeError> {
            let native_type_nlimbs = ct_mp_nlimbs(mem::size_of::<$nt>());
            let nbytes_from_last = ((mem::size_of::<$nt>() - 1) % LIMB_BYTES) + 1;
            let mut head_is_nonzero = 0;
            for i in native_type_nlimbs..self.nlimbs() {
                head_is_nonzero |= self.load_l(i);
            }
            let last_val = self.load_l(native_type_nlimbs - 1);
            let last_mask = ct_lsb_mask_l(8 * nbytes_from_last as u32);
            head_is_nonzero |= last_val & !last_mask;
            if ct_is_nonzero_l(head_is_nonzero) != 0 {
                return Err(MpUIntCommonTryIntoNativeError {});
            }
            let mut result: $nt = (last_val & last_mask) as $nt;
            let mut i = native_type_nlimbs - 1;
            while i > 0 {
                i -= 1;
                // This loop only gets executed in case
                // LIMB_BITS < $nt::BITS. Avoid a compiler error for smaller
                // native types due to a too large shift distance.
                result <<= LIMB_BITS.min(<$nt>::BITS - 1);
                result |= self.load_l_full(i) as $nt;
            }
            Ok(result)
        }
    };
}

pub trait MpUIntCommon: MpUIntCommonPriv + fmt::LowerHex {
    fn len(&self) -> usize {
        self._len()
    }

    fn nlimbs(&self) -> usize {
        ct_mp_nlimbs(self.len())
    }

    fn is_empty(&self) -> bool;

    fn load_l_full(&self, i: usize) -> LimbType;
    fn load_l(&self, i: usize) -> LimbType;

    fn test_bit(&self, pos: usize) -> LimbChoice {
        let limb_index = pos / LIMB_BITS as usize;
        if limb_index >= self.nlimbs() {
            return LimbChoice::from(0);
        }
        let pos_in_limb = pos % LIMB_BITS as usize;
        let l = self.load_l(limb_index);
        LimbChoice::from((l >> pos_in_limb) & 1)
    }

    fn fmt_lower_hex(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn nibble_to_hexchar(nibble: u8) -> char {
            let c = match nibble {
                0x0..=0x9 => b'0' + nibble,
                0xa..=0xf => b'a' + (nibble - 0xa),
                _ => unreachable!(),
            };
            c as char
        }

        fn fmt_byte(f: &mut fmt::Formatter<'_>, v: u8) -> fmt::Result {
            <fmt::Formatter<'_> as fmt::Write>::write_char(f, nibble_to_hexchar(v >> 4))?;
            <fmt::Formatter<'_> as fmt::Write>::write_char(f, nibble_to_hexchar(v & 0xf))?;
            Ok(())
        }

        fn fmt_l(f: &mut fmt::Formatter<'_>, v: LimbType, len: usize) -> fmt::Result {
            for i in 0..len {
                fmt_byte(f, (v >> (8 * (len - i - 1))) as u8)?;
            }
            Ok(())
        }

        if f.alternate() {
            f.write_str("0x")?;
        }
        if self.is_empty() {
            f.write_str("(empty)")?;
            return Ok(());
        }

        let v = self.load_l(self.nlimbs() - 1);
        fmt_l(f, v, (self.len() - 1) % LIMB_BYTES + 1)?;

        let mut i = 0;
        while i + 1 < self.nlimbs() {
            <fmt::Formatter<'_> as fmt::Write>::write_char(f, '_')?;
            let v = self.load_l(self.nlimbs() - 2 - i);
            fmt_l(f, v, LIMB_BYTES)?;
            i += 1;
        }

        Ok(())
    }

    fn len_is_compatible_with(&self, len: usize) -> bool {
        if self._len() <= len {
            true
        } else if !Self::SUPPORTS_UNALIGNED_BUFFER_LENGTHS && self._len() - len < LIMB_BYTES {
            // Allow for aligment excess
            self.load_l(self.nlimbs() - 1) >> (8 * (LIMB_BYTES - (self._len() - len))) == 0
        } else {
            false
        }
    }

    _mpu_try_into_native_u!(u8, try_into_u8);
    _mpu_try_into_native_u!(u16, try_into_u16);
    _mpu_try_into_native_u!(u32, try_into_u32);
    _mpu_try_into_native_u!(u64, try_into_u64);
}

pub trait MpUInt: MpUIntCommon {}

macro_rules! _mpu_set_to_native_u {
    ($nt:ty, $name:ident) => {
        fn $name(&mut self, mut value: $nt) {
            debug_assert!(self.len() >= mem::size_of::<$nt>());
            self.clear_bytes_above(mem::size_of::<$nt>());
            let native_type_nlimbs = ct_mp_nlimbs(mem::size_of::<$nt>());
            for i in 0..native_type_nlimbs {
                self.store_l(i, (value & (!(0 as LimbType) as $nt)) as LimbType);
                // This loop only gets executed more than once in case LIMB_BITS < $nt::BITS.
                // Avoid a compiler error for smaller native types due to a too large
                // shift distance.
                value >>= LIMB_BITS.min(<$nt>::BITS - 1);
            }
        }
    };
}

pub trait MpMutUInt: MpUIntCommon {
    fn store_l_full(&mut self, i: usize, value: LimbType);
    fn store_l(&mut self, i: usize, value: LimbType);
    fn clear_bytes_above(&mut self, begin: usize);
    fn clear_bytes_below(&mut self, end: usize);

    fn clear_bytes_above_cond(&mut self, begin: usize, cond: LimbChoice) {
        if begin >= self.len() {
            return;
        }

        let mask = cond.select(!0, 0);
        let begin_limb = begin / LIMB_BYTES;
        let begin_in_limb = begin % LIMB_BYTES;
        let val = self.load_l(begin_limb);
        let first_mask = !ct_lsb_mask_l(8 * begin_in_limb as u32) | mask;
        let val = val & first_mask;
        self.store_l(begin_limb, val);

        for i in begin_limb + 1..self.nlimbs() {
            self.store_l(i, self.load_l(i) & mask);
        }
    }

    fn copy_from<S: MpUIntCommon>(&'_ mut self, src: &S) {
        let src_nlimbs = src.nlimbs();
        let dst_nlimbs = self.nlimbs();
        debug_assert!(find_last_set_byte_mp(src) <= self.len());

        if src_nlimbs == 0 {
            self.clear_bytes_above(0);
            return;
        } else if dst_nlimbs == 0 {
            return;
        }
        let common_nlimbs = src_nlimbs.min(dst_nlimbs);
        for i in 0..common_nlimbs - 1 {
            self.store_l_full(i, src.load_l_full(i));
        }
        let high_limb = src.load_l(common_nlimbs - 1);
        debug_assert!(src_nlimbs < dst_nlimbs || (high_limb & !self.partial_high_mask()) == 0);
        self.store_l(common_nlimbs - 1, high_limb);
        self.clear_bytes_above(src.len());
    }

    fn copy_from_cond<S: MpUIntCommon>(&'_ mut self, src: &S, cond: LimbChoice) {
        let src_nlimbs = src.nlimbs();
        let dst_nlimbs = self.nlimbs();
        debug_assert!(find_last_set_byte_mp(src) <= self.len());

        if src_nlimbs == 0 {
            self.clear_bytes_above_cond(0, cond);
            return;
        } else if dst_nlimbs == 0 {
            return;
        }
        let common_nlimbs = src_nlimbs.min(dst_nlimbs);
        for i in 0..common_nlimbs - 1 {
            let val = cond.select(self.load_l_full(i), src.load_l_full(i));
            self.store_l_full(i, val);
        }
        let high_limb = cond.select(
            self.load_l(common_nlimbs - 1),
            src.load_l(common_nlimbs - 1),
        );
        debug_assert!(src_nlimbs < dst_nlimbs || (high_limb & !self.partial_high_mask()) == 0);
        self.store_l(common_nlimbs - 1, high_limb);
        self.clear_bytes_above_cond(src.len(), cond);
    }

    fn set_bit_to(&mut self, pos: usize, val: bool) {
        let limb_index = pos / LIMB_BITS as usize;
        debug_assert!(limb_index < self.nlimbs());
        let pos_in_limb = pos % LIMB_BITS as usize;
        let mut l = self.load_l(limb_index);
        let bit_pos_mask = (1 as LimbType) << pos_in_limb;
        let val_mask = LimbChoice::from(val as LimbType).select(0, bit_pos_mask);
        l &= !bit_pos_mask;
        l |= val_mask;
        self.store_l(limb_index, l)
    }

    _mpu_set_to_native_u!(u8, set_to_u8);
    _mpu_set_to_native_u!(u16, set_to_u16);
    _mpu_set_to_native_u!(u32, set_to_u32);
    _mpu_set_to_native_u!(u64, set_to_u64);
}

pub trait MpUIntSliceCommonPriv: MpUIntCommonPriv {
    type BackingSliceElementType: Sized + Copy + cmp::PartialEq + convert::From<u8> + fmt::Debug;

    const BACKING_ELEMENT_SIZE: usize = mem::size_of::<Self::BackingSliceElementType>();

    fn _limbs_align_len(nbytes: usize) -> usize {
        if Self::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
            nbytes
        } else {
            ct_mp_limbs_align_len(nbytes)
        }
    }

    fn n_backing_elements_for_len(nbytes: usize) -> usize {
        let nbytes = Self::_limbs_align_len(nbytes);
        nbytes / Self::BACKING_ELEMENT_SIZE + (nbytes % Self::BACKING_ELEMENT_SIZE != 0) as usize
    }

    fn n_backing_elements(&self) -> usize;

    fn take(self, nbytes: usize) -> (Self, Self);
}

pub trait MpUIntSliceCommon: MpUIntSliceCommonPriv + MpUIntCommon {}

pub trait MpUIntSlicePriv: MpUIntSliceCommon + MpUInt {
    type SelfT<'a>: MpUIntSlice<BackingSliceElementType = Self::BackingSliceElementType>
    where
        Self: 'a;

    type FromSliceError: fmt::Debug;

    fn from_slice<'a: 'b, 'b>(
        s: &'a [Self::BackingSliceElementType],
    ) -> Result<Self::SelfT<'b>, Self::FromSliceError>
    where
        Self: 'b;

    fn _shrink_to(&self, nbytes: usize) -> Self::SelfT<'_>;
}

pub trait MpUIntSlice: MpUIntSlicePriv + MpUInt {
    fn coerce_lifetime(&self) -> Self::SelfT<'_>;

    fn shrink_to(&self, nbytes: usize) -> Self::SelfT<'_> {
        let nbytes = nbytes.min(self._len());
        debug_assert!(nbytes >= find_last_set_byte_mp(self));
        self._shrink_to(nbytes)
    }
}

pub trait MpMutUIntSlicePriv: MpUIntSliceCommon + MpMutUInt {
    type SelfT<'a>: MpMutUIntSlice<BackingSliceElementType = Self::BackingSliceElementType>
    where
        Self: 'a;

    type FromSliceError: fmt::Debug;

    fn from_slice<'a: 'b, 'b>(
        s: &'a mut [Self::BackingSliceElementType],
    ) -> Result<Self::SelfT<'b>, Self::FromSliceError>
    where
        Self: 'b;

    fn _shrink_to(&mut self, nbytes: usize) -> Self::SelfT<'_>;
}

pub trait MpMutUIntSlice: MpMutUIntSlicePriv + MpMutUInt {
    fn coerce_lifetime(&mut self) -> Self::SelfT<'_>;

    fn shrink_to(&mut self, nbytes: usize) -> Self::SelfT<'_> {
        let nbytes = nbytes.min(self._len());
        debug_assert!(nbytes >= find_last_set_byte_mp(self));
        self._shrink_to(nbytes)
    }
}

#[derive(Clone)]
pub struct MpBigEndianUIntByteSlice<'a> {
    bytes: &'a [u8],
}

impl<'a> MpBigEndianUIntByteSlice<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &'a [u8] {
        &self.bytes
    }
}

impl<'a> MpUIntCommonPriv for MpBigEndianUIntByteSlice<'a> {
    const SUPPORTS_UNALIGNED_BUFFER_LENGTHS: bool = true;

    fn _len(&self) -> usize {
        self.bytes.len()
    }
}

impl<'a> MpUIntCommon for MpBigEndianUIntByteSlice<'a> {
    fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn load_l_full(&self, i: usize) -> LimbType {
        be_mp_load_l_full(self.bytes, i)
    }

    fn load_l(&self, i: usize) -> LimbType {
        be_mp_load_l(self.bytes, i)
    }
}

impl<'a> MpUInt for MpBigEndianUIntByteSlice<'a> {}

impl<'a> MpUIntSliceCommonPriv for MpBigEndianUIntByteSlice<'a> {
    type BackingSliceElementType = u8;

    fn n_backing_elements(&self) -> usize {
        self.bytes.len()
    }

    fn take(self, nbytes: usize) -> (Self, Self) {
        let (h, l) = self.bytes.split_at(self.bytes.len() - nbytes);
        (Self { bytes: h }, Self { bytes: l })
    }
}

impl<'a> MpUIntSliceCommon for MpBigEndianUIntByteSlice<'a> {}

impl<'a> MpUIntSlicePriv for MpBigEndianUIntByteSlice<'a> {
    type SelfT<'b>
        = MpBigEndianUIntByteSlice<'b>
    where
        Self: 'b;

    type FromSliceError = convert::Infallible;

    fn from_slice<'b: 'c, 'c>(
        s: &'b [Self::BackingSliceElementType],
    ) -> Result<Self::SelfT<'c>, Self::FromSliceError>
    where
        Self: 'c,
    {
        Ok(Self::SelfT::<'c> { bytes: s })
    }

    fn _shrink_to(&self, nbytes: usize) -> Self::SelfT<'_> {
        MpBigEndianUIntByteSlice {
            bytes: &self.bytes[self.bytes.len() - nbytes..],
        }
    }
}

impl<'a> MpUIntSlice for MpBigEndianUIntByteSlice<'a> {
    fn coerce_lifetime(&self) -> Self::SelfT<'_> {
        MpBigEndianUIntByteSlice { bytes: &self.bytes }
    }
}

impl<'a> fmt::LowerHex for MpBigEndianUIntByteSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_lower_hex(f)
    }
}

impl<'a, 'b> From<&'a MpMutBigEndianUIntByteSlice<'b>> for MpBigEndianUIntByteSlice<'a> {
    fn from(value: &'a MpMutBigEndianUIntByteSlice<'b>) -> Self {
        Self { bytes: value.bytes }
    }
}

impl<'a> From<MpBigEndianUIntByteSlice<'a>> for &'a [u8] {
    fn from(value: MpBigEndianUIntByteSlice<'a>) -> Self {
        value.bytes
    }
}

pub struct MpMutBigEndianUIntByteSlice<'a> {
    bytes: &'a mut [u8],
}

impl<'a> MpMutBigEndianUIntByteSlice<'a> {
    pub fn from_bytes(bytes: &'a mut [u8]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes<'b>(&'b self) -> &'b [u8] {
        &self.bytes
    }

    pub fn as_bytes_mut<'b>(&'b mut self) -> &'b mut [u8] {
        &mut self.bytes
    }
}

impl<'a> MpUIntCommonPriv for MpMutBigEndianUIntByteSlice<'a> {
    const SUPPORTS_UNALIGNED_BUFFER_LENGTHS: bool = true;

    fn _len(&self) -> usize {
        self.bytes.len()
    }
}

impl<'a> MpUIntCommon for MpMutBigEndianUIntByteSlice<'a> {
    fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn load_l_full(&self, i: usize) -> LimbType {
        be_mp_load_l_full(self.bytes, i)
    }

    fn load_l(&self, i: usize) -> LimbType {
        be_mp_load_l(self.bytes, i)
    }
}

impl<'a> MpMutUInt for MpMutBigEndianUIntByteSlice<'a> {
    fn store_l_full(&mut self, i: usize, value: LimbType) {
        be_mp_store_l_full(self.bytes, i, value)
    }

    fn store_l(&mut self, i: usize, value: LimbType) {
        be_mp_store_l(self.bytes, i, value)
    }

    fn clear_bytes_above(&mut self, begin: usize) {
        be_mp_clear_bytes_above(self.bytes, begin)
    }

    fn clear_bytes_below(&mut self, end: usize) {
        be_mp_clear_bytes_below(self.bytes, end)
    }
}

impl<'a> MpUIntSliceCommonPriv for MpMutBigEndianUIntByteSlice<'a> {
    type BackingSliceElementType = u8;

    fn n_backing_elements(&self) -> usize {
        self.bytes.len()
    }

    fn take(self, nbytes: usize) -> (Self, Self) {
        let (h, l) = self.bytes.split_at_mut(self.bytes.len() - nbytes);
        (Self { bytes: h }, Self { bytes: l })
    }
}

impl<'a> MpUIntSliceCommon for MpMutBigEndianUIntByteSlice<'a> {}

impl<'a> MpMutUIntSlicePriv for MpMutBigEndianUIntByteSlice<'a> {
    type SelfT<'b>
        = MpMutBigEndianUIntByteSlice<'b>
    where
        Self: 'b;

    type FromSliceError = convert::Infallible;

    fn from_slice<'b: 'c, 'c>(
        s: &'b mut [Self::BackingSliceElementType],
    ) -> Result<Self::SelfT<'c>, Self::FromSliceError>
    where
        Self: 'c,
    {
        Ok(Self::SelfT::<'c> { bytes: s })
    }

    fn _shrink_to(&mut self, nbytes: usize) -> Self::SelfT<'_> {
        let l = self.bytes.len();
        MpMutBigEndianUIntByteSlice {
            bytes: &mut self.bytes[l - nbytes..],
        }
    }
}

impl<'a> MpMutUIntSlice for MpMutBigEndianUIntByteSlice<'a> {
    fn coerce_lifetime(&mut self) -> Self::SelfT<'_> {
        MpMutBigEndianUIntByteSlice {
            bytes: &mut self.bytes,
        }
    }
}

impl<'a> fmt::LowerHex for MpMutBigEndianUIntByteSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_lower_hex(f)
    }
}

impl<'a, 'b> From<&'a mut MpMutBigEndianUIntByteSlice<'b>> for &'a mut [u8] {
    fn from(value: &'a mut MpMutBigEndianUIntByteSlice<'b>) -> Self {
        &mut value.bytes
    }
}

impl<'a, 'b> From<&'a MpMutBigEndianUIntByteSlice<'b>> for &'a [u8] {
    fn from(value: &'a MpMutBigEndianUIntByteSlice<'b>) -> Self {
        &value.bytes
    }
}

#[derive(Clone)]
pub struct MpLittleEndianUIntByteSlice<'a> {
    bytes: &'a [u8],
}

impl<'a> MpLittleEndianUIntByteSlice<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &'a [u8] {
        &self.bytes
    }
}

impl<'a> MpUIntCommonPriv for MpLittleEndianUIntByteSlice<'a> {
    const SUPPORTS_UNALIGNED_BUFFER_LENGTHS: bool = true;

    fn _len(&self) -> usize {
        self.bytes.len()
    }
}

impl<'a> MpUIntCommon for MpLittleEndianUIntByteSlice<'a> {
    fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn load_l_full(&self, i: usize) -> LimbType {
        le_mp_load_l_full(self.bytes, i)
    }

    fn load_l(&self, i: usize) -> LimbType {
        le_mp_load_l(self.bytes, i)
    }
}

impl<'a> MpUInt for MpLittleEndianUIntByteSlice<'a> {}

impl<'a> MpUIntSliceCommonPriv for MpLittleEndianUIntByteSlice<'a> {
    type BackingSliceElementType = u8;

    fn n_backing_elements(&self) -> usize {
        self.bytes.len()
    }

    fn take(self, nbytes: usize) -> (Self, Self) {
        let (l, h) = self.bytes.split_at(nbytes);
        (Self { bytes: h }, Self { bytes: l })
    }
}

impl<'a> MpUIntSliceCommon for MpLittleEndianUIntByteSlice<'a> {}

impl<'a> MpUIntSlicePriv for MpLittleEndianUIntByteSlice<'a> {
    type SelfT<'b>
        = MpLittleEndianUIntByteSlice<'b>
    where
        Self: 'b;

    type FromSliceError = convert::Infallible;

    fn from_slice<'b: 'c, 'c>(
        s: &'b [Self::BackingSliceElementType],
    ) -> Result<Self::SelfT<'c>, Self::FromSliceError>
    where
        Self: 'c,
    {
        Ok(Self::SelfT::<'c> { bytes: s })
    }

    fn _shrink_to(&self, nbytes: usize) -> Self::SelfT<'_> {
        MpLittleEndianUIntByteSlice {
            bytes: &self.bytes[..nbytes],
        }
    }
}

impl<'a> MpUIntSlice for MpLittleEndianUIntByteSlice<'a> {
    fn coerce_lifetime(&self) -> Self::SelfT<'_> {
        MpLittleEndianUIntByteSlice { bytes: &self.bytes }
    }
}

impl<'a> fmt::LowerHex for MpLittleEndianUIntByteSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_lower_hex(f)
    }
}

impl<'a, 'b> From<&'a MpMutLittleEndianUIntByteSlice<'b>> for MpLittleEndianUIntByteSlice<'a> {
    fn from(value: &'a MpMutLittleEndianUIntByteSlice<'b>) -> Self {
        Self { bytes: value.bytes }
    }
}

impl<'a> From<MpLittleEndianUIntByteSlice<'a>> for &'a [u8] {
    fn from(value: MpLittleEndianUIntByteSlice<'a>) -> Self {
        value.bytes
    }
}

pub struct MpMutLittleEndianUIntByteSlice<'a> {
    bytes: &'a mut [u8],
}

impl<'a> MpMutLittleEndianUIntByteSlice<'a> {
    pub fn from_bytes(bytes: &'a mut [u8]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes<'b>(&'b self) -> &'b [u8] {
        &self.bytes
    }

    pub fn as_bytes_mut<'b>(&'b mut self) -> &'b mut [u8] {
        &mut self.bytes
    }
}

impl<'a> MpUIntCommonPriv for MpMutLittleEndianUIntByteSlice<'a> {
    const SUPPORTS_UNALIGNED_BUFFER_LENGTHS: bool = true;

    fn _len(&self) -> usize {
        self.bytes.len()
    }
}

impl<'a> MpUIntCommon for MpMutLittleEndianUIntByteSlice<'a> {
    fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn load_l_full(&self, i: usize) -> LimbType {
        le_mp_load_l_full(self.bytes, i)
    }

    fn load_l(&self, i: usize) -> LimbType {
        le_mp_load_l(self.bytes, i)
    }
}

impl<'a> MpMutUInt for MpMutLittleEndianUIntByteSlice<'a> {
    fn store_l_full(&mut self, i: usize, value: LimbType) {
        le_mp_store_l_full(self.bytes, i, value)
    }

    fn store_l(&mut self, i: usize, value: LimbType) {
        le_mp_store_l(self.bytes, i, value)
    }

    fn clear_bytes_above(&mut self, begin: usize) {
        le_mp_clear_bytes_above(self.bytes, begin)
    }

    fn clear_bytes_below(&mut self, end: usize) {
        le_mp_clear_bytes_below(self.bytes, end)
    }
}

impl<'a> MpUIntSliceCommonPriv for MpMutLittleEndianUIntByteSlice<'a> {
    type BackingSliceElementType = u8;

    fn n_backing_elements(&self) -> usize {
        self.bytes.len()
    }

    fn take(self, nbytes: usize) -> (Self, Self) {
        let (l, h) = self.bytes.split_at_mut(nbytes);
        (Self { bytes: h }, Self { bytes: l })
    }
}

impl<'a> MpUIntSliceCommon for MpMutLittleEndianUIntByteSlice<'a> {}

impl<'a> MpMutUIntSlicePriv for MpMutLittleEndianUIntByteSlice<'a> {
    type SelfT<'b>
        = MpMutLittleEndianUIntByteSlice<'b>
    where
        Self: 'b;

    type FromSliceError = convert::Infallible;

    fn from_slice<'b: 'c, 'c>(
        s: &'b mut [Self::BackingSliceElementType],
    ) -> Result<Self::SelfT<'c>, Self::FromSliceError>
    where
        Self: 'c,
    {
        Ok(Self::SelfT::<'c> { bytes: s })
    }

    fn _shrink_to(&mut self, nbytes: usize) -> Self::SelfT<'_> {
        MpMutLittleEndianUIntByteSlice {
            bytes: &mut self.bytes[..nbytes],
        }
    }
}

impl<'a> MpMutUIntSlice for MpMutLittleEndianUIntByteSlice<'a> {
    fn coerce_lifetime(&mut self) -> Self::SelfT<'_> {
        MpMutLittleEndianUIntByteSlice {
            bytes: &mut self.bytes,
        }
    }
}

impl<'a> fmt::LowerHex for MpMutLittleEndianUIntByteSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_lower_hex(f)
    }
}

impl<'a, 'b> From<&'a mut MpMutLittleEndianUIntByteSlice<'b>> for &'a mut [u8] {
    fn from(value: &'a mut MpMutLittleEndianUIntByteSlice<'b>) -> Self {
        &mut value.bytes
    }
}

impl<'a, 'b> From<&'a MpMutLittleEndianUIntByteSlice<'b>> for &'a [u8] {
    fn from(value: &'a MpMutLittleEndianUIntByteSlice<'b>) -> Self {
        &value.bytes
    }
}

pub struct MpNativeEndianUIntLimbsSlice<'a> {
    limbs: &'a [LimbType],
}

impl<'a> MpNativeEndianUIntLimbsSlice<'a> {
    pub fn from_limbs(limbs: &'a [LimbType]) -> Self {
        Self { limbs }
    }

    pub fn nlimbs_for_len(nbytes: usize) -> usize {
        Self::n_backing_elements_for_len(nbytes)
    }
}

impl<'a> MpUIntCommonPriv for MpNativeEndianUIntLimbsSlice<'a> {
    const SUPPORTS_UNALIGNED_BUFFER_LENGTHS: bool = false;

    fn _len(&self) -> usize {
        self.limbs.len() * Self::BACKING_ELEMENT_SIZE
    }
}

impl<'a> MpUIntCommon for MpNativeEndianUIntLimbsSlice<'a> {
    fn is_empty(&self) -> bool {
        self.limbs.is_empty()
    }

    #[inline(always)]
    fn load_l_full(&self, i: usize) -> LimbType {
        self.limbs[i]
    }

    #[inline(always)]
    fn load_l(&self, i: usize) -> LimbType {
        self.load_l_full(i)
    }
}

impl<'a> MpUInt for MpNativeEndianUIntLimbsSlice<'a> {}

impl<'a> MpUIntSliceCommonPriv for MpNativeEndianUIntLimbsSlice<'a> {
    type BackingSliceElementType = LimbType;

    fn n_backing_elements(&self) -> usize {
        self.limbs.len()
    }

    fn take(self, nbytes: usize) -> (Self, Self) {
        debug_assert_eq!(nbytes % LIMB_BYTES, 0);
        let (l, h) = self
            .limbs
            .split_at(Self::n_backing_elements_for_len(nbytes));
        (Self { limbs: h }, Self { limbs: l })
    }
}

impl<'a> MpUIntSliceCommon for MpNativeEndianUIntLimbsSlice<'a> {}

impl<'a> MpUIntSlicePriv for MpNativeEndianUIntLimbsSlice<'a> {
    type SelfT<'b>
        = MpNativeEndianUIntLimbsSlice<'b>
    where
        Self: 'b;

    type FromSliceError = convert::Infallible;

    fn from_slice<'b: 'c, 'c>(
        s: &'b [Self::BackingSliceElementType],
    ) -> Result<Self::SelfT<'c>, Self::FromSliceError>
    where
        Self: 'c,
    {
        Ok(Self::SelfT::<'c> { limbs: s })
    }

    fn _shrink_to(&self, nbytes: usize) -> Self::SelfT<'_> {
        let nlimbs = Self::n_backing_elements_for_len(nbytes);
        MpNativeEndianUIntLimbsSlice {
            limbs: &self.limbs[..nlimbs],
        }
    }
}

impl<'a> MpUIntSlice for MpNativeEndianUIntLimbsSlice<'a> {
    fn coerce_lifetime(&self) -> Self::SelfT<'_> {
        MpNativeEndianUIntLimbsSlice { limbs: &self.limbs }
    }
}

impl<'a> fmt::LowerHex for MpNativeEndianUIntLimbsSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_lower_hex(f)
    }
}

impl<'a, 'b> From<&'a MpMutNativeEndianUIntLimbsSlice<'b>> for MpNativeEndianUIntLimbsSlice<'a> {
    fn from(value: &'a MpMutNativeEndianUIntLimbsSlice<'b>) -> Self {
        Self { limbs: value.limbs }
    }
}

pub struct MpMutNativeEndianUIntLimbsSlice<'a> {
    limbs: &'a mut [LimbType],
}

impl<'a> MpMutNativeEndianUIntLimbsSlice<'a> {
    pub fn from_limbs(limbs: &'a mut [LimbType]) -> Self {
        Self { limbs }
    }

    pub fn nlimbs_for_len(nbytes: usize) -> usize {
        Self::n_backing_elements_for_len(nbytes)
    }
}

impl<'a> MpUIntCommonPriv for MpMutNativeEndianUIntLimbsSlice<'a> {
    const SUPPORTS_UNALIGNED_BUFFER_LENGTHS: bool = false;

    fn _len(&self) -> usize {
        self.limbs.len() * Self::BACKING_ELEMENT_SIZE
    }
}

impl<'a> MpUIntCommon for MpMutNativeEndianUIntLimbsSlice<'a> {
    fn is_empty(&self) -> bool {
        self.limbs.is_empty()
    }

    #[inline(always)]
    fn load_l_full(&self, i: usize) -> LimbType {
        self.limbs[i]
    }

    #[inline(always)]
    fn load_l(&self, i: usize) -> LimbType {
        self.load_l_full(i)
    }
}

impl<'a> MpMutUInt for MpMutNativeEndianUIntLimbsSlice<'a> {
    #[inline(always)]
    fn store_l_full(&mut self, i: usize, value: LimbType) {
        self.limbs[i] = value;
    }

    #[inline(always)]
    fn store_l(&mut self, i: usize, value: LimbType) {
        self.store_l_full(i, value);
    }

    fn clear_bytes_above(&mut self, begin: usize) {
        let mut begin_limb = begin / LIMB_BYTES;
        if begin_limb >= self.limbs.len() {
            return;
        }
        let begin_in_limb = begin % LIMB_BYTES;
        if begin_in_limb != 0 {
            self.limbs[begin_limb] &= ct_lsb_mask_l(8 * begin_in_limb as u32);
            begin_limb += 1;
        }
        self.limbs[begin_limb..].fill(0);
    }

    fn clear_bytes_below(&mut self, end: usize) {
        let mut end_limb = ct_mp_nlimbs(end).min(self.limbs.len());
        let end_in_limb = end % LIMB_BYTES;
        if end_in_limb != 0 {
            end_limb -= 1;
            let mut l = self.load_l(end_limb);
            l >>= 8 * end_in_limb;
            l <<= 8 * end_in_limb;
            self.limbs[end_limb] = l;
        }
        self.limbs[..end_limb].fill(0);
    }
}

impl<'a> MpUIntSliceCommonPriv for MpMutNativeEndianUIntLimbsSlice<'a> {
    type BackingSliceElementType = LimbType;

    fn n_backing_elements(&self) -> usize {
        self.limbs.len()
    }

    fn take(self, nbytes: usize) -> (Self, Self) {
        debug_assert_eq!(nbytes % LIMB_BYTES, 0);
        let (l, h) = self
            .limbs
            .split_at_mut(Self::n_backing_elements_for_len(nbytes));
        (Self { limbs: h }, Self { limbs: l })
    }
}

impl<'a> MpUIntSliceCommon for MpMutNativeEndianUIntLimbsSlice<'a> {}

impl<'a> MpMutUIntSlicePriv for MpMutNativeEndianUIntLimbsSlice<'a> {
    type SelfT<'b>
        = MpMutNativeEndianUIntLimbsSlice<'b>
    where
        Self: 'b;

    type FromSliceError = convert::Infallible;

    fn from_slice<'b: 'c, 'c>(
        s: &'b mut [Self::BackingSliceElementType],
    ) -> Result<Self::SelfT<'c>, Self::FromSliceError>
    where
        Self: 'c,
    {
        Ok(Self::SelfT::<'c> { limbs: s })
    }

    fn _shrink_to(&mut self, nbytes: usize) -> Self::SelfT<'_> {
        let nlimbs = Self::n_backing_elements_for_len(nbytes);
        MpMutNativeEndianUIntLimbsSlice {
            limbs: &mut self.limbs[..nlimbs],
        }
    }
}

impl<'a> MpMutUIntSlice for MpMutNativeEndianUIntLimbsSlice<'a> {
    fn coerce_lifetime(&mut self) -> Self::SelfT<'_> {
        MpMutNativeEndianUIntLimbsSlice {
            limbs: &mut self.limbs,
        }
    }
}

impl<'a> fmt::LowerHex for MpMutNativeEndianUIntLimbsSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_lower_hex(f)
    }
}

fn find_last_set_limb<T0: MpUIntCommon>(op0: &T0) -> usize {
    let mut nlimbs = op0.nlimbs();
    if nlimbs == 0 {
        return 0;
    }

    if op0.load_l(nlimbs - 1) == 0 {
        nlimbs -= 1;
        while nlimbs > 0 {
            if op0.load_l_full(nlimbs - 1) != 0 {
                break;
            }
            nlimbs -= 1;
        }
    }

    nlimbs
}

#[cfg(test)]
fn test_find_last_set_limb_with_unaligned_lengths<
    T0: MpMutUIntSlice<BackingSliceElementType = u8>,
>() {
    let mut op0 = [0u8; 2 * LIMB_BYTES + 2];
    let mut op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
    op0.store_l(0, 1);
    assert_eq!(find_last_set_limb(&op0), 1);

    op0.store_l(1, 1);
    assert_eq!(find_last_set_limb(&op0), 2);

    op0.store_l(2, 1);
    assert_eq!(find_last_set_limb(&op0), 3);
}

#[cfg(test)]
fn test_find_last_set_limb_with_aligned_lengths<T0: MpMutUIntSlice>() {
    let mut op0 = tst_mk_mp_backing_vec!(T0, 0);
    let op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
    assert_eq!(find_last_set_limb(&op0), 0);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
    assert_eq!(find_last_set_limb(&op0), 0);

    op0.store_l(0, 1);
    assert_eq!(find_last_set_limb(&op0), 1);

    op0.store_l(1, 1);
    assert_eq!(find_last_set_limb(&op0), 2);
}

#[test]
fn test_find_last_set_limb_be() {
    test_find_last_set_limb_with_unaligned_lengths::<MpMutBigEndianUIntByteSlice>();
    test_find_last_set_limb_with_aligned_lengths::<MpMutBigEndianUIntByteSlice>();
}

#[test]
fn test_find_last_set_limb_le() {
    test_find_last_set_limb_with_unaligned_lengths::<MpMutLittleEndianUIntByteSlice>();
    test_find_last_set_limb_with_aligned_lengths::<MpMutLittleEndianUIntByteSlice>();
}

#[test]
fn test_find_last_set_limb_ne() {
    test_find_last_set_limb_with_aligned_lengths::<MpMutNativeEndianUIntLimbsSlice>();
}

pub fn find_last_set_byte_mp<T0: MpUIntCommon>(op0: &T0) -> usize {
    let nlimbs = find_last_set_limb(op0);
    if nlimbs == 0 {
        return 0;
    }
    let nlimbs = nlimbs - 1;
    nlimbs * LIMB_BYTES + ct_find_last_set_byte_l(op0.load_l(nlimbs))
}

#[cfg(test)]
fn test_find_last_set_byte_mp_with_unaligned_lengths<
    T0: MpMutUIntSlice<BackingSliceElementType = u8>,
>() {
    let mut op0 = [0u8; 2 * LIMB_BYTES + 2];
    let mut op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
    op0.store_l(0, 1);
    assert_eq!(find_last_set_byte_mp(&op0), 1);

    op0.store_l(1, 1);
    assert_eq!(find_last_set_byte_mp(&op0), LIMB_BYTES + 1);

    op0.store_l(2, 1);
    assert_eq!(find_last_set_byte_mp(&op0), 2 * LIMB_BYTES + 1);
}

#[cfg(test)]
fn test_find_last_set_byte_mp_with_aligned_lengths<T0: MpMutUIntSlice>() {
    let mut op0 = tst_mk_mp_backing_vec!(T0, 0);
    let op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
    assert_eq!(find_last_set_byte_mp(&op0), 0);

    let mut op0 = tst_mk_mp_backing_vec!(T0, 2 * LIMB_BYTES);
    let mut op0 = T0::from_slice(op0.as_mut_slice()).unwrap();
    op0.store_l(0, (1 as LimbType) << LIMB_BITS - 1);
    assert_eq!(find_last_set_byte_mp(&op0), LIMB_BYTES);

    op0.store_l(1, (1 as LimbType) << LIMB_BITS - 1);
    assert_eq!(find_last_set_byte_mp(&op0), 2 * LIMB_BYTES);
}

#[test]
fn test_find_last_set_byte_be() {
    test_find_last_set_byte_mp_with_unaligned_lengths::<MpMutBigEndianUIntByteSlice>();
    test_find_last_set_byte_mp_with_aligned_lengths::<MpMutBigEndianUIntByteSlice>();
}

#[test]
fn test_find_last_set_byte_le() {
    test_find_last_set_byte_mp_with_unaligned_lengths::<MpMutLittleEndianUIntByteSlice>();
    test_find_last_set_byte_mp_with_aligned_lengths::<MpMutLittleEndianUIntByteSlice>();
}

#[test]
fn test_find_last_set_byte_ne() {
    test_find_last_set_byte_mp_with_aligned_lengths::<MpMutNativeEndianUIntLimbsSlice>();
}

pub fn ct_find_first_set_bit_mp<T0: MpUIntCommon>(op0: &T0) -> (LimbChoice, usize) {
    let mut tail_is_zero = LimbChoice::from(1);
    let mut ntrailing_zeroes: usize = 0;
    for i in 0..op0.nlimbs() {
        let op0_val = op0.load_l(i);
        ntrailing_zeroes +=
            tail_is_zero.select(0, ct_find_first_set_bit_l(op0_val) as LimbType) as usize;
        tail_is_zero &= LimbChoice::from(ct_is_zero_l(op0_val));
    }
    (
        !tail_is_zero,
        tail_is_zero.select_usize(ntrailing_zeroes, 0),
    )
}

#[cfg(test)]
fn test_ct_find_first_set_bit_mp<T0: MpMutUIntSlice>() {
    let mut limbs = tst_mk_mp_backing_vec!(T0, 3 * LIMB_BYTES);
    let limbs = T0::from_slice(&mut limbs).unwrap();
    let (is_nonzero, first_set_bit_pos) = ct_find_first_set_bit_mp(&limbs);
    assert_eq!(is_nonzero.unwrap(), 0);
    assert_eq!(first_set_bit_pos, 0);

    for i in 0..3 * LIMB_BITS as usize {
        let mut limbs = tst_mk_mp_backing_vec!(T0, 3 * LIMB_BYTES);
        let mut limbs = T0::from_slice(&mut limbs).unwrap();
        let limb_index = i / LIMB_BITS as usize;
        let bit_pos_in_limb = i % LIMB_BITS as usize;

        limbs.store_l(limb_index, 1 << bit_pos_in_limb);
        let (is_nonzero, first_set_bit_pos) = ct_find_first_set_bit_mp(&limbs);
        assert!(is_nonzero.unwrap() != 0);
        assert_eq!(first_set_bit_pos, i);

        limbs.store_l(limb_index, !((1 << bit_pos_in_limb) - 1));
        for j in limb_index + 1..limbs.nlimbs() {
            limbs.store_l(j, !0);
        }
        let (is_nonzero, first_set_bit_pos) = ct_find_first_set_bit_mp(&limbs);
        assert!(is_nonzero.unwrap() != 0);
        assert_eq!(first_set_bit_pos, i);
    }
}

#[test]
fn test_ct_find_first_set_bit_be() {
    test_ct_find_first_set_bit_mp::<MpMutBigEndianUIntByteSlice>()
}

#[test]
fn test_ct_find_first_set_bit_le() {
    test_ct_find_first_set_bit_mp::<MpMutLittleEndianUIntByteSlice>()
}

#[test]
fn test_ct_find_first_set_bit_ne() {
    test_ct_find_first_set_bit_mp::<MpMutNativeEndianUIntLimbsSlice>()
}

pub fn ct_find_last_set_bit_mp<T0: MpUIntCommon>(op0: &T0) -> (LimbChoice, usize) {
    let mut head_is_zero = LimbChoice::from(1);
    let mut nleading_zeroes: usize = 0;
    let mut i = op0.nlimbs();
    while i > 0 {
        i -= 1;
        let op0_val = op0.load_l(i);
        nleading_zeroes += head_is_zero.select(
            0,
            LIMB_BITS as LimbType - ct_find_last_set_bit_l(op0_val) as LimbType,
        ) as usize;
        head_is_zero &= LimbChoice::from(ct_is_zero_l(op0_val));
    }
    (
        !head_is_zero,
        op0.nlimbs() * LIMB_BITS as usize - nleading_zeroes,
    )
}

#[cfg(test)]
fn test_ct_find_last_set_bit_mp<T0: MpMutUIntSlice>() {
    let mut limbs = tst_mk_mp_backing_vec!(T0, 3 * LIMB_BYTES);
    let limbs = T0::from_slice(&mut limbs).unwrap();
    let (is_nonzero, first_set_bit_pos) = ct_find_last_set_bit_mp(&limbs);
    assert_eq!(is_nonzero.unwrap(), 0);
    assert_eq!(first_set_bit_pos, 0);

    for i in 0..3 * LIMB_BITS as usize {
        let mut limbs = tst_mk_mp_backing_vec!(T0, 3 * LIMB_BYTES);
        let mut limbs = T0::from_slice(&mut limbs).unwrap();
        let limb_index = i / LIMB_BITS as usize;
        let bit_pos_in_limb = i % LIMB_BITS as usize;

        limbs.store_l(limb_index, 1 << bit_pos_in_limb);
        let (is_nonzero, last_set_bit_pos) = ct_find_last_set_bit_mp(&limbs);
        assert!(is_nonzero.unwrap() != 0);
        assert_eq!(last_set_bit_pos, i + 1);

        limbs.store_l(limb_index, (1 << bit_pos_in_limb) - 1);
        for j in 0..limb_index {
            limbs.store_l(j, !0);
        }
        let (is_nonzero, last_set_bit_pos) = ct_find_last_set_bit_mp(&limbs);
        assert_eq!(is_nonzero.unwrap() != 0, i != 0);
        assert_eq!(last_set_bit_pos, i);
    }
}

#[test]
fn test_ct_find_last_set_bit_be() {
    test_ct_find_last_set_bit_mp::<MpMutBigEndianUIntByteSlice>()
}

#[test]
fn test_ct_find_last_set_bit_le() {
    test_ct_find_last_set_bit_mp::<MpMutLittleEndianUIntByteSlice>()
}

#[test]
fn test_ct_find_last_set_bit_ne() {
    test_ct_find_last_set_bit_mp::<MpMutNativeEndianUIntLimbsSlice>()
}

pub fn ct_clear_bits_above_mp<T0: MpMutUInt>(op0: &mut T0, begin: usize) {
    let first_limb_index = begin / LIMB_BITS as usize;
    let first_limb_retain_nbits = begin % LIMB_BITS as usize;
    let first_limb_mask = ct_lsb_mask_l(first_limb_retain_nbits as u32);
    let mut next_mask = !0;
    for i in 0..op0.nlimbs() {
        let is_first_limb = ct_eq_usize_usize(i, first_limb_index);
        let mask = is_first_limb.select(next_mask, first_limb_mask);
        next_mask = is_first_limb.select(next_mask, 0);
        let val = op0.load_l(i);
        op0.store_l(i, val & mask)
    }
}

#[cfg(test)]
fn test_ct_clear_bits_above_mp_common<T0: MpMutUIntSlice>(op0_len: usize) {
    fn fill_with_ones<T0: MpMutUInt>(op0: &mut T0) {
        for i in 0..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                op0.store_l_full(i, !0);
            } else {
                op0.store_l(i, op0.partial_high_mask());
            }
        }
    }

    let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    fill_with_ones(&mut op0);
    for begin in [
        8 * op0_len,
        8 * op0_len + 1,
        8 * op0.nlimbs() * LIMB_BYTES,
        8 * op0.nlimbs() * LIMB_BYTES + 1,
    ] {
        ct_clear_bits_above_mp(&mut op0, begin);
        for i in 0..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                assert_eq!(op0.load_l_full(i), !0);
            } else {
                assert_eq!(op0.load_l(i), op0.partial_high_mask());
            }
        }
    }

    for j in 0..ct_mp_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize;

        let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
        let mut op0 = T0::from_slice(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        ct_clear_bits_above_mp(&mut op0, begin);
        for i in 0..j {
            assert_eq!(op0.load_l_full(i), !0);
        }

        for i in j..op0.nlimbs() {
            assert_eq!(op0.load_l(i), 0);
        }
    }

    for j in 1..ct_mp_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize - 1;
        let begin = begin.min(8 * op0_len - 1);

        let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
        let mut op0 = T0::from_slice(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        ct_clear_bits_above_mp(&mut op0, begin);
        for i in 0..j - 1 {
            assert_eq!(op0.load_l_full(i), !0);
        }

        let expected = ct_lsb_mask_l((begin % LIMB_BITS as usize) as u32);
        assert_eq!(op0.load_l(j - 1), expected);

        for i in j..op0.nlimbs() {
            assert_eq!(op0.load_l(i), 0);
        }
    }
}

#[cfg(test)]
fn test_ct_clear_bits_above_mp_with_aligned_lengths<T0: MpMutUIntSlice>() {
    test_ct_clear_bits_above_mp_common::<T0>(0);
    test_ct_clear_bits_above_mp_common::<T0>(LIMB_BYTES);
    test_ct_clear_bits_above_mp_common::<T0>(2 * LIMB_BYTES);
    test_ct_clear_bits_above_mp_common::<T0>(3 * LIMB_BYTES);
}

#[cfg(test)]
fn test_ct_clear_bits_above_mp_with_unaligned_lengths<T0: MpMutUIntSlice>() {
    test_ct_clear_bits_above_mp_common::<T0>(LIMB_BYTES - 1);
    test_ct_clear_bits_above_mp_common::<T0>(2 * LIMB_BYTES - 1);
    test_ct_clear_bits_above_mp_common::<T0>(3 * LIMB_BYTES - 1);
}

#[test]
fn test_ct_clear_bits_above_be() {
    test_ct_clear_bits_above_mp_with_aligned_lengths::<MpMutBigEndianUIntByteSlice>();
    test_ct_clear_bits_above_mp_with_unaligned_lengths::<MpMutBigEndianUIntByteSlice>();
}

#[test]
fn test_ct_clear_bits_above_le() {
    test_ct_clear_bits_above_mp_with_aligned_lengths::<MpMutLittleEndianUIntByteSlice>();
    test_ct_clear_bits_above_mp_with_unaligned_lengths::<MpMutLittleEndianUIntByteSlice>();
}

#[test]
fn test_ct_clear_bits_above_ne() {
    test_ct_clear_bits_above_mp_with_aligned_lengths::<MpMutNativeEndianUIntLimbsSlice>();
}

pub fn clear_bits_above_mp<T0: MpMutUInt>(op0: &mut T0, begin: usize) {
    let first_limb_index = begin / LIMB_BITS as usize;
    if op0.nlimbs() <= first_limb_index {
        return;
    }
    let first_limb_retain_nbits = begin % LIMB_BITS as usize;
    let first_limb_mask = ct_lsb_mask_l(first_limb_retain_nbits as u32);
    op0.store_l(
        first_limb_index,
        op0.load_l(first_limb_index) & first_limb_mask,
    );

    op0.clear_bytes_above((begin + LIMB_BITS as usize - 1) / LIMB_BITS as usize * LIMB_BYTES);
}

#[cfg(test)]
fn test_clear_bits_above_mp_common<T0: MpMutUIntSlice>(op0_len: usize) {
    fn fill_with_ones<T0: MpMutUInt>(op0: &mut T0) {
        for i in 0..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                op0.store_l_full(i, !0);
            } else {
                op0.store_l(i, op0.partial_high_mask());
            }
        }
    }

    let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    fill_with_ones(&mut op0);
    for begin in [
        8 * op0_len,
        8 * op0_len + 1,
        8 * op0.nlimbs() * LIMB_BYTES,
        8 * op0.nlimbs() * LIMB_BYTES + 1,
    ] {
        clear_bits_above_mp(&mut op0, begin);
        for i in 0..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                assert_eq!(op0.load_l_full(i), !0);
            } else {
                assert_eq!(op0.load_l(i), op0.partial_high_mask());
            }
        }
    }

    for j in 0..ct_mp_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize;

        let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
        let mut op0 = T0::from_slice(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        clear_bits_above_mp(&mut op0, begin);
        for i in 0..j {
            assert_eq!(op0.load_l_full(i), !0);
        }

        for i in j..op0.nlimbs() {
            assert_eq!(op0.load_l(i), 0);
        }
    }

    for j in 1..ct_mp_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize - 1;
        let begin = begin.min(8 * op0_len - 1);

        let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
        let mut op0 = T0::from_slice(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        clear_bits_above_mp(&mut op0, begin);
        for i in 0..j - 1 {
            assert_eq!(op0.load_l_full(i), !0);
        }

        let expected = ct_lsb_mask_l((begin % LIMB_BITS as usize) as u32);
        assert_eq!(op0.load_l(j - 1), expected);

        for i in j..op0.nlimbs() {
            assert_eq!(op0.load_l(i), 0);
        }
    }
}

#[cfg(test)]
fn test_clear_bits_above_mp_with_aligned_lengths<T0: MpMutUIntSlice>() {
    test_clear_bits_above_mp_common::<T0>(0);
    test_clear_bits_above_mp_common::<T0>(LIMB_BYTES);
    test_clear_bits_above_mp_common::<T0>(2 * LIMB_BYTES);
    test_clear_bits_above_mp_common::<T0>(3 * LIMB_BYTES);
}

#[cfg(test)]
fn test_clear_bits_above_mp_with_unaligned_lengths<T0: MpMutUIntSlice>() {
    test_clear_bits_above_mp_common::<T0>(LIMB_BYTES - 1);
    test_clear_bits_above_mp_common::<T0>(2 * LIMB_BYTES - 1);
    test_clear_bits_above_mp_common::<T0>(3 * LIMB_BYTES - 1);
}

#[test]
fn test_clear_bits_above_be() {
    test_clear_bits_above_mp_with_aligned_lengths::<MpMutBigEndianUIntByteSlice>();
    test_clear_bits_above_mp_with_unaligned_lengths::<MpMutBigEndianUIntByteSlice>();
}

#[test]
fn test_clear_bits_above_le() {
    test_clear_bits_above_mp_with_aligned_lengths::<MpMutLittleEndianUIntByteSlice>();
    test_clear_bits_above_mp_with_unaligned_lengths::<MpMutLittleEndianUIntByteSlice>();
}

#[test]
fn test_clear_bits_above_ne() {
    test_clear_bits_above_mp_with_aligned_lengths::<MpMutNativeEndianUIntLimbsSlice>();
}

pub fn ct_clear_bits_below_mp<T0: MpMutUInt>(op0: &mut T0, end: usize) {
    let last_limb_index = end / LIMB_BITS as usize;
    let last_limb_clear_nbits = end % LIMB_BITS as usize;
    let last_limb_mask = !ct_lsb_mask_l(last_limb_clear_nbits as u32);
    let mut next_mask = 0;
    for i in 0..op0.nlimbs() {
        let is_last_limb = ct_eq_usize_usize(i, last_limb_index);
        let mask = is_last_limb.select(next_mask, last_limb_mask);
        next_mask = is_last_limb.select(next_mask, !0);
        let val = op0.load_l(i);
        op0.store_l(i, val & mask)
    }
}

#[cfg(test)]
fn test_ct_clear_bits_mp_below_common<T0: MpMutUIntSlice>(op0_len: usize) {
    fn fill_with_ones<T0: MpMutUInt>(op0: &mut T0) {
        for i in 0..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                op0.store_l_full(i, !0);
            } else {
                op0.store_l(i, op0.partial_high_mask());
            }
        }
    }

    let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    fill_with_ones(&mut op0);
    ct_clear_bits_below_mp(&mut op0, 0);
    for i in 0..op0.nlimbs() {
        if i + 1 != op0.nlimbs() {
            assert_eq!(op0.load_l_full(i), !0);
        } else {
            assert_eq!(op0.load_l(i), op0.partial_high_mask());
        }
    }

    for j in 0..ct_mp_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize;

        let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
        let mut op0 = T0::from_slice(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        ct_clear_bits_below_mp(&mut op0, begin);
        for i in 0..j {
            assert_eq!(op0.load_l_full(i), 0);
        }

        for i in j..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                assert_eq!(op0.load_l(i), !0);
            } else {
                assert_eq!(op0.load_l(i), op0.partial_high_mask());
            }
        }
    }

    for j in 1..ct_mp_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize - 1;
        let begin = begin.min(8 * op0_len - 1);

        let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
        let mut op0 = T0::from_slice(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        ct_clear_bits_below_mp(&mut op0, begin);
        for i in 0..j - 1 {
            assert_eq!(op0.load_l_full(i), 0);
        }

        let expected = !ct_lsb_mask_l((begin % LIMB_BITS as usize) as u32);
        assert_eq!(op0.load_l(j - 1), expected);

        for i in j..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                assert_eq!(op0.load_l(i), !0);
            } else {
                assert_eq!(op0.load_l(i), op0.partial_high_mask());
            }
        }
    }
}

#[cfg(test)]
fn test_ct_clear_bits_below_mp_with_aligned_lengths<T0: MpMutUIntSlice>() {
    test_ct_clear_bits_mp_below_common::<T0>(0);
    test_ct_clear_bits_mp_below_common::<T0>(LIMB_BYTES);
    test_ct_clear_bits_mp_below_common::<T0>(2 * LIMB_BYTES);
    test_ct_clear_bits_mp_below_common::<T0>(3 * LIMB_BYTES);
}

#[cfg(test)]
fn test_ct_clear_bits_below_mp_with_unaligned_lengths<T0: MpMutUIntSlice>() {
    test_ct_clear_bits_mp_below_common::<T0>(LIMB_BYTES - 1);
    test_ct_clear_bits_mp_below_common::<T0>(2 * LIMB_BYTES - 1);
    test_ct_clear_bits_mp_below_common::<T0>(3 * LIMB_BYTES - 1);
}

#[test]
fn test_ct_clear_bits_below_be() {
    test_ct_clear_bits_below_mp_with_aligned_lengths::<MpMutBigEndianUIntByteSlice>();
    test_ct_clear_bits_below_mp_with_unaligned_lengths::<MpMutBigEndianUIntByteSlice>();
}

#[test]
fn test_ct_clear_bits_below_le() {
    test_ct_clear_bits_below_mp_with_aligned_lengths::<MpMutLittleEndianUIntByteSlice>();
    test_ct_clear_bits_below_mp_with_unaligned_lengths::<MpMutLittleEndianUIntByteSlice>();
}

#[test]
fn test_ct_clear_bits_below_ne() {
    test_ct_clear_bits_below_mp_with_aligned_lengths::<MpMutNativeEndianUIntLimbsSlice>();
}

pub fn clear_bits_below_mp<T0: MpMutUInt>(op0: &mut T0, end: usize) {
    let last_limb_index = end / LIMB_BITS as usize;
    op0.clear_bytes_below(last_limb_index * LIMB_BYTES);
    if last_limb_index >= op0.nlimbs() {
        return;
    }
    let last_limb_clear_nbits = end % LIMB_BITS as usize;
    let last_limb_mask = !ct_lsb_mask_l(last_limb_clear_nbits as u32);
    op0.store_l(
        last_limb_index,
        op0.load_l(last_limb_index) & last_limb_mask,
    );
}

#[cfg(test)]
fn test_clear_bits_below_mp_common<T0: MpMutUIntSlice>(op0_len: usize) {
    fn fill_with_ones<T0: MpMutUInt>(op0: &mut T0) {
        for i in 0..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                op0.store_l_full(i, !0);
            } else {
                op0.store_l(i, op0.partial_high_mask());
            }
        }
    }

    let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
    let mut op0 = T0::from_slice(&mut op0).unwrap();
    fill_with_ones(&mut op0);
    clear_bits_below_mp(&mut op0, 0);
    for i in 0..op0.nlimbs() {
        if i + 1 != op0.nlimbs() {
            assert_eq!(op0.load_l_full(i), !0);
        } else {
            assert_eq!(op0.load_l(i), op0.partial_high_mask());
        }
    }

    for j in 0..ct_mp_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize;

        let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
        let mut op0 = T0::from_slice(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        clear_bits_below_mp(&mut op0, begin);
        for i in 0..j {
            assert_eq!(op0.load_l_full(i), 0);
        }

        for i in j..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                assert_eq!(op0.load_l(i), !0);
            } else {
                assert_eq!(op0.load_l(i), op0.partial_high_mask());
            }
        }
    }

    for j in 1..ct_mp_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize - 1;
        let begin = begin.min(8 * op0_len - 1);

        let mut op0 = tst_mk_mp_backing_vec!(T0, op0_len);
        let mut op0 = T0::from_slice(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        clear_bits_below_mp(&mut op0, begin);
        for i in 0..j - 1 {
            assert_eq!(op0.load_l_full(i), 0);
        }

        let expected = !ct_lsb_mask_l((begin % LIMB_BITS as usize) as u32);
        assert_eq!(op0.load_l(j - 1), expected);

        for i in j..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                assert_eq!(op0.load_l(i), !0);
            } else {
                assert_eq!(op0.load_l(i), op0.partial_high_mask());
            }
        }
    }
}

#[cfg(test)]
fn test_clear_bits_below_mp_with_aligned_lengths<T0: MpMutUIntSlice>() {
    test_clear_bits_below_mp_common::<T0>(0);
    test_clear_bits_below_mp_common::<T0>(LIMB_BYTES);
    test_clear_bits_below_mp_common::<T0>(2 * LIMB_BYTES);
    test_clear_bits_below_mp_common::<T0>(3 * LIMB_BYTES);
}

#[cfg(test)]
fn test_clear_bits_below_mp_with_unaligned_lengths<T0: MpMutUIntSlice>() {
    test_clear_bits_below_mp_common::<T0>(LIMB_BYTES - 1);
    test_clear_bits_below_mp_common::<T0>(2 * LIMB_BYTES - 1);
    test_clear_bits_below_mp_common::<T0>(3 * LIMB_BYTES - 1);
}

#[test]
fn test_clear_bits_below_be() {
    test_clear_bits_below_mp_with_aligned_lengths::<MpMutBigEndianUIntByteSlice>();
    test_clear_bits_below_mp_with_unaligned_lengths::<MpMutBigEndianUIntByteSlice>();
}

#[test]
fn test_clear_bits_below_le() {
    test_clear_bits_below_mp_with_aligned_lengths::<MpMutLittleEndianUIntByteSlice>();
    test_clear_bits_below_mp_with_unaligned_lengths::<MpMutLittleEndianUIntByteSlice>();
}

#[test]
fn test_clear_bits_below_ne() {
    test_clear_bits_below_mp_with_aligned_lengths::<MpMutNativeEndianUIntLimbsSlice>();
}

pub fn ct_swap_cond_mp<T0: MpMutUInt, T1: MpMutUInt>(op0: &mut T0, op1: &mut T1, cond: LimbChoice) {
    debug_assert_eq!(op0.nlimbs(), op1.nlimbs());
    let nlimbs = op0.nlimbs();
    let cond_mask = cond.select(0, !0);
    for i in 0..nlimbs {
        let mut op0_val = op0.load_l(i);
        let mut op1_val = op1.load_l(i);
        op0_val ^= op1_val;
        op1_val ^= op0_val & cond_mask;
        op0_val ^= op1_val;
        op0.store_l(i, op0_val);
        op1.store_l(i, op1_val);
    }
}

#[cfg(test)]
fn test_ct_swap_cond_mp<T0: MpMutUIntSlice, T1: MpMutUIntSlice>() {
    use super::cmp_impl::ct_eq_mp_mp;

    let len = 2 * LIMB_BYTES - 1;
    let mut op0_orig = tst_mk_mp_backing_vec!(T0, len);
    op0_orig.fill(0xccu8.into());
    let mut op0 = op0_orig.clone();
    let op0_orig = T0::from_slice(&mut op0_orig).unwrap();
    let mut op0 = T0::from_slice(&mut op0).unwrap();

    let mut op1_orig = tst_mk_mp_backing_vec!(T1, len);
    op1_orig.fill(0xbbu8.into());
    let mut op1 = op1_orig.clone();
    let op1_orig = T1::from_slice(&mut op1_orig).unwrap();
    let mut op1 = T1::from_slice(&mut op1).unwrap();

    ct_swap_cond_mp(&mut op0, &mut op1, LimbChoice::from(0));
    assert_ne!(ct_eq_mp_mp(&op0, &op0_orig).unwrap(), 0);
    assert_ne!(ct_eq_mp_mp(&op1, &op1_orig).unwrap(), 0);

    ct_swap_cond_mp(&mut op0, &mut op1, LimbChoice::from(1));
    assert_ne!(ct_eq_mp_mp(&op0, &op1_orig).unwrap(), 0);
    assert_ne!(ct_eq_mp_mp(&op1, &op0_orig).unwrap(), 0);
}

#[test]
fn test_ct_swap_cond_be_be() {
    test_ct_swap_cond_mp::<MpMutBigEndianUIntByteSlice, MpMutBigEndianUIntByteSlice>();
}

#[test]
fn test_ct_swap_cond_le_le() {
    test_ct_swap_cond_mp::<MpMutLittleEndianUIntByteSlice, MpMutLittleEndianUIntByteSlice>();
}

#[test]
fn test_ct_swap_cond_ne_ne() {
    test_ct_swap_cond_mp::<MpMutNativeEndianUIntLimbsSlice, MpMutNativeEndianUIntLimbsSlice>();
}

/// Internal data structure describing a single one of a
/// [`CompositeLimbsBuffer`]'s constituting segments.
struct CompositeLimbsBufferSegment<'a, ST: MpUIntCommon> {
    /// The total number of limbs whose least significant bytes are held in this
    /// or less significant segments each (note that a limb may span
    /// multiple segments).
    end: usize,
    /// The actual data, interpreted as a segment within a composed
    /// multiprecision integer byte buffer.
    ///
    /// The `segment` slice, if non-empty, will be made to align with the least
    /// significant limb whose least significant byte is found in this
    /// segment. That is, the slice's last byte (in memory order)
    /// corresponds to the least significant byte of the least significant limb.
    segment: ST,
    /// Continuation bytes of this or preceeding, less significant segment's
    /// most significant limb.
    ///
    /// A segment's most significant limb might be covered only partially by it
    /// and extend across one or more subsequent, more significant segments.
    /// In this case its remaining, more significant continutation bytes are
    /// collected from this and the subsequent, more significant
    /// segments' `high_next_partial` slices.
    high_next_partial: ST,

    _phantom: marker::PhantomData<&'a [u8]>,
}

/// Access multiprecision integers composed of multiple virtually concatenated
/// byte buffers in units of [`LimbType`].
///
/// A [`CompositeLimbsBuffer`] provides a composed multiprecision integer byte
/// buffer view on `N_SEGMENTS` virtually concatenated byte slices, of endianess
/// as specified by the `E` generic parameter each. Primitives are provided
/// alongside for accessing the composed integer in units of [`LimbType`].
///
/// Certain applications need to truncate the result of some multiprecision
/// integer arithmetic operation result and dismiss the rest. An example would
/// be key generation modulo some large integer by the method of oversampling. A
/// `CompositeLimbsBuffer` helps such applications to reduce the memory
/// footprint of the truncation operation: instead of allocating a smaller
/// destination buffer and copying the to be retained parts over from the
/// result, the arithmetic primitive can, if supported, operate directly on a
/// multiprecision integer byte buffer composed of several independent smaller
/// slices by virtual concatenation. Assuming the individual segments' slice
/// lengths align properly with the needed and unneeded parts of the result, the
/// latter ones can get truncated away trivially when done by simply dismissing
/// the corresponding underlying buffers.
///
/// See also [`CompositeLimbsBuffer`] for a non-mutable variant.
pub struct CompositeLimbsBuffer<'a, ST: MpUIntCommon, const N_SEGMENTS: usize> {
    /// The composed view's individual segments, ordered from least to most
    /// significant.
    segments: [CompositeLimbsBufferSegment<'a, ST>; N_SEGMENTS],
}

impl<'a, ST: MpUIntSliceCommon, const N_SEGMENTS: usize> CompositeLimbsBuffer<'a, ST, N_SEGMENTS> {
    /// Construct a `CompositeLimbsBuffer` view from the individual byte buffer
    /// segments.
    ///
    /// # Arguments
    ///
    /// * `segments` - An array of `N_SEGMENTS` byte slices to compose the
    ///   multiprecision integer byte buffer view from by virtual concatenation.
    ///   Ordered from least to most significant relative with respect to their
    ///   position within the resulting view.
    pub fn new(segments: [ST; N_SEGMENTS]) -> Self {
        let mut segments = <[ST; N_SEGMENTS] as IntoIterator>::into_iter(segments);
        let mut segments: [Option<ST>; N_SEGMENTS] = core::array::from_fn(|_| segments.next());
        let mut n_bytes_total = 0;
        let mut create_segment = |i: usize| {
            let segment = segments[i].take().unwrap();
            let segment_len = segment.len();
            n_bytes_total += segment_len;

            let n_high_partial = n_bytes_total % LIMB_BYTES;
            let (high_next_partial, segment) = if i + 1 != segments.len() && n_high_partial != 0 {
                let next_segment = segments[i + 1].take().unwrap();
                let next_segment_len = next_segment.len();
                let n_from_next = (LIMB_BYTES - n_high_partial).min(next_segment_len);
                let (next_segment, high_next_partial) = next_segment.take(n_from_next);
                segments[i + 1] = Some(next_segment);
                (high_next_partial, segment)
            } else {
                let (high_next_partial, segment) = segment.take(segment_len);
                (high_next_partial, segment)
            };

            let high_next_partial_len = high_next_partial.len();
            n_bytes_total += high_next_partial_len;
            let end = ct_mp_nlimbs(n_bytes_total);
            CompositeLimbsBufferSegment {
                end,
                segment,
                high_next_partial,
                _phantom: marker::PhantomData,
            }
        };

        let segments: [CompositeLimbsBufferSegment<'a, ST>; N_SEGMENTS] =
            core::array::from_fn(&mut create_segment);
        Self { segments }
    }

    /// Lookup the least significant buffer segment holding the specified limb's
    /// least significant byte.
    ///
    /// Returns a pair of segment index and, as a by-product of the search, the
    /// corresponding segment's offset within the composed multiprecision
    /// integer byte buffer in terms of limbs.
    ///
    /// Execution time depends on the composed multiprecision integer's
    /// underlying segment layout as well as as on on the limb index
    /// argument `i` as far as branching is concerned.
    ///
    /// # Arguments
    ///
    /// * `i` - The index of the limb to load, counted from least to most
    ///   significant.
    fn limb_index_to_segment(&self, i: usize) -> (usize, usize) {
        let mut segment_offset = 0;
        for segment_index in 0..N_SEGMENTS {
            let segment_end = self.segments[segment_index].end;
            if i < segment_end {
                return (segment_index, segment_offset);
            } else {
                segment_offset = segment_end;
            }
        }
        unreachable!();
    }

    /// Load a limb from the composed multiprecision integer byte buffer.
    ///
    /// Execution time depends on the composed multiprecision integer's
    /// underlying segment layout as well as as on on the limb index
    /// argument `i`, and is otherwise constant as far as branching
    /// is concerned.
    ///
    /// * `i` - The index of the limb to load, counted from least to most
    ///   significant.
    pub fn load(&self, i: usize) -> LimbType {
        let (segment_index, segment_offset) = self.limb_index_to_segment(i);
        let segment = &self.segments[segment_index];
        let segment_slice = &segment.segment;
        if i != segment.end - 1 || !ST::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
            segment_slice.load_l_full(i - segment_offset)
        } else if segment_index + 1 == N_SEGMENTS || segment_slice.len() % LIMB_BYTES == 0 {
            // The last (highest) segment's most significant bytes don't necessarily occupy
            // a full limb.
            segment_slice.load_l(i - segment_offset)
        } else {
            let mut npartial = segment_slice.len() % LIMB_BYTES;
            let mut value = segment_slice.load_l(i - segment_offset);
            let mut segment_index = segment_index;
            while npartial != LIMB_BYTES && segment_index < self.segments.len() {
                let partial = &self.segments[segment_index].high_next_partial;
                if !partial.is_empty() {
                    value |= partial.load_l(0) << (8 * npartial);
                    npartial += partial.len();
                }
                segment_index += 1;
            }
            value
        }
    }
}

impl<'a, ST: MpMutUIntSlice, const N_SEGMENTS: usize> CompositeLimbsBuffer<'a, ST, N_SEGMENTS> {
    /// Update a limb in the composed multiprecision integer byte buffer.
    ///
    /// Execution time depends on the composed multiprecision integer's
    /// underlying segment layout as well as as on on the limb index
    /// argument `i`, and is otherwise constant as far as branching
    /// is concerned.
    ///
    /// * `i` - The index of the limb to update, counted from least to most
    ///   significant.
    /// * `value` - The value to store in the i'th limb.
    pub fn store(&mut self, i: usize, value: LimbType) {
        let (segment_index, segment_offset) = self.limb_index_to_segment(i);
        let segment = &mut self.segments[segment_index];
        let segment_slice = &mut segment.segment;
        if i != segment.end - 1 || !ST::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
            segment_slice.store_l_full(i - segment_offset, value);
        } else if segment_index + 1 == N_SEGMENTS || segment_slice.len() % LIMB_BYTES == 0 {
            // The last (highest) part's most significant bytes don't necessarily occupy a
            // full limb.
            segment_slice.store_l(i - segment_offset, value)
        } else {
            let mut value = value;
            let mut npartial = segment_slice.len() % LIMB_BYTES;
            let value_mask = ct_lsb_mask_l(8 * npartial as u32);
            segment_slice.store_l(i - segment_offset, value & value_mask);
            value >>= 8 * npartial;
            let mut segment_index = segment_index;
            while npartial != LIMB_BYTES && segment_index < self.segments.len() {
                let partial = &mut self.segments[segment_index].high_next_partial;
                if !partial.is_empty() {
                    let value_mask = ct_lsb_mask_l(8 * partial.len() as u32);
                    partial.store_l(0, value & value_mask);
                    value >>= 8 * partial.len();
                    npartial += partial.len();
                }
                segment_index += 1;
            }
            debug_assert!(value == 0);
        }
    }
}

#[test]
fn test_composite_limbs_buffer_load_be() {
    let mut buf0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let buf1: [u8; 0] = [0; 0];
    let mut buf2: [u8; 2 * LIMB_BYTES + 2] = [0; 2 * LIMB_BYTES + 2];

    // 0x02 00 .. 00 01 00 .. 00
    buf0[LIMB_BYTES - 1 + LIMB_BYTES / 2] = 0x1;
    buf0[LIMB_BYTES - 1] = 0x2;

    // 0x05 04 00 .. 00 03 .. 00
    buf0[LIMB_BYTES / 2 - 1] = 0x3;
    buf0[0] = 0x4;
    buf2[1 + 2 * LIMB_BYTES] = 0x5;

    // 0x07 00 .. 00 06 00 .. 00
    buf2[1 + LIMB_BYTES + LIMB_BYTES / 2] = 0x6;
    buf2[1 + LIMB_BYTES] = 0x7;

    // 0x09 00 .. 00 08 00 .. 00
    buf2[1 + LIMB_BYTES / 2] = 0x8;
    buf2[1] = 0x9;

    // 0x00 .. 00 0x0a
    buf2[0] = 0xa;

    let buf0 = MpBigEndianUIntByteSlice::from_bytes(buf0.as_slice());
    let buf1 = MpBigEndianUIntByteSlice::from_bytes(buf1.as_slice());
    let buf2 = MpBigEndianUIntByteSlice::from_bytes(buf2.as_slice());
    let limbs = CompositeLimbsBuffer::new([buf0, buf1, buf2]);

    let l0 = limbs.load(0);
    assert_eq!(l0, 0x2 << LIMB_BITS - 8 | 0x1 << LIMB_BITS / 2 - 8);
    let l1 = limbs.load(1);
    assert_eq!(l1, 0x0504 << LIMB_BITS - 16 | 0x3 << LIMB_BITS / 2 - 8);
    let l2 = limbs.load(2);
    assert_eq!(l2, 0x7 << LIMB_BITS - 8 | 0x6 << LIMB_BITS / 2 - 8);
    let l3 = limbs.load(3);
    assert_eq!(l3, 0x9 << LIMB_BITS - 8 | 0x8 << LIMB_BITS / 2 - 8);
    let l4 = limbs.load(4);
    assert_eq!(l4, 0xa);
}

#[test]
fn test_composite_limbs_buffer_load_le() {
    let mut buf0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let buf1: [u8; 0] = [0; 0];
    let mut buf2: [u8; 2 * LIMB_BYTES + 2] = [0; 2 * LIMB_BYTES + 2];

    // 0x00 .. 00 01 00 .. 00 02
    buf0[LIMB_BYTES / 2 - 1] = 0x1;
    buf0[LIMB_BYTES - 1] = 0x2;

    // 0x00 .. 03 00 .. 00 04 05
    buf0[LIMB_BYTES + LIMB_BYTES / 2 - 1] = 0x3;
    buf0[2 * LIMB_BYTES - 2] = 0x4;
    buf2[0] = 0x5;

    // 0x00 .. 00 06 00 .. 00 07
    buf2[1 + LIMB_BYTES / 2 - 1] = 0x6;
    buf2[1 + LIMB_BYTES - 1] = 0x7;

    // 0x00 .. 00 08 00 .. 00 09
    buf2[1 + LIMB_BYTES + LIMB_BYTES / 2 - 1] = 0x8;
    buf2[1 + 2 * LIMB_BYTES - 1] = 0x9;

    // 0x00 .. 00 0x0a
    buf2[1 + 2 * LIMB_BYTES] = 0xa;

    let buf0 = MpLittleEndianUIntByteSlice::from_bytes(buf0.as_slice());
    let buf1 = MpLittleEndianUIntByteSlice::from_bytes(buf1.as_slice());
    let buf2 = MpLittleEndianUIntByteSlice::from_bytes(buf2.as_slice());
    let limbs = CompositeLimbsBuffer::new([buf0, buf1, buf2]);

    let l0 = limbs.load(0);
    assert_eq!(l0, 0x2 << LIMB_BITS - 8 | 0x1 << LIMB_BITS / 2 - 8);
    let l1 = limbs.load(1);
    assert_eq!(l1, 0x0504 << LIMB_BITS - 16 | 0x3 << LIMB_BITS / 2 - 8);
    let l2 = limbs.load(2);
    assert_eq!(l2, 0x7 << LIMB_BITS - 8 | 0x6 << LIMB_BITS / 2 - 8);
    let l3 = limbs.load(3);
    assert_eq!(l3, 0x9 << LIMB_BITS - 8 | 0x8 << LIMB_BITS / 2 - 8);
    let l4 = limbs.load(4);
    assert_eq!(l4, 0xa);
}

#[test]
fn test_composite_limbs_buffer_load_ne() {
    let mut buf0 = [0 as LimbType; 2];
    let buf1 = [0 as LimbType; 0];
    let mut buf2 = [0 as LimbType; 2 * LIMB_BYTES];

    buf0[0] = 0x1;
    buf0[1] = 0x2;
    buf2[0] = 0x3;
    buf2[1] = 0x4;

    let buf0 = MpNativeEndianUIntLimbsSlice::from_limbs(buf0.as_slice());
    let buf1 = MpNativeEndianUIntLimbsSlice::from_limbs(buf1.as_slice());
    let buf2 = MpNativeEndianUIntLimbsSlice::from_limbs(buf2.as_slice());
    let limbs = CompositeLimbsBuffer::new([buf0, buf1, buf2]);

    let l0 = limbs.load(0);
    assert_eq!(l0, 0x1);
    let l1 = limbs.load(1);
    assert_eq!(l1, 0x2);
    let l2 = limbs.load(2);
    assert_eq!(l2, 0x3);
    let l3 = limbs.load(3);
    assert_eq!(l3, 0x4);
}

#[cfg(test)]
fn test_composite_limbs_buffer_store_with_unaligned_lengths<
    ST: MpMutUIntSlice<BackingSliceElementType = u8>,
>() {
    debug_assert_eq!(ST::SUPPORTS_UNALIGNED_BUFFER_LENGTHS, true);

    let mut buf0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let mut buf1: [u8; 0] = [0; 0];
    let mut buf2: [u8; 2 * LIMB_BYTES + 2] = [0; 2 * LIMB_BYTES + 2];
    let buf0 = ST::from_slice(&mut buf0).unwrap();
    let buf1 = ST::from_slice(&mut buf1).unwrap();
    let buf2 = ST::from_slice(&mut buf2).unwrap();
    let mut limbs = CompositeLimbsBuffer::new([buf0, buf1, buf2]);

    let l0 = 0x2 << LIMB_BITS - 8 | 0x1 << LIMB_BITS / 2 - 8;
    let l1 = 0x0504 << LIMB_BITS - 16 | 0x3 << LIMB_BITS / 2 - 8;
    let l2 = 0x7 << LIMB_BITS - 8 | 0x6 << LIMB_BITS / 2 - 8;
    let l3 = 0x9 << LIMB_BITS - 8 | 0x8 << LIMB_BITS / 2 - 8;
    let l4 = 0xa;

    limbs.store(0, l0);
    limbs.store(1, l1);
    limbs.store(2, l2);
    limbs.store(3, l3);
    limbs.store(4, l4);
    assert_eq!(l0, limbs.load(0));
    assert_eq!(l1, limbs.load(1));
    assert_eq!(l2, limbs.load(2));
    assert_eq!(l3, limbs.load(3));
    assert_eq!(l4, limbs.load(4));
}

#[cfg(test)]
fn test_composite_limbs_buffer_store_with_aligned_lengths<ST: MpMutUIntSlice>() {
    let mut buf0 = tst_mk_mp_backing_vec!(ST, 2 * LIMB_BYTES);
    let mut buf1 = tst_mk_mp_backing_vec!(ST, 0);
    let mut buf2 = tst_mk_mp_backing_vec!(ST, 2 * LIMB_BYTES);
    let buf0 = ST::from_slice(&mut buf0).unwrap();
    let buf1 = ST::from_slice(&mut buf1).unwrap();
    let buf2 = ST::from_slice(&mut buf2).unwrap();
    let mut limbs = CompositeLimbsBuffer::new([buf0, buf1, buf2]);

    let l0 = 0x2 << LIMB_BITS - 8 | 0x1 << LIMB_BITS / 2 - 8;
    let l1 = 0x0504 << LIMB_BITS - 16 | 0x3 << LIMB_BITS / 2 - 8;
    let l2 = 0x7 << LIMB_BITS - 8 | 0x6 << LIMB_BITS / 2 - 8;
    let l3 = 0x9 << LIMB_BITS - 8 | 0x8 << LIMB_BITS / 2 - 8;

    limbs.store(0, l0);
    limbs.store(1, l1);
    limbs.store(2, l2);
    limbs.store(3, l3);
    assert_eq!(l0, limbs.load(0));
    assert_eq!(l1, limbs.load(1));
    assert_eq!(l2, limbs.load(2));
    assert_eq!(l3, limbs.load(3));
}

#[test]
fn test_composite_limbs_buffer_store_be() {
    test_composite_limbs_buffer_store_with_unaligned_lengths::<MpMutBigEndianUIntByteSlice>();
    test_composite_limbs_buffer_store_with_aligned_lengths::<MpMutBigEndianUIntByteSlice>();
}

#[test]
fn test_composite_limbs_buffer_store_le() {
    test_composite_limbs_buffer_store_with_unaligned_lengths::<MpMutLittleEndianUIntByteSlice>();
    test_composite_limbs_buffer_store_with_aligned_lengths::<MpMutLittleEndianUIntByteSlice>();
}

#[test]
fn test_composite_limbs_buffer_store_ne() {
    test_composite_limbs_buffer_store_with_aligned_lengths::<MpMutNativeEndianUIntLimbsSlice>();
}

pub fn limb_slice_as_bytes_mut(limbs: &mut [LimbType]) -> &mut [u8] {
    let len = mem::size_of_val(limbs);
    let ptr = limbs.as_mut_ptr() as *mut u8;
    unsafe { slice::from_raw_parts_mut(ptr, len) }
}

pub fn limb_slice_as_bytes(limbs: &[LimbType]) -> &[u8] {
    let len = mem::size_of_val(limbs);
    let ptr = limbs.as_ptr() as *const u8;
    unsafe { slice::from_raw_parts(ptr, len) }
}
