//! Accessors for multiprecision integers as stored in byte buffers of certain endian layouts.
//!
//! Multiprecision integers are stored in bytes buffers in big-, little- or internal,
//! "native"-endian order. The arithmetic primitives all operate on those in units of [`LimbType`]
//! for efficiency reasons.  This helper module provides a couple of utilities for accessing these
//! byte buffers in units of [`LimbType`].

use core::{self, convert, fmt, marker};

use super::limb::{LimbType, LIMB_BYTES, LIMB_BITS, ct_find_last_set_byte_l, ct_lsb_mask_l, LimbChoice, ct_find_first_set_bit_l, ct_find_last_set_bit_l, ct_is_zero_l};
use super::usize_ct_cmp::ct_eq_usize_usize;

/// Determine the number of [`LimbType`] limbs stored in a multiprecision integer big-endian byte
/// buffer.
///
/// # Arguments
///
/// * `len` - The multiprecision integer's underlying big-endian byte buffer's length in bytes.
///
pub const fn mp_ct_nlimbs(len: usize) -> usize {
    (len + LIMB_BYTES - 1) / LIMB_BYTES
}

pub const fn mp_ct_limbs_align_len(len: usize) -> usize {
    mp_ct_nlimbs(len) * LIMB_BYTES
}

#[test]
fn test_mp_ct_nlimbs() {
    assert_eq!(mp_ct_nlimbs(0), 0);
    assert_eq!(mp_ct_nlimbs(1), 1);
    assert_eq!(mp_ct_nlimbs(LIMB_BYTES - 1), 1);
    assert_eq!(mp_ct_nlimbs(LIMB_BYTES), 1);
    assert_eq!(mp_ct_nlimbs(LIMB_BYTES + 1), 2);
    assert_eq!(mp_ct_nlimbs(2 * LIMB_BYTES - 1), 2);
}


/// Internal helper to load some fully contained limb from a multiprecision integer big-endian byte
/// buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. This internal helper is used whenever the
/// specified limb is not such a partially covered high limb.
///
/// Runs in constant time as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in big-endian order.
/// * `src_end` - The position past the end of the limb to be loaded from `limbs`.
///               Must be `>= core::mem::size_of::<LimbType>()`.
///
fn _mp_be_load_l_full(limbs: &[u8], src_end: usize) -> LimbType {
    let src_begin = src_end - LIMB_BYTES;
    let src = &limbs[src_begin..src_end];
    let src = <[u8; LIMB_BYTES] as TryFrom<&[u8]>>::try_from(src).unwrap();
    LimbType::from_be_bytes(src)
}

/// Internal helper to load some partially contained limb from a multiprecision integer big-endian
/// byte buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. This internal helper is used whenever the
/// specified limb is such a partially covered high limb.
///
/// Execution time depends only on the `src_end` argument and is otherwise constant as far as
/// branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in big-endian order.
/// * `src_end` - The position past the end of the limb to be loaded from `limbs`.
///               Must be `< core::mem::size_of::<LimbType>()`.
///
fn _mp_be_load_l_high_partial(limbs: &[u8], src_end: usize) -> LimbType {
    let mut src: [u8; LIMB_BYTES] = [0; LIMB_BYTES];
    src[LIMB_BYTES - src_end..LIMB_BYTES].copy_from_slice(&limbs[0..src_end]);
    let l = LimbType::from_be_bytes(src);
    l
}

/// Load some fully contained limb from a multiprecision integer big-endian byte buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. The generic [`mp_be_load_l()`] implements some extra logic for
/// handling this special case of a partially stored high limb, which can be avoided if it is known
/// that the limb to be accessed is completely covered by the `limbs` buffer. This function here
/// provides such a less generic, but more performant variant.
///
/// Runs in constant time as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in big-endian order.
/// * `i` - The index of the limb to load, counted from least to most significant.
///
fn mp_be_load_l_full(limbs: &[u8], i: usize) -> LimbType {
    debug_assert!(i * LIMB_BYTES < limbs.len());
    let src_end = limbs.len() - i * LIMB_BYTES;
    _mp_be_load_l_full(limbs, src_end)
}

/// Load some limb from a multiprecision integer big-endian byte buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. If it is known that the limb to be accessed at position `i` is
/// fully covered by the `limbs` buffer, consider using [`mp_be_load_l_full()`] for improved
/// performance instead.
///
/// Execution time depends only on the multiprecision integer's length and the limb index argument
/// `i`, and is otherwise constant as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in big-endian order.
/// * `i` - The index of the limb to load, counted from least to most significant.
///
fn mp_be_load_l(limbs: &[u8], i: usize) -> LimbType {
    debug_assert!(i * LIMB_BYTES <= limbs.len());
    let src_end = limbs.len() - i * LIMB_BYTES;
    if src_end >= LIMB_BYTES {
        _mp_be_load_l_full(limbs, src_end)
    } else {
        _mp_be_load_l_high_partial(limbs, src_end)
    }
}

#[test]
fn test_mp_be_load_l() {
    use super::limb::LIMB_BITS;

    let limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    assert_eq!(mp_be_load_l(&limbs, 0), 0);
    assert_eq!(mp_be_load_l(&limbs, 1), 0);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    limbs[LIMB_BYTES] = 0x80;
    limbs[LIMB_BYTES - 1] = 1;
    assert_eq!(mp_be_load_l(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(mp_be_load_l_full(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(mp_be_load_l(&limbs, 1), 1);
    assert_eq!(mp_be_load_l_full(&limbs, 1), 1);


    let limbs: [u8; 1] = [0; 1];
    assert_eq!(mp_be_load_l(&limbs, 0), 0);

    let limbs: [u8; LIMB_BYTES + 1] = [0; LIMB_BYTES + 1];
    assert_eq!(mp_be_load_l(&limbs, 0), 0);
    assert_eq!(mp_be_load_l_full(&limbs, 0), 0);
    assert_eq!(mp_be_load_l(&limbs, 1), 0);


    let limbs: [u8; 2] = [0, 1];
    assert_eq!(mp_be_load_l(&limbs, 0), 1);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    limbs[1] = 1;
    assert_eq!(mp_be_load_l(&limbs, 0), 0);
    assert_eq!(mp_be_load_l_full(&limbs, 0), 0);
    assert_eq!(mp_be_load_l(&limbs, 1), 1);


    let limbs: [u8; 2] = [1, 0];
    assert_eq!(mp_be_load_l(&limbs, 0), 0x0100);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    limbs[0] = 1;
    assert_eq!(mp_be_load_l(&limbs, 0), 0);
    assert_eq!(mp_be_load_l_full(&limbs, 0), 0);
    assert_eq!(mp_be_load_l(&limbs, 1), 0x0100);
}

/// Internal helper to update some fully contained limb in a multiprecision integer big-endian
/// byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. This internal helper is used whenever the
/// specified limb is not such a partially covered high limb.
///
/// Runs in constant time as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in big-endian order.
/// * `dst_end` - The position past the end of the limb to be updated in `limbs`.
///               Must be `>= core::mem::size_of::<LimbType>()`.
/// * `value` - The value to store in `limbs` at the specified position.
///
fn _mp_be_store_l_full(limbs: &mut [u8], dst_end: usize, value: LimbType) {
    let dst_begin = dst_end - LIMB_BYTES;
    let dst = &mut limbs[dst_begin..dst_end];
    let dst = <&mut [u8; LIMB_BYTES] as TryFrom<&mut [u8]>>::try_from(dst).unwrap();
    *dst = value.to_be_bytes();
}

/// Internal helper to update some partially contained limb in a multiprecision integer big-endian
/// byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. This internal helper is used whenever the
/// specified limb is not such a partially covered high limb.
///
/// Execution time depends only on the `dst_end` argument and is otherwise constant as far as
/// branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in big-endian order.
/// * `dst_end` - The position past the end of the limb to be updated in `limbs`.
///               Must be `< core::mem::size_of::<LimbType>()`.
/// * `value` - The value to store in `limbs` at the specified position. The high
///             bits corresponding to any of the excess limb bytes not covered by
///             `limbs` can have arbitrary values, but are ignored.
///
fn _mp_be_store_l_high_partial(limbs: &mut [u8], dst_end: usize, value: LimbType) {
    let dst = &mut limbs[0..dst_end];
    let src: [u8; LIMB_BYTES] = value.to_be_bytes();
    dst.copy_from_slice(&src[LIMB_BYTES - dst_end..LIMB_BYTES]);
}

/// Update some fully contained limb in a multiprecision integer big-endian byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. The generic [`mp_be_store_l()`] implements some extra logic for
/// handling this special case of a partially stored high limb, which can be avoided if it is known
/// that the limb to be accessed is completely covered by the `limbs` buffer. This function here
/// provides such a less generic, but more performant variant.
///
/// Runs in constant time.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in big-endian order.
/// * `i` - The index of the limb to update, counted from least to most significant.
/// * `value` - The value to store at limb index `i` in `limbs`.
///
fn mp_be_store_l_full(limbs: &mut [u8], i: usize, value: LimbType) {
    debug_assert!(i * LIMB_BYTES < limbs.len());
    let dst_end = limbs.len() - i * LIMB_BYTES;
    _mp_be_store_l_full(limbs, dst_end, value);
}

/// Update some limb in a multiprecision integer big-endian byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. If the most significant limb is to be stored, all bytes in `value`
/// corresponding to these excess bytes must be zero accordingly.

/// If it is known that the limb to be stored at position `i` is fully covered by the `limbs`
/// buffer, consider using [`mp_be_store_l_full()`] for improved performance instead.
///
/// Execution time depends only on the multiprecision integer's length and the limb index argument
/// `i`, and is otherwise constant as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in big-endian order.
/// * `i` - The index of the limb to update, counted from least to most significant.
/// * `value` - The value to store at limb index `i` in `limbs`.
///
fn mp_be_store_l(limbs: &mut [u8], i: usize, value: LimbType) {
    debug_assert!(i * LIMB_BYTES <= limbs.len());
    let dst_end = limbs.len() - i * LIMB_BYTES;
    if dst_end >= LIMB_BYTES {
        _mp_be_store_l_full(limbs, dst_end, value);
    } else {
        debug_assert_eq!(value >> 8 * dst_end, 0);
        _mp_be_store_l_high_partial(limbs, dst_end, value);
    }
}

#[test]
fn test_mp_be_store_l() {
    use super::limb::LIMB_BITS;

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_be_store_l(&mut limbs, 0, 1 << (LIMB_BITS - 1));
    mp_be_store_l(&mut limbs, 1, 1);
    assert_eq!(mp_be_load_l(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(mp_be_load_l(&limbs, 1), 1);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_be_store_l_full(&mut limbs, 0, 1 << (LIMB_BITS - 1));
    mp_be_store_l_full(&mut limbs, 1, 1);
    assert_eq!(mp_be_load_l(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(mp_be_load_l(&limbs, 1), 1);

    let mut limbs: [u8; 1] = [0; 1];
    mp_be_store_l(&mut limbs, 0, 1);
    assert_eq!(mp_be_load_l(&limbs, 0), 1);

    let mut limbs: [u8; LIMB_BYTES - 1] = [0; LIMB_BYTES - 1];
    mp_be_store_l(&mut limbs, 0, 1);
    assert_eq!(mp_be_load_l(&limbs, 0), 1);

    let mut limbs: [u8; LIMB_BYTES - 1] = [0; LIMB_BYTES - 1];
    mp_be_store_l(&mut limbs, 0, 1 << LIMB_BITS - 8 - 1);
    assert_eq!(mp_be_load_l(&limbs, 0), 1 << LIMB_BITS - 8 - 1);

    let mut limbs: [u8; 2] = [0; 2];
    mp_be_store_l(&mut limbs, 0, 1);
    assert_eq!(mp_be_load_l(&limbs, 0), 1);

    let mut limbs: [u8; 2] = [0; 2];
    mp_be_store_l(&mut limbs, 0, 0x0100);
    assert_eq!(mp_be_load_l(&limbs, 0), 0x0100);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    mp_be_store_l(&mut limbs, 1, 1);
    assert_eq!(mp_be_load_l(&limbs, 0), 0);
    assert_eq!(mp_be_load_l(&limbs, 1), 1);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    mp_be_store_l(&mut limbs, 1, 0x0100);
    assert_eq!(mp_be_load_l(&limbs, 0), 0);
    assert_eq!(mp_be_load_l(&limbs, 1), 0x0100);
}

fn mp_be_zeroize_bytes_above(limbs: &mut [u8], begin: usize) {
    let limbs_len = limbs.len();
    if limbs_len <= begin {
        return;
    }
    limbs[..limbs_len - begin].fill(0);
}

#[test]
fn test_mp_be_zeroize_bytes_above() {
    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    mp_be_store_l(&mut limbs, 0, !0);
    mp_be_store_l(&mut limbs, 1, !0 >> 8);
    mp_be_zeroize_bytes_above(&mut limbs, LIMB_BYTES + 1);
    assert_eq!(mp_be_load_l(&mut limbs, 0), !0);
    assert_eq!(mp_be_load_l(&mut limbs, 1), !0 & 0xff);

    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    mp_be_store_l(&mut limbs, 0, !0);
    mp_be_store_l(&mut limbs, 1, !0 >> 8);
    mp_be_zeroize_bytes_above(&mut limbs, LIMB_BYTES - 1);
    assert_eq!(mp_be_load_l(&mut limbs, 0), !0 >> 8);
    assert_eq!(mp_be_load_l(&mut limbs, 1), 0);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_be_store_l(&mut limbs, 0, !0);
    mp_be_store_l(&mut limbs, 1, !0);
    mp_be_zeroize_bytes_above(&mut limbs, 2 * LIMB_BYTES);
    assert_eq!(mp_be_load_l(&mut limbs, 0), !0);
    assert_eq!(mp_be_load_l(&mut limbs, 1), !0);
}

fn mp_be_zeroize_bytes_below(limbs: &mut [u8], end: usize) {
    let limbs_len = limbs.len();
    let end = end.min(limbs_len);
    limbs[limbs_len - end..].fill(0);
}

#[test]
fn test_mp_be_zeroize_bytes_below() {
    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    mp_be_store_l(&mut limbs, 0, !0);
    mp_be_store_l(&mut limbs, 1, !0 >> 8);
    mp_be_zeroize_bytes_below(&mut limbs, LIMB_BYTES + 1);
    assert_eq!(mp_be_load_l(&mut limbs, 0), 0);
    assert_eq!(mp_be_load_l(&mut limbs, 1), (!0 >> 8) & !0xff);

    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    mp_be_store_l(&mut limbs, 0, !0);
    mp_be_store_l(&mut limbs, 1, !0 >> 8);
    mp_be_zeroize_bytes_below(&mut limbs, LIMB_BYTES - 1);
    assert_eq!(mp_be_load_l(&mut limbs, 0), 0xff << 8 * (LIMB_BYTES - 1));
    assert_eq!(mp_be_load_l(&mut limbs, 1), !0 >> 8);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_be_store_l(&mut limbs, 0, !0);
    mp_be_store_l(&mut limbs, 1, !0);
    mp_be_zeroize_bytes_below(&mut limbs, 0);
    assert_eq!(mp_be_load_l(&mut limbs, 0), !0);
    assert_eq!(mp_be_load_l(&mut limbs, 1), !0);
}

/// Internal helper to load some fully contained limb from a multiprecision integer little-endian byte
/// buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. This internal helper is used whenever the
/// specified limb is not such a partially covered high limb.
///
/// Runs in constant time as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in little-endian order.
/// * `src_begin` - The position of the limb to be loaded from `limbs`.
///
fn _mp_le_load_l_full(limbs: &[u8], src_begin: usize) -> LimbType {
    let src_end = src_begin + LIMB_BYTES;
    let src = &limbs[src_begin..src_end];
    let src = <[u8; LIMB_BYTES] as TryFrom<&[u8]>>::try_from(src).unwrap();
    LimbType::from_le_bytes(src)
}

/// Internal helper to load some partially contained limb from a multiprecision integer little-endian
/// byte buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. This internal helper is used whenever the
/// specified limb is such a partially covered high limb.
///
/// Execution time depends only on the `src_end` argument and is otherwise constant as far as
/// branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in little-endian order.
/// * `src_begin` - The position of the limb to be loaded from `limbs`.
///
fn _mp_le_load_l_high_partial(limbs: &[u8], src_begin: usize) -> LimbType {
    let mut src: [u8; LIMB_BYTES] = [0; LIMB_BYTES];
    src[..limbs.len() - src_begin].copy_from_slice(&limbs[src_begin..]);
    let l = LimbType::from_le_bytes(src);
    l
}

/// Load some fully contained limb from a multiprecision integer little-endian byte buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. The generic [`mp_le_load_l()`] implements some extra logic for
/// handling this special case of a partially stored high limb, which can be avoided if it is known
/// that the limb to be accessed is completely covered by the `limbs` buffer. This function here
/// provides such a less generic, but more performant variant.
///
/// Runs in constant time as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in little-endian order.
/// * `i` - The index of the limb to load, counted from least to most significant.
///
fn mp_le_load_l_full(limbs: &[u8], i: usize) -> LimbType {
    let src_begin = i * LIMB_BYTES;
    debug_assert!(src_begin < limbs.len());
    _mp_le_load_l_full(limbs, src_begin)
}

/// Load some limb from a multiprecision integer little-endian byte buffer.
///
/// The specified limb is returned in the host's native endianess.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. If it is known that the limb to be accessed at position `i` is
/// fully covered by the `limbs` buffer, consider using [`mp_le_load_l_full()`] for improved
/// performance instead.
///
/// Execution time depends only on the multiprecision integer's length and the limb index argument
/// `i`, and is otherwise constant as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in little-endian order.
/// * `i` - The index of the limb to load, counted from least to most significant.
///
fn mp_le_load_l(limbs: &[u8], i: usize) -> LimbType {
    let src_begin = i * LIMB_BYTES;
    debug_assert!(src_begin < limbs.len());
    if src_begin + LIMB_BYTES <= limbs.len() {
        _mp_le_load_l_full(limbs, src_begin)
    } else {
        _mp_le_load_l_high_partial(limbs, src_begin)
    }
}

#[test]
fn test_mp_le_load_l() {
    use super::limb::LIMB_BITS;

    let limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    assert_eq!(mp_le_load_l(&limbs, 0), 0);
    assert_eq!(mp_le_load_l(&limbs, 1), 0);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    limbs[LIMB_BYTES - 1] = 0x80;
    limbs[LIMB_BYTES] = 1;
    assert_eq!(mp_le_load_l(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(mp_le_load_l_full(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(mp_le_load_l(&limbs, 1), 1);
    assert_eq!(mp_le_load_l_full(&limbs, 1), 1);


    let limbs: [u8; 1] = [0; 1];
    assert_eq!(mp_le_load_l(&limbs, 0), 0);

    let limbs: [u8; LIMB_BYTES + 1] = [0; LIMB_BYTES + 1];
    assert_eq!(mp_le_load_l(&limbs, 0), 0);
    assert_eq!(mp_le_load_l_full(&limbs, 0), 0);
    assert_eq!(mp_le_load_l(&limbs, 1), 0);


    let limbs: [u8; 2] = [1, 0];
    assert_eq!(mp_le_load_l(&limbs, 0), 1);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    limbs[LIMB_BYTES] = 1;
    assert_eq!(mp_le_load_l(&limbs, 0), 0);
    assert_eq!(mp_le_load_l_full(&limbs, 0), 0);
    assert_eq!(mp_le_load_l(&limbs, 1), 1);


    let limbs: [u8; 2] = [0, 1];
    assert_eq!(mp_le_load_l(&limbs, 0), 0x0100);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    limbs[LIMB_BYTES + 1] = 1;
    assert_eq!(mp_le_load_l(&limbs, 0), 0);
    assert_eq!(mp_le_load_l_full(&limbs, 0), 0);
    assert_eq!(mp_le_load_l(&limbs, 1), 0x0100);
}

/// Internal helper to update some fully contained limb in a multiprecision integer little-endian
/// byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. This internal helper is used whenever the
/// specified limb is not such a partially covered high limb.
///
/// Runs in constant time as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in little-endian order.
/// * `dst_begin` - The position of the limb to be updated in `limbs`.
/// * `value` - The value to store in `limbs` at the specified position.
///
fn _mp_le_store_l_full(limbs: &mut [u8], dst_begin: usize, value: LimbType) {
    let dst_end = dst_begin + LIMB_BYTES;
    let dst = &mut limbs[dst_begin..dst_end];
    let dst = <&mut [u8; LIMB_BYTES] as TryFrom<&mut [u8]>>::try_from(dst).unwrap();
    *dst = value.to_le_bytes();
}

/// Internal helper to update some partially contained limb in a multiprecision integer little-endian
/// byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. This internal helper is used whenever the
/// specified limb is not such a partially covered high limb.
///
/// Execution time depends only on the `dst_end` argument and is otherwise constant as far as
/// branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in little-endian order.
/// * `dst_begin` - The position of the limb to be updated in `limbs`.
/// * `value` - The value to store in `limbs` at the specified position. The high
///             bits corresponding to any of the excess limb bytes not covered by
///             `limbs` can have arbitrary values, but are ignored.
///
fn _mp_le_store_l_high_partial(limbs: &mut [u8], dst_begin: usize, value: LimbType) {
    let dst_end = limbs.len();
    let dst = &mut limbs[dst_begin..];
    let src: [u8; LIMB_BYTES] = value.to_le_bytes();
    dst.copy_from_slice(&src[0..dst_end - dst_begin]);
}

/// Update some fully contained limb in a multiprecision integer little-endian byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. The generic [`mp_le_store_l()`] implements some extra logic for
/// handling this special case of a partially stored high limb, which can be avoided if it is known
/// that the limb to be accessed is completely covered by the `limbs` buffer. This function here
/// provides such a less generic, but more performant variant.
///
/// Runs in constant time.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in little-endian order.
/// * `i` - The index of the limb to update, counted from least to most significant.
/// * `value` - The value to store at limb index `i` in `limbs`.
///
fn mp_le_store_l_full(limbs: &mut [u8], i: usize, value: LimbType) {
    let dst_begin = i * LIMB_BYTES;
    debug_assert!(dst_begin < limbs.len());
    _mp_le_store_l_full(limbs, dst_begin, value);
}

/// Update some limb in a multiprecision integer little-endian byte buffer.
///
/// The byte buffer's length doesn't necessarily align with the size of a [`LimbType`], in which
/// case the most significant limb's might be stored only partially, with its virtual excess high
/// bytes defined to equal zero. If the most significant limb is to be stored, all bytes in `value`
/// corresponding to these excess bytes must be zero accordingly.

/// If it is known that the limb to be stored at position `i` is fully covered by the `limbs`
/// buffer, consider using [`mp_le_store_l_full()`] for improved performance instead.
///
/// Execution time depends only on the multiprecision integer's length and the limb index argument
/// `i`, and is otherwise constant as far as branching is concerned.
///
/// # Arguments
///
/// * `limbs` - The multiprecision integer's underlying byte buffer in little-endian order.
/// * `i` - The index of the limb to update, counted from least to most significant.
/// * `value` - The value to store at limb index `i` in `limbs`.
///
fn mp_le_store_l(limbs: &mut [u8], i: usize, value: LimbType) {
    let dst_begin = i * LIMB_BYTES;
    debug_assert!(dst_begin < limbs.len());
    if dst_begin + LIMB_BYTES <= limbs.len() {
        _mp_le_store_l_full(limbs, dst_begin, value);
    } else {
        debug_assert_eq!(value >> 8 * (limbs.len() - dst_begin), 0);
        _mp_le_store_l_high_partial(limbs, dst_begin, value);
    }
}

#[test]
fn test_mp_le_store_l() {
    use super::limb::LIMB_BITS;

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_le_store_l(&mut limbs, 0, 1 << (LIMB_BITS - 1));
    mp_le_store_l(&mut limbs, 1, 1);
    assert_eq!(mp_le_load_l(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(mp_le_load_l(&limbs, 1), 1);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_le_store_l_full(&mut limbs, 0, 1 << (LIMB_BITS - 1));
    mp_le_store_l_full(&mut limbs, 1, 1);
    assert_eq!(mp_le_load_l(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(mp_le_load_l(&limbs, 1), 1);

    let mut limbs: [u8; 1] = [0; 1];
    mp_le_store_l(&mut limbs, 0, 1);
    assert_eq!(mp_le_load_l(&limbs, 0), 1);

    let mut limbs: [u8; LIMB_BYTES - 1] = [0; LIMB_BYTES - 1];
    mp_le_store_l(&mut limbs, 0, 1);
    assert_eq!(mp_le_load_l(&limbs, 0), 1);

    let mut limbs: [u8; LIMB_BYTES - 1] = [0; LIMB_BYTES - 1];
    mp_le_store_l(&mut limbs, 0, 1 << LIMB_BITS - 8 - 1);
    assert_eq!(mp_le_load_l(&limbs, 0), 1 << LIMB_BITS - 8 - 1);

    let mut limbs: [u8; 2] = [0; 2];
    mp_le_store_l(&mut limbs, 0, 1);
    assert_eq!(mp_le_load_l(&limbs, 0), 1);

    let mut limbs: [u8; 2] = [0; 2];
    mp_le_store_l(&mut limbs, 0, 0x0100);
    assert_eq!(mp_le_load_l(&limbs, 0), 0x0100);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    mp_le_store_l(&mut limbs, 1, 1);
    assert_eq!(mp_le_load_l(&limbs, 0), 0);
    assert_eq!(mp_le_load_l(&limbs, 1), 1);

    let mut limbs: [u8; LIMB_BYTES + 2] = [0; LIMB_BYTES + 2];
    mp_le_store_l(&mut limbs, 1, 0x0100);
    assert_eq!(mp_le_load_l(&limbs, 0), 0);
    assert_eq!(mp_le_load_l(&limbs, 1), 0x0100);
}

fn mp_le_zeroize_bytes_above(limbs: &mut [u8], begin: usize) {
    if limbs.len() <= begin {
        return;
    }
    limbs[begin..].fill(0);
}

#[test]
fn test_mp_le_zeroize_bytes_above() {
    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    mp_le_store_l(&mut limbs, 0, !0);
    mp_le_store_l(&mut limbs, 1, !0 >> 8);
    mp_le_zeroize_bytes_above(&mut limbs, LIMB_BYTES + 1);
    assert_eq!(mp_le_load_l(&mut limbs, 0), !0);
    assert_eq!(mp_le_load_l(&mut limbs, 1), !0 & 0xff);

    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    mp_le_store_l(&mut limbs, 0, !0);
    mp_le_store_l(&mut limbs, 1, !0 >> 8);
    mp_le_zeroize_bytes_above(&mut limbs, LIMB_BYTES - 1);
    assert_eq!(mp_le_load_l(&mut limbs, 0), !0 >> 8);
    assert_eq!(mp_le_load_l(&mut limbs, 1), 0);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_le_store_l(&mut limbs, 0, !0);
    mp_le_store_l(&mut limbs, 1, !0);
    mp_le_zeroize_bytes_above(&mut limbs, 2 * LIMB_BYTES);
    assert_eq!(mp_le_load_l(&mut limbs, 0), !0);
    assert_eq!(mp_le_load_l(&mut limbs, 1), !0);
}

fn mp_le_zeroize_bytes_below(limbs: &mut [u8], end: usize) {
    let end = end.min(limbs.len());
    limbs[..end].fill(0);
}

#[test]
fn test_mp_le_zeroize_bytes_below() {
    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    mp_le_store_l(&mut limbs, 0, !0);
    mp_le_store_l(&mut limbs, 1, !0 >> 8);
    mp_le_zeroize_bytes_below(&mut limbs, LIMB_BYTES + 1);
    assert_eq!(mp_le_load_l(&mut limbs, 0), 0);
    assert_eq!(mp_le_load_l(&mut limbs, 1), (!0 >> 8) & !0xff);

    let mut limbs: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    mp_le_store_l(&mut limbs, 0, !0);
    mp_le_store_l(&mut limbs, 1, !0 >> 8);
    mp_le_zeroize_bytes_below(&mut limbs, LIMB_BYTES - 1);
    assert_eq!(mp_le_load_l(&mut limbs, 0), 0xff << 8 * (LIMB_BYTES - 1));
    assert_eq!(mp_le_load_l(&mut limbs, 1), !0 >> 8);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_le_store_l(&mut limbs, 0, !0);
    mp_le_store_l(&mut limbs, 1, !0);
    mp_le_zeroize_bytes_below(&mut limbs, 0);
    assert_eq!(mp_le_load_l(&mut limbs, 0), !0);
    assert_eq!(mp_le_load_l(&mut limbs, 1), !0);
}

fn _mp_ne_load_l_full(limbs: &[u8], src_begin: usize) -> LimbType {
    let src_end = src_begin + LIMB_BYTES;
    let src = &limbs[src_begin..src_end];
    let src = <[u8; LIMB_BYTES] as TryFrom<&[u8]>>::try_from(src).unwrap();
    LimbType::from_ne_bytes(src)
}

fn mp_ne_load_l_full(limbs: &[u8], i: usize) -> LimbType {
    let src_begin = i * LIMB_BYTES;
    debug_assert!(src_begin < limbs.len());
    _mp_ne_load_l_full(limbs, src_begin)
}

fn mp_ne_load_l(limbs: &[u8], i: usize) -> LimbType {
    mp_ne_load_l_full(limbs, i)
}

#[cfg(test)]
const fn test_ne_is_le() -> bool {
    let mut bytes: [u8; LIMB_BYTES] = [0; LIMB_BYTES];
    bytes[0] = 1;
    LimbType::from_ne_bytes(bytes) == 1
}

#[test]
fn test_mp_ne_load_l() {
    let limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    assert_eq!(mp_ne_load_l(&limbs, 0), 0);
    assert_eq!(mp_ne_load_l(&limbs, 1), 0);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    if test_ne_is_le() {
        limbs[0] = 1;
        limbs[LIMB_BYTES] = 2;
    } else {
        limbs[LIMB_BYTES - 1] = 1;
        limbs[2 * LIMB_BYTES - 1] = 2;
    }
    assert_eq!(mp_ne_load_l(&limbs, 0), 1);
    assert_eq!(mp_ne_load_l(&limbs, 1), 2);
}

fn _mp_ne_store_l_full(limbs: &mut [u8], dst_begin: usize, value: LimbType) {
    let dst_end = dst_begin + LIMB_BYTES;
    let dst = &mut limbs[dst_begin..dst_end];
    let dst = <&mut [u8; LIMB_BYTES] as TryFrom<&mut [u8]>>::try_from(dst).unwrap();
    *dst = value.to_ne_bytes();
}

fn mp_ne_store_l_full(limbs: &mut [u8], i: usize, value: LimbType) {
    let dst_begin = i * LIMB_BYTES;
    debug_assert!(dst_begin < limbs.len());
    _mp_ne_store_l_full(limbs, dst_begin, value);
}

fn mp_ne_store_l(limbs: &mut [u8], i: usize, value: LimbType) {
    mp_ne_store_l_full(limbs, i, value)
}

#[test]
fn test_mp_ne_store_l() {
    use super::limb::LIMB_BITS;

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_ne_store_l(&mut limbs, 0, 1 << (LIMB_BITS - 1));
    mp_ne_store_l(&mut limbs, 1, 1);
    assert_eq!(mp_ne_load_l(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(mp_ne_load_l(&limbs, 1), 1);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_ne_store_l_full(&mut limbs, 0, 1 << (LIMB_BITS - 1));
    mp_ne_store_l_full(&mut limbs, 1, 1);
    assert_eq!(mp_ne_load_l(&limbs, 0), 1 << (LIMB_BITS - 1));
    assert_eq!(mp_ne_load_l(&limbs, 1), 1);
}

fn mp_ne_zeroize_bytes_above(limbs: &mut [u8], begin: usize) {
    if begin >= limbs.len() {
        return;
    }
    let begin_limb = begin / LIMB_BYTES;
    let mut begin_aligned = begin_limb * LIMB_BYTES;
    let begin_in_limb = begin - begin_aligned;
    if begin_in_limb != 0 {
        let mut l = mp_ne_load_l(limbs, begin_limb);
        l &= ct_lsb_mask_l(8 * begin_in_limb as u32);
        mp_ne_store_l(limbs, begin_limb, l);
        begin_aligned += LIMB_BYTES;
    }
    limbs[begin_aligned..].fill(0);
}

#[test]
fn test_mp_ne_zeroize_bytes_above() {
    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_ne_store_l(&mut limbs, 0, !0);
    mp_ne_store_l(&mut limbs, 1, !0);
    mp_ne_zeroize_bytes_above(&mut limbs, LIMB_BYTES + 1);
    assert_eq!(mp_ne_load_l(&mut limbs, 0), !0);
    assert_eq!(mp_ne_load_l(&mut limbs, 1), !0 & 0xff);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_ne_store_l(&mut limbs, 0, !0);
    mp_ne_store_l(&mut limbs, 1, !0);
    mp_ne_zeroize_bytes_above(&mut limbs, LIMB_BYTES - 1);
    assert_eq!(mp_ne_load_l(&mut limbs, 0), !0 >> 8);
    assert_eq!(mp_ne_load_l(&mut limbs, 1), 0);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_ne_store_l(&mut limbs, 0, !0);
    mp_ne_store_l(&mut limbs, 1, !0);
    mp_ne_zeroize_bytes_above(&mut limbs, 2 * LIMB_BYTES);
    assert_eq!(mp_ne_load_l(&mut limbs, 0), !0);
    assert_eq!(mp_ne_load_l(&mut limbs, 1), !0);
}

fn mp_ne_zeroize_bytes_below(limbs: &mut [u8], end: usize) {
    let end = end.min(limbs.len());
    let end_limb = mp_ct_nlimbs(end);
    let mut end_aligned = end_limb * LIMB_BYTES;
    let retain_in_limb = end_aligned - end;
    if retain_in_limb != 0 {
        let end_in_limb = LIMB_BYTES - retain_in_limb;
        let mut l = mp_ne_load_l(limbs, end_limb - 1);
        l = l >> 8 * end_in_limb;
        l = l << 8 * end_in_limb;
        mp_ne_store_l(limbs, end_limb - 1, l);
        end_aligned -= LIMB_BYTES;
    }
    limbs[..end_aligned].fill(0);
}

#[test]
fn test_mp_ne_zeroize_bytes_below() {
    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_ne_store_l(&mut limbs, 0, !0);
    mp_ne_store_l(&mut limbs, 1, !0);
    mp_ne_zeroize_bytes_below(&mut limbs, LIMB_BYTES + 1);
    assert_eq!(mp_ne_load_l(&mut limbs, 0), 0);
    assert_eq!(mp_ne_load_l(&mut limbs, 1), !0xff);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_ne_store_l(&mut limbs, 0, !0);
    mp_ne_store_l(&mut limbs, 1, !0);
    mp_ne_zeroize_bytes_below(&mut limbs, LIMB_BYTES - 1);
    assert_eq!(mp_ne_load_l(&mut limbs, 0), 0xff << 8 * (LIMB_BYTES - 1));
    assert_eq!(mp_ne_load_l(&mut limbs, 1), !0);

    let mut limbs: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    mp_ne_store_l(&mut limbs, 0, !0);
    mp_ne_store_l(&mut limbs, 1, !0);
    mp_ne_zeroize_bytes_below(&mut limbs, 0);
    assert_eq!(mp_ne_load_l(&mut limbs, 0), !0);
    assert_eq!(mp_ne_load_l(&mut limbs, 1), !0);
}

pub trait MPIntByteSliceCommonPriv: Sized {
    const SUPPORTS_UNALIGNED_BUFFER_LENGTHS: bool;

    fn _len(&self) -> usize;

    fn nlimbs(&self) -> usize {
        mp_ct_nlimbs(self._len())
    }

    fn partial_high_mask(&self) -> LimbType {
        if Self::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
            let high_npartial = if self._len() != 0 {
                ((self._len()) - 1) % LIMB_BYTES as usize + 1
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
            let high_npartial = self._len() % LIMB_BYTES as usize;
            if high_npartial == 0 {
                0
            } else {
                8 * high_npartial as u32
            }
        } else {
            0
        }
    }

    fn take(self, nbytes: usize) -> (Self, Self);
}

pub trait MPIntByteSliceCommon: MPIntByteSliceCommonPriv + fmt::LowerHex {
    fn len(&self) -> usize {
        self._len()
    }

    fn limbs_align_len(nbytes: usize) -> usize {
        if Self::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
            nbytes
        } else {
            mp_ct_limbs_align_len(nbytes)
        }
    }

    fn is_empty(&self) -> bool;

    fn load_l_full(&self, i: usize) -> LimbType;
    fn load_l(&self, i: usize) -> LimbType;

    fn fmt_lower_hex(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn nibble_to_hexchar(nibble: u8) -> char {
            let c = match nibble {
                0x0..=0x9 => b'0' + (nibble - 0x0),
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
                fmt_byte(f, (v >> 8 * (len - i - 1)) as u8)?;
            }
            Ok(())
        }

        if f.alternate() {
            f.write_str("0x")?;
        }
        if self.is_empty() {
            f.write_str("(empty)")?;
            return Ok(())
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
}

pub trait MPIntByteSlicePriv: MPIntByteSliceCommon {
    type SelfT<'a>: MPIntByteSlice where Self: 'a;

    fn split_at<'a>(&'a self, nbytes: usize) -> (Self::SelfT<'a>, Self::SelfT<'a>)
    where Self: 'a;
}

pub trait MPIntByteSlice: MPIntByteSlicePriv {
    type FromBytesError: fmt::Debug;

    fn from_bytes<'a: 'b, 'b>(bytes: &'a [u8]) -> Result<Self::SelfT<'b>, Self::FromBytesError>
    where Self: 'b;

    fn coerce_lifetime<'a>(self: &'a Self) -> Self::SelfT<'a>;
}

pub trait MPIntMutByteSlicePriv: MPIntByteSliceCommon {
    type SelfT<'a>: MPIntMutByteSlice where Self: 'a;

    fn split_at<'a>(&'a mut self, nbytes: usize) -> (Self::SelfT<'a>, Self::SelfT<'a>)
    where Self: 'a;
}

pub trait MPIntMutByteSlice: MPIntMutByteSlicePriv {
    type FromBytesError: fmt::Debug;

    fn from_bytes<'a: 'b, 'b>(bytes: &'a mut [u8]) -> Result<Self::SelfT<'b>, Self::FromBytesError>
    where Self: 'b;

    fn coerce_lifetime<'a>(self: &'a mut Self) -> Self::SelfT<'a>;

    fn store_l_full(&mut self, i: usize, value: LimbType);
    fn store_l(&mut self, i: usize, value: LimbType);
    fn zeroize_bytes_above(&mut self, begin: usize);
    fn zeroize_bytes_below(&mut self, end: usize);

    fn copy_from<S: MPIntByteSliceCommon>(&'_ mut self, src: &S) {
        let src_nlimbs = src.nlimbs();
        let dst_nlimbs = self.nlimbs();
        debug_assert!(dst_nlimbs >= src_nlimbs);

        if src_nlimbs == 0 {
            self.zeroize_bytes_above(0);
            return;
        }
        for i in 0..src_nlimbs - 1 {
            self.store_l_full(i, src.load_l_full(i));
        }
        let high_limb = src.load_l(src_nlimbs - 1);
        debug_assert!(
            src_nlimbs < dst_nlimbs ||
            (high_limb & !self.partial_high_mask()) == 0
        );
        self.store_l(src_nlimbs - 1, high_limb);
        self.zeroize_bytes_above(src.len());
    }
}

pub struct MPBigEndianByteSlice<'a> {
    bytes: &'a [u8]
}

impl<'a> MPIntByteSliceCommonPriv for MPBigEndianByteSlice<'a> {
    const SUPPORTS_UNALIGNED_BUFFER_LENGTHS: bool = true;

    fn _len(&self) -> usize {
        self.bytes.len()
    }

    fn take(self, nbytes: usize) -> (Self, Self) {
        let (h, l) = self.bytes.split_at(self.bytes.len() - nbytes);
        (Self { bytes: h }, Self { bytes: l })
    }
}

impl<'a> MPIntByteSliceCommon for MPBigEndianByteSlice<'a> {
    fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn load_l_full(&self, i: usize) -> LimbType {
        mp_be_load_l_full(self.bytes, i)
    }

    fn load_l(&self, i: usize) -> LimbType {
        mp_be_load_l(self.bytes, i)
    }
}

impl<'a> MPIntByteSlicePriv for MPBigEndianByteSlice<'a> {
    type SelfT<'b> = MPBigEndianByteSlice<'b> where Self: 'b;

    fn split_at<'b>(&'b self, nbytes: usize) -> (Self::SelfT<'b>, Self::SelfT<'b>)
    where Self: 'b
    {
        let (h, l) = self.bytes.split_at(self.bytes.len() - nbytes);
        (Self::SelfT::<'b> { bytes: h }, Self::SelfT::<'b> { bytes: l })
    }
}

impl<'a> MPIntByteSlice for MPBigEndianByteSlice<'a> {
    type FromBytesError = convert::Infallible;

    fn from_bytes<'b: 'c, 'c>(bytes: &'b [u8]) -> Result<Self::SelfT<'c>, Self::FromBytesError>
    where Self: 'c
    {
        Ok(Self::SelfT::<'c> { bytes })
    }

    fn coerce_lifetime<'b>(self: &'b Self) -> Self::SelfT<'b> {
        Self::from_bytes(self.bytes).unwrap()
    }
}

impl<'a> fmt::LowerHex for MPBigEndianByteSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_lower_hex(f)
    }
}

pub struct MPBigEndianMutByteSlice<'a> {
    bytes: &'a mut [u8]
}

impl<'a> MPIntByteSliceCommonPriv for MPBigEndianMutByteSlice<'a> {
    const SUPPORTS_UNALIGNED_BUFFER_LENGTHS: bool = true;

    fn _len(&self) -> usize {
        self.bytes.len()
    }

    fn take(self, nbytes: usize) -> (Self, Self) {
        let (h, l) = self.bytes.split_at_mut(self.bytes.len() - nbytes);
        (Self { bytes: h }, Self { bytes: l })
    }
}

impl<'a> MPIntByteSliceCommon for MPBigEndianMutByteSlice<'a> {
    fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn load_l_full(&self, i: usize) -> LimbType {
        mp_be_load_l_full(self.bytes, i)
    }

    fn load_l(&self, i: usize) -> LimbType {
        mp_be_load_l(self.bytes, i)
    }
}

impl<'a> MPIntMutByteSlicePriv for MPBigEndianMutByteSlice<'a> {
    type SelfT<'b> = MPBigEndianMutByteSlice<'b> where Self: 'b;

    fn split_at<'b>(&'b mut self, nbytes: usize) -> (Self::SelfT<'b>, Self::SelfT<'b>)
    where Self: 'b
    {
        let (h, l) = self.bytes.split_at_mut(self.bytes.len() - nbytes);
        (Self::SelfT { bytes: h }, Self::SelfT { bytes: l })
    }
}

impl<'a> MPIntMutByteSlice for MPBigEndianMutByteSlice<'a> {
    type FromBytesError = convert::Infallible;

    fn from_bytes<'b: 'c, 'c>(bytes: &'b mut [u8]) -> Result<Self::SelfT<'c>, Self::FromBytesError>
    where Self: 'c
    {
        Ok(Self::SelfT::<'c> { bytes })
    }

    fn coerce_lifetime<'b>(self: &'b mut Self) -> Self::SelfT<'b> {
        Self::from_bytes(self.bytes.as_mut()).unwrap()
    }

    fn store_l_full(&mut self, i: usize, value: LimbType) {
        mp_be_store_l_full(self.bytes, i, value)
    }

    fn store_l(&mut self, i: usize, value: LimbType) {
        mp_be_store_l(self.bytes, i, value)
    }

    fn zeroize_bytes_above(&mut self, begin: usize) {
        mp_be_zeroize_bytes_above(self.bytes, begin)
    }

    fn zeroize_bytes_below(&mut self, end: usize) {
        mp_be_zeroize_bytes_below(self.bytes, end)
    }
}

impl<'a> fmt::LowerHex for MPBigEndianMutByteSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_lower_hex(f)
    }
}

pub struct MPLittleEndianByteSlice<'a> {
    bytes: &'a [u8]
}

impl<'a> MPIntByteSliceCommonPriv for MPLittleEndianByteSlice<'a> {
    const SUPPORTS_UNALIGNED_BUFFER_LENGTHS: bool = true;

    fn _len(&self) -> usize {
        self.bytes.len()
    }

    fn take(self, nbytes: usize) -> (Self, Self) {
        let (l, h) = self.bytes.split_at(nbytes);
        (Self { bytes: h }, Self { bytes: l })
    }
}

impl<'a> MPIntByteSliceCommon for MPLittleEndianByteSlice<'a> {
    fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn load_l_full(&self, i: usize) -> LimbType {
        mp_le_load_l_full(self.bytes, i)
    }

    fn load_l(&self, i: usize) -> LimbType {
        mp_le_load_l(self.bytes, i)
    }
}

impl<'a> MPIntByteSlicePriv for MPLittleEndianByteSlice<'a> {
    type SelfT<'b> = MPLittleEndianByteSlice<'b> where Self: 'b;

    fn split_at<'b>(&'b self, nbytes: usize) -> (Self::SelfT<'b>, Self::SelfT<'b>)
    where Self: 'b
    {
        let (l, h) = self.bytes.split_at(nbytes);
        (Self::SelfT::<'b> { bytes: h }, Self::SelfT::<'b> { bytes: l })
    }
}

impl<'a> MPIntByteSlice for MPLittleEndianByteSlice<'a> {
    type FromBytesError = convert::Infallible;

    fn from_bytes<'b: 'c, 'c>(bytes: &'b [u8]) -> Result<Self::SelfT<'c>, Self::FromBytesError>
    where Self: 'c
    {
        Ok(Self::SelfT::<'c> { bytes })
    }

    fn coerce_lifetime<'b>(self: &'b Self) -> Self::SelfT<'b> {
        Self::from_bytes(self.bytes).unwrap()
    }
}

impl<'a> fmt::LowerHex for MPLittleEndianByteSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_lower_hex(f)
    }
}

pub struct MPLittleEndianMutByteSlice<'a> {
    bytes: &'a mut [u8]
}

impl<'a> MPIntByteSliceCommonPriv for MPLittleEndianMutByteSlice<'a> {
    const SUPPORTS_UNALIGNED_BUFFER_LENGTHS: bool = true;

    fn _len(&self) -> usize {
        self.bytes.len()
    }

    fn take(self, nbytes: usize) -> (Self, Self) {
        let (l, h) = self.bytes.split_at_mut(nbytes);
        (Self { bytes: h }, Self { bytes: l })
    }
}

impl<'a> MPIntByteSliceCommon for MPLittleEndianMutByteSlice<'a> {
    fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn load_l_full(&self, i: usize) -> LimbType {
        mp_le_load_l_full(self.bytes, i)
    }

    fn load_l(&self, i: usize) -> LimbType {
        mp_le_load_l(self.bytes, i)
    }
}

impl<'a> MPIntMutByteSlicePriv for MPLittleEndianMutByteSlice<'a> {
    type SelfT<'b> = MPLittleEndianMutByteSlice<'b> where Self: 'b;

    fn split_at<'b>(&'b mut self, nbytes: usize) -> (Self::SelfT<'b>, Self::SelfT<'b>)
    where Self: 'b
    {
        let (l, h) = self.bytes.split_at_mut(nbytes);
        (Self::SelfT { bytes: h }, Self::SelfT { bytes: l })
    }
}

impl<'a> MPIntMutByteSlice for MPLittleEndianMutByteSlice<'a> {
    type FromBytesError = convert::Infallible;

    fn from_bytes<'b: 'c, 'c>(bytes: &'b mut [u8]) -> Result<Self::SelfT<'c>, Self::FromBytesError>
    where Self: 'c
    {
        Ok(Self::SelfT::<'c> { bytes })
    }

    fn coerce_lifetime<'b>(self: &'b mut Self) -> Self::SelfT<'b> {
        Self::from_bytes(self.bytes.as_mut()).unwrap()
    }

    fn store_l_full(&mut self, i: usize, value: LimbType) {
        mp_le_store_l_full(self.bytes, i, value)
    }

    fn store_l(&mut self, i: usize, value: LimbType) {
        mp_le_store_l(self.bytes, i, value)
    }

    fn zeroize_bytes_above(&mut self, begin: usize) {
        mp_le_zeroize_bytes_above(self.bytes, begin)
    }

    fn zeroize_bytes_below(&mut self, end: usize) {
        mp_le_zeroize_bytes_below(self.bytes, end)
    }
}

impl<'a> fmt::LowerHex for MPLittleEndianMutByteSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_lower_hex(f)
    }
}

#[derive(Debug)]
pub struct UnalignedMPByteSliceLenError {}

pub struct MPNativeEndianByteSlice<'a> {
    bytes: &'a [u8]
}

impl<'a> MPIntByteSliceCommonPriv for MPNativeEndianByteSlice<'a> {
    const SUPPORTS_UNALIGNED_BUFFER_LENGTHS: bool = false;

    fn _len(&self) -> usize {
        self.bytes.len()
    }

    fn take(self, nbytes: usize) -> (Self, Self) {
        debug_assert_eq!(nbytes % LIMB_BYTES, 0);
        let (l, h) = self.bytes.split_at(nbytes);
        (Self { bytes: h }, Self { bytes: l })
    }
}

impl<'a> MPIntByteSliceCommon for MPNativeEndianByteSlice<'a> {
    fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn load_l_full(&self, i: usize) -> LimbType {
        mp_ne_load_l_full(self.bytes, i)
    }

    fn load_l(&self, i: usize) -> LimbType {
        mp_ne_load_l(self.bytes, i)
    }
}

impl<'a> MPIntByteSlicePriv for MPNativeEndianByteSlice<'a> {
    type SelfT<'b> = MPNativeEndianByteSlice<'b> where Self: 'b;

    fn split_at<'b>(&'b self, nbytes: usize) -> (Self::SelfT<'b>, Self::SelfT<'b>)
    where Self: 'b
    {
        debug_assert_eq!(nbytes % LIMB_BYTES, 0);
        let (l, h) = self.bytes.split_at(nbytes);
        (Self::SelfT::<'b> { bytes: h }, Self::SelfT::<'b> { bytes: l })
    }
}

impl<'a> MPIntByteSlice for MPNativeEndianByteSlice<'a> {
    type FromBytesError = UnalignedMPByteSliceLenError;

    fn from_bytes<'b: 'c, 'c>(bytes: &'b [u8]) -> Result<Self::SelfT<'c>, Self::FromBytesError>
    where Self: 'c
    {
        if bytes.len() % LIMB_BYTES == 0 {
            Ok(Self::SelfT::<'c> { bytes })
        } else {
            Err(UnalignedMPByteSliceLenError {})
        }
    }

    fn coerce_lifetime<'b>(self: &'b Self) -> Self::SelfT<'b> {
        Self::from_bytes(self.bytes).unwrap()
    }
}

impl<'a> fmt::LowerHex for MPNativeEndianByteSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_lower_hex(f)
    }
}

pub struct MPNativeEndianMutByteSlice<'a> {
    bytes: &'a mut [u8]
}

impl<'a> MPIntByteSliceCommonPriv for MPNativeEndianMutByteSlice<'a> {
    const SUPPORTS_UNALIGNED_BUFFER_LENGTHS: bool = false;

    fn _len(&self) -> usize {
        self.bytes.len()
    }

    fn take(self, nbytes: usize) -> (Self, Self) {
        debug_assert_eq!(nbytes % LIMB_BYTES, 0);
        let (l, h) = self.bytes.split_at_mut(nbytes);
        (Self { bytes: h }, Self { bytes: l })
    }
}

impl<'a> MPIntByteSliceCommon for MPNativeEndianMutByteSlice<'a> {
    fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    fn load_l_full(&self, i: usize) -> LimbType {
        mp_ne_load_l_full(self.bytes, i)
    }

    fn load_l(&self, i: usize) -> LimbType {
        mp_ne_load_l(self.bytes, i)
    }
}

impl<'a> MPIntMutByteSlicePriv for MPNativeEndianMutByteSlice<'a> {
    type SelfT<'b> = MPNativeEndianMutByteSlice<'b> where Self: 'b;

    fn split_at<'b>(&'b mut self, nbytes: usize) -> (Self::SelfT<'b>, Self::SelfT<'b>)
    where Self: 'b
    {
        debug_assert_eq!(nbytes % LIMB_BYTES, 0);
        let (l, h) = self.bytes.split_at_mut(nbytes);
        (Self::SelfT { bytes: h }, Self::SelfT { bytes: l })
    }
}

impl<'a> MPIntMutByteSlice for MPNativeEndianMutByteSlice<'a> {
    type FromBytesError = UnalignedMPByteSliceLenError;

    fn from_bytes<'b: 'c, 'c>(bytes: &'b mut [u8]) -> Result<Self::SelfT<'c>, Self::FromBytesError>
    where Self: 'c
    {
        if bytes.len() % LIMB_BYTES == 0 {
            Ok(Self::SelfT::<'c> { bytes })
        } else {
            Err(UnalignedMPByteSliceLenError {})
        }
    }

    fn coerce_lifetime<'b>(self: &'b mut Self) -> Self::SelfT<'b> {
        Self::from_bytes(self.bytes.as_mut()).unwrap()
    }

    fn store_l_full(&mut self, i: usize, value: LimbType) {
        mp_ne_store_l_full(self.bytes, i, value)
    }

    fn store_l(&mut self, i: usize, value: LimbType) {
        mp_ne_store_l(self.bytes, i, value)
    }

    fn zeroize_bytes_above(&mut self, begin: usize) {
        mp_ne_zeroize_bytes_above(self.bytes, begin)
    }

    fn zeroize_bytes_below(&mut self, end: usize) {
        mp_ne_zeroize_bytes_below(self.bytes, end)
    }
}

impl<'a> fmt::LowerHex for MPNativeEndianMutByteSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_lower_hex(f)
    }
}

pub fn mp_find_last_set_limb_mp<'a, T0: MPIntByteSliceCommon>(op0: &T0) -> usize {
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
fn test_mp_find_last_set_limb_mp_with_unaligned_lengths<T0: MPIntMutByteSlice>()  {
    let mut op0: [u8; 2 * LIMB_BYTES + 2] = [0; 2 * LIMB_BYTES + 2];
    let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
    op0.store_l(0, 1);
    assert_eq!(mp_find_last_set_limb_mp(&op0), 1);

    op0.store_l(1, 1);
    assert_eq!(mp_find_last_set_limb_mp(&op0), 2);

    op0.store_l(2, 1);
    assert_eq!(mp_find_last_set_limb_mp(&op0), 3);
}

#[cfg(test)]
fn test_mp_find_last_set_limb_mp_with_aligned_lengths<T0: MPIntMutByteSlice>()  {
    let mut op0: [u8; 0] = [0; 0];
    let op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
    assert_eq!(mp_find_last_set_limb_mp(&op0), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
    assert_eq!(mp_find_last_set_limb_mp(&op0), 0);

    op0.store_l(0, 1);
    assert_eq!(mp_find_last_set_limb_mp(&op0), 1);

    op0.store_l(1, 1);
    assert_eq!(mp_find_last_set_limb_mp(&op0), 2);
}

#[test]
fn test_mp_find_last_set_limb_be()  {
    test_mp_find_last_set_limb_mp_with_unaligned_lengths::<MPBigEndianMutByteSlice>();
    test_mp_find_last_set_limb_mp_with_aligned_lengths::<MPBigEndianMutByteSlice>();
}

#[test]
fn test_mp_find_last_set_limb_le()  {
    test_mp_find_last_set_limb_mp_with_unaligned_lengths::<MPLittleEndianMutByteSlice>();
    test_mp_find_last_set_limb_mp_with_aligned_lengths::<MPLittleEndianMutByteSlice>();
}

#[test]
fn test_mp_find_last_set_limb_ne()  {
    test_mp_find_last_set_limb_mp_with_aligned_lengths::<MPNativeEndianMutByteSlice>();
}

pub fn mp_find_last_set_byte_mp<'a, T0: MPIntByteSliceCommon>(op0: &T0) -> usize {
    let nlimbs = mp_find_last_set_limb_mp(op0);
    if nlimbs == 0 {
        return 0;
    }
    let nlimbs = nlimbs - 1;
    nlimbs * LIMB_BYTES + ct_find_last_set_byte_l(op0.load_l(nlimbs))
}

#[cfg(test)]
fn test_mp_find_last_set_byte_mp_with_unaligned_lengths<T0: MPIntMutByteSlice>() {
    let mut op0: [u8; 2 * LIMB_BYTES + 2] = [0; 2 * LIMB_BYTES + 2];
    let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
    op0.store_l(0, 1);
    assert_eq!(mp_find_last_set_byte_mp(&op0), 1);

    op0.store_l(1, 1);
    assert_eq!(mp_find_last_set_byte_mp(&op0), LIMB_BYTES + 1);

    op0.store_l(2, 1);
    assert_eq!(mp_find_last_set_byte_mp(&op0), 2 * LIMB_BYTES + 1);
}

#[cfg(test)]
fn test_mp_find_last_set_byte_mp_with_aligned_lengths<T0: MPIntMutByteSlice>() {
    use super::limb::LIMB_BITS;

    let mut op0: [u8; 0] = [0; 0];
    let op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
    assert_eq!(mp_find_last_set_byte_mp(&op0), 0);

    let mut op0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut op0 = T0::from_bytes(op0.as_mut_slice()).unwrap();
    op0.store_l(0, (1 as LimbType) << LIMB_BITS - 1);
    assert_eq!(mp_find_last_set_byte_mp(&op0), LIMB_BYTES);

    op0.store_l(1, (1 as LimbType) << LIMB_BITS - 1);
    assert_eq!(mp_find_last_set_byte_mp(&op0), 2 * LIMB_BYTES);
}

#[test]
fn test_mp_find_last_set_byte_be() {
    test_mp_find_last_set_byte_mp_with_unaligned_lengths::<MPBigEndianMutByteSlice>();
    test_mp_find_last_set_byte_mp_with_aligned_lengths::<MPBigEndianMutByteSlice>();
}

#[test]
fn test_mp_find_last_set_byte_le() {
    test_mp_find_last_set_byte_mp_with_unaligned_lengths::<MPLittleEndianMutByteSlice>();
    test_mp_find_last_set_byte_mp_with_aligned_lengths::<MPLittleEndianMutByteSlice>();
}

#[test]
fn test_mp_find_last_set_byte_ne() {
    test_mp_find_last_set_byte_mp_with_aligned_lengths::<MPNativeEndianMutByteSlice>();
}

pub fn mp_ct_find_first_set_bit_mp<T0: MPIntByteSliceCommon>(op0: &T0) -> (LimbChoice, usize) {
    let mut tail_is_zero = LimbChoice::from(1);
    let mut ntrailing_zeroes: usize = 0;
    for i in 0..op0.nlimbs() {
        let op0_val = op0.load_l(i);
        ntrailing_zeroes += tail_is_zero.select(0, ct_find_first_set_bit_l(op0_val) as LimbType) as usize;
        tail_is_zero &= LimbChoice::from(ct_is_zero_l(op0_val));
    }
    (!tail_is_zero, tail_is_zero.select_usize(ntrailing_zeroes, 0))
}

#[cfg(test)]
fn test_mp_find_first_set_bit_mp<T0: MPIntMutByteSlice>() {
    let mut limbs: [u8; 3 * LIMB_BYTES] = [0u8; 3 * LIMB_BYTES];
    let limbs = T0::from_bytes(&mut limbs).unwrap();
    let (is_nonzero, first_set_bit_pos) = mp_ct_find_first_set_bit_mp(&limbs);
    assert_eq!(is_nonzero.unwrap(), 0);
    assert_eq!(first_set_bit_pos, 0);

    for i in 0..3 * LIMB_BITS as usize {
    let mut limbs: [u8; 3 * LIMB_BYTES] = [0u8; 3 * LIMB_BYTES];
        let mut limbs = T0::from_bytes(&mut limbs).unwrap();
        let limb_index = i / LIMB_BITS as usize;
        let bit_pos_in_limb = i % LIMB_BITS as usize;

        limbs.store_l(limb_index, 1 << bit_pos_in_limb);
        let (is_nonzero, first_set_bit_pos) = mp_ct_find_first_set_bit_mp(&limbs);
        assert!(is_nonzero.unwrap() != 0);
        assert_eq!(first_set_bit_pos, i);

        limbs.store_l(limb_index, !((1 << bit_pos_in_limb) - 1));
        for j in limb_index + 1..limbs.nlimbs() {
            limbs.store_l(j, !0);
        }
        let (is_nonzero, first_set_bit_pos) = mp_ct_find_first_set_bit_mp(&limbs);
        assert!(is_nonzero.unwrap() != 0);
        assert_eq!(first_set_bit_pos, i);
    }
}

#[test]
fn test_mp_find_first_set_bit_be() {
    test_mp_find_first_set_bit_mp::<MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_find_first_set_bit_le() {
    test_mp_find_first_set_bit_mp::<MPLittleEndianMutByteSlice>()
}

#[test]
fn test_mp_find_first_set_bit_ne() {
    test_mp_find_first_set_bit_mp::<MPNativeEndianMutByteSlice>()
}

pub fn mp_ct_find_last_set_bit_mp<T0: MPIntByteSliceCommon>(op0: &T0) -> (LimbChoice, usize) {
    let mut head_is_zero = LimbChoice::from(1);
    let mut nleading_zeroes: usize = 0;
    let mut i = op0.nlimbs();
    while i > 0 {
        i -= 1;
        let op0_val = op0.load_l(i);
        nleading_zeroes += head_is_zero.select(
            0,
            LIMB_BITS as LimbType - ct_find_last_set_bit_l(op0_val) as LimbType
        ) as usize;
        head_is_zero &= LimbChoice::from(ct_is_zero_l(op0_val));
    }
    (!head_is_zero, op0.nlimbs() * LIMB_BITS as usize - nleading_zeroes)
}

#[cfg(test)]
fn test_mp_find_last_set_bit_mp<T0: MPIntMutByteSlice>() {
    let mut limbs: [u8; 3 * LIMB_BYTES] = [0u8; 3 * LIMB_BYTES];
    let limbs = T0::from_bytes(&mut limbs).unwrap();
    let (is_nonzero, first_set_bit_pos) = mp_ct_find_last_set_bit_mp(&limbs);
    assert_eq!(is_nonzero.unwrap(), 0);
    assert_eq!(first_set_bit_pos, 0);

    for i in 0..3 * LIMB_BITS as usize {
        let mut limbs: [u8; 3 * LIMB_BYTES] = [0u8; 3 * LIMB_BYTES];
        let mut limbs = T0::from_bytes(&mut limbs).unwrap();
        let limb_index = i / LIMB_BITS as usize;
        let bit_pos_in_limb = i % LIMB_BITS as usize;

        limbs.store_l(limb_index, 1 << bit_pos_in_limb);
        let (is_nonzero, last_set_bit_pos) = mp_ct_find_last_set_bit_mp(&limbs);
        assert!(is_nonzero.unwrap() != 0);
        assert_eq!(last_set_bit_pos, i + 1);

        limbs.store_l(limb_index, (1 << bit_pos_in_limb) - 1);
        for j in 0..limb_index {
            limbs.store_l(j, !0);
        }
        let (is_nonzero, last_set_bit_pos) = mp_ct_find_last_set_bit_mp(&limbs);
        assert_eq!(is_nonzero.unwrap() != 0, i != 0);
        assert_eq!(last_set_bit_pos, i);
    }
}

#[test]
fn test_mp_find_last_set_bit_be() {
    test_mp_find_last_set_bit_mp::<MPBigEndianMutByteSlice>()
}

#[test]
fn test_mp_find_last_set_bit_le() {
    test_mp_find_last_set_bit_mp::<MPLittleEndianMutByteSlice>()
}

#[test]
fn test_mp_find_last_set_bit_ne() {
    test_mp_find_last_set_bit_mp::<MPNativeEndianMutByteSlice>()
}

pub fn mp_ct_zeroize_bits_above<T0: MPIntMutByteSlice>(op0: &mut T0, begin: usize) {
    let first_limb_index = begin / LIMB_BITS  as usize;
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
fn test_mp_ct_zeroize_bits_above_common<T0: MPIntMutByteSlice>(op0_len: usize) {
    fn fill_with_ones<T0: MPIntMutByteSlice>(op0: &mut T0) {
        for i in 0..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                op0.store_l_full(i, !0);
            } else {
                op0.store_l(i, op0.partial_high_mask());
            }
        }
    }

    let mut op0 = vec![0u8; op0_len];
    let mut op0 = T0::from_bytes(&mut op0).unwrap();
    fill_with_ones(&mut op0);
    for begin in [8 * op0_len, 8 * op0_len + 1, 8 * op0.nlimbs() * LIMB_BYTES, 8 * op0.nlimbs() * LIMB_BYTES + 1] {
        mp_ct_zeroize_bits_above(&mut op0, begin);
        for i in 0..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                assert_eq!(op0.load_l_full(i), !0);
            } else {
                assert_eq!(op0.load_l(i), op0.partial_high_mask());
            }
        }
    }

    for j in 0..mp_ct_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize;

        let mut op0 = vec![0u8; op0_len];
        let mut op0 = T0::from_bytes(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        mp_ct_zeroize_bits_above(&mut op0, begin);
        for i in 0..j {
            assert_eq!(op0.load_l_full(i), !0);
        }

        for i in j..op0.nlimbs() {
            assert_eq!(op0.load_l(i), 0);
        }
    }

    for j in 1..mp_ct_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize - 1;
        let begin = begin.min(8 * op0_len  - 1);

        let mut op0 = vec![0u8; op0_len];
        let mut op0 = T0::from_bytes(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        mp_ct_zeroize_bits_above(&mut op0, begin);
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
fn test_mp_ct_zeroize_bits_above_with_aligned_lengths<T0: MPIntMutByteSlice>() {
    test_mp_ct_zeroize_bits_above_common::<T0>(0);
    test_mp_ct_zeroize_bits_above_common::<T0>(LIMB_BYTES);
    test_mp_ct_zeroize_bits_above_common::<T0>(2 * LIMB_BYTES);
    test_mp_ct_zeroize_bits_above_common::<T0>(3 * LIMB_BYTES);
}

#[cfg(test)]
fn test_mp_ct_zeroize_bits_above_with_unaligned_lengths<T0: MPIntMutByteSlice>() {
    test_mp_ct_zeroize_bits_above_common::<T0>(LIMB_BYTES - 1);
    test_mp_ct_zeroize_bits_above_common::<T0>(2 * LIMB_BYTES - 1);
    test_mp_ct_zeroize_bits_above_common::<T0>(3 * LIMB_BYTES - 1);
}

#[test]
fn test_mp_ct_zeroize_bits_above_be() {
    test_mp_ct_zeroize_bits_above_with_aligned_lengths::<MPBigEndianMutByteSlice>();
    test_mp_ct_zeroize_bits_above_with_unaligned_lengths::<MPBigEndianMutByteSlice>();
}

#[test]
fn test_mp_ct_zeroize_bits_above_le() {
    test_mp_ct_zeroize_bits_above_with_aligned_lengths::<MPLittleEndianMutByteSlice>();
    test_mp_ct_zeroize_bits_above_with_unaligned_lengths::<MPLittleEndianMutByteSlice>();
}

#[test]
fn test_mp_ct_zeroize_bits_above_ne() {
    test_mp_ct_zeroize_bits_above_with_aligned_lengths::<MPNativeEndianMutByteSlice>();
}

pub fn mp_zeroize_bits_above<T0: MPIntMutByteSlice>(op0: &mut T0, begin: usize) {
    let first_limb_index = begin / LIMB_BITS  as usize;
    if op0.nlimbs() <= first_limb_index {
        return;
    }
    let first_limb_retain_nbits = begin % LIMB_BITS as usize;
    let first_limb_mask = ct_lsb_mask_l(first_limb_retain_nbits as u32);
    op0.store_l(
        first_limb_index,
        op0.load_l(first_limb_index) & first_limb_mask
    );

    op0.zeroize_bytes_above((begin + LIMB_BITS as usize - 1) / LIMB_BITS as usize * LIMB_BYTES);
}

#[cfg(test)]
fn test_mp_zeroize_bits_above_common<T0: MPIntMutByteSlice>(op0_len: usize) {
    fn fill_with_ones<T0: MPIntMutByteSlice>(op0: &mut T0) {
        for i in 0..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                op0.store_l_full(i, !0);
            } else {
                op0.store_l(i, op0.partial_high_mask());
            }
        }
    }

    let mut op0 = vec![0u8; op0_len];
    let mut op0 = T0::from_bytes(&mut op0).unwrap();
    fill_with_ones(&mut op0);
    for begin in [8 * op0_len, 8 * op0_len + 1, 8 * op0.nlimbs() * LIMB_BYTES, 8 * op0.nlimbs() * LIMB_BYTES + 1] {
        mp_zeroize_bits_above(&mut op0, begin);
        for i in 0..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                assert_eq!(op0.load_l_full(i), !0);
            } else {
                assert_eq!(op0.load_l(i), op0.partial_high_mask());
            }
        }
    }

    for j in 0..mp_ct_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize;

        let mut op0 = vec![0u8; op0_len];
        let mut op0 = T0::from_bytes(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        mp_zeroize_bits_above(&mut op0, begin);
        for i in 0..j {
            assert_eq!(op0.load_l_full(i), !0);
        }

        for i in j..op0.nlimbs() {
            assert_eq!(op0.load_l(i), 0);
        }
    }

    for j in 1..mp_ct_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize - 1;
        let begin = begin.min(8 * op0_len  - 1);

        let mut op0 = vec![0u8; op0_len];
        let mut op0 = T0::from_bytes(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        mp_zeroize_bits_above(&mut op0, begin);
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
fn test_mp_zeroize_bits_above_with_aligned_lengths<T0: MPIntMutByteSlice>() {
    test_mp_zeroize_bits_above_common::<T0>(0);
    test_mp_zeroize_bits_above_common::<T0>(LIMB_BYTES);
    test_mp_zeroize_bits_above_common::<T0>(2 * LIMB_BYTES);
    test_mp_zeroize_bits_above_common::<T0>(3 * LIMB_BYTES);
}

#[cfg(test)]
fn test_mp_zeroize_bits_above_with_unaligned_lengths<T0: MPIntMutByteSlice>() {
    test_mp_zeroize_bits_above_common::<T0>(LIMB_BYTES - 1);
    test_mp_zeroize_bits_above_common::<T0>(2 * LIMB_BYTES - 1);
    test_mp_zeroize_bits_above_common::<T0>(3 * LIMB_BYTES - 1);
}

#[test]
fn test_mp_zeroize_bits_above_be() {
    test_mp_zeroize_bits_above_with_aligned_lengths::<MPBigEndianMutByteSlice>();
    test_mp_zeroize_bits_above_with_unaligned_lengths::<MPBigEndianMutByteSlice>();
}

#[test]
fn test_mp_zeroize_bits_above_le() {
    test_mp_zeroize_bits_above_with_aligned_lengths::<MPLittleEndianMutByteSlice>();
    test_mp_zeroize_bits_above_with_unaligned_lengths::<MPLittleEndianMutByteSlice>();
}

#[test]
fn test_mp_zeroize_bits_above_ne() {
    test_mp_zeroize_bits_above_with_aligned_lengths::<MPNativeEndianMutByteSlice>();
}

pub fn mp_ct_zeroize_bits_below<T0: MPIntMutByteSlice>(op0: &mut T0, end: usize) {
    let last_limb_index = end / LIMB_BITS  as usize;
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
fn test_mp_ct_zeroize_bits_below_common<T0: MPIntMutByteSlice>(op0_len: usize) {
    fn fill_with_ones<T0: MPIntMutByteSlice>(op0: &mut T0) {
        for i in 0..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                op0.store_l_full(i, !0);
            } else {
                op0.store_l(i, op0.partial_high_mask());
            }
        }
    }

    let mut op0 = vec![0u8; op0_len];
    let mut op0 = T0::from_bytes(&mut op0).unwrap();
    fill_with_ones(&mut op0);
    mp_ct_zeroize_bits_below(&mut op0, 0);
    for i in 0..op0.nlimbs() {
        if i + 1 != op0.nlimbs() {
            assert_eq!(op0.load_l_full(i), !0);
        } else {
            assert_eq!(op0.load_l(i), op0.partial_high_mask());
        }
    }

    for j in 0..mp_ct_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize;

        let mut op0 = vec![0u8; op0_len];
        let mut op0 = T0::from_bytes(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        mp_ct_zeroize_bits_below(&mut op0, begin);
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

    for j in 1..mp_ct_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize - 1;
        let begin = begin.min(8 * op0_len  - 1);

        let mut op0 = vec![0u8; op0_len];
        let mut op0 = T0::from_bytes(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        mp_ct_zeroize_bits_below(&mut op0, begin);
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
fn test_mp_ct_zeroize_bits_below_with_aligned_lengths<T0: MPIntMutByteSlice>() {
    test_mp_ct_zeroize_bits_below_common::<T0>(0);
    test_mp_ct_zeroize_bits_below_common::<T0>(LIMB_BYTES);
    test_mp_ct_zeroize_bits_below_common::<T0>(2 * LIMB_BYTES);
    test_mp_ct_zeroize_bits_below_common::<T0>(3 * LIMB_BYTES);
}

#[cfg(test)]
fn test_mp_ct_zeroize_bits_below_with_unaligned_lengths<T0: MPIntMutByteSlice>() {
    test_mp_ct_zeroize_bits_below_common::<T0>(LIMB_BYTES - 1);
    test_mp_ct_zeroize_bits_below_common::<T0>(2 * LIMB_BYTES - 1);
    test_mp_ct_zeroize_bits_below_common::<T0>(3 * LIMB_BYTES - 1);
}

#[test]
fn test_mp_ct_zeroize_bits_below_be() {
    test_mp_ct_zeroize_bits_below_with_aligned_lengths::<MPBigEndianMutByteSlice>();
    test_mp_ct_zeroize_bits_below_with_unaligned_lengths::<MPBigEndianMutByteSlice>();
}

#[test]
fn test_mp_ct_zeroize_bits_below_le() {
    test_mp_ct_zeroize_bits_below_with_aligned_lengths::<MPLittleEndianMutByteSlice>();
    test_mp_ct_zeroize_bits_below_with_unaligned_lengths::<MPLittleEndianMutByteSlice>();
}

#[test]
fn test_mp_ct_zeroize_bits_below_ne() {
    test_mp_ct_zeroize_bits_below_with_aligned_lengths::<MPNativeEndianMutByteSlice>();
}

pub fn mp_zeroize_bits_below<T0: MPIntMutByteSlice>(op0: &mut T0, end: usize) {
    let last_limb_index = end / LIMB_BITS  as usize;
    op0.zeroize_bytes_below(last_limb_index * LIMB_BYTES);
    if last_limb_index >= op0.nlimbs() {
        return;
    }
    let last_limb_clear_nbits = end % LIMB_BITS as usize;
    let last_limb_mask = !ct_lsb_mask_l(last_limb_clear_nbits as u32);
    op0.store_l(
        last_limb_index,
        op0.load_l(last_limb_index) & last_limb_mask
    );
}

#[cfg(test)]
fn test_mp_zeroize_bits_below_common<T0: MPIntMutByteSlice>(op0_len: usize) {
    fn fill_with_ones<T0: MPIntMutByteSlice>(op0: &mut T0) {
        for i in 0..op0.nlimbs() {
            if i + 1 != op0.nlimbs() {
                op0.store_l_full(i, !0);
            } else {
                op0.store_l(i, op0.partial_high_mask());
            }
        }
    }

    let mut op0 = vec![0u8; op0_len];
    let mut op0 = T0::from_bytes(&mut op0).unwrap();
    fill_with_ones(&mut op0);
    mp_zeroize_bits_below(&mut op0, 0);
    for i in 0..op0.nlimbs() {
        if i + 1 != op0.nlimbs() {
            assert_eq!(op0.load_l_full(i), !0);
        } else {
            assert_eq!(op0.load_l(i), op0.partial_high_mask());
        }
    }

    for j in 0..mp_ct_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize;

        let mut op0 = vec![0u8; op0_len];
        let mut op0 = T0::from_bytes(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        mp_zeroize_bits_below(&mut op0, begin);
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

    for j in 1..mp_ct_nlimbs(op0_len) {
        let begin = j * LIMB_BITS as usize - 1;
        let begin = begin.min(8 * op0_len  - 1);

        let mut op0 = vec![0u8; op0_len];
        let mut op0 = T0::from_bytes(&mut op0).unwrap();
        fill_with_ones(&mut op0);
        mp_zeroize_bits_below(&mut op0, begin);
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
fn test_mp_zeroize_bits_below_with_aligned_lengths<T0: MPIntMutByteSlice>() {
    test_mp_zeroize_bits_below_common::<T0>(0);
    test_mp_zeroize_bits_below_common::<T0>(LIMB_BYTES);
    test_mp_zeroize_bits_below_common::<T0>(2 * LIMB_BYTES);
    test_mp_zeroize_bits_below_common::<T0>(3 * LIMB_BYTES);
}

#[cfg(test)]
fn test_mp_zeroize_bits_below_with_unaligned_lengths<T0: MPIntMutByteSlice>() {
    test_mp_zeroize_bits_below_common::<T0>(LIMB_BYTES - 1);
    test_mp_zeroize_bits_below_common::<T0>(2 * LIMB_BYTES - 1);
    test_mp_zeroize_bits_below_common::<T0>(3 * LIMB_BYTES - 1);
}

#[test]
fn test_mp_zeroize_bits_below_be() {
    test_mp_zeroize_bits_below_with_aligned_lengths::<MPBigEndianMutByteSlice>();
    test_mp_zeroize_bits_below_with_unaligned_lengths::<MPBigEndianMutByteSlice>();
}

#[test]
fn test_mp_zeroize_bits_below_le() {
    test_mp_zeroize_bits_below_with_aligned_lengths::<MPLittleEndianMutByteSlice>();
    test_mp_zeroize_bits_below_with_unaligned_lengths::<MPLittleEndianMutByteSlice>();
}

#[test]
fn test_mp_zeroize_bits_below_ne() {
    test_mp_zeroize_bits_below_with_aligned_lengths::<MPNativeEndianMutByteSlice>();
}

pub fn mp_ct_swap_cond<T0: MPIntMutByteSlice, T1: MPIntMutByteSlice>(
    op0: &mut T0, op1: &mut T1, cond: LimbChoice
) {
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
fn test_mp_ct_swap_cond_common<T0: MPIntMutByteSlice, T1: MPIntMutByteSlice>() {
    use super::cmp_impl::mp_ct_eq_mp_mp;

    let len = T0::limbs_align_len(2 * LIMB_BYTES - 1);
    let len = T1::limbs_align_len(len);

    let mut op0_orig = vec![0xccu8; len];
    let mut op0 = op0_orig.clone();
    let op0_orig = T0::from_bytes(&mut op0_orig).unwrap();
    let mut op0 = T0::from_bytes(&mut op0).unwrap();

    let mut op1_orig = vec![0xbbu8; len];
    let mut op1 = op1_orig.clone();
    let op1_orig = T0::from_bytes(&mut op1_orig).unwrap();
    let mut op1 = T0::from_bytes(&mut op1).unwrap();

    mp_ct_swap_cond(&mut op0, &mut op1, LimbChoice::from(0));
    assert_ne!(mp_ct_eq_mp_mp(&op0, &op0_orig).unwrap(), 0);
    assert_ne!(mp_ct_eq_mp_mp(&op1, &op1_orig).unwrap(), 0);

    mp_ct_swap_cond(&mut op0, &mut op1, LimbChoice::from(1));
    assert_ne!(mp_ct_eq_mp_mp(&op0, &op1_orig).unwrap(), 0);
    assert_ne!(mp_ct_eq_mp_mp(&op1, &op0_orig).unwrap(), 0);
}

#[test]
fn test_mp_ct_swap_cond_be_be() {
    test_mp_ct_swap_cond_common::<MPBigEndianMutByteSlice, MPBigEndianMutByteSlice>();
}

#[test]
fn test_mp_ct_swap_cond_le_le() {
    test_mp_ct_swap_cond_common::<MPLittleEndianMutByteSlice, MPLittleEndianMutByteSlice>();
}

#[test]
fn test_mp_ct_swap_cond_ne_ne() {
    test_mp_ct_swap_cond_common::<MPNativeEndianMutByteSlice, MPNativeEndianMutByteSlice>();
}

/// Internal data structure describing a single one of a [`CompositeLimbsBuffer`]'s constituting
/// segments.
struct CompositeLimbsBufferSegment<'a, ST: MPIntByteSliceCommon> {
    /// The total number of limbs whose least significant bytes are held in this or less significant
    /// segments each (note that a limb may span multiple segments).
    end: usize,
    /// The actual data, interpreted as a segment within a composed multiprecision integer
    /// byte buffer.
    ///
    /// The `segment` slice, if non-empty, will be made to align with the least significant limb
    /// whose least significant byte is found in this segment. That is, the slice's last byte
    /// (in memory order) corresponds to the least significant byte of the least significant limb.
    ///
    segment: ST,
    /// Continuation bytes of this or preceeding, less significant segment's most significant limb.
    ///
    /// A segment's most significant limb might be covered only partially by it and extend across
    /// one or more subsequent, more significant segments. In this case its remaining, more
    /// significant continutation bytes are collected from this and the subsequent, more significant
    /// segments' `high_next_partial` slices.
    high_next_partial: ST,

    _phantom: marker::PhantomData<&'a [u8]>
}

/// Access multiprecision integers composed of multiple virtually concatenated byte buffers in units
/// of [`LimbType`].
///
/// A [`CompositeLimbsBuffer`] provides a composed multiprecision integer byte buffer view on
/// `N_SEGMENTS` virtually concatenated byte slices, of endianess as specified by the `E` generic
/// parameter each. Primitives are provided alongside for accessing the composed integer in units of
/// [`LimbType`].
///
/// Certain applications need to truncate the result of some multiprecision integer arithmetic
/// operation result and dismiss the rest. An example would be key generation modulo some large
/// integer by the method of oversampling. A `CompositeLimbsBuffer` helps such applications to
/// reduce the memory footprint of the truncation operation: instead of allocating a smaller
/// destination buffer and copying the to be retained parts over from the result, the arithmetic
/// primitive can, if supported, operate directly on a multiprecision integer byte buffer composed
/// of several independent smaller slices by virtual concatenation. Assuming the individual
/// segments' slice lengths align properly with the needed and unneeded parts of the result, the
/// latter ones can get truncated away trivially when done by simply dismissing the corresponding
/// underlying buffers.
///
/// See also [`CompositeLimbsBuffer`] for a non-mutable variant.
///
pub struct CompositeLimbsBuffer<'a, ST: MPIntByteSliceCommon, const N_SEGMENTS: usize> {
    /// The composed view's individual segments, ordered from least to most significant.
    segments: [CompositeLimbsBufferSegment<'a, ST>; N_SEGMENTS],
}

impl<'a, ST: MPIntByteSliceCommon, const N_SEGMENTS: usize> CompositeLimbsBuffer<'a, ST, N_SEGMENTS> {
    /// Construct a `CompositeLimbsBuffer` view from the individual byte buffer segments.
    ///
    /// # Arguments
    ///
    /// * `segments` - An array of `N_SEGMENTS` byte slices to compose the multiprecision integer
    ///                byte buffer view from by virtual concatenation. Ordered from least to most
    ///                significant relative with respect to their position within the resulting
    ///                view.
    ///
    pub fn new(segments: [ST; N_SEGMENTS]) -> Self
    {
        let mut segments = <[ST; N_SEGMENTS] as IntoIterator>::into_iter(segments);
        let mut segments: [Option<ST>; N_SEGMENTS]
            = core::array::from_fn(|_| segments.next());
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
            let end = mp_ct_nlimbs(n_bytes_total);
            CompositeLimbsBufferSegment { end, segment, high_next_partial, _phantom: marker::PhantomData }
        };

        let segments: [CompositeLimbsBufferSegment<'a, ST>; N_SEGMENTS] = core::array::from_fn(&mut create_segment);
        Self { segments }
    }

    /// Lookup the least significant buffer segment holding the specified limb's least significant
    /// byte.
    ///
    /// Returns a pair of segment index and, as a by-product of the search, the corresponding segment's
    /// offset within the composed multiprecision integer byte buffer in terms of limbs.
    ///
    /// Execution time depends on the composed multiprecision integer's underlying segment layout as
    /// well as as on on the limb index argument `i` as far as branching is concerned.
    ///
    /// # Arguments
    ///
    /// * `i` - The index of the limb to load, counted from least to most significant.
    ///
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
    /// Execution time depends on the composed multiprecision integer's underlying segment layout as
    /// well as as on on the limb index argument `i`, and is otherwise constant as far as branching
    /// is concerned.
    ///
    /// * `i` - The index of the limb to load, counted from least to most significant.
    ///
    pub fn load<'b>(&'b self, i: usize) -> LimbType {
        let (segment_index, segment_offset) = self.limb_index_to_segment(i);
        let segment = &self.segments[segment_index];
        let segment_slice = &segment.segment;
        if i != segment.end - 1 || !ST::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
            segment_slice.load_l_full(i - segment_offset)
        } else if segment_index + 1 == N_SEGMENTS || segment_slice.len() % LIMB_BYTES == 0 {
            // The last (highest) segment's most significant bytes don't necessarily occupy a full
            // limb.
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

impl<'a, ST: MPIntMutByteSlice, const N_SEGMENTS: usize> CompositeLimbsBuffer<'a, ST, N_SEGMENTS> {
    /// Update a limb in the composed multiprecision integer byte buffer.
    ///
    /// Execution time depends on the composed multiprecision integer's underlying segment layout as
    /// well as as on on the limb index argument `i`, and is otherwise constant as far as branching
    /// is concerned.
    ///
    /// * `i` - The index of the limb to update, counted from least to most significant.
    /// * `value` - The value to store in the i'th limb.
    ///
    pub fn store(&mut self, i: usize, value: LimbType) {
        let (segment_index, segment_offset) = self.limb_index_to_segment(i);
        let segment = &mut self.segments[segment_index];
        let segment_slice = &mut segment.segment;
        if i != segment.end - 1 || !ST::SUPPORTS_UNALIGNED_BUFFER_LENGTHS {
            segment_slice.store_l_full(i - segment_offset, value);
        } else if segment_index + 1 == N_SEGMENTS || segment_slice.len() % LIMB_BYTES == 0 {
            // The last (highest) part's most significant bytes don't necessarily occupy a full
            // limb.
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
    use super::limb::LIMB_BITS;

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
    buf2[1 + LIMB_BYTES ] = 0x7;

    // 0x09 00 .. 00 08 00 .. 00
    buf2[1 + LIMB_BYTES / 2] = 0x8;
    buf2[1] = 0x9;

    // 0x00 .. 00 0x0a
    buf2[0] = 0xa;

    let buf0 = MPBigEndianByteSlice::from_bytes(buf0.as_slice()).unwrap();
    let buf1 = MPBigEndianByteSlice::from_bytes(buf1.as_slice()).unwrap();
    let buf2 = MPBigEndianByteSlice::from_bytes(buf2.as_slice()).unwrap();
    let limbs =  CompositeLimbsBuffer::new(
        [buf0, buf1, buf2]
    );

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
    use super::limb::LIMB_BITS;

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

    let buf0 = MPLittleEndianByteSlice::from_bytes(buf0.as_slice()).unwrap();
    let buf1 = MPLittleEndianByteSlice::from_bytes(buf1.as_slice()).unwrap();
    let buf2 = MPLittleEndianByteSlice::from_bytes(buf2.as_slice()).unwrap();
    let limbs =  CompositeLimbsBuffer::new(
        [buf0, buf1, buf2]
    );

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
    use super::limb::LIMB_BITS;

    let mut buf0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let buf1: [u8; 0] = [0; 0];
    let mut buf2: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];

    if test_ne_is_le() {
        // 0x00 .. 00 01 00 .. 00 02
        buf0[LIMB_BYTES / 2 - 1] = 0x1;
        buf0[LIMB_BYTES - 1] = 0x2;

        // 0x00 .. 03 00 .. 00 04 05
        buf0[LIMB_BYTES + LIMB_BYTES / 2 - 1] = 0x3;
        buf0[2 * LIMB_BYTES - 2] = 0x4;
        buf0[2 * LIMB_BYTES - 1] = 0x5;

        // 0x00 .. 00 06 00 .. 00 07
        buf2[LIMB_BYTES / 2 - 1] = 0x6;
        buf2[LIMB_BYTES - 1] = 0x7;

        // 0x00 .. 00 08 00 .. 00 09
        buf2[LIMB_BYTES + LIMB_BYTES / 2 - 1] = 0x8;
        buf2[2 * LIMB_BYTES - 1] = 0x9;
    } else {
        // 0x02 00 .. 00 01 00 .. 00
        buf0[0] = 0x2;
        buf0[LIMB_BYTES / 2] = 0x1;

        // 0x05 04 00 .. 00 03 00 .. 00
        buf0[LIMB_BYTES + LIMB_BYTES / 2] = 0x3;
        buf0[LIMB_BYTES + 1] = 0x4;
        buf0[LIMB_BYTES] = 0x5;

        // 0x07 00 .. 00 06 00 .. 00
        buf2[LIMB_BYTES / 2] = 0x6;
        buf2[0] = 0x7;

        // 0x09 00 .. 00 08 00 .. 00
        buf2[LIMB_BYTES + LIMB_BYTES / 2] = 0x8;
        buf2[0] = 0x9;
    }

    let buf0 = MPNativeEndianByteSlice::from_bytes(buf0.as_slice()).unwrap();
    let buf1 = MPNativeEndianByteSlice::from_bytes(buf1.as_slice()).unwrap();
    let buf2 = MPNativeEndianByteSlice::from_bytes(buf2.as_slice()).unwrap();
    let limbs =  CompositeLimbsBuffer::new(
        [buf0, buf1, buf2]
    );

    let l0 = limbs.load(0);
    assert_eq!(l0, 0x2 << LIMB_BITS - 8 | 0x1 << LIMB_BITS / 2 - 8);
    let l1 = limbs.load(1);
    assert_eq!(l1, 0x0504 << LIMB_BITS - 16 | 0x3 << LIMB_BITS / 2 - 8);
    let l2 = limbs.load(2);
    assert_eq!(l2, 0x7 << LIMB_BITS - 8 | 0x6 << LIMB_BITS / 2 - 8);
    let l3 = limbs.load(3);
    assert_eq!(l3, 0x9 << LIMB_BITS - 8 | 0x8 << LIMB_BITS / 2 - 8);
}

#[cfg(test)]
fn test_composite_limbs_buffer_store_with_unaligned_lengths<ST: MPIntMutByteSlice>() {
    use super::limb::LIMB_BITS;

    debug_assert_eq!(ST::SUPPORTS_UNALIGNED_BUFFER_LENGTHS, true);

    let mut buf0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let mut buf1: [u8; 0] = [0; 0];
    let mut buf2: [u8; 2 * LIMB_BYTES + 2] = [0; 2 * LIMB_BYTES + 2];
    let buf0 = ST::from_bytes(&mut buf0).unwrap();
    let buf1 = ST::from_bytes(&mut buf1).unwrap();
    let buf2 = ST::from_bytes(&mut buf2).unwrap();
    let mut limbs =  CompositeLimbsBuffer::new(
        [buf0, buf1, buf2]
    );

    let l0 = 0x2 << LIMB_BITS - 8 | 0x1 << LIMB_BITS / 2 - 8;
    let l1 = 0x0504 << LIMB_BITS - 16 | 0x3 << LIMB_BITS / 2 - 8;
    let l2 = 0x7 << LIMB_BITS - 8 | 0x6 << LIMB_BITS / 2 - 8;
    let l3 = 0x9 << LIMB_BITS - 8 | 0x8 << LIMB_BITS / 2 - 8;
    let l4  = 0xa;

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
fn test_composite_limbs_buffer_store_with_aligned_lengths<ST: MPIntMutByteSlice>() {
    use super::limb::LIMB_BITS;

    let mut buf0: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let mut buf1: [u8; 0] = [0; 0];
    let mut buf2: [u8; 2 * LIMB_BYTES] = [0; 2 * LIMB_BYTES];
    let buf0 = ST::from_bytes(&mut buf0).unwrap();
    let buf1 = ST::from_bytes(&mut buf1).unwrap();
    let buf2 = ST::from_bytes(&mut buf2).unwrap();
    let mut limbs =  CompositeLimbsBuffer::new(
        [buf0, buf1, buf2]
    );

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
    test_composite_limbs_buffer_store_with_unaligned_lengths::<MPBigEndianMutByteSlice>();
    test_composite_limbs_buffer_store_with_aligned_lengths::<MPBigEndianMutByteSlice>();
}

#[test]
fn test_composite_limbs_buffer_store_le() {
    test_composite_limbs_buffer_store_with_unaligned_lengths::<MPLittleEndianMutByteSlice>();
    test_composite_limbs_buffer_store_with_aligned_lengths::<MPLittleEndianMutByteSlice>();
}

#[test]
fn test_composite_limbs_buffer_store_ne() {
    test_composite_limbs_buffer_store_with_aligned_lengths::<MPNativeEndianMutByteSlice>();
}
