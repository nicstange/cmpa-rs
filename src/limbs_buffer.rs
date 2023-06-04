//! Accessors for multiprecision integers as stored in big-endian byte buffers.
//!
//! Multiprecision integers are stored in bytes buffers in big-endian order. The arithmetic
//! primitives all operate on those in units of [`LimbType`] for efficiency reasons.
//! This helper module provides a couple of utilities for accessing these byte buffers in units of
//! [`LimbType`].

#[cfg(feature = "zeroize")]
use zeroize::Zeroize as _;

use super::limb::{LimbType, LIMB_BITS, LIMB_BYTES};
use super::zeroize::Zeroizing;

/// Determine the number of [`LimbType`] limbs stored in a multiprecision integer big-endian byte
/// buffer.
///
/// # Arguments
///
/// * `len` - The multiprecision integer's underlying big-endian byte buffer's length in bytes.
///
pub fn mp_ct_nlimbs(len: usize) -> usize {
    (len + LIMB_BYTES - 1) / LIMB_BYTES
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
    #[cfg(feature = "zeroize")]
    src.zeroize();
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
pub fn mp_be_load_l_full(limbs: &[u8], i: usize) -> LimbType {
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
pub fn mp_be_load_l(limbs: &[u8], i: usize) -> LimbType {
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
    let src: Zeroizing<[u8; LIMB_BYTES]> = value.to_be_bytes().into();
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
pub fn mp_be_store_l_full(limbs: &mut [u8], i: usize, value: LimbType) {
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
pub fn mp_be_store_l(limbs: &mut [u8], i: usize, value: LimbType) {
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


/// Internal data structure describing a single one of a [`CompositeLimbsBuffer`]'s constituting
/// segments.
///
struct CompositeLimbsBufferSegment<'a> {
    /// The total number of limbs whose least significant bytes are held in this or less significant
    /// segments each (note that a limb may span multiple segments).
    end: usize,
    /// The actual data, interpreted as a segment within a composed multiprecision integer
    /// big-endian byte buffer.
    ///
    /// The `segment` slice, if non-empty, will be made to align with the least significant limb
    /// whose least significant byte is found in this segment. That is, the slice's last byte
    /// (in memory order) corresponds to the least significant byte of the least significant limb.
    ///
    segment: &'a mut [u8],
    /// Continuation bytes of this or preceeding, less significant segment's most significant limb.
    ///
    /// A segment's most significant limb might be covered only partially by it and extend across
    /// one or more subsequent, more significant segments. In this case its remaining, more
    /// significant continutation bytes are collected from this and the subsequent, more significant
    /// segments' `high_next_partial` slices.
    high_next_partial: &'a mut [u8],
}

/// Access multiprecision integers composed of multiple virtually concatenated big-endian byte
/// buffers in units of [`LimbType`].
///
/// A [`CompositeLimbsBuffer`] provides a composed multiprecision integer big-endian byte buffer
/// view on `N_SEGMENTS` virtually concatenated big-endian byte slices. Primitives are provided
/// alongside for accessing the composed integer in units of [`LimbType`].
///
/// Certain applications need to truncate the result of some multiprecision integer arithmetic
/// operation result and dismiss the rest. An example would be key generation modulo some large
/// integer by the method of oversampling. A `CompositeLimbsBuffer` helps such applications to
/// reduce the memory footprint of the truncation operation: instead of allocating a smaller
/// destination buffer and copying the to be retained parts over from the result, the arithmetic
/// primitive can, if supported, operate directly on a multiprecision integer big-endian byte buffer
/// composed of several independent smaller slices by virtual concatenation. Assuming the individual
/// segments' slice lengths align properly with the needed and unneeded parts of the result, the
/// latter ones can get truncated away trivially when done by simply dismissing the corresponding
/// underlying buffers.
///
pub struct  CompositeLimbsBuffer<'a, const N_SEGMENTS: usize> {
    /// The composed view's individual segments, ordered from least to most significant.
    segments: [CompositeLimbsBufferSegment<'a>; N_SEGMENTS]
}

impl<'a, const N_SEGMENTS: usize>  CompositeLimbsBuffer<'a, N_SEGMENTS> {
    /// Construct a `CompositeLimbsBuffer` view from the individual big-endian byte buffer segments.
    ///
    /// # Arguments
    ///
    /// * `segments` - An array of `N_SEGMENTS` byte slices to compose the multiprecision integer
    ///                big-endian byte buffer view from by virtual concatenation. Ordered from least
    ///                to most significant relative with respect to their position within the
    ///                resulting view.
    ///
    pub fn new(segments: [&'a mut [u8]; N_SEGMENTS]) -> Self {
        let mut segments = <[&'a mut [u8]; N_SEGMENTS] as IntoIterator>::into_iter(segments);
        let mut segments: [Option<&'a mut [u8]>; N_SEGMENTS]
            = core::array::from_fn(|_| segments.next());
        let mut n_bytes_total = 0;
        let mut create_segment = |i: usize| {
            let segment = segments[i].take().unwrap();
            n_bytes_total += segment.len();

            let n_high_partial = n_bytes_total % LIMB_BYTES;
            let (segment, high_next_partial) = if i + 1 != segments.len() && n_high_partial != 0 {
                let next_segment = segments[i + 1].take().unwrap();
                let n_from_next = (LIMB_BYTES - n_high_partial).min(next_segment.len());
                let (next_segment, high_next_partial) = next_segment.split_at_mut(next_segment.len() - n_from_next);
                segments[i + 1] = Some(next_segment);
                (segment, high_next_partial)
            } else {
                let (high_next_partial, segment) = segment.split_at_mut(0);
                (segment, high_next_partial)
            };

            n_bytes_total += high_next_partial.len();
            let end = mp_ct_nlimbs(n_bytes_total);
            CompositeLimbsBufferSegment { end, segment, high_next_partial }
        };

        let segments: [CompositeLimbsBufferSegment; N_SEGMENTS] = core::array::from_fn(&mut create_segment);
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

    /// Load a limb from the composed multiprecision integer big-endian byte buffer.
    ///
    /// Execution time depends on the composed multiprecision integer's underlying segment layout as
    /// well as as on on the limb index argument `i`, and is otherwise constant as far as branching
    /// is concerned.
    ///
    /// * `i` - The index of the limb to load, counted from least to most significant.
    ///
    pub fn load(&self, i: usize) -> LimbType {
        let (segment_index, segment_offset) = self.limb_index_to_segment(i);
        let segment = &self.segments[segment_index];
        if i != segment.end - 1 {
            mp_be_load_l_full(segment.segment, i - segment_offset)
        } else if segment_index + 1 == N_SEGMENTS || segment.segment.len() % LIMB_BYTES == 0 {
            // The last (highest) segment's most significant bytes don't necessarily occupy a full
            // limb.
            mp_be_load_l(segment.segment, i - segment_offset)
        } else {
            let mut npartial = segment.segment.len() % LIMB_BYTES;
            let mut value = _mp_be_load_l_high_partial(segment.segment, npartial);
            let mut segment_index = segment_index;
            while npartial != LIMB_BYTES && segment_index < self.segments.len() {
                let partial = &self.segments[segment_index].high_next_partial;
                value |= _mp_be_load_l_high_partial(partial, partial.len()) << (npartial * 8);
                npartial += partial.len();
                segment_index += 1;
            }
            value
        }
    }

    /// Update a limb in the composed multiprecision integer big-endian byte buffer.
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
        if i != segment.end - 1 {
            mp_be_store_l_full(segment.segment, i - segment_offset, value);
        } else if segment_index + 1 == N_SEGMENTS || segment.segment.len() % LIMB_BYTES == 0 {
            // The last (highest) part's most significant bytes don't necessarily occupy a full
            // limb.
            mp_be_store_l(segment.segment, i - segment_offset, value)
        } else {
            let mut value = value;
            let mut npartial = segment.segment.len() % LIMB_BYTES;
            _mp_be_store_l_high_partial(segment.segment, npartial, value);
            value >>= npartial * 8;
            let mut segment_index = segment_index;
            while npartial != LIMB_BYTES && segment_index < self.segments.len() {
                let partial = &mut self.segments[segment_index].high_next_partial;
                _mp_be_store_l_high_partial(partial, partial.len(), value);
                value >>= partial.len() * 8;
                npartial += partial.len();
                segment_index += 1;
            }
            debug_assert!(value == 0);
        }
    }
}

#[test]
fn test_composite_limbs_buffer_load() {
    let mut buf0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let mut buf1: [u8; 0] = [0; 0];
    let mut buf2: [u8; 2 * LIMB_BYTES + 2] = [0; 2 * LIMB_BYTES + 2];

    // 0x02 00 .. 00 01 00 .. 00
    buf0[LIMB_BYTES - 1 + LIMB_BYTES / 2] = 0x1;
    buf0[LIMB_BYTES - 1] = 0x2;

    // 0x05 04 00 .. 00 03 .. 00
    buf0[LIMB_BYTES / 2 - 1] = 0x3;
    buf0[0] = 0x4;
    buf2[1 + 2 * LIMB_BYTES] = 0x05;

    // 0x07 00 .. 00 06 00 .. 00
    buf2[1 + LIMB_BYTES + LIMB_BYTES / 2] = 0x6;
    buf2[1 + LIMB_BYTES ] = 0x7;

    // 0x08 00 .. 00 08 00 .. 00
    buf2[1 + LIMB_BYTES / 2] = 0x8;
    buf2[1] = 0x9;

    // 0x00 .. 00 0x0a
    buf2[0] = 0xa;

    let limbs =  CompositeLimbsBuffer::new([buf0.as_mut_slice(), buf1.as_mut_slice(), buf2.as_mut_slice()]);

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
fn test_composite_limbs_buffer_store() {
    let mut buf0: [u8; 2 * LIMB_BYTES - 1] = [0; 2 * LIMB_BYTES - 1];
    let mut buf1: [u8; 0] = [0; 0];
    let mut buf2: [u8; 2 * LIMB_BYTES + 2] = [0; 2 * LIMB_BYTES + 2];
    let mut limbs =  CompositeLimbsBuffer::new([buf0.as_mut_slice(), buf1.as_mut_slice(), buf2.as_mut_slice()]);

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
