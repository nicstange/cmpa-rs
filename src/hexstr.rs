extern crate alloc;

use alloc::string::String;

#[derive(PartialEq, Eq, Debug)]
pub enum BytesFromHexStrError {
    InvalidHexStrLen,
    InvalidHexChar,
}

pub const fn be_bytes_from_hexstr<const N: usize>(
    hexstr: &str,
) -> Result<[u8; N], BytesFromHexStrError> {
    const fn byte_from_hex(hexstr: &[u8; 2]) -> Result<u8, BytesFromHexStrError> {
        let mut result = 0u8;
        let mut i = 0;
        while i < 2 {
            let hex_char = hexstr[i];
            let nibble = hex_char
                - match hex_char {
                    b'0'..=b'9' => b'0',
                    b'a'..=b'f' => b'a' - 0xa,
                    b'A'..=b'F' => b'A' - 0xa,
                    _ => return Err(BytesFromHexStrError::InvalidHexChar),
                };
            result = result << 4 | nibble;
            i += 1;
        }
        Ok(result)
    }

    let hexstr = hexstr.as_bytes();
    if hexstr.len() > 2 * N {
        return Err(BytesFromHexStrError::InvalidHexStrLen);
    }

    let mut result: [u8; N] = [0; N];
    let result_offset = N - (hexstr.len() + 1) / 2;
    let mut i = hexstr.len();
    while i > 1 {
        i -= 2;
        let hexstr: [u8; 2] = [hexstr[i], hexstr[i + 1]];
        result[result_offset + (i + 1) / 2] = match byte_from_hex(&hexstr) {
            Ok(b) => b,
            Err(e) => return Err(e),
        };
    }
    if i == 1 {
        let hexstr: [u8; 2] = [b'0', hexstr[0]];
        result[result_offset] = match byte_from_hex(&hexstr) {
            Ok(b) => b,
            Err(e) => return Err(e),
        };
    }
    Ok(result)
}

pub const fn bytes_from_hexstr<const N: usize>(
    hexstr: &str,
) -> Result<[u8; N], BytesFromHexStrError> {
    if hexstr.len() != 2 * N {
        return Err(BytesFromHexStrError::InvalidHexStrLen);
    }
    be_bytes_from_hexstr::<N>(hexstr)
}

pub const fn bytes_from_hexstr_cnst<const N: usize>(hexstr: &str) -> [u8; N] {
    // Result::unwrap() is not a const fn :/, so provide a wrapper for
    // use in const contexts.
    match bytes_from_hexstr::<N>(hexstr) {
        Ok(result) => result,
        Err(_) => panic!("invalid hex string"),
    }
}

#[test]
fn test_be_bytes_from_hexstr() {
    assert_eq!(
        be_bytes_from_hexstr::<11>("0123456789abcdefABCDEF").unwrap(),
        [0x01u8, 0x23u8, 0x45u8, 0x67u8, 0x89u8, 0xabu8, 0xcdu8, 0xefu8, 0xabu8, 0xcdu8, 0xefu8],
    );
    assert_eq!(
        be_bytes_from_hexstr::<11>("123456789abcdefABCDEF").unwrap(),
        [0x01u8, 0x23u8, 0x45u8, 0x67u8, 0x89u8, 0xabu8, 0xcdu8, 0xefu8, 0xabu8, 0xcdu8, 0xefu8],
    );
    assert_eq!(
        be_bytes_from_hexstr::<12>("0123456789abcdefABCDEF").unwrap(),
        [
            0x00u8, 0x01u8, 0x23u8, 0x45u8, 0x67u8, 0x89u8, 0xabu8, 0xcdu8, 0xefu8, 0xabu8, 0xcdu8,
            0xefu8
        ],
    );
    assert_eq!(
        be_bytes_from_hexstr::<12>("123456789abcdefABCDEF").unwrap(),
        [
            0x00u8, 0x01u8, 0x23u8, 0x45u8, 0x67u8, 0x89u8, 0xabu8, 0xcdu8, 0xefu8, 0xabu8, 0xcdu8,
            0xefu8
        ],
    );
    assert_eq!(
        be_bytes_from_hexstr::<13>("0123456789abcdefABCDEF").unwrap(),
        [
            0x00u8, 0x00u8, 0x01u8, 0x23u8, 0x45u8, 0x67u8, 0x89u8, 0xabu8, 0xcdu8, 0xefu8, 0xabu8,
            0xcdu8, 0xefu8
        ],
    );
    assert_eq!(
        be_bytes_from_hexstr::<13>("123456789abcdefABCDEF").unwrap(),
        [
            0x00u8, 0x00u8, 0x01u8, 0x23u8, 0x45u8, 0x67u8, 0x89u8, 0xabu8, 0xcdu8, 0xefu8, 0xabu8,
            0xcdu8, 0xefu8
        ],
    );
}

#[test]
fn test_bytes_from_hexstr() {
    assert_eq!(
        bytes_from_hexstr::<11>("0123456789abcdefABCDEF").unwrap(),
        [0x01u8, 0x23u8, 0x45u8, 0x67u8, 0x89u8, 0xabu8, 0xcdu8, 0xefu8, 0xabu8, 0xcdu8, 0xefu8],
    );
    assert_eq!(
        bytes_from_hexstr::<11>("123456789abcdefABCDEF"),
        Err(BytesFromHexStrError::InvalidHexStrLen)
    );
}

pub fn bytes_to_hexstr(bytes: &[u8]) -> Result<String, alloc::collections::TryReserveError> {
    fn nibble_to_hexchar(nibble: u8) -> char {
        let c = match nibble {
            0x0..=0x9 => b'0' + nibble,
            0xa..=0xf => b'a' + (nibble - 0xa),
            _ => unreachable!(),
        };
        c as char
    }

    let mut result = String::new();
    result.try_reserve_exact(2 * bytes.len())?;
    for b in bytes {
        result.push(nibble_to_hexchar(b >> 4));
        result.push(nibble_to_hexchar(b & 0xf))
    }
    Ok(result)
}

#[test]
fn test_bytes_to_hexstr() {
    assert_eq!(
        bytes_to_hexstr(&[0x01u8, 0x23u8, 0x45u8, 0x67u8, 0x89u8, 0xabu8, 0xcdu8, 0xefu8,])
            .unwrap(),
        "0123456789abcdef",
    );
}
