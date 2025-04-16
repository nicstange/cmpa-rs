# *C*ryptographic *M*ulti*P*recision *A*rithmetic (cmpa)

Pure Rust implementation of multiprecision arithmetic primitives
commonly needed for asymmetric cryptography.

`rustdoc` comments are WIP.

## Features
* `[no_std]`
* Rigorously constant-time throughout.
* Complete user control over buffers: cmpa does not allocate any own
  its own or places any on the stack. Whenever scratch space is needed
  for some computation, a buffer is provided as an additional argument
  to the API.

### Functionality overview
* Operates on multiprecision integers in big or little endian format
  (stored as byte slices) or more effeciently on native format
  multiprecision integers (stored as a slice of machine words).
* "Standard" airhtmetic: addition, subtraction, multiplication,
  division, shifts, comparisons etc.
* Modular arithmetic: Montogomery multiplication and related functionality,
  modular inversions etc.
* Primality testing.

## Cargo features
* `zeroize` - implement traits from the
  [`zeroize`](https://github.com/RustCrypto/utils/tree/master/zeroize)
  crate.
* `enable_arch_math_asm` - Use (rather trivial) assembly
   implementations for certain basic sinlge-word arithmetic
   primitives. You most likely want to enable it for performance
   reason, but it does introduce some `unsafe{}` blocks.
