#![macro_use]

macro_rules! tst_mk_mp_backing_vec {
    ($mpt:ty, $len:expr) => {{
        extern crate alloc;
        let mut v = alloc::vec::Vec::<<$mpt>::BackingSliceElementType>::new();
        v.resize(<$mpt>::n_backing_elements_for_len($len), 0u8.into());
        v
    }};
}
