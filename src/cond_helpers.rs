use subtle::{self, ConditionallySelectable as _};

use super::limb::LimbType;

pub fn cond_choice_to_mask(choice: subtle::Choice) -> LimbType {
    LimbType::conditional_select(&0, &!0, choice)
}

pub fn cond_select_with_mask(a: LimbType, b: LimbType, cond_mask: LimbType) -> LimbType {
    a ^ ((a ^ b) & cond_mask)
}
