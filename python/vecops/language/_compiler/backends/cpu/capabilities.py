"""CPU C++ type spelling and target representation capabilities."""

CPP_TYPES = {
  "float16": "vecops::float16_t",
  "bfloat16": "vecops::bfloat16_t",
  "float32": "float",
  "float64": "double",
  "bool": "bool",
  **{f"{sign}int{bits}": f"std::{sign}int{bits}_t" for sign in ("", "u") for bits in (8, 16, 32, 64)},
}
