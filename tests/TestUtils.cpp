#include "TestUtils.h"

namespace test_utils {

std::vector<bool> make_pattern(vecops::nint_t size, int mask_bits, int match_val) {
  std::vector<bool> pattern((size_t)size);
  for (vecops::nint_t i = 0; i < size; ++i)
    pattern[(size_t)i] = ((int)i & mask_bits) == match_val;
  return pattern;
}

std::vector<bool> scalar_mask_and(const std::vector<bool>& a, const std::vector<bool>& b) {
  std::vector<bool> r(a.size());
  for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] && b[i];
  return r;
}

std::vector<bool> scalar_mask_or(const std::vector<bool>& a, const std::vector<bool>& b) {
  std::vector<bool> r(a.size());
  for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] || b[i];
  return r;
}

std::vector<bool> scalar_mask_xor(const std::vector<bool>& a, const std::vector<bool>& b) {
  std::vector<bool> r(a.size());
  for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] != b[i];
  return r;
}

std::vector<bool> scalar_mask_andnot(const std::vector<bool>& a, const std::vector<bool>& b) {
  std::vector<bool> r(a.size());
  for (size_t i = 0; i < a.size(); ++i) r[i] = !a[i] && b[i];
  return r;
}

std::vector<bool> scalar_mask_not(const std::vector<bool>& a) {
  std::vector<bool> r(a.size());
  for (size_t i = 0; i < a.size(); ++i) r[i] = !a[i];
  return r;
}

std::vector<bool> scalar_mask_lower(const std::vector<bool>& a) {
  size_t n = a.size() / 2;
  std::vector<bool> r(n);
  for (size_t i = 0; i < n; ++i) r[i] = a[i];
  return r;
}

std::vector<bool> scalar_mask_upper(const std::vector<bool>& a) {
  size_t n = a.size() / 2;
  std::vector<bool> r(n);
  for (size_t i = 0; i < n; ++i) r[i] = a[i + n];
  return r;
}

std::vector<bool> scalar_mask_concat(const std::vector<bool>& lo, const std::vector<bool>& hi) {
  size_t n = lo.size();
  std::vector<bool> r(n * 2);
  for (size_t i = 0; i < n; ++i) {
    r[i] = lo[i];
    r[i + n] = hi[i];
  }
  return r;
}

} // namespace test_utils
