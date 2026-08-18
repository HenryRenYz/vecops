#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
#include "LayerNormTestShared.h"

TEST(LayerNormDTypeTest, CoversAllScaleBiasOutputCombinationsWithBf16Input) {
  run_all_dtype_combos(TypeList<vecops::bfloat16_t>{});
}
#endif
