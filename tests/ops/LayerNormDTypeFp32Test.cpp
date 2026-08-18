#include "LayerNormTestShared.h"

TEST(LayerNormDTypeTest, CoversAllScaleBiasOutputCombinationsWithFp32Input) {
  run_all_dtype_combos(TypeList<float32_t>{});
}
