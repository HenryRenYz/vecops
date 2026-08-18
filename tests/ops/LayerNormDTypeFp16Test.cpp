#include "LayerNormTestShared.h"

TEST(LayerNormDTypeTest, CoversAllScaleBiasOutputCombinationsWithFp16Input) {
  run_all_dtype_combos(TypeList<vecops::float16_t>{});
}
