#include "LayerNormTestShared.h"

TEST(LayerNormDTypeTest, CoversAllScaleBiasOutputCombinationsWithFp64Input) {
  run_all_dtype_combos(TypeList<float64_t>{});
}
