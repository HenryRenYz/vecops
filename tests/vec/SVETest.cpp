//
// SVETest.cpp
// Test file for SVE type infrastructure (SVE_Types.h)
//
// NOTE: This test requires ARM platform with SVE enabled to compile.
// It verifies compile-time type properties and does not require SVE hardware.
//

#include <gtest/gtest.h>
#include <type_traits>
#include "vecops/vec/impl/SVE_Types.h"

using namespace vecops;
using namespace vecops::vec;

// ============================================================================
// Tag Tests -- ScalableTag on SVE
// ============================================================================

TEST(SVETest, ScalableTagBasic) {
  ScalableTag<float32_t> t;

  EXPECT_TRUE((std::is_same_v<decltype(t)::Type, float32_t>));
  EXPECT_EQ(t.N, -1);
  EXPECT_EQ(t.POW2, 0);
  EXPECT_TRUE(t.is_runtime_size);
  EXPECT_EQ(t.Bytes, -1);
}

TEST(SVETest, ScalableTagWithPOW2) {
  ScalableTag<float32_t, 1> t;

  EXPECT_EQ(t.N, -1);
  EXPECT_EQ(t.POW2, 1);
  EXPECT_EQ(t.AdjustedN, -1);
  EXPECT_TRUE(t.is_runtime_size);
}

TEST(SVETest, ScalableTagAllTypes) {
  EXPECT_TRUE((std::is_same_v<ScalableTag<float32_t>, Tag<float32_t, -1, 0>>));
  EXPECT_TRUE((std::is_same_v<ScalableTag<float64_t>, Tag<float64_t, -1, 0>>));
  EXPECT_TRUE((std::is_same_v<ScalableTag<int32_t>,  Tag<int32_t,  -1, 0>>));
  EXPECT_TRUE((std::is_same_v<ScalableTag<int8_t>,   Tag<int8_t,   -1, 0>>));
}

// ============================================================================
// VecDefs: Single-Word SVE  (POW2 = 0)
// ============================================================================

TEST(SVETest, VecDefsSingleWord) {
  using Defs = VecDefs<float32_t, -1, 0>;

  EXPECT_EQ(Defs::num_words, 1);
  EXPECT_TRUE(Defs::is_scalable);
  EXPECT_FALSE(Defs::is_default_impl);
  EXPECT_TRUE(Defs::is_word_vec);

  EXPECT_GT(Defs::max_word_size, 0);
  EXPECT_GT(Defs::max_size, 0);
}

TEST(SVETest, VecDefsSingleWordVecType) {
  using Defs = VecDefs<float32_t, -1, 0>;

  EXPECT_TRUE((std::is_same_v<typename Defs::VecType, svfloat32_t>));
  EXPECT_TRUE((std::is_same_v<typename Defs::MaskType, svbool_t>));
}

TEST(SVETest, VecDefsSingleWordAllTypes) {
  EXPECT_TRUE((std::is_same_v<typename VecDefs<float32_t, -1, 0>::VecType, svfloat32_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<float64_t, -1, 0>::VecType, svfloat64_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<int32_t,  -1, 0>::VecType, svint32_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<uint8_t,  -1, 0>::VecType, svuint8_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<vecops::bfloat16_t, -1, 0>::VecType, svvecops::bfloat16_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<vecops::float16_t, -1, 0>::VecType, svfloat16_t>));
}

// ============================================================================
// VecDefs: 2-Word SVE  (POW2 = 1)
// ============================================================================

TEST(SVETest, VecDefsTwoWord) {
  using Defs = VecDefs<float32_t, -1, 1>;

  EXPECT_EQ(Defs::num_words, 2);
  EXPECT_TRUE(Defs::is_scalable);
  EXPECT_FALSE(Defs::is_default_impl);
  EXPECT_FALSE(Defs::is_word_vec);

  EXPECT_GT(Defs::max_word_size, 0);
  EXPECT_GT(Defs::max_size, Defs::max_word_size);
}

TEST(SVETest, VecDefsTwoWordVecType) {
  using Defs = VecDefs<float32_t, -1, 1>;

  EXPECT_TRUE((std::is_same_v<typename Defs::VecType, svfloat32x2_t>));
  EXPECT_TRUE((std::is_same_v<typename Defs::MaskType, svboolx2_t>));
}

TEST(SVETest, VecDefsTwoWordAllTypes) {
  EXPECT_TRUE((std::is_same_v<typename VecDefs<float32_t, -1, 1>::VecType, svfloat32x2_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<float64_t, -1, 1>::VecType, svfloat64x2_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<int32_t,  -1, 1>::VecType, svint32x2_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<uint8_t,  -1, 1>::VecType, svuint8x2_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<vecops::bfloat16_t, -1, 1>::VecType, svbfloat16x2_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<vecops::float16_t, -1, 1>::VecType, svfloat16x2_t>));
}

// ============================================================================
// VecDefs: 4-Word SVE  (POW2 = 2)
// ============================================================================

TEST(SVETest, VecDefsFourWord) {
  using Defs = VecDefs<float32_t, -1, 2>;

  EXPECT_EQ(Defs::num_words, 4);
  EXPECT_TRUE(Defs::is_scalable);
  EXPECT_FALSE(Defs::is_default_impl);
  EXPECT_FALSE(Defs::is_word_vec);
}

TEST(SVETest, VecDefsFourWordVecType) {
  using Defs = VecDefs<float32_t, -1, 2>;

  EXPECT_TRUE((std::is_same_v<typename Defs::VecType, svfloat32x4_t>));
  EXPECT_TRUE((std::is_same_v<typename Defs::MaskType, svboolx4_t>));
}

TEST(SVETest, VecDefsFourWordAllTypes) {
  EXPECT_TRUE((std::is_same_v<typename VecDefs<float32_t, -1, 2>::VecType, svfloat32x4_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<float64_t, -1, 2>::VecType, svfloat64x4_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<int32_t,  -1, 2>::VecType, svint32x4_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<uint8_t,  -1, 2>::VecType, svuint8x4_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<vecops::bfloat16_t, -1, 2>::VecType, svbfloat16x4_t>));
  EXPECT_TRUE((std::is_same_v<typename VecDefs<vecops::float16_t, -1, 2>::VecType, svfloat16x4_t>));
}

// ============================================================================
// Vec / Mask Type Alias Tests
// ============================================================================

TEST(SVETest, VecTypeAliasSingleWord) {
  using T = ScalableTag<float32_t>;
  using V = Vec<T>;

  EXPECT_TRUE((std::is_same_v<V, svfloat32_t>));
}

TEST(SVETest, VecTypeAliasTwoWord) {
  using T = ScalableTag<float32_t, 1>;
  using V = Vec<T>;

  EXPECT_TRUE((std::is_same_v<V, svfloat32x2_t>));
}

TEST(SVETest, VecTypeAliasFourWord) {
  using T = ScalableTag<float32_t, 2>;
  using V = Vec<T>;

  EXPECT_TRUE((std::is_same_v<V, svfloat32x4_t>));
}

TEST(SVETest, MaskTypeAliasSingleWord) {
  using T = ScalableTag<float32_t>;
  using M = Mask<T>;

  EXPECT_TRUE((std::is_same_v<M, svbool_t>));
}

TEST(SVETest, MaskTypeAliasTwoWord) {
  using T = ScalableTag<float32_t, 1>;
  using M = Mask<T>;

  EXPECT_TRUE((std::is_same_v<M, svboolx2_t>));
}

TEST(SVETest, MaskTypeAliasFourWord) {
  using T = ScalableTag<float32_t, 2>;
  using M = Mask<T>;

  EXPECT_TRUE((std::is_same_v<M, svboolx4_t>));
}

TEST(SVETest, VecOfMacro) {
  ScalableTag<float32_t> t;
  EXPECT_TRUE((std::is_same_v<VecOf(t), svfloat32_t>));
}

TEST(SVETest, MaskOfMacro) {
  ScalableTag<float32_t> t;
  EXPECT_TRUE((std::is_same_v<MaskOf(t), svbool_t>));
}

// ============================================================================
// Vec2Tag -- reverse deduction from SVE type to Tag
// ============================================================================

TEST(SVETest, Vec2TagSingleWord) {
  using TagT = Vec2Tag<svfloat32_t>;

  EXPECT_TRUE((std::is_same_v<TagT, Tag<float32_t, -1, 0>>));
  EXPECT_TRUE((std::is_same_v<typename TagT::Type, float32_t>));
  EXPECT_EQ(TagT::N, -1);
  EXPECT_EQ(TagT::POW2, 0);
}

TEST(SVETest, Vec2TagTwoWord) {
  using TagT = Vec2Tag<svfloat64x2_t>;

  EXPECT_TRUE((std::is_same_v<TagT, Tag<float64_t, -1, 1>>));
  EXPECT_EQ(TagT::POW2, 1);
}

TEST(SVETest, Vec2TagFourWord) {
  using TagT = Vec2Tag<svint32x4_t>;

  EXPECT_TRUE((std::is_same_v<TagT, Tag<int32_t, -1, 2>>));
  EXPECT_EQ(TagT::POW2, 2);
}

TEST(SVETest, Vec2TagAllTypes) {
  EXPECT_TRUE((std::is_same_v<Vec2Tag<svfloat32_t>,    Tag<float32_t, -1, 0>>));
  EXPECT_TRUE((std::is_same_v<Vec2Tag<svfloat64_t>,    Tag<float64_t, -1, 0>>));
  EXPECT_TRUE((std::is_same_v<Vec2Tag<svint32_t>,      Tag<int32_t,  -1, 0>>));
  EXPECT_TRUE((std::is_same_v<Vec2Tag<svuint32_t>,     Tag<uint32_t, -1, 0>>));
  EXPECT_TRUE((std::is_same_v<Vec2Tag<svint16_t>,      Tag<int16_t,  -1, 0>>));
  EXPECT_TRUE((std::is_same_v<Vec2Tag<svuint16_t>,     Tag<uint16_t, -1, 0>>));
  EXPECT_TRUE((std::is_same_v<Vec2Tag<svint8_t>,       Tag<int8_t,   -1, 0>>));
  EXPECT_TRUE((std::is_same_v<Vec2Tag<svuint8_t>,      Tag<uint8_t,  -1, 0>>));
  EXPECT_TRUE((std::is_same_v<Vec2Tag<svvecops::bfloat16_t>,   Tag<vecops::bfloat16_t,-1, 0>>));
  EXPECT_TRUE((std::is_same_v<Vec2Tag<svfloat16_t>,    Tag<vecops::float16_t,-1, 0>>));
  EXPECT_TRUE((std::is_same_v<Vec2Tag<svint64_t>,      Tag<int64_t,  -1, 0>>));
  EXPECT_TRUE((std::is_same_v<Vec2Tag<svuint64_t>,     Tag<uint64_t, -1, 0>>));
}

// ============================================================================
// is_vec / is_mask
// ============================================================================

TEST(SVETest, IsVecSVE) {
  EXPECT_TRUE((is_vec<svfloat32_t>));
  EXPECT_TRUE((is_vec<svfloat64_t>));
  EXPECT_TRUE((is_vec<svint32_t>));
  EXPECT_TRUE((is_vec<svfloat32x2_t>));
  EXPECT_TRUE((is_vec<svfloat32x4_t>));
}

TEST(SVETest, IsMaskSVE) {
  EXPECT_TRUE((is_mask<svbool_t>));
  EXPECT_TRUE((is_mask<svboolx2_t>));
  EXPECT_TRUE((is_mask<svboolx4_t>));
}

TEST(SVETest, SVEVecIsNotMask) {
  EXPECT_FALSE((is_mask<svfloat32_t>));
  EXPECT_FALSE((is_mask<svfloat32x2_t>));
}

TEST(SVETest, SVEMaskIsNotVec) {
  EXPECT_FALSE((is_vec<svbool_t>));
  EXPECT_FALSE((is_vec<svboolx2_t>));
}

// ============================================================================
// WordDefs / word_tag
// ============================================================================

TEST(SVETest, WordDefsPOW2_0) {
  using Defs = VecDefs<float32_t, -1, 0>;

  EXPECT_TRUE((std::is_same_v<typename Defs::WordDefs, VecDefs<float32_t, -1, 0>>));
}

TEST(SVETest, WordDefsPOW2_1) {
  using Defs = VecDefs<float32_t, -1, 1>;

  EXPECT_TRUE((std::is_same_v<typename Defs::WordDefs, VecDefs<float32_t, -1, 0>>));
  EXPECT_TRUE((std::is_same_v<typename Defs::WordDefs::VecType, svfloat32_t>));
  EXPECT_TRUE((std::is_same_v<typename Defs::WordDefs::MaskType, svbool_t>));
}

TEST(SVETest, WordDefsPOW2_2) {
  using Defs = VecDefs<float32_t, -1, 2>;

  EXPECT_TRUE((std::is_same_v<typename Defs::WordDefs, VecDefs<float32_t, -1, 0>>));
}

TEST(SVETest, WordTagSingleWord) {
  ScalableTag<float32_t> t;

  auto wt = word_tag(t);

  EXPECT_TRUE((std::is_same_v<decltype(wt), Tag<float32_t, -1, 0>>));
  EXPECT_EQ(wt.POW2, 0);
}

TEST(SVETest, WordTagMultiWord) {
  ScalableTag<float32_t, 1> t;

  auto wt = word_tag(t);

  EXPECT_TRUE((std::is_same_v<decltype(wt), Tag<float32_t, -1, 0>>));
  EXPECT_EQ(wt.POW2, 0);
  EXPECT_TRUE((std::is_same_v<decltype(wt)::Type, float32_t>));
}

// ============================================================================
// WordOf
// ============================================================================

TEST(SVETest, WordOfSingleWord) {
  using T = ScalableTag<float32_t>;
  using W = WordOf<T>;

  EXPECT_TRUE((std::is_same_v<W, Tag<float32_t, -1, 0>>));
}

TEST(SVETest, WordOfMultiWord) {
  using T = ScalableTag<float32_t, 1>;
  using W = WordOf<T>;

  EXPECT_TRUE((std::is_same_v<W, Tag<float32_t, -1, 0>>));
  EXPECT_EQ(W::POW2, 0);
}

// ============================================================================
// Half / Twice
// ============================================================================

TEST(SVETest, Half) {
  using T = ScalableTag<float32_t, 2>;

  EXPECT_TRUE((std::is_same_v<Half<T>, Tag<float32_t, -1, 1>>));
}

TEST(SVETest, Twice) {
  using T = ScalableTag<float32_t, 1>;

  EXPECT_TRUE((std::is_same_v<Twice<T>, Tag<float32_t, -1, 2>>));
}

// ============================================================================
// Rebind / ViewAs
// ============================================================================

TEST(SVETest, RebindScalable) {
  using T = ScalableTag<float32_t>;
  using R = Rebind<float64_t, T>;

  EXPECT_TRUE((std::is_same_v<R, Tag<float64_t, -1, 1>>));
}

TEST(SVETest, ViewAsScalable) {
  using T = ScalableTag<float32_t>;
  using V = ViewAs<float64_t, T>;

  EXPECT_TRUE((std::is_same_v<V, Tag<float64_t, -1, 0>>));
}

// ============================================================================
// Index Type Tests  (repeated from BaseTest for completeness)
// ============================================================================

TEST(SVETest, IndexTypeFloat32) {
  EXPECT_TRUE((std::is_same_v<Index<float32_t>, int32_t>));
}

TEST(SVETest, IndexTypeFloat64) {
  EXPECT_TRUE((std::is_same_v<Index<float64_t>, int64_t>));
}

// ============================================================================
// Helper Functions
// ============================================================================

TEST(SVETest, NumWordsScalable) {
  ScalableTag<float32_t> t;
  EXPECT_EQ(num_words(t), 1);
}

TEST(SVETest, NumWordsScalableMulti) {
  ScalableTag<float32_t, 1> t;
  EXPECT_EQ(num_words(t), 2);
}

TEST(SVETest, IsScalableTrue) {
  ScalableTag<float32_t> t;
  EXPECT_TRUE(is_scalable(t));
}

TEST(SVETest, IsDefaultImplFalse) {
  ScalableTag<float32_t> t;
  EXPECT_FALSE(is_default_impl(t));
}

TEST(SVETest, IsWordVec) {
  ScalableTag<float32_t> t;
  EXPECT_TRUE(is_word_vec(t));
}

TEST(SVETest, IsWordVecFalseForMulti) {
  ScalableTag<float32_t, 1> t;
  EXPECT_FALSE(is_word_vec(t));
}

// ============================================================================
// SVE MultiRegType
// ============================================================================

TEST(SVETest, MultiRegTypePOW2_0) {
  EXPECT_TRUE((std::is_same_v<typename SVE::MultiRegType<float32_t, 0>::Type, svfloat32_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::MultiRegType<int32_t, 0>::Type, svint32_t>));
}

TEST(SVETest, MultiRegTypePOW2_1) {
  EXPECT_TRUE((std::is_same_v<typename SVE::MultiRegType<float32_t, 1>::Type, svfloat32x2_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::MultiRegType<float64_t, 1>::Type, svfloat64x2_t>));
}

TEST(SVETest, MultiRegTypePOW2_2) {
  EXPECT_TRUE((std::is_same_v<typename SVE::MultiRegType<float32_t, 2>::Type, svfloat32x4_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::MultiRegType<int16_t, 2>::Type, svint16x4_t>));
}

// ============================================================================
// SVE MultiPredType
// ============================================================================

TEST(SVETest, MultiPredType) {
  EXPECT_TRUE((std::is_same_v<typename SVE::MultiPredType<0>::Type, svbool_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::MultiPredType<1>::Type, svboolx2_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::MultiPredType<2>::Type, svboolx4_t>));
}

// ============================================================================
// SVE RegType
// ============================================================================

TEST(SVETest, RegTypeAll12Types) {
  EXPECT_TRUE((std::is_same_v<typename SVE::RegType<vecops::bfloat16_t>::Type, svvecops::bfloat16_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::RegType<vecops::float16_t>::Type,   svfloat16_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::RegType<float32_t>::Type,   svfloat32_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::RegType<float64_t>::Type,   svfloat64_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::RegType<int8_t>::Type,      svint8_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::RegType<uint8_t>::Type,     svuint8_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::RegType<int16_t>::Type,     svint16_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::RegType<uint16_t>::Type,    svuint16_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::RegType<int32_t>::Type,     svint32_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::RegType<uint32_t>::Type,    svuint32_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::RegType<int64_t>::Type,     svint64_t>));
  EXPECT_TRUE((std::is_same_v<typename SVE::RegType<uint64_t>::Type,    svuint64_t>));
}

// ============================================================================
// Constexpr Tests
// ============================================================================

TEST(SVETest, ConstexprNumWords) {
  constexpr auto n0 = num_words(ScalableTag<float32_t>{});
  constexpr auto n1 = num_words(ScalableTag<float32_t, 1>{});
  constexpr auto n2 = num_words(ScalableTag<float32_t, 2>{});

  EXPECT_EQ(n0, 1);
  EXPECT_EQ(n1, 2);
  EXPECT_EQ(n2, 4);
}

TEST(SVETest, ConstexprIsScalable) {
  constexpr auto sc = is_scalable(ScalableTag<float32_t>{});

  EXPECT_TRUE(sc);
}

TEST(SVETest, ConstexprIsDefaultImpl) {
  constexpr auto di = is_default_impl(ScalableTag<float32_t>{});

  EXPECT_FALSE(di);
}

TEST(SVETest, ConstexprIsWordVec) {
  constexpr auto wv0 = is_word_vec(ScalableTag<float32_t>{});
  constexpr auto wv1 = is_word_vec(ScalableTag<float32_t, 1>{});

  EXPECT_TRUE(wv0);
  EXPECT_FALSE(wv1);
}

TEST(SVETest, ConstexprMaxWordSize) {
  constexpr auto mws = max_word_size(ScalableTag<float32_t>{});

  EXPECT_GT(mws, 0);
}

// ============================================================================
// Mask multi-word get_mask / set_mask type verification
// ============================================================================

TEST(SVETest, TwoWordMaskWordDefs) {
  using Defs = VecDefs<float32_t, -1, 1>;

  EXPECT_TRUE((std::is_same_v<typename Defs::WordDefs::MaskType, svbool_t>));
}

TEST(SVETest, FourWordMaskWordDefs) {
  using Defs = VecDefs<float32_t, -1, 2>;

  EXPECT_TRUE((std::is_same_v<typename Defs::WordDefs::MaskType, svbool_t>));
}

// ============================================================================
// POW2 < 0  --  fractional single-register SVE vectors
// ============================================================================

TEST(SVETest, ScalableTagPOW2Negative) {
  ScalableTag<float32_t, -1> t;

  EXPECT_EQ(t.N, -1);
  EXPECT_EQ(t.POW2, -1);
  EXPECT_TRUE(t.is_runtime_size);
  EXPECT_EQ(t.Bytes, -1);
}

TEST(SVETest, VecDefsFractionalSingleWord) {
  using Defs = VecDefs<float32_t, -1, -1>;

  EXPECT_EQ(Defs::num_words, 1);
  EXPECT_TRUE(Defs::is_scalable);
  EXPECT_FALSE(Defs::is_default_impl);
  EXPECT_TRUE(Defs::is_word_vec);

  EXPECT_GT(Defs::max_word_size, 0);
  EXPECT_GT(Defs::max_size, 0);
  EXPECT_LE(Defs::max_size, Defs::max_word_size);
}

TEST(SVETest, VecDefsFractionalVecType) {
  using Defs = VecDefs<float32_t, -1, -1>;

  EXPECT_TRUE((std::is_same_v<typename Defs::VecType, svfloat32_t>));
  EXPECT_TRUE((std::is_same_v<typename Defs::MaskType, svbool_t>));
}

TEST(SVETest, VecDefsFractionalWordDefs) {
  using Defs = VecDefs<float32_t, -1, -1>;

  EXPECT_TRUE((std::is_same_v<typename Defs::WordDefs, VecDefs<float32_t, -1, 0>>));
  EXPECT_TRUE((std::is_same_v<typename Defs::WordDefs::VecType, svfloat32_t>));
  EXPECT_TRUE((std::is_same_v<typename Defs::WordDefs::MaskType, svbool_t>));
  EXPECT_EQ(Defs::WordDefs::num_words, 1);
}

TEST(SVETest, VecDefsFractionalPOW2Negative2) {
  using Defs = VecDefs<float32_t, -1, -2>;

  EXPECT_EQ(Defs::num_words, 1);
  EXPECT_TRUE(Defs::is_word_vec);
  EXPECT_GT(Defs::max_size, 0);
}

TEST(SVETest, VecDefsFractionalPOW2Negative3) {
  using Defs = VecDefs<float32_t, -1, -3>;

  EXPECT_EQ(Defs::num_words, 1);
  EXPECT_TRUE(Defs::is_word_vec);
  EXPECT_GT(Defs::max_size, 0);
}

TEST(SVETest, VecTypeAliasFractional) {
  using T = ScalableTag<float32_t, -1>;
  using V = Vec<T>;

  EXPECT_TRUE((std::is_same_v<V, svfloat32_t>));
}

TEST(SVETest, MaskTypeAliasFractional) {
  using T = ScalableTag<float32_t, -1>;
  using M = Mask<T>;

  EXPECT_TRUE((std::is_same_v<M, svbool_t>));
}

TEST(SVETest, VecOfMacroFractional) {
  ScalableTag<float32_t, -1> t;
  EXPECT_TRUE((std::is_same_v<VecOf(t), svfloat32_t>));
}

TEST(SVETest, MaskOfMacroFractional) {
  ScalableTag<float32_t, -1> t;
  EXPECT_TRUE((std::is_same_v<MaskOf(t), svbool_t>));
}

TEST(SVETest, HalfFromPOW2_0) {
  using T = ScalableTag<float32_t>;

  EXPECT_TRUE((std::is_same_v<Half<T>, Tag<float32_t, -1, -1>>));
}

TEST(SVETest, TwiceFromPOW2_Neg1) {
  using T = ScalableTag<float32_t, -1>;

  EXPECT_TRUE((std::is_same_v<Twice<T>, Tag<float32_t, -1, 0>>));
}

TEST(SVETest, HalfTwiceRoundTrip) {
  using T = ScalableTag<float32_t>;

  EXPECT_TRUE((std::is_same_v<Twice<Half<T>>, T>));
  EXPECT_TRUE((std::is_same_v<Half<Twice<T>>, T>));
}

TEST(SVETest, HalfFromPOW2_1) {
  using T = ScalableTag<float32_t, 1>;

  EXPECT_TRUE((std::is_same_v<Half<T>, Tag<float32_t, -1, 0>>));
}

TEST(SVETest, WordTagFractional) {
  ScalableTag<float32_t, -1> t;

  auto wt = word_tag(t);

  EXPECT_TRUE((std::is_same_v<decltype(wt), Tag<float32_t, -1, 0>>));
  EXPECT_EQ(wt.POW2, 0);
}

TEST(SVETest, WordOfFractional) {
  using T = ScalableTag<float32_t, -1>;
  using W = WordOf<T>;

  EXPECT_TRUE((std::is_same_v<W, Tag<float32_t, -1, 0>>));
  EXPECT_EQ(W::POW2, 0);
}

TEST(SVETest, NumWordsFractional) {
  ScalableTag<float32_t, -1> t;
  EXPECT_EQ(num_words(t), 1);
}

TEST(SVETest, IsWordVecFractional) {
  ScalableTag<float32_t, -1> t;
  EXPECT_TRUE(is_word_vec(t));
}

TEST(SVETest, IsScalableFractional) {
  ScalableTag<float32_t, -1> t;
  EXPECT_TRUE(is_scalable(t));
}

TEST(SVETest, IsDefaultImplFractional) {
  ScalableTag<float32_t, -1> t;
  EXPECT_FALSE(is_default_impl(t));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
