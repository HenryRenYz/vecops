//
// Created by renyz on 2026/6/3.
//

#include <gtest/gtest.h>
#include <sstream>
#include <string>

#include "vecops/gemm/Layout.h"

using namespace vecops;
using namespace vecops::gemm;

// ======================================================================
// Helper: capture operator<< output to std::string
// ======================================================================

template <typename T>
static std::string to_string(const T& v) {
  std::ostringstream oss;
  oss << v;
  return oss.str();
}

// ======================================================================
// Value Suite — Value / Const / Dynamic / Any / ToValue
// ======================================================================

class ValueTest : public ::testing::Test {};

TEST_F(ValueTest, ConstBasicProps) {
  EXPECT_TRUE((Const<128>::is_const));
  EXPECT_FALSE((Const<128>::is_runtime));
  EXPECT_EQ((Const<128>::value), 128);
}

TEST_F(ValueTest, ConstConforms) {
  EXPECT_TRUE((Const<128>::conforms(128)));
  EXPECT_FALSE((Const<128>::conforms(64)));
}

TEST_F(ValueTest, ConstAligns) {
  EXPECT_TRUE((Const<128>::aligns(4)));
  EXPECT_FALSE((Const<128>::aligns(3)));
}

TEST_F(ValueTest, ConstIsAligned) {
  EXPECT_TRUE((Const<128>{}.is_aligned(4)));
  EXPECT_FALSE((Const<128>{}.is_aligned(3)));
}

TEST_F(ValueTest, ConstOperatorNint) {
  EXPECT_EQ(nint_t(Const<128>{}), 128);
}

TEST_F(ValueTest, ConstDefaultCtor) {
  Const<128> c;
  EXPECT_EQ(nint_t(c), 128);
}

TEST_F(ValueTest, ConstExplicitCtorMatch) {
  Const<128> c(128);
  EXPECT_EQ(nint_t(c), 128);
}

TEST_F(ValueTest, DynamicBasicProps) {
  EXPECT_FALSE((Dynamic<16>::is_const));
  EXPECT_TRUE((Dynamic<16>::is_runtime));
  EXPECT_EQ((Dynamic<16>::alignment), 16);
  EXPECT_EQ((Dynamic<16>::lo), kLoInf);
  EXPECT_EQ((Dynamic<16>::hi), kHiInf);
  EXPECT_FALSE((Dynamic<16>::has_lower));
  EXPECT_FALSE((Dynamic<16>::has_upper));
}

TEST_F(ValueTest, DynamicConforms) {
  EXPECT_TRUE((Dynamic<16>::conforms(64)));
  EXPECT_FALSE((Dynamic<16>::conforms(63)));
  EXPECT_TRUE((Dynamic<16>::conforms(0)));
}

TEST_F(ValueTest, DynamicAligns) {
  EXPECT_TRUE((Dynamic<16>::aligns(4)));
  EXPECT_TRUE((Dynamic<16>::aligns(8)));
  EXPECT_FALSE((Dynamic<16>::aligns(32)));
}

TEST_F(ValueTest, DynamicIsAligned) {
  Dynamic<16> a(64);
  EXPECT_TRUE(a.is_aligned(4));
  EXPECT_FALSE(a.is_aligned(7));
}

TEST_F(ValueTest, DynamicOperatorNint) {
  Dynamic<16> a(64);
  EXPECT_EQ(nint_t(a), 64);
}

TEST_F(ValueTest, AnyIsAligned1) {
  EXPECT_FALSE((Any::is_const));
  EXPECT_TRUE((Any::is_runtime));
  EXPECT_EQ((Any::alignment), 1);
}

TEST_F(ValueTest, AnyConforms) {
  EXPECT_TRUE((Any::conforms(0)));
  EXPECT_TRUE((Any::conforms(1)));
  EXPECT_TRUE((Any::conforms(-1)));
  EXPECT_TRUE((Any::conforms(999999)));
}

TEST_F(ValueTest, CintTemplate) {
  auto c = cint<128>;
  EXPECT_EQ(nint_t(c), 128);
  EXPECT_TRUE((std::is_same_v<decltype(c), Const<128>>));
}

TEST_F(ValueTest, CintDifferentValues) {
  EXPECT_EQ(nint_t(cint<0>), 0);
  EXPECT_EQ(nint_t(cint<1>), 1);
  EXPECT_EQ(nint_t(cint<1024>), 1024);
}

TEST_F(ValueTest, ValueBaseDefaults) {
  EXPECT_FALSE(Value::is_const);
  EXPECT_FALSE(Value::is_runtime);
  EXPECT_FALSE(Value::conforms(0));
  EXPECT_FALSE(Value::aligns(1));
}

// --- Dynamic bounds tests ---

TEST_F(ValueTest, DynamicWithBounds) {
  using D = Dynamic<8, 0, 128>;
  EXPECT_FALSE(D::is_const);
  EXPECT_TRUE(D::is_runtime);
  EXPECT_EQ(D::alignment, 8);
  EXPECT_EQ(D::lo, 0);
  EXPECT_EQ(D::hi, 128);
  EXPECT_TRUE(D::has_lower);
  EXPECT_TRUE(D::has_upper);
}

TEST_F(ValueTest, DynamicBoundsConforms) {
  using D = Dynamic<8, 0, 128>;
  EXPECT_TRUE(D::conforms(0));
  EXPECT_TRUE(D::conforms(8));
  EXPECT_TRUE(D::conforms(128));
  EXPECT_FALSE(D::conforms(4));     // unaligned
  EXPECT_FALSE(D::conforms(-8));    // below lower bound
  EXPECT_FALSE(D::conforms(136));   // above upper bound
}

TEST_F(ValueTest, DynamicLowerBoundOnly) {
  using D = Dynamic<16, 32, kHiInf>;
  EXPECT_TRUE(D::has_lower);
  EXPECT_FALSE(D::has_upper);
  EXPECT_TRUE(D::conforms(32));
  EXPECT_TRUE(D::conforms(48));
  EXPECT_FALSE(D::conforms(16));    // below lower bound
}

TEST_F(ValueTest, DynamicNoBoundsIsAny) {
  EXPECT_TRUE((std::is_same_v<Dynamic<1>, Any>));
  EXPECT_FALSE((Dynamic<1>::has_lower));
  EXPECT_FALSE((Dynamic<1>::has_upper));
}

// --- dyn<> helper ---

TEST_F(ValueTest, DynFull) {
  auto d = dyn<8, 0, 128>(64);
  EXPECT_EQ(nint_t(d), 64);
  EXPECT_TRUE((std::is_same_v<decltype(d), Dynamic<8, 0, 128>>));
}

TEST_F(ValueTest, DynAlignmentOnly) {
  auto d = dyn<16>(64);
  EXPECT_EQ(nint_t(d), 64);
  EXPECT_TRUE((std::is_same_v<decltype(d), Dynamic<16>>));
  EXPECT_FALSE(decltype(d)::has_lower);
  EXPECT_FALSE(decltype(d)::has_upper);
}

// --- Operator: Const + Const ---

TEST_F(ValueTest, ConstAddConst) {
  auto r = cint<10> + cint<20>;
  EXPECT_EQ(nint_t(r), 30);
  EXPECT_TRUE((std::is_same_v<decltype(r), Const<30>>));
}

TEST_F(ValueTest, ConstSubConst) {
  auto r = cint<30> - cint<10>;
  EXPECT_EQ(nint_t(r), 20);
  EXPECT_TRUE((std::is_same_v<decltype(r), Const<20>>));
}

TEST_F(ValueTest, ConstMulConst) {
  auto r = cint<4> * cint<5>;
  EXPECT_EQ(nint_t(r), 20);
  EXPECT_TRUE((std::is_same_v<decltype(r), Const<20>>));
}

TEST_F(ValueTest, ConstDivConst) {
  auto r = cint<20> / cint<4>;
  EXPECT_EQ(nint_t(r), 5);
  EXPECT_TRUE((std::is_same_v<decltype(r), Const<5>>));
}

TEST_F(ValueTest, ConstModConst) {
  auto r = cint<17> % cint<5>;
  EXPECT_EQ(nint_t(r), 2);
  EXPECT_TRUE((std::is_same_v<decltype(r), Const<2>>));
}

TEST_F(ValueTest, ConstNeg) {
  auto r = -cint<42>;
  EXPECT_EQ(nint_t(r), -42);
  EXPECT_TRUE((std::is_same_v<decltype(r), Const<-42>>));
}

// --- Operator: Const + Dynamic ---

TEST_F(ValueTest, ConstAddDyn) {
  auto d = Dynamic<8>{16};
  auto r = cint<4> + d;
  EXPECT_EQ(nint_t(r), 20);
  // gcd(8, 4) = 4,  constraints: lo=kLoInf+4=inf, hi=kHiInf+4=inf
  using RT = Dynamic<4>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, DynAddConst) {
  auto d = Dynamic<8>{16};
  auto r = d + cint<4>;
  EXPECT_EQ(nint_t(r), 20);
  using RT = Dynamic<4>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, ConstSubDyn) {
  auto d = Dynamic<8, 0, 128>{64};
  auto r = cint<100> - d;
  EXPECT_EQ(nint_t(r), 36);
  // gcd(8, 100) = 4, lo = 100-128 = -28, hi = 100-0 = 100
  using RT = Dynamic<4, -28, 100>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, DynSubConst) {
  auto d = Dynamic<8, 0, 128>{64};
  auto r = d - cint<20>;
  EXPECT_EQ(nint_t(r), 44);
  // gcd(8, 20) = 4, lo = 0-20=-20, hi=128-20=108
  using RT = Dynamic<4, -20, 108>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

// --- Operator: Dynamic + Dynamic ---

TEST_F(ValueTest, DynAddDyn) {
  auto d1 = Dynamic<16, 0, 64>{32};
  auto d2 = Dynamic<8, 16, 128>{48};
  auto r = d1 + d2;
  EXPECT_EQ(nint_t(r), 80);
  // min(16, 8)=8, lo=0+16=16, hi=64+128=192
  using RT = Dynamic<8, 16, 192>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, DynSubDyn) {
  auto d1 = Dynamic<16, 32, 128>{64};
  auto d2 = Dynamic<8, 0, 64>{32};
  auto r = d1 - d2;
  EXPECT_EQ(nint_t(r), 32);
  // min(16,8)=8, lo=32-64=-32, hi=128-0=128
  using RT = Dynamic<8, -32, 128>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, DynNeg) {
  auto d = Dynamic<16, 0, 128>{64};
  auto r = -d;
  EXPECT_EQ(nint_t(r), -64);
  using RT = Dynamic<16, -128, 0>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

// --- Operator: multiplication ---

TEST_F(ValueTest, ConstMulDyn) {
  auto d = Dynamic<8, 0, 16>{8};
  auto r = cint<3> * d;
  EXPECT_EQ(nint_t(r), 24);
  // lsb(3) = 1, alg = 8*1=8, lo=3*0=0, hi=3*16=48
  using RT = Dynamic<8, 0, 48>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, ConstMulDynNegative) {
  auto d = Dynamic<8, 0, 16>{8};
  auto r = cint<-2> * d;
  EXPECT_EQ(nint_t(r), -16);
  // lsb(-2)=2, alg=8*2=16, lo=-2*16=-32, hi=-2*0=0
  using RT = Dynamic<16, -32, 0>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, ConstMulDynZero) {
  auto d = Dynamic<8, 0, 128>{64};
  auto r = cint<0> * d;
  EXPECT_EQ(nint_t(r), 0);
  EXPECT_TRUE((std::is_same_v<decltype(r), Const<0>>));
}

TEST_F(ValueTest, DynMulDyn) {
  auto d1 = Dynamic<4, 0, 8>{4};
  auto d2 = Dynamic<2, 1, 3>{2};
  auto r = d1 * d2;
  EXPECT_EQ(nint_t(r), 8);
  // alg=4*2=8, lo=min(0,0,8,24)=0, hi=max(0,0,8,24)=24
  using RT = Dynamic<8, 0, 24>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

// --- Operator: division ---

TEST_F(ValueTest, DynDivConstExact) {
  auto d = Dynamic<8, 0, 64>{32};
  auto r = d / cint<4>;
  EXPECT_EQ(nint_t(r), 8);
  // 8%4==0, alg=8/4=2, lo=0/4=0, hi=64/4=16
  using RT = Dynamic<2, 0, 16>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, DynDivConstInexact) {
  auto d = Dynamic<8, 0, 64>{32};
  auto r = d / cint<3>;
  EXPECT_EQ(nint_t(r), 10);
  // 8%3!=0, alg=1
  using RT = Dynamic<1, 0, 21>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, ConstDivDyn) {
  auto d = Dynamic<4, 4, 16>{8};
  auto r = cint<100> / d;
  EXPECT_EQ(nint_t(r), 12);
  // k=[1,4], 100/4=25, 100/8=12, 100/12=8, 100/16=6, hi=25, lo=6
  using RT = Dynamic<1, 6, 25>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

// --- Operator: remainder ---

TEST_F(ValueTest, DynModConstExact) {
  auto d = Dynamic<8, 0, 64>{32};
  auto r = d % cint<4>;
  EXPECT_EQ(nint_t(r), 0);
  // A%N==0: → Const<0>
  EXPECT_TRUE((std::is_same_v<decltype(r), Const<0>>));
}

TEST_F(ValueTest, DynModConstInexact) {
  auto d = Dynamic<8, 0, 64>{16};
  auto r = d % cint<7>;
  EXPECT_EQ(nint_t(r), 2);
  using RT = Dynamic<1, 0, 6>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, ConstModDyn) {
  auto d = Dynamic<4, 4, 16>{8};
  auto r = cint<100> % d;
  EXPECT_EQ(nint_t(r), 4);
  // k=1:100%4=0, k=2:100%8=4, k=3:100%12=4, k=4:100%16=4
  using RT = Dynamic<1, 0, 4>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

// --- Operator: Value + int wrapping ---

TEST_F(ValueTest, ConstAddInt) {
  auto r = cint<10> + 5;
  EXPECT_EQ(nint_t(r), 15);
  // cint<10> + Any{5} = Any{15} (sentinel bounds propagate)
  EXPECT_TRUE((std::is_same_v<decltype(r), Any>));
}

TEST_F(ValueTest, IntAddConst) {
  auto r = 5 + cint<10>;
  EXPECT_EQ(nint_t(r), 15);
  EXPECT_TRUE((std::is_same_v<decltype(r), Any>));
}

TEST_F(ValueTest, DynAddInt) {
  auto d = Dynamic<8, 0, 128>{64};
  auto r = d + 10;
  EXPECT_EQ(nint_t(r), 74);
  // gcd(8, 1)=1, sentinel bounds propagate → no bounds
  EXPECT_FALSE(decltype(r)::has_lower);
  EXPECT_EQ(decltype(r)::alignment, 1);
}

TEST_F(ValueTest, IntMulDyn) {
  auto d = Dynamic<8, 0, 16>{8};
  auto r = 3 * d;
  EXPECT_EQ(nint_t(r), 24);
  // 3 * Any → Any (no bounds)
  EXPECT_FALSE(decltype(r)::has_lower);
}

// ======================================================================
// A: Operator gap tests (Dyn/Dyn / %, value-int wrapping -, /, %)
// ======================================================================

TEST_F(ValueTest, DynDivDyn) {
  auto d1 = Dynamic<4, 0, 16>{8};
  auto d2 = Dynamic<2, 2, 8>{4};
  auto r = d1 / d2;
  EXPECT_EQ(nint_t(r), 2);
  // Alignment → 1, bounds: 0/2=0, 0/8=0, 16/2=8, 16/8=2 → lo=0, hi=8
  using RT = Dynamic<1, 0, 8>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, DynModDyn) {
  auto d1 = Dynamic<4, 0, 16>{12};
  auto d2 = Dynamic<2, 2, 8>{4};
  auto r = d1 % d2;
  EXPECT_EQ(nint_t(r), 0);
  // v1∈[0,16], v2∈[2,8], max_abs_v2=8
  // L1≥0 → lo=0, H1≥0 → hi=7
  using RT = Dynamic<1, 0, 7>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, ConstSubInt) {
  auto r = cint<10> - 3;
  EXPECT_EQ(nint_t(r), 7);
  EXPECT_TRUE((std::is_same_v<decltype(r), Any>));
}

TEST_F(ValueTest, IntSubConst) {
  auto r = 20 - cint<10>;
  EXPECT_EQ(nint_t(r), 10);
  EXPECT_TRUE((std::is_same_v<decltype(r), Any>));
}

TEST_F(ValueTest, ConstDivInt) {
  auto r = cint<20> / 3;
  EXPECT_EQ(nint_t(r), 6);
  EXPECT_TRUE((std::is_same_v<decltype(r), Any>));
}

TEST_F(ValueTest, IntDivConst) {
  auto r = 20 / cint<7>;
  EXPECT_EQ(nint_t(r), 2);
  EXPECT_TRUE((std::is_same_v<decltype(r), Any>));
}

TEST_F(ValueTest, ConstModInt) {
  auto r = cint<17> % 5;
  EXPECT_EQ(nint_t(r), 2);
  EXPECT_TRUE((std::is_same_v<decltype(r), Any>));
}

TEST_F(ValueTest, IntModConst) {
  auto r = 20 % cint<7>;
  EXPECT_EQ(nint_t(r), 6);
  EXPECT_TRUE((std::is_same_v<decltype(r), Any>));
}

TEST_F(ValueTest, DynMulInt) {
  auto d = Dynamic<8, 0, 16>{8};
  auto r = d * 3;
  EXPECT_EQ(nint_t(r), 24);
  EXPECT_FALSE(decltype(r)::has_lower);
}

TEST_F(ValueTest, DynDivInt) {
  auto d = Dynamic<8, 0, 64>{32};
  auto r = d / 3;
  EXPECT_EQ(nint_t(r), 10);
  EXPECT_FALSE(decltype(r)::has_lower);
}

TEST_F(ValueTest, DynSubInt) {
  auto d = Dynamic<8, 0, 128>{64};
  auto r = d - 10;
  EXPECT_EQ(nint_t(r), 54);
  EXPECT_EQ(decltype(r)::alignment, 1);
}

// ======================================================================
// B: Sentinel propagation tests
// ======================================================================

TEST_F(ValueTest, AnyAddAny) {
  auto r = Any{10} + Any{20};
  EXPECT_EQ(nint_t(r), 30);
  using RT = Dynamic<1>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
  EXPECT_FALSE(decltype(r)::has_lower);
  EXPECT_FALSE(decltype(r)::has_upper);
}

TEST_F(ValueTest, AnyMulAny) {
  auto r = Any{5} * Any{6};
  EXPECT_EQ(nint_t(r), 30);
  using RT = Dynamic<1>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, AnyAddDyn) {
  auto r = Any{100} + Dynamic<8, 0, 128>{64};
  EXPECT_EQ(nint_t(r), 164);
  // Any's sentinel bounds dominate → unbounded
  using RT = Dynamic<1>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, ConstAddAny) {
  auto r = cint<10> + Any{5};
  EXPECT_EQ(nint_t(r), 15);
  using RT = Dynamic<1>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, DynSubAny) {
  auto d = Dynamic<16, 32, 128>{64};
  auto r = d - Any{10};
  EXPECT_EQ(nint_t(r), 54);
  // Any's sentinel bounds → unbounded
  using RT = Dynamic<1>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

// ======================================================================
// C: Bounds edge case tests
// ======================================================================

TEST_F(ValueTest, DynDivConstNegativeBounds) {
  auto d = Dynamic<8, -64, -16>{-32};
  auto r = d / cint<4>;
  EXPECT_EQ(nint_t(r), -8);
  // A=8, N=4, A%N=0, g=8/4=2
  // k in [-8, -2], f(k)=2*k → [-16, -4]
  using RT = Dynamic<2, -16, -4>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, ConstDivDynAllNeg) {
  auto d = Dynamic<4, -16, -4>{-8};
  auto r = cint<100> / d;
  EXPECT_EQ(nint_t(r), -12);
  // k_min=-4, k_max=-1. has_neg=true. c3=100/(-16)=-6, c4=100/(-4)=-25.
  using RT = Dynamic<1, -25, -6>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, ConstDivDynBothSides) {
  auto d = Dynamic<8, -16, 16>{8};
  auto r = cint<100> / d;
  EXPECT_EQ(nint_t(r), 12);
  // k_min=-2, k_max=2. has_pos && has_neg → full endpoint analysis
  using RT = Dynamic<1, -12, 12>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, DynModConstCrossZero) {
  auto d = Dynamic<8, -32, 32>{16};
  auto r = d % cint<7>;
  EXPECT_EQ(nint_t(r), 2);
  // L=-32<0, H=32>0 → sign-aware bounds [-6, 6]
  using RT = Dynamic<1, -6, 6>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, ConstMulDynPowerOfTwo) {
  auto d = Dynamic<16, 0, 32>{16};
  auto r = cint<8> * d;
  EXPECT_EQ(nint_t(r), 128);
  // lsb(8)=8, alg=16*8=128, lo=0, hi=256
  using RT = Dynamic<128, 0, 256>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, DynNegSentinelBounds) {
  auto d = Dynamic<16, 0, kHiInf>{64};
  auto r = -d;
  EXPECT_EQ(nint_t(r), -64);
  // -hi = -(kHiInf) = kLoInf, -lo = -(0) = 0
  using RT = Dynamic<16, kLoInf, 0>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

// ======================================================================
// D: Chaining and type verification tests
// ======================================================================

TEST_F(ValueTest, ChainedAddMul) {
  auto d1 = Dynamic<8, 0, 128>{64};
  auto d2 = Dynamic<4, 0, 64>{32};
  auto r = (d1 + cint<10>) * d2;
  EXPECT_EQ(nint_t(r), 2368);
  // Step1: Dyn<8,0,128>+Const<10> → Dyn<2,10,138>{74}
  // Step2: Dyn<2,10,138>*Dyn<4,0,64> → Dyn<8,0,8832>{2368}
  using RT = Dynamic<8, 0, 8832>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, ChainedSubAdd) {
  auto d1 = Dynamic<16, 32, 128>{64};
  auto d2 = Dynamic<8, 0, 64>{32};
  auto r = d1 - (cint<50> + d2);
  EXPECT_EQ(nint_t(r), -18);
  // Step1: Const<50>+Dyn<8,0,64> → Dyn<2,50,114>{82}
  // Step2: Dyn<16,32,128> - Dyn<2,50,114> → Dyn<2,-82,78>{-18}
  using RT = Dynamic<2, -82, 78>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, ChainedDivMul) {
  auto d = Dynamic<8, 0, 64>{32};
  auto r = (d / cint<4>) * cint<3>;
  EXPECT_EQ(nint_t(r), 24);
  // Step1: Dyn<8,0,64>/Const<4> → Dyn<2,0,16>{8}
  // Step2: Dyn<2,0,16>*Const<3> → Dyn<2,0,48>{24}
  using RT = Dynamic<2, 0, 48>;
  EXPECT_TRUE((std::is_same_v<decltype(r), RT>));
}

TEST_F(ValueTest, TypeVerificationAdd) {
  using T = decltype(Const<10>{} + Dynamic<8, 0, 128>{64});
  // gcd(8,10): lsb(10)=2, min(8,2)=2. L+N=10, H+N=138.
  EXPECT_TRUE((std::is_same_v<T, Dynamic<2, 10, 138>>));
}

TEST_F(ValueTest, OperatorOnPackedValue) {
  vecops::gemm::details::PackedStorage<Dynamic<8, 0, 64>> ps(Dynamic<8, 0, 64>{32});
  auto v = ps.template get<0>();
  auto d = Dynamic<16, 0, 128>{64};
  auto r = d + v;
  EXPECT_EQ(nint_t(r), 96);
  // d + nint_t(v) → d + Any{32} → sentinel lo/hi
  EXPECT_FALSE(decltype(r)::has_lower);
}

TEST_F(ValueTest, ToValuePromotesInt) {
  EXPECT_TRUE((std::is_same_v<ToValue<int>, Any>));
  EXPECT_TRUE((std::is_same_v<ToValue<uint32_t>, Any>));
}

TEST_F(ValueTest, ToValuePreservesValueType) {
  EXPECT_TRUE((std::is_same_v<ToValue<Const<5>>, Const<5>>));
  EXPECT_TRUE((std::is_same_v<ToValue<Dynamic<16>>, Dynamic<16>>));
  EXPECT_TRUE((std::is_same_v<ToValue<Any>, Any>));
}

// ======================================================================
// PackedStorage Suite
// ======================================================================

class PackedStorageTest : public ::testing::Test {};

TEST_F(PackedStorageTest, ConstructAllRuntime) {
  vecops::gemm::details::PackedStorage<Any, Any> ps(Any{3}, Any{4});
  EXPECT_EQ(ps.template get<0>(), 3);
  EXPECT_EQ(ps.template get<1>(), 4);
  EXPECT_EQ(ps[0], 3);
  EXPECT_EQ(ps[1], 4);
}

TEST_F(PackedStorageTest, ConstructAllConst) {
  vecops::gemm::details::PackedStorage<Const<3>, Const<4>> ps(Const<3>{}, Const<4>{});
  EXPECT_EQ(ps.template get<0>(), 3);
  EXPECT_EQ(ps.template get<1>(), 4);
  EXPECT_EQ(ps[0], 3);
  EXPECT_EQ(ps[1], 4);
}

TEST_F(PackedStorageTest, ConstructMixed) {
  vecops::gemm::details::PackedStorage<Const<128>, Any> ps(Const<128>{}, Any{100});
  EXPECT_EQ(ps.template get<0>(), 128);
  EXPECT_EQ(ps.template get<1>(), 100);
}

TEST_F(PackedStorageTest, ConstructFromPtrAllRuntime) {
  nint_t data[] = {7, 8};
  vecops::gemm::details::PackedStorage<Any, Any> ps(data);
  EXPECT_EQ(ps[0], 7);
  EXPECT_EQ(ps[1], 8);
}

TEST_F(PackedStorageTest, ConstructFromPtrMixed) {
  nint_t data[] = {128, 100};
  vecops::gemm::details::PackedStorage<Const<128>, Any> ps(data);
  EXPECT_EQ(ps.template get<0>(), 128);
  EXPECT_EQ(ps.template get<1>(), 100);
  EXPECT_EQ(ps[0], 128);  // const
  EXPECT_EQ(ps[1], 100);  // runtime
}

TEST_F(PackedStorageTest, OperatorBracketConstDim) {
  vecops::gemm::details::PackedStorage<Const<5>, Any, Const<10>> ps(Const<5>{}, Any{7}, Const<10>{});
  // operator[] 对 const 维度必须返回正确的 const_values（回归 bug #2）
  EXPECT_EQ(ps[0], 5);
  EXPECT_EQ(ps[1], 7);
  EXPECT_EQ(ps[2], 10);
}

TEST_F(PackedStorageTest, ToArrayRoundtrip) {
  vecops::gemm::details::PackedStorage<Const<3>, Any, Const<5>> ps(Const<3>{}, Any{7}, Const<5>{});
  auto arr = ps.to_array();
  EXPECT_EQ(arr[0], 3);
  EXPECT_EQ(arr[1], 7);
  EXPECT_EQ(arr[2], 5);
  EXPECT_EQ(arr.size(), 3u);
}

TEST_F(PackedStorageTest, ToPackedArray) {
  vecops::gemm::details::PackedStorage<Const<3>, Any, Const<5>> ps(Const<3>{}, Any{7}, Const<5>{});
  auto& packed = ps.to_packed_array();
  EXPECT_EQ(packed.size(), 1u);  // only Any dimension
  EXPECT_EQ(packed[0], 7);
}

TEST_F(PackedStorageTest, FromPackedArrayRoundtrip) {
  nint_t packed_data[] = {42};
  auto ps = vecops::gemm::details::PackedStorage<Const<3>, Any, Const<5>>::from_packed_array(packed_data);
  EXPECT_EQ(ps.template get<1>(), 42);
}

TEST_F(PackedStorageTest, OffsetsAllRuntime) {
  auto& offsets = vecops::gemm::details::PackedStorage<Any, Any, Any>::offsets;
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 1);
  EXPECT_EQ(offsets[2], 2);
}

TEST_F(PackedStorageTest, OffsetsMixed) {
  auto& offsets = vecops::gemm::details::PackedStorage<Const<1>, Any, Const<2>, Any>::offsets;
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 0);
  EXPECT_EQ(offsets[2], 1);
  EXPECT_EQ(offsets[3], 1);
}

TEST_F(PackedStorageTest, NumStorAllConst) {
  EXPECT_EQ((vecops::gemm::details::PackedStorage<Const<1>, Const<2>, Const<3>>::num_stor), 0);
}

TEST_F(PackedStorageTest, NumStorAllRuntime) {
  EXPECT_EQ((vecops::gemm::details::PackedStorage<Any, Any, Any>::num_stor), 3);
}

TEST_F(PackedStorageTest, NumStorMixed) {
  EXPECT_EQ((vecops::gemm::details::PackedStorage<Const<1>, Any, Const<2>>::num_stor), 1);
}

TEST_F(PackedStorageTest, SingleDimRuntime) {
  vecops::gemm::details::PackedStorage<Any> ps(Any{42});
  EXPECT_EQ(ps.template get<0>(), 42);
  EXPECT_EQ(ps[0], 42);
  auto arr = ps.to_array();
  EXPECT_EQ(arr[0], 42);
}

TEST_F(PackedStorageTest, SingleDimConst) {
  vecops::gemm::details::PackedStorage<Const<99>> ps(Const<99>{});
  EXPECT_EQ(ps.template get<0>(), 99);
  EXPECT_EQ(ps[0], 99);
}

TEST_F(PackedStorageTest, WithBoundedDynamic) {
  vecops::gemm::details::PackedStorage<Dynamic<8, 0, 128>> ps(Dynamic<8, 0, 128>{64});
  EXPECT_EQ(ps.template get<0>(), 64);
  EXPECT_EQ(ps[0], 64);
  EXPECT_EQ(ps.to_array()[0], 64);
}

// ======================================================================
// ArrayMeta Suite
// ======================================================================

class ArrayMetaTest : public ::testing::Test {};

TEST_F(ArrayMetaTest, ConstructFromValues) {
  ArrayMeta<Const<128>, Any> am(128, 100);
  EXPECT_EQ(am.template get<0>(), 128);
  EXPECT_EQ(am.template get<1>(), 100);
}

TEST_F(ArrayMetaTest, GetCompiletimeIndex) {
  ArrayMeta<Const<128>, Any> am(128, 100);
  EXPECT_EQ(am.template get<0>(), 128);
}

TEST_F(ArrayMetaTest, GetRuntimeIndex) {
  ArrayMeta<Const<128>, Any> am(128, 100);
  EXPECT_EQ(am.template get<1>(), 100);
}

TEST_F(ArrayMetaTest, OperatorBracket) {
  ArrayMeta<Any, Any> am(3, 4);
  EXPECT_EQ(am[0], 3);
  EXPECT_EQ(am[1], 4);
}

TEST_F(ArrayMetaTest, IsConst) {
  ArrayMeta<Const<128>, Any> am(128, 100);
  EXPECT_TRUE(am.template is_const<0>());
  EXPECT_FALSE(am.template is_const<1>());
}

TEST_F(ArrayMetaTest, IsRuntime) {
  ArrayMeta<Const<128>, Any> am(128, 100);
  EXPECT_FALSE(am.template is_runtime<0>());
  EXPECT_TRUE(am.template is_runtime<1>());
}

TEST_F(ArrayMetaTest, Ndim) {
  ArrayMeta<Const<128>, Any> am(128, 100);
  EXPECT_EQ(am.ndim(), 2);
}

TEST_F(ArrayMetaTest, NdimMatchesTemplate) {
  EXPECT_EQ((ArrayMeta<Any, Any, Any>::Ndim), 3);
  EXPECT_EQ((ArrayMeta<Const<1>>::Ndim), 1);
}

TEST_F(ArrayMetaTest, SingleDim) {
  ArrayMeta<Any> am(42);
  EXPECT_EQ(am.ndim(), 1);
  EXPECT_EQ(am[0], 42);
  EXPECT_EQ(am.template get<0>(), 42);
}

TEST_F(ArrayMetaTest, AllConst) {
  ArrayMeta<Const<1>, Const<2>, Const<3>> am(1, 2, 3);
  EXPECT_EQ(am[0], 1);
  EXPECT_EQ(am[1], 2);
  EXPECT_EQ(am[2], 3);
}

// ======================================================================
// Shape Suite
// ======================================================================

class ShapeTest : public ::testing::Test {};

TEST_F(ShapeTest, MakeShapeLiteral) {
  auto s = make_shape(128, 64, 32);
  EXPECT_EQ(s.ndim(), 3);
  EXPECT_EQ(s[0], 128);
  EXPECT_EQ(s[1], 64);
  EXPECT_EQ(s[2], 32);
  EXPECT_TRUE((std::is_same_v<decltype(s), Shape<Any, Any, Any>>));
}

TEST_F(ShapeTest, MakeShapeConst) {
  auto s = make_shape(cint<128>, cint<64>);
  EXPECT_EQ(s.ndim(), 2);
  EXPECT_EQ(s.template get<0>(), 128);
  EXPECT_TRUE(s.template is_const<0>());
  EXPECT_TRUE(s.template is_const<1>());
}

TEST_F(ShapeTest, MakeShapeMixed) {
  auto s = make_shape(cint<128>, 100, cint<32>);
  EXPECT_EQ(s.ndim(), 3);
  EXPECT_EQ(s[0], 128);
  EXPECT_EQ(s[1], 100);
  EXPECT_EQ(s[2], 32);
  EXPECT_TRUE(s.template is_const<0>());
  EXPECT_FALSE(s.template is_const<1>());
  EXPECT_TRUE(s.template is_const<2>());
}

TEST_F(ShapeTest, MakeShapeSingle) {
  auto s = make_shape(128);
  EXPECT_EQ(s.ndim(), 1);
  EXPECT_EQ(s[0], 128);
}

TEST_F(ShapeTest, MakeShapeForwardLvalue) {
  auto c = cint<128>;
  auto s = make_shape(c, 64);
  EXPECT_EQ(s.ndim(), 2);
  EXPECT_EQ(s.template get<0>(), 128);
  EXPECT_EQ(s.template get<1>(), 64);
}

TEST_F(ShapeTest, DefaultCtor) {
  Shape<Any, Any> s;
  (void)s;
}

TEST_F(ShapeTest, MakeShapeWithDynamicBounds) {
  auto s = make_shape(Dynamic<8, 0, 128>{64}, 100);
  EXPECT_EQ(s.ndim(), 2);
  EXPECT_EQ(s[0], 64);
  EXPECT_EQ(s[1], 100);
  EXPECT_FALSE(s.template is_const<0>());
  EXPECT_FALSE(s.template is_const<1>());
}

// ======================================================================
// Strides Suite
// ======================================================================

class StridesTest : public ::testing::Test {};

TEST_F(StridesTest, MakeStridesLiteral) {
  auto st = make_strides(64, 1);
  EXPECT_EQ(st.ndim(), 2);
  EXPECT_EQ(st[0], 64);
  EXPECT_EQ(st[1], 1);
  EXPECT_TRUE((std::is_same_v<decltype(st), Strides<Any, Any>>));
}

TEST_F(StridesTest, MakeStridesConst) {
  auto st = make_strides(cint<128>, cint<1>);
  EXPECT_EQ(st.ndim(), 2);
  EXPECT_TRUE(st.template is_const<0>());
  EXPECT_TRUE(st.template is_const<1>());
}

TEST_F(StridesTest, MakeStridesNegative) {
  auto st = make_strides(-1, 10);
  EXPECT_EQ(st[0], -1);
  EXPECT_EQ(st[1], 10);
}

TEST_F(StridesTest, MakeStridesForwardLvalue) {
  int x = 64, y = 1;
  auto st = make_strides(x, y);
  EXPECT_EQ(st[0], 64);
  EXPECT_EQ(st[1], 1);
}

TEST_F(StridesTest, DefaultCtor) {
  Strides<Any, Any> s;
  (void)s;
}

TEST_F(StridesTest, MakeStridesWithDynamicBounds) {
  auto st = make_strides(Dynamic<16, -128, 128>{64}, cint<1>);
  EXPECT_EQ(st.ndim(), 2);
  EXPECT_EQ(st[0], 64);
  EXPECT_EQ(st[1], 1);
}

// ======================================================================
// Layout Suite
// ======================================================================

class LayoutTest : public ::testing::Test {};

TEST_F(LayoutTest, ConstructAndAccess) {
  auto s = make_shape(cint<128>, cint<64>);
  auto st = make_strides(cint<64>, cint<1>);
  Layout layout(s, st);
  EXPECT_EQ(layout.ndim(), 2);
  EXPECT_EQ(layout.shape().template get<0>(), 128);
  EXPECT_EQ(layout.strides().template get<0>(), 64);
  EXPECT_EQ(layout.shape().template get<1>(), 64);
  EXPECT_EQ(layout.strides().template get<1>(), 1);
}

TEST_F(LayoutTest, MakeLayout) {
  auto layout = make_layout(make_shape(128, 64), make_strides(64, 1));
  EXPECT_EQ(layout.ndim(), 2);
  EXPECT_EQ(layout.shape()[0], 128);
  EXPECT_EQ(layout.strides()[0], 64);
}

TEST_F(LayoutTest, MakeLayoutForward) {
  auto layout = make_layout(make_shape(cint<128>, cint<64>),
                            make_strides(cint<64>, cint<1>));
  EXPECT_EQ(layout.ndim(), 2);
  EXPECT_TRUE(layout.shape().template is_const<0>());
}

TEST_F(LayoutTest, SingleDimLayout) {
  auto layout = make_layout(make_shape(128), make_strides(1));
  EXPECT_EQ(layout.ndim(), 1);
  EXPECT_EQ(layout.shape()[0], 128);
  EXPECT_EQ(layout.strides()[0], 1);
}

TEST_F(LayoutTest, WithDynamicBounds) {
  auto s = make_shape(Dynamic<8, 0, 128>{64}, cint<32>);
  auto st = make_strides(Dynamic<16, -128, 128>{32}, cint<1>);
  Layout layout(s, st);
  EXPECT_EQ(layout.ndim(), 2);
  EXPECT_EQ(layout.shape()[0], 64);
  EXPECT_EQ(layout.strides()[0], 32);
  EXPECT_EQ(layout.shape()[1], 32);
  EXPECT_EQ(layout.strides()[1], 1);
}

// ======================================================================
// Free Functions Suite — get / is_const / is_runtime / size / stride
// ======================================================================

class FreeFunctionTest : public ::testing::Test {};

TEST_F(FreeFunctionTest, FreeGet) {
  auto s = make_shape(cint<128>, 100);
  EXPECT_EQ(get<0>(s), 128);
  EXPECT_EQ(get<1>(s), 100);
}

TEST_F(FreeFunctionTest, FreeIsConst) {
  auto s = make_shape(cint<128>, 100);
  EXPECT_EQ(is_const<0>(s), true);
  EXPECT_EQ(is_const<1>(s), false);
}

TEST_F(FreeFunctionTest, FreeIsRuntime) {
  auto s = make_shape(cint<128>, 100);
  EXPECT_EQ(is_runtime<0>(s), false);
  EXPECT_EQ(is_runtime<1>(s), true);
}

TEST_F(FreeFunctionTest, FreeSize) {
  auto layout = make_layout(make_shape(cint<128>, cint<64>),
                            make_strides(cint<64>, cint<1>));
  EXPECT_EQ(size<0>(layout), 128);
  EXPECT_EQ(size<1>(layout), 64);
}

TEST_F(FreeFunctionTest, FreeStride) {
  auto layout = make_layout(make_shape(cint<128>, cint<64>),
                            make_strides(cint<64>, cint<1>));
  EXPECT_EQ(stride<0>(layout), 64);
  EXPECT_EQ(stride<1>(layout), 1);
}

TEST_F(FreeFunctionTest, OffsetAtFromIntegralPack) {
  auto layout = make_layout(make_shape(3, 4, 5), make_strides(20, 5, 1));
  EXPECT_EQ(offset_at(layout, 2, 3, 4), 59);
}

TEST_F(FreeFunctionTest, OffsetAtFromArray) {
  auto layout = make_layout(make_shape(3, 4), make_strides(10, 2));
  std::array<nint_t, 2> coords{2, 3};
  EXPECT_EQ(offset_at(layout, coords), 26);
}

TEST_F(FreeFunctionTest, OffsetAtSupportsCompileTimeLayout) {
  auto layout = make_layout(make_shape(cint<3>, cint<4>),
                            make_strides(cint<7>, cint<1>));
  EXPECT_EQ(offset_at(layout, 2, 3), 17);
}

TEST_F(FreeFunctionTest, OffsetAtSupportsNegativeStrides) {
  auto layout = make_layout(make_shape(3, 4), make_strides(-4, 1));
  EXPECT_EQ(offset_at(layout, 2, 3), -5);
}

// ======================================================================
// MetaOps Suite — remove / set / insert on ArrayMeta, Shape, Strides, Layout
// ======================================================================

class MetaOpsTest : public ::testing::Test {};

// --- remove ---

TEST_F(MetaOpsTest, RemoveDimMeta) {
  auto m = ArrayMeta<Const<1>, Any, Const<3>>(1, 2, 3);
  auto r = remove<1>(m);
  EXPECT_EQ(r.ndim(), 2);
  EXPECT_EQ(get<0>(r), 1);
  EXPECT_EQ(get<1>(r), 3);
}

TEST_F(MetaOpsTest, RemoveDimShape) {
  auto s = make_shape(cint<1>, cint<2>, 100);
  auto r = remove<0>(s);
  EXPECT_EQ(r.ndim(), 2);
  EXPECT_EQ(get<0>(r), 2);
  EXPECT_EQ(get<1>(r), 100);
}

TEST_F(MetaOpsTest, RemoveDimStrides) {
  auto st = make_strides(cint<64>, cint<1>, 10);
  auto r = remove<1>(st);
  EXPECT_EQ(r.ndim(), 2);
  EXPECT_EQ(get<0>(r), 64);
  EXPECT_EQ(get<1>(r), 10);
}

TEST_F(MetaOpsTest, RemoveDimLayout) {
  auto layout = make_layout(make_shape(cint<128>, cint<64>, 100),
                            make_strides(cint<64>, cint<1>, 10));
  auto r = remove<1>(layout);
  EXPECT_EQ(r.ndim(), 2);
  EXPECT_EQ(size<0>(r), 128);
  EXPECT_EQ(size<1>(r), 100);
  EXPECT_EQ(stride<0>(r), 64);
  EXPECT_EQ(stride<1>(r), 10);
}

TEST_F(MetaOpsTest, RemoveFirstDim) {
  auto s = make_shape(cint<1>, 2, 3);
  auto r = remove<0>(s);
  EXPECT_EQ(r.ndim(), 2);
  EXPECT_EQ(get<0>(r), 2);
}

TEST_F(MetaOpsTest, RemoveLastDim) {
  auto s = make_shape(1, 2, cint<3>);
  auto r = remove<2>(s);
  EXPECT_EQ(r.ndim(), 2);
  EXPECT_EQ(get<1>(r), 2);
}

// --- set ---

TEST_F(MetaOpsTest, SetDimMeta) {
  auto m = ArrayMeta<Const<1>, Any, Const<3>>(1, 2, 3);
  auto s = set<1>(m, Any{99});
  EXPECT_EQ(s.ndim(), 3);
  EXPECT_EQ(get<0>(s), 1);
  EXPECT_EQ(get<1>(s), 99);
  EXPECT_EQ(get<2>(s), 3);
}

TEST_F(MetaOpsTest, SetDimShape) {
  auto shape = make_shape(cint<128>, cint<64>);
  auto s = set<1>(shape, Any{32});
  EXPECT_EQ(s.ndim(), 2);
  EXPECT_EQ(get<0>(s), 128);
  EXPECT_EQ(get<1>(s), 32);
}

TEST_F(MetaOpsTest, SetDimStrides) {
  auto st = make_strides(cint<64>, cint<1>);
  auto s = set<0>(st, Any{32});
  EXPECT_EQ(get<0>(s), 32);
  EXPECT_EQ(get<1>(s), 1);
}

TEST_F(MetaOpsTest, SetDimLayout) {
  auto layout = make_layout(make_shape(cint<128>, cint<64>),
                            make_strides(cint<64>, cint<1>));
  auto s = set<0>(layout, Any{256}, Any{128});
  EXPECT_EQ(size<0>(s), 256);
  EXPECT_EQ(stride<0>(s), 128);
  EXPECT_EQ(size<1>(s), 64);
  EXPECT_EQ(stride<1>(s), 1);
}

// --- insert ---

TEST_F(MetaOpsTest, InsertDimMeta) {
  auto m = ArrayMeta<Const<1>, Const<3>>(1, 3);
  auto r = insert<1>(m, Any{2});
  EXPECT_EQ(r.ndim(), 3);
  EXPECT_EQ(get<0>(r), 1);
  EXPECT_EQ(get<1>(r), 2);
  EXPECT_EQ(get<2>(r), 3);
}

TEST_F(MetaOpsTest, InsertDimShape) {
  auto s = make_shape(cint<1>, cint<3>);
  auto r = insert<1>(s, Any{2});
  EXPECT_EQ(r.ndim(), 3);
  EXPECT_EQ(get<0>(r), 1);
  EXPECT_EQ(get<1>(r), 2);
  EXPECT_EQ(get<2>(r), 3);
}

TEST_F(MetaOpsTest, InsertAtBegin) {
  auto s = make_shape(cint<2>, cint<3>);
  auto r = insert<0>(s, cint<1>);
  EXPECT_EQ(r.ndim(), 3);
  EXPECT_EQ(get<0>(r), 1);
  EXPECT_EQ(get<1>(r), 2);
  EXPECT_EQ(get<2>(r), 3);
}

TEST_F(MetaOpsTest, InsertAtEnd) {
  auto s = make_shape(cint<1>, cint<2>);
  auto r = insert<2>(s, Any{3});
  EXPECT_EQ(r.ndim(), 3);
  EXPECT_EQ(get<0>(r), 1);
  EXPECT_EQ(get<1>(r), 2);
  EXPECT_EQ(get<2>(r), 3);
}

TEST_F(MetaOpsTest, InsertDimLayout) {
  auto layout = make_layout(make_shape(cint<128>, cint<64>),
                            make_strides(cint<64>, cint<1>));
  auto r = insert<1>(layout, cint<3>, cint<30>);
  EXPECT_EQ(r.ndim(), 3);
  EXPECT_EQ(size<0>(r), 128);
  EXPECT_EQ(size<1>(r), 3);
  EXPECT_EQ(size<2>(r), 64);
  EXPECT_EQ(stride<0>(r), 64);
  EXPECT_EQ(stride<1>(r), 30);
  EXPECT_EQ(stride<2>(r), 1);
}

// --- combined operations ---

TEST_F(MetaOpsTest, SetAfterRemove) {
  auto s = make_shape(cint<1>, 100, cint<3>);
  auto r = remove<1>(s);
  auto t = set<1>(r, Any{99});
  EXPECT_EQ(t.ndim(), 2);
  EXPECT_EQ(get<0>(t), 1);
  EXPECT_EQ(get<1>(t), 99);
}

TEST_F(MetaOpsTest, InsertAfterRemove) {
  auto s = make_shape(cint<1>, 100, cint<3>);
  auto r = remove<1>(s);
  auto t = insert<0>(r, cint<0>);
  EXPECT_EQ(t.ndim(), 3);
  EXPECT_EQ(get<0>(t), 0);
  EXPECT_EQ(get<1>(t), 1);
  EXPECT_EQ(get<2>(t), 3);
}

// ======================================================================
// I/O Suite — operator<< for ArrayMeta and Layout
// ======================================================================

class LayoutIOTest : public ::testing::Test {};

TEST_F(LayoutIOTest, PrintAllConst) {
  auto s = make_shape(cint<128>, cint<64>);
  EXPECT_EQ(to_string(s), "(128!, 64!)");
}

TEST_F(LayoutIOTest, PrintAllAny) {
  auto s = make_shape(128, 64);
  EXPECT_EQ(to_string(s), "(128, 64)");
}

TEST_F(LayoutIOTest, PrintMixed) {
  auto s = make_shape(cint<128>, 100, cint<64>);
  EXPECT_EQ(to_string(s), "(128!, 100, 64!)");
}

TEST_F(LayoutIOTest, PrintSingleDim) {
  auto s = make_shape(128);
  EXPECT_EQ(to_string(s), "(128)");
}

TEST_F(LayoutIOTest, PrintSingleDimConst) {
  auto s = make_shape(cint<128>);
  EXPECT_EQ(to_string(s), "(128!)");
}

TEST_F(LayoutIOTest, PrintDynamic) {
  ArrayMeta<Dynamic<16>, Dynamic<4>> am(Dynamic<16>{64}, Dynamic<4>{8});
  EXPECT_EQ(to_string(am), "(64@16, 8@4)");
}

TEST_F(LayoutIOTest, PrintDynamicWithAny) {
  ArrayMeta<Any, Dynamic<16>> am(Any{100}, Dynamic<16>{64});
  EXPECT_EQ(to_string(am), "(100, 64@16)");
}

TEST_F(LayoutIOTest, PrintLayoutBasic) {
  auto layout = make_layout(make_shape(cint<128>, cint<64>),
                            make_strides(cint<64>, cint<1>));
  EXPECT_EQ(to_string(layout), "Layout(s=(128!, 64!), st=(64!, 1!))");
}

TEST_F(LayoutIOTest, PrintLayoutMixed) {
  auto layout = make_layout(make_shape(128, 64),
                            make_strides(cint<64>, 1));
  EXPECT_EQ(to_string(layout), "Layout(s=(128, 64), st=(64!, 1))");
}

TEST_F(LayoutIOTest, PrintLayoutSingleDim) {
  auto layout = make_layout(make_shape(128), make_strides(1));
  EXPECT_EQ(to_string(layout), "Layout(s=(128), st=(1))");
}

// --- I/O: Dynamic bounds output ---

TEST_F(LayoutIOTest, PrintBoundsBoth) {
  ArrayMeta<Dynamic<8, 0, 128>> am(Dynamic<8, 0, 128>{64});
  EXPECT_EQ(to_string(am), "(64@8[0,128])");
}

TEST_F(LayoutIOTest, PrintBoundsLowerOnly) {
  ArrayMeta<Dynamic<8, 16, kHiInf>> am(Dynamic<8, 16, kHiInf>{32});
  EXPECT_EQ(to_string(am), "(32@8[16,∞))");
}

TEST_F(LayoutIOTest, PrintBoundsUpperOnly) {
  ArrayMeta<Dynamic<4, kLoInf, 64>> am(Dynamic<4, kLoInf, 64>{16});
  EXPECT_EQ(to_string(am), "(16@4[0,64])");
}

// ======================================================================
// Type Traits Suite
// ======================================================================

class TypeTraitsTest : public ::testing::Test {};

TEST_F(TypeTraitsTest, IsArrayMeta) {
  EXPECT_TRUE((is_array_meta<ArrayMeta<Any, Any>>));
  EXPECT_FALSE((is_array_meta<int>));
  EXPECT_FALSE((is_array_meta<Layout<Shape<Any>, Strides<Any>>>));
}

TEST_F(TypeTraitsTest, IsShape) {
  EXPECT_TRUE((is_shape<Shape<Any, Any>>));
  EXPECT_TRUE((is_shape<Shape<Const<128>, Any>>));
  EXPECT_FALSE((is_shape<int>));
  EXPECT_FALSE((is_shape<ArrayMeta<Any, Any>>));
  EXPECT_FALSE((is_shape<Strides<Any, Any>>));
}

TEST_F(TypeTraitsTest, IsStrides) {
  EXPECT_TRUE((is_strides<Strides<Any, Any>>));
  EXPECT_TRUE((is_strides<Strides<Const<64>>>));
  EXPECT_FALSE((is_strides<int>));
  EXPECT_FALSE((is_strides<ArrayMeta<Any, Any>>));
  EXPECT_FALSE((is_strides<Shape<Any, Any>>));
}

TEST_F(TypeTraitsTest, IsLayout) {
  EXPECT_TRUE((is_layout<Layout<Shape<Any, Any>, Strides<Any, Any>>>));
  EXPECT_FALSE((is_layout<int>));
  EXPECT_FALSE((is_layout<ArrayMeta<Any, Any>>));
}

TEST_F(TypeTraitsTest, IsArrayMetaSubsumesShape) {
  EXPECT_TRUE((is_array_meta<Shape<Any, Any>>));
}

TEST_F(TypeTraitsTest, IsArrayMetaSubsumesStrides) {
  EXPECT_TRUE((is_array_meta<Strides<Any, Any>>));
}

// ======================================================================
// Compile-Time Suite — constexpr evaluation & static_assert equivalents
// (only valid in non-Debug builds; VECOPS_DEBUG makes VECOPS_ASSERT non-constexpr)
// ======================================================================

#ifndef VECOPS_DEBUG

class CompileTimeTest : public ::testing::Test {};

TEST_F(CompileTimeTest, ConstexprShape) {
  constexpr auto s = make_shape(cint<128>, cint<64>);
  EXPECT_EQ(s.ndim(), 2);
  EXPECT_EQ(s[0], 128);
}

TEST_F(CompileTimeTest, ConstexprGet) {
  constexpr auto s = make_shape(cint<128>, cint<64>);
  constexpr nint_t v0 = get<0>(s);
  constexpr nint_t v1 = get<1>(s);
  EXPECT_EQ(v0, 128);
  EXPECT_EQ(v1, 64);
}

TEST_F(CompileTimeTest, ConstexprRemove) {
  constexpr auto s = make_shape(cint<128>, cint<64>, cint<32>);
  constexpr auto r = remove<1>(s);
  EXPECT_EQ(r.ndim(), 2);
  EXPECT_EQ(r[0], 128);
  EXPECT_EQ(r[1], 32);
}

TEST_F(CompileTimeTest, ConstexprToArray) {
  constexpr auto s = make_shape(cint<128>, 100);
  constexpr auto arr = s._stor.to_array();
  EXPECT_EQ(arr[0], 128);
  EXPECT_EQ(arr[1], 100);
}

TEST_F(CompileTimeTest, ConstexprLayout) {
  constexpr auto layout = make_layout(make_shape(cint<128>, cint<64>),
                                      make_strides(cint<64>, cint<1>));
  constexpr nint_t sz = size<0>(layout);
  constexpr nint_t st = stride<1>(layout);
  EXPECT_EQ(sz, 128);
  EXPECT_EQ(st, 1);
}

#endif  // VECOPS_DEBUG

// ======================================================================
// Metaprogramming Details Suite
// ======================================================================

class MetaDetailsTest : public ::testing::Test {};

TEST_F(MetaDetailsTest, PickConstValue) {
  using namespace vecops::gemm::details;
  EXPECT_EQ((PickConstValue<Const<42>>::value), 42);
  EXPECT_EQ((PickConstValue<Any>::value), 0);
  EXPECT_EQ((PickConstValue<Dynamic<16>>::value), 0);
}

TEST_F(MetaDetailsTest, ValuePromote) {
  using namespace vecops::gemm::details;
  EXPECT_TRUE((std::is_same_v<ValuePromote<int>::Type, Any>));
  EXPECT_TRUE((std::is_same_v<ValuePromote<Const<5>>::Type, Const<5>>));
  EXPECT_TRUE((std::is_same_v<ValuePromote<Dynamic<16>>::Type, Dynamic<16>>));
}

TEST_F(MetaDetailsTest, IsArrayMetaDetails) {
  using namespace vecops::gemm::details;
  EXPECT_TRUE((IsArrayMeta<ArrayMeta<Any, Any>>::value));
  EXPECT_FALSE((IsArrayMeta<int>::value));
}

TEST_F(MetaDetailsTest, IsShapeDetails) {
  using namespace vecops::gemm::details;
  EXPECT_TRUE((IsShape<Shape<Any, Any>>::value));
  EXPECT_FALSE((IsShape<ArrayMeta<Any, Any>>::value));
}

TEST_F(MetaDetailsTest, IsStridesDetails) {
  using namespace vecops::gemm::details;
  EXPECT_TRUE((IsStrides<Strides<Any, Any>>::value));
  EXPECT_FALSE((IsStrides<ArrayMeta<Any, Any>>::value));
}

TEST_F(MetaDetailsTest, IsLayoutDetails) {
  using namespace vecops::gemm::details;
  EXPECT_TRUE((IsLayout<Layout<Shape<Any>, Strides<Any>>>::value));
  EXPECT_FALSE((IsLayout<ArrayMeta<Any>>::value));
}

// ======================================================================
// Death Tests — VECOPS_ASSERT failures (debug-only)
// ======================================================================

#ifdef VECOPS_DEBUG

class LayoutDeathTest : public ::testing::Test {};

TEST_F(LayoutDeathTest, ConstConstructorWrongValue) {
  EXPECT_DEATH(Const<128>(64), "!= 128");
}

TEST_F(LayoutDeathTest, DynamicConstructorUnaligned) {
  EXPECT_DEATH(Dynamic<16>(63), "Dynamic<");
}

TEST_F(LayoutDeathTest, ShapeNegativeValue) {
  EXPECT_DEATH(make_shape(cint<128>, -1), "non-negative");
}

TEST_F(LayoutDeathTest, PackedStorageOutOfBounds) {
  vecops::gemm::details::PackedStorage<Any, Any> ps(Any{1}, Any{2});
  EXPECT_DEATH((void)ps[-1], "!in 0..");
  EXPECT_DEATH((void)ps[99], "!in 0..");
}

TEST_F(LayoutDeathTest, OffsetAtOutOfBounds) {
  auto layout = make_layout(make_shape(3, 4), make_strides(4, 1));
  EXPECT_DEATH((void)offset_at(layout, 2, 4), "index out of range");
  EXPECT_DEATH((void)offset_at(layout, -1, 0), "index out of range");
}

TEST_F(LayoutDeathTest, DynamicBoundsViolationLower) {
  using D = Dynamic<8, 0, 128>;
  EXPECT_DEATH(D(-8), "Dynamic<");
}

TEST_F(LayoutDeathTest, DynamicBoundsViolationUpper) {
  using D = Dynamic<8, 0, 128>;
  EXPECT_DEATH(D(136), "Dynamic<");
}

TEST_F(LayoutDeathTest, DynamicBoundsViolationUnaligned) {
  using D = Dynamic<8, 0, 128>;
  EXPECT_DEATH(D(4), "Dynamic<");
}

#endif  // VECOPS_DEBUG

// ======================================================================
// Continuity Traits Suite
// ======================================================================

class ContiguityTest : public ::testing::Test {};

// ======================================================================
// Type Trait Selection Suite
// ======================================================================

class TypeTraitSelectionTest : public ::testing::Test {};

template <typename T, typename = void>
struct HasTypeMember : std::false_type {};

template <typename T>
struct HasTypeMember<T, std::void_t<typename T::type>> : std::true_type {};

TEST_F(TypeTraitSelectionTest, ChainIfSelectsFirstTrueOption) {
  using T = chain_if_t<
      chain_opt<false, int>,
      chain_opt<true, float>,
      chain_opt<true, double>>;
  EXPECT_TRUE((std::is_same_v<T, float>));
}

TEST_F(TypeTraitSelectionTest, ChainIfHasNoTypeWhenNoOptionMatches) {
  using T = chain_if<
      chain_opt<false, int>,
      chain_opt<false, float>>;
  EXPECT_FALSE((HasTypeMember<T>::value));
}

TEST_F(TypeTraitSelectionTest, ChainIfDuplicateTrueOptionsUseFirst) {
  using T = chain_if_t<
      chain_opt<true, int>,
      chain_opt<true, float>>;
  EXPECT_TRUE((std::is_same_v<T, int>));
}

// ======================================================================
// Lenient Meta Matching Suite
// ======================================================================

class LenientMetaTest : public ::testing::Test {};

TEST_F(LenientMetaTest, MoreLenientMetaMovedToLayout) {
  using SrcShape = Shape<Const<4>, Const<8>>;
  using DstShape = Shape<Dynamic<4>, Any>;
  using SrcStrides = Strides<Const<8>, Const<1>>;
  using DstStrides = Strides<Any, Any>;

  EXPECT_TRUE((vecops::gemm::details::IsMoreLenientMeta<SrcShape, DstShape>::value));
  EXPECT_TRUE((vecops::gemm::details::IsMoreLenientMeta<SrcStrides, DstStrides>::value));
  EXPECT_FALSE((vecops::gemm::details::IsMoreLenientMeta<DstShape, SrcShape>::value));
}

TEST_F(LenientMetaTest, IsLenientMatchesWildcardAndValueConstraints) {
  using St = Strides<Const<64>, Const<8>, Const<2>, Const<2>>;

  EXPECT_TRUE((is_lenient_v<St, any, any, Dynamic<1, -1, 3>, Const<2>>));
  EXPECT_FALSE((is_lenient_v<St, any, any, Dynamic<4, -1, 3>, Const<2>>));
}

TEST_F(LenientMetaTest, IsLenientWildcardIsDistinctFromAnyValue) {
  EXPECT_FALSE((std::is_same_v<any, Any>));

  using St = Strides<Const<7>>;
  EXPECT_TRUE((is_lenient_v<St, any>));
  EXPECT_TRUE((is_lenient_v<St, Any>));

  using DynamicSt = Strides<Dynamic<2>>;
  EXPECT_TRUE((is_lenient_v<DynamicSt, any>));
  EXPECT_TRUE((is_lenient_v<DynamicSt, Any>));
  EXPECT_FALSE((is_lenient_v<Any, any>));
}

TEST_F(LenientMetaTest, IsLenientRequiresMatchingRank) {
  using St = Strides<Const<8>, Const<1>>;
  EXPECT_FALSE((is_lenient_v<St, any>));
  EXPECT_FALSE((is_lenient_v<St, any, any, any>));
}

// ======================================================================
// Layout Conversion Suite
// ======================================================================

class LayoutConversionTest : public ::testing::Test {};

TEST_F(LayoutConversionTest, AsCastsToTargetMetaTypes) {
  auto layout = make_layout(make_shape(Any{4}, Any{5}),
                            make_strides(Any{5}, Any{1}));

  auto typed = layout.as<Shape<Const<4>, Const<5>>, Strides<Const<5>, Const<1>>>();

  EXPECT_TRUE((std::is_same_v<decltype(typed)::Shape, Shape<Const<4>, Const<5>>>));
  EXPECT_TRUE((std::is_same_v<decltype(typed)::Strides, Strides<Const<5>, Const<1>>>));
  EXPECT_EQ(typed.shape()[0], 4);
  EXPECT_EQ(typed.shape()[1], 5);
  EXPECT_EQ(typed.strides()[0], 5);
  EXPECT_EQ(typed.strides()[1], 1);
}

TEST_F(LayoutConversionTest, ImplicitConversionToMoreLenientLayout) {
  auto strict = make_layout(make_shape(cint<4>, cint<5>),
                            make_strides(cint<5>, cint<1>));

  Layout<Shape<Dynamic<4>, Any>, Strides<Any, Any>> lenient = strict;

  EXPECT_EQ(lenient.shape()[0], 4);
  EXPECT_EQ(lenient.shape()[1], 5);
  EXPECT_EQ(lenient.strides()[0], 5);
  EXPECT_EQ(lenient.strides()[1], 1);
}

TEST_F(LayoutConversionTest, ImplicitConversionRejectsMoreStrictLayout) {
  using Strict = Layout<Shape<Const<4>, Const<5>>, Strides<Const<5>, Const<1>>>;
  using Lenient = Layout<Shape<Any, Any>, Strides<Any, Any>>;

  EXPECT_FALSE((std::is_convertible_v<Lenient, Strict>));
  EXPECT_TRUE((std::is_convertible_v<Strict, Lenient>));
}

TEST_F(ContiguityTest, CtLastContiguous_FullConst) {
  auto layout = make_layout(make_shape(cint<4>, cint<6>),
                            make_strides(cint<6>, cint<1>));
  EXPECT_TRUE((is_ct_last_contiguous<decltype(layout), 2>::value));
  EXPECT_TRUE((is_ct_last_contiguous<decltype(layout), 1>::value));
  EXPECT_TRUE((is_ct_contiguous<decltype(layout)>::value));
}

TEST_F(ContiguityTest, CtLastContiguous_LastOneConst) {
  auto layout = make_layout(make_shape(Any{4}, Any{6}),
                            make_strides(Any{6}, cint<1>));
  EXPECT_TRUE((is_ct_last_contiguous<decltype(layout), 1>::value));
}

TEST_F(ContiguityTest, CtLastContiguous_ShapeNotConst) {
  // shape[1] must be non-const because the contiguity check at D=0
  // uses Zd1=shape[1]; shape[0] is not checked in that position.
  auto layout = make_layout(make_shape(cint<4>, Any{6}),
                            make_strides(cint<6>, cint<1>));
  EXPECT_FALSE((is_ct_last_contiguous<decltype(layout), 2>::value));
}

TEST_F(ContiguityTest, CtLastContiguous_StrideNotConst) {
  auto layout = make_layout(make_shape(cint<4>, cint<6>),
                            make_strides(Any{6}, cint<1>));
  EXPECT_FALSE((is_ct_last_contiguous<decltype(layout), 2>::value));
}

TEST_F(ContiguityTest, CtLastContiguous_N_Exceeds_Ndim) {
  auto layout = make_layout(make_shape(cint<4>, cint<6>),
                            make_strides(cint<6>, cint<1>));
  EXPECT_FALSE((is_ct_last_contiguous<decltype(layout), 3>::value));
}

TEST_F(ContiguityTest, CtLastContiguous_N_Zero) {
  auto layout = make_layout(make_shape(cint<4>, cint<6>),
                            make_strides(cint<6>, cint<1>));
  EXPECT_TRUE((is_ct_last_contiguous<decltype(layout), 0>::value));
}

TEST_F(ContiguityTest, CtLastContiguous_NonUnitLastStride) {
  auto layout = make_layout(make_shape(cint<4>, cint<6>),
                            make_strides(cint<6>, cint<2>));
  EXPECT_FALSE((is_ct_last_contiguous<decltype(layout), 1>::value));
}

TEST_F(ContiguityTest, CtLastContiguous_4D_FullConst) {
  auto s = make_shape(cint<2>, cint<3>, cint<4>, cint<5>);
  auto st = make_strides(cint<60>, cint<20>, cint<5>, cint<1>);
  auto layout = make_layout(s, st);
  EXPECT_TRUE((is_ct_last_contiguous<decltype(layout), 4>::value));
  EXPECT_TRUE((is_ct_last_contiguous<decltype(layout), 2>::value));
  EXPECT_TRUE((is_ct_contiguous<decltype(layout)>::value));
}

TEST_F(ContiguityTest, CtLastContiguous_4D_Last2) {
  auto s = make_shape(cint<2>, cint<3>, cint<4>, cint<5>);
  auto st = make_strides(cint<20>, cint<5>, cint<1>, Any{1});
  auto layout = make_layout(s, st);
  EXPECT_FALSE((is_ct_last_contiguous<decltype(layout), 4>::value));
}

TEST_F(ContiguityTest, CtIsContiguous_NotContiguous) {
  auto layout = make_layout(make_shape(cint<4>, cint<6>),
                            make_strides(cint<5>, cint<1>));
  EXPECT_FALSE((is_ct_contiguous<decltype(layout)>::value));
}

TEST_F(ContiguityTest, CtContiguous_1D) {
  auto layout = make_layout(make_shape(cint<10>), make_strides(cint<1>));
  EXPECT_TRUE((is_ct_contiguous<decltype(layout)>::value));
}

TEST_F(ContiguityTest, CtContiguous_3D_FullConst) {
  auto s = make_shape(cint<3>, cint<4>, cint<5>);
  auto st = make_strides(cint<20>, cint<5>, cint<1>);
  auto layout = make_layout(s, st);
  EXPECT_TRUE((is_ct_contiguous<decltype(layout)>::value));
}

TEST_F(ContiguityTest, CtContiguous_3D_NotContiguous_Mismatch) {
  auto s = make_shape(cint<3>, cint<4>, cint<5>);
  auto st = make_strides(cint<30>, cint<5>, cint<1>);
  auto layout = make_layout(s, st);
  EXPECT_FALSE((is_ct_contiguous<decltype(layout)>::value));
}

// ======================================================================
// Runtime Contiguity Suite
// ======================================================================

class RuntimeContiguityTest : public ::testing::Test {};

TEST_F(RuntimeContiguityTest, IsLastContiguous_AllRuntimeRowMajor) {
  auto layout = make_layout(make_shape(4, 6), make_strides(6, 1));
  EXPECT_TRUE(is_last_contiguous<2>(layout));
  EXPECT_TRUE(is_contiguous(layout));
}

TEST_F(RuntimeContiguityTest, IsLastContiguous_AllRuntimeNonContiguous) {
  auto layout = make_layout(make_shape(4, 6), make_strides(10, 2));
  EXPECT_FALSE(is_last_contiguous<2>(layout));
  EXPECT_FALSE(is_last_contiguous<1>(layout));
}

TEST_F(RuntimeContiguityTest, IsLastContiguous_CompileTimeShortcut) {
  auto layout = make_layout(make_shape(cint<4>, cint<6>),
                            make_strides(cint<6>, cint<1>));
  EXPECT_TRUE(is_last_contiguous<2>(layout));
  EXPECT_TRUE(is_contiguous(layout));
}

TEST_F(RuntimeContiguityTest, IsLastContiguous_N_Greater_Than_Ndim) {
  auto layout = make_layout(make_shape(4, 6), make_strides(6, 1));
  EXPECT_FALSE(is_last_contiguous<3>(layout));
}

TEST_F(RuntimeContiguityTest, IsLastContiguous_3D_Last1) {
  auto layout = make_layout(make_shape(3, 4, 5), make_strides(20, 5, 1));
  EXPECT_TRUE(is_last_contiguous<1>(layout));
}

TEST_F(RuntimeContiguityTest, IsLastContiguous_3D_Last2) {
  auto layout = make_layout(make_shape(3, 4, 5), make_strides(20, 5, 1));
  EXPECT_TRUE(is_last_contiguous<2>(layout));
}

TEST_F(RuntimeContiguityTest, IsLastContiguous_3D_Full) {
  auto layout = make_layout(make_shape(3, 4, 5), make_strides(20, 5, 1));
  EXPECT_TRUE(is_last_contiguous<3>(layout));
}

TEST_F(RuntimeContiguityTest, IsContiguous_NonContiguous) {
  auto layout = make_layout(make_shape(3, 4, 5), make_strides(30, 5, 1));
  EXPECT_FALSE(is_contiguous(layout));
  EXPECT_TRUE(is_last_contiguous<2>(layout));
}

TEST_F(RuntimeContiguityTest, IsContiguous_4D) {
  auto layout = make_layout(make_shape(2, 3, 4, 5), make_strides(60, 20, 5, 1));
  EXPECT_TRUE(is_contiguous(layout));
}

TEST_F(RuntimeContiguityTest, IsContiguous_1D) {
  auto layout = make_layout(make_shape(10), make_strides(1));
  EXPECT_TRUE(is_contiguous(layout));
}

// ======================================================================
// Transpose / Swap Dim Trait Suite
// ======================================================================

class TransposeTraitTest : public ::testing::Test {};

TEST_F(TransposeTraitTest, SwapDim_Basic) {
  auto s = make_shape(cint<1>, cint<2>, cint<3>);
  auto r = gemm::details::swap_dim<0, 2>(s);
  EXPECT_EQ(r.ndim(), 3);
  EXPECT_EQ(get<0>(r), 3);
  EXPECT_EQ(get<1>(r), 2);
  EXPECT_EQ(get<2>(r), 1);
  EXPECT_TRUE(r.template is_const<0>());
  EXPECT_TRUE(r.template is_const<2>());
}

TEST_F(TransposeTraitTest, SwapDim_PreserveConst) {
  auto s = make_shape(cint<128>, Any{64}, cint<32>);
  auto r = gemm::details::swap_dim<0, 1>(s);
  // After swap: Shape<Any, Const<128>, Const<32>>
  EXPECT_FALSE(r.template is_const<0>());  // Any moved to pos 0
  EXPECT_TRUE(r.template is_const<1>());   // Const<128> moved to pos 1
  EXPECT_TRUE(r.template is_const<2>());   // Const<32> unchanged
  EXPECT_EQ(get<0>(r), 64);
  EXPECT_EQ(get<1>(r), 128);
  EXPECT_EQ(get<2>(r), 32);
}

TEST_F(TransposeTraitTest, SwapDim_RuntimeValues) {
  auto s = make_shape(10, 20, 30);
  auto r = gemm::details::swap_dim<0, 2>(s);
  EXPECT_EQ(get<0>(r), 30);
  EXPECT_EQ(get<1>(r), 20);
  EXPECT_EQ(get<2>(r), 10);
}

TEST_F(TransposeTraitTest, SwapDim_AllConst) {
  auto s = make_shape(cint<1>, cint<2>, cint<3>, cint<4>);
  auto r = gemm::details::swap_dim<1, 2>(s);
  EXPECT_EQ(get<0>(r), 1);
  EXPECT_EQ(get<1>(r), 3);
  EXPECT_EQ(get<2>(r), 2);
  EXPECT_EQ(get<3>(r), 4);
}

TEST_F(TransposeTraitTest, SwapDim_Strides) {
  auto st = make_strides(cint<60>, cint<20>, cint<5>, cint<1>);
  auto r = gemm::details::swap_dim<0, 3>(st);
  EXPECT_EQ(get<0>(r), 1);
  EXPECT_EQ(get<1>(r), 20);
  EXPECT_EQ(get<2>(r), 5);
  EXPECT_EQ(get<3>(r), 60);
}

// ======================================================================
// Layout Transpose Suite
// ======================================================================

class LayoutTransposeTest : public ::testing::Test {};

TEST_F(LayoutTransposeTest, TransposeCT_Basic) {
  auto layout = make_layout(make_shape(cint<4>, cint<6>),
                            make_strides(cint<6>, cint<1>));
  auto t = transpose<0, 1>(layout);
  EXPECT_EQ(t.ndim(), 2);
  EXPECT_EQ(size<0>(t), 6);
  EXPECT_EQ(size<1>(t), 4);
  EXPECT_EQ(stride<0>(t), 1);
  EXPECT_EQ(stride<1>(t), 6);
}

TEST_F(LayoutTransposeTest, TransposeCT_3D) {
  auto layout = make_layout(make_shape(cint<2>, cint<3>, cint<4>),
                            make_strides(cint<12>, cint<4>, cint<1>));
  auto t = transpose<0, 2>(layout);
  EXPECT_EQ(t.ndim(), 3);
  EXPECT_EQ(size<0>(t), 4);
  EXPECT_EQ(size<1>(t), 3);
  EXPECT_EQ(size<2>(t), 2);
  EXPECT_EQ(stride<0>(t), 1);
  EXPECT_EQ(stride<1>(t), 4);
  EXPECT_EQ(stride<2>(t), 12);
}

TEST_F(LayoutTransposeTest, TransposeCT_ConstPreserved) {
  auto layout = make_layout(make_shape(cint<128>, cint<64>, cint<32>),
                            make_strides(cint<2048>, cint<32>, cint<1>));
  auto t = transpose<0, 1>(layout);
  EXPECT_TRUE(t.shape().template is_const<0>());
  EXPECT_TRUE(t.shape().template is_const<1>());
  EXPECT_TRUE(t.strides().template is_const<0>());
}

TEST_F(LayoutTransposeTest, TransposeCT_SameAxis) {
  auto layout = make_layout(make_shape(4, 6), make_strides(6, 1));
  auto t = transpose<1, 1>(layout);
  EXPECT_EQ(size<0>(t), 4);
  EXPECT_EQ(size<1>(t), 6);
}

TEST_F(LayoutTransposeTest, TransposeRT_Basic) {
  auto layout = make_layout(make_shape(4, 6), make_strides(6, 1));
  auto t = transpose(layout, 0, 1);
  EXPECT_EQ(t.ndim(), 2);
  EXPECT_EQ(size<0>(t), 6);
  EXPECT_EQ(size<1>(t), 4);
  EXPECT_EQ(stride<0>(t), 1);
  EXPECT_EQ(stride<1>(t), 6);
}

TEST_F(LayoutTransposeTest, TransposeRT_DegradesToAny) {
  auto layout = make_layout(make_shape(cint<4>, cint<6>),
                            make_strides(cint<6>, cint<1>));
  auto t = transpose(layout, 0, 1);
  EXPECT_FALSE(t.shape().template is_const<0>());
  EXPECT_FALSE(t.shape().template is_const<1>());
  EXPECT_FALSE(t.strides().template is_const<0>());
  EXPECT_FALSE(t.strides().template is_const<1>());
}

TEST_F(LayoutTransposeTest, TransposeRT_3D) {
  auto layout = make_layout(make_shape(2, 3, 4), make_strides(12, 4, 1));
  auto t = transpose(layout, 0, 2);
  EXPECT_EQ(size<0>(t), 4);
  EXPECT_EQ(size<1>(t), 3);
  EXPECT_EQ(size<2>(t), 2);
}

TEST_F(LayoutTransposeTest, TransposeRT_SameAxis) {
  auto layout = make_layout(make_shape(4, 6), make_strides(6, 1));
  auto t = transpose(layout, 1, 1);
  EXPECT_EQ(size<0>(t), 4);
  EXPECT_EQ(size<1>(t), 6);
}

#ifdef VECOPS_DEBUG
TEST_F(LayoutTransposeTest, TransposeRT_InvalidIndexDeath) {
  auto layout = make_layout(make_shape(4, 6), make_strides(6, 1));
  EXPECT_DEATH(transpose(layout, -1, 0), "out of range");
  EXPECT_DEATH(transpose(layout, 0, 2), "out of range");
}
#endif
