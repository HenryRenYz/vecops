#ifndef VECOPS_VEC_DETAILS_SVE_MATH_LOG_H
#define VECOPS_VEC_DETAILS_SVE_MATH_LOG_H

/**
 * @file Log.h
 * @brief SVE backend implementations for the log family (log, log2, log10).
 *
 * The f64 Strict kernels, the special-case structure, and all f64 minimax
 * coefficients and 128-entry invc/logc tables are derived from the Arm
 * optimized-routines project (https://github.com/ARM-software/optimized-routines),
 * files math/aarch64/sve/{log,log2,log10}.c and
 * math/aarch64/v_log{,2,10}_data.c (dual-licensed under "MIT OR Apache-2.0
 * WITH LLVM-exception"). The f32 kernels follow the table-free shape of
 * math/aarch64/sve/logf.c with its AdvSIMD minimax polynomial for Strict.
 *
 * Shared design, mirroring the exp/reciprocal families: a branch-free
 * inline core covers the normal domain and one NOINLINE special routine
 * owns every tail (subnormals, zero, negatives, +inf, NaN). The special
 * routine renormalizes subnormal lanes by an exact power of two (2^23 /
 * 2^52), reruns the exact-tier core, and adds the matching logarithmic
 * correction; zero/inf/NaN lanes select their constants through the same
 * masked add, which also propagates NaN from negative lanes.
 *
 * Reduction: bits(x) - off splits x into 2^k * z with z inside a fixed
 * window (table kernels: [0x3fe6900900000000-window] via invc/logc
 * gathers; polynomial kernels: [2/3, 4/3) via a float-subtraction r = z-1
 * that is exact by Sterbenz). The table index is masked to the table size,
 * so any input bit pattern gathers in range.
 *
 * Tiers: Strict is the upstream table algorithm for f64 (measured
 * 2.64/2.58/2.46 ULP for log/log2/log10) and the upstream degree-7
 * polynomial for f32 (measured 3.34 ULP), both within the documented
 * 4-ULP contract. Fast drops the table and keeps six Taylor terms
 * (relative error ~2^-13.9); Estimate keeps three (~2^-10.3). Non-e bases
 * scale the natural-log result once at the end, which preserves the
 * relative-error tier bound; the f64 Strict kernels instead use the
 * upstream per-base tables and coefficients verbatim.
 *
 * f16 and bf16 widen through the f32 kernels; every f16 input is normal in
 * f32 and log outputs are never subnormal, so the f32 pipelines own all
 * narrow-format tails exactly.
 */

#include <arm_sve.h>
#include <cstdint>
#include <limits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/sve/Basic.h"
#include "vecops/vec/details/sve/Bf16.h"

namespace vecops::vec::details {

/* **************************************************************************** */
//    f64 tables (upstream optimized-routines, verbatim)                       //
/* **************************************************************************** */

/**
 * Interleaved invc/logc layout, as upstream: the gather index is the
 * even element index 2*j, the logc gather just uses a base pointer that is
 * one double higher, so both loads share one index computation.
 */
struct SVELogTabEntry {
  double invc, logc;
};

inline constexpr SVELogTabEntry kSVELogTab64[128] = {
    {0x1.6a133d0dec120p+0, -0x1.62fe995eb963ap-2}, {0x1.6815f2f3e42edp+0, -0x1.5d5a48dad6b67p-2},
    {0x1.661e39be1ac9ep+0, -0x1.57bde257d2769p-2}, {0x1.642bfa30ac371p+0, -0x1.52294fbf2af55p-2},
    {0x1.623f1d916f323p+0, -0x1.4c9c7b598aa38p-2}, {0x1.60578da220f65p+0, -0x1.47174fc5ff560p-2},
    {0x1.5e75349dea571p+0, -0x1.4199b7fa7b5cap-2}, {0x1.5c97fd387a75ap+0, -0x1.3c239f48cfb99p-2},
    {0x1.5abfd2981f200p+0, -0x1.36b4f154d2aebp-2}, {0x1.58eca051dc99cp+0, -0x1.314d9a0ff32fbp-2},
    {0x1.571e526d9df12p+0, -0x1.2bed85cca3cffp-2}, {0x1.5554d555b3fcbp+0, -0x1.2694a11421af9p-2},
    {0x1.539015e2a20cdp+0, -0x1.2142d8d014fb2p-2}, {0x1.51d0014ee0164p+0, -0x1.1bf81a2c77776p-2},
    {0x1.50148538cd9eep+0, -0x1.16b452a39c6a4p-2}, {0x1.4e5d8f9f698a1p+0, -0x1.11776ffa6c67ep-2},
    {0x1.4cab0edca66bep+0, -0x1.0c416035020e0p-2}, {0x1.4afcf1a9db874p+0, -0x1.071211aa10fdap-2},
    {0x1.495327136e16fp+0, -0x1.01e972e293b1bp-2}, {0x1.47ad9e84af28fp+0, -0x1.f98ee587fd434p-3},
    {0x1.460c47b39ae15p+0, -0x1.ef5800ad716fbp-3}, {0x1.446f12b278001p+0, -0x1.e52e160484698p-3},
    {0x1.42d5efdd720ecp+0, -0x1.db1104b19352ep-3}, {0x1.4140cfe001a0fp+0, -0x1.d100ac59e0bd6p-3},
    {0x1.3fafa3b421f69p+0, -0x1.c6fced287c3bdp-3}, {0x1.3e225c9c8ece5p+0, -0x1.bd05a7b317c29p-3},
    {0x1.3c98ec29a211ap+0, -0x1.b31abd229164fp-3}, {0x1.3b13442a413fep+0, -0x1.a93c0edadb0a3p-3},
    {0x1.399156baa3c54p+0, -0x1.9f697ee30d7ddp-3}, {0x1.38131639b4cdbp+0, -0x1.95a2efa9aa40ap-3},
    {0x1.36987540fbf53p+0, -0x1.8be843d796044p-3}, {0x1.352166b648f61p+0, -0x1.82395ecc477edp-3},
    {0x1.33adddb3eb575p+0, -0x1.7896240966422p-3}, {0x1.323dcd99fc1d3p+0, -0x1.6efe77aca8c55p-3},
    {0x1.30d129fefc7d2p+0, -0x1.65723e117ec5cp-3}, {0x1.2f67e6b72fe7dp+0, -0x1.5bf15c0955706p-3},
    {0x1.2e01f7cf8b187p+0, -0x1.527bb6c111da1p-3}, {0x1.2c9f518ddc86ep+0, -0x1.491133c939f8fp-3},
    {0x1.2b3fe86e5f413p+0, -0x1.3fb1b90c7fc58p-3}, {0x1.29e3b1211b25cp+0, -0x1.365d2cc485f8dp-3},
    {0x1.288aa08b373cfp+0, -0x1.2d13758970de7p-3}, {0x1.2734abcaa8467p+0, -0x1.23d47a721fd47p-3},
    {0x1.25e1c82459b81p+0, -0x1.1aa0229f25ec2p-3}, {0x1.2491eb1ad59c5p+0, -0x1.117655ddebc3bp-3},
    {0x1.23450a54048b5p+0, -0x1.0856fbf83ab6bp-3}, {0x1.21fb1bb09e578p+0, -0x1.fe83fabbaa106p-4},
    {0x1.20b415346d8f7p+0, -0x1.ec6e8507a56cdp-4}, {0x1.1f6fed179a1acp+0, -0x1.da6d68c7cc2eap-4},
    {0x1.1e2e99b93c7b3p+0, -0x1.c88078462be0cp-4}, {0x1.1cf011a7a882ap+0, -0x1.b6a786a423565p-4},
    {0x1.1bb44b97dba5ap+0, -0x1.a4e2676ac7f85p-4}, {0x1.1a7b3e66cdd4fp+0, -0x1.9330eea777e76p-4},
    {0x1.1944e11dc56cdp+0, -0x1.8192f134d5ad9p-4}, {0x1.18112aebb1a6ep+0, -0x1.70084464f0538p-4},
    {0x1.16e013231b7e9p+0, -0x1.5e90bdec5cb1fp-4}, {0x1.15b1913f156cfp+0, -0x1.4d2c3433c5536p-4},
    {0x1.14859cdedde13p+0, -0x1.3bda7e219879ap-4}, {0x1.135c2dc68cfa4p+0, -0x1.2a9b732d27194p-4},
    {0x1.12353bdb01684p+0, -0x1.196eeb2b10807p-4}, {0x1.1110bf25b85b4p+0, -0x1.0854be8ef8a7ep-4},
    {0x1.0feeafd2f8577p+0, -0x1.ee998cb277432p-5}, {0x1.0ecf062c51c3bp+0, -0x1.ccadb79919fb9p-5},
    {0x1.0db1baa076c8bp+0, -0x1.aae5b1d8618b0p-5}, {0x1.0c96c5bb3048ep+0, -0x1.89413015d7442p-5},
    {0x1.0b7e20263e070p+0, -0x1.67bfe7bf158dep-5}, {0x1.0a67c2acd0ce3p+0, -0x1.46618f83941bep-5},
    {0x1.0953a6391e982p+0, -0x1.2525df1b0618ap-5}, {0x1.0841c3caea380p+0, -0x1.040c8e2f77c6ap-5},
    {0x1.07321489b13eap+0, -0x1.c62aad39f738ap-6}, {0x1.062491aee9904p+0, -0x1.847fe3bdead9cp-6},
    {0x1.05193497a7cc5p+0, -0x1.43183683400acp-6}, {0x1.040ff6b5f5e9fp+0, -0x1.01f31c4e1d544p-6},
    {0x1.0308d19aa6127p+0, -0x1.82201d1e6b69ap-7}, {0x1.0203beedb0c67p+0, -0x1.00dd0f3e1bfd6p-7},
    {0x1.010037d38bcc2p+0, -0x1.ff6fe1feb4e53p-9}, {1.0, 0.0},
    {0x1.fc06d493cca10p-1, 0x1.fe91885ec8e20p-8}, {0x1.f81e6ac3b918fp-1, 0x1.fc516f716296dp-7},
    {0x1.f44546ef18996p-1, 0x1.7bb4dd70a015bp-6}, {0x1.f07b10382c84bp-1, 0x1.f84c99b34b674p-6},
    {0x1.ecbf7070e59d4p-1, 0x1.39f9ce4fb2d71p-5}, {0x1.e91213f715939p-1, 0x1.7756c0fd22e78p-5},
    {0x1.e572a9a75f7b7p-1, 0x1.b43ee82db8f3ap-5}, {0x1.e1e0e2c530207p-1, 0x1.f0b3fced60034p-5},
    {0x1.de5c72d8a8be3p-1, 0x1.165bd78d4878ep-4}, {0x1.dae50fa5658ccp-1, 0x1.3425d2715ebe6p-4},
    {0x1.d77a71145a2dap-1, 0x1.51b8bd91b7915p-4}, {0x1.d41c51166623ep-1, 0x1.6f15632c76a47p-4},
    {0x1.d0ca6ba0bb29fp-1, 0x1.8c3c88ecbe503p-4}, {0x1.cd847e8e59681p-1, 0x1.a92ef077625dap-4},
    {0x1.ca4a499693e00p-1, 0x1.c5ed5745fa006p-4}, {0x1.c71b8e399e821p-1, 0x1.e27876de1c993p-4},
    {0x1.c3f80faf19077p-1, 0x1.fed104fce4cdcp-4}, {0x1.c0df92dc2b0ecp-1, 0x1.0d7bd9c17d78bp-3},
    {0x1.bdd1de3cbb542p-1, 0x1.1b76986cef97bp-3}, {0x1.baceb9e1007a3p-1, 0x1.295913d24f750p-3},
    {0x1.b7d5ef543e55ep-1, 0x1.37239fa295d17p-3}, {0x1.b4e749977d953p-1, 0x1.44d68dd78714bp-3},
    {0x1.b20295155478ep-1, 0x1.52722ebe5d780p-3}, {0x1.af279f8e82be2p-1, 0x1.5ff6d12671f98p-3},
    {0x1.ac5638197fdf3p-1, 0x1.6d64c2389484bp-3}, {0x1.a98e2f102e087p-1, 0x1.7abc4da40fddap-3},
    {0x1.a6cf5606d05c1p-1, 0x1.87fdbda1e8452p-3}, {0x1.a4197fc04d746p-1, 0x1.95295b06a5f37p-3},
    {0x1.a16c80293dc01p-1, 0x1.a23f6d34abbc5p-3}, {0x1.9ec82c4dc5bc9p-1, 0x1.af403a28e04f2p-3},
    {0x1.9c2c5a491f534p-1, 0x1.bc2c06a85721ap-3}, {0x1.9998e1480b618p-1, 0x1.c903161240163p-3},
    {0x1.970d9977c6c2dp-1, 0x1.d5c5aa93287ebp-3}, {0x1.948a5c023d212p-1, 0x1.e274051823fa9p-3},
    {0x1.920f0303d6809p-1, 0x1.ef0e656300c16p-3}, {0x1.8f9b698a98b45p-1, 0x1.fb9509f05aa2ap-3},
    {0x1.8d2f6b81726f6p-1, 0x1.04041821f37afp-2}, {0x1.8acae5bb55badp-1, 0x1.0a340a49b3029p-2},
    {0x1.886db5d9275b8p-1, 0x1.105a7918a126dp-2}, {0x1.8617ba567c13cp-1, 0x1.1677819812b84p-2},
    {0x1.83c8d27487800p-1, 0x1.1c8b405b40c0ep-2}, {0x1.8180de3c5dbe7p-1, 0x1.2295d16cfa6b1p-2},
    {0x1.7f3fbe71cdb71p-1, 0x1.28975066318a2p-2}, {0x1.7d055498071c1p-1, 0x1.2e8fd855d86fcp-2},
    {0x1.7ad182e54f65ap-1, 0x1.347f83d605e59p-2}, {0x1.78a42c3c90125p-1, 0x1.3a666d1244588p-2},
    {0x1.767d342f76944p-1, 0x1.4044adb6f8ec4p-2}, {0x1.745c7ef26b00ap-1, 0x1.461a5f077558cp-2},
    {0x1.7241f15769d0fp-1, 0x1.4be799e20b9c8p-2}, {0x1.702d70d396e41p-1, 0x1.51ac76a6b79dfp-2},
    {0x1.6e1ee3700cd11p-1, 0x1.57690d5744a45p-2}, {0x1.6c162fc9cbe02p-1, 0x1.5d1d758e45217p-2}
};

inline constexpr SVELogTabEntry kSVELog2Tab64[128] = {
    {0x1.6a133d0dec120p+0, -0x1.00130d57f5fadp-1}, {0x1.6815f2f3e42edp+0, -0x1.f802661bd725ep-2},
    {0x1.661e39be1ac9ep+0, -0x1.efea1c6f73a5bp-2}, {0x1.642bfa30ac371p+0, -0x1.e7dd1dcd06f05p-2},
    {0x1.623f1d916f323p+0, -0x1.dfdb4ae024809p-2}, {0x1.60578da220f65p+0, -0x1.d7e484d101958p-2},
    {0x1.5e75349dea571p+0, -0x1.cff8ad452f6ep-2}, {0x1.5c97fd387a75ap+0, -0x1.c817a666c997fp-2},
    {0x1.5abfd2981f200p+0, -0x1.c04152d640419p-2}, {0x1.58eca051dc99cp+0, -0x1.b87595a3f64b2p-2},
    {0x1.571e526d9df12p+0, -0x1.b0b4526c44d07p-2}, {0x1.5554d555b3fcbp+0, -0x1.a8fd6d1a90f5ep-2},
    {0x1.539015e2a20cdp+0, -0x1.a150ca2559fc6p-2}, {0x1.51d0014ee0164p+0, -0x1.99ae4e62cca29p-2},
    {0x1.50148538cd9eep+0, -0x1.9215df1a1e842p-2}, {0x1.4e5d8f9f698a1p+0, -0x1.8a8761fe1f0d9p-2},
    {0x1.4cab0edca66bep+0, -0x1.8302bd1cc9a54p-2}, {0x1.4afcf1a9db874p+0, -0x1.7b87d6fb437f6p-2},
    {0x1.495327136e16fp+0, -0x1.741696673a86dp-2}, {0x1.47ad9e84af28fp+0, -0x1.6caee2b3c6fe4p-2},
    {0x1.460c47b39ae15p+0, -0x1.6550a3666c27ap-2}, {0x1.446f12b278001p+0, -0x1.5dfbc08de02a4p-2},
    {0x1.42d5efdd720ecp+0, -0x1.56b022766c84ap-2}, {0x1.4140cfe001a0fp+0, -0x1.4f6db1c955536p-2},
    {0x1.3fafa3b421f69p+0, -0x1.4834579063054p-2}, {0x1.3e225c9c8ece5p+0, -0x1.4103fd2249a76p-2},
    {0x1.3c98ec29a211ap+0, -0x1.39dc8c3fe6dabp-2}, {0x1.3b13442a413fep+0, -0x1.32bdeed4b5c8fp-2},
    {0x1.399156baa3c54p+0, -0x1.2ba80f41e20ddp-2}, {0x1.38131639b4cdbp+0, -0x1.249ad8332f4a7p-2},
    {0x1.36987540fbf53p+0, -0x1.1d96347e7f3ebp-2}, {0x1.352166b648f61p+0, -0x1.169a0f7d6604ap-2},
    {0x1.33adddb3eb575p+0, -0x1.0fa654a221909p-2}, {0x1.323dcd99fc1d3p+0, -0x1.08baefcf8251ap-2},
    {0x1.30d129fefc7d2p+0, -0x1.01d7cd14deecdp-2}, {0x1.2f67e6b72fe7dp+0, -0x1.f5f9b1ad55495p-3},
    {0x1.2e01f7cf8b187p+0, -0x1.e853ff76a77afp-3}, {0x1.2c9f518ddc86ep+0, -0x1.dabe5d624cba1p-3},
    {0x1.2b3fe86e5f413p+0, -0x1.cd38a5cef4822p-3}, {0x1.29e3b1211b25cp+0, -0x1.bfc2b38d315f9p-3},
    {0x1.288aa08b373cfp+0, -0x1.b25c61f5edd0fp-3}, {0x1.2734abcaa8467p+0, -0x1.a5058d18e9cacp-3},
    {0x1.25e1c82459b81p+0, -0x1.97be1113e47a3p-3}, {0x1.2491eb1ad59c5p+0, -0x1.8a85cafdf5e27p-3},
    {0x1.23450a54048b5p+0, -0x1.7d5c97e8fc45bp-3}, {0x1.21fb1bb09e578p+0, -0x1.704255d6486e4p-3},
    {0x1.20b415346d8f7p+0, -0x1.6336e2cedd7bfp-3}, {0x1.1f6fed179a1acp+0, -0x1.563a1d9b0cc6ap-3},
    {0x1.1e2e99b93c7b3p+0, -0x1.494be541aaa6fp-3}, {0x1.1cf011a7a882ap+0, -0x1.3c6c1964dd0f2p-3},
    {0x1.1bb44b97dba5ap+0, -0x1.2f9a99f19a243p-3}, {0x1.1a7b3e66cdd4fp+0, -0x1.22d747344446p-3},
    {0x1.1944e11dc56cdp+0, -0x1.1622020d4f7f5p-3}, {0x1.18112aebb1a6ep+0, -0x1.097aabb3553f3p-3},
    {0x1.16e013231b7e9p+0, -0x1.f9c24b48014c5p-4}, {0x1.15b1913f156cfp+0, -0x1.e0aaa3bdc858ap-4},
    {0x1.14859cdedde13p+0, -0x1.c7ae257c952d6p-4}, {0x1.135c2dc68cfa4p+0, -0x1.aecc960a03e58p-4},
    {0x1.12353bdb01684p+0, -0x1.9605bb724d541p-4}, {0x1.1110bf25b85b4p+0, -0x1.7d595ca7147cep-4},
    {0x1.0feeafd2f8577p+0, -0x1.64c74165002d9p-4}, {0x1.0ecf062c51c3bp+0, -0x1.4c4f31c86d344p-4},
    {0x1.0db1baa076c8bp+0, -0x1.33f0f70388258p-4}, {0x1.0c96c5bb3048ep+0, -0x1.1bac5abb3037dp-4},
    {0x1.0b7e20263e070p+0, -0x1.0381272495f21p-4}, {0x1.0a67c2acd0ce3p+0, -0x1.d6de4eba2de2ap-5},
    {0x1.0953a6391e982p+0, -0x1.a6ec4e8156898p-5}, {0x1.0841c3caea380p+0, -0x1.772be542e3e1bp-5},
    {0x1.07321489b13eap+0, -0x1.479cadcde852dp-5}, {0x1.062491aee9904p+0, -0x1.183e4265faa5p-5},
    {0x1.05193497a7cc5p+0, -0x1.d2207fdaa1b85p-6}, {0x1.040ff6b5f5e9fp+0, -0x1.742486cb4a6a2p-6},
    {0x1.0308d19aa6127p+0, -0x1.1687d77cfc299p-6}, {0x1.0203beedb0c67p+0, -0x1.7293623a6b5dep-7},
    {0x1.010037d38bcc2p+0, -0x1.70ec80ec8f25dp-8}, {1.0, 0.0},
    {0x1.fc06d493cca10p-1, 0x1.704c1ca6b6bc9p-7}, {0x1.f81e6ac3b918fp-1, 0x1.6eac8ba664beap-6},
    {0x1.f44546ef18996p-1, 0x1.11e67d040772dp-5}, {0x1.f07b10382c84bp-1, 0x1.6bc665e2105dep-5},
    {0x1.ecbf7070e59d4p-1, 0x1.c4f8a9772bf1dp-5}, {0x1.e91213f715939p-1, 0x1.0ebff10fbb951p-4},
    {0x1.e572a9a75f7b7p-1, 0x1.3aaf4d7805d11p-4}, {0x1.e1e0e2c530207p-1, 0x1.664ba81a4d717p-4},
    {0x1.de5c72d8a8be3p-1, 0x1.9196387da6de4p-4}, {0x1.dae50fa5658ccp-1, 0x1.bc902f2b7796p-4},
    {0x1.d77a71145a2dap-1, 0x1.e73ab5f584f28p-4}, {0x1.d41c51166623ep-1, 0x1.08cb78510d232p-3},
    {0x1.d0ca6ba0bb29fp-1, 0x1.1dd2fe2f0dcb5p-3}, {0x1.cd847e8e59681p-1, 0x1.32b4784400df4p-3},
    {0x1.ca4a499693e00p-1, 0x1.47706f3d49942p-3}, {0x1.c71b8e399e821p-1, 0x1.5c0768ee4a4dcp-3},
    {0x1.c3f80faf19077p-1, 0x1.7079e86fc7c6dp-3}, {0x1.c0df92dc2b0ecp-1, 0x1.84c86e1183467p-3},
    {0x1.bdd1de3cbb542p-1, 0x1.98f377a34b499p-3}, {0x1.baceb9e1007a3p-1, 0x1.acfb803bc924bp-3},
    {0x1.b7d5ef543e55ep-1, 0x1.c0e10098b025fp-3}, {0x1.b4e749977d953p-1, 0x1.d4a46efe103efp-3},
    {0x1.b20295155478ep-1, 0x1.e8463f45b8d0bp-3}, {0x1.af279f8e82be2p-1, 0x1.fbc6e3228997fp-3},
    {0x1.ac5638197fdf3p-1, 0x1.079364f2e5aa8p-2}, {0x1.a98e2f102e087p-1, 0x1.1133306010a63p-2},
    {0x1.a6cf5606d05c1p-1, 0x1.1ac309631bd17p-2}, {0x1.a4197fc04d746p-1, 0x1.24432485370c1p-2},
    {0x1.a16c80293dc01p-1, 0x1.2db3b5449132fp-2}, {0x1.9ec82c4dc5bc9p-1, 0x1.3714ee1d7a32p-2},
    {0x1.9c2c5a491f534p-1, 0x1.406700ab52c94p-2}, {0x1.9998e1480b618p-1, 0x1.49aa1d87522b2p-2},
    {0x1.970d9977c6c2dp-1, 0x1.52de746d7ecb2p-2}, {0x1.948a5c023d212p-1, 0x1.5c0434336b343p-2},
    {0x1.920f0303d6809p-1, 0x1.651b8ad6c90d1p-2}, {0x1.8f9b698a98b45p-1, 0x1.6e24a56ab5831p-2},
    {0x1.8d2f6b81726f6p-1, 0x1.771fb04ec29b1p-2}, {0x1.8acae5bb55badp-1, 0x1.800cd6f19c25ep-2},
    {0x1.886db5d9275b8p-1, 0x1.88ec441df11dfp-2}, {0x1.8617ba567c13cp-1, 0x1.91be21b7c93f5p-2},
    {0x1.83c8d27487800p-1, 0x1.9a8298f8c7454p-2}, {0x1.8180de3c5dbe7p-1, 0x1.a339d255c04ddp-2},
    {0x1.7f3fbe71cdb71p-1, 0x1.abe3f59f43db7p-2}, {0x1.7d055498071c1p-1, 0x1.b48129deca9efp-2},
    {0x1.7ad182e54f65ap-1, 0x1.bd119575364c1p-2}, {0x1.78a42c3c90125p-1, 0x1.c5955e23ebcbcp-2},
    {0x1.767d342f76944p-1, 0x1.ce0ca8f4e1557p-2}, {0x1.745c7ef26b00ap-1, 0x1.d6779a5a75774p-2},
    {0x1.7241f15769d0fp-1, 0x1.ded6563550d27p-2}, {0x1.702d70d396e41p-1, 0x1.e728ffafd840ep-2},
    {0x1.6e1ee3700cd11p-1, 0x1.ef6fb96c8d739p-2}, {0x1.6c162fc9cbe02p-1, 0x1.f7aaa57907219p-2}
};

inline constexpr SVELogTabEntry kSVELog10Tab64[128] = {
    {0x1.6a133d0dec120p+0, -0x1.345825f221684p-3}, {0x1.6815f2f3e42edp+0, -0x1.2f71a1f0c554ep-3},
    {0x1.661e39be1ac9ep+0, -0x1.2a91fdb30b1f4p-3}, {0x1.642bfa30ac371p+0, -0x1.25b9260981a04p-3},
    {0x1.623f1d916f323p+0, -0x1.20e7081762193p-3}, {0x1.60578da220f65p+0, -0x1.1c1b914aeefacp-3},
    {0x1.5e75349dea571p+0, -0x1.1756af5de404dp-3}, {0x1.5c97fd387a75ap+0, -0x1.12985059c90bfp-3},
    {0x1.5abfd2981f200p+0, -0x1.0de0628f63df4p-3}, {0x1.58eca051dc99cp+0, -0x1.092ed492e08eep-3},
    {0x1.571e526d9df12p+0, -0x1.0483954caf1dfp-3}, {0x1.5554d555b3fcbp+0, -0x1.ffbd27a9adbcp-4},
    {0x1.539015e2a20cdp+0, -0x1.f67f7f2e3d1ap-4}, {0x1.51d0014ee0164p+0, -0x1.ed4e1071ceebep-4},
    {0x1.50148538cd9eep+0, -0x1.e428bb47413c4p-4}, {0x1.4e5d8f9f698a1p+0, -0x1.db0f6003028d6p-4},
    {0x1.4cab0edca66bep+0, -0x1.d201df6749831p-4}, {0x1.4afcf1a9db874p+0, -0x1.c9001ac5c9672p-4},
    {0x1.495327136e16fp+0, -0x1.c009f3c78c79p-4}, {0x1.47ad9e84af28fp+0, -0x1.b71f4cb642e53p-4},
    {0x1.460c47b39ae15p+0, -0x1.ae400818526b2p-4}, {0x1.446f12b278001p+0, -0x1.a56c091954f87p-4},
    {0x1.42d5efdd720ecp+0, -0x1.9ca3332f096eep-4}, {0x1.4140cfe001a0fp+0, -0x1.93e56a3f23e55p-4},
    {0x1.3fafa3b421f69p+0, -0x1.8b3292a3903bp-4}, {0x1.3e225c9c8ece5p+0, -0x1.828a9112d9618p-4},
    {0x1.3c98ec29a211ap+0, -0x1.79ed4ac35f5acp-4}, {0x1.3b13442a413fep+0, -0x1.715aa51ed28c4p-4},
    {0x1.399156baa3c54p+0, -0x1.68d2861c999e9p-4}, {0x1.38131639b4cdbp+0, -0x1.6054d40ded21p-4},
    {0x1.36987540fbf53p+0, -0x1.57e17576bc9a2p-4}, {0x1.352166b648f61p+0, -0x1.4f7851798bb0bp-4},
    {0x1.33adddb3eb575p+0, -0x1.47194f5690ae3p-4}, {0x1.323dcd99fc1d3p+0, -0x1.3ec456d58ec47p-4},
    {0x1.30d129fefc7d2p+0, -0x1.36794ff3e5f55p-4}, {0x1.2f67e6b72fe7dp+0, -0x1.2e382315725e4p-4},
    {0x1.2e01f7cf8b187p+0, -0x1.2600b8ed82e91p-4}, {0x1.2c9f518ddc86ep+0, -0x1.1dd2fa85efc12p-4},
    {0x1.2b3fe86e5f413p+0, -0x1.15aed136e3961p-4}, {0x1.29e3b1211b25cp+0, -0x1.0d94269d1a30dp-4},
    {0x1.288aa08b373cfp+0, -0x1.0582e4a7659f5p-4}, {0x1.2734abcaa8467p+0, -0x1.faf5eb655742dp-5},
    {0x1.25e1c82459b81p+0, -0x1.eaf888487e8eep-5}, {0x1.2491eb1ad59c5p+0, -0x1.db0d75ef25a82p-5},
    {0x1.23450a54048b5p+0, -0x1.cb348a49e6431p-5}, {0x1.21fb1bb09e578p+0, -0x1.bb6d9c69acdd8p-5},
    {0x1.20b415346d8f7p+0, -0x1.abb88368aa7ap-5}, {0x1.1f6fed179a1acp+0, -0x1.9c1517476af14p-5},
    {0x1.1e2e99b93c7b3p+0, -0x1.8c833051bfa4dp-5}, {0x1.1cf011a7a882ap+0, -0x1.7d02a78e7fb31p-5},
    {0x1.1bb44b97dba5ap+0, -0x1.6d93565e97c5fp-5}, {0x1.1a7b3e66cdd4fp+0, -0x1.5e351695db0c5p-5},
    {0x1.1944e11dc56cdp+0, -0x1.4ee7c2ba67adcp-5}, {0x1.18112aebb1a6ep+0, -0x1.3fab35ba16c01p-5},
    {0x1.16e013231b7e9p+0, -0x1.307f4ad854bc9p-5}, {0x1.15b1913f156cfp+0, -0x1.2163ddf4f988cp-5},
    {0x1.14859cdedde13p+0, -0x1.1258cb5d19e22p-5}, {0x1.135c2dc68cfa4p+0, -0x1.035defdba3188p-5},
    {0x1.12353bdb01684p+0, -0x1.e8e651191bce4p-6}, {0x1.1110bf25b85b4p+0, -0x1.cb30a62be444cp-6},
    {0x1.0feeafd2f8577p+0, -0x1.ad9a9b3043823p-6}, {0x1.0ecf062c51c3bp+0, -0x1.9023ecda1ccdep-6},
    {0x1.0db1baa076c8bp+0, -0x1.72cc592bd82dp-6}, {0x1.0c96c5bb3048ep+0, -0x1.55939eb1f9c6ep-6},
    {0x1.0b7e20263e070p+0, -0x1.38797ca6cc5ap-6}, {0x1.0a67c2acd0ce3p+0, -0x1.1b7db35c2c072p-6},
    {0x1.0953a6391e982p+0, -0x1.fd400812ee9a2p-7}, {0x1.0841c3caea380p+0, -0x1.c3c05fb4620f1p-7},
    {0x1.07321489b13eap+0, -0x1.8a7bf3c40e2e3p-7}, {0x1.062491aee9904p+0, -0x1.517249c15a75cp-7},
    {0x1.05193497a7cc5p+0, -0x1.18a2ea5330c91p-7}, {0x1.040ff6b5f5e9fp+0, -0x1.c01abc8cdc4e2p-8},
    {0x1.0308d19aa6127p+0, -0x1.4f6261750dec9p-8}, {0x1.0203beedb0c67p+0, -0x1.be37b6612afa7p-9},
    {0x1.010037d38bcc2p+0, -0x1.bc3a8398ac26p-10}, {1.0, 0.0},
    {0x1.fc06d493cca10p-1, 0x1.bb796219f30a5p-9}, {0x1.f81e6ac3b918fp-1, 0x1.b984fdcba61cep-8},
    {0x1.f44546ef18996p-1, 0x1.49cf12adf8e8cp-7}, {0x1.f07b10382c84bp-1, 0x1.b6075b5217083p-7},
    {0x1.ecbf7070e59d4p-1, 0x1.10b7466fc30ddp-6}, {0x1.e91213f715939p-1, 0x1.4603e4db6a3a1p-6},
    {0x1.e572a9a75f7b7p-1, 0x1.7aeb10e99e105p-6}, {0x1.e1e0e2c530207p-1, 0x1.af6e49b0f0e36p-6},
    {0x1.de5c72d8a8be3p-1, 0x1.e38f064f41179p-6}, {0x1.dae50fa5658ccp-1, 0x1.0ba75abbb7623p-5},
    {0x1.d77a71145a2dap-1, 0x1.25575ee2dba86p-5}, {0x1.d41c51166623ep-1, 0x1.3ed83f477f946p-5},
    {0x1.d0ca6ba0bb29fp-1, 0x1.582aa79af60efp-5}, {0x1.cd847e8e59681p-1, 0x1.714f400fa83aep-5},
    {0x1.ca4a499693e00p-1, 0x1.8a46ad3901cb9p-5}, {0x1.c71b8e399e821p-1, 0x1.a311903b6b87p-5},
    {0x1.c3f80faf19077p-1, 0x1.bbb086f216911p-5}, {0x1.c0df92dc2b0ecp-1, 0x1.d4242bdda648ep-5},
    {0x1.bdd1de3cbb542p-1, 0x1.ec6d167c2af1p-5}, {0x1.baceb9e1007a3p-1, 0x1.0245ed8221426p-4},
    {0x1.b7d5ef543e55ep-1, 0x1.0e40856c74f64p-4}, {0x1.b4e749977d953p-1, 0x1.1a269a31120fep-4},
    {0x1.b20295155478ep-1, 0x1.25f8718fc076cp-4}, {0x1.af279f8e82be2p-1, 0x1.31b64ffc95bfp-4},
    {0x1.ac5638197fdf3p-1, 0x1.3d60787ca5063p-4}, {0x1.a98e2f102e087p-1, 0x1.48f72ccd187fdp-4},
    {0x1.a6cf5606d05c1p-1, 0x1.547aad6602f1cp-4}, {0x1.a4197fc04d746p-1, 0x1.5feb3989d3acbp-4},
    {0x1.a16c80293dc01p-1, 0x1.6b490f3978c79p-4}, {0x1.9ec82c4dc5bc9p-1, 0x1.76946b3f5e703p-4},
    {0x1.9c2c5a491f534p-1, 0x1.81cd895717c83p-4}, {0x1.9998e1480b618p-1, 0x1.8cf4a4055c30ep-4},
    {0x1.970d9977c6c2dp-1, 0x1.9809f4c48c0ebp-4}, {0x1.948a5c023d212p-1, 0x1.a30db3f9899efp-4},
    {0x1.920f0303d6809p-1, 0x1.ae001905458fcp-4}, {0x1.8f9b698a98b45p-1, 0x1.b8e15a2e3a2cdp-4},
    {0x1.8d2f6b81726f6p-1, 0x1.c3b1ace2b0996p-4}, {0x1.8acae5bb55badp-1, 0x1.ce71456edfa62p-4},
    {0x1.886db5d9275b8p-1, 0x1.d9205759882c4p-4}, {0x1.8617ba567c13cp-1, 0x1.e3bf1513af0dfp-4},
    {0x1.83c8d27487800p-1, 0x1.ee4db0412c414p-4}, {0x1.8180de3c5dbe7p-1, 0x1.f8cc5998de3a5p-4},
    {0x1.7f3fbe71cdb71p-1, 0x1.019da085eaeb1p-3}, {0x1.7d055498071c1p-1, 0x1.06cd4acdb4e3dp-3},
    {0x1.7ad182e54f65ap-1, 0x1.0bf542bef813fp-3}, {0x1.78a42c3c90125p-1, 0x1.11159f14da262p-3},
    {0x1.767d342f76944p-1, 0x1.162e761c10d1cp-3}, {0x1.745c7ef26b00ap-1, 0x1.1b3fddc60d43ep-3},
    {0x1.7241f15769d0fp-1, 0x1.2049ebac86aa6p-3}, {0x1.702d70d396e41p-1, 0x1.254cb4fb7836ap-3},
    {0x1.6e1ee3700cd11p-1, 0x1.2a484e8d0d252p-3}, {0x1.6c162fc9cbe02p-1, 0x1.2f3ccce1c860bp-3}
};

/* **************************************************************************** */
//    f64 families and kernels                                                 //
/* **************************************************************************** */

/** Constant set of one f64 table kernel (Strict tier and special paths). */
struct SVELogF64Family {
  const SVELogTabEntry* table;
  double lead;     //< multiplier of r in the hi term: 1, 1/ln2, 1/ln10
  double kd_scale; //< multiplier of k: ln2, 1, log10(2)
  double c0, c1, c2, c3, c4;
  double sub;      //< special-path correction: -52*log_base(2)
};

inline constexpr SVELogF64Family kSVELogF64E{
    kSVELogTab64,
    1.0,
    0x1.62e42fefa39efp-1,
    -0x1.ffffffffffff7p-2, 0x1.55555555170d4p-2, -0x1.0000000399c27p-2,
    0x1.999b2e90e94cap-3, -0x1.554e550bd501ep-3,
    -0x1.205966f2b4f12p+5};

inline constexpr SVELogF64Family kSVELogF64B2{
    kSVELog2Tab64,
    0x1.71547652b82fep+0,
    1.0,
    -0x1.71547652b83p-1, 0x1.ec709dc340953p-2, -0x1.71547651c8f35p-2,
    0x1.2777ebe12dda5p-2, -0x1.ec738d616fe26p-3,
    -52.0};

inline constexpr SVELogF64Family kSVELogF64B10{
    kSVELog10Tab64,
    0x1.bcb7b1526e50ep-2,
    0x1.34413509f79ffp-2,
    -0x1.bcb7b1526e506p-3, 0x1.287a7636be1d1p-3, -0x1.bcb7b158af938p-4,
    0x1.63c78734e6d07p-4, -0x1.287461742fee4p-4,
    -0x1.f4e9f6303263ep+3};

/** Constant set of one f64 table-free kernel (Fast/Estimate tiers). */
struct SVELogF64PolyFamily {
  double ln2;
  double scale; //< end-of-kernel base scale: 1, log2(e), 1/ln(10)
  double c0, c1, c2, c3, c4, c5;
};

inline constexpr SVELogF64PolyFamily kSVELogF64PolyE{
    0x1.62e42fefa39efp-1,
    1.0,
    -0x1.0000000000000p-1, 0x1.5555555555555p-2, -0x1.0000000000000p-2,
    0x1.999999999999ap-3, -0x1.5555555555555p-3, 0x1.2492492492492p-3};

inline constexpr SVELogF64PolyFamily kSVELogF64PolyB2{
    0x1.62e42fefa39efp-1,
    0x1.71547652b82fep+0,
    -0x1.0000000000000p-1, 0x1.5555555555555p-2, -0x1.0000000000000p-2,
    0x1.999999999999ap-3, -0x1.5555555555555p-3, 0x1.2492492492492p-3};

inline constexpr SVELogF64PolyFamily kSVELogF64PolyB10{
    0x1.62e42fefa39efp-1,
    0x1.bcb7b1526e50ep-2,
    -0x1.0000000000000p-1, 0x1.5555555555555p-2, -0x1.0000000000000p-2,
    0x1.999999999999ap-3, -0x1.5555555555555p-3, 0x1.2492492492492p-3};

/**
 * f64 table kernel: log(x) = (logc + r*lead + k*kd_scale) +
 * r^2*(c0 + r*c1 + r^2*(c2 + r*c3 + r^2*c4)), r = invc*z - 1. Gather
 * indices are always in range by construction, so inactive or garbage
 * lanes can never form wild addresses.
 */
template <const SVELogF64Family& C>
VECOPS_ALWAYS_INLINE svfloat64_t sve_log_f64_table_core(
    svfloat64_t x, svbool_t pg) {
  const auto ix = svreinterpret_u64_f64(x);
  const auto tmp = svsub_n_u64_x(pg, ix, 0x3fe6900900000000ull);
  const auto i = svand_n_u64_x(pg, svlsr_n_u64_x(pg, tmp, 44), 0xfeull);
  const auto k = svcvt_f64_s64_x(
      pg, svasr_n_s64_x(pg, svreinterpret_s64_u64(tmp), 52));
  const auto z = svreinterpret_f64_u64(
      svsub_u64_x(pg, ix, svand_n_u64_x(pg, tmp, 0xfffull << 52)));
  const auto invc = svld1_gather_u64index_f64(pg, &C.table[0].invc, i);
  const auto logc = svld1_gather_u64index_f64(pg, &C.table[0].logc, i);
  const auto r = svmad_n_f64_x(pg, invc, z, -1.0);
  const auto w = svmla_n_f64_x(pg, logc, r, C.lead);
  const auto hi = svmla_n_f64_x(pg, w, k, C.kd_scale);
  const auto r2 = svmul_f64_x(pg, r, r);
  auto y = svmla_n_f64_x(pg, svdup_n_f64(C.c2), r, C.c3);
  y = svmla_n_f64_x(pg, y, r2, C.c4);
  const auto p = svmla_n_f64_x(pg, svdup_n_f64(C.c0), r, C.c1);
  y = svmla_f64_x(pg, p, r2, y);
  return svmla_f64_x(pg, hi, r2, y);
}

/**
 * f64 table-free kernel for the Fast and Estimate tiers: r = z - 1 on the
 * [2/3, 4/3) window (exact by Sterbenz), six Taylor terms for Fast and
 * three for Estimate, and one final base scale for non-e bases.
 */
template <const SVELogF64PolyFamily& C, Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64_t sve_log_f64_poly_core(
    svfloat64_t x, svbool_t pg) {
  const auto ix = svreinterpret_u64_f64(x);
  const auto tmp = svsub_n_u64_x(pg, ix, 0x3fe5555555555555ull);
  const auto n = svcvt_f64_s64_x(
      pg, svasr_n_s64_x(pg, svreinterpret_s64_u64(tmp), 52));
  const auto z = svreinterpret_f64_u64(
      svsub_u64_x(pg, ix, svand_n_u64_x(pg, tmp, 0xfffull << 52)));
  const auto r = svsub_n_f64_x(pg, z, 1.0);
  const auto r2 = svmul_f64_x(pg, r, r);
  svfloat64_t y;
  if constexpr (Tier == Accuracy::Fast) {
    const auto a = svmla_n_f64_x(pg, svdup_n_f64(C.c0), r, C.c1);
    const auto b = svmla_n_f64_x(pg, svdup_n_f64(C.c2), r, C.c3);
    const auto d = svmla_n_f64_x(pg, svdup_n_f64(C.c4), r, C.c5);
    y = svmla_f64_x(pg, b, r2, d);
    y = svmla_f64_x(pg, a, r2, y);
  } else {
    // P = c0 + r*(c1 + r*c2): true Horner, both steps multiply by r.
    y = svmla_n_f64_x(pg, svdup_n_f64(C.c1), r, C.c2);
    y = svmla_f64_x(pg, svdup_n_f64(C.c0), r, y);
  }
  const auto hi = svmla_n_f64_x(pg, r, n, C.ln2);
  auto result = svmla_f64_x(pg, hi, r2, y);
  if constexpr (C.scale != 1.0)
    result = svmul_n_f64_x(pg, result, C.scale);
  return result;
}

/** Tails: subnormals renormalized by 2^52, constants for 0/inf/NaN. */
template <const SVELogF64Family& C>
VECOPS_NOINLINE inline svfloat64_t sve_log_f64_special(
    svbool_t pg, svbool_t special, svfloat64_t x) {
  const auto is_sub = svcmpgt_n_f64(pg, x, 0.0);
  const auto is_minf = svcmpeq_n_f64(pg, x, 0.0);
  const auto is_pinf =
      svcmpeq_n_f64(pg, x, std::numeric_limits<double>::infinity());
  const auto scaled = svmul_m(special, x, 0x1p52);
  auto y = sve_log_f64_table_core<C>(scaled, svptrue_b64());
  auto corr = svsel_f64(is_sub, svdup_n_f64(C.sub), svdup_n_f64(
                         std::numeric_limits<double>::quiet_NaN()));
  corr = svsel_f64(
      is_minf, svdup_n_f64(-std::numeric_limits<double>::infinity()),
      corr);
  corr = svsel_f64(
      is_pinf, svdup_n_f64(std::numeric_limits<double>::infinity()),
      corr);
  y = svadd_m(special, y, corr);
  return y;
}

template <const SVELogF64Family& C, const SVELogF64PolyFamily& P,
          Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat64_t sve_log_f64(svfloat64_t x, svbool_t pg) {
  const auto ix = svreinterpret_u64_f64(x);
  const svbool_t special = svcmpge_n_u64(
      pg, svsub_n_u64_x(pg, ix, 0x0010000000000000ull), 0x7fe0000000000000ull);
  if (svptest_any(special, special))
    return sve_log_f64_special<C>(pg, special, x);
  if constexpr (Tier == Accuracy::Strict)
    return sve_log_f64_table_core<C>(x, pg);
  else
    return sve_log_f64_poly_core<P, Tier>(x, pg);
}

/* **************************************************************************** */
//    base selection                                                            //
/* **************************************************************************** */

template <LogBase Base>
consteval const SVELogF64Family& sve_log_f64_family() {
  if constexpr (Base == LogBase::E) return kSVELogF64E;
  else if constexpr (Base == LogBase::Base2) return kSVELogF64B2;
  else return kSVELogF64B10;
}

template <LogBase Base>
consteval const SVELogF64PolyFamily& sve_log_f64_poly_family() {
  if constexpr (Base == LogBase::E) return kSVELogF64PolyE;
  else if constexpr (Base == LogBase::Base2) return kSVELogF64PolyB2;
  else return kSVELogF64PolyB10;
}

/* **************************************************************************** */
//    f32 families and kernels                                                 //
/* **************************************************************************** */

/**
 * Constant set of one f32 table-free kernel for the Fast and Estimate
 * tiers: the Taylor series of ln(1+r) - r (Estimate truncates it after
 * c2). The Strict tier instead widens and evaluates the f64 table kernel.
 */
struct SVELogF32Family {
  float ln2;
  float scale; //< end-of-kernel base scale: 1, log2(e), 1/ln(10)
  float c0, c1, c2, c3, c4, c5;
};

inline constexpr SVELogF32Family kSVELogF32FastE{
    0x1.62e43p-1f,
    1.0f,
    -0x1.000000p-1f, 0x1.555556p-2f, -0x1.000000p-2f, 0x1.99999ap-3f,
    -0x1.555556p-3f, 0x1.24924ap-3f};

inline constexpr SVELogF32Family kSVELogF32FastB2{
    0x1.62e43p-1f,
    0x1.715476p+0f,
    -0x1.000000p-1f, 0x1.555556p-2f, -0x1.000000p-2f, 0x1.99999ap-3f,
    -0x1.555556p-3f, 0x1.24924ap-3f};

inline constexpr SVELogF32Family kSVELogF32FastB10{
    0x1.62e43p-1f,
    0x1.bcb7b2p-2f,
    -0x1.000000p-1f, 0x1.555556p-2f, -0x1.000000p-2f, 0x1.99999ap-3f,
    -0x1.555556p-3f, 0x1.24924ap-3f};

template <LogBase Base>
consteval const SVELogF32Family& sve_log_f32_fast_family() {
  if constexpr (Base == LogBase::E) return kSVELogF32FastE;
  else if constexpr (Base == LogBase::Base2) return kSVELogF32FastB2;
  else return kSVELogF32FastB10;
}

/**
 * f32 table-free kernel on the [2/3, 4/3) window; r = z - 1 is exact by
 * Sterbenz. Fast keeps six Taylor terms, Estimate three.
 */
template <const SVELogF32Family& C, Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32_t sve_log_f32_poly_core(
    svfloat32_t x, svbool_t pg) {
  constexpr std::uint32_t kOff = 0x3f2aaaabu;
  const auto u_off = svsub_n_u32_x(pg, svreinterpret_u32_f32(x), kOff);
  const auto n = svcvt_f32_s32_x(
      pg, svasr_n_s32_x(pg, svreinterpret_s32_u32(u_off), 23));
  const auto z = svreinterpret_f32_u32(
      svadd_n_u32_x(pg, svand_n_u32_x(pg, u_off, 0x007fffffu), kOff));
  const auto r = svsub_n_f32_x(pg, z, 1.0f);
  const auto r2 = svmul_f32_x(pg, r, r);
  svfloat32_t y;
  if constexpr (Tier == Accuracy::Fast) {
    const auto a = svmla_n_f32_x(pg, svdup_n_f32(C.c0), r, C.c1);
    const auto b = svmla_n_f32_x(pg, svdup_n_f32(C.c2), r, C.c3);
    const auto d = svmla_n_f32_x(pg, svdup_n_f32(C.c4), r, C.c5);
    y = svmla_f32_x(pg, b, r2, d);
    y = svmla_f32_x(pg, a, r2, y);
  } else {
    // P = c0 + r*(c1 + r*c2): true Horner, both steps multiply by r.
    y = svmla_n_f32_x(pg, svdup_n_f32(C.c1), r, C.c2);
    y = svmla_f32_x(pg, svdup_n_f32(C.c0), r, y);
  }
  const auto hi = svmla_n_f32_x(pg, r, n, C.ln2);
  auto result = svmla_f32_x(pg, hi, r2, y);
  if constexpr (C.scale != 1.0f)
    result = svmul_n_f32_x(pg, result, C.scale);
  return result;
}

/**
 * f32 accuracy path: widens to two f64 words, evaluates the f64 table
 * kernel (with its own special handling), and narrows back. Every f32
 * value including subnormals is exact and normal in f64, so this path owns
 * all f32 tails exactly and delivers Strict-tier accuracy (<= 1 ULP after
 * the single narrowing rounding). Used for the whole Strict tier and for
 * the Fast/Estimate special lanes.
 */
template <LogBase Base>
VECOPS_ALWAYS_INLINE svfloat32_t sve_log_f32_via_f64(svfloat32_t x) {
  const auto bits32 = svreinterpret_u32_f32(x);
  // The widening FCVT reads the bottom 32-bit half of every 64-bit slot,
  // so it takes the even lanes of its source directly; rotating the source
  // by one 32-bit element first yields the odd lanes.
  const auto lo = svcvt_f64_f32_x(svptrue_b64(), x);
  const auto hi = svcvt_f64_f32_x(
      svptrue_b64(),
      svreinterpret_f32_u32(svext_u32(bits32, bits32, 1)));
  constexpr Accuracy Tier = Accuracy::Strict;
  const auto result_lo = sve_log_f64<sve_log_f64_family<Base>(),
                                     sve_log_f64_poly_family<Base>(),
                                     Tier>(lo, svptrue_b64());
  const auto result_hi = sve_log_f64<sve_log_f64_family<Base>(),
                                     sve_log_f64_poly_family<Base>(),
                                     Tier>(hi, svptrue_b64());
  // FCVT Zn.S, Zm.D parks each narrowed value in the bottom half of its
  // 64-bit slot; UZP1 compacts the even lanes before the final ZIP.
  const auto lo_narrow = svreinterpret_u32_f32(
      svcvt_f32_f64_x(svptrue_b32(), result_lo));
  const auto hi_narrow = svreinterpret_u32_f32(
      svcvt_f32_f64_x(svptrue_b32(), result_hi));
  const auto lo_compact = svuzp1_u32(lo_narrow, lo_narrow);
  const auto hi_compact = svuzp1_u32(hi_narrow, hi_narrow);
  return svreinterpret_f32_u32(svzip1_u32(lo_compact, hi_compact));
}

template <LogBase Base, Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat32_t sve_log_f32(svfloat32_t x, svbool_t pg) {
  if constexpr (Tier == Accuracy::Strict)
    return sve_log_f32_via_f64<Base>(x);
  const auto u = svreinterpret_u32_f32(x);
  const svbool_t special = svcmpge_n_u32(
      pg, svsub_n_u32_x(pg, u, 0x00800000u), 0x7f000000u);
  if (svptest_any(special, special))
    return sve_log_f32_via_f64<Base>(x);
  return sve_log_f32_poly_core<sve_log_f32_fast_family<Base>(), Tier>(
      x, pg);
}

/**
 * Widens an f16 word to two f32 words, evaluates the f32 pipeline on both
 * halves, and narrows back. Every f16 value (including subnormals) is
 * normal in f32 and log never produces subnormal outputs, so the f32
 * special routine already owns all f16 tails exactly.
 */
template <LogBase Base, Accuracy Tier>
VECOPS_ALWAYS_INLINE svfloat16_t sve_log_f16_via_f32(svfloat16_t raw) {
  const auto low = svcvt_f32_f16_x(svptrue_b32(), raw);
#if defined(__ARM_FEATURE_SVE2)
  const auto high = svcvtlt_f32_f16_x(svptrue_b32(), raw);
#else
  const auto odd = svuzp2_u16(
      svreinterpret_u16_f16(raw), svreinterpret_u16_f16(raw));
  const auto high = svcvt_f32_f16_x(
      svptrue_b32(), svreinterpret_f16_u16(svzip1_u16(odd, odd)));
#endif
  const auto result_low = sve_log_f32<Base, Tier>(low, svptrue_b32());
  const auto result_high = sve_log_f32<Base, Tier>(high, svptrue_b32());
  const auto packed_low = svcvt_f16_f32_z(svptrue_b32(), result_low);
#if defined(__ARM_FEATURE_SVE2)
  return svcvtnt_f16_f32_m(packed_low, svptrue_b32(), result_high);
#else
  const auto packed_high = svcvt_f16_f32_z(svptrue_b32(), result_high);
  return svtrn1_f16(packed_low, packed_high);
#endif
}

template <LogBase Base, Accuracy Tier, FloatingTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> sve_log_dispatch(
    Tag, NativeWordVec<Tag> value, svbool_t pg) {
  using T = ElementOf<Tag>;
  const auto raw = sve_basic_raw_word(value);
  if constexpr (std::same_as<T, float32_t>) {
    return sve_basic_wrap_word<Tag>(sve_log_f32<Base, Tier>(raw, pg));
  } else if constexpr (std::same_as<T, float64_t>) {
    return sve_basic_wrap_word<Tag>(
        sve_log_f64<sve_log_f64_family<Base>(),
                    sve_log_f64_poly_family<Base>(), Tier>(raw, pg));
  } else if constexpr (std::same_as<T, float16_t>) {
    return sve_basic_wrap_word<Tag>(
        sve_log_f16_via_f32<Base, Tier>(raw));
  } else {
    // bf16 widens exactly to f32 and the f32 exponent range is shared, so
    // the f32 pipeline owns every bf16 tail directly.
    const auto low = sve_bf16_to_f32_lo(raw);
    const auto high = sve_bf16_to_f32_hi(raw);
    return sve_basic_wrap_word<Tag>(sve_f32_pair_to_bf16(
        sve_log_f32<Base, Tier>(low, svptrue_b32()),
        sve_log_f32<Base, Tier>(high, svptrue_b32())));
  }
}

/**
 * Shared word implementation for the log family. Wide element formats are
 * elementwise, so an explicit mask only guards the special-trigger compare
 * and the final blend. Narrow formats widen through whole-word f32
 * conversions; sanitize inactive lanes first so the widening (and the
 * special-tail tests inside it) cannot observe stale values.
 */
template <LogBase Base, Accuracy A>
struct SVELogWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LogOp<Base, A>, Tag tag, NativeWordVec<Tag> value) {
    return sve_log_dispatch<Base, A>(
        tag, value, sve_full_predicate<ElementOf<Tag>>());
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LogOp<Base, A> op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    if constexpr (sizeof(ElementOf<Tag>) >= 4) {
      const auto computed = sve_log_dispatch<Base, A>(
          tag, value, sve_basic_raw_word(mask));
      return NativeWordImpl<SVEBackend, BlendOp>::template call<Index>(
          BlendOp{}, tag, inactive, mask, computed);
    } else {
      const auto zero =
          NativeWordImpl<SVEBackend, FillOp>::template call<Index>(
              FillOp{}, tag, ElementOf<Tag>{});
      const auto safe = NativeWordImpl<SVEBackend, BlendOp>::template call<
          Index>(BlendOp{}, tag, zero, mask, value);
      const auto computed = sve_log_dispatch<Base, A>(
          tag, safe, sve_full_predicate<ElementOf<Tag>>());
      return NativeWordImpl<SVEBackend, BlendOp>::template call<Index>(
          BlendOp{}, tag, inactive, mask, computed);
    }
  }
};

template <LogBase Base, Accuracy A>
struct NativeWordImpl<SVEBackend, LogOp<Base, A>> : SVELogWordImpl<Base, A> {
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_MATH_LOG_H
