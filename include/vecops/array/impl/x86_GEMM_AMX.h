//
// Created by renyz on 2026/5/9.
//

#ifndef VECOPS_X86_GEMM_AMX_H
#define VECOPS_X86_GEMM_AMX_H

#include "vecops/Features.h"
#include "vecops/CoreDefs.h"
#include "vecops/Assertion.h"
#include "vecops/vec/Vec.h"

#ifndef HAS_AMX_TILE
  #error "No AMX"
#endif
#ifndef HAS_AVX512F
  #errpr "No AVX512"
#endif

#include <syscall.h>
#include <immintrin.h>

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif // ARCH_REQ_XCOMP_PERM
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif // XFEATURE_XTILEDATA

namespace vecops::array::AMX {

VECOPS_ALWAYS_INLINE static bool init_amx_env()
{
  return !syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
}

static constexpr int MaxRows = 16;
static constexpr int MaxBytes = 1024;
static constexpr int MaxColBytes = 64;
template <typename T>
static constexpr int MaxCols = MaxColBytes / sizeof(T);
static constexpr int NumTiles = 8;

class alignas(64) AMXConfig
{
public:
  AMXConfig() = default;

  template <typename T> VECOPS_INLINE
  AMXConfig& define(int tile, int rows, int cols)
  {
    VECOPS_ASSERT(0 <= tile && tile < NumTiles, "%d !in 0:%d", tile, NumTiles);
    VECOPS_ASSERT(0 <= rows && rows <= MaxRows, "%d !in 0:%d", rows, MaxRows);
    VECOPS_ASSERT(0 <= cols && cols * sizeof(T) <= MaxColBytes, "%d !in 0:%d", cols * sizeof(T), MaxColBytes);
    if (rows == 0 || cols == 0) {
      rows = cols = 0;  // Must be both zero, otherwise a #GP is raised
    }
    this->colsb[tile] = cols * sizeof(T);
    this->rows[tile] = rows;
    return  *this;
  }

  VECOPS_ALWAYS_INLINE
  AMXConfig& define_fp32(int tile, int rows, int cols)
  {
    return define<int32_t>(tile, rows, cols);
  }

  VECOPS_ALWAYS_INLINE
  AMXConfig& define_fp16(int tile, int rows, int cols)
  {
    return define<int16_t>(tile, rows, cols);
  }

  VECOPS_ALWAYS_INLINE
  AMXConfig& define_int32(int tile, int rows, int cols)
  {
    return define<int32_t>(tile, rows, cols);
  }

  VECOPS_ALWAYS_INLINE
  AMXConfig& define_bf16(int tile, int rows, int cols)
  {
    return define<int16_t>(tile, rows, cols);
  }

  VECOPS_ALWAYS_INLINE
  AMXConfig& define_int8(int tile, int rows, int cols)
  {
    return define<int8_t>(tile, rows, cols);
  }

private:
  uint8_t palette_id = 1;
  uint8_t start_row = 0;
  uint8_t reserved_0[14] = {};
  uint16_t colsb[16] = {};
  uint8_t rows[16] = {};
}; // class AMXConfig

static_assert(sizeof(AMXConfig) == 64);

VECOPS_ALWAYS_INLINE void load_config(const AMXConfig& config)
{
  _tile_loadconfig(&config);
}

VECOPS_ALWAYS_INLINE AMXConfig get_config()
{
  AMXConfig cfg;
  _tile_storeconfig(&cfg);
  return cfg;
}

VECOPS_ALWAYS_INLINE void release()
{
  _tile_release();
}

/**
 * Perform a matrix mul-add loop on packed matrix A and B to tile register 4-7:
 * @code
 *    TILE #          2: B[kb,:,:,0,:]  3: B[kb,:,:,1,:]
 *    0: A[0,:,kb,:]  4: 0 @ 2          5: 0 @ 3
 *    1: A[1,:,kb,:]  6: 1 @ 2          7: 1 @ 3
 * @endcode
 *
 * @note Caller ensures that tile config is set before calling this function.
 * @note Kernel will not clear tile 4-7 before the loop
 * @note K must be divisible by 32
 *
 * @note To pack inputs: suppose unpacked matrix A (32, K) & B (K, 32)
 *   - A(32, K) -> view as (2, 16, K)
 *              -> pad K with 0 (2, 16, ceil(K/32)*32)
 *              -> reshape (2, 16, ceil(K/32), 32), stride (512, 32, 1024, 1)
 *   - B(K, 32) -> view as (K, 2, 16)
 *              -> pad K with 0 (ceil(K/32)*32, 2, 16)
 *              -> view as (ceil(K/32), 32, 2, 16)
 *              -> reshape (ceil(K/32), 16, 2, 2, 16), stride (1024, 32, 1, 512, 2)
 * @param A packed array of (nMb=2, Mb=16, nKb=ceil(K/32), Kb=32), stride (512, 32, 1024, 1)
 * @param B packed array of (nKb=ceil(K/32), Kb0=16, Kb1=2, nNb=2, Nb=16), stride (1024, 32, 1, 512, 2)
 * @param K the size of accumulation axis
 */
void gemm_bf16_32x32(const bfloat16_t * A, const bfloat16_t * B, int K) {
  constexpr int KTileSz = MaxCols<bfloat16_t>;
  constexpr int KTileBytes = MaxColBytes;
  VECOPS_ASSERT(K % KTileSz == 0, "K must be divisible by %d: %d", KTileSz, K);

  _tile_loadd(0, A, KTileBytes);
  _tile_loadd(1, A + MaxRows * KTileSz, KTileBytes);
  _tile_loadd(2, B, KTileBytes);
  _tile_loadd(3, B + MaxRows * KTileSz, KTileBytes);
  int k = KTileSz;
  for (; k < K; k += KTileSz) {
    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(6, 1, 2);
    _tile_loadd(2, B + k * MaxRows * 2, KTileBytes);
    _tile_dpbf16ps(5, 0, 3);
    _tile_loadd(0, A + k * MaxRows * 2, KTileBytes);
    _tile_dpbf16ps(7, 1, 3);
    _tile_loadd(1, A + k * MaxRows * 2 + MaxRows * KTileSz, KTileBytes);
    _tile_loadd(3, B + k * MaxRows * 2 + MaxRows * KTileSz, KTileBytes);
  }
  _tile_dpbf16ps(4, 0, 2);
  _tile_dpbf16ps(6, 1, 2);
  _tile_dpbf16ps(5, 0, 3);
  _tile_dpbf16ps(7, 1, 3);
  // store back tile 4, 5, 6, 7 and apply epilog
}

template <
    
    >
class Gemm {

};

/**
 * Perform a matrix multiplication C = A @ B
 * @tparam TAInputFn
 * @tparam TBInputFn
 * @tparam TOutputFn
 * @param A
 * @param B
 * @param C
 * @param K
 */
template <typename TAInputFn, typename TBInputFn, typename TOutputFn>
void gemm_bf16_32x32(TAInputFn&& A, TBInputFn&& B, TOutputFn&& C, int K) {
  constexpr int KTileSz = MaxCols<bfloat16_t>;
  constexpr int KTileBytes = MaxColBytes;
  VECOPS_ASSERT(K % KTileSz == 0, "K must be divisible by %d: %d", KTileSz, K);

  _tile_loadd(0, A, KTileBytes);
  _tile_loadd(1, A + MaxRows * KTileSz, KTileBytes);
  _tile_loadd(2, B, KTileBytes);
  _tile_loadd(3, B + MaxRows * KTileSz, KTileBytes);
  int k = KTileSz;
  for (; k < K; k += KTileSz) {
    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(6, 1, 2);
    _tile_loadd(2, B + k * MaxRows * 2, KTileBytes);
    _tile_dpbf16ps(5, 0, 3);
    _tile_loadd(0, A + k * MaxRows * 2, KTileBytes);
    _tile_dpbf16ps(7, 1, 3);
    _tile_loadd(1, A + k * MaxRows * 2 + MaxRows * KTileSz, KTileBytes);
    _tile_loadd(3, B + k * MaxRows * 2 + MaxRows * KTileSz, KTileBytes);
  }
  _tile_dpbf16ps(4, 0, 2);
  _tile_dpbf16ps(6, 1, 2);
  _tile_dpbf16ps(5, 0, 3);
  _tile_dpbf16ps(7, 1, 3);
  // store back tile 4, 5, 6, 7 and apply epilog
}



} // namespace vecops::array::AMX

#endif //VECOPS_X86_GEMM_AMX_H
