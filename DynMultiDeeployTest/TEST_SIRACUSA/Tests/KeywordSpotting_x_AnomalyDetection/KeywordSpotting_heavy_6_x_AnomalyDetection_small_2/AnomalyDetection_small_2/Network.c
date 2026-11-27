#include "DeeployPULPMath.h"
#include "bsp/ram.h"
#include "dory_mem.h"
#include "mchan_siracusa.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "stdint.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Network.h"

int8_t *AnomalyDetection_small_2_MEMORYARENA_L1; // L2_init_template
int8_t *AnomalyDetection_small_2_MEMORYARENA_L2; // L2_init_template
int8_t *AnomalyDetection_small_2_input_0;        // L2_init_template
int8_t *AnomalyDetection_small_2_output_0;       // L2_init_template

static PI_L2 int8_t AnomalyDetection_small_2_layer0_weight_tensor[1280] = {
    9,   21,  14,  16,   -13, 4,   -13, 1,   15,  -21, 24,  -10, 16,  7,   17,  -16, -1,  -9,  -14, -6,   -14, -2,  -20, 27,  -17, 37,  -11, 10,  15,  -12, 13,
    -11, 25,  3,   -31,  2,   6,   17,  -23, 47,  45,  -6,  1,   14,  -8,  5,   -9,  -3,  -30, -22, -21,  16,  -16, 28,  1,   9,   12,  -9,  -5,  -32, 7,   -13,
    -9,  -5,  15,  -11,  22,  4,   -27, -7,  17,  39,  47,  -25, -14, -12, 9,   20,  -5,  19,  -5,  24,   22,  33,  63,  9,   60,  -41, 15,  -40, 17,  -41, 2,
    5,   7,   -21, -9,   -12, -12, -6,  31,  4,   6,   -6,  -37, -31, -24, -12, -30, 18,  3,   21,  17,   -72, -9,  59,  2,   -16, -45, -42, 10,  2,   -31, 12,
    -9,  23,  -85, -115, -8,  29,  18,  6,   -16, 8,   -16, -9,  -18, -15, 5,   -10, 3,   -15, 23,  -30,  -18, -14, 10,  -3,  9,   -1,  -2,  13,  -7,  18,  8,
    -7,  4,   12,  -2,   0,   12,  2,   -24, -20, 8,   -5,  -16, 67,  45,  -17, -10, -1,  -2,  2,   -4,   -10, -24, 10,  -6,  5,   -2,  22,  -12, -15, 0,   -12,
    8,   -42, 1,   -3,   -3,  4,   4,   4,   -8,  9,   -16, -5,  10,  20,  12,  -19, -10, -22, 7,   7,    -12, -6,  5,   38,  20,  22,  24,  -1,  29,  6,   44,
    -34, 25,  -12, -4,   -12, 19,  -30, -33, 28,  12,  6,   19,  -24, -8,  18,  -10, 8,   -13, 7,   9,    9,   -14, 0,   29,  -47, -11, 69,  11,  -4,  -20, -35,
    8,   29,  -21, 20,   15,  26,  -62, -94, -7,  4,   7,   0,   22,  2,   11,  -6,  9,   15,  -16, -4,   -9,  -16, 1,   -26, 16,  15,  21,  -24, -10, 4,   7,
    -9,  -26, 35,  -22,  8,   12,  2,   -16, 13,  8,   -10, -5,  -26, -16, -14, -11, 35,  12,  1,   2,    -14, -23, -7,  -12, -13, -47, 11,  -30, 2,   -25, 12,
    -2,  -4,  -4,  -7,   13,  -40, -3,  6,   -1,  10,  -10, -5,  3,   9,   -20, -1,  -1,  7,   10,  -8,   3,   -7,  5,   1,   15,  -5,  -6,  37,  8,   34,  50,
    -7,  40,  -16, 8,    -44, 12,  -22, -2,  -11, 16,  -3,  -4,  -1,  -13, -8,  33,  7,   -3,  -11, -22,  -21, -42, 16,  -4,  17,  -11, 0,   -3,  -46, 6,   65,
    -10, -8,  -28, -33,  14,  32,  5,   32,  12,  14,  -71, -77, -3,  30,  -4,  -13, 10,  -7,  -6,  -13,  7,   -10, 6,   -1,  -10, -1,  11,  -35, -21, 17,  -3,
    -13, -4,  -29, -9,   5,   -16, 37,  -3,  8,   8,   8,   25,  -14, -3,  -8,  -28, -35, -7,  10,  -11,  68,  30,  7,   -6,  -10, -6,  11,  11,  3,   -30, 10,
    -31, 20,  -20, 5,    -20, -8,  -13, -16, -16, -33, 9,   1,   -7,  23,  13,  -24, 7,   0,   -31, 6,    18,  28,  5,   26,  -3,  -13, 3,   29,  -9,  10,  3,
    24,  -5,  20,  60,   23,  49,  -21, 41,  -20, -2,  -10, -9,  -6,  18,  -6,  -11, -15, -17, -9,  30,   -18, 22,  3,   -5,  16,  -31, 34,  -5,  8,   3,   13,
    20,  -16, -25, 49,   5,   -15, -18, -37, 31,  16,  -39, 18,  21,  31,  -78, -83, -1,  12,  -13, -12,  26,  2,   -11, 8,   -1,  9,   13,  12,  -6,  6,   1,
    -23, -14, -5,  -4,   -28, 3,   4,   26,  13,  -15, 45,  -10, 26,  -2,  11,  -13, -17, 16,  -12, -11,  -8,  17,  12,  -20, 68,  40,  -6,  -7,  -29, -12, -5,
    1,   7,   -25, 2,    -21, 6,   -24, -11, -13, -14, -3,  -29, -4,  -56, -5,  13,  -9,  13,  16,  8,    3,   -4,  -12, -9,  7,   33,  28,  -8,  -6,  -1,  21,
    16,  10,  26,  2,    35,  16,  15,  54,  1,   62,  2,   27,  -15, 2,   -28, 18,  -15, -4,  -18, -7,   27,  1,   -23, 21,  24,  9,   5,   -31, -6,  -24, 22,
    -9,  37,  -30, 2,    32,  -43, -22, 83,  -13, -19, -42, -45, 16,  30,  -46, 47,  7,   32,  -93, -100, 16,  -23, 1,   21,  -19, -11, 8,   8,   6,   -13, 25,
    0,   25,  -1,  4,    36,  19,  -3,  -3,  34,  -20, 5,   -19, 7,   6,   -49, 16,  -27, 4,   -36, 7,    0,   19,  15,  4,   34,  -7,  -1,  2,   -60, -17, 3,
    -6,  19,  16,  7,    2,   -1,  18,  -6,  3,   -2,  2,   14,  12,  17,  18,  32,  11,  37,  9,   -12,  6,   -24, 0,   -16, 11,  -11, -13, -2,  5,   -17, -4,
    -28, 12,  14,  -16,  -34, 5,   -4,  -18, -27, 4,   -22, -56, -6,  -32, -8,  -58, 34,  6,   11,  -10,  32,  -13, 29,  14,  -21, 17,  3,   -31, 8,   1,   13,
    18,  -4,  20,  -25,  -17, -15, 34,  -7,  -21, 45,  17,  -87, 2,   5,   14,  52,  -25, -20, 20,  -51,  -41, -24, 83,  65,  13,  -7,  11,  11,  -23, -1,  4,
    -5,  -25, -11, 6,    1,   11,  -23, 9,   20,  -3,  -9,  18,  31,  2,   6,   -4,  -6,  17,  -61, 35,   -38, -5,  -7,  -8,  12,  6,   14,  5,   12,  -5,  -21,
    8,   -32, -13, -13,  -18, 5,   25,  1,   3,   -6,  19,  21,  20,  -11, 19,  10,  1,   -8,  6,   28,   17,  24,  4,   -2,  11,  -12, -10, 4,   -14, -1,  -2,
    1,   2,   -28, -29,  -19, 14,  6,   -17, -34, -1,  -24, -1,  -2,  10,  -20, -77, -6,  -50, 39,  -20,  37,  9,   26,  -8,  12,  -5,  10,  -11, 19,  34,  14,
    -37, -32, -22, 35,   33,  21,  23,  -10, 11,  -30, 8,   -31, -5,  45,  10,  -67, 2,   10,  23,  37,   -41, -6,  12,  -49, -25, -39, 74,  76,  13,  -35, -1,
    2,   21,  -8,  34,   -5,  7,   21,  -16, 6,   1,   -27, -10, 23,  28,  20,  31,  8,   -17, 14,  4,    -24, -1,  -44, 3,   -20, 4,   -19, -19, 25,  4,   4,
    26,  5,   -26, -29,  14,  -62, -46, 7,   -7,  -3,  2,   -6,  -6,  -9,  -1,  19,  -1,  -14, -3,  -1,   11,  5,   1,   33,  26,  23,  1,   6,   14,  -2,  -27,
    0,   -4,  -1,  -5,   3,   -9,  -40, -27, -8,  28,  24,  -18, -35, 23,  -19, -9,  -5,  3,   -9,  -44,  -11, -40, 15,  -59, 25,  -5,  15,  -6,  12,  -10, 31,
    22,  -8,  7,   -4,   -18, -7,  -18, 12,  25,  -10, -2,  -8,  1,   -24, 7,   -32, -36, 38,  28,  -71,  -23, 5,   11,  35,  -29, -7,  38,  -43, -20, -47, 57,
    83,  18,  -9,  -6,   -10, 10,  -15, 14,  -8,  5,   -1,  4,   12,  1,   -12, -2,  17,  -12, 25,  6,    20,  -14, -12, -13, -8,  9,   -41, 22,  -23, 2,   -11,
    21,  1,   -10, 7,    4,   -4,  -15, -4,  13,  -27, -28, 11,  -16, 1,   16,  14,  17,  11,  19,  18,   -1,  6,   3,   -5,  -8,  4,   -9,  27,  -1,  30,  16,
    3,   11,  12,  -6,   -22, 1,   -11, -15, 9,   12,  -21, -29, 24,  20,  15,  -23, -6,  -1,  -3,  -5,   -21, -13, -28, -37, 16,  -34, 10,  -30, 39,  -19, 31,
    -17, 8,   -9,  23,   11,  -22, 0,   -4,  -23, -32, 9,   22,  49,  24,  10,  5,   7,   -36, 23,  -24,  -11, 68,  -3,  -82, -7,  -2,  29,  35,  -7,  -15, 2,
    -61, -11, -31, 56,   79,  17,  -28, -16, -4,  23,  -9,  11,  14,  0,   22,  7,   28,  5,   -6,  -10,  31,  -1,  3,   10,  12,  -10, 26,  20,  2,   5,   -33,
    12,  -5,  -13, -7,   -17, 1,   10,  4,   25,  22,  8,   -4,  5,   -34, -24, -2,  -21, -12, 9,   -1,   8,   14,  30,  16,  12,  -3,  -1,  -20, 1,   -1,  -1,
    21,  13,  14,  6,    13,  15,  2,   -5,  7,   -4,  -18, 5,   -9,  4,   -26, -10, -9,  14,  29,  -13,  -27, 14,  2,   -17, -25, -9,  -51, -63, -19, -36, 26,
    -57, 30,  -22, 11,   -4,  -8,  -43, 10,  17,  6,   15,  -24, -45, 0,   -8,  25,  29,  1,   21,  -18,  8,   -13, -13, -36, -5,  48,  2,   -53, -17, -1,  26,
    40,  -13, 3,   11,   -39, -9,  -31, 72,  103};

static PI_L2 int32_t AnomalyDetection_small_2_layer0_bias_tensor[2] = {-44506, -3738};

static PI_L2 int32_t AnomalyDetection_small_2_layer0_mul_tensor[2] = {1, 1};

static PI_L1 uint8_t AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[2] = {0, 1};

void *AnomalyDetection_small_2_inputs[1];
void *AnomalyDetection_small_2_outputs[1];
extern struct pi_device cluster_dev;
// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref;
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref;
  int32_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref;
  int32_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref;
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref;
} AnomalyDetection_small_2_RequantizedGemm_0_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void AnomalyDetection_small_2_RequantizedGemm_0_tiling_closure(void *AnomalyDetection_small_2_RequantizedGemm_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  AnomalyDetection_small_2_RequantizedGemm_0_tiling_closure_args_t *args =
      (AnomalyDetection_small_2_RequantizedGemm_0_tiling_closure_args_t *)AnomalyDetection_small_2_RequantizedGemm_0_tiling_closure_args;
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref = args->AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref;
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref = args->AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref;
  int32_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref = args->AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref;
  int32_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref = args->AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref;
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref =
      args->AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN GEMM
  int8_t *ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref =
      AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref;
  int8_t *ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref =
      AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref;
  int8_t
      *ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref =
          AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 1; j++) {
// LMACAN: In some edge cases sporadic errors happen if this loop is not added.
// We believe this is due to missing bubbles in the pipeline that break operator forwarding.
// Breaking test:
//   `python testRunner_tiled_siracusa.py -t=Tests/Transformer --defaultMemLevel=L3 --doublebuffer --l1=30000`
#pragma unroll 1
      for (int k = 0; k < 3; k++) {
        asm volatile("nop" ::);
      }
      pulp_nn_linear_i8_i8_i8(
          ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref,
          NULL,
          ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref,
          ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref,
          AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref, AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref, 1, 10, 640,
          2, 1, 1, NUM_CORES);
      ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref += 640;
      ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref +=
          2;
    }
    ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref += 640 * 2;
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref;
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref;
  int32_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref;
  int32_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref;
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref;
} AnomalyDetection_small_2_RequantizedGemm_0_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void AnomalyDetection_small_2_RequantizedGemm_0_cluster_fork(void *AnomalyDetection_small_2_RequantizedGemm_0_cluster_fork_args) {
  // CLOSURE ARG CAST
  AnomalyDetection_small_2_RequantizedGemm_0_cluster_fork_args_t *args =
      (AnomalyDetection_small_2_RequantizedGemm_0_cluster_fork_args_t *)AnomalyDetection_small_2_RequantizedGemm_0_cluster_fork_args;
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref = args->AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref;
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref = args->AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref;
  int32_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref = args->AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref;
  int32_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref = args->AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref;
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref =
      args->AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref;

  // CLOSURE FUNCTION CALL
  AnomalyDetection_small_2_RequantizedGemm_0_tiling_closure_args_t AnomalyDetection_small_2_AnomalyDetection_small_2_RequantizedGemm_0_tiling_closure_args =
      (AnomalyDetection_small_2_RequantizedGemm_0_tiling_closure_args_t){
          .AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref = AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref,
          .AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref = AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref,
          .AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref = AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref,
          .AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref = AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref,
          .AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref =
              AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref};

  // AnomalyDetection_small_2_RequantizedGemm_0_tiling_closure CLOSURE CALL
  AnomalyDetection_small_2_RequantizedGemm_0_tiling_closure(&AnomalyDetection_small_2_AnomalyDetection_small_2_RequantizedGemm_0_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;
} AnomalyDetection_small_2_RequantizedGemm_0_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void AnomalyDetection_small_2_RequantizedGemm_0_closure(void *AnomalyDetection_small_2_RequantizedGemm_0_closure_args) {
  // CLOSURE ARG CAST
  AnomalyDetection_small_2_RequantizedGemm_0_closure_args_t *args =
      (AnomalyDetection_small_2_RequantizedGemm_0_closure_args_t *)AnomalyDetection_small_2_RequantizedGemm_0_closure_args;
  uint8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
      args->AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref = (int8_t *)((char *)AnomalyDetection_small_2_MEMORYARENA_L1 + 1282);
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref = (int8_t *)((char *)AnomalyDetection_small_2_MEMORYARENA_L1 + 2);
  int32_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref = (int32_t *)((char *)AnomalyDetection_small_2_MEMORYARENA_L1 + 1930);
  int32_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref = (int32_t *)((char *)AnomalyDetection_small_2_MEMORYARENA_L1 + 1922);
  int8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref = (int8_t *)((char *)AnomalyDetection_small_2_MEMORYARENA_L1 + 0);
  void *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_input_0_ref = (void *)((char *)AnomalyDetection_small_2_input_0 + 0);
  void *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_weight_tensor_ref =
      (void *)((char *)AnomalyDetection_small_2_layer0_weight_tensor + 0);
  void *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_bias_tensor_ref = (void *)((char *)AnomalyDetection_small_2_layer0_bias_tensor + 0);
  void *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_mul_tensor_ref = (void *)((char *)AnomalyDetection_small_2_layer0_mul_tensor + 0);
  void *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_output_0_ref = (void *)((char *)AnomalyDetection_small_2_output_0 + 0);

  const static char RequantizedGemm_0_L2_suffix[] = " cycles \n";

  const static char RequantizedGemm_0_L2_prefix[] = "[RequantizedGemm_0_L2][SB][4486 ops][Tile ";

  uint32_t RequantizedGemm_0_L2_egress_dma_wait_end_measurements[1];

  uint32_t RequantizedGemm_0_L2_egress_dma_wait_start_measurements[1];

  uint32_t RequantizedGemm_0_L2_ingress_dma_wait_end_measurements[1];

  uint32_t RequantizedGemm_0_L2_ingress_dma_wait_start_measurements[1];

  uint32_t RequantizedGemm_0_L2_kernel_end_measurements[1];

  uint32_t RequantizedGemm_0_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t AnomalyDetection_small_2_channel_output = (uint32_t)-1;
  uint32_t AnomalyDetection_small_2_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[*AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr];
       TILING_I <
       AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[(*AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr) + 1];
       TILING_I++) {

    RequantizedGemm_0_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    AnomalyDetection_small_2_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442432, AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref,
                      AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_input_0_ref);
    mchan_transfer_1d(1443072, AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref,
                      AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_weight_tensor_ref);
    mchan_transfer_1d(1441800, AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref,
                      AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_bias_tensor_ref);
    mchan_transfer_1d(1441800, AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref,
                      AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_mul_tensor_ref);

    // Wait for input tiles

    if (AnomalyDetection_small_2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_small_2_channel_input);
      mchan_channel_free(AnomalyDetection_small_2_channel_input);
    }

    RequantizedGemm_0_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    RequantizedGemm_0_L2_kernel_start_measurements[TILING_I] = getCycles();

    AnomalyDetection_small_2_RequantizedGemm_0_cluster_fork_args_t AnomalyDetection_small_2_AnomalyDetection_small_2_RequantizedGemm_0_cluster_fork_args =
        (AnomalyDetection_small_2_RequantizedGemm_0_cluster_fork_args_t){
            .AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref = AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref,
            .AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref = AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref,
            .AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref = AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref,
            .AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref = AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref,
            .AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref =
                AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)AnomalyDetection_small_2_RequantizedGemm_0_cluster_fork,
                    &AnomalyDetection_small_2_AnomalyDetection_small_2_RequantizedGemm_0_cluster_fork_args);

    RequantizedGemm_0_L2_kernel_end_measurements[TILING_I] = getCycles();

    RequantizedGemm_0_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    AnomalyDetection_small_2_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310722, AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref,
                      AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_output_0_ref);

    // Wait for output tiles

    if (AnomalyDetection_small_2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_small_2_channel_output);
      mchan_channel_free(AnomalyDetection_small_2_channel_output);
    }

    RequantizedGemm_0_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr > 0)
                              ? AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[(
                                    *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[*AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", RequantizedGemm_0_L2_prefix, PROFILING_I, "Input DMA took ",
           RequantizedGemm_0_L2_ingress_dma_wait_end_measurements[PROFILING_I] - RequantizedGemm_0_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           RequantizedGemm_0_L2_suffix);

    printf("%s%u] %s%u%s", RequantizedGemm_0_L2_prefix, PROFILING_I, "Kernel took ",
           RequantizedGemm_0_L2_kernel_end_measurements[PROFILING_I] - RequantizedGemm_0_L2_kernel_start_measurements[PROFILING_I],
           RequantizedGemm_0_L2_suffix);

    printf("%s%u] %s%u%s", RequantizedGemm_0_L2_prefix, PROFILING_I, "Output DMA took ",
           RequantizedGemm_0_L2_egress_dma_wait_end_measurements[PROFILING_I] - RequantizedGemm_0_L2_egress_dma_wait_start_measurements[PROFILING_I],
           RequantizedGemm_0_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;
} AnomalyDetection_small_2_RequantizedGemm_0_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void AnomalyDetection_small_2_RequantizedGemm_0_closure_L3(void *AnomalyDetection_small_2_RequantizedGemm_0_closure_L3_args) {
  // CLOSURE ARG CAST
  AnomalyDetection_small_2_RequantizedGemm_0_closure_L3_args_t *args =
      (AnomalyDetection_small_2_RequantizedGemm_0_closure_L3_args_t *)AnomalyDetection_small_2_RequantizedGemm_0_closure_L3_args;
  uint8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
      args->AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  AnomalyDetection_small_2_RequantizedGemm_0_closure_args_t AnomalyDetection_small_2_AnomalyDetection_small_2_RequantizedGemm_0_closure_args =
      (AnomalyDetection_small_2_RequantizedGemm_0_closure_args_t){.AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
                                                                      AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr};

  // AnomalyDetection_small_2_RequantizedGemm_0_closure CLOSURE CALL
  AnomalyDetection_small_2_RequantizedGemm_0_closure(&AnomalyDetection_small_2_AnomalyDetection_small_2_RequantizedGemm_0_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

void RunNetwork() {

  uint8_t bu_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr = 0;
  uint8_t *AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
      &bu_AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;
  // RUNNETRWORK LAYER AnomalyDetection_small_2_RequantizedGemm_0 CALL START

  AnomalyDetection_small_2_RequantizedGemm_0_closure_L3_args_t AnomalyDetection_small_2_AnomalyDetection_small_2_RequantizedGemm_0_closure_L3_args =
      (AnomalyDetection_small_2_RequantizedGemm_0_closure_L3_args_t){.AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
                                                                         AnomalyDetection_small_2_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr};

  // AnomalyDetection_small_2_RequantizedGemm_0_closure_L3 CLOSURE CALL
  AnomalyDetection_small_2_RequantizedGemm_0_closure_L3(&AnomalyDetection_small_2_AnomalyDetection_small_2_RequantizedGemm_0_closure_L3_args);

  // RUNNETRWORK LAYER AnomalyDetection_small_2_RequantizedGemm_0 CALL END
}

void InitNetwork() {

  AnomalyDetection_small_2_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 1938);

  AnomalyDetection_small_2_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 642);

  AnomalyDetection_small_2_input_0 = (int8_t *)((char *)AnomalyDetection_small_2_MEMORYARENA_L2 + 2);
  AnomalyDetection_small_2_output_0 = (int8_t *)((char *)AnomalyDetection_small_2_MEMORYARENA_L2 + 0);
  AnomalyDetection_small_2_inputs[0] = (void *)AnomalyDetection_small_2_input_0;
  AnomalyDetection_small_2_outputs[0] = (void *)AnomalyDetection_small_2_output_0;
}
