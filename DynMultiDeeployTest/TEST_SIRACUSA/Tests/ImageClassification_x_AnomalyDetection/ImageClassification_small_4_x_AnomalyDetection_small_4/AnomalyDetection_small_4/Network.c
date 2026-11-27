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

int8_t *AnomalyDetection_small_4_MEMORYARENA_L1; // L2_init_template
int8_t *AnomalyDetection_small_4_MEMORYARENA_L2; // L2_init_template
int8_t *AnomalyDetection_small_4_input_0;        // L2_init_template
int8_t *AnomalyDetection_small_4_output_0;       // L2_init_template

static PI_L2 int8_t AnomalyDetection_small_4_layer0_weight_tensor[1280] = {
    1,  1,  1,  1,  0,  1,  0,  1,  1,  0,  1,  0,  1,  1,  1,  0,  0,  0, 0,  0,  0,  0,  0,  1,  0,  2,  0,  1,  1,  0,  1,  0,  1,  1,  -1, 1,  1,  1,  0,
    2,  2,  0,  1,  1,  0,  1,  0,  0,  -1, 0,  0,  1,  0,  2,  1,  1,  1, 0,  0,  -1, 1,  0,  0,  0,  1,  0,  1,  1,  -1, 0,  1,  2,  2,  0,  0,  0,  1,  1,
    0,  1,  0,  1,  1,  2,  3,  1,  3,  -1, 1,  -1, 1,  -1, 1,  1,  1,  0, 0,  0,  0,  0,  2,  1,  1,  0,  -1, -1, 0,  0,  -1, 1,  1,  1,  1,  -2, 0,  3,  1,
    0,  -1, -1, 1,  1,  -1, 1,  0,  1,  -3, -4, 0,  2,  1,  1,  0,  1,  0, 0,  0,  0,  1,  0,  1,  0,  1,  -1, 0,  0,  1,  0,  1,  0,  0,  1,  0,  1,  1,  0,
    1,  1,  0,  0,  1,  1,  0,  0,  1,  0,  0,  3,  2,  0,  0,  0,  0,  1, 0,  0,  0,  1,  0,  1,  0,  1,  0,  0,  1,  0,  1,  -1, 1,  0,  0,  1,  1,  1,  0,
    1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  2,  1,  1,  1, 0,  2,  1,  2,  -1, 1,  0,  0,  0,  1,  -1, -1, 2,  1,  1,  1,  0,  0,  1,  0,  1,
    0,  1,  1,  1,  0,  0,  2,  -1, 0,  3,  1,  0,  0,  -1, 1,  2,  0,  1, 1,  1,  -2, -3, 0,  1,  1,  1,  1,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1,  0,  1,
    1,  1,  0,  0,  1,  1,  0,  0,  2,  0,  1,  1,  1,  0,  1,  1,  0,  0, 0,  0,  0,  0,  2,  1,  1,  1,  0,  0,  0,  0,  0,  -1, 1,  -1, 1,  0,  1,  0,  0,
    0,  0,  1,  -1, 0,  1,  0,  1,  0,  0,  1,  1,  0,  0,  0,  1,  1,  0, 1,  0,  1,  1,  1,  0,  0,  2,  1,  2,  2,  0,  2,  0,  1,  -1, 1,  0,  0,  0,  1,
    0,  0,  0,  0,  0,  2,  1,  0,  0,  0,  0,  -1, 1,  0,  1,  0,  1,  0, -1, 1,  3,  0,  0,  -1, -1, 1,  2,  1,  2,  1,  1,  -2, -2, 0,  2,  0,  0,  1,  0,
    0,  0,  1,  0,  1,  0,  0,  0,  1,  -1, 0,  1,  0,  0,  0,  -1, 0,  1, 0,  2,  0,  1,  1,  1,  1,  0,  0,  0,  -1, -1, 0,  1,  0,  3,  2,  1,  0,  0,  0,
    1,  1,  1,  -1, 1,  -1, 1,  0,  1,  0,  0,  0,  0,  0,  -1, 1,  1,  0, 1,  1,  0,  1,  0,  -1, 1,  1,  2,  1,  1,  0,  0,  1,  2,  0,  1,  1,  1,  0,  1,
    3,  1,  2,  0,  2,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  2,  0, 1,  1,  0,  1,  -1, 2,  0,  1,  1,  1,  1,  0,  0,  2,  1,  0,  0,  -1, 2,  1,  -1,
    1,  1,  2,  -2, -3, 0,  1,  0,  0,  1,  1,  0,  1,  0,  1,  1,  1,  0, 1,  1,  0,  0,  0,  0,  -1, 1,  1,  1,  1,  0,  2,  0,  1,  0,  1,  0,  0,  1,  0,
    0,  0,  1,  1,  0,  3,  2,  0,  0,  -1, 0,  0,  1,  1,  0,  1,  0,  1, 0,  0,  0,  0,  0,  -1, 0,  -2, 0,  1,  0,  1,  1,  1,  1,  0,  0,  0,  1,  2,  2,
    0,  0,  0,  1,  1,  1,  1,  1,  2,  1,  1,  3,  1,  3,  1,  1,  0,  1, -1, 1,  0,  0,  0,  0,  1,  1,  0,  1,  1,  1,  1,  -1, 0,  0,  1,  0,  2,  -1, 1,
    2,  -1, 0,  4,  0,  0,  -1, -1, 1,  2,  -1, 2,  1,  2,  -3, -3, 1,  0, 1,  1,  0,  0,  1,  1,  1,  0,  1,  0,  1,  0,  1,  2,  1,  0,  0,  2,  0,  1,  0,
    1,  1,  -1, 1,  0,  1,  -1, 1,  1,  1,  1,  1,  2,  0,  0,  1,  -2, 0, 1,  0,  1,  1,  1,  1,  0,  1,  0,  1,  0,  1,  1,  1,  1,  1,  2,  1,  2,  1,  0,
    1,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  -1, 1,  1,  0,  -1, 1,  0, 0,  -1, 1,  0,  -2, 0,  -1, 0,  -2, 2,  1,  1,  0,  2,  0,  2,  1,  0,  1,  1,  -1,
    1,  1,  1,  1,  0,  1,  0,  0,  0,  2,  0,  0,  2,  1,  -3, 1,  1,  1, 2,  0,  0,  1,  -1, -1, 0,  4,  3,  1,  0,  1,  1,  0,  0,  1,  0,  0,  0,  1,  1,
    1,  0,  1,  1,  0,  0,  1,  2,  1,  1,  0,  0,  1,  -2, 2,  -1, 0,  0, 0,  1,  1,  1,  1,  1,  0,  0,  1,  -1, 0,  0,  0,  1,  1,  1,  1,  0,  1,  1,  1,
    0,  1,  1,  1,  0,  1,  2,  1,  1,  1,  0,  1,  0,  0,  1,  0,  0,  0, 1,  1,  -1, -1, 0,  1,  1,  0,  -1, 0,  0,  0,  0,  1,  0,  -2, 0,  -1, 2,  0,  2,
    1,  1,  0,  1,  0,  1,  0,  1,  2,  1,  -1, -1, 0,  2,  2,  1,  1,  0, 1,  -1, 1,  -1, 0,  2,  1,  -2, 1,  1,  1,  2,  -1, 0,  1,  -1, 0,  -1, 3,  3,  1,
    -1, 0,  1,  1,  0,  2,  0,  1,  1,  0,  1,  1,  -1, 0,  1,  2,  1,  2, 1,  0,  1,  1,  0,  0,  -1, 1,  0,  1,  0,  0,  1,  1,  1,  1,  1,  0,  -1, 1,  -2,
    -1, 1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  1,  1,  2, 1,  1,  1,  1,  1,  0,  -1, 0,  0,  0,  0,  1,  0,  -1, -1, 0,  2,  1,  0,  -1, 1,
    0,  0,  0,  1,  0,  -1, 0,  -1, 1,  -2, 1,  0,  1,  0,  1,  0,  2,  1, 0,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  1,  0,  1,  -1, -1, 2,  2,  -2, 0,  1,
    1,  2,  -1, 0,  2,  -1, 0,  -1, 3,  4,  1,  0,  0,  0,  1,  0,  1,  0, 1,  0,  1,  1,  1,  0,  0,  1,  0,  1,  1,  1,  0,  0,  0,  0,  1,  -1, 1,  0,  1,
    0,  1,  1,  0,  1,  1,  0,  0,  0,  1,  -1, -1, 1,  0,  1,  1,  1,  1, 1,  1,  1,  0,  1,  1,  0,  0,  1,  0,  1,  0,  2,  1,  1,  1,  1,  0,  0,  1,  0,
    0,  1,  1,  0,  -1, 1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  -1, -1, 1, -1, 1,  -1, 2,  0,  2,  0,  1,  0,  1,  1,  0,  1,  0,  0,  -1, 1,  1,  2,  1,  1,
    1,  1,  -1, 1,  0,  0,  3,  0,  -3, 0,  0,  2,  2,  0,  0,  1,  -2, 0, -1, 3,  3,  1,  -1, 0,  0,  1,  0,  1,  1,  1,  1,  1,  2,  1,  0,  0,  2,  0,  1,
    1,  1,  0,  1,  1,  1,  1,  -1, 1,  0,  0,  0,  0,  1,  1,  1,  1,  1, 1,  0,  1,  -1, 0,  0,  0,  0,  1,  0,  1,  1,  2,  1,  1,  0,  0,  0,  1,  0,  0,
    1,  1,  1,  1,  1,  1,  1,  0,  1,  0,  0,  1,  0,  1,  0,  0,  0,  1, 2,  0,  0,  1,  1,  0,  0,  0,  -1, -2, 0,  -1, 1,  -2, 2,  0,  1,  0,  0,  -1, 1,
    1,  1,  1,  0,  -1, 1,  0,  1,  2,  1,  1,  0,  1,  0,  0,  -1, 0,  2, 1,  -1, 0,  0,  1,  2,  0,  1,  1,  -1, 0,  -1, 3,  4};

static PI_L2 int32_t AnomalyDetection_small_4_layer0_bias_tensor[2] = {-1470, -124};

static PI_L2 int32_t AnomalyDetection_small_4_layer0_mul_tensor[2] = {1, 1};

static PI_L1 uint8_t AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[2] = {0, 1};

void *AnomalyDetection_small_4_inputs[1];
void *AnomalyDetection_small_4_outputs[1];
extern struct pi_device cluster_dev;
// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref;
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref;
  int32_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref;
  int32_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref;
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref;
} AnomalyDetection_small_4_RequantizedGemm_0_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void AnomalyDetection_small_4_RequantizedGemm_0_tiling_closure(void *AnomalyDetection_small_4_RequantizedGemm_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  AnomalyDetection_small_4_RequantizedGemm_0_tiling_closure_args_t *args =
      (AnomalyDetection_small_4_RequantizedGemm_0_tiling_closure_args_t *)AnomalyDetection_small_4_RequantizedGemm_0_tiling_closure_args;
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref = args->AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref;
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref = args->AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref;
  int32_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref = args->AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref;
  int32_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref = args->AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref;
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref =
      args->AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN GEMM
  int8_t *ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref =
      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref;
  int8_t *ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref =
      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref;
  int8_t
      *ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref =
          AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref;
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
          ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref,
          NULL,
          ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref,
          ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref,
          AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref, AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref, 1, 8, 640,
          2, 1, 1, NUM_CORES);
      ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref += 640;
      ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref +=
          2;
    }
    ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref += 640 * 2;
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref;
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref;
  int32_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref;
  int32_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref;
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref;
} AnomalyDetection_small_4_RequantizedGemm_0_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void AnomalyDetection_small_4_RequantizedGemm_0_cluster_fork(void *AnomalyDetection_small_4_RequantizedGemm_0_cluster_fork_args) {
  // CLOSURE ARG CAST
  AnomalyDetection_small_4_RequantizedGemm_0_cluster_fork_args_t *args =
      (AnomalyDetection_small_4_RequantizedGemm_0_cluster_fork_args_t *)AnomalyDetection_small_4_RequantizedGemm_0_cluster_fork_args;
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref = args->AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref;
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref = args->AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref;
  int32_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref = args->AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref;
  int32_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref = args->AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref;
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref =
      args->AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref;

  // CLOSURE FUNCTION CALL
  AnomalyDetection_small_4_RequantizedGemm_0_tiling_closure_args_t AnomalyDetection_small_4_AnomalyDetection_small_4_RequantizedGemm_0_tiling_closure_args =
      (AnomalyDetection_small_4_RequantizedGemm_0_tiling_closure_args_t){
          .AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref = AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref,
          .AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref = AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref,
          .AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref = AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref,
          .AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref = AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref,
          .AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref =
              AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref};

  // AnomalyDetection_small_4_RequantizedGemm_0_tiling_closure CLOSURE CALL
  AnomalyDetection_small_4_RequantizedGemm_0_tiling_closure(&AnomalyDetection_small_4_AnomalyDetection_small_4_RequantizedGemm_0_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;
} AnomalyDetection_small_4_RequantizedGemm_0_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void AnomalyDetection_small_4_RequantizedGemm_0_closure(void *AnomalyDetection_small_4_RequantizedGemm_0_closure_args) {
  // CLOSURE ARG CAST
  AnomalyDetection_small_4_RequantizedGemm_0_closure_args_t *args =
      (AnomalyDetection_small_4_RequantizedGemm_0_closure_args_t *)AnomalyDetection_small_4_RequantizedGemm_0_closure_args;
  uint8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
      args->AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref = (int8_t *)((char *)AnomalyDetection_small_4_MEMORYARENA_L1 + 1282);
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref = (int8_t *)((char *)AnomalyDetection_small_4_MEMORYARENA_L1 + 2);
  int32_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref = (int32_t *)((char *)AnomalyDetection_small_4_MEMORYARENA_L1 + 1930);
  int32_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref = (int32_t *)((char *)AnomalyDetection_small_4_MEMORYARENA_L1 + 1922);
  int8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref = (int8_t *)((char *)AnomalyDetection_small_4_MEMORYARENA_L1 + 0);
  void *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_input_0_ref = (void *)((char *)AnomalyDetection_small_4_input_0 + 0);
  void *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_weight_tensor_ref =
      (void *)((char *)AnomalyDetection_small_4_layer0_weight_tensor + 0);
  void *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_bias_tensor_ref = (void *)((char *)AnomalyDetection_small_4_layer0_bias_tensor + 0);
  void *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_mul_tensor_ref = (void *)((char *)AnomalyDetection_small_4_layer0_mul_tensor + 0);
  void *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_output_0_ref = (void *)((char *)AnomalyDetection_small_4_output_0 + 0);

  const static char RequantizedGemm_0_L2_suffix[] = " cycles \n";

  const static char RequantizedGemm_0_L2_prefix[] = "[RequantizedGemm_0_L2][SB][4486 ops][Tile ";

  uint32_t RequantizedGemm_0_L2_egress_dma_wait_end_measurements[1];

  uint32_t RequantizedGemm_0_L2_egress_dma_wait_start_measurements[1];

  uint32_t RequantizedGemm_0_L2_ingress_dma_wait_end_measurements[1];

  uint32_t RequantizedGemm_0_L2_ingress_dma_wait_start_measurements[1];

  uint32_t RequantizedGemm_0_L2_kernel_end_measurements[1];

  uint32_t RequantizedGemm_0_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t AnomalyDetection_small_4_channel_output = (uint32_t)-1;
  uint32_t AnomalyDetection_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[*AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr];
       TILING_I <
       AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[(*AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr) + 1];
       TILING_I++) {

    RequantizedGemm_0_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    AnomalyDetection_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442432, AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref,
                      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_input_0_ref);
    mchan_transfer_1d(1443072, AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref,
                      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_weight_tensor_ref);
    mchan_transfer_1d(1441800, AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref,
                      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_bias_tensor_ref);
    mchan_transfer_1d(1441800, AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref,
                      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_layer0_mul_tensor_ref);

    // Wait for input tiles

    if (AnomalyDetection_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_small_4_channel_input);
      mchan_channel_free(AnomalyDetection_small_4_channel_input);
    }

    RequantizedGemm_0_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    RequantizedGemm_0_L2_kernel_start_measurements[TILING_I] = getCycles();

    AnomalyDetection_small_4_RequantizedGemm_0_cluster_fork_args_t AnomalyDetection_small_4_AnomalyDetection_small_4_RequantizedGemm_0_cluster_fork_args =
        (AnomalyDetection_small_4_RequantizedGemm_0_cluster_fork_args_t){
            .AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref = AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_A_ref,
            .AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref = AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_B_ref,
            .AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref = AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_mul_ref,
            .AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref = AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_C_ref,
            .AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref =
                AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)AnomalyDetection_small_4_RequantizedGemm_0_cluster_fork,
                    &AnomalyDetection_small_4_AnomalyDetection_small_4_RequantizedGemm_0_cluster_fork_args);

    RequantizedGemm_0_L2_kernel_end_measurements[TILING_I] = getCycles();

    RequantizedGemm_0_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    AnomalyDetection_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310722, AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_data_out_ref,
                      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_output_0_ref);

    // Wait for output tiles

    if (AnomalyDetection_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(AnomalyDetection_small_4_channel_output);
      mchan_channel_free(AnomalyDetection_small_4_channel_output);
    }

    RequantizedGemm_0_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr > 0)
                              ? AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[(
                                    *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_numTiles[*AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr];
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
  uint8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;
} AnomalyDetection_small_4_RequantizedGemm_0_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void AnomalyDetection_small_4_RequantizedGemm_0_closure_L3(void *AnomalyDetection_small_4_RequantizedGemm_0_closure_L3_args) {
  // CLOSURE ARG CAST
  AnomalyDetection_small_4_RequantizedGemm_0_closure_L3_args_t *args =
      (AnomalyDetection_small_4_RequantizedGemm_0_closure_L3_args_t *)AnomalyDetection_small_4_RequantizedGemm_0_closure_L3_args;
  uint8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
      args->AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  AnomalyDetection_small_4_RequantizedGemm_0_closure_args_t AnomalyDetection_small_4_AnomalyDetection_small_4_RequantizedGemm_0_closure_args =
      (AnomalyDetection_small_4_RequantizedGemm_0_closure_args_t){.AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
                                                                      AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr};

  // AnomalyDetection_small_4_RequantizedGemm_0_closure CLOSURE CALL
  AnomalyDetection_small_4_RequantizedGemm_0_closure(&AnomalyDetection_small_4_AnomalyDetection_small_4_RequantizedGemm_0_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

void RunNetwork() {

  uint8_t bu_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr = 0;
  uint8_t *AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
      &bu_AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr;
  // RUNNETRWORK LAYER AnomalyDetection_small_4_RequantizedGemm_0 CALL START

  AnomalyDetection_small_4_RequantizedGemm_0_closure_L3_args_t AnomalyDetection_small_4_AnomalyDetection_small_4_RequantizedGemm_0_closure_L3_args =
      (AnomalyDetection_small_4_RequantizedGemm_0_closure_L3_args_t){.AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr =
                                                                         AnomalyDetection_small_4_TILING_CODEGEN_L1_RequantizedGemm_0_tileIdxPtr};

  // AnomalyDetection_small_4_RequantizedGemm_0_closure_L3 CLOSURE CALL
  AnomalyDetection_small_4_RequantizedGemm_0_closure_L3(&AnomalyDetection_small_4_AnomalyDetection_small_4_RequantizedGemm_0_closure_L3_args);

  // RUNNETRWORK LAYER AnomalyDetection_small_4_RequantizedGemm_0 CALL END
}

void InitNetwork() {

  AnomalyDetection_small_4_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 1938);

  AnomalyDetection_small_4_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 642);

  AnomalyDetection_small_4_input_0 = (int8_t *)((char *)AnomalyDetection_small_4_MEMORYARENA_L2 + 2);
  AnomalyDetection_small_4_output_0 = (int8_t *)((char *)AnomalyDetection_small_4_MEMORYARENA_L2 + 0);
  AnomalyDetection_small_4_inputs[0] = (void *)AnomalyDetection_small_4_input_0;
  AnomalyDetection_small_4_outputs[0] = (void *)AnomalyDetection_small_4_output_0;
}
