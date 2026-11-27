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

int8_t *KeywordSpotting_small_4_MEMORYARENA_L1; // L2_init_template
int8_t *KeywordSpotting_small_4_MEMORYARENA_L2; // L2_init_template
int8_t *KeywordSpotting_small_4_input_0;        // L2_init_template
int8_t *KeywordSpotting_small_4_output_0;       // L2_init_template

static PI_L2 int8_t KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor[72] = {
    99,   107, -30,  119, -28, 26,   -63, 76,   114, -95, 112, 24,  96, 18,  62, -18, 100, 19,  -60, 33,  -60, -15, -53, 86,
    -102, -60, -37,  -78, 12,  -128, 117, -110, 100, 22,  -42, 80,  20, 104, 14, -41, 35,  -35, 54,  115, 75,  -57, 75,  23,
    66,   -79, -128, -50, -99, 106,  37,  54,   41,  -2,  101, -92, 8,  -88, 40, -45, 40,  -27, 107, -77, -77, -77, 116, 43};

static PI_L2 int32_t KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor[8] = {473, 473, 473, 473, 473, 473, 473, 473};

static PI_L2 int32_t KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor[8] = {117993, 46419,   391189,  -30944,
                                                                                                               166165, -104412, -222232, 132940};

static PI_L2 int8_t KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor[72] = {
    123, -106, -127, -100, -86, 52,  46,  106,  -66, -87, 68,   -52, 78,   -30, 73,  -99, -65,  39,   27,  -33, 76, 87,   -93, -68,
    117, -43,  -45,  -124, -73, 32,  -17, -93,  3,   -87, -109, -71, -112, -82, 127, 24,  39,   -119, -84, -43, 20, -113, -55, -77,
    0,   -48,  -9,   -87,  -88, -75, -44, -101, 107, -25, 110,  40,  -108, 89,  -35, -49, -106, -127, 37,  -28, 50, -105, 95,  -94};

static PI_L2 int32_t KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor[8] = {688, 688, 688, 688, 688, 688, 688, 688};

static PI_L2 int32_t KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1add_tensor[8] = {327062, -113131, 551862, 189068,
                                                                                                                      471827, -36476,  449284, 255184};

static PI_L2 int8_t KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor[64] = {
    -106, 127, 127,  127,  127,  -128, 127, -128, -67,  -128, 127,  -128, 127,  127,  127, -128, 127, 127,  79,   -128, 127, -92,
    -128, 127, 127,  -128, -128, -29,  127, -128, -128, -128, -128, 127,  127,  -128, 127, 127,  111, -128, 127,  127,  127, -107,
    -128, 65,  -108, -128, 127,  127,  -60, -128, 9,    -36,  127,  127,  -128, -128, 127, 127,  -30, -128, -128, -128};

static PI_L2 int32_t KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2mul_tensor[8] = {151, 151, 151, 151, 151, 151, 151, 151};

static PI_L2 int32_t KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2add_tensor[8] = {68573,  53293,  43196,   1520,
                                                                                                                      458300, -45796, -477186, 155164};

static PI_L2 int8_t KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor[72] = {
    -42, 43,  82, 59, -114, -77, -20,  124,  19, -33, 53,  -49, -83, 94,   -58, -26,  -128, 86, 97,  47,  -90,  -127, -104, 96,
    62,  108, 67, 33, -1,   -98, -110, -120, 53, -63, -26, -74, -23, -91,  -84, 43,   -38,  79, -41, -94, -23,  -62,  -39,  -122,
    72,  -90, 65, 58, 92,   -99, 92,   -61,  48, 121, -18, -1,  -30, -107, 62,  -128, 80,   96, 122, -30, -106, 29,   71,   -128};

static PI_L2 int32_t KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor[8] = {679, 679, 679, 679, 679, 679, 679, 679};

static PI_L2 int32_t KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3add_tensor[8] = {481546, 200993, 186984, 189067,
                                                                                                                      284144, 276920, 139472, 397262};

static PI_L2 int8_t KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor[64] = {
    -29, -78,  -11,  -64,  -53,  -41, -123, 106, 109, -20,  -15,  -53, -117, -126, 48,  -71, -83, -10, -43, -42, 4,  -28,
    -45, -62,  -106, 109,  -52,  34,  -45,  11,  121, 60,   -112, 51,  123,  34,   87,  127, -20, 27,  -90, -27, 96, 66,
    -82, -104, -89,  -128, -100, -32, 88,   22,  -99, -104, 65,   -96, -16,  62,   -60, -14, -11, -31, -66, -116};

static PI_L2 int32_t KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4mul_tensor[8] = {715, 715, 715, 715, 715, 715, 715, 715};

static PI_L2 int32_t KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4add_tensor[8] = {86813,  135178, -194987, 58608,
                                                                                                                      360564, -53830, 198203,  49973};

static PI_L2 int8_t KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor[96] = {
    -104, -69, 125, -62, -87,  31,  36,  71,  98,  72, -128, 11,   78,  -12,  -76,  123, -48, -74, -116, 6,   57,  29,  26,  -98,
    -121, 2,   118, 74,  -75,  -17, -95, -62, 23,  70, 107,  -119, 86,  -91,  48,   109, 2,   117, -110, -49, 75,  -28, -26, -54,
    89,   63,  41,  -73, -105, 14,  38,  -60, -36, 87, 10,   6,    -32, -117, -121, -62, -66, 40,  -38,  -51, 123, 45,  92,  -62,
    -53,  47,  -86, -84, -6,   -47, -97, 77,  104, 21, -22,  -120, -47, 33,   61,   -16, -51, 72,  -103, 82,  -50, 2,   -26, 16};

static PI_L2 int32_t KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor[12] = {-3461, 8543, -240, 9167,  11274, -5617,
                                                                                                        -8443, 9418, 9201, -8839, -838,  -11434};

static PI_L2 int32_t KeywordSpotting_small_4_RequantShift_final_mul_tensor[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles[2] = {0, 1};

static PI_L1 uint16_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x[2] = {47, 4};

static PI_L1 uint16_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x[2] = {23, 2};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top[2] = {1, 0};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom[2] = {0, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[2] = {0, 2};

static PI_L1 uint32_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cmd[2] = {1442262, 1441832};

static PI_L1 uint32_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[2] = {1311640, 1310800};

static PI_L1 uint8_t
    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_numTiles
        [2] = {0, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[2] = {0, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[2] = {0, 1};

static PI_L1 uint8_t
    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_numTiles
        [2] = {0, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[2] = {0, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[2] = {0, 1};

static PI_L1 uint8_t
    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_numTiles
        [2] = {0, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_numTiles[2] = {0, 1};

static PI_L1 uint8_t KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles[2] = {0, 1};

void *KeywordSpotting_small_4_inputs[1];
void *KeywordSpotting_small_4_outputs[1];
extern struct pi_device cluster_dev;
// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure(
    void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 1, 49, 10] -> [1, 49, 10, 1] (Name: _MERGE_CONVRQ_PASS_0_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 1;

  uint16_t dimLen_2 = 49;

  uint16_t dimLen_3 = 10;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_2 / NUM_CORES;
    const uint32_t leftover = dimLen_2 - baseChunk * NUM_CORES;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_2 = offset; i_2 < offset + chunk; i_2++) {

      for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

          ((int8_t (*)[dimLen_2][dimLen_3][dimLen_1])
               KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref)[i_0][i_2][i_3][i_1] =
              ((int8_t (*)[dimLen_1][dimLen_2][dimLen_3])
                   KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork(
    void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t
      KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args =
          (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t){
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure(
      &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void
KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args;
  int8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 490);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref = (void *)((char *)KeywordSpotting_small_4_input_0 + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed + 0);

  const static char _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_prefix[] = "[_MERGE_CONVRQ_PASS_0_input_0_transpose_L2][SB][0 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t KeywordSpotting_small_4_channel_output = (uint32_t)-1;
  uint32_t KeywordSpotting_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
           [*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr];
       TILING_I < KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
                      [(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442282, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t
        KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args =
            (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t){
                .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
                    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
                    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork,
                    &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311210, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr > 0)
                              ? KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles[(
                                    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
                         [*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_end_measurements[PROFILING_I] -
               _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3(
    void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args;
  int8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t
      KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args =
          (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t){
              .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure(
      &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer;
  uint16_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref;
  uint16_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_tiling_closure(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args;
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer;
  uint16_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref;
  uint16_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_i8_u8_i8(
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref, KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer, NULL,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref, 1, 18, 10,
      *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref, 1, 5,
      *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref, 8, 3, 3,
      *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref,
      *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref, 1, 1, 2, 2, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer;
  uint16_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref;
  uint16_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_cluster_fork(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args;
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer;
  uint16_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref;
  uint16_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args_t){
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_tiling_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_tiling_closure(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_args;
  int8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint16_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref =
      (uint16_t *)((char *)KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x + 0);
  uint16_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref =
      (uint16_t *)((char *)KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x + 0);
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref =
      (uint8_t *)((char *)KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top + 0);
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref =
      (uint8_t *)((char *)KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom + 0);
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 920);
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1544);
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1616);
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1648);
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor + 0);
  void *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed +
                   0);
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer = (void *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1400);

  const static char _MERGE_CONVRQ_PASS_0_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_0_L2_prefix[] = "[_MERGE_CONVRQ_PASS_0_L2][SB][18375 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_end_measurements[2];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_start_measurements[2];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_end_measurements[2];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_start_measurements[2];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_kernel_end_measurements[2];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_kernel_start_measurements[2];

  // Initialize DMA futures
  uint32_t KeywordSpotting_small_4_channel_output = (uint32_t)-1;
  uint32_t KeywordSpotting_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       TILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr) +
                                                                               1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cmd[TILING_I],
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref);

    // UPDATE VARIABLE KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref
    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
        (void *)((char *)(KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref) + 450);

    mchan_transfer_1d(1441864, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_0weight_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0add_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0mul_tensor_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_L2_kernel_start_measurements[TILING_I] = getCycles();

    // UPDATE VARIABLE KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref
    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref =
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom[TILING_I];

    // UPDATE VARIABLE KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref
    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref =
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top[TILING_I];

    // UPDATE VARIABLE KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref
    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref =
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x[TILING_I];

    // UPDATE VARIABLE KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref
    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref =
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x[TILING_I];

    KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args =
        (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args_t){
            .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_top_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_padding_y_bottom_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_cluster_fork,
                    &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_cluster_fork_args);

    _MERGE_CONVRQ_PASS_0_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[TILING_I],
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref);

    // UPDATE VARIABLE
    // KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref
    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref =
        (void
             *)((char
                     *)(KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref) +
                920);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr > 0)
                              ? KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(
                                    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_0_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_0_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_L3(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args;
  int8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_args_t){
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tiling_closure(
    void
        *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tiling_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tiling_closure_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tiling_closure_args;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 25, 5, 8] -> [1, 8, 25, 5] (Name: _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose,
  // Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 25;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 8;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_3 / NUM_CORES;
    const uint32_t leftover = dimLen_3 - baseChunk * NUM_CORES;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_3 = offset; i_3 < offset + chunk; i_3++) {

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {

          ((uint8_t (*)[dimLen_3][dimLen_1][dimLen_2])
               KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref)
              [i_0][i_3][i_1][i_2] =
                  ((uint8_t (*)[dimLen_1][dimLen_2][dimLen_3])
                       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref)
                      [i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_cluster_fork(
    void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_cluster_fork_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_cluster_fork_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_cluster_fork_args;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tiling_closure_args_t
      KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tiling_closure_args =
          (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tiling_closure_args_t){
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tiling_closure CLOSURE
  // CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tiling_closure(
      &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure(
    void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_args;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref =
          (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref =
          (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1000);
  void *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed +
                   0);
  void *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_ref =
          (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor + 0);

  const static char _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_prefix[] =
      "[_MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2][SB][0 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t KeywordSpotting_small_4_channel_output = (uint32_t)-1;
  uint32_t KeywordSpotting_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (
      int TILING_I = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_numTiles
          [*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr];
      TILING_I <
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_numTiles
          [(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr) +
           1];
      TILING_I++) {

    _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] =
        getCycles();

    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442792,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] =
        getCycles();

    _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_kernel_start_measurements[TILING_I] =
        getCycles();

    KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_cluster_fork_args_t
        KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_cluster_fork_args =
            (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_cluster_fork_args_t){
                .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref =
                    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_in_ref,
                .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref =
                    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref};

    pi_cl_team_fork(
        NUM_CORES,
        (void *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_cluster_fork,
        &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_start_measurements[TILING_I] =
        getCycles();

    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311720,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_data_out_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_end_measurements[TILING_I] =
        getCycles();

    // CLOSE TILING LOOP
  }
  *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr +=
      1;

  // Deinitialize DMA futures

  StopTimer();
  for (
      int PROFILING_I =
          ((*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr >
            0)
               ? KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_numTiles
                     [(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr -
                       1)]
               : 0);
      PROFILING_I <
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_numTiles
          [*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr];
      PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_prefix, PROFILING_I,
           "Input DMA took ",
           _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_end_measurements
                   [PROFILING_I] -
               _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_start_measurements
                   [PROFILING_I],
           _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_suffix);

    printf(
        "%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_prefix, PROFILING_I,
        "Kernel took ",
        _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_kernel_end_measurements[PROFILING_I] -
            _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_kernel_start_measurements[PROFILING_I],
        _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_prefix, PROFILING_I,
           "Output DMA took ",
           _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_end_measurements
                   [PROFILING_I] -
               _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_start_measurements
                   [PROFILING_I],
           _MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_L3(
    void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_L3_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_L3_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_L3_args;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_args_t
      KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_args =
          (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_args_t){
              .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
                  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed,
              .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor =
                  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure(
      &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_tiling_closure(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args;
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_depthwise_u8_u8_i8(
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref, KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer, NULL,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
      NULL, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref, 1,
      18, 5, 25, 8, 5, 25, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_cluster_fork(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args;
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args_t){
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_tiling_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_tiling_closure(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_args;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1152);
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 3152);
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 3224);
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 3256);
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2152);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1add_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1add_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor + 0);
  void *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed +
                   0);
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer = (void *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);

  const static char _MERGE_CONVRQ_PASS_1_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_1_L2_prefix[] = "[_MERGE_CONVRQ_PASS_1_L2][SB][18375 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_1_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_1_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t KeywordSpotting_small_4_channel_output = (uint32_t)-1;
  uint32_t KeywordSpotting_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr];
       TILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr) +
                                                                               1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442792, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_ref);
    mchan_transfer_1d(1441864, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_1weight_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1add_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1mul_tensor_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_1_L2_kernel_start_measurements[TILING_I] = getCycles();

    KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args =
        (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args_t){
            .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_cluster_fork,
                    &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_cluster_fork_args);

    _MERGE_CONVRQ_PASS_1_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311720, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr > 0)
                              ? KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[(
                                    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_1_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_1_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_1_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_1_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_1_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_1_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_1_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_1_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_1_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_1_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_L3(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_args_t){
          .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor =
              KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_tiling_closure(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args;
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_u8_u8_i8(
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref, KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer, NULL,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref, 1, 18, 5,
      25, 8, 5, 25, 8, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_cluster_fork(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args;
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args_t){
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_tiling_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_tiling_closure(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_args;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2128);
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2192);
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2224);
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1000);
  void *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed +
                   0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2add_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2add_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2mul_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2mul_tensor + 0);
  void *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed +
                   0);
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer = (void *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2000);

  const static char _MERGE_CONVRQ_PASS_2_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_2_L2_prefix[] = "[_MERGE_CONVRQ_PASS_2_L2][SB][16375 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t KeywordSpotting_small_4_channel_output = (uint32_t)-1;
  uint32_t KeywordSpotting_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr];
       TILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr) +
                                                                               1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442792, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(1441856, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_2weight_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2add_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_blocks0_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2mul_tensor_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_2_L2_kernel_start_measurements[TILING_I] = getCycles();

    KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args =
        (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args_t){
            .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_in_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_weight_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_mul_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_add_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_cluster_fork,
                    &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_cluster_fork_args);

    _MERGE_CONVRQ_PASS_2_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311720, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_data_out_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr > 0)
                              ? KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[(
                                    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_2_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_2_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_2_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_2_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_2_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_2_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_2_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_2_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_2_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_2_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_L3(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_args_t){
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tiling_closure(
    void
        *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tiling_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tiling_closure_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tiling_closure_args;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 25, 5, 8] -> [1, 8, 25, 5] (Name: _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose,
  // Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 25;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 8;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_3 / NUM_CORES;
    const uint32_t leftover = dimLen_3 - baseChunk * NUM_CORES;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_3 = offset; i_3 < offset + chunk; i_3++) {

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {

          ((uint8_t (*)[dimLen_3][dimLen_1][dimLen_2])
               KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref)
              [i_0][i_3][i_1][i_2] =
                  ((uint8_t (*)[dimLen_1][dimLen_2][dimLen_3])
                       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref)
                      [i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_cluster_fork(
    void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_cluster_fork_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_cluster_fork_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_cluster_fork_args;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tiling_closure_args_t
      KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tiling_closure_args =
          (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tiling_closure_args_t){
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tiling_closure CLOSURE
  // CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tiling_closure(
      &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure(
    void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_args;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref =
          (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref =
          (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1000);
  void *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed +
                   0);
  void *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref =
          (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor + 0);

  const static char _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_prefix[] =
      "[_MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2][SB][0 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t KeywordSpotting_small_4_channel_output = (uint32_t)-1;
  uint32_t KeywordSpotting_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (
      int TILING_I = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_numTiles
          [*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr];
      TILING_I <
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_numTiles
          [(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr) +
           1];
      TILING_I++) {

    _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] =
        getCycles();

    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442792,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] =
        getCycles();

    _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_kernel_start_measurements[TILING_I] =
        getCycles();

    KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_cluster_fork_args_t
        KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_cluster_fork_args =
            (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_cluster_fork_args_t){
                .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref =
                    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_in_ref,
                .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref =
                    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref};

    pi_cl_team_fork(
        NUM_CORES,
        (void *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_cluster_fork,
        &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_start_measurements[TILING_I] =
        getCycles();

    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311720,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_data_out_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_end_measurements[TILING_I] =
        getCycles();

    // CLOSE TILING LOOP
  }
  *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr +=
      1;

  // Deinitialize DMA futures

  StopTimer();
  for (
      int PROFILING_I =
          ((*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr >
            0)
               ? KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_numTiles
                     [(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr -
                       1)]
               : 0);
      PROFILING_I <
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_numTiles
          [*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr];
      PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_prefix, PROFILING_I,
           "Input DMA took ",
           _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_end_measurements
                   [PROFILING_I] -
               _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_start_measurements
                   [PROFILING_I],
           _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_suffix);

    printf(
        "%s%u] %s%u%s", _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_prefix, PROFILING_I,
        "Kernel took ",
        _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_kernel_end_measurements[PROFILING_I] -
            _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_kernel_start_measurements[PROFILING_I],
        _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_prefix, PROFILING_I,
           "Output DMA took ",
           _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_end_measurements
                   [PROFILING_I] -
               _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_start_measurements
                   [PROFILING_I],
           _MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_L3(
    void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_L3_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_L3_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_L3_args;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_args_t
      KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_args =
          (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_args_t){
              .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
                  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
              .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
                  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure(
      &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_tiling_closure(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args;
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_depthwise_u8_u8_i8(
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref, KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer, NULL,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
      NULL, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref, 1,
      18, 5, 25, 8, 5, 25, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_cluster_fork(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args;
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args_t){
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_tiling_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_tiling_closure(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_args;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1152);
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 3152);
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 3224);
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 3256);
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2152);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3add_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3add_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor + 0);
  void *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed +
                   0);
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer = (void *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);

  const static char _MERGE_CONVRQ_PASS_3_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_3_L2_prefix[] = "[_MERGE_CONVRQ_PASS_3_L2][SB][18375 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_3_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_3_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t KeywordSpotting_small_4_channel_output = (uint32_t)-1;
  uint32_t KeywordSpotting_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr];
       TILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr) +
                                                                               1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442792, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_ref);
    mchan_transfer_1d(1441864, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_3weight_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3add_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3mul_tensor_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_3_L2_kernel_start_measurements[TILING_I] = getCycles();

    KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args =
        (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args_t){
            .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_in_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_weight_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_mul_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_add_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_cluster_fork,
                    &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_cluster_fork_args);

    _MERGE_CONVRQ_PASS_3_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311720, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_data_out_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr > 0)
                              ? KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[(
                                    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_3_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_3_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_3_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_3_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_3_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_3_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_3_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_3_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_3_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_3_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_L3(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_args_t){
          .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
              KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_tiling_closure(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args;
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_u8_u8_i8(
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref, KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer, NULL,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref, 1, 18, 5,
      25, 8, 5, 25, 8, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_cluster_fork(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args;
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer = args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args_t){
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_tiling_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_tiling_closure(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_args;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2128);
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2192);
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2224);
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1000);
  void *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed +
                   0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4add_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4add_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4mul_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4mul_tensor + 0);
  void *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed +
                   0);
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer = (void *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 2000);

  const static char _MERGE_CONVRQ_PASS_4_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_4_L2_prefix[] = "[_MERGE_CONVRQ_PASS_4_L2][SB][16375 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t KeywordSpotting_small_4_channel_output = (uint32_t)-1;
  uint32_t KeywordSpotting_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr];
       TILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr) +
                                                                               1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442792, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed_ref);
    mchan_transfer_1d(1441856, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_PACT_CONV2D_PASS_4weight_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4add_tensor_ref);
    mchan_transfer_1d(1441824, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_blocks1_QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4mul_tensor_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_4_L2_kernel_start_measurements[TILING_I] = getCycles();

    KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args =
        (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args_t){
            .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_in_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_weight_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_mul_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_add_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_cluster_fork,
                    &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_cluster_fork_args);

    _MERGE_CONVRQ_PASS_4_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311720, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_data_out_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr > 0)
                              ? KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[(
                                    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_4_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_4_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_4_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_4_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_4_L2_kernel_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_4_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_4_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_4_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_CONVRQ_PASS_4_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_4_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_L3(void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args_t *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_args_t){
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tiling_closure(
    void
        *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tiling_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tiling_closure_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tiling_closure_args;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 25, 5, 8] -> [1, 8, 25, 5] (Name: _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose,
  // Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 25;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 8;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_3 / NUM_CORES;
    const uint32_t leftover = dimLen_3 - baseChunk * NUM_CORES;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_3 = offset; i_3 < offset + chunk; i_3++) {

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {

          ((uint8_t (*)[dimLen_3][dimLen_1][dimLen_2])
               KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref)
              [i_0][i_3][i_1][i_2] =
                  ((uint8_t (*)[dimLen_1][dimLen_2][dimLen_3])
                       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref)
                      [i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_cluster_fork(
    void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_cluster_fork_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_cluster_fork_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_cluster_fork_args;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tiling_closure_args_t
      KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tiling_closure_args =
          (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tiling_closure_args_t){
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tiling_closure CLOSURE
  // CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tiling_closure(
      &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure(
    void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_args;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref =
          (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref =
          (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1000);
  void *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref =
          (void *)((char *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed +
                   0);
  void *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_ref =
          (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor + 0);

  const static char _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_prefix[] =
      "[_MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2][SB][0 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t KeywordSpotting_small_4_channel_output = (uint32_t)-1;
  uint32_t KeywordSpotting_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (
      int TILING_I = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_numTiles
          [*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr];
      TILING_I <
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_numTiles
          [(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr) +
           1];
      TILING_I++) {

    _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] =
        getCycles();

    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442792,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }

    _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] =
        getCycles();

    _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_kernel_start_measurements[TILING_I] =
        getCycles();

    KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_cluster_fork_args_t
        KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_cluster_fork_args =
            (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_cluster_fork_args_t){
                .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref =
                    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_in_ref,
                .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref =
                    KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref};

    pi_cl_team_fork(
        NUM_CORES,
        (void *)KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_cluster_fork,
        &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_start_measurements[TILING_I] =
        getCycles();

    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311720,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_data_out_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

    _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_end_measurements[TILING_I] =
        getCycles();

    // CLOSE TILING LOOP
  }
  *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr +=
      1;

  // Deinitialize DMA futures

  StopTimer();
  for (
      int PROFILING_I =
          ((*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr >
            0)
               ? KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_numTiles
                     [(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr -
                       1)]
               : 0);
      PROFILING_I <
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_numTiles
          [*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr];
      PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_prefix, PROFILING_I,
           "Input DMA took ",
           _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_end_measurements
                   [PROFILING_I] -
               _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_ingress_dma_wait_start_measurements
                   [PROFILING_I],
           _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_suffix);

    printf(
        "%s%u] %s%u%s", _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_prefix, PROFILING_I,
        "Kernel took ",
        _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_kernel_end_measurements[PROFILING_I] -
            _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_kernel_start_measurements[PROFILING_I],
        _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_prefix, PROFILING_I,
           "Output DMA took ",
           _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_end_measurements
                   [PROFILING_I] -
               _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_egress_dma_wait_start_measurements
                   [PROFILING_I],
           _MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_L3(
    void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_L3_args_t *args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_L3_args_t *)
          KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_L3_args;
  uint8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      args->KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_args_t
      KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_args =
          (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_args_t){
              .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
                  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed,
              .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor =
                  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure(
      &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_in_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref;
} KeywordSpotting_small_4_poolGlobalAveragePool_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4_poolGlobalAveragePool_tiling_closure(void *KeywordSpotting_small_4_poolGlobalAveragePool_tiling_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4_poolGlobalAveragePool_tiling_closure_args_t *args =
      (KeywordSpotting_small_4_poolGlobalAveragePool_tiling_closure_args_t *)KeywordSpotting_small_4_poolGlobalAveragePool_tiling_closure_args;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_in_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_in_ref;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref;

  // CLOSURE FUNCTION CALL

  // ReduceMean (Name: poolGlobalAveragePool, Op: ReduceMean)
  BEGIN_SINGLE_CORE
  int32_t KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_accumulator = 0;

  uint8_t *dummy_KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref =
      KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref;

  for (uint32_t i_0 = 0; i_0 < 1; i_0++) {
    for (uint32_t i_1 = 0; i_1 < 8; i_1++) {
      KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_accumulator = 128 * 125;
      for (uint32_t i_2 = 0; i_2 < 25; i_2++) {
        for (uint32_t i_3 = 0; i_3 < 5; i_3++) {
          KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_accumulator +=
              ((uint8_t (*)[8][25][5])KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
      int8_t KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_sgn = 0;
      KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_sgn =
          -(KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_accumulator < 0) +
          (KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_accumulator >= 0);
      *dummy_KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref++ =
          (uint8_t)((KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_accumulator +
                     KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref_sgn * (125 >> 1)) /
                        125 +
                    0);
    }
  }
  END_SINGLE_CORE

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor;
  uint8_t *KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr;
} KeywordSpotting_small_4_poolGlobalAveragePool_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4_poolGlobalAveragePool_closure(void *KeywordSpotting_small_4_poolGlobalAveragePool_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4_poolGlobalAveragePool_closure_args_t *args =
      (KeywordSpotting_small_4_poolGlobalAveragePool_closure_args_t *)KeywordSpotting_small_4_poolGlobalAveragePool_closure_args;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor;
  uint8_t *KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor = args->KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_in_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 1000);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_poolGlobalAveragePool_output_0_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor + 0);

  const static char poolGlobalAveragePool_L2_suffix[] = " cycles \n";

  const static char poolGlobalAveragePool_L2_prefix[] = "[poolGlobalAveragePool_L2][SB][0 ops][Tile ";

  uint32_t poolGlobalAveragePool_L2_egress_dma_wait_end_measurements[1];

  uint32_t poolGlobalAveragePool_L2_egress_dma_wait_start_measurements[1];

  uint32_t poolGlobalAveragePool_L2_ingress_dma_wait_end_measurements[1];

  uint32_t poolGlobalAveragePool_L2_ingress_dma_wait_start_measurements[1];

  uint32_t poolGlobalAveragePool_L2_kernel_end_measurements[1];

  uint32_t poolGlobalAveragePool_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t KeywordSpotting_small_4_channel_output = (uint32_t)-1;
  uint32_t KeywordSpotting_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_numTiles
           [*KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr];
       TILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_numTiles[(*KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr) +
                                                                                1];
       TILING_I++) {

    poolGlobalAveragePool_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1442792, KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_in_ref,
        KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }

    poolGlobalAveragePool_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    poolGlobalAveragePool_L2_kernel_start_measurements[TILING_I] = getCycles();

    KeywordSpotting_small_4_poolGlobalAveragePool_tiling_closure_args_t
        KeywordSpotting_small_4_KeywordSpotting_small_4_poolGlobalAveragePool_tiling_closure_args =
            (KeywordSpotting_small_4_poolGlobalAveragePool_tiling_closure_args_t){
                .KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_in_ref =
                    KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_in_ref,
                .KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref =
                    KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref};

    // KeywordSpotting_small_4_poolGlobalAveragePool_tiling_closure CLOSURE CALL
    KeywordSpotting_small_4_poolGlobalAveragePool_tiling_closure(&KeywordSpotting_small_4_KeywordSpotting_small_4_poolGlobalAveragePool_tiling_closure_args);

    poolGlobalAveragePool_L2_kernel_end_measurements[TILING_I] = getCycles();

    poolGlobalAveragePool_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310728, KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_data_out_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_poolGlobalAveragePool_output_0_tensor_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

    poolGlobalAveragePool_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr > 0)
                              ? KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_numTiles[(
                                    *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", poolGlobalAveragePool_L2_prefix, PROFILING_I, "Input DMA took ",
           poolGlobalAveragePool_L2_ingress_dma_wait_end_measurements[PROFILING_I] - poolGlobalAveragePool_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           poolGlobalAveragePool_L2_suffix);

    printf("%s%u] %s%u%s", poolGlobalAveragePool_L2_prefix, PROFILING_I, "Kernel took ",
           poolGlobalAveragePool_L2_kernel_end_measurements[PROFILING_I] - poolGlobalAveragePool_L2_kernel_start_measurements[PROFILING_I],
           poolGlobalAveragePool_L2_suffix);

    printf("%s%u] %s%u%s", poolGlobalAveragePool_L2_prefix, PROFILING_I, "Output DMA took ",
           poolGlobalAveragePool_L2_egress_dma_wait_end_measurements[PROFILING_I] - poolGlobalAveragePool_L2_egress_dma_wait_start_measurements[PROFILING_I],
           poolGlobalAveragePool_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor;
  uint8_t *KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr;
} KeywordSpotting_small_4_poolGlobalAveragePool_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4_poolGlobalAveragePool_closure_L3(void *KeywordSpotting_small_4_poolGlobalAveragePool_closure_L3_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4_poolGlobalAveragePool_closure_L3_args_t *args =
      (KeywordSpotting_small_4_poolGlobalAveragePool_closure_L3_args_t *)KeywordSpotting_small_4_poolGlobalAveragePool_closure_L3_args;
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor =
      args->KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor;
  uint8_t *KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor = args->KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4_poolGlobalAveragePool_closure_args_t KeywordSpotting_small_4_KeywordSpotting_small_4_poolGlobalAveragePool_closure_args =
      (KeywordSpotting_small_4_poolGlobalAveragePool_closure_args_t){
          .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor =
              KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor,
          .KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor = KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr};

  // KeywordSpotting_small_4_poolGlobalAveragePool_closure CLOSURE CALL
  KeywordSpotting_small_4_poolGlobalAveragePool_closure(&KeywordSpotting_small_4_KeywordSpotting_small_4_poolGlobalAveragePool_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref;
} KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure(void *KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args_t *)KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref = args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref = args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref = args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN GEMM
  int8_t *ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref =
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref;
  int8_t *ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref =
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref;
  int8_t *
      ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref =
          KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref;
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
      pulp_nn_linear_u8_i8_i8(
          ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref,
          NULL,
          ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref,
          ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref,
          KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref, 1, 7, 8,
          12, 1, 1, NUM_CORES);
      ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref += 8;
      ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref +=
          12;
    }
    ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref +=
        8 * 12;
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref;
} KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork(void *KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args_t *args =
      (KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args_t *)KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref = args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref = args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref;
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref = args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref;
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args =
      (KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args_t){
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref};

  // KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4_Flatten_output_0_tensor;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure(void *KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_args_t *args =
      (KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_args_t *)KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_args;
  uint8_t *KeywordSpotting_small_4_Flatten_output_0_tensor = args->KeywordSpotting_small_4_Flatten_output_0_tensor;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 204);
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 12);
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 156);
  int32_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref = (int32_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 108);
  int8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L1 + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_Flatten_output_0_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_Flatten_output_0_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_RequantShift_final_mul_tensor_ref =
      (void *)((char *)KeywordSpotting_small_4_RequantShift_final_mul_tensor + 0);
  void *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_output_0_ref = (void *)((char *)KeywordSpotting_small_4_output_0 + 0);

  const static char _MERGE_GEMMRQ_PASS_0_L2_suffix[] = " cycles \n";

  const static char _MERGE_GEMMRQ_PASS_0_L2_prefix[] = "[_MERGE_GEMMRQ_PASS_0_L2][SB][252 ops][Tile ";

  uint32_t _MERGE_GEMMRQ_PASS_0_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_GEMMRQ_PASS_0_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_GEMMRQ_PASS_0_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_GEMMRQ_PASS_0_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_GEMMRQ_PASS_0_L2_kernel_end_measurements[1];

  uint32_t _MERGE_GEMMRQ_PASS_0_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t KeywordSpotting_small_4_channel_output = (uint32_t)-1;
  uint32_t KeywordSpotting_small_4_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr];
       TILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles[(*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr) +
                                                                               1];
       TILING_I++) {

    _MERGE_GEMMRQ_PASS_0_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    KeywordSpotting_small_4_channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441800, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_Flatten_output_0_tensor_ref);
    mchan_transfer_1d(1441888, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0weight_tensor_ref);
    mchan_transfer_1d(1441840, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0__QL_REPLACED__INTEGERIZE_PACT_LIN_PASS_0bias_tensor_ref);
    mchan_transfer_1d(1441840, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_RequantShift_final_mul_tensor_ref);

    // Wait for input tiles

    if (KeywordSpotting_small_4_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_input);
      mchan_channel_free(KeywordSpotting_small_4_channel_input);
    }

    _MERGE_GEMMRQ_PASS_0_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMMRQ_PASS_0_L2_kernel_start_measurements[TILING_I] = getCycles();

    KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args =
        (KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args_t){
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_A_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_B_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_mul_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref = KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_C_ref,
            .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref =
                KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork,
                    &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_cluster_fork_args);

    _MERGE_GEMMRQ_PASS_0_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_GEMMRQ_PASS_0_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    KeywordSpotting_small_4_channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310732, KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_data_out_ref,
                      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_output_0_ref);

    // Wait for output tiles

    if (KeywordSpotting_small_4_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(KeywordSpotting_small_4_channel_output);
      mchan_channel_free(KeywordSpotting_small_4_channel_output);
    }

    _MERGE_GEMMRQ_PASS_0_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr > 0)
                              ? KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles[(
                                    *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I <
       KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_numTiles[*KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_GEMMRQ_PASS_0_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_GEMMRQ_PASS_0_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_GEMMRQ_PASS_0_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMMRQ_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMMRQ_PASS_0_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_GEMMRQ_PASS_0_L2_kernel_end_measurements[PROFILING_I] - _MERGE_GEMMRQ_PASS_0_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_GEMMRQ_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_GEMMRQ_PASS_0_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_GEMMRQ_PASS_0_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_GEMMRQ_PASS_0_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_GEMMRQ_PASS_0_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *KeywordSpotting_small_4_Flatten_output_0_tensor;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;
} KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_L3(void *KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args_t *args =
      (KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args_t *)KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args;
  uint8_t *KeywordSpotting_small_4_Flatten_output_0_tensor = args->KeywordSpotting_small_4_Flatten_output_0_tensor;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
      args->KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_args =
      (KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_args_t){.KeywordSpotting_small_4_Flatten_output_0_tensor =
                                                                        KeywordSpotting_small_4_Flatten_output_0_tensor,
                                                                    .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
                                                                        KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure CLOSURE CALL
  KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

void RunNetwork() {
  int8_t *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed; // L2_init_template
  uint8_t
      *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor;                                 // L2_init_template
  uint8_t
      *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t
      *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor;                                 // L2_init_template
  uint8_t
      *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t
      *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed; // L2_init_template
  uint8_t *KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor;                                 // L2_init_template
  uint8_t *KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor;                                                                 // L2_init_template
  uint8_t *KeywordSpotting_small_4_Flatten_output_0_tensor;                                                                               // L2_init_template
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_buffer;                                                                              // L2_init_template
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_buffer;                                                                              // L2_init_template
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_buffer;                                                                              // L2_init_template
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_buffer;                                                                              // L2_init_template
  void *KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_buffer;                                                                              // L2_init_template

  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  uint8_t
      bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          0;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr;
  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr;
  uint8_t
      bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          0;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr;
  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr;
  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr;
  uint8_t
      bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          0;
  uint8_t *
      KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr =
          &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr;
  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr;
  uint8_t bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr = 0;
  uint8_t *KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
      &bu_KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr;
  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose CALL START

  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 1000);
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t
      KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args =
          (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t){
              .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3 CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3(
      &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args);

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transpose CALL END

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0 CALL START

  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 0);
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args_t){
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed = KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_input_0_transposed,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_L3 CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_L3(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0_closure_L3_args);

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0 CALL END

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose CALL
  // START

  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 1000);
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_L3_args_t
      KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_L3_args =
          (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_L3_args_t){
              .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed =
                  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transposed,
              .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor =
                  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_L3 CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_L3(
      &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose_closure_L3_args);

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_0__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor_pre_transpose CALL END

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1 CALL START

  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 0);
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args_t){
          .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor =
              KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_0Clip_output_0_tensor,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_L3 CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_L3(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1_closure_L3_args);

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1 CALL END

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2 CALL START

  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 1000);
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args_t){
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_1__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_1Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_L3 CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_L3(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2_closure_L3_args);

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2 CALL END

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose CALL
  // START

  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 0);
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_L3_args_t
      KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_L3_args =
          (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_L3_args_t){
              .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed =
                  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transposed,
              .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
                  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_L3 CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_L3(
      &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose_closure_L3_args);

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_2__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor_pre_transpose CALL END

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3 CALL START

  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 1000);
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args_t){
          .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor =
              KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_2Clip_output_0_tensor,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_3_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_L3 CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_L3(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3_closure_L3_args);

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3 CALL END

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4 CALL START

  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 0);
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args =
      (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args_t){
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_3__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_3Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
              KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_L3 CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_L3(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4_closure_L3_args);

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4 CALL END

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose CALL
  // START

  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor =
      (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 1000);
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_L3_args_t
      KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_L3_args =
          (KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_L3_args_t){
              .KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed =
                  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transposed,
              .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor =
                  KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor,
              .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr =
                  KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_L3 CLOSURE CALL
  KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_L3(
      &KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose_closure_L3_args);

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_CONVRQ_PASS_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor_pre_transpose CALL END

  // RUNNETRWORK LAYER KeywordSpotting_small_4_poolGlobalAveragePool CALL START

  KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor = (uint8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 0);
  KeywordSpotting_small_4_poolGlobalAveragePool_closure_L3_args_t KeywordSpotting_small_4_KeywordSpotting_small_4_poolGlobalAveragePool_closure_L3_args =
      (KeywordSpotting_small_4_poolGlobalAveragePool_closure_L3_args_t){
          .KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor =
              KeywordSpotting_small_4__QL_REPLACED__INTEGERIZE_BN2D_UNSIGNED_ACT_PASS_4Clip_output_0_tensor,
          .KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor = KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor,
          .KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr =
              KeywordSpotting_small_4_TILING_CODEGEN_L1_poolGlobalAveragePool_tileIdxPtr};

  // KeywordSpotting_small_4_poolGlobalAveragePool_closure_L3 CLOSURE CALL
  KeywordSpotting_small_4_poolGlobalAveragePool_closure_L3(&KeywordSpotting_small_4_KeywordSpotting_small_4_poolGlobalAveragePool_closure_L3_args);

  // RUNNETRWORK LAYER KeywordSpotting_small_4_poolGlobalAveragePool CALL END

  // RUNNETRWORK LAYER KeywordSpotting_small_4_Flatten CALL START

  // Reshape (Name: Flatten, Op: Flatten)
  KeywordSpotting_small_4_Flatten_output_0_tensor = KeywordSpotting_small_4_poolGlobalAveragePool_output_0_tensor;

  // RUNNETRWORK LAYER KeywordSpotting_small_4_Flatten CALL END

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0 CALL START

  KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args_t KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args =
      (KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args_t){.KeywordSpotting_small_4_Flatten_output_0_tensor =
                                                                           KeywordSpotting_small_4_Flatten_output_0_tensor,
                                                                       .KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr =
                                                                           KeywordSpotting_small_4_TILING_CODEGEN_L1__MERGE_GEMMRQ_PASS_0_tileIdxPtr};

  // KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_L3 CLOSURE CALL
  KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_L3(&KeywordSpotting_small_4_KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0_closure_L3_args);

  // RUNNETRWORK LAYER KeywordSpotting_small_4__MERGE_GEMMRQ_PASS_0 CALL END
}

void InitNetwork() {

  KeywordSpotting_small_4_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 3288);

  KeywordSpotting_small_4_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 2000);

  KeywordSpotting_small_4_input_0 = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 0);
  KeywordSpotting_small_4_output_0 = (int8_t *)((char *)KeywordSpotting_small_4_MEMORYARENA_L2 + 8);
  KeywordSpotting_small_4_inputs[0] = (void *)KeywordSpotting_small_4_input_0;
  KeywordSpotting_small_4_outputs[0] = (void *)KeywordSpotting_small_4_output_0;
}
