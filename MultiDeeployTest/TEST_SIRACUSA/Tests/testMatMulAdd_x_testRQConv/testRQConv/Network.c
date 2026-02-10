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

int8_t *testRQConv_MEMORYARENA_L1;
int8_t *testRQConv_MEMORYARENA_L2;
int8_t *testRQConv_input_0;
int8_t *testRQConv_output_0;

static PI_L2 int8_t testRQConv_weight_tensor[512] = {
    -63,  41,   -81,  -108, -104, 63,   -83,  2,    -8,   -46,  -86,  -94,  -115, 48,   -109, 75,   102,  -64,  25,   92,   7,    -73,  17,   -1,   119,  -44,
    -6,   -34,  26,   -77,  -53,  -14,  8,    41,   7,    26,   -36,  72,   -81,  -51,  -67,  -42,  -125, -88,  52,   112,  0,    33,   45,   -113, 23,   89,
    49,   -3,   -57,  108,  -123, 44,   -93,  -39,  91,   67,   41,   83,   115,  16,   93,   89,   124,  -20,  45,   -72,  -56,  -103, 35,   125,  -52,  50,
    -9,   -103, -87,  39,   114,  -35,  -122, 84,   74,   -8,   20,   -33,  4,    104,  20,   14,   -126, -71,  21,   52,   -78,  -62,  -47,  -21,  -60,  94,
    121,  -121, -79,  19,   -68,  -60,  -51,  -49,  -27,  93,   -82,  38,   9,    -128, 2,    47,   119,  120,  -64,  23,   108,  109,  109,  34,   -87,  -122,
    69,   59,   95,   104,  35,   -115, -128, -20,  -100, -37,  87,   97,   63,   -37,  -8,   21,   61,   122,  3,    -8,   -54,  1,    41,   63,   -68,  121,
    96,   72,   -29,  -15,  13,   -115, -74,  -85,  123,  -59,  95,   40,   88,   -75,  94,   72,   -44,  -109, -45,  101,  110,  -95,  112,  -112, -102, 90,
    34,   -50,  18,   -102, 108,  -49,  113,  90,   -119, 75,   29,   -18,  -78,  96,   81,   -88,  39,   -18,  107,  -49,  6,    125,  7,    91,   -58,  67,
    122,  76,   101,  -74,  15,   83,   35,   -19,  -90,  -33,  32,   53,   74,   -127, 41,   -79,  83,   25,   -81,  44,   60,   -63,  46,   23,   114,  118,
    -58,  -39,  -56,  -75,  55,   11,   64,   -14,  -81,  -16,  -89,  47,   84,   -9,   -77,  81,   61,   86,   -102, -61,  53,   -88,  -72,  30,   126,  123,
    77,   -7,   33,   53,   -76,  -126, 43,   -60,  -11,  84,   -62,  -123, -43,  -68,  0,    115,  15,   -9,   59,   93,   -18,  -13,  -65,  -119, -21,  20,
    64,   23,   -121, 47,   71,   112,  -62,  36,   8,    -106, -63,  61,   103,  -23,  -109, -103, -86,  -16,  113,  -121, -45,  -1,   55,   -114, -24,  74,
    -107, 117,  -68,  48,   50,   38,   -63,  19,   -117, 14,   -82,  -128, -78,  -17,  -126, 2,    -1,   26,   -122, 4,    -125, -42,  -77,  107,  -85,  92,
    110,  -73,  -52,  90,   -53,  -50,  15,   -88,  -31,  17,   -65,  44,   3,    -116, -19,  18,   100,  -23,  -115, -112, -110, 126,  47,   12,   -6,   99,
    -37,  -5,   -88,  -51,  10,   28,   -46,  -12,  -78,  -11,  -72,  -59,  56,   -52,  -94,  -6,   -55,  -3,   17,   35,   -28,  -73,  -37,  33,   -52,  -19,
    -52,  92,   57,   84,   -41,  -3,   -70,  107,  -11,  67,   119,  80,   -112, 112,  71,   60,   66,   -125, -38,  -94,  23,   -15,  -10,  40,   -114, 39,
    84,   -25,  107,  103,  21,   24,   43,   -93,  36,   95,   53,   118,  -48,  121,  -55,  46,   126,  95,   53,   -109, -49,  117,  -125, 125,  9,    8,
    -61,  96,   42,   21,   3,    -2,   -109, -121, 109,  -7,   -121, 45,   -103, 74,   -93,  -78,  90,   -121, 40,   19,   0,    -91,  112,  -73,  -46,  -48,
    -82,  44,   63,   -87,  122,  -8,   112,  -20,  -34,  71,   54,   39,   -4,   -88,  43,   59,   -92,  -25,  106,  85,   -78,  105,  -101, -32,  111,  28,
    65,   8,    37,   -49,  -41,  94,   77,   49,   70,   102,  93,   -62,  30,   -21,  -65,  44,   -44,  -46};

static PI_L2 int32_t testRQConv_RQS1mul_tensor[4] = {71, 74, 58, 80};

static PI_L2 int32_t testRQConv_RQS1add_tensor[4] = {-997157, 852762, 832762, -677151};

static PI_L1 uint8_t testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[2] = {0, 1};

static PI_L1 uint8_t testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles[2] = {0, 1};

void *testRQConv_inputs[1];
void *testRQConv_outputs[1];
extern struct pi_device cluster_dev;
// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;
} testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure(void *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 2, 64, 32] -> [1, 64, 32, 2] (Name: _MERGE_CONVRQ_PASS_0_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 2;

  uint16_t dimLen_2 = 64;

  uint16_t dimLen_3 = 32;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_2 / NUM_CORES;
    const uint32_t leftover = dimLen_2 - baseChunk * NUM_CORES;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_2 = offset; i_2 < offset + chunk; i_2++) {

      for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

          ((int8_t (*)[dimLen_2][dimLen_3][dimLen_1])testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref)[i_0][i_2][i_3][i_1] =
              ((int8_t (*)[dimLen_1][dimLen_2][dimLen_3])testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;
} testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork(void *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args =
      (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t){
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
              testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
              testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref};

  // testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure CLOSURE CALL
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure(void *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref = (int8_t *)((char *)testRQConv_MEMORYARENA_L1 + 0);
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref = (int8_t *)((char *)testRQConv_MEMORYARENA_L1 + 4096);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref = (void *)((char *)testRQConv_input_0 + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed + 0);

  const static char _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_prefix[] = "[_MERGE_CONVRQ_PASS_0_input_0_transpose_L2][SB][0 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
           [*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr];
       TILING_I < testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
                      [(*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1445888, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                      testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args =
        (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t){
            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
                testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
                testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork,
                    &testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1314816, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref,
                      testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    _MERGE_CONVRQ_PASS_0_input_0_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr > 0)
                              ? testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles[(
                                    *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
                         [*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr];
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
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3(void *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args =
      (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t){.testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed =
                                                                             testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed,
                                                                         .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                                                                             testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr};

  // testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure CLOSURE CALL
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *testRQConv__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
} testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure(void *testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args;
  void *testRQConv__MERGE_CONVRQ_PASS_0_buffer = args->testRQConv__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN CONV

  pulp_nn_conv_i8_i8_i8(testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref, testRQConv__MERGE_CONVRQ_PASS_0_buffer, NULL,
                        testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                        testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref, 1, 16, 32, 64, 2,
                        27, 59, 4, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  void *testRQConv__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
} testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork(void *testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args;
  void *testRQConv__MERGE_CONVRQ_PASS_0_buffer = args->testRQConv__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL
  testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args =
      (testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args_t){
          .testRQConv__MERGE_CONVRQ_PASS_0_buffer = testRQConv__MERGE_CONVRQ_PASS_0_buffer,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref};

  // testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure CLOSURE CALL
  testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testRQConv__MERGE_CONVRQ_PASS_0_closure(void *testRQConv__MERGE_CONVRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_closure_args_t *args = (testRQConv__MERGE_CONVRQ_PASS_0_closure_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_closure_args;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = (int8_t *)((char *)testRQConv_MEMORYARENA_L1 + 6372);
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = (int8_t *)((char *)testRQConv_MEMORYARENA_L1 + 12516);
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = (int32_t *)((char *)testRQConv_MEMORYARENA_L1 + 13028);
  int32_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = (int32_t *)((char *)testRQConv_MEMORYARENA_L1 + 13044);
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = (int8_t *)((char *)testRQConv_MEMORYARENA_L1 + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_tensor_ref = (void *)((char *)testRQConv_weight_tensor + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_RQS1add_tensor_ref = (void *)((char *)testRQConv_RQS1add_tensor + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_RQS1mul_tensor_ref = (void *)((char *)testRQConv_RQS1mul_tensor + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref =
      (void *)((char *)testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed + 0);
  void *testRQConv__MERGE_CONVRQ_PASS_0_buffer = (void *)((char *)testRQConv_MEMORYARENA_L1 + 10468);

  const static char _MERGE_CONVRQ_PASS_0_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_0_L2_prefix[] = "[_MERGE_CONVRQ_PASS_0_L2][SB][1636011 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       TILING_I < testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr) + 1]; TILING_I++) {

    _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1445888, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
                      testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref);
    mchan_transfer_1d(1442304, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                      testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_tensor_ref);
    mchan_transfer_1d(1441808, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_RQS1add_tensor_ref);
    mchan_transfer_1d(1441808, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_RQS1mul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _MERGE_CONVRQ_PASS_0_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_L2_kernel_start_measurements[TILING_I] = getCycles();

    testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args =
        (testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args_t){
            .testRQConv__MERGE_CONVRQ_PASS_0_buffer = testRQConv__MERGE_CONVRQ_PASS_0_buffer,
            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork, &testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_cluster_fork_args);

    _MERGE_CONVRQ_PASS_0_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1317092, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
                      testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    _MERGE_CONVRQ_PASS_0_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr > 0)
                              ? testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]; PROFILING_I++) {

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
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testRQConv__MERGE_CONVRQ_PASS_0_closure_L3(void *testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  testRQConv__MERGE_CONVRQ_PASS_0_closure_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_closure_args = (testRQConv__MERGE_CONVRQ_PASS_0_closure_args_t){
      .testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed,
      .testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
      .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

  // testRQConv__MERGE_CONVRQ_PASS_0_closure CLOSURE CALL
  testRQConv__MERGE_CONVRQ_PASS_0_closure(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;
} testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void
testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure(void *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t *)
          testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 59, 27, 4] -> [1, 4, 59, 27] (Name: _MERGE_CONVRQ_PASS_0_output_0_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 59;

  uint16_t dimLen_2 = 27;

  uint16_t dimLen_3 = 4;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

      const uint32_t baseChunk = dimLen_1 / NUM_CORES;
      const uint32_t leftover = dimLen_1 - baseChunk * NUM_CORES;
      const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
      const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
      for (uint32_t i_1 = offset; i_1 < offset + chunk; i_1++) {

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {

          ((int8_t (*)[dimLen_3][dimLen_1][dimLen_2])
               testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref)[i_0][i_3][i_1][i_2] =
              ((int8_t (*)[dimLen_1][dimLen_2][dimLen_3])
                   testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref)[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;
} testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void
testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork(void *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t
      testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args =
          (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t){
              .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
                  testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref,
              .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
                  testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref};

  // testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure CLOSURE CALL
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure(void *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref = (int8_t *)((char *)testRQConv_MEMORYARENA_L1 + 6372);
  int8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref = (int8_t *)((char *)testRQConv_MEMORYARENA_L1 + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref =
      (void *)((char *)testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed + 0);
  void *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref = (void *)((char *)testRQConv_output_0 + 0);

  const static char _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_suffix[] = " cycles \n";

  const static char _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_prefix[] = "[_MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2][SB][0 ops][Tile ";

  uint32_t _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_egress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_egress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_ingress_dma_wait_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_ingress_dma_wait_start_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_kernel_end_measurements[1];

  uint32_t _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles
           [*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr];
       TILING_I < testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles
                      [(*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1448164, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref,
                      testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_kernel_start_measurements[TILING_I] = getCycles();

    testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t
        testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args =
            (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t){
                .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
                    testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref,
                .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
                    testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork,
                    &testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args);

    _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1317092, testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref,
                      testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr > 0)
                              ? testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles[(
                                    *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles
                         [*testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr];
       PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_ingress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_kernel_end_measurements[PROFILING_I] -
               _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_egress_dma_wait_end_measurements[PROFILING_I] -
               _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;
} testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3(void *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t *args =
      (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t *)testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
      args->testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args =
      (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t){
          .testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
              testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr};

  // testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure CLOSURE CALL
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

void RunNetwork() {
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  void *testRQConv__MERGE_CONVRQ_PASS_0_buffer;

  uint8_t bu_testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr = 0;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      &bu_testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
  uint8_t bu_testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = 0;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = &bu_testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
  uint8_t bu_testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr = 0;
  uint8_t *testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
      &bu_testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = (int8_t *)((char *)testRQConv_MEMORYARENA_L2 + 6372);
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args =
      (testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t){.testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed =
                                                                                testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed,
                                                                            .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                                                                                testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr};

  // testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3 CLOSURE CALL
  testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args);

  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = (int8_t *)((char *)testRQConv_MEMORYARENA_L2 + 0);
  testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args =
      (testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args_t){
          .testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed = testRQConv__MERGE_CONVRQ_PASS_0_input_0_transposed,
          .testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

  // testRQConv__MERGE_CONVRQ_PASS_0_closure_L3 CLOSURE CALL
  testRQConv__MERGE_CONVRQ_PASS_0_closure_L3(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_closure_L3_args);

  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args =
      (testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t){
          .testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
          .testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
              testRQConv_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr};

  // testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3 CLOSURE CALL
  testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3(&testRQConv_testRQConv__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args);
}

void InitNetwork() {

  testRQConv_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 13060);

  testRQConv_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 12744);

  testRQConv_input_0 = (int8_t *)((char *)testRQConv_MEMORYARENA_L2 + 0);
  testRQConv_output_0 = (int8_t *)((char *)testRQConv_MEMORYARENA_L2 + 6372);
  testRQConv_inputs[0] = (void *)testRQConv_input_0;
  testRQConv_outputs[0] = (void *)testRQConv_output_0;
}
