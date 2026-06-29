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

int8_t *TestRQAdd_MEMORYARENA_L1; // L2_init_template
int8_t *TestRQAdd_MEMORYARENA_L2; // L2_init_template
int8_t *TestRQAdd_input_0;        // L2_init_template
int8_t *TestRQAdd_input_1;        // L2_init_template
int8_t *TestRQAdd_output_0;       // L2_init_template

static PI_L1 uint16_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size[6] = {5332, 172, 5332, 172, 5208, 168};

static PI_L1 uint8_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles[2] = {0, 6};

static PI_L1 uint32_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_cmd[6] = {1971412, 1966252, 1971412, 1966252, 1971288, 1966248};

static PI_L1 uint8_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_size_1d[6] = {43, 43, 43, 43, 42, 42};

static PI_L1 int16_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_relativeOffset[6] = {15872, -15829, 15872, -15829, 15872, 0};

static PI_L1 uint32_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_cmd[6] = {1971412, 1966252, 1971412, 1966252, 1971288, 1966248};

static PI_L1 uint8_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_size_1d[6] = {43, 43, 43, 43, 42, 42};

static PI_L1 int16_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_relativeOffset[6] = {15872, -15829, 15872, -15829, 15872, 0};

static PI_L1 uint32_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_cmd[6] = {1840340, 1835180, 1840340, 1835180, 1840216, 1835176};

static PI_L1 uint8_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_size_1d[6] = {43, 43, 43, 43, 42, 42};

static PI_L1 int16_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_relativeOffset[6] = {15872, -15829, 15872, -15829, 15872, 0};

void *TestRQAdd_inputs[2];
void *TestRQAdd_outputs[1];
extern struct pi_device cluster_dev;
// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;
} TestRQAdd__MERGE_ADDRQ_PASS_0_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void TestRQAdd__MERGE_ADDRQ_PASS_0_tiling_closure(void *TestRQAdd__MERGE_ADDRQ_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  TestRQAdd__MERGE_ADDRQ_PASS_0_tiling_closure_args_t *args =
      (TestRQAdd__MERGE_ADDRQ_PASS_0_tiling_closure_args_t *)TestRQAdd__MERGE_ADDRQ_PASS_0_tiling_closure_args;
  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP NN RQADD
  pulp_nn_add_i8_i8_i8(TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref, TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
                       TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref, 32768, 32768, 16, 32768, 32768, 16, 65536, 32768, 16, 1,
                       *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref, 1, 1, NUM_CORES);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;
} TestRQAdd__MERGE_ADDRQ_PASS_0_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void TestRQAdd__MERGE_ADDRQ_PASS_0_cluster_fork(void *TestRQAdd__MERGE_ADDRQ_PASS_0_cluster_fork_args) {
  // CLOSURE ARG CAST
  TestRQAdd__MERGE_ADDRQ_PASS_0_cluster_fork_args_t *args =
      (TestRQAdd__MERGE_ADDRQ_PASS_0_cluster_fork_args_t *)TestRQAdd__MERGE_ADDRQ_PASS_0_cluster_fork_args;
  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL
  TestRQAdd__MERGE_ADDRQ_PASS_0_tiling_closure_args_t TestRQAdd_TestRQAdd__MERGE_ADDRQ_PASS_0_tiling_closure_args =
      (TestRQAdd__MERGE_ADDRQ_PASS_0_tiling_closure_args_t){
          .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref,
          .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref,
          .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
          .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref};

  // TestRQAdd__MERGE_ADDRQ_PASS_0_tiling_closure CLOSURE CALL
  TestRQAdd__MERGE_ADDRQ_PASS_0_tiling_closure(&TestRQAdd_TestRQAdd__MERGE_ADDRQ_PASS_0_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;
} TestRQAdd__MERGE_ADDRQ_PASS_0_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void TestRQAdd__MERGE_ADDRQ_PASS_0_closure(void *TestRQAdd__MERGE_ADDRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  TestRQAdd__MERGE_ADDRQ_PASS_0_closure_args_t *args = (TestRQAdd__MERGE_ADDRQ_PASS_0_closure_args_t *)TestRQAdd__MERGE_ADDRQ_PASS_0_closure_args;
  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = (uint16_t *)((char *)TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size + 0);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 5332);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 10664);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 0);
  void *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_0_ref = (void *)((char *)TestRQAdd_input_0 + 0);
  void *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_1_ref = (void *)((char *)TestRQAdd_input_1 + 0);
  void *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_output_0_ref = (void *)((char *)TestRQAdd_output_0 + 0);

  const static char _MERGE_ADDRQ_PASS_0_L2_suffix[] = " cycles \n";

  const static char _MERGE_ADDRQ_PASS_0_L2_prefix[] = "[_MERGE_ADDRQ_PASS_0_L2][SB][16384 ops][Tile ";

  uint32_t _MERGE_ADDRQ_PASS_0_L2_egress_dma_wait_end_measurements[6];

  uint32_t _MERGE_ADDRQ_PASS_0_L2_egress_dma_wait_start_measurements[6];

  uint32_t _MERGE_ADDRQ_PASS_0_L2_ingress_dma_wait_end_measurements[6];

  uint32_t _MERGE_ADDRQ_PASS_0_L2_ingress_dma_wait_start_measurements[6];

  uint32_t _MERGE_ADDRQ_PASS_0_L2_kernel_end_measurements[6];

  uint32_t _MERGE_ADDRQ_PASS_0_L2_kernel_start_measurements[6];

  // Initialize DMA futures
  uint32_t TestRQAdd_channel_output = (uint32_t)-1;
  uint32_t TestRQAdd_channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles[*TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr];
       TILING_I < TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles[(*TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr) + 1]; TILING_I++) {

    _MERGE_ADDRQ_PASS_0_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    TestRQAdd_channel_input = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_cmd[TILING_I],
                                  TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref, TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_0_ref,
                                  TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_size_1d[TILING_I], 128);

    // UPDATE VARIABLE TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_0_ref
    TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_0_ref = (void *)((char *)(TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_0_ref) +
                                                                           TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_relativeOffset[TILING_I]);

    mchan_transfer_2d_ext_strided(TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_cmd[TILING_I],
                                  TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref, TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_1_ref,
                                  TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_size_1d[TILING_I], 128);

    // UPDATE VARIABLE TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_1_ref
    TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_1_ref = (void *)((char *)(TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_1_ref) +
                                                                           TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_relativeOffset[TILING_I]);

    // Wait for input tiles

    if (TestRQAdd_channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(TestRQAdd_channel_input);
      mchan_channel_free(TestRQAdd_channel_input);
    }

    _MERGE_ADDRQ_PASS_0_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    _MERGE_ADDRQ_PASS_0_L2_kernel_start_measurements[TILING_I] = getCycles();

    // UPDATE VARIABLE TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref
    *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size[TILING_I];

    TestRQAdd__MERGE_ADDRQ_PASS_0_cluster_fork_args_t TestRQAdd_TestRQAdd__MERGE_ADDRQ_PASS_0_cluster_fork_args =
        (TestRQAdd__MERGE_ADDRQ_PASS_0_cluster_fork_args_t){
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)TestRQAdd__MERGE_ADDRQ_PASS_0_cluster_fork, &TestRQAdd_TestRQAdd__MERGE_ADDRQ_PASS_0_cluster_fork_args);

    _MERGE_ADDRQ_PASS_0_L2_kernel_end_measurements[TILING_I] = getCycles();

    _MERGE_ADDRQ_PASS_0_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    TestRQAdd_channel_output = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_cmd[TILING_I],
                                  TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref, TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_output_0_ref,
                                  TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_size_1d[TILING_I], 128);

    // UPDATE VARIABLE TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_output_0_ref
    TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_output_0_ref = (void *)((char *)(TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_output_0_ref) +
                                                                            TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_relativeOffset[TILING_I]);

    // Wait for output tiles

    if (TestRQAdd_channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(TestRQAdd_channel_output);
      mchan_channel_free(TestRQAdd_channel_output);
    }

    _MERGE_ADDRQ_PASS_0_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I = ((*TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr > 0)
                              ? TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles[(*TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr - 1)]
                              : 0);
       PROFILING_I < TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles[*TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr]; PROFILING_I++) {

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_0_L2_prefix, PROFILING_I, "Input DMA took ",
           _MERGE_ADDRQ_PASS_0_L2_ingress_dma_wait_end_measurements[PROFILING_I] - _MERGE_ADDRQ_PASS_0_L2_ingress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_0_L2_prefix, PROFILING_I, "Kernel took ",
           _MERGE_ADDRQ_PASS_0_L2_kernel_end_measurements[PROFILING_I] - _MERGE_ADDRQ_PASS_0_L2_kernel_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_0_L2_suffix);

    printf("%s%u] %s%u%s", _MERGE_ADDRQ_PASS_0_L2_prefix, PROFILING_I, "Output DMA took ",
           _MERGE_ADDRQ_PASS_0_L2_egress_dma_wait_end_measurements[PROFILING_I] - _MERGE_ADDRQ_PASS_0_L2_egress_dma_wait_start_measurements[PROFILING_I],
           _MERGE_ADDRQ_PASS_0_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;
} TestRQAdd__MERGE_ADDRQ_PASS_0_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void TestRQAdd__MERGE_ADDRQ_PASS_0_closure_L3(void *TestRQAdd__MERGE_ADDRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  TestRQAdd__MERGE_ADDRQ_PASS_0_closure_L3_args_t *args = (TestRQAdd__MERGE_ADDRQ_PASS_0_closure_L3_args_t *)TestRQAdd__MERGE_ADDRQ_PASS_0_closure_L3_args;
  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  TestRQAdd__MERGE_ADDRQ_PASS_0_closure_args_t TestRQAdd_TestRQAdd__MERGE_ADDRQ_PASS_0_closure_args = (TestRQAdd__MERGE_ADDRQ_PASS_0_closure_args_t){
      .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr};

  // TestRQAdd__MERGE_ADDRQ_PASS_0_closure CLOSURE CALL
  TestRQAdd__MERGE_ADDRQ_PASS_0_closure(&TestRQAdd_TestRQAdd__MERGE_ADDRQ_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

void RunNetwork() {

  uint8_t bu_TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = 0;
  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = &bu_TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;
  // RUNNETRWORK LAYER TestRQAdd__MERGE_ADDRQ_PASS_0 CALL START

  TestRQAdd__MERGE_ADDRQ_PASS_0_closure_L3_args_t TestRQAdd_TestRQAdd__MERGE_ADDRQ_PASS_0_closure_L3_args = (TestRQAdd__MERGE_ADDRQ_PASS_0_closure_L3_args_t){
      .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr};

  // TestRQAdd__MERGE_ADDRQ_PASS_0_closure_L3 CLOSURE CALL
  TestRQAdd__MERGE_ADDRQ_PASS_0_closure_L3(&TestRQAdd_TestRQAdd__MERGE_ADDRQ_PASS_0_closure_L3_args);

  // RUNNETRWORK LAYER TestRQAdd__MERGE_ADDRQ_PASS_0 CALL END
}

void InitNetwork() {

  TestRQAdd_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 15996);

  TestRQAdd_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 49152);

  TestRQAdd_input_0 = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L2 + 16384);
  TestRQAdd_input_1 = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L2 + 32768);
  TestRQAdd_output_0 = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L2 + 0);
  TestRQAdd_inputs[0] = (void *)TestRQAdd_input_0;
  TestRQAdd_inputs[1] = (void *)TestRQAdd_input_1;
  TestRQAdd_outputs[0] = (void *)TestRQAdd_output_0;
}
