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

static PI_L1 uint16_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size[4] = {10664, 344, 5208, 168};

static PI_L1 uint8_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles[2] = {0, 4};

static PI_L1 uint32_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_cmd[4] = {1976744, 1966424, 1971288, 1966248};

static PI_L1 uint8_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_size_1d[4] = {86, 86, 42, 42};

static PI_L1 int16_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_relativeOffset[4] = {15872, -15786, 15872, 0};

static PI_L1 uint32_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_cmd[4] = {1976744, 1966424, 1971288, 1966248};

static PI_L1 uint8_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_size_1d[4] = {86, 86, 42, 42};

static PI_L1 int16_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_relativeOffset[4] = {15872, -15786, 15872, 0};

static PI_L1 uint32_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_cmd[4] = {1845672, 1835352, 1840216, 1835176};

static PI_L1 uint8_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_size_1d[4] = {86, 86, 42, 42};

static PI_L1 int16_t TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_relativeOffset[4] = {15872, -15786, 15872, 0};

void *TestRQAdd_inputs[2];
void *TestRQAdd_outputs[1];
extern struct pi_device cluster_dev;

int cores_map[1][2] = {{0, 3}};
int nb_dedicated_cores[1] = {4};

typedef struct {
  uint8_t group_start;
  uint8_t group_size;
  volatile uint32_t arrived[NUM_CORES];
  volatile uint32_t epoch;
} subgroup_barrier_t;

static inline void subgroup_barrier_init(subgroup_barrier_t *b, int group_start, int group_end) {
  b->group_start = group_start;
  b->group_size = group_end - group_start + 1;
  b->epoch = 0;
  for (int i = b->group_start; i < NUM_CORES; i++) {
    b->arrived[i] = 0;
  }
}

static inline void subgroup_barrier_wait(subgroup_barrier_t *b, uint32_t core_id) {
  b->arrived[core_id] = 1;
  uint32_t starting_epoch = b->epoch;
  int all_arrived = 0;

  for (int i = b->group_start; i < b->group_start + b->group_size; i++) {
    if (b->arrived[i] == 1) {
      all_arrived++;
    }
  }

  if (all_arrived != b->group_size) {
    while (b->epoch == starting_epoch) {}
  } else {
    b->epoch++;
  }
  
  for (int i = b->group_start; i < b->group_start + b->group_size; i++) {
    b->arrived[i] = 0;
  }
}
PI_L1 static subgroup_barrier_t g_barrier_0;

// ===== FUSED FUNCTIONS: Scheduling Round 0 =====

void *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_0_ref;
void *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_1_ref;
void *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_output_0_ref;

typedef struct {
  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;

  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;

} Round_0_tiling_closure_args_t;

static void Round_0_tiling_closure(Round_0_tiling_closure_args_t* Round_0_tiling_closure_args) {
    Round_0_tiling_closure_args_t* args = (Round_0_tiling_closure_args_t*) Round_0_tiling_closure_args;
  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;

  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;




        int core_id = pi_core_id();
    
        uint32_t TestRQAdd_channel_input = (uint32_t)-1;
        uint32_t TestRQAdd_channel_output = (uint32_t)-1;

        if (cores_map[0][0] <= core_id && core_id <= cores_map[0][1]) {
uint32_t nb_cycles_start_TestRQAdd = getCycles();
      for (int TILING_I = 0; TILING_I < 1; TILING_I++) {
        if (core_id == cores_map[0][0]) {
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


    // UPDATE VARIABLE TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref
    *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size[TILING_I];

            }
        subgroup_barrier_wait(&g_barrier_0, core_id);

  // PULP NN RQADD
  pulp_nn_add_i8_i8_i8(TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref, TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
                       TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref, 32768, 32768, 16, 32768, 32768, 16, 65536, 32768, 16, 1,
                       *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref, 1, 1, 4);

        subgroup_barrier_wait(&g_barrier_0, core_id);
        if (core_id == cores_map[0][0]) {
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

        }
        subgroup_barrier_wait(&g_barrier_0, core_id);
uint32_t nb_cycles_TestRQAdd__MERGE_ADDRQ_PASS_0_TestRQAdd = getCycles();
      }
    }
        pi_cl_team_barrier();
}

typedef struct {
  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;

} Round_0_closure_args_t;

static void Round_0_closure(Round_0_closure_args_t* Round_0_closure_args) {
    Round_0_closure_args_t* args = (Round_0_closure_args_t*) Round_0_closure_args;
  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;



  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = (uint16_t *)((char *)TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size + 0);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 21328);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 10664);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 0);


    Round_0_tiling_closure_args_t Round_0_tiling_closure_args = (Round_0_tiling_closure_args_t) {
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref,
      .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr};
    pi_cl_team_fork(NUM_CORES, (void *)Round_0_tiling_closure, &Round_0_tiling_closure_args);
}


// ===== FUSED FUNCTIONS: Scheduling Round 1 =====

typedef struct {
  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;

  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;

} Round_1_tiling_closure_args_t;

static void Round_1_tiling_closure(Round_1_tiling_closure_args_t* Round_1_tiling_closure_args) {
    Round_1_tiling_closure_args_t* args = (Round_1_tiling_closure_args_t*) Round_1_tiling_closure_args;
  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;

  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;




        int core_id = pi_core_id();
    
        uint32_t TestRQAdd_channel_input = (uint32_t)-1;
        uint32_t TestRQAdd_channel_output = (uint32_t)-1;

        if (cores_map[0][0] <= core_id && core_id <= cores_map[0][1]) {
uint32_t nb_cycles_start_TestRQAdd = getCycles();
      for (int TILING_I = 1; TILING_I < 2; TILING_I++) {
        if (core_id == cores_map[0][0]) {
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


    // UPDATE VARIABLE TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref
    *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size[TILING_I];

            }
        subgroup_barrier_wait(&g_barrier_0, core_id);

  // PULP NN RQADD
  pulp_nn_add_i8_i8_i8(TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref, TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
                       TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref, 32768, 32768, 16, 32768, 32768, 16, 65536, 32768, 16, 1,
                       *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref, 1, 1, 4);

        subgroup_barrier_wait(&g_barrier_0, core_id);
        if (core_id == cores_map[0][0]) {
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

        }
        subgroup_barrier_wait(&g_barrier_0, core_id);
uint32_t nb_cycles_TestRQAdd__MERGE_ADDRQ_PASS_0_TestRQAdd = getCycles();
      }
    }
        pi_cl_team_barrier();
}

typedef struct {
  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;

} Round_1_closure_args_t;

static void Round_1_closure(Round_1_closure_args_t* Round_1_closure_args) {
    Round_1_closure_args_t* args = (Round_1_closure_args_t*) Round_1_closure_args;
  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;



  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = (uint16_t *)((char *)TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size + 0);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 21328);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 10664);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 0);


    Round_1_tiling_closure_args_t Round_1_tiling_closure_args = (Round_1_tiling_closure_args_t) {
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref,
      .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr};
    pi_cl_team_fork(NUM_CORES, (void *)Round_1_tiling_closure, &Round_1_tiling_closure_args);
}


// ===== FUSED FUNCTIONS: Scheduling Round 2 =====

typedef struct {
  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;

  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;

} Round_2_tiling_closure_args_t;

static void Round_2_tiling_closure(Round_2_tiling_closure_args_t* Round_2_tiling_closure_args) {
    Round_2_tiling_closure_args_t* args = (Round_2_tiling_closure_args_t*) Round_2_tiling_closure_args;
  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;

  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;




        int core_id = pi_core_id();
    
        uint32_t TestRQAdd_channel_input = (uint32_t)-1;
        uint32_t TestRQAdd_channel_output = (uint32_t)-1;

        if (cores_map[0][0] <= core_id && core_id <= cores_map[0][1]) {
uint32_t nb_cycles_start_TestRQAdd = getCycles();
      for (int TILING_I = 2; TILING_I < 3; TILING_I++) {
        if (core_id == cores_map[0][0]) {
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


    // UPDATE VARIABLE TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref
    *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size[TILING_I];

            }
        subgroup_barrier_wait(&g_barrier_0, core_id);

  // PULP NN RQADD
  pulp_nn_add_i8_i8_i8(TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref, TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
                       TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref, 32768, 32768, 16, 32768, 32768, 16, 65536, 32768, 16, 1,
                       *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref, 1, 1, 4);

        subgroup_barrier_wait(&g_barrier_0, core_id);
        if (core_id == cores_map[0][0]) {
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

        }
        subgroup_barrier_wait(&g_barrier_0, core_id);
uint32_t nb_cycles_TestRQAdd__MERGE_ADDRQ_PASS_0_TestRQAdd = getCycles();
      }
    }
        pi_cl_team_barrier();
}

typedef struct {
  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;

} Round_2_closure_args_t;

static void Round_2_closure(Round_2_closure_args_t* Round_2_closure_args) {
    Round_2_closure_args_t* args = (Round_2_closure_args_t*) Round_2_closure_args;
  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;



  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = (uint16_t *)((char *)TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size + 0);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 21328);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 10664);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 0);


    Round_2_tiling_closure_args_t Round_2_tiling_closure_args = (Round_2_tiling_closure_args_t) {
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref,
      .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr};
    pi_cl_team_fork(NUM_CORES, (void *)Round_2_tiling_closure, &Round_2_tiling_closure_args);
}


// ===== FUSED FUNCTIONS: Scheduling Round 3 =====

typedef struct {
  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;

  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;

} Round_3_tiling_closure_args_t;

static void Round_3_tiling_closure(Round_3_tiling_closure_args_t* Round_3_tiling_closure_args) {
    Round_3_tiling_closure_args_t* args = (Round_3_tiling_closure_args_t*) Round_3_tiling_closure_args;
  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref;
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref;

  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;




        int core_id = pi_core_id();
    
        uint32_t TestRQAdd_channel_input = (uint32_t)-1;
        uint32_t TestRQAdd_channel_output = (uint32_t)-1;

        if (cores_map[0][0] <= core_id && core_id <= cores_map[0][1]) {
uint32_t nb_cycles_start_TestRQAdd = getCycles();
      for (int TILING_I = 3; TILING_I < 4; TILING_I++) {
        if (core_id == cores_map[0][0]) {
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


    // UPDATE VARIABLE TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref
    *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size[TILING_I];

            }
        subgroup_barrier_wait(&g_barrier_0, core_id);

  // PULP NN RQADD
  pulp_nn_add_i8_i8_i8(TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref, TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
                       TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref, 32768, 32768, 16, 32768, 32768, 16, 65536, 32768, 16, 1,
                       *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref, 1, 1, 4);

        subgroup_barrier_wait(&g_barrier_0, core_id);
        if (core_id == cores_map[0][0]) {
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

        }
        subgroup_barrier_wait(&g_barrier_0, core_id);
uint32_t nb_cycles_TestRQAdd__MERGE_ADDRQ_PASS_0_TestRQAdd = getCycles();
      }
    }
        pi_cl_team_barrier();
}

typedef struct {
  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;

} Round_3_closure_args_t;

static void Round_3_closure(Round_3_closure_args_t* Round_3_closure_args) {
    Round_3_closure_args_t* args = (Round_3_closure_args_t*) Round_3_closure_args;
  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = args->TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;



  uint16_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = (uint16_t *)((char *)TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size + 0);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 21328);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 10664);
  int8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L1 + 0);


    Round_3_tiling_closure_args_t Round_3_tiling_closure_args = (Round_3_tiling_closure_args_t) {
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_size_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
            .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref,
      .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr};
    pi_cl_team_fork(NUM_CORES, (void *)Round_3_tiling_closure, &Round_3_tiling_closure_args);
}


void RunNetwork() {

  uint8_t bu_TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = 0;
  uint8_t *TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = &bu_TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;


TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_0_ref= (void *)((char *)TestRQAdd_input_0 + 0);
TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_1_ref= (void *)((char *)TestRQAdd_input_1 + 0);
TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_output_0_ref= (void *)((char *)TestRQAdd_output_0 + 0);

// ===== Scheduling Round 0 =====

Round_0_closure_args_t Round_0_closure_args = (Round_0_closure_args_t) {

      .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr,
};
    Round_0_closure(&Round_0_closure_args);
// ===== Scheduling Round 1 =====

Round_1_closure_args_t Round_1_closure_args = (Round_1_closure_args_t) {

      .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr,
};
    Round_1_closure(&Round_1_closure_args);
// ===== Scheduling Round 2 =====

Round_2_closure_args_t Round_2_closure_args = (Round_2_closure_args_t) {

      .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr,
};
    Round_2_closure(&Round_2_closure_args);
// ===== Scheduling Round 3 =====

Round_3_closure_args_t Round_3_closure_args = (Round_3_closure_args_t) {

      .TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = TestRQAdd_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr,
};
    Round_3_closure(&Round_3_closure_args);

}

void InitNetwork() {

  TestRQAdd_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 31992);

  TestRQAdd_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 49152);

  TestRQAdd_input_0 = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L2 + 32768);
  TestRQAdd_input_1 = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L2 + 16384);
  TestRQAdd_output_0 = (int8_t *)((char *)TestRQAdd_MEMORYARENA_L2 + 0);
  TestRQAdd_inputs[0] = (void *)TestRQAdd_input_0;
  TestRQAdd_inputs[1] = (void *)TestRQAdd_input_1;
  TestRQAdd_outputs[0] = (void *)TestRQAdd_output_0;
subgroup_barrier_init(&g_barrier_0, cores_map[0][0], cores_map[0][1]);
}
