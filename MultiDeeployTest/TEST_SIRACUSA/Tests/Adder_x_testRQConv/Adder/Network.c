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

int8_t *Adder_MEMORYARENA_L1; // L2_init_template
int8_t *Adder_MEMORYARENA_L2; // L2_init_template
int8_t *Adder_input_0;        // L2_init_template
int8_t *Adder_input_1;        // L2_init_template
int32_t *Adder_output_0;      // L2_init_template

static PI_L1 uint8_t Adder_TILING_CODEGEN_L1_Add_numTiles[2] = {0, 1};

void *Adder_inputs[2];
void *Adder_outputs[1];
extern struct pi_device cluster_dev;
// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *Adder_TILING_CODEGEN_L1_Add_data_in_1_ref;
  int8_t *Adder_TILING_CODEGEN_L1_Add_data_in_2_ref;
  int32_t *Adder_TILING_CODEGEN_L1_Add_data_out_ref;
} Adder_Add_tiling_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void Adder_Add_tiling_closure(void *Adder_Add_tiling_closure_args) {
  // CLOSURE ARG CAST
  Adder_Add_tiling_closure_args_t *args = (Adder_Add_tiling_closure_args_t *)Adder_Add_tiling_closure_args;
  int8_t *Adder_TILING_CODEGEN_L1_Add_data_in_1_ref = args->Adder_TILING_CODEGEN_L1_Add_data_in_1_ref;
  int8_t *Adder_TILING_CODEGEN_L1_Add_data_in_2_ref = args->Adder_TILING_CODEGEN_L1_Add_data_in_2_ref;
  int32_t *Adder_TILING_CODEGEN_L1_Add_data_out_ref = args->Adder_TILING_CODEGEN_L1_Add_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Add (Name: Add, Op: Add)
  BEGIN_SINGLE_CORE
  for (uint32_t i = 0; i < 125; i++) {
    Adder_TILING_CODEGEN_L1_Add_data_out_ref[i] = Adder_TILING_CODEGEN_L1_Add_data_in_1_ref[i] + Adder_TILING_CODEGEN_L1_Add_data_in_2_ref[i] + 0;
  }
  END_SINGLE_CORE

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  int8_t *Adder_TILING_CODEGEN_L1_Add_data_in_1_ref;
  int8_t *Adder_TILING_CODEGEN_L1_Add_data_in_2_ref;
  int32_t *Adder_TILING_CODEGEN_L1_Add_data_out_ref;
} Adder_Add_cluster_fork_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void Adder_Add_cluster_fork(void *Adder_Add_cluster_fork_args) {
  // CLOSURE ARG CAST
  Adder_Add_cluster_fork_args_t *args = (Adder_Add_cluster_fork_args_t *)Adder_Add_cluster_fork_args;
  int8_t *Adder_TILING_CODEGEN_L1_Add_data_in_1_ref = args->Adder_TILING_CODEGEN_L1_Add_data_in_1_ref;
  int8_t *Adder_TILING_CODEGEN_L1_Add_data_in_2_ref = args->Adder_TILING_CODEGEN_L1_Add_data_in_2_ref;
  int32_t *Adder_TILING_CODEGEN_L1_Add_data_out_ref = args->Adder_TILING_CODEGEN_L1_Add_data_out_ref;

  // CLOSURE FUNCTION CALL
  Adder_Add_tiling_closure_args_t Adder_Adder_Add_tiling_closure_args =
      (Adder_Add_tiling_closure_args_t){.Adder_TILING_CODEGEN_L1_Add_data_in_1_ref = Adder_TILING_CODEGEN_L1_Add_data_in_1_ref,
                                        .Adder_TILING_CODEGEN_L1_Add_data_in_2_ref = Adder_TILING_CODEGEN_L1_Add_data_in_2_ref,
                                        .Adder_TILING_CODEGEN_L1_Add_data_out_ref = Adder_TILING_CODEGEN_L1_Add_data_out_ref};

  // Adder_Add_tiling_closure CLOSURE CALL
  Adder_Add_tiling_closure(&Adder_Adder_Add_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *Adder_TILING_CODEGEN_L1_Add_tileIdxPtr;
} Adder_Add_closure_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void Adder_Add_closure(void *Adder_Add_closure_args) {
  // CLOSURE ARG CAST
  Adder_Add_closure_args_t *args = (Adder_Add_closure_args_t *)Adder_Add_closure_args;
  uint8_t *Adder_TILING_CODEGEN_L1_Add_tileIdxPtr = args->Adder_TILING_CODEGEN_L1_Add_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *Adder_TILING_CODEGEN_L1_Add_data_in_1_ref = (int8_t *)((char *)Adder_MEMORYARENA_L1 + 500); // ARENA ALLOCATE
  int8_t *Adder_TILING_CODEGEN_L1_Add_data_in_2_ref = (int8_t *)((char *)Adder_MEMORYARENA_L1 + 625); // ARENA ALLOCATE
  int32_t *Adder_TILING_CODEGEN_L1_Add_data_out_ref = (int32_t *)((char *)Adder_MEMORYARENA_L1 + 0);  // ARENA ALLOCATE
  void *Adder_TILING_CODEGEN_L1_Add_input_0_ref = (void *)((char *)Adder_input_0 + 0);
  void *Adder_TILING_CODEGEN_L1_Add_input_1_ref = (void *)((char *)Adder_input_1 + 0);
  void *Adder_TILING_CODEGEN_L1_Add_output_0_ref = (void *)((char *)Adder_output_0 + 0);

  const static char Add_L2_suffix[] = " cycles \n";

  const static char Add_L2_prefix[] = "[Add_L2][SB][125 ops][Tile ";

  uint32_t Add_L2_egress_dma_wait_end_measurements[1];

  uint32_t Add_L2_egress_dma_wait_start_measurements[1];

  uint32_t Add_L2_ingress_dma_wait_end_measurements[1];

  uint32_t Add_L2_ingress_dma_wait_start_measurements[1];

  uint32_t Add_L2_kernel_end_measurements[1];

  uint32_t Add_L2_kernel_start_measurements[1];

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = Adder_TILING_CODEGEN_L1_Add_numTiles[*Adder_TILING_CODEGEN_L1_Add_tileIdxPtr];
       TILING_I < Adder_TILING_CODEGEN_L1_Add_numTiles[(*Adder_TILING_CODEGEN_L1_Add_tileIdxPtr) + 1]; TILING_I++) {

    Add_L2_ingress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441917, Adder_TILING_CODEGEN_L1_Add_data_in_1_ref, Adder_TILING_CODEGEN_L1_Add_input_0_ref);
    mchan_transfer_1d(1441917, Adder_TILING_CODEGEN_L1_Add_data_in_2_ref, Adder_TILING_CODEGEN_L1_Add_input_1_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    Add_L2_ingress_dma_wait_end_measurements[TILING_I] = getCycles();

    Add_L2_kernel_start_measurements[TILING_I] = getCycles();

    Adder_Add_cluster_fork_args_t Adder_Adder_Add_cluster_fork_args =
        (Adder_Add_cluster_fork_args_t){.Adder_TILING_CODEGEN_L1_Add_data_in_1_ref = Adder_TILING_CODEGEN_L1_Add_data_in_1_ref,
                                        .Adder_TILING_CODEGEN_L1_Add_data_in_2_ref = Adder_TILING_CODEGEN_L1_Add_data_in_2_ref,
                                        .Adder_TILING_CODEGEN_L1_Add_data_out_ref = Adder_TILING_CODEGEN_L1_Add_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)Adder_Add_cluster_fork, &Adder_Adder_Add_cluster_fork_args);

    Add_L2_kernel_end_measurements[TILING_I] = getCycles();

    Add_L2_egress_dma_wait_start_measurements[TILING_I] = getCycles();

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311220, Adder_TILING_CODEGEN_L1_Add_data_out_ref, Adder_TILING_CODEGEN_L1_Add_output_0_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    Add_L2_egress_dma_wait_end_measurements[TILING_I] = getCycles();

    // CLOSE TILING LOOP
  }
  *Adder_TILING_CODEGEN_L1_Add_tileIdxPtr += 1;

  // Deinitialize DMA futures

  StopTimer();
  for (int PROFILING_I =
           ((*Adder_TILING_CODEGEN_L1_Add_tileIdxPtr > 0) ? Adder_TILING_CODEGEN_L1_Add_numTiles[(*Adder_TILING_CODEGEN_L1_Add_tileIdxPtr - 1)] : 0);
       PROFILING_I < Adder_TILING_CODEGEN_L1_Add_numTiles[*Adder_TILING_CODEGEN_L1_Add_tileIdxPtr]; PROFILING_I++) {

    printf("%s%u] %s%u%s", Add_L2_prefix, PROFILING_I, "Input DMA took ",
           Add_L2_ingress_dma_wait_end_measurements[PROFILING_I] - Add_L2_ingress_dma_wait_start_measurements[PROFILING_I], Add_L2_suffix);

    printf("%s%u] %s%u%s", Add_L2_prefix, PROFILING_I, "Kernel took ",
           Add_L2_kernel_end_measurements[PROFILING_I] - Add_L2_kernel_start_measurements[PROFILING_I], Add_L2_suffix);

    printf("%s%u] %s%u%s", Add_L2_prefix, PROFILING_I, "Output DMA took ",
           Add_L2_egress_dma_wait_end_measurements[PROFILING_I] - Add_L2_egress_dma_wait_start_measurements[PROFILING_I], Add_L2_suffix);
  }
  StartTimer();

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

// LAYER FUNCTION STRUCT DEFINITION, START
typedef struct {
  uint8_t *Adder_TILING_CODEGEN_L1_Add_tileIdxPtr;
} Adder_Add_closure_L3_args_t;
// LAYER FUNCTION STRUCT DEFINITION, END

// LAYER FUNCTION CLOSURE, START
static void Adder_Add_closure_L3(void *Adder_Add_closure_L3_args) {
  // CLOSURE ARG CAST
  Adder_Add_closure_L3_args_t *args = (Adder_Add_closure_L3_args_t *)Adder_Add_closure_L3_args;
  uint8_t *Adder_TILING_CODEGEN_L1_Add_tileIdxPtr = args->Adder_TILING_CODEGEN_L1_Add_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  Adder_Add_closure_args_t Adder_Adder_Add_closure_args =
      (Adder_Add_closure_args_t){.Adder_TILING_CODEGEN_L1_Add_tileIdxPtr = Adder_TILING_CODEGEN_L1_Add_tileIdxPtr};

  // Adder_Add_closure CLOSURE CALL
  Adder_Add_closure(&Adder_Adder_Add_closure_args);

  // CLOSURE ARG WRITEBACK
}
// LAYER FUNCTION CLOSURE, END

void RunNetwork() {

  uint8_t bu_Adder_TILING_CODEGEN_L1_Add_tileIdxPtr = 0;
  uint8_t *Adder_TILING_CODEGEN_L1_Add_tileIdxPtr = &bu_Adder_TILING_CODEGEN_L1_Add_tileIdxPtr;
  // RUNNETRWORK LAYER Adder_Add CALL START

  Adder_Add_closure_L3_args_t Adder_Adder_Add_closure_L3_args =
      (Adder_Add_closure_L3_args_t){.Adder_TILING_CODEGEN_L1_Add_tileIdxPtr = Adder_TILING_CODEGEN_L1_Add_tileIdxPtr};

  // Adder_Add_closure_L3 CLOSURE CALL
  Adder_Add_closure_L3(&Adder_Adder_Add_closure_L3_args);

  // RUNNETRWORK LAYER Adder_Add CALL END
}

void InitNetwork() {

  Adder_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 750);

  Adder_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 750);

  Adder_input_0 = (int8_t *)((char *)Adder_MEMORYARENA_L2 + 500);
  Adder_input_1 = (int8_t *)((char *)Adder_MEMORYARENA_L2 + 625);
  Adder_output_0 = (int32_t *)((char *)Adder_MEMORYARENA_L2 + 0);
  Adder_inputs[0] = (void *)Adder_input_0;
  Adder_inputs[1] = (void *)Adder_input_1;
  Adder_outputs[0] = (void *)Adder_output_0;
}
