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

int8_t *TestRQAdd1_MEMORYARENA_L1; // L2_init_template
int8_t *TestRQAdd1_MEMORYARENA_L2; // L2_init_template
int8_t *TestRQAdd1_input_0;        // L2_init_template
int8_t *TestRQAdd1_input_1;        // L2_init_template
int8_t *TestRQAdd1_output_0;       // L2_init_template

static PI_L1 uint8_t TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles[2] = {0, 1};

void *TestRQAdd1_inputs[2];
void *TestRQAdd1_outputs[1];
extern struct pi_device cluster_dev;


int8_t *TestRQAdd2_MEMORYARENA_L1; // L2_init_template
int8_t *TestRQAdd2_MEMORYARENA_L2; // L2_init_template
int8_t *TestRQAdd2_input_0;        // L2_init_template
int8_t *TestRQAdd2_input_1;        // L2_init_template
int8_t *TestRQAdd2_output_0;       // L2_init_template

static PI_L1 uint8_t TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_numTiles[2] = {0, 1};

void *TestRQAdd2_inputs[2];
void *TestRQAdd2_outputs[1];

int cores_map[2][2] = {{0, 7}, {8, 15}};
int nb_dedicated_cores[2] = {8, 8};

#include "hal/eu/eu_v3.h"
#include "archi/eu/eu_v3.h"

static inline uint32_t subgroup_core_mask(uint32_t group_start, uint32_t group_end)
{
    uint32_t mask = 0;

    for (uint32_t c = group_start; c <= group_end; c++) {
        for (volatile int i = 0; i < 8; i++) {
            asm volatile("nop");
        }
        // printf("Adding core %u to subgroup barrier mask (bit %08x)\n", c, bit);
        mask |= (1u << c);
    }

    return mask;
}

static inline void subgroup_barrier_init(uint32_t barrier_id,
                                         uint32_t group_start,
                                         uint32_t group_end)
{
    uint32_t mask = subgroup_core_mask(group_start, group_end);
    eu_bar_setup(eu_bar_addr(barrier_id), mask);
}

static inline void subgroup_barrier_wait(uint32_t barrier_id, uint32_t core_id)
{
    // printf("Core %u waiting on barrier %u\n", core_id, barrier_id);
    eu_evt_maskSet(1u << PULP_HW_BAR_EVENT);
    eu_bar_trig_wait_clr(eu_bar_addr(barrier_id));
    // printf("Core %u passed barrier %u\n", core_id, barrier_id);
}

uint32_t round_0_tiles_timings[2][3] = {{0, 0, 0},{0, 0, 0}};

void print_tiles_timings() {
 printf("=== Round 0 ===\n");
 printf("Tile TestRQAdd1.TestRQAdd1__MERGE_ADDRQ_PASS_0\n[DMA_in] %d cycles\n[Kernel] %d cycles\n[DMA_out] %d cycles\n", round_0_tiles_timings[0][0], round_0_tiles_timings[0][1], round_0_tiles_timings[0][2]);
 printf("Tile TestRQAdd2.TestRQAdd2__MERGE_ADDRQ_PASS_0\n[DMA_in] %d cycles\n[Kernel] %d cycles\n[DMA_out] %d cycles\n", round_0_tiles_timings[1][0], round_0_tiles_timings[1][1], round_0_tiles_timings[1][2]);
 printf("\n");
}

void RunNetworks() {
	
	// GLOBALS INITS
	void *TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_0_ref;
	void *TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_1_ref;
	void *TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_output_0_ref;
	void *TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_0_ref;
	void *TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_1_ref;
	void *TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_output_0_ref;

	uint8_t bu_TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = 0;
	uint8_t *TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = &bu_TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;
	int8_t *TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = (int8_t *)((char *)TestRQAdd1_MEMORYARENA_L1 + 32768);
	int8_t *TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = (int8_t *)((char *)TestRQAdd1_MEMORYARENA_L1 + 16384);
	int8_t *TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = (int8_t *)((char *)TestRQAdd1_MEMORYARENA_L1 + 0);
	TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_0_ref= (void *)((char *)TestRQAdd1_input_0 + 0);
	TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_1_ref= (void *)((char *)TestRQAdd1_input_1 + 0);
	TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_output_0_ref= (void *)((char *)TestRQAdd1_output_0 + 0);

	uint8_t bu_TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = 0;
	uint8_t *TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr = &bu_TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_tileIdxPtr;
	int8_t *TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref = (int8_t *)((char *)TestRQAdd2_MEMORYARENA_L1 + 32768);
	int8_t *TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref = (int8_t *)((char *)TestRQAdd2_MEMORYARENA_L1 + 16384);
	int8_t *TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref = (int8_t *)((char *)TestRQAdd2_MEMORYARENA_L1 + 0);
	TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_0_ref= (void *)((char *)TestRQAdd2_input_0 + 0);
	TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_1_ref= (void *)((char *)TestRQAdd2_input_1 + 0);
	TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_output_0_ref= (void *)((char *)TestRQAdd2_output_0 + 0);

	// SUBGROUP BARRIERS CONFIGS
	int core_id = pi_core_id();
	uint32_t temp_start, temp_end;
	if (core_id == 0) {
		subgroup_barrier_init(2, cores_map[0][0], cores_map[0][1]);
		subgroup_barrier_init(3, cores_map[1][0], cores_map[1][1]);
	}
   	subgroup_barrier_init(4, 0, NUM_CORES - 1);
	pi_cl_team_barrier();
	eu_evt_maskSet(1u << PULP_HW_BAR_EVENT);

	// TestRQAdd1 execution
	uint32_t TestRQAdd1_channel_input = (uint32_t)-1;
	uint32_t TestRQAdd1_channel_output = (uint32_t)-1;
	if (cores_map[0][0] <= core_id && core_id <= cores_map[0][1]) {
		for (int TILING_I = 0; TILING_I < 1; TILING_I++) {
        	if (core_id == cores_map[0][0]) {
            	temp_start = getCycles();
				// Transfer input tiles
				TestRQAdd1_channel_input = mchan_channel_alloc();
				mchan_transfer_1d(1458176, TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref, TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_0_ref);
				mchan_transfer_1d(1458176, TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref, TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_1_ref);

				// Wait for input tiles

				if (TestRQAdd1_channel_input <= MCHAN_CHANNEL_ID_MAX) {
				mchan_channel_wait(TestRQAdd1_channel_input);
				mchan_channel_free(TestRQAdd1_channel_input);
				}


                temp_end = getCycles();
				round_0_tiles_timings[0 + TILING_I][0] = temp_end - temp_start;
				temp_start = getCycles();
			}
			subgroup_barrier_wait(2, core_id);

			// PULP NN RQADD
			pulp_nn_add_i8_i8_i8(TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref, TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
								TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref, 32768, 32768, 16, 32768, 32768, 16, 65536, 32768, 16, 1, 16384, 1, 1,
								8);

			subgroup_barrier_wait(2, core_id);
			if (core_id == cores_map[0][0]) {
				temp_end = getCycles();
				round_0_tiles_timings[0 + TILING_I][1] = temp_end - temp_start;
				temp_start = getCycles();
				// Transfer output tiles
				TestRQAdd1_channel_output = mchan_channel_alloc();
				mchan_transfer_1d(1327104, TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref, TestRQAdd1_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_output_0_ref);

				// Wait for output tiles

				if (TestRQAdd1_channel_output <= MCHAN_CHANNEL_ID_MAX) {
				mchan_channel_wait(TestRQAdd1_channel_output);
				mchan_channel_free(TestRQAdd1_channel_output);
				}

				temp_end = getCycles();
				round_0_tiles_timings[0 + TILING_I][2] = temp_end - temp_start;
			}
			subgroup_barrier_wait(2, core_id);
		}
	}

	// TestRQAdd2 execution
	uint32_t TestRQAdd2_channel_input = (uint32_t)-1;
	uint32_t TestRQAdd2_channel_output = (uint32_t)-1;
	if (cores_map[1][0] <= core_id && core_id <= cores_map[1][1]) {
		uint32_t nb_cycles_start_TestRQAdd2 = getCycles();
      	for (int TILING_I = 0; TILING_I < 1; TILING_I++) {
        	if (core_id == cores_map[1][0]) {
            	temp_start = getCycles();
    			// Transfer input tiles
				TestRQAdd2_channel_input = mchan_channel_alloc();
				mchan_transfer_1d(1458176, TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref, TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_0_ref);
				mchan_transfer_1d(1458176, TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref, TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_input_1_ref);

				// Wait for input tiles
				if (TestRQAdd2_channel_input <= MCHAN_CHANNEL_ID_MAX) {
				mchan_channel_wait(TestRQAdd2_channel_input);
				mchan_channel_free(TestRQAdd2_channel_input);
				}

                temp_end = getCycles();
            	round_0_tiles_timings[1 + TILING_I][0] = temp_end - temp_start;
            	temp_start = getCycles();
        	}
        	subgroup_barrier_wait(3, core_id);

			// PULP NN RQADD
			pulp_nn_add_i8_i8_i8(TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_1_ref, TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_in_2_ref,
								TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref, 32768, 32768, 16, 32768, 32768, 16, 65536, 32768, 16, 1, 16384, 1, 1,
								8);

        	subgroup_barrier_wait(3, core_id);
        	if (core_id == cores_map[1][0]) {
				temp_end = getCycles();
				round_0_tiles_timings[1 + TILING_I][1] = temp_end - temp_start;
				temp_start = getCycles();

				// Transfer output tiles
				TestRQAdd2_channel_output = mchan_channel_alloc();
				mchan_transfer_1d(1327104, TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_data_out_ref, TestRQAdd2_TILING_CODEGEN_L1__MERGE_ADDRQ_PASS_0_output_0_ref);

				// Wait for output tiles
				if (TestRQAdd2_channel_output <= MCHAN_CHANNEL_ID_MAX) {
				mchan_channel_wait(TestRQAdd2_channel_output);
				mchan_channel_free(TestRQAdd2_channel_output);
				}

				temp_end = getCycles();
				round_0_tiles_timings[1 + TILING_I][2] = temp_end - temp_start;
        	}
        	subgroup_barrier_wait(3, core_id);
      	}
		uint32_t nb_cycles_TestRQAdd2__MERGE_ADDRQ_PASS_0_TestRQAdd2 = getCycles();
	}

	// BOTH GROUPS BARRIER
	subgroup_barrier_wait(4, core_id);
}

void RunNetwork() {
	pi_cl_team_fork(NUM_CORES, (void *)RunNetworks, NULL);
}

void InitNetwork() {

	TestRQAdd1_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 49152);

	TestRQAdd1_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 49152);

	TestRQAdd1_input_0 = (int8_t *)((char *)TestRQAdd1_MEMORYARENA_L2 + 32768);
	TestRQAdd1_input_1 = (int8_t *)((char *)TestRQAdd1_MEMORYARENA_L2 + 16384);
	TestRQAdd1_output_0 = (int8_t *)((char *)TestRQAdd1_MEMORYARENA_L2 + 0);
	TestRQAdd1_inputs[0] = (void *)TestRQAdd1_input_0;
	TestRQAdd1_inputs[1] = (void *)TestRQAdd1_input_1;
	TestRQAdd1_outputs[0] = (void *)TestRQAdd1_output_0;

	TestRQAdd2_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 49152);

	TestRQAdd2_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 49152);

	TestRQAdd2_input_0 = (int8_t *)((char *)TestRQAdd2_MEMORYARENA_L2 + 32768);
	TestRQAdd2_input_1 = (int8_t *)((char *)TestRQAdd2_MEMORYARENA_L2 + 16384);
	TestRQAdd2_output_0 = (int8_t *)((char *)TestRQAdd2_MEMORYARENA_L2 + 0);
	TestRQAdd2_inputs[0] = (void *)TestRQAdd2_input_0;
	TestRQAdd2_inputs[1] = (void *)TestRQAdd2_input_1;
	TestRQAdd2_outputs[0] = (void *)TestRQAdd2_output_0;
}
