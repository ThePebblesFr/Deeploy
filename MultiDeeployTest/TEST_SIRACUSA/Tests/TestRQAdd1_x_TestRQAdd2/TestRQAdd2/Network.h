
#ifndef __DEEPLOY_HEADER__
#define __DEEPLOY_HEADER__
#include "DeeployPULPMath.h"
#include "bsp/ram.h"
#include "dory_mem.h"
#include "mchan_siracusa.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "stdint.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void RunNetwork();
void InitNetwork();

extern int8_t *TestRQAdd2_MEMORYARENA_L1; // L2_init_template
static const uint32_t TestRQAdd2_MEMORYARENA_L1_len = 31992;
extern int8_t *TestRQAdd2_MEMORYARENA_L2; // L2_init_template
static const uint32_t TestRQAdd2_MEMORYARENA_L2_len = 49152;
extern int8_t *TestRQAdd2_input_0; // L2_init_template
static const uint32_t TestRQAdd2_input_0_len = 16384;
extern int8_t *TestRQAdd2_input_1; // L2_init_template
static const uint32_t TestRQAdd2_input_1_len = 16384;
extern int8_t *TestRQAdd2_output_0; // L2_init_template
static const uint32_t TestRQAdd2_output_0_len = 16384;
static const uint32_t TestRQAdd2_num_inputs = 2;
static const uint32_t TestRQAdd2_num_outputs = 1;
extern void *TestRQAdd2_inputs[2];
extern void *TestRQAdd2_outputs[1];
static const uint32_t TestRQAdd2_inputs_bytes[2] = {16384, 16384};
static const uint32_t TestRQAdd2_outputs_bytes[1] = {16384};
#endif
