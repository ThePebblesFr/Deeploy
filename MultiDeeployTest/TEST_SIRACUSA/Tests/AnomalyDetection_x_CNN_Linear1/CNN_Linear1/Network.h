
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

extern int8_t *CNN_Linear1_MEMORYARENA_L1; // L2_init_template
static const uint32_t CNN_Linear1_MEMORYARENA_L1_len = 31840;
extern int8_t *CNN_Linear1_MEMORYARENA_L2; // L2_init_template
static const uint32_t CNN_Linear1_MEMORYARENA_L2_len = 20480;
extern int8_t *CNN_Linear1_input_0; // L2_init_template
static const uint32_t CNN_Linear1_input_0_len = 1024;
extern uint8_t *CNN_Linear1_output_0; // L2_init_template
static const uint32_t CNN_Linear1_output_0_len = 10;
static const uint32_t CNN_Linear1_num_inputs = 1;
static const uint32_t CNN_Linear1_num_outputs = 1;
extern void *CNN_Linear1_inputs[1];
extern void *CNN_Linear1_outputs[1];
static const uint32_t CNN_Linear1_inputs_bytes[1] = {1024};
static const uint32_t CNN_Linear1_outputs_bytes[1] = {10};
#endif
