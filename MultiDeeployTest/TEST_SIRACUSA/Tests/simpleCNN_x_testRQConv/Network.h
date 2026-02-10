
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


extern int8_t *simpleCNN_MEMORYARENA_L1;
static const uint32_t simpleCNN_MEMORYARENA_L1_len = 63640;
extern int8_t *simpleCNN_MEMORYARENA_L2;
static const uint32_t simpleCNN_MEMORYARENA_L2_len = 20480;
extern int8_t *simpleCNN_input_0;
static const uint32_t simpleCNN_input_0_len = 1024;
extern uint8_t *simpleCNN_output_0;
static const uint32_t simpleCNN_output_0_len = 10;
static const uint32_t simpleCNN_num_inputs = 1;
static const uint32_t simpleCNN_num_outputs = 1;
extern void *simpleCNN_inputs[1];
extern void *simpleCNN_outputs[1];
static const uint32_t simpleCNN_inputs_bytes[1] = {1024};
static const uint32_t simpleCNN_outputs_bytes[1] = {10};


extern int8_t *testRQConv_MEMORYARENA_L1;
static const uint32_t testRQConv_MEMORYARENA_L1_len = 13060;
extern int8_t *testRQConv_MEMORYARENA_L2;
static const uint32_t testRQConv_MEMORYARENA_L2_len = 12744;
extern int8_t *testRQConv_input_0;
static const uint32_t testRQConv_input_0_len = 4096;
extern int8_t *testRQConv_output_0;
static const uint32_t testRQConv_output_0_len = 6372;
static const uint32_t testRQConv_num_inputs = 1;
static const uint32_t testRQConv_num_outputs = 1;
extern void *testRQConv_inputs[1];
extern void *testRQConv_outputs[1];
static const uint32_t testRQConv_inputs_bytes[1] = {4096};
static const uint32_t testRQConv_outputs_bytes[1] = {6372};
#endif
