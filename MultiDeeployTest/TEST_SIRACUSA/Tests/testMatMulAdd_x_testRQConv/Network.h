
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


extern int8_t *testMatMulAdd_MEMORYARENA_L1;
static const uint32_t testMatMulAdd_MEMORYARENA_L1_len = 18432;
extern int8_t *testMatMulAdd_MEMORYARENA_L2;
static const uint32_t testMatMulAdd_MEMORYARENA_L2_len = 16384;
extern int8_t *testMatMulAdd_input_0;
static const uint32_t testMatMulAdd_input_0_len = 1536;
extern int32_t *testMatMulAdd_output_0;
static const uint32_t testMatMulAdd_output_0_len = 2048;
static const uint32_t testMatMulAdd_num_inputs = 1;
static const uint32_t testMatMulAdd_num_outputs = 1;
extern void *testMatMulAdd_inputs[1];
extern void *testMatMulAdd_outputs[1];
static const uint32_t testMatMulAdd_inputs_bytes[1] = {1536};
static const uint32_t testMatMulAdd_outputs_bytes[1] = {8192};


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
